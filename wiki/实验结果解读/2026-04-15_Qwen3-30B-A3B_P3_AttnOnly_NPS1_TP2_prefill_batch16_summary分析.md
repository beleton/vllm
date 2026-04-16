# 2026-04-15 Qwen3-30B-A3B P3_AttnOnly NPS1_TP2 prefill batch_16 summary 分析

## 范围
- 数据：
  - `test_results/P3_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/prefill-like/global-fixed/batch_16/summary.csv`
  - `test_results/P3_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/prefill-like/global-fixed/batch_16/summary_l3_source_latency.csv`
  - `test_results/P3_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/prefill-like/global-fixed/batch_16/detail_l3_source_latency.csv`
- 场景：`NPS1_TP2 / prefill-like / global-fixed / batch_16`
- 对照：`balanced_span4` 与 `acc-local-l3_span4`
- 边界：这批结果仍属于 `group_span=4` 诊断口径，只用于判断当前实现是否改变 locality 指标，以及这种改变是否兑现为 latency 收益
- 说明：`summary_l3_source_latency.csv` 来自单独补跑的自定义 `PCM` 批次；其中 `Derived Avg L3 Miss Latency (ns)` 与标准 `summary.csv` 的 `Ave L3 Miss Latency (ns)` 基本对齐，除 `q256/balanced` 外偏差都在 `3.1%` 内，`q256/balanced` 偏差为 `6.45%`

## 核心结论
- `acc-local-l3_span4` 仍不是稳定有效优化：`q=64/128/256` 时 `slowest rank mean (ms)` 相对 `balanced_span4` 下降 `10.45%/11.18%/3.29%`，`q=512/1024/2048` 时上升 `6.45%/21.76%/28.65%`
- 新自定义 `PCM` 已把短 `q` 的高 `L3 miss` 延迟主因锁定到 `same-node another CCX`
  - `q64`：`472.22 -> 147.88 ns`
  - `q128`：`430.22 -> 226.67 ns`
  - `q256`：`186.34 -> 152.20 ns`
  - 这些点的 `L3 miss` 延迟份额都仍由这一路径主导：`balanced=98.94%/98.56%/91.26%`，`acc-local-l3=98.08%/98.35%/95.73%`
- `acc-local-l3` 没有显式 `L2` 优化；短 `q` 下 `Local L2` fill 增加来自调度映射变化
  - 源码里两条路径都沿用同一套 `tile size / scratchpad` 计算
  - 变化点是 `balanced` 用全局原子计数器抢 `kv_head × legacy_thread_offset` 任务，`acc-local-l3` 改成按 `subgroup -> kv_head -> legacy_thread_offset` 静态分片
  - `q64/q128/q256` 上 `All Demand DC Fills (pti)` 基本不变，但 `Demand DC Fills From Local L2 (pti)` 升到 `12.99/12.37/11.04`，同时 `Demand DC Fills From another CCX in same node (pti)` 降到 `0.24/0.22/0.18`，`L3 Access (pti)` 降到 `0.87/1.00/1.30`
- `balanced` 下 `q` 增大后总平均 `L3 miss` 延迟下降，不是因为 `Local Memory` 更快地“接管”了分母，而是两条主路径各自的平均 `L3 miss` 延迟都在下降，且请求构成从 `same-node another CCX` 逐步转向 `Local Memory`
- 从 scheduler 机制看，小 `batch` 更容易放大 `balanced` 的跨 `CCD` `K/V` 共享，因此理论上更可能为 `acc-local-l3` 提供 locality 收益空间
  - scheduler 先按全局 `total_kv_len` 计算 `kv_len_per_thread`；`batch` 变小而单请求长度不变时，同一请求更容易被切成更多 `workitem` 并落到更多 `legacy thread bucket`
  - `balanced` 运行时又允许所有物理线程全局抢 `kv_head × legacy_thread_offset` 任务，不限制 `subgroup`
  - 这条机制推导在现有 `qhead32_kvhead16` 的 `batch_1` 数据上只得到部分支持：`q256/q512/q1024` 时 `acc-local-l3` 比 `balanced` 快 `4.34%/1.84%/2.85%`，但 `q64/q128` 仍慢 `7.08%/1.38%`，`q2048` 更慢 `37.59%`
- `q512~2048` 上 `acc-local-l3_span4` 的 slowdown 不能归因到更差的 `L3 miss` 来源延迟
  - `acc-local-l3` 的总平均 `L3 miss` 延迟是 `144.13/135.13/132.83 ns`
  - `balanced` 是 `153.84/147.99/133.42 ns`
  - 同时 `acc-local-l3` 的 `L3 Access (pti)`、`L3 Miss (pti)` 也更低
- `q512~2048` 上的 slowdown 已落到执行效率层面
  - `IPC (Sys + User)`：`2.86/2.97/3.07 -> 2.74/2.57/2.55`
  - `CPI (Sys + User)`：`0.35/0.34/0.33 -> 0.36/0.39/0.39`
  - 源码可直接确认：`balanced` 仍保留全局动态抢任务；`acc-local-l3` 去掉了跨 `subgroup` 的动态抢任务，并把每个本地 `kv_head` 固定到 `4` 个 `subgroup`
- 因此，这批新数据把旧结论推进到“按来源拆分后的平均 `L3 miss` 延迟 + 调度层执行效率”层面：`q64~256` 的收益来自 `same-node another CCX` 单次 miss 延迟下降，并伴随更多 demand fill 停在 `Local L2`；`q512~2048` 的 slowdown 主要不在当前测到的 `L3 miss` 服务路径内，而在更低的执行效率上

## 1. `q64/q128/q256`：收益来自 `same-node another CCX` 单次 miss 延迟被压低
- `q64`
  - `balanced`：`same-node another CCX` 请求占比 `98.40%`，这一路径的平均 `L3 miss` 延迟 `472.22 ns`，来源延迟份额 `98.94%`
  - `acc-local-l3`：请求占比 `98.83%`，这一路径的平均 `L3 miss` 延迟 `147.88 ns`，来源延迟份额 `98.08%`
  - 两种模式的主来源相同，但 `balanced` 的这一路径单次 miss 延迟高出 `219.26%`
- `q128`
  - `balanced`：`430.22 ns`
  - `acc-local-l3`：`226.67 ns`
  - 差值仍有 `203.55 ns`
- `q256`
  - `balanced`：`186.34 ns`
  - `acc-local-l3`：`152.20 ns`
  - 差值收窄到 `34.14 ns`，对应端到端收益也收窄到 `3.29%`
- `Local Memory` 不是短 `q` 的主因
  - `q64/q128` 下其请求占比仅 `1.24%/1.93%` 与 `0.41%/0.57%`
  - `q256` 下虽然 `balanced` 的 `Local Memory` 请求占比升到 `10.08%`，但这一路径的平均 `L3 miss` 延迟 `144.78 ns` 仍低于 `same-node another CCX` 的 `186.34 ns`

## 2. 为什么没有显式 `L2` 优化，却出现更多 `Local L2` fills
- 从源码看，`acc-local-l3` 没有单独增加 `L2` 特化逻辑
  - `balanced` 与 `acc-local-l3` 都用同一套 `AttentionScheduler::calcu_default_tile_size(...)` 计算默认 tile size
  - `acc-local-l3` 真正新增的是 `thread -> subgroup -> kv_head -> legacy_thread_offset` 映射
- 两条路径的调度差异是：
  - `balanced`：每个物理线程都通过全局原子计数器抢 `kv_head_idx / thread_offset` 任务
  - `acc-local-l3`：每个物理线程先固定 `subgroup_id` 与 `local_offset`，再只执行本 `subgroup` 覆盖的 `kv_head` 与 `legacy_thread_offset`
- `q64` 的 rank 级 metadata 已直接显示这种静态归属
  - `thread_num=127`
  - `subgroup_num=8`
  - `group_span=4`
  - `actual_kv_head_num=2`
  - `kv_head_to_subgroup=[0,4]`
  - 这表示每个本地 `kv_head` 固定覆盖 `4` 个 `subgroup`，不是仍由全部线程自由抢同一批任务
- 对应到 `PCM` 指标，短 `q` 上 demand fill 的总量几乎没变，但停留层级变了
  - `q64`：`All Demand DC Fills (pti) 13.21 -> 13.45`，`Demand DC Fills From Local L2 (pti) 11.44 -> 12.99`，`Demand DC Fills From another CCX in same node (pti) 0.84 -> 0.24`，`L3 Access (pti) 5.92 -> 0.87`
  - `q128`：`12.46 -> 12.84`，`11.14 -> 12.37`，`0.74 -> 0.22`，`4.63 -> 1.00`
  - `q256`：`11.16 -> 11.50`，`10.36 -> 11.04`，`0.36 -> 0.18`，`3.22 -> 1.30`
- 因此，短 `q` 下更多 fill 被 `Local L2` 吸收，应写成调度映射改变后的结果，而不是 `acc-local-l3` 额外实现了专门的 `L2` 优化

## 3. 为什么 `balanced` 下 `Local Memory` 占比升高，但总平均延迟反而下降
- 现在可以直接同时看“来源请求占比”和“各来源路径的平均 `L3 miss` 延迟”，不必只靠 `pti` 与来源占比间接推断
- `balanced` 的来源构成变化是：
  - `q64`：`same-node another CCX` 请求占比 `98.40%`，`Local Memory` `1.24%`
  - `q512`：`61.47%` 与 `38.06%`
  - `q1024`：`28.48%` 与 `71.12%`
  - `q2048`：`18.19%` 与 `81.35%`
- 同时两条路径各自的平均 `L3 miss` 延迟也在下降：
  - `same-node another CCX`：`472.22 -> 430.22 -> 186.34 -> 155.18 -> 153.48 -> 152.99 ns`
  - `Local Memory`：`226.27 -> 208.66 -> 144.78 -> 150.20 -> 144.92 -> 128.10 ns`
- 因此 `balanced` 的总平均 `L3 miss` 延迟才会从 `469.65 ns` 持续降到 `133.42 ns`
- `Local Memory` 延迟份额在长 `q` 升到 `69.64%/78.11%`，表示长 `q` 下总 miss 延迟更多由这一路径贡献，不表示它的单次服务时间在变差

## 4. 为什么 `q512~2048` 上 `acc-local-l3` 的 locality 更好却更慢
- `q512`
  - `slowest rank mean (ms)`：`1.493269 -> 1.589635`，变慢 `6.45%`
  - 总平均 `L3 miss` 延迟：`153.84 -> 144.13 ns`
  - `same-node another CCX` 平均延迟：`155.18 -> 148.78 ns`
  - `Local Memory` 平均延迟：`150.20 -> 115.21 ns`
- `q1024`
  - `slowest rank mean (ms)`：`5.571502 -> 6.783606`，变慢 `21.76%`
  - 总平均 `L3 miss` 延迟：`147.99 -> 135.13 ns`
  - `same-node another CCX` 平均延迟：`153.48 -> 141.52 ns`
  - `Local Memory` 平均延迟：`144.92 -> 113.15 ns`
- `q2048`
  - `slowest rank mean (ms)`：`21.323087 -> 27.431844`，变慢 `28.65%`
  - 总平均 `L3 miss` 延迟：`133.42 -> 132.83 ns`
  - `same-node another CCX` 平均延迟：`152.99 -> 145.89 ns`
  - `Local Memory` 平均延迟：`128.10 -> 124.62 ns`
- 这三点里，`acc-local-l3` 的 `L3 miss` 路径没有更差；`summary.csv` 里的 `L3 Access (pti)` 与 `L3 Miss (pti)` 也更低
- 但执行效率更差
  - `IPC (Sys + User)`：`2.86 -> 2.74`、`2.97 -> 2.57`、`3.07 -> 2.55`
  - `CPI (Sys + User)`：`0.35 -> 0.36`、`0.34 -> 0.39`、`0.33 -> 0.39`
  - `DC Fills From Local L2 (pti)` 也不再像短 `q` 那样上升，而是 `36.58 -> 35.55`、`36.89 -> 33.66`、`36.92 -> 32.99`
- 源码能直接确认的调度差异是：
  - `balanced` 仍保留全局原子计数器抢任务；长 `q` 下 `effective_thread_num=64`、`workitem_group_num=79`，每个 `legacy_thread_offset` 对应 `1` 或 `2` 个 workitem group，任意物理线程都可继续抢这些任务
  - `acc-local-l3` 则改成按 `subgroup` 与 `local_offset` 静态分派；长 `q` 下 metadata 固定为 `subgroup_num=8`、`group_span=4`、`kv_head_to_subgroup=[0,4]`，每个本地 `kv_head` 只由固定的 `4` 个 `subgroup` 处理，线程不会再跨 `subgroup` 动态抢任务
- 这两条路径的根本区别是：
  - `balanced` 抢的是 task；数据访问路径只是随 task 分配一起改变
  - `acc-local-l3` 固定的是 task 资格；线程先被限制在固定 `subgroup -> kv_head -> legacy_thread_offset` 范围内，再去执行对应 task
- 因而 locality 优势不会自动转成 latency 优势
  - `acc-local-l3` 确实更容易把同一 `kv_head` 的 `K/V` 访问限制在较小的 `CCD/L3` 范围内
  - 但它同时拿掉了 `balanced` 的全局动态均衡能力；当 `legacy slot/workitem` 成本不完全相等时，这部分执行效率收益可能更大
- `qhead32_kvhead16/group_span=1/q=kv=1024` 的单次日志已经直接给出当前实现里的尾部来源
  - `8` 个本地 `kv_head` 被固定到 `8` 个 `subgroup`
  - 其中一个 `subgroup` 只有 `15` 个线程，导致 `thread 112` 同时承担 `legacy_slots=[0,15]`
  - 这说明现有 `acc-local-l3` 不是“先按 locality 分组后再做局部均衡调度”，而是“先按 locality 卡住执行范围，再复用 legacy slot 映射”
- 因此长 `q` 的 slowdown 不能再写成“另一路 miss 延迟变高”；当前已定位到“更低的执行效率”
- 相比 `balanced`，`acc-local-l3` 已知具体失去的是：
  - 跨 `subgroup` 的全局动态抢任务能力
  - 单个本地 `kv_head` 可动用的 `subgroup` 范围，从不受 `subgroup` 约束变为固定 `4` 个 `subgroup`
- 如果继续优化 `acc-local-l3`，优先级应放在调度代码而不是 attention 数学 kernel
  - 当前证据里，`tile size / scratchpad` 逻辑没有分叉，`L3 miss` 路径也没有更差
  - slowdown 已定位到 `subgroup` 静态分派与动态均衡能力缺失这一层
- 现有数据还不能继续证明长 `q` 的主导子因子究竟是负载不均、线程空转、聚合缓存容量变小，还是其他执行层代价；这一步需要额外的 per-thread 时间线或热点证据
- 结合现有 `group_span=4` 边界，文档仍只能把这批结果视为诊断结果，不能把 `acc-local-l3_span4` 写成已被证明的稳定优化配置

## 5. 为什么理论上小 `batch` 更容易为 `acc-local-l3` 提供收益空间
- 这条结论不是来自本页 `batch_16` 指标本身，而是来自 scheduler 机制与现有低 `batch` 样例的交叉观察
- scheduler 的切分口径是：
  - 先遍历所有请求的可见 `KV` 区间，累计全局 `total_kv_len`
  - 再据此计算每个逻辑线程桶的 `kv_len_per_thread`
  - `batch` 越小、单请求长度不变时，`kv_len_per_thread` 越小，同一请求越容易跨更多逻辑线程桶，被切成更多 `workitem`
- 现成样例已经能直接看到这一点
  - `qhead32_kvhead16/q=kv=1024`
  - `batch_1` 时，单个请求被切成 `15` 个 `workitem`
  - `batch_8` 时，`8` 个请求总共只切成 `23` 个 `workitem`，平均每请求 `2.875`
- 这会放大 `balanced` 的 locality 风险
  - `balanced` 运行时按全局原子计数器抢 `kv_head × legacy_thread_offset` 任务
  - 同一请求若占了更多 `legacy thread bucket`，同一 `kv_head` 的 `K/V` 就更容易被更多 `CCD` 的线程消费
- `acc-local-l3` 的设计靶点正是限制这种跨 `CCD` 扩散
  - 它先按 `L3/CCX` 拓扑把线程切成 `subgroup`
  - 再把每个本地 `kv_head` 限在固定 `subgroup` 范围内执行
- 因此，小 `batch` 更适合 `acc-local-l3` 的更准确表述是：
  - 小 `batch` 更容易让 locality 成为主导矛盾
  - 因而更容易出现 `acc-local-l3` 可以兑现收益的窗口
  - 这不等于小 `batch` 下 `acc-local-l3` 必然更快
- 当前 `qhead32_kvhead16` 的 `batch_1` 数据只支持“概率提高”，不支持“稳定更优”
  - `q256/q512/q1024` 上 `acc-local-l3` 快于 `balanced`
  - `q64/q128` 仍慢于 `balanced`
  - `q2048` 则明显更慢

## 6. 仍然不能推出的结论
- 自定义 `PCM` 只把结论推进到“按来源拆分后的平均 `L3 miss` 延迟”
- 仍不能把 `q64/balanced` 的 `472.22 ns` 继续细拆成某一级 `MSHR`、bank conflict、snoop 仲裁或 fabric 队列单独主导
- 仍不能把长 `q` 的执行效率下降继续细拆成负载不均、线程空转、聚合缓存容量变化或其他单一因素主导
- 仍不能把这批 `span4` 诊断结果外推到 `group_span=1`、其他 `batch`、其他 `NPS/TP` 或真实服务全链路
