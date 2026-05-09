# CPU attention `balanced` 与 `acc-local-l3` 线程级任务划分简述

## `balanced` 的任务生成与执行

`balanced` 的任务划分先发生在调度阶段。调度器先遍历请求与 `q token` 分段，按当前可见的 `KV` 区间累计工作量，生成一批 `workitem_group`。每个 `workitem_group` 对应某个请求上一段连续 `q token` 的 attention 计算，以及它对应的一段 `KV` 计算范围。随后，这些 `workitem_group` 会按顺序装入若干逻辑线程桶。这里的“逻辑线程桶”只是调度阶段的任务槽位，不是后面固定执行的物理线程。

连续非空的逻辑线程桶数量会被统计为 `effective_thread_num`。真正执行 attention 时，运行时 task 不是“一个线程拿一个 `workitem_group`”，而是先把工作沿两个维度展开：

- `kv_head`
- 非空逻辑线程桶

因此，一个运行时 task 对应的是“一个 `kv_head` + 一个逻辑线程桶”。线程拿到 task 后，会顺序处理这个逻辑线程桶里挂着的整段 `workitem_group`。整个 task 执行期间 `kv_head` 保持不变，但物理线程并不会在调度阶段预先绑定到某个 `kv_head`。

`balanced` 的领取方式是全线程池共享同一个全局计数器。谁先空闲，谁就去抢下一个 `kv_head × 逻辑线程桶` task。它优先保证全线程池的负载均衡，不保证 `kv_head` 与 `L3/CCX` 的局部绑定。

## `acc-local-l3` 的改动范围

现有 `acc-local-l3` 不改 attention 的数学 kernel，也不重写底层 `workitem_group` 和 reduction item 的生成逻辑；它直接复用 legacy scheduler 产出的这批任务。它真正改的是线程组织方式和任务领取范围。

线程启动后，会先按局部缓存拓扑把线程划成若干 `subgroup`。分组依据是 `numa_node + socket + l3_cache_id`。然后为每个 `kv_head` 指定一个起始 `subgroup`，并用 `group_span` 决定它覆盖多少个连续 `subgroup`：

- `group_span=1`：一个 `kv_head` 只在一个局部线程组内执行
- `group_span>1`：一个 `kv_head` 可以交给多个相邻 `subgroup` 联合处理

进入执行阶段后，`acc-local-l3` 先限制“只有覆盖到该 `kv_head` 的 `subgroup` 内线程才有资格执行这个 `kv_head`”。在这个覆盖范围内部，线程再通过该 `kv_head` 自己的局部计数器，动态领取原有逻辑线程桶对应的任务和归约项。也就是说，它不是把 `balanced` 的全局动态均衡改成静态分派，而是把动态领取范围从“全线程池”收缩到“负责该 `kv_head` 的局部线程池”。

因此，两者在线程级的核心差异可以概括为：

- `balanced`：全线程池上的全局动态均衡
- `acc-local-l3`：先按 `L3/CCX` 局部域限制 `kv_head` 的可执行线程范围，再在这个局部范围内做动态均衡

## 与 GPU 论文 `ACC` 的对应关系

若讨论范围限定为：

- `GQA`
- 单请求
- `group_span=1`

那么当前 CPU `acc-local-l3` 的 locality 单元已经基本和论文里的 `ACC` 对齐。

原因是：

- `GQA` 下外层调度使用的 `actual_kv_head_num` 等于 `num_heads_kv`
- 一个 `kv_head` 对应一组共享同一份 `K/V` 的 query heads
- 当前 `acc-local-l3` 正是按 `kv_head` 把任务限制到某个 `subgroup/CCD`

在这个限定场景里，不应再把问题扩展成“还缺少别的共享对象”或者“必须显式做 `KV` 页迁移”。当前讨论的核心就是：让一个 `ACC` 的共享 `K/V` 由同一个 `CCD/L3` 局部域反复访问。现有 `acc-local-l3` 的基本思路已经是这样。

## `legacy workitem` 与 `ACC` 映射的关系

对单请求 GQA，复用 `legacy workitem` 不等于和 `ACC` 思路冲突。关键要看“全局切分后再映射到 `ACC`”和“先按 `ACC` 再局部切分”在具体 case 里是否只是同构重编号。

例如 `test_results/P3_AttnOnly/Qwen3-30B-A3B/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1/q65536_kv65536/acc-local-l3/debug.log` 这个 case 里，可以直接看到：

- `actual_kv_head_num=8`
- `group_span=1`
- `legacy_effective_thread_num=16`
- 每个 `kv_head` 只覆盖一个 `subgroup`
- 大多数 `subgroup` 的 `covered_thread_num=16`
- 每个 `kv_head` 的 `legacy_slot_pool` 都是同一套 `[0..15]`

这意味着当前 case 基本就是：

- 先用 legacy scheduler 生成 `16` 个逻辑 slot
- 再把每个 `kv_head` 的这 `16` 个 slot 固定交给一个 `CCD/L3 subgroup` 去消费

若此时改成“每个 `ACC` 单独按本地线程池生成 local workitem”，只有在每个 `ACC` 生成出来的 local slot 数与全局投影后的 slot 数不同，或者切分边界不同，才会引入新的变量。若只是换一个入口重新生成同一套 `16` 个 slot，本质上仍是同构重编号，不会天然更接近论文。

因此，不能把“先按 `ACC` 局部生成 workitem”本身当成必然更正确的方向。对这类单请求 GQA case，更重要的是判断它是否真的改变了 task partition，而不是只换了编号方式。

## 当前主线下更合适的受控对照

若目标是相对 `balanced` 证明提升是否来自 `ACC -> CCD/L3` 映射，那么当前做法反而更干净：

- 保持 `workitem_group` 与 reduction item 不变
- 只改 `kv_head -> subgroup/CCD` 映射和运行时领取范围

这样一旦观察到性能、`IPC`、`L3` 指标变化，就更容易把差异归因到：

- 共享 `K/V` 工作集的活动范围是否被收缩到单个 `CCD/L3`
- 映射收缩后局部缓存复用是否变化
- 局部线程池约束是否影响了有效并行度

如果同时重写 workitem 切分，归因就会混入第二个变量：`task partition` 本身是否改变。那时必须用四组对照同时拆分“映射效应”和“切分效应”。

## 什么时候才需要怀疑 `legacy workitem` 不再等价

只有当下面这些条件出现时，“先全局切分再映射”才更可能和“先按 `ACC` 局部切分”真正不同：

- `covered_thread_num` 明显不等于 `legacy_effective_thread_num`
- `group_span > 1`，同一 `kv_head` 由多个 `subgroup` 联合处理
- 开启 split-KV，reduction item 的切分也需要跟随局部线程池重排
- 多请求或请求长度分布不均，global scheduler 的 `kv_len_per_thread` 被全局负载共同决定
- 有意改变 task 粒度或 task 顺序，而不是只改变映射关系

在这些场景里，local workitem 才不再只是同构投影，而会成为独立的优化变量。
