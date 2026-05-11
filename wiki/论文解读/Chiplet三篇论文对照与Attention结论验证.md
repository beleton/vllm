# Chiplet 三篇论文对照与 Attention 结论验证

## 依据版本
- `OLAP on Modern Chiplet-Based Processors`, PVLDB 2024
- `Optimizing Sorting for Chiplet-Based CPUs`, ADMS 2024
- `CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System`, EuroSys 2026

## 问题
三篇 chiplet 架构强相关论文各自优化了什么，选择的应用具有怎样的特征使得优化有效，以及这些特征是否与当前假设「适合 chiplet 优化的场景」一致：
- 场景 1：应用对 L3 大小敏感，此时优化数据局部性、减少冗余可以带来提升
- 场景 2：应用的各线程/进程需要经常跨核心访问，此时可优化进程/线程的放置或计算过程来减少跨 CCD 的访问

## 各论文核心内容

### OLAP on Modern Chiplet-Based Processors
- **优化方法**：提出 WICP (Worker per Chiplet) 部署策略，将查询引擎 worker 以 chiplet 粒度绑定。根据 working set 大小与 L3 容量关系选择 WICP_Local（只用单 chiplet 核心）或 WICP_Mixed（每 chiplet 一个核心）。在 Presto、SingleStore、SparkSQL 上评估 (Fig.7, Fig.8)。
- **应用特征**：OLAP 查询涉及大量表扫描（TableScan）和重分区（Repartition），这些操作的 working set 大小决定了 L3 cache 是否够用。论文观察到 SparkSQL 在 WIM 部署下数据分布不均导致互联拥塞 34% 时间 (Fig.8, §4.2)。WICP 通过强制数据局部性，使 Q13 的 Orders 表扫描从 5.36s 降至 0.4s (§4.2)。
- **最高 speedup**：SparkSQL on Intel Sapphire Rapids 达 6.97x (Fig.7)。

### Optimizing Sorting for Chiplet-Based CPUs
- **优化方法**：四项策略：(1) chiplet 粒度分区输入数据；(2) 将 `in-cache` 阶段拆为 `in-local-chiplet-cache` 和 `in-remote-chiplet-cache`；(3) 根据数据大小相对 L3 容量调度任务；(4) 避免 data shuffling，用 chiplet-aware 线程绑定替代 (§4)。实现了 chiplet-aware LSB Radix-Sort 和 Comparison-Sort。
- **应用特征**：排序是 memory-intensive 操作，核心行为是反复扫描和重新组织数据。排序的 `in-cache` 阶段直接依赖 L3 容量——数据在 L3 内排序 vs 溢出到主存，性能差异巨大。Data shuffling 占排序总时间 20%-32% (Tab.1)，在 chiplet 架构上代价极高。论文说明 L3 cache 的 partitioned 特性改变了排序算法设计的根本假设 (§3.1)。
- **最高 speedup**：Comparison-Sort 达 4.5x，Radix-Sort 达 2x，均对比 NUMA-aware 方案 (Fig.1)。

### CHARM
- **优化方法**：运行时系统，包含 chiplet-aware 任务调度、自适应 cache 分区、coroutine 轻量级并发、PMU 性能监控。去中心化决策：每个 worker 独立监控 remote cache access rate（`ANY_DATA_CACHE_FILLS_FROM_SYSTEM`），超过阈值则 spread 到更多 chiplet（追求更大 cache），否则 consolidate（追求局部性）(Algorithm 1, Algorithm 2)。
- **应用特征**：覆盖图计算（BFS/PageRank/SSSP 等，不规则内存访问）、统计分析（SGD，内存带宽密集）、OLAP（TPC-H on DuckDB）、OLTP（YCSB/TPC-C on ERMIA）(§5.1)。论文说明没有一种固定放置策略适合所有应用——需根据 remote cache access rate 动态调整 (§4.2)。
- **最高 speedup**：SGD 达 3.85x，Graph500 达 2.81x，OLAP (TPC-H) 达 2.34x (Fig.1)。图计算上 CHARM 将 remote chiplet access 降低了 1-3 个数量级 (Tab.1)。

## 三篇论文的共性规律

### 共性 1：L3 局部性 vs 聚合容量的权衡是核心决策维度

三篇论文不约而同地将优化问题建模为同一个权衡：按 chiplet 收紧任务以获得更低访问延迟（Chiplet_Local），还是跨 chiplet 分散任务以获得更大聚合缓存（Chiplet_Mixed）。

- OLAP 论文 Fig.10：数据 < 32MB（单 chiplet L3）时 WICP_Local 带宽更高；超过后 WICP_Mixed 更优
- Sorting 论文 Fig.3 / Fig.13：相同的 Chiplet_Local vs Chiplet_Mixed 边界在 32MB 处出现
- CHARM Algorithm 1：`spread_rate` 的增减直接由 remote cache access rate 驱动，本质是同一决策

### 共性 2：避免跨 chiplet 数据移动是共同优化方向

- OLAP 论文：WICP 强制数据局部性，缓解了 SparkSQL 下 34% 时间的互联拥塞
- Sorting 论文：去掉 data shuffling 是 ablation study 中贡献最大的优化项 (Fig.15)
- CHARM 论文：Tab.1 展示 chiplet-aware 调度将 remote chiplet access 降低 1-3 个数量级

### 共性 3：有效的前提是应用对 L3 cache 敏感

三篇论文优化的应用——OLAP 表扫描、排序 in-cache 阶段、图遍历——都有共同的特性：**working set 大小在 L3 容量边界附近**。能否 fit in L3 直接决定是否会频繁溢出到主存，因此优化数据在 L3 中的布局可以本质性地改变性能。

## 与当前结论的对照

### 对照场景 1：「应用对 L3 大小敏感」

**三篇论文高度一致地支持此结论。**

- Sorting 论文最直接：将 `in-cache` 拆为 chiplet-local 和 chiplet-remote 两个阶段，因为排序性能在「数据是否 fit in L3」这个边界上发生阶跃变化 (§3.1, Fig.13)
- OLAP 论文 Fig.10：32MB（单 chiplet L3）是带宽曲线的拐点
- CHARM Algorithm 1：`spread_rate` 调整以 remote cache access rate 为信号，隐含假设是「避免 L3 miss 带来的 main memory access」是性能关键

CPU Attention 不符合此条件的证据：Attention 的 working set 中，head 维度在 L2 内完成复用，sequence 维度是 streaming access 不依赖 L3 复用。限制 L3 可用大小对性能影响极小（见 [2026-05-06 Qwen3-30B-A3B CPU Attention ACC 局部性优化结论](../实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md) 第 5.2 节），说明 Attention 不满足 L3 敏感性前提。

### 对照场景 2：「线程/进程需要经常跨核心访问」

**三篇论文也支持此结论，但有重要限定条件。**

- OLAP 论文 Fig.7/Fig.8：interconnect 拥塞是 WIM 部署下性能降级的主因
- Sorting 论文 Tab.1：shuffling 占排序时间 20-32%，是跨 chiplet 访问的主要来源
- CHARM 论文 Tab.1：图计算中 NUMA-aware 调度产生大量 remote chiplet access

**但限定在于**：减少跨 CCD 访问如果不同时改善 L3 cache 局部性，效果可能有限。更根本的是，当前 attention 中跨 CCD 访问本身就不是主要瓶颈——PCM 数据（见 [2026-05-06 结论](../实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md) 第 5.3 节）显示 q2048+ 区间跨 CCD demand fill（another CCX same node）仅占 0.14%–0.81%，主体 demand fill 来自本地 L2。这意味着即便把跨 CCD 通信降到零，能换回的收益也被上限锁死在很小的范围。这与三篇论文中 workload 的情况不同——OLAP 的 WIM 部署下互联拥塞占 34% 执行时间，排序的 shuffling 占 15%–32%，图计算 NUMA-aware 的 remote chiplet access 达百万次量级。在这些 workload 中跨 chiplet 通信本身就是重要瓶颈，降低它能直接换回显著收益；在 attention 中这个前提不成立。

### Attention 与三篇论文应用的对比

| 特征 | OLAP/Sorting/Graph | CPU Attention |
|------|-------------------|---------------|
| Working set 与 L3 关系 | Working set 在 L3 容量边界附近 | 要么 fit in L2（head 维度），要么 streaming 无复用（seq 维度） |
| 数据复用模式 | 同一数据被多次访问（排序多轮 pass、OLAP 扫描后 repartition/join 重读、图遍历反复触碰邻居）。数据以**只读共享**为主——多 chiplet 同时读同一份数据，coherence 协议在各 chiplet L3 中产生副本 | K/V 在 sequence 维度 streaming access，每个元素只读一次。虽然 K/V 也是只读共享，但无复用意味着 coherence 副本扩散的代价没有被放大 |
| 副本扩散的后果 | 多 chiplet L3 中同时驻留同一数据的副本，挤占本地 L3 容量；跨 chiplet 的 coherence 消息（snoop/probe/64B cache line 搬运）占用 Infinity Fabric 带宽。**Chiplet 放置通过限制"谁读哪份数据"来消除副本扩散** | 副本虽存在但影响小——每个 cache line 被读一次后就成为冷数据，不占用后续访问的 L3 空间，coherence 消息量也仅与单次读取量成正比 |
| L3 容量敏感度 | 高——复用期间数据必须留在 L3，容量不够 → eviction → 下次访问走 DRAM（120 ns）。L3 容量直接决定复用访问的命中率 | 低——无复用意味着 L3 只是数据从 DRAM 到 L2 的通道。CAT 0001 将 L3 容量压到 1/16，q8192+ 只退化 0.03%–2.11% [2026-05-06 结论文档 5.2] |
| 优化杠杆点 | 数据布局 + 任务放置 → 控制跨 chiplet 的 cache coherence 流量与复用时的 L3 命中率 | 内存带宽/计算吞吐 → L3 命中率和 coherence 流量不是瓶颈 |

## 分析

从三篇论文可直接提炼的事实是：**chiplet 级任务/数据放置要有效，必须满足一个条件——应用的性能瓶颈出在"跨 chiplet 的数据访问"上，且这种瓶颈能通过改变放置来消除。** 在三篇论文的 workload 中，"跨 chiplet 数据访问"具体体现为两种形式：

1. **同一份只读数据被多个 chiplet 上的线程反复访问**（OLAP 的表扫描被后续 repartition/join 重读、排序的多轮 pass、图遍历的邻居反复触碰）。硬件 cache coherence 把这份数据在各 chiplet L3 中复制多份，产生副本扩散。副本占用 L3 容量，coherence 协议的消息（64 字节 cache line 的 snoop/probe/搬运）占用 Infinity Fabric 带宽。若每次复用都走本地 L3（24 ns）而非跨 chiplet（~106 ns）或 DRAM（~120 ns），延迟差距被复用次数放大。**Chiplet 放置通过限制"谁读哪份数据"消除副本扩散，让复用访问稳定走本地 L3。**

2. **数据在不同 chiplet 之间主动搬迁**（排序的 shuffling、OLAP 的 WIM 下数据跨 chiplet 不均分配导致 join 阶段互联拥塞）。搬迁后的数据在原 chiplet L3 中的副本失效，后续访问全部变成跨 chiplet 或 DRAM 访问。

这两种形式的共同点是：**数据的访问路径（走本地 L3、跨 chiplet L3、还是 DRAM）取决于数据被哪个 chiplet 上的线程访问，而这正是 chiplet 放置可以控制的。**

CPU Attention 既不满足形式 1（K/V streaming access 无复用，coherence 副本扩散的代价不被放大），也不满足形式 2（没有主动的数据搬迁步骤）。因此 chiplet 放置——即 acc-local-l3 做的事——能改变 L3 中的驻留形态（L3 occupancy 降了 23%–32%，L3 Miss/task 降了 17%–53%），但改不动性能，因为在关键路径上本来就不是跨 chiplet 访问在瓶颈。

## 对寻找新研究方向的启示

三篇论文对当前课题的价值不仅是"验证 attention 不走 chiplet 局部性这条路"，更在于它们提供了一套**筛选 chiplet 优化适用场景的方法**：

1. **先确认瓶颈在哪里**：用 PCM / PMU 计数器测量目标阶段（如 prefill、decode、KV cache 管理等）的 demand fill 来源分布。只有当跨 chiplet 访问（another CCX same node）或远端 DRAM 访问（remote DRAM）占到有意义的份额时，chiplet 级放置才有杠杆可撬。三篇论文各自 workload 的全流程执行中跨 chiplet 通信占比都在 15%–34%，而 attention 仅 0.14%–0.81%。

2. **再确认瓶颈能否被放置改变**：即使跨 chiplet 访问占比高，还要看这些访问是否来自"数据和处理它的线程不在同一个 chiplet"。如果是（如 OLAP WIM 部署），改变放置就能消除；如果不是（如数据本身在远端 socket 的 DRAM 中，且无法搬迁），放置改变不了。

3. **用 CAT 实验验证 L3 敏感性**：三篇论文都直接或间接依赖"working set 在 L3 容量边界附近"这一前提。CHARM 用 PMU 事件率来检测这一边界，排序论文用 STREAM benchmark 来标定。当前课题已有的 CAT 0001 实验（[2026-05-06 结论](../实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md) 第 5.2 节）提供了同等效力的验证手段——如果极端限制 L3 容量几乎不影响性能，说明该阶段的 working set 不走 L3 复用路径，chiplet 局部性优化就不对症。

这三条筛选条件可以直接用于扫描 LLM 推理中 attention 以外的其他环节。具体而言：

- **Decode**：PD_Test 已确认 decode 的 L3 Miss% 远高于 prefill（96.51% vs 49.43%），但 CAT 实验若也显示 decode 对 L3 容量不敏感，则说明 decode 同理不走 L3 复用路径。
- **KV cache 管理 / 分布式 prefill 的数据交换**：这些环节涉及跨 rank 或跨 chiplet 的数据搬迁。如果搬迁的 volume 和频率使得跨 chiplet 通信成为可观测的瓶颈，chiplet-aware 放置可能适用。
- **Whole-model serving 全链路**：attention-only 和 PD_Test 的口径只覆盖单 kernel 或单 request。全链路中可能存在其他产生跨 chiplet 通信的环节（线程池调度、内存分配、batch 拼装等），需要 full-stack profiling 才能识别。

总的原则是：**先找到跨 chiplet 数据访问确实在关键路径上的环节，再考虑 chiplet 级放置优化。** 这条路在 attention 上行不通不意味着在 LLM 推理的其他环节也行不通。

## 边界

- 三篇论文均在 AMD EPYC Milan / Intel Sapphire Rapids 上实验，未覆盖 ARM Graviton 3 之外的其他 ARM 平台。其中 ARM Graviton 3 因其单计算 chiplet 设计，chiplet-aware 优化效果显著弱于 AMD/Intel（OLAP 论文 Fig.7，CHARM §5.1）。
- 三篇论文评估的 workload（OLAP 查询、排序、图计算、SGD）与 LLM inference 的 Attention kernel 在工作集特征和数据复用模式上有本质差异，不可直接外推方法。
- 三篇论文的优化有效性都建立在一个共同前提上：应用的 working set 大小与 L3 cache 容量处于可比较的量级。当前 Attention kernel 不满足此前提，因此同类方法不能直接迁移。

## 可迁移点

- 「按 chiplet 粒度分配任务以减少跨 chiplet 数据移动」作为通用设计原则，可应用于 LLM inference 中满足 L3 敏感条件的其他环节
- CHARM 的去中心化自适应调度思路（每个 worker 独立根据 PMU 计数器决策）可作为 runtime 优化的设计参考
- 「先通过 micro-benchmark 验证 L3 敏感性再投入优化」的方法论可直接采纳

## 不可直接迁移点

- 三篇论文的 chiplet-aware 调度策略均依赖「working set 可被分区且各分区独立计算」，Attention 的 K/V 被所有 head/query 共享，不具备此特性
- 论文中的 WICP 部署策略需要修改引擎/框架的调度层，在 vllm 的 CPU backend 中对应层面不同
