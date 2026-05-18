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
- **优化方法**：提出 WICP (Worker per Chiplet) 部署策略—每 chiplet 一个独立 worker 进程，线程只绑定本 chiplet 核心，各 worker 数据不跨 chiplet 共享地址空间。根据单 worker 的工作集大小与 L3 容量关系进一步选择线程放置：工作集 < 单 chiplet L3（32 MB）时 worker 线程全部绑在本 chiplet 核心（WICP_Local）；工作集超出时 worker 的线程分散到多个 chiplet 以换取更大聚合缓存（WICP_Mixed）。在 Presto、SingleStore、SparkSQL 上评估 (Fig.7, Fig.8)。
- **应用特征**：OLAP 查询涉及大量表扫描（TableScan）和重分区（Repartition）。WIM 下多 chiplet 线程在同一地址空间交叉访问共享内存，产生大量跨 chiplet cache coherence 流量，互联拥塞占 SparkSQL 执行时间 34% (Fig.8, §4.2)。WICP 用独立进程地址空间切断 coherence 路径——Q13 的 Orders 表扫描从 5.36s 降至 0.4s (§4.2)。此外 worker 内 working set < 单 chiplet L3 时 WICP_Local 可叠加 L3 局部性收益。
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

三篇论文的优化问题可归结为同一权衡：按 chiplet 收紧任务获更低访问延迟（Chiplet_Local），还是跨 chiplet 分散任务获更大聚合缓存（Chiplet_Mixed）。

- OLAP 论文 Fig.10：数据 < 32MB（单 chiplet L3）时 WICP_Local 带宽更高；超过后 WICP_Mixed 更优
- Sorting 论文 Fig.3 / Fig.13：相同的 Chiplet_Local vs Chiplet_Mixed 边界在 32MB 处出现
- CHARM Algorithm 1：`spread_rate` 的增减直接由 remote cache access rate 驱动，本质是同一决策

### 共性 2：避免跨 chiplet 数据移动是共同优化方向

- OLAP 论文：WICP 强制数据局部性，缓解了 SparkSQL 下 34% 时间的互联拥塞
- Sorting 论文：去掉 data shuffling 是 ablation study 中贡献最大的优化项 (Fig.15)
- CHARM 论文：Tab.1 展示 chiplet-aware 调度将 remote chiplet access 降低 1-3 个数量级

### 共性 3：跨 chiplet 数据访问开销在关键路径上占比显著

三种 workload 前提相同：存在显著的跨 chiplet 数据访问开销，可通过改变 chiplet 级放置来消除。形式：

- **OLAP**：WIM 下多 chiplet 线程在同一地址空间交叉访问共享内存，触发硬件 cache coherence 风暴。WICP 用独立进程地址空间切断 coherence 路径。coherence 消除在所有数据规模下有效。
- **Sorting**：NUMA-aware 排序的 shuffling 步骤跨 chiplet 主动搬迁数据，搬走后数据在原 chiplet L3 中失效，后续 pass 全部变成远端访问。chiplet-aware 排序直接去掉 shuffling。
- **CHARM**：图计算中 NUMA-aware 调度产生百万次量级的 remote chiplet cache access（Tab.1），`spread_rate` 的去中心化调整将其降低 1–3 个数量级。

**"working set 在 L3 容量边界附近"不是 chiplet 优化的必要条件。** OLAP 的 coherence 消除和 Sorting 的 shuffling 消除在 L3 边界之外仍然有效。L3 容量边界只决定 WICP_Local vs WICP_Mixed 的二阶选择——是否能在消除跨 chiplet 开销的基础上，再通过"命中本地 L3 vs 聚合 L3 vs DRAM"的延迟差异获取额外收益。判断 workload 是否值得做 chiplet 优化，主条件是跨 chiplet 数据访问开销在关键路径上的占比，而非 working set 是否在 L3 边界。

## 与当前结论的对照

### 对照场景 1：「应用对 L3 大小敏感」

**L3 敏感是 Chiplet_Local vs Chiplet_Mixed 的选择条件，不是 chiplet 优化的总前提。** OLAP 的 coherence 消除和 Sorting 的 shuffling 消除在 L3 边界外仍有效；L3 敏感只决定能否叠加"本地 L3 命中"的额外收益。

- Sorting：将 `in-cache` 拆为 chiplet-local 和 chiplet-remote 两个阶段，pass 间复用数据量在 L3 边界附近时，Local vs Mixed 的选择决定 L3 命中率 (§3.1, Fig.13)
- OLAP 论文 Fig.10：32 MB 是 WICP_Local vs WICP_Mixed 的分界线；WICP 的 coherence 消除在 > 256 MB 后仍然存在
- CHARM Algorithm 1：`spread_rate` 以远程 cache fill 事件率为信号，检测当前工作集是否超出本地 chiplet L3

CPU Attention 不满足 L3 敏感性：tile 复用发生在 L2 内（~500 KB–1 MB），L3 是数据从 DRAM 到 L2 的通道。CAT 0001 将 L3 压到 1/16 仅退化 0.03%–2.11%（[2026-05-06 结论](../实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md) 第 5.2 节）。此外 attention 也通不过场景 2——跨 CCD 通信占比极低。

### 对照场景 2：「线程/进程需要经常跨核心访问」

**但 attention 的跨 CCD 通信占比远低于三篇论文的 workload。**

- OLAP 论文 Fig.7/Fig.8：WIM 下互联拥塞占执行时间 34%
- Sorting 论文 Tab.1：shuffling 占排序时间 20%–32%
- CHARM 论文 Tab.1：NUMA-aware 调度产生百万次量级 remote chiplet access
- Attention：PCM 数据（[2026-05-06 结论](../实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md) 第 5.3 节）显示 q2048+ 区间跨 CCD demand fill（another CCX same node）仅 0.14%–0.81%，主体 demand fill 来自本地 L2。跨 CCD 通信降到零能换回的收益上限极低。

### Attention 与三篇论文应用的对比

| 特征 | OLAP/Sorting/Graph | CPU Attention |
|------|-------------------|---------------|
| Working set 与 L3 关系 | Working set 在 L3 容量边界附近 | 要么 fit in L2（head 维度），要么 streaming 无复用（seq 维度） |
| 数据复用模式 | 同一数据被多次访问（排序多轮 pass、OLAP 扫描后 repartition/join 重读、图遍历反复触碰邻居）。数据以**只读共享**为主——多 chiplet 同时读同一份数据，coherence 协议在各 chiplet L3 中产生副本 | K/V 在 sequence 维度 streaming access，每个元素只读一次。虽然 K/V 也是只读共享，但无复用意味着 coherence 副本扩散的代价没有被放大 |
| 副本扩散的后果 | 多 chiplet L3 中同时驻留同一数据的副本，挤占本地 L3 容量；跨 chiplet 的 coherence 消息（snoop/probe/64B cache line 搬运）占用 Infinity Fabric 带宽。**Chiplet 放置通过限制"谁读哪份数据"来消除副本扩散** | 副本虽存在但影响小——每个 cache line 被读一次后就成为冷数据，不占用后续访问的 L3 空间，coherence 消息量也仅与单次读取量成正比 |
| L3 容量敏感度 | 高——复用期间数据必须留在 L3，容量不够 → eviction → 下次访问走 DRAM（120 ns）。L3 容量直接决定复用访问的命中率 | 低——无复用意味着 L3 只是数据从 DRAM 到 L2 的通道。CAT 0001 将 L3 容量压到 1/16，q8192+ 只退化 0.03%–2.11% [2026-05-06 结论文档 5.2] |
| 优化杠杆点 | 数据布局 + 任务放置 → 控制跨 chiplet 的 cache coherence 流量与复用时的 L3 命中率 | 内存带宽/计算吞吐 → L3 命中率和 coherence 流量不是瓶颈 |

## 分析

从三篇论文可得：**chiplet 级任务/数据放置要有效，应用的性能瓶颈必须出在"跨 chiplet 的数据访问"上，且能通过改变放置来消除。** 跨 chiplet 数据访问在三种 workload 中有两种形式：

1. **同一份只读数据被多个 chiplet 上的线程反复访问**（OLAP 的表扫描被后续 repartition/join 重读、排序的多轮 pass、图遍历的邻居反复触碰）。硬件 cache coherence 把这份数据在各 chiplet L3 中复制多份，产生副本扩散。副本占用 L3 容量，coherence 协议的消息（64 字节 cache line 的 snoop/probe/搬运）占用 Infinity Fabric 带宽。若每次复用都走本地 L3（24 ns）而非跨 chiplet（~106 ns）或 DRAM（~120 ns），延迟差距被复用次数放大。**Chiplet 放置通过限制"谁读哪份数据"消除副本扩散，让复用访问稳定走本地 L3。**

2. **数据在不同 chiplet 之间主动搬迁**（排序的 shuffling、OLAP 的 WIM 下数据跨 chiplet 不均分配导致 join 阶段互联拥塞）。搬迁后的数据在原 chiplet L3 中的副本失效，后续访问全部变成跨 chiplet 或 DRAM 访问。

两种形式的共同点：**数据走本地 L3、跨 chiplet L3 还是 DRAM，取决于数据被哪个 chiplet 上的线程访问。chiplet 放置可以控制这一点。**

CPU Attention 既不满足形式 1（K/V streaming access 无复用，coherence 副本扩散的代价不被放大），也不满足形式 2（没有主动的数据搬迁步骤）。因此 chiplet 放置——即 acc-local-l3 做的事——能改变 L3 中的驻留形态（L3 occupancy 降了 23%–32%，L3 Miss/task 降了 17%–53%），但改不动性能，因为在关键路径上本来就不是跨 chiplet 访问在瓶颈。

## 对寻找新研究方向的启示

三篇论文提供了一套筛选 chiplet 优化适用场景的方法：

1. **确认瓶颈位置**：用 PCM / PMU 计数器测量目标阶段的 demand fill 来源分布。跨 chiplet 访问（another CCX same node）或远端 DRAM 访问（remote DRAM）占比有意义的份额时，chiplet 级放置才有杠杆。三篇论文的 workload 中跨 chiplet 通信占 15%–34%，attention 仅 0.14%–0.81%。

2. **确认瓶颈能否被放置改变**：跨 chiplet 访问占比高时，还需确认这些访问来自"数据和处理它的线程不在同一 chiplet"。如果是（OLAP WIM 部署），改变放置可消除；如果不是（数据在远端 socket DRAM 且无法搬迁），放置无效。

3. **CAT 实验验证 L3 敏感性**：CHARM 用 PMU 事件率检测工作集是否超出本地 L3，排序用 STREAM benchmark 标定 L3 边界。CAT 0001（[2026-05-06 结论](../实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md) 第 5.2 节）提供了等效验证——极端限制 L3 容量几乎不影响性能，说明 working set 不走 L3 复用路径。

用这三条扫描 LLM 推理中 attention 以外的环节：

- **Decode**：L3 Miss% 远高于 prefill（96.51% vs 49.43%），但 CAT 实验若也显示对 L3 容量不敏感，则 decode 同理不走 L3 复用路径。
- **KV cache 管理 / 分布式 prefill 的数据交换**：涉及跨 rank 或跨 chiplet 的数据搬迁。搬迁 volume 和频率使跨 chiplet 通信成为可观测瓶颈时，chiplet-aware 放置可能适用。
- **Whole-model serving 全链路**：attention-only 和 PD_Test 只覆盖单 kernel 或单 request。全链路中（线程池调度、内存分配、batch 拼装等）可能另有跨 chiplet 通信的环节，需 full-stack profiling。

**先找到跨 chiplet 数据访问在关键路径上的环节，再考虑 chiplet 级放置优化。** Attention 上行不通不代表其他环节不行。

## 并行度与工作集的共同约束

三篇论文没有把 Chiplet_Local vs Chiplet_Mixed 简化为"只看工作集是否小于单 chiplet L3"。工作集大小决定本地 L3 与聚合 L3 的取舍；所需线程数、worker 数或任务并行度决定单个 chiplet 是否有足够核心承载任务。

- Sorting 论文在方法部分明确给出双条件：输入可放入单 chiplet L3 且所需线程数不超过单 chiplet 核数时使用 Chiplet_Local；线程数超过单 chiplet 核数，或输入超过单 chiplet L3 但仍可放入聚合 L3 时使用 Chiplet_Mixed。Fig. 13 固定 8 threads，只验证数据规模对 Local/Mixed 的影响；Fig. 8/Fig. 10 另行验证核心数扩展。论文没有给出"数据规模 × 核心数"完整二维矩阵。
- OLAP 论文的 WICP_Local 与 WICP_Mixed 使用相同数量的 core，只改变 core 在 chiplet 内集中还是跨 chiplet 分散。WICP 的主实验不是减少全机核心使用，而是每 chiplet 启动 worker，使整机多个 chiplet 同时处理不同数据 fragment。worker 数敏感性实验显示最优 worker 数接近 chiplet 数。
- CHARM 在 `UpdateLocation` 中显式检查 `spread_rate` 覆盖的物理核心数是否足够容纳全部 worker thread；若不足，跳过 remap。其调度规则不是盲目压到单 chiplet，而是在 remote cache fill 事件率和可用核心数约束下调整扩散范围。

因此，三篇论文覆盖了"工作集大小 + 所需并行度"这一基本决策关系，但实验主要针对 memory/cache-sensitive workload。未充分覆盖的是系统性的二维实验：在同一 workload 上同时扫数据规模与核心数，并进一步区分 memory-bound 与 compute-bound 阶段。

对小工作集、计算密集型任务的判据应先看所需并行度：若单 chiplet 核心数不足以提供目标计算吞吐，即使工作集可放入单 chiplet L3，也需要扩到更多 chiplet；此时 L3 locality 是次级约束。三篇论文的实测收益不能直接外推到这类 compute-bound kernel，因为它们的主要收益来自降低跨 chiplet 数据访问、提高 L3/内存局部性或减少数据搬迁。

## 边界

- 三篇论文均在 AMD EPYC Milan / Intel Sapphire Rapids 上实验，未覆盖 ARM Graviton 3 之外的其他 ARM 平台。其中 ARM Graviton 3 因其单计算 chiplet 设计，chiplet-aware 优化效果显著弱于 AMD/Intel（OLAP 论文 Fig.7，CHARM §5.1）。
- 三篇论文评估的 workload（OLAP 查询、排序、图计算、SGD）与 LLM inference 的 Attention kernel 在工作集特征和数据复用模式上有本质差异，不可直接外推方法。
- 三篇论文的方法均依赖"跨 chiplet 数据访问开销显著"这一前提。OLAP 靠 coherence 消除（所有数据规模有效），Sorting 靠 shuffling 消除 + L3 局部性，CHARM 靠去中心化 spread_rate 调整。Attention 上跨 CCD 通信占比（0.14%–0.81%）不足以构成可优化的瓶颈。

## 可迁移点

- 按 chiplet 粒度分配任务以减少跨 chiplet 数据移动，可用于 LLM inference 中跨 chiplet 通信占比显著的环节
- CHARM 的去中心化自适应调度（每个 worker 独立根据 PMU 计数器决策）
- 先通过 micro-benchmark 验证瓶颈层级再投入优化

## 不可直接迁移点

- 三篇论文的 chiplet-aware 调度依赖 working set 可被分区且各分区独立计算。Attention 的 K/V 被所有 head/query 共享，不具备此特性。
- WICP 部署策略在 vllm CPU backend 中无直接对应层面。
