# CHARM 解读

## 说明
- 依据版本：EuroSys 2026 最终中稿版（`CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System`）。
- 以下内容均以本地 PDF 为准，未对照 arXiv 或其他版本。

## 问题
Chiplet CPU 在同一 NUMA domain 内仍存在不同的 L3 访问延迟和核间通信代价——AMD EPYC Milan 上同 NUMA domain 内延迟分三组：约 25 ns（intra-chiplet）、约 80–90 ns（同 NUMA 域内中间层）、超过 150 ns（更远层级）（Fig. 3）。只按 NUMA node 建模的调度与内存放置策略无法覆盖 chiplet 级延迟差异。CHARM 要回答的是：能否构建一个运行时系统，使各类并行应用自动在"chiplet 局部性"与"更大聚合 L3 容量"之间做出自适应选择。

## 背景：CHARM 覆盖的四类 Workload 及其访存特征

CHARM 覆盖四类场景（Sec. 5.1），每类的访存特征不同，对 chiplet-aware 调度的敏感点也不同：

### 图计算（BFS / PageRank / CC / SSSP / Graph500）
- **计算流程**：基于 frontier 的迭代遍历。每轮从当前活跃节点集（active frontier）出发，访问其邻居，更新邻居状态，将新激活的节点加入下一轮 frontier，直到收敛或遍历完成。
- **访存特征**：不规则内存访问（irregular access patterns）。任务是按 active frontier node 动态生成的，每个 frontier node 的处理会访问图中任意位置的邻居数据。邻居数据可能散布在所有 chiplet 的内存和 cache 中。
- **与 chiplet 架构的冲突**：每次访问邻居节点都是随机地址跳转，cache line 的局部性很差。如果一个 chiplet 上的线程要访问另一个 chiplet 主存中的邻居数据，不仅产生远端主存访问（~120 ns），还会在本地 L3 中留下仅使用一次的 cache line，挤占本地 L3 容量。
- **论文选择原因**：chiplet-aware task partitioning 对不规则内存访问 workload 尤其有效，用它们验证 CHARM 的主要收益场景。

### 随机访问（GUPS, Giga Updates Per Second）
- **计算流程**：对一个大数组（通常远大于 L3 容量）做随机位置的读-修改-写。每个线程独立生成随机索引，读取该位置的值，做一次更新，写回。
- **访存特征**：完全非连续内存访问（non-contiguous memory accesses）。每次访问的地址是随机的，几乎每次都是 cache miss。性能完全取决于内存系统的随机访问延迟和带宽。
- **与 chiplet 架构的冲突**：随机地址落在哪个 chiplet 的内存控制器管辖范围内决定了访问延迟。若线程绑在 chiplet 0 但随机地址落在 chiplet 3 的内存区域，每次访问都是远端 NUMA + 跨 chiplet。
- **论文选择原因**：GUPS 直接考察非连续访问带来的数据移动和缓存/内存访问代价，适合观察 chiplet-aware 放置对吞吐和扩展性的影响。

### 统计分析（DimmWitted + SGD）
- **计算流程**：逻辑回归的随机梯度下降（SGD）。每轮 epoch 遍历全部训练数据（约 6250 MB），对每个样本计算梯度并更新模型参数。DimmWitted 框架把 workload 切成数百个细粒度 chunk，跨 600+ 线程执行。
- **访存特征**：每个 epoch 完整触碰整份 6 GB 输入。工作集 = 全部训练数据 + 模型参数，超过单 socket 聚合 L3（256 MB）。计算过程是 memory-bandwidth-intensive——梯度计算只需一次乘加，数据搬运量远超计算量。同时，DimmWitted 的细粒度 chunk 机制产生了大量线程创建/切换开销（32 cores 下创建了 641 个线程）。
- **与 chiplet 架构的冲突**：训练数据散布在多个 chiplet 的内存中，600+ 线程竞争 chiplet L3 和内存带宽。线程创建/切换开销（OS 级 context switch）进一步消耗了本可用于数据搬运的 CPU 时间。
- **论文选择原因**：检验 chiplet-aware 放置 + coroutine 减少线程切换开销的组合效果，同时测试工作集远超聚合 L3 时的收益边界。

### OLAP（DuckDB + TPC-H SF100）
- **计算流程**：同 OLAP 论文的背景，Coordinator 解析 SQL → 生成执行计划 → Worker 在本地数据 fragment 上执行 scan/filter/join/aggregate。
- **访存特征**：两类查询混合：(1) 大表 hash join / inner join（Q3/Q4/Q5/Q7/Q9/Q10），build 端和 probe 端的数据量在 GB 级，工作集远大于 L3；(2) 小工作集 scan/filter 查询，处理 2–4 MB 数据，运行时间不到 1s。
- **与 chiplet 架构的冲突**：大 join 查询需要更大的聚合 L3 来缓存 build 端的 hash table，小查询则更适合收紧在少数 chiplet 上以避免 cache coherence 开销。
- **论文选择原因**：验证 CHARM 能否随 query type 和 working set 大小自动在"跨 chiplet 扩大聚合 L3"和"收紧到少数 chiplet 保持 locality"之间切换。

### OLTP（ERMIA + YCSB / TPC-C）
- **计算流程**：短事务处理。YCSB：单表，45% read + 55% read-modify-write。TPC-C：更复杂的事务组合，含跨分区（cross-partition）访问。事务执行路径：读 → 修改 → 提交（commit log 写入）。
- **访存特征**：短事务、频繁提交和同步。事务本身的数据访问量很小（几条到几十条记录），但 commit 的同步开销（日志刷盘、锁、latch）主导了执行时间。
- **与 chiplet 架构的冲突**：数据访问量极小，L3 容量不构成瓶颈。跨 chiplet 访问即使存在，绝对时间也被事务同步开销淹没。
- **论文选择原因**：测试 chiplet-aware 调度的适用边界——当瓶颈是同步/提交而非 cache 时，CHARM 的策略是否还重要。

## 观察：局部性与聚合缓存容量的 Trade-off 是普遍存在的

论文在单路 8-chiplet AMD EPYC Milan 上做 8-thread 向量写微基准（Fig. 5）：
- **LocalCache**：线程和数据限制在一个 chiplet（可用 32 MB L3）
- **DistributedCache**：线程分散到多个 chiplet（可用 256 MB 聚合 L3）

结果：
- 数据 < 38 MB 时，LocalCache 执行时间更低（本地 L3 延迟优势）
- 超过 32 MB 后，DistributedCache 开始更有效（聚合容量优势）
- 38 GB 数据规模上，DistributedCache 相对 LocalCache 加速最高 2.5x

论文由此确立核心判断：不是"永远收紧 locality"或"永远铺开到更多 chiplet"，而是要按工作集动态切换。这一判断与 OLAP 论文的 WICP_Local vs WICP_Mixed 和排序论文的 Chiplet_Local vs Chiplet_Mixed 在逻辑上一致。

## 方法：CHARM 运行时系统设计

### 整体架构（Fig. 6）
四个组件：
- **Performance Profiler**：持续监控 cache/memory 行为（直接读 PMU 事件计数器）
- **Adaptive Controller**：根据 profiler 结果生成调度策略
- **Task and Memory Manager**：管理 coroutine、任务队列和内存放置
- **Global Scheduler**：协调任务分发和迁移

### 核心调度机制：去中心化的 spread_rate 调整（Algorithm 1, Sec. 4.2）

CHARM 给每个 worker thread 分配一个独立 physical core，作为最小独立调度单位。每个 worker 周期性（`SCHEDULER_TIMER = 500 ms`）读取自己的 cache fill 事件计数器。

**阈值驱动的决策**：若 remote cache fill 事件率超过阈值（`RMT_CHIP_ACCESS_RATE = 300`），说明当前工作集已超出本地 chiplet L3 容量，频繁产生远端访问，worker 增大 `spread_rate`，将任务分散到更多 chiplet 以换取更大的聚合缓存容量。若低于阈值，减小 `spread_rate`，将任务收紧到更少 chiplet 以增强局部性。

**去中心化的含义**：不是先全局收集所有 worker 的统计数据再统一调度，而是每个 worker 根据本地 PMU 观测独立触发调整。论文认为这降低了同步开销，使系统能更快响应各 worker 的局部访存压力变化。

**与 workload 特征的耦合关系**：
- 图计算（不规则访问）：`spread_rate` 的变化直接影响 frontier node 数据和邻居数据在哪个 chiplet 的 L3 中，跨 chiplet remote cache fill 的减少直接转化为遍历吞吐提升。
- SGD（大工作集）：工作集 6 GB 远超聚合 L3（256 MB），`spread_rate` 的主要作用是平衡各 chiplet 上的本地主存访问压力，减少远端 NUMA 内存访问。
- OLAP（混合大小查询）：大 join 查询自动推高 `spread_rate`（需要更大聚合缓存），小 scan/filter 查询自动降低（精简单 chiplet 即可），实现查询粒度的自适应。
- OLTP（短事务）：事务数据量极小，remote cache fill 事件率始终很低，CHARM 保持低 `spread_rate`，不触发不必要的任务扩散。这也解释了为什么 CHARM 在 OLTP 上无收益。

### 核心映射机制：UpdateLocation（Algorithm 2）

CHARM 给每个 worker 分配唯一 ID，再据此计算唯一的 `(chiplet, slot, core)` 三元组：
- `chiplet = floor(unique_worker_ID / (CORES_PER_CHIPLET / spread_rate))`
- `slot = unique_worker_ID mod (CORES_PER_CHIPLET / spread_rate)`
- `core = chiplet × CORES_PER_CHIPLET + slot`
- 若 `chiplet >= CHIPLETS`，做回卷修正：`chiplet = chiplet mod CHIPLETS`，`slot = slot + floor(unique_worker_ID / CORES_PER_CHIPLET)`

**前提**：1 个 worker thread 独占 1 个 physical core。只有当当前 `spread_rate` 覆盖的物理核心数不少于 worker thread 数时，映射才有效。若核心数不够，跳过此次 remap，保持原有 affinity。

同时设置 `thread affinity` 和 `set_mempolicy(MPOL_BIND, ...)`，让任务落点和主存分配一起跟着拓扑走，尽量压过 OS 的默认 NUMA balancing。

### Coroutine 任务模型（Sec. 4.3）

CHARM 不依赖大量 OS 线程来支持细粒度并发。每个任务有自己的栈、状态和调度器，可以在开发者定义的位置 `yield`，运行时结合 profiling 结果决定是否迁移。每个 core 上维护本地低开销任务队列；若本地队列为空，优先从同 chiplet 其他 core 偷任务，再考虑别的 chiplet。

**与 workload 特征的关联**：DimmWitted + SGD 实验直接展示了效果——32 cores 下 DimmWitted 创建了 641 个 OS 线程，CHARM 只用了 34 个（Fig. 12）。OS 线程的创建/切换开销被 coroutine 的用户态切换替代。对于 DimmWitted 这种把 workload 切成数百个细粒度 chunk 的框架，减少线程切换开销的收益与 chiplet-aware 放置的收益是加性的。

### Profiling 机制（Sec. 4.2, Sec. 4.4）

CHARM 用 `libpfm` 直接读 PMU 事件计数器：
- AMD：`ANY_DATA_CACHE_FILLS_FROM_SYSTEM`（对应 AMDuProfPcm 的 `All DC Fills (pti)`）
- Intel：`OFFCORE_RESPONSE`

高频轮询时 profiling 开销约 5%–10%，轮询频率可调。论文将这些计数器的来源层次概括为 on-chip（intra-CCX）、on-die（inter-CCX）和 remote memory（inter-NUMA）。

## 实验设置

### 平台
- AMD：双路 EPYC Milan 7713，每 socket 64 cores、512 GB RAM、8 chiplets，每 chiplet 32 MB L3
- Intel：双路 Xeon Platinum 8488C，每 socket 48 cores、512 GB RAM，共享 105 MB L3

### Baseline（Sec. 5.1）
总 baseline 四个，各子实验对照的子集不同：
- **RING**：NUMA-aware message-batching runtime system，面向高性能内存内数据密集型工作负载。图计算和 GUPS 的主要对比对象。
- **SHOAL**：NUMA multi-core runtime，提供数组抽象优化内存分配与访问模式。Streamcluster 的对比对象。
- **AsymSched**：bandwidth-centric NUMA scheduler，在不对称互连下优化线程与内存放置。
- **SAM**：面向 multi-programmed 多核机器的 CPU scheduler，通过 latency tolerance 和 hyperthreading awareness 缓解数据共享与竞争。

### 子实验对照（Sec. 5.2–5.7）
- 图计算 + GUPS：对比 RING / AsymSched / SAM
- Streamcluster：对比 SHOAL
- DimmWitted + SGD：额外用 DW-per-core、DW-NUMA-node、DW-per-machine、DW+CHARM+std::async 作为对照
- DuckDB + TPC-H：对比未修改的 DuckDB 与 DuckDB+CHARM
- ERMIA + YCSB/TPC-C：CHARM 自适应逻辑未直接接入，以 LocalCache 和 DistributedCache 两种静态策略近似

## 结果与解释

### 图计算与 GUPS（Fig. 7, Tab. 1, Sec. 5.2）
- AMD 上优势最明显。CHARM 基本线性扩展到 64 cores。64 cores 时 BFS/CC/SSSP 相对 baseline 加速约 1.8x/1.9x/2.3x。96 核以上优势扩大到 2x–2.8x，同时 RING/AsymSched/SAM 的吞吐开始下降。
- Tab. 1 显示 SSSP 的 Remote NUMA Chiplet 访问：CHARM 为 6×10³，RING 为 230,939×10³，减少 4 个数量级。CHARM 的去中心化 spread_rate 调整直接压低了远端 chiplet cache fill。

### Intel 上的表现（Fig. 8, Sec. 5.3）
- 全部 benchmark 上仍优于 AsymSched/RING/SAM，单 socket 范围内优势最大。
- 跨 socket 后吞吐会先下降再恢复，与 AsymSched/RING 的差距缩小到"略优或接近持平"。
- 论文归因于 Intel 与 AMD 在 chiplet 数量和互连带宽上的架构差异（Intel 4 tile vs AMD 8 chiplet，Intel EMIB 带宽更高、延迟更均匀）。

### Streamcluster（Fig. 9, Tab. 2, Sec. 5.4）
- CHARM 在 24 cores 时达到 21x 峰值加速（SHOAL 峰值为 32 cores 下 16x）。16 cores 时 CHARM 相对 SHOAL 为 2x。
- 超过 40 cores 后两者收益逐渐掉回接近 1x。线程太多后每线程工作量过小，调度优化空间变小。

### 图规模敏感性（Fig. 10, Sec. 5.5）
- 图规模从 19 MB 扩到 5300 MB，CHARM 稳定优于 RING。
- 效果更受 working set 是否落入 cache 影响，而不是总数据量。这一观察与 OLAP 和排序论文的结论一致：chiplet-aware 优化的有效边界由 working set 与 L3 容量的关系决定。

### SGD（Fig. 11, Fig. 12, Sec. 5.6）
- Logistic loss 吞吐最高 165 GB/s，gradient computation + model update 最高 106 GB/s。对比：DimmWitted-NUMA-node 分别约 50 GB/s 和 40 GB/s。
- 收益来自两个独立因素：(1) chiplet-aware task placement 改善 chiplet 内数据局部性；(2) coroutine 替代 std::async，线程数从 641 降到 34，消除了大量 OS 线程创建/切换开销。

### OLAP（Fig. 13, Sec. 5.7）
- DuckDB + TPC-H SF100，只用 8 cores（保证执行时间足够长以观察调度影响）。
- 大表 hash join / inner join 查询（Q3/Q4/Q5/Q7/Q9/Q10）加速约 1.2x–1.5x。论文说明对这些查询 CHARM 自动把线程分散到更多 chiplet 以利用更大聚合 L3 来缓存 hash table。
- 小工作集查询自动把线程压缩到更少 chiplet。

### OLTP（Fig. 14, Sec. 5.8）
- YCSB 和 TPC-C 上，LocalCache 和 DistributedCache 两种静态策略几乎没有性能差异。
- 论文结论：高同步、高提交开销的 OLTP 负载对 chiplet 级 cache 放置不敏感，瓶颈不在 cache locality。这与 workload 的访存特征一致——事务数据量极小（几条记录），事务执行时间由 commit 同步主导。

## 分析
- CHARM 不把单 NUMA 域视为均匀资源，而是把 chiplet 级异构延迟和分片 L3 当作调度输入。这是它与传统 NUMA-aware runtime 的根本区别。
- 主优化目标是"局部性"与"聚合缓存容量"之间的动态切换，而非常规的"尽量减少远端访问"。当工作集压力高时，CHAR 主动选择扩散以获取更大聚合缓存。
- PMU 计数器直接接入 runtime 决策（而非仅用于事后分析），`spread_rate` 由 remote cache fill 事件率驱动。这一设计说明硬件计数器可以直接参与放置策略的自动化。
- 四类 workload 的结果共同揭示了一条规律：**CHARM 有效的充要条件是 workload 的访存瓶颈在 cache/memory 层级能被 chiplet-aware 放置改变**。图计算（不规则访问）和 SGD（大工作集带宽瓶颈）满足此条件，OLTP（同步瓶颈）不满足。

## 边界
- 研究对象是通用并行 runtime，不是 LLM inference runtime。
- 研究粒度是任务级调度、线程/内存放置和 cache 利用，不是 attention kernel 内部映射。
- 主实验围绕图计算、数据库和统计分析，不包含 vLLM、KV cache 或 LLM serving。
- CHARM 假设单应用独占硬件、chiplet 布局对称、多级 NUMA 下优先吃满一个 socket。
- CHARM 是源码可改、可接入应用 runtime 的用户态框架，不是通用 OS 调度器。

## 可迁移点
- 不把单 NUMA 域视为均匀资源池，调度粒度应下沉到 chiplet 级。
- locality 与聚合缓存容量之间要显式建模为 trade-off，而不是默认固定一种策略。CHARM 的 `spread_rate` 提供了这种建模的具体实现参考。
- 调度决策可以直接参考硬件 PMU 计数器（如 `All DC Fills`），而不只依赖静态拓扑信息。
- CHARM 在 OLTP 上的负结果提供了反面参考：如果 workload 的主瓶颈是同步/提交而不是 cache，chiplet-aware 放置可能没有效果。对于 LLM 推理而言，这意味着需要先确认目标阶段（prefill/decode/attention）的瓶颈是否在 cache 或数据局部性上。

## 不可直接迁移点
- CHARM 的收益来自通用 task runtime 的任务级粒度，不等于当前 kernel 级 `acc-local-l3` 一定有同量级收益。
- CHARM 的 coroutine 模型针对的是"数百个细粒度 chunk 跨 600+ 线程"的场景（如 DimmWitted），与 vLLM attention kernel 的 tile/workitem 并行模型在粒度上不匹配。
- CHARM 的 spread_rate 以 500ms 为周期调整，调整频率远低于 attention kernel 中一次 forward pass 的耗时（ms 级），不能直接用于 per-kernel-invocation 的决策。

## 证据
- 原始资料：`wiki/原始资料/papers/CHARM.pdf`
- 关键锚点：摘要；Fig. 1–14；Tab. 1–2；Algorithm 1–2；Sec. 2.1–2.3；Sec. 4.1–4.4；Sec. 5.1–5.8
