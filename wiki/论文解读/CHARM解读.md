# CHARM 解读
## 问题
- 论文要解决的问题是：`chiplet-based CPU` 即使在同一个 `NUMA domain` 内，也存在不同的 `L3` 访问延迟和核间通信代价；只按 `NUMA node` 建模的调度与内存放置策略不足以覆盖这种差异。

## 核心内容
- `CHARM` 是一个面向 `chiplet CPU` 的 runtime system。
- 它把三件事绑在一起处理：
  - 基于 `chiplet` 的任务放置与迁移
  - 在 `locality` 与更大聚合 `L3` 容量之间做自适应取舍
  - 用轻量级 coroutine 降低细粒度任务切换和同步开销
- 论文覆盖的工作负载包括图计算、并行计算、统计分析、`OLAP` 和 `OLTP`。

## 论文先证明了什么

### 1. 单 `NUMA` 域内并不均匀
- 论文在双路 `AMD EPYC Milan` 上测 `core-to-core latency`，给出三层现象：
  - 同 chiplet 最快
  - 跨 chiplet 但仍在同一 `NUMA` 域内更慢
  - 跨 `NUMA` 最慢
- 对应论文 `Fig. 3`。
- 论文还指出 `within NUMA` 曲线内部本身分成三组：约 `25 ns` 的 `intra-chiplet`、约 `80-90 ns` 的同一 `NUMA` 域内中间延迟层，以及超过 `150 ns` 的更高延迟层。需要注意：论文原文把中间层写成 `inter-chiplet but intra-CCX`，但这与文中 `Milan` 拓扑描述并不自洽，因此这里不直接沿用该术语，只保留“同一 `NUMA` 域内存在三组明显不同的延迟层”这一事实。
- 论文据此把“单 `NUMA` 域内仍有明显异构延迟”作为 chiplet-aware runtime 的出发点。

### 2. `local cache` 与更大聚合缓存容量存在直接 trade-off
- 论文在单路 `8-chiplet AMD EPYC Milan` 上做了 `8-thread` 向量写微基准；该平台每个 chiplet 共享 `32 MB L3`，因此单路聚合 `L3` 总量是 `256 MB`。论文对比：
  - `LocalCache`：线程和数据限制在一个 chiplet
  - `DistributedCache`：同样线程数分散到多个 chiplet
- 结果是：
  - 数据规模不超过 `38 MB` 时，`LocalCache` 执行时间更低
  - 超过单 chiplet 的 `32 MB L3` 容量后，`DistributedCache` 开始更有效
  - 在 `38 GB` 数据规模上，`DistributedCache` 相对 `LocalCache` 的加速最高到 `2.5x`
- 对应论文 `Fig. 5`。
- 因此论文的核心判断不是“永远收紧 locality”或“永远铺开到更多 chiplet”，而是要按工作集动态切换。

## 方法与系统设计

### 1. 四个核心组件
- `performance profiler`：持续监控 cache/memory 行为
- `adaptive controller`：根据 profiler 结果生成调度策略
- `task and memory manager`：管理 coroutine、任务队列和内存放置
- `global scheduler`：协调任务分发和迁移
- 对应论文 `Fig. 6`。

### 2. 调度核心是按 worker 独立调整 `spread_rate`
- 这里的 `worker` 是 `worker thread` 粒度。论文明确写的是 `each worker thread independently decides on migration`，并写明 `CHARM dedicates one physical core to each worker thread`，把 physical core 作为最小独立调度单位。
- 每个 worker 周期性读取自己的 `cache fill event counter`；`Algorithm 1` 里把它标成 `Cache fill events`，`§4.2` 说明它用于跟踪 `remote memory accesses`。
- 若该访问率高于阈值，就把 `spread_rate` 增大，让任务分散到更多 chiplet，以换取更大的聚合缓存容量。
- 若该访问率较低，就把 `spread_rate` 减小，把任务收紧到更少 chiplet，以增强与其他线程的数据局部性。
- 论文明确强调这是去中心化决策：不是先全局收集数据再统一调度，而是每个 worker 根据本地观测独立触发调整。
- 对应论文 `Algorithm 1`。

### 3. `UpdateLocation` 负责无冲突映射
- `CHARM` 给每个 worker 分配唯一 `ID`，再据此计算唯一的 `(chiplet, slot, core)`。
- 映射规则是：`chiplet = floor(unique_worker_ID / (CORES_PER_CHIPLET / spread_rate))`，`slot = unique_worker_ID mod (CORES_PER_CHIPLET / spread_rate)`，再由 `core = chiplet * CORES_PER_CHIPLET + slot` 得到目标物理核心。
- 若上一步算出的 `chiplet >= CHIPLETS`，则继续做回卷修正：`chiplet = chiplet mod CHIPLETS`，`slot = slot + floor(unique_worker_ID / CORES_PER_CHIPLET)`。
- 这套映射以“`1` 个 `worker thread` 独占 `1` 个 `physical core`”为前提；只有当当前 `spread_rate` 覆盖的物理核心数不少于 `worker thread` 数时，映射才有效。若核心数不够，论文算法会直接跳过这次 remap，保持原有 `affinity`。
- 它同时设置：
  - `thread affinity`
  - `set_mempolicy(MPOL_BIND, ...)`
- 这样做的目标是让任务落点和主存分配一起跟着拓扑走，并尽量压过 OS 的默认 `NUMA balancing`。
- 对应论文 `Algorithm 2`。

### 4. 细粒度并发依赖 coroutine，不依赖大量 OS 线程
- `CHARM` 的任务有自己的栈、状态和调度器。
- 任务可以在开发者定义的位置 `yield`，运行时再结合 profiling 结果决定是否迁移。
- 每个 core 上维护本地低开销任务队列；若本地队列为空，优先从同 chiplet 其他 core 偷任务，再考虑别的 chiplet。

### 5. profiling 直接读 PMU 事件
- `CHARM` 用 `libpfm` 和现成 PMU counter 收集 profiling 数据。
- 论文说明高频轮询时 profiling 开销约为 `5%-10%`，但轮询频率可调。
- 论文对事件名给了更具体说明：
  - `AMD`：`ANY_DATA_CACHE_FILLS_FROM_SYSTEM`
  - `Intel`：`OFFCORE_RESPONSE`
- 若映射到 `AMDuProfPcm`，`ANY_DATA_CACHE_FILLS_FROM_SYSTEM` 对应 `All DC Fills (pti)`。
- 论文原文把这些计数器的来源层次概括为 `on-chip (intra-CCX)`、`on-die (inter-CCX)` 和 `remote memory (inter-NUMA)`。但按文中前面对 `Milan` 的 `chiplet/CCX` 描述，这组术语并不完全自洽；更稳妥的理解是：它试图区分更近的片上来源、更远的片内/片间来源，以及跨 `NUMA` 的远端主存来源。
- 在实验实现里：
  - `RMT_CHIP_ACCESS_RATE = 300`
  - `SCHEDULER_TIMER = 500 ms`

### 6. 实现假设
- `CHARM` 假设：
  - 单应用独占硬件
  - chiplet 布局是对称的
  - 多级 `NUMA` 下优先吃满一个 socket，再扩到另一个 socket
- 它不是通用 OS 调度器，而是源码可改、可接入应用 runtime 的用户态框架。

## 实验设置

### 延迟测量方法与工具
- 论文在 `§2.1 Inter-core latencies` 里说明，`Fig. 3` 的 `Inter-core latencies` 是在双路 `AMD EPYC Milan` 上测的 `core-to-core latency`。
- 方法上，作者把线程绑定到不同核心，并用 `compare-and-swap (CAS)` 操作测量核心间通信延迟；论文把结果画成 `CDF`（`Fig. 3`）。
- 就正文当前给出的信息，论文没有像 `BenchIT`、`lmbench` 这类那样明确点名一个独立 benchmark/tool 名称；能直接确认的只有“绑核 + CAS 测通信延迟”这一实现口径。
- 因此若后续在本机复现，最接近论文口径的做法是：固定线程亲和性，构造两核心之间基于共享 cache line 的 `CAS` 往返/竞争测试，再统计不同核心对之间的延迟分布。

### 平台
- `AMD`：双路 `EPYC Milan 7713`，每 socket `64 cores`、`512 GB RAM`、`8 chiplets`，每 chiplet `32 MB L3`
- `Intel`：双路 `Xeon Platinum 8488C`，每 socket `48 cores`、`512 GB RAM`、共享 `105 MB L3`

### baseline
- 论文在 `§5.1 Experimental setup` 里给出的总 baseline 一共四个：
  - `RING`：`NUMA-aware` 的 `message-batching runtime system`，面向高性能、内存内、数据密集型工作负载。
  - `SHOAL`：面向 `NUMA multi-core` 的 runtime，提供数组抽象，用来优化内存分配与访问模式。
  - `AsymSched`：`bandwidth-centric NUMA scheduler`，目标是在不对称互连下优化线程与内存放置。
  - `SAM`：面向现代 `multi-programmed` 多核机器的 CPU scheduler，通过识别 `latency tolerance` 并结合 `hyperthreading awareness` 缓解数据共享与竞争带来的性能问题。
- 但论文后面的具体子实验并不总是同时对比这四个系统：
  - 图计算和 `GUPS` 主要对比 `RING / AsymSched / SAM`。
  - `Streamcluster` 只对比 `SHOAL`。
  - `DimmWitted + SGD` 额外用了 `DW-per-core`、`DW-NUMA-node`、`DW-per-machine` 和 `DW+CHARM+std::async` 作为对照。
  - `DuckDB + TPC-H` 对比的是未修改的 `DuckDB` 与 `DuckDB+CHARM`。
  - `ERMIA + YCSB/TPC-C` 没有直接接入 `CHARM` 自适应逻辑，而是比较 `LocalCache` 和 `DistributedCache` 两种静态策略，作为近似对照。

### 工作负载
- 图计算：`BFS / PR / CC / SSSP / Graph500`
- 并行计算：`RandomAccess (GUPS)`、`Streamcluster`
- 统计分析：`DimmWitted + SGD`
- `OLAP`：`DuckDB + TPC-H`
- `OLTP`：`ERMIA + YCSB / TPC-C`

### 工作负载特征与选取原因
- 论文选择这些 workload 的总口径是：覆盖四类场景 `graph processing / high performance parallel processing / statistical analytics / database management`，并观察 `irregular memory access`、`working set`、`locality`、`data sharing`、`synchronization` 和 `off-chip traffic` 对 chiplet-aware runtime 的影响。
- 图计算 `BFS / PR / CC / SSSP / Graph500`
  - 负载特征：论文明确写这类图算法具有 `irregular access patterns`；实验里图任务按 `active frontier node` 动态生成，并一直运行到完整遍历或收敛。
  - 选择原因：论文明确指出 chiplet-aware task partitioning 对这类不规则内存访问 workload 尤其有效，因此用它们验证 `CHARM` 的主要收益场景。
- `RandomAccess (GUPS)`
  - 负载特征：论文定义它用于评估分布式共享内存架构中的 `non-contiguous memory accesses`。
  - 选择原因：它直接考察非连续访问带来的数据移动和缓存/内存访问代价，适合观察 chiplet-aware 放置对吞吐和扩展性的影响。
- `Streamcluster`
  - 负载特征：论文定义它是 `compute-intensive clustering`，同时对 `memory access patterns` 敏感；它还能提供关于 `working sets`、`locality`、`data sharing`、`synchronization` 和 `off-chip traffic` 的观察。
  - 选择原因：论文专门用它在共享内存多核场景下对比 `CHARM` 和 `SHOAL`，考察并行 workload 中 locality 与可用聚合缓存容量的取舍。
- `DimmWitted + SGD`
  - 负载特征：论文使用 `logistic regression` 的 `SGD`，数据集约 `6250 MB`；`DimmWitted` 会把 workload 切成数百个细粒度 chunk，并跨 `600+` 线程执行；每个 epoch 会多次触碰整份 `6 GB` 输入，工作集超过单 socket 聚合 `L3`。
  - 选择原因：论文在 `§5.5` 明确用它研究 `irregular data and memory access patterns` 对 `CHARM` 的影响，同时检验 coroutine 相对 `std::async` 的线程切换与同步开销。
- `DuckDB + TPC-H`
  - 负载特征：既包含大表 `hash join / inner join`，也包含更小工作集的 `scan/filter` 类查询；论文还说明这里的任务可小到处理 `2-4 MB` 数据、运行时间不到 `1 s`。
  - 选择原因：论文用它验证 `CHARM` 是否能随 query type 和 working set 大小，在“跨 chiplet 扩大聚合 `L3`”和“收紧到少数 chiplet 保持 locality”之间切换。
- `ERMIA + YCSB / TPC-C`
  - 负载特征：`YCSB` 是单表、`45% read + 55% read-modify-write`；`TPC-C` 有更复杂的事务组合和 `cross-partition` 访问。论文把这类 `OLTP` 负载概括为 `short transactions with frequent commits and synchronizations`。
  - 选择原因：论文用它们测试 chiplet-aware 调度的适用边界，判断在事务提交、同步和 `ACID` 开销主导时，cache locality 与更大聚合 cache 是否仍然重要。

## 结果与解释

### 1. 图计算与随机访问：`AMD` 上优势最明显
- 论文报告 `CHARM` 在图计算和 `GUPS` 上能基本线性扩展到 `64 cores`。
- 到 `64 cores` 时，`BFS / CC / SSSP` 相对 baseline 的加速分别约为 `1.8x / 1.9x / 2.3x`。
- 到 `96` 核以上时，论文称 `CHARM` 相对 baseline 的优势扩大到 `2x-2.8x`，同时 `RING / AsymSched / SAM` 的吞吐开始下降。
- 论文还给出 `64 cores` 下的访问统计：`Tab. 1` 中各 workload 的 `Remote NUMA Chiplet` 访问都低于 `RING`；例如 `SSSP` 条目下，`CHARM` 为 `6 x 10^3`，`RING` 为 `230939 x 10^3`。
- 吞吐扩展对应论文 `Fig. 7`，访问统计对应论文 `Tab. 1`。

### 2. `Intel` 上仍然有效，但优势小于 `AMD`
- 在双路 `Xeon Platinum 8488C` 上，论文说 `CHARM` 在全部 benchmark 上仍优于 `AsymSched / RING / SAM`，且在单 socket 覆盖范围内优势最大。
- 但论文明确说：
  - 跨 socket 后吞吐会先下降再恢复
  - 与 `AsymSched / RING` 的差距会缩小到“略优或接近持平”
- 论文把这一点归因于 `Intel` 与 `AMD` 在 chiplet 数量和互连带宽上的架构差异。
- 对应论文 `Fig. 8`。

### 3. `Streamcluster`：相对 `SHOAL` 的高点出现在中等并行度
- `CHARM` 在 `24 cores` 时达到 `21x` 峰值加速。
- `SHOAL` 的峰值是 `32 cores` 下的 `16x`。
- 在 `16 cores` 时，论文报告 `CHARM` 相对 `SHOAL` 为 `2x`。
- 超过 `40 cores` 后，两者收益都逐渐掉回接近 `1x`，论文解释是线程太多后每线程工作量过小，调度优化空间变小。
- 速度对比对应论文 `Fig. 9`，访问模式对比对应论文 `Tab. 2`。

### 4. 图规模敏感性：更受工作集大小影响，而不是总数据量
- 论文把图规模从 `19 MB` 扩到 `5300 MB`，比较 `CHARM` 相对 `RING` 的速度。
- 结论是 `CHARM` 在不同图规模上都稳定优于 `RING`。
- 论文特别指出，效果更受 `working set` 是否落进 cache 影响，而不是只看总输入有多大。
- 对应论文 `Fig. 10`。

### 5. 统计分析：coroutine 对 `SGD` 提升很大
- 在 `DimmWitted` 的 `SGD` 实验里：
  - `logistic loss` 吞吐最高到 `165 GB/s`
  - `gradient computation + model update` 吞吐最高到 `106 GB/s`
- 与之对比：
  - `DimmWitted-NUMA-node` 在两项任务里分别约到 `50 GB/s` 和 `40 GB/s`
  - `DW + CHARM + std::async` 低于 `DW + CHARM`
- 论文把结果归因于两点：
  - chiplet-aware task placement 改善 chiplet 内数据局部性、优化 cache 使用并减少主存访问
  - coroutine 让多个任务可以复用同一线程，低于 `std::async` 的线程创建和 OS 级切换开销
- 文中还给出一个直接对比：`32 cores` 下，`DimmWitted` 创建了 `641` 个线程，而 `CHARM` 用了 `34` 个。
- 吞吐结果对应论文 `Fig. 11`，线程并发对比对应论文 `Fig. 12`。

### 6. `OLAP`：大表 join 查询收益更明显
- 在 `DuckDB + TPC-H SF100` 上，论文只用 `8 cores` 做对比，以保证执行时间足够长、能观察到调度影响。
- 所有查询都受益于 `CHARM`，而且额外开销很小。
- 论文点名 `Q3/Q4/Q5/Q7/Q9/Q10` 这类大表 `hash join / inner join` 查询，速度提升大约在 `1.2x-1.5x`。
- 对较小工作集查询，论文写的是把线程压缩到更少 chiplet；对大 join 查询，则把线程分散到更多 chiplet，以利用更大的聚合 `L3` 容量。
- 对应论文 `Fig. 13`。

### 7. `OLTP`：`YCSB/TPC-C` 几乎没有收益
- 论文没有把 `CHARM` 的自适应逻辑直接接进 `ERMIA`，而是用两种静态策略近似：
  - `LocalCache`
  - `DistributedCache`
- 在 `YCSB` 和 `TPC-C` 上，两种策略几乎没有性能差异。
- 论文结论很明确：高同步、高提交开销的 `OLTP` 负载，对 chiplet 级 cache 放置不敏感，瓶颈主要不在 cache locality。
- 对应论文 `Fig. 14`。

## 分析
- 从论文可直接提炼的相关事实之一是：它没有把单 `NUMA` 域视为均匀资源，而是明确把 `chiplet` 级异构延迟和分片 `L3` 当作调度输入。
- 论文的主优化目标也不是单纯减少远端访问，而是在：
  - 更强局部性
  - 更大聚合缓存容量
  之间动态切换。
- 论文还把 PMU 计数器直接接入 runtime 决策，并用 `remote cache fill` 驱动 `spread_rate` 调整；这一做法说明硬件计数器可以直接参与放置策略切换。

## 边界
- 论文对象是通用并行 runtime，不是 `LLM inference runtime`。
- 它研究的是任务级调度、线程/内存放置和 cache 利用，不是 attention kernel 内部映射。
- 论文主实验围绕图计算、数据库和统计分析，不包含 `vLLM`、`KV cache` 或 `LLM serving`。
- 因此，文中的收益数字不能直接外推到当前 `CPU attention` 或整模型推理。

## 可迁移点
- 不能把单 `NUMA` 域视为均匀资源池。
- `locality` 与聚合缓存容量之间要显式建模，而不是默认固定一种策略。
- 调度决策可以直接参考硬件计数器，而不只依赖静态拓扑。

## 不可直接迁移点
- `CHARM` 的收益来自通用 task runtime，不等于当前 kernel 级 `acc-local-l3` 一定有同量级收益。
- 论文里的 `OLTP` 结论也说明：如果主瓶颈是同步/提交而不是 cache，chiplet-aware 放置本身可能几乎没有效果。

## 证据
- [../原始资料/papers/CHARM.pdf](../原始资料/papers/CHARM.pdf)
