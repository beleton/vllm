# CHARM 解读

## 问题
Chiplet CPU 在同一 NUMA domain 内仍存在不同的 L3 访问延迟和核间通信代价——AMD EPYC Milan 上同 NUMA domain 内延迟分三组：约 25 ns（intra-chiplet）、约 80–90 ns（同 NUMA 域内中间层）、超过 150 ns（更远层级）（Fig. 3）。只按 NUMA node 建模的调度与内存放置策略无法覆盖 chiplet 级延迟差异。CHARM 要回答的是：能否构建一个运行时系统，使各类并行应用自动在"chiplet 局部性"与"更大聚合 L3 容量"之间做出自适应选择。

## 背景：CHARM 覆盖的四类 Workload 及其访存特征

CHARM 覆盖四类场景（Sec. 5.1），每类的访存特征不同，对 chiplet-aware 调度的敏感点也不同。

### 图计算（BFS / PageRank / CC / SSSP / Graph500）

图算法在 CHARM 中是一个核心场景。这些算法的共同特征是：对一个大规模图（论文使用 Kronecker 图，约 4 GB，含 2^24 个顶点和 16×2^24 条边）做多轮迭代计算，直到结果收敛或遍历完成。

> **frontier（活跃边界）是什么**：frontier 是图迭代算法中的一个关键概念——它是**当前轮次正在处理的"活跃节点集合"**。每一轮只处理 frontier 中的节点，处理完之后会生成一批新的活跃节点，成为下一轮的 frontier。不同算法的 frontier 含义略有差异：BFS 中是当前深度正在扩展的节点层；PageRank 中是尚未收敛的节点；SSSP 中是距离值刚被更新的节点。它们的共同点是——每轮只需要操作 frontier 中的节点，不需要遍历全图。

- **计算流程**：基于 frontier 的迭代遍历。每轮从当前活跃节点集（active frontier）出发，访问其邻居，更新邻居状态，将新激活的节点加入下一轮 frontier，直到收敛或遍历完成。算法实现来自 PBBS（Problem Based Benchmark Suite）。
- **访存特征**：不规则内存访问（irregular access patterns）。每个 frontier node 需要访问图中任意位置的邻居数据——邻居边可能散布在图文件的任何位置。因此每次迭代都是大量随机地址跳转，cache line 局部性极差。
- **与 chiplet 架构的冲突**：若一个 chiplet 上的线程访问了另一个 chiplet 主存中的邻居数据，不仅产生远端主存访问（~120 ns），还会在本地 L3 中留下"只用一次就被逐出"的 cache line，白白挤占本地 L3 容量。
- **论文选择原因**：chiplet-aware task partitioning 对不规则内存访问 workload 尤其有效——跨 chiplet remote cache fill 的减少直接转化为遍历吞吐提升。图计算是验证 CHARM 主要收益的理想场景。

### 随机访问（GUPS, Giga Updates Per Second）

GUPS 是一个衡量内存系统随机访问吞吐的标准微基准（HPC Challenge Benchmark 之一）。它做的事情极简：对一个大数组做随机位置的原子更新，几乎每次访问都是 cache miss，性能完全由内存子系统的随机访问延迟和带宽决定。

- **计算流程**：对一个远大于 L3 的大数组做随机位置的读-修改-写（`A[random_index] = A[random_index] XOR random_value`）。每个线程独立生成随机索引，读取 → 更新 → 写回。计算量极小，瓶颈全在内存访问。
- **访存特征**：完全非连续内存访问（non-contiguous memory accesses）。每次访问的地址是随机的，几乎每次都是 cache miss。性能直接反映内存系统的随机访问延迟和带宽。
- **与 chiplet 架构的冲突**：随机地址落在哪个 chiplet 的内存控制器范围决定了访问延迟。若线程绑在 chiplet 0 但随机地址落在 chiplet 3 的内存区域，每次访问都是远端 NUMA + 跨 chiplet。
- **论文选择原因**：GUPS 以最纯粹的形式暴露 chiplet 间数据移动和远端内存访问的代价——没有计算干扰，没有复杂数据依赖，适合观察 chiplet-aware 放置对吞吐和扩展性的影响。

### 统计分析（DimmWitted + SGD）

DimmWitted 是一个面向统计分析的 NUMA 优化框架，CHARM 以它为基础验证 chiplet-aware 调度对数据分析 workload 的效果。SGD（Stochastic Gradient Descent）是逻辑回归的标准训练算法。

- **计算流程**：逻辑回归的随机梯度下降。每轮 epoch 遍历全部训练数据（约 6250 MB，10,000 样本 × 8,192 特征），对每个样本计算梯度并更新模型参数。DimmWitted 框架把 workload 切成数百个细粒度 chunk，每个 chunk 用 `std::async` 提交，产生大量 OS 线程（32 cores 下创建了 641 个线程）。
- **访存特征**：每个 epoch 完整触碰整份 ~6 GB 输入。工作集 = 全部训练数据 + 模型参数，远超单 socket 聚合 L3（256 MB）。计算过程是 memory-bandwidth-intensive——梯度计算只需一次乘加，数据搬运量远超计算量。同时，600+ 线程在 32 个物理核上竞争，context switch 开销进一步消耗了本可用于数据搬运的 CPU 时间。
- **与 chiplet 架构的冲突**：两个层面的问题叠加——(1) 训练数据散布在多个 chiplet 的内存中，访问模式虽规律（顺序扫数据）但跨 chiplet 访问不可避免；(2) OS 线程数远超物理核数，线程切换开销已经很大，跨 chiplet 任务迁移进一步恶化 cache 局部性。
- **论文选择原因**：SGD 同时检验 CHARM 的两个独立优化——chiplet-aware 放置减少远端内存访问 + coroutine 替代 OS 线程减少切换开销。这两个收益是加性的，在实验中被分别测量。

### OLAP（DuckDB + TPC-H SF100）

DuckDB 是一个嵌入式 OLAP 数据库，CHARM 将其作为 chiplet-aware 调度的宿主运行时，验证在查询粒度上自适应切换的能力。TPC-H SF100 是一个 100 GB 规模的 OLAP 标准测试集。

- **计算流程**：Coordinator 解析 SQL → 生成执行计划 → Worker 在本地数据 fragment 上执行 scan/filter/join/aggregate。论文只用 8 个 core，以保证足够的查询执行时间以便观察调度效果。
- **访存特征**：两类查询混合：(1) 大表 hash join / inner join（Q3/Q4/Q5/Q7/Q9/Q10），build 端和 probe 端的数据量在 GB 级，工作集远大于 L3；(2) 小工作集 scan/filter 查询，处理 2–4 MB 数据，运行时间不到 1s。
- **与 chiplet 架构的冲突**：大 join 查询需要更大的聚合 L3 来缓存 build 端的 hash table（hash table 大小可能几十 MB，超出单 chiplet L3）；小 scan/filter 查询则更适合精简单 chiplet，避免跨 chiplet cache coherence 开销。
- **论文选择原因**：验证 CHARM 能否随查询类型和 working set 大小自动切换策略——大 join 自动扩散（要更大聚合缓存），小 scan 自动收紧（要更低延迟）。这是对 CHARM 自适应逻辑的粒度和响应能力的关键检验。

### OLTP（ERMIA + YCSB / TPC-C）

ERMIA 是一个内存优化的 OLTP 引擎，以它验证 OLTP workload 对 chiplet-aware 调度的敏感性。论文没有将 CHARM 的自适应逻辑直接接入 ERMIA（因为 ERMIA 的线程管理高度封闭），而是用 LocalCache 和 DistributedCache 两种静态策略近似 CHRAM 的行为范围。

- **计算流程**：短事务处理。YCSB：单表，45% read + 55% read-modify-write。TPC-C：更复杂的事务组合，含跨分区（cross-partition）访问。事务执行路径：读 → 修改 → 提交（commit log 写入）。
- **访存特征**：短事务、频繁提交和同步。事务本身的数据访问量很小（几条到几十条记录），但 commit 的同步开销（日志刷盘、锁、latch）主导了执行时间。
- **与 chiplet 架构的冲突**：数据访问量极小，L3 容量不构成瓶颈。跨 chiplet 访问即使存在，绝对时间也被事务同步开销淹没。
- **论文选择原因**：测试 chiplet-aware 调度的适用边界——当瓶颈是同步/提交而非 cache 时，CHARM 的策略是否还重要。结果证明 OLTP 对 chiplet 级放置不敏感，这也是 CHARM 有效性的反面证据。

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
- **Global Scheduler**：协调任务分发和迁移。去中心化调度，且非抢占式

### 运行时层次与进程边界

CHARM 的执行层次是三层结构：**进程 → 工作线程 → 协程任务**。

**进程（MPI rank）**

CHARM 使用 MPI 实现多机并行——一个 MPI rank 就是一个操作系统进程。调用 `CHARM_Init()` 后，进程内创建 `global_scheduler`、`global_comm`、PGAS 内存管理器和通信缓冲区；这些组件只在本进程内部可见，不是跨进程共享的。

CHARM 在代码层面强制约束**一台物理机器上只允许运行一个 CHARM 进程**。具体做法是：初始化时调用 `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`，让 MPI 把"共享同一块物理内存的进程"归为一组，然后检查组内进程数——如果不是 1，直接断言失败退出。这样做的原因是 CHARM 的调度器只管理本进程内部的线程，不具备跨进程协调能力；如果同机多进程强行运行，彼此之间没有任何全局的 worker placement 协调，会导致各进程争抢同一组物理核（绑核冲突）、PMU 计数器互相污染、内存绑定相互覆盖。

> **MPI 背景**：MPI 用"共享内存节点"（shared-memory node）来描述一组可以直接访问同一块物理内存的进程——通常就是同一台物理服务器。这个概念和 Linux 的 NUMA node 完全不同：一台物理机器内部可以有多个 NUMA node，但在 MPI 视角下它们同属一个 shared-memory node。

当涉及多机时，不同机器的 CHARM 进程之间通过 MPI 或 RPC 通信。

**工作线程（worker thread）**

每个 CHARM 进程启动时创建固定数量的 OS 线程作为 worker pool（本地代码 `THREAD_SIZE = 8`）。每个 worker 独占一个物理核，是调度器进行 chiplet-aware 放置的最小单位——`spread_rate` 调整的就是这些 worker 线程的 CPU 亲和性和内存绑定范围。

**协程任务（coroutine task）**

worker 线程是执行载体，协程任务才是真正跑工作负载的单元。当前代码初始化 256 个协程（`INIT_CORO_NUM = 256`），它们在用户态切换，不涉及系统调用，开销远小于 OS 线程。协程不参与 `spread_rate` 的绑核映射——被绑定到物理核上的是 worker 线程，不是协程。

**任务入口与分发**

主任务只在 rank 0（编号为 0 的那个 MPI 进程）上提交。`all_do()` 和 `call()` 按 `rank × THREAD_SIZE + thread_rank` 的规则计算出全局 core ID，将任务分发到对应 worker 线程上执行。rank 0 的角色是"任务入口和分发起点"，它不负责统一指挥所有 worker——每个 worker 的 `spread_rate` 调整是独立、去中心化的（见下一节）。

### 核心调度机制：去中心化的 spread_rate 调整（Algorithm 1, Sec. 4.2）

**设计前提**：CHARM 给每个 worker thread 分配一个独立的 physical core，worker thread 是 CPU affinity 绑定的最小单位。所有 worker thread 的总数在启动时确定（`THREAD_SIZE`），运行期间不变。

**决策循环**：每个 worker 以固定周期独立运行 Algorithm 1：

1. 检查距上次决策是否已过 `SCHEDULER_TIMER`
2. 读取本 worker 的 PMU cache fill 事件计数器，归一化为事件率（events / `SCHEDULER_TIMER`）
3. 与阈值 `RMT_CHIP_ACCESS_RATE`（论文取 300）比较
4. 高于阈值 → 增大 `spread_rate`（向更多 chiplet 扩散，换取更大聚合缓存）
5. 低于阈值 → 减小 `spread_rate`（向更少 chiplet 收紧，增强局部性）
6. 调用 `UpdateLocation()` 将新的 `spread_rate` 转化为具体的 core 绑定

`spread_rate` 是每个 worker 自己的成员变量，调整对象是该 worker 自身的 CPU affinity 和内存绑定范围，不改变 worker 总数，也不控制其他 worker。

**与 workload 特征的耦合关系**：
- 图计算（不规则访问）：`spread_rate` 的变化直接影响 frontier node 数据和邻居数据在哪个 chiplet 的 L3 中，跨 chiplet remote cache fill 的减少直接转化为遍历吞吐提升。
- SGD（大工作集）：工作集 6 GB 远超聚合 L3（256 MB），`spread_rate` 的主要作用是平衡各 chiplet 上的本地主存访问压力，减少远端 NUMA 内存访问。
- OLAP（混合大小查询）：大 join 查询自动推高 `spread_rate`（需要更大聚合缓存），小 scan/filter 查询自动降低（精简单 chiplet 即可），实现查询粒度的自适应。
- OLTP（短事务）：事务数据量极小，remote cache fill 事件率始终很低，CHARM 保持低 `spread_rate`，不触发不必要的任务扩散。这也解释了为什么 CHARM 在 OLTP 上无收益。

### 核心映射机制：UpdateLocation（Algorithm 2）——无冲突的确定性绑核

**问题**：上一节的去中心化调度中，每个 worker 独立决定自己的 `spread_rate`，各自调用 `UpdateLocation()`。没有全局锁、没有集中仲裁，如何保证 N 个 worker 不会把两个线程绑到同一个 physical core 上？

**答案**：每个 worker 在初始化时被分配一个**全局唯一的 worker ID**（0, 1, 2, ...），`UpdateLocation` 的公式将这个唯一 ID **确定性地** 映射到一个唯一的 `(chiplet, slot, core)` 三元组。因为 ID 不同，算出来的 core 就不可能相同——冲突从数学上被排除了。公式的本质是按唯一 worker ID 顺序填入这些槽位——ID 0 进 chiplet 0 的 slot 0，ID 1 进 chiplet 0 的 slot 1……ID 小的优先填满前面的 chiplet，再填下一个。

**具体映射公式**：

```
chiplet = floor(unique_worker_ID / (CORES_PER_CHIPLET / spread_rate))
slot    = unique_worker_ID mod (CORES_PER_CHIPLET / spread_rate)
core    = chiplet × CORES_PER_CHIPLET + slot
```

若 `chiplet >= CHIPLETS`，做修正：
```
chiplet = chiplet mod CHIPLETS
slot    = slot + floor(unique_worker_ID / CORES_PER_CHIPLET)
```

**以 AMD EPYC Milan 为例**（单 socket 8 chiplet，每 chiplet 8 核，即 `CORES_PER_CHIPLET = 8`，`CHIPLETS = 8`）：

| 场景  | spread_rate | 每 chiplet 槽位数 | 8 个 worker 的落点                                |
| --- | ----------- | ------------- | --------------------------------------------- |
| 收紧  | 1           | 8             | 全部在 chiplet 0，各占 1 个 core                     |
| 扩散  | 2           | 4             | worker 0–3 在 chiplet 0，worker 4–7 在 chiplet 1 |
| 全铺开 | 8           | 1             | 每个 chiplet 各 1 个 worker                       |

**容量约束**：remap 前先检查 `worker_count <= spread_rate × CORES_PER_CHIPLET`。若不满足（例如 64 个 worker 但 `spread_rate = 1` 只覆盖 8 个 core），跳过此次 remap，保持现有 affinity，等待下一轮调度周期重试。这个检查同样是每个 worker 本地完成的，不需要全局协调。

**实际执行**：检查通过后，该 worker 调用 `pthread_setaffinity_np(core)` 把自己绑定到计算出的 core，同时调用 `set_mempolicy(MPOL_BIND, numa_node)` 把内存分配也绑定到对应的 NUMA node。每个 worker 只操作自己的 affinity，不碰其他 worker。论文提到 remap 时机选在 task 完成之后，以减少对正在执行的任务的影响。

### Coroutine 任务模型（Sec. 4.3）

CHARM 不依赖大量 OS 线程来支持细粒度并发。worker thread 是执行载体，coroutine task 是被执行的工作单元。一个进程内有固定数量的 worker thread；每个 worker thread 可以顺序执行多个 coroutine task；一个 coroutine task 在某一时刻只运行在一个 worker thread 上。task 可以通过调度队列和 RPC 被投递到其他 worker，worker thread 本身不随 task 数量增加而增加。

每个任务有自己的栈、状态和调度器，可以在开发者定义的位置 `yield`，运行时结合 profiling 结果决定是否迁移。每个 core 上维护本地低开销任务队列；若本地队列为空，优先从同 chiplet 其他 core 偷任务，再考虑别的 chiplet。

**与 workload 特征的关联**：DimmWitted + SGD 实验（Fig. 11, Fig. 12）直接展示了效果——32 cores 下 DimmWitted 创建了 641 个 OS 线程，CHARM 只用了 34 个。

**DimmWitted 为什么有 641 个线程**：DimmWitted 把训练数据切分成数百个细粒度 chunk，对每个 chunk 调用 `std::async` 提交给线程池。`std::async` 的默认行为是为每个任务创建（或从池中分配）一个 OS 线程。32 个物理核上同时存在 641 个线程，意味着 OS 调度器必须频繁在 641 个线程之间做 context switch——每次切换都需要保存/恢复寄存器、可能触发 TLB flush、污染 cache。当切换开销超过了实际计算时间，系统就陷入了"线程管理比干活还贵"的状态。

**CHARM 为什么只有 34 个线程**：CHARM 的线程数在启动时确定，不随 chunk 数量增长。32 个 worker 线程（每个绑一个物理核）+ ~2 个辅助线程（通信、调度），共约 34 个。DimmWitted 的每个 chunk 对应 CHARM 的一个 coroutine（`INIT_CORO_NUM = 256`），但这些 coroutine 不是 OS 线程——它们是用户态的轻量执行单元，在 worker 线程之间按需调度。一个 worker 跑完一个 coroutine → 换下一个 coroutine → 继续跑，切换在用户态完成（换栈指针、换状态），不经过 OS，开销比 OS context switch 低几个数量级。

**本质**：CHARM 把"一个任务 = 一个 OS 线程"的模型，替换为"固定线程池 + coroutine 复用"的模型。线程数回归物理核数，剩下的并发由用户态协程消化，OS 不再参与频繁的线程调度。这个收益与 chiplet-aware 放置是独立的——实验证明两者是加性关系：chiplet-aware 放置减少远端内存访问，coroutine 减少切换开销，合起来进一步提升吞吐。

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

## 可迁移点
- 不把单 NUMA 域视为均匀资源池，调度粒度应下沉到 chiplet 级。
- locality 与聚合缓存容量之间要显式建模为 trade-off，而不是默认固定一种策略。CHARM 的 `spread_rate` 提供了这种建模的具体实现参考。
- 调度决策可以直接参考硬件 PMU 计数器（如 `All DC Fills`），而不只依赖静态拓扑信息。
- CHARM 在 OLTP 上的负结果提供了反面参考：如果 workload 的主瓶颈是同步/提交而不是 cache，chiplet-aware 放置可能没有效果。对于 LLM 推理而言，这意味着需要先确认目标阶段（prefill/decode/attention）的瓶颈是否在 cache 或数据局部性上。

## 证据
- 原始资料：`wiki/原始资料/papers/CHARM.pdf`
- 关键锚点：摘要；Fig. 1–14；Tab. 1–2；Algorithm 1–2；Sec. 2.1–2.3；Sec. 4.1–4.4；Sec. 5.1–5.8
