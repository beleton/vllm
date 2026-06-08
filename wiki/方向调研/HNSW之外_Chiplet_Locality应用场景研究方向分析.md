# HNSW 之外的 Chiplet Locality 应用场景研究方向分析

生成时间：2026-05-27

## 结论

当前 HNSW 方向已经证明“相似请求在同一 CCD 内并行执行可能提高本地 L3 命中”的现象，但 HNSW 单 query 通常只有 `1-3 ms`。这种时间尺度限制了在线调度空间：相似性判断、分组等待、路由、负载均衡和结果合并都必须在几十微秒到数百微秒内完成，否则收益容易被抵消。

跳出 HNSW 后，更值得关注的场景不是“另一个近似最近邻索引”，而是具备以下特征的系统：

1. 单个任务、批次或阶段的执行时间至少达到数毫秒到秒级。
2. 存在可识别的共享工作集，例如热 embedding row、图邻域、特征行、数据库中间状态、文件系统元数据或局部仿真状态。
3. 任务之间可以在短窗口内分组，或系统本身已经以 batch、mini-batch、compaction、query stage、sandbox session 等较粗粒度执行。
4. 放置机制可控，包括线程绑核、per-CCD 队列、数据分片、热数据复制、阶段切换、session 亲和或共享状态分片。
5. 调度开销可被任务粒度摊销，不需要在每个亚毫秒请求上做复杂决策。

综合研究价值、时间尺度、实验可实现性和论文创新空间，最值得优先投入的方向是：

| 优先级 | 方向 | 推荐结论 |
| --- | --- | --- |
| P0 | GNN neighbor sampling 与 feature gather | 最符合“相似任务共享工作集”的长时间尺度版本。mini-batch 粒度更大，调度预算明显优于 HNSW，论文空间强。 |
| P1 | Agent tool sandbox 的共享 OS 路径分片与 session 亲和 | 时间尺度最大、应用新；归因难度高，但与已有 chiplet cache 调度工作重叠较少。 |
| P2 | LSM/HTAP 的 phase-aware Local/Mixed 切换 | 阶段长、开销可摊销，数据库系统价值明确；风险是工程面较大，且与已有 chiplet OLAP/CHARM 方法线接近。 |
| P3 | 推荐系统 embedding lookup | 不建议作为第一主线。AMD embedding bag 论文已覆盖表到 CCD/核心组的任务划分和 hot row 复用，剩余空间较窄。 |
| P4 | HPC/CFD/FEA/stencil 的 CCD 级块调度 | L3 容量敏感证据强，但领域已有 cache blocking、NUMA、MPI/OpenMP 优化，chiplet 增量需要较强应用知识。 |
| P5 | EDA/RTL/离散事件仿真 | 时间尺度和局部状态都合适，3D V-Cache 证据支持；开源可改系统较少，实验不确定性高。 |
| P6 | 网络 I/O、RPC、KV serving | 可做 queue/worker/mempool 分区和 tail 隔离，但 AMD Zen 上不应使用 TiNA 式 remote cache 容量套利作为核心动机。 |
| P7 | OLAP 查询引擎 worker 部署 | 已有 OLAP on Chiplet 论文覆盖较充分，适合作为方法基线，不适合作为新主线。 |
| P8 | Genomics k-mer counting | 分桶、热 minimizer 和 hash locality 存在，但应用域较窄，I/O 与数据规模可能掩盖 L3 收益。 |

最终建议收敛到三个候选：

1. **GNN neighbor sampling / feature gather**：作为主要论文方向继续深挖，时间尺度和共享工作集最符合目标。
2. **Agent tool sandbox**：作为高创新、高风险备选，前提是先用 `perf c2c`、lock contention、page fault、NUMA 统计证明共享 OS 路径确实进入关键路径。
3. **LSM/HTAP phase-aware 调度**：作为工程型数据库备选，适合复用已有 AMDuProf/PMU 分析方法验证阶段工作集和前后台隔离收益。

## 筛选框架

### 与 HNSW 的核心差异

HNSW 的问题不是没有 locality，而是 locality 可利用窗口太短。一个 query 的图搜索通常在毫秒级完成，单 query 内部又不适合拆给多个 CCD 执行。相似 query 调度只能在 batch 行级做文章，任何分组等待都会直接进入用户可见延迟。

更适合 chiplet locality 研究的场景应满足：

| 维度 | HNSW similar-query scheduler | 更理想场景 |
| --- | --- | --- |
| 单位粒度 | 单 query，`1-3 ms` | mini-batch、operator、compaction、sandbox step，`5 ms-秒级` |
| 可等待窗口 | 很小，通常只能短窗口聚合 | 可在 batch 队列、后台阶段或 pipeline 中摊销 |
| 相似性来源 | query path 重叠，需要在线估计 | ID 热度、图分区、session、文件路径、阶段元数据等更容易获得 |
| 共享对象 | HNSW 节点块和邻接表 | embedding row、feature row、邻接表、hash table、page cache、event state |
| 放置变量 | query 到 CCD queue | 数据副本、分片、任务 home domain、阶段 Local/Mixed、OS 路径分片 |
| 风险 | 调度开销覆盖收益 | 归因和工程复杂度上升，但调度预算更大 |

### 评价维度

每个候选方向按以下问题评估：

1. 系统解决的问题是什么。
2. 典型 workload 如何执行。
3. locality、共享工作集或重复访问从哪里来。
4. CCD-aware 调度为什么可能有效。
5. 单次请求、任务、批次或阶段的时间尺度。
6. 相比 HNSW 是否有更大的设计空间。
7. 调度、分组和路由开销能否摊销。
8. 可行的系统设计。
9. 可测指标和实验可实现性。
10. 与已有研究的关系和创新空间。
11. 研究难度、实验成本和潜在论文价值。

## 候选方向总览

| 方向 | 共享工作集来源 | 时间尺度 | 调度预算 | 实验成本 | 创新空间 |
| --- | --- | --- | --- | --- | --- |
| GNN sampling / feature gather | 邻域重叠、热节点、特征行、图社区 | `毫秒-秒级` mini-batch | 高 | 中 | 高 |
| Agent tool sandbox | page cache、dentry/inode、cgroup、日志、IPC、session 文件集 | `100 ms-秒级` | 很高 | 中高 | 高 |
| LSM/HTAP | memtable、Bloom/filter、compaction run、delta/main、hash/join 状态 | `毫秒-秒级` | 高 | 中高 | 中高 |
| 推荐系统 embedding lookup | 热 ID、热表、同 batch 重复 sparse feature | `100 us-几十 ms`，取决于 batch 和表规模 | 中高 | 低 | 低中 |
| HPC/CFD/FEA/stencil | mesh block、stencil tile、halo、sparse row block | `毫秒-分钟级` | 高 | 中高 | 中 |
| EDA/RTL/PDES | logical process、event queue、模块状态、时间戳局部性 | `毫秒-小时级` | 高 | 高 | 中高 |
| RPC/KV serving | queue、mempool、热 key 元数据、连接状态 | `微秒-毫秒` 请求，批次/流可更长 | 中 | 中 | 中 |
| OLAP worker placement | split、hash table、Page、shuffle buffer | `秒级` query stage | 高 | 中 | 低中 |
| k-mer counting | minimizer bucket、hash bucket、计数结构 | `秒级-小时级` | 高 | 中 | 中低 |

## 方向一：推荐系统 Embedding Lookup

### 背景与问题定义

推荐系统用于广告、内容排序、商品推荐和信息流排序。典型模型会同时处理两类特征：

- dense feature：连续数值特征，例如价格、年龄、统计计数。
- sparse feature：离散 ID 特征，例如用户 ID、商品 ID、广告 ID、类别 ID、上下文 token。

Embedding lookup 指把离散 ID 映射到向量。一个请求会包含多个 sparse feature，每个 feature 对应一个或多个 ID。系统从 embedding table 中读取这些 ID 对应的向量，再对同一字段内的多个向量求和或求平均。`EmbeddingBag`（嵌入向量袋）就是 PyTorch 中用于“读取多个 embedding 并做聚合”的算子。FBGEMM/TorchRec 中的 Table Batched Embedding（TBE，表批量嵌入）则把多个表、多个请求的查表聚合成高效批处理。

推荐模型通常由 sparse embedding 部分和 dense neural network 部分组成。dense 部分主要是矩阵计算，适合 GPU 或 CPU SIMD；sparse 部分主要是查表，访问位置由请求中的 ID 决定，随机性强。工业推荐模型的 embedding table 通常很大，远超单 CCD L3，甚至远超 LLC 聚合容量。请求之间的计算量差异也很大：有些请求只查少量 ID，有些请求包含较长的 bag 或多个字段。

现有系统通常从两个层级优化这类问题。第一层是 embedding bag operator 的计算任务划分，例如按 batch、table、CCD 或核心组划分任务，配合预取、SIMD、cache reuse 和负载均衡。第二层是服务系统优化，例如 batching、hot embedding cache、CPU/GPU 混合放置、表分片和请求调度。

`Parallelization Strategies for DLRM Embedding Bag Operator on AMD CPUs` 已经覆盖了第一层中的关键 chiplet 设计：TT（Table Threading，表级线程划分）把每张表交给单线程处理；HT（Hierarchical Threading，层次化线程划分）把表分配到 CCD，同一 CCD 内多个核心并行处理分配给该 CCD 的表，并避免同一表 hot rows 在跨 CCD 之间产生共享流量。因此，本文不能再把“表按 CCD 划分、CCD 内多线程查同一张表、利用 hot row cache reuse”作为创新点。若保留推荐系统方向，问题必须收窄到更高层的 serving runtime：多个独立请求、多个模型实例或多租户 workload 之间，是否还能在已有 HT/TT kernel 之上通过请求 co-location 和跨 batch hot row 管理获得增量收益。

### 核心 workload 与执行模式

在线 serving 路径通常是：

```text
请求进入 batch
  -> 解析 sparse feature
  -> 按 table / feature field 生成 ID 列表
  -> embedding lookup
  -> embedding bag reduce
  -> dense MLP 或 ranking tower
  -> 返回排序分数
```

训练路径还会包含反向传播和 embedding 更新。本文更推荐先研究 serving 或读多写少的 inference，因为热 row 副本和 per-CCD cache 不需要处理高频一致性。

典型执行单元不是单个 ID lookup，而是一个 batch 内多张表、多条请求的查表与聚合。实际时间尺度依赖 batch size、表数、embedding 维度和数据类型。单个 operator 可从数百微秒到数十毫秒不等；端到端推荐请求往往由多个 stage 组成，调度空间大于 HNSW 的单 query 图搜索。

### Locality 来源

Embedding lookup 的 locality 来自四个层面：

1. **热 ID 分布**：用户、商品、广告、类目等 ID 往往服从偏斜分布。热门 ID 被大量请求重复访问。
2. **字段稳定性**：同一 feature field 的 ID 范围、表、维度和访问模式稳定。
3. **batch 内重复**：同一 batch 中多个请求可能访问相同 hot ID 或同一组热门表。
4. **serving 读多写少**：在线推理中 embedding 权重多数时间只读，适合复制。

这些 locality 比 HNSW 的 query path overlap 更容易识别。调度器不需要先运行图搜索才能知道路径，只要读取 sparse ID、table id 或字段 id 即可获得 locality signature。

### CCD-aware 调度可能有效的原因

在 AMD Zen 平台上，每个 CCD 内核心共享本地 L3。若同一批请求中多个核心访问相同 hot embedding row，把它们放在同一 CCD 内执行，可以让该 row 的 cache line 在本地 L3 中复用。若默认线程池把同一 hot row 的访问分散到多个 CCD，系统会在多个 L3 中重复填充同一数据。

可用机制必须避开 HT 已覆盖的表级任务划分。可能仍有增量空间的机制包括：

- request-level co-location：在多个独立 embedding bag 调用之间，把 hot ID overlap 高的请求放到同一 CCD group 或同一模型实例队列。
- cross-batch hot row management：让跨 batch 反复出现的 hot rows 在固定 CCD group 内持续复用，而不是每个 batch 独立调度。
- multi-model / multi-tenant placement：多个推荐模型或多个租户共享同一 EPYC 时，按 hot table/hot ID overlap 做模型实例放置。
- SLO-aware bounded spill：本地队列过载时允许远端执行，并显式记录 locality 损失和 tail latency 权衡。

相比 HNSW，embedding 的优势是 signature 便宜、batch 天然存在、数据结构只读或低频更新。问题是已有 HT/TT 已经把最直接的 CCD-local table reuse 做掉，剩余空间必须来自 operator 外部，创新空间和可行性都低于 GNN sampling。

### 时间尺度与开销摊销

Embedding lookup 的单个 ID 读取是纳秒级到微秒级，但系统调度不应作用在单个 ID 上，而应作用在 table batch、request batch 或 feature field batch 上。一个 batch 中可包含几百到几万次 ID lookup，分组和路由开销可被整个 operator 摊销。

与 HNSW 对比：

- HNSW：每个 query 只有 `1-3 ms`，相似性判断本身可能吃掉预算。
- Embedding：signature 直接来自输入 ID，不需要额外图遍历；分组粒度是 batch/operator，不是单次随机读。
- HNSW：相似 query 是否访问相同节点需要实验采样。
- Embedding：hot ID 重复可直接统计，热度分布和复用率可离线/在线测量。

### 系统设计方式

若继续验证，最小系统设计应把 AMD HT/TT 作为 baseline，而不是从普通 embedding bag 线程池开始：

```text
多个待服务请求
  -> 解析 table id / sparse id
  -> 计算跨请求 hot-ID overlap
  -> 按 overlap 和 SLO 选择 model instance / CCD group
  -> 每个实例内部调用已有 HT/TT embedding bag kernel
  -> 记录跨 batch hot row reuse 与 tail latency
```

第一版不需要修改完整推荐系统，但必须使用强 baseline：

1. 生成或加载 DLRM/TorchRec 风格 sparse trace。
2. 支持 Zipf 热度、batch size、表数、embedding 维度、ID 重复率配置。
3. baseline 至少包含 BT、TT、HT 或等价的表到 CCD/core-group 任务划分。
4. 只在 HT/TT 之上增加 request-level grouping、cross-batch affinity 或 multi-model placement。
5. 使用 `numactl/taskset`、`pthread_setaffinity_np`、first-touch 控制线程和数据布局。

### 可测指标

- operator latency：p50/p95/p99。
- throughput：lookups/s、requests/s。
- realized bandwidth。
- hot ID reuse distance。
- batch 内 duplicate ID ratio。
- L3 miss / lookup。
- `another CCD` demand fill 占比。
- IBS load latency。
- owner queue imbalance。
- replica memory overhead。

### 与已有研究的关系

已有研究已覆盖：

- CPU embedding bag 并行、预取、超线程、缓存复用。
- GPU/CPU 混合 embedding cache。
- TorchRec/FBGEMM 的 table batched embedding。
- 热 ID 软件缓存和频率感知缓存。

其中 `Parallelization Strategies for DLRM Embedding Bag Operator on AMD CPUs` 覆盖的不只是 kernel 微优化，还包括表到 CCD/core-group 的计算任务划分。它的 HT 方案已经把表分配到 CCD，同一 CCD 内多个核心并行处理同一表，目标正是减少 shared hot rows 的跨 CCD 流量并提高 L3 复用。因此，它不是普通 baseline，而是强相关工作。

尚未充分覆盖的空间只剩更高层的 serving runtime：跨多个 embedding bag 调用、多个模型实例或多租户请求，是否能利用 hot-ID overlap 做 co-location。这个空间较窄，且必须证明 HT/TT 之后仍有跨请求可利用的 L3 驻留。论文贡献不能写成“embedding lookup 随机访存”，而只能聚焦：

```text
HT/TT embedding bag kernel 之上的 request-level co-location
  + cross-batch hot row affinity
  + multi-model / multi-tenant placement
  + SLO-aware bounded spill
```

### 难度与论文价值

实验成本低，但论文价值和可行性需要下调。直接的 CCD-aware 表级任务划分已被 AMD 论文覆盖；剩余的 request-level co-location 依赖三个条件：真实 serving 中存在多个可重排请求或多个模型实例；hot-ID overlap 能跨 batch 保持；额外排队和实例选择不会抬高 p99。若这三个条件不同时成立，该方向只能作为负结果或工程备忘，不适合作为第一主线。

## 方向二：GNN Neighbor Sampling 与 Feature Gather

### 背景与问题定义

GNN（Graph Neural Network，图神经网络）用于社交网络、推荐、欺诈检测、知识图谱和分子图。图中每个节点有特征，边表示关系。GNN 训练或推理时，一个目标节点需要聚合邻居节点的信息。

大图通常无法一次放入 GPU 显存，也不能对全图做 full-batch 训练。常用方法是 neighbor sampling（邻居采样）：每次从一批 seed nodes 出发，采样若干跳邻居，构造一个 mini-batch 子图，再把相关节点特征送入 GNN 模型。

在常见 CPU-GPU pipeline 中，CPU 负责采样邻居、读取邻接表、收集节点 feature、构造 mini-batch；GPU 负责神经网络前向和反向计算。已有 GNN 系统大量优化 feature caching、graph partitioning、sampling pipeline 和 CPU-GPU 数据搬运，说明采样和特征读取是实际瓶颈。

GNN 的数据结构通常由两部分组成。第一部分是图结构，例如 CSR（Compressed Sparse Row，压缩稀疏行）格式的邻接表，用于快速找到某个节点的邻居。第二部分是节点特征矩阵，每一行是一个节点的 feature vector。采样阶段主要访问邻接表，feature gather 阶段根据采样结果读取特征行。二者都不是规则连续扫描，访问位置由 seed nodes 和采样结果决定。

当前行业和学术界常见优化包括：把图按社区或顶点 ID 分区，缓存高频节点特征，把采样和 GPU 训练流水线化，提前预取下一批 mini-batch，把部分 feature cache 放在 GPU 显存中。chiplet 方向与这些优化不同，关注 CPU 侧采样和特征读取是否能在 CCD L3 内形成更高复用。

### 核心 workload 与执行模式

典型训练流程：

```text
seed batch
  -> 第 1 层邻居采样
  -> 第 2 层邻居采样
  -> 生成 sampled subgraph
  -> gather node features
  -> CPU 到 GPU 传输
  -> GNN forward/backward
  -> 下一 mini-batch
```

一个 mini-batch 可能包含几百到几千个 seed，扩展后涉及成千上万甚至更多邻居节点。执行时间通常是毫秒到百毫秒级，若图在磁盘或远端存储上则更长。相比 HNSW 的 `1-3 ms` 单 query，GNN 的调度预算明显更大。

### Locality 来源

GNN 的 locality 比 HNSW 更强，原因是：

1. **图社区结构**：真实图常有 community，同一批 seed 的邻域可能重叠。
2. **高频节点**：高入度或高重要性节点会被多个 batch 反复采样。
3. **feature row 复用**：同一节点 feature 可能被多个 seed 或多个 mini-batch 读取。
4. **采样可控**：seed batch 的组合方式会直接影响邻域重叠。
5. **训练 pipeline 可预取**：下一批 seed 通常可提前知道，允许预调度和预加载。

### CCD-aware 调度可能有效的原因

GNN 的 CPU 侧工作包括邻接表随机读、feature row gather 和 mini-batch 构造。这些都是典型的随机访存和共享热集问题。若把邻域重叠高的 seed 放在同一 CCD 的采样 worker 上，可以让邻接表和 feature row 在本地 L3 中复用。

可用机制：

- seed locality grouping：按图分区、社区、metis partition、node id range 或历史邻域 overlap 分组 seed。
- per-CCD adjacency cache：每个 CCD 缓存热节点邻接表。
- per-CCD feature hot cache：每个 CCD 缓存高频 feature row。
- partition-owned sampler：图分区有 home CCD，采样优先在 owner CCD 执行。
- split gather：跨分区 feature 由 owner CCD 本地读取后返回 compact buffer。

### 时间尺度与开销摊销

GNN 的调度粒度是 mini-batch，不是单个请求。即使每个 mini-batch 只有几毫秒，分组和路由可以提前在 DataLoader 阶段完成，并通过 pipeline 与 GPU 计算重叠。

与 HNSW 对比：

- HNSW query 到达后才知道实际访问路径；GNN seed list 在采样前已知。
- HNSW 单 query 的调度等待直接进入服务延迟；GNN mini-batch 可在训练管线中预取。
- HNSW 目标是降低 query latency；GNN 可用 sampling throughput、GPU wait time 和 batch ready latency 评估，优化空间更宽。

### 系统设计方式

推荐设计：

```text
seed queue
  -> locality signature: partition id / community id / hot neighbor set
  -> per-CCD seed queues
  -> local neighbor sampling
  -> local feature gather
  -> cross-CCD feature exchange only at batch boundary
  -> compact batch to GPU
```

第一版原型可基于 PyG NeighborLoader 或 DGL GraphBolt 外围实现，不必立即改框架内部：

1. 离线图分区，记录每个 seed 的 home partition。
2. DataLoader 生成 seed batch 前做 locality grouping。
3. 采样线程按 CCD 绑定，图 CSR 和 feature matrix first-touch 到分区 owner。
4. 热节点 feature 做 per-CCD replica。
5. 对比随机 seed batch、默认 DataLoader、partition grouping、hot feature replica。

### 可测指标

- sampling throughput。
- mini-batch ready latency。
- GPU wait time。
- end-to-end epoch time。
- feature gather bandwidth。
- sampled node duplicate ratio。
- seed batch neighborhood overlap。
- L3 miss / sampled edge。
- remote CCD demand fill。
- feature cache hit ratio。
- 模型精度或收敛曲线。

### 与已有研究的关系

已有 GNN 系统如 PaGraph、GNNLab、SALIENT、DistDGL、GraphBolt 等已经研究 feature cache、GPU cache、graph partitioning、pipeline 和 sampling 加速。它们主要面向 GPU 显存、CPU-GPU 传输或分布式训练。

未充分研究的是：在单机 AMD Zen 多 CCD CPU 上，CPU-side sampling 和 feature gather 如何利用分片 L3、本地热 feature cache 和 per-CCD seed scheduling。创新点应避免重复“图分区提升 locality”，而要落在：

```text
mini-batch seed scheduling
  + per-CCD feature/adjacency hot cache
  + CPU-side sampling pipeline
  + GPU wait time reduction
```

### 难度与论文价值

研究价值高，时间尺度合适，locality 强。难点是 GNN 框架复杂，端到端收益需要证明 GPU 没有完全掩盖 CPU 侧优化。建议作为第一阶段主线：先用 CPU-only sampling microbenchmark 验证，再接 GPU pipeline。

## 方向三：Agent Tool Sandbox 与共享 OS 路径

### 背景与问题定义

Agent 指由 LLM 驱动、能够多轮调用工具完成任务的系统。代码修复、数据分析、浏览器操作和企业自动化 agent 都会运行外部命令或工具。为了安全，工具通常在 sandbox 中执行，例如容器、namespace、cgroup、微虚拟机或预热环境池。

Agent 的端到端开销不只来自 LLM 推理。近年测量论文显示，tool call、容器初始化和 OS 资源管理可以占据很高比例的任务延迟；内存峰值、文件系统访问、日志、IPC 和 cgroup 管理会成为并发瓶颈。

与普通 LLM serving 不同，Agent workload 是“模型推理 + 外部工具执行 + OS 资源管理”的混合系统。一次用户任务可能包含多轮 `grep`、Python 脚本、测试命令、浏览器动作或数据库查询。每个 tool call 都会触发进程创建、文件访问、依赖加载、日志收集和权限控制。并发 Agent 系统还需要同时管理多个 sandbox 的 CPU、内存、I/O 和文件系统状态。

当前系统通常从资源隔离和生命周期管理入手优化：预热 sandbox 池降低冷启动，使用 cgroup 限制资源，复用容器镜像和 page cache，按任务队列做 admission control，或把 LLM 推理和工具执行流水线化。chiplet 方向不应把每个 sandbox 的私有内存当作共享工作集，而应关注多个 sandbox 共同访问的 OS 控制路径和同一 session 的文件缓存复用。

### 核心 workload 与执行模式

典型代码 agent 的循环：

```text
LLM 生成 tool call
  -> agent runtime 解析命令
  -> 选择 sandbox
  -> 准备工作目录和文件系统视图
  -> 启动 grep/python/pytest/npm/mypy 等工具
  -> 收集 stdout/stderr/log
  -> 返回 observation 给 LLM
  -> 下一轮
```

单个 tool call 常为 `100 ms` 到数秒。一个任务包含多轮 tool call，甚至多次重试。时间尺度远大于 HNSW，调度、分片和 session 亲和开销更容易摊销。

### Locality 来源

Agent 方向的关键不是 sandbox 私有堆栈。私有数据若由本地 first-touch 分配并固定进程，不会天然造成跨 CCD 共享热点。

真正可能产生 chiplet 相关性的对象是共享 OS 路径：

- VFS（虚拟文件系统）路径解析。
- dentry/inode 元数据缓存。
- page cache 中的 repo 文件、依赖库、测试文件。
- overlayfs lower/upper/work 元数据。
- cgroup 计数器和资源统计。
- pipe、Unix socket、日志队列和 IPC broker。
- 同一 session 反复访问的 repo、venv、node_modules、pytest cache。

### CCD-aware 调度可能有效的原因

若所有 sandbox 共用一个 cgroup subtree、一个日志队列、一个 tmp/workdir 根目录和一个 IPC acceptor，多 CCD 上的 worker 会反复访问同一批内核对象或用户态队列，可能造成 cache line bounce、锁争用和远端页访问。

可用机制：

- per-domain sandbox pool：每个 CCD/NPS/socket 一个 sandbox 池。
- per-domain cgroup subtree：减少共享 cgroup 计数器竞争。
- per-domain tmp/log/workdir：减少文件系统元数据热点。
- per-domain IPC acceptor：分散 pipe/socket buffer 和日志队列。
- session affinity：同一 repo/session/tool-class 固定 home domain。
- page-cache affinity：同一 session 的后续 tool call 尽量复用本域 page cache。

### 时间尺度与开销摊销

Agent tool call 的执行时间通常远大于调度开销。即使分配 home domain、选择 sandbox pool、写入 per-domain log queue 需要几十微秒到数百微秒，也可被 `100 ms-秒级` tool 执行摊销。

与 HNSW 对比：

- HNSW 要优化用户可见的毫秒级 query。
- Agent 优化的是 tool call、sandbox step 或 session 级执行。
- HNSW locality 在只读索引内；Agent locality 在 OS 共享路径和 page cache 内。
- Agent 论文新颖性高，但硬件归因更难。

### 系统设计方式

推荐从 counterfactual 做起，不宜直接实现复杂 agent runtime：

```text
baseline:
  global sandbox pool + global cgroup/log/tmp/IPC

counterfactual:
  per-domain sandbox pool
  per-domain cgroup subtree
  per-domain tmp/log/workdir
  per-domain IPC/log queue
  session-to-domain affinity
```

实验 workload 可选择：

- 多个 repo 上并发 `pytest`。
- 同一 repo 上连续 `grep -> edit -> pytest -> mypy -> pytest`。
- 多租户 sandbox 并发执行 Python/Node 工具。
- 与 vLLM CPU serving 混部，测 TTFT 和 decode p99 干扰。

### 可测指标

- tool call p50/p95/p99。
- sandbox cold/warm start latency。
- task completion time。
- page fault、major fault、workingset refault。
- `perf c2c` HITM 热点。
- lock contention。
- cgroup/memcg 更新热点。
- dentry/inode slab 增长。
- PSI 中 CPU/memory/io pressure。
- vLLM TTFT/decode p99 干扰。

### 与已有研究的关系

AgentCgroup、CPU-centric agentic AI、agent serving 论文关注 OS 资源、tool execution、cgroup 控制、workflow scheduling 和 agent throughput。它们尚未系统研究 AMD EPYC 多 CCD 拓扑下的共享 OS 路径分片。

创新点可以是：

```text
agent tool execution
  + shared OS path coherence
  + session/page-cache affinity
  + chiplet/NUMA domain as isolation boundary
```

### 难度与论文价值

应用新，时间尺度理想，潜在论文价值高。风险也高：系统噪声大，热点可能在磁盘、解释器初始化、网络或外部服务，而不是 CCD/L3。只有在 `perf c2c`、lock、NUMA、page cache 指标证明共享 OS 路径进入关键路径后才值得深入。

## 方向四：LSM 与 HTAP 数据库阶段调度

### 背景与问题定义

LSM（Log-Structured Merge-tree，日志结构合并树）用于 RocksDB、LevelDB、Pebble 等存储引擎。写入先进入内存中的 MemTable，随后 flush 成磁盘或内存中的有序 run，再通过 compaction 把多个 run 合并，控制读放大和空间放大。

HTAP（Hybrid Transactional/Analytical Processing，混合事务分析处理）系统同时处理事务更新和分析查询。典型结构包括热 delta、冷 main、后台 merge、scan、join 和 aggregation。

这类系统的关键特点是阶段边界清晰，阶段时间尺度较长。与 HNSW 的短 query 不同，flush、compaction、delta merge、join build/probe 可以持续毫秒到秒级。

LSM 的设计目标是把随机写转换为顺序写。代价是数据会分散在多个 level 和多个 SST 文件中，读请求需要查询 MemTable、Bloom filter、index block 和多个数据文件。Compaction 是后台维护过程，它把多个有序 run 合并成更少的 run，减少读放大和空间放大，但会消耗大量 CPU、内存带宽和 I/O。

HTAP 的目标是在同一系统中同时服务短事务和长分析查询。常见做法是把最近更新放在较小的 delta 区，把稳定历史数据放在较大的 main 区，后台周期性 merge。数据库系统通常通过 compaction policy、block cache、Bloom filter、NUMA-aware memory placement、前后台任务隔离来优化。chiplet 方向可以利用这些阶段边界，把不同阶段映射到不同 CCD 放置策略。

### 核心 workload 与执行模式

LSM 路径：

```text
foreground write
  -> MemTable insert
  -> flush MemTable to SST
  -> background compaction
  -> build index/filter/compression block
  -> foreground read consult MemTable + SST + Bloom/filter + block cache
```

HTAP 路径：

```text
OLTP update
  -> append to delta
  -> analytical scan on main + delta
  -> periodic delta merge
  -> join / aggregation / materialized view update
```

### Locality 来源

- MemTable、skiplist、hash index、Bloom filter 是小而热的结构。
- small compaction 的 active set 可能接近单 CCD L3。
- large compaction 是多路 merge，输入远超 L3，但 merge heap、block index、filter、输出 buffer 有局部状态。
- HTAP delta 较热且较小，main 较冷且大。
- hash join build table 或 aggregation state 的大小可能落在单 CCD L3 与聚合 L3 之间。

### CCD-aware 调度可能有效的原因

可以按阶段选择 Local 或 Mixed：

- 小 MemTable、small flush、small compaction：收紧到单 CCD，获得低延迟本地 L3。
- 中等 hash table、filter build、delta merge：扩展到多个 CCD，利用聚合 L3 容量。
- 大 scan、大 compaction：转向 DRAM bandwidth 和 NUMA locality，不强求 L3。
- 前台短请求与后台 compaction 隔离到不同 CCD，降低 p99 干扰。

### 时间尺度与开销摊销

compaction、flush、merge 都是粗粒度任务，调度开销可以忽略不计。即使 PMU 采样、任务分类和队列迁移需要毫秒级，也可被秒级后台任务摊销。

相比 HNSW，设计空间更大：

- 可以在 task 开始前估算输入 run 大小、level、filter/index 大小。
- 可以在后台任务上试错，不直接影响单请求正确性。
- 可用 phase metadata 辅助调度，不需要昂贵相似性判断。

### 系统设计方式

```text
DB task scheduler
  -> classify: MemTable / flush / small compaction / large compaction / scan / join
  -> estimate active working set
  -> choose Local / Mixed / bandwidth mode
  -> bind threads and memory
  -> collect PMU feedback
  -> adjust thresholds
```

RocksDB 原型可从独立 compaction benchmark 或 `db_bench` 开始。HTAP 可先做简化 microbenchmark：point update + scan + periodic delta merge。

### 可测指标

- compaction throughput。
- write stall time。
- foreground read/write p99。
- flush latency。
- scan throughput。
- L3 miss/op。
- DRAM bandwidth。
- `another CCD` demand fill。
- background/foreground interference。

### 与已有研究的关系

OLAP on Chiplet、CHARM、P-MOSS、排序论文已经证明数据库类 workload 会受 chiplet 放置影响。RocksDB/LSM 研究大量关注 compaction policy、I/O、write amplification、block cache 和 Bloom filter。

创新点应是：

```text
phase-aware chiplet scheduling for storage engine internals
  + active working-set estimation
  + foreground/background isolation
  + PMU feedback
```

### 难度与论文价值

数据库系统价值明确，时间尺度合适。风险是工程复杂，端到端收益可能被 SSD、压缩、锁或 write stall 机制掩盖。该方向适合作为系统备选，不建议先于 embedding/GNN。

## 方向五：HPC、CFD、FEA 与 Stencil

### 背景与问题定义

CFD（Computational Fluid Dynamics，计算流体力学）、FEA（Finite Element Analysis，有限元分析）和 stencil 计算是技术计算中的典型内存敏感 workload。它们在网格、单元、稀疏矩阵或邻域模板上反复迭代。

AMD 3D V-Cache 面向 CFD、FEA、EDA 等 workload 的公开资料和近期 CFD 评测表明，L3 容量会显著影响部分技术计算应用。CFD 论文使用 OpenFOAM motorBike 和 Urban Air Pollution 等 memory-bound 模型，分析不同 EPYC 架构和 L3 容量对性能的影响。

这类应用通常把物理空间离散成网格或单元。每个时间步或迭代步中，程序读取一个 cell 及其邻居的状态，计算新的速度、压力、温度、应力或其他物理量。Stencil（模板计算）是这种邻域访问的简化抽象，例如每个网格点读取上下左右或三维邻居。FEA 和 CFD 中还常见稀疏矩阵向量乘、预条件器和迭代求解器。

当前优化主要包括 domain decomposition（区域分解）、MPI rank mapping、OpenMP thread affinity、cache blocking、NUMA first-touch、halo exchange 批量化和向量化。chiplet 方向需要在这些已有优化之上进一步判断：网格块或稀疏块的活跃工作集是否能落入单 CCD L3 或少数 CCD 聚合 L3。

### 核心 workload 与执行模式

典型流程：

```text
mesh/grid partition
  -> 每个 time step 遍历 cell/block
  -> 读取邻近 cell 或 sparse row
  -> 更新物理量
  -> halo exchange
  -> convergence check
```

每个 step 中，同一网格块及其 halo 会被多次访问。任务时间从毫秒到分钟不等，取决于网格规模和迭代次数。

### Locality 来源

- 空间邻域访问稳定。
- 网格块、stencil tile 或 sparse block 可以控制大小。
- halo 数据在边界重复使用。
- 小 block 可能装入单 CCD L3。
- 中等 block 可能适合跨少数 CCD 的聚合 L3。

### CCD-aware 调度可能有效的原因

若 block working set 小于 32 MiB，绑定到单 CCD 可提高本地 L3 命中。若 block 介于单 CCD L3 和多个 CCD 聚合 L3 之间，可使用 Mixed 策略。若 block 远超聚合 L3，则应优先 NUMA 和 DRAM bandwidth。

可用机制：

- mesh block per-CCD ownership。
- stencil tile size 与 CCD L3 容量匹配。
- halo exchange 显式批量化。
- 小 block Local，中 block Mixed，大 block bandwidth mode。
- OpenMP thread affinity 与 first-touch 控制。

### 时间尺度与开销摊销

HPC 任务时间尺度远大于 HNSW。调度和分区开销可在迭代或 time step 间摊销。问题不在开销预算，而在已有优化强度很高，chiplet 增量需要超过成熟的 NUMA、MPI、OpenMP、cache blocking baseline。

### 可测指标

- time step time。
- cell updates/s。
- solver iteration time。
- L3 miss/cell。
- memory bandwidth。
- halo exchange time。
- scaling efficiency。
- OpenFOAM 或 proxy-app 端到端时间。

### 与已有研究的关系

已有 HPC 领域长期研究 cache blocking、domain decomposition、NUMA placement、MPI rank mapping 和 OpenMP affinity。AMD 3D V-Cache 资料已经证明大 L3 对部分 CFD/FEA 有收益。

创新空间在于 CCD 级 block scheduling 与 Local/Mixed 自适应，而不是证明“大 L3 有用”。若没有具体应用知识，该方向容易变成调参实验。

### 难度与论文价值

硬件事实强，应用价值高，但领域门槛高。适合作为与 HPC 团队合作的方向，不适合作为当前项目的第一主线。

## 方向六：EDA、RTL 与并行离散事件仿真

### 背景与问题定义

EDA（Electronic Design Automation，电子设计自动化）中的 RTL simulation、functional simulation 和 verification 会模拟大量模块、信号和事件。并行离散事件仿真（PDES，Parallel Discrete Event Simulation）也有类似结构：系统由多个 logical process 组成，每个 process 维护局部状态和事件队列，事件按时间戳传播。

这类 workload 的执行时间通常很长，状态访问密集，事件传播具有局部性。AMD 和 EDA 工具厂商的资料显示，部分 EDA 仿真能从 3D V-Cache 获益。

RTL（Register Transfer Level，寄存器传输级）仿真把硬件设计表示为寄存器、组合逻辑和时钟周期上的信号变化。仿真器维护每个模块的状态，按时间戳处理事件：某个信号变化后，可能触发下游逻辑重新计算，并产生新的事件。PDES 把类似结构抽象为 logical process，每个 process 有本地状态和事件队列，跨 process 事件需要通信和同步。

当前优化通常围绕事件队列、模块划分、增量求值、并行调度、NUMA placement 和同步开销展开。商用 EDA 工具还会针对特定 CPU cache 容量、内存带宽和编译后的仿真代码做优化。chiplet 方向的切入点是把强通信模块或 logical process 放到同一 CCD，并把跨 CCD 事件从细粒度共享队列变成批量交换。

### 核心 workload 与执行模式

```text
event queue
  -> 取出当前时间戳事件
  -> 更新 module/process state
  -> 产生新事件
  -> 投递到本地或远端 process queue
  -> barrier / synchronization / rollback
```

### Locality 来源

- 强通信模块之间反复传递事件。
- module/process state 会被局部事件反复访问。
- 每个 logical process 有本地 event queue。
- 全局 priority queue 或同步结构可能成为共享热点。

### CCD-aware 调度可能有效的原因

可以把强通信的 logical process 映射到同一 CCD，将本地事件队列和模块状态放在本地 L3。跨 CCD 事件通过显式 batch 交换，避免细粒度共享队列的 cache line bounce。

可用机制：

- process-to-CCD graph partitioning。
- per-CCD local event queue。
- cross-CCD event batching。
- hierarchical barrier。
- hot module state local placement。

### 时间尺度与开销摊销

仿真任务通常为秒级到小时级。调度、分区和 profiling 开销非常容易摊销。问题在于 correctness、determinism 和工具可改性。

### 与已有研究的关系

PDES 研究长期关注 event locality、NUMA-aware priority queue、logical process mapping 和同步机制。EDA 商用工具多为闭源，只能通过外部绑核和配置做有限控制。开源仿真器可改，但代表性可能不足。

### 难度与论文价值

潜在价值高，时间尺度理想。实验成本和领域门槛高。若能找到可改开源 RTL/PDES benchmark，该方向有论文空间；否则不建议优先投入。

## 方向七：网络 I/O、RPC 与 KV Serving

### 背景与问题定义

网络 I/O、RPC 和 KV serving 处理大量短请求。典型系统包括 DPDK 用户态网络栈、eRPC、memcached、RocksDB-backed KV、Redis-like server 等。请求经过 NIC RX queue、polling core、worker queue、mempool、应用处理和 TX queue。

这类系统的单请求可能是微秒到毫秒级，看似与 HNSW 一样短。但它们通常以 flow、queue、worker pool 和 request class 为单位运行，调度粒度可比单请求更粗。

DPDK 这类用户态网络栈通常让固定核心轮询 NIC queue，绕过内核网络栈以降低延迟。RPC 系统在网络包之上提供请求-响应语义，KV serving 则在应用层处理 get、set、scan、delete 等操作。系统性能取决于单次处理速度、queue 分配、mempool 缓存、连接状态、热 key、长短请求混跑和 tail latency。

当前优化通常包括 RSS 或 flow steering 把流导向不同 queue，polling core 绑核，per-core mempool cache，small/large request 分队列，hot key 缓存，work stealing 和 admission control。chiplet 方向可以把这些 queue、worker 和 mempool 组织成 CCD-local group，但不能假设远端 CCD L3 是比 DRAM 更好的容量池。

### 核心 workload 与执行模式

```text
NIC RX queue
  -> polling core
  -> packet buffer / mempool
  -> RPC dispatch
  -> worker handles get/set/scan
  -> response buffer
  -> TX queue
```

### Locality 来源

- per-flow 连接状态。
- mempool cache。
- hot key metadata。
- slab class metadata。
- request class 队列。
- small/large request 分离后形成稳定性更高的工作集。

### CCD-aware 调度可能有效的原因

可将 `RX/TX queue -> polling core -> worker group -> mempool` 固定到拓扑域，减少 queue 共享、mempool 远端页和 worker 迁移。KV 元数据可复制到每个 CCD，大 value 或冷 value 分片。

需要注意：TiNA 的 remote-tier DCA 思路不适合直接迁移到 AMD Zen。Zen 上跨 CCD L3/cache 路径接近本地 DRAM，不能把远端 CCD L3 当作比 DRAM 更优的容量池。保留的是本地分区、tail 隔离和 bounded spill，不是远端 cache 容量套利。

### 时间尺度与开销摊销

单请求很短，不能做复杂 per-request 相似性判断。但可以按 flow、queue、request class 和 worker pool 做静态或低频动态调度。相比 HNSW，优势是 routing key 更清楚；劣势是 p99 SLO 更紧。

### 系统设计方式

- per-CCD queue/worker/mempool。
- small RPC 与 large RPC 分队列。
- hot metadata per-CCD replica。
- cold value shard。
- tail-latency bounded spill/steal。
- per-domain credit/window 控制。

### 可测指标

- QPS。
- p50/p95/p99/p999 latency。
- queue depth。
- mempool cache miss。
- request class head-of-line blocking。
- L3 miss/request。
- remote CCD demand fill。
- packet drop/retransmit。

### 难度与论文价值

网络/RPC 系统论文价值存在，但该方向偏 tail scheduling 与 queue isolation，不是最纯粹的 shared working-set exploitation。作为 HNSW 替代主线不如 embedding/GNN。

## 方向八：OLAP 查询引擎 Worker 部署

### 背景与问题定义

OLAP（Online Analytical Processing，联机分析处理）查询引擎执行 scan、filter、join、aggregation 和 shuffle。Presto、SparkSQL、SingleStore、DuckDB 等系统会把 SQL 拆成 stage、task、driver 和 operator pipeline。

已有 OLAP on Modern Chiplet-Based Processors 论文已经系统研究了 WIM、WIN、WICP、WIC 等 worker 部署策略，并证明每 chiplet 一个 worker 可改善局部性和扩展性。

OLAP 系统面向大规模分析查询，不是单条记录读写。一个查询可能扫描数 GB 到数 TB 数据，先过滤，再按 key 做 join 或 aggregation。分布式或多 worker 查询引擎会把表切成 fragment，把查询计划拆成多个 stage，每个 worker 处理自己的 split，并在 shuffle 或 exchange 阶段交换中间结果。

当前优化包括列式存储、向量化执行、predicate pushdown、hash join、Bloom filter、runtime filter、NUMA-aware worker 部署和 exchange 优化。chiplet 相关研究已经证明 worker 边界与 CCD 对齐可以减少无结构的跨 CCD 共享内存访问，因此该方向更适合作为已有方法线基线。

### 核心 workload 与执行模式

典型查询路径：

```text
SQL query
  -> coordinator 生成 query plan
  -> scan/filter 读取表 fragment
  -> build hash table 或 aggregation state
  -> probe / aggregate
  -> shuffle 或 exchange 中间结果
  -> 汇总结果
```

时间尺度通常是秒级到分钟级。其主要共享工作集来自 worker 内部的 Page、hash table、operator state 和 shuffle buffer。

### Locality 来源

- Worker 内部 task queue、Page、hash table、Bloom filter、scan state。
- Join build/probe 的局部 hash table。
- Repartition/shuffle buffer。
- 表 fragment 与 worker 的放置关系。

### CCD-aware 调度可能有效的原因

Worker 边界与 CCD 对齐后，局部状态更可能由同一 CCD 内的线程生产和消费，减少无结构的跨 CCD cache coherence。中等工作集可用 WICP_Mixed 获取聚合 L3。

### 时间尺度与开销摊销

SQL query stage 常为秒级，调度开销可忽略。相比 HNSW，这是理想时间尺度。

### 研究空间判断

该方向的问题是已有工作覆盖较充分。若继续做，需要找到不同于 WICP 的新对象，例如：

- query-stage 内 Local/Mixed 自适应。
- per-operator PMU feedback。
- HTAP/LSM 与 OLAP 的阶段统一调度。
- 与 vLLM/RAG 混部时的资源隔离。

不建议把“OLAP worker per CCD”作为新主线。

## 方向九：Genomics k-mer Counting

### 背景与问题定义

k-mer counting 是生物信息学中的基础任务。系统读取 DNA 序列，把长度为 k 的子串计数，用于组装、纠错、分类和比较。常见实现使用 hash table、Bloom filter、sort/compact、minimizer partition 等方法。

DNA 序列可以看作由 A/C/G/T 四种字符组成的长字符串。长度为 k 的连续子串称为 k-mer。相邻 k-mer 共享 `k-1` 个字符，因此原始数据中存在强顺序相关性；但计数时通常要对 k-mer 做 hash 或按 minimizer 分桶，访问会转化为大量哈希表更新、桶写入和后续合并。

当前优化通常包括压缩输入、minimizer-based partitioning（按代表性短子串分桶）、cache-friendly hashing、Bloom filter 过滤低频项、外部排序和磁盘 spill 控制。chiplet 方向应关注 bucket 内计数结构或 hot minimizer metadata 是否能在 CCD L3 内复用，而不是整份 genomics 数据是否能放入 cache。

### 核心 workload 与执行模式

```text
读取 reads
  -> 生成 k-mers
  -> 计算 minimizer 或 hash
  -> 写入 bucket / hash table
  -> 合并计数
  -> sort / compact
```

### Locality 来源

- 相邻 k-mer 高度重叠。
- minimizer 将相近 k-mer 聚到同一 bucket。
- 热 minimizer bucket 会被反复访问。
- bucket 内 hash table 或计数结构可局部化。

### CCD-aware 调度可能有效的原因

可按 minimizer bucket 将任务分配到 CCD，bucket 内计数结构本地化。跨 CCD 合并只在 batch 边界进行。热 bucket 可以复制或分裂。

### 时间尺度与开销摊销

任务时间通常很长，调度开销容易摊销。但总体数据量远超 L3，很多工具瓶颈在 I/O、压缩、磁盘 spill 或 DRAM bandwidth。只有 bucket metadata 和中等大小计数结构适合 L3。

### 研究空间判断

可做，但应用域窄。若目标是通用系统论文或 AI serving 相关方向，优先级低。

## 与 HNSW 的系统性对比

| 方向 | 时间尺度 | 相似性识别成本 | 调度开销摊销 | 预期收益来源 | 实现难度 | 相比 HNSW 的优势 |
| --- | --- | --- | --- | --- | --- | --- |
| HNSW | `1-3 ms/query` | 中，需要 query signature 或路径采样 | 低 | 相似 query 共享节点块 | 已有基础 | 已观察现象，但预算小 |
| GNN sampling | `毫秒-秒级/mini-batch` | 中，seed/partition 可预知 | 高 | 邻域与 feature row 复用 | 中 | mini-batch 可预取，GPU wait time 可衡量 |
| Agent sandbox | `100 ms-秒级/tool call` | 低，session/repo/tool-class 明确 | 很高 | OS 路径和 page cache 亲和 | 中高 | 时间预算最大，应用新 |
| LSM/HTAP | `毫秒-秒级/stage` | 低，阶段元数据明确 | 高 | 阶段工作集与前后台隔离 | 中高 | 不受单请求 ms 限制 |
| Embedding lookup | `100 us-几十 ms/operator` | 低，直接读取 table/id | 中高 | 跨请求 hot ID 复用 | 中 | signature 便宜，但表级 CCD 复用已被 HT/TT 覆盖 |
| HPC/stencil | `毫秒-分钟级` | 低，block/tile 已知 | 高 | mesh block 与 halo 复用 | 中高 | 时间尺度理想，L3 证据强 |
| EDA/PDES | `秒级-小时级` | 中，process/event graph 可分析 | 高 | local state 与 event queue | 高 | 调度预算充足 |
| RPC/KV | `微秒-毫秒/request` | 低，flow/key/class 明确 | 中 | queue/mempool/key metadata | 中 | routing key 清楚，但 SLO 紧 |
| OLAP | `秒级/query stage` | 低，operator/stage 已知 | 高 | worker 局部状态 | 中 | 已有研究覆盖 |
| k-mer | `秒级-小时级` | 低，bucket/minimizer 已知 | 高 | bucket locality | 中 | 时间尺度好，应用窄 |

## 排序与优先级分析

### P0：GNN neighbor sampling / feature gather

选择理由：

- 图邻域和 feature row 共享工作集天然存在。
- mini-batch 时间尺度明显大于 HNSW。
- DataLoader 可以提前分组和预取。
- 指标完整：sampling throughput、GPU wait time、epoch time、模型精度。
- 论文 framing 清楚：chiplet-aware CPU-side data pipeline for GNN。

主要风险：

- 框架复杂。
- 端到端收益可能被 GPU 计算或 PCIe 传输掩盖。
- 已有 GNN partition/cache 研究多，需要突出 CCD/L3 贡献。

### P1：Agent tool sandbox

选择理由：

- 时间尺度最大。
- Agent 系统快速发展，现有工作尚少触及 chiplet 拓扑。
- OS 共享路径分片与 session/page-cache 亲和是新问题，不是 HNSW/graph locality 的重复。

主要风险：

- 归因难。
- 噪声大。
- 需要先证明热点来自共享 OS 路径，而不是外部 I/O 或解释器初始化。

### P2：LSM/HTAP

选择理由：

- 阶段长，调度预算充足。
- 与 CHARM、OLAP、sorting 方法线自然衔接。
- foreground/background isolation 有实际系统价值。

主要风险：

- 工程成本高。
- 贡献容易被认为是已有 chiplet DB 调度的扩展。
- 端到端收益可能被 I/O 和压缩掩盖。

### P3：推荐系统 embedding lookup

选择理由：

- 与当前 AI serving 语境接近。
- 输入 ID 自带 locality signature，不需要复杂相似性判断。
- 可用合成 trace 快速验证。
- 剩余可做空间集中在 HT/TT kernel 之上的 request-level co-location、cross-batch affinity 和 multi-model placement。

主要风险：

- AMD HT/TT 已经覆盖表到 CCD/core-group 的任务划分和同表 hot row 的 CCD 内复用。
- 真实 serving batch 可能很小，额外分组等待会抬高 p99。
- 若 HT/TT 已接近 LLC/DDR 带宽上限，operator 外部调度难以产生可发表增量。

### P4-P8

这些方向均有研究空间，但不适合作为当前第一主线：

- HPC/stencil：硬件证据强，但领域门槛高。
- EDA/PDES：方向新，工具可改性差。
- RPC/KV：SLO 紧，容易偏离 shared working-set exploitation。
- OLAP：已有研究覆盖充分。
- k-mer：应用域较窄。

## 推荐研究路线

### 第一阶段：GNN Sampling 原型

目标：验证更长时间尺度下，seed locality grouping 和 per-CCD feature cache 是否降低 CPU sampling/gather 开销和 GPU wait time。

实验设计：

1. 选择 OGBN 或公开大图数据集。
2. 使用 PyG/DGL 的 neighbor sampling pipeline。
3. 离线图分区，生成 seed locality signature。
4. 实现 partition-aware seed batch。
5. 实现 per-CCD hot feature replica 或 feature cache。
6. 对比随机 seed、默认 DataLoader、partition grouping、hot cache。

指标：

- mini-batch ready latency。
- sampling throughput。
- GPU wait time。
- epoch time。
- feature cache hit ratio。
- L3/DRAM/CCD 指标。
- 精度和收敛曲线。

判停条件：

- seed grouping 不提高邻域重叠。
- CPU sampling 不是瓶颈。
- GPU wait time 不下降。
- 分组改变训练随机性导致精度问题。

论文 framing：

```text
Chiplet-aware GNN input pipeline:
将 seed scheduling、图分区和 per-CCD feature cache 结合，利用 CPU chiplet L3 缓解大图 mini-batch 采样与特征读取瓶颈。
```

### 第二阶段：Agent Tool Sandbox 高风险验证

目标：判断共享 OS 路径是否存在跨 CCD coherence 或锁竞争热点。

实验设计：

1. 构造多 sandbox 并发执行 `grep/pytest/mypy/npm test`。
2. 对比 global sandbox pool 与 per-domain pool。
3. 分片 cgroup、tmp/log/workdir、IPC acceptor。
4. 固定 session home domain。
5. 使用 `perf c2c`、lock contention、page fault、NUMA stats、PSI 归因。

判停条件：

- 热点不在共享 OS 路径。
- per-domain 分片不降低 p99。
- page-cache affinity 不降低 fault/refault 或运行时间。
- 工具执行主要受外部 I/O 支配。

论文 framing：

```text
Chiplet-aware OS path sharding for AI agents:
把 agent tool execution 的共享内核路径和 session page cache 映射到拓扑域，降低多租户 sandbox 并发下的共享状态争用和尾延迟。
```

### 第三阶段：LSM/HTAP 阶段调度验证

目标：验证数据库后台阶段和前台请求能否通过 CCD-aware phase scheduling 获得稳定收益。

实验设计：

1. 先做独立 compaction/merge microbenchmark，控制输入 run 大小、filter/index 大小和输出 buffer。
2. 对比默认线程池、单 CCD Local、多 CCD Mixed、NUMA bandwidth mode。
3. 在 RocksDB `db_bench` 或简化 HTAP microbenchmark 中加入前台读写和后台 compaction/merge 干扰。
4. 使用 PMU/IBS 区分 L3 容量 miss、DRAM bandwidth、跨 CCD demand fill 和锁/队列干扰。

判停条件：

- compaction/merge 对 L3 容量或 CCD 放置不敏感。
- 后台阶段改善不能转化为 foreground p99 或 write stall 改善。
- 收益主要来自 I/O 或压缩参数变化，无法归因到 chiplet locality。

论文 framing：

```text
Phase-aware chiplet scheduling for storage engines:
利用数据库阶段元数据和硬件计数器，在 Local/Mixed/bandwidth 模式之间切换，降低后台阶段对前台请求的尾延迟干扰。
```

## 最终建议

若目标是做一篇“利用 chiplet locality 的系统优化研究”，下一步最值得投入的是 **GNN neighbor sampling / feature gather**。原因是：

1. mini-batch 粒度明显大于 HNSW query，调度、分组和预取开销更容易摊销。
2. seed nodes、图分区、邻域 overlap 和 feature row 访问天然提供 locality signature。
3. CPU 侧 sampling 和 feature gather 是真实 GNN pipeline 中的可测瓶颈。
4. 相比推荐系统 embedding，GNN 的 chiplet-aware seed scheduling 没有被现有 AMD embedding bag HT/TT 工作直接覆盖。
5. 可先做 CPU-only sampling/gather microbenchmark，再扩展到 GPU wait time 和 epoch time。

推荐系统 embedding lookup 不再作为第一主线。`Parallelization Strategies for DLRM Embedding Bag Operator on AMD CPUs` 已经覆盖表到 CCD/core-group 的任务划分、CCD 内多线程查同一张表、hot row cache reuse 和负载均衡。剩余 request-level co-location 空间较窄，只适合在有真实 serving trace 和 HT/TT baseline 后作为补充验证。

Agent tool sandbox 不建议作为第一实验，但建议保留。它不是传统 L3 数据复用问题，而是共享 OS 路径和 session locality 问题。若早期归因能证明跨 CCD 共享状态进入关键路径，它可能形成更有新意的系统论文。

## 参考资料

- PyTorch EmbeddingBag：<https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html>
- TorchRec introduction：<https://docs.pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html>
- FBGEMM Table Batched Embedding：<https://docs.pytorch.org/FBGEMM/fbgemm_gpu/cpp-api/split_table_batched_embeddings.html>
- Optimizing CPU Performance for Recommendation Systems At-Scale：<https://pure.psu.edu/en/publications/optimizing-cpu-performance-for-recommendation-systems-at-scale/>
- Parallelization Strategies for DLRM Embedding Bag Operator on AMD CPUs：<https://pure.psu.edu/en/publications/parallelization-strategies-for-dlrm-embedding-bag-operator-on-amd/>
- PyTorch Geometric NeighborLoader：<https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html>
- DGL GraphBolt：<https://www.dgl.ai/dgl_docs/en/2.2.x/api/python/dgl.graphbolt.html>
- GNNLab：<https://colab.ws/articles/10.1145%2F3492321.3519557>
- PaGraph：<https://paperswithcode.com/paper/pagraph-scaling-gnn-training-on-large-graphs>
- AgentCgroup：<https://arxiv.org/abs/2602.09345>
- A CPU-Centric Perspective on Agentic AI：<https://arxiv.org/abs/2511.00739>
- Evaluating the impact of L3 cache size of AMD EPYC CPUs on CFD：<https://arxiv.org/abs/2505.17934>
- AMD EPYC CPUs with 3D V-Cache：<https://www.amd.com/en/blogs/2024/amd-epyc-cpus-deliver-breakthrough-performance-wi.html>
- OLAP on Modern Chiplet-Based Processors：<https://www.doc.ic.ac.uk/~af6618/publication/OLAP_on_Modern_Chiplet-Based_Processors/>
- P-MOSS：<https://arxiv.org/abs/2411.02933>
- RocksDB Tuning Guide：<https://github.com/facebook/rocksdb/wiki/RocksDB-Tuning-Guide>
