# 主内存数据库索引、OLAP、Stream Processing 在 Chiplet/NUMA CPU 上的软件优化研究备忘

## 范围

平台约束沿用当前主线：`2 x AMD EPYC 9745`，每 CPU 8 个 CCD，每 CCD 16 核，共享约 32 MiB 本地 `L3`。目标不是再做 attention 局部化，而是寻找 **可在现有 AMD EPYC CCD/L3/NUMA 上落地** 的数据库与数据处理软件优化方向。

本文重点覆盖：
- 主内存索引：`B+-Tree`
- 分析型执行：`OLAP`、sorting
- 流处理：`stream processing`
- 混合负载：`LSM`、`HTAP`
- 向量检索：`HNSW` / vector DB

## 结论

现有文献可迁移的共性机制只有四类：
- **可路由分片**：请求先映射到稳定分片，再决定执行位置。`P-MOSS` 的 `index slice`、`OLAP` 的 `worker/fragment`、`BriskStream` 的 operator graph 都属于这一类。
- **相位切换**：局部 `L3` 优先与聚合 `L3` 优先不是同一策略。`OLAP` 与 sorting 都证明 `Local/Mixed` 需要按工作集切换。
- **显式边界替代隐式跨 chiplet 共享**：把随机的远端 cache/coherence 访问改成显式的 repartition、shuffle、batched exchange，通常更可控。
- **运行时反馈**：调度不能只靠静态分区。可用 `PMU`、队列长度、带宽、热点分布做轻量在线重排。

直接可做、且与现有论文有明确差异的新研究点有五个：
- `B+-Tree/HNSW` 顶层热点结构 `CCD` 复制 + 底层分片本地化
- `LSM/HTAP` 的 phase-aware `Local/Mixed` 切换
- 面向 `EPYC CCD` 的轻量 `PMU` 驱动空间调度，替代 `P-MOSS` 的离线训练 + 分钟级推理
- `stream processing` / `HTAP` 的 `CCD` 级 producer-consumer pipeline 共置
- 面向向量检索与点查的 locality-signature query grouping

## 现有工作的可迁移机制

### P-MOSS：可路由索引分片 + 执行核/数据联合放置

可迁移机制：
- 按 key range 切出稳定 `index slice`
- 先路由请求，再决定 core/data placement
- 用 `PMU` 统计做在线反馈
- 必要时迁移页面或更新 slice-to-core 映射

可直接借用的不是 `Decision Transformer`，而是“**先找稳定分片，再调度分片**”的建模方式。

边界：
- `P-MOSS` 的数据放置单位是 `NUMA node / IMC`，不是 `CCD/L3`
- 其主对象是 `B+-Tree`
- 迁移路径依赖页面移动，适合慢变化 workload，不适合高频切换

### OLAP on Modern Chiplet-Based Processors：worker 粒度降到 chiplet

可迁移机制：
- `worker per chiplet`
- `WICP_Local` 与 `WICP_Mixed` 按工作集相对单 `L3` / 聚合 `L3` 切换
- 用显式 `exchange/repartition` 换掉隐式跨 chiplet 共享访问

对当前平台最重要的结论：
- 仅做到 `NUMA-aware` 不够，仍会留下同一 socket 内跨 `CCD` 的代价
- 优化粒度应落到 `CCD/L3`

### Optimizing Sorting for Chiplet-Based CPUs：去掉跨域 shuffling + Local/Mixed 切换

可迁移机制：
- 明确区分 `local-L3 / remote-L3 / DRAM`
- 小工作集走 `Chiplet_Local`
- 中等工作集走 `Chiplet_Mixed`
- 避免为“均衡”而做额外跨域 shuffling

这对数据库类工作负载的含义是：
- 不是所有均衡都值得换取一次额外搬运
- build / histogram / partition 一类阶段适合强局部化
- merge / large scan / repartition 一类阶段可能更适合聚合 `L3`

### NUMA-aware data placement / morsel-driven / BWAP：页放置与任务放置解耦

可迁移机制：
- `Morsel-driven`：小任务块 + NUMA-local operator state
- `Adaptive NUMA-aware data placement`：数据与任务放置按 workload 调整，而不是固定静态分区
- `BWAP`：页分布不必平均 interleave，可按带宽与拥塞做非对称加权

对 EPYC 的直接启发：
- 当前机器的 DRAM 放置粒度通常仍是 socket/NPS，而不是 `CCD`
- 因此更现实的设计不是“把页面放到 CCD”，而是“**CCD 绑核 + socket/NPS 绑页 + 热结构复制/分片 + 运行时切换**”

### BriskStream：producer-consumer 相对位置感知

可迁移机制：
- 调度时显式考虑 producer-consumer 的拓扑距离
- 状态算子与上下游共置，减少跨域传递
- 优化目标不是单算子最快，而是整条 pipeline 的吞吐/可扩展性

这与 `OLAP` 的 `WICP` 一致：真正需要局部化的是 **状态与消费链**。

### AHA-Tree / Quake：结构自适应，但尚未做 EPYC CCD 拓扑感知

`AHA-Tree` 给出的可迁移机制：
- 在 `B+-Tree` 与 `LSM` 风格之间按 workload 自适应切换

`Quake` 给出的可迁移机制：
- workload-aware partitioning
- NUMA-aware intra-query parallelism

两者都说明：**索引结构本身可自适应，拓扑放置也可自适应**。现有工作尚未把二者与 `CCD/L3` 绑定到一起。

## 新研究点

### 方向一：`B+-Tree/HNSW` 顶层热点结构 `CCD` 复制，底层分片本地化

核心想法：
- `B+-Tree` 的 root/upper internal nodes，或 `HNSW` 的 top layers / entry points，容量小、命中频繁
- 对这部分做 `per-CCD` 复制
- 底层 leaf / posting / graph neighborhood 按 key range、cluster 或 graph partition 做分片
- 请求先命中本 `CCD` 的顶层副本，再落到本地优先的数据分片；远端仅在底层 miss 或跨分片扩展时发生

与已有论文差异：
- 不同于 `P-MOSS` 的整段 slice 迁移；这里强调 **热点上层复制 + 冷/大结构分片**
- 不同于 `OLAP/sorting` 的阶段性 worker 放置；这里直接改索引层级结构
- 不同于 `Quake` 的 NUMA-aware parallelism；这里把拓扑单位从 `NUMA` 压到 `CCD/L3`

适用对象：
- `B+-Tree` 点查、短 scan
- `HNSW` / graph ANN 查询
- 带热点前缀的 range lookup

验证方案：
- 基线：全局共享索引、纯分片索引、全复制索引
- 对照：`per-CCD upper replica + lower partition`
- 工作负载：`YCSB` read-heavy / scan，`HNSW` search，热点 Zipf 分布
- 指标：吞吐、`p95/p99`、`L3 miss/op`、`another-CCD share`、更新放大、复制内存开销
- 工具：`AMDuProfPcm`、`IBS`、`perf c2c`、`numastat`

### 方向二：`LSM/HTAP` 的 phase-aware `Local/Mixed` 切换

核心想法：
- `MemTable` flush、sorted run build、Bloom/filter build、small compaction、delta merge build 走 `Local`
- 多路 compaction、长 scan、merge join、large range read 走 `Mixed`
- 选择依据不是语义，而是 **阶段工作集相对 32 MiB 单 CCD L3 与多 CCD 聚合 L3 的关系**

与已有论文差异：
- `sorting` 和 `OLAP` 已证明 `Local/Mixed` 切换有效，但对象是 sort pass 和 query worker
- 这里把同一规则落到 `LSM` compaction、`HTAP` delta merge、`AHA-Tree` 形态切换后的执行阶段
- 现有 `HTAP` / `LSM` 文献更关注逻辑结构、Bloom/filter 或 merge policy，较少显式绑定 `CCD/L3`

适用对象：
- `RocksDB` / `Pebble` 类 `LSM`
- `HTAP` 中 delta merge、冷热层合并
- `AHA-Tree` 一类自适应索引

验证方案：
- 先做 compaction/sort/merge 微基准，再接 `db_bench`
- 基线：线程自由调度、socket 级绑核、`NPS1/NPS4`
- 方案：按阶段切换 `CCD` 局部执行与跨 `CCD` 分散执行
- 指标：throughput、write amplification、compaction stall、scan latency、`L3 miss/MB merged`

### 方向三：面向 `EPYC CCD` 的轻量 `PMU` 驱动空间调度

核心想法：
- 不做 `P-MOSS` 式离线预训练与分钟级推理
- 直接用在线指标触发重排：`L3 miss share`、`another CCD` 访问占比、memory bandwidth、队列长度、skew、`IPC/CPI`
- 调整对象是：分片到 `CCD` 的映射、热点副本数、`Local/Mixed` 模式、是否启用 query grouping

与已有论文差异：
- 不同于 `P-MOSS` 的 `DT + page migration`
- 不同于 `BWAP` 的纯页分布
- 不同于 `OLAP` 的静态 worker-per-chiplet
- 核心创新点是 **把 EPYC 上可直接采的 `PMU` 信号闭环到 `CCD` 级软件策略**

适用对象：
- `B+-Tree` / `HNSW`
- `stream state store`
- `HTAP` 的冷热分区

验证方案：
- 先离线回放 workload，确定阈值策略
- 再在线跑热点漂移、读写切换、scan burst
- 比较：静态分片、固定 `WICP` 风格、轻量反馈调度
- 关注调度开销与收益交叉点

### 方向四：`stream processing` / `HTAP` 的 `CCD` 级 pipeline 共置

核心想法：
- 将 producer-consumer 强相关算子链与其状态放在同一 `CCD`
- 仅在明确的 repartition / window merge / sink 边界跨 `CCD`
- 小状态窗口、局部聚合、局部 join state 走 `Local`
- 大窗口合并、全局排序、跨 key 重分布走 `Mixed`

与已有论文差异：
- `BriskStream` 做的是 `NUMA distance` 感知
- 这里进一步收缩到 `CCD/L3`
- 同时把 pipeline 共置与 `OLAP` 的 `worker per chiplet`、`sorting` 的 `avoid shuffling` 合并起来

适用对象：
- 本地单机 stream engine
- 流批一体 / streaming DB
- 带本地状态表的 `HTAP` 增量视图维护

验证方案：
- 微基准：`map -> filter -> local agg -> repartition -> global agg`
- 状态版：`stream join` / windowed agg
- 变量：算子是否跨 `CCD`、状态是否复制、窗口大小、skew
- 指标：records/s、端到端延迟、state access 的本地/远端占比

### 方向五：locality-signature query grouping

核心想法：
- 不只给数据选位置，也给请求分组
- `B+-Tree` 以 key range，`HNSW` 以 entry point / centroid / coarse cluster，`HTAP` 以 fragment id，给请求打 locality signature
- 调度器把 signature 相近的请求短时间聚到同一 `CCD`
- 目标是放大顶层索引、热点 leaf、graph frontier 的 `L3` 复用

与已有论文差异：
- `P-MOSS` 只对 slice 选 core，不显式做 request grouping
- `Quake` 自适应 index，但重点不是批内 locality amplification
- 该方向更接近“shared-prefix batching”在数据库索引中的等价形式

适用对象：
- 点查密集型 `B+-Tree`
- `HNSW` / IVF-PQ rerank
- 高频热点区间查询

验证方案：
- 设定可控 batching window，例如 `10-100 us`
- 比较 FCFS、纯哈希分片、locality grouping
- 观察 throughput 与 tail latency 的 tradeoff

## 优先级判断

优先级最高的不是 `LSM` 全系统实现，而是两类最短路径：
- **短路径 A**：`B+-Tree/HNSW` 顶层复制 + 底层分片
- **短路径 B**：`phase-aware Local/Mixed` 切换

原因：
- 二者都能直接复用 `EPYC CCD + 32 MiB L3` 这一平台特征
- 与 `P-MOSS / OLAP / sorting` 都有明确方法继承关系
- 与当前 attention 负结果不同，这两类对象都存在稳定分片、可复用热点结构或明确阶段边界

`stream processing` 与 `HTAP` 更适合作为第二阶段扩展，因为需要更完整的系统框架。

## 与现有论文的总体差异

可形成论文差异的位置有三处：
- **拓扑粒度差异**：已有很多工作停在 `NUMA node`；目标是落到 `AMD EPYC CCD/L3`
- **对象差异**：已有 chiplet 论文主要是 `OLAP` worker、sorting pass、attention/GPU；目标是 `main-memory index / HTAP / vector DB / stream state`
- **机制组合差异**：已有工作通常只做分片、或只做放置、或只做结构自适应；目标是 **分片 + 热结构复制 + phase switching + PMU feedback** 的组合

## 实验建议

### 平台变量

- `NPS1 / NPS4`
- 固定 socket 级绑页
- `CCD` 级绑核
- 若 BIOS 支持，再补 `L3 Cache as NUMA Domain`

### 基础指标

- 吞吐、`p50/p95/p99`
- `IPC/CPI`
- `L3 miss/op`
- `L3 source share`：本地、same-node another `CCD/CCX`、local memory
- DRAM bandwidth、queue depth、stall time

### 负载集合

- `B+-Tree`：`YCSB-A/B/C/E`
- `HNSW`：`SIFT/GIST/DEEP1B` 子集或本地向量集
- `LSM`：`db_bench` 的 fillrandom/readrandom/seek/compaction
- `stream`：固定 operator DAG 的单机 shared-memory 微基准
- `HTAP`：小型混合读写 + scan + merge workload

### 论文组织方式

最稳妥的论文主线是：
1. 先证明 `CCD` 级热点结构与阶段边界真实存在
2. 再给出 `Local/Mixed/Replica/Grouping` 的软件机制
3. 最后用 `PMU` 或轻量反馈做动态切换

## 本次检索范围内的空白

截至 `2026-05-14`，本次检索范围内未见：
- 直接针对 `AMD EPYC CCD/L3` 的 `B+-Tree` / `HNSW` 顶层复制论文
- 直接把 `Local/Mixed` 规则落到 `LSM compaction` 或 `HTAP delta merge` 的 chiplet 论文
- 直接把 `BriskStream` 风格的拓扑感知调度压到 `CCD/L3` 粒度的 stream-processing 论文

这意味着新工作不需要证明“首次发现 chiplet 差异”，而需要证明“**在现有 EPYC 上，哪些数据库对象具备可迁移的 `CCD` 级软件优化性**”。

## 来源 URL

- `P-MOSS`：https://arxiv.org/abs/2411.02933
- `OLAP on Modern Chiplet-Based Processors`：https://www.vldb.org/pvldb/vol17/p3428-fogli.pdf
- `Optimizing Sorting for Chiplet-Based CPUs`：https://vldb.org/workshops/2024/proceedings/ADMS/ADMS24_03.pdf
- `Morsel-Driven Parallelism`：https://portal.fis.tum.de/en/publications/morsel-driven-parallelism-a-numa-aware-query-evaluation-framework/
- `Adaptive NUMA-aware data placement and task scheduling for analytical workloads in main-memory column-stores`：https://www.vldb.org/pvldb/vol10/p37-psaroudakis.pdf
- `Bandwidth-Aware Page Placement in NUMA`：https://arxiv.org/abs/2003.03304
- `BriskStream`：https://arxiv.org/abs/1904.03604
- `The AHA-Tree: An Adaptive Index for HTAP Workloads`：https://arxiv.org/abs/2406.08746
- `Quake: Adaptive Indexing for Vector Search`：https://arxiv.org/abs/2506.03437
- `AMD EPYC 9004 BIOS & Workload Tuning Guide`：https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58011-epyc-9004-tg-bios-and-workload.pdf
- `AMD EPYC 9004 High Performance Computing Tuning Guide`：https://docs.amd.com/v/u/en-US/58002_amd-epyc-9004-tg-hpc
