# d-HNSW 解读

**一句话概述**：d-HNSW 是面向 RDMA-based disaggregated memory（基于 RDMA 的解耦内存）的 HNSW 向量检索系统。论文把轻量级 meta-HNSW 缓存在 compute pool，把 sub-HNSW 和原始向量放在 memory pool，通过 representative index caching、RDMA-friendly layout 和 batched query-aware loading 减少远端读取。实验在 SIFT1M/GIST1M 上显示，d-HNSW 相比 naive d-HNSW 最高降低 117x/121x 查询延迟，但主要收益来自减少 RDMA network round trip，不能直接外推到单机 chiplet CPU 的 CCD/L3 场景。

---

## 背景：HNSW 与解耦内存

### 向量检索与 ANN

`Vector similarity search`（向量相似性搜索）给定查询向量，在大规模向量集合中查找最相似的 top-k 向量。论文将其作为 LLM、RAG、搜索和推荐系统的基础能力（Sec. 1）。

`ANN`（Approximate Nearest Neighbor，近似最近邻）通过近似结果换取更低查询延迟。论文关注 ANN，不是精确 top-k 最近邻搜索（Sec. 1）。

### HNSW

`HNSW`（Hierarchical Navigable Small World，层次化可导航小世界图）是图 ANN 索引。上层图用于粗粒度导航，下层图提供密集连接和精细搜索。查询从 entry point 开始，在每层执行 greedy routing（贪婪路由），逐步移动到与查询向量距离更近的邻居（Sec. 2.1, Fig. 1）。

HNSW 在本地内存中通常依赖指针或邻接表随机访问；在解耦内存中，这种不可预测访问会转化为大量远端读取。

### Disaggregated memory 与 RDMA

`Disaggregated memory`（解耦内存）将 compute hardware 和 memory hardware 分离，资源池通过高速网络连接，便于独立扩展计算和内存容量（Sec. 1, Sec. 2.2）。

`RDMA`（Remote Direct Memory Access，远程直接内存访问）允许一台机器直接读写另一台机器的注册内存，绕过远端 CPU。论文提到 RDMA READ/WRITE、CAS、FAA 等 one-sided primitives（单边操作）（Sec. 2.2）。

`Doorbell batching` 将多个 RDMA 操作批量提交，使多个非连续地址读取可在一个网络 round trip 中完成，但 RDMA NIC 仍可能发起多个 PCIe transaction；批量过大可能干扰其他 RDMA command 并增加延迟（Sec. 3.2）。

## 研究问题

论文要解决的问题是：HNSW 在内存解耦系统中不能直接按本地内存结构访问。HNSW 贪婪路径不可预测，每一步都可能访问不同节点；如果每个节点或每个 sub-HNSW 都通过 RDMA 单独读取，会产生大量 network round trips，导致延迟过高（Sec. 1, Sec. 3.1）。

论文还处理两个系统问题：

1. 动态插入会导致 sub-HNSW 数据碎片化，降低 RDMA 读取效率（Sec. 3.2）。
2. 批量查询中不同查询可能需要相同 sub-HNSW，重复加载会浪费 compute pool 与 memory pool 之间的带宽（Sec. 3.3）。

## 核心内容

d-HNSW 的核心是把 HNSW 拆成两级结构（Sec. 3, Fig. 2）：

1. **meta-HNSW**：从全量数据均匀采样 500 个向量，构建三层 representative HNSW，作为轻量级全局分类索引，缓存在 compute pool。SIFT1M 上 meta-HNSW 大小为 0.373MB，GIST1M 上为 1.960MB（Sec. 3.1, Fig. 3）。
2. **sub-HNSW**：meta-HNSW 底层 L0 的每个向量定义一个 partition，对应一个 sub-HNSW。sub-HNSW 和原始 floating-point vectors 存在 memory pool（Sec. 3.1, Fig. 3）。
3. **两阶段查询**：compute instance 先在本地 meta-HNSW 上定位最相关 sub-HNSW，再通过 RDMA 从 memory pool 读取少量 sub-HNSW 做细粒度搜索（Sec. 3.1）。

## 方法与系统设计

### Representative index caching

compute pool 的 DRAM 只作为 cache，不能为每个查询加载完整 HNSW。d-HNSW 使用 meta-HNSW 代表全局索引，只缓存这个轻量结构；真正的大索引拆成多个 sub-HNSW 存在 memory pool（Sec. 3.1, Fig. 3）。

该设计利用 HNSW 贪婪搜索的分类能力：先用 meta-HNSW 找到可能包含 top-k 的 partitions，再只读取这些 partitions 对应的 sub-HNSW。目标是减少未访问节点的传输，降低带宽和延迟（Sec. 3.1）。

### RDMA-friendly graph index layout

memory instance 上注册一段连续内存。开头放 global metadata，记录各 sub-HNSW 的 offset；后续空间按 group 组织，每个 group 容纳两个相邻 sub-HNSW，中间共享 overflow memory space，用于动态插入的新向量和元数据（Sec. 3.2, Fig. 4）。

SIFT1M 的共享 overflow space 为 0.75MB，GIST1M 为 3.92MB。查询读取 sub-HNSW 时，同时读取对应 overflow space，使新插入数据仍能与原 sub-HNSW 连续读取，避免碎片化导致多次 RDMA read（Sec. 3.2, Fig. 4）。

对非连续 sub-HNSW，d-HNSW 使用 doorbell batching。它在一次网络 round trip 中提交多个 RDMA 读取请求，由 NIC 发起多个 PCIe transaction。论文明确指出，doorbell batch 过大可能干扰其他 RDMA command，并因 NIC 扩展性问题产生更长延迟（Sec. 3.2）。

### Query-aware batched data loading

批量查询中，每个 query 需要访问 `b` 个 closest sub-HNSW；batch size 为 `s` 时，原始需求为 `b*s` 次 sub-HNSW 访问。d-HNSW 在线分析这些访问需求，保证同一个 sub-HNSW 在同一 batch 内只从 memory pool 加载一次（Sec. 3.3, Fig. 5）。

compute instance 还保留最近加载的 `c` 个 sub-HNSW 给下一批查询复用。如果目标 sub-HNSW 已在本地 cache，则不再发起 RDMA 加载（Sec. 3.3）。

## RDMA 与解耦内存机制

论文假设 compute pool 有充足 CPU 资源，但每个 compute instance 的 DRAM cache 有限；memory instances 计算能力弱，主要负责轻量级 memory registration。compute 与 memory pool 通过 RDMA 连接（Sec. 3, Fig. 2）。

d-HNSW 主要使用 RDMA READ 从 memory pool 拉取 sub-HNSW 和向量数据。memory node CPU 不参与查询计算，compute node 通过 one-sided RDMA 直接访问远端注册内存（Sec. 2.2, Sec. 4）。

RDMA 优化目标有三类：

1. 减少 network round trips。
2. 避免非连续远端地址造成多次小读取。
3. 利用 doorbell batching 批量读取非连续 sub-HNSW。

## 实验设置

实验在 CloudLab 上运行。测试床包含 4 台 Dell PowerEdge R650，每台有 2 颗 36-core Intel Xeon Platinum CPU、256GB RAM、1.6TB NVMe SSD、Mellanox ConnectX-6 100Gb NIC。3 台服务器作为 compute pool，1 台作为 memory instance（Sec. 4）。

每台服务器有 144 个 hyperthreads，划分为 8 个 compute instances。每个 instance 运行一个 vector query worker，发送 RDMA command 做 top-k 检索；每个 instance 使用 18 个 OpenMP 线程做 HNSW search（Sec. 4）。

运行设置：

- compute instance cache 只存 memory pool 中 10% 的 sub-HNSW clusters。
- batch size 为 2000。
- 数据集为 SIFT1M 和 GIST1M。
- 评估 top-1 与 top-10。
- `efSearch` 从 1 到 48（Sec. 4, Fig. 6）。

baseline：

- **Naive d-HNSW / Native-HNSW**：论文 Sec. 4 写作 Native-HNSW，Fig. 6 和 Table 1-2 写作 naive d-HNSW。该基线对每个涉及的 sub-HNSW 发起 RDMA read round trip。
- **d-HNSW w./o. doorbell**：使用 meta-HNSW caching 和 query-aware loading，但读取 sub-HNSW 时不用 doorbell batching。
- **d-HNSW**：完整系统，包含 representative index caching、RDMA-friendly layout、query-aware loading 和 doorbell batching（Sec. 4, Fig. 6, Table 1-2）。

## 结果与解释

### SIFT1M

SIFT1M@10 中，d-HNSW 相比 naive d-HNSW 最高降低 117x 延迟，相比 without-doorbell 版本降低 1.12x 延迟；`efSearch=48` 时 recall 约 0.86（Sec. 4, Fig. 6a）。

SIFT1M@1 中，`efSearch=48` 时最高 recall 约 0.85。论文解释 top-1 查询只需选择最近向量，因此所有方案延迟都比 top-10 更低（Sec. 4, Fig. 6b）。

SIFT1M@1、`efSearch=48` 的延迟分解显示，naive d-HNSW network 为 90271.2us，sub-HNSW 为 6564.5us，meta-HNSW 为 13.52us；d-HNSW network 降到 527.6us，sub-HNSW 为 269.2us，meta-HNSW 为 9.75us（Sec. 4, Table 1）。

SIFT1M@1 的 round trips per query 分别为 naive d-HNSW 3.547、without-doorbell 0.896、d-HNSW 4.75e-3。该结果直接支持论文主张：主要收益来自减少 RDMA round trips 和网络传输延迟（Sec. 4, Table 1 附近正文）。

### GIST1M

GIST1M 中，d-HNSW 相比 naive d-HNSW 最高降低 121x 延迟，相比 without-doorbell 版本降低 1.30x 延迟。论文指出 GIST1M 向量维度更高，因此查询延迟整体高于 SIFT1M（Sec. 4, Fig. 6c-d）。

GIST1M@1、`efSearch=48` 的延迟分解显示，naive d-HNSW network 为 422.9ms，sub-HNSW 为 35.3ms；d-HNSW network 降到 1.3ms，sub-HNSW 为 1.48ms，meta-HNSW 为 46.9us（Sec. 4, Table 2）。

### 收益来源

d-HNSW 的核心不是改变 HNSW 搜索数学，而是改变索引在 disaggregated memory 下的切分、放置和加载方式。meta-HNSW 承担本地粗筛，sub-HNSW 承担远端精搜；这把不可预测的逐节点远端访问转化为少量 partition 级远端读取（Sec. 3.1）。

Table 1 和 Table 2 中 network latency 的下降幅度远大于 meta-HNSW computation 的变化；meta-HNSW 计算本身只有微秒级。doorbell batching 的收益小于 representative index caching 的收益：SIFT1M@10 中完整 d-HNSW 相对 without-doorbell 为 1.12x，GIST1M 中为 1.30x；相对 naive d-HNSW 的 117x/121x 主要来自避免对每个涉及 sub-HNSW 单独发起 RDMA round trip（Sec. 4, Fig. 6, Table 1-2）。

## 分析

- d-HNSW 将 HNSW 的细粒度随机访问提升为 partition 粒度访问，适合 RDMA 网络场景。
- meta-HNSW 是低成本本地分类器。SIFT1M 上只有 0.373MB，GIST1M 上为 1.960MB，可常驻 compute pool cache（Sec. 3.1, Fig. 3）。
- RDMA-friendly layout 同时服务查询和插入。overflow memory 解决新增数据连续读取问题，但论文未评估 overflow 耗尽后的行为（Sec. 3.2）。
- Query-aware loading 将 batch 内共享访问显式去重，适合批量查询场景；该机制与在线低 batch 查询的收益可能不同（Sec. 3.3）。

## 边界

- 论文没有评估 CXL、NUMA、chiplet、CCD、本地 L3 cache、remote L3 或片间互连。其 disaggregation 是跨服务器 RDMA memory pool，不是单机 chiplet CPU 内部缓存/内存层次（Sec. 2.2, Sec. 4）。
- 论文没有给出 build time、insert throughput、delete、并发更新一致性、内存放大、load balancing 策略的详细实验结果。动态插入只在 layout 设计中说明 overflow space 处理方式（Sec. 3.2）。
- 论文只报告 SIFT1M 和 GIST1M，没有报告 billion-scale 数据集结果。Abstract 和 Sec. 4 的结论不能直接外推到更大规模索引。
- 论文没有报告 CPU 利用率、cache miss、PCIe counter、NIC queue depth、RDMA doorbell batch size sweep。doorbell batching 的最佳批量大小没有在实验中展开（Sec. 3.2, Sec. 4）。
- 论文主要报告 latency-recall 与 latency breakdown，没有给出独立 throughput 曲线（Sec. 4, Fig. 6, Table 1-2）。

## 可迁移点

- HNSW 在远端内存场景下应避免逐节点远端读取；可用轻量级代表索引先做 partition 选择，再批量读取候选 partition（Sec. 3.1）。
- 图 ANN 索引布局应按远端读取粒度设计，而不是只按本地指针结构序列化（Sec. 3.2）。
- 批量查询可合并重复 sub-HNSW 访问，把 batch 内相同 partition 的远端传输去重（Sec. 3.3）。
- 远端索引系统需要区分 metadata、graph neighbor array、floating-point vectors 和 overflow data 的放置方式（Sec. 3.2, Fig. 4）。

## 不可直接迁移点

- 不能直接把 d-HNSW 的 117x/121x 结论迁移到单机 Chiplet CPU。论文瓶颈是 RDMA network round trip，不是 CCD 间访问或 L3 miss（Sec. 4, Table 1-2）。
- 不能直接套用 meta-HNSW 采样 500 个向量的参数。论文只给出 SIFT1M/GIST1M 下的 meta-HNSW 大小和结果，没有参数敏感性分析（Sec. 3.1）。
- 不能据此证明 HNSW 一定适合 L3/CCD 局部化。论文没有测本地 cache 行为，也没有讨论线程绑定、NUMA placement 或 cache-aware graph layout。
- 不能把 memory pool 弱计算能力假设等同于当前 CPU 本地内存场景。论文假设 memory instances 只做轻量 registration，查询计算在 compute pool（Sec. 3）。

## 与当前 Chiplet CPU 研究的关系

论文能支持的结论是：HNSW/vector search 的性能可以由索引访问粒度、数据布局和批量访问去重显著影响；图 ANN 的不规则访问路径需要系统级布局优化（Sec. 3, Sec. 4）。

论文不能支持“Chiplet CPU 上 HNSW 的主要瓶颈是跨 CCD/L3 访问”。其主要证据指向 RDMA network latency 和 round-trip 数量（Sec. 4, Table 1-2）。

对当前 Chiplet CPU 研究的可用价值在方法层面：把 HNSW 访问路径拆成 meta 层、partition 层、subgraph 层，分别分析哪些结构应复制、缓存、分片或批量加载。是否能转化为 CCD/L3 优化，需要本地实验补证 CPU cache、NUMA/CCD locality、线程调度和查询 batch 行为。

## 证据

- 原始资料：`wiki/原始资料/papers/HNSW/Efficient Vector Search on Disaggregated Memory with d-HNSW.pdf`
- 论文元数据：arXiv `2505.11783v1`，2025-05-17，DOI `10.48550/arXiv.2505.11783`
- 关键锚点：Abstract；Sec. 1；Sec. 2.1-2.2；Sec. 3.1-3.3；Sec. 4；Fig. 1-6；Table 1-2；Sec. 6
