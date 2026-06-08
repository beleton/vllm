# HNSW 论文总览

## 总体结论

分类目标：哪些论文说明 HNSW 的访问路径和数据布局会影响 cache，哪些论文说明相似 query 存在请求级复用机会，哪些论文说明硬件层次或远端内存会改变索引组织，哪些论文只适合作为 baseline 或边界材料。
这些论文总体支持一个事实：graph ANN 查询对访问路径、节点布局、查询分布和内存层次敏感。

## 分类视角

### 算法与评测基线

这一类论文定义 HNSW 的基本搜索语义、参数空间、recall-latency 口径和强 baseline。它们回答“当前实验应该和谁比、哪些变量必须控制”，不直接提供 chiplet 优化方法。

| 文献 | 作用 | 对当前研究的含义 |
|---|---|---|
| HNSW 原论文 | 定义多层近邻图、上层 greedy search、第 0 层 best-first search、`M/efConstruction/efSearch` 等核心机制 | 所有调度或布局实验必须保持搜索语义清晰，并报告 recall |
| ANN-Benchmarks | 提供跨 ANN 实现的统一评测框架 | 指标应覆盖 recall、query time、build/index size，避免只看 QPS |
| Graph-based ANNS Survey | 将 graph ANN 拆成初始化、候选获取、邻居选择、连通性、seed/routing 等组件 | chiplet 实验需要控制算法组件差异，避免把图质量变化误判为硬件收益 |
| Graph-Based Vector Search | 将方法归纳为 SS、II、NP、ND、DC，并评估 HNSW、Vamana、ELPIS 等 | 后续不能只对比未优化 HNSW；HNSW、Vamana、ELPIS 和优化实现应作为候选 baseline |
| Intrinsic Dimensionality and HNSW Recall | 说明 recall 受数据内在维度、插入顺序和类别顺序影响 | 改 internal id、构建顺序、分区或入口点时必须报告 recall@k |

对当前实验，`recall@k` 不是附属指标。HNSW 的插入顺序、节点编号、图边、入口点和搜索宽度都可能改变结果质量。任何只报告 latency/QPS 的优化都不足以支撑结论。

### 节点布局与普通 CPU cache

`Graph Reordering for Cache-Efficient Near Neighbor Search` 是当前 HNSW/chiplet 方向最重要的普通 cache baseline。它说明 HNSW 类图搜索沿邻接表随机跳转，节点向量和邻接表的物理布局会影响 L1/L2/L3/TLB miss；Gorder、RCM、Porder 等方法可在不改变搜索算法和 recall 的前提下改善平均延迟和尾延迟。

这类工作对当前研究的作用是建立边界：

- 第 0 层节点块布局低局部性是通用 cache locality 问题，不是 chiplet 特有问题。
- 若新方案只优于原始插入顺序布局，不能证明 CCD/L3 拓扑感知有独立价值。
- 强 baseline 至少应包含原始布局、随机重排负对照、RCM/Gorder、Porder 或 trace-guided 普通重排序。
- 若提出 chiplet-aware layout、热点副本或 per-CCD owner，需要证明其相对普通 graph reordering 仍有增量，并用硬件计数说明收益来自 CCD/L3 域。

因此，graph reordering 不应作为当前主线贡献本身，而应作为实验前置基线。当前主线更适合研究“相似 query 是否能在同一个 CCD 的共享 L3 中复用同一批只读节点块”，而不是重新证明节点重排可降低 cache miss。

### 查询流调度与相似 query 复用

这一类论文与当前“同一 CCD 内相似 query 并行调度缺失”最接近。它们共同使用一个思想：相似 query 往往共享近邻集合、索引分区或访问路径，因此请求顺序、缓存和调度可以成为系统优化对象。

| 文献 | 核心机制 | 可迁移内容 | 不可直接迁移内容 |
|---|---|---|---|
| Dynamic Query Reordering | 在有限缓冲区内按 query 相似性重排执行顺序，提高索引分区缓存复用 | 请求级 locality scheduling；吞吐与排队延迟的权衡 | 实验是 M-Index 磁盘索引，不是 HNSW，也没有 CCD/L3 |
| Active Caching / CES | 用共享邻居和倒排近邻列表估计未缓存 query 的 top-k | shared-neighbor 可作为 query 相似度或复用潜力信号 | CES 是应用级结果缓存，会改变是否执行后端搜索，不是 CPU cache 优化 |
| QVCache | 在后端向量库前增加 query-aware vector cache，用 mini-index 和阈值判断缓存命中 | workload 局部性建模、滑动窗口工作集、query 相似度扰动实验 | 命中时绕过后端 HNSW，不能证明 HNSW 内部 chiplet 优化 |
| CCD-Level Thread Orchestration | 按 CCD 维护任务映射、负载监控和 CCD-aware stealing | 直接关联 HNSW/IVF、AMD CCD、本地 L3、线程调度和生产 workload | 论文 HNSW 场景是多表共置，当前单表相似 query 需要重新验证 |

### 内存层次、分区与远端访问粒度

HM-ANN、Pyramid、d-HNSW、DiskANN 系列、SPANN、AiSAQ 和 MemANNS 都说明向量检索系统可以通过数据放置、分区、批量加载、热点复制或查询路由改变性能。但这些论文的瓶颈分别是 PMM、RDMA、SSD、PIM、分布式扇出或压缩码扫描，不能把 speedup 数字迁移到单机 CCD/L3。

| 类别 | 代表论文 | 可借鉴思想 | 当前边界 |
|---|---|---|---|
| 快慢内存 | HM-ANN | 区分热导航结构和冷大图；用入口点和层级组织降低慢内存访问 | CCD L3 不是可显式管理的大容量 PMM，放置粒度和代价不同 |
| 分布式/远端内存 | Pyramid、d-HNSW | meta-HNSW 作为查询路由器；按 partition/sub-HNSW 批量访问；batch 内访问去重 | 目标是减少跨机器或 RDMA round trip，不是提高本地 L3 hit |
| SSD-resident ANN | DiskANN、SPANN、AiSAQ、DiskANN++ | 用访问粒度、页面布局、query-sensitive entry 和缓存减少 I/O | 主要瓶颈是 SSD page 或 PQ/graph 驻留，不是 CCD 间 cache 行为 |
| PIM/压缩索引 | MemANNS | 按 `size * frequency` 建模聚类工作量；热点复制；工作单元共置 | 对象是 IVFPQ 与 DPU/MRAM，不是 HNSW 图遍历 |
| 持久内存一致性 | P-HNSW | hnswlib 数据结构可迁移到 PMem，并需日志保证恢复 | 关注插入和恢复，不提供查询阶段 chiplet 优化证据 |

这些论文适合作为方法启发：把 HNSW 访问路径拆成 metadata、上层导航、L0 邻接表、向量、visited 状态和结果堆，分别分析访问频率、数据大小、生命周期和放置域。它们不适合作为“chiplet-aware HNSW 一定有效”的证据。


## 论文索引

| 类别 | 文档 |
|---|---|
| HNSW 基础 | [HNSW 原论文解读](./HNSW原论文解读.md) |
| Graph ANN 评测 | [Graph-Based Vector Search 实验评估解读](./Graph-Based_Vector_Search实验评估解读.md) |
| 普通 cache locality | [Graph Reordering for Cache-Efficient Near Neighbor Search 解读](./Graph_Reordering_for_Cache-Efficient_Near_Neighbor_Search解读.md) |
| CCD 调度 | [CCD-Level and Load-Aware Thread Orchestration 解读](./CCD_Level_Load_Aware_Thread_Orchestration解读.md) |
| 查询重排 | [Enhancing Similarity Search Throughput by Dynamic Query Reordering 解读](./Enhancing_Similarity_Search_Throughput_by_Dynamic_Query_Reordering解读.md) |
| 查询级缓存 | [QVCache 解读](./QVCache_A_Query-Aware_Vector_Cache解读.md)、[Active Caching 解读](./Active_Caching_for_Similarity_Queries_Based_on_Shared-Neighbor_Information解读.md) |
| 异构内存 | [HM-ANN 解读](./HM-ANN解读.md) |
| 分布式/远端内存 | [Pyramid 解读](./Pyramid_Distributed_Similarity_Search解读.md)、[d-HNSW 解读](./d-HNSW解读.md) |
| PIM/压缩索引 | [MemANNS 解读](./MemANNS解读.md) |

