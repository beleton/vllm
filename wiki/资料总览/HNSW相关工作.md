# HNSW 相关工作

整理 `HNSW`（Hierarchical Navigable Small World，层次化可导航小世界图）与图近似最近邻检索、缓存局部性、内存层次和向量检索系统相关的论文。本文用于支撑 [HNSW 向量检索的 Chiplet 研究方向](../方向调研/HNSW向量检索_Chiplet研究方向.md) 的后续复核。

## 一、结论

当前未找到直接研究 `chiplet CPU + HNSW` 的成熟论文。已有论文可支撑三类事实：
1. `HNSW` 是主流图近似最近邻检索方法，查询过程依赖多层图导航和第 0 层图扩展。
2. 图近似最近邻检索存在缓存局部性问题，节点重排、入口点选择、距离计算裁剪和内存布局会影响查询时延。
3. 大规模向量检索已有异构内存、SSD、持久内存和解耦内存系统工作，但这些工作优化的是 DRAM/SSD/PMem/RDMA 层次，不等同于 `CCD/L3` 局部性优化。

对当前 chiplet 课题，最值得优先深读的是：

1. **HNSW 原论文**：理解算法结构、`efSearch`、上层导航和邻居选择机制。
2. **Graph Reordering for Cache-Efficient Near Neighbor Search**：直接证明 HNSW 类图搜索存在缓存局部性优化空间。
3. **HM-ANN**：异构内存下的图 ANN 设计，最接近“快小内存 + 慢大内存”的层次化问题。
4. **d-HNSW**：解耦内存场景下的 HNSW 拆分、缓存和 RDMA-friendly layout。
5. **DiskANN / SPANN / DiskANN++**：大规模向量索引在内存和 SSD 间分层的代表系统。

## 二、HNSW 本体与图 ANN 背景

### Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs

- 本地论文：[Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.pdf](../原始资料/papers/HNSW/Efficient%20and%20robust%20approximate%20nearest%20neighbor%20search%20using%20Hierarchical%20Navigable%20Small%20World%20graphs.pdf)
- 作用：`HNSW` 原始论文。
- 主要内容：提出多层 proximity graph。上层节点少、边跨度大，用于贪婪导航；第 0 层节点多，用于高 recall 搜索。节点最高层级按指数衰减概率随机生成。论文同时使用邻居选择启发式改善高 recall 和聚簇数据上的性能。
- 对当前课题的意义：定义了需要优化的对象，包括 entry point、上层导航图、第 0 层邻接表、候选队列和 `efSearch` 控制的搜索宽度。

### A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search

- 本地论文：[A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search.pdf](../原始资料/papers/HNSW/A%20Comprehensive%20Survey%20and%20Experimental%20Comparison%20of%20Graph-Based%20Approximate%20Nearest%20Neighbor%20Search.pdf)
- 作用：图近似最近邻检索综述与实验对比。
- 主要内容：对 13 种代表性 graph-based ANNS 算法做分类和统一环境实验，对比 8 个真实数据集和 12 个合成数据集。
- 对当前课题的意义：用于确认 HNSW 所属方法线及其和 NSG、KGraph、SSG 等图 ANN 的关系，避免只围绕单个实现做结论。

### Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art

- 本地论文：[Graph-Based Vector Search - An Experimental Evaluation of the State-of-the-Art.pdf](../原始资料/papers/HNSW/Graph-Based%20Vector%20Search%20-%20An%20Experimental%20Evaluation%20of%20the%20State-of-the-Art.pdf)
- 作用：图向量检索方法的大规模实验评估。
- 主要内容：按 seed selection、incremental insertion、neighborhood propagation、neighborhood diversification、divide-and-conquer 五类设计范式梳理图检索方法，并在最高 10 亿规模数据上评估 12 种方法。
- 对当前课题的意义：用于选择 baseline 和解释 HNSW 之外的图 ANN 方法；其结论不能直接替代 chiplet 本地实验。

### The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds

- 本地论文：[The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds.pdf](../原始资料/papers/HNSW/The%20Impacts%20of%20Data,%20Ordering,%20and%20Intrinsic%20Dimensionality%20on%20Recall%20in%20Hierarchical%20Navigable%20Small%20Worlds.pdf)
- 作用：HNSW 在真实 embedding 数据上的敏感性分析。
- 主要内容：研究 HNSW recall 与 intrinsic dimensionality（内在维度）、数据插入顺序和真实向量数据库默认参数的关系。
- 对当前课题的意义：说明 HNSW 性能和 recall 受数据集、构建顺序和参数影响。后续做 layout/partition 时必须固定 recall@k，并区分布局收益和构图顺序收益。

## 三、缓存局部性与查询执行优化

### THE IMPACTS OF DATA, ORDERING, AND INTRINSIC DIMENSIONALITY ON RECALL IN HIERARCHICAL NAVIGABLE SMALL WORLDS
- 本地论文：[Graph Reordering for Cache-Efficient Near Neighbor Search](../原始资料/papers/HNSW/The%20Impacts%20of%20Data,%20Ordering,%20and%20Intrinsic%20Dimensionality%20on%20Recall%20in%20Hierarchical%20Navigable%20Small%20Worlds.pdf)
- 作用：HNSW 缓存局部性优化的直接相关工作。
- 主要内容：论文指出 HNSW 等图近似最近邻索引存在 poor cache miss performance，将图遍历建模为 cache hit maximization 问题，并用 graph reordering 将常共同访问的节点放在相近内存位置。实验中，重排最高使 query time 改善 40%，重排时间相对建索引时间较小。
- 对当前课题的意义：这是 trace-guided layout、节点重排和 `CCD` owner 分区最直接的论文依据。它证明的是缓存布局有效，不证明 `CCD/L3` 分区一定有效。

### FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search
- 本地论文：[FINGER - Fast Inference for Graph-based Approximate Nearest Neighbor Search.pdf](../原始资料/papers/HNSW/FINGER%20-%20Fast%20Inference%20for%20Graph-based%20Approximate%20Nearest%20Neighbor%20Search.pdf)
- 作用：图 ANN 查询过程的距离计算裁剪。
- 主要内容：观察到 greedy graph search 中许多距离计算不会改变搜索状态，提出用低秩基和分布匹配近似距离，跳过部分不必要计算。论文展示该方法可加速 HNSW。
- 对当前课题的意义：用于区分“距离计算瓶颈”和“图节点访存瓶颈”。若高维向量距离计算占主导，chiplet 局部性优化空间会被压缩。

### ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms
- 本地论文：[ANN-Benchmarks - A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.pdf](../原始资料/papers/HNSW/ANN-Benchmarks%20-%20A%20Benchmarking%20Tool%20for%20Approximate%20Nearest%20Neighbor%20Algorithms.pdf)
- 作用：ANN 算法评测工具。
- 主要内容：提供统一接口，在标准数据集上评估不同近似最近邻算法的速度和质量，支持参数扫描和可视化。
- 对当前课题的意义：可作为 baseline 选择和 recall-latency 曲线的参考，但 chiplet 实验仍需额外加入绑核、内存放置和硬件计数器采集。

## 四、内存层次与大规模向量检索系统

### HM-ANN: Efficient Billion-Point Nearest Neighbor Search on Heterogeneous Memory

- 本地论文：[HM-ANN - Efficient Billion-Point Nearest Neighbor Search on Heterogeneous Memory.pdf](../原始资料/papers/HNSW/HM-ANN%20-%20Efficient%20Billion-Point%20Nearest%20Neighbor%20Search%20on%20Heterogeneous%20Memory.pdf)
- 作用：异构内存中的图近似最近邻检索。
- 主要内容：面向 fast-but-small memory 与 slow-but-large memory 的层次化内存，目标是在单机上支持十亿规模数据并避免压缩带来的精度损失。论文同时考虑 memory heterogeneity 和 data heterogeneity。
- 对当前课题的意义：可迁移的是“热结构留在快层、冷结构放在慢层”的方法论。`CCD` 本地 L3 不是可显式分配的大容量内存，不能直接照搬其内存分配策略。

### Efficient Vector Search on Disaggregated Memory with d-HNSW

- 本地论文：[Efficient Vector Search on Disaggregated Memory with d-HNSW.pdf](../原始资料/papers/HNSW/Efficient%20Vector%20Search%20on%20Disaggregated%20Memory%20with%20d-HNSW.pdf)
- 作用：RDMA 解耦内存系统中的 HNSW。
- 主要内容：提出面向 disaggregated memory 的 HNSW 向量搜索引擎，包含 representative index caching、RDMA-friendly data layout 等机制，用于减少 compute pool 与 memory pool 之间的数据移动。
- 对当前课题的意义：representative index caching 与 per-CCD navigator replica 在思想上接近；RDMA-friendly layout 可作为“访问路径驱动布局”的系统参照。

### DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node

- 本地论文：[DiskANN - Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.pdf](../原始资料/papers/HNSW/DiskANN%20-%20Fast%20Accurate%20Billion-point%20Nearest%20Neighbor%20Search%20on%20a%20Single%20Node.pdf)
- 作用：SSD + DRAM 混合图 ANN 系统。
- 主要内容：在单机 64GB DRAM 和 SSD 上进行十亿规模近似最近邻搜索，目标同时满足 high recall、low latency 和 high density。
- 对当前课题的意义：提供“完整索引无法全放快层时，如何组织慢层访问”的系统背景。其主要瓶颈是 SSD I/O，不等同于 chiplet 远端 L3/DRAM 访问。

### SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search

- 本地论文：[SPANN - Highly-efficient Billion-scale Approximate Nearest Neighbor Search.pdf](../原始资料/papers/HNSW/SPANN%20-%20Highly-efficient%20Billion-scale%20Approximate%20Nearest%20Neighbor%20Search.pdf)
- 作用：内存-磁盘混合 ANN 系统。
- 主要内容：采用 inverted index 方法，将 centroid points 放在内存，将大 posting lists 放在磁盘；通过 balanced clustering、posting list closure 和 query-aware search 降低磁盘访问次数并保持 recall。
- 对当前课题的意义：可借鉴“轻量入口/路由结构常驻快层，大数据结构分层放置”的设计方式。

### DiskANN++: Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex

- 本地论文：[DiskANN++ - Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex.pdf](../原始资料/papers/HNSW/DiskANN++%20-%20Efficient%20Page-based%20Search%20over%20Isomorphic%20Mapped%20Graph%20Index%20using%20Query-sensitivity%20Entry%20Vertex.pdf)
- 作用：DiskANN 的页级 I/O 优化。
- 主要内容：针对 DiskANN routing path 长和冗余 I/O 请求问题，引入 query-sensitive entry vertex selection 和 page-based search。
- 对当前课题的意义：query-sensitive entry vertex 可对应 HNSW 中 per-query entry routing 或 hot-path query grouping，但 DiskANN++ 的主要目标是减少 SSD I/O。

### AiSAQ: All-in-Storage ANNS with Product Quantization for DRAM-free Information Retrieval

- 本地论文：[AiSAQ - All-in-Storage ANNS with Product Quantization for DRAM-free Information Retrieval.pdf](../原始资料/papers/HNSW/AiSAQ%20-%20All-in-Storage%20ANNS%20with%20Product%20Quantization%20for%20DRAM-free%20Information%20Retrieval.pdf)
- 作用：面向 DRAM-free 检索的全存储 ANN。
- 主要内容：基于 DiskANN，将压缩向量 offload 到 SSD index，在十亿规模数据上将查询阶段内存使用压到约 10 MB，并强调可用于 RAG retriever。
- 对当前课题的意义：主要作为存储层次背景。若当前目标是内存 HNSW 的 `CCD/L3` 优化，该论文优先级低于 HM-ANN、d-HNSW 和 Graph Reordering。

### P-HNSW: Crash-Consistent HNSW for Vector Databases on Persistent Memory

- 本地论文：[P-HNSW - Crash-Consistent HNSW for Vector Databases on Persistent Memory.pdf](../原始资料/papers/HNSW/P-HNSW%20-%20Crash-Consistent%20HNSW%20for%20Vector%20Databases%20on%20Persistent%20Memory.pdf)
- 作用：持久内存上的 HNSW 崩溃一致性。
- 主要内容：面向 vector database 中的 HNSW 索引，讨论 DRAM 实现的持久性问题，并在 persistent memory 上设计 crash-consistent HNSW。
- 对当前课题的意义：适合了解 HNSW 的系统化存储和恢复问题；与 chiplet locality 的直接关系较弱。

## 五、对 chiplet HNSW 的可迁移方法

### 热导航结构复制

HNSW 原论文和 d-HNSW 都把上层导航结构视为影响搜索入口和访问路径的关键结构。对 chiplet CPU，可验证每个 `CCD` 复制 entry point、上层节点和少量热点 hub 的收益。该方法只适合只读或低更新率索引。

### 第 0 层图布局与分区

Graph Reordering 证明 HNSW 图节点内存顺序会影响查询时延和 cache miss。chiplet 方向可以在普通 graph reordering baseline 之外，加入 `CCD` owner、first-touch、NPS domain 和 query trace 约束。实验必须证明 chiplet-aware layout 相对普通 reordering 仍有增量。

### 查询入口与路径分组

DiskANN++ 的 query-sensitive entry vertex 和 SPANN 的 query-aware search 都说明入口选择会影响搜索路径和访问量。对 HNSW，可尝试 coarse centroid、上层路径或热点 hub 作为 query grouping signature。该方向需要控制排队时延，不能只看平均 QPS。

### 内存层次判据

HM-ANN、DiskANN、SPANN、AiSAQ 和 d-HNSW 都采用“快层保存小而关键的结构，慢层保存大而冷的结构”的设计。对 AMD EPYC chiplet CPU，快层不是传统意义上的可分配内存，而是本地 `CCD` L3 和本地 NUMA 内存路径。因此必须用硬件计数器验证：

- `L3 miss/query`
- `DTLB miss/query`
- IBS load latency 分布
- `another CCD/CCX same node` 来源占比
- `local memory` 与 `remote memory` 来源占比
- p50/p95/p99 latency 与 `recall@k`

## 六、当前论文空白

1. 未找到直接面向 AMD EPYC `CCD/L3` 或 chiplet CPU 的 HNSW 优化论文。
2. 现有 HNSW 缓存优化主要关注通用 cache locality，不区分本地 `CCD` L3、远端 `CCD` L3 和本地 DRAM。
3. 现有内存层次系统主要面向 DRAM/SSD/PMem/RDMA，不能直接证明 `per-CCD navigator replica` 或 `trace-guided cold graph partition/layout` 有效。
4. 后续论文论证应避免写成“已有研究证明 chiplet-aware HNSW 有效”，只能写成“已有研究证明图 ANN 对布局和内存层次敏感，chiplet-aware HNSW 是可验证的新变量”。

## 七、优先阅读顺序

1. HNSW 原论文。
2. Graph Reordering for Cache-Efficient Near Neighbor Search。
3. HM-ANN。
4. d-HNSW。
5. DiskANN、SPANN、DiskANN++。
6. The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in HNSW。
7. FINGER。
8. 两篇综述和 ANN-Benchmarks。
9. AiSAQ、P-HNSW。
