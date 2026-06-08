# Graph-Based Vector Search 实验评估解读

**一句话概述**：论文系统梳理 graph-based vector search（基于图的向量检索）的设计范式，并在 7 个真实数据集、多个合成分布和最高 10 亿向量规模上评估 12 种 state-of-the-art 方法。论文将方法归纳为 seed selection、incremental insertion、neighborhood propagation、neighborhood diversification、divide-and-conquer 五类范式，结论是 HNSW、Vamana、ELPIS 属于综合表现较强的方法组，但方法排名受数据规模、查询 recall、图构建方式、实现优化和内存布局影响。论文不研究 chiplet CPU、CCD/L3 或 NUMA locality，不能直接作为 chiplet-aware HNSW 收益证据。

---

## 背景：图向量检索基础

### Vector search

`Vector search`（向量检索）在向量集合中查找与查询向量最相似的对象。相似度通常由欧氏距离、余弦距离或内积等度量定义。暴力扫描需要比较查询向量和所有数据向量，时间复杂度为 `O(nd)`，其中 `n` 是向量数量，`d` 是维度（Sec. 1）。

向量检索支撑推荐、信息检索、聚类、分类、异常检测、实体解析、数据发现和 RAG（Retrieval-Augmented Generation，检索增强生成）等任务。论文指出，embedding 数据规模可达到数十亿向量和上千维，传统精确搜索难以满足毫秒级延迟需求（Sec. 1）。

### Graph-based ANN

`ANN`（Approximate Nearest Neighbor，近似最近邻）通过牺牲部分精确性换取速度。Graph-based ANN 将向量组织为图，节点是向量，边连接近邻或有导航价值的向量。查询从一个或多个种子节点开始，沿图搜索逐步接近查询向量（Sec. 2-3）。

论文强调，graph-based 方法在许多实际应用中成为主流，因为它们可在 terabyte-scale collection 上实现低毫秒级查询延迟，并在实践中保持较高 accuracy（Sec. 1）。但图方法内部设计差异大，单纯比较算法名称无法解释性能差异。

## 研究问题

论文要回答三个问题：

1. graph-based vector search 方法可按哪些核心设计范式分类（Sec. 3）。
2. 不同范式对索引构建、内存占用、查询时间、distance calculations 和 recall 有何影响（Sec. 4）。
3. 在数据规模增大、数据分布变化和实现优化后，哪些方法仍保持稳定表现（Sec. 4-6）。

论文范围限定为 in-memory dense vector search，不覆盖 out-of-core 方法，也不做理论复杂度分析（Sec. 1）。

## 核心内容

论文贡献包括：

1. 提出 graph-based vector search 的五类设计范式（Sec. 3.2）。
2. 构建 taxonomy，按时间线和影响关系梳理代表方法（Sec. 3.5, Fig. 3）。
3. 评估 12 种 state-of-the-art 方法，在 7 个真实数据集和合成数据上覆盖最高 1B 向量规模（Sec. 4）。
4. 分析 seed selection 和 neighborhood diversification 的独立影响（Sec. 4.2-4.3）。
5. 讨论可扩展性、实现优化和后续研究方向（Sec. 5-6）。

## 方法分类

### 五类设计范式

| 范式 | 英文全称 | 含义 |
|---|---|---|
| SS | Seed Selection | 查询开始时选择一个或多个搜索入口点 |
| II | Incremental Insertion | 逐个插入向量并在插入过程中建立邻接关系 |
| NP | Neighborhood Propagation | 通过邻居传播扩展候选近邻并优化图 |
| ND | Neighborhood Diversification | 对候选邻居做多样化剪枝，降低冗余边 |
| DC | Divide-and-Conquer | 将数据划分为子集，分区构图或分区搜索 |

这五类范式不是互斥分类，一个方法可同时包含多种范式（Sec. 3.2, Fig. 3）。

### 代表方法归类

论文将代表方法归类如下（Sec. 3.5-3.6, Fig. 3）：

| 方法 | 主要范式 |
|---|---|
| HNSW | II + ND |
| Vamana | ND |
| NSG / SSG | NP + ND |
| SPTAG | DC + SS + ND |
| ELPIS | DC + II + ND |
| HCNNG | DC + SS |
| KGraph / EFANNA | NP |

该分类说明，HNSW 的核心不只是层次结构，也包含 incremental insertion 和 neighborhood diversification；Vamana、ELPIS 等方法虽然没有完全相同层级结构，但可在图构建、邻域剪枝和分区组织上形成强 baseline。

## 实验设置
### 硬件与软件

实验平台为 Intel Xeon Platinum 8276 四路 28 核服务器，1.5TB RAM。操作系统为 Ubuntu 20.04 或 Rocky Linux 8.5，编译器为 GCC 8.2。论文报告 wall-clock time、distance calculations、recall、memory footprint 和 beam width（Sec. 4.1）。

查询设置为顺序单条执行，不是 batch。默认查询为 10-NN；复杂度分析和 seed selection 实验使用 100-NN。每组实验运行 6 次，去掉最好和最差结果。单次索引构建设置 48 小时上限（Sec. 4.1）。

### 数据集

真实数据集包括 Deep、Sift、SALD、Seismic、Text-to-Image、GIST、ImageNet1M，规模最高到 1B 向量。合成数据包括 RandPow0、RandPow5、RandPow50，用于控制数据分布 skewness（Sec. 4.1）。

论文用 `LID`（Local Intrinsic Dimensionality，局部内在维度）和 `LRC`（Local Relative Contrast，局部相对对比度）衡量数据难度。Fig. 4 显示，Pow0/5/50、Seismic、Text-to-Image 更难，Sift、Deep、ImageNet 更易（Sec. 4.1, Fig. 4）。

### Neighborhood diversification 实验

ND 实验使用 II 图作为底座，固定 `R=60, L=800`，比较 RND、RRND、MOND 和 NoND。NoND 表示不做邻域多样化，作为 baseline（Sec. 4.2, Fig. 5）。

### Seed selection 实验

SS 实验在 II+RND 图上比较 SN、KD、KS、MD，并加入 SF baseline。该实验用于分离入口点选择对索引和搜索性能的影响（Sec. 4.3, Fig. 6）。

## 结果与解释

### Neighborhood diversification 是核心变量

ND 实验显示，RND 和 MOND 表现最好，RRND 次之，NoND 最差；数据规模越大，差距越明显（Sec. 4.2, Fig. 5, Table 1）。

该结果说明，邻域剪枝不能只追求更强 pruning。图搜索需要保留足够连通性和导航方向。过度删除边可能降低图可达性，使查询需要更多扩展或陷入局部区域（Sec. 4.2, Table 1）。

### Seed selection 的最优选择依赖规模

SS 实验显示，SN 和 KS 表现最强。KD 在 25GB/100GB 规模上仍可接受，SF 和 MD 表现较差。SN 在 1B 规模上优于 KS，但在小/中规模上 KS 距离计算更少。Table 2 显示，SN 比 KS 多 182M 次距离计算（Deep1M）和 22.3B 次距离计算（Deep25GB）（Sec. 4.3, Fig. 6, Table 2）。

该结果提示，HNSW 的层级 seed selection 对大规模有效，但小/中规模场景中，较简单的 K-random 或其他入口策略可能更省构建和搜索成本。

### 索引构建可扩展性

索引构建实验显示，II 基方法通常较快。100GB 以上规模时，HNSW、ELPIS、Vamana 仍能保持可接受构建时间。ELPIS 构建最快，索引内存更低，最高比 HNSW 少 40%（Sec. 4.4, Fig. 7-9）。

论文也指出，HNSW 的 contiguous block allocation 可减少指针间接访问和 cache miss，但在大 out-degree 和大规模数据上可能带来近似二次内存增长（Sec. 4.4）。

### 查询性能

查询实验显示，25GB 及以上规模 ELPIS 常位于前列；1B 规模上，ELPIS 在 0.95 recall 可比其他方法快一个数量级。Table 3 总结 HNSW、Vamana、ELPIS 属于搜索和索引效率最好的方法组（Sec. 4.5, Fig. 12-16, Table 3）。

数据分布实验中，ELPIS 对 skewness 0 到 50 的分布变化保持较强或较稳表现（Sec. 4.5, Data Distributions）。

### 实现优化会改变排序

Fig. 17 比较优化实现后，ParlayANN 的 HNSW_Opt 和 Vamana_Opt 在 recall < 0.97 时更快。该结果说明，算法名称不是唯一决定因素；内存布局、并行化和实现细节会改变同类方法的性能排序（Fig. 17）。

## 分析

- HNSW 是强 baseline，但不是唯一强 baseline。Vamana 和 ELPIS 在若干规模和 recall 目标下也具有竞争力（Table 3）。
- Graph ANN 的性能不能只归因于搜索阶段。seed selection、图构建、邻域多样化、分区策略和内存布局均会影响最终曲线。
- 数据集难度会改变方法排序。LID/LRC、skewness、维度、距离分布都会影响图导航效率（Fig. 4）。
- 实现层影响必须单独控制。若后续做 chiplet-aware layout，应与优化实现、普通 contiguous layout、普通 graph reordering 分开对比（Fig. 17）。

## 边界

- 论文只覆盖 in-memory dense vector search，不覆盖 out-of-core、SSD、PMem、RDMA、CXL 或 disaggregated memory（Sec. 1）。
- 实验平台为四路 Intel Xeon Platinum 8276，不是 AMD Zen chiplet CPU；论文没有 CCD/L3、NPS、NUMA source latency 或 IBS 计数器数据（Sec. 4.1）。
- 查询为顺序单条执行，不是 batch，也不是多租户在线服务；不能直接外推到 RAG retriever 的批量请求场景（Sec. 4.1）。
- 1M queries 结果部分由 100-query set 外推，不等于全部逐条真实运行（Sec. 4.1, Fig. 4）。
- 部分方法因实现问题或性能差被排除，结论不覆盖全部 graph ANN 实现（Sec. 4.1）。
- 结果依赖作者统一参数调优和代码修改，跨库复现时排名可能变化（Sec. 4.1, Fig. 17）。

## 可迁移点

- HNSW、Vamana、ELPIS 可作为后续 HNSW-on-chiplet 实验 baseline 候选，避免只与未优化 HNSW 对比。
- ND、SS、DC 是必须控制的算法变量。chiplet-aware layout 若有收益，应证明其相对普通 ND/SS/DC 优化仍有增量。
- HNSW 的 contiguous block allocation 与 cache miss 关系可作为布局优化切入点，但必须同时报告内存开销（Sec. 4.4）。
- 数据集难度指标 LID/LRC 可用于选择 HNSW 实验数据集，避免只在简单数据上得出过强结论。

## 不可直接迁移点

- 论文没有验证 chiplet/CCD 拓扑、跨 CCD 访存、L3 分片调度或 per-CCD replica。
- 论文没有证明图方法的 cache miss 主要来自跨 CCD 访问。
- 论文的最优点建立在 Euclidean、10-NN、顺序单查询、特定实现优化和超大内存服务器上；当前 workload 若使用 batch、不同 embedding、不同 `k` 或不同 recall 目标，排序可能改变。

## 与当前 Chiplet CPU 研究的关系

该论文适合作为 HNSW 方向的 baseline 和变量控制依据。后续研究不能只把 HNSW 原实现作为唯一对照；至少应考虑 HNSW 优化实现、Vamana、ELPIS、普通 graph reordering 和不同 seed/ND 参数。若 chiplet-aware HNSW 只优于一个弱 baseline，不足以说明 CCD/L3 优化有效。

该论文不能作为 chiplet-aware HNSW 有效性的直接证据。它说明 graph ANN 对算法结构、数据分布和实现敏感，不说明 AMD EPYC CCD/L3 是瓶颈。

## 证据

- 原始资料：`wiki/原始资料/papers/HNSW/Graph-Based Vector Search - An Experimental Evaluation of the State-of-the-Art.pdf`
- 关键锚点：Abstract；Sec. 1-6；Fig. 3-17；Table 1-3
