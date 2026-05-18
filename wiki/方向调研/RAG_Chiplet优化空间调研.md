# RAG 在 Chiplet CPU 上的优化空间调研

生成时间：2026-05-11

## 目的

基于三篇 chiplet 论文（OLAP / Sorting / CHARM）提炼的三条筛选条件，调研 RAG（Retrieval-Augmented Generation）在 LLM 推理管线中是否存在 chiplet L3 局部性优化的有效空间。调研方式为文献搜索 + 理论分析，未做实验验证。

## 基础概念

### RAG

`RAG`（Retrieval-Augmented Generation，检索增强生成）把外部知识库接入 LLM。模型回答问题前，系统先从文档库中检索与用户问题相关的片段，再把这些片段拼进 prompt，让 LLM 基于检索到的上下文生成答案。

RAG 解决的问题不是让模型参数记住所有知识，而是在推理时从外部文档取证。典型用途包括企业知识库问答、代码库问答、论文问答、客服问答和需要引用来源的问答。

RAG 通常分为离线建库和在线查询两部分：

| 阶段 | 发生时间 | 主要工作 |
|---|---|---|
| 文档切分 | 离线 | 将 PDF、网页、代码、Markdown 等切成 chunk，每个 chunk 通常是几百到数千 token |
| 文档向量化 | 离线 | 用 embedding 模型把每个 chunk 编码成向量 |
| 索引构建 | 离线 | 将向量放入向量索引，如 Flat、IVF、HNSW、DiskANN |
| 查询向量化 | 在线 | 将用户问题编码成 query embedding |
| 向量检索 | 在线 | 在索引中查找与 query embedding 最相似的 top-k 文档 chunk |
| 重排序 | 在线，可选 | 用 cross-encoder 或 LLM 对候选 chunk 重新打分 |
| 上下文拼装 | 在线 | 去重、截断、排序，将检索结果拼入 prompt |
| 增强生成 | 在线 | LLM 对拼接后的 prompt 做 prefill 和 decode，生成答案 |

### 向量、相似度和 top-k

Embedding 模型会把文本映射为一个定长向量。例如一个文档 chunk 可以被表示为 `768` 维或 `1536` 维浮点向量。语义相近的文本，其向量距离较近。

检索时，系统计算 query 向量与文档向量的相似度，常见指标包括：

- `dot product`：点积，越大越相似。
- `cosine similarity`：余弦相似度，衡量方向相近程度。
- `L2 distance`：欧氏距离，越小越相似。

如果逐个计算 query 与全部文档向量的距离，就是 Flat / brute-force search。它结果精确，但数据量大时成本高。生产系统通常使用 ANN（Approximate Nearest Neighbor，近似最近邻）索引，用少量精度损失换取更低延迟。

### 常见向量索引

| 索引 | 基本思想 | 访存特征 | Chiplet 相关性 |
|---|---|---|---|
| Flat | 扫描全部向量并计算距离 | 顺序流式读取 | 通常是 DRAM bandwidth / SIMD 问题，L3 复用少 |
| IVF | 先找最近聚类中心，再扫描少数 cluster | cluster 内顺序扫描 | batch query 若共享 cluster，可能有 L3 复用 |
| HNSW | 在近邻图上做贪婪 / best-first 遍历 | 不规则图访问 | 最可能出现 L3/CCD 局部性问题 |
| DiskANN | 图索引 + SSD/DRAM 分层 | I/O + 内存混合 | 更偏存储层级，不是当前 CPU L3 主线 |

本文认为 HNSW 是最值得优先验证的点，不是因为“HNSW 一定有跨 chiplet 开销”，而是因为它具备可验证的优化抓手：图节点热度、层次结构、query 访问路径和数据布局都可被测量和改写。

## 筛选条件回顾

1. **跨 chiplet 数据访问在关键路径上占比有意义的份额**——PCM/PMU 可测。三篇论文 workload 中跨 chiplet 通信占 15%–34%，CPU attention 仅 0.14%–0.81%。
2. **该访问能通过改变数据/计算的放置来消除**——"数据被哪个 chiplet 的线程访问"是可控变量。
3. **工作集在 L3 容量边界附近且存在复用**——数据在 L3 内被多次访问。单 chiplet L3 约 32 MB，聚合 L3 约 256–512 MB。

补充判据：仅证明存在跨 chiplet 开销不够，还要证明该开销可被优化。每个候选方向都必须同时满足以下条件：

| 判据 | 必须回答的问题 | 不满足时的结论 |
|---|---|---|
| 开销归因 | 跨 CCD cache fill、远端 NUMA 页、共享锁、L3 容量 miss 中哪一项进入关键路径 | 只能作为现象观察 |
| 放置可控 | 线程、数据页、队列、索引分区或 OS 共享对象能否被稳定放到目标拓扑域 | 不能做 topology-aware 优化 |
| 替代路径 | 能否用本地复制、本地执行、远端执行、分片队列或 first-touch 替代原跨域访问 | 只能测，不能优化 |
| 代价可摊销 | 复制、分区、消息传递、排队等待、负载不均的代价是否小于节省的 stall | 可能像 acc-local-l3 一样指标变好但变慢 |
| 语义约束 | HNSW 的 recall、RAG 的端到端质量、LLM 生成吞吐和 p99 是否保持 | 不能作为有效系统优化 |

判定流程：
1. 先用 PMU / tracing 证明开销在关键路径上
2. 再构造最小 counterfactual：把目标数据或共享对象人工放到本地、复制、分片或隔离
3. 若 counterfactual 不能降低 runtime 或 p99，停止该方向
4. 若 counterfactual 有效，再设计自动化调度或布局策略

## RAG 管线拆解

```
用户Query → [1.Query Embedding] → [2.Vector Retrieval] → [3.Context Assembly] → [4.Augmented Generation(Prefill+Decode)]
                                                                                       ↑
                                                                            [5.Re-Ranking(optional)]
```

上图是在线查询路径。各阶段含义如下：

1. **Query Embedding**：把用户问题转换成向量。输入是一段文本，输出是一个向量。
2. **Vector Retrieval**：在向量库中找与 query 向量最接近的文档 chunk。该阶段通常决定检索延迟。
3. **Re-Ranking**：对初步检索出的候选文档重新打分。它更准，但计算成本更高，因此通常只处理几十到几百个候选。
4. **Context Assembly**：把候选文档去重、排序、截断，拼接成 LLM prompt 的上下文部分。
5. **Augmented Generation**：LLM 读取“问题 + 检索上下文”，生成答案。这个阶段在 vLLM 中对应 prefill 和 decode。

## 各阶段 Chiplet 敏感性分析

### 阶段 1：Query Embedding（编码查询向量）

- **访存模式**：小模型（BGE/GTE/sentence-transformer）对单条 query 做一次 forward pass
- **工作集**：query 本身 < 1 KB，模型权重 streaming 读取
- **判定**：与已分析的 GEMM/线性层相同——工作集要么 fit in L2，要么远超 L3。**条件 3 不满足。排除。**

### 阶段 2：Vector Retrieval（向量检索）——最相关阶段

#### 2a. Flat（暴力搜索）：不满足条件

- `scores = query @ doc_embeddings^T`，对整个 embedding 矩阵做 GEMV
- 每个 document vector 恰好被读一次 → streaming 无复用
- 与 attention K/V 扫描的访存模式相同 → **条件 3 不满足。排除。**

#### 2b. IVF（倒排索引 + 聚类扫描）：batch 场景可能满足条件

- 两阶段：(1) 找最近 centroids → (2) 在选中 cluster 内做精确搜索
- **Centroids**（数 MB 级）：被所有 query 共享，但体量太小，单一 chiplet L3 就能装下
- **单 query cluster 扫描**：streaming 读，无复用 → 排除
- **Batch retrieval**：多个 query 同时到达时，如果共享相同 cluster（CaGR-RAG 发现非相邻 query 间 cluster 重叠率 >60%），cluster 数据在同一批次内被多个 query 复用 → **条件 3 可能满足**
- 复用模式类似于 GPU Attention NUMA 论文中的 ACC 概念——共享同一 cluster 的 query 构成 "Retrieval Compute Cluster"（RCC）

#### 2c. HNSW（图遍历近似搜索）：最可能满足条件

- 从入口节点出发，沿边遍历多层近邻图，每一步都要比较当前候选节点与 query 的距离
- **不规则的随机内存访问**，但它是否足以形成可优化的跨 chiplet 开销，需要实测验证，而不能只凭“随机访问”下结论

| 特征 | CHARM 图计算 | HNSW 检索 | Attention |
|------|-------------|-----------|-----------|
| 访问模式 | 不规则随机访问（邻居遍历） | 不规则随机访问（图边遍历） | 流式顺序访问 |
| 跨 chiplet 后果 | 百万次量级 remote cache fill | **待测量** | 0.14%–0.81% |
| 数据复用 | 图节点被多轮 frontier 回访 | 热门节点可能被多 query 回访 | 无 |
| CHARM 优化效果 | 2.81x (Graph500) | **未知** | 无效 |

HNSW 的关键不是“有图”，而是“图的层次结构 + query 的访问路径可以被改写”。

### HNSW 的基本流程

#### 建索引

1. 每个向量作为图中的一个节点。
2. 新节点插入时，随机分配一个层级。
3. 先从最高层的入口点开始，逐层向下搜索，找到与新节点接近的已有节点。
4. 在每一层，把新节点连接到若干个最近邻，形成稀疏近邻图。
5. 低层节点更多，边更密；高层节点更少，像“导航层”。

#### 查索引

1. 从最高层的入口点开始。
2. 在每一层做贪婪下降：只要某个邻居离 query 更近，就沿着更近的方向走。
3. 一直下降到第 0 层。
4. 在第 0 层做 best-first 搜索：维护候选集合和结果集合，优先扩展更接近 query 的节点。
5. 扩展到 `efSearch` 指定的候选规模后，返回 top-k 结果。

#### 关键参数

| 参数 | 含义 | 影响 |
|---|---|---|
| `M` | 每个节点保留的邻居数 | 越大，图越密，召回通常更高，内存更大 |
| `efConstruction` | 建图时搜索候选规模 | 越大，图质量越高，建图更慢 |
| `efSearch` | 查询时搜索候选规模 | 越大，召回更高，查询更慢 |
| `top-k` | 最终返回结果数 | 业务要求，通常很小，如 5、10、20 |

#### 为什么 HNSW 可能有优化空间

- 高层入口节点和热点节点会被很多 query 反复访问，可能形成可复制的热区。
- 第 0 层的候选扩展会产生大量随机读，容易受内存布局影响。
- query 之间的访问路径可能重叠，batch 场景存在共享热路径。
- 如果把热节点、副本、布局和 query 分组做对，HNSW 的 p99 可能下降；如果只做线程绑核，不改布局，收益可能很小。

#### 为什么 HNSW 也可能没有收益

- 如果距离计算本身占主导，布局优化会被 SIMD 计算掩盖。
- 如果 query 分布很散、热路径不稳定，复制和分组会增加维护成本。
- 如果索引已经被高质量重排，chiplet 级再优化的增量可能有限。

### 阶段 3：Context Assembly（上下文拼装）

- tokenization + embedding lookup
- 数据量：数百到数千 tokens
- **工作集小，无 L3 级复用。排除。**

### 阶段 4：Augmented Generation（增强生成）

- Prefill（含检索文档的长前缀）+ Decode
- **已有实验结论**（[2026-05-06 ACC 结论](../实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md)）：attention prefill 和 decode 对 L3 容量不敏感，跨 CCD 通信仅 0.14%–0.81%
- **RAG 特有差异**：检索文档使前缀大幅增长，但 KV 仍是 streaming access → 量变非质变。多轮 RAG 中同一文档的 KV cache 可跨请求复用，但前提是 batch 内请求检索结果重叠率高。
- **判定**：主路径已排除。共享 prefix 的 chiplet 放置是边际机会。

### 阶段 5：Re-Ranking（可选的重排序）

- Cross-encoder 对 query-document pair 打分
- 相似于 embedding 模型推理，工作集小
- **排除。**

## 两个存在优化空间的路径

### 路径 A：Chiplet-Aware HNSW 图分区与遍历

**对应三条件验证**：

| 条件 | 论证 |
|------|------|
| 条件 1（跨 chiplet 访问占比） | HNSW 图遍历的不规则随机访问会产生跨 chiplet cache fill，类似 CHARM 中的图计算（Tab.1 显示 NUMA-aware 调度下 remote chiplet access 达百万次量级）。**需 PMU 实测验证占比** |
| 条件 2（放置可控性） | 图可以按 chiplet 分区（如 METIS 图分区），将相邻节点放在同一 chiplet 内存中，减少跨 chiplet 边。线程绑定到 chiplet → 遍历尽量落在本地 |
| 条件 3（L3 边界复用） | 图的高层入口节点少 + 被高频访问 → 适合完整驻留在每个 chiplet 的本地 L3（复制）。中低层节点按 chiplet 分区驻留 |

**已有间接证据**：
- CHARM 在图计算上的 2.81x speedup 证明了 chiplet 放置对不规则图遍历的有效性
- CaGR-RAG 证明了 query 按 cluster 分组的收益（51.55% 尾延迟降低），但未落到 chiplet 层级
- Elasticsearch simdvec 证明了向量搜索的 cache miss 是核心瓶颈（预取将 cache miss 从 139K 降到 19K，IPC 翻倍）

**与三篇 chiplet 论文的机制对照**：

| 论文机制 | HNSW 中可能的对应 | 适用性 |
|----------|-----------------|--------|
| OLAP WICP：独立地址空间切断 coherence | HNSW 只读索引无 write coherence，机制不直接适用 | 低 |
| Sorting：去 shuffling、chiplet 本地写缓冲 | ANN 检索无 pass 间数据搬迁，不适用 | 低 |
| Sorting L3 容量驱动的 Local vs Mixed | 冷热分层驻留涉及 L3 边界决策 | 部分相似 |
| CHARM 图分区 + 线程绑定 | HNSW 图分区 + 线程绑定方法线相似，但 HNSW 还需处理 recall、efSearch、候选堆和 query-dependent path | 部分相似 |
| CHARM spread_rate | 检索广度调整，但 HNSW 的搜索半径由 efSearch 和 query 决定，不同于 frontier-based 迭代 | 有限相似 |

关键差异：三篇论文不处理 ANN search 的 recall、efSearch、候选堆、入口层复制、query-dependent path 和 boundary 节点。

### 路径 B：Batch Retrieval 的 Cluster-Aware Query Grouping

**对应三条件验证**：

| 条件 | 论证 |
|------|------|
| 条件 1（跨 chiplet 访问占比） | 多 query batch 检索时，不同 chiplet 上的线程交叉访问同一 cluster 向量 → coherence 流量。**需 PMU 实测** |
| 条件 2（放置可控性） | 将共享相同 cluster 的 query 分组，每组绑定到一个 chiplet → cluster 数据只在该 chiplet L3 驻留 |
| 条件 3（L3 边界复用） | 热门 cluster 的数据量在 10–50 MB 范围 → 处于单 chiplet L3 (32 MB) 边界附近，placement 策略决定本地 L3 命中还是 DRAM |

**复用模式与 ACC 的类比**：

```
GPU Attention NUMA: 同一 ACC = 共享同一组 K/V 的 workgroup 集合
                     ↓ 迁移到 CPU 检索
Batch Retrieval:    同一 RCC = 共享同一 cluster 的 query 集合
```

核心逻辑相同：先识别"谁在共享同一份只读数据"，再把共享者放到同一个局部域。

## 方法相似性分析：与 CHRAM 等论文的差异

### OLAP / Sorting / CHARM 是否已覆盖 HNSW/向量检索？

**没有。** 三篇论文覆盖的 workload：

| 论文 | 覆盖的应用 | 是否含向量检索/ANN |
|------|-----------|-------------------|
| OLAP | Presto, SingleStore, SparkSQL (TPC-H/TPC-DS) | 否 |
| Sorting | LSB Radix-Sort, Comparison-Sort | 否 |
| CHARM | BFS/PageRank/SSSP/Graph500, GUPS, SGD, DuckDB OLAP, YCSB/TPC-C OLTP | 否 |

CHARM 做了图计算（BFS/PageRank/SSSP），但那是迭代式图算法（frontier-based traversal，每轮更新节点状态），不是 ANN 检索中的贪婪图搜索（beam search + distance computation）。

### 方法相似性风险

如果只是把 CHARM 的图分区方法搬到 HNSW 上（"分区 → 绑线程 → 减少跨 chiplet 边"），审稿人大概率会指出方法线雷同。逐机制对照：

| 论文机制 | 直接搬到 HNSW/向量检索 | 相似度评估 |
|----------|----------------------|-----------|
| OLAP WICP（独立地址空间切断 coherence） | 向量索引是只读的，本来就没有 write coherence 风暴 | **不适用** |
| Sorting 去 shuffling | ANN 检索没有 pass 间数据搬迁步骤 | **不适用** |
| Sorting L3 容量驱动的 Local vs Mixed 选择 | 热门 cluster/图节点的驻留策略涉及 L3 边界决策 | 部分相似 |
| CHARM spread_rate | HNSW 的搜索半径由 efSearch 和 query 决定，不同于 frontier-based 迭代 | 有限相似 |
| CHARM 图分区 + 线程绑定 | HNSW 图分区方法线表面相似，但 HNSW 还需处理 recall、efSearch、候选堆、入口层和 query-dependent path | 部分相似 |

关键区别：CHARM 不处理 ANN search 的 recall、efSearch、候选堆、入口层复制、boundary 节点和 query-dependent path。直接把 CHARM 的图分区搬到 HNSW 上创新性不足。HNSW chiplet 优化的真正抓手是四个设计变量（见下文方向）。

这些差异只说明“有哪些变量可以动”，不说明“动了之后一定有效”。判断是否值得做 chiplet 优化，还必须继续问三个问题：

1. 该差异对应的开销是否在关键路径上。
2. 该开销是否能被放置、复制、分区或远端执行替代。
3. 替代后的收益是否大于维护副本、消息传递或负载不均的代价。

### 三个差异化角度

#### 差异 1：只读数据使能热复制和远端执行

三篇论文面对的都是**读写混合**数据。向量索引在检索期间是**纯只读**的，但只读本身不是贡献点——它只说明两个设计可行：

- **热数据复制**：入口层节点、热点 hub 节点、短 neighbor list、压缩向量码可复制到每个 CCD 的本地热区，不需要处理写失效。多 chiplet 同时持有同一 cache line 的 shared 副本完全合法
- **远端执行**：搜索过程中访问远端分区时，可以把 query 向量、候选堆摘要和待扩展节点 ID 发给远端 CCD worker，由远端 worker 在本地索引上扩展，再返回少量候选结果。只读索引允许多个 worker 并行读同一图，不需要锁住图结构

不可直接得出的结论：
- 不能说复制免费。复制消耗 L3 容量、DRAM 容量和构建时间
- 不能复制完整索引。大规模 HNSW 的向量和邻接表远超单 CCD L3（~32 MiB），只能复制高层和热点子集

核心 tradeoff 从"消除 coherence 风暴"变为副本放置策略：全复制 vs 分区 vs 混合（热复制 + 冷分区）。该 tradeoff 类似 cache coherence 中的 replication vs migration，但应用于算法层面的数据放置决策。

#### 差异 2：距离计算的计算强度可变

三篇论文的 workload 几乎都是 memory-intensive。ANN 检索不同：
- 低维 embedding（128d）：距离计算很轻，瓶颈在 graph traversal → 偏 memory-bound
- 高维 embedding（4096d）：距离计算成为瓶颈 → 偏 compute-bound
- **维度参数决定了瓶颈在 chiplet 的哪一层**，这决定了 chiplet 优化应该在哪个层面介入

这个维度依赖性是三篇论文完全没有的。

#### 差异 3：多 query 并发 × 只读共享图的独特交互

CHARM 的图计算是一个算法实例遍历一张图。生产环境的向量检索是几十到几百个 query 同时遍历同一张只读图：
- 多个 query 的遍历路径在图上可能交叉（同一节点被多个 query 访问）
- 由于图是只读的，这种交叉不会产生 coherence 冲突
- 但会产生**隐式的 L3 共享**：query A 加载的节点恰好也被 query B 需要
- 这引入了 CHARM 没有的问题：**query 之间的调度顺序是否应基于它们在 HNSW 图上的访问重叠来优化？**

#### 差异 4：层次化图结构 × 缓存层级的自然映射

HNSW 有天然的层次结构：
- **上层**（入口层）：节点少（数十到数百），被几乎所有 query 访问 → 自然适合 L2/L3 复制
- **下层**（0 层）：节点多（数百万到十亿），每个 query 只访问少数 → 适合按 chiplet 分区

这种层次结构与 chiplet 的缓存层级（L2 私有 → L3 chiplet 本地 → 聚合 L3 → DRAM）可以天然对齐。

### 四个可行的 HNSW Chiplet-Aware 设计

#### 设计 1：Replicated Navigator（per-CCD 导航层复制）（推荐）

**对应差异**：差异 1（只读）+ 差异 4（层次化结构）

每个 CCD 保留一份小型导航层，预算建议先设为每 CCD 4–8 MiB，避免挤占单 CCD 32 MiB L3：

- HNSW 第 1 层及以上所有节点（几十到数百个）
- 第 0 层 top-k 热点节点的邻接表和压缩向量码
- 全局 entry point 和若干 per-cluster entry points

目标：减少每个 query 开头阶段的全局热节点争用和重复 DRAM 读取。

**可优化性条件**：
- 热节点访问必须占 query stall 的可观比例（需 PMU 验证）
- 热副本规模必须明显小于单 CCD L3。若热集超过 8–16 MiB/CCD，复制会挤占底层搜索数据
- query 入口路径要稳定。若热点随 workload 快速变化，维护成本会上升
- 该设计只优化高层导航和热点 hub，不优化底层冷节点随机访问

#### 设计 2：Cold Graph Topology Layout（冷图拓扑布局）（推荐）

**对应差异**：差异 2（计算强度）+ 差异 3（多 query 并发）

底层大图不复制，按 sampled query trace 分区和重排：

1. 用 sampled query 得到节点转移权重 T(u,v)
2. 先按向量 cluster 粗分区，保证近邻大概率同区
3. 再在分区内按访问转移顺序重排节点，提升空间局部性
4. 每个分区 first-touch 到一个 CCD 所属的 NUMA/NPS domain
5. 边界节点保留跨分区边，不限制 recall 正确性

目标：高概率连续访问落在同一 CCD/L3 局部域。

**与 graph reordering 的区别**：已有方法优化单一线性布局；此处优化的是"线性布局 + CCD 分区 + 热点复制预算"。与 graph reordering baseline 比较后仍有增量才算有效。

**可优化性条件**：
- sampled query trace 的转移概率要稳定。若训练和线上 query 分布差异大，layout 会失效
- 分区后本地转移比例必须上升。仅按节点 ID 或向量 cluster 分区不足以证明有效
- first-touch / NUMA placement 必须能把分区页面稳定放到目标 NPS domain

#### 设计 3：Remote Expansion by Message Passing（远端扩展执行）（高风险高回报）

**对应差异**：差异 1（只读使能远端执行）

默认 HNSW 搜索在本线程访问远端节点时会把远端 cache line 拉到当前 CCD。更合理的 chiplet 设计是移动小的 query state 而不是移动大的索引数据：

- 本地 CCD worker 扩展本地候选节点
- 遇到远端分区节点时，把 (query_id, query_vector, candidate_ids, heap_threshold) 发送到远端 worker
- 远端 worker 在本地索引分区扩展候选，返回 top-m 候选和距离
- 本地 worker 合并候选堆，继续搜索或结束

这是 chiplet-aware 的核心创新点，比简单线程绑定更强。

**可优化性条件**：
- 远端节点扩展的数据量必须大于消息状态。若只读一两个 cache line，发送消息不划算
- 跨分区访问要集中成批。若远端节点零散出现，队列和同步开销会超过收益
- 远端 worker 必须有可用并行度，否则排队拉高 p99
- 必须证明"移动 query state"比"硬件拉远端 cache line"更快（remote expansion on/off 对照）

#### 设计 4：Query Grouping for Hot-path Locality（batch 场景 query 分组）

**对应差异**：差异 3（多 query 并发共享图）

batch 场景下，query 先经过轻量路由：

1. 用 coarse centroid 或上层 HNSW 前几跳预测 home partition
2. 将 home partition 相同或热节点重叠高的 query 放入同一 CCD queue
3. 同一 queue 内按 cluster / entry path 排序执行

**与 CaGR-RAG 的关系**：CaGR-RAG 面向 disk-based vector search，目标是减少 disk/page-cache 随机访问。这里目标是 in-memory HNSW 在 CCD/L3/NUMA 层面的热路径复用。可借鉴 query grouping，但不能直接引用其收益作为 chiplet 收益。

**可优化性条件**：
- batch 内 query 必须存在可预测的 hot-path overlap
- 分组延迟必须小于 locality 收益。对在线低 QPS 请求，等待成 batch 不可接受
- 同组 query 不能造成单 CCD 负载热点，需要 work stealing 或动态阈值
- 该方法适合 batch/p99 优化，不适合单 query latency 主线

#### 补充方向：RAG Pipeline 的 Chiplet 资源隔离（低优先级）

检索和生成在同一 chiplet CPU 上运行时可能相互污染 L3。可按 chiplet 做空间分区：一部分专属检索、一部分专属生成。但 attention 对 L3 不敏感的已有结论意味着生成端 chiplet 隔离可能收益很低，需先测检索端是否 L3 敏感。

## 建议的下一步：最小验证实验

先不做完整实现，按以下顺序验证前提：

1. **PMU 基线**：复现未优化 HNSW 的 PCM/IBS——p99、recall、L3 miss latency、`another CCX same node`、local/remote memory。确认跨 CCD 开销占比进入关键路径
2. **CAT 上界**：把 L3 从 `ffff` 压到 `0001`。若 HNSW p99 明显退化而 attention 不退化，说明 HNSW 有 L3 敏感性
3. **来源上界**：用 PCM/IBS 看 another CCX same node、local memory、remote memory、L3 miss latency。如果跨 CCD/远端内存来源占比只有 ~1%，chiplet 优化上限很低
4. **Layout 上界**：复现 graph reordering。已有工作报告 HNSW query time 最高可降约 40%；chiplet-aware layout 的增量应与该基线比较，而非与未优化 HNSW 比
5. **人工 counterfactual**：手动复制高层节点、固定分区 first-touch、强制 query 固定到 home CCD，确认开销可被放置改变
6. **加入 per-CCD navigator replica + cold graph partition**，对比 graph reordering 后的增量
7. **加入 remote expansion + query grouping**，只在 batch/p99 场景评估

预期范围（研究假设，非结论）：

| 场景 | 合理预期 |
|---|---|
| 未优化 HNSW，低维，in-memory，大 batch | 可能有 10%–30% p99 改善 |
| 已做 graph reordering / prefetch 的 HNSW | chiplet 增量可能只有 3%–15% |
| 高维 fp32，距离计算主导 | 可能低于 5% |
| query 分布无热点、batch overlap 低 | 可能无收益 |

## 关键约束

1. **检索服务与 LLM 推理是独立部署的**：vllm CPU backend 本身不包含向量检索。chiplet 优化落在 FAISS / 向量数据库侧。这既是约束（不能直接在 vllm 代码上做），也是机会（独立的 benchmark 和实验空间）。
2. **数据规模与 L3 边界的关系需实测**：论文中的 32 MB 边界针对 EPYC Milan。HNSW 图节点 + 邻接表的数据量、IVF cluster 的数据量在不同数据集上差异很大。
3. **Chiplet 数量是关键前提**：OLAP 论文在 ARM Graviton 3（单 compute chiplet）上 WICP 几乎无收益。chiplet 数量少时收益会显著小于 8 chiplet 的 AMD EPYC。
4. **与现有优化的关系**：Elasticsearch simdvec、FAISS 的预取/SIMD 优化是计算和微架构层面的，chiplet 放置是数据分布和线程映射层面的，两者正交可叠加。
5. **CCD 不包含内存控制器**：内存控制器在 IOD，OS 能直接控制的是 socket / NUMA / NPS domain，不是"每 CCD 一个 DRAM 池"。chiplet 优化的对象是 CCD 的 L3 cache 局部性和 NUMA locality，不是 CCD-local DRAM。
6. **维度决定瓶颈位置**：HNSW 检索的瓶颈可能在内存访问也可能在距离计算，取决于 embedding 维度和编码方式。实验必须做维度和编码 sweep：

| 配置 | 预期瓶颈 | 对 chiplet layout 的价值 |
|---|---|---|
| 128d fp32 / bf16 | 图访问 + 部分距离计算 | 高 |
| 768d fp32 | 距离计算占比升高 | 中 |
| 1536d fp32 | SIMD compute 占比高 | 低到中 |
| PQ/SQ codes | 图和 codebook locality | 高 |

## 核心结论

**RAG 在 chiplet CPU 上存在有意义的优化空间，但不在 LLM 推理环节，而在向量检索环节（HNSW 图遍历）。**

- **Attention / GEMM / Decode / Prefill**：已通过 CAT 实验和 PCM 数据排除（跨 CCD 通信 <1%，L3 容量不敏感）
- **Vector Retrieval（HNSW 图遍历）**：不规则的图随机访问可能产生跨 CCD 开销。研究价值不来自"只读数据不同于读写混合数据"这类泛化差异，而来自四个可操作设计变量：per-CCD 导航层复制、query-trace 引导的冷图布局、远端扩展执行、batch query 热路径分组
- **Vector Retrieval（IVF batch 查询）**：当 query batch 内 cluster 重叠率高时，存在类似 ACC 的共享工作集复用优化空间
- **差异化关键**：不是简单复用 CHARM 的图分区方法，而是在 HNSW 的 recall、efSearch、候选堆、入口层和 query-dependent path 约束下，同时利用层次结构、只读特性和 query trace 做 chiplet-aware 设计。所有设计必须通过 PMU counterfactual 验证，并与 graph reordering baseline 比较增量

## 创新性边界

### 与 CHARM 的区别

CHARM 优化通用 graph analytics / OLAP / SGD 的任务放置和 spread_rate，不处理 ANN search 的 recall、efSearch、候选堆、入口层、边界节点复制和 query-dependent path。

本方向的创新点：
- 面向 ANN/RAG retrieval 的 chiplet 微架构测量
- 利用 HNSW 层次结构做 per-CCD navigator replication
- 用 sampled query trace 优化图布局（而非 general graph partitioning）
- 用 remote expansion 将计算移到数据所在 CCD
- 同时约束 recall、p99 和 PMU 来源

### 与 graph reordering 的区别

Graph reordering 证明"内存布局影响 HNSW 性能"，但通常是单一线性布局。这里增加 CCD 分区目标、热点复制预算、NPS/NUMA first-touch、batch query 到 CCD queue 的调度、远端扩展执行。若最终只做线性重排而不做 CCD/NPS 相关设计，创新性不足。

### 与 CaGR-RAG 的区别

CaGR-RAG 关注 RAG 中 query 对 disk-based vector search cluster 的重叠，核心收益在 disk/page-cache 和 cluster 级调度。这里关注 in-memory ANN 在 chiplet CPU 上的 L3/NUMA/CCD 放置。可引用其"query overlap 存在"，不能直接继承其方法和收益。

## 参考文献

- [RAGO: Systematic Performance Optimization for RAG Serving (ISCA 2025)](https://arxiv.org/abs/2503.14649)
- [CaGR-RAG: Context-aware Query Grouping for Disk-based Vector Search](https://arxiv.org/abs/2505.01164)
- [The Faiss Library (2024)](https://arxiv.org/abs/2401.08281)
- [VectorLiteRAG: Latency-Aware Resource Partitioning for Efficient RAG](https://arxiv.org/abs/2504.08930)
- [RAG at Scale: Accelerating Prompt Assembly with CXL (UT Austin 2025)](https://repositories.lib.utexas.edu/items/1f3c7695-b6b3-4a23-a7e1-ca17e45b6483)
- [TeleRAG: Efficient RAG Inference with Lookahead Retrieval](https://arxiv.org/abs/2502.20969)
- [PipeRAG: Fast RAG via Algorithm-System Co-design](https://arxiv.org/abs/2403.05676)
- [FusionANNS: CPU/GPU Cooperative Processing for Billion-scale ANNS (FAST 2025)](https://www.usenix.org/conference/fast25/presentation/tian-bing)
- [Graph Reordering for Cache-Efficient Near Neighbor Search](https://arxiv.org/abs/2104.03221)
- [Elasticsearch simdvec: SIMD-accelerated vector search engine](https://www.elastic.co/search-labs/blog/elasticsearch-vector-search-simdvec-engine)

## 相关内部文档

- 三篇 chiplet 论文解读：`wiki/论文解读/OLAP_on_Modern_Chiplet_Based_Processors解读.md`、`Optimizing_Sorting_for_Chiplet_Based_CPUs解读.md`、`CHARM解读.md`
- 三篇论文对照：`wiki/论文解读/Chiplet三篇论文对照与Attention结论验证.md`
- 2026-05-06 ACC 结论：`wiki/实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md`
- LLM 推理链路扫描：`wiki/资料总览/LLM推理链路Chiplet优化空间扫描.md`
- 论文方法线对比总览：`wiki/论文解读/论文方法线对比总览.md`
