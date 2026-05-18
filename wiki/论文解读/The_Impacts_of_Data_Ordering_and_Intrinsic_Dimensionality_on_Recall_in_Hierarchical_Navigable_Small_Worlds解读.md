# HNSW 的召回受数据顺序和内在维度影响

**一句话概述**：论文研究 HNSW（Hierarchical Navigable Small World，层次化可导航小世界图）在真实深度学习向量场景下的召回稳定性。论文固定 HNSW 参数，分别用合成向量、标准文本检索数据集和真实电商图像数据验证：HNSW 的 recall（召回率）与向量空间的 intrinsic dimensionality（内在维度）相关，且会受到数据插入顺序影响；按 pointwise Local Intrinsic Dimensionality（逐点局部内在维度，简称 LID）或商品类别组织插入顺序，可让 recall 出现最高 12.8 个百分点的差异（Abstract, Table 5）。

---

## 问题

论文关注的不是 HNSW 参数调优，而是固定参数下 HNSW 对数据性质的敏感性。

现有 ANN（Approximate Nearest Neighbour，近似最近邻）评测常使用 MNIST、SIFT1M 等相对简单或低维的数据集。论文认为这类数据集不能充分代表当前 RAG、推荐和多模态检索中常见的 embedding（嵌入向量）空间，因为这些向量通常来自深度学习模型，且不同模型、数据集、类别和局部区域会形成不同的内在维度结构（Sec. 1）。

论文要回答三个问题：

1. 向量空间的内在维度升高时，固定参数 HNSW 的 recall 是否下降。
2. 数据插入 HNSW 图的顺序是否会改变最终图结构和 recall。
3. 用 exact KNN（精确 K 近邻）建立的模型排行榜，是否能直接代表 HNSW 近似检索下的模型表现。

---

## 核心结论

HNSW 的 recall 不是只由 `M`、`efConstruction`、`efSearch` 决定。固定这些参数后，数据的内在维度、局部内在维度和插入顺序仍会显著影响结果。

论文报告的关键现象如下：

- synthetic vectors（合成向量）接近 full rank（满秩）时，HNSW recall 约下降 50%（Sec. 5.2.1, Fig. 2）。
- 在 7 个标准检索数据集和 16 个 embedding 模型上，按 descending LID（LID 降序）插入数据时，平均 recall@10 高于 ascending LID（LID 升序）和 random order（随机顺序）（Table 5, Table 6）。
- `efSearch = 10` 时，descending LID 相对 ascending LID 的平均收益为 HNSWLib 2.6 个百分点、FAISS 6.2 个百分点；最大差异为 HNSWLib 5.6 个百分点、FAISS 12.8 个百分点（Table 5）。
- `efSearch = 40` 时，整体 recall 升高，但插入顺序差异仍存在；最大差异为 HNSWLib 7.1 个百分点、FAISS 11.5 个百分点（Table 6）。
- exact KNN、HNSWLib 和 FAISS-HNSW 评估同一批模型时，NDCG@10 排名可上下移动最多 3 位（Table 4）。
- 电商图像数据中，按商品类别形成的非随机插入顺序也会改变 recall；Fashion 数据集在 `efSearch = 10` 下 ViT-B-32 的 recall 差异达到约 7.7 个百分点（Table 10）。

---

## HNSW 参数固定化

论文使用 FAISS 和 HNSWLib 两个实现，因为它们构建单个 HNSW 图，不引入生产系统中的 sharding（分片）、segmentation（分段）、replica（副本）和可变更索引机制（Sec. 3）。

HNSW 的三个主要参数如下（Sec. 3.1）：

| 参数 | 含义 | 影响 |
|---|---|---|
| `M` | 每个节点形成的双向连接数；底层通常使用 `2 * M` | 影响 recall、内存和延迟 |
| `efConstruction` | 构图时候选堆大小 | 影响构图质量和构建时间，不直接影响查询延迟 |
| `efSearch` | 查询时候选堆大小 | 提高 recall，但增加查询延迟 |

论文没有为每个数据集搜索最优参数，而是调查多个向量数据库或 ANN 系统的默认值，并使用接近默认配置的固定设置。Table 1 覆盖 Marqo、HNSWLib、FAISS、Chroma、Weaviate、Qdrant、Milvus、Vespa、OpenSearch、Elasticsearch、Redis 和 PGVector 等系统。论文选择 `M = 16`，因为 16 是调查结果中的中位数；选择 `efConstruction = 128`，因为 128 是调查结果中的中位数；`efSearch` 根据任务结果数 `k` 和各系统默认值确定，实验中重点报告 `efSearch = 10` 与 `efSearch = 40`（Table 1, Sec. 3.1）。

这种设置让论文的问题变成：在工业常见默认参数附近，HNSW 对数据分布和插入顺序是否稳健。

---

## 数据集

论文使用三类数据。

### 合成向量

合成向量通过 Gram-Schmidt Orthonormalization（格拉姆-施密特正交归一化）生成正交基，再用不同数量的正交基线性组合形成数据。向量表观维度为 1024，但实际 intrinsic dimensionality（内在维度）由参与组合的正交基数量控制（Sec. 4.1）。

论文用 PCA（Principal Component Analysis，主成分分析）估计内在维度。具体做法是计算 cumulative explained variance ratio（累计解释方差比例），找到达到 `theta = 0.99` 所需的最少主成分数量，将其作为估计内在维度（Sec. 5.1）。Fig. 1 用 PCA 曲线验证不同正交基数量确实形成不同内在维度。

### 标准文本检索数据集

论文从 MTEB（Massive Text Embedding Benchmark，大规模文本嵌入评测）中选择 7 个检索数据集（Table 2）：

| 数据集 | 查询数 | 语料规模 | 任务类型 |
|---|---:|---:|---|
| NFCorpus | 323 | 3.6K | Asymmetric |
| Quora | 10k | 523k | Symmetric |
| SCIDOCS | 1k | 25k | Asymmetric |
| SciFact | 300 | 25k | Asymmetric |
| CQADupstack | 13.1k | 547k | Asymmetric |
| TRECCOVID | 50 | 171k | Asymmetric |
| ArguAna | 1.4k | 8.7k | Symmetric |

Asymmetric（非对称检索）表示查询和文档类型不同，例如用问题检索答案或文档。Symmetric（对称检索）表示查询和语料文本类型相同，例如 Quora 中用标题检索相似标题（Sec. 4.2）。

论文使用 16 个 embedding 模型生成向量，包括 bge、e5、multilingual-e5、ember-v1、all-MiniLM-L6-v2、gte-base、stella-base-en-v2 等，维度覆盖 384、768 和 1024（Table 3）。

### 真实电商图像数据

论文使用两个私有电商图像数据集（Sec. 4.3）：

- Fashion：collectibles、handbags、streetwear、sneakers、watches。
- Homewares：home、furniture、kitchenware、wall、renovation、bed、rugs、lighting、baby、lifestyle、pet、office。

图像 embedding 由 CLIP（Contrastive Language-Image Pre-training，对比式图文预训练）模型生成，使用 ViT-B-32 与 ViT-L-14 两种架构。查询是 GPT-4 生成的电商搜索词（Sec. 5.3）。

---

## 评价指标

论文把 exact KNN 作为精确检索器，把 HNSW 作为近似检索器。对查询集合 `Q` 和每个查询返回 `k` 个结果，recall@k 定义为 HNSW 返回结果与 exact KNN 返回结果的交集占 exact KNN 结果集合的比例。除非特别说明，论文使用 `k = 10`（Sec. 5）。

模型检索效果使用 NDCG@10（Normalised Discounted Cumulative Gain，归一化折损累计增益）衡量。NDCG@10 不只看是否命中，还看相关文档在前 10 个结果中的排序位置（Sec. 5.2）。

---

## 合成向量实验显示内在维度升高会降低 recall

论文首先用合成数据隔离内在维度变量。随着生成数据所用正交基数量增加，向量空间的内在维度升高。Fig. 2 显示，在 `M = 16`、`efConstruction = 128`、`efSearch = 40` 下，HNSWLib 和 FAISS 的 recall 都随正交基数量增加而下降（Fig. 2）。

论文在后续分析中概括这一现象：当 synthetic data（合成数据）接近 full rank（满秩）时，recall 约下降 50%（Sec. 5.2.1）。这说明在固定参数下，HNSW 图搜索对数据空间几何结构敏感。内在维度越高，局部近邻集合之间的重叠可能越少，图搜索更难从一个局部区域跳到另一个局部区域；论文在 Related Work 中也引用 Lin 等人的结论，指出 LID 增大时 HNSW 层次结构相对 flat search（平面搜索）的优势会减弱（Sec. 2）。

---

## LID 排序实验显示插入顺序会改变图质量

论文对每个向量计算 pointwise LID（逐点局部内在维度），估计方法是 Maximum Likelihood Estimation（MLE，最大似然估计），使用每个向量的 100 个 exact nearest neighbours（精确最近邻）（Sec. 5.2.1）。

论文比较三种插入顺序（Table 5, Table 6）：

- descending LID：先插入高 LID 向量，再插入低 LID 向量。
- ascending LID：先插入低 LID 向量，再插入高 LID 向量。
- random order：随机顺序。

作者给出的解释是：descending LID 类似 simulated annealing（模拟退火式构图过程），高 LID 区域先形成全局结构，低 LID 的紧密簇后接入；ascending LID 会较早形成 tight localities（紧密局部区域），后续高 LID 向量更难优化全局连接。论文还在 ArguAna、NFCorpus、SciFact、SCIDOCS 四个较小数据集上计算 HNSWLib 最终层平均路径长度，发现 recall 与平均路径长度的 Pearson correlation coefficient（皮尔逊相关系数）为 0.61，论文据此支持“较长路径长度对应更好 recall”的解释（Sec. 5.2.1）。

### `efSearch = 10` 的结果

在 7 个标准检索数据集上，论文对每个模型计算 recall@10，再跨数据集取平均。Table 5 显示，除 FAISS 上的 gte-base 外，descending LID 都获得最高 recall（Table 5）。

关键数值如下：

| 实现 | descending LID 相对 ascending LID 的平均提升 | 最大差异 |
|---|---:|---:|
| HNSWLib | 2.6 个百分点 | 5.6 个百分点，出现在 e5-small-v2 |
| FAISS | 6.2 个百分点 | 12.8 个百分点，出现在 all-MiniLM-L6-v2 |

随机插入通常位于 ascending LID 和 descending LID 之间。论文据此认为 LID 可用于有方向地改变 HNSW recall，而不只是造成随机波动（Table 5）。

### `efSearch = 40` 的结果

Table 6 显示，提高 `efSearch` 会提高整体 recall，但不能消除插入顺序影响。最大差异仍达到 HNSWLib 7.1 个百分点、FAISS 11.5 个百分点，均出现在 all-MiniLM-L6-v2 或 e5-small-v2 等模型上（Table 6）。

这说明增加查询候选堆可以部分缓解近似搜索漏召回，但并不完全修复构图阶段由插入顺序造成的结构差异。

---

## HNSW 近似检索会改变模型排行榜

论文用 exact KNN、HNSWLib 和 FAISS-HNSW 分别评估 16 个模型的平均 NDCG@10。Table 4 使用 `efSearch = 10`，并按 exact KNN 的 NDCG@10 降序排列（Table 4）。

Table 4 的结果显示，模型排名可上下移动最多 3 位。例如：

- all-MiniLM-L6-v2 在 HNSWLib 下上升 2 位，在 FAISS 下上升 3 位。
- bge-micro 在 HNSWLib 和 FAISS 下都上升 3 位。
- e5-small-v2 在 FAISS 下下降 3 位。

论文进一步报告，recall@10 与 NDCG@10 的 Pearson correlation coefficient 为 0.71，说明 HNSW recall 差异会传导到下游检索任务指标（Sec. 5.2.3）。Table 7 还显示，在 HNSWLib 中仅改变插入顺序，也会改变 NDCG 排名；不同模型对插入顺序的鲁棒性不同（Table 7）。

这一结果的含义是：只用 exact KNN 做 embedding 模型评测，不能完全代表模型在 HNSW 近似检索系统中的线上表现。

---

## 类别插入顺序实验显示真实数据中也存在类似现象

论文承认按 LID 排序插入是人为构造场景，不一定等同真实业务数据流。为了验证现实相关性，论文用电商类别顺序模拟实际系统中“某一类商品集中导入”或“新类别分批上线”的非随机插入模式（Sec. 5.3）。

类别级实验的前提是：同一类别商品的 embedding 往往比跨类别商品更接近，因此类别顺序可能与局部几何结构相关。论文用 PCA 方法计算类别和全数据集的内在维度，发现类别内在维度通常低于全数据集（Table 8, Table 9）。

Fashion 数据集的例子如下：

| 类别 | ViT-B-32 内在维度 | ViT-L-14 内在维度 |
|---|---:|---:|
| Watches | 312 | 336 |
| Streetwear | 434 | 463 |
| Collectibles | 440 | 463 |
| Sneakers | 389 | 427 |
| Handbags | 418 | 450 |
| All Data | 442 | 469 |

Homewares 数据集中，单类 ViT-B-32 内在维度范围为 313 到 426，全数据集为 439；单类 ViT-L-14 内在维度范围为 369 到 456，全数据集为 475（Table 9）。

Fashion 数据集在 `efSearch = 10` 下，ViT-B-32 的 recall 从 0.435639 到 0.512328，差异约 7.7 个百分点；ViT-L-14 的 recall 从 0.405639 到 0.466295，差异约 6.1 个百分点（Table 10）。在 `efSearch = 40` 下，差异仍存在：ViT-B-32 从 0.744918 到 0.799016，ViT-L-14 从 0.712656 到 0.777410（Table 11）。

Homewares 数据集中差异较小。`efSearch = 10` 下，ViT-B-32 从 0.668467 到 0.690070，ViT-L-14 从 0.672265 到 0.695470；`efSearch = 40` 下，ViT-B-32 从 0.912510 到 0.919790，ViT-L-14 从 0.918570 到 0.925330（Table 12, Table 13）。论文指出 Homewares 类别更多，穷举所有类别排列不可行，因此该实验只覆盖部分顺序（Sec. 5.3）。

---

## 机制解释

论文的机制解释集中在 HNSW 构图对早期节点和局部结构的依赖。

HNSW 是增量构图算法。新节点插入时，从已有入口点开始搜索候选邻居，再建立连接。早期插入的数据会影响上层入口点、局部图结构和后续节点接入路径。若早期数据主要来自低 LID 紧密簇，图可能先形成局部性很强的连接；后续高 LID 节点接入时，需要跨越这些局部结构，可能降低全局导航质量。若先插入高 LID 数据，图更早覆盖复杂区域，低 LID 簇后续接入时对全局结构破坏较小（Sec. 5.2.1）。

论文没有给出形式化证明，也没有提出新的 HNSW 构图算法。其证据来自合成数据趋势、LID 排序实验、路径长度相关性、NDCG 排名变化和电商类别顺序实验。

---

## 与既有 HNSW 认识的关系

HNSW 原论文强调多层图结构、邻居选择和 `ef` 参数对搜索质量的影响。这篇论文补充了一个工程层面的变量：即使 `M`、`efConstruction` 和 `efSearch` 固定，数据插入顺序仍会影响图质量。

论文还呼应 HNSWLib 源码注释中关于 `M` 与数据内在维度的经验判断：高内在维度数据可能需要更大的 `M`，如 48 到 64，才能获得较好 recall。但该论文没有通过调大 `M` 系统消除问题，而是固定 `M = 16`，观察工业默认参数附近的数据敏感性（Sec. 2, Sec. 3.1）。

---

## 边界

论文结论有明确边界：

1. 实验集中在 FAISS 和 HNSWLib 的单图 HNSW 实现，不覆盖带 sharding、segmentation、replica、增量压缩、冷热分层或在线重构策略的生产级向量数据库（Sec. 3）。
2. 论文固定参数观察数据影响，不提供每个数据集的最优 HNSW 参数，也不证明调大 `M`、`efConstruction` 或 `efSearch` 后仍保持相同差异（Sec. 3.1）。
3. LID 排序是构造性实验，可用于揭示敏感性，但真实业务中通常不会按全量 LID 排序批量建库（Sec. 5.3）。
4. 电商图像数据是私有数据，论文只给出类别、模型和结果，不提供可复现实验所需的完整原始商品数据（Sec. 4.3, Sec. 5.3）。
5. 论文只把 future work（未来工作）扩展到 DiskANN、IVFPQ、ANNOY、MRPT、KD-Trees 等其他 ANN 方法，没有报告这些方法上的实测结果（Sec. 7）。

---

## 对当前 Chiplet 研究的意义

该论文不研究 chiplet CPU、CCD、L3 cache、NUMA 或内存层次。它不能直接作为 chiplet-aware HNSW 优化收益证据。

可迁移的结论是：HNSW 查询性能和 recall 不只受查询阶段参数影响，也受构图阶段形成的图拓扑影响。若后续研究 HNSW 在 chiplet CPU 上的分片、复制或布局优化，需要同时记录图构建顺序、数据类别顺序、模型类型、LID 分布和 recall 变化。只测查询延迟而不控制构图条件，可能把图质量差异误判为硬件布局差异。

对 HNSW-on-chiplet 的启发包括：

- 图分区或 per-CCD replica（每 CCD 副本）方案需要区分“访问局部性收益”和“召回变化”。任何改变图构建、节点排列或入口点选择的优化都可能改变 recall。
- 如果按类别、时间或业务批次增量建库，插入顺序可能与 LID 或局部簇结构相关，后续性能评估应同时报告 recall@k。
- 若实验使用 exact KNN 排行榜选择 embedding 模型，不能直接假定该模型在 HNSW 近似检索下仍保持相同排序。

---

## 结论

论文证明，在常见默认参数附近，HNSW 的 recall 会受到数据内在维度和插入顺序影响。合成数据显示内在维度升高会降低 recall；标准检索数据集显示 descending LID 插入通常优于 ascending LID 插入；真实电商图像数据表明按类别形成的非随机插入顺序也会造成 recall 差异。该论文的主要价值不是提出新算法，而是提醒 HNSW 评测必须同时记录数据几何性质、插入顺序、近似检索参数和下游指标。
