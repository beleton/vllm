# Graph Reordering for Cache-Efficient Near Neighbor Search 解读

**一句话概述**：论文把 HNSW 类图近邻搜索的查询延迟建模为 cache hit maximization（缓存命中最大化）问题，通过 graph reordering（图重排序，即只改变节点在连续内存块中的排列）改善节点访问的空间局部性。论文包含理论分析、profile-guided reordering（基于查询 profile 的重排序）和消融实验；结果显示重排序最高提升平均查询时间 40%，最高提升 P99 latency 20%，Porder 相对 Gorder 还能带来 5%-10% 额外收益（Abstract；Sec. 1.1；Fig. 4-5；Tab. 5）。

---

## 论文基本信息

- 论文：`Graph Reordering for Cache-Efficient Near Neighbor Search`
- 作者：Benjamin Coleman、Santiago Segarra、Alex Smola、Anshumali Shrivastava
- 会议：NeurIPS 2022，36th Conference on Neural Information Processing Systems
- 研究对象：基于 HNSW 的图近邻搜索索引
- 核心问题：图近邻搜索的主查询路径存在随机节点访问和 cache miss，能否只通过节点内存布局重排降低查询延迟
- 原始资料：`wiki/原始资料/papers/HNSW/graph-reordering-for-cache-efficient-near-neighbor-search.pdf`

## 问题

近邻搜索给定数据集 `D = {x1, x2, ..., xN}` 和查询点 `q`，返回距离 `q` 最近的 `k` 个数据点。embedding 检索中，向量维度通常为 100 到 1000，数据规模常超过 1 亿，推荐系统、NLP、视觉检索和语义搜索都依赖高 recall 区间下的低延迟查询（Sec. 1）。

HNSW 的查询不是顺序扫描，而是在剪枝后的近邻图上沿边遍历。论文用 `perf` profile nmslib-HNSW，Fig. 1 将查询时间分为初始化、visited lists、节点遍历、距离计算和向量取数，其中约 40% 查询时间落在距离计算第一维对应的向量取数路径上，作者据此把主要优化目标定位为 memory fetching（Fig. 1；Sec. 1）。

论文强调，常见 graph reordering 在图分析任务上的传统解释是把多个节点装入同一 cache line；该解释不适用于近邻图，因为单个 HNSW 节点包含向量、邻接表等内容，通常跨越多条 cache line。会议版的核心修正是：收益主要来自硬件 prefetcher 对更连续节点布局的利用，以及 visited list、priority queue 等小型辅助结构的局部性改善（Sec. 1.1；Sec. 7）。

## 核心结论

1. Graph reordering 只改变节点存储顺序，不改变图结构、搜索算法和 recall；论文认为它应作为图近邻搜索的标准预处理步骤（Sec. 3；Sec. 7）。
2. Gorder、RCM、Porder 这类 objective-based reordering（基于目标函数的重排序）最有效。平均 query time 在小数据集上典型提升约 10%，在大数据集和高 recall 区间最高提升 40%（Fig. 4；Sec. 7）。
3. P99 latency 没有因布局变化恶化。SIFT100M 上，RCM P99 latency 改善 17%，Gorder 改善 30%；正文总结为高 recall 区间至少改善 20%（Fig. 5；Sec. 6）。
4. Porder 使用 1000 个查询统计边遍历频率，在 SIFT10M 和 Yandex 10M 上相对 Gorder 额外提升 5%-10%；SIFT10M 上 query time 从无重排的 22.8 降到 Gorder 17.9、Porder 16.8，Yandex 10M 上从 16.3 降到 Gorder 12.8、Porder 12.4（Tab. 5；Sec. 6）。
5. Cache miss 降低与延迟收益一致。SIFT100M 上 perf 显示 Gorder 将 L1/L2/L3/TLB miss 分别降到 14.46%、9.6%、4.0%、2.14%，RCM 分别降到 17.37%、7.61%、5.1%、2.56%，原始布局分别为 19.53%、13.9%、6.5%、3.85%（Tab. 1）。
6. 重排序成本低于 HNSW 索引构建成本。Fig. 2 和 Fig. 3 显示，多种重排序算法耗时小于图构建，且随节点数和最大度数 `kc` 增长仍可扩展；正文称多数数据集上重排序时间比 HNSW construction 小一个数量级（Fig. 2-3；Sec. 7）。

## 方法

### Graph Reordering

Graph reordering 构造标号函数 `P: V -> {1, ..., N}`，给图中每个节点分配唯一整数编号。内存布局按 `P(v)` 放置节点，使相连或常共同访问的节点位于相近地址。该方法作用于已构建索引的节点排列，不改变边、距离函数、查询参数和 recall（Sec. 3）。

论文使用 HNSW 作为实验对象，但论证对象是图近邻搜索中的共同访问模式。HNSW、PANNG、ONNG 等索引都包含近邻图、剪枝/多样化、起点选择和沿边搜索；重排序只依赖搜索时的节点访问模式，不依赖特定索引的上层组织（Sec. 2）。

### Gorder

Gorder 最大化相近编号节点之间的邻域重叠。若两个节点直接相连，或共享很多入邻居，则 Gorder 倾向于把它们放到同一个大小为 `w` 的局部窗口中。论文使用 `Ss(u, v)` 表示 `u` 与 `v` 的有向连接数，用 `Sn(u, v)` 表示共享入邻居数，目标函数为窗口内 `Ss + Sn` 之和（Eq. 1；Sec. 3）。

Gorder 的目标函数是 NP-hard，论文沿用已有 greedy 近似算法。会议版将 Gorder 与 cache complexity 的上界联系起来，给出它在特定条件下改善 cache efficiency 的理论依据（Theorem 2-3）。

### RCM

RCM（Reverse Cuthill-McKee，反向 Cuthill-McKee）最初用于降低稀疏对称矩阵 bandwidth。用于图重排序时，它试图降低相连节点之间的最大编号距离。论文将 RCM 归为 objective-driven methods，但也指出 RCM 只考虑最大 label discrepancy，理论上不能像 Gorder 那样直接保证 cache complexity 降低（Sec. 3；Sec. 4）。

### Corder

Corder（cache order）是论文中的消融方法。Theorem 1 给出 DFS cache efficiency 的上下界，其中 Gorder 近似优化上界；Corder 在目标函数中加入 shared-parent triplet 惩罚项，直接优化下界，用于检验 Gorder 上界和真实 cache efficiency 之间的差距（Eq. 2；Sec. 4.1）。

实验中，Corder 在 SIFT10M 上比 Gorder 慢 4%。作者据此认为 Gorder 得到的布局已经接近 cache efficiency 最优，Theorem 1 中的组合项在实际任务中较小（Sec. 6）。

### Porder

Porder（profile order）先用少量查询统计边 `e` 在搜索中的遍历次数 `Te`，再将边权设为 `1 + Te`，把 Gorder 目标函数改为 weighted graph order 问题。论文默认用 test set 前 1000 个查询生成 weighted graph，再在 10000 个查询上报告结果（Eq. 3；Sec. 4.1；Sec. 5）。

Porder 的含义是：搜索中更常经过的边和邻域更应被放近。它不是完全 query-agnostic 的静态图布局，而是根据近期查询分布调整内存布局。论文在讨论中指出，生产数据库本来会根据访问频率和 workload 统计周期性重排内容，Porder 可用于 query distribution 变化后的动态布局更新（Sec. 7）。

### 轻量级方法

论文还评估 degree sorting、hub sorting、hub clustering 和 degree-based grouping。这些方法只根据节点 degree 或 hub 标签重排，成本低，但在近邻图上缺乏稳定收益。作者在理论部分给出反例：近似 regular graph 中只有一个节点度数不同，degree-based sorting 可能把该节点从邻居旁移开，反而增加访问成本（Sec. 3；Sec. 4）。

论文没有采用 graph partitioning。原因是已有研究显示这类方法生成的顺序过粗，难以在 cache 访问粒度上加速（Sec. 5）。

## 理论分析

论文用 idealized cache model（理想化缓存模型）分析图布局与 cache performance 的关系。模型假设一条 cache line 包含 `B` 个对象、cache 有 `T` 条 line，并采用能最小化总 miss 的最优替换策略；论文在 tall cache 假设 `T > B^2` 下分析一次 BFS/DFS 风格节点扩展的 cache miss 数（Sec. 4）。

论文将访问节点 `vi` 及其出邻居的成本写为 `Ci`，将命中数写为 `CEi`，并定义平均 DFS cache efficiency `CEDFS(P)`。目标是寻找布局 `P* = arg max CEDFS(P)`（Sec. 4）。

Theorem 1 给出 `CEDFS(P)` 的上下界。上界包含相邻窗口内的直接边和共享入邻居项，即与 Gorder 目标函数相近；下界再减去同一 cache line 内三元组组合项 `Triplets(vi)`（Theorem 1）。

Lemma 1 证明寻找最优 DFS cache cost 的布局是 maximum weight perfect hypergraph matching 的 NP-hard 实例。该结论解释了为什么论文使用 Gorder、RCM、Corder、Porder 等近似方法，而不是求精确最优布局（Lemma 1）。

Theorem 2 将 Gorder objective 与 `CEDFS(P)` 连接起来。若图最大出度为 `M`，对 Gorder 布局加上 cache boundary alignment 后，`CEDFS(P')` 可由 `FGO(P)`、`B`、`N`、`M` 的组合项给出上下界（Theorem 2）。

Theorem 3 给出 Gorder 改善 cache efficiency 的充分条件。若新布局 `P2` 的 Gorder objective 相比 `P1` 提升足够大，则 `CEDFS(P2) >= CEDFS(P1)`。论文用 `B = 2` 计算 SIFT1M、F-MNIST、NYTimes 的初始和最终 Gorder 分数，发现提升足以满足实际任务上的 cache benefit 条件：SIFT1M 从 6.6k 到 2.6M，F-MNIST 从 64 到 102k，NYTimes 从 417 到 736k（Tab. 2；Theorem 3）。

理论边界：

- 该理论分析 DFS/BFS 式一步扩展，不直接分析 HNSW 实际 beam search。作者在 checklist 中说明，这是因为 beam search 难以直接分析（Checklist）。
- 模型假设单条 line 可容纳 `B` 个对象，而真实 HNSW 节点跨多条 cache line。作者用 next-line prefetcher 和 visited list 等小对象解释该抽象仍有意义（Sec. 4；Sec. 7）。
- 轻量级 degree 方法在没有 power-law 假设时不能保证降低 cache complexity；近邻图通常不满足图分析任务中常见的幂律度分布前提（Sec. 4）。

## 实验设置

### 平台

实验服务器配置为 252 GB RAM、28 个 Intel Xeon E5-2697 CPU、共享 36 MB L3 cache。查询 benchmark 采用 warm start，即 index、query、data 预先加载到 RAM；每次执行 10000 个查询，报告平均 query time。为降低干扰，查询程序限制为单核运行，并保证服务器上没有其他程序运行（Sec. 5）。

### 实现

论文扩展 nmslib 的 HNSW 实现以支持 graph reordering，并使用 Naidan 等人手册中的 flat layout（扁平布局）避免 memory fragmentation。作者验证修改版实现与 nmslib-HNSW 使用相同 memory layout、相同距离计算次数，并产生相同图结构（Sec. 5）。

该 baseline 已经是连续预分配内存块，不是指针碎片布局。重排序收益来自图结构感知的节点顺序，而不是从碎片化实现切换到连续内存实现（Sec. 2；Sec. 5）。

### 数据集

论文使用 ANN-benchmarks 数据集，以及 SIFT1B 和 DEEP1B 的 10M/100M 子集；所有实验查询 top 100 neighbors，并报告 `R100@100`，即返回 top 100 结果中覆盖 top 100 ground truth neighbors 的比例（Sec. 5）。

| 数据集 | 规模 `N` | 维度 `d` | 每向量存储 |
|---|---:|---:|---:|
| GIST | 1M | 960 | 3.8 kB |
| SIFT | 10M-100M | 128 | 128 B |
| DEEP | 10M-100M | 96 | 384 B |
| MNIST | 60k | 784 | 3.1 kB |

SIFT 和 DEEP 单向量更小，cache 中可容纳更大的子图；MNIST 和 GIST 单向量更大，向量数据读取占比更高。论文在讨论中用该差异解释重排序在 SIFT/DEEP 上收益更大（Tab. 4；Sec. 7）。

### HNSW 参数

构建阶段设置 `Mc = 100`，并构建 `kc ∈ {4, 8, 16, 32, 64, 96}` 的多个索引。论文选择 recall > 0.95 区间中 recall-latency tradeoff 最好的索引用于重排序实验。查询阶段改变 beam search buffer size `Mq`，范围为 100 到 5000，以得到不同 recall-latency 点（Sec. 5）。

Gorder 使用窗口大小 `w = 5`。Porder 用前 1000 个测试查询构造 weighted graph，再对同一批排序方法在 10000 个查询上报告结果（Sec. 5）。

### 指标

- Query latency：10000 个查询的 wall-clock 总时间换算平均 query time；speedup 定义为无重排序平均 latency / 有重排序平均 latency，平均值来自 10000 queries 和 5 runs（Sec. 5-6）。
- P99 latency：SIFT100M 上统计 99th percentile latency，验证布局变化不会恶化尾延迟（Fig. 5；Sec. 6）。
- Cache miss：用 Linux `perf` 统计 L1、L2、L3、TLB miss rate，miss rate = data cache misses / data references（Tab. 1；Sec. 5）。
- Cachegrind：只标注节点遍历和距离计算代码行，验证 miss 降低确实来自核心查询路径（Tab. 3；Sec. 5）。

## 结果与解释

### Query Time

Fig. 4 展示不同数据集和 recall 区间下的 query time vs recall tradeoff。黑线代表原始布局，位于黑线之上的算法更快，位于黑线之下表示变慢。Gorder 和 RCM 在多个数据集上稳定正收益；轻量级 degree/hub 方法在不同数据集上表现不稳定，部分点可能退化（Fig. 4）。

论文总结：objective-based reordering 在小数据集 `N < 1M` 上典型 speedup 约 10%，在大数据集和高 recall 区间最高 speedup 40%。该结论覆盖 Gorder、Porder、RCM 等方法，不等同于所有轻量级重排方法均有效（Fig. 4；Sec. 7）。

### P99 Latency

SIFT100M 上，Fig. 5 报告 P99 latency。RCM 相对原始布局降低 17%，Gorder 降低 30%；正文总结为高 recall 区间 Gorder 和 RCM 至少改善 20%。该实验排除了“平均延迟下降但尾延迟上升”的主要风险（Fig. 5；Sec. 6）。

### Porder

Tab. 5 比较无重排、Gorder 和 Porder：

| 算法 | SIFT10M query time | SIFT10M speedup | Yandex 10M query time | Yandex 10M speedup |
|---|---:|---:|---:|---:|
| None | 22.8 | 0% | 16.3 | 0% |
| Gorder | 17.9 | 27% | 12.8 | 22% |
| Porder | 16.8 | 36% | 12.4 | 24% |

Porder 说明 query distribution 信息可以进一步优化布局。该结果依赖前 1000 个查询的 profile；若线上查询分布变化，Porder 布局需要重新 profile 和重排（Tab. 5；Sec. 6-7）。

### Cache Miss

Tab. 1 的 perf 结果显示 Gorder 和 RCM 同时降低多级 cache miss 与 TLB miss：

| 算法 | L1 miss | L2 miss | L3 miss | TLB miss |
|---|---:|---:|---:|---:|
| Original | 19.53% | 13.9% | 6.5% | 3.85% |
| RCM | 17.37% | 7.61% | 5.1% | 2.56% |
| Gorder | 14.46% | 9.6% | 4.0% | 2.14% |

Gorder 在 L1、L3、TLB miss 上最低；RCM 在 L2 miss 上最低。论文用该结果支持“延迟收益来自 cache miss 降低”的解释（Tab. 1；Sec. 6）。

Tab. 3 的 cachegrind 结果只统计节点遍历和距离计算代码行：

| 算法 | L1 miss | L3 miss |
|---|---:|---:|
| Original | 22.76% | 13.56% |
| Gorder | 19.28% | 8.32% |
| RCM | 20.61% | 8.91% |
| DegSort (in) | 22.76% | 13.55% |
| DegSort (out) | 22.76% | 13.56% |
| HubSort | 22.85% | 13.63% |
| HubCluster | 22.77% | 13.51% |
| DBG | 22.81% | 13.34% |

Gorder 和 RCM 的 L3 miss 明显低于原始布局和轻量级方法。论文指出，cache 与 RAM 访问延迟约有 100 倍差异，小幅 miss rate 降低也可能产生明显速度收益；cachegrind 的 miss 排序与 Fig. 4 中 speedup 排序一致（Tab. 3；Sec. 5-6）。

### 重排序成本

Fig. 2 和 Fig. 3 报告重排序时间低于 HNSW graph construction。Fig. 3 分别展示不同数据集、SIFT 在 `k = 16` 下随节点数增长、GIST1M 在 `k <= 120` 下随最大度数增长的重排序成本。论文结论是重排序随节点数和最大度数扩展良好，多数数据集上重排序比 HNSW construction 小一个数量级（Fig. 2-3；Sec. 7）。

该结果的系统含义是：近邻搜索索引通常构建昂贵、生命周期内被查询数百万次，离线重排序成本可由在线查询收益摊销。它不同于 PageRank 等只处理图少数几次的 workload，后者更依赖低成本轻量级重排（Sec. 7）。

### Prefetcher 与 Node Size 消融

论文包含两个消融实验：

1. 关闭 L1/L2 hardware prefetcher 后，重排序 speedup 明显变小：MNIST 减少 10%，DEEP10M 减少 5%，SIFT10M 减少 5%。实验仍保留 TLB caching 和 L3 prefetch（Sec. 6）。
2. SIFT1M 使用同一图，比较 32-bit float 向量和 8-bit integer 向量。512 B/vector 的大向量平均 speedup 为 11%，128 B/vector 的小向量平均 speedup 为 19%。该结果支持“向量越小，cache 中可容纳的子图越大，重排序收益越强”的解释（Sec. 6）。

作者解释：数据向量大于单条 cache line，因此 Table 1 中的收益很可能来自 prefetcher 对空间局部性的利用；visited list 和 priority queue 等辅助结构用少量字节表示节点，可直接受益于同一 cache line 内的并发加载和预取（Sec. 7）。

## 边界

1. 实验平台是 Intel Xeon E5-2697、共享 36 MB L3 cache；论文没有评估 AMD Zen CCD、CCX、NPS、跨 CCD latency、L3 slice 或 IBS 来源级延迟（Sec. 5）。
2. 查询程序限制为单核运行；论文没有评估多线程 batch query、跨核共享 L3、per-CCD worker、线程迁移、多租户服务或吞吐饱和场景（Sec. 5）。
3. 理论分析使用理想化 cache model，并分析 DFS/BFS 式节点扩展，不是对 HNSW beam search 的完整理论建模（Sec. 4；Checklist）。
4. 论文实现基于 nmslib-HNSW flat layout；hnswlib、Faiss、DiskANN 等系统的节点块布局、visited list 实现、prefetch 指令、SIMD 距离计算和内存分配策略可能改变收益幅度（Sec. 5）。
5. Graph reordering 不改变访问节点数、距离计算次数、图结构和 recall；它不能替代 HNSW 参数调优，也不能证明图遍历路径本身减少（Sec. 3；Sec. 5）。
6. Porder 依赖查询分布 profile。论文使用前 1000 个测试查询构造权重，未评估长期分布漂移、在线更新成本和错误 profile 对尾延迟的影响（Sec. 5-7）。
7. 论文没有证明 miss 来自跨 CCD 或跨 socket 访问。它只证明布局改善可降低缓存层次 miss 和 TLB miss（Tab. 1；Tab. 3）。

## 与当前 Chiplet CPU 研究的关系

### 可迁移点

- HNSW 类图近邻搜索对节点内存布局和 cache miss 敏感。普通 graph reordering 可在不改变 recall 的情况下带来显著 query latency 收益，是后续 chiplet-aware HNSW 研究必须纳入的强 baseline（Fig. 4-5；Tab. 1；Tab. 3）。
- 当前实验若只相对原始插入顺序提升，不能证明 CCD/L3 拓扑感知策略有独立价值；至少应对比 Gorder、RCM、Porder 或等价的普通 cache-friendly layout。
- 论文提供了可复现实验指标模板：mean latency、P99 latency、recall、L1/L2/L3 miss、TLB miss、核心查询代码段 cache miss。Zen 平台应补充 IBS load latency、L3 miss source、remote/local source、CCD 级 LLC occupancy 和跨 CCD 访问比例。
- Porder 对当前 query grouping 方向有直接启发。若相似 query 或近期 query 分布会重复经过相同边和节点，可用 profile 统计边/节点热度，再分析这些热路径在 CCD/L3 内是否可复用。
- Prefetcher 与 node size 消融说明，压缩向量、float16、PQ/SQ 或低维 embedding 可能放大布局收益。当前 HNSW 实验应覆盖不同向量大小，而不是只测单一维度/类型。

### 不可直接迁移点

- 论文平台是共享 L3 的单核实验，不等同于 AMD EPYC 每 CCD 约 32 MiB 本地 L3 的分片结构。共享 L3 上的 Gorder/RCM/Porder 收益不能直接外推为 per-CCD layout 收益。
- 论文没有多 query 并行调度实验。当前若研究同一 CCD 内相似 query 并行处理，需要独立测多线程 batch 下的 cache line 复用、tail latency 和吞吐。
- 论文没有 per-CCD replica、CCD owner、query routing、remote expansion 或 chiplet-aware graph partitioning。使用这些机制时，需要证明收益来自 chiplet 拓扑，而不是普通 cache-friendly reorder。
- 论文没有评估跨 NUMA/socket 内存放置。当前平台若用 `numactl`、resctrl CAT、CCD 绑定或 NPS 配置，必须单独报告内存归属和硬件计数器来源。

## 后续实验设计启发

1. Baseline 至少包含原始 hnswlib/HNSW 布局、随机重排负对照、Gorder/RCM、Porder 或 trace-guided 普通重排序。
2. 指标至少包含 recall、mean latency、P50/P95/P99、访问节点数、距离计算次数、L1/L2/L3 miss、DTLB miss、IBS load latency、L3 miss source、CCD 级 LLC occupancy。
3. 实验应区分单 query 单核、同一 CCD 多 query、多 CCD 并发和跨 socket 场景；论文结论只能直接支撑单核 warm-start 查询路径。
4. 若做 chiplet-aware layout，应报告相对普通 Gorder/RCM/Porder 的增量，并给出跨 CCD 访问减少或本地 L3 residency 增加的硬件证据。
5. 若做热节点副本，应区分上层导航节点、第 0 层热点节点、邻接表、向量数据、visited list/priority queue 等不同对象。论文支持“节点访问局部性重要”，不支持任意粒度复制都有效。

## 证据锚点

- Abstract：HNSW cache miss、cache hit maximization、graph reordering、最高 40% query time 改善。
- Fig. 1：HNSW query subroutine profile，约 40% 查询时间用于从内存获取向量。
- Sec. 1.1：graph reordering 在大节点场景下依赖 prefetcher；新增理论分析、Porder、实验贡献。
- Sec. 3：graph reordering 定义、Gorder 目标函数、评估的重排序方法。
- Sec. 4：idealized cache model、`CEDFS(P)`、Theorem 1-3、Lemma 1。
- Sec. 4.1：Corder 和 Porder 定义，Porder 使用边遍历频率构造 weighted graph。
- Tab. 1：SIFT100M 上 perf 的 L1/L2/L3/TLB miss。
- Tab. 2：`B = 2` 下 Gorder 初始/最终分数。
- Tab. 3：SIFT100M 上 cachegrind 的核心查询代码 miss rate。
- Fig. 4：query time vs recall tradeoff，展示不同重排序方法的 speedup/slowdown。
- Tab. 4：GIST、SIFT、DEEP、MNIST 的规模、维度和每向量存储大小。
- Tab. 5：SIFT10M 和 Yandex 10M 上 Gorder/Porder query time 与 speedup。
- Fig. 5：SIFT100M 上 P99 latency，RCM 改善 17%，Gorder 改善 30%。
- Sec. 6：Porder、Corder、reordering cost、cache miss、prefetcher 和 node size 消融。
- Sec. 7：重排序成本摊销、收益机制、软件 prefetch 边界、与压缩/降维/生产系统动态重排的关系。
