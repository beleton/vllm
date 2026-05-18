# HM-ANN 解读

**一句话概述**：HM-ANN 面向 heterogeneous memory（异构内存）中的十亿级近似最近邻检索，目标是在单机 DRAM + Optane PMM 上不压缩向量也能获得高 recall 和低查询延迟。论文把 HNSW 式分层图改造为适配快慢内存的结构：热的上层导航图放入 fast memory，包含全量点的第 0 层放入 slow memory；构建时用 top-down insertion 和 bottom-up promotion 选择高阶 hub 节点，上层搜索产生多个入口点后在第 0 层并行搜索。实验显示 HM-ANN 在 BIGANN/DEEP1B 上相同 accuracy 下比 HNSW 快 2x、比 NSG 快 5.8x，但该结论建立在显式可管理快慢内存上，不能直接外推到 CCD/L3 cache。

---

## 背景：异构内存与大规模 ANN

### ANNS 与图索引

`ANNS`（Approximate Nearest Neighbor Search，近似最近邻搜索）通过允许近似结果降低查询开销。图 ANN 方法如 HNSW 和 NSG 通常在 latency-accuracy tradeoff 上优于 tree、LSH 和 inverted index 等方法，但图结构需要保存大量邻接边，内存占用高，在十亿级数据上容易超过单机 DRAM 容量（Sec. 1, Sec. 2.1）。

压缩方法如 product quantization（PQ，乘积量化）可降低内存占用，但使用压缩向量计算近似距离，高 recall 目标下准确性会下降。论文以 L&C 和 IMI+OPQ 作为代表压缩 baseline（Sec. 1, Sec. 4.1）。

### Heterogeneous memory

`Heterogeneous memory`（异构内存）指单机同时包含 fast but small memory 和 slow but large memory。论文实验平台使用 96GB DDR4 作为 fast memory，1.5TB Intel Optane DC PMM 作为 slow memory。PMM 单机容量可达 TB 级，但论文给出的特征是：随机访问延迟约为 DRAM 的 3x，带宽约为 DRAM 的 1/6；相对 SSD，PMM 随机访问仍约快 80x（Sec. 1, Sec. 2.2, Sec. 4.1）。如果把访问频繁的数据放在 PMM，会显著拖慢查询。

PMM 支持 Memory Mode 和 App-direct Mode。Memory Mode 由硬件把 DRAM 当作 PMM cache；App-direct Mode 允许程序显式控制 DRAM/PMM 放置。HM-ANN 使用 App-direct Mode，以便把上层导航图放入 DRAM、底层大图放入 PMM（Sec. 2.2）。

HM-ANN 的核心问题是：如何在不压缩原始向量的前提下，把图 ANN 索引和全精度向量放入 HM，并避免 slow memory 随机访问成为主瓶颈。

## 研究问题

论文聚焦三个问题：

1. 如何在单机 HM 上支持 billion-point ANNS，不依赖向量压缩（Abstract, Sec. 1）。
2. 如何将图 ANN 的热导航结构放入 fast memory，将大而冷的数据放入 slow memory（Sec. 3）。
3. 如何选择搜索参数，使查询满足指定 latency 和 recall 约束（Sec. 3.3）。

## 核心内容

1. **HM-aware 图构建**：将 HNSW 构建拆成 top-down insertion 和 bottom-up promotion（Sec. 3.1, Algorithm 1）。
2. **hub promotion**：根据第 0 层节点度数选择高阶 hub 节点提升到上层，而不是完全随机提升（Sec. 3.1, Fig. 3）。
3. **分层内存放置**：上层图放 fast memory，最大的第 0 层图和全量向量放 slow memory（Sec. 3.2）。
4. **L1 多入口 + L0 并行搜索**：在 L1 搜索多个 entry points，再分配给多个线程在 L0 做 multi-start 1-greedy search（Sec. 3.2, Algorithm 2）。
5. **性能模型选参**：用模型选择 `efSearch_L1` 和 `efSearch_L0`，在响应时间和 recall 约束下筛选配置（Sec. 3.3, Fig. 7-8）。

## 方法与系统设计

### 图构建：top-down insertion

HM-ANN 先执行类似 HNSW 的 top-down insertion。新节点从当前高层入口开始逐层搜索并插入，在底层形成包含全部节点的 L0 图。与 HNSW 不同，HM-ANN 的目标不是只构建随机层级图，而是为后续 HM 放置和 hub promotion 提供可导航底层图（Sec. 3.1, Algorithm 1）。

### 图构建：bottom-up promotion

bottom-up promotion 统计 L0 中每个节点的 degree（度数），将高度数节点视为 hub。论文按度数从高到低选择节点提升到上层；上层节点数由 fast memory 容量决定，L1 最大度数设置为 `2*M`，提升率为 `1/M`（Sec. 3.1, Algorithm 1）。

该设计基于图搜索直觉：高度数节点通常连接更多区域，可作为更好的导航点。Fig. 3 显示，高度数 promotion 比随机 promotion 更快达到目标 recall（Fig. 3）。

### 查询流程

查询算法见 Algorithm 2：

1. 从顶层开始执行 1-greedy search，逐层接近查询。
2. 到 L1 后，使用 `efSearch_L1` 做 beam search，得到多个 entry points。
3. 将这些入口点分配给多个线程。
4. 多个线程在 L0 执行 multi-start 1-greedy search。
5. 合并各线程结果，返回近邻。

该流程把搜索工作拆为两部分：L1 在 fast memory 中寻找高质量入口点；L0 在 slow memory 中做并行局部搜索，尽量减少 slow memory 的访问步数（Sec. 3.2, Algorithm 2）。

### 异构内存放置

HM-ANN 的放置原则是：小而热的上层图放 fast memory，大而冷的 L0 图放 slow memory。第 0 层包含全量节点和主要邻接信息，占用最大；上层只包含 promoted hub，规模小，更适合放入 fast memory（Sec. 3.2）。

论文还保留约 2GB migration space 作为软件管理缓存。在 L1 搜索期间，系统异步将可能访问的 L0 节点及其连接从 slow memory 迁移到 fast memory，降低后续 L0 访问代价（Sec. 3.2）。

论文给出内存消耗公式：fast memory 主要保存第 1 层及以上连接和第 1 层元素，slow memory 保存第 0 层连接以及只出现在第 0 层的元素。对应公式为 `fast_memory_size = sum_{i=1..l}(Ni * Mi) * byte_per_link + N1 * byte_per_element`，`slow_memory_size = (N0 * M0) * byte_per_link + (N0 - N1) * byte_per_element`（Sec. 3.4）。

### 性能模型

论文将查询时间建模为：

```
T = T_L1 + T_L0
```

其中 L1 搜索在 fast memory 中完成，L0 搜索涉及 slow memory。模型用于筛选 `efSearch_L1` 和 `efSearch_L0`，在满足 latency 与 recall 约束的配置中选择更优参数。Fig. 7-8 显示，该模型可过滤大量不满足约束的配置，并选择更优搜索参数（Sec. 3.3, Fig. 7-8）。

## 实验设置

### 硬件

实验平台为 Intel Xeon Gold 6252，配置 96GB DDR4 和 1.5TB Intel Optane DC PMM。DDR4 作为 fast memory，PMM 作为 slow memory。PMM 使用 App-direct mode，使软件可显式管理数据放置（Sec. 4.1）。

### 数据集

十亿级数据集：

- BIGANN：10 亿个 128 维 SIFT descriptor，10,000 queries。
- DEEP1B：10 亿个 96 维自然图像特征，10,000 queries。

每个十亿级数据集使用 10,000 queries（Sec. 4.1）。

百万级数据集：

- SIFT1M
- DEEP1M
- GIST1M：1M 个 960 维图像 descriptor

这些数据用于对比 HM-ANN、HNSW 和 NSG 在较小规模上的行为（Sec. 4.1, Fig. 2）。

### Baseline

论文使用的 baseline 包括：

- **IMI+OPQ**：inverted multi-index + optimized product quantization，压缩向量方法。
- **L&C**：Link-and-Code，结合图和量化编码的压缩方案。
- **HNSW on HM**：作者实现的 HM 上 HNSW 强基线。
- **NSG on HM**：作者实现的 HM 上 NSG 强基线。
- **Memory Mode / first-touch NUMA**：用于对比 HM 放置策略（Sec. 4.1-4.2, Fig. 6）。

## 结果与解释

### 十亿级 recall-latency 结果

Fig. 1 显示，在 BIGANN 和 DEEP1B 上，HM-ANN 可在 1ms 内达到超过 95% top-1 recall。达到相同 accuracy 时，HM-ANN 比 HNSW 快 2x、比 NSG 快 5.8x；同等 latency 下，HM-ANN 比 L&C 和 IMI+OPQ recall 高 46%（Abstract, Fig. 1, Sec. 4.2）。

论文将该收益归因于两点：不压缩原始向量，避免高 recall 下压缩距离误差；同时将热导航层放入 fast memory，减少 slow memory 访问（Sec. 3, Sec. 4.2）。

### 百万级数据结果

Fig. 2 显示，在 SIFT1M、DEEP1M、GIST1M 上，HM-ANN 与 HNSW 持平或更好，并优于 NSG。达到 99% recall 时，HM-ANN 平均约 850 次 distance computations/query，HNSW 约 900 次（Fig. 2）。

该结果说明，HM-ANN 的改造没有只在十亿级场景有效，在较小数据上也未明显牺牲查询效率。

### 构建时间与内存

Table 1 显示，BIGANN 上 HM-ANN graph size 为 536GB，indexing time 为 96h，promotion rate 为 0.16，fast memory 使用 96GB，slow memory 使用 462GB；DEEP1B 上 HM-ANN graph size 为 756GB，indexing time 为 117h，promotion rate 为 0.11，fast memory 使用 96GB，slow memory 使用 681GB。整体上，HM-ANN 构建时间比 HNSW 约长 8%，索引大小比 HNSW 大 5%-13%，但 fast memory 占用更低（Table 1）。这说明 HM-ANN 不是无代价优化；其收益来自将内存需求转移到 slow memory，并用上层结构减少慢层访问。

### Hub promotion 效果

Fig. 3 显示，高度数 promotion 比随机 promotion 更有效。达到 95%/99%/99.5% recall 时，高度数 promotion 分别快 1.8x/4.3x/3.9x（Fig. 3）。

这直接支持论文对 hub 节点的判断：第 0 层高度数节点更适合作为上层导航点。

消融实验还测得慢内存和快内存中 10k 次距离计算的平均时间分别为 421ns 和 183ns，用于支撑快慢内存访问代价差异（Sec. 4.3）。

### 组件消融

Fig. 4 显示，只改并行 L0 搜索的收益有限。Fig. 5 显示，BP + PL0 + DP 逐步叠加后，在 99% recall 下相对 HNSW 查询时间降低 1.75x（Fig. 4-5）。

该结果说明，HM-ANN 的收益来自构建、promotion、并行搜索和数据放置组合，而不是单一并行化。

### HM 放置策略对比

Fig. 6 显示，HM-ANN 比 HNSW 的 Memory Mode 和 first-touch NUMA 分别快 2x 和 3.7x（Fig. 6）。论文由此说明，简单依赖系统默认 HM 管理或 first-touch 放置不足以释放 HM 性能，需要索引结构感知的放置策略。

## 分析

- HM-ANN 的核心是让算法结构匹配显式内存层次。HNSW 的上层本来就是导航结构，HM-ANN 将其映射到 fast memory；L0 体积最大，放入 slow memory。
- Hub promotion 改变了 HNSW 原始随机层级假设。它利用图结构统计选择导航节点，目标是减少 slow memory 中的搜索步数。
- 多入口并行搜索将 L1 的候选质量转化为 L0 并行性，但其收益依赖候选入口质量和线程开销。
- 性能模型将 `efSearch_L1/efSearch_L0` 从经验调参转成受 latency/recall 约束的搜索问题，可用于复现实验口径。

## 边界

- HM-ANN 的 fast/slow memory 是可显式管理的内存层，不是 CPU cache。论文机制依赖 DDR4 + Optane PMM 的容量差、延迟差和 App-direct mode（Sec. 2.2, Sec. 4.1）。
- 论文没有评估 CCD/L3、跨 CCD cache-to-cache、NPS、IBS load latency 或 LLC occupancy。
- 论文没有覆盖动态更新、删除、在线重构和多租户干扰。
- PMM 与 DRAM 的层次差异大于同 socket 内 CCD/L3/DRAM 差异；论文中的 speedup 不能直接外推到 chiplet CPU。
- 论文收益与 full-precision vectors 有关，不能简单迁移到已经使用 PQ、SQ 或其他压缩索引的系统。
- Table 1 显示 BIGANN/DEEP1B 构建时间为 96h/117h，构建成本是实际部署边界之一（Table 1）。
- Broader Impact 指出，高度数节点被结构化放入特定内存区域可能暴露关键图信息；Optane 非易失内存也引入安全风险（Broader Impact）。

## 可迁移点

- “热导航结构靠近计算，冷大图放远”的思想可迁移到 per-CCD navigator replica、hot hub replica 或第 0 层 cold graph partition。
- Hub promotion 提示可用访问频率、度数、trace 热度选择需要复制或靠近本地的 HNSW 节点。
- 多入口并行搜索可作为减少单一路径慢访问的启发，但必须固定 recall 并报告候选合并成本。
- 性能模型选参思路可迁移到 chiplet 实验：同时约束 latency、recall 和访问层级，而不是只调 `efSearch`。

## 不可直接迁移点

- CCD 本地 L3 不是可显式分配的大容量 fast memory，不能原样实现 HM-ANN 的 fast memory placement 或 2GB migration space。
- Optane PMM 的随机访问延迟约为 DRAM 的 3x，和当前 AMD EPYC CCD/L3/本地 DRAM 路径不是同一层次问题。
- HM-ANN 没有证明跨 CCD 访问是 HNSW 的主瓶颈，也没有证明 L3 局部化可带来相同量级收益。
- bottom-up promotion 改变了图结构，若用于 chiplet 研究，必须区分“图质量改善”与“硬件局部性改善”。

## 与当前 Chiplet CPU 研究的关系

HM-ANN 适合作为 HNSW 内存层次优化的系统参照。它支持的结论是：HNSW 类图索引可以通过识别热导航结构、冷大图和搜索入口，将数据放置与查询路径结合，从而改善快慢内存系统中的 latency-recall 曲线。

HM-ANN 不能作为 chiplet-aware HNSW 有效的直接证据。当前 AMD EPYC chiplet CPU 上，CCD 本地 L3 不是 PMM/DRAM 这种可显式管理的大容量内存层；远端 CCD 或本地 DRAM 的代价也不同于 PMM。后续应在本地平台测量 `recall@k`、p50/p95/p99 latency、L3 miss/query、IBS load latency、local memory 与 another CCD/source share，并与普通 HNSW、普通 graph reordering 和优化实现对照。

## 证据

- 原始资料：`wiki/原始资料/papers/HNSW/HM-ANN - Efficient Billion-Point Nearest Neighbor Search on Heterogeneous Memory.pdf`
- 关键锚点：Abstract；Sec. 1；Sec. 2.1-2.2；Sec. 3.1-3.4；Algorithm 1-2；Eq. 1；Sec. 4.1-4.3；Fig. 1-8；Table 1；Broader Impact
