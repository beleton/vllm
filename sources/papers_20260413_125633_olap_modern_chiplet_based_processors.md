# 检索记录：OLAP on Modern Chiplet-Based Processors

- 检索时间：2026-04-13 12:57:56 +0800
- 查询：`"OLAP on modern chiplet-based processors"`

## 命中文献

- 标题：`OLAP on Modern Chiplet-Based Processors`
- 作者：Alessandro Fogli、Bo Zhao、Peter R. Pietzuch、Maximilian Bandle、Jana Giceva
- 来源：`Proceedings of the VLDB Endowment 17(11): 3428-3441 (2024)`
- DOI：`10.14778/3681954.3682011`
- Open Access PDF：https://www.vldb.org/pvldb/vol17/p3428-fogli.pdf
- DBLP：https://dblp.org/rec/journals/pvldb/FogliZPBG24
- 作者页镜像 PDF：https://www.doc.ic.ac.uk/~af6618/files/papers/VLDB24.pdf

## 已确认事实

- 论文面向 modern chiplet-based processors 上的 `OLAP`，摘要列出的代表平台为 `AMD EPYC Milan`、`Intel Sapphire Rapids`、`ARM Graviton 3`。
- 论文摘要列出的代表 query engines 为 `Presto`、`SingleStore`、`SparkSQL`。
- 论文提出 `within-chiplet` 与 `across-chiplet` 两类部署思路，以降低 chiplet 架构下的 cache misses 与 remote accesses。
- 论文摘要明确写明：相对 hardware-oblivious deployment，chiplet-aware deployment 在其实验中可带来 `up to 7x` 的性能提升。
- 论文第 6 节明确给出部署边界：当 working set 小于单个 chiplet 的 `L3` 容量时优先 `WICP_Local`；当 working set 超过单个 chiplet `L3` 但仍小于多个 chiplet 聚合 `L3` 容量时优先 `WICP_Mixed`。

## 检索入口

- VLDB open-access 页面：https://www.vldb.org/pvldb/vol17/p3428-fogli.pdf
- DBLP 记录：https://dblp.org/rec/journals/pvldb/FogliZPBG24
