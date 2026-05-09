# Chiplet 三篇论文对照与 Attention 结论验证

## 依据版本
- `OLAP on Modern Chiplet-Based Processors`, PVLDB 2024
- `Optimizing Sorting for Chiplet-Based CPUs`, ADMS 2024
- `CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System`, EuroSys 2026

## 问题
三篇 chiplet 架构强相关论文各自优化了什么，选择的应用具有怎样的特征使得优化有效，以及这些特征是否与当前假设「适合 chiplet 优化的场景」一致：
- 场景 1：应用对 L3 大小敏感，此时优化数据局部性、减少冗余可以带来提升
- 场景 2：应用的各线程/进程需要经常跨核心访问，此时可优化进程/线程的放置或计算过程来减少跨 CCD 的访问

## 各论文核心内容

### OLAP on Modern Chiplet-Based Processors
- **优化方法**：提出 WICP (Worker per Chiplet) 部署策略，将查询引擎 worker 以 chiplet 粒度绑定。根据 working set 大小与 L3 容量关系选择 WICP_Local（只用单 chiplet 核心）或 WICP_Mixed（每 chiplet 一个核心）。在 Presto、SingleStore、SparkSQL 上评估 (Fig.7, Fig.8)。
- **应用特征**：OLAP 查询涉及大量表扫描（TableScan）和重分区（Repartition），这些操作的 working set 大小决定了 L3 cache 是否够用。论文观察到 SparkSQL 在 WIM 部署下数据分布不均导致互联拥塞 34% 时间 (Fig.8, §4.2)。WICP 通过强制数据局部性，使 Q13 的 Orders 表扫描从 5.36s 降至 0.4s (§4.2)。
- **最高 speedup**：SparkSQL on Intel Sapphire Rapids 达 6.97x (Fig.7)。

### Optimizing Sorting for Chiplet-Based CPUs
- **优化方法**：四项策略：(1) chiplet 粒度分区输入数据；(2) 将 `in-cache` 阶段拆为 `in-local-chiplet-cache` 和 `in-remote-chiplet-cache`；(3) 根据数据大小相对 L3 容量调度任务；(4) 避免 data shuffling，用 chiplet-aware 线程绑定替代 (§4)。实现了 chiplet-aware LSB Radix-Sort 和 Comparison-Sort。
- **应用特征**：排序是 memory-intensive 操作，核心行为是反复扫描和重新组织数据。排序的 `in-cache` 阶段直接依赖 L3 容量——数据在 L3 内排序 vs 溢出到主存，性能差异巨大。Data shuffling 占排序总时间 20%-32% (Tab.1)，在 chiplet 架构上代价极高。论文说明 L3 cache 的 partitioned 特性改变了排序算法设计的根本假设 (§3.1)。
- **最高 speedup**：Comparison-Sort 达 4.5x，Radix-Sort 达 2x，均对比 NUMA-aware 方案 (Fig.1)。

### CHARM
- **优化方法**：运行时系统，包含 chiplet-aware 任务调度、自适应 cache 分区、coroutine 轻量级并发、PMU 性能监控。去中心化决策：每个 worker 独立监控 remote cache access rate（`ANY_DATA_CACHE_FILLS_FROM_SYSTEM`），超过阈值则 spread 到更多 chiplet（追求更大 cache），否则 consolidate（追求局部性）(Algorithm 1, Algorithm 2)。
- **应用特征**：覆盖图计算（BFS/PageRank/SSSP 等，不规则内存访问）、统计分析（SGD，内存带宽密集）、OLAP（TPC-H on DuckDB）、OLTP（YCSB/TPC-C on ERMIA）(§5.1)。论文说明没有一种固定放置策略适合所有应用——需根据 remote cache access rate 动态调整 (§4.2)。
- **最高 speedup**：SGD 达 3.85x，Graph500 达 2.81x，OLAP (TPC-H) 达 2.34x (Fig.1)。图计算上 CHARM 将 remote chiplet access 降低了 1-3 个数量级 (Tab.1)。

## 三篇论文的共性规律

### 共性 1：L3 局部性 vs 聚合容量的权衡是核心决策维度

三篇论文不约而同地将优化问题建模为同一个权衡：按 chiplet 收紧任务以获得更低访问延迟（Chiplet_Local），还是跨 chiplet 分散任务以获得更大聚合缓存（Chiplet_Mixed）。

- OLAP 论文 Fig.10：数据 < 32MB（单 chiplet L3）时 WICP_Local 带宽更高；超过后 WICP_Mixed 更优
- Sorting 论文 Fig.3 / Fig.13：相同的 Chiplet_Local vs Chiplet_Mixed 边界在 32MB 处出现
- CHARM Algorithm 1：`spread_rate` 的增减直接由 remote cache access rate 驱动，本质是同一决策

### 共性 2：避免跨 chiplet 数据移动是共同优化方向

- OLAP 论文：WICP 强制数据局部性，缓解了 SparkSQL 下 34% 时间的互联拥塞
- Sorting 论文：去掉 data shuffling 是 ablation study 中贡献最大的优化项 (Fig.15)
- CHARM 论文：Tab.1 展示 chiplet-aware 调度将 remote chiplet access 降低 1-3 个数量级

### 共性 3：有效的前提是应用对 L3 cache 敏感

三篇论文优化的应用——OLAP 表扫描、排序 in-cache 阶段、图遍历——都有共同的特性：**working set 大小在 L3 容量边界附近**。能否 fit in L3 直接决定是否会频繁溢出到主存，因此优化数据在 L3 中的布局可以本质性地改变性能。

## 与当前结论的对照

### 对照场景 1：「应用对 L3 大小敏感」

**三篇论文高度一致地支持此结论。**

- Sorting 论文最直接：将 `in-cache` 拆为 chiplet-local 和 chiplet-remote 两个阶段，因为排序性能在「数据是否 fit in L3」这个边界上发生阶跃变化 (§3.1, Fig.13)
- OLAP 论文 Fig.10：32MB（单 chiplet L3）是带宽曲线的拐点
- CHARM Algorithm 1：`spread_rate` 调整以 remote cache access rate 为信号，隐含假设是「避免 L3 miss 带来的 main memory access」是性能关键

CPU Attention 不符合此条件的证据：Attention 的 working set 中，head 维度在 L2 内完成复用，sequence 维度是 streaming access 不依赖 L3 复用。限制 L3 可用大小对性能影响极小（见 `wiki/实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md`），说明 Attention 不满足 L3 敏感性前提。

### 对照场景 2：「线程/进程需要经常跨核心访问」

**三篇论文也支持此结论，但有重要限定条件。**

- OLAP 论文 Fig.7/Fig.8：interconnect 拥塞是 WIM 部署下性能降级的主因
- Sorting 论文 Tab.1：shuffling 占排序时间 20-32%，是跨 chiplet 访问的主要来源
- CHARM 论文 Tab.1：图计算中 NUMA-aware 调度产生大量 remote chiplet access

**但限定在于**：减少跨 CCD 访问如果不同时改善 L3 cache 局部性，效果可能有限。在 chiplet 架构上，跨 chiplet 访问的主要代价不仅是通信延迟，更是丢失了 L3 cache 局部性收益。若数据无论如何都要去主存（如 Attention 的 K/V 在 sequence 维度上 streaming access 无复用），仅减少跨 CCD 通信本身带来的收益有限。

### Attention 与三篇论文应用的对比

| 特征 | OLAP/Sorting/Graph | CPU Attention |
|------|-------------------|---------------|
| Working set 与 L3 关系 | Working set 在 L3 容量边界附近 | 要么 fit in L2（head 维度），要么 streaming 无复用（seq 维度） |
| 数据复用模式 | 排序反复扫描 partition；OLAP 扫描过滤后复用；图遍历 frontier 复用 | K/V 在 sequence 维度 streaming access，一次读取后不再复用 |
| L3 容量敏感度 | 高——能否 fit in L3 决定是否溢出到主存 | 低——限制 L3 大小对性能影响极小（已有实验证据） |
| 优化杠杆点 | 数据布局 + 任务放置 → 控制 L3 命中率 | 内存带宽/计算吞吐 → L3 命中率不是瓶颈 |

## 分析

从三篇论文可直接提炼的事实是：**chiplet 优化有效的充要条件近似于「应用是 L3-cache-sensitive 的」**。具体而言：

1. 通过优化数据布局可以实质性地改变 L2/L3 命中率；或
2. 线程间有频繁的细粒度数据交换（排序的 shuffling、图计算的 frontier 传播），使得跨 chiplet 通信本身成为独立于 cache 的瓶颈

三篇论文中，条件 1 是主要的，条件 2 往往与条件 1 共存——因为跨 chiplet 通信的代价很大一部分正是由于 L3 局部性被破坏。

## 边界

- 三篇论文均在 AMD EPYC Milan / Intel Sapphire Rapids 上实验，未覆盖 ARM Graviton 3 之外的其他 ARM 平台。其中 ARM Graviton 3 因其单计算 chiplet 设计，chiplet-aware 优化效果显著弱于 AMD/Intel（OLAP 论文 Fig.7，CHARM §5.1）。
- 三篇论文评估的 workload（OLAP 查询、排序、图计算、SGD）与 LLM inference 的 Attention kernel 在工作集特征和数据复用模式上有本质差异，不可直接外推方法。
- 三篇论文的优化有效性都建立在一个共同前提上：应用的 working set 大小与 L3 cache 容量处于可比较的量级。当前 Attention kernel 不满足此前提，因此同类方法不能直接迁移。

## 可迁移点

- 「按 chiplet 粒度分配任务以减少跨 chiplet 数据移动」作为通用设计原则，可应用于 LLM inference 中满足 L3 敏感条件的其他环节
- CHARM 的去中心化自适应调度思路（每个 worker 独立根据 PMU 计数器决策）可作为 runtime 优化的设计参考
- 「先通过 micro-benchmark 验证 L3 敏感性再投入优化」的方法论可直接采纳

## 不可直接迁移点

- 三篇论文的 chiplet-aware 调度策略均依赖「working set 可被分区且各分区独立计算」，Attention 的 K/V 被所有 head/query 共享，不具备此特性
- 论文中的 WICP 部署策略需要修改引擎/框架的调度层，在 vllm 的 CPU backend 中对应层面不同
