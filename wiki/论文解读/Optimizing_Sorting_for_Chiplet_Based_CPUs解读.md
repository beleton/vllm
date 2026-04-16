# Optimizing Sorting for Chiplet-Based CPUs 解读

## 说明
- 当前依据的原始资料是 `wiki/原始资料/papers/Optimizing Sorting for Chiplet-Based CPUs.pdf`。
- 该 PDF 首页给出的版本信息是：`VLDB 2024 Workshop: Fifteenth International Workshop on Accelerating Analytics and Data Management Systems Using Modern Processor and Storage Architectures (ADMS 2024)`。
- 本文未额外对照 `arXiv` 或其他版本，以下内容均以这份本地 PDF 为准。

## 问题
- 论文要解决的是：传统排序算法通常只按 `NUMA` 假设做放置与并行，而现代 chiplet CPU 在同一 socket 内还存在分片 `L3`、不同核间延迟和带宽差异，因此单纯 `NUMA-aware` 仍可能导致次优性能（摘要，Sec. 1，Sec. 2.3）。

## 核心内容
- 论文提出四类 chiplet-aware 优化：按 chiplet 粒度切分输入、把内存层级中的 `in-cache` 再细分、按数据规模相对本地/聚合 `L3` 容量选择放置策略、避免昂贵的数据 `shuffling`（摘要，Sec. 1，Sec. 3）。
- 论文不是只给一个调度器，而是把这套思路落到了两类排序实现上：`LSB Radix-Sort` 和基于 range partitioning 的 `Comparison-Sort`（摘要，Sec. 4）。
- 论文摘要报告：相对 `NUMA-aware` 基线，chiplet-aware 方案最高可带来 `2x` 的 `Radix-Sort` 提升和 `4.5x` 的 `Comparison-Sort` 提升（摘要，Fig. 1）。

## 观察
- 论文先指出 chiplet CPU 的异构性不只体现在 `NUMA` 级远端内存访问，还体现在分片 `L3`、核间延迟和带宽差异；`Fig. 2` 给出的 AMD Ryzen 例子中，同一 socket 内核间延迟可相差最高约 `6x`（Fig. 2，Sec. 2.2）。
- 论文进一步说明，标准 `NUMA` 优化无法把数据直接定向放入某个 chiplet 的 `L3`。即使使用 Intel `CAT`，也只能做 LLC way partition/隔离，不能决定数据具体落到哪个 chiplet 的缓存分片（Sec. 2.3）。
- 用 `8` 个线程做 `STREAM` 时，若都放在单个 chiplet 上，带宽在数组大小达到单 chiplet `L3` 容量 `32 MB` 之前更高；超过 `32 MB` 后带宽骤降，因为开始回落到主存访问。相对地，把 `8` 个线程分布到 `8` 个 chiplet 的 `Chiplet_Mixed` 策略带宽更平稳，因为能利用多个 chiplet 的聚合 `L3` 容量（Fig. 3，Sec. 2.3）。
- 在双路 `AMD EPYC Milan 7713` 上，`1 process per chiplet` 的共享无关执行方式在接近总 `L3` 容量时可达到约 `4.8 TB/s` 聚合带宽，而 `1 process per NUMA node` 约为 `3.6 TB/s`；前者在 `L2` 驻留区间的峰值约为 `7 TB/s`（Fig. 4，Sec. 3.2）。

## 方法与系统设计
### 内存层级重解释
- 论文把传统排序里的三段式层级 `in-register / in-cache / out-of-cache` 重新细分为四段：`in-register`、`in-local-chiplet-cache`、`in-remote-chiplet-cache`、`out-of-cache`。其中新增的两段分别对应“数据落在单 chiplet 本地 `L3` 内”和“数据超过单 chiplet `L3`、但仍落在多个 chiplet 聚合 `L3` 内”的情况（Sec. 3.1）。
- 论文的直接含义是：chiplet CPU 上不能把所有 `L3` 命中都视为同一种情况，因为单 chiplet 本地 `L3` 与跨 chiplet 聚合 `L3` 的访问延迟和带宽条件不同（Sec. 3.1，Fig. 3）。

### 按数据规模选择放置
- 论文给出的调度原则是：若数据能放进单个 chiplet 的 `L3`，则优先把任务放在该 chiplet 内；若数据超过单 chiplet `L3`，但仍小于所有 chiplet 聚合 `L3`，则把任务分散到多个 chiplet 以利用更大的缓存容量；若数据超过总 `L3` 容量，则要求核心只访问本地主存，避免远端 `NUMA` 访问（Sec. 4.4）。
- 论文在实验中把这一选择具体写成两种策略：`Chiplet_Local` 表示尽量把任务压在最少 chiplet 上，`Chiplet_Mixed` 表示尽量把任务分散到更多 chiplet 上（Sec. 2.3，Fig. 13）。

### LSB Radix-Sort
- 论文把传统 `NUMA-aware` 的 `LSB Radix-Sort` 与自己的 chiplet-aware 版本并排画成 `Fig. 5` 和 `Fig. 6`。原始版本包含按采样做分区、按 `NUMA` 区域切大块、局部直方图、分区写缓冲、`NUMA` 间数据 `shuffling` 和多轮 radix pass（Fig. 5，Sec. 4.1）。
- chiplet-aware 改动有四点：去掉采样，直接按并行度分区；线程绑定到具体 chiplet 内核心；分区缓冲尽量留在本地 chiplet cache；去掉数据 `shuffling`；并按输入规模动态调整 pass 数（Fig. 6，Sec. 4.1）。
- 论文说明最终通过对各局部直方图做前缀和，确保各分区输出写到互不重叠的位置，再把这些有序段串接为最终输出（Sec. 4.1）。

### Comparison-Sort
- 论文的 `Comparison-Sort` 不是基于 radix，而是基于 `range partitioning`；它仍保留直方图阶段和利用 cache-resident index 记录 range partition 的做法，以避免重复计算 partition function（Sec. 4.2）。
- 其 chiplet-aware 改动与 `Radix-Sort` 基本同向：去掉随机采样，按并行度分区；线程调度考虑 chiplet 边界；根据直方图计算分区偏移；避免 `NUMA` 间 `shuffling`；每个分区内部用 SIMD 优化的 `comb sort` 做原地排序，最后再合并各有序分区（Sec. 4.2）。

### 避免数据 Shuffling
- 论文专门比较了 `data shuffling` 与 `core affinity` 两类思路。其结论不是“所有场景都不该移动数据”，而是：在高 chiplet 数量场景下，`shuffling` 会显著增加跨 chiplet cache 访问，因此对 chiplet-aware 方案反而不利（Sec. 4.3，Fig. 7）。
- 在 `32-bit Radix-Sort + 1B tuples` 下，带 `shuffling` 的 chiplet-aware 方案对本地 `NUMA` 远端 chiplet cache 的访问约 `630K`，对远端 `NUMA` 远端 chiplet cache 的访问约 `562K`；去掉 `shuffling` 后，这两个值分别降到约 `162K` 和 `111K`（Fig. 7，Sec. 4.3）。
- 对 `NUMA-aware LSB Radix-Sort`，`Tab. 1` 显示数据 `shuffling` 可占总排序时间的 `7%` 到 `32%`；例如 `8` 核时最高约 `20%`，`64` 核时最高约 `32%`（Tab. 1，Sec. 4.3）。

### 额外实现优化
- 论文还在直方图和分区阶段加了 `_mm_prefetch(..., _MM_HINT_T0)` 预取，并通过拆分对齐/未对齐路径、减少分支深度和 `__builtin_expect` 改善分支预测（Sec. 4.5）。

## 实验设置
- 主实验平台是双路 `AMD EPYC Milan 7713`。每个 socket 有 `64` 个 CPU core、`512 GB RAM`、`8` 个 chiplet，每个 chiplet 带 `32 MB L3`；整机共 `16` 个 chiplet（Sec. 3.2，Sec. 5.1）。
- OS 是 `Ubuntu 23.04`，编译器是 `GCC 12 -O3`。论文保持与基线一致，使用 `SSE (128-bit SIMD)` 指令在 `AVX (256-bit)` 寄存器上运行，以隔离 chiplet-aware 优化本身的影响（Sec. 5.1）。
- 吞吐统一用 `GB/s` 计，而不是 `tuples/s`；除非另有说明，输入为均匀随机分布，实验默认使用 `1B tuples`、`16 cores`，每个结果取 `10` 次执行平均值（Sec. 5.1）。
- `Radix-Sort` 和 `Comparison-Sort` 的扩展性图使用 `100M tuples`，比较 `16/32/64-bit` key，在不同核心数下的 chiplet-aware 与 `NUMA-aware` 两种策略（Fig. 8，Fig. 10）。

## 结果与解释
- 总体上，`Fig. 1` 给出的是跨输入规模与 key 宽度的 speedup 汇总，摘要和引言把最高收益概括为：`Radix-Sort` 最高约 `2x`，`Comparison-Sort` 最高约 `4.5x`（摘要，Fig. 1）。
- 对 `Radix-Sort`，论文说明 chiplet-aware 在所有 key 宽度和核心数上都持续优于 `NUMA-aware`。以 `100M tuples` 为例，`32 cores` 时 chiplet-aware 吞吐约 `24 GB/s`，而 `NUMA-aware` 约 `16.2 GB/s`（Fig. 8，Sec. 5.2）。
- 论文同时指出 `Radix-Sort` 在 `24` 核之后扩展性开始放缓，原因是更多线程共享同一 chiplet 时，每线程可用本地 cache 份额变小，主存访问增加，算法更偏 memory-bound；`Fig. 9` 中主存访问量也在这一点之后明显上升（Fig. 9，Sec. 5.2）。
- 对 `Comparison-Sort`，chiplet-aware 在 `8-96` 核区间同样整体更好。论文给出更具体的 speedup：`16-bit` 平均 `1.23x`、最大 `1.41x`；`32-bit` 平均 `1.40x`、最大 `1.64x`；`64-bit` 整体 `1.34x`、峰值 `1.61x`（Fig. 10，Sec. 5.3）。
- 论文还比较了两类排序之间的差异：`16/32-bit` 时 `Radix-Sort` 整体吞吐更高，但 `Comparison-Sort` 扩展性更好；到 `64-bit` 时则反转为 `Comparison-Sort` 更优。作者把这一点归因于 `Radix-Sort` 面对更宽 key 时需要更多 pass，重复做直方图和分区，额外开销更重；而 `Comparison-Sort` 只需一次直方图/分区后在 cache 内排序（Sec. 5.4，Fig. 11）。
- 在输入规模实验中，`64-bit Radix-Sort` 上 chiplet-aware 对 `NUMA-aware` 在 `1M/10M/100M/1B tuples` 四个点都约快 `39%-41%`（Fig. 11a，Sec. 5.5）。
- `64-bit Comparison-Sort` 上结果更依赖规模：`1M tuples` 时两者接近；到 `10M` 和 `100M tuples` 时 chiplet-aware 约快 `17%-49%`；到 `1B tuples` 时优势扩大到约 `4.29x`，因为 `NUMA-aware` 在大输入下明显退化（Fig. 11b，Sec. 5.5）。
- `Tab. 2` 给出的主存访问数支持这一点。`NUMA-aware` 从 `100M` 到 `1B tuples` 时，本地主存访问从 `39,683 x10^3` 增至 `3,236,810 x10^3`，远端主存访问从 `12,492 x10^3` 增至 `652,202 x10^3`；而 chiplet-aware 分别从 `11,681 x10^3`、`20,084 x10^3` 增到 `140,405 x10^3`、`218,315 x10^3`，增长更接近输入规模本身（Tab. 2，Sec. 5.5）。
- 在 skew 实验中，`32-bit Radix-Sort` 的 Zipf 参数从 `1.2` 增到 `2.0` 时，chiplet-aware 吞吐从约 `6.1 GB/s` 升到约 `7.5 GB/s`，而 `NUMA-aware` 从约 `5.0 GB/s` 降到约 `3.8 GB/s`。论文说明这是因为 chiplet-aware 的 cache miss rate 从 `6%` 下降到 `1.5%`（Fig. 12，Sec. 5.6）。
- 在调度策略实验中，`32-bit Radix-Sort + 8 threads` 下，小数据更适合 `Chiplet_Local`，大数据更适合 `Chiplet_Mixed`。论文明确把这一现象解释为：前者更贴近单 chiplet 本地 cache，后者能利用多个 chiplet 的聚合 `L3`（Fig. 13，Sec. 5.7）。
- 消融实验进一步说明收益来源。以 `32-bit`、`100M tuples`、`32 cores` 为例，`Radix-Sort` 从 `NUMA-aware` 的 `10.71 GB/s`，到去掉 `shuffling` 的 `10.73 GB/s`、加 `Chiplet_Local` 的 `10.93 GB/s`、加 `Chiplet_Mixed` 的 `14.87 GB/s`、再加分支预测到 `15.83 GB/s`、再加预取到 `16.25 GB/s`。`Comparison-Sort` 对应值是 `3.98/4.69/4.73/5.77/5.79/5.82 GB/s`，其中最大增益同样来自去掉 `shuffling` 和引入 chiplet-aware 调度（Fig. 15，Sec. 5.8）。

## 分析
- 从论文可直接提炼的事实是：chiplet CPU 上“更局部”不总是更快。若工作集能放进单 chiplet `L3`，`Chiplet_Local` 更优；若工作集超过单 chiplet `L3`、但仍在聚合 `L3` 范围内，跨多个 chiplet 的 `Chiplet_Mixed` 反而更优（Fig. 3，Fig. 13）。
- 论文也直接说明：只做到 `NUMA-aware` 不足以处理 chiplet CPU 上的分片 `L3` 问题，因为 `NUMA` 级本地性和 chiplet 级 cache 本地性不是同一层次（Sec. 2.3，Sec. 3.1）。
- 论文报告的收益依赖具体对象：固定长度整数排序、`EPYC Milan` 平台、`GB/s` 吞吐口径，以及论文中的分区/缓冲/直方图实现；不能脱离这些条件单独解读（Sec. 4，Sec. 5）。

## 边界
- 论文研究对象是排序，不是 LLM 推理，也不是 attention / KV cache。
- 论文的输入是固定长度 `key/payload` 数组，主要工作是分区、直方图、原地排序和合并；这与张量算子、KV 读写和调度开销的结构不同。
- 主实验平台是 `AMD EPYC Milan 7713`，并辅以 `AMD Ryzen` 的延迟图示，不覆盖当前所有 CPU 平台与 workload。
- 论文没有直接给出 `L3 miss latency`、`L3 slice/CCX` 粒度计数器，也没有测 `vLLM` 或大模型服务负载。

## 可迁移点
- 可以直接借鉴的方法是：不要把 chiplet CPU 只看成 `NUMA` 的粗粒度问题，而要把“单 chiplet 本地 `L3`”和“跨 chiplet 聚合 `L3`”作为不同工作区间来处理（Sec. 3.1，Fig. 3）。
- 论文给出的尺寸驱动调度规则可迁移为通用启发：先估工作集是否落在单局部缓存，再决定是否扩大到多个局部域共享更大的聚合缓存（Sec. 4.4，Fig. 13）。
- 若负载能在前置分区阶段做均衡，避免后续昂贵数据 `shuffling` 可能比“先做 NUMA 均衡、再重排数据”更有效（Tab. 1，Fig. 7，Fig. 15）。

## 不可直接迁移点
- 论文里的吞吐增益不能直接映射到当前 `vLLM CPU`、`attention-only` 或整模型推理，因为任务类型、数据结构和热路径不同。
- 论文中的 `Chiplet_Local/Chiplet_Mixed` 是针对排序工作集和分区缓冲设计的，不等于当前 attention kernel 的 head/KV 调度策略。
- 论文没有直接研究 `CCX/L3 slice` 计数器、来源分布或 `L3 miss` 延迟，因此不能把它的结果直接当作当前 profiling 结论。

## 证据
- 原始资料：`wiki/原始资料/papers/Optimizing Sorting for Chiplet-Based CPUs.pdf`
- 关键锚点：摘要；`Fig. 1-15`；`Tab. 1-2`；`Sec. 2.3`；`Sec. 3.1-3.2`；`Sec. 4.1-4.5`；`Sec. 5.1-5.8`；`Sec. 8`
