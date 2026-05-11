# Optimizing Sorting for Chiplet-Based CPUs 解读

## 说明
- 依据版本：VLDB 2024 Workshop ADMS 2024（Fifteenth International Workshop on Accelerating Analytics and Data Management Systems Using Modern Processor and Storage Architectures）。
- 以下内容均以本地 PDF 为准，未对照 arXiv 或其他版本。

## 问题
传统排序算法通常只按 NUMA 假设做放置与并行。Chiplet CPU 在同一 socket 内还存在分片 L3、不同核间延迟和带宽差异（同 socket 内核间延迟最高相差约 6x，Fig. 2），只做到 NUMA-aware 仍可能次优（摘要，Sec. 1, Sec. 2.3）。论文要回答的是：如何在排序算法中显式建模 chiplet 级 L3 分片和核间延迟差异，并在不改变排序数学正确性的前提下获得性能提升。

## 背景：排序算法的计算流程与访存特征

### LSB Radix Sort 的原理与访存模式

Radix Sort（基数排序）不比较 key 大小，而是按 key 的二进制位逐轮分桶。以 32 位整数为例，每次处理 8 位（一个 byte），共 4 轮（pass）。每轮操作完全一致（Sec. 4.1）：

1. **扫描全数组**：逐个读入 tuple，提取当前轮对应的 8 位作为桶号（0–255）。
2. **直方图统计**：统计每个桶各有多少条 tuple。
3. **前缀和计算偏移量**：根据直方图算出每个桶的输出起始位置。
4. **分区写输出**：再次扫描全数组，根据桶号写入输出缓冲区的对应位置。

一轮结束后数据按当前 8 位有序。下一轮处理下一个更高 8 位，在上一轮有序基础上稳定化。4 轮后整个 32 位数组有序。

**Radix Sort 的访存特征决定了它在 chiplet 架构上的性能瓶颈**：

- **每轮 pass 扫描和重写整个输入数组**。输入 10 GB，每 pass 产生 10 GB 读 + 10 GB 写。32-bit key 需要 4 轮 pass，总数据移动量 = 输入体积 × 8。
- **直方图缓冲区被频繁访问**。每个线程维护自己的局部直方图（256 个桶 × 每个桶的计数），在分区阶段需要反复读直方图来确定写入位置。直方图缓冲区的放置位置直接影响访问延迟。
- **多轮 pass 间，数据布局不断变化**。每轮输出是下一轮的输入，数据在各 chiplet L3 之间的分布取决于上一轮的分区逻辑。如果分区逻辑不考虑 chiplet 边界，数据就会散布到所有 chiplet 的 L3 中，后续访问全部变成跨 chiplet 命中。

### Comparison-Sort 的流程

Comparison-Sort 只做一次 range partitioning（按 key 值范围分区），然后对每个分区内部做比较排序，最后合并（Sec. 4.2）：
1. **直方图**：每个线程扫描数据，统计各值域范围的 key 数量。
2. **分区偏移量计算**：根据直方图计算各分区输出位置。
3. **分区内排序**：每个分区内部用 SIMD 优化的 comb sort 原地排序。
4. **合并**：各分区已内部有序，拼接即为全局有序。

与 Radix Sort 相比，Comparison-Sort 只做一次直方图/分区，减少了数据扫描次数，但分区内排序是计算密集型操作。两者对 chiplet 局部性的敏感点不同。

### NUMA-aware 排序的标准流程及其在 chiplet 上的问题

传统 NUMA-aware LSB Radix-Sort 的一轮 pass 包含 6 步（Fig. 5, Sec. 4.1）：

1. **Key Sampling**：随机抽取部分 key，根据采样分布确定分区边界（delimiter），使各 NUMA node 分到的数据量大致均衡。
2. **数据按 NUMA 切分**：按 Step 1 的边界将输入数组切成 NUMA 大块。
3. **局部直方图**：每个线程对分配给自己的数据段扫描，统计各桶的 tuple 数量。
4. **分区写缓冲**：根据直方图，各线程将数据按桶写入本地 NUMA node 主存中的输出缓冲。
5. **跨 NUMA Shuffling**：因采样不精确，各 NUMA node 数据量不均衡时，进行跨 NUMA 数据搬迁以平衡负载。
6. **重复**：Steps 3–5 作为一轮。根据 key 位宽决定 pass 数。

**这套流程在 chiplet CPU 上的根本问题是 Step 5（Shuffling）**：shuffling 把数据从当前 chiplet 的 L3 搬到另一个 chiplet 的主存。搬完后，后续对该数据的访问全部变成跨 chiplet cache 命中（106 ns, 6 GB/s），而不是本地 chiplet L3 命中（24 ns, 12 GB/s）。Tab. 1 显示 shuffling 占 NUMA-aware LSB Radix-Sort 总时间的 15%–32%（64 核时），核数越多占比越大。

## 观察：Chiplet CPU 上的内存层级需要重新建模

### 传统三层模型不适用

传统多核 CPU 上排序算法把访存分为三层：in-register（寄存器）、in-cache（L3）、out-of-cache（主存）。在这个模型下，"in-cache" 被假设为同质的——一次 L3 命中就是一次 L3 命中，不分来源。

Chiplet CPU 上这个假设不成立。AMD EPYC Milan 的 L3 被切成 8 个独立分片（每 chiplet 32 MB），chiplet 内命中约 24 ns，跨 chiplet 命中约 106 ns（Fig. 3a）。两者都叫 L3 命中，性能差 4 倍。论文因此把 in-cache 拆成两层（Sec. 3.1）：
- **in-local-chiplet-cache**：数据在本地 chiplet L3 内，24 ns，12 GB/s
- **in-remote-chiplet-cache**：数据在另一 chiplet L3 内，106 ns，6 GB/s

### 工作集大小与放置策略的关系：STREAM 基准的启发

论文在单路 AMD EPYC Milan 7713 上用 8 线程跑 STREAM TRIAD，按数组大小分三段对比两种线程放置策略（Fig. 3, Sec. 2.3）：

- **Chiplet_Local**：8 线程全部放在同一 chiplet 的 8 个 core 上，共享 32 MB L3。
- **Chiplet_Mixed**：8 线程分散到 8 个 chiplet，每 chiplet 取一个 core，可用 256 MB 聚合 L3。

结果分三段：
1. **数组 < 32 MB**（单 chiplet L3 容量内）：Chiplet_Local 带宽更高。数据全部命中本地 L3（24 ns），带宽可达数百 GB/s，远超 DRAM 带宽。
2. **数组 32–256 MB**（超出单 chiplet L3，在聚合 L3 内）：Chiplet_Local 带宽骤降，Chiplet_Mixed 保持平稳。Local 的 32 MB L3 已装不下，所有线程回退到主存；Mixed 的 256 MB 聚合 L3 仍能容纳。
3. **数组 > 256 MB**（超出聚合 L3）：两者趋同，全部回退到 DRAM。

这一结果说明：Chiplet CPU 上的最优放置策略不是固定的，而是取决于数据规模与 L3 容量之间的关系。核心权衡是"用延迟换容量"——单 chiplet 内 L3 延迟最低但容量有限，跨 chiplet 聚合 L3 容量更大但命中延迟更高。

## 方法：Chiplet-Aware 排序的四项改动

论文在两套排序实现（LSB Radix-Sort 和 Comparison-Sort）上做了四项 chiplet-aware 改动（Sec. 4）。

### 改动一：去掉采样（删除 NUMA-aware 流程的 Step 1）

不再通过随机采样猜测 key 分布，而是直接按并行度均匀切分输入数据。

**与 chiplet 架构的关联**：采样的流程是线程随机扫描全数组来估计 key 分布，读到的数据可能散布在所有 chiplet 的内存和 cache 中。这个过程本身产生跨 chiplet 访问。均匀切分消除了这一步的额外扫描开销和随之而来的跨 chiplet 流量。

**为什么均匀切分足够**：每个 chiplet 处理的数据段大小相同，最终通过直方图的前缀和来精确确定输出位置。不均衡问题通过后续的精确偏移量计算来解决，不需要预先的统计采样（Sec. 4.1）。

### 改动二：线程绑定到 chiplet 内核心（替换 Step 2）

不再把数据段绑定到 NUMA 区域，而是把每个线程直接固定到 chiplet 内的一个 core 上，数据分配也改为按 chiplet 粒度：一个 chiplet 一个数据分区，该 chiplet 上的线程只处理这个分区。

**关键输入参数**：输入数据大小、单 chiplet L3 容量（32 MB）、需要的并行度、chiplet 数量。线程放置策略（Chiplet_Local 还是 Chiplet_Mixed）在此步骤根据数据大小与 L3 容量的关系决定（Sec. 4.4）。

### 改动三：分区缓冲留在本地 chiplet L3（修改 Step 4）

每个线程做分区时，输出缓冲尽量保持在本地 chiplet 的 L3 内。依赖改动二的线程绑定——线程跑在哪个 chiplet 上，写缓冲就被哪个 chiplet 的 L3 缓存。

**与排序访存特征的关联**：分区阶段每个线程要反复读取直方图缓冲区（确定写入偏移量）和写入输出缓冲。如果这两块缓冲区在本地 chiplet L3 内，每次访问约 24 ns；如果被 coherence 拉到远端 chiplet L3，每次约 106 ns。对于每轮 pass 都要重写整个输入的 Radix Sort，这个延迟差异会放大到整个运行时间。

### 改动四：去掉跨 NUMA Shuffling（删除 Step 5）

四项改动中贡献最大的一项（Fig. 15）。论文在 Sec. 4.3 和 Fig. 7 中专门论证了 shuffling 在 chiplet 上适得其反。

**机制**：shuffling 把数据从原本处理的 chiplet 的 L3 搬到另一个 chiplet 的主存。搬完后，原本对这份数据的所有本地 chiplet L3 访问全部变成了远端 chiplet cache 访问。

**数据证据（Fig. 7）**：
- chiplet-aware + shuffling：local NUMA remote chiplet cache fill = 630K，remote NUMA remote chiplet cache fill = 562K
- chiplet-aware 不加 shuffling：分别降至 162K 和 111K
- NUMA-aware 去掉 shuffling 后远端 chiplet cache fill 反而更高（521K），因为没有了 shuffling 的均衡作用，NUMA-aware 的访存集中在少数热点区域，引发了更严重的 cache 冲突

**与排序访存特征的关联**：Radix Sort 每轮 pass 都要扫描和重写整个输入，多轮 pass 间同一份数据会被反复访问。如果 shuffling 把数据搬离了第一次处理时所在的 chiplet L3，后续多轮 pass 的所有访问都是跨 chiplet 的。shuffling 在 chiplet 上的代价被多轮 pass 放大了。

### 额外实现优化（Sec. 4.5）

- 在直方图和分区阶段加 `_mm_prefetch(..., _MM_HINT_T0)` 预取
- 通过拆分对齐/未对齐路径、减少分支深度和 `__builtin_expect` 改善分支预测
- 消融实验显示这两项对 Radix-Sort 合计贡献约 15%（Fig. 15a），对 Comparison-Sort 基本无效（Fig. 15b）

## 实验设置（Sec. 5.1）

- 主平台：双路 AMD EPYC Milan 7713（每 socket 64 core, 512 GB RAM, 8 chiplet, 每 chiplet 32 MB L3），共 16 chiplet。
- OS: Ubuntu 23.04，编译器: GCC 12 -O3。
- 使用 SSE (128-bit SIMD) 指令在 AVX (256-bit) 寄存器上运行，以隔离 chiplet-aware 优化本身的影响。
- 吞吐用 GB/s（非 tuples/s）。除非另有说明，输入为均匀随机分布，默认 1B tuples、16 cores，每个结果取 10 次执行平均值。

## 结果与解释

### 整体收益（Fig. 1）
Radix-Sort 最高约 2x，Comparison-Sort 最高约 4.5x，均对比 NUMA-aware 方案。

### Radix-Sort（Fig. 8, Fig. 9, Sec. 5.2）
- 所有 key 宽度（16/32/64-bit）和核心数上持续优于 NUMA-aware。100M tuples、32 cores 时吞吐约 24 GB/s vs 16.2 GB/s。
- 24 核后扩展性放缓。论文归因于更多线程共享同一 chiplet 时，每线程可用本地 cache 份额变小，主存访问增加，算法更偏 memory-bound（Fig. 9）。

### Comparison-Sort（Fig. 10, Sec. 5.3）
- chiplet-aware 在 8–96 核区间整体更优：16-bit 平均 1.23x / 最大 1.41x，32-bit 平均 1.40x / 最大 1.64x，64-bit 整体 1.34x / 峰值 1.61x。

### 两类排序互相对比（Fig. 11, Sec. 5.4）
- 16/32-bit：Radix-Sort 整体吞吐更高，但 Comparison-Sort 扩展性更好。
- 64-bit：Comparison-Sort 反超。因为 Radix-Sort 面对更宽 key 需要更多 pass（64-bit 要 8 轮），重复做直方图和分区的额外开销更重；Comparison-Sort 只需一次直方图/分区。

### 输入规模影响（Fig. 11, Tab. 2, Sec. 5.5）
- 64-bit Radix-Sort：chiplet-aware 在 1M / 10M / 100M / 1B tuples 四个点都约快 39%–41%。
- 64-bit Comparison-Sort：1B tuples 时 chiplet-aware 优势扩大到约 4.29x，因为 NUMA-aware 在大输入下明显退化。Tab. 2 显示：NUMA-aware 从 100M 到 1B tuples，本地主存访问从 39,683×10³ 增至 3,236,810×10³，远端主存访问从 12,492×10³ 增至 652,202×10³；chiplet-aware 分别从 11,681×10³ / 20,084×10³ 增到 140,405×10³ / 218,315×10³，增长更接近输入规模本身的线性比例。

### Skew 的影响（Fig. 12, Sec. 5.6）
32-bit Radix-Sort，Zipf 参数从 1.2 增到 2.0：chiplet-aware 吞吐从约 6.1 GB/s 升到约 7.5 GB/s（cache miss rate 从 6% 降到 1.5%），NUMA-aware 从约 5.0 GB/s 降到约 3.8 GB/s。Skew 增强了局部性，chiplet-aware 从中受益更多。

### 调度策略选择（Fig. 13, Sec. 5.7）
32-bit Radix-Sort + 8 threads：小数据（15 MB）时 Chiplet_Local 吞吐更高；大数据（150 MB+）时 Chiplet_Mixed 反超。分界线是单 chiplet L3 容量 32 MB，与 STREAM benchmark 的观察一致。

### 消融实验（Fig. 15, Sec. 5.8）
32-bit、100M tuples、32 cores 下，Radix-Sort 从 NUMA-aware 的 10.71 GB/s 逐步提升：去 shuffling → 10.73 → 加 Chiplet_Local → 10.93 → 加 Chiplet_Mixed → 14.87 → 加分支预测 → 15.83 → 加预取 → 16.25 GB/s。Comparison-Sort 对应值：3.98 → 4.69 → 4.73 → 5.77 → 5.79 → 5.82 GB/s。最大单次增益来自去 shuffling 和引入 chiplet-aware 调度。

## 分析
- chiplet CPU 上"更局部"不总是更快。工作集 < 单 chiplet L3 → Chiplet_Local 更优；工作集超出单 chiplet L3 但 < 聚合 L3 → Chiplet_Mixed 更优（Fig. 3, Fig. 13）。
- NUMA-aware 不足以处理 chiplet CPU 上的分片 L3 问题。NUMA 级本地性和 chiplet 级 cache 本地性不是同一层次（Sec. 2.3, Sec. 3.1）。
- 论文报告的收益依赖具体对象：固定长度整数排序、EPYC Milan 平台、论文中的分区/缓冲/直方图实现，不能脱离这些条件解读。

## 边界
- 研究对象是排序，不是 LLM 推理，不是 attention / KV cache。
- 输入是固定长度 key/payload 数组，主要工作是分区、直方图、原地排序和合并；与张量算子、KV 读写和调度开销的结构不同。
- 主平台是 AMD EPYC Milan 7713，不覆盖 Genoa/Bergamo 等 newer 平台。
- 论文未直接测量 L3 miss latency、L3 slice/CCX 粒度计数器，也未测 vLLM 或大模型服务负载。

## 可迁移点
- 把 chiplet CPU 的内存层级从三层（register/cache/DRAM）显式拆为四层（register/local-L3/remote-L3/DRAM），在算法设计中区分 local-L3 和 remote-L3 命中（Sec. 3.1）。
- 按工作集大小驱动放置策略选择：先估工作集是否落在单 chiplet 本地 L3，再决定是否跨 chiplet 共享更大聚合缓存（Sec. 4.4, Fig. 13）。
- 避免跨 chiplet 数据 shuffling 的启发：若负载能在前置分区阶段做均衡，避免后续昂贵的数据搬迁可能比"先做 NUMA 均衡再重排"更有效（Tab. 1, Fig. 7, Fig. 15）。这项启发尤其适用于多轮 pass 反复扫描的工作负载。

## 不可直接迁移点
- 吞吐增益不能直接映射到当前 vLLM CPU attention-only 或整模型推理，任务类型和数据结构不同。
- Chiplet_Local/Chiplet_Mixed 针对排序工作集和分区缓冲设计，不等于当前 attention kernel 的 head/KV 调度策略。
- 论文的"多轮 pass 放大跨 chiplet 代价"效应，在 attention 中不存在对等机制——attention 对每个 KV tile 只读取一次（streaming），不存在排序中同一数据被多轮反复访问的模式。

## 证据
- 原始资料：`wiki/原始资料/papers/Optimizing Sorting for Chiplet-Based CPUs.pdf`
- 关键锚点：摘要；Fig. 1–15；Tab. 1–2；Sec. 2.3；Sec. 3.1–3.2；Sec. 4.1–4.5；Sec. 5.1–5.8；Sec. 8
