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
- 论文使用 `STREAM` 基准测试来探测 chiplet 的 cache/内存层级带宽特性。`STREAM` 是由 John McCalpin 开发的微基准，用于测量可持续主存带宽，包含四种向量操作：`COPY`（`a[i] = b[i]`，每元素 16 bytes）、`SCALE`（`a[i] = q * b[i]`，每元素 16 bytes）、`ADD`（`a[i] = b[i] + c[i]`，每元素 24 bytes）、`TRIAD`（`a[i] = b[i] + q * c[i]`，每元素 24 bytes）。其核心特点是计算极少（每迭代一次乘加）、访存极多，瓶颈完全在内存带宽。`STREAM` 的线程之间无需通信，每个线程只处理数组中互不重叠的一段，因此线程间带宽差异与同步开销无关，纯粹反映缓存和内存层级特性。
- 论文在单路 `AMD EPYC Milan 7713` 上用 `8` 线程跑 `STREAM TRIAD`，对比了两种线程放置策略（Fig. 3，Sec. 2.3）：
    - **Chiplet_Local**：把 `8` 个线程全部放在同一个 chiplet 的 `8` 个 core 上，所有线程共享该 chiplet 的 `32 MB L3`。
    - **Chiplet_Mixed**：把 `8` 个线程分布到 `8` 个 chiplet，每个 chiplet 取一个 core，可利用 `8 × 32 = 256 MB` 聚合 `L3`。
    - 实验结果按数组大小分为三段：
        1. **数组 < 32 MB（单 chiplet L3 容量内）**：`Chiplet_Local` 带宽更高。原因是数据全部命中本地 L3，单 chiplet L3 的延迟约 24 ns、带宽可达几百 GB/s，远超 DRAM 带宽（每内存通道约 20-25 GB/s）。`Chiplet_Mixed` 虽能利用 8 个内存控制器，但走的还是 DRAM，无法匹敌 L3 带宽。
        2. **数组在 32 MB ~ 256 MB（超出单 chiplet L3，但在聚合 L3 内）**：`Chiplet_Local` 带宽骤降，`Chiplet_Mixed` 保持平稳。原因是 Local 的 32 MB L3 已装不下数据，所有线程回退到主存访问；而 Mixed 的 256 MB 聚合 L3 仍能容纳，继续享受 L3 命中。
        3. **数组 > 256 MB（超出聚合 L3 容量）**：两者趋同，全部回退到 DRAM，谁也占不到 cache 的便宜。
    - 这一结果说明：并非 `Chiplet_Mixed` 在所有情况下都更优，而是取决于数据规模与 L3 容量之间的关系。这是一个"用延迟换容量"的权衡——单 chiplet 内 L3 延迟最低但容量有限，跨 chiplet 聚合 L3 容量更大但命中延迟更高（本地 24 ns vs 跨 chiplet 106 ns）。
- 在双路 `AMD EPYC Milan 7713` 上，`1 process per chiplet` 的共享无关执行方式在接近总 `L3` 容量时可达到约 `4.8 TB/s` 聚合带宽，而 `1 process per NUMA node` 约为 `3.6 TB/s`；前者在 `L2` 驻留区间的峰值约为 `7 TB/s`（Fig. 4，Sec. 3.2）。

## 方法与系统设计

### 内存层级重解释：为什么 chiplet CPU 上需要重新划分排序阶段

传统多核 CPU 上的排序算法把访存分为三个层级：`in-register`（寄存器内）、`in-cache`（L3 cache 内）、`out-of-cache`（超出 L3，落到主存）。在这个模型下，`in-cache` 阶段被假设为同质的——一次 L3 命中就是一次 L3 命中，不分来源。

chiplet CPU 上这个假设不再成立。AMD EPYC Milan 的 L3 被切分为 8 个独立分片，每个 chiplet 独占 32 MB，chiplet 内命中延迟约 24 ns，跨 chiplet 命中延迟约 106 ns（Fig. 3a）。两者都叫 L3 命中，性能差 4 倍以上。论文因此把 `in-cache` 拆成两层：`in-local-chiplet-cache`（数据在本地 chiplet L3 内，24 ns、12 GB/s）和 `in-remote-chiplet-cache`（数据在另一个 chiplet L3 内，106 ns、6 GB/s）（Sec. 3.1，Fig. 3）。

这个拆分的实际后果是：排序算法需要在两个新的工作区间上做不同决策。当数据能放入单个 chiplet L3 时，把线程绑在同一个 chiplet 上获得最低延迟；当数据超出单 chiplet L3 但仍在所有 chiplet 聚合 L3 内时，需要把线程分散以利用更大的总缓存容量；当数据超出总 L3 时，两种策略都回退到主存，但需要确保主存本地分配避免远端 NUMA 访问（Sec. 3.1，Sec. 3.2，Fig. 4）。

### 按数据规模选择放置：Chiplet_Local 与 Chiplet_Mixed 的决策逻辑

论文把这个决策具体化为两种线程放置策略（Sec. 4.4，Fig. 13）：

**Chiplet_Local**：把所有工作线程集中在尽可能少的 chiplet 上（极限情况是一个 chiplet 的 8 个 core）。优点是所有 cache 命中都是本地 chiplet L3，零跨 chiplet 通信；缺点是可用 L3 只有 32 MB，数据超过这个量时立即回退到主存。

**Chiplet_Mixed**：把线程分散到多个 chiplet 上（每个 chiplet 取一个 core）。优点是能用多个 chiplet 的聚合 L3（256 MB），数据更大时仍可留在 cache 内；缺点是每个线程的邻居数据在别的 chiplet L3 里，访存时发生跨 chiplet cache coherence。

两种策略的分界线是单 chiplet L3 容量 32 MB。STREAM benchmark 在单 socket AMD Milan 上用 8 线程测量：数组 < 32 MB 时 Chiplet_Local 带宽更高；数组在 32 MB 到 256 MB（聚合 L3 大小）之间时 Chiplet_Mixed 带宽更平稳；超过 256 MB 后两者趋同（Fig. 3，Sec. 2.3）。在 32-bit Radix-Sort 实际排序中，这个转折同样存在：15 MB 输入时 Chiplet_Local 吞吐更高，150 MB 和更大输入时 Chiplet_Mixed 反超（Fig. 13，Sec. 5.7）。

### LSB Radix-Sort 的方法与 chiplet-aware 改动

#### Radix Sort 的基本原理

Radix Sort（基数排序）不通过比较两个 key 的大小来排序，而是按 key 的二进制位（digit）逐轮分桶。以 32 位整数为例，如果每次处理 8 位（一个 byte），就需要 4 轮（pass）。每轮的操作完全一致：扫描全部数据 → 统计每个桶（0-255）各有多少条 tuple（直方图）→ 根据直方图算出每个桶的输出偏移量 → 把每条 tuple 写到对应桶的输出位置。这一轮结束后，数据已经按这一轮处理的 8 位排好序。下一轮再处理下一个 8 位，在上一轮的有序基础上进一步稳定化。4 轮以后整个数组有序。LSB（Least Significant Bit）意味着从最低位开始，先按低位分桶，再逐轮向高位推进，最终高位决定总体顺序，低位决定同等高位内的顺序（Sec. 4.1）。

这种算法的一个关键特征是：**每轮 pass 都要扫描和重写整个输入数组**。输入数据 10 GB，每 pass 就产生 10 GB 读 + 10 GB 写。如果有多轮 pass（32-bit 要 4 轮，64-bit 要 8 轮），数据总量就是输入体积的若干倍。这些读写请求在所有 chiplet L3 之间穿行，每次命中的是本地 chiplet L3 还是远端 chiplet L3，对延迟和带宽的影响完全不同。

#### NUMA-aware LSB Radix-Sort 的流程（Fig. 5）

论文在 Fig. 5 中画出了传统 NUMA-aware 版本的一个 radix pass 的 6 步流程：

**Step 1 — Key Sampling（采样确定分区边界）**：从输入数据中随机抽取一部分 key，根据这些采样 key 的分布确定分区边界（delimiter）。这些边界决定哪些 key 范围属于哪个 NUMA node。采样的目的是让各 NUMA node 分到的数据量大致均衡。

**Step 2 — 数据按 NUMA 切分**：根据 Step 1 确定的边界，把整个输入数组按 NUMA 区域切成大块。每个 NUMA node 上的线程只处理属于本 node 的数据段。

**Step 3 — 局部直方图**：每个线程对分配给自己的数据段做一次扫描，统计各桶（radix bucket，例如 8 位 → 256 个桶）的 tuple 数量。直方图是每线程独立的，记录了本线程数据段内每个桶的计数。

**Step 4 — 分区写缓冲**：根据直方图，每个线程把数据按桶写到输出缓冲区。缓冲区分配在本地 NUMA node 的主存中，目的是让后续访问尽量走本地内存控制器。

**Step 5 — 跨 NUMA 数据 Shuffling**：由于采样的不精确性，各 NUMA node 分到的数据量可能不均衡。这一步把数据在 NUMA node 之间搬迁，以平衡负载。

**Step 6 — 重复**：Steps 3-5 作为一轮 pass。根据 key 的位宽（16/32/64 bit）决定需要多少轮 pass。每轮 pass 结束后的输出数据成为下一轮 pass 的输入。

#### chiplet-aware LSB Radix-Sort 的改动（Fig. 6）

论文对上述流程做了四处关键改动，对应 Fig. 6 中的简化流程：

**改动一：去掉采样（删除 Step 1）。** 不再通过采样猜测数据分布，而是直接根据并行度均匀切分输入数据。这避免了采样的额外扫描开销和采样不精确导致的后续不均衡。论文的论点是：在 chiplet 粒度做负载均衡时，均匀切分已经足够有效，因为每个 chiplet 处理的数据段大小相同，且最终通过直方图的前缀和来精确确定输出位置（Sec. 4.1）。

**改动二：线程绑定到 chiplet 内核心（替换 Step 2）。** 不再把数据段绑定到 NUMA 区域，而是把每个线程直接固定到 chiplet 内的一个 core 上。数据分配也从按 NUMA 改为按 chiplet 粒度：一个 chiplet 一个数据分区，该 chiplet 上的线程只处理这个分区。这里的关键输入参数是：输入数据大小、单 chiplet L3 容量、需要的并行度、和 chiplet 数量。线程调度策略（Chiplet_Local 还是 Chiplet_Mixed）就是在这个步骤根据数据大小和 L3 容量的关系来决定的（Sec. 4.4）。

**改动三：分区缓冲留在本地 chiplet cache（修改 Step 4）。** 每个线程做分区时，输出缓冲尽量保持在本地 chiplet 的 L3 内，不主动触及远端 chiplet 的 cache line。这一步依赖于改动二的线程绑定——线程跑在哪个 chiplet 上，写缓冲就被那个 chiplet 的 L3 缓存。

**改动四：去掉跨 NUMA 数据 shuffling（删除 Step 5）。** 改动二和三保证了数据在 chiplet 内局部处理，不需要再像 NUMA-aware 那样在每个 pass 结尾做数据搬迁来平衡负载。论文在 Sec. 4.3 和 Fig. 7 中专门论证了：当 chiplet 数量多时，shuffling 反而增加跨 chiplet cache 访问（因为它把数据移动到了不属于自己 chiplet 的位置，后续访问变成了远端 chiplet cache 命中）。

**最终合并**：所有线程完成各自的直方图和分区后，对所有局部直方图做一次前缀和（prefix sum），计算出每个 bucket 的输出在整个数组中的起始偏移量。各线程据此把各自有序的分区数据写到互不重叠的全局输出位置。最终输出就是把所有有序段拼接在一起得到的完整有序数组（Sec. 4.1）。

#### 为什么这些改动对 chiplet CPU 有效

去掉 shuffling 后，每个线程从头到尾只访问自己 chiplet 本地的直方图缓冲和分区输出缓冲。所有 L3 命中都是 `in-local-chiplet-cache`（24 ns），不产生跨 chiplet cache coherence 流量。Fig. 7 的实测数据直接支持：chiplet-aware 不加 shuffling 时，本地 NUMA 远端 chiplet cache fill 仅 162K 次（对比加 shuffling 的 630K 次），远端 NUMA 远端 chiplet cache fill 仅 111K 次（对比加 shuffling 的 562K 次）（Sec. 4.3, Fig. 7）。

去掉采样也减少了跨 chiplet 访问。采样的流程是：一个线程先随机扫一遍全数组来估计 key 分布，这个过程读到的数据可能散布在所有 chiplet 的内存和 cache 中。cut 直接按并行度均匀切分，各线程只需要知道自己负责哪一段，不需要预扫描。

### Comparison-Sort 的方法与 chiplet-aware 改动

Comparison-Sort 与 Radix-Sort 的核心区别在于：Radix-Sort 按二进制位逐轮分桶，多轮 pass；Comparison-Sort 只做一次 range partitioning（按 key 值范围分区），然后对每个分区内部做比较排序，最后合并。其流程是：生成直方图（每个线程扫描数据统计各值域范围的 key 数量）→ 根据直方图计算分区偏移量 → 每个分区内部用 SIMD 优化的 comb sort 原地排序 → 合并各分区。论文在此算法中保留了用 cache-resident index 记录 range partition 映射的做法，避免在排序阶段重复计算 partition function（Sec. 4.2）。

chiplet-aware 的改动与 Radix-Sort 同向：去掉随机采样（均匀切分代替）、线程绑定考虑 chiplet 边界、避免 NUMA 间 shuffling。主要的计算量集中在一次直方图/分区阶段，后续 in-cache 排序在各分区内独立完成。这使得 Comparison-Sort 受 chiplet-aware 优化的影响模式与 Radix-Sort 不同：直方图/分区阶段的收益与 Radix-Sort 类似（消除跨 chiplet 通信），但排序阶段属于分区内操作，天然在 chiplet 本地（Sec. 4.2）。

### 避免数据 Shuffling 的独立论证

论文把 shuffling 单独拿出来讨论，是因为传统 NUMA-aware 排序严重依赖 shuffling 来平衡负载。论文用实验证明这个做法在 chiplet 上适得其反（Sec. 4.3）：

- Tab. 1 给出 shuffling 占 NUMA-aware LSB Radix-Sort 总时间的比例：8 核时占 7%-20%，64 核时占 15%-32%。核数越多 shuffling 占比越大（Tab. 1）。
- Fig. 7 从 cache fill 来源的角度解释了原因：chiplet-aware + shuffling 时本地 NUMA 远端 chiplet cache fill 为 630K，远端 NUMA 远端 chiplet cache fill 为 562K；取消 shuffling 后分别降至 162K 和 111K。shuffling 把数据搬离了原本 chiplet 的 L3，导致后续访问全部变成跨 chiplet 命中。
- 有趣的是，NUMA-aware 取消 shuffling 后远端 chiplet cache fill 反而更高（521K），因为没有了 shuffling 的均衡作用，NUMA-aware 的访存集中在少数热点区域，引发了更严重的 cache 冲突。

### 额外实现优化
- 论文还在直方图和分区阶段加了 `_mm_prefetch(..., _MM_HINT_T0)` 预取，并通过拆分对齐/未对齐路径、减少分支深度和 `__builtin_expect` 改善分支预测（Sec. 4.5）。消融实验显示这两项对 Radix-Sort 加起来贡献约 15%（Fig. 15a），对 Comparison-Sort 基本无效（Fig. 15b）（Sec. 5.8）。

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
