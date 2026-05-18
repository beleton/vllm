# Optimizing Sorting for Chiplet-Based CPUs 解读

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

传统 NUMA-aware LSB Radix-Sort 的初始 range/radix 分区包含 6 步（Fig. 5, Sec. 4.1）：

1. **Key Sampling**：随机抽取部分 key，对样本排序后取分位点作为分区边界（delimiter）。`numa=2` 时取近似中位数，`numa=4` 时取近似 25%/50%/75% 分位点，目标是让各 NUMA node 的 key range 数据量接近均衡。
2. **初始数据按 NUMA/线程连续切分**：输入数组先按位置切成 NUMA 大块，再在每个 NUMA 内按线程切成连续片段。此时每个 NUMA 的数据仍可能同时包含多个目标 key range。
3. **局部直方图**：每个线程扫描自己的连续片段，按 `(目标 NUMA range, 当前 radix bucket)` 联合统计 tuple 数量。目标 NUMA range 由 Step 1 的 delimiter 判断，radix bucket 由当前处理的低位 bit 判断。
4. **分区写缓冲**：根据直方图和前缀和，各线程把本地数据写入当前 NUMA 的临时输出缓冲。缓冲内数据已按目标 NUMA range 和 radix bucket 分组，但仍位于源 NUMA。
5. **跨 NUMA Shuffling**：把每个源 NUMA 缓冲中属于目标 NUMA range 的分区复制到目标 NUMA。完成后，各 NUMA 持有互不重叠的全局 key range，NUMA 间顺序已经确定。
6. **后续本地 radix pass**：跨 NUMA shuffling 只在初始 range/radix 分区后执行一次。后续 pass 在各 NUMA 本地继续按剩余 bit 做 histogram、partition 和缓冲交换，不再跨 NUMA 搬迁；最终按 NUMA key range 顺序拼接。

**这套流程在 chiplet CPU 上的根本问题是 Step 5（Shuffling）**：shuffling 把数据从源 NUMA/chiplet 的缓冲复制到目标 NUMA/chiplet。完成搬迁后，后续本地 pass 会在目标 NUMA 上反复扫描这些数据；搬迁本身带来额外数据移动，并破坏源 chiplet 上已有的 L3 局部性。Tab. 1 显示 shuffling 占 NUMA-aware LSB Radix-Sort 总时间的 15%–32%（64 核时），核数越多占比越大。

## 观察：Chiplet CPU 上的内存层级需要重新建模

### 传统三层模型不适用

传统多核 CPU 上排序算法把访存分为三层：in-register（寄存器）、in-cache（L3）、out-of-cache（主存）。在这个模型下，"in-cache" 被假设为同质的——一次 L3 命中就是一次 L3 命中，不分来源。

Chiplet CPU 上这个假设不成立。AMD EPYC Milan 的 L3 被切成 8 个独立分片（每 chiplet 32 MB），chiplet 内访问延迟低、带宽高，跨 chiplet 访问须经 Infinity Fabric，延迟和带宽明显劣化（Fig. 3）。两者都叫 L3 命中，性能差数倍。论文因此把 in-cache 拆成两层（Sec. 3.1）：
- **in-local-chiplet-cache**：数据在本地 chiplet L3 内，低延迟、高带宽
- **in-remote-chiplet-cache**：数据在另一 chiplet L3 内，高延迟、低带宽

### 工作集大小与放置策略的关系：STREAM 基准的启发

论文在单路 AMD EPYC Milan 7713 上用 8 线程跑 STREAM TRIAD，按数组大小分三段对比两种线程放置策略（Fig. 3, Sec. 2.3）：

>  STREAM TRIAD 做 `a[i] = b[i] + q * c[i]`，同时存在读和写

- **Chiplet_Local**：8 线程全部放在同一 chiplet 的 8 个 core 上，共享 32 MB L3。
- **Chiplet_Mixed**：8 线程分散到 8 个 chiplet，每 chiplet 取一个 core，可用 256 MB 聚合 L3。

结果分三段：
1. **数组 < 32 MB**（单 chiplet L3 容量内）：Chiplet_Local 带宽更高。数据全部命中本地 L3，带宽可达数百 GB/s，远超 DRAM 带宽。
2. **数组 32–256 MB**（超出单 chiplet L3，在聚合 L3 内）：Chiplet_Local 带宽骤降，Chiplet_Mixed 保持平稳。Local 的 32 MB L3 已装不下，所有线程回退到主存；Mixed 的 256 MB 聚合 L3 仍能容纳。
3. **数组 > 256 MB**（超出聚合 L3）：两者趋同，全部回退到 DRAM。

这一结果说明：Chiplet CPU 上的最优放置策略不是固定的，而是取决于数据规模与 L3 容量之间的关系。核心权衡是"用延迟换容量"——单 chiplet 内 L3 延迟最低但容量有限，跨 chiplet 聚合 L3 容量更大但命中延迟更高。

## 方法：Chiplet-Aware 排序的四项改动

论文在两套排序实现（LSB Radix-Sort 和 Comparison-Sort）上做了四项 chiplet-aware 改动（Sec. 4）。

### 改动一：去掉采样（删除 NUMA-aware 流程的 Step 1）

不再通过随机采样猜测 key 分布，而是直接按并行度均匀切分输入数据。

**与 chiplet 架构的关联**：论文采用 shared-nothing、load-balanced 的输入分区，不再按 NUMA node 单独做采样、直方图生成和直方图聚合，避免后续 NUMA shuffling 的两阶段流程。

**负载均衡方式**：输入按线程/核心数切成等量 chunk，并在开始阶段绑定到对应 chiplet/core；Comparison-Sort 中数据负载在 Steps 1/2 完成均衡。直方图和前缀和用于计算 bucket 的输出偏移，保证各线程写入不同位置，不负责解决负载不均衡（Sec. 4.1, Sec. 4.2）。

### 改动二：线程绑定到 chiplet 内核心（替换 Step 2）

不再把数据段绑定到 NUMA 区域，而是把线程固定到具体 chiplet/core 上，数据分配按 chiplet 粒度组织。线程放置不固定追求“越局部越好”，而是由输入数据大小、单 chiplet L3 容量（32 MB）、聚合 L3 容量、所需线程/核心数和 chiplet 数共同决定。

工作集可放入单 chiplet L3，且所需线程数不超过单 chiplet 核心数时，调度器使用 Chiplet_Local，把任务放在尽可能少的 chiplet 内，优先获得本地 L3 的低延迟和高带宽。所需线程数超过单 chiplet 核心数，或工作集超过单 chiplet L3 但仍可放入多个 chiplet 的聚合 L3 时，调度器使用 Chiplet_Mixed，把任务分布到多个 chiplet，以换取更大的聚合 L3 容量。工作集超过聚合 L3 容量后，调度目标转为使用本地 main memory，避免昂贵的 remote NUMA 访问（Sec. 4.4）。

Fig. 13 展示了这个判据：32-bit Radix-Sort、8 threads 下，15 MB 输入使用 Chiplet_Local 更快；150 MB 及以上输入使用 Chiplet_Mixed 更快。

### 改动三：分区缓冲留在本地 chiplet L3（修改 Step 4）

论文对 Step 4 的优化表述是：将 partitioning buffers 主要保持在本地 chiplet cache 中。该表述不是硬件级锁定 L3 residency，而是通过线程绑定和分区写入位置降低缓冲被远端 chiplet 消费或搬迁的概率。

源码中 `lsb_64_chiplet.c` 的实现方式是：`cpu_bind(id)` 将排序线程绑定到按 `calculatePattern()` 计算出的核心；线程只扫描当前 `numa_node` 的连续片段，并将分区结果写入同一 `numa_node` 的 `keys_buf[numa_node]` / `rids_buf[numa_node]`；chiplet-aware LSB 路径不调用 `data_shuffling()`，`numa_shuffle_time` 保持为 0。写入线程所在核心的 cache hierarchy 会首先承接这些写入。项目 benchmark 对 `chiplet_lsb_64` / `chiplet_lsb_32` 默认传入 `numa=1`，后续 pass 在同一全局分区空间内继续处理对应缓冲，从而提高本地 chiplet L3 命中概率。

**与排序访存特征的关联**：分区阶段每个线程要反复读取直方图缓冲区（确定写入偏移量）和写入输出缓冲。如果这两块缓冲区在本地 chiplet L3 内，命中延迟低；如果被 coherence 拉到远端 chiplet L3，须经 Infinity Fabric，延迟明显更高。对于每轮 pass 都要重写整个输入的 Radix Sort，这个延迟差异会放大到整个运行时间。

### 改动四：去掉跨 NUMA Shuffling（删除 Step 5）

论文在 Sec. 4.3 和 Fig. 7 中比较了 shuffling 与 core affinity 对 remote chiplet cache 访问的影响。

**机制**：shuffling 在分区后跨 NUMA node 重新分布数据以平衡负载。该步骤会引入额外数据移动；在 chiplet-aware 配置中，它显著增加 remote chiplet cache fill，削弱本地 chiplet cache 的访问收益。

**数据证据（Fig. 7）**：
- chiplet-aware + shuffling：local NUMA remote chiplet cache fill = 630K，remote NUMA remote chiplet cache fill = 562K
- chiplet-aware 不加 shuffling：分别降至 162K 和 111K
- NUMA-aware 去掉 shuffling 后远端 chiplet cache fill 反而更高（521K），因为没有了 shuffling 的均衡作用，NUMA-aware 的访存集中在少数热点区域，引发了更严重的 cache 冲突

**与排序访存特征的关联**：Radix Sort 每轮 pass 都要扫描和重写整个输入，多轮 pass 间数据会被反复读取和重写。若分区与线程放置不能保持 chiplet 局部性，remote chiplet cache 访问和数据移动开销会在多轮 pass 中持续出现。

### 跨 Pass 数据局部性：四项改动如何协同
 
**全局有序依赖分桶与偏移计算。** NUMA-aware 方案先通过 sampling 得到 delimiter，再通过 shuffling 把属于不同 key range 的数据搬到目标 NUMA domain。论文的 chiplet-aware LSB Radix-Sort 删除这条路径：输入数据按并行度分区，线程从开始阶段绑定到具体 chiplet/core；每个线程对自己的 chunk 建 histogram，并根据 histogram 计算 partition offsets；分区缓冲主要保留在本地 chiplet cache。后续 pass 继续按 radix bucket 做本地 histogram、partition 和缓冲交换，排序顺序由 radix pass 的分桶顺序和偏移计算保证。

**删除 shuffling 的优化目标。** 论文原文将 shuffling 描述为 partitioning phase 之后用于平衡 NUMA domains 负载的数据重分布。chiplet-aware 方法不再依赖这个跨 NUMA 重分布步骤，目标是避免频繁数据传输，尤其是跨 chiplet 传输。Fig. 7 显示，chiplet-aware 配置下保留 shuffling 会显著增加 remote chiplet cache fill；删除 shuffling 后，本地 cache 访问收益更高。

**跨 pass 的数据局部性也并非由显式的"数据追随线程"机制实现，而是改动二~四叠加的效果：**

| 改动                    | 跨 pass 作用                                                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 改动二：线程绑定 chiplet      | Pass 间线程始终在同一 chiplet 核心上，上一轮写进本地 L3 的数据在下一轮仍可本地命中                                                                                     |
| 改动三：缓冲留在本地 chiplet L3 | 对比 NUMA-aware 的 `numa_alloc_onnode()` 把缓冲分配在主存，chiplet-aware 通过调度策略（Chiplet_Local/Chiplet_Mixed）让工作集匹配 chiplet L3 容量，输出优先留在 L3 而非写穿到主存 |
| 改动四：去掉 shuffling      | 不在 pass 间主动搬迁数据，让 cache coherence 协议自然处理——谁用谁拿，不用不搬                                                                                    |

**对比两种策略在 pass 间的行为：**

```
NUMA-aware（有 shuffling）:
Pass 1: 线程写 → NUMA node 0 主存
         ↓ shuffling 主动搬迁
Pass 2: 数据已被搬去 NUMA node 1 主存 → 线程跨 NUMA 取数 → 全变成远端访问

Chiplet-aware（无 shuffling）:
Pass 1: 线程0(Ch0) 写数据 → 留在 Ch0 L3
         ↓ 不做搬迁
Pass 2: 线程0(Ch0) 仍绑在 Ch0 → 读到的数据可能在 Ch0 L3 或其他 chiplet L3
        但数据至少还在某个 chiplet 的 L3 内，没有被人为搬去远端主存
```

Shuffling 的代价是"主动把数据搬到远端，而后所有人访问都是远端命中"。去掉 shuffling 后，cache coherence 是被动的——跨 chiplet 访问仅在确实需要时才发生，自然情况下远端比例远低于主动搬迁。Fig. 7 的数据证实了这一点：chiplet-aware 无 shuffling 比 chiplet-aware + shuffling 远端 chiplet cache fill 降低了 75–80%。

### 额外实现优化（Sec. 4.5）

- 在直方图和分区阶段加 `_mm_prefetch(..., _MM_HINT_T0)` 预取
- 通过拆分对齐/未对齐路径、减少分支深度和 `__builtin_expect` 改善分支预测
- 消融实验显示这两项对 Radix-Sort 合计贡献约 15%（Fig. 15a），对 Comparison-Sort 基本无效（Fig. 15b）

## 代码层面实现：NUMA-aware LSB Radix-Sort 的流程与关键机制

本节基于论文源码 `lsb_64.c` 说明 NUMA-aware 排序的实现细节，包括线程任务划分、第一趟 NUMA 感知的直方图与分区、跨 NUMA 数据洗牌，以及后续纯本地基数排序的完整流程。
### 线程任务划分与两阶段同步

**线程分布**：`threads` 个线程均匀分布在 `numa` 个 NUMA 节点之间，每 NUMA 节点 `threads_per_numa = threads / numa` 个线程。`schedule_threads()` 负责给每个线程分配物理 CPU 和 NUMA 节点：优先使用系统上实际存在的 CPU，将同 NUMA 的线程尽可能放在该 NUMA 内的不同核心上；若请求的线程数超过系统可用核心数，则退化为顺序分配。

**数据切分**：每个线程处理其所在 NUMA 节点数据的一段连续子区间。子区间大小 `size = numa_size / threads_per_numa`，偏移量 `offset = size * numa_local_id`（末线程承接余数）。线程通过 `cpu_bind()` 和 `memory_bind()` 分别绑定到指定 CPU 和 NUMA 内存节点。

**两阶段同步**：

```
全局屏障 (global_barrier):  采样阶段 8 趟 × 3 个屏障/趟 = 24 个全局屏障
本地屏障 (local_barrier):   每个 NUMA 节点独立，仅该 NUMA 内的 threads_per_numa 线程参与
```

采样阶段所有线程共同参与，因此使用全局屏障。NUMA 洗牌完成后，后续各 pass 完全在 NUMA 本地进行，同步降级为本地屏障——每个 NUMA 节点独立推进，不与其他 NUMA 节点同步。

### 整体流程

```
输入数据（已按位置均分到 N 个 NUMA 节点，值分布随机）
  │
  ├─ Phase A: 分配与初始化
  │   每个 NUMA 节点分配 capacity = numa_size × 1.1 的输出缓冲区
  │
  ├─ Phase B: 采样
  │   所有线程全局协作，随机抽取 ~0.1% 样本，完整排序后提取 NUMA 分隔符
  │
  ├─ Phase C: 第一趟 ── NUMA 感知的直方图 (histogram_numa_{2,4,8})
  │   同时提取 key 的低 radix_bits 位（基数分桶）+ 判定 key 归属的 NUMA 节点
  │
  ├─ Phase D: 第一趟 ── NUMA 感知的分区 (partition_numa_{2,4,8})
  │   每个线程将数据按合并的 partition ID 写入当前 NUMA 的输出缓冲区
  │
  ├─ Phase E: 跨 NUMA 数据洗牌 (data_shuffling)
  │   每个 NUMA 节点收集属于自己的数据（可能来自所有远程 NUMA 节点）
  │   洗牌后: NUMA 间值有序，每个 NUMA 内部按低 radix_bits 位已部分有序
  │
  └─ Phase F: 后续 pass ── 纯本地 LSB 基数排序
      使用普通 histogram() / partition()，无 NUMA 逻辑，仅本地屏障
      每 pass 排序 radix_bits 个更高位，pass 间交换 keys_a / keys_b
```

### Phase B：采样与 delimiter 提取

采样仅在 NUMA 数大于 1 时执行。样本量取全量数据的 0.1%（上限 100K 条），使用 `numa_alloc_interleaved` 分配，物理内存跨所有 NUMA 节点交错分布，确保各线程访问延迟均衡。

每个线程从自己的数据片段中随机抽取等量样本。所有线程协作完成样本的完整 LSB 基数排序：通过 8 趟（每趟处理 8 位）将样本完整排序，每趟包含 3 个全局屏障同步点（直方图完成 → 分区完成 → 冲刷完成），共 24 个全局屏障。

排序后调用 `extract_delimiters()`：将排序后的样本按等间距取 `numa - 1` 个分位点作为分隔符。分隔符之间即为各 NUMA 节点应持有的 key 范围边界。例如 `numa=2` 时取近似中位数，`numa=4` 时取近似 25%/50%/75% 分位点。

### Phase C：NUMA 感知的直方图 —— 同时完成基数分桶与 NUMA 归属判定

第一趟直方图同时做两件事：**提取 key 的低 radix_bits 位**（基数分桶）和**判断 key 属于哪个 NUMA 节点**（值是大于还是小于采样得出的分隔符）。两者被编码进一个合并的 partition ID：

```
partition_id = (key 的低 radix_bits 位) | (目标 NUMA 号 << radix_bits)

例如 numa=4, radix_bits=10（第一趟共 12 位）:
┌──────────────┬──────────────────┐
│ NUMA 目标    │   radix 桶号     │
│  (2 bits)    │   (10 bits)      │
└──────────────┴──────────────────┘
 bits 11-10       bits 9-0
```

代码按 NUMA 数量分为三个版本：`histogram_numa_2`（1 bit NUMA + 1 次分隔符比较）、`histogram_numa_4`（2 bits NUMA + 2 次分隔符比较）、`histogram_numa_8`（3 bits NUMA + 3 次分隔符比较）。具体做法是：对每个 key，先提取其低 radix_bits 位得到 radix 桶号，然后与 `numa - 1` 个分隔符做多级比较确定 NUMA 归属，最后将 NUMA 位左移 radix_bits 位与 radix 桶号合并。第一趟直方图总桶数为 `(1 << radix_bits) × numa`。

每条 key 的 NUMA 归属信息同时被累计到一个辅助数组 `numa_local_count` 中（每个线程记录自己片段内有多少数据属于各目标 NUMA），供后续洗牌计算传输量使用。

### Phase D：NUMA 感知的分区

分区函数同样处理合并的 partition ID。每个 partition 拥有一个独立的小型输出缓冲区（128 字节 = 双 cache line），缓冲区末尾槽位用作写指针追踪填充进度。处理每对 key+rid 时，按其 partition ID 定位对应缓冲区，写入键值对。当缓冲区接近填满时（填满 14 个槽位），整条 cache line 通过非时间流存储写出到该 NUMA 的输出数组（`keys_buf` / `rids_buf`），绕开 CPU 缓存以避免污染 L3。各线程完成自身片段后通过本地屏障同步，最后将各缓冲区中的残留项冲刷到输出位置。

第一趟分区的关键结果是：每个 NUMA 节点的输出缓冲区中，数据已按 `(目标 NUMA, radix 桶号)` 分组排列。例如 `numa=4, radix_bits=10` 时，NUMA 0 的输出缓冲区包含 4096 个桶，前 1024 个桶属于目标 NUMA 0，接下来 1024 个属于目标 NUMA 1，以此类推。但物理上所有数据仍位于源 NUMA——NUMA 1 的目标数据可能还在 NUMA 0 的缓冲区里。

### Phase E：跨 NUMA 数据洗牌 (data_shuffling) —— Chiplet 感知的核心切入点

这是 NUMA-aware 排序中最关键的一步，也是 chiplet-aware 方案明确删除的步骤（改动四）。`data_shuffling()` 在第一趟分区之后执行，完成数据的物理搬迁。

**输入状态**：第一趟分区后，每个 NUMA 节点的输出缓冲区 `keys_buf[numa_node]` / `rids_buf[numa_node]` 中，数据已按 `(目标 NUMA, radix 桶号)` 排列。但物理上仍位于源 NUMA——属于 NUMA 1 的 key 可能还在 NUMA 0 的缓冲区里。

**洗牌过程**（对每个目标 NUMA 节点）：

1. **计算传输矩阵** `transfer[src][dst]`：汇总所有线程的 `numa_local_count`，构建各源 NUMA 到各目标 NUMA 的数据量矩阵。
2. **验证容量**：洗牌后每个 NUMA 的总数据量必须 ≤ `numa_size × 1.1`（初始化时的 fudge 因子）。若超出，打印警告。
3. **逐 radix 分区处理**：对每个 radix 桶号 `i`，计算合并 partition ID `j = i | (numa_node << radix_bits)`，统计所有源 NUMA 为该分区贡献的数据量，随机化源 NUMA 的读取顺序（见下文），然后按随机顺序将数据从各源 NUMA 复制到目标 NUMA。
4. **全局同步**：洗牌完成后通过全局屏障同步所有线程。

**NUMA 顺序随机化的目的**：对于每个 radix 分区，每个线程使用独立种子随机化其读取源 NUMA 的顺序（Fisher-Yates 洗牌）。这样不同线程的远端访问交错模式各不相同，避免了所有线程以相同步幅同时访问同一远端 NUMA 节点造成的即时热点和互连拥塞。

**洗牌后的状态**：

```
NUMA 0: 持有 key ≤ delimiter[0] 的所有数据，已按低 radix_bits 位部分有序
NUMA 1: 持有 delimiter[0] < key ≤ delimiter[1] 的所有数据，已按低 radix_bits 位部分有序
NUMA 2: 持有 delimiter[1] < key ≤ delimiter[2] 的所有数据，已按低 radix_bits 位部分有序
...
```

NUMA 之间值大小有序（NUMA 0 所有 key < NUMA 1 所有 key < ...），后续无需再访问远端 NUMA。

### Phase F：后续纯本地基数排序 (pass 2, 3, ...)

洗牌后各 NUMA 独立运行，不再需要全局同步。每个 pass 的流程相同：直方图 → 本地屏障 → 计算偏移量 → 分区 → 本地屏障 → 冲刷残留项 → 交换源/目标缓冲区。所用的是普通直方图和分区函数（`histogram()` / `partition()`），不含分隔符比较，无 NUMA 逻辑。每 pass 处理 `radix_bits` 个更高位，通过乒乓缓冲将前一趟输出作为下一趟输入。

对比第一趟与后续趟的关键差异：

| 维度 | 第一趟 (NUMA-aware) | 后续趟 (纯本地) |
|------|-------------------|----------------|
| 直方图函数 | `histogram_numa_{2,4,8}()` | `histogram()` |
| 分区函数 | `partition_numa_{2,4,8}()` | `partition()` |
| 直方图桶数 | `(1 << radix_bits) × numa` | `1 << radix_bits` |
| 分隔符比较 | 有 | 无（不需要 NUMA 判定） |
| 屏障类型 | 本地屏障 | 本地屏障 |
| 跨 NUMA 访存 | 洗牌时有 | 无 |

通过乒乓缓冲（`keys_a` / `keys_b` 交替），每趟 pass 的输出成为下一趟的输入。

### 非时间流存储与软件输出缓冲

分区阶段大量使用非时间流存储，将写满一条 cache line 的输出缓冲区直接写入内存而不经 CPU 缓存。这对 Radix Sort 尤为重要——分区输出是下一趟的输入，当前并不消费，写入缓存只会驱逐有用数据。软件管理的输出缓冲（每条分区 128 字节，填满后整条写出）相比逐元素散写减少了写合并缓冲区压力，也使对齐的流存储成为可能。

## 代码层面实现：Chiplet-aware LSB Radix-Sort 的改动与流程

本节基于论文源码 `lsb_64_chiplet.c`，在与上一节 NUMA-aware 流程对比的基础上，说明 chiplet-aware 版本做了哪些删除、替换和新增，以及这些改动如何映射到论文的四项改动。

### 线程绑定

**NUMA-aware（lsb_64.c）**：`schedule_threads()` 按 `t % numa` 轮询将线程均分到各 NUMA 节点，每个 NUMA 内取该节点上空闲的低编号 CPU。

**Chiplet-aware（lsb_64_chiplet.c）**：`cpu_bind()` 通过 `calculatePattern()` 将连续的线程 ID 重映射，使相邻线程落在不同 chiplet 上（每 chiplet 取一个空闲的低编号核心），从而让线程分散到多个 chiplet 以扩大聚合 L3 容量。内存绑定简化为 `set_mempolicy(MPOL_BIND)`，NUMA 归属由映射后的物理核心位置自然决定。

### 整体流程

```
输入数据（numa=1，单 NUMA 域，值分布随机）
  │
  ├─ Phase A: 分配与初始化
  │   使用 mmap + 2MB Huge Pages 分配输出缓冲区
  │
  ├─ Phase B: 采样 ── 被删除
  │   分配 delimiter 内存并置零，不抽取样本，不排序
  │
  ├─ Phase C: 第一趟 ── 普通直方图 histogram()（加预取）
  │   纯基数分桶，无 NUMA 归属判定，无分隔符比较
  │
  ├─ Phase D: 第一趟 ── 普通分区 partition()（加预取 + __builtin_expect）
  │   纯基数分区，通过线程绑定、删除 shuffling 等方式让数据尽可能留在核心所在 chiplet 的 L3
  │
  ├─ Phase E: 跨 NUMA 数据洗牌 ── 被删除
  │   不调用 data_shuffling()，numa_shuffle_time 恒为 0
  │
  └─ Phase F: 后续 pass ── 纯本地 LSB 基数排序
      每 pass 继续 histogram → barrier → partition → barrier → finalize
      数据可能在本 chiplet L3 或远端 chiplet L3，由 cache coherence 自然决定
```

### 改动一（删除采样）：skip sampling

`sort_thread()` 第 1406-1411 行分配了 `delimiter` 数组并以 `~0` 初始化末尾，但**从不填充实际分隔符值**。`a->sample_time` 接近于零（仅包含内存分配开销）。这意味着不再通过采样猜测 key 分布的 NUMA 边界——chiplet-aware 方案中数据不存在"归属哪个 NUMA"的概念，所有数据在同一 NUMA 地址空间内，数据放置由分区写入位置决定，负载均衡通过等量数据切分和直方图偏移计算实现。

### 改动二（线程绑定）：calculatePattern 跨 chiplet 交错

见上节"线程绑定"的详细说明。

### 改动三（分区缓冲留本地 L3）：大页 + 预取 + 分支预测

除了线程交错的调度策略，chiplet 版本在实现层面新增了三项措施来提升本地 chiplet L3 的利用率：

**2MB Huge Pages**：`mamalloc()` 通过 `mmap(MAP_HUGETLB)` 分配 2MB 大页，取代 `lsb_64.c` 的 `posix_memalign`。大页减少 TLB miss，尤其在多 pass 反复扫描大数组时效果明显。

**`_mm_prefetch` 预取**：直方图函数中在加载当前 128 字节前预取 128 字节后的下一个 cache line（`_mm_prefetch(keys+128, _MM_HINT_T0)`）；分区函数中预取 keys 和 rids 的 `[+32]` 位置（即下一个 4 元素组）。这些预取提示将数据提前拉入本地 chiplet L3，减少后续使用的等待延迟。

**`__builtin_expect` 分支预测**：在分区函数的主循环中，对对齐检查、未对齐末尾处理、缓冲区满判断等低频路径使用 `__builtin_expect(cond, 0)` 提示编译器生成有利于高频路径的分支预测布局，减少流水线停顿。

### 改动四（删除 shuffling）：不搬迁数据

后续 pass 直接进入普通直方图/分区循环（第 1459-1493 行）。第一趟分区后不做物理搬迁，让数据尽可能留在核心所在 chiplet 的 L3。后续 pass 的跨 chiplet 访问由 cache coherence 协议自然处理，其他 chiplet 需要时通过 Infinity Fabric 读取。论文 Fig. 7 的数据显示这种被动方式产生的远端 chiplet cache fill 仅为主动 shuffling 的 20–25%（162K/111K vs 630K/562K）。

### 改动之外的差异

| 维度                      | NUMA-aware                 | chiplet-aware                        |
| ----------------------- | -------------------------- | ------------------------------------ |
| NUMA 参数                 | numa >= 1，支持多 NUMA         | 通常 numa=1，单 NUMA 域                   |
| 第一趟直方图函数                | `histogram_numa_{2,4,8}()` | `histogram()` 纯基数                    |
| 第一趟分区函数                 | `partition_numa_{2,4,8}()` | `partition()` 纯基数                    |
| 全局屏障数                   | 24（采样 8 趟）                 | 1（仅 sample_barrier 用于全局同步点）          |
| `distribute_bits` limit | {12, 23, 34, 45, 56, 67}   | 大数据集（≥100M）时提高为 {14, 27, 40, 53, 66} |
| 内存释放                    | `free()`                   | `munmap()`（对应大页 mmap）                |

### 两种策略的数据访问模式对比

```
NUMA-aware (lsb_64.c):
  Pass 1:  Thread → 本地 NUMA 主存读 → NUMA 判定 + 分桶 → 本地缓冲写
           → data_shuffling: 把属于远程 NUMA 的数据搬到目标 NUMA 主存
  Pass 2~N: Thread → 本地 NUMA 主存读/写（零远端访问）

  代价: 一次全局搬迁（占总时间 15-32%）

Chiplet-aware (lsb_64_chiplet.c):
  Pass 1:  Thread → 全局数组索引段读 → 分桶 → 写入输出缓冲（尽可能保留本地 L3 局部性）
  Pass 2~N: Thread → 同一索引段，数据部分在本地 L3、部分在远端 chiplet L3
            cache coherence 被动处理跨 chiplet 访问

  代价: 每 pass 可能有少量远端 L3 访问，但无一次性搬迁
```

两种策略的选择取决于工作集大小与单 chiplet L3 容量（32MB）的关系：工作集小于 32MB 时应使用 Chiplet_Local（线程集中），工作集超出单 chiplet L3 但在聚合容量内（256MB）时 Chiplet_Mixed 更优（论文 Fig. 13）。

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
