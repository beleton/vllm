# Load and MLP-Aware Thread Orchestration for Recommendation Systems Inference on CPUs 解读

**概述**：论文面向 CPU 上的深度推荐模型推理，研究 embedding 阶段在多 CCD AMD EPYC 处理器上的并行调度问题。作者指出现有 Hierarchical Threading 虽然把每张 embedding table 分配到一个 CCD，已经能减少跨 CCD 的缓存复制和 L3 竞争，但仍存在 CCD 间负载不均、每张表固定使用 8 个核心导致核心利用率低、以及 MLP 资源利用不均的问题。论文提出 `Balance` 调度器，用表负载和 MLP 需求两个指标指导核心分组、任务排队和 work stealing，在 96 核 AMD EPYC 上相对 HT 最高取得 `1.67x` 扩展性提升（摘要；Fig. 16-Fig. 24）。

本文不是 HNSW 论文。它放在 HNSW 目录下的价值是作为 CPU 多 CCD 线程编排的强相关参考：它说明在只读或读多写少、随机访存、表/任务异构的 workload 中，调度器不能只按任务数量均衡，还需要同时考虑每个任务的工作量、缓存复用和可并发未命中请求压力。对当前 HNSW 研究而言，它更适合作为“CCD 调度方法与边界”的参照，而不是可直接迁移的 HNSW 算法设计。

## 论文基本信息

- 论文：`Load and MLP-Aware Thread Orchestration for Recommendation Systems Inference on CPUs`
- 作者：Rishabh Jain、Teyuh Chou、Onur Kayiran、John Kalamatianos、Gabriel H. Loh、Mahmut T. Kandemir、Chita R. Das
- 会议：ASPLOS 2025
- 研究对象：CPU 上的 DLRM 推理，重点是 embedding bag operator 的线程调度
- 目标平台：单路 96 核 AMD 4th Generation EPYC，12 个 CCD，每 CCD 8 核、32 MB L3（Table 2）
- 原始资料：`wiki/原始资料/papers/RM/Load and MLP-Aware Thread Orchestration forRecommendation Systems Inference on CPUs.pdf`

## 背景

### DLRM 推理流程

DLRM（Deep Learning Recommendation Model，深度学习推荐模型）用于广告、商品推荐、内容排序等场景。它同时处理两类特征：

- **连续特征**：数值型输入，例如价格、年龄、统计计数。
- **类别特征**：离散 ID 输入，例如用户 ID、商品 ID、广告 ID、频道 ID。

DLRM 的推理路径包含四个阶段（Fig. 2）：

1. **Bottom MLP**：对连续特征做多层感知机计算。MLP 是由多层矩阵乘法和非线性函数组成的神经网络模块，主要消耗计算资源。
2. **Embedding**：对类别特征查 embedding table。Embedding table 是一个大矩阵，每一行对应一个离散 ID 的向量表示。
3. **Feature interaction**：把 dense 特征和 sparse embedding 特征做交互。
4. **Top MLP**：基于交互后的特征输出推荐分数。

论文关注第二阶段，即 embedding。原因是该阶段访问大量 embedding table 行，访问位置由输入 ID 决定，随机性强，通常成为 DLRM 推理中最耗时、最受内存带宽限制的阶段（Sec. 1；Sec. 2.1）。

### Embedding bag 计算

Embedding bag 是推荐模型中常见的查表聚合算子。一次 batch 包含多个用户样本；每个样本会在多张表中查多个 ID；每个 ID 对应一行 embedding 向量；同一字段内的多行向量会被累加或聚合。

论文用 Algorithm 1 概括 embedding 阶段的五层循环：

```text
batch
  -> table
    -> sample in batch
      -> pooling factor
        -> embedding dimension
```

其中 `pooling factor` 表示一次样本在某张表中要查多少个 ID。每次查表需要从 embedding table 中加载一行向量，再和累加器相加。若 embedding 维度是 128，单行向量按 FP32 计算就是 512 字节。论文举例说明：100 张表、batch size 1024、pooling factor 150 时，一个 batch 要加载约 7.5 GB 数据（Sec. 2.1）。

这个算子的核心特征是：

- 读多、随机、数据量大。
- 算术操作只是向量加法，计算强度低。
- 硬件预取很难预测下一行 ID。
- 性能取决于 CPU 能否持续从 L3 和 DRAM 向核心供应数据。

### CCD、L3 和 MLP

论文目标 CPU 是 AMD EPYC 9654。一个 CCD（Core Complex Die）包含 8 个核心，共享一个 32 MB L3。CPU 总共有 12 个 CCD、96 个核心、12 通道 DDR5，内存带宽约 460 GB/s（Table 2）。

在这种架构中，同一 CCD 内的核心共享本地 L3；不同 CCD 的 L3 是分片的。若同一张表的热点行被多个 CCD 读取，数据可能在多个 L3 中重复填充，浪费容量和带宽。若多个高随机访存任务集中在一个 CCD，它们会共同争抢该 CCD 的 L3 未命中处理资源。

论文反复提到 MLP（Memory-Level Parallelism，内存级并行）。它指处理器在等待一次内存访问完成时，同时发起和跟踪其他未完成内存访问的能力。随机访存 workload 中，单次 DRAM 访问延迟较高，如果硬件能同时挂起很多未命中请求，就能用并发度隐藏部分延迟。

MSHR（Miss Status Holding Register，未命中状态保持寄存器）是实现 MLP 的关键结构之一。缓存未命中后，MSHR 记录该请求的地址、状态和等待者。MSHR 数量有限；若随机访问产生的未命中太多，MSHR 被占满，核心就无法继续发起新的 miss，出现 backpressure。论文把 L3 MSHR 视为 CCD 级 MLP 资源（Sec. 2.3；Sec. 2.6）。

## 研究问题

论文研究的问题是：在多 CCD CPU 上执行 DLRM embedding 推理时，如何调度不同 embedding table 的计算，使核心、L3 和 MLP 资源都被更充分利用。

这个问题有三个具体来源。

### 现有并行方式的扩展性不足

论文比较三类已有 embedding 并行方式（Fig. 1；Fig. 5）：

| 策略 | 含义 | 优点 | 问题 |
|---|---|---|---|
| Batch Threading (BT) | 一次只处理一张表，把该表在 batch 内的工作切给整个 SoC 的所有核心 | 单张表可用全部片上缓存容量 | 线程同步频繁；同一表数据在多个 L1/L2/L3 中复制；轻量任务线程开销高 |
| Table Threading (TT) | 每张表分配给一个核心，多个表并行处理 | 线程粒度较粗；私有缓存复制更少 | 每个核心处理的表工作量差异大；多张表可能冲击同一个 L3 |
| Hierarchical Threading (HT) | 每张表分配给一个 CCD，再在 CCD 内用 8 个核心并行处理 | 减少跨 CCD 的同表数据复制；降低 L3 竞争；比 BT/TT 更快 | 仍有 CCD 间负载不均、每表核心数过度分配、MLP 利用不均 |

HT 是论文采用的最强 baseline。它比 BT 和 TT 最高分别快 `5.2x` 和 `1.56x`，但在生产异构数据集上，96 核只达到约 `13x` 单核加速，远低于理想 `96x`（Fig. 1；Sec. 2.2）。

### 生产表高度异构

论文使用 Meta 的生产 DLRM trace，模型约 110 GB，包含 129 张 embedding table（Sec. 2.1）。这些表在三方面差异很大：

1. **表大小差异**：表行数从 100 到 500 万不等。小表可完全放入 L2 或 L3，大表远超缓存容量（Fig. 3a）。
2. **访问热度差异**：不同表的 unique index 比例和 top 10% index 覆盖率差异明显。说明有些表访问集中、缓存复用强，有些表访问接近随机、缓存复用弱（Fig. 3b）。
3. **pooling factor 差异**：最大 pooling factor 从 0 到 164 不等。pooling factor 为 0 表示该表在该 batch 中没有实际查表工作（Fig. 3c）。

这些差异最终表现为每张表执行时间差异可达 `37.8x`（Fig. 4）。若 HT 只是按 round-robin 把表分到 CCD 队列，它均衡的是表数量，不是执行时间，因此多个重表可能落到同一 CCD，其他 CCD 过早空闲（Fig. 6；Sec. 2.3）。

### 固定 8 核处理每张表并不合理

HT 对每张表固定使用一个 CCD 的全部 8 个核心。但论文测得：129 张表中只有 5 张表能从 8 核中获益，35 张表适合 6 核，其余 89 张表在 2 核或 4 核后就基本不再扩展（Fig. 11；Sec. 2.5）。

这说明对多数表使用 8 核会造成两类浪费：

- 轻量表每个线程分到的工作太少，线程调度、同步、fork/join 开销占比升高。
- 一张表占满 CCD 后，其他可并行表不能利用该 CCD 剩余核心。

论文进一步说明，表负载越高，越可能从更多核心中获益。TL 大的表适合 4 核或 8 核；TL 小的表通常 2 核就足够（Fig. 12；Table 1）。

### MLP 资源也会不均衡

不同表的访问复用不同，对 MLP 的需求也不同。低复用表会产生大量 L3 miss，需要更多 MSHR 和更高 DRAM 请求并发；高复用表更多命中 L3，占用 MLP 少，但占用 L3 容量更有价值。

论文用一个微基准隔离核心数量和缓存/MLP 资源：固定 8 个线程和 8 个核心，但把线程分布到 1、2、4、8 个 CCD 上（Fig. 13a）。这样核心数不变，但可用 L3 容量和 MSHR 数量随 CCD 数增加。结果显示，低复用随机访问表在多 CCD 分布时最高可达到 `7.7x` 加速；高复用表 2 核后就趋于饱和（Fig. 13b；Sec. 2.6）。

这个结果表明：对随机访问表，瓶颈不是算力，而是单 CCD 可同时处理的未命中请求数和数据供给速率。若多个高 MLP 需求表同时落到一个 CCD，它们会抢 MSHR；若所有高复用表同时执行，系统又可能没有充分使用 MLP 和内存带宽。

## 关键指标

### TL：表负载

TL（Table Load，表负载）用于估计某张表的执行时间。它由两部分相乘得到（Sec. 2.4）：

```text
TL = APF * AMAT
```

其中：

- `APF` 是 average pooling factor，表示该表在 batch 内平均要查多少行。
- `AMAT` 是 average memory access time，表示该表行访问的平均内存访问时间。

论文没有只用 pooling factor 估计工作量，因为 pooling factor 只能说明查表次数，不能说明每次访问是 L3 命中还是 DRAM 访问。两个表可能查表次数相近，但一个高度复用、一个近似随机，执行时间会明显不同。

AMAT 的估计过程如下（Fig. 7；Sec. 2.4）：

1. 记录第一批输入中每张表被访问的 row index 序列，把 row index 当作内存地址代理。
2. 计算 reuse distance，即同一 row 两次访问之间经过了多少个不同 row。
3. 用一个 fully-associative、32 MB、LRU 替换策略的 L3 cache 模型判断 row 访问是 hit 还是 miss。
4. 假设 embedding 维度为 128，每行 512 字节；L3 hit 代价为 30 个时间单位，L3 miss 代价为 450 个时间单位，on-chip/off-chip 访问代价比为 1:15。

例如某表 APF 为 81.3，模型估计 L3 hit rate 为 67.4%，则：

```text
AMAT = (67.4 * 1 + (100 - 67.4) * 15) / 100 = 5.57
TL = 81.3 * 5.57 = 452.7
```

论文用实测表执行时间验证 TL，发现 TL 与 measured execution time 相关性高（Fig. 8b）。相比只用 APF，TL 能避免把“查得多但复用好”和“查得少但 miss 多”的表误判为同一类负载（Fig. 8c）。

作者还把表按 TL 分为三类（Fig. 9）：

| 类别 | 条件 | 含义 |
|---|---|---|
| Heavy weight | `TL >= 200` | 工作量大，通常能从更多核心中获益 |
| Medium weight | `35 < TL < 200` | 中等工作量 |
| Light weight | `TL <= 35` | 工作量小，过多核心会放大调度开销 |

论文认为 TL 不需要频繁重建。表的执行时间、APF 和 AMAT 在 120 个 batch 中变化很小；若 embedding table 每隔数小时更新一次，TL 可用新 batch 的 row 访问模式快速重新生成（Fig. 10；Sec. 2.4）。

### WMLP：加权 MLP 需求

WMLP（Weighted MLP，加权内存级并行需求）用于估计某张表对 MLP 资源的压力（Fig. 14；Sec. 2.6）。它的作用不是预测表执行时间，而是帮助调度器判断哪些表不应同时放到同一个 CCD。

论文提取文本中出现一个表述歧义：公式句写作 TL 与 L3 cache hit ratio 的乘积，但后文立即说明 high pooling factor 和 high cache miss rate 会产生 large WMLP，低 pooling factor 和 low cache miss rate 会产生 small WMLP。按上下文，WMLP 应表达的是“工作量和 L3 未命中压力共同决定的 MLP 需求”，而不是单纯的 L3 命中率。本文在后续解读中按“TL 结合 L3 miss 压力”理解 WMLP。

WMLP 在生产 trace 中从 0 到 46600 变化很大（Fig. 14）。调度器据此把表分为：

- **HWMLP 表**：高 MLP 需求表，通常低复用、miss 多、需要更多未完成请求并发。
- **LWMLP 表**：低 MLP 需求表，通常复用好或工作量小，对 MSHR 和内存带宽压力较低。

该分类的调度意义是：每个 CCD 同时最多放一个 HWMLP 表，再搭配一个或多个 LWMLP 表，以避免多个高 miss 压力任务挤在同一个 CCD。

## HT 的实现与低效来源

HT 使用 PyTorch 的 `torch.jit._fork` 和 `torch.jit._wait` 在 Python 顶层为每张表生成任务，再由 PyTorch 后端 C++ thread pool 执行。每个 CCD 绑定一个 task queue；表任务按 round-robin 注入不同 CCD 队列；某个 CCD 取到一张表后，在该 CCD 的 8 个核心上用 OpenMP 执行 embedding bag operator（Fig. 6；Sec. 2.3）。

该实现的优点是降低跨 CCD 的同表数据复制。与 TT 相比，HT 让同一张表在一个 CCD 内处理，多个核心共享该 CCD 的 L3。与 BT 相比，HT 不会把同一张表扩散到整个 SoC 的所有 L3 中。

但 HT 的调度粒度仍过粗，主要有三类低效。

### CCD 间负载不均

HT round-robin 分发表任务，只保证每个 CCD 队列中的表数量接近。生产表执行时间差异可达 `37.8x`，因此表数量均衡不能代表工作量均衡。论文称在最差分布下，负载最高 CCD 的工作量可比最低 CCD 高 `37.8x`，导致大量核心等待 straggler CCD（Sec. 2.4）。

### 每表固定 8 核导致过度分配

HT 默认每张表使用整 CCD 的 8 核。对于轻量表，8 核不会带来线性加速，反而使每线程工作变小、同步开销占比增大。Fig. 11 显示绝大多数表不需要 8 核。Fig. 12 和 Table 1 进一步给出 TL 与核心数扩展性的关系：TL 高的表 4 核和 8 核收益较明显，TL 低的表 2 核即可达到主要收益。

### MLP 利用不均

HT 不区分高 miss 表和高复用表。若多个高 miss 表同时落在一个 CCD，MSHR 和内存请求通路容易过载；若多个高复用表同时执行，缓存容量被使用，但 MLP 和内存带宽可能没有被充分使用。Fig. 13 的微基准说明，在低复用表上增加 CCD 数能显著提升吞吐，因为每个线程获得更多 L3 容量和 MSHR；高复用表不需要这么多 MLP 资源（Sec. 2.6）。

## Balance 设计

Balance 在 HT 的基础上修改 thread pool 调度。它的核心设计包括三部分（Fig. 15；Sec. 3）：

1. low-core grouping：每张表不再固定占用 8 核，而是使用 2 核或 4 核。
2. MLP-aware scheduler：把表按 WMLP 分成高/低 MLP 需求队列，并按 WMLP 降序调度。
3. work stealing：当一个队列空而另一个队列仍有任务时，从另一队列窃取任务，减少长尾等待。

### 低核心分组

低核心分组的目标是避免“轻量表占满整个 CCD”。Balance 静态选择 2 核或 4 核作为每张表的并行核心数。一个 8 核 CCD 因此最多可同时处理：

- 4 张表：每张表 2 核；
- 2 张表：每张表 4 核。

论文没有实现对每张表都动态选择 2/4/6/8 核，而是采用统一的 2-core grouping 或 4-core grouping 做评估。这样设计复杂度低，也减少调度器需要处理的 worker group 类型。

低核心分组的收益来自两点：

- 对轻量表，每个线程获得更多连续工作，thread overhead 更容易被摊销。
- 一个 CCD 可以同时执行多张表，提高核心利用率和表级并行度。

但它也有风险：2-core grouping 同时处理的表更多，可能导致一个 CCD 中 L3 miss 更高，数据 fabric 流量上升。论文实验显示 4-core grouping 更稳定，2-core grouping 在高 batch size 和高核心数下不总是优于 HT（Fig. 18-Fig. 21）。

### MLP 感知任务调度

Balance 为一次 embedding 推理任务中的所有表计算 WMLP，然后按 cutoff point 分成两个集合：

- `HWMLP queue`：高 MLP 需求表；
- `LWMLP queue`：低 MLP 需求表。

每个队列内部按 WMLP 降序排列。调度时先把 HWMLP 表以 round-robin 方式分布到不同 CCD，并限制每个 CCD 同时最多一个 HWMLP 表；随后在同一个 CCD 上搭配一个或多个 LWMLP 表（Fig. 15）。

这个策略的目标是让每个 CCD 同时包含：

```text
1 个高 MLP 需求表 + 若干低 MLP 需求表
```

这样可以避免两个极端：

- 多个高 miss 表同处一个 CCD，导致 MSHR 和 DRAM 请求通路过载。
- 多个高复用或轻量表同处一个 CCD，导致 MLP 和内存带宽没有被充分使用。

论文认为，充分使用每个 CCD 的 MLP 资源也会提高整个 SoC 的内存带宽利用率，因为全芯片 DRAM 带宽取决于所有 CCD 发出的 in-flight memory requests（Sec. 3）。

### 两队列设计

HT 为每个 CCD 维护一个队列，队列数等于 CCD 数。Balance 只维护两个全局队列，即 HWMLP queue 和 LWMLP queue。这样做有两个效果：

- 调度器的队列数量不随 CCD 数增长。
- 任务分配由 WMLP 类型驱动，而不是由表进入顺序驱动。

每次调度表任务时，Balance 根据该表预定义的 core grouping 使用 OpenMP 线程在目标 core group 上执行。对 4-core grouping，一个 CCD 可以并行运行两个表；对 2-core grouping，一个 CCD 可以并行运行四个表（Sec. 3；Sec. 5）。

### Work stealing

仅靠 cutoff point 分成两个队列仍可能出现不均衡。例如某个 batch 中 HWMLP 表很少，HWMLP queue 很快为空，而 LWMLP queue 仍有大量任务；或者相反。若严格从各自队列取任务，会出现 worker 空闲。

Balance 的 work stealing 规则较简单：当 HWMLP 或 LWMLP 中一个队列为空时，从另一个队列窃取表任务继续执行（Sec. 3）。它不是 HNSW/IVF 论文中那种 CCD 内优先、跨 CCD 次之的 topology-aware stealing；本文 work stealing 的目标主要是平衡两个 WMLP 队列，而不是维护某个表与 CCD 的长期亲和性。

在 400 表模型的敏感性实验中，work stealing 显著降低 cutoff point 选择带来的性能波动，并在最优 cutoff point 下使 4-core grouping 和 2-core grouping 分别最高提升 `1.5x` 和 `2.06x`（Fig. 22；Sec. 5.2）。

### 三个机制的协同

Balance 的设计不是单独解决一个问题，而是把三个资源维度一起处理：

| 机制 | 解决的问题 | 依赖指标 |
|---|---|---|
| 低核心分组 | 多数表不需要 8 核，固定整 CCD 处理导致核心利用率低 | TL 与 per-table scaling |
| MLP 感知调度 | 高 miss 表与低 miss 表混合放置，提高 MSHR 和内存带宽利用率 | WMLP |
| Work stealing | cutoff point 或表分布导致两个队列任务量不均 | 队列状态 |

这也是论文与普通“负载均衡”工作的差别：它不是只把表执行时间均匀分摊到 CCD，而是同时考虑每张表适合多少核心、是否会冲击 MLP、以及两个队列是否出现动态不均衡。

## 实验设置

### 硬件

论文所有实验使用单 socket 配置（Table 2；Sec. 4）：

| 参数 | 配置 |
|---|---|
| CPU | AMD 4th Generation EPYC |
| 核心数 | 96 |
| CCD 数 | 12 |
| Core/CCD | 8 |
| L3/CCD | 32 MB |
| L2/Core | 1 MB |
| L1 I/D | 32 KB / 32 KB |
| 内存带宽 | 12 通道 DDR5，总带宽约 460 GB/s |
| 内存容量 | 768 GB/socket |
| 主频 | 3.5 GHz |
| TDP | 360 W |

系统只测试单路处理器，不涉及跨 socket NUMA。论文中 CCD 是主要拓扑单位。

### 软件环境

软件配置如下（Sec. 4）：

- PyTorch v1.13.1，从源码编译。
- ZenDNN，包含 AVX512 优化的 embedding bag operator。
- GCC 11.4.0。
- Ubuntu 22.04.2 LTS，Linux kernel 5.19.0-32-generic。
- 开启 Transparent Huge Pages 和 frequency boost。
- 关闭 SMT。
- 关闭 L1/L2 stream hardware prefetcher。

作者说明，Transparent Huge Pages 可把地址转换开销移出 embedding bag 关键路径，使 batch latency 提升 20%-30%；stream prefetcher 对不规则 embedding 访问无帮助；SMT 由于每线程 MLP 有限，也不能提升 embedding bag 性能（Sec. 4）。

### Workload

论文使用两类数据集（Sec. 4）：

| 数据集 | 含义 | 用途 |
|---|---|---|
| Homogeneous dataset | 96 张表，每表 500K 行，每用户 150 次 lookup；表大小、batch size、pooling factor、embedding dimension 等参数均匀 | 作为已有研究中的规则随机/热点数据集，展示理想化场景 |
| Heterogeneous dataset | Meta 生产 trace，表大小、pooling factor、访问模式均不均匀；129 表模型和 400 表模型 | 主要评估对象，代表生产推荐模型 |

默认设置：

- embedding dimension = 128；
- batch size = 1024，除非实验明确改变；
- batch size 敏感性覆盖 2048、4096、8192；
- 129 表模型运行 128 个 batch，前 8 个 batch 作为冷启动不计入结果，后 120 个 warm batch 取平均（Sec. 4）。

### Baseline 与比较对象

主要 baseline 是 HT，因为论文在 Sec. 2.2 中已经证明 HT 是 BT/TT/HT 中性能最强的既有方案。评估中比较：

- `HT`：每张表分到一个 CCD，使用该 CCD 的 8 个核心。
- `Balance 4-core grouping`：每张表使用 4 核，一个 CCD 最多并行 2 张表。
- `Balance 2-core grouping`：每张表使用 2 核，一个 CCD 最多并行 4 张表。

性能指标主要是：

- 相对 HT 的 speedup；
- 相对单核实现的 speedup；
- embedding bag kernel latency；
- memory bandwidth；
- cutoff point 和 work stealing 对性能波动的影响。

## 实验结果与解释

### 129 表模型，batch size 1024

Fig. 16 比较 Balance 和 HT 在 1、8、16、24、48、96 核下的性能。batch size 为 1024，模型包含 129 张表。

主要结果：

- 24 核配置下，Balance 4-core grouping 相对 HT 最高约 `1.3x`，2-core grouping 相对 HT 最高约 `1.41x`（Sec. 5.1；Fig. 16）。
- 核心数继续增加后，性能扩展趋于停滞。论文解释为调度到更多 CCD 后，data fabric 网络中的内存流量升高，成为新的限制（Sec. 5.1）。
- 即便在 96 核下，4-core grouping 仍优于 HT；2-core grouping 的稳定性较差（Fig. 16）。

Fig. 17 展示 Balance 相对单核的扩展性。曲线在 48 核附近出现停滞，说明此时单纯增加核心/CCD 已难以继续提高 embedding bag 性能，瓶颈转向内存系统和 data fabric（Fig. 17；Sec. 5.1）。

### 不同 batch size

论文将 batch size 提高到 2048、4096、8192，分别在 Fig. 18-Fig. 21 中展示相对 HT 和相对单核的扩展性。

4-core grouping 的结果：

- 在多个 batch size 和核心数下持续优于 HT（Fig. 18）。
- batch size 增大后，每张表的工作量更大，线程创建/销毁和同步开销更容易被摊销，因此 Balance 扩展性更好（Sec. 5.1）。
- 相对单核的最高加速达到 `21.65x`，发生在 batch size 8192（Fig. 20；Sec. 5.1）。

2-core grouping 的结果：

- 在 batch size 较大或核心数较高时不总是优于 HT（Fig. 19）。
- 原因是 2-core grouping 让一个 CCD 同时处理最多 4 张表，带来更多并发表任务，容易产生更高 L3 miss 和 data fabric 流量（Sec. 5.1）。
- 论文还指出，若少数 high-work 表被映射到同一 CCD，2-core grouping 可能产生 straggler，类似 TT 在 96 核下的异常（Fig. 19；Sec. 5.1）。

综合结论：2-core grouping 在某些小 batch 或较少核心配置下收益更高，但 4-core grouping 是更稳定的设计点。

### 96 核扩展性与绝对延迟

论文报告 Balance 在 batch size 1024、96 核配置下，相比 HT 实现 `1.67x` 更好的扩展性（摘要；Sec. 5.1）。此外，Balance 下 embedding bag kernel 绝对延迟为 `2.5 ms`，作者认为这更适合实时推荐推理的延迟要求（Sec. 5.1）。

这里的 `1.67x` 是相对 HT 的多核扩展能力，不代表相对单核达到 96 倍。论文同时显示，即使 Balance 也远未达到 96 核理想线性扩展，内存系统和 data fabric 仍是主要限制。

### Cutoff point 与 work stealing

400 表模型用于分析 cutoff point 和 work stealing。cutoff point 决定 WMLP 排序后多少张表进入 HWMLP queue，其余进入 LWMLP queue。

Fig. 22 在 batch size 2048、48 核下比较不同 cutoff point 的归一化执行时间。

主要结果：

- 不使用 work stealing 时，不同 cutoff point 导致执行时间波动明显（Fig. 22）。
- 2-core grouping 的波动大于 4-core grouping，因为一个 CCD 内并发表更多，更容易受表分布不均影响（Fig. 22）。
- 启用 work stealing 后，cutoff point 带来的性能波动显著降低（Fig. 22）。
- 在最优 cutoff point 下，work stealing 对 4-core grouping 最高提升 `1.5x`，对 2-core grouping 最高提升 `2.06x`（Sec. 5.2）。
- 4-core grouping 平均每 batch 触发 work stealing 35.4 次；2-core grouping 平均触发 71.2 次。2-core grouping 因 worker 数更多、队列比例更偏，依赖 stealing 更强（Sec. 5.2）。

这个实验说明 WMLP 队列划分不是一次性静态最优。生产表数量和 WMLP 分布变化时，单纯依赖 cutoff point 会有风险；work stealing 是必要补偿机制。

### 400 表模型扩展性

Fig. 23 比较 400 表模型下 HT、Balance 4-core grouping、Balance 2-core grouping 在 batch size 1024 和 2048 下相对单核的 speedup。

论文结论是：Balance 继续优于 HT，尤其在高核心数下 4-core grouping 优势更明显（Fig. 23；Sec. 5.2）。这说明 Balance 不只适用于 129 表模型，也能扩展到更多表的推荐模型。

### 内存带宽

Fig. 24 在 48 核、batch size 2048 下比较 HT 和 Balance 的内存带宽。

论文报告：

- Balance 的内存带宽利用率高于 HT（Fig. 24）。
- 作者解释为 Balance 改善了缓存利用和执行时间；即使到 DRAM 的总流量更低，单位时间完成更多工作也会表现为更高带宽利用（Sec. 5.2）。
- 2-core grouping 的内存带宽比 HT 高 `1.39x`，但性能只提高 `1.2x`。原因是 2-core grouping 一个 CCD 最多并行 4 张表，更容易产生共享 L3 miss，导致更多 DRAM 流量（Fig. 24；Sec. 5.2）。

这个结果是 2-core grouping 的边界证据：带宽更高不一定代表更高效率，也可能表示 L3 冲突和 miss 增加。

## 论文结论

论文的直接结论如下：

1. DLRM embedding 生产表高度异构，表大小、访问复用、pooling factor 和执行时间差异明显，HT 的 round-robin 表分配不足以保证 CCD 间负载均衡（Fig. 3-Fig. 6）。
2. 多数表不需要整 CCD 的 8 个核心。用 2 核或 4 核分组能提高核心利用率，并降低轻量表的调度开销占比（Fig. 11-Fig. 12；Table 1）。
3. 随机低复用表对 L3 MSHR 和 MLP 资源敏感。固定核心数但增加 CCD 数可显著提升低复用表的扩展性，说明调度需要感知 MLP 需求（Fig. 13-Fig. 14）。
4. Balance 通过 TL、WMLP、低核心分组、双队列调度和 work stealing 组合解决 HT 的三个低效来源（Fig. 15）。
5. 在 96 核 AMD EPYC 上，Balance 相对 HT 最高获得 `1.67x` 扩展性提升；在多个 batch size 上相对 HT 平均/稳定取得更高性能，摘要称整体范围内性能提升约 `1.22x`（摘要；Fig. 16-Fig. 24）。

## 对当前 Chiplet/HNSW 研究的启发

### 可迁移思想

这篇论文对当前研究最有价值的部分不是推荐模型本身，而是调度器如何把“任务异构”和“硬件资源异构”映射起来。

可迁移思想包括：

1. **任务负载不能只按数量均衡**。HT 的失败点之一是每个 CCD 表数量接近，但执行时间差异巨大。HNSW 中也类似：相同 query 数不代表相同 visited node 数、距离计算量或 L3 miss 压力。
2. **调度 key 需要低成本获得**。TL 只用 row index trace、pooling factor 和简单缓存模型估计，不依赖高开销 PMU 在线采样。HNSW 若做 dispatcher，也应优先选择低成本信号，例如 upper-entry、entry region、粗粒度路径签名，而不是运行复杂 oracle。
3. **局部性和负载要一起处理**。只把相似任务放到同一 CCD 可能形成热点 CCD；只做负载均衡可能打散 L3 复用。Balance 用 WMLP 区分高/低 MLP 任务，再限制每 CCD 的高 MLP 表数量，体现了“资源互补放置”的思路。
4. **核心分组粒度需要匹配任务扩展性**。DLRM 表不一定适合 8 核；HNSW 单 query 通常也不适合拆到多个核心。若做 HNSW 调度，更合理的粒度可能是 query-to-CCD 或 batch-to-CCD，而不是把单 query 拆给整 CCD。
5. **work stealing 是必要补偿，但会影响 locality**。本文 work stealing 用于弥补双队列不均；HNSW 中若 stealing 跨 CCD，会破坏本地 L3 复用。因此 HNSW 的 stealing 应比本文更强调 CCD 内优先、跨 CCD 保底。

### 不可直接迁移点

这篇论文不能直接证明 HNSW CCD-aware dispatcher 有效，原因如下：

1. **任务粒度不同**：DLRM embedding 一个 batch 的表任务可被调度器集中组织，HNSW 单 query 通常只有毫秒级，在线调度预算更小。
2. **调度对象不同**：DLRM 的 table id 和 row index 在执行前已知；HNSW 的第 0 层访问路径要通过搜索过程才能知道，完整路径不能作为零成本调度 key。
3. **并行结构不同**：DLRM 一次 embedding stage 要处理上百张表，天然存在大量可重排任务；HNSW 服务端是否有足够相似 query 同时到达，需要 workload 证明。
4. **复用对象不同**：DLRM 表行是直接按 ID 查表，热 ID 复用容易统计；HNSW 节点复用取决于图结构、入口点、`efSearch`、查询分布和向量空间局部性。
5. **贡献边界不同**：本文已经覆盖推荐系统 embedding lookup 的表到 CCD/core group 调度。若当前课题转向推荐系统，不能把表级 CCD 划分、低核心分组、hot row 复用作为新贡献，只能研究更高层 request-level co-location 或跨 batch affinity。

### 对“推荐系统 embedding lookup”方向的影响

当前 `HNSW 之外的 Chiplet Locality 应用场景研究方向分析` 已将推荐系统 embedding lookup 降为较低优先级。这篇 ASPLOS 2025 论文进一步强化了该判断：

- 它已经把 HT 作为强 baseline，并在 HT 之上继续优化 core grouping、WMLP 调度和 work stealing。
- 它直接使用 AMD CCD 拓扑、每 CCD L3、MSHR/MLP 资源和生产 Meta trace，覆盖了我们原本可能提出的很多 chiplet-aware embedding 设计点。
- 剩余空间主要在 serving runtime 层，例如跨 batch 的 request-level hot-ID co-location、多模型/多租户实例放置、SLO-aware spill。这些问题不再是 embedding bag operator 内部调度，实验和论文定位都需要改变。

因此，推荐系统 embedding lookup 不适合作为当前 Chiplet locality 的第一主线。它更适合作为方法参照和强相关工作，用于提醒后续研究：若选择一个随机访存 workload，必须先确认已有工作是否已经覆盖“表/任务到 CCD、CCD 内核心组、负载和 MLP 感知调度”。

## 边界与注意事项

1. 论文主要评估 embedding bag kernel，不是完整端到端推荐服务。Bottom MLP、feature interaction、top MLP、请求排队和服务框架开销不在主要结果中。
2. 实验为单 socket AMD EPYC，不涉及跨 socket NUMA。结论不能直接外推到双路服务器的 socket 间访问。
3. Balance 的核心分组是静态 2 核或 4 核评估，没有展示每张表动态选择最优核心数的完整在线算法。
4. WMLP 的公式表述存在 hit/miss 方向歧义。论文上下文支持“WMLP 表达 L3 miss/MLP 压力”，但引用时应避免写成严格公式，除非核对最终排版原文。
5. 2-core grouping 的结果说明，增加并发表数可能提高带宽但降低效率。Chiplet 调度不能只追求更多并发，也要控制 L3 miss 和 data fabric 流量。
6. Work stealing 在本文中主要跨两个 WMLP 队列平衡任务，没有深入讨论 stealing 对某张表本地 L3 驻留的破坏。对 HNSW 或其他更依赖 CCD 本地复用的 workload，需要重新设计 stealing 策略。

## 证据索引

| 结论 | 证据 |
|---|---|
| HT 在异构生产数据集上 96 核只达到约 13x，远低于理想 96x | Fig. 1 |
| DLRM 包含 bottom MLP、embedding、feature interaction、top MLP | Fig. 2 |
| 生产 trace 表大小、访问模式、pooling factor 差异明显 | Fig. 3 |
| 表执行时间差异可达 37.8x | Fig. 4 |
| BT、TT、HT 的缓存复制、负载和 MLP 问题 | Fig. 5 |
| HT 使用每 CCD 一个 task queue，表按 round-robin 注入 | Fig. 6 |
| TL 用 reuse distance 模型估计 AMAT | Fig. 7 |
| TL 与实测执行时间相关性高，APF 单独估计不充分 | Fig. 8 |
| TL 将表划分为 heavy/medium/light | Fig. 9 |
| TL 在 120 个 batch 内稳定 | Fig. 10 |
| 多数表不需要 8 核 | Fig. 11 |
| TL 与 2/4/8 核扩展性相关 | Fig. 12；Table 1 |
| 固定 8 线程跨多个 CCD 可提高低复用表扩展性 | Fig. 13 |
| WMLP 捕捉表的 MLP 需求差异 | Fig. 14 |
| Balance 使用 HWMLP/LWMLP 双队列、低核心分组和 work stealing | Fig. 15 |
| Balance 在 129 表模型、batch size 1024 下优于 HT | Fig. 16-Fig. 17 |
| 4-core grouping 在多 batch size 下更稳定 | Fig. 18-Fig. 21 |
| Work stealing 降低 cutoff point 敏感性并最高提升 1.5x/2.06x | Fig. 22 |
| 400 表模型中 Balance 仍优于 HT | Fig. 23 |
| Balance 提高内存带宽利用，2-core grouping 带宽更高但效率不一定更好 | Fig. 24 |
