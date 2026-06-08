# MemANNS 解读

**一句话概述**：MemANNS 面向十亿级近似最近邻搜索中的 IVFPQ 压缩索引，把最耗内存带宽的查询阶段迁移到 UPMEM PIM 硬件上执行。论文的核心不是改造 HNSW 图搜索，而是围绕 IVFPQ（Inverted File with Product Quantization，倒排文件加乘积量化）设计 PIM 感知的数据放置、查询调度、DPU 内线程与 WRAM 复用、共现感知编码和 Top-K 剪枝。实验显示，在 SIFT1B 和 SPACEV1B 上，MemANNS 相对 CPU 版 Faiss IVFPQ 的 QPS 提升 2.1x-4.3x，多数配置下达到接近 GPU 版 Faiss 的 QPS，并在 QPS/W 上约为 GPU 的 2x，摘要中报告最高 2.3x（Abstract；Fig. 13；Sec. 5.2）。

---

## 论文基本信息

- 论文：`MemANNS: Enhancing Billion-Scale ANNS Efficiency with Practical PIM Hardware`
- 作者：Sitian Chen、Amelie Chi Zhou、Yucheng Shi、Yusen Li、Xin Yao
- 时间：arXiv 2024-10-31
- 研究问题：十亿级 ANNS（Approximate Nearest Neighbor Search，近似最近邻搜索）中 IVFPQ 压缩索引的查询效率
- 硬件平台：UPMEM Processing-in-Memory（商用 PIM 硬件，把简单处理核心集成到 DRAM 芯片内部）
- 原始资料：`wiki/原始资料/papers/HNSW/MemANNS: Enhancing Billion-Scale ANNS Efficiency with Practical PIM Hardware.pdf`

## 问题

论文从 Faiss 的 CPU 和 GPU IVFPQ 实现出发，用 SIFT 数据集在 1M、100M、1B 三个规模上测量各阶段耗时占比（Fig. 1），发现两个关键现象：

1. **CPU 端瓶颈随数据规模转移**：数据从 1M 增长到 1B 后，CPU 版 IVFPQ 的瓶颈从计算密集的查找表构造（LUT construction）转为内存密集的距离计算（distance calculation）。原因是数据量增大后，每个查询需要访问的压缩编码点数量急剧增长，内存访问时间占据主导。论文估算：数据量 10^9、聚类数 |C|=4096、nprobe=32、编码长度 M=32 时，单查询需要约 2.5 亿次随机内存访问来读取编码并查表累加（Sec. 2.3）。
2. **GPU 端 Top-K 成为新瓶颈**：GPU 带宽高，距离计算不再是瓶颈，但 Top-K 选择阶段并行度有限，无法充分利用 GPU 资源，在大规模下消耗大量时间（Fig. 1；Sec. 1）。

基于上述观察，论文要解决的问题是：

1. CPU 内存带宽不足以支撑十亿级 IVFPQ 的距离计算阶段（2.5 亿次随机内存访问/查询）。
2. GPU 带宽高但显存容量有限（A100 80GB）、成本高，且 Top-K 阶段可能成为新瓶颈，能效和成本不适合大规模线上检索（Fig. 1；Fig. 18；Sec. 5.2）。
3. 商用 PIM（UPMEM）提供高聚合带宽（理论 7.2 TB/s），但单个 DPU 算力弱、局部内存小（64 KB WRAM、64 MB MRAM）、跨 DPU 通信必须经主机 CPU，需要重新设计数据放置、调度和 DPU 内执行方式才能利用聚合带宽（Sec. 2.2-2.3）。

## 背景

### ANNS 与 IVFPQ

ANNS（Approximate Nearest Neighbor Search，近似最近邻搜索）给定查询向量，在向量集合中返回距离最近的 k 个向量，允许近似结果。十亿级场景下，精确搜索需扫描全量向量，成本过高；压缩索引通过聚类缩小搜索范围、通过量化压缩向量表示，以少量精度损失换取更低延迟和内存占用。

IVFPQ 由两部分组成：

- **IVF**（Inverted File，倒排文件）：把全量向量用 K-means 等聚类算法划分到 |C| 个聚类。查询时只选择离查询向量最近的 `nprobe` 个聚类，不扫描全量。
- **PQ**（Product Quantization，乘积量化）：把高维向量切成 M 个子向量，每个子向量用离线训练的码本（codebook）中最近码字编号（uint8，0-255）表示。距离计算时不读取原始向量，而是读取压缩编码并查表累加。

IVFPQ 的离线阶段：聚类 → 计算每个点相对聚类中心的残差 → 对残差做 PQ 编码。压缩率约 4D/M（原始维度 D，编码维度 M）。在线查询阶段分四步（Fig. 2；Sec. 2.1）：

1. **聚类过滤**：计算查询向量与各聚类中心的距离，选 `nprobe` 个最近聚类。
2. **查找表构造**：为每个选中聚类建立 LUT。LUT[j][i] = 查询残差第 i 子段与子码本 B_i 第 j 个码字的距离。LUT 的粒度是 **(query, cluster) 对**——不同查询 q 不同则残差 q−c 不同，同一查询不同聚类 c 不同则残差也不同，因此每个 (query, cluster) 对需要独立的 LUT。距离计算时查表取值即可，避免每次重复计算子向量距离。
3. **距离计算**：遍历选中聚类内所有编码点，通过查 LUT 并累加 M 个子段的部分距离，得到查询到候选点的近似距离。
4. **Top-K 选择**：从所有候选距离中取最小的 k 个。

十亿级数据放大了距离计算阶段的内存访问量，这解释了 CPU 版在大规模下的内存带宽瓶颈。

### UPMEM PIM 硬件

UPMEM 是第一款商用 PIM 硬件，形态为标准 DDR4 DIMM。一条 UPMEM DIMM 含 16 个 PIM chip，每个 chip 含 8 个 DPU（DRAM Processing Unit），一条 DIMM 共 128 个 DPU。现有系统最多支持 20 条 DIMM（2560 个 DPU），理论聚合带宽 7.2 TB/s（Fig. 3；Sec. 2.2）。

每个 DPU 的规格：

- 32 位 RISC 核，350 MHz，14 级流水线，最多 24 个硬件线程
- 64 MB **MRAM**：DPU 独占的 DRAM bank，DPU 不能直接寻址 MRAM，必须通过 DMA 引擎搬移到 WRAM 后才能访问；MRAM 访问延迟远高于 WRAM
- 64 KB **WRAM**：快速 scratchpad，单周期访问
- 24 KB **IRAM**：指令内存
- 无 MMU，DPU 侧使用物理地址管理内存

关键约束：

- **跨 DPU 通信必须经主机 CPU**，不能 DPU 间直连
- **CPU-DPU 数据传输**：仅当所有 MRAM bank 的传输 buffer 大小相同时才能并发，否则串行执行
- **WRAM 容量极小**（64 KB），需要显式管理数据生命周期
- **DPU 算力远弱于 CPU 大核和 GPU**：UPMEM 最大整数处理能力 896 GOPS，A100 为 156 TFLOPS；乘法和加法延迟差距大，需要规避乘法

UPMEM 的优势来自大量 DPU 的聚合带宽（每个 DPU 独占一条到本地 MRAM 的通道），适合数据访问密集、计算强度低的工作负载（Sec. 2.2-2.3）。

## 核心内容

MemANNS 保持 IVFPQ 的搜索语义，不改变召回率。主机 CPU 负责聚类过滤和调度，DPU 负责查找表构造、距离计算和局部 Top-K。五项主要设计（Fig. 5；Sec. 3-4）：

1. **PIM 感知的数据放置**：离线把聚类分布到 DPU，以 `size * frequency` 估算工作量；热门或大聚类复制到多个 DPU；相邻聚类共置以减少 CPU-DPU 通信（Algorithm 1；Sec. 4.1）。
2. **在线查询调度**：根据每个查询选中的聚类和离线放置结果，把查询-聚类任务分配到负载较低且包含对应副本的 DPU（Algorithm 2；Sec. 4.1）。
3. **DPU 内资源管理**：同一聚类内部多线程并行；复用 64 KB WRAM 的生命周期；通过流水化隐藏 MRAM→WRAM 搬移延迟（Fig. 8-9；Sec. 4.2）。
4. **共现感知编码**：利用压缩编码中某些码字组合频繁共同出现的事实，预存这些组合的部分和地址，减少在线查表和累加次数（Fig. 10-11；Table 1；Sec. 4.3）。
5. **DPU 内 Top-K 剪枝**：每个线程维护局部堆，DPU 内合并时提前剪掉不可能进入总 Top-K 的元素，减少 DPU→CPU 结果传输和合并开销（Fig. 12；Sec. 4.4）。

## 方法与系统设计

### 系统边界

离线阶段在主机 CPU 上执行：IVFPQ 编码、数据放置决策、共现组合选择、数据传输到 DPU MRAM。

在线阶段：CPU 执行聚类过滤和查询调度，DPU 执行 LUT 构造、距离计算和局部 Top-K，CPU 最后汇总各 DPU 的部分 Top-K 得最终结果（Fig. 5；Sec. 3）。

这个划分与 IVFPQ 的瓶颈匹配：聚类过滤计算量小，放 CPU 不会成瓶颈；距离计算需要大量读压缩编码，适合交给 DPU 利用本地 MRAM 带宽（Sec. 3；Sec. 4.1）。

### 数据放置（Algorithm 1）

UPMEM 每个 DPU 只能快速访问本地 MRAM。数据放置的两个目标：减少跨 DPU/CPU-DPU 通信，均衡各 DPU 在线内存访问量（Sec. 4.1）。

论文以聚类为放置单位。一个聚类内的所有编码点尽量放同一 DPU，这样该聚类产生的局部 Top-K 可在 DPU 内完成，CPU 只汇总更少的部分结果。

聚类 i 的工作量估计为 `w_i = s_i * f_i`，其中 `s_i` 是聚类大小（向量数），`f_i` 是聚类被查询选中的历史频率。给定 n 个 DPU，目标平均工作量为 `W = (1/n) * Σ s_i * f_i`。当 `w_i` 显著超过 W 时，为该聚类创建 `ncopy = ceil(s_i * f_i / W)` 个副本并分布到多个 DPU。在线查询命中该聚类时可调度到不同副本以平衡负载（Algorithm 1；Sec. 4.1）。

放置算法按聚类工作量从高到低处理。每个副本迭代所有 DPU，选择同时满足两个条件的：剩余容量足够，且放入后当前 DPU 工作量不超过 `W * thld`。若一轮未找到合适 DPU，放宽阈值 `thld` 后重试（Algorithm 1 line 5, 9）。算法还受 `MAX_DPU_SIZE` 约束，防止单 DPU 超容量。

论文还处理聚类共置。实际查询选中的多个聚类通常在向量空间中相邻。若这些聚类分散在多个 DPU，CPU 需接收更多部分 Top-K；若放同一 DPU，DPU 可先本地合并。MemANNS 在为聚类选择 DPU 后，继续把距离相近的聚类放入同一 DPU，直到该 DPU 工作量接近 W（Fig. 6；Sec. 4.1）。

Fig. 4 展示 SPACEV1B 上聚类访问频率（最高差 500x）、聚类大小（最高差 106x）和工作量（size × frequency）均高度偏斜。Fig. 7 显示使用该放置策略后，SIFT1B 上各 DPU 的工作量和内存使用更均衡（Fig. 4；Fig. 7；Sec. 2.3；Sec. 4.1）。

### 查询调度（Algorithm 2）

在线阶段每个 query batch 的每个查询先在 CPU 上执行聚类过滤，得到 `nprobe` 个聚类。调度器根据离线得到的聚类→DPU 映射，把每个查询-聚类任务分配给具体 DPU。

调度本质是带约束的负载均衡：任务只能分配给包含该聚类副本的 DPU。算法分两阶段：
1. 先处理只有一个副本的聚类（无选择余地，直接分配）。
2. 再处理多副本聚类，按聚类大小降序，每次选当前负载最小且包含该副本的 DPU（Algorithm 2；Sec. 4.1）。

复杂度 `O(|Q| * nprobe)`。论文认为 batch 大小和 `nprobe` 相比十亿级数据很小，调度开销可忽略。实验每批处理 1000 个查询（Algorithm 2；Sec. 4.1；Sec. 5.1）。

### DPU 内线程调度

每个 DPU 最多 24 个硬件线程，14 级流水线，所有线程共享 64 KB WRAM。论文比较了三种并行粒度：

- 不同查询之间并行
- 同一查询的不同聚类之间并行
- 同一聚类内部并行

MemANNS 选择同一聚类内部并行。原因：WRAM 太小。SIFT 数据集上，码本大小约 `D * 256 = 32 KB`；当 M=16 时，LUT 大小约 `M * 256 * sizeof(uint16) = 8 KB`。LUT 是 (query, cluster) 粒度，若并行处理多个查询或多个聚类，每增加一个并行单元就多一份 LUT——超过 4 份 LUT 总和即超 64 KB。因此 MemANNS 让一个 DPU 顺序处理被分配到的 (query, cluster) 对，但在单个 (query, cluster) 内部用多线程按数据分片并行。具体分工（Fig. 8；Sec. 4.2）：

- **LUT 构造阶段**：所有线程共享同一份码本，各自并行计算 LUT 的不同子段（每人负责一段列范围）。
- **距离计算阶段**：所有线程共享同一份 LUT，各自从 MRAM 并行读取**同一聚类内不同批次的编码点**（每批 16 个向量），查表累加距离，各自维护线程局部 Top-K 堆。

```
线程 0: 读编码点批次 [0..15]   → 查共享 LUT → 维护 local top-k
线程 1: 读编码点批次 [16..31]  → 查共享 LUT → 维护 local top-k
线程 2: 读编码点批次 [32..47]  → 查共享 LUT → 维护 local top-k
...
```

所有线程共享一份码本（32 KB）和一份 LUT（8 KB），WRAM 总共只需约 40 KB。系统级并行度由 DPU 间并行保证（不同 DPU 同时处理不同的 query-cluster 对），DPU 内专注于把一个聚类的距离计算做快。

DPU 内流程含四个同步点（Fig. 8；Sec. 4.2）：

- **Barrier 0**：防止有线程仍基于旧 LUT 计算距离时，其他线程提前更新 LUT。
- **Barrier 1**：确保 LUT 构造完成后再计算共现组合部分和。
- **Barrier 2**：确保 LUT 和组合部分和已更新后再读取。
- **Barrier 3**：确保所有线程完成距离计算和局部结果插入后，再合并 Top-K。

线程数选择：DPU 的 14 级流水线中，同线程内仅最后 3 级可与下条指令的前 2 级并行执行。需足够线程才能填满流水线、隐藏访存延迟。实验显示 QPS 在 1 到 11 个 tasklet 之间近似线性提升（11 tasklet 接近单 tasklet 的 11x），超过 11 后饱和。论文默认每 DPU 使用 11 个线程（Fig. 16；Sec. 5.4.2）。

### WRAM 复用与 MRAM 读粒度

WRAM 只有 64 KB，MemANNS 根据 IVFPQ 在线阶段的先后关系复用同一段 WRAM。构造 LUT 时，WRAM 保存码本（32 KB）和 LUT（8 KB）；LUT 完成后，码本不再需要，该空间可覆盖为共现组合部分和或编码点缓冲。距离计算阶段，多线程从 MRAM 批量读取编码点到 WRAM，再查表累加。Fig. 8 展示了完整的 WRAM 内容生命周期（Fig. 8；Sec. 4.2）。

论文还通过流水化（pipelining）重叠 MRAM 读取和距离计算，以隐藏 MRAM 高延迟（Sec. 4.2）。

MRAM 读粒度需要在两种成本之间平衡：读太小，DMA 调用频繁；读太大，占用 WRAM 且延迟增长。论文测得 MRAM 读延迟在 8B 到 256B 之间增长缓慢，超过 256B 后近似线性增长。MRAM→WRAM 传输须为 8B 的倍数，不小于 8B 且不大于 2048B。后续敏感性实验显示每次读 16 个向量时 QPS 趋于稳定。论文默认每次 MRAM 读取 16 个向量（Fig. 9；Fig. 15；Sec. 4.2；Sec. 5.4.1）。

### 共现感知编码

IVFPQ 压缩编码由 M 个 0-255 的码字组成。论文观察到真实数据集中某些码字组合在特定列上频繁共同出现。例如 SIFT1B 中长度为 3 的最高频组合出现在 5.7% 的编码点中。若在线距离计算每次都分别读取这些码字对应的 LUT 项并累加，会重复执行相同的部分求和（Fig. 10；Sec. 4.3）。

MemANNS 使用 Item Co-occurrence Graph（ICG）选择高频组合。关键约束：组合必须同时考虑**码字和列位置**。例如 (1, 15, 26) 只有在列 (0, 1, 2) 上出现时，才能对应固定的 LUT 项之和；若相同码字出现在其他列，距离含义不同，不能复用同一部分和（Fig. 11；Sec. 4.3）。

离线阶段，系统为每个聚类选择 `m` 个长度为 3 的高频组合，默认 `m=256`，并为这些组合在 WRAM 中预留部分和地址。原始编码被重编码：不属于高频组合的码字保留为带列信息的原始编码（uint16 存储列号和码字），属于高频组合的部分替换为对应部分和的缓存地址。在线阶段，LUT 构造完成后，DPU 先计算这些高频组合的部分和并写入预留 WRAM 地址；距离计算时直接读取部分和地址，减少查表和累加次数（Fig. 11；Sec. 4.3）。

论文进一步把原始码字地址和组合码字地址都转换为直接地址，避免 DPU 上代价较高的乘法。原始码字地址由列号和码字预先折算；组合地址由 LUT 总大小和组合编号折算。在线距离计算直接按地址读取，无需乘法（Fig. 11；Sec. 4.3）。

Table 1 给出经验关系：平均编码长度缩减 0.25、0.5、0.75 时，查询时间分别降低 18%、36%、54%。论文只在平均长度缩减超过 50% 时启用该编码（Table 1；Sec. 4.3）。

### Top-K 剪枝

每个线程维护大小为 k 的最大堆存储线程局部 Top-K。若直接把所有线程的局部结果传给 CPU，CPU-DPU 通信开销大。MemANNS 先在 DPU 内合并线程局部结果，再把每个 DPU 的部分 Top-K 发给 CPU（Fig. 12；Sec. 4.4）。

具体做法：合并阶段，把线程局部最大堆转换为最小堆，使用信号量（sem_take/sem_give）并发地把各线程最小堆堆顶插入 DPU 总 Top-K 最大堆。若某线程局部最小堆的堆顶大于总 Top-K 最大堆中的当前最大值，该线程剩余元素不可能进入全局 Top-K，直接剪掉（Fig. 12；Sec. 4.4）。

该设计降低了 DPU 内合并和 CPU-DPU 传输压力，避免在 PIM 方案中重现 GPU 版 Faiss 的 Top-K 瓶颈（Fig. 1；Fig. 12；Fig. 18）。

## 实验设置

### Baseline

使用 Meta Faiss 的 CPU 版和 GPU 版 IVFPQ 作为主要 baseline。FANNS 用 FPGA 加速 IVFPQ，但设备内存受限不适合十亿级数据；ANNA 是模拟器上的专用架构，不能与真实 UPMEM 硬件直接对比；Juno 用 GPU RT core 加速 IVFPQ，论文认为其性能接近普通 A100 GPU Faiss，不作为主要对照（Sec. 5.1）。

实验比较的是同一压缩索引（IVFPQ）在不同硬件上的系统效率，不是不同 ANN 方法之间的算法比较。

### 硬件

Table 2 给出三类硬件：

| 平台 | 配置 | 近似价格 | 内存容量 | 峰值功耗 | 带宽 |
|---|---|---:|---:|---:|---:|
| CPU | 2 x Intel Xeon Silver 4110@2.10GHz，4 x DDR4 DRAM | 1400 USD | 128 GB | 190 W | 85.3 GB/s |
| GPU | NVIDIA A100 PCIe 80GB | 20000 USD | 80 GB | 300 W | 1935 GB/s |
| PIM | 7 x UPMEM PIM，共 896 DPU | 2800 USD | 56 GB | 162 W | 612.5 GB/s |

PIM 聚合带宽（612.5 GB/s）介于 CPU（85.3 GB/s）和 A100（1935 GB/s）之间；峰值功耗最低（162 W）；硬件成本远低于 A100（2800 vs 20000 USD）（Table 2；Sec. 5.1）。

### 数据集

- **SIFT1B**：10 亿个 128 维向量，编码为 16 维压缩编码。
- **SPACEV1B**：10 亿个 100 维向量，编码为 20 维压缩编码。

两个数据集均含 10000 个查询向量和对应 ground truth。实验每批处理 1000 个查询。MemANNS 的优化保持 IVFPQ 搜索语义，不改变召回率（Sec. 5.1）。

### 参数与指标

主要指标：**QPS**（Queries Per Second）。与 GPU 对比时，还报告 **QPS/W**（单位峰值功耗下的查询吞吐），使用峰值功耗而非运行时实测功耗近似。

IVF 聚类数：4096、8192、16384。nprobe：64、128、256。nprobe 越大候选点越多、召回率通常越高，但查询工作量也越大。结果按 Faiss-CPU 或 Faiss-GPU 在 nprobe=256 时的值归一化展示（Fig. 13；Sec. 5.2）。

## 实验结果与解释

### 整体性能

**相对 Faiss-CPU**：MemANNS 在所有设置下 QPS 更高。SIFT1B 提升 2.3x-4.3x，SPACEV1B 提升 2.1x-4.0x。nprobe 增大时两者 QPS 均下降（搜索更多聚类）。相同 nprobe 下，IVF 聚类数越大，MemANNS 相对 CPU 的提升越明显（Fig. 13a；Sec. 5.2）。

原因：聚类数增大后单个聚类更小，CPU 多级缓存的局部性收益变弱，QPS 不能线性提升；DPU 只有小 WRAM 和本地 MRAM，执行模型围绕批量 MRAM 读取设计，对 CPU cache 局部性变化不那么敏感（Sec. 5.2）。

**相对 Faiss-GPU**：MemANNS 在多数配置下达到接近 GPU 的 QPS。例外：IVF=16384、nprobe=64 时 GPU 明显更强。论文用 Nsight 分析，该设置下 GPU 的 Top-K 阶段并行度异常高（SIFT1B 上比 SPACEV1B 同参数高 9x）（Fig. 13b；Sec. 5.2）。

**能效**：7 个 UPMEM DIMM 峰值功耗 162 W，A100 为 300 W。MemANNS 多数情况下相对 Faiss-GPU 有约 2x QPS/W，摘要和结论报告最高 2.3x。按价格计算的 QPS 最高可达 Faiss-GPU 的 9.3x（Abstract；Fig. 13c；Table 2；Sec. 5.2）。

### 可扩展性

论文仅有 7 个 UPMEM DIMM，真实硬件测量覆盖约 500-900 DPU，并用线性回归预测到 2560 DPU。由于低 DPU 数下容量有限，可扩展性实验使用 SIFT 和 SPACE 的 5 亿规模版本而非完整 10 亿规模（Sec. 5.3）。

Fig. 14 显示 500-900 DPU 实测点与回归曲线吻合，QPS 随 DPU 数接近线性增长。预测到 2560 DPU 时，MemANNS 可达最高 2.7x Faiss-GPU QPS。若按与 A100 相同 300 W 峰值功耗约束估计，可用 1654 个 DPU，论文称该点在所有设置下 QPS 均高于 Faiss-GPU（Fig. 14；Sec. 5.3）。

注意 900 到 2560 DPU 部分属于回归预测而非硬件实测。

### MRAM 读粒度

SIFT1B：每次 MRAM 读取向量数从 2 到 64，对应 64 B 到 2 KB。SPACEV1B：每次 2 到 32 个向量（编码维度 20，单次传输不超过 2048 B）。固定 IVF=4096、top-k=10、14 线程（Sec. 5.4.1）。

Fig. 15 显示两组数据集上 QPS 在读取向量数从 2 增至 16 时增长较快，超过 16 后趋于稳定。小到中等读粒度能摊薄 DMA 开销，过大读粒度占用 WRAM 且延迟增长明显。论文默认每次读取 16 个向量（Fig. 9；Fig. 15；Sec. 5.4.1）。

### DPU 线程数

tasklet 数从 1 到 24，按单 tasklet 归一化。SIFT1B 和 SPACEV1B 在不同 nprobe 下现象一致：1 到 11 tasklet 线性提升，11 tasklet QPS 接近单 tasklet 的 11x，超过 11 后饱和（Fig. 16；Sec. 5.4.2）。

原因：DPU 的 14 级流水线需足够线程隐藏流水线空泡和访存等待；超过 11 线程后 DPU 已保持忙碌，继续增加无明显收益。论文默认每 DPU 使用 11 个线程（Fig. 16；Sec. 5.4.2）。

### Top-K 大小

k 取 1、10、100。相同 k 下，MemANNS 相对 Faiss-CPU 平均 QPS 提升 2.6x；相对 Faiss-GPU 的 QPS/W 超过 2x。k 增大时：Faiss-CPU QPS 基本不变（依然受距离计算支配）；MemANNS QPS 略降（DPU→CPU Top-K 通信和合并成本增加）；Faiss-GPU 也下降（Top-K 阶段大量 CUDA stream synchronization 开销）（Fig. 17；Sec. 5.4.3）。

时间分解（Fig. 18）进一步说明三类硬件的瓶颈差异。相同 k 下，MemANNS 将距离计算阶段占比从 Faiss-CPU 的 99.5% 降至 75.5%（指 MemANNS 总时间中距离计算占 75.5%，而 CPU 上占 99.5%），说明 PIM 有效缓解了 CPU 内存瓶颈。GPU 的 Top-K 阶段占比超过 85%，成为大规模下的主导瓶颈。DPU 上 Top-K 阶段占比随 k 从 9% 增至 17%，未达到 GPU 的主导程度（Fig. 18；Sec. 5.4.3）。

### 组件验证情况

论文对 MRAM 读粒度、DPU 线程数和 Top-K 大小做了敏感性实验，通过 Table 1 给出共现编码长度缩减与查询时间缩减的经验关系。但论文没有提供完整的逐项消融（ablation）表（如依次关闭数据放置、查询调度、共现编码、WRAM 复用和 Top-K 剪枝后的整体 QPS 对比）。Fig. 13 的整体收益应理解为整套系统设计的综合效果，不是某个单独优化的独立收益（Table 1；Fig. 13-18；Sec. 5）。

## 分析

MemANNS 的关键是把 IVFPQ 的内存访问形态匹配到 UPMEM 硬件结构上。IVFPQ 距离计算需大量读取短编码并查表累加——计算强度低、访存量大。PIM 的大量 DPU 各自独占一条通到本地 MRAM 的通道，能把访问分散到多个独立内存通道，绕开 CPU 内存带宽瓶颈（Sec. 2.3；Sec. 3）。

**数据放置和查询调度**是 PIM 收益成立的前提。若热门聚类集中在少数 DPU，聚合带宽无法发挥；若相邻聚类分散在多个 DPU，部分 Top-K 结果增多，CPU-DPU 通信增加。MemANNS 同时使用热门聚类复制和相邻聚类共置，分别处理负载均衡和通信减少（Algorithm 1；Algorithm 2；Fig. 6-7）。

**DPU 内线程策略**体现 PIM 与 CPU/GPU 的根本差异。CPU 依赖 cache 和乱序执行隐藏延迟，GPU 依赖大量线程和高带宽。UPMEM DPU 频率低（350 MHz）、WRAM 小（64 KB）、MRAM 访问需 DMA，必须显式安排 WRAM 内容生命周期、读粒度和线程数。MemANNS 的 WRAM 复用和 11 tasklet 默认值均来自这些硬件约束（Fig. 8-9；Fig. 15-16）。

**共现感知编码**不是普通压缩率优化，而是把部分查表累加结果显式物化到 WRAM。它利用真实数据中压缩码字组合的重复性，减少在线距离计算的访问次数和加法次数。该设计依赖码字组合频率、位置一致性和 WRAM 空间——不能无条件认为所有 IVFPQ 数据集都能达到相同缩减率（Fig. 10-11；Table 1；Sec. 4.3）。

**Top-K 剪枝**解决 PIM 方案中可能出现的新瓶颈。PIM 把距离计算从 CPU 内存带宽瓶颈中释放后，结果合并和通信比例会上升。DPU 内先合并、再剪枝、最后只向 CPU 发送部分结果，是避免 CPU-DPU 通信吞噬收益的必要步骤（Fig. 12；Fig. 18）。

## 边界

1. 论文研究 IVFPQ，不研究 HNSW 图遍历。不能直接证明 HNSW 的邻接图访问、候选队列、入口点选择或图重排序在 PIM/Chiplet 上有效（Sec. 2.1；Sec. 3）。
2. MemANNS 依赖 UPMEM DPU 独占 MRAM 的局部性模型。CPU CCD 的本地 L3、远端 CCD cache 和 DRAM 不是这种可把数据静态放入独占 bank 的结构，不能直接套用 Algorithm 1 的放置含义（Sec. 2.2；Sec. 4.1）。
3. 实验主要报告 QPS 和 QPS/W，没有报告 p50/p95/p99 延迟、服务级尾延迟、在线并发队列或多租户干扰。对延迟敏感系统，QPS 不足以替代尾延迟评估（Sec. 5）。
4. QPS/W 使用峰值功耗近似，不是运行时能耗实测。CPU、GPU、PIM 在不同负载下的实际功耗可能不同，能效结论应按论文口径理解（Table 2；Sec. 5.2）。
5. 可扩展性到 2560 DPU 的结果含回归预测。真实硬件实测覆盖 500-900 DPU，且该实验使用 5 亿规模数据而非完整 10 亿规模（Fig. 14；Sec. 5.3）。
6. 论文没有完整逐项消融，无法精确分离数据放置、查询调度、共现编码、WRAM 复用和 Top-K 剪枝各自贡献（Sec. 5）。
7. 共现感知编码只在平均长度缩减超过 50% 时启用，收益依赖数据集的码字共现模式。若数据分布或 PQ 码本导致高频组合不明显，该机制收益可能较低（Table 1；Fig. 10；Sec. 4.3）。
8. UPMEM DPU 算力有限：最大整数处理能力 896 GOPS，而 A100 为 156 TFLOPS；DPU 上乘法和加法延迟差距也需规避。这限制了 PIM 方案适合的算法类型——低计算强度、访存占比高的阶段更合适（Sec. 5.5）。
9. 论文没有评估动态插入、删除、在线重构、查询分布长期漂移和副本重放置成本。Algorithm 1 使用历史查询频率预测 `f_i`，若线上分布变化，放置和复制策略需重新评估（Algorithm 1；Sec. 4.1）。

## 与当前 Chiplet/HNSW 研究的关系

### 可迁移点

- MemANNS 支撑一个明确事实：十亿级向量检索中，压缩索引的距离计算阶段可能由内存访问主导，且工作量受聚类大小和访问频率共同影响（Fig. 1；Fig. 4；Sec. 2.3）。
- 负载均衡不能只看数据大小。MemANNS 用 `s_i * f_i` 描述聚类工作量，提示 Chiplet/HNSW 实验也应同时记录节点/子图大小和访问频率，而非仅按静态数据量分片（Algorithm 1；Sec. 4.1）。
- 热点复制是处理访问倾斜的有效系统手段。对 HNSW 或其他图索引，若某些入口点、hub 节点或高频路径被大量查询共享，可考虑复制这些结构，但须区分复制带来的负载均衡收益与缓存局部性收益（Algorithm 1；Fig. 4）。
- 相邻工作单元共置能减少中间结果合并和跨域通信。MemANNS 用聚类距离决定共置；对 HNSW 方向可类比为把常共同访问的入口点、邻接块或 query group 放在同一 CCD 工作域，但需硬件计数器证明跨 CCD 访问或共享缓存复用确实发生变化（Fig. 6；Sec. 4.1）。
- DPU 内 WRAM 复用说明局部存储容量小并不自动带来收益，必须让算法阶段显式匹配存储生命周期。Chiplet L3 优化也需证明被复用的数据在同一 CCD 内持续驻留，而非只改变任务归属（Fig. 8；Sec. 4.2）。

### 不可直接迁移点

- MemANNS 的硬件单元是 DPU + 私有 MRAM，不是 AMD EPYC 的 CCD + 本地 L3。DPU 间通信经 CPU，CCD 间访问通过 CPU 片上互联和缓存/内存层次，延迟、带宽和一致性机制不同（Sec. 2.2）。
- MemANNS 优化 IVFPQ 的聚类扫描和压缩码查表，不是 HNSW 的图搜索。HNSW 的瓶颈包含随机邻接访问、向量距离计算、visited 标记、候选队列和图结构质量，不能用 IVFPQ 的 QPS 结论直接替代（Sec. 2.1；Sec. 4）。
- MemANNS 召回率不变是因为优化保持 IVFPQ 搜索语义。HNSW 中若改变图结构、入口点、候选扩展顺序或分片搜索边界，召回率可能变化，必须独立报告 recall-latency 曲线（Sec. 5.1）。
- 论文没有测 L3 miss、跨 CCD 访问、cache-to-cache、IBS load latency、LLC occupancy 或 NUMA source。它只能作为"内存访问主导型向量检索如何做硬件感知调度"的参考，不能作为 chiplet-aware HNSW 有效性的直接证据。

## 后续实验启发

1. 对 HNSW 查询记录访问热度时，应同时统计访问频率和访问数据量，形成类似 `size * frequency` 的工作量指标，再比较按大小分片、按热度分片和按访问频率复制的效果。
2. 研究相似 query 调度时，应记录同一批 query 共同访问的节点、邻接块和 cache line，而非只记录 query embedding 相似度。MemANNS 的聚类共置依据是"同一查询常共同选择"，对应 HNSW 应转化为"同一批查询常共同访问"。
3. HNSW 的强 baseline 应包括普通 cache-friendly layout、图重排序和热点复制。若只相对原始插入顺序提升，不能证明收益来自 chiplet 拓扑。
4. 指标应覆盖 recall、平均延迟、p95/p99 延迟、QPS、L3 miss/query、IBS load latency、local/remote source、CCD 级 LLC occupancy 和跨 CCD 访问比例。MemANNS 的 QPS 口径不足以支撑服务级尾延迟结论。
5. 若考虑把 IVFPQ 作为当前平台的新 workload，MemANNS 提供了明确切入点：聚类访问倾斜、查询批调度、热点聚类复制和分片内结果合并。这比直接把 MemANNS 结论套到 HNSW 更可控。

## 证据

- 原始资料：`wiki/原始资料/papers/HNSW/MemANNS: Enhancing Billion-Scale ANNS Efficiency with Practical PIM Hardware.pdf`
- 关键锚点：Abstract；Sec. 1；Sec. 2.1-2.3；Sec. 3；Sec. 4.1-4.4；Algorithm 1-2；Table 1-2；Fig. 1-18；Sec. 5.1-5.5；Sec. 7
