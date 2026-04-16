# 2026-04-09 补充相关工作取证摘要

生成时间：`2026-04-09 21:30:31 +0800`

## 1. Leveraging Chiplet-Locality for Efficient Memory Mapping in Multi-Chip Module GPUs

- 原始链接：`https://dl.acm.org/doi/10.1145/3725843.3756090`
- 访问情况：`dl.acm.org` 直连受限；改用可访问的摘要镜像与论文信息页核对。
- 已核对事实：
  - 标题为 `Leveraging Chiplet-Locality for Efficient Memory Mapping in Multi-Chip Module GPUs`。
  - 论文提出 `CLAP`。
  - 可访问摘要写明：作者观察到 GPU 应用存在稳定的 `chiplet-locality` 页组模式；`CLAP` 依据这一模式，把更可能由同一 chiplet 访问的页预组织到该 chiplet 的连续物理帧中，并通过合并 TLB 项兼顾 locality 与大页收益。
  - 可访问摘要写明：相对既有 paging 方案，性能最高提升 `19.2%`。

## 2. Memory Performance of AMD EPYC Rome and Intel Cascade Lake SP Server Processors

- 原始链接：`https://research.spec.org/icpe_proceedings/2022/proceedings/p165.pdf`
- 已核对事实：
  - 标题为 `Memory Performance of AMD EPYC Rome and Intel Cascade Lake SP Server Processors`。
  - 摘要写明：论文详细实验评估了 `AMD EPYC Rome` 与 `Intel Cascade Lake SP` 的 memory hierarchy。
  - 摘要写明：两者在 memory latency 尤其是 `remote cache accesses` 上表现不同。
  - 摘要写明：论文分析了 `NUMA` 特性，以及 data placement 和 cache coherence state 对本地/远程访问延迟的影响。

## 3. TiNA: Tiered Network Buffer Architecture for Fast Networking in Chiplet-based CPUs

- 原始链接：`https://saksham.web.illinois.edu/assets/pdf/tina.pdf`
- 已核对事实：
  - 标题为 `TiNA: Tiered Network Buffer Architecture for Fast Networking in Chiplet-based CPUs`。
  - 摘要写明：论文关注 chiplet CPU 中跨 chiplet 访问 `LLC slices` / `DRAM controllers` 带来的更长且更不稳定的内存访问延迟。
  - 摘要写明：论文利用 `Sub-NUMA Clustering (SNC)` 的低延迟特性，同时指出其会缩小可用 LLC 容量。
  - 摘要写明：`TiNA` 通过分层网络缓冲与 opportunistic 使用远端 chiplet LLC slices 做 `DCA`，平均把 mean/tail latency 分别相对 `SNC` 降低 `25%/18%`，相对 `non-SNC` 降低 `28%/22%`。

## 4. Measuring Data Access Latency in Large CPU Caches

- 原始链接：`https://www.memsys.io/wp-content/uploads/ninja-forms/5/Measuring_Data_Access_Latency_in_Large_CPU_Caches-2.pdf`
- 已核对事实：
  - 标题为 `Measuring Data Access Latency in Large CPU Caches`。
  - 摘要写明：论文提出新的 `multi-locality benchmark`，用于测量近期 AMD `V-Cache` 机器的大 cache 访问延迟。
  - 摘要写明：结果显示这些大 cache 与传统 LLC 至少有两点不同：`V-Cache is partitioned rather than shared`，且 replacement policy 更接近随机而非 `LRU`。
  - 引言写明：单核场景下，只会使用本地 `V-Cache`，更多 `V-Cache modules` 不会自动带来收益。

## 5. Server Chiplet Networking

- 原始链接：`https://conferences.sigcomm.org/hotnets/2025/papers/hotnets25-final85.pdf`
- 已核对事实：
  - 标题为 `Server Chiplet Networking`。
  - 摘要写明：论文把 chiplet server 内部由 compute chiplets、I/O chiplets、off-chip memory 与外设组成的通信系统抽象为新的 `server chiplet network substrate`。
  - 摘要写明：论文系统刻画了两代 `AMD EPYC chiplet servers`，识别出四类通信特性：更长的数据路径、异构带宽域、不一致的 bandwidth-delay product、以及 sender-driven aggressive bandwidth partitioning。
  - 论文把这些观察总结为后续系统与应用设计含义。
