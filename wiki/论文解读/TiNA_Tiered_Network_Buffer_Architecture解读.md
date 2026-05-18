# TiNA: Tiered Network Buffer Architecture for Fast Networking in Chiplet-based CPUs 解读

本文依据版本为 ASPLOS 2026 论文 `TiNA: Tiered Network Buffer Architecture for Fast Networking in Chiplet-based CPUs`，原始 PDF 位于 `wiki/原始资料/papers/TiNA Tiered Network Buffer Architecture for Fast Networking in Chiplet-based CPUs.pdf`。论文研究对象是 Intel Sapphire Rapids chiplet CPU 上的微秒级网络包处理，不是 LLM 推理或 attention kernel。核心问题是：Sub-NUMA Clustering（SNC，按 chiplet 划分 socket 内资源的 NUMA 模式）能降低本地 cache/DRAM 访问延迟，但会减少单个处理核心可用的 DCA cache 容量；长突发网络流量下，较小 DCA 容量会触发更多 DMA leak 和 DMA bloat，使 SNC 反而慢于 non-SNC（摘要，Sec. 1，Sec. 3）。

## 问题

Sapphire Rapids CPU 由最多 4 个 chiplet 组成。论文测试平台 Intel Xeon Gold 6414U 的每个 chiplet 包含 8 个 CPU core、8 个 1.875 MB LLC slice、2 个 DDR5-4800 DRAM controller 和 12 条 PCIe 5.0 lane，单个 chiplet 本身接近一个小型 CPU（Sec. 2.1）。

默认 non-SNC 模式把 4 个 chiplet 组成一个统一 socket。CPU core 可以访问所有 LLC slice 和 DRAM controller，地址跨 chiplet 交织，优点是可用 LLC 容量和 DRAM 带宽更大，缺点是访问会经过 on-package interconnect，延迟和方差更高。Fig. 1 给出的示意延迟为本地 LLC 约 25 ns，跨相邻 chiplet LLC 约 50 ns，更远 chiplet LLC 约 75 ns。

SNC 模式把单 socket 逻辑切成多个 sub-NUMA node，每个 chiplet 对 OS 表现为一个 NUMA node。默认情况下，某个 chiplet 内的 CPU core 只访问本 chiplet 的 LLC slice 和 DRAM controller，除非显式做 NUMA-aware allocation。SNC 的优势是访问延迟更低、方差更小；代价是该 core 可直接使用的 LLC 容量和 DRAM 带宽约为 non-SNC 的 1/4（Sec. 2.1）。

论文关注的是微秒级网络应用。每个 packet 的处理路径包含 NIC DMA、descriptor/mbuf 访问、CPU 读包、应用处理和回包。网络栈对几十 ns 到数百 ns 的内存访问差异敏感，chiplet 内外访问差异会放大到 packet round-trip latency（摘要，Sec. 1）。

## 背景：DCA 与 DPDK 包处理

### DCA 的作用

Direct Cache Access：网卡如果先把数据包写进内存（DRAM），CPU 再去读，太慢了。DCA（Intel 实现称 DDIO）允许 NIC 把收到的 packet 通过 DMA 写入专用 LLC ways，论文称这些 ways 为 DCA ways。若目标 buffer entry 已在 DCA ways 中，NIC 做 write-update；若不在，NIC 需要 write-allocate，并驱逐已有 cacheline（Sec. 2.2）。

DCA 的目标是减少 DRAM 访问和 packet processing latency。问题出现在 DCA ways 容量不足时：

1. **DMA leak**：被 NIC 驱逐的 cacheline 里仍有未处理 packet。后续 CPU 处理这些 packet 时必须从 DRAM 拉回，增加内存带宽和延迟（Sec. 2.2）。
2. **DMA bloat**：已处理 packet 所在 cacheline 从 MLC 被驱逐到 LLC 的任意 ways，与应用自身 working set 或新收到 packet 竞争 cache。论文指出这与现代 CPU 的 non-inclusive cache hierarchy 有关，会破坏 DCA ways 与 non-DCA ways 的隔离（Sec. 2.2）。

SNC 对 DCA 的关键影响是容量。论文测试系统中，SNC 模式下本地 DCA ways 容量为 2 MB；non-SNC 模式下可用 DCA ways 容量为 8 MB（Sec. 3.2）。

### DPDK pipeline 模型

Data Plane Development Kit（DPDK，用户态 kernel-bypass 网络框架）使用 descriptor buffer 和 mbuf 接收 packet。每个 descriptor 指向一个 mbuf；mbuf 通常是 2 kB 的 packet buffer，来自 1 GB huge page mempool（Sec. 2.3）。

论文区分两种 DPDK 模型（Fig. 2）：

1. **Run-to-Completion (RTC)**：处理 core 轮询 descriptor buffer，直接处理 descriptor 指向的 packet。长突发流量要求 descriptor buffer 足够大，否则丢包；大 buffer 会超过 DCA ways 和 MLC 容量，导致 DMA write miss 和 DMA bloat。
2. **Pipeline**：polling core 只把新 packet 的 mbuf 指针传给 processing buffer，processing core 独立处理。短突发下，LIFO 方式分配和释放 mbuf 可以让活跃 mbuf 集合保持在 DCA ways 和 MLC 内；长突发或高频突发下，活跃 mbuf 集合仍会膨胀。

论文后续使用 pipeline 模型，因为它对 DCA 更友好，但仍会在长突发下暴露 DCA 容量不足问题（Sec. 2.3）。

## 观察：SNC 的收益和退化条件

### 内存访问延迟

论文用 Intel Memory Latency Checker 在 Sapphire Rapids 服务器上测 pointer chasing（Fig. 3 left）。结果分为四段：

1. **working set <= 2 MB**：SNC 与 non-SNC 延迟相同，因为 working set 放入与 core 绑定的 MLC（Sec. 3.1）。
2. **2-15 MB**：SNC 比 non-SNC 低约 45% 或约 20 ns，因为 working set 放入单个 chiplet 的 LLC slices（Sec. 3.1，Fig. 3）。
3. **15-60 MB**：SNC 最多比 non-SNC 高约 100%，因为 SNC 下单个 chiplet 只有约 15 MB LLC 容量，超过后转向 DRAM；non-SNC 仍可使用 4 个 chiplet 聚合出的 60 MB LLC（Sec. 3.1，Fig. 3）。
4. **>= 60 MB**：两者都访问 DRAM，但 SNC 比 non-SNC 低约 20%，原因是 SNC 把访问限制在本 chiplet 的两个 DRAM controller（Sec. 3.1，Fig. 3）。

Fig. 3 right 测 DRAM 带宽压力对延迟的影响。低带宽时 SNC 与 non-SNC 延迟接近；当带宽消耗超过约 30 GB/s 后，SNC 延迟开始高于 non-SNC。论文解释为 SNC 只暴露 non-SNC 约 1/4 的 DRAM 带宽，DRAM controller 排队更快出现（Sec. 3.1）。

### 网络包处理延迟

论文用 DPDK microbenchmark `L2TouchFwd` 评估网络性能。该网络函数读取每个 packet 的完整 1 kB payload，修改 MAC header 后转发。实验使用 100 Gbps line rate 的周期性 burst，burst length 从 100 µs 到 800 µs，平均链路利用率只有 10%（Sec. 3.2）。

Fig. 4 显示：

- 约 100 µs burst 下，SNC 的 p50 和 p99 packet round-trip latency 比 non-SNC 低约 60%。
- burst length 增加到约 400 µs 时，SNC 和 non-SNC 延迟接近。
- 超过约 400 µs 后，SNC 延迟开始高于 non-SNC。

退化原因来自 active mbuf size（活跃 mbuf 总大小）。pipeline 模型中，未处理 packet 对应的 mbuf 会随 burst length 增长。burst length 小于约 450 µs 时，SNC 下 active mbuf size 增长较慢，且能放入 2 MB DCA ways；超过该范围后，active mbuf size 超过 SNC 的 DCA capacity，PCIe miss rate 上升，DMA leak 增多。non-SNC 有 8 MB DCA ways，容量更大，因此长 burst 下更晚触发 leak（Sec. 3.2，Fig. 5）。

Fig. 6 进一步说明 break-even point 依赖 processing rate。CPU 频率从 2.0 GHz 降到 1.5 GHz 后，packet 处理变慢，mbuf 生命周期变长，SNC 与 non-SNC 的延迟分界移动到更短 burst。论文强调，burst length 只是简化表达；多个间隔很小的短 burst 也能造成同样的 queue buildup（Sec. 3.2，Fig. 6）。

## 核心内容：TiNA 的两级网络 buffer

TiNA 是 Tiered Network Buffer Architecture，由增强 NIC 和增强网络栈组成，分别称为 TiNA-NIC 和 TiNA-stack。目标是在 active mbuf size 小时获得 SNC 的低本地 LLC 延迟，在 active mbuf size 大时使用远端 chiplet 的 DCA ways 扩大有效 DCA 容量，避免 DMA leak（Sec. 4）。

### Local-tier 与 Remote-tier

TiNA 在 SNC 模式下建立两类 network buffer（Fig. 7）：

- **Local-tier**：由 processing core 所在本地 chiplet 的 memory region 支撑，DMA 写入后缓存在本地 DCA ways。
- **Remote-tier**：由其余 `N - 1` 个远端 chiplet 的 memory region 支撑，DMA 写入后缓存在远端 chiplet 的 DCA ways。

这一设计利用了 SNC 的一个属性：属于某个 chiplet 的 memory region 只会被缓存到该 chiplet 的 LLC slices。TiNA 因此可以通过选择 descriptor buffer 所在的 memory region，控制 packet 进入 local DCA ways 还是 remote DCA ways（Sec. 4.1）。

### descriptor buffer 与硬件队列

传统网络栈对每个 packet processing core 使用一个 descriptor buffer。TiNA-stack 为每个 processing core 分配 `N` 个 descriptor buffers，每个 chiplet 一个：本地 descriptor buffer 组成 Local-tier，其余 descriptor buffers 组成 Remote-tier（Sec. 4.2）。

TiNA-NIC 依赖 NIC 的多 hardware queue 和 Receive-Side Scaling（RSS，按 packet header hash 把 packet 分发到接收队列的机制）。每个 descriptor buffer 对应一个 hardware queue，因此每个 processing core 需要 `N` 个 hardware queues。论文指出 ConnectX-5 及后续 NIC 支持超过 512 个 hardware queues，在 4 chiplet 设置下可覆盖 128 个 processing cores（Sec. 4.2）。

## 方法：packet placement policy

### 基本策略

TiNA 的 placement policy 根据 active mbuf size 决定 packet 写入 Local-tier 或 Remote-tier。记号如下（Sec. 4.3）：

- `A_local`：Local-tier 中 active mbuf size。
- `A_remote`：Remote-tier 中 active mbuf size。
- `P`：当前收到的一批 packet 大小。
- `D_local`：Local-tier DCA ways 容量。
- `D_remote`：Remote-tier DCA ways 容量。
- `C`：packet consumption rate，由 TiNA-stack 周期性上报。

TiNA-NIC 默认从 local 状态开始。若 `A_local + P <= D_local`，packet 放入 Local-tier；若 `A_local + P > D_local`，切到 remote 状态，packet 放入 Remote-tier。基本策略只有在 Local-tier 中 packet 先处理完、Remote-tier 中 packet 再处理完后，才回到 local 状态，以保证 packet 处理顺序与接收顺序一致（Sec. 4.3）。

`P` 由 NIC 直接测得；`A_local` 与 `A_remote` 需要估算。TiNA-NIC 在接收 packet 时按 placement 决策增加对应 `A`；TiNA-stack 每 `I` 秒向 TiNA-NIC 上报平均 consumption rate `C`，TiNA-NIC 再按 `C` 周期性递减 `A_local` 或 `A_remote`。若 `A_remote` 非零，TiNA-NIC 按处理顺序先递减 `A_local`，再递减 `A_remote`（Sec. 4.3）。

### 顺序保证与高级策略

基本策略会错过两个机会（Sec. 4.4）：

1. **C1**：`A_local = D_local` 且 `A_remote + P > D_remote`。长 burst 下 remote tier 也将溢出，DMA leak 无法避免。此时把 packet 放到 Local-tier 更合适，因为 local DRAM 访问延迟低于 remote DRAM。
2. **C2**：`A_local < D_local` 且 `A_remote > 0`。CPU 已处理一部分 Local-tier packet，本地 DCA ways 出现空闲。此时可机会性切回 Local-tier 利用本地空余容量。

这两个优化会让后到的 packet 重新进入 Local-tier，因此不能再简单按 tier 顺序处理。TiNA-stack 使用 sequence number 保序：比较 Local-tier 与 Remote-tier 队头 packet 的 `S_local` 和 `S_remote`，处理 sequence number 更小的 packet。若传输协议已有 TCP sequence number，TiNA 直接使用；否则 TiNA-NIC 添加自己的 sequence number（Sec. 4.4，Algorithm 1）。

Algorithm 1 给出完整策略：placement 侧在 local 溢出时切 remote；remote 状态下若 remote 也将溢出，或 `A_local + P` 低于略小于 `D_local` 的阈值 `D_local'`，切回 local。processing 侧根据 tier 是否为空以及 `S_local`、`S_remote` 的大小选择处理 tier。`D_local'` 是抑制 local/remote 频繁振荡的 hysteresis threshold（Sec. 4.4，Algorithm 1）。

## 实现

### TiNA-NIC

论文把 TiNA-NIC 设想为 commodity NIC RSS 逻辑的增强。原型实现使用 Xilinx U280 FPGA，以 bump-in-the-wire 方式连接 ConnectX-6 NIC 和服务器。该原型约 500 行 Verilog，资源开销约 1k LUTs 和 2k registers，分别远低于 U280 的 1M LUTs 和 2M registers（Sec. 5.1）。

TiNA-NIC 的新增操作包括：

1. 对没有可靠传输协议 sequence number 的 packet 添加 sequence number。
2. 根据 Algorithm 1 的 `PlacementTier` 选择 Local-tier 或 Remote-tier。
3. 通过增强 RSS 逻辑把 packet 导向对应 hardware queue。

对于每个 processing core，TiNA-NIC 使用 1 个 Local-tier hardware queue 和 `N - 1` 个 Remote-tier hardware queues。传统 5-tuple hash 先选择 logical hardware queue；TiNA-NIC 再用内部 bit 选择 local 或 remote，并用 sequence number 的 `ceil(log2(N - 1))` 个 bit 在远端 queues 间均衡。论文使用 sequence number 高位，例如第 5-7 位，把连续 packet 批量导向同一 remote queue，减少 `rte_eth_rx_burst` 返回 packet 时的乱序和 TiNA-stack 检查 sequence number 的开销（Sec. 5.1）。

### TiNA-stack

TiNA-stack 的改动限定在 DPDK library 内，不改变 DPDK API。它修改 `rte_eth_rx_burst`，让应用仍看到类似单 descriptor buffer 的接口，但内部会轮询 `N` 个 descriptor buffers，并按 sequence number 保序交付 packet（Sec. 5.2）。

为降低保序开销，TiNA-stack 只在两个以上 descriptor buffers 都有待处理 packet 时查看队头 sequence number；随后持续从 sequence number 更小的 buffer 批量处理，直到遇到超过另一个 buffer 已记录 sequence number 的 packet，再重新选择 buffer（Sec. 5.2）。

TiNA-stack 还负责估算 consumption rate。它记录应用调用 `rte_eth_rx_burst` 取到 packet 的时间，以及调用 `rte_pktmbuf_free` 释放 mbuf 的时间，按 processed bytes 计算平均 `C`。每 `I` 秒计算 weighted moving average：`C_wma = 1/8 * C + 7/8 * C_wma`，并通过 posted MMIO write 发送给 TiNA-NIC。论文评估中 `I = 100 µs` 对 microbenchmark 和三类端到端应用都有效，单次上报开销为几百 cycles，最多小于 400 ns，并由 DPDK service core 处理（Sec. 5.2）。

## 实验设置

### 平台

评估平台为 Intel Xeon Gold 6414U、8 个 DDR5-4800 DRAM modules、Mellanox ConnectX-6 100GbE NIC、Ubuntu 22.04 LTS。服务器通过 FPGA 与一台 load generator client 相连，FPGA 同时用于 TiNA-NIC 原型和 packet timestamp（Sec. 6.1）。

### 应用

论文评估 4 类应用（Sec. 6.1）：

- **L2TouchFwd**：读取 1 kB payload，修改 MAC header 后转发，代表 deep packet access 的简单转发函数。
- **NAT**：Network Address Translation，只处理 header，代表 shallow network function。
- **RSA**：对 payload 做 RSA 相关处理，代表处理更重的 deep network function。
- **KVS**：Key-Value Store，代表 client-serving application，有更大的非 I/O working set。

### workload 与指标

客户端生成 1 kB payload packet。论文说明 1 kB 是 datacenter 中两类主要 payload size 之一，并指出 TiNA 对 payload size 不敏感，因为 placement 主要根据 burst 内收到的字节数 `P` 决策（Sec. 6.1）。

端到端实验使用来自 hyperscaler 微突发分布的三类 traces。T1 与 T2 对应 user-facing applications，T3 对应 batch job。Fig. 8 显示 active mbuf size 可达到 20 MB，且根据应用 consumption rate 和 trace 不同，active mbuf size 超过 local DCA ways 2 MB 的时间占比约 20%-50%（Sec. 6.1，Fig. 8）。

指标是 per-packet latency：从服务器 NIC 首次收到 packet 到服务器发出 response packet 的时间，包含 NIC receive data path、DMA、host processing 和 transmit data path。论文用实现 TiNA-NIC 的同一 FPGA 打 timestamp。每条 trace 固定运行 1 分钟，收集 30M 到 120M packet samples（Sec. 6.1）。

## 结果与解释

### microbenchmark：固定 burst length

Fig. 9 显示，TiNA 在所有测试 burst length 下都与 SNC/non-SNC 中较优者接近或更低。具体表现为：

- burst length 小于 400 µs 时，SNC 优于 non-SNC，TiNA 匹配或略优于 SNC（Sec. 6.2，Fig. 9）。
- 250-400 µs 区间，TiNA 比 SNC 的 p99 latency 低约 8%-10%，原因是 local DCA ways 即将填满时，TiNA 可动态使用 remote DCA ways，减少 DMA leak（Sec. 6.2，Fig. 9）。
- 400-700 µs 区间，non-SNC 优于 SNC，TiNA 比 non-SNC 的 p99 latency 低最多约 5%-10%，原因是 TiNA 仍优先使用 local DCA ways，同时避免过早发生 DMA leak（Sec. 6.2，Fig. 9）。
- 极短和极长 burst 下，TiNA 相比 SNC 或 non-SNC 有不超过 2% 的 p99 overhead，来源是 TiNA-stack 轮询多个 descriptor buffers 的额外开销；这些极端点 TiNA 本身没有额外可利用空间（Sec. 6.2，Fig. 9）。

Fig. 10 测固定 100 µs burst、按 Poisson arrival process 增加网络负载。SNC 在超过 38 Gbps 后出现 latency inflation 和 packet drop；non-SNC 超过 43 Gbps 后才出现。TiNA 使用 remote DCA ways 后，DMA leak 和 packet drop 出现得与 non-SNC 一样晚，同时在不可避免 leak 前保持低于 non-SNC 的 p99 latency（Sec. 6.2，Fig. 10）。

### 端到端应用与 datacenter traces

摘要报告的总体结果是：跨多种网络应用和 traces，TiNA 相比 SNC 平均降低 mean latency 25%、tail latency 18%；相比 non-SNC 平均降低 mean latency 28%、tail latency 22%（摘要，Sec. 1）。

Fig. 11 显示，L2TouchFwd、NAT、KVS、RSA 在 T1/T2/T3 上，TiNA 的 p50 和 p99 latency 均与 SNC/non-SNC 中较优者接近或更低。论文按应用解释如下：

- **L2TouchFwd**：TiNA 的 p99 latency 相比 SNC 和 non-SNC 均低约 15%-25%。p50 latency 低于 100 µs，对应 active mbuf size 小于约 700 kB，因此 TiNA 更接近 SNC；p99 latency 超过 4000 µs，对应 active mbuf size 大于 28 MB，SNC 和 non-SNC 都发生大量 LLC miss，TiNA 通过让 Remote-tier packet 仍能获得 LLC hit 降低 p99（Sec. 6.3，Fig. 11，Fig. 12a）。
- **KVS**：TiNA 的 p99 latency 降幅最大，最高约 50%-55%。论文解释为 KVS 有较大的非 I/O working set，受 DMA bloat 影响更明显；TiNA 同时利用 non-SNC 的 DCA 容量优势和 SNC 的低 core-to-LLC 延迟（Sec. 6.3，Fig. 11，Fig. 12b）。
- **NAT**：TiNA 性能接近 SNC，相比 non-SNC 在 p50 和 p99 上约有 20% 收益。Fig. 12c 显示 NAT p99 latency 仍小于 10 µs，说明 active mbuf size 很小；NAT 每 packet 只访问单个 cacheline，consumption rate 高，因此 SNC 已接近最优（Sec. 6.3，Fig. 12c）。
- **RSA**：TiNA 的 p99 改善小于 L2TouchFwd。论文解释为 RSA 造成更大的 active mbuf size，p99 latency 约为 L2TouchFwd 的 1.7 倍；active mbuf size 过大时，SNC、non-SNC 和 TiNA 都会遇到不可避免的 DMA leak，TiNA 放入 Remote-tier 并避免 leak 的 packet 占比下降（Sec. 6.3，Fig. 12d）。

### 与非网络应用共跑

论文还评估 TiNA 与非网络 workload 共跑。网络 workload 为 L2TouchFwd 800 µs bursts；32-core 系统中 2 cores 跑 L2TouchFwd，其余 30 cores 跑 SPEC CPU2017 memory-intensive benchmarks 或 DCPerf web-serving workloads。作者先验证非网络应用对网络应用性能影响接近 0，原因是 CAT 对 networking traffic 的 DCA ways 做了隔离（Sec. 6.4）。

Fig. 13 显示，TiNA 与 SNC 的 SPEC 性能接近，平均比 non-SNC 快约 7%；DCPerf web-serving benchmarks 对 SNC/non-SNC 不敏感。TiNA 使用 Remote-tier DCA ways 未降低非网络应用性能（Sec. 6.4，Fig. 13）。

## 设计取舍

### active mbuf size 不能只在软件侧计算

论文讨论了由 TiNA-stack 精确计算 `A_local` 和 `A_remote` 后通过 MMIO 发给 NIC 的方案。作者认为该方案会过慢：TiNA-NIC 必须在 packet 到达 NIC 后立即决定 hardware queue；但 TiNA-stack 只有在 packet 已经 DMA 到内存后才能感知该 packet。NIC receive data path 的排队可能达到几十 µs，软件侧精确值到达 NIC 时已错过 placement 时机（Sec. 7）。

### 原始 RSS 不能替代 TiNA placement

未修改 RSS 可按 5-tuple 静态把 flow 分到 Local-tier 或 Remote-tier，但它不感知 active mbuf size。active mbuf size 小时，RSS 可能仍把 flow 放到 Remote-tier，浪费 local DCA capacity；active mbuf size 大且存在 elephant flow 时，RSS 会把该 flow 固定到单个 chiplet，其他 chiplet DCA ways 空闲也无法利用。TiNA 可随 active mbuf size 切换 tier，并可把单个 flow 拆到 Local-tier 和 Remote-tier（Sec. 7）。

### TiNA 默认运行在 SNC 模式

论文没有选择 non-SNC 作为 TiNA 默认模式。原因是 non-SNC 下 memory access 以 256-byte 粒度跨 chiplet 交织。若 packet 大于 256 bytes，TiNA 需要让一个 tier 中的 descriptor buffer 和 mbuf 横跨多个非连续 memory regions，导致 split-DMA transaction。论文评估说明 split-DMA 会带来明显 CPU 和 latency overhead（Sec. 7）。

## 分析

- TiNA 的核心判据不是固定选择 local 或 remote，而是根据 active mbuf size 在低延迟本地 DCA capacity 与更大聚合 DCA capacity 之间切换（Sec. 4.3，Algorithm 1）。
- SNC 对短 burst 有利，对长 burst 不一定有利。短 burst 受益于 local LLC/DRAM 低延迟；长 burst 的 active mbuf size 超过 local DCA ways 后，DMA leak 和 DMA bloat 会抵消甚至反转 SNC 的低延迟收益（Sec. 3.2，Fig. 4，Fig. 5）。
- TiNA 的 tiering 与排序论文中的 Chiplet_Local/Chiplet_Mixed 有相似方法论：工作集在单 chiplet cache 容量内时优先本地；超过本地容量时利用跨 chiplet 聚合容量。但 TiNA 的对象是 DCA ways 中的 packet mbufs，不是通用 LLC 工作集（Sec. 4.1，Fig. 7）。
- 论文强调 packet placement 必须发生在 NIC receive path 中。该约束来自网络 I/O 的实时性，与 CPU 内部算子调度不同（Sec. 7）。

## 边界

- 硬件平台是 Intel Sapphire Rapids，关键机制是 SNC、DDIO/DCA、CAT、RSS、多 hardware queue NIC；不能直接映射到 AMD EPYC Bergamo/Genoa 的 CCD/L3/IO die 行为。
- TiNA 依赖 NIC 可按 packet 动态选择 descriptor buffer，并依赖 DCA ways 的容量与 cache placement 语义；当前 vLLM CPU attention 没有 NIC DMA、DCA ways、mbuf、RSS 或 packet ordering 问题。
- 实验 workload 是 100 Gbps 网络包处理，payload 主要为 1 kB，应用包括 L2TouchFwd、NAT、RSA、KVS；不覆盖 LLM prefill/decode、KV cache streaming 或 OpenMP attention task 调度。
- TiNA 的原型 NIC 逻辑由 FPGA bump-in-the-wire 实现，不是 commodity NIC 原生实现。论文说明资源开销很小，但仍属于原型验证（Sec. 5.1）。
- 论文中的阈值依赖该平台 DCA capacity：SNC 2 MB、non-SNC 8 MB。不能把 2 MB、8 MB 或 400 µs break-even 直接作为其他 CPU/NIC/网络栈的通用阈值（Sec. 3.2，Fig. 4，Fig. 5）。

## 可迁移点

- chiplet CPU 上的本地低延迟与跨 chiplet 聚合容量存在条件化取舍。优化策略需要先估计当前活跃工作集是否超过本地 cache/tier 容量，再决定是否使用远端容量（Sec. 3.1，Sec. 4.3）。
- 动态策略优于静态拓扑绑定。TiNA 不把 packet 永久固定到 local 或 remote，而是随 active mbuf size 和 consumption rate 调整 placement（Sec. 4.3，Algorithm 1）。
- 衡量局部性收益时必须同时检查容量溢出路径。SNC 在短 burst 下更快，但当 active mbuf size 超过 local DCA ways 后，locality 优势被 DMA leak/bloat 反转（Sec. 3.2，Fig. 5）。
- 负载速率影响局部性策略分界。Fig. 6 表明 processing rate 下降会让 active mbuf 生命周期变长，使 break-even point 提前；对应到其他 workload 时，不能只按输入规模判断，还要看消费速度。

## 不可直接迁移点

- TiNA 的 Local-tier/Remote-tier 是网络 buffer tier，不是普通线程任务池或 attention KV cache 分片策略。
- TiNA 的收益依赖 DCA write-hit、DMA leak、DMA bloat 这些 I/O cache path 机制；CPU attention 主要是 CPU load path，不存在 NIC DMA 写入 DCA ways。
- TiNA 的保序机制依赖 packet sequence number；attention kernel 没有 packet arrival order 与 descriptor buffer 多队列交付问题。
- 当前 vLLM CPU attention 的已测结论显示，按 `kv_head` 静态限制到本地 L3 分组会降低跨组补空能力，长序列退化明显。TiNA 不能作为“固定本地化一定有效”的证据，只能作为“本地低延迟与聚合容量需要动态切换”的方法启发。

## 证据

- 原始资料：`wiki/原始资料/papers/TiNA Tiered Network Buffer Architecture for Fast Networking in Chiplet-based CPUs.pdf`
- 关键锚点：摘要；Fig. 1-13；Algorithm 1；Sec. 2.1-2.3；Sec. 3.1-3.2；Sec. 4.1-4.4；Sec. 5.1-5.2；Sec. 6.1-6.4；Sec. 7；Artifact Appendix
