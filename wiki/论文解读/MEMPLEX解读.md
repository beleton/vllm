# MEMPLEX: A Memory System with Replication and Migration of Data for Multi-Chiplet NUMA Architectures 解读

本文依据版本为 ICS 2025 论文 `MEMPLEX: A Memory System with Replication and Migration of Data for Multi-Chiplet NUMA Architectures`，原始 PDF 位于 `wiki/原始资料/papers/MEMPLEX.pdf`。论文研究对象是多 chiplet NUMA 系统中的 HBM/DDR 主存组织，不是 CPU L3 调度。核心问题是：多 chiplet 系统中即使使用 NUMA-aware 数据放置，远端 HBM 和外部 DDR 访问仍会造成显著性能损失；现有商用系统通常只能把 HBM 全部作为 flat address space，或全部作为外部 DDR 的 DRAM cache，缺少在多个 HBM node 之间动态复制和迁移数据的硬件机制（摘要，Sec. 1，Sec. 2.2）。

## 问题

论文关注的系统由多个 CPU chiplet、多个 HBM node 和外部 DDR 组成。每个 NUMA node 由一个 processor chiplet 和一个本地 HBM 组成。某个 chiplet 访问自己的 HBM 时是 Local Memory（LM，本地内存）；访问其他 chiplet 的 HBM 或外部 DDR 时是 Remote Memory（RM，远端内存）（Fig. 1，Sec. 3）。

在这种系统中，HBM 既有容量价值，也有带宽和延迟优势。若把 HBM 全部作为 flat address space，软件或 OS 需要负责数据放置；数据放置不理想时会产生远端 HBM 或 DDR 访问。若把 HBM 全部作为 DRAM cache，可以减少远端访问，但会牺牲 HBM 作为主存容量的价值（Sec. 1，Sec. 2.2）。

论文用理想系统作为上界：Ideal 系统假设每次 LLC miss 都由最近 HBM channel 服务，相当于本地 HBM 容量无限。与 NUMA-aware baseline 相比，4-chiplet 系统的 Ideal 性能高 26%，16-chiplet 系统高 31%。这说明仅靠 NUMA-aware 分配仍有明显远端访问开销（摘要，Sec. 1，Fig. 7，Fig. 9）。

## 背景：复制、迁移与混合内存

### COMA 与 ccNUMA

Cache-Coherent NUMA（ccNUMA，缓存一致的非统一内存访问）允许远端数据进入本地 cache hierarchy，但远端数据的 home node 仍负责页面初始分配和一致性维护。由于远端 cache 容量有限，ccNUMA 对数据放置敏感（Sec. 2.1）。

Cache Only Memory Architecture（COMA，把主存组织成可迁移缓存的体系结构）允许远端页面自由迁移到本地内存，提高本地命中概率。传统 COMA 没有固定 home node，miss 后定位数据复杂；FLAT-COMA 通过固定目录 home node 解决定位问题；S-COMA 把部分复杂性转移给 OS（Sec. 2.1）。

MEMPLEX 借鉴的是 COMA 的数据迁移思想和 DRAM cache 的复制思想，但目标平台不是传统多 socket NUMA，而是多个 processor chiplet + 多个 HBM node + 外部 DDR 的 chiplet NUMA 系统。

### Hybrid Memory 的两类路径

混合内存系统通常结合高带宽小容量 HBM 和大容量低带宽外部 DRAM。论文归纳了两类路径（Sec. 2.2）：

1. **HBM + DDR 共同作为 flat address space**：通过软件或硬件把 hot data 迁移到 HBM。
2. **HBM 作为 DRAM cache**：HBM 缓存外部 DDR 内容，主要挑战是 tag metadata 管理开销。

Hybrid2 将一部分 HBM 用作 cache，剩余 HBM 作为 main memory。MEMPLEX 扩展了这条思路：目标从单处理器 chip + 两级 HBM/DDR，变成多个 processor chiplet、多个 HBM node 和外部 DDR。因此，MEMPLEX 需要处理分布式 remap metadata、跨 NUMA node 的迁移决策，以及多个私有 DRAM cache 与目录一致性协议的兼容问题（Sec. 2.2）。

## 核心内容

MEMPLEX 是一个硬件内存系统。它把每个 HBM node 逻辑划分成两部分（Fig. 1，Fig. 2）：

- 一小部分作为该 chiplet 私有的 sectored DRAM cache（分 sector 管理的 DRAM cache）。
- 剩余 HBM 容量与外部 DDR 共同组成 shared flat address space，并支持硬件数据迁移。

DRAM cache 的目标是把本 chiplet 经常访问的数据吸引到 LM。迁移机制的目标是在 DRAM cache eviction 时，根据 sector 使用情况和迁移带来的远端流量开销，决定该 sector 是否迁移到 LM（摘要，Sec. 3.1，Sec. 3.7）。

论文中的数据粒度分为两层（Sec. 3.1）：

- **Cache line 粒度**：DRAM cache 按 64 B cache line 抓取数据。
- **Sector 粒度**：DRAM cache tag 和迁移决策按 sector 管理，论文为简化把一个 sector 设为一个 OS page，即 4 KB。

这意味着一次 LLC miss 只需要抓取请求的 cache line，但 tag、valid/dirty 状态、访问计数和迁移决策都按 4 KB sector 维护。

## 方法与系统设计

### NUMA node 与 LM/RM

MEMPLEX 把每个 processor chiplet + HBM 视为一个 NUMA node。某个 node 的本地 HBM 是 LM；其他 HBM 和外部 DDR 是 RM。外部 DDR 被逻辑划分成与 node 数量相同的区域，每个区域分配一个 Memory Node Identifier（MNID），使其能进入同一套 remap 和迁移机制（Fig. 1，Sec. 3.2）。

该设计保留大部分 HBM 作为主存容量，同时通过每 node 私有 DRAM cache 复制远端数据，并在 eviction 时决定是否把数据迁移到 LM。

### DRAM Cache Controller 与 DCTA

每个 processor chiplet 内有一个 DRAM Cache Controller（DCC，DRAM cache 控制器）。DCC 负责处理 processor memory request、访问片上 tag、处理 miss、eviction、dirty writeback，以及管理 sector migration（Sec. 3.2）。

DCC 维护 DRAM Cache Tag Array（DCTA，DRAM cache tag 数组）。DCTA 存在 processor chiplet 的 SRAM 中，按 set-associative 组织。每个 DCTA entry 包含（Fig. 1，Fig. 2，Sec. 3.2）：

- sector tag 和 cache state。
- 每个 cache line 的 valid bit 与 dirty bit。
- Access Counter（AC，访问计数器），用于 eviction 时判断 sector 是否值得迁移。
- Cache Pointer（CP），指向 sector 在 LM 中的 cache/data 位置。
- Memory Pointer（MP），指向 sector 在 RM 中的主存位置；若 sector 属于 LM 或已完全迁移到 LM，MP 与 CP 相同。
- Memory Node Identifier（MNID），标识 sector 当前主存位置所在 node。

DCTA 不只是 cache tag，也缓存 remap metadata。由于所有 memory request 都先经过 DCTA，DCTA 中保存 CP、MP 和 MNID 可以减少 remap table lookup 的开销（Sec. 3.1，Sec. 3.2）。

### Metadata 结构

MEMPLEX 在每个 memory node 中维护三类迁移元数据（Sec. 3.3）：

1. **Remap Table**：记录 processor physical address 到实际 memory location 的映射。它既记录本 node 原生 sector 迁移到其他 node 的情况，也记录其他 node 的 sector 迁移到本地 HBM 的情况。论文把 remap table 设计为 hash table；若本地原生 sector 在 remap table miss，则默认该 sector 仍在原生位置。
2. **Inverted Remap Table**：按 memory node 内的实际位置反查 processor physical address，并保存 cached sector 的 sharer bitmap。该表用于把 sector 从某个 memory node 迁出时更新映射。
3. **Free Memory Stack**：保存本 node 当前可用的空闲 sector location。每个 node 还为其他 memory node 预留部分 free entries，用于远端 node 迁入数据。

论文报告上述 metadata 的空间开销较小，即使考虑完整 remap table，也只占 memory node capacity 的 0.5%（Sec. 3.3）。

### Memory access path

LLC miss 到达请求 node 的 DCC 后，DCC 用 physical address 查询 DCTA，判断 sector 和请求 cache line 是否在 DRAM cache 中。论文把路径分成四类（Fig. 3，Sec. 3.4）：

1. **DCTA miss，sector 在 LM**：DCC 分配新的 DCTA entry，CP 和 MP 都指向 LM 中该 sector 的位置，MNID 设为 self ID，所有 cache line 标记为 valid 和 dirty。这样避免复制已经在 LM 的数据，DCTA 只作为该 sector 的 remap/cache metadata。
2. **DCTA miss，sector 在 RM**：DCC 先查 LM remap table；若查不到，再查该 physical address 的 Home Node remap table 或 RM node remap table，定位 sector 当前实际位置。随后在 LM 中分配 DRAM cache space，只把请求 cache line 从 RM 拉到 LM，并更新 DCTA、valid/dirty bit、CP、MP、MNID 和 inverted remap table。
3. **DCTA hit，但请求 cache line 不在 DRAM cache**：sector entry 已存在，但该 cache line 的 valid bit 为 false。DCC 用 MP 从 RM 拉取该 cache line，再用 CP 写入 LM 中的 DRAM cache 位置。
4. **DCTA hit，cache line 在 DRAM cache**：DCC 直接通过 CP 从 LM 读取请求 cache line。此时 sector 可能原本属于 LM，也可能来自 RM，但被请求的 cache line 已在本地 DRAM cache。

关键点是：DCTA miss 不等于必须复制完整 4 KB sector；对 RM sector，初始只拉取请求 cache line。完整 sector 是否迁移到 LM，要等 DRAM cache eviction 时再判断（Sec. 3.4，Sec. 3.6）。

### Local Memory 中的 sector 分配

当请求 sector 位于 RM 且发生 DCTA miss 时，DCC 需要在 LM 中为该 sector 分配位置。流程包括（Fig. 4，Sec. 3.5）：

1. 在 LM 中选择 victim sector。
2. 从最近 RM 的 Free Memory Stack 中找到空闲 sector。
3. 把 LM victim sector 复制到 RM 的空闲位置。
4. 更新 LM、RM 或 Home Node 中的 remap table 和 inverted remap table。

LM victim 的选择使用 FIFO counter。DCC 通过 inverted remap table 找到 counter 对应 sector 的 physical address，再查询 DCTA；若该 sector 当前在 DRAM cache 中，则跳过，继续寻找下一个可用 sector。论文指出这比纯 FIFO 更合理，因为频繁访问的 sector 更可能仍在 DRAM cache 中，不会被迁出 LM。为降低关键路径延迟，每个 DCC 还维护少量 spare unused DC data entries，例如两个可立即使用的位置（Sec. 3.5.1）。

### DRAM cache eviction 与迁移决策

DCC 在 DRAM cache eviction 时用 LRU 选择被驱逐的 DCTA entry。被驱逐 sector 分为三类（Fig. 5，Sec. 3.6）：

- sector 已在 LM。
- sector 已迁移到 LM。
- sector 位于 RM，仅部分或全部 cache line 被拉到 LM 的 DRAM cache。

前两类不需要数据搬移；相关 remap metadata 已在迁移或首次 cache 时更新。第三类需要在两个动作中选择（Sec. 3.6.2，Sec. 3.7）：

1. **Evict back to RM**：只把 dirty cache line 写回 RM。
2. **Migrate to LM**：把未 valid 的 cache line 从 RM 拉到 LM，使完整 sector 常驻 LM；同时把 LM 中的 victim sector 迁出到 RM，并更新 remap metadata。

论文用远端访问次数衡量迁移开销。设一个 sector 中总 cache line 数为 `Nall`，valid cache line 数为 `Nvalid`，dirty cache line 数为 `Ndirty`。Eviction 的远端访问为 `ERM = Ndirty`；Migration 需要拉取 `Nall - Nvalid` 个 missing cache line，并把 LM victim sector 的 `Nall` 个 cache line 写到 RM。论文给出的迁移开销为（Sec. 3.7.1）：

```text
Om = 2 * Nall - Nvalid - Ndirty + 1
```

`Om` 的范围从 1 到 `2 * Nall`。当 sector 的 cache line 大多 valid 且 dirty 时，迁移增量开销低；当只取过少量 clean cache line 时，迁移开销高。

### 远端流量调节

DCC 维护一个 remote access counter，用于控制 migration traffic 不抢占过多 processor request 带宽。该 counter 在每次 DRAM cache miss 需要从 RM 拉取数据时递增；当 sector 被迁移时，counter 按 `Om` 递减。DCC 还检查 DCTA entry 的 AC，要求该 sector 的访问计数相对同 set 其他 sector 足够高，才考虑迁移（Sec. 3.7.2）。

迁移判据是：`Om` 小于 remote access counter，且 AC 检查通过。remote access counter 每 100K cycles 重置一次，以适应 workload phase 变化。这使 MEMPLEX 不在每次 miss 时立即迁移，而是在 eviction 点按“使用热度 + 迁移流量预算”做决策（Sec. 3.7.2）。

### Cache coherence

MEMPLEX 的每个 memory node 都有私有 DRAM cache，且一个 page 可能同时存在于多个 DRAM cache 中，因此需要 cache coherence。论文没有给出完整一致性协议实现和评估，而是说明 MEMPLEX 与目录式协议兼容，例如 CANDY。目录可放在 memory node 的 DRAM 中，并在 processor chiplet 上维护 SRAM directory cache 以降低访问延迟（Sec. 3.9）。

论文明确把 cache coherence 优化和 multi-threaded workload 评估留作 future work。当前评估使用 multi-programmed workloads，因此不能把结果视为共享数据多线程程序上的完整一致性开销评估（Sec. 3.9）。

## 实验设置

### 模拟平台

论文使用 BZSim 进行微架构模拟。BZSim 基于 ZSim，并集成 BookSim2 做 cycle-accurate 的 intra-chiplet 和 inter-chiplet network 建模；DRAMSim3 用于 DRAM 建模，CACTI 用于 cache access time 估算（Sec. 4.2）。

由于详细模拟开销高，论文把系统缩小到真实 chiplet 的 1/4。默认配置是 4 个 chiplet，每个 chiplet 4 个 out-of-order cores，频率 3.2 GHz。每 chiplet 有 1 GB HBM2，4 个 HBM channels，128-bit/channel；外部 DDR4 为 4 GB，单 channel。L3 为每 core 1 MB、16-way，默认 4 MB L3 时访问延迟 12 cycles（Tab. 1，Sec. 4.1）。

系统把所有 HBM 和外部 DDR 都视为 unified flat address space。NUMA-aware allocation policy 按距离分配页面：页面优先分配到首次访问它的 chiplet 所在 HBM；若最近 HBM 不足，再分配到邻近 HBM 或外部 DDR（Sec. 4.2）。

### Workloads

论文使用 multi-programmed workloads，不是单个多线程程序。workload 来自 SPEC CPU2017、GraphBIG、GUPS Random Access 和 XSBench。SPEC CPU2017 和 GraphBIG 使用 SimPoint 选择 10 亿指令代表片段。共选择 21 个 workload，并随机组成 multi-programmed mixes，每个 core 映射一个 benchmark（Tab. 2，Sec. 4.3）。

每个 mix 的总 memory footprint 至少 7 GB，geomean LLC MPKI 至少 11。对于 32-core 和 64-core 系统，论文把 16-application mix 复制 2 次或 4 次。每个实验平均 warm-up 125M instructions per core，随后详细模拟 250M instructions per core（Sec. 4.3）。

### 对比系统

论文评估 4 个系统（Sec. 4.4）：

| 系统 | 含义 |
|------|------|
| Baseline (BS) | 多 chiplet 系统，private LLC，NUMA-aware data placement，无 DRAM caching 或 migration |
| DRAM Cache-Only (CO) | 每个 chiplet 把整个本地 HBM 作为私有 DRAM cache；外部 DDR 容量增大，以容纳 workload mix |
| MEMPLEX (MP) | NUMA-aware data placement；每个 HBM 的一部分作为私有 DRAM cache，其余作为 main memory，并支持硬件 migration |
| Ideal (IL) | 理想系统；每次 LLC miss 总能由最近 HBM 服务，等价于本地 HBM 容量无限 |

默认 MEMPLEX 使用 1:16 的 DRAM cache to main memory ratio，即每个 HBM 的 1/16 用作 DRAM cache（Fig. 7，Sec. 5.1）。

## 结果与解释

### 性能与 AMAT

Fig. 7(a) 显示，在默认 4-chiplet 系统上，MEMPLEX 相比 baseline 的 IPC speedup 为 3%-7%，平均 5%。DRAM Cache-Only 平均只比 baseline 快 1%，且不同 mix 表现不稳定。Ideal 比 baseline 快 26%（Fig. 7，Sec. 5.1）。

Fig. 7(b) 显示，MEMPLEX 的 Average Memory Access Time（AMAT，平均内存访问时间）平均降低 10%，归一化到 baseline 后平均约为 0.90。CO 平均约为 0.92，Ideal 约为 0.74。论文把 MEMPLEX 的性能收益解释为远端访问减少带来的 AMAT 下降（Fig. 7，Sec. 5.1）。

Fig. 7(c) 给出访问来源。Baseline 平均 39% 数据访问是远端访问，其中 10% 到 remote HBM，29% 到 external DDR。CO 把远端访问降到 14%，全部是 external DDR。MEMPLEX 使 90% 数据访问落在 local HBM，只剩 10% 访问 remote memory（Fig. 7，Sec. 5.1）。

### Memory traffic

Fig. 8(a)(b) 显示，CO 相比 baseline 增加 55% local memory traffic，同时减少 62% remote traffic。MEMPLEX 增加 58% local memory traffic，同时减少 80% remote traffic，优于 CO 的 remote traffic reduction（Fig. 8，Sec. 5.2）。

论文指出 MEMPLEX 仍然会访问 external DDR，因此没有达到 Ideal。但它只牺牲 HBM 的 1/16 作为 cache，而 CO 把整个 HBM 用作 cache。论文据此认为 MEMPLEX 的 HBM 资源效率高于 CO（Sec. 5.2）。

### Dynamic memory energy

Fig. 8(c) 显示，CO 相比 baseline 降低 23% dynamic memory energy；MEMPLEX 降低 44%；Ideal 降低 51%。论文解释为 MEMPLEX 减少了 remote HBM 和 external DDR 访问，从而减少高能耗 memory operation。论文未报告 processor energy 和 static memory energy，理由是它们大体与 runtime 成比例（Fig. 8，Sec. 5.3）。

### System size 敏感性

Fig. 9(a) 比较 4-chiplet 和 16-chiplet 系统。4-chiplet 系统中，Ideal 比 baseline 高 26%，MEMPLEX 平均 speedup 5%，单个 mix 最高 7%。16-chiplet 系统中，Ideal 比 baseline 高 31%，MEMPLEX 平均 speedup 增至 10%，单个 mix 最高 15%。论文解释为系统规模增大后 NUMA overhead 增加，MEMPLEX 的收益也随之扩大（Fig. 9，Sec. 5.4）。

### DRAM cache size 敏感性

Fig. 9(b) 比较 MEMPLEX 的 DRAM Cache to Main Memory ratio：1:8、1:16、1:32。论文报告三者平均 speedup 分别为 8%、5%、4%；1:8 在部分场景最高可达 10%。论文同时给出 DCTA SRAM 开销：按 8-byte entry 假设，1:8、1:16、1:32 分别需要 512 KB、1 MB、2 MB DCTA（Fig. 9，Sec. 5.5）。

## 分析

- MEMPLEX 的核心不是固定“本地优先”，而是在每个 node 中用小容量 DRAM cache 复制远端数据，并在 eviction 时按访问热度与迁移流量预算决定是否把完整 sector 迁移到 LM（Sec. 3.6，Sec. 3.7）。
- MEMPLEX 同时利用两种机制：cache line 粒度复制降低首次访问成本，sector 粒度迁移把反复访问的远端 sector 变成本地主存数据。复制发生在访问路径上，迁移决策延迟到 eviction，以避免每次 miss 都做完整 sector 迁移（Fig. 3，Fig. 5）。
- 论文中的 baseline 已经有 NUMA-aware data placement。MEMPLEX 的收益说明，静态或首次访问驱动的数据放置仍可能与后续访问模式不匹配；硬件可见的访问计数和远端流量反馈能进一步降低 remote traffic（Fig. 7，Fig. 8）。
- CO 的表现说明“把全部 HBM 当 cache”不是无条件更优。它减少 remote traffic，但牺牲 HBM 主存容量，需要增加 external DDR 容量才能公平容纳 workload mix；平均性能收益只有 1%，低于 MEMPLEX 的 5%（Sec. 4.4，Fig. 7）。
- MEMPLEX 的结果来自模拟系统，不是实机测量。论文使用缩小到 1/4 的 chiplet 配置，并有意缩小 L2/L3 以提高 LLC MPKI；结论依赖该模拟设置和 workload mix（Sec. 4.1，Sec. 4.2）。

## 边界

- 论文对象是主存系统架构，关键资源是 HBM node、external DDR、DRAM cache、remap metadata 和 migration traffic；不是 CPU L3 slice/CCD 内任务调度。
- MEMPLEX 需要硬件 DCC、DCTA、remap table、free memory stack 和迁移控制逻辑。当前 vLLM CPU attention 不能通过软件线程绑定直接实现 MEMPLEX 的数据迁移机制。
- 评估 workload 是 multi-programmed mixes，每个 core 跑一个应用实例；论文未评估共享数据多线程程序，也未完成 DRAM cache coherence 优化和 multi-threaded workload 评估（Sec. 3.9，Sec. 4.3）。
- 论文平台是模拟的 HBM + external DDR chiplet NUMA 架构。当前 AMD EPYC CPU 平台的 DDR 内存层级、CCD L3 行为、NUMA/NPS 模式与该 HBM 系统不同，不能直接套用 1:16 cache ratio、80% remote traffic reduction 或 5%-10% speedup。
- 论文没有覆盖 LLM prefill/decode、KV cache streaming、vLLM scheduler、OpenMP attention task 或 batch-level serving 行为。

## 可迁移点

- “远端访问开销”需要拆成访问来源统计，而不是只看总 memory traffic。MEMPLEX 同时报告 local HBM、remote HBM、external DDR 占比，并用 AMAT 和 energy 解释性能变化（Fig. 7，Fig. 8）。
- 数据放置策略应基于运行期访问行为。MEMPLEX 用 AC 和 remote access counter 在 eviction 时决定 migration，避免静态 NUMA placement 无法适应 phase 变化的问题（Sec. 3.7）。
- 复制和迁移可以分层使用。MEMPLEX 先按 cache line 复制请求数据，再按 sector 决定是否迁移完整数据；这比每次 miss 都迁移整页更谨慎（Sec. 3.4，Sec. 3.6）。
- 成本模型需要显式计入迁移本身产生的远端流量。MEMPLEX 不只看访问热度，还用 `Om` 和 remote access counter 限制 migration traffic（Sec. 3.7）。

## 不可直接迁移点

- MEMPLEX 的 Local Memory/Remote Memory 是 HBM/DDR 主存层级，不是 CCD 本地 L3 与跨 CCD L3。当前 attention 的 `acc-local-l3` 尝试不能从 MEMPLEX 推导出“静态本地 L3 绑定有效”。
- MEMPLEX 的收益来自硬件 remap 和 sector migration；vLLM CPU attention 目前只能改变线程和任务领取范围，无法透明改变 KV page 的物理主存归属。
- MEMPLEX 面对的是多程序高 LLC MPKI workload mix；attention prefill/decode 的数据访问模式是张量/KV streaming 与算子调度，不具备同样的 sector eviction migration 决策点。
- MEMPLEX 仍有未评估的一致性和多线程共享数据问题。对当前 LLM 推理系统，只能借鉴“访问来源量化”和“热度/迁移成本联合判据”，不能把论文性能数字作为当前系统可达收益。

## 证据

- 原始资料：`wiki/原始资料/papers/MEMPLEX.pdf`
- 关键锚点：摘要；Fig. 1-9；Tab. 1-2；Sec. 1；Sec. 2.1-2.3；Sec. 3.1-3.9；Sec. 4.1-4.4；Sec. 5.1-5.5；Sec. 6
