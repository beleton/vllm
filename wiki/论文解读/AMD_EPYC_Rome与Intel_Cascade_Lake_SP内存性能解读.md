# AMD EPYC Rome 与 Intel Cascade Lake SP 内存性能解读

## 说明
- 当前依据的原始资料是 `wiki/原始资料/papers/Memory Performance of AMD EPYC Rome and Intel Cascade Lake SP Server Processors.pdf`。
- 本地 PDF 首页给出的版本信息是：`ICPE 2022, April 9–13, Beijing, China`。
- 以下内容仅基于这份本地 PDF，不额外对照其他版本。

## 问题
- 论文要回答的是：`AMD EPYC Rome` 与 `Intel Cascade Lake SP` 的内存层级在本地访问、远端缓存访问、跨 `NUMA` 访问和双路访问下分别表现如何，以及这些差异会怎样影响数据放置与共享内存程序的性能（Abstract，Sec. 6-9）。

## 核心内容
- 论文系统测量两代服务器 CPU 的 `cache` / `NUMA` / `socket-socket` 延迟与带宽，并把差异归到具体硬件组织上：`Rome` 是 `CCD + CCX + I/O-die` 的 chiplet 设计，`CLX` 是单片 `mesh` 设计（Sec. 3，Sec. 4，Sec. 9）。
- 论文主结论有两条：
  - `Rome` 的本地 `L3` 延迟更低、整 socket 内存带宽更高，但访问层级必须至少区分为“本地 `CCX`”“同 socket 内其他 `CCX/NUMA node`”“其他 socket 的 `NUMA node`”，不同层级的代价差异很大，也更依赖数据放置与线程绑定（Fig. 7-13，Fig. 16，Sec. 9）。
  - `CLX` 的本地 `L3` 更慢、整 socket 带宽更低，但跨核共享数据时延迟更均匀，适合更多核心共同访问共享数据的场景（Fig. 7，Fig. 10，Fig. 15，Sec. 7.4，Sec. 9）。

## 观察
- `Rome` 每个处理器最多有 `8` 个 `CCD`，每个 `CCD` 有 `2` 个 `CCX`，每个 `CCX` 最多 `4` 个 `Zen 2` 核，共享 `16 MiB L3`；`CCD` 之间、`CCX` 到内存控制器之间都经过中央 `I/O-die` 与 Infinity Fabric（Sec. 3.1，Fig. 2，Fig. 3）。
- `Rome` 可以配置为 `1/2/4` 个 `NUMA` node，也可以把 `L3` 暴露成更多 `NUMA` 节点；双路之间通过 `xGMI` 相连，且接口在 `I/O-die` 上的位置不对称，因此远端 socket 延迟会随节点组合变化（Sec. 3.2）。
- `CLX` 使用单片 `mesh`，每个 core tile 含 `core + 1.375 MiB L3 slice + CHA`；`L2` 为 `1 MiB`，`L3` 为非包含式；处理器可按两个 `SNC` 划分本地内存控制器与本地 `L3` slice（Sec. 4.1，Sec. 4.2，Fig. 5，Fig. 6）。
- `CLX` 的 `L3` 不是像 `Rome` 的 `CCX-local L3` 那样由一小组核心独占共享，而是由多个 `L3 slice` 组成、经 `mesh` 互连的 LLC 体系；单个 `L3 slice` 只有 `1.375 MiB`，但同 socket 内所有核心都能以较接近的代价访问整个 `L3` 体系（Sec. 4.2，Sec. 7.4）。
- 测试平台是双路 `EPYC 7702` 与双路 `Xeon Gold 6248`。`Rome` 按 `NPS4` 暴露 `8` 个 `NUMA` 节点，`CLX` 按 `SNC` 暴露 `4` 个 `NUMA` 节点；延迟测试使用 `BenchIT/x86-membench` 的 `memory_latency`，带宽测试使用 `throughput` 与 `STREAM`（Table 1，Sec. 5）。

### Rome 拓扑术语速记
- `CCD`：`Core Complex Die`，是 `Rome` 的计算 chiplet。
- `CCX`：`Core Complex`，位于一个 `CCD` 内；论文口径下每个 `CCD` 含 `2` 个 `CCX`，每个 `CCX` 最多 `4` 个核，共享 `16 MiB L3`（Sec. 3.1）。
- `I/O-die`：位于处理器中央，负责连接各个 `CCD`、主存控制器和外部 `I/O`（Fig. 2）。
- `IF-switch`：`Infinity Fabric` 在 `I/O-die` 内部的硬件路由节点。论文明确写到，数据经过这些 `IF-switch` 时会产生至少 `2` 个 `FCLK` 周期的延迟（Sec. 3.1）。
- `IF-repeater`：`Infinity Fabric` 路径上的中继单元，论文写明每个会再增加 `1` 个 `FCLK` 周期延迟（Sec. 3.1）。
- 因此这里的 `IF-switch` 不是软件对象，而是 `I/O-die` 内部 `Infinity Fabric` 网络上的硬件转发/路由节点。

### Rome 文字拓扑图
- 可按论文口径把单 socket 理解成：
  - `core -> L2 -> CCX-local L3`
  - `CCX -> IFOP -> I/O-die(IF-switch / IF-repeater) -> 其他 CCD / 内存控制器 / xGMI`
- 展开写就是：
  - `socket`
  - `  -> I/O-die`
  - `     -> IF network (IF-switch / IF-repeater)`
  - `     -> memory controllers`
  - `     -> xGMI to another socket`
  - `     -> CCD0 ... CCD7`
  - `CCD`
  - `  -> CCX0 (cores + 16 MiB L3)`
  - `  -> CCX1 (cores + 16 MiB L3)`
- 论文还特别指出：同一 `CCD` 里的两个 `CCX` 也不是直接互连，而是通过 `IFOP` 接到 `I/O-die`；这正是跨 `CCX` 延迟明显升高的直接硬件原因（Sec. 3.1，Sec. 7.1）。

## 实验设置
- `Rome` 平台：`2 x AMD EPYC 7702`，每路 `64` 核，`L2` 总计 `64 MiB`，`L3` 总计 `512 MiB`，主频使用 `2.0 GHz`（Table 1）。
- `CLX` 平台：`2 x Intel Xeon Gold 6248`，每路 `20` 核，`L2` 总计 `40 MiB`，`L3` 总计 `49.5 MiB`；带宽测试固定在 `AVX-512 nominal frequency 1.6 GHz`，`uncore` 固定 `2.4 GHz`（Table 1，Sec. 5，Sec. 6.2，Sec. 7.4）。
- 延迟测试通过指针追逐控制缓存行状态，并显式构造 `Modified / Owned / Exclusive / Shared / Forward` 等一致性状态；带宽测试使用不同 SIMD 宽度的流式 load，`STREAM` 另外使用 non-temporal store（Sec. 5，Listing 1）。

### 延迟测量方法与工具
- 论文明确使用 `BenchIT 4` 的 `x86-membench` 扩展做延迟与带宽分析；延迟对应的 benchmark 名称是 `memory_latency`（Sec. 5）。
- `memory_latency` 的核心方法是 `pointer-chasing`：通过分配 buffer 大小控制目标内存层级，再沿随机生成的地址链逐次读取，测单次依赖访问延迟（Sec. 5）。
- 为降低 prefetcher 影响，访问地址由额外线程用随机数生成；为控制来源位置和一致性状态，作者用额外线程先把 cache line 准备到指定核心，并构造 `Modified / Owned / Exclusive / Shared / Forward` 等状态（Sec. 5，Listing 1）。
- 计时方式是用序列化的 `rdtsc`，在前后配 `mfence` / `lfence`；访问循环是展开的 `mov (%rbx), %rbx`，最终再扣除纯测量开销（Sec. 5）。
- 线程与内存放置也被显式固定：线程用 `sched_set_affinity()` 绑核，内存用 `numa_set_membind()` 绑定到目标节点（Listing 1）。
- 为了复现性，论文还在每次运行前 flush `L1/L2/L3`，默认使用 `512 B` 对齐，并开启 transparent huge pages；每个点重复多次取最小值/中位数汇总（Sec. 5）。
- 因此若后续在本机复现，最接近论文口径的工具链是：`BenchIT + x86-membench/memory_latency`，而不是只做普通 load latency 或简单双线程 ping-pong。

## 结果与解释

### 本地延迟
- 两个平台本地 `L1d` 延迟都为 `4 cycles`，但换算到时间上 `Rome` 是 `2.0 ns`，`CLX` 是 `1.6 ns`；本地 `L2` 延迟分别是 `12 cycles / 6.0 ns` 与 `14 cycles / 5.6 ns`（Fig. 7）。
- 本地 `L3` 上，`Rome` 是 `39 cycles / 19.5 ns`，`CLX` 是 `54 cycles / 21.6 ns`；论文据此指出，`Rome` 的 `CCX` 本地 `L3` 延迟更低。容量上，更准确的说法是：`CLX` 的本地 `L2` 更大（`1 MiB` 对 `512 KiB`），但 `L3` 不能写成“本地 `L3` 更大”，因为 `CLX` 的单个 `L3 slice` 只有 `1.375 MiB`，而 `Rome` 按 `CCX` 口径是 `4` 核共享 `16 MiB`。`CLX` 真正的优势是：单核可通过 `mesh` 以较接近的延迟访问整个共享 `L3` 体系，而不是拥有更大的本地 `L3 slice`（Fig. 7，Sec. 4.2，Sec. 6.1，Sec. 7.4）。
- 本地主存延迟方面，`Rome` 约 `220 cycles / 110 ns`，`CLX` 约 `200 cycles / 80 ns`。论文认为，`Rome` 更高的主存延迟可能与数据路径经过 `I/O-die` 有关，而 `CLX` 的内存控制器集成在 `mesh` 上（Fig. 7，Sec. 6.1）。

### 带宽
- `Rome` 的单核 `L1` 读带宽达到理论值 `128 GB/s`，`L2` 实测 `63.7 GB/s`，接近理论上限 `64 GB/s`；单核 `L3` 读带宽最高约 `46 GB/s`，但 `L3` 带宽不会随 `CCX` 内核心数线性扩展，`4` 核时整个 `CCX` 约 `151 GB/s`（Fig. 8，Sec. 6.2）。
- `Rome` 的一个 `CCD` 上，单 `CCX` 用 `3` 个核已可把该 `CCD` 的 RAM 带宽跑满；`STREAM` 在单 `NUMA` 节点上测得最高 `42.9 GB/s`，论文说明这可扩展到每 socket 约 `171 GB/s`，而且只需每 `CCD` 很少几个核心（Fig. 8，Fig. 9，Sec. 6.2）。
- `CLX` 的 `L1`、`L2` 带宽都低于手册给出的最大值或 sustained 值。论文给出的例子是：`AVX-512` 下 `L1` 实测 `116.25 B/cycle`，低于理论 `128 B/cycle`；`L3` 约 `11.3 B/cycle`，低于文档里的 `15 B/cycle`；单个 `SNC` 的 RAM 带宽需要 `8` 个核才饱和（Fig. 10，Sec. 6.2，Sec. 9）。
- 论文最后汇总为：单个 `NUMA` 节点上，`CLX` 因每节点有 `3` 条内存通道而带宽高于 `Rome`；但整 socket 上，`Rome` 因 `8` 条通道对 `6` 条通道，占优明显，论文测到 `CLX` 整 socket RAM 带宽比 `Rome` 低 `41%`（Sec. 6.2，Sec. 9）。

### Rome 的同 socket 访问层级
- `Rome` 上，同一 `CCX` 内读取其他核心的 `L1/L2` 数据时，`Owned/Shared` 大约是 `36-37 ns`，`Modified/Exclusive` 约 `39 ns`；这比本地 `L3` 的 `19.5 ns` 高一倍左右（Fig. 11，Table 2）。
- 若把主存一起纳入同 socket 对比，本地 `RAM` 是 `110 ns`；这意味着同一 `CCD` 的另一 `CCX` 上很多远端缓存访问，已经与本地内存访问同量级，甚至略慢（Table 2，Fig. 11）。
- 同一 socket 内、同一 `CCD` 的另一 `CCX` 上，远端 `CCX` 的 `L1/L2/L3` 访问延迟约 `102.5-126.5 ns`；论文明确说明这是因为两个 `CCX` 并不直连，而是必须经过 `I/O-die`（Fig. 11，Table 2，Sec. 7.1）。
- 同一 socket 内、其他 `NUMA node` 上的远端缓存访问，若来源是这些节点上核心缓存中的 `L1/L2/L3` 数据，其延迟约为 `107-152 ns`，取决于目标节点、缓存层级和一致性状态（Fig. 11，Table 2，Sec. 7.1）。
- 同一 socket 内、其他 `NUMA node` 上的远端主存访问，则是访问这些节点本地连接的 `RAM`；`NUMA 1`、`NUMA 2`、`NUMA 3` 的远端 `RAM` 延迟分别约为 `115 ns`、`124 ns`、`127.5 ns`（Fig. 11，Table 2，Sec. 7.1）。
- 因而论文数据直接表明：在 `Rome` 上，某些同 socket 内的远端缓存访问，尤其是远端 `L3` 或其他 `NUMA node` 上的远端缓存行访问，会慢于本地 `RAM = 110 ns`；这里慢的是“远端缓存访问”，不是本地 `L3`（Table 2，Fig. 11）。
- 仅看同 socket 主存层级时，本地 `RAM -> NUMA 1 RAM -> NUMA 2 RAM -> NUMA 3 RAM` 的延迟大致是 `110 ns -> 115 ns -> 124 ns -> 127.5 ns`；因此同 socket 内“远端内存”本身也应继续区分不同 `NUMA node`，不能合成一个统一数值（Table 2，Sec. 7.1）。
- 论文按路径估算：访问同 socket 内其他 `NUMA node` 时，每经过一个通往其他 `NUMA` 节点的 `IF-switch`，单向约增加 `2-2.5 ns`；因此同一 socket 内不同远端 `NUMA node` 的代价本身也不相同（Sec. 7.1）。
- 论文还指出 `Rome` 的 `CCX` 内部也并非完全均匀：在固定 `L3 slice` 的实验里，core 0 访问 core 3 的 `L1/L2` 比访问 core 1/2 额外多 `5-7 cycles`，作者据此推测 `CCX` 内部可能存在 ring 式距离差异（Fig. 12，Sec. 7.2）。

### Rome 的复杂一致性请求流
- 论文专门构造了三节点场景：请求节点、home node、forwarding node 分别不同，用来测迁移线程或共享数据写回后的真实代价（Sec. 7.3）。
- 在这类场景下，`Rome` 的最坏 `L2` 读取延迟达到 `330 cycles / 165 ns`，高于任何“本地来源一致”的原生访问（Fig. 13，Sec. 7.3）。
- 当请求节点固定为 `NUMA 0` 且 `home node` 与 `forwarding node` 都不在本地时，最终延迟主要由 `home node` 决定；若请求节点也是 `home node`，则更受 `forwarding node` 位置影响，且矩阵并不对称（Fig. 13，Sec. 7.3）。
- 论文据此明确写出：若 `Rome` 被配置为对 OS 只暴露单一 `NUMA` 域，仍应依赖线程绑定，否则会承受这类隐藏的远端延迟（Sec. 7.3）。

### CLX 的单 socket 远端访问
- `CLX` 上，同一 `SNC` 内远端 `Shared/Forwarded` 缓存行从 `L3` 读取约 `20 ns`；`Modified/Exclusive` 从远端 `L2/L1` 读取约 `48-52.8 ns`（Fig. 15，Table 3）。
- 跨到同 socket 的另一个 `SNC` 时，各 cache 层统一只额外增加约 `15 cycles / 6 ns`。论文把这和 `Rome` 跨 `CCX` 额外 `130-200 cycles` 的代价做了直接对比（Fig. 15，Sec. 7.4）。
- 论文据此判断：如果一个共享内存工作负载需要超过 `4` 个核心共同使用同一批数据，`CLX` 这种延迟更均匀、全核都能以相近延迟访问共享 `L3` 体系的设计更有优势（Sec. 7.4，Sec. 9）。

### Rome 的其他 socket 访问层级
- `Rome` 的双路远端主存延迟在不同 `NUMA` 节点组合之间差异明显，最低是 `203-204 ns`，最高约 `218 ns`；这里测的是“本 socket 请求节点访问另一 socket 上不同 `NUMA node` 的主存”，已经明显高于同 socket 内其他 `NUMA node` 的 `115-127.5 ns`（Fig. 16，Sec. 8，Table 2）。
- 论文据测量结果推断，`xGMI` 到另一颗处理器的路径在 `I/O-die` 上不是对称可达的，因此即便都是跨 socket，节点组合不同也会再拉开约 `15 ns` 的差距（Fig. 16，Sec. 8）。
- `CLX` 的远端 socket 主存访问更规整：访问第二颗 socket 的 `SNC 2` 最低约 `138 ns`，访问 `SNC 3` 再多约 `10 ns`，不同请求核心位置还会再引入少量附加差异（Fig. 17，Sec. 8）。
- 论文对两平台都给出同一结论：如果必须跨 socket 共享数据，应把通信线程放在更合适的节点/核心上，避免无谓的路径绕行（Sec. 8）。

## 分析
- 从论文可直接提炼的事实是：`Rome` 上访问层级不能只粗分成“本地”和“远端”。至少应区分 `CCX` 本地、同 socket 内同 `CCD` 的另一 `CCX`、同 socket 内其他 `NUMA node`、其他 socket 的 `NUMA node` 这几层；这些层级的延迟大致对应 `19.5 ns`、`102.5-126.5 ns`、`115-152 ns`、`203-218 ns`（Fig. 11，Fig. 16，Table 2）。
- 其中“同一 socket”也远不是均匀共享域。真正低延迟共享 `L3` 的粒度只有 `CCX` 内 `4` 个核；一旦跨 `CCX`，即使仍在同 `CCD` 或同 `NUMA` 节点内，延迟也会立刻接近甚至达到主存量级（Fig. 11，Table 2，Sec. 7.1）。
- 论文对这一现象的解释是：`Rome` 的 `CCX` 之间不直连，远端缓存请求需要穿过 `I/O-die`；因此某些远端缓存行虽然名义上仍在“cache”里，但访问代价已经高于本地主存访问。论文据此强调 `Rome` 上的数据放置与线程绑定必须谨慎处理，尤其是共享数据会跨 `CCX/NUMA` 流动时（Sec. 7.1，Sec. 7.3，Sec. 9）。
- 论文也直接说明：`CLX` 的优势不是本地 `L3` 更快，而是远端缓存访问的额外代价较小，且所有核心都能以相近代价访问公共 `L3`。这更适合共享数据较多的多核共享内存程序（Fig. 15，Sec. 7.4，Sec. 9）。
- 在带宽上，论文给出的区分是“单节点 vs 整 socket”。`CLX` 单 `SNC` 因 `3` 条内存通道有更高本地带宽；`Rome` 整 socket 因 `8` 条通道获得更高总带宽，前提是尽量减少跨核心共享带来的远端访问（Sec. 6.2，Sec. 9）。

## 边界
- 论文对象是 `EPYC 7702 (Rome)` 与 `Xeon Gold 6248 (Cascade Lake SP)`，不覆盖其他 `Rome/CLX` SKU，更不覆盖后续 `Milan`、`Sapphire Rapids` 或当前其它平台（Table 1，Sec. 9）。
- `Rome` 的测试是 `NPS4`，`CLX` 的测试是 `SNC`；这些延迟和带宽值依赖 BIOS/NUMA 配置、固定频率设置、内存条规格与主板拓扑，不能脱离这些条件使用（Table 1，Sec. 5）。
- 论文用的是 `BenchIT/x86-membench` 与 `STREAM` 的微基准，不是 LLM 推理、attention 或真实服务吞吐；因此图中的具体数值不能直接等同于当前工作负载的端到端收益或损失（Sec. 5，Sec. 9）。

## 可迁移点
- 在 chiplet CPU 上，应把“本地共享缓存域”的粒度看得比 `socket` 或粗粒度 `NUMA` 更细；对 `Rome` 而言，论文直接支持把 `CCX-local L3` 视为真正低延迟共享域（Fig. 11，Sec. 7.1）。
- 若工作负载主要受带宽限制且跨核共享较少，`Rome` 这类高通道数设计更有利；若大量核心必须持续共享同一批数据，像 `CLX` 这样远端缓存访问更均匀的设计更有利（Sec. 7.4，Sec. 9）。
- 线程绑定和数据放置在 `Rome` 上是一级问题。论文没有把它写成“可选优化”，而是直接指出单 NUMA 暴露下也应显式 pinning，避免隐藏的远端延迟（Sec. 7.3）。

## 不可直接迁移点
- 论文没有研究 attention、KV cache、batch 调度或 `vLLM` 线程池，因此不能把文中的 `CCX` / `SNC` 延迟结论直接替换成当前推理代码的性能结论。
- 论文的带宽测试偏流式、顺序访问；共享内存部分虽然研究了一致性状态，但仍是微基准，不等价于真实模型推理中的混合读写、同步和算子交错。

## 证据
- 原始资料：`wiki/原始资料/papers/Memory Performance of AMD EPYC Rome and Intel Cascade Lake SP Server Processors.pdf`
- 关键锚点：Abstract；`Table 1-3`；`Fig. 7-17`；`Sec. 3-9`
