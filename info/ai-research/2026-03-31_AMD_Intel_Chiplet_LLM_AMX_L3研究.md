# AMD / Intel Chiplet CPU、LLM推理与 `L3`-感知放置 / `AMX` 关系研究

时间：`2026-03-31 14:50:00 +0800`
最近修订：`2026-04-01 12:41:56 +0800`

## 0. 本文边界与本地锚点

事实：
- 本文的“启发论文2”特指 `OPTIMIZING ATTENTION ON GPUS BY EXPLOITING GPU ARCHITECTURAL NUMA EFFECTS`；在 `info/进展.md` 中，它被概括为“把共享同一组 K/V 的 workgroup 尽量放在同一局部域，以提升局部 cache 复用、减少重复外部访存”。证据：`info/进展.md:14-18`。
- 本文的“实验观察1”特指：当前 CPU prefill attention 在同一 `kv_head` 下，会把工作进一步拆到多个线程桶；这些线程会从同一组 `key_cache/value_cache` 读取高度重叠的 KV 前缀。证据：`info/进展.md:21-25`，以及 `csrc/cpu/cpu_attn_impl.hpp:436-589`、`1424-1459`、`1527-1561`、`1631-1727`。
- 当前机器是 `2 x AMD EPYC 9745 128-Core Processor`；`lscpu` 记录了 `L3 cache: 512 MiB (16 instances)`、`NUMA node(s): 2`。证据：`sources/local_20260331_lscpu.txt`。
- 当前机器的 CPU flags 含 `avx_vnni`、`avx512_bf16`、`avx512_vnni`，但未检出任何 `amx_*` flags。证据：`sources/local_20260331_cpu_flags.txt`。

## 1. AMD / Intel 的 chiplet / tile / disaggregated CPU 对比

事实：
- Intel 术语不要混写成一个层级：
  - `chiplet / MCM` 是封装层概念。Intel 官方支持文档明确写了：4th Gen Xeon 是 `MCM`，Xeon 6 则进入更彻底的 `modular, tile-based architecture`。证据：`Are the Intel Xeon Scalable Processor Families Single-Chip or MCM?`, `https://www.intel.com/content/www/us/en/support/articles/000099181/processors/intel-xeon-processors.html`。
  - `tile` 是封装内的物理 die / tile。对 Sapphire Rapids，Intel 新闻稿明确给出“最多 `4` 个 tiles，通过 `EMIB` 封装”；官方 datasheet 又把处理器描述为“由多个 `I/O interface` 与 `CHA/core modules` 组成，并通过 `mesh interconnect` 连接”。这说明 tile 内部仍然是“core + cache/home agent + I/O 接口 + mesh”的 SoC 组织，而不是“一个 tile = 一个核心簇”的极简结构。证据：`Intel Launches 4th Gen Xeon...`, p.5, `https://download.intel.com/newsroom/archive/2025/en-us-2023-01-10-intel-launches-4th-gen-xeon-scalable-processors-max-series-cpus.pdf`；`Sapphire Rapids Datasheet, Overview Features and Topologies`, `https://edc.intel.com/content/www/vn/vi/design/products-and-solutions/processors-and-chipsets/eagle-stream/sapphire-rapids-datasheet-vol-one/overview-features-and-topologies/`。
  - `SNC` 是软件可见的 socket 内 NUMA 划分，不等于 tile。Intel 官方写了 `SNC2` / `SNC4` 是在 `Hemisphere` / `Quadrant` cache clustering 基础上继续把 socket 切成 `2` 或 `4` 个 locality domains；例如 `SNC2` 下每个 domain 拥有一半处理器、一半 `LLC banks` 和一个内存控制器，`SNC4` 下则进一步切成四分之一处理器、四分之一 `LLC banks`、四分之一内存控制器。证据：`4th Gen Intel Xeon Processor Scalable Family, sapphire rapids`, “Memory Controllers and Sub-NUMA Clusters (SNC)”, `https://www.intel.com/content/www/us/en/developer/articles/technical/fourth-generation-xeon-scalable-family-overview.html`。
- 把 Sapphire Rapids 写具体，Intel 侧更像“四层结构”：
  - 封装层：单 socket 是一个 `MCM` package，最多 `4` 个 compute tiles，通过 `EMIB` 拼成一颗 CPU。证据：`Intel Launches 4th Gen Xeon...`, p.5。
  - tile 内部层：每 tile 不是只有核心，还包含多组 `CHA/core modules`、I/O interface，并通过 `mesh` 互联。证据：`Sapphire Rapids Datasheet, Overview Features and Topologies`。
  - cache 层：官方总览表给出每核 `2 MB private L2`、每核 `1.875 MB LLC`，说明 LLC 是按 slice/bank 分布在整颗 package 上，而不是像 AMD `CCX 32 MB L3` 那样直接对应一个显式共享块。证据：`4th Gen Intel Xeon Processor Scalable Family, sapphire rapids`, Table 1, `https://www.intel.com/content/www/us/en/developer/articles/technical/fourth-generation-xeon-scalable-family-overview.html`。
  - socket 互连层：Sapphire Rapids 还提供最多 `4` 条 `UPI 2.0` 链路、`4` 个 integrated memory controllers、共 `8` 个 DDR5 channels，所以跨 tile、跨 memory controller、跨 socket 是三个不同层次的路径。证据：`4th Gen Intel Xeon Processor Scalable Family, sapphire rapids`, Table 1。
- Intel 的 locality 机制也要拆开理解：
  - 默认 `UMA` 下，地址会在全部内存控制器间交错，所有核心都可访问全 socket 的 `LLC slices` 和内存。
  - `Hemisphere` / `Quadrant` 更像 cache clustering / snoop 层面的本地化。
  - `SNC` 则把这种局部性进一步显式暴露给 OS / runtime，使 thread、page、memory controller 和一组 `LLC banks` 更容易绑定到同一 locality domain。
  证据：`4th Gen Intel Xeon Processor Scalable Family, sapphire rapids`, “Memory Controllers and Sub-NUMA Clusters (SNC)”.
- Xeon 6 的“tile-based / disaggregated”比 Sapphire Rapids 更进一步：
  - Intel 官方支持文档明确说 Xeon 6 不再只是“多个 compute tiles 的 MCM”，而是更彻底的 `modular, tile-based architecture`，可以由多个 `compute tiles` 与 `I/O tiles` 组成。证据：`Are the Intel Xeon Scalable Processor Families Single-Chip or MCM?`。
  - Intel Xeon 6 产品简介已经把不同 tile 组合映射到不同 SKU 家族：`P-core` SKU 最多 `128` cores / `504 MB total L3 cache`，`E-core` SKU 最多 `288` cores / `216 MB total L3 cache`。这至少说明 Xeon 6 时代 Intel 公开描述缓存容量时，已经更像“整包聚合后的总量”，而不是直接用一个单 die LLC 来叙述。证据：`Intel Xeon 6 Product Brief`, `https://www.intel.com/content/www/us/en/products/docs/xeon-6-product-brief.html`。
  - Intel 在 `2026-03-12` 发布的 Xeon 6 技术文章里，直接把这代产品称为 `newer disaggregated SoC architecture`。结合上一条“compute tiles + I/O tiles”的官方说法，更稳妥的理解是：Intel 这里的 `disaggregated`，指的是把原本可做成单一大 die 的 compute、I/O、SoC 功能拆成多个可组合 tiles，再按 SKU 目标重新拼装，而不是单纯指双路 NUMA。证据：`Intel Xeon 6 Processors: Performance and Power Profiles...`, 发布日期 `2026-03-12`, `https://www.intel.com/content/www/us/en/content-details/826934/intel-xeon-6-processors-performance-and-power-profiles-default-latency-optimized-mode-and-other-options-technical-article.html`。这里“对 disaggregated 的含义解释”是基于两份 Intel 官方材料的推断。
- 矩阵乘法相关硬件能力：Intel 4th Gen Xeon 提供 `AMX_INT8`、`AMX_BF16`，同时也支持 `AVX512_VNNI`、`AVX512_BF16`；Intel 官方调优指南明确写了框架会优先选 `AMX` 矩阵路径，`FP32` 则继续使用 `AVX-512`。Intel ISA 手册则把 `AMX` 描述为 `tile registers + TMUL`。证据：`Tuning Guide for AI on the 4th Generation Intel Xeon Scalable Processors`, `https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-guide-for-ai-on-the-4th-generation.html`；`Intel Architecture Instruction Set Extensions Programming Reference`, p.15, p.98-p.101, `https://cdrdv2.intel.com/v1/dl/getContent/671368`。
- 作为对照，AMD 官方资料给出的 EPYC 9005 组织仍然是：多个 `CCD` 围绕中心 `I/O Die`，一个 `CCX` 共享一份 `32 MB L3`；`Zen5` 最多 `8` 核 / `32 MB L3`，`Zen5c` 最多 `16` 核 / `32 MB L3`。证据：`AMD EPYC 9005 Processor Architecture Overview`, p.3-p.8, `https://docs.amd.com/api/khub/documents/iZ5eCtQ5v6PyPN7wEd8HQg/content`。
- AMD 是否有与 AMX 对应的矩阵扩展单元：本次检索覆盖的 AMD 官方材料里，我只检到“`Zen 5` 从核心开始扩成 `512-bit` 数据通路、可提升向量/FP 吞吐”的表述，没有检到与 Intel `AMX tile register + TMUL` 对等的独立矩阵单元描述。证据：`5th Gen AMD EPYC Processor Architecture`, p.8；本机 flags 仅见 `avx_vnni`、`avx512_bf16`、`avx512_vnni`，未见 `amx_*`，证据：`sources/local_20260331_cpu_flags.txt`。

基于事实的推断：
- 如果后续要在 Intel 平台上做 locality 实验，至少要分清 `socket`、`tile`、`SNC domain`、`UPI peer socket` 四层；把它们混成“跨 chiplet”会丢掉关键信息。证据基底：前述 Intel 官方 `MCM / tile / mesh / SNC / UPI` 描述。
- AMD 矩阵执行路径：在当前可证据化范围内，更稳妥的表述是：AMD 没有在这批官方资料里披露与 `AMX` 直接对等的矩阵扩展单元；矩阵乘法主要仍应理解为走 `SIMD / AVX-512 / VNNI / BF16 / FMA` 这类向量执行路径，而不是 `tile + TMUL` 路径。证据基底同上。
- 当前机器的 slice 形态：`lscpu` 显示本机双路共有 `16` 个 `L3 instances`，即每 socket `8` 个 `L3 slices`；把这点与 AMD 官方“每个 `CCX` 共享 `32 MB L3`，`Zen5c` 最多 `16` 核 / `32 MB L3`”对照，当前机器更像“每 socket 8 个 `32 MB` slice、每 slice 约 16 物理核心”的形态。证据：`sources/local_20260331_lscpu.txt`；`AMD EPYC 9005 Processor Architecture Overview`, p.3。这里是推断，不是 AMD 产品页直接声明。

## 2. 两个场景分别分析

### 2.1 矩阵乘法（GEMM）

事实：
- Intel 官方把 `AMX` 定义为矩阵乘法路径：`tile registers` 装载矩阵块，`TMUL` 执行 `C += A * B` 风格的 tile-level multiply-accumulate；Intel 官方 AI 指南明确区分了 `AMX_INT8 / AMX_BF16` 与 `AVX512_VNNI / AVX512_BF16` 两条路径，并说明框架会优先 AMX。证据：`Intel Architecture Instruction Set Extensions Programming Reference`, p.98-p.101；`Tuning Guide for AI on the 4th Generation Intel Xeon Scalable Processors`。
- AMD 官方能直接确认的是：EPYC 9005 的 `Zen 5` 浮点 / 向量 ALU 具备 `full, 512-bit data path`。证据：`5th Gen AMD EPYC Processor Architecture`, p.8。

基于事实的推断：
- GEMM 更偏“计算内核吞吐优化”的一侧：`AMX`、`AVX-512/VNNI/BF16`、kernel blocking、寄存器 tiling，首先影响的是单核 / 单线程块的乘加吞吐，而不是线程跨 chiplet 的访问拓扑。
- 但 GEMM 也不是完全脱离拓扑：当参与线程跨 `CCX / CCD / SNC cluster / socket`，且矩阵块超出本地 `L2/L3` 时，`A/B/C` tile 的 refill 延迟、带宽和 `LLC` 命中率仍会受 `NPS / SNC / NUMA memory policy / 绑核` 影响。其依据是 AMD 明确把 `LLC as NUMA` 暴露出来，Intel 明确写了 `SNC` 的目标就是降低本地域 `LLC / memory latency`。证据：`AMD EPYC 9005 Processor Architecture Overview`, p.6-p.8；`4th Gen Intel Xeon Processor Scalable Family, sapphire rapids`, “SNC”.

### 2.2 LLM 推理（prefill / decode / attention / KV cache）

事实：
- `SARATHI` 明确区分了 `prefill` 和 `decode`：`prefill` 在小 batch 下也能较充分占用算力，而 `decode` 因逐 token 生成会出现低计算利用率。证据：`SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills`, `https://www.microsoft.com/en-us/research/publication/sarathi-efficient-llm-inference-by-piggybacking-decodes-with-chunked-prefills/`。
- 本地 PCM 结果显示，在 `B16/I1024/O1` vs `B16/I1/O1024` 对照下，`decode` 的 `CPI=1.64`、`L3 Miss %=96.51`、`Ave L3 Miss Latency=326.57 ns`，都显著高于 `prefill` 的 `0.44 / 49.43 / 180.12 ns`；但两者 `Remote DRAM Reads %` 都只有 `0.03`。证据：`test_results/PD_Test/Qwen3-30B-A3B/PCm_res/2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md:23-39`。
- 本地源码证据显示，当前 CPU prefill attention 的 workitem 切分先按 `kv_len_per_thread` 和剩余长度做负载均衡，再展开成 `actual_kv_head_num * effective_thread_num`；不同线程会按同一 `curr_kv_head_idx` 去全局 `key_cache/value_cache` 取数，并在 causal prefill 下读取彼此高度重叠的 KV 前缀。证据：`csrc/cpu/cpu_attn_impl.hpp:436-589`、`1424-1459`、`1527-1561`、`1631-1727`。
- `vAttention` 指出，把 KV cache 从连续虚拟地址打散成分页式映射会带来额外编程和性能开销，说明 KV 布局本身就是性能变量。证据：`vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention`, `https://arxiv.org/abs/2405.04437`。

基于事实的推断：
- 因此，LLM 推理比纯 GEMM 更明显地分成两类热点：
  - 线性层 / GEMM：更偏计算内核吞吐，`AMX` 或 `AVX-512/VNNI/BF16` 影响更直接。
  - attention / KV cache：更偏数据放置、cache 局部性、线程放置、NUMA / chiplet 拓扑。
- 对当前课题，`prefill` 不应只被当成“算子计算侧”问题：本地源码已经给出“同一 `kv_head` 多线程重复读重叠 KV 前缀”的直接证据；这类访问模式更像 `启发论文2` 里的“共享 K/V 工作集”问题，而不是单纯 GEMM 问题。
- 对当前机器，`decode` 的主要压力现在更像 `L3/locality` 问题而不是“远端 DRAM”问题，因为 `Remote DRAM Reads %` 很低，但 `L3 Miss %` 与 `Ave L3 Miss Latency` 很高。证据基底：`2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md:23-39`、`:73-85`。

## 3. 单独回答：`L3` 感知数据放置是否与 `AMX` 正交

### 3.1 理论上：在支持 `AMX` 的 Intel CPU 上

事实：
- `AMX` 作用在“计算内核执行层”：Intel 官方把它定义为 `tile registers + TMUL` 的矩阵运算路径；`FP32` 仍走 `AVX-512`，`INT8/BF16` 框架会优先选 `AMX`。证据：`Intel Architecture Instruction Set Extensions Programming Reference`, p.98-p.101；`Tuning Guide for AI on the 4th Generation Intel Xeon Scalable Processors`。
- `L3` 感知的数据放置 / 任务放置作用在“数据和拓扑层”：AMD 的 `NPS / LLC as NUMA`、Intel 的 `SNC` 都是在改变线程更常访问哪一片 `LLC / memory controller / NUMA domain`。证据：`AMD EPYC 9005 Processor Architecture Overview`, p.6-p.8；`4th Gen Intel Xeon Processor Scalable Family, sapphire rapids`, “SNC”.

基于事实的推断：
- 所以两者不是同一层优化，不能混为一谈。更准确的说法是：**它们“部分正交、整体互补”，而不是“完全无关”**。
- 在纯 dense GEMM 上，这种“部分正交”最明显：
  - `AMX` 先改变单核 / 单 thread block 的乘加吞吐。
  - `L3`-感知放置先改变 A/B/C tile 的 refill 延迟、本地 `LLC` 命中率、跨 cluster / cross-socket 流量。
  - 如果 working set 足够小、block 足够好，二者接近正交。
- 但它们不是完全正交，耦合点至少有三处：
  - 更快的 `GEMM` 会改变 bottleneck 占比。若 `AMX` 把计算时间压低，而数据供给改善幅度较小，瓶颈会更快转向 `L3 / memory / topology`。
  - 更强的矩阵单元会提高对数据供给连续性和局部性的要求。Intel 官方本身就把 `AMX` 描述为 tile load/store + TMUL 协作的多周期执行路径，这意味着数据供给失配时，矩阵单元也会空等。证据：`Intel Architecture Instruction Set Extensions Programming Reference`, p.98-p.101。
  - `AMX_INT8 / BF16` 还会改变数据类型和字节数；Intel 官方调优指南明确写了 `BF16 / INT8` 可降低内存占用，因此 `AMX` 既可能放大供给瓶颈，也可能因为精度缩减而减轻部分带宽压力。证据：`Tuning Guide for AI on the 4th Generation Intel Xeon Scalable Processors`。
- 在 attention / KV cache 场景里，两者更不对称：`AMX` 直接作用的是 GEMM-like kernel；而 KV cache 的读取重排、线程放置、cache 复用、拓扑局部性，并不会因为有 `AMX` 就自动解决。所以 attention / KV 场景下，`L3`-感知放置通常比 AMX 更“相对独立”。

### 3.2 结合当前课题：在当前 `2 x AMD EPYC 9745 128-Core Processor` 上

事实：
- 当前机器没有 `AMX` flags，但有 `avx_vnni`、`avx512_bf16`、`avx512_vnni`。证据：`sources/local_20260331_cpu_flags.txt`。
- AMD 官方材料明确给出了 `NPS`、`LLC as NUMA`、`CCX`-shared `L3`、`Infinity Fabric / xGMI` 这套 locality / topology 机制。证据：`AMD EPYC 9005 Processor Architecture Overview`, p.3-p.8。
- 本地源码与实验已经分别给出了两个和 `L3` 直接相关的证据：
  - prefill attention 存在“共享同一 `kv_head` 的多线程重复读取重叠 KV 前缀”。证据：`csrc/cpu/cpu_attn_impl.hpp:436-589`、`1424-1459`、`1527-1561`、`1631-1727`。
  - 当前 decode 的 `L3 Miss` 与 miss latency 明显高，而 `Remote DRAM Reads %` 仍很低。证据：`2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md:23-39`、`:73-85`。

基于事实的推断：
- 所以，“当前机器不支持 `AMX`”并不等于“`L3` 感知数据放置问题无意义”。恰恰相反：在这台 AMD 机器上，**`L3` 感知数据放置 / 任务放置仍然值得独立研究**，因为现有本地证据已经把问题指向了 `CCX/L3/locality`，而不是 `AMX` 缺失。
- 更直接地说：当前课题的主问题可以独立表述为“如何减少共享 KV 工作集的跨 `CCX/L3` 扩散与重复读取”，这个问题与有没有 `AMX` 不是一回事。

尚待验证的假设：
- 如果未来拿支持 `AMX` 的 Intel CPU 做对照，同样仍应分别研究两层：
  - 计算内核层：`AMX` 是否改善线性层 / GEMM 吞吐；
  - 拓扑与缓存层：共享 KV 工作集是否仍需要 `SNC / LLC / NUMA` 感知的放置策略。

## 4. 对当前课题的直接启示

事实：
- `启发论文2` 的核心不是泛泛“就近调度”，而是把**共享同一组 K/V 的工作**尽量放进同一个局部域，以提高局部 cache 复用、减少重复外部读取。证据：`info/进展.md:14-18`。
- `实验观察1` 指向的正是一个 CPU 版对应物：当前 attention 调度先按负载均衡切，再展开到 `actual_kv_head_num * effective_thread_num`，于是同一 `kv_head` 的 KV 前缀会被多个线程重复读。证据：`info/进展.md:21-25`；`csrc/cpu/cpu_attn_impl.hpp:436-589`、`1424-1459`、`1527-1561`、`1631-1727`。

基于事实的推断：
- 因而，对当前课题最直接的研究主线不是“先找 AMX 等价物”，而是把 `启发论文2` 的“共享 K/V 工作集局部化”思想，翻译到这台 AMD chiplet CPU 上的 `CCX/L3/NUMA` 语境：
  - 让共享同一 `kv_head`、或共享更大 KV 前缀的 tasks 尽量落在同一 `CCX` / 同一 `LLC-as-NUMA` 域。
  - 让对应 KV pages 尽量 first-touch / 保持在同一局部域，减少 same-socket cross-CCX 流量。
  - 再用 `CPI`、`L3 Miss %`、`Ave L3 Miss Latency`、`L3 Miss Latency From another CCX in same node (%)`、`Remote DRAM Reads %` 做验证。

尚待验证的假设：
- 对 prefill，可优先做“`kv_head`-clustered 绑核 + 局部分配”实验，观察 `same-node another CCX %` 与 `Ave L3 Miss Latency` 是否下降。依据：AMD 官方 `LLC as NUMA` / `NPS` 机制 + 本地“重叠 KV 前缀”源码证据。
- 对 decode，现阶段更应优先确认“高 `L3 Miss` 是否来自 KV cache 读取形态本身”，而不是先假设远端 DRAM 是主因；因为当前 `Remote DRAM Reads %` 很低。依据：`2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md:23-39`、`:73-85`。

## 5. 一句话结论

基于事实的推断：
- `L3` 感知的数据放置 / 任务放置，与 `AMX` 这类矩阵乘法扩展**不是同一层优化**；在 Intel `AMX` CPU 上它们是**部分正交、整体互补**，在 attention / KV cache 场景下更偏向“拓扑 / 缓存局部性优化相对独立”。在当前这台不支持 `AMX` 的 AMD EPYC 9745 上，`L3` 感知数据放置不仅仍有意义，而且已经被本地源码和 PCM 结果直接支撑为值得单独研究的方向。
