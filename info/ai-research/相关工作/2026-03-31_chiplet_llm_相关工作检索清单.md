# Chiplet LLM 相关工作检索清单

## 任务目标
围绕 `info/进展.md` 中的当前课题，系统整理与 chiplet/NUMA/分片缓存硬件上应用优化相关的资料，优先关注 LLM 推理、attention、KV 共享工作集、线程放置、绑核、NUMA 内存策略、拓扑感知调度。

## 当前状态
- 2026-03-31：文档骨架已建立。
- 2026-03-31 20:27:37 +0800：已完成任务 `25` 的第一轮广覆盖检索；原始检索结果见 `../../../sources/search_20260331_202737_chiplet_llm_round1.md`。
- 2026-03-31 21:00:30 +0800：已完成任务 `26` 的第二轮补漏检索；原始检索结果见 `../../../sources/search_20260331_210030_chiplet_llm_round2.md`。
- 2026-03-31 22:30:55 +0800：已完成任务 `27`；已从候选池筛出正式清单并按四类整理。
- 主交接入口：本文件。
- 候选池入口：`2026-03-31_chiplet_llm_相关工作候选池.md`。

## 一、最直接相关：LLM 推理 + chiplet/NUMA/分片缓存

### 1. ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs
- 来源：2026-03-08，arXiv 预印本。https://arxiv.org/abs/2603.07770
- 平台/对象：many-core CPU、NUMA 多节点 CPU；CPU 上的 LLM 推理。
- 核心点：显式针对 cross-NUMA 开销做内存管理、线程调度和更细粒度 tensor parallel。
- 与当前课题关系：直接对应本机 `TP`、线程绑核、NUMA 内存策略，是目前最直接的论文型 CPU LLM 候选之一。
- 相关性：高。

### 2. Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving
- 来源：2025-05-19，arXiv 预印本。https://arxiv.org/abs/2507.18454
- 平台/对象：commodity CPU、x86 AVX2/AVX512、ARM NEON、NUMA CPU；CPU LLM serving。
- 核心点：显式拆分 prefill 和 decode 的执行计划，并指出静态 per-NUMA 分区对两阶段并不同时最优。
- 与当前课题关系：直接命中当前 `prefill/decode` 分相和 NUMA 划分主线。
- 相关性：高。

### 3. CPU - vLLM / vllm.platforms.cpu
- 来源：2026-03-31 访问，vLLM 官方文档与 API 参考。https://docs.vllm.ai/en/stable/getting_started/installation/cpu/ ； https://docs.vllm.ai/en/stable/api/vllm/platforms/cpu/
- 平台/对象：x86 CPU、NUMA 多 socket / 多节点 CPU；vLLM CPU backend。
- 核心点：文档直接给出 `VLLM_CPU_OMP_THREADS_BIND`、`CPU_VISIBLE_MEMORY_NODES`、`tensor-parallel-size` 等调优入口，API 侧暴露 `discover_numa_topology`。
- 与当前课题关系：是当前实验栈最贴近的一手实现资料，可直接对照仓库里的 `vLLM CPU + Chiplet` 实验配置。
- 相关性：高。

### 4. Unlocking Optimal LLM Performance on AMD EPYC CPUs with vLLM
- 来源：2025-11-05，AMD 官方技术文章。https://www.amd.com/en/blogs/2025/unlocking-optimal-llm-performance-on-amd-epyc--cpus-with-vllm.html
- 平台/对象：AMD EPYC、vLLM、ZenDNN；CPU-only LLM serving。
- 核心点：给出 EPYC 上 `vLLM` 的 socket / 内存 / 软件栈配置，并强调系统级参数对吞吐和延迟的影响。
- 与当前课题关系：这是当前最直接的 `AMD EPYC + vLLM` 工程资料之一，可补论文证据不足的缺口。
- 相关性：高。

### 5. ZenDNN 5.2: Accelerating vLLM Inference on AMD EPYC CPUs
- 来源：2026-03-13，AMD 官方技术文章。https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-accelerating-vllm-inference-on-amd-epyc-cpus.html
- 平台/对象：`2 x AMD EPYC 9755 128-Core Processor`、vLLM `0.15.1+cpu`、ZenTorch `5.2`。
- 核心点：指出单个超大实例会遇到 wide compute fabric 和 memory contention，建议用 `numactl` 拆成多实例并绑定本地内存池。
- 与当前课题关系：直接命中 `AMD EPYC + vLLM + NUMA/绑核/内存策略`，而且显式谈到 cache locality 与本地内存池。
- 相关性：高。

### 6. Speculative LLM Inference on the 5th Gen AMD EPYC Processors with PARD and PACE
- 来源：2025-07-23，AMD 官方技术文章。https://www.amd.com/en/developer/resources/technical-articles/2025/speculative-llm-inference-on-the-5th-gen-amd-epyc-processors-wit.html
- 平台/对象：dual-socket 5th Gen AMD EPYC 9005；CPU-only LLM serving。
- 核心点：把 speculative decoding、parallel draft model 和 AMD PACE 组合到 CPU serving 栈中，缓解 auto-regressive 阶段的带宽压力。
- 与当前课题关系：虽然主线不是 chiplet/L3，但它提供了 AMD EPYC 双路 CPU-only serving 的直接工程参照。
- 相关性：中高。

### 7. Llama 2 Inference from Intel with DeepSpeed
- 来源：2026-03-31 访问，Intel 官方技术文章。https://www.intel.com/content/www/us/en/developer/articles/technical/llama-2-on-xeon-scalable-processor-with-deepspeed.html
- 平台/对象：4th Gen Intel Xeon Scalable、双 socket / SNC CPU；DeepSpeed CPU inference。
- 核心点：明确使用 `--bind_cores_to_rank`、`--bind_core_list`、`numactl`、低延迟 SHM all-reduce 和 Indirect Access KV Cache。
- 与当前课题关系：虽然平台是 Intel，但“TP rank 绑定 socket / sub-NUMA + KV cache 访存重排”的思路可直接映射到当前主线。
- 相关性：高。

## 二、高相关方法启发：attention / KV / 拓扑感知放置

### 1. Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects
- 来源：2025-11-03，arXiv 预印本。https://arxiv.org/abs/2511.02132
- 平台/对象：AMD MI300X、多 chiplet GPU；attention / MHA / GPU NUMA。
- 核心点：把共享同一组 K/V 的 attention workgroup 映射到同一 GPU NUMA 域，以提高 cache reuse 并减少重复 HBM 读取。
- 与当前课题关系：这是当前最直接的“共享 KV 工作集 + 拓扑感知放置”证据来源，虽然平台是 GPU，但方法论与 CPU prefill 局部性假设高度同构。
- 相关性：高。

### 2. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- 来源：2023-05-22，arXiv 预印本 / EMNLP 2023。https://arxiv.org/abs/2305.13245
- 平台/对象：Transformer 架构层面；decoder inference。
- 核心点：提出 grouped-query attention，让多个 query heads 共享一组 KV heads。
- 与当前课题关系：为“共享同一组 K/V 的 head 集合就是一个工作集单元”提供了结构基础。
- 相关性：中高。

### 3. Hydragen: High-Throughput LLM Inference with Shared Prefixes
- 来源：2024-02-07，arXiv 预印本 / ICML 2024。https://arxiv.org/abs/2402.05099
- 平台/对象：GPU 为主的 LLM serving 环境；shared-prefix batch decoding。
- 核心点：把 attention 拆成 shared prefix 与 unique suffix 两部分，跨序列批处理 prefix attention，减少重复 KV 读取。
- 与当前课题关系：为“共享 KV 前缀就是一个工作集单元”提供了直接系统实现证据。
- 相关性：高。

### 4. ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition
- 来源：2024，ACL 2024 论文。https://aclanthology.org/2024.acl-long.623/
- 平台/对象：LLM serving 系统；多租户 self-attention。
- 核心点：通过 prefix-aware KV cache 与 two-phase partition 改进 self-attention 的 data locality，并共享匹配前缀的 KV tensor。
- 与当前课题关系：把“共享前缀 -> 共享 KV -> locality-aware attention kernel”链条写得最完整，适合映射到当前工作集分析。
- 相关性：高。

### 5. HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading
- 来源：2025-02-18，arXiv 预印本。https://arxiv.org/abs/2502.12574
- 平台/对象：GPU + CPU RAM；long-context LLM inference。
- 核心点：按 attention head 粒度把 KV cache 在 GPU 和 CPU RAM 间分放，只保留部分 heads 在 GPU 上。
- 与当前课题关系：说明“head 粒度的 KV 管理”可以落成系统设计，对当前 `kv_head` 局部性分析是直接启发。
- 相关性：中高。

### 6. KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference
- 来源：2025-03，arXiv 预印本。https://arxiv.org/abs/2503.16525
- 平台/对象：LLM / MLLM serving 系统；多用户 KV cache 共享。
- 核心点：把严格前缀匹配扩展到基于语义相似度的细粒度 KV 复用。
- 与当前课题关系：补上“共享工作集不一定要求完全相同前缀”的系统视角。
- 相关性：中高。

### 7. NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention
- 来源：2024-03-02，arXiv 预印本。https://arxiv.org/abs/2403.01273
- 平台/对象：CPU、SIMD；CPU attention 推理。
- 核心点：把 attention 里的 MAD 计算替换成基于 SIMD register lookup 的实现。
- 与当前课题关系：它不讨论 chiplet/NUMA，但属于少数直接研究 CPU attention 推理的论文，可作为 attention 阶段 CPU 优化对照组。
- 相关性：中高。

### 8. MagicPIG: LSH Sampling for Efficient LLM Generation
- 来源：2024，arXiv 预印本。https://arxiv.org/abs/2410.16179
- 平台/对象：GPU + CPU 异构系统；long-context LLM generation。
- 核心点：把 LSH hash table 和部分 attention 相关计算放到 CPU，缓解 GPU KV cache 压力。
- 与当前课题关系：提供了“CPU 侧承担 attention / KV 相关工作”的异构路线参考，但与当前精确 prefill attention 不是同一问题。
- 相关性：中。

## 三、相近硬件启发：multi-die GPU / GPU NUMA / 分片 LLC

### 1. Realizing the AMD Exascale Heterogeneous Processor Vision
- 来源：2024-06，ISCA 2024 industry paper。https://www.computermachines.org/joe/publications/pdfs/isca2024_exascale.pdf
- 平台/对象：AMD MI300A / MI300X、chiplet GPU / APU。
- 核心点：给出 MI300X 的 `NPS1/NPS4` 与多种 XCD partitioning 设计背景，把 chiplet 互连与 AI/HPC 工作负载放在统一架构里讨论。
- 与当前课题关系：能作为 `MI300X attention NUMA` 论文背后的硬件一手背景，也便于对照 `CCD/XCD` 分片与局部性概念。
- 相关性：中高。

### 2. Deep dive into the MI300 compute and memory partition modes
- 来源：2025-03-27，ROCm 官方博客。https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
- 平台/对象：AMD Instinct MI300X、ROCm、`CPX/SPX`、`NPS1/NPS4`。
- 核心点：解释 compute partition 与 memory partition 的软件暴露形式，并说明 `CPX + NPS4` 对局部化访问的意义。
- 与当前课题关系：这是把 `multi-die GPU / GPU NUMA` 落到可执行软件接口上的一手资料，可直接辅助理解 `XCD/NPS` 与拓扑感知放置。
- 相关性：高。

### 3. Optimizing Sorting for Chiplet-Based CPUs
- 来源：2024-08，VLDB ADMS 2024 workshop 论文。https://vldb.org/workshops/2024/proceedings/ADMS/ADMS24_03.pdf
- 平台/对象：AMD Ryzen / chiplet CPU；memory-intensive sorting。
- 核心点：按 chiplet 粒度切分输入，并根据数据规模在本地 L3 与聚合 L3 之间切换。
- 与当前课题关系：它把“数据规模相对本地 L3 / 聚合 L3 大小”的调度规则说得很清楚，对当前 decode/prefill 的 L3 行为分析高度相关。
- 相关性：高。

### 4. A Performance Analysis of Chiplet-Based Systems
- 来源：2025，DATE 2025 会议论文。https://research.chalmers.se/en/publication/546843
- 平台/对象：multi-chiplet systems、NUMA systems。
- 核心点：系统比较 chiplet 与 monolithic 的性能代价，以及 memory hierarchy、interconnect、设计选择的影响。
- 与当前课题关系：为“chiplet 本身为什么会带来性能损失、哪些硬件层因素最关键”提供背景证据。
- 相关性：中高。

### 5. MEMPLEX: A Memory System with Replication and Migration of Data for Multi-Chiplet NUMA Architectures
- 来源：2025，ICS 2025 会议论文。https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-36.pdf
- 平台/对象：multi-chiplet NUMA architecture、HBM + DDR。
- 核心点：在 multi-chiplet NUMA memory system 中结合数据复制与迁移，降低 remote access 开销。
- 与当前课题关系：虽然更偏硬件/内存系统，但给“数据迁移/复制是否能缓解 chiplet NUMA 代价”提供了底层相关工作入口。
- 相关性：中。

## 四、背景与系统方法：NUMA-aware / cache-aware / topology-aware runtime

### 1. ARCAS: Adaptive Runtime System for Chiplet-Aware Scheduling
- 来源：2025-03-14，arXiv 预印本。https://arxiv.org/abs/2503.11460
- 平台/对象：chiplet-based CPU；memory-intensive parallel workloads。
- 核心点：chiplet-aware task scheduling、hardware-aware memory allocation、fine-grained monitoring，并在局部性和聚合 cache 之间自适应切换。
- 与当前课题关系：它是“chiplet-aware runtime”主线代表，可直接支撑当前“局部性 vs 聚合 L3 容量”的分析框架。
- 相关性：高。

### 2. CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System
- 来源：2026，EuroSys 2026 预出版 PDF。https://zbjob.github.io/EuroSys26.pdf
- 平台/对象：AMD EPYC Milan 单 socket 8 chiplets；runtime mapping。
- 核心点：显式比较 `LocalCache` 与 `DistributedCache`，按运行时行为与数据规模调整任务放置。
- 与当前课题关系：比 ARCAS 更强调 chiplet 内/跨 chiplet cache 策略切换，对当前 L3 slice 与 CCD 级放置分析有直接参考价值。
- 相关性：高。

### 3. LFOC+: A Fair OS-level Cache-Clustering Policy for Commodity Multicore Systems
- 来源：IEEE TC 2022，对应 2024 arXiv 开放版本。https://arxiv.org/abs/2402.07693
- 平台/对象：commodity multicore、支持 LLC partitioning 的 Linux 系统。
- 核心点：基于 PMC 动态分类应用的 cache sensitivity / contentiousness，在 Linux kernel 中实现 fairness-aware cache clustering。
- 与当前课题关系：补足了“OS 级如何基于 LLC 行为做运行时聚类与隔离”的背景。
- 相关性：中。

### 4. Adaptive NUMA-aware data placement and task scheduling for analytical workloads in main-memory column-stores
- 来源：2016-10-01，PVLDB 论文。https://www.vldb.org/pvldb/vol10/p37-psaroudakis.pdf
- 平台/对象：多 socket NUMA server；main-memory column-store。
- 核心点：根据 workload intensity 自适应调整数据放置与跨 socket task stealing，避免内存密集型任务的有害远程窃取。
- 与当前课题关系：虽然不是 LLM / chiplet 场景，但它是 topology-aware data placement + task scheduling 的经典系统工作，适合作为基线背景。
- 相关性：中。

### 5. CacheAware: Data Locality-Aware Scheduling for Distributed Memory Systems
- 来源：2026-03，期刊论文。https://www.mdpi.com/2073-431X/15/3/181
- 平台/对象：NUMA / distributed memory systems；compiler-runtime 协同调度。
- 核心点：编译器标注数据访问模式，runtime 按 locality hint 和 cache affinity 做放置，并根据运行时 cache miss 动态迁移。
- 与当前课题关系：不是 chiplet/LLM 专题，但“cache affinity + 动态迁移”方法线值得保留。
- 相关性：中。

## 总结

### 哪些方向资料最多，哪些方向最缺
- 按正式清单条目数看，资料最多的是“高相关方法启发”方向，共 `8` 条；其次是“最直接相关”方向，共 `7` 条。
- 但“最直接相关”里真正贴近当前问题的论文型资料仍少，主要是 `ArcLight` 和 `Sandwich` 两篇预印本；其余更偏官方文档或工程技术文章。这说明“`AMD EPYC + vLLM + CPU NUMA` 有工程资料，但论文化证据不足”。
- “相近硬件启发”和“背景与系统方法”各有 `5` 条，数量不算少，足够支撑 `local cache vs aggregated cache`、`topology-aware placement`、`data migration/replication` 这些分析框架。
- 当前最缺的仍是两类资料：
  - 直接把 `AMD EPYC / vLLM CPU backend` 与 `chiplet / CCD / CCX / L3 slice` 放置策略绑定起来的论文。
  - 直接把 `shared KV prefix` 或 `kv_head` 工作集映射到 `CPU remote L3 / remote CCX / Ave L3 Miss Latency / Remote DRAM Reads %` 等指标的资料。

### 哪几篇/哪几类资料最值得先读
- 第一优先级：`ArcLight`、`Sandwich`、`CPU - vLLM / vllm.platforms.cpu`、`ZenDNN 5.2`、`Unlocking Optimal LLM Performance on AMD EPYC CPUs with vLLM`。
  - 理由：最贴近当前 `AMD EPYC + vLLM + prefill/decode + NUMA` 主线，能直接指导 `TP`、绑核、内存节点和执行相位拆分。
- 第二优先级：`Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects`、`Hydragen`、`ChunkAttention`、`GQA`。
  - 理由：最直接补“共享 KV 工作集 -> 拓扑感知放置”这条方法链，其中 GPU NUMA attention 论文给放置策略，`GQA/Hydragen/ChunkAttention` 给工作集定义和系统化实现。
- 第三优先级：`ARCAS`、`CHARM`、`Optimizing Sorting for Chiplet-Based CPUs`、`LFOC+`。
  - 理由：补齐 `local cache vs aggregated cache`、runtime 映射、OS 级 cache clustering 的系统背景。
- 第四优先级：`Realizing the AMD Exascale Heterogeneous Processor Vision`、`Deep dive into the MI300 compute and memory partition modes`、`MEMPLEX`、`A Performance Analysis of Chiplet-Based Systems`。
  - 理由：作为 `multi-die GPU / chiplet memory hierarchy` 的硬件与系统背景，用来校准对 `CCD/XCD/NPS`、数据迁移和远端访问代价的理解。

## 任务交接/下一步
- 2026-03-31 22:30:55 +0800：任务 `27` 已完成。本文件已从“检索清单骨架”升级为该主题的主交接文档。
- 原始检索轨迹保留在：
  - `../../../sources/search_20260331_202737_chiplet_llm_round1.md`
  - `../../../sources/search_20260331_210030_chiplet_llm_round2.md`
- 候选池继续保留在 `2026-03-31_chiplet_llm_相关工作候选池.md`，用于保存暂未进入主清单的边缘条目或后续补检结果。
- 如果后续继续做深读或补实验映射，建议从以下顺序进入：
  - 先读第一优先级资料，抽出对 `TP`、线程绑核、NUMA 内存策略、prefill/decode 分相最直接的规则。
  - 再读第二优先级资料，抽出“共享 KV 工作集”的可执行定义，判断能否映射到当前 `kv_head` / prefix 重叠模式。
  - 最后用第三、第四优先级资料补 `local L3 vs aggregated L3`、`CCD/XCD`、迁移/复制策略的背景。
- 当前仍未补到的空白：
  - `AMD EPYC + vLLM + chiplet / L3 slice` 的论文型直接证据。
  - 用 `Ave L3 Miss Latency`、`same-node another CCX %`、`Remote DRAM Reads %` 解释 LLM attention 调度的直接文献。
