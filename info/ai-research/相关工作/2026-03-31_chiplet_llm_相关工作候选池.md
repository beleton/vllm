# Chiplet LLM 相关工作候选池

## 用途
本文件用于保存检索阶段的候选资料、待核实条目和临时分组结果。允许先粗收集，再去重，再筛到正式清单。

## 记录模板
- 标题：
- 作者/机构：
- 年份或绝对日期：
- 类型：
- 硬件/平台：
- 研究对象或工作负载：
- 核心优化点：
- 与当前课题的关联点：
- 相关性评级：
- 资料链接：
- 备注：若为推断，明确标注“推断，不是原文直接结论”。

## 候选条目
### 1. ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs
- 标题：ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs
- 作者/机构：Yuzhuang Xu、Xu Han、Yuxuan Li、Wanxiang Che
- 年份或绝对日期：2026-03-08
- 类型：arXiv 预印本
- 硬件/平台：many-core CPU、NUMA 多节点 CPU
- 研究对象或工作负载：CPU 上的 LLM 推理
- 核心优化点：显式针对 cross-NUMA 开销做内存管理、线程调度和更细粒度的 tensor parallel
- 与当前课题的关联点：这是目前命中的最直接“LLM 推理 + many-core CPU/NUMA”资料，能直接对照本机 `TP`、线程绑核、NUMA 内存策略
- 相关性评级：高
- 资料链接：https://arxiv.org/abs/2603.07770
- 备注：原文声称相对主流框架最高提升 `46%` 吞吐；是否适用于 `AMD EPYC 9745` 仍需后续细读实验平台

### 2. CPU - vLLM / vllm.platforms.cpu
- 标题：CPU - vLLM / vllm.platforms.cpu
- 作者/机构：vLLM Team
- 年份或绝对日期：2026-03-31 访问（滚动更新文档）
- 类型：官方文档 / API 参考
- 硬件/平台：x86 CPU、NUMA 多 socket / 多节点 CPU
- 研究对象或工作负载：vLLM CPU backend 的部署、TP 与 NUMA 绑定
- 核心优化点：给出 `VLLM_CPU_OMP_THREADS_BIND`、`CPU_VISIBLE_MEMORY_NODES`、`VLLM_CPU_KVCACHE_SPACE` 的调优口径，并说明 `tensor-parallel-size` 建议对齐 NUMA node 数；API 文档还暴露了 `discover_numa_topology`
- 与当前课题的关联点：直接对应本仓库当前 `vLLM CPU + Chiplet` 分析对象，是最贴近现有实验栈的一手资料
- 相关性评级：高
- 资料链接：https://docs.vllm.ai/en/stable/getting_started/installation/cpu/ ； https://docs.vllm.ai/en/stable/api/vllm/platforms/cpu/
- 备注：这是工程文档，不是论文；可用于确认当前实现是否已有 NUMA 感知入口

### 3. Llama 2 Inference from Intel with DeepSpeed
- 标题：Llama 2 Inference from Intel with DeepSpeed
- 作者/机构：Intel
- 年份或绝对日期：2026-03-31 访问（页面未显式标出发布日期）
- 类型：官方技术文章
- 硬件/平台：4th Gen Intel Xeon Scalable、双 socket / SNC CPU
- 研究对象或工作负载：CPU 上的 LLM 推理、DeepSpeed AutoTP
- 核心优化点：`--bind_cores_to_rank` / `--bind_core_list`、每个 TP worker 用 `numactl` 绑核与绑内存、单节点低延迟 SHM AllReduce、Indirect Access KV Cache、oneDNN/AMX
- 与当前课题的关联点：虽然是 Intel 平台，但“TP rank 绑定 socket / sub-NUMA + KV cache 访存重排优化”的思路可直接映射到当前 `TP`、绑核、NUMA 内存策略主线
- 相关性评级：高
- 资料链接：https://www.intel.com/content/www/us/en/developer/articles/technical/llama-2-on-xeon-scalable-processor-with-deepspeed.html
- 备注：不是 chiplet 论文，但属于 CPU LLM 推理的一手实现资料

### 4. NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention
- 标题：NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention
- 作者/机构：Tianyi Zhang、Jonah Wonkyu Yi、Bowen Yao、Zhaozhuo Xu、Anshumali Shrivastava
- 年份或绝对日期：2024-03-02
- 类型：arXiv 预印本
- 硬件/平台：CPU、SIMD
- 研究对象或工作负载：CPU 上 attention 的加速
- 核心优化点：把 attention 里的 MAD 计算替换成基于 SIMD register lookup 的实现
- 与当前课题的关联点：虽然不讨论 chiplet/NUMA，但它是少数直接研究“CPU attention 推理”的论文，适合作为 attention 阶段 CPU 优化的并行参考
- 相关性评级：高
- 资料链接：https://arxiv.org/abs/2403.01273
- 备注：原文给出 16k context、4-bit quantized LLaMA-7B 最高 `2x` 加速；属于算法/内核路线，不是数据放置路线

### 5. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- 标题：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- 作者/机构：Joshua Ainslie、James Lee-Thorp、Michiel de Jong、Yury Zemlyanskiy、Federico Lebrón、Sumit Sanghai
- 年份或绝对日期：2023-05-22
- 类型：arXiv 预印本 / EMNLP 2023
- 硬件/平台：Transformer 架构层面
- 研究对象或工作负载：decoder inference、共享 KV heads
- 核心优化点：提出 grouped-query attention，让多个 query heads 共享一组 KV heads，在质量与速度间折中
- 与当前课题的关联点：它给“共享同一组 K/V 的 head 集合就是一个工作集单元”提供了结构基础，能支撑当前按 `kv_head` / 共享 KV 前缀讨论局部性的视角
- 相关性评级：中高
- 资料链接：https://arxiv.org/abs/2305.13245
- 备注：这是模型结构论文，不是 NUMA/CPU 优化论文；关联点主要是“共享 KV 工作集”的定义方式

### 6. HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading
- 标题：HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading
- 作者/机构：Cheng Luo、Zefan Cai、Hanshi Sun、Jinqi Xiao、Bo Yuan、Wen Xiao、Junjie Hu、Jiawei Zhao、Beidi Chen、Anima Anandkumar
- 年份或绝对日期：2025-02-18
- 类型：arXiv 预印本
- 硬件/平台：GPU + CPU RAM
- 研究对象或工作负载：long-context LLM inference、KV cache 管理
- 核心优化点：按 attention head 粒度把 KV cache 在 GPU 与 CPU RAM 间分放，只保留部分 heads 在 GPU 上
- 与当前课题的关联点：说明“head 粒度的 KV 管理”可以落成系统设计，对当前 `kv_head` 局部性分析是直接启发
- 相关性评级：中高
- 资料链接：https://arxiv.org/abs/2502.12574
- 备注：原文主要目标是极长上下文和显存压缩，不直接讨论 CPU chiplet/L3

### 7. MagicPIG: LSH Sampling for Efficient LLM Generation
- 标题：MagicPIG: LSH Sampling for Efficient LLM Generation
- 作者/机构：Zhuoming Chen、Ranajoy Sadhukhan、Zihao Ye、Yang Zhou、Jianyu Zhang、Niklas Nolte、Yuandong Tian、Matthijs Douze、Leon Bottou、Zhihao Jia、Beidi Chen
- 年份或绝对日期：2024-10-21（v1），2024-12-18（v4）
- 类型：arXiv 预印本
- 硬件/平台：GPU + CPU 异构系统
- 研究对象或工作负载：long-context LLM generation、KV cache / attention bottleneck
- 核心优化点：把 LSH hash table 和 attention computation 放到 CPU，缓解 GPU KV cache 压力
- 与当前课题的关联点：提供了“CPU 侧承担 attention / KV 相关工作”的异构方案参考，也说明 KV 管理和计算可沿工作集特征拆分
- 相关性评级：中高
- 资料链接：https://arxiv.org/abs/2410.16179
- 备注：这是近似 attention 路线，和当前精确 prefill attention 不是同一问题，但仍值得保留

### 8. Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects
- 标题：Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects
- 作者/机构：Mansi Choudhary、Karthik Sangaiah、Sonali Singh、Muhammad Osama、Lisa Wu Wills、Ganesh Dasika
- 年份或绝对日期：2025-11-03
- 类型：arXiv 预印本
- 硬件/平台：AMD MI300X、多 chiplet GPU
- 研究对象或工作负载：attention、MHA、GPU NUMA
- 核心优化点：把 attention heads 对齐到 GPU NUMA domain，利用 intra-chiplet cache reuse；原文称在 MI300X 上最高 `50%` 提升、L2 hit rate `80-97%`
- 与当前课题的关联点：这是当前最直接的“共享 KV 工作集 + 拓扑感知映射”证据来源，虽然平台是 GPU，但方法论与当前 CPU prefill 局部性假设高度同构
- 相关性评级：高
- 资料链接：https://arxiv.org/abs/2511.02132
- 备注：与 `info/进展.md` 里现有启发论文一致，可作为相关工作主干之一

### 9. ARCAS: Adaptive Runtime System for Chiplet-Aware Scheduling
- 标题：ARCAS: Adaptive Runtime System for Chiplet-Aware Scheduling
- 作者/机构：Alessandro Fogli、Bo Zhao、Peter Pietzuch、Jana Giceva
- 年份或绝对日期：2025-03-14
- 类型：arXiv 预印本
- 硬件/平台：chiplet-based CPU
- 研究对象或工作负载：memory-intensive parallel workloads、runtime scheduling
- 核心优化点：chiplet-aware task scheduling、hardware-aware memory allocation、fine-grained monitoring、自适应在局部性和聚合 cache 间切换
- 与当前课题的关联点：它是“chiplet-aware runtime”主线的代表性资料，可直接支撑当前“局部性 vs 聚合 L3 容量”的分析框架
- 相关性评级：高
- 资料链接：https://arxiv.org/abs/2503.11460
- 备注：不是 LLM 论文；更偏通用 runtime / 调度系统

### 10. CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System
- 标题：CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System
- 作者/机构：Alessandro Fogli、Bo Zhao、Peter Pietzuch、Jana Giceva
- 年份或绝对日期：2026-04-27（EuroSys 2026 会期；PDF 标注）
- 类型：会议论文 PDF / 预出版版本
- 硬件/平台：AMD EPYC Milan 单 socket 8 chiplets
- 研究对象或工作负载：runtime mapping、chiplet heterogeneity、local cache vs distributed cache
- 核心优化点：显式比较 `LocalCache` 与 `DistributedCache`，按运行时行为与数据规模调整任务放置
- 与当前课题的关联点：比 ARCAS 更强调 chiplet 内/跨 chiplet cache 策略切换，对当前 L3 slice 与 CCD 级放置分析有直接参考价值
- 相关性评级：高
- 资料链接：https://zbjob.github.io/EuroSys26.pdf
- 备注：命中的是 PDF，后续需补正式 DOI/出版页

### 11. Optimizing Sorting for Chiplet-Based CPUs
- 标题：Optimizing Sorting for Chiplet-Based CPUs
- 作者/机构：Alessandro Fogli、Peter Pietzuch、Jana Giceva
- 年份或绝对日期：2024-08
- 类型：VLDB ADMS 2024 论文
- 硬件/平台：AMD Ryzen / chiplet CPU
- 研究对象或工作负载：memory-intensive sorting
- 核心优化点：按 chiplet 粒度切分输入、扩展 memory hierarchy 以显式纳入分片 L3、按数据规模在单 chiplet 本地 L3 和多 chiplet 聚合 L3 之间切换
- 与当前课题的关联点：它把“数据规模相对本地 L3/聚合 L3 大小”的调度规则说得很清楚，和当前 decode/prefill 的 L3 行为分析高度相关
- 相关性评级：高
- 资料链接：https://vldb.org/workshops/2024/proceedings/ADMS/ADMS24_03.pdf
- 备注：ADMS 2024 workshop paper

### 12. A Performance Analysis of Chiplet-Based Systems
- 标题：A Performance Analysis of Chiplet-Based Systems
- 作者/机构：Neethu Bal Mallya、Panagiotis Strikos、Bhavishya Goel、Ahsen Ejaz、Ioannis Sourdis
- 年份或绝对日期：2025
- 类型：DATE 2025 会议论文
- 硬件/平台：multi-chiplet systems、NUMA systems
- 研究对象或工作负载：chiplet 架构的性能/成本分析
- 核心优化点：系统比较 chiplet 与 monolithic 的性能代价、memory hierarchy / interconnect / design choices 的影响；原文摘要称可能损失超过三分之一 monolithic 性能，但部分可通过设计选择收回
- 与当前课题的关联点：为“chiplet 本身为什么会带来性能损失、哪些硬件层因素最关键”提供背景证据
- 相关性评级：中高
- 资料链接：https://research.chalmers.se/en/publication/546843
- 备注：偏硬件分析，不直接讨论 LLM 或 runtime

### 13. MEMPLEX: A Memory System with Replication and Migration of Data for Multi-Chiplet NUMA Architectures
- 标题：MEMPLEX: A Memory System with Replication and Migration of Data for Multi-Chiplet NUMA Architectures
- 作者/机构：Neethu Bal Mallya、Bhavishya Goel、Ioannis Sourdis
- 年份或绝对日期：2025
- 类型：ICS 2025 会议论文
- 硬件/平台：multi-chiplet NUMA architecture、HBM + DDR
- 研究对象或工作负载：chiplet memory system、data replication / migration
- 核心优化点：在 multi-chiplet NUMA memory system 中结合 replication 与 migration，降低 remote access 开销
- 与当前课题的关联点：虽然更偏硬件/内存系统，但它给“数据迁移/复制是否能缓解 chiplet NUMA 代价”提供了更底层的相关工作入口
- 相关性评级：中
- 资料链接：https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-36.pdf
- 备注：不是软件 runtime 方案；更适合放在硬件启发或背景类

### 14. CacheAware: Data Locality-Aware Scheduling for Distributed Memory Systems
- 标题：CacheAware: Data Locality-Aware Scheduling for Distributed Memory Systems
- 作者/机构：Haifa A. Alanazi、Abdulaziz G. Alanazi、Nasser S. Albalawi
- 年份或绝对日期：2026-03
- 类型：期刊论文
- 硬件/平台：NUMA / distributed memory systems
- 研究对象或工作负载：compiler-runtime 协同的数据局部性感知调度
- 核心优化点：编译器标注数据访问模式、runtime 按 locality hint 和 cache affinity 做放置，并根据运行时 cache miss 动态迁移
- 与当前课题的关联点：不是 chiplet/LLM 专题，但“编译/运行时协同 + cache affinity + 动态迁移”这条方法线值得保留
- 相关性评级：中
- 资料链接：https://www.mdpi.com/2073-431X/15/3/181
- 备注：更偏 HPC / distributed memory；与当前课题的连接属于方法启发，不是直接证据

### 15. Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving
- 标题：Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving
- 作者/机构：Juntao Zhao、Jiuru Li、Chuan Wu
- 年份或绝对日期：2025-05-19
- 类型：arXiv 预印本
- 硬件/平台：commodity CPU、x86 AVX2/AVX512、ARM NEON、NUMA CPU
- 研究对象或工作负载：CPU LLM serving、prefill/decode 分相执行
- 核心优化点：显式把 prefill 和 decode 的执行计划拆开，并指出现有静态 per-NUMA 模型分片对两阶段并不同时最优
- 与当前课题的关联点：它直接命中当前仓库的 `prefill/decode` 分相主线，也比第一轮资料更直接地把 CPU LLM serving 和 NUMA 执行计划绑在一起
- 相关性评级：高
- 资料链接：https://arxiv.org/abs/2507.18454
- 备注：不是 chiplet 专题，但摘要已明确提到现有 CPU 解法的静态 per-NUMA 分区问题

### 16. Hydragen: High-Throughput LLM Inference with Shared Prefixes
- 标题：Hydragen: High-Throughput LLM Inference with Shared Prefixes
- 作者/机构：Jordan Juravsky、Bradley Brown、Ryan Ehrlich、Daniel Y. Fu、Christopher Ré、Azalia Mirhoseini
- 年份或绝对日期：2024-02-07
- 类型：arXiv 预印本 / ICML 2024
- 硬件/平台：GPU 为主的 LLM serving 环境
- 研究对象或工作负载：shared-prefix batch decoding、KV cache overlap
- 核心优化点：把 attention 拆成 shared prefix 与 unique suffix 两部分，跨序列批处理 prefix attention，减少重复 KV 读取
- 与当前课题的关联点：它为“共享 KV 前缀就是一个工作集单元”提供了很直接的系统实现证据，适合映射到当前 `kv_head` 局部性分析
- 相关性评级：高
- 资料链接：https://arxiv.org/abs/2402.05099
- 备注：平台不是 CPU chiplet；但 shared-prefix attention 的工作集定义非常贴近当前问题

### 17. ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition
- 标题：ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition
- 作者/机构：Lu Ye、Ze Tao、Yong Huang、Yang Li
- 年份或绝对日期：2024-08
- 类型：ACL 2024 论文
- 硬件/平台：LLM serving 系统
- 研究对象或工作负载：多租户 LLM self-attention、shared system prompt、KV cache 管理
- 核心优化点：通过 prefix-aware KV cache 与 two-phase partition 改进 self-attention 的 data locality，并在运行时共享匹配前缀的 KV tensor
- 与当前课题的关联点：它把“共享前缀 -> 共享 KV -> locality-aware attention kernel”这条链条写得很完整，是当前 shared-workset 主线的重要补充
- 相关性评级：高
- 资料链接：https://aclanthology.org/2024.acl-long.623/
- 备注：不是 NUMA 论文；更像 attention kernel / KV cache 组织的系统工作

### 18. KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference
- 标题：KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference
- 作者/机构：Huan Yang、Renji Zhang、Deyu Zhang
- 年份或绝对日期：2025-03-17
- 类型：arXiv 预印本
- 硬件/平台：LLM / MLLM serving 系统
- 研究对象或工作负载：多用户 KV cache 共享、prefix/semantic cache reuse
- 核心优化点：把严格前缀匹配扩展到基于语义相似度的细粒度 KV 复用，缓解只靠 prefix cache 的命中限制
- 与当前课题的关联点：虽然不直接讨论 chiplet/L3，但它补上了“共享工作集不一定要求完全相同前缀”的系统视角
- 相关性评级：中高
- 资料链接：https://arxiv.org/abs/2503.16525
- 备注：相关性更偏 KV reuse 策略，不是线程/NUMA 放置

### 19. Unlocking Optimal LLM Performance on AMD EPYC CPUs with vLLM
- 标题：Unlocking Optimal LLM Performance on AMD EPYC CPUs with vLLM
- 作者/机构：AMD
- 年份或绝对日期：2025-11（搜索结果相对时间推断）
- 类型：官方技术文章
- 硬件/平台：AMD EPYC 9555、vLLM、ZenDNN
- 研究对象或工作负载：CPU-only vLLM 推理、socket / NUMA 配置比较
- 核心优化点：给出 AMD EPYC 上 `vLLM` 的软硬件栈与单/双 socket 测试口径，强调配置透明度、SLA 与系统级参数对结果的影响
- 与当前课题的关联点：这是当前最直接的 `AMD EPYC + vLLM` 一手资料之一，可补足第一轮“更多是官方文档、缺直接工程结果”的缺口
- 相关性评级：高
- 资料链接：https://www.amd.com/en/blogs/2025/unlocking-optimal-llm-performance-on-amd-epyc--cpus-with-vllm.html
- 备注：不是 peer-reviewed 论文；`2025-11` 来自搜索结果“Published: 4 months ago”按当前日期推断

### 20. Speculative LLM Inference on the 5th Gen AMD EPYC Processors with Parallel Draft Models (PARD) & AMD Platform Aware Compute Engine (AMD PACE)
- 标题：Speculative LLM Inference on the 5th Gen AMD EPYC Processors with Parallel Draft Models (PARD) & AMD Platform Aware Compute Engine (AMD PACE)
- 作者/机构：Dinesh Chitlangia、Chinmaya Chavan、Karthik Mudur、Srinivas Bhimavarapu
- 年份或绝对日期：2025-07-23
- 类型：官方技术文章
- 硬件/平台：dual-socket 5th Gen AMD EPYC 9005
- 研究对象或工作负载：CPU-only LLM serving、speculative decoding
- 核心优化点：把 speculative decoding、parallel draft model 和 AMD PACE 组合到 CPU serving 栈中，缓解 auto-regressive 的 memory-bandwidth 限制
- 与当前课题的关联点：虽然主线是 speculative decoding，不是 chiplet/L3，但它提供了 AMD EPYC 上 CPU-only LLM serving 的一手工程证据
- 相关性评级：中高
- 资料链接：https://www.amd.com/en/developer/resources/technical-articles/2025/speculative-llm-inference-on-the-5th-gen-amd-epyc-processors-wit.html
- 备注：更偏 serving 工程与解码加速，不直接讨论 NUMA 绑核

### 21. ZenDNN 5.2: Accelerating vLLM Inference on AMD EPYC CPUs
- 标题：ZenDNN 5.2: Accelerating vLLM Inference on AMD EPYC CPUs
- 作者/机构：AMD
- 年份或绝对日期：2026-03（搜索结果标注 Published: 2 weeks ago）
- 类型：官方技术文章
- 硬件/平台：`2 x AMD EPYC 9755 128-Core Processor`、vLLM 0.15.1+cpu、ZenTorch 5.2
- 研究对象或工作负载：vLLM V1 engine 在 AMD EPYC 上的 LLM 推理
- 核心优化点：强调单个超大实例会遇到 wide compute fabric 与 memory contention，建议用 `numactl` 拆成多实例并绑定本地内存池，以提升 decode 吞吐
- 与当前课题的关联点：它直接命中本课题最关心的 `AMD EPYC + vLLM + NUMA/绑核/内存策略`，而且显式提到了 cache locality 与本地内存池
- 相关性评级：高
- 资料链接：https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-accelerating-vllm-inference-on-amd-epyc-cpus.html
- 备注：不是论文；日期只精确到 `2026-03`，因为当前证据来自搜索结果摘要

### 22. Realizing the AMD Exascale Heterogeneous Processor Vision
- 标题：Realizing the AMD Exascale Heterogeneous Processor Vision
- 作者/机构：Alan Smith、Gabriel H. Loh、Michael J. Schulte、Mike Ignatowski、Samuel Naffziger 等，AMD
- 年份或绝对日期：2024-06
- 类型：ISCA 2024 industry paper
- 硬件/平台：AMD MI300A / MI300X、chiplet GPU / APU
- 研究对象或工作负载：MI300 系列架构、partitioning、heterogeneous integration
- 核心优化点：给出 MI300X 的 `NPS1/NPS4` 与多种 XCD partitioning 设计背景，并把 chiplet 互连与 AI/HPC 工作负载放在统一架构里讨论
- 与当前课题的关联点：它能作为 `MI300X attention NUMA` 论文背后的硬件一手背景，也能帮助对照 `CCD/XCD` 分片与局部性概念
- 相关性评级：中高
- 资料链接：https://www.computermachines.org/joe/publications/pdfs/isca2024_exascale.pdf
- 备注：不是 attention 优化论文；更偏硬件架构一手资料

### 23. Deep dive into the MI300 compute and memory partition modes
- 标题：Deep dive into the MI300 compute and memory partition modes
- 作者/机构：David Doscher、AMD ROCm
- 年份或绝对日期：2025-03-27
- 类型：ROCm 官方博客
- 硬件/平台：AMD Instinct MI300X、ROCm、CPX / SPX、NPS1 / NPS4
- 研究对象或工作负载：MI300X 的 compute partition 与 memory partition 使用方式
- 核心优化点：解释 `CPX` / `SPX`、`NPS1` / `NPS4` 的软件暴露形式，并指出 `CPX + NPS4` 有助于把 XCD 访问局部化到最近的 HBM 域
- 与当前课题的关联点：这是把 `multi-die GPU / GPU NUMA` 落到可执行软件接口上的一手资料，可直接辅助理解 `XCD/NPS` 与拓扑感知放置
- 相关性评级：高
- 资料链接：https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
- 备注：不是论文；偏硬件/软件接口说明，但与当前 topic 高度相关

### 24. LFOC+: A Fair OS-level Cache-Clustering Policy for Commodity Multicore Systems
- 标题：LFOC+: A Fair OS-level Cache-Clustering Policy for Commodity Multicore Systems
- 作者/机构：Juan Carlos Saez、Fernando Castro、Graziano Fanizzi、Manuel Prieto-Matias
- 年份或绝对日期：2024-02-12
- 类型：arXiv 预印本 / IEEE TC 2022 对应开放版本
- 硬件/平台：commodity multicore、支持 LLC partitioning 的 Linux 系统
- 研究对象或工作负载：OS 级 cache clustering / LLC partitioning
- 核心优化点：基于 PMC 动态分类应用的 cache sensitivity / contentiousness，在 Linux kernel 中实现 fairness-aware cache clustering
- 与当前课题的关联点：它不是 chiplet/LLM 论文，但补足了“OS 级如何基于 LLC 行为做运行时聚类与隔离”的背景
- 相关性评级：中
- 资料链接：https://arxiv.org/abs/2402.07693
- 备注：更偏通用 multicore cache management；不直接讨论 NUMA 或 chiplet

### 25. Adaptive NUMA-aware data placement and task scheduling for analytical workloads in main-memory column-stores
- 标题：Adaptive NUMA-aware data placement and task scheduling for analytical workloads in main-memory column-stores
- 作者/机构：Iraklis Psaroudakis、Tobias Scheuer、Norman May、Abdelkader Sellami、Anastasia Ailamaki
- 年份或绝对日期：2016-10-01
- 类型：PVLDB 论文
- 硬件/平台：多 socket NUMA server、main-memory column-store
- 研究对象或工作负载：NUMA-aware data placement、inter-socket task stealing
- 核心优化点：根据 workload intensity 自适应调整数据放置与跨 socket task stealing，避免内存密集型任务的有害远程窃取
- 与当前课题的关联点：虽然不是 LLM / chiplet 场景，但它是 topology-aware data placement + task scheduling 的经典系统工作，适合作为调度背景基线
- 相关性评级：中
- 资料链接：https://www.vldb.org/pvldb/vol10/p37-psaroudakis.pdf
- 备注：属于较早的 NUMA-aware runtime 论文，用于背景参照而非直接迁移结论

## 临时分组
### 可能属于“最直接相关”
- ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs
- Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving
- CPU - vLLM / vllm.platforms.cpu
- Unlocking Optimal LLM Performance on AMD EPYC CPUs with vLLM
- ZenDNN 5.2: Accelerating vLLM Inference on AMD EPYC CPUs
- Speculative LLM Inference on the 5th Gen AMD EPYC Processors with PARD and PACE
- Llama 2 Inference from Intel with DeepSpeed
- NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention

### 可能属于“高相关方法启发”
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- Hydragen: High-Throughput LLM Inference with Shared Prefixes
- ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition
- KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference
- HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading
- MagicPIG: LSH Sampling for Efficient LLM Generation
- Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects

### 可能属于“相近硬件启发”
- Realizing the AMD Exascale Heterogeneous Processor Vision
- Deep dive into the MI300 compute and memory partition modes
- A Performance Analysis of Chiplet-Based Systems
- MEMPLEX: A Memory System with Replication and Migration of Data for Multi-Chiplet NUMA Architectures

### 可能属于“背景与系统方法”
- ARCAS: Adaptive Runtime System for Chiplet-Aware Scheduling
- CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System
- Optimizing Sorting for Chiplet-Based CPUs
- LFOC+: A Fair OS-level Cache-Clustering Policy for Commodity Multicore Systems
- Adaptive NUMA-aware data placement and task scheduling for analytical workloads in main-memory column-stores
- CacheAware: Data Locality-Aware Scheduling for Distributed Memory Systems
