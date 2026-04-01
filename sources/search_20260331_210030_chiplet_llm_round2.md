# 2026-03-31 Chiplet LLM 相关工作第二轮补漏检索原始结果

## 元数据
- 时间：2026-03-31 21:00:30 +0800
- 任务：`tasks.json` 任务 `26`
- 执行方式：内置联网检索
- 说明：shell 环境里仍没有 `PARALLEL_API_KEY` / `OPENROUTER_API_KEY`，且仓库内未找到 `parallel_web.py` / `research_lookup.py`，因此继续沿用手工保存检索轨迹的 fallback
- 目标：补齐第一轮命中较少方向，至少覆盖 `LLM + chiplet/NUMA CPU`、`attention/KV 共享工作集`、`multi-die GPU / GPU NUMA`、`分片 LLC / 拓扑感知调度`

## 查询组 1：LLM serving on CPU / AMD EPYC / vLLM / NUMA
- 查询词：
  - `Sandwich CPU LLM Serving prefill decode NUMA arxiv`
  - `Unlocking Optimal LLM Performance on AMD EPYC CPUs with vLLM`
  - `ZenDNN 5.2 Accelerating vLLM Inference on AMD EPYC CPUs`
  - `Speculative LLM Inference on the 5th Gen AMD EPYC Processors with PARD and PACE`
- 主要命中：
  - `Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving`
    - 链接：https://arxiv.org/abs/2507.18454
    - 命中原因：直接研究 CPU LLM serving，并明确指出现有方案对 prefill / decode 套同一套 per-NUMA 划分是次优
  - `Unlocking Optimal LLM Performance on AMD EPYC™ CPUs with vLLM`
    - 链接：https://www.amd.com/en/blogs/2025/unlocking-optimal-llm-performance-on-amd-epyc--cpus-with-vllm.html
    - 命中原因：AMD 官方一手工程文章，直接给出 EPYC + vLLM 的 socket / 内存 / 软件栈配置
  - `ZenDNN 5.2: Accelerating vLLM Inference on AMD EPYC™ CPUs`
    - 链接：https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-accelerating-vllm-inference-on-amd-epyc-cpus.html
    - 命中原因：直接讨论 `vLLM V1`、`numactl` 多实例、内存交错和 AMD EPYC 上的 vLLM 吞吐提升
  - `Speculative LLM Inference on the 5th Gen AMD EPYC Processors with Parallel Draft Models (PARD) & AMD Platform Aware Compute Engine (AMD PACE)`
    - 链接：https://www.amd.com/en/developer/resources/technical-articles/2025/speculative-llm-inference-on-the-5th-gen-amd-epyc-processors-wit.html
    - 命中原因：AMD 官方文章，直接给出 dual-socket EPYC 9005 上的 CPU-only LLM serving / speculative decoding 方案
- 备注：
  - 第一轮已收录 `ArcLight`、`vLLM CPU docs`、Intel `DeepSpeed CPU`，本轮没有重复加入候选池
  - 直接同时覆盖 `AMD EPYC + vLLM + chiplet/L3` 的 peer-reviewed 论文仍未命中

## 查询组 2：shared prefix / KV reuse / attention working set
- 查询词：
  - `Hydragen shared prefixes arxiv`
  - `ChunkAttention prefix-aware KV cache`
  - `KVShare semantic-aware key-value cache sharing`
  - `shared prefix KV cache LLM inference`
- 主要命中：
  - `Hydragen: High-Throughput LLM Inference with Shared Prefixes`
    - 链接：https://arxiv.org/abs/2402.05099
    - 命中原因：直接把 shared prefix attention 分解成 shared prefix + unique suffix 两部分，减少重复 KV 读取
  - `ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition`
    - 链接：https://aclanthology.org/2024.acl-long.623/
    - 命中原因：prefix-aware KV cache + two-phase partition，明确把 data locality 写进 self-attention kernel 设计
  - `KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference`
    - 链接：https://arxiv.org/abs/2503.16525
    - 命中原因：从严格前缀匹配扩展到语义相似 KV 复用，补上“共享工作集不一定来自完全相同前缀”的资料
- 备注：
  - 第一轮已收录 `GQA`、`HeadInfer`、`MagicPIG`，本轮重点补 shared-prefix / KV-reuse 方向
  - 仍未找到把 `kv_head` / shared KV prefix 直接映射到 `CCD/CCX/L3 slice` 的 CPU 论文

## 查询组 3：MI300X / multi-die GPU / GPU NUMA / partitioning
- 查询词：
  - `Realizing the AMD Exascale Heterogeneous Processor Vision MI300X NUMA`
  - `Deep dive into the MI300 compute and memory partition modes`
  - `MI300X XCD NPS4 NUMA`
  - `multi-die GPU NUMA attention MI300X`
- 主要命中：
  - `Realizing the AMD Exascale Heterogeneous Processor Vision`
    - 链接：https://www.computermachines.org/joe/publications/pdfs/isca2024_exascale.pdf
    - 命中原因：AMD / ISCA 2024 一手架构论文，明确给出 `MI300X` 支持 `NPS1/NPS4` 与多种 XCD partitioning
  - `Deep dive into the MI300 compute and memory partition modes`
    - 链接：https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
    - 命中原因：ROCm 官方博客，补充 `SPX/CPX`、`NPS1/NPS4` 的软件暴露方式与带宽差异
  - 第一轮重复确认：
    - `Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects`
    - 链接：https://arxiv.org/abs/2511.02132
    - 说明：仍然是当前最直接的“attention + NUMA + shared KV workset”论文，本轮不重复加入候选池

## 查询组 4：LLC-aware / NUMA-aware / topology-aware scheduling
- 查询词：
  - `LFOC+ cache clustering LLC arxiv`
  - `Adaptive NUMA-aware data placement and task scheduling pdf`
  - `LLC aware scheduling runtime paper`
  - `partitioned LLC topology aware scheduling paper`
- 主要命中：
  - `LFOC+: A Fair OS-level Cache-Clustering Policy for Commodity Multicore Systems`
    - 链接：https://arxiv.org/abs/2402.07693
    - 命中原因：直接研究 OS 级 LLC partition / cache clustering，并在 Linux kernel 中实现
  - `Adaptive NUMA-aware data placement and task scheduling for analytical workloads in main-memory column-stores`
    - 链接：https://www.vldb.org/pvldb/vol10/p37-psaroudakis.pdf
    - 命中原因：经典 NUMA-aware data placement + task stealing 调度论文，可作为 `拓扑感知调度` 背景基线
- 次级命中：
  - Linux `cache-aware scheduling` 补丁讨论
    - 说明：是系统实现一手资料，但当前更像内核工程推进，不如上面两篇适合放进正式清单主干；先不入候选池

## 本轮新增保留候选
- Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving
- Unlocking Optimal LLM Performance on AMD EPYC™ CPUs with vLLM
- ZenDNN 5.2: Accelerating vLLM Inference on AMD EPYC™ CPUs
- Speculative LLM Inference on the 5th Gen AMD EPYC Processors with PARD and PACE
- Hydragen: High-Throughput LLM Inference with Shared Prefixes
- ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition
- KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference
- Realizing the AMD Exascale Heterogeneous Processor Vision
- Deep dive into the MI300 compute and memory partition modes
- LFOC+: A Fair OS-level Cache-Clustering Policy for Commodity Multicore Systems
- Adaptive NUMA-aware data placement and task scheduling for analytical workloads in main-memory column-stores

## 本轮结论性备注
- `CPU LLM serving + prefill/decode phase-aware + NUMA` 方向补到了更直接的资料，其中 `Sandwich` 最贴近当前 `prefill/decode` 分相主线。
- `shared prefix / KV reuse / prefix-aware locality` 方向补到了三篇更直接的工作，能把“共享 KV 工作集”从结构、系统和 cache reuse 三个角度补齐。
- `MI300X` 方向补到了 AMD 一手架构和 ROCm 分区资料，能更扎实地支撑 `XCD/NPS` 与 `attention NUMA` 的硬件背景。
- 仍未命中把 `Ave L3 Miss Latency`、`Remote L3/CCX` 或 `Remote DRAM Reads %` 这类 CPU 指标直接映射到 LLM attention 调度的论文。
