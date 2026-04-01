# 2026-03-31 Chiplet LLM 相关工作第一轮广覆盖检索原始结果

## 元数据
- 时间：2026-03-31 20:27:37 +0800
- 任务：`tasks.json` 任务 `25`
- 执行方式：内置联网检索
- 说明：shell 环境里没有 `PARALLEL_API_KEY` / `OPENROUTER_API_KEY`，因此未使用 `parallel_web.py` / `research_lookup.py`
- 目标：建立尽量全的候选资料池，优先一手来源；正式整理延后到任务 `26` / `27`

## 查询组 1：LLM inference + CPU + NUMA / chiplet
- 查询词：
  - `LLM inference CPU NUMA paper arxiv llama.cpp`
  - `site:docs.vllm.ai CPU_VISIBLE_MEMORY_NODES VLLM_CPU_OMP_THREADS_BIND vLLM CPU docs`
  - `site:intel.com CPU-optimized LLM inference NUMA OpenVINO oneDNN`
- 主要命中：
  - `ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs`
    - 链接：https://arxiv.org/abs/2603.07770
    - 命中原因：直接研究 many-core CPU / NUMA 上的 LLM 推理
  - `CPU - vLLM`
    - 链接：https://docs.vllm.ai/en/stable/getting_started/installation/cpu/
    - 命中原因：官方 CPU 文档直接给出 `VLLM_CPU_OMP_THREADS_BIND`、`CPU_VISIBLE_MEMORY_NODES`、`tensor-parallel-size` 与 NUMA 的口径
  - `vllm.platforms.cpu`
    - 链接：https://docs.vllm.ai/en/stable/api/vllm/platforms/cpu/
    - 命中原因：源码/API 级别暴露 `discover_numa_topology`
  - `Llama 2 Inference from Intel with DeepSpeed`
    - 链接：https://www.intel.com/content/www/us/en/developer/articles/technical/llama-2-on-xeon-scalable-processor-with-deepspeed.html
    - 命中原因：官方 CPU LLM 推理文章，明确 `--bind_cores_to_rank`、`numactl`、`Indirect Access KV Cache`
- 低质量或次级命中：
  - `llama.cpp` discussion / gist / benchmark blog
    - 说明：能看到实践者对 NUMA 的经验反馈，但大多不是一手学术来源，本轮不作为主候选

## 查询组 2：attention / KV 共享工作集 / head 粒度管理
- 查询词：
  - `attention KV cache locality NUMA paper GPU XCD grouped query attention`
  - `NoMAD-Attention arxiv CPU attention inference`
  - `HeadInfer arxiv KV cache heads inference`
  - `MagicPIG arxiv KV cache CPU`
  - `grouped-query attention paper arxiv 2023`
- 主要命中：
  - `NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention`
    - 链接：https://arxiv.org/abs/2403.01273
    - 命中原因：少数直接做 CPU attention 推理的论文
  - `GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints`
    - 链接：https://arxiv.org/abs/2305.13245
    - 命中原因：提供“多个 query heads 共享一组 KV heads”的结构背景
  - `HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading`
    - 链接：https://arxiv.org/abs/2502.12574
    - 命中原因：按 head 粒度管理 KV cache，和当前 `kv_head` 视角接近
  - `MagicPIG: LSH Sampling for Efficient LLM Generation`
    - 链接：https://arxiv.org/abs/2410.16179
    - 命中原因：把部分 attention / KV 相关工作放到 CPU，属于 GPU/CPU 异构路线

## 查询组 3：multi-die GPU / GPU NUMA / attention locality
- 查询词：
  - `multi-die GPU NUMA attention MI300X XCD paper arxiv`
  - `GPU NUMA scheduling cache locality multi-die GPU paper`
- 主要命中：
  - `Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects`
    - 链接：https://arxiv.org/abs/2511.02132
    - 命中原因：当前最直接的 `attention + NUMA + chiplet cache reuse` 论文，平台是 AMD MI300X
- 观察：
  - 该方向一手资料不算多，但命中质量高，后续任务应继续沿 `MI300X`、`XCD`、`head-first mapping` 扩展

## 查询组 4：chiplet runtime / 分片 L3 / cache-aware scheduling
- 查询词：
  - `site:arxiv.org chiplet-aware scheduling runtime ARCAS paper`
  - `chiplet CPU sliced LLC scheduling paper topology aware cache locality`
  - `chiplet CPU cache locality runtime scheduling L3 paper`
  - `cache-aware scheduling Linux LLC NUMA paper`
- 主要命中：
  - `ARCAS: Adaptive Runtime System for Chiplet-Aware Scheduling`
    - 链接：https://arxiv.org/abs/2503.11460
    - 命中原因：chiplet-aware runtime 主线
  - `CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System`
    - 链接：https://zbjob.github.io/EuroSys26.pdf
    - 命中原因：更强调 `LocalCache` vs `DistributedCache`
  - `Optimizing Sorting for Chiplet-Based CPUs`
    - 链接：https://vldb.org/workshops/2024/proceedings/ADMS/ADMS24_03.pdf
    - 命中原因：把“数据规模相对 chiplet-local L3 / aggregated L3”的调度口径写得很明确
  - `CacheAware: Data Locality-Aware Scheduling for Distributed Memory Systems`
    - 链接：https://www.mdpi.com/2073-431X/15/3/181
    - 命中原因：不是 chiplet 专题，但给出 cache affinity + dynamic migration 的方法框架

## 查询组 5：chiplet memory hierarchy / NUMA penalty / system background
- 查询词：
  - `"A Performance Analysis of Chiplet-Based Systems" authors`
  - `MEMPLEX chiplet NUMA replication migration pdf`
- 主要命中：
  - `A Performance Analysis of Chiplet-Based Systems`
    - 链接：https://research.chalmers.se/en/publication/546843
    - 命中原因：概括 chiplet 系统相对 monolithic 的性能代价和设计权衡
  - `MEMPLEX: A Memory System with Replication and Migration of Data for Multi-Chiplet NUMA Architectures`
    - 链接：https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-36.pdf
    - 命中原因：从 memory system 角度讨论 multi-chiplet NUMA 的 replication / migration

## 本轮去重前保留的核心候选
- ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs
- CPU - vLLM / vllm.platforms.cpu
- Llama 2 Inference from Intel with DeepSpeed
- NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading
- MagicPIG: LSH Sampling for Efficient LLM Generation
- Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects
- ARCAS: Adaptive Runtime System for Chiplet-Aware Scheduling
- CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System
- Optimizing Sorting for Chiplet-Based CPUs
- A Performance Analysis of Chiplet-Based Systems
- MEMPLEX: A Memory System with Replication and Migration of Data for Multi-Chiplet NUMA Architectures
- CacheAware: Data Locality-Aware Scheduling for Distributed Memory Systems

## 本轮结论性备注
- “直接针对 `LLM inference + chiplet CPU`” 的一手论文命中不多，`ArcLight` 是目前最直接的一篇。
- “`vLLM CPU + NUMA`” 命中更多是官方文档而非论文。
- “共享 KV 工作集 + 拓扑感知放置” 这一思路在 GPU NUMA 方向证据更强，在 CPU chiplet 方向仍需继续补检。
