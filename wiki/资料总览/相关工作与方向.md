# 相关工作与方向

整理 chiplet/NUMA/分片缓存硬件上应用优化相关的资料，优先关注 LLM 推理、attention、KV 共享工作集、拓扑感知调度。

## 一、核心论文（建议深读）

### Chiplet CPU 直接相关

1. **CHARM: Chiplet Heterogeneity-Aware Runtime Mapping System** (EuroSys 2026)
   - 显式比较 LocalCache 与 DistributedCache，按运行时行为与数据规模调整任务放置。直接启发 `group_span` 或 L3 subgroup 扩张策略。

2. **Optimizing Sorting for Chiplet-Based CPUs** (VLDB ADMS 2024)
   - 按 chiplet 粒度切分输入，根据数据规模在本地 L3 与聚合 L3 之间切换。适合映射 prefill 不同长度点的最优放置策略。

3. **OLAP on Modern Chiplet-Based Processors** (PVLDB 2024)
   - 提出 within-chiplet 与 across-chiplet 两类部署思路，给出工作集小于/大于单 chiplet L3 时的策略切换规则。

4. **Memory Performance of AMD EPYC Rome and Intel Cascade Lake SP** (ICPE 2022)
   - 详细评估 EPYC Rome memory hierarchy，补强 CCD/CCX/L3/remote cache 成本理解。

5. **TiNA: Tiered Network Buffer Architecture for Fast Networking in Chiplet-based CPUs** (ASPLOS 2026)
   - 分析 SNC 降低跨 chiplet 延迟但缩小 LLC 容量的权衡，与 LocalCache/DistributedCache 主线高度同构。

6. **Effects of Poor Workload Partitioning on System Performance for Chiplet-Based Systems** (Electronics, 2026.03)
   - University of Florida。系统量化 chiplet 分区不当的代价：inter-chiplet 延迟恶化最高 10×，通信开销在 16-chiplet 系统中占 85% 执行时间。优化分区后 inter-chiplet 流量降低 87.4%，吞吐提升 8.75×。可作为 chiplet 优化必要性的定量 motivation 引用。
   - 论文：`原始资料/papers/Effects_of_Poor_Workload_Partitioning_Chiplet.pdf`

### GPU Chiplet + Attention

6. **Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects** (arXiv 2025)
   - 将共享同一组 K/V 的 attention workgroup 映射到同一 GPU NUMA 域。方法论与 CPU prefill 局部性假设高度同构，是最直接的"共享 KV 工作集 + 拓扑感知放置"证据来源。

7. **Leveraging Chiplet-Locality for Efficient Memory Mapping in Multi-Chip Module GPUs** (MICRO 2025)
   - 识别 GPU 应用的 chiplet-locality 页组模式，方法线与"KV/workset → L3 subgroup"映射接近。

8. **Deep dive into the MI300 compute and memory partition modes** (ROCm 官方博客 2025)
   - 解释 CPX/SPX、NPS1/NPS4 的软件暴露形式，是理解 XCD/NPS 与拓扑感知放置的一手资料。

### CPU LLM 推理

9. **ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs** (arXiv 2026)
   - 显式针对 cross-NUMA 开销做内存管理、线程调度和细粒度 TP，直接对应本机绑核、NUMA 内存策略。

10. **Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving** (arXiv 2025)
    - 显式拆分 prefill/decode 执行计划，指出静态 per-NUMA 分区对两阶段并不同时最优。直接命中当前分相和 NUMA 划分主线。

### LLM Serving / KV / Locality

11. **GQA: Training Generalized Multi-Query Transformer Models** (EMNLP 2023)
    - 提出 grouped-query attention，为"共享同一组 K/V 的 head 集合 = 工作集单元"提供结构基础。

12. **Hydragen: High-Throughput LLM Inference with Shared Prefixes** (ICML 2024)
    - 将 shared prefix 与 unique suffix 分开批处理，为"共享 KV 前缀 = locality 单元"提供系统实现证据。

13. **ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache** (ACL 2024)
    - 把 prefix-aware KV cache + two-phase partition + locality-aware attention 链条写得最完整。

## 二、方法支撑论文

- **NoMAD-Attention** (arXiv 2024)：CPU attention 的 MAD-free 优化路线，可作为算子公式路线的对照。
- **HeadInfer** (arXiv 2025)：按 head 粒度管理 KV cache 可落成系统设计，但主场景是 GPU+CPU 异构。
- **KVShare** (arXiv 2025)：共享工作集不一定要求严格同前缀，补足视角。
- **Realizing the AMD Exascale Heterogeneous Processor Vision** (ISCA 2024)：MI300X/XCD/NPS 硬件背景。
- **LFOC+** (IEEE TC 2022)：OS 级基于 cache 行为做运行时聚类的可行性证据。
- **Adaptive NUMA-aware data placement for analytical workloads** (PVLDB 2016)：topology-aware data placement 经典基线。
- **MEMPLEX** (ICS 2025)：multi-chiplet NUMA 内存系统中数据复制与迁移，偏硬件。
- **A Performance Analysis of Chiplet-Based Systems** (DATE 2025)：chiplet vs monolithic 性能代价的系统比较。
- **P-MOSS** (SIGMOD 2026)：用 Decision Transformer + PMU 硬件计数器指导 NUMA 节点上索引查询与数据的协同放置，B⁺-Tree 索引实现 6× 吞吐提升。ML-guided chiplet/NUMA 调度决策的方法线可参考。论文：`原始资料/papers/P-MOSS.pdf`
- **Delegato** (MICRO 2025)：提出 delegated/migrating 远原子操作减少 chiplet 间数据搬运，比集中式 AMO 加速 1.07–1.13×。BSC/UPC 团队。若 attention 实现涉及跨 chiplet 原子操作（如 softmax reduction），可参考其思路。论文：`原始资料/papers/Delegato.pdf`
- **Performance Analysis of Memory Systems for Multi-Chiplet NUMA Architectures** (Chalmers Licentiate Thesis, 2025)：系统量化 chiplet CPU 相比 monolithic 设计最高 33% 的性能损失，MEMPLEX 的前置工作。论文：`原始资料/papers/Chalmers_Multi-Chiplet_NUMA_Thesis.pdf`
- **The Fake-Busy and True-Idle Problems of Running Graph Applications on Chiplet-Based Multi-Cores** (IISWC 2025)：分析图计算在 chiplet OoO 核心上因流水线利用不充分导致的"假忙真闲"问题，提出 Vector Runahead + Virtualized Multi-Threading。方法线上与 attention 计算图的 irregular memory access 有可类比之处。论文：`原始资料/papers/The_Fake-Busy_and_True-Idle_Problems_of_Running_Graph_Applications_on_Chiplet-Based_Multi-Cores.pdf`
- **SEEChiplet** (电子与信息学报, 2024.12)：中科院计算所基于 gem5 扩展的 chiplet CPU 系统级模拟器，支持私有 L3、MCM/2.5D 封装、IO-Die/Mesh 拓扑建模。若需要 chiplet 模拟实验可参考。注意：需通过 jeit.ac.cn 手动下载 PDF（约 5 MB），本地未下载成功。

## 三、工程资料（不混入论文结论）

- [vLLM CPU 官方文档](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/)
- [Unlocking Optimal LLM Performance on AMD EPYC CPUs with vLLM](https://www.amd.com/en/blogs/2025/unlocking-optimal-llm-performance-on-amd-epyc--cpus-with-vllm.html)
- [ZenDNN 5.2: Accelerating vLLM Inference on AMD EPYC CPUs](https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-accelerating-vllm-inference-on-amd-epyc-cpus.html)
- [Speculative LLM Inference on 5th Gen AMD EPYC with PARD and PACE](https://www.amd.com/en/developer/resources/technical-articles/2025/speculative-llm-inference-on-the-5th-gen-amd-epyc-processors-wit.html)
- [Llama 2 Inference from Intel with DeepSpeed](https://www.intel.com/content/www/us/en/developer/articles/technical/llama-2-on-xeon-scalable-processor-with-deepspeed.html)

## 四、当前最大的论文空白

1. 直接把 AMD EPYC / vLLM CPU backend 与 chiplet/CCD/CCX/L3 slice 放置策略绑定的论文。
2. 直接把 shared KV prefix 或 kv_head 工作集映射到 Ave L3 Miss Latency、same-node another CCX % 等 CPU 计数器的论文。

这意味着不能声称"文献已证明该策略有效"，只能表述为"文献提供方法论，本地实验提供可验证靶点"。

另有一篇 IEEE 论文仅能通过机构订阅获取，本地未下载：
- **On Design Space Exploration of Cache System in Multi-Chiplet Systems** (DAC 2025)：chiplet cache 层级 + 互联双级优化，相比 Zen 4 减少 39.7% 执行时间。

## 五、研究方向建议（ACC 被排除后）

ACC-local-L3 已被证明不可行——它改善了 locality 指标但无法转化为时延收益，核心矛盾在调度并行效率。后续可考虑的方向：

1. **prefill/decode 相位感知策略**：不再追求统一策略，按 phase 和长度选择不同 locality mode（参考 Sandwich、CHARM）。
2. **NPS/TP 与 locality 协同**：把 rank 级分散与 rank 内局部化分开控制，判断两者是否正交。
3. **非 attention 路径**：既然 attention 的 L3 局部化未带来收益，可重新审视 P0 归因结论——decode 高 L3 miss 的主路径可能不在 attention。

## 六、暂不建议投入

- 语义级 KV 共享（KVShare）：更偏服务侧复用，不是当前 L3/CCX 主矛盾。
- KV 在 CPU/GPU 间 offload（HeadInfer）：适合异构显存受限场景。
- 近似 attention/LSH 路线（MagicPIG）：问题设定不同，不适合回答精确 attention 的拓扑局部性问题。
