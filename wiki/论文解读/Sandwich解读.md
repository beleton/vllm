# Sandwich 解读

## 说明

- 依据版本：本地 PDF `wiki/原始资料/papers/Sandwich.pdf`。
- 论文标题：`Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving`。
- 以下内容只基于论文 PDF，不使用外部资料。

## 问题

Sandwich 研究 CPU 上的 LLM serving。论文认为现有 CPU serving 系统通常沿用 GPU 执行计划，或使用静态 per-NUMA model partition，忽略 prefill 与 decode 两个阶段的 workload 差异。Sandwich 的目标是在现有 CPU/NUMA 系统上分别优化 prefill 和 decode 的执行计划、模型分区、核心使用方案与算子实现（Sec. 1, Sec. 2）。

该论文属于现有硬件上的软件优化工作，不要求修改 cache coherence protocol、ISA、directory 或芯片互联。

## 核心内容

- **应用场景**：CPU LLM serving，包括单请求 sequential serving 和 continuous batching/batched serving（Sec. 8.1）。
- **核心思路**：prefill 和 decode 使用不同 service configuration。service configuration 指模型分区计划和 core utilization plan 的组合，包括 process 数量、core 数量、core 位置、NUMA 关联关系（Sec. 2, Sec. 4）。
- **prefill 优化对象**：动态 shape GEMM tensor program。Sandwich 生成 micro-kernel、computation slice 和 thread polymerization scheme，并用 fast-start + finetune 缩小 tuning 成本（Sec. 3.2, Sec. 6）。
- **decode 优化对象**：memory-bound 阶段的共享资源竞争。Sandwich 用 TopoTree 枚举 active core 的数量和位置，并用 remove transformation 降低内存控制器、LLC tag 等共享结构竞争（Sec. 3.3, Sec. 5）。
- **运行方式**：service configuration generation 和 kernel orchestration 均离线执行；运行时在 prefill 和 decode 阶段切换到各自的 service configuration 与 tensor programs（Fig. 4）。

## Prefill 与 Decode 差异

论文用 BF16 Llama3.2-1B 和 Llama3.2-3B，在双路 Intel Xeon Platinum 8275CL 上运行 vLLM，并按推荐把模型均匀分到两个 NUMA node。请求来自 ShareGPT，batch size 为 1、4、16（Sec. 3.1）。

prefill 阶段处理完整 prompt，生成首 token 和初始 KV cache。它计算密集，需要适配输入序列长度变化的 dynamic-shape tensor program。decode 阶段每次生成一个 token，频繁读取模型参数和 KV cache，memory-bound 特征更强（Sec. 2）。

Fig. 1 显示 prefill 的 IPC 更高、BUSPKI 更低；decode 的 IPC 更低、BUSPKI 和 memory-controller 相关压力更高。论文据此认为 prefill 需要更多计算能力，decode 需要降低 NUMA 系统内 memory controller 和共享资源拥塞（Sec. 3.1, Fig. 1）。

论文没有把 decode 的瓶颈归因于单一的跨 NUMA 远端访问。它同时讨论 cache contention、thread synchronization、remote memory access、memory bus contention 和共享结构竞争（Sec. 1, Sec. 3.1, Sec. 3.3）。

## TopoTree 与 Service Configuration

TopoTree 是 Sandwich 的硬件拓扑抽象。叶子节点是 processing unit，包括物理 CPU core 和虚拟 CPU core；非叶子节点表示后代节点共享的硬件资源，例如 L3 cache、NUMA 资源或潜在的共享结构（Sec. 5.1, Fig. 5）。

Sandwich 先用 `lstopo` 等系统工具解析可见拓扑，构建 fundamental TopoTree。系统工具不一定暴露全部共享结构。论文以 Kunpeng920 为例，`lscpu` 只能看到 24 cores 属于同一 NUMA/L3 cache，不能暴露每 4 cores 共享一个 L3-Tag 的 CCL 结构（Sec. 5.2, Fig. 5）。

TopoTree 上有两个 transformation：

- `group(n, t, d)`：在指定深度插入新层，把节点按大小 `n` 和 stride `t` 分组，用于枚举潜在的中间共享结构（Sec. 5.2）。
- `remove(n, d)`：在指定层级从每个父节点移除 `n` 个 right-most children，用于减少某一级共享资源的竞争（Sec. 5.2）。

Sandwich 先用 group 枚举潜在共享结构，再对这些树应用 remove。每个 remove 产生的 TopoTree 对应一组新的 service configuration。group 变换的顺序不影响结果；remove 的搜索空间通过 hash 去重和 transformation tree 剪枝缩小（Sec. 5.2）。

`sandwich-config` 算法把 TopoTree 转成候选 service configuration。算法按层横切 TopoTree，横切到的节点决定 process 数量和 memory hierarchy 关联，节点下方子树决定 core locality 与 core utilization。论文给出的例子中，144 cores 可解释为 8 个 process、每个 process 16 cores，也可解释为 48 个 process、每个 process 3 cores（Sec. 5.3, Fig. 6）。

Sandwich 在选择 service configuration 时先用默认 tensor schedule 做 latency simulation，从所有 TopoTree 中选 top-k service configurations，再对这些配置做后续 kernel orchestration。实验中使用 `k = 10`（Sec. 5.3）。

## Group 与 Remove 的含义

`group` 的作用不是减少 core，而是补全系统工具未暴露的层级。它把“可能共享某种资源的一组 core”显式建模为 TopoTree 中间节点。论文用 Kunpeng 的 4-core CCL/L3-Tag 作为例子（Sec. 5.2, Fig. 5）。

`remove` 的作用是减少某些共享结构下的 active core。论文的动机是 decode 阶段可能被共享资源竞争限制，减少 active core 反而可能降低 memory bus 或 LLC tag 等结构的压力（Sec. 3.3, Sec. 5.2）。

group 服务于“找出可能的拓扑粒度”，remove 服务于“在某个拓扑粒度上减少竞争”。二者共同生成 core utilization 与 model partition 的搜索空间（Sec. 5.2, Fig. 4, Fig. 5）。

## Decode Memory Contention

论文在 Kunpeng-920 上做 active core 位置实验。平台为 4 颗 Kunpeng-920 CPU、总计 192 cores、8 个 NUMA nodes，TP size 为 8。实验故意停用 6 个 cores，比较按每 4 cores 的 LLC Tag cluster 停用、按 NUMA node 末尾 core 停用、只使用 fast NUMA、只使用 slow NUMA 等方案（Sec. 3.3, Fig. 3(a)）。

Fig. 3(a) 显示使用所有 NUMA nodes 的 E2E latency 为 1053.0 s；SC-RM 为 196.63 s；NUMA-RM 为 197.75 s；Fast NUMA 为 283.6 s；Slow NUMA 为 300.62 s。论文据此说明 active core 的数量和物理位置会影响 decode 性能，减少 core 可以缓解共享结构竞争（Sec. 3.3, Fig. 3(a)）。

论文还在 AMD EPYC 系统上比较不同 TP degree。Fig. 3(b) 给出 TP-2、TP-4、TP-8、TP-16 的 E2E latency 分别为 556.03 s、418.35 s、383.12 s、639.05 s，TP-8 最低。正文解释为：在 2-NUMA EPYC 系统中，每 CPU 64 cores，总计 16 个 LLC，每 4 cores 共享一个 LLC；TP-8 可减少每个 NUMA node 下跨多个 LLC group 的数据访问，从而缓解 cache coherence 压力（Sec. 3.3, Fig. 3(b)）。

PDF 在该处存在型号标注不一致：Fig. 3(b) caption 写 `AMD EPYC 7H21`，正文写 `EPYC 7H11`，实验设置中写 `EPYC 7H12`。解读时只保留论文报告的拓扑关系和结果，不把具体型号作为独立结论。

## Model Partition 与 Core Utilization

论文把 model partition 和 core utilization 合并为 service configuration。model partition 决定模型切成多少 process 或 TP degree；core utilization 决定每个 process 使用哪些 cores、这些 cores 位于哪些 NUMA 或共享层级下（Sec. 2, Sec. 5.3）。

现有 CPU serving 方案通常按 NUMA group 静态分配进程和 cores，并在整个推理过程中保持不变。Sandwich 的差异是：prefill 使用 all-core 方案和计算优化 kernels；decode 使用针对 memory-bound 操作搜索出的 core utilization 与 model partition（Sec. 1, Sec. 4）。

这个方法的关键不是“固定减少 core”，而是离线枚举 TopoTree 后用 trace latency simulation 选配置。论文的 Table 2 显示 top-k 越大，tuning time 增加，throughput 略升：`k=5/10/15/20` 时 tuning time 为 4716.86/9609.13/13443.91/16497.87 s，throughput 为 15.46/15.58/16.19/16.42 token/s（Table 2）。

## Prefill GEMM 编译优化

prefill 的主要问题是输入序列长度变化导致 dynamic-shape GEMM。Sandwich 生成 micro-kernel（MK）、computation slice 和 polymerization scheme。MK 的大小为 `mu_M x mu_N`，沿 reduction dimension 加载、乘法和累加数据。Sandwich 枚举不超过 32 个 vector registers 的 MK 候选，并让 `b_K` 与 cache line size 对齐（Sec. 6）。

论文指出 computation slice 和 polymerization scheme 必须联合优化。较大的 computation slice 可改善低层 cache 数据局部性，但会限制可并行化的维度，导致 CPU cores 使用不足或错过更高层 cache locality 的 polymerization 方案（Sec. 3.2, Fig. 2(a)）。

Fig. 2(a) 的 GEMM 例子中，`(b_M, b_N, b_K)=(64,96,576)` 单线程性能最好，但只能扩展到 24 threads，无法同时利用 32 cores。`(32,144,1152)` 明显优于 `(16,288,1152)`，但 cost-model-based 方法会把二者视为等价。论文据此说明只优化单线程 cache 利用或只用固定 MK 的 cost model 都会漏掉更优组合（Sec. 3.2, Fig. 2(a)）。

Sandwich 的 kernel search 分两阶段：

- **Fast Start**：按 `2^{t_D} x tilesize` 的指数增长步长扩展 computation slice。若某维度扩展不带来性能提升或降低性能，则回滚到上一步；若继续扩展会限制并行化，也停止扩展该维度（Sec. 6）。
- **Finetune**：枚举使用全部可用 CPU cores 的 thread polymerization schemes，在每个方案下沿最有效的维度继续增长 computation slice，最后比较所有 MK、slice 和 polymerization 的组合（Sec. 6）。

Sandwich 还利用 prefill shape 的相似性降低 tuning 成本。sequence length 小时 shape 偏 skinny，需要保留更多 MK 候选；sequence length 增大后，最优 MK 和 polymerization 会趋于稳定。论文使用 sliding window，实验中 `sigma = 16`；当新最优 slice 与前序平均差异低于阈值时，后续更大 shape 复用该 tensor schedule（Sec. 6）。

## 通信实现

Sandwich 使用 shared memory 做进程间通信。master process 获取 file descriptor 并分发给 workers，workers 缓存并映射共享区域。all-reduce 时 workers 在 shared memory 中累加数据，再把结果复制回本地内存（Sec. 7.2）。

论文提出 rank-shifted adaptive SHM communication：每个 process 使用不同 rank-based shift，形成环形写入模式，减少等待；shared memory block size 动态调整，并保证超过 cache line size 以避免 false sharing（Sec. 7.2）。

Table 1 的 ablation 显示通信优化贡献较大。Llama3.2-3B 在 Xeon 8272CL 上，BS=1 时 vLLM 为 4.09 token/s，加入 communication op 后为 13.46 token/s；继续加入 service config、kernel tuning、split-k 后分别为 14.90、17.54、17.09 token/s。BS=8 时对应为 2.35、3.66、5.40、8.16、8.78 token/s（Table 1）。

## 实验设置

### 平台

论文覆盖 5 类 CPU 平台（Sec. 8.1）：

- 2 x Intel Xeon Gold 6151，每 CPU 18 physical cores / 36 virtual cores，2 NUMA nodes，AVX-512。
- 2 x Intel Xeon Gold 6230，每 CPU 20 physical cores / 40 virtual cores，2 NUMA nodes，AVX-512。
- 2 x Intel Xeon Platinum 8272CL，每 CPU 24 physical cores / 48 virtual cores，2 NUMA nodes。
- 2 x AMD EPYC 7H12，每 CPU 64 physical cores / 128 virtual cores，2 NUMA nodes，AVX-2。
- 4 x Kunpeng 920，总计 8 NUMA nodes、192 physical cores，NEON。

### 精度

AVX-512 CPU 上使用 BF16。AVX-2 和 Kunpeng 平台使用 FP32。由于 TVM 不支持 CPU BF16 tuning，kernel 对比实验使用 FP32（Sec. 8.1）。

### 模型与 workload

模型来自 Llama 系列。dynamic-shape benchmark 通过 payload generator 根据最大序列长度和模型配置生成所有可能 operator shapes，包括 hidden size、intermediate size、KV heads 和 query heads。该实验显式限制在单 NUMA node 上运行，以避免干扰（Sec. 8.1）。

serving benchmark 包括 single sequence serving 和 batched serving。single sequence serving 从 ShareGPT 和 LMsys-Chat-1M 采样 90 个请求并顺序输入系统。batched serving 的请求到达服从 Poisson distribution，并改变 request rate（Sec. 8.1）。

### Baseline

dynamic-shape benchmark 对比两类 baseline（Sec. 8.1）：

- vendor/DNN library：x86 上的 MKL-DNN 和 OpenVINO，ARM 上的 Bolt 和 Torch with XNNPACK。
- compiler：TVM Ansor、DietCode、Roller、MikPoly。论文将原本面向 GPU 的动态 shape scheduling policy 移植到 CPU，并提供与 Sandwich 相同的 MK candidates。

serving benchmark 对比（Sec. 8.1）：

- `ipex`：intel-extension-for-pytorch。
- `ov`：OpenVINO。
- `vLLM`：默认 CPU backend，无 model partition。
- `vLLM++`：枚举 TP strategies，NUMA node 和 cores 均匀分区，最小 TP size 设为 2。
- `llama.cpp`。

论文未与 xFasterTransformer 做主实验对比，理由是其主要面向至少 AVX512-bf16 的高级指令平台，且对论文 CPU 平台支持较差（Sec. 8.1）。附录给出 Xeon Gold 6230 BF16 下的额外比较（Fig. 14）。

## 结果与解释

### Dynamic-Shape GEMM

相对 vendor baseline，Sandwich kernel 在 Llama-1.3B 的 GEMM shapes 上取得 1.27-4.02x speedup，并在多数 cases 中优于 OpenVINO 和 Bolt。论文指出 Sandwich 对较小 M-size 的 kernels 更有效；当 token size 大于 100，shape 不再那么 skinny，vendor solution 的表现变好，Sandwich 的 speedup 下降（Sec. 8.2, Fig. 7）。

相对 compiler baseline，Sandwich 的 kernel performance 与 TVM 相当或更好，tuning time 低于 TVM；DietCode 和 MikPoly tuning 很快，但 kernel performance 低于 Sandwich。论文将 Sandwich 的优势归因于联合优化 computation slice 和 concurrent polymerization，而不是只优化单线程 slice 或只用 cost model（Sec. 8.2, Fig. 8）。

### Single Sequence Serving

论文在 1.3B 和 8B 两种模型、ShareGPT 和 LMsys-Chat-1M 两个数据集、四类平台上评估 single sequence serving。SLO 由论文经验设定：1.3B 模型 TTFT SLO 为 2200 ms，TPOT SLO 为 70 ms；8B 模型 TTFT SLO 为 8000 ms，TPOT SLO 为 240 ms（Sec. 8.3）。

Fig. 9 显示 Sandwich 可在更严格的 SLO scale 下保持 90% SLO attainment。Sec. 8.3 文本报告：相对 OpenVINO 和 vLLM，Sandwich 可支持最高 3.40x 和 4.45x 更严格的 SLO。Fig. 10 显示 Sandwich 的 TTFT 与最好的 vendor/open-source solutions 接近；Fig. 11 显示 Sandwich 的 token generation throughput 平均提升 2.01x（Sec. 8.3, Fig. 9, Fig. 10, Fig. 11）。

### Batched Serving

batched serving 使用 160M 和 1.3B Llama models。由于 vLLM 仅支持 BF16 和 AVX-512 inference，实验在 Xeon 6151 和 Xeon 6230 上进行。SLO 为：160M 模型 P90 TTFT 不超过 200 ms、P90 TPOT 不超过 100 ms；1.3B 模型 P90 TTFT 不超过 2000 ms、P90 TPOT 不超过 280 ms（Sec. 8.3）。

Fig. 12 显示 Sandwich 在满足 TTFT 和 TPOT SLO 的前提下可承载更高 request rate：Llama-160M 在 Xeon Gold 6151 上为 1.84 requests/s，Llama-1.3B 在 Xeon Gold 6230 上为 0.92 requests/s。论文指出原始 vLLM 持续不能满足 TTFT 要求（Sec. 8.3, Fig. 12）。

### Tuning 成本

Sec. 8.3 报告 service configuration tuning time 范围为 78-3279 s，平均 1309 s；kernel tuning time 范围为 2022-23115 s，平均 10155 s。Table 4 列出不同 CPU、模型、dtype、dataset 下的 graph tune time 和 kernel tune time（Sec. 8.3, Table 4）。

这说明 Sandwich 的优化主要适合离线 tuning 后复用的 serving 场景，不适合频繁改变模型、平台或核心拓扑且无法接受离线搜索成本的场景。

## 分析

Sandwich 对当前课题的价值在于：它把 CPU LLM serving 的优化对象从单个 attention kernel 扩展到全链路 phase-aware execution plan。论文明确区分 prefill 和 decode，并把 active core selection、model partition、NUMA/topology-aware placement 与 kernel generation 统一为 service configuration 搜索（Sec. 2, Sec. 4, Sec. 5）。

它与“现有 chiplet/NUMA CPU 上的软件优化”方向匹配。TopoTree、group/remove、sandwich-config、latency simulation、kernel tuning 都是软件侧机制。论文没有提出需要新硬件支持的设计。

论文的 decode 结论对当前平台有启发：decode 不一定适合使用所有 cores；active core 的数量和位置会影响 memory-bound 阶段的共享资源竞争。但论文是在 Kunpeng、Xeon 和 EPYC 7H12/7H11/7H21 标注混合的实验上下文中得到结果，不能直接推出 AMD EPYC 9745 上相同的最优 core 数或最优 CCD 粒度（Sec. 3.3, Fig. 3）。

论文的 prefill 结论对当前平台有启发：prefill 的动态 shape GEMM 可能比 attention-only L3 局部化更适合作为优化对象。论文证明的是 computation slice 与 polymerization 需要联合搜索，不是证明 `kv_head` 静态绑定到 L3/CCD 能提升当前 vLLM attention 性能（Sec. 3.2, Sec. 6）。

## 边界

- 论文的核心应用仍是 LLM serving，不是新的非 LLM 应用场景。
- 论文没有以 chiplet 为主术语，主要建模对象是 CPU/NUMA、多级 cache、潜在共享结构和 active core。
- 论文没有覆盖当前平台 `2 x AMD EPYC 9745 128-Core`，不能直接复用其最佳 TP degree 或 core removal 策略。
- TopoTree 能否发现 AMD EPYC 9745 的 CCD/L3 粒度和潜在共享瓶颈，需要本地 `lstopo`、PMU 和 end-to-end latency 实验确认。
- 论文结果包含大量离线 tuning 成本。若模型、batching 策略或请求分布频繁变化，收益需要扣除重新 tuning 成本。
- 论文证明的是全链路服务配置和 kernel 编译优化的组合收益。Table 1 显示 communication、service config、kernel tuning 都有贡献，不能把整体 speedup 单独归因于 core utilization 或 TopoTree。

## 可迁移点

- prefill 和 decode 应使用不同执行计划。prefill 优先计算能力和动态 shape kernel，decode 优先降低内存系统共享资源竞争（Sec. 3.1, Fig. 1）。
- core utilization 需要同时考虑 active core 数量和物理位置。只按 NUMA node 均匀分配不一定覆盖 LLC、L3 tag、CCD/L3 等更细层级（Sec. 3.3, Sec. 5.1, Fig. 3, Fig. 5）。
- 可把硬件拓扑转成树结构，再用 group 枚举潜在共享层级，用 remove 枚举降低竞争的 core 子集。该方法适合在现有 CPU 上实现为离线搜索器（Sec. 5.2）。
- model partition 和 core utilization 应联合搜索。固定 TP degree 或固定每 NUMA 均分可能漏掉更优配置（Sec. 5.3, Fig. 6, Table 2）。
- prefill GEMM 优化需要联合考虑 computation slice 和 polymerization。只优化单线程 cache 利用可能降低并行度，只用固定 MK 的 cost model 可能错过更优 slice（Sec. 3.2, Fig. 2）。

## 不可直接迁移点

- Sandwich 的整体系统包含自定义 kernel compiler、Dual-IR、shared memory communication 和 vLLM 集成。不能把 TopoTree 作为单独绑核策略直接等同于论文完整收益（Sec. 7, Table 1）。
- 论文没有验证当前已尝试的 attention `kv_head` 静态 L3 局部化策略。当前课题中 attention-only 路线的负结果不能由 Sandwich 直接推翻。
- Sandwich 的 EPYC 结果不等同于 EPYC 9745。当前平台每 CPU 8 个 CCD、每 16 cores 共享约 32 MiB L3，拓扑粒度与论文中的 EPYC 描述不同。
- 论文的 best configuration 依赖 trace autotune 和 sampled payload。若本地 workload 是固定 batch、attention-only benchmark 或不同模型结构，搜索目标和收益可能变化。

## 证据

- 原始资料：[../原始资料/papers/Sandwich.pdf](../原始资料/papers/Sandwich.pdf)
- 关键图表：Fig. 1、Fig. 2、Fig. 3、Fig. 4、Fig. 5、Fig. 6、Fig. 7、Fig. 8、Fig. 9、Fig. 10、Fig. 11、Fig. 12、Table 1、Table 2、Table 4。
