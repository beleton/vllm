# P-MOSS 解读

## 说明
- 依据版本：本地 PDF，标题为 `P-MOSS: Scheduling Main-Memory Indexes Over NUMA Servers Using Next Token Prediction`，页眉标注 `Accepted to SIGMOD'26`。

## 问题
P-MOSS 研究的是主内存数据库索引的空间调度。目标对象是 B+-Tree 主内存索引。论文关注两个决策：查询在哪个 logical core 执行，以及索引数据放在哪个 NUMA node 或 memory controller 上。论文把二者合称为 `spatial scheduling`（空间调度）（Sec. 1, Sec. 3）。

论文给出的动机是：同一个 B+-Tree 在不同空间调度策略、不同 NUMA/chiplet 服务器上吞吐差异可达 5.91x；同一数据放置策略下，仅 core scheduling 不同也会导致性能差异；不存在一个策略在 AMD EPYC Milan、Intel Skylake X、NVIDIA Grace Hopper 上都最优（Fig. 1, Sec. 1）。

## 应用场景
P-MOSS 的应用是 **main-memory B+-Tree index query processing**，主要面向数据库或 key-value 类系统中的索引查询。论文实验使用基于 optimistic lock coupling 的 B+-Tree 实现，B+-Tree leaf node 最多容纳 255 个 entry（Sec. 6）。

Workload 来自 YCSB，包括读写、点查、scan、scan/insert 混合等组合；主实验使用 1000M records，每条记录包含 64-bit key 和 64-bit value，读写 workload 为 50% read + 50% write，Zipfian constant 为 0.99（Sec. 7.2）。论文还用 OSM 数据集测试未见过的数据分布，规模为 800M records（Sec. 7.4）。

## 核心内容
P-MOSS 是软件调度框架。它不提出新硬件，不修改 cache coherence 协议，不要求 ISA 支持。它依赖现有服务器上的 PMU、Linux Perf Event API、Intel PCM、Linux `migrate_pages` / `move_pages` 等机制完成观测、推理和页面迁移（Sec. 5.4, Sec. 6, Sec. 7.7）。

P-MOSS 的基本策略：
- 将 B+-Tree 按 key range 逻辑切分为多个 `index slice`。每个 slice 对应一个 key range，是调度单位（Sec. 1, Sec. 6）。
- Decision Transformer 为每个 index slice 预测一个 worker core。slice 中的 index nodes 被放到该 core 所属本地 NUMA node 的 DIMM 上（Fig. 4, Sec. 5.2）。
- Router core 根据查询 predicate 判断访问哪个 index slice，再把查询发给该 slice 对应的 worker core（Fig. 7, Sec. 6）。
- 当策略更新时，P-MOSS 迁移对应 slice 的 index nodes，并更新 router core 保存的映射（Sec. 6）。

## Spatial Scheduling
Spatial scheduling 包含两部分（Sec. 3）：
- **core scheduling**：决定查询由哪个 core 执行。
- **data partitioning / placement**：决定查询访问或生成的数据放在哪个 NUMA node。

论文的目标不是改变 B+-Tree 算法，而是为每个 index slice 选择 core 和 NUMA node，使吞吐最大化。它假设 core 和数据放置存在亲和性：某个 slice 被调度到某个 core 后，该 slice 的 index nodes 放在该 core 本地 NUMA node 连接的 DIMM 上（Fig. 4, Sec. 5.2）。

## PMU Token 与 Decision Transformer
P-MOSS 使用硬件 PMU 统计作为学习特征，避免依赖 DBMS 内部 bookkeeping。论文理由是 PMU 能从硬件层面反映 query workload 和数据分布，例如热点 key range 会产生更多 cache 和 memory accesses，range scan 通常比 lookup 产生更多 cache 和 memory accesses（Sec. 3, Sec. 5.4）。

P-MOSS 最多采集 39 个硬件计数器，其中包括 15 个 core counters 和 24 个 off-core counters；off-core counters 的描述限定为 Intel servers only（Sec. 5.4）。计数器覆盖四类硬件组件（Sec. 5.4）：
- front-end：instructions executed、L1-I misses、branch mispredictions per 1k instructions。
- cache subsystem：L1D、LLC、DTLB、LLC-write misses per 1k instructions。
- memory subsystem：memory accesses、local/remote DRAM serviced retired load instructions、memory-write misses per 1k instructions。
- network interconnects：每个 NUMA node 的 memory channel bandwidth、off-chip interconnect incoming/outgoing traffic。

Token 组成（Fig. 5, Sec. 5.4）：
- **State token**：由 View token、Position token、Machine token 融合得到。View token 表示哪些 core 已被占用；Position token 表示哪些 core 对下一个 slice 可选；Machine token 聚合 front-end、cache、memory PMU 统计。
- **Meta token**：表示 system-wide interconnect 状态，并包含 logical cores 数、NUMA nodes 数、sockets 数、CPU vendor 等 processor-level 信息。
- **Action token**：worker core ID，即当前 slice 的调度目标。
- **RTG token**：Return-To-Go，以目标吞吐和已观察 reward 的差值表示。reward 使用 index query throughput。

P-MOSS 将调度写成 Next Token Prediction：给定前面 slice 的调度动作、硬件状态和期望吞吐，预测下一个 slice 的 core ID。Action token 是要预测的 token，State、RTG、Meta token 提供上下文（Sec. 5.1, Sec. 5.3）。

Decision Transformer 使用 GPT-1 作为 backbone。每条 scheduling policy 输入 `3T+1` 个 token，其中 `T` 是 index slice 数；P-MOSS 中 context length 设为 256，等于 index slice 数（Sec. 5.3, Sec. 7.1）。

## Pre-training / Inference / Post-training

### Pre-training
Pre-training 完全离线执行。P-MOSS 建立 hardware pool、workload pool 和 policy pool，从中采样服务器、workload 和调度策略，运行 workload，周期性采集 PMU，生成 Hardware Snapshot，再 token 化为 state/action/reward/meta tokens，写入 offline dataset。随后用 offline RL 以监督方式训练 Decision Transformer，生成 base model（Fig. 3, Sec. 4）。

Pre-training dataset 的硬件池包括（Sec. 5.5）：
- AMD Milan 1-NUMA-Per-Socket。
- AMD Milan 4-NUMA-Per-Socket。
- 4-Socket 8-NUMA Node Intel Skylake X。
- 4-Socket Intel Sandy Bridge。
- NVIDIA H200 Grace Hopper。

Workload 包括 50% read + 50% write、100% read、100% scan、20% read + 30% scan + 50% insert、25% read + 50% scan + 25% insert（Sec. 5.5）。Policy pool 包括 grouped、mixed、spread、random，离线数据占比分别为 14.63%、12.75%、1.76%、71.03%（Sec. 5.5）。

### Inference
Inference 是在线执行路径。若 assistant model 不可用，P-MOSS 使用 pre-training 生成的 base model；否则使用最新 assistant model。模型根据当前 Hardware Snapshot 和 RTG 逐个输出 index slice 的 core ID，形成完整 scheduling policy（Fig. 6, Sec. 5.6）。

论文主实验中，target workload 和 target server 上 inference 的初始 RTG 设置为 fine-tuning dataset 中已观察最佳吞吐的 2x。Inference 在 NVIDIA A30 GPU 上最多 1.5 分钟（Sec. 7.1）。

### Post-training
Post-training 使用 target hardware、target workload 和 learned scheduling policies 产生的 token 构造 fine-tuning dataset。它从 base model 继续训练，生成 assistant model，并随 fine-tuning dataset 增长增量更新。Post-training 也完全离线，不阻塞 DBMS 学习流程（Fig. 3, Sec. 4）。

主实验中 fine-tuning 最多 5 分钟，fine-tuning dataset 最多 1900 samples，准确率超过 90%；相比之下 pre-training 在 NVIDIA A30 Tensor Core GPU 上约 4 小时，offline dataset accuracy 达 91.2%（Sec. 7.1）。

## Runtime System
P-MOSS 在启动时创建与硬件 core 数相同的线程，并将线程绑定到 core；论文中 threads 和 cores 可互换使用（Sec. 6）。核心角色包括：
- **router core**：每个 NUMA node 至少一个。根据 query predicate 判断 key range 对应的 index slice，再按 scheduling policy 路由到 worker core。range scan 跨多个 key range 时，从多个候选 core 中随机选择以平衡负载（Fig. 7, Sec. 6）。
- **worker core**：执行 index traversal 并产生结果，同时维护 job queue 和 core stats queue（Sec. 6）。
- **core PMU sweeper**：每个 NUMA node 一个，汇总本地 worker core 的 core PMU 统计（Fig. 7, Sec. 6）。
- **off-core PMU sweeper**：采集 memory controller 和 interconnect PMU，用于生成 Meta token；实现使用 Intel PCM（Sec. 6）。
- **stitcher / enforcer**：拼接不同 sweeping round 的 snapshots；必要时执行新策略，迁移 index slice 并更新 router mapping（Sec. 6）。

Core PMU profiling 粒度设为 100，即一条硬件 trace 包含 100 个已执行查询的累计 core PMU 统计。核心 PMU 通过 Linux Perf Event API 的 C++ wrapper 实时读取（Sec. 6）。

Slice 迁移通过 migratory range scan 触达该 key range 的 index nodes，再调用 Linux `migrate_pages` 迁移到目标 NUMA node。论文报告平均迁移一个 index slice 需要 0.1 秒，单线程迁移 1000M records 的 B+-Tree 最多 25 秒（Sec. 6）。动态实验中使用 `move_pages` 建立新策略（Sec. 7.7）。

## 实验设置

### 硬件平台
主实验覆盖五类 NUMA 机器（Sec. 7.1）：
- AMD Milan 1NPS：dual socket，64 cores × 2，AMD 7543，Ubuntu 22.04，CloudLab r6525，NPS=1。
- AMD Milan 4NPS：dual socket，64 cores × 2，AMD 7543，Ubuntu 22.04，CloudLab r6525，NPS=4，即每 socket 4 个 NUMA domains。两台 AMD 机器均为 251GB DDR4 RAM。
- Intel Skylake X：4 socket，92 (×2) core machine，Ubuntu 22.04，2.95TB DDR4 RAM，Sub-NUMA Clustering enabled，每 socket 2 个 NUMA domains。
- Intel Sandy Bridge：4 socket，32 (×2) core machine，Ubuntu 22.04，CloudLab d820，128GB DDR4 RAM，Intel Xeon E5-4620，不支持 SNC/COD。
- NVIDIA H200 Grace Hopper：single socket，72 cores × 1，Ubuntu 22.04，CloudLab nvidiagh，ARM Neoverse V2，480GB DDR4 RAM。

未见环境实验额外使用 IBM Power System S822LC：dual socket，20 cores × 8 SMT，Ubuntu 20.04，CloudLab ibm8335，255GB DDR4 RAM，PowerPC RISC，每物理 core 支持 8 simultaneous threads，通过 Centaur buffer chips 连接 DRAM，buffer chips 带 16MB L4 cache（Sec. 7.4）。

### Baseline
论文区分 OS baseline、NUMA-aware baseline 和 SN:T baseline（Sec. 7.1）：
- **OS:D**：Linux 默认策略。OS 处理数据放置和 core scheduling；默认 NUMA memory policy 是 local，内存分配在发起分配请求的 core 本地 NUMA node；查询可在任意 core 执行。
- **OS:I**：OS interleave memory allocation，core scheduling 仍为默认 OS 策略。
- **SE:N**：Shared Everything-NUMA。相邻 key range 的 index slices 放在同一 NUMA node，core scheduling 由 OS 处理。
- **SN:N**：Shared Nothing-NUMA。数据放置沿用 SE:N；core scheduling 保持 slice 数据亲和性，查询只能调度到该 slice 数据所在 NUMA node 的本地 core。
- **SN:T**：Shared Nothing-Thread。数据放置和 core scheduling 都遵循 shared-nothing 策略；每个 index slice 有指定 core。P-MOSS 学到的调度也属于这一类。
- **Grouped / Spread / Mixed / Random**：SN:T 的四种启发式。Grouped 将相邻 key range 放在同一 NUMA node 并调度到邻近 core；Spread 将相邻 key range 分散到所有 NUMA nodes；Mixed 是 Grouped 和 Spread 的折中；Random 随机做数据放置和 core scheduling，形成 checkerboard pattern。

除 OS baseline 外，其余 baseline 均假设 index 已逻辑切分为 slices。所有系统启用 automatic NUMA balancing（Sec. 7.1）。

## 结果与解释

### 读写 workload
在 YCSB 50% read + 50% write、1000M records、Zipfian constant 0.99 下，P-MOSS 在五台服务器上均优于 OS:D、OS:I、SE:N、SN:N，平均 speedup 分别为 1.92x、1.90x、1.91x、3.66x（Fig. 8a, Sec. 7.2）。

对 SN:T 启发式，P-MOSS 相对 Grouped、Mixed、Spread、Random 最高分别提升 3.09%、2.48%、3.27%、3.95%。论文同时指出没有单一 SN:T 启发式在所有服务器上最优：Random 在 AMD Milan 1NPS 上更好，Mixed 在 AMD Milan 4NPS 和 Intel Skylake X 上更好，Grouped 在 Intel Sandy Bridge 上更好（Fig. 8b, Sec. 7.2）。

论文对 OS baseline 的解释是：P-MOSS 在 AMD 上减少了来自远端 socket cache/memory 的 data cache fills，也减少了同 socket 外部 off-chip cache fills；在 Intel 上减少 remote memory accesses，从而降低 cache/memory stalls；还改善 DTLB cache 行为并减少 executed instructions（Sec. 7.2）。

### Point lookup
100% point lookup 下，P-MOSS 相对 OS baselines、SE:N、SN:N 平均提升 2.57x（Fig. 9a, Sec. 7.3）。对 SN:T 启发式，P-MOSS 在所有 testbed servers 上表现最好，最高提升 14.39%；相对 Mixed 策略在五台服务器上分别提升 1.78%、1.58%、1.76%、4.99%、0.73%（Fig. 9b, Sec. 7.3）。

### Scan
100% scan workload 中，range scan selectivity 均匀分布在 `[0.001%, 0.01%]`。论文报告 P-MOSS 在 AMD 和 Intel 服务器上接近最佳 baseline，但改进幅度较小；原因是低选择性 range scan 最多可取回 10M records，容易造成 cache thrashing，core scheduling 的影响下降（Fig. 10, Sec. 7.3）。

对 SN:T baseline，P-MOSS 平均提升 4.94%。它在 AMD Milan 1NPS、AMD Milan 4NPS、Intel Skylake、NVIDIA 服务器上相对 Mixed 分别提升 3.58%、0.37%、1.27%、8.51%，但在 Intel Sandy Bridge 上回退 3.68%。NVIDIA 单 NUMA 服务器上 P-MOSS 出现小幅回退，论文解释为采集硬件统计的开销超过 learned schedule 收益（Fig. 10, Sec. 7.3）。

### Mixed workload
Mixed workload 为 25% reads + 50% scans + 25% inserts，scan selectivity 均匀分布到 0.01%。P-MOSS 相对 OS、SE:N、SN:N 平均提升 1.15x；由于该 workload 由 50% range scans 主导，论文认为 scan workload 的观察同样适用。相对 SN:T 策略，P-MOSS 平均提升 3.70%，在 AMD 和 Intel 服务器上最高提升 1.33x，但在 NVIDIA 服务器上有小幅回退（Fig. 11, Sec. 7.3）。

### 未见环境
IBM Power System S822LC 未出现在 offline dataset 中。P-MOSS 在该硬件上对 YCSB read-write 和 lookup 平均相对竞争 baseline 提升 1.46x（Fig. 12, Sec. 7.4）。

OSM 数据集同样未出现在 offline dataset 中，规模为 800M records，分布不同于 YCSB。对 read-write 和 lookup，P-MOSS 相对 OS baselines 平均分别提升 2.01x、1.94x；相对 SN:T 策略平均分别提升 2.41%、3.09%（Fig. 12, Sec. 7.4）。

在未见 AMD、Intel、NVIDIA 平台上，P-MOSS 对 Read-Write、Lookup、Scan、Mixed YCSB workload 平均相对最佳 SN:T baseline 提升 2.8%。Fig. 13 同时显示个别组合存在负值，最低可见为 -7.8%，说明泛化不是无条件收益（Fig. 13, Sec. 7.4）。

### Runtime overhead 与 learned component
PMU probing 在 Intel SKX read-write、lookup、scan、mixed workload 上平均造成 0.73% 性能下降。SIMD sweep hardware snapshot 相对 scalar processing 平均提升 P-MOSS 性能 60.37%，论文将其归因于 AVX-512 8x parallelism 降低 PMU 统计累积造成的 cache pollution（Fig. 14, Sec. 7.5）。

Decision Transformer 默认配置为 6 layers、8 attention heads、embedding size 128。消融实验显示增加 layer/head count 提高吞吐，但增加参数和训练/推理开销；embedding size 增大反而降低吞吐，因此采用上述默认配置（Fig. 15a, Sec. 7.6）。

与 Behavior Cloning 和 Conservative Q-Learning 相比，BC 和 CQL 最高准确率分别为 71.1% 和 80%，训练约 17 和 28 小时；DT 约 4 小时超过 90% 准确率。性能上，DT 相对 BC 和 CQL 最高分别提升 1.82% 和 1.52%；在 IBM 未见环境中最高差距扩大到 4.85%（Fig. 15b, Fig. 16, Sec. 7.6）。

### 动态 workload
动态实验在 Intel Skylake X 上执行 YCSB Lookup -> Scan -> Read-Write，B+-Tree 初始 500M records。P-MOSS 先用 SN:N bootstraps，异步执行 inference，生成新策略后使用 `move_pages` 迁移页面。迁移与 query execution 并发执行（Fig. 17, Sec. 7.7）。

整个 workload span 内，P-MOSS 相对 OS:D、OS:I、SE:N、SN:N 分别提升 1.72x、1.73x、1.73x、2.94x。单个 workload 上，P-MOSS 相对最佳 baseline 在 Lookup、Scan、Read-Write 上分别最高提升 1.53x、2.18x、1.71x（Fig. 17, Sec. 7.7）。

## 分析
P-MOSS 对当前“现有 chiplet/NUMA CPU 上的软件优化”课题有直接参考价值，原因是它把 chiplet/NUMA 服务器差异转化为软件可操作的三类对象：
- **数据切分对象**：index slice，即可稳定识别、可迁移、可绑定 key range 的数据分片。
- **执行位置对象**：logical core / worker core，而不是只停留在 NUMA node。
- **反馈信号**：PMU token，用硬件计数器描述当前策略对 cache、memory、interconnect 的影响。

P-MOSS 的重点不是“PMU + 大模型”本身，而是把调度问题表达成“按固定序列逐个放置 slice”的形式。该形式适合索引类 workload，因为查询 predicate 能直接映射到 key range，router 能以低成本决定目标 slice。对应用场景选择的启发是：优先寻找具有稳定数据分片、请求可路由、分片可迁移、吞吐受 core/data 放置影响的系统。

## 边界
- 论文只在 B+-Tree 主内存索引上做实验。结论不能直接外推到 LLM attention、KV cache 或 tensor kernel（Sec. 6, Sec. 7）。
- 主要指标是 query throughput，不是 tail latency 或单请求延迟（Sec. 7）。
- P-MOSS 的收益依赖可识别的 index slice 和 key range。没有稳定分片边界、请求无法按 predicate 路由、数据不可迁移的 workload 不能直接套用。
- Scan-dominated workload 收益有限，低选择性 range scan 会造成 cache thrashing；Intel Sandy Bridge scan 场景出现 3.68% 回退，NVIDIA 单 NUMA 场景也出现小幅回退（Fig. 10, Sec. 7.3）。
- 策略更新需要页面迁移。论文报告单 slice 平均 0.1 秒，1000M records B+-Tree 单线程迁移最多 25 秒；该开销不适合过高频率重排（Sec. 6）。
- Off-core PMU token 的实现与平台相关。论文明确提到 24 个 off-core counters 为 Intel servers only，并使用 Intel PCM 采集 off-core PMU（Sec. 5.4, Sec. 6）。
- 模型训练和推理需要额外资源。Pre-training 在 NVIDIA A30 上约 4 小时，inference 最多 1.5 分钟，fine-tuning 最多 5 分钟（Sec. 7.1）。

## 可迁移点
- 在现有 chiplet/NUMA CPU 上，软件可通过“数据分片 + core binding + NUMA page placement”联合优化性能，不必提出新硬件（Sec. 1, Sec. 6）。
- 调度粒度应落到可路由的数据分片。P-MOSS 的 index slice 是 key range；对应到其他应用时，需要找到同等粒度的稳定分片。
- PMU 可以作为在线/离线调度反馈，而不只是事后 profiling。P-MOSS 用 front-end、cache、memory、interconnect 统计构造 state/meta token（Sec. 5.4）。
- 对目标 workload 的 post-training 是必要组成。论文将 pre-training 用于跨硬件/跨 workload 的基础模型，将 post-training 用于 target hardware/workload 对齐（Fig. 3, Sec. 4）。
- Baseline 需要同时覆盖 OS 调度、NUMA-aware 数据放置、shared-nothing core+data 放置。只和 OS 默认策略比较不足以证明 chiplet-aware 调度有效（Sec. 7.1）。

## 不可直接迁移点
- P-MOSS 的 slice-to-core 序列预测不能直接映射到 attention head、KV block 或 vLLM workitem。attention 的数据访问和 B+-Tree key range 路由结构不同。
- P-MOSS 的推理周期是分钟级，适合 workload 变化较慢、策略可复用的数据库索引服务；不适合每次 forward pass 都重新推理调度。
- B+-Tree 的 query predicate 能天然定位 key range；LLM 推理请求通常没有类似 key range 的数据路由谓词。
- P-MOSS 通过页面迁移建立数据放置。若目标应用的数据结构生命周期短、分片频繁变化或迁移成本接近请求执行时间，该机制不成立。

## 证据
- 原始资料：`wiki/原始资料/papers/P-MOSS.pdf`
- 关键锚点：摘要；Fig. 1-17；Sec. 1；Sec. 3；Sec. 4；Sec. 5.1-5.6；Sec. 6；Sec. 7.1-7.7；Sec. 8；Sec. 9。
