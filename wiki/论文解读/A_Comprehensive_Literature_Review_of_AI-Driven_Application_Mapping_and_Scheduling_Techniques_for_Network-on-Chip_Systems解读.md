# A Comprehensive Literature Review of AI-Driven Application Mapping and Scheduling Techniques for Network-on-Chip Systems 解读

本文依据版本为 `Computer Modeling in Engineering & Sciences` 2026 年第 146 卷第 1 期论文 `A Comprehensive Literature Review of AI-Driven Application Mapping and Scheduling Techniques for Network-on-Chip Systems`，DOI 为 `10.32604/cmes.2025.074902`，原始 PDF 位于 `wiki/原始资料/papers/A Comprehensive Literature Review of AI-Driven Application Mapping and Scheduling Techniques for Network-on-Chip Systems.pdf`。

论文类型是综述，研究对象是 Network-on-Chip（NoC，片上网络）中的 application mapping（应用映射，即把任务分配到 Processing Element/PE 等片上处理节点）与 scheduling（调度，即任务与通信在时间和资源上的安排）。论文不提出新的 NoC 映射算法，也不提供统一实验平台上的原创性能对比。

论文要回答的问题是：现有 NoC 应用映射与调度方法如何分类，AI/机器学习方法在其中承担什么角色，不同方法在延迟、吞吐、能耗、容错、负载均衡和实时适应性上的取舍是什么。论文重点覆盖 2D/3D NoC、动态/静态/混合映射、机器学习映射、功耗感知映射、容错映射和负载均衡方案（摘要，Sec. 1，Sec. 4）。

## 问题

NoC 用分布式 packet-switched fabric（分组交换通信结构）替代传统共享总线或点到点互连，用于连接高核数、异构、可扩展的 MPSoC（多处理器片上系统）。论文说明，随着系统转向 massively parallel megacore designs，传统总线和点对点互连难以满足扩展性、延迟和功耗要求，因此 NoC 中的应用映射与调度成为性能、扩展性和可靠性的关键因素（Sec. 1）。

论文对 application mapping 的基本定义是：把应用划分到不同节点，并决定执行期间每个节点承担的工作。映射目标包括降低节点间通信开销、减少延迟、平衡片上计算负载、控制能耗、避免拥塞、提升容错能力（Sec. 1，Fig. 1，Fig. 4）。

NoC 映射问题的核心难点来自两类因素：

- **搜索空间大**：NoC 应用映射具有 NP-hard 特征，精确优化方法可提供较高准确性，但难以扩展到大规模 NoC（Sec. 1）。
- **运行期状态变化**：节点位置、流量负载、工作负载、温度、网络状态和故障都会影响映射效果。静态映射低开销但难以适应变化；动态和学习型映射具备适应性，但引入运行期开销、训练成本和安全约束问题（Sec. 3，Sec. 7）。

## 研究方法与资料范围

论文采用系统性文献综述方法，搜索时间范围为 2014-2025 年，只纳入英文、同行评议论文。搜索短语包括 `Network-on-Chip AND application mapping`、`NoC AND dynamic scheduling`、`Machine learning-based NoC mapping` 和 `fault-tolerant NoC techniques`。检索来源包括 ScienceDirect、Springer、IEEE Xplore、ACM Digital Library、Google Scholar、Research Rabbit 和 Connected Papers（Sec. 2.1）。

论文的筛选流程分为初检、Filter 1 和 Filter 2。Table 1 给出的数量为：

| 数据源 | 初始扫描 | Filter 1 后 | Filter 2 后 | 最终选择 |
| --- | ---: | ---: | ---: | ---: |
| IEEE Xplore | 243 | 123 | 94 | 70 |
| Google Scholar | 114 | 93 | 67 | 63 |
| ACM Digital Library | 160 | 87 | 39 | 8 |
| ScienceDirect | 89 | 34 | 26 | 5 |
| Springer | 37 | 17 | 12 | 6 |
| 合计 | 643 | 354 | 238 | 152 |

论文说明 grey literature（技术报告、学位论文等）被短暂检查但排除，原因是同行评议标准和方法细节不一致（Sec. 2.2，Table 1）。

论文明确排除 routing algorithm design（路由算法设计）、router microarchitecture（路由器微架构）、低层电路优化、buffer 或 link-level 优化。其范围限定在 NoC 系统的应用映射与调度方法，包括任务分配、通信优化和跨拓扑的性能评估（Sec. 2.3）。

## NoC 与应用映射背景

### NoC 的架构作用

论文把 NoC 描述为片上系统中的结构化、模块化、packet-switched 通信基础设施。相比共享总线，NoC 的优势包括更好的可扩展性、模块化、功耗效率、测试性、故障隔离能力，以及对多电压/多频率域的支持（Sec. 1，Fig. 1）。

NoC 的拓扑可包含 2D mesh、3D mesh、ring、hybrid wireless NoC、optical NoC、3D hybrid optical-electrical NoC 等。论文用 Fig. 4 展示 NoC 拓扑，并在后文把拓扑差异作为映射复杂度来源之一（Fig. 4，Sec. 4.2）。

### 3D NoC 的额外约束

论文特别关注 3D NoC。3D 堆叠系统带来更高功率密度和热热点风险，映射策略需要考虑 vertical traffic concentration（垂直通信集中）、thermal gradients（层间热梯度）、TSV/ILV 结构和 dense stacks 中的 fault clustering（Sec. 1，Sec. 6）。

在 3D NoC 中，映射不仅决定通信路径长度，还会影响温度分布、链路拥塞、垂直互连利用率和故障恢复空间。论文把 thermal-aware mapping、traffic-aware mapping、fault-tolerant mapping 和 runtime task migration 都列为 3D NoC 的重要方向（Sec. 1，Sec. 6）。

## 映射挑战

论文在 Sec. 3 中把应用映射挑战归结为多目标、多状态、多约束的组合问题。映射器需要同时考虑节点状态、节点位置、流量、工作负载、温度、网络状态和故障。Fig. 7 将主要挑战概括为动态工作负载适应与能效保持（Sec. 3，Fig. 7）。

具体挑战包括：

- **规模扩展**：processing element 和通信链路数量增加后，通信开销、资源共享和搜索空间同步增加（Sec. 3）。
- **实时性**：ERT-EAM 等实时嵌入式映射方法需要同时满足延迟、面积和通信能耗目标，但动态运行期决策会带来额外复杂度（Sec. 3，Sec. 4.2.1）。
- **热与可靠性**：重负载 PE 可能形成热热点，温度传感器可用于预测潜在失效并触发自适应调整，但热状态与映射选择本身相互影响（Sec. 3，Sec. 6）。
- **学习模型可靠性**：FANC、RL、Deep RL 等方法可优化通信路径或资源配置，但错误预测可能导致性能损失，训练和部署开销仍是障碍（Sec. 3，Sec. 7）。
- **拓扑差异**：2D/3D mesh、ring、hybrid wireless NoC、optical NoC 等拓扑对延迟、吞吐、功耗和容错约束不同，单一映射策略难以通用（Sec. 3，Sec. 4.2）。

## 分类框架

### 分类依据

论文在 Sec. 4.1 给出四类主分类：dynamic mapping、static mapping、hybrid mapping 和 machine learning-based mapping。分类依据是映射决策发生的时间、是否组合多种技术、是否使用学习方法。论文也承认分类存在重叠，例如 reinforcement learning（RL，强化学习）本身是机器学习方法，但通常用于动态运行期映射（Sec. 4.1，Fig. 9）。

引言和结论中还使用了更宽的主题分类：machine learning-based、power-aware、dynamic、fault-tolerant、static 和 load-balancing mapping。正文实际写法是以四类方法为骨架，把能耗、容错、热管理、拥塞控制和负载均衡作为目标维度嵌入各类方法中（摘要，Sec. 1，Sec. 4.1，Sec. 8）。

### Dynamic Mapping

Dynamic mapping（动态映射）在运行期根据系统状态调整任务到资源的分配。论文列出的子方向包括 energy-aware mapping、congestion-aware mapping、fault-aware mapping，以及 latency/execution-time-based mapping（Sec. 4.2.1）。

代表性方法包括：

- **SaHNoC**：self-adaptive hybrid NoC，用 wired/wireless switch 的组合在 2D mesh 中权衡能效和通信速度（Sec. 4.2.1，Table 2）。
- **ERT-EAM**：面向实时嵌入式应用，在 Xilinx Zynq UltraScale+ MPSoC ZCU104 上使用 Minimum Core Average Distance（CAD）原则，目标是改善 latency/delay、throughput 和通信能效（Sec. 4.2.1，Table 2）。
- **RL + MOPSO**：将 reinforcement learning 与 multi-objective particle swarm optimization 结合，在异构 NoC 中处理永久 PE failure，并优化可靠性和能耗（Sec. 4.2.1，Table 2）。
- **RLARA / RL-FTR / Q-function adaptive routing**：使用 Q-learning、拥塞预测、fault-aware routing 或 topology awareness 改善故障场景下的延迟、delivery rate、吞吐和热平衡（Table 2）。

动态映射的优势是能根据拥塞、故障、温度和负载变化调整映射；代价是训练、推理、状态采集和运行期控制开销（Sec. 4.3，Sec. 7，Table 3）。

### Static Mapping

Static mapping（静态映射）在部署前决定任务与资源的映射。论文认为它的优势是可预测、运行期开销低，适合固定 workload；缺点是难以应对运行期故障、温度变化和负载波动（Sec. 4.2.2，Sec. 4.3）。

代表性方法包括：

- **Ring NoC / 2D 和 3D Mesh NoC 设计**：在 Virtex-5 FPGA、Xilinx ISE 和 Modelsim 等环境中评估 ring、2D mesh、3D mesh 的延迟、吞吐、面积和功耗（Sec. 4.2.2，Table 2）。
- **GAPSO**：将 Particle Swarm Optimization（PSO，粒子群优化）与 Genetic Algorithm（GA，遗传算法）结合，用 Core Graph 优化通信效率，论文表格称其表现优于单独 PSO（Sec. 4.2.2，Table 2）。
- **Double PSO / ABC**：用于 3D NoC 或 3D-NoC 映射，目标包括降低能耗、改善散热、减少延迟或接近 ILP 结果（Sec. 4.2.2，Table 2）。
- **MILP / SA / GA**：针对 2D mesh NoC 和 energy-aware mapping，将线性规划、模拟退火、遗传算法和 3-stage heuristic 用于通信能耗、实时约束、DVFS 和 contention 控制（Sec. 4.2.2，Table 2）。
- **FTTM / FTMAP**：面向 fault-tolerant mapping，处理 offline/runtime job、spare core、failed PE 和通信能耗更新（Sec. 4.2.2，Table 2）。

静态方法的核心边界是：当 workload、故障和热状态在运行期显著变化时，预先生成的映射难以持续保持最优（Sec. 4.3，Sec. 6）。

### Hybrid Mapping

Hybrid mapping（混合映射）组合静态映射、动态重映射、精确优化、启发式搜索或多种元启发式算法。论文把 hybrid 方法视为折中路径：离线阶段生成较强的 backbone mapping，运行期只在热点、拥塞、故障或输入场景变化时做局部修正（Sec. 4.2.4，Sec. 6）。

代表性方法包括：

- **Scenario-based DVFS-aware HAM**：把输入空间分成 scenario，在设计期映射基础上结合 DVFS（动态电压频率调整）做低开销适配。论文说明该类方法在 deadline miss 和能耗上优于仅重映射或不感知 DVFS 的方法（Sec. 4.2.4，Table 2）。
- **HyDra**：Hybrid Task Mapping，将 design-time mapping 与 runtime remapping 结合。Table 2 报告其相对基线降低 14% communication cost、15% energy、19% latency（Table 2）。
- **IWOA-IGA**：Improved Whale Optimization Algorithm 与 GA 结合，目标是降低能耗、功耗和延迟，并改善大 task graph 收敛（Table 2）。
- **GA + ACO / Hyb ACOA-3D-NoC / DPSO**：组合遗传算法、蚁群优化、蜂鸟/Coati 优化、双粒子群等方法，目标多集中在通信成本、功耗、延迟和 3D NoC 能耗（Table 2）。

混合方法的主要问题是集成复杂度、参数调优和设计期映射数量增加。论文强调，部分研究显示 hybrid 方法优于纯静态或纯动态方法，但也有研究指出在特定 workstation 或 workload 下 hybrid overhead 会抵消收益（Sec. 4.3）。

### Machine Learning-Based Mapping

Machine learning-based mapping（基于机器学习的映射）使用 RL、Deep RL、神经网络、Transformer、SNN、LSM 等模型学习映射、路由、仲裁、热点预测或资源控制策略。论文把该类方法视为 NoC 映射的近期增长方向（Sec. 1，Sec. 4.2.5）。

代表性方法包括：

- **RL-MAP**：使用 Actor-Critic 网络与 local search，在 Xilinx Zynq UltraScale+ MPSoC ZCU104 上改善 NoC performance、latency 和 throughput（Sec. 4.2.5，Table 2）。
- **Deep reinforcement learning**：动态调整 voltage/frequency level。Table 2 报告 energy-delay product 改善 30%-40%（Table 2）。
- **FANC**：Fault-Tolerant Multi-Application Mapping，使用 ML-based model 简化设计探索搜索过程。Table 2 报告 communication cost reduction 为 266%（Table 2）。该数值来自论文表格，本文不对原始研究的计算口径做外推。
- **低复杂度 ML mapping model**：针对 non-symmetric NoC mapping dataset，Table 2 报告至少 99.6% accuracy、平均 96.3% mapping accuracy，但同时指出错误预测可能导致性能损失（Table 2）。
- **DREAM NoC**：将 ant colony optimization 与 RL 用于 routing。Table 2 报告 latency 降低 63%、throughput 增加 25%，但评估场景限于 uniform random traffic 下的不规则拓扑（Table 2）。
- **HAS-RL / NCTPAM / CSAP / HP-LSM**：分别面向近似计算 accelerator 的质量损失与通信延迟权衡、3D NoC 应用映射与 TSV placement、DNN accelerator 中通信同步感知仲裁、基于 liquid state machine 的热点预测（Table 2，Sec. 4.2.5）。

论文对 ML 方法的核心判断是：学习型映射具备处理高维状态和长程依赖的潜力，但训练成本、数据效率、部署安全、约束满足、推理延迟和可复现实验协议仍是开放问题（Sec. 6，Sec. 7）。

## 代表性趋势与数量结果

Sec. 4.3 声称对 selected 75 studies 做趋势分析。该节给出的类别比例为（Table 3）：

| 方法类型 | 论文占比 | 常见优化目标 | 典型取舍 |
| --- | ---: | --- | --- |
| Machine learning | 28% | energy、latency、fault tolerance | 高复杂度、训练开销 |
| Heuristic/Metaheuristic | 24% | energy、throughput | 近似最优但无保证 |
| Static mapping | 21% | predictability、low overhead | 难以适应运行期变化 |
| Hybrid mapping | 17% | adaptability、energy | 集成复杂 |
| Exact algorithms | 10% | optimality | 扩展性问题、计算开销高 |

论文还报告：energy efficiency 和 latency 是最常评估的指标，出现在超过 65% 的研究中；reliability 和 thermal management 分别约出现在 30% 和 22% 的研究中。论文认为动态和 ML-based 方法更适合故障容忍与实时重配置，但运行期复杂度更高（Sec. 4.3，Table 3）。

## 性能指标

论文在 Sec. 5 汇总 NoC 映射常用评估指标。指标定义如下：

- **Latency**：PE 间通信交换所需时间。论文给出公式 `L = Response Time - Transmission Time`（Eq. 1）。
- **Throughput**：单位时间内 PE 间传输的数据量。公式为 `T = Data Transferred / Time`（Eq. 2）。
- **Packet loss**：传输中丢失 packet 占总 packet 的比例。公式为 `PL = Number of Lost Packets / Total Number of Packets * 100`（Eq. 3）。
- **Bandwidth utilization**：通信通道和片上互连资源利用率。公式为 `BU = Data Transferred / Maximum Bandwidth * 100`（Eq. 4）。
- **Response time**：受 topology、routing、congestion 和 flow control 影响。公式为 `RT = Processing Time + Transmission Time + Queuing Time + Wait Time`（Eq. 5）。
- **Error rate**：通信请求中的错误比例。公式为 `ER = Number of Errors / Total Number of Requests * 100`（Eq. 6）。

这些指标覆盖通信性能、资源利用和可靠性，但论文没有提供统一 benchmark 上的跨方法实测对比。不同研究的 workload、模拟器、拓扑和指标口径不完全一致（Sec. 5，Sec. 8）。

## 讨论与未来方向

### 静态、动态、混合与 AI 方法的取舍

论文在 Sec. 6 将 NoC 映射方案概括为 static、dynamic、hybrid 和 AI-driven 四组。静态方法可预测、低运行期开销，但在 3D NoC 中难以响应热热点、拥塞和故障。动态方法和在线迁移能利用 queueing delay、router load、thermal sensor 等运行期遥测信息，但控制面必须足够轻量，否则会抵消性能收益（Sec. 6）。

论文提出的近期路径是 hybrid mapping：先离线生成强 backbone mapping，并加入通信和热约束；运行期只在必要时根据 telemetry 做 migration 或 local remapping。该思路用于降低全局集中式运行期决策的开销（Sec. 6）。

### Chiplet 与层次化映射

论文在讨论中把 chiplet-based fabrics 纳入未来 NoC 映射趋势，指出 inter-chiplet latency、cache/memory locality 等 non-uniformity 会使纯全局集中式运行期决策变昂贵。论文提出 hierarchical mapping（层次化映射）方向，粒度为 `cluster -> chiplet -> PE`，并引用 CHARM、MEMPLEX、REX 等 chiplet runtime 或 memory system 工作作为参考（Sec. 6，Refs. 136-138）。

该部分是论文的趋势判断，不是本文原创实验。其可直接提炼的事实是：NoC 映射研究开始把 chiplet 层级视为映射层次之一，而不是只在扁平 PE 集合上做任务分配（Sec. 6）。

### AI 方法的开放问题

论文列出的 AI/ML 映射开放问题包括：

- training cost：RL 需要大量迭代和仿真 episode，在大 NoC 上成本高（Sec. 7）。
- data efficiency：学习型方法需要足够训练数据，数据不足会影响泛化（Sec. 6，Sec. 7）。
- inference latency：实时系统中策略推理必须极低延迟，否则不能部署在 live system（Sec. 7）。
- robustness under faults：许多 RL 方法假设固定拓扑；link/core failure 出现时，若训练未覆盖，恢复能力不足（Sec. 7）。
- safety and constraints：部署时必须满足热、故障、安全和实时约束，不能只优化平均性能（Sec. 6，Sec. 7）。
- reproducible evaluation：需要标准化 benchmark、真实 task graph 和 traffic model，以便公平比较 ML 与经典方法（Sec. 6，Sec. 8）。

### 评估栈问题

论文指出，NoC 映射方法常在 simulator 或受限 benchmark 中评估，缺少 realistic traffic generator、coherent task-graph modeling 和统一评价协议。若评估栈不统一，ML-based mapping 的收益可能只成立于特定 benchmark 或仿真设置，难以转化为端到端改进（Sec. 6，Sec. 8）。

## 论文内部一致性问题

以下问题来自论文正文内部对照，不依赖外部资料：

- Table 1 给出最终选择论文数为 152，但 Sec. 4.3 写作趋势分析基于 selected 75 studies。论文未解释 152 与 75 的关系，可能是进一步子集筛选，也可能是表述不一致。
- Sec. 1 结尾写 Section 7 是 conclusion，但正文 Section 7 是 `Limitations of Current Techniques`，Section 8 才是 `Conclusion`。
- Sec. 4.3 末尾写 comparative summary provided in Table 1，但紧邻上下文实际讨论的是 Table 2/3。Table 1 是搜索来源筛选表。
- Table 2 中部分数值和表述来自被综述论文，当前综述没有重新统一验证口径。例如 FANC 的 `266% reduction in communication costs` 直接来自表格，不能在未查原论文前作为统一可比性能结论使用。

这些问题不影响论文作为文献索引和分类入口的价值，但会限制其作为定量证据来源的可靠性。

## 分析

- 论文的主要价值是分类和索引，不是证明某种映射策略在统一平台上优于其他方法。所有跨方法比较都应理解为综述层面的汇总，而不是同一 benchmark 下的实验结论。
- 论文把 NoC 映射问题放在多目标优化框架下：通信延迟、吞吐、功耗、热、故障、负载均衡和安全性同时约束映射策略。该框架可用于组织后续文献，但不能直接给出当前 CPU attention 的优化方案。
- 动态和 AI-based mapping 的适用前提是运行期状态能被足够低开销地观测，并且策略推理开销低于被优化掉的通信/拥塞/热管理成本（Sec. 6，Sec. 7）。
- Hybrid mapping 是论文最明确的工程化方向：离线生成稳定映射，运行期做局部修正。这比完全在线搜索更符合大规模 NoC 和 chiplet fabric 的控制开销约束（Sec. 6）。
- 论文在 chiplet 讨论中强调 `cluster -> chiplet -> PE` 层次，但没有展开 chiplet CPU 的 L3 分片、cache coherence、NUMA 绑核或 vLLM CPU attention 行为。该文不能作为当前 `acc-local-l3` 或 attention kernel 局部性策略有效性的证据。

## 边界

- 研究对象是 NoC/SoC/MPSoC 的应用映射和调度，不是通用服务器 CPU 上的 OpenMP kernel、vLLM 推理或 attention KV cache。
- 论文覆盖 2D/3D NoC、DNN accelerator、neuromorphic hardware、hybrid wireless NoC、optical NoC 等多种系统，硬件抽象通常是 PE、router、link、TSV、NoC topology，而不是 AMD EPYC CCD/L3/NUMA。
- 论文没有给出统一实验平台、统一 workload、统一性能计数器，也没有提供可复现代码。Table 2 中的性能提升来自不同被综述论文，不能直接横向比较。
- 论文明确排除了路由算法设计、router microarchitecture、buffer/link-level 优化。若当前研究问题落在硬件互连延迟、cache coherence、L3 fill source 或具体微架构路径上，该综述覆盖不足。
- AI/RL 方法在论文中主要作为趋势和类别讨论。论文没有证明 RL 可以自动解决任意 NoC 或 chiplet 映射问题，也没有证明学习型映射一定优于启发式映射。

## 可迁移点

- 用分层映射描述复杂系统：先按更粗的 cluster/chiplet 层级确定局部域，再在 PE/core 层级做细粒度分配。该表述可作为整理 chiplet CPU 任务放置文献的统一语言（Sec. 6）。
- 把调度策略拆成离线 backbone 和在线 correction。对于运行期全局重排代价较高的系统，只做局部迁移或局部重映射更符合论文给出的工程方向（Sec. 6）。
- 评价映射方案时至少同时记录 latency、throughput、bandwidth utilization、response time、error/error rate，以及是否存在 packet/task loss 或 fault recovery 约束（Sec. 5）。
- 对 AI/RL 映射方法，应先检查训练成本、推理延迟、状态观测成本、故障泛化能力和约束满足方式，再讨论性能收益（Sec. 6，Sec. 7）。
- 对 chiplet/NoC 系统，任务映射不应只看计算均衡，还要把 interconnect latency、cache/memory locality、thermal hotspot 和 fault cluster 作为一等约束（Sec. 6）。

## 不可直接迁移点

- NoC 的 PE/router/link/TSV 映射与当前 AMD EPYC 上的 CCD/L3/NUMA 绑核不是同一硬件抽象。不能把 NoC 映射算法直接改名为 CPU thread placement。
- 论文没有分析 L3 miss、remote cache fill、Infinity Fabric、DRAM source latency 或 cache coherence traffic，不能用于解释当前 CPU attention 的 L3 行为。
- 论文中的 DNN accelerator NoC 映射、CSAP arbitration、HP-LSM hotspot prediction 等方法面向专用加速器或片上 router，不对应 vLLM CPU attention 的 task queue 和 KV tile 读取路径。
- Table 2 的算法收益来自不同论文、不同平台、不同模拟器和不同 workload。不能把其中任一百分比作为当前平台的预期收益。
- 当前项目已验证 `acc-local-l3` 按 `kv_head` 静态本地化在长序列上因跨组补空能力下降而退化。本文只能提供“映射策略需要考虑动态状态和层次化约束”的方法论启发，不能推翻该实测结论。

## 证据

- 原始资料：`wiki/原始资料/papers/A Comprehensive Literature Review of AI-Driven Application Mapping and Scheduling Techniques for Network-on-Chip Systems.pdf`
- 关键锚点：摘要；Fig. 1-10；Table 1-3；Eq. 1-6；Sec. 1；Sec. 2.1-2.3；Sec. 3；Sec. 4.1-4.3；Sec. 5；Sec. 6；Sec. 7；Sec. 8；Refs. 136-138
