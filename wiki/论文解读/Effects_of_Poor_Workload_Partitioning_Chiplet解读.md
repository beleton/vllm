# Effects of Poor Workload Partitioning on System Performance for Chiplet-Based Systems 解读

本文依据版本为 `Electronics 2026, 15, 1139`，原始 PDF 位于 `wiki/原始资料/papers/Effects_of_Poor_Workload_Partitioning_Chiplet.pdf`。论文研究对象是多 chiplet DNN 推理加速系统中的 workload partitioning（工作负载分区，即把任务图中的计算任务映射到不同 chiplet）。核心问题是：当任务映射忽略任务间通信关系时，跨 chiplet 通信、链路拥塞和排队延迟会随 chiplet 数量非线性放大，最终使大规模 chiplet 系统从性能下降变成不可稳定运行（摘要，Sec. 1，Sec. 3.6，Sec. 4.4）。

## 问题

Chiplet 系统把功能拆到多个 die，并通过 interposer 或 D2D link 互连。论文关注的是任务图中的强通信任务是否被放在同一 chiplet 或相邻 chiplet。若随机分配任务，通信边容易跨 chiplet，导致 inter-chiplet traffic（跨 chiplet 流量）增加、NoI（Network-on-Interposer，封装内互连网络）拥塞和执行时间中的通信占比上升（Sec. 2.2，Sec. 2.5）。

论文把 partitioning 定义为“决定计算任务运行在哪个 chiplet 上”的过程。它不研究单个 kernel 的线程绑定，也不研究 cache line 在 CPU CCD/L3 slice 间的迁移。其核心实验是控制 partitioning quality（分区质量）从 0% 到 100%，观察通信延迟、跨 chiplet traffic、网络拥塞、吞吐和能效的变化（Sec. 2.5，Sec. 3.1）。

## 背景：任务图、DNN 推理与通信模式

论文使用 application task graph（应用任务图）描述 DNN 推理。任务图记为 `G = (T, E)`：`T` 是计算任务集合，`E` 是任务之间的有向通信边，每条边带有通信量 `wij`。一个 partitioning `pi` 把每个任务 `ti` 映射到 chiplet 集合 `C` 中的一个 chiplet（Sec. 3.2）。

DNN workload 被选为主要对象，原因是其通信模式较规则、数据复用明显，适合通过静态任务映射降低通信边跨 chiplet 的概率。论文选择三类模型（Sec. 3.5）：

| 模型 | 论文给出的特征 | 选择目的 |
| --- | --- | --- |
| `ResNet-50` | regular convolution，计算与通信较均衡，moderate sparsity | 常用生产 DNN baseline |
| `VGG-16` | 大 fully connected layer，memory-intensive，通信密集 | DNN 类中的带宽压力测试 |
| `DarkNet-19` | lightweight，sparse attention mechanisms，只有 12% layers 显著通信 | 低通信强度/稀疏网络代表 |

论文明确说明结论直接适用于 DNN inference、tensor computation、matrix multiplication、convolution 这类 structured communication workload；不保证适用于 sorting、graph traversal、runtime scheduling、sparse neural network 等 irregular 或 control dominated workload（Sec. 3.5.1，Sec. 3.5.3）。

## 核心内容

论文不是提出一个完整的新调度系统，而是构建一个模拟/建模框架，系统刻画“分区质量下降会怎样破坏 chiplet 系统性能”。核心变量是 `Q`，核心结果是跨 chiplet 通信随 `Q` 下降而快速恶化，并随 chiplet 数量增加进入 congestion collapse（拥塞崩溃）区域（Sec. 3.2，Sec. 3.6，Sec. 4）。

主要结论如下：

- 8-chiplet ResNet-50 场景下，`Q=0%` 的 inter-chiplet latency 为 850 ns，`Q=100%` 为 85 ns，相差 10x（Fig. 5a，Tab. A1）。
- inter-chiplet traffic 从 95% 降到 12%，相对 `Q=0%` 降低 87.4%（Fig. 5b，Tab. A1）。
- throughput 从 28 images/s 增至 245 images/s，提升 8.75x；energy efficiency 从 1.2 TOPS/W 增至 12.4 TOPS/W，提升约 10.3x（Fig. 5c，Tab. A2）。
- 16-chiplet 下，poor partitioning 的 congestion 达到 320%，optimized partitioning 为 58%；论文把 70% 以下视为可持续运行区间，100% 以上视为链路负载超过容量并引发排队、溢出或重传（Fig. 5d，Tab. A3，Sec. 3.7.3）。
- 16-chiplet 下，poor partitioning 的通信开销占执行时间 85%，optimized partitioning 为 35%（Fig. 4，Sec. 4.5）。

## 方法与模型

### Partitioning quality

论文用跨 chiplet traffic 衡量分区质量。给定 partitioning `pi`，跨 chiplet traffic 定义为所有跨 chiplet 通信边的权重之和（Sec. 3.2.1）：

```text
V(pi) = sum wij, for edges (ti, tj) where pi(ti) != pi(tj)
```

最优分区 `pi*` 是使 `V(pi)` 最小的任务映射。论文希望 `Q` 满足三个性质：`Q in [0, 1]`，`0` 表示 random/worst placement，`1` 表示 optimal placement，并且可跨 workload 和系统规模比较（Sec. 3.2.2）。

需要注意：`Table 1` 中 `Q` 的排版公式与“worst placement 时 Q=0”的文字性质不完全一致。正文使用时按“0% = random allocation，100% = optimal placement”的语义解释（Sec. 3.2.2，Sec. 4.1，Tab. 1）。

### 最优 baseline

由于 task mapping 是 NP-hard，论文没有穷举所有映射，而是用两类近似建立 `pi*`（Sec. 3.3）：

- 2-4 chiplet 小系统使用 spectral partitioning、min-cut 和 sequential min-cut，目标是最小化 task graph edge cut。
- 更大系统使用 prolonged simulated annealing（长时间模拟退火），报告 10 个随机种子中最好的结果作为 `pi*`。

模拟退火参数为：初始温度 `T0=1000`，cooling rate `alpha=0.95`，每个温度 10,000 次迭代；每次 move 随机把一个 task 移到一个随机 chiplet；若 cost 下降则接受，若 cost 上升则按 `exp(-delta/T)` 接受；温度低于 `1.0` 或 50 个温度阶段无改进时停止（Sec. 3.4.1-Sec. 3.4.3）。

中间质量点 `20%/40%/60%/80%` 通过从随机初始状态开始运行退火，并在对应 iteration count 停止获得。论文报告参数敏感性：`alpha in {0.90, 0.95, 0.99}` 时结果变化 <3%；`IT in {5000, 10000, 20000}` 时在 10,000 收敛；10 个随机种子方差 <2%（Sec. 3.4.4，Sec. 3.4.5，Fig. A1）。

### 系统与网络模型

模拟系统包含 2、4、8、16 个 chiplet，使用 silicon interposer 上的 2D mesh NoI。4 chiplet 为 `2 x 2`，8 chiplet 为 `2 x 4`，16 chiplet 为 `4 x 4`。intra-chiplet bandwidth 为 512 GB/s，inter-chiplet link bandwidth 为 128 GB/s（Sec. 3.1，Sec. 3.7.1）。

网络延迟模型为（Sec. 3.7.1）：

```text
L(i, j) = Lbase + dij * Lhop + Lserialization
```

其中 `Lbase=85 ns`，`Lhop=45 ns/hop`，`Lserialization=160 ns`，`dij` 为 Manhattan hop 数。论文示例给出 1 hop 为 290 ns，2 hop 为 335 ns，16-chiplet 最大 3 hop 为 380 ns。实验结果中的 850 ns 不是纯物理 hop 延迟，而是 poor partitioning 下包含 congestion-induced queueing（拥塞排队）的有效通信延迟（Sec. 3.7.1，Sec. 4.1，Tab. A1）。

路由策略为 deterministic XY routing，不使用 adaptive routing。router 使用 credit-based flow control；packet 切成 64-byte flit；拥塞时由 virtual cut-through 转为存储/排队。论文把 cache coherence traffic 作为 data traffic 的额外膨胀项处理，baseline run 中按 15% communication volume inflation 建模（Sec. 3.7.2）。

### 拥塞定义

单条链路在时间 `t` 的 congestion 定义为（Sec. 3.7.3）：

```text
Cong(i, j, t) = traffic_offered(i, j, t) / link_capacity * 100%
```

总体 congestion 是所有 inter-chiplet links 和所有 time steps 上的平均 link utilization。论文把 `Cong < 70%` 定义为 sustainable operation；`Cong > 100%` 表示 offered load 超过链路容量，packet 需要排队，可能发生 buffer overflow 和 retransmission（Sec. 3.7.3）。

### 吞吐与能效 baseline

论文使用两个 baseline（Sec. 3.8.1）：

| Baseline | 含义 |
| --- | --- |
| primary baseline | 同等计算资源的 monolithic chip，单 die、64 cores、16 MB unified L3、单 memory controller、NoC 而非 NoI |
| secondary baseline | 均匀/随机任务分配的 baseline chiplet system，用于衡量 optimized partitioning 相对 naive deployment 的收益 |

吞吐用 ResNet-50 inference 的 images/s 表示。论文把 8-chiplet ResNet-50 的 `Q=100%` throughput 写为 245 images/s，`Q=0%` 为 28 images/s，因此提升为 `245/28=8.75x`（Sec. 3.8.2，Tab. 2，Tab. A2）。

能效用 TOPS/W 表示。论文将 ResNet-50 定义为 94 billion MACs，并把 execution time 中的 computation、memory latency 和 inter-chiplet communication latency 都计入。功耗由 compute power 和 inter-chiplet communication power 组成（Sec. 3.8.3）。

## 实验设置

### Workload profile 来源

任务图与通信量来自三类来源（Sec. 3.5.2）：

- PyTorch profiler 的 layer-by-layer execution analysis。
- NVIDIA V100 上的 memory access traces。
- 已发表的 DNN kernel characterization papers。

论文没有给出真实芯片上的端到端实测结果；结果来自 `Chiplet-Based DNN Accelerator Simulator (Chip-DNN-SIM)` 和论文自述的 network/power model（Sec. 3.2，Sec. 3.5）。

### 评价指标

论文评价 6 类指标（Sec. 3.9）：

| 指标 | 含义 |
| --- | --- |
| Inter-Chiplet Communication Latency | 跨 chiplet 数据传输平均延迟 |
| Inter-Chiplet Traffic Percentage | 总 traffic 中跨 chiplet links 的比例 |
| Network Congestion | inter-chiplet link utilization，超过 70% 表示不可持续运行风险 |
| System Throughput | DNN inference images/s |
| Energy Efficiency | TOPS/W，包含计算和通信开销 |
| Network Communication Overhead | 执行时间中被跨 chiplet 通信消耗的比例 |

## 结果与解释

### 分区质量与 latency/traffic

8-chiplet ResNet-50 场景下，分区质量越高，latency 和跨 chiplet traffic 越低（Fig. 5a，Fig. 5b，Tab. A1）。

| Partitioning Quality | Inter-Chiplet Latency | Inter-Chiplet Traffic |
| --- | ---: | ---: |
| 0% | 850 ns | 95% |
| 20% | 720 ns | 78% |
| 40% | 580 ns | 62% |
| 60% | 380 ns | 45% |
| 80% | 185 ns | 28% |
| 100% | 85 ns | 12% |

论文解释为：poor partitioning 把强通信任务放在远距离 chiplet 上，并破坏本地 cache/data reuse；optimized partitioning 通过 co-location（把频繁通信任务放在同一 chiplet）降低跨 chiplet 边数量和物理距离（Sec. 4.1，Sec. 4.2）。

### 吞吐和能效

8-chiplet ResNet-50 场景下，throughput 与 energy efficiency 随 `Q` 单调提高（Fig. 5c，Tab. A2）。

| Partitioning Quality | Throughput | Energy Efficiency |
| --- | ---: | ---: |
| 0% | 28 images/s | 1.2 TOPS/W |
| 20% | 52 images/s | 2.1 TOPS/W |
| 40% | 89 images/s | 3.8 TOPS/W |
| 60% | 142 images/s | 6.2 TOPS/W |
| 80% | 198 images/s | 9.1 TOPS/W |
| 100% | 245 images/s | 12.4 TOPS/W |

论文正文报告中等质量的 heuristic partitioning 已能获得大量收益，举例为 throughput 从 28 images/s 增至约 142 images/s，energy efficiency 从 1.2 TOPS/W 增至约 6.2 TOPS/W。附录 `Tab. A2` 中，142 images/s 和 6.2 TOPS/W 对应 `Q=60%`，引用具体质量点时以 `Tab. A2` 为准（Sec. 4.3，Tab. A2）。

### Network congestion 与系统规模

网络拥塞随 chiplet 数量增加而放大，poor partitioning 的放大速度高于 optimized partitioning（Fig. 5d，Tab. A3）。

| Chiplet Count | Poor Partitioning | Good Partitioning |
| --- | ---: | ---: |
| 2 | 15% | 8% |
| 4 | 52% | 18% |
| 8 | 148% | 32% |
| 16 | 320% | 58% |

论文把 `Cong > 100%` 解释为 offered load 超过链路容量。8-chiplet poor partitioning 已达到 148%，16-chiplet poor partitioning 达到 320%，意味着链路需要排队或重传，系统无法满足稳定实时运行。optimized partitioning 在 16-chiplet 下保持 58%，低于论文定义的 70% sustainable threshold（Sec. 3.7.3，Sec. 4.4）。

### 通信开销占执行时间

Fig. 4 显示通信开销占总执行时间的比例随 chiplet 数量上升：

- 2-chiplet：poor partitioning 为 45%，optimized partitioning 为 12%，差距 33 percentage points。
- 16-chiplet：poor partitioning 为 85%，optimized partitioning 为 35%，差距 50 percentage points。

论文解释为：poor partitioning 下，8+ chiplet 系统的执行时间已经主要由通信而非计算决定；16-chiplet 时 85% 时间用于跨 chiplet 通信，计算优化、频率提升或局部 cache 扩容不再是主导杠杆（Sec. 4.5，Sec. 5.1）。

### Workload 类型差异

Fig. 6 比较 8-chiplet 下不同 workload 类型或 DNN 的表现。论文给出的 DNN communication overhead 为（Fig. 6b）：

| DNN | Poor Partitioning | Optimized Partitioning |
| --- | ---: | ---: |
| ResNet-50 | 85% | 35% |
| VGG-16 | 92% | 42% |
| DarkNet-19 | 45% | 18% |

VGG-16 的通信开销最高，DarkNet-19 较低。论文还给出 memory-bound workload 的假设：若 memory latency 远大于 communication latency，则 partitioning quality 的影响会较小，8.75x 可能缩小到 1.5-2x；该假设未做实验验证（Sec. 3.5.4，Fig. 6a）。

## 设计规则

论文提出 feasibility boundary（可行性边界）：随着 chiplet 数量增加，naive partitioning 不只是变慢，而是会在跨 chiplet link offered load 超过容量后进入 congestion collapse，导致 buffer overflow、task failure 或 5-10x task completion delay（Sec. 3.6.1，Sec. 3.7.3）。

论文给出的最低分区质量规则为（Sec. 3.6.2）：

```text
threshold(m) ~= 0.3 + 0.05 * m
```

| Chiplet Count | Required Qmin |
| --- | ---: |
| 2 | >= 0.4 |
| 4 | >= 0.5 |
| 8 | >= 0.7 |
| 16 | >= 0.95 |

该规则是 workload-dependent，论文只在 DNN 通信模式和敏感性曲线内使用。它不能直接作为所有 chiplet CPU workload 的通用阈值（Sec. 3.6.2，Sec. 3.6.3）。

## 分析

- 论文的核心不是“本地 L3 优先”，而是“强通信任务 co-location”。优化对象是 task graph edge cut 和 NoI traffic，不是 CPU CCD 的 cache fill 来源（Sec. 3.2，Sec. 4.2）。
- 850 ns latency 是拥塞和排队后的有效延迟，不等于 chiplet interposer 的无拥塞物理 hop 延迟。论文自己的 hop latency 示例最大为 380 ns，850 ns 来自 poor partitioning 下的 congestion-induced queueing（Sec. 3.7.1，Sec. 4.1，Tab. A1）。
- 8.75x throughput 和 10.3x energy efficiency 来自模型计算和仿真，不是商业 CPU 或真实 chiplet accelerator 的硬件测量结果（Sec. 3.8，Sec. 4.3，Tab. A2）。
- 论文对 DNN 之外 workload 的描述主要是 hypothesis。memory-bound workload 只能达到 1.5-2x 的说法没有实验证据，不能引用为已验证结论（Sec. 3.5.4）。
- 论文的“16+ chiplets 不能使用 naive partitioning”成立条件是 DNN-style communication-intensive task graph、2D mesh NoI、128 GB/s inter-chiplet link 和文中拥塞模型。不能直接推广到 AMD EPYC CPU 的 CCD/L3 层级（Sec. 3.1，Sec. 3.7，Sec. 4.4）。

## 证据口径与异常

论文包含若干引用时需要标注的口径问题：

- `Table 1` 的 `Q` 公式排版与正文定义的 `Q=0` worst、`Q=1` optimal 不完全一致。正文实验和图表按百分比质量点解释即可（Sec. 3.2.2，Tab. 1）。
- buffer 配置描述存在不一致：Sec. 3.7.2 先写每个 router node 有 4 个 input buffers、capacity 为 32 flits，Sec. 3.7.3 又写 maximum queue depth 为 8 flits。引用拥塞结果时应以最终 `Cong%` 与表格数据为主，不推导具体 buffer 容量。
- Sec. 4.3 正文写从 0% 到 50% quality 后 throughput 为 142 images/s、energy efficiency 为 6.2 TOPS/W；`Tab. A2` 中这两个值对应 60% quality。引用质量点和数值组合时应以 `Tab. A2` 为主。
- 能效计算中，Sec. 3.8.4 的 raw TOPS/W 为 0.58 与 2.07，随后又说明论文报告值为 1.2 与 12.4，差异来自分布式供电、frequency scaling、排队功耗和 cache efficiency 等模型补正。引用 10.3x 时应说明这是论文模型报告值，不是 raw power equation 直接算出的结果（Sec. 3.8.4，Sec. 3.8.5）。
- Data Availability Statement 指向一个 `claude.ai` public artifact，并且 acknowledgments 说明使用 `ChatGPT5.2` 做 grammar and structural editing。该信息不否定文中数据，但后续引用核心结论时应优先使用论文正文表格和图，而不是外部 artifact 的可复现性假设。

## 边界

- 研究对象是 DNN inference task mapping，不是 CPU attention、KV cache、OpenMP task scheduling 或 vLLM serving pipeline。
- 任务粒度是 DNN layer/task graph，不是 CPU 线程、cache line、page 或 attention tile。
- 系统是参数化 chiplet DNN accelerator simulator，不是 AMD EPYC Milan/Genoa/Bergamo 实机。
- 网络是 2D mesh NoI，链路带宽、延迟、flow control、coherence traffic 都由模型设定。
- 缓存建模是简化的。论文明确列出 simplified task graphs、uniform intra-chiplet latency、no cache coherence misses 等限制，并估计真实系统会增加 20-40% communication overhead，但该估计不是实机验证结果（Sec. 3.5.5）。
- 论文未评估 dynamic repartitioning。其主要结果来自 static partitioning 质量变化；runtime scheduling 被列为 future work（Sec. 5.4）。

## 可迁移点

- 对任何 chiplet 优化方案，先把 workload 表达为通信图：谁与谁交换数据、边权多大、哪些边在关键路径上。若通信图可被 co-location 降低 edge cut，partitioning 才有优化空间（Sec. 3.2，Sec. 4.2）。
- 评估分区策略时不能只看平均 latency，应同时报告 traffic share、link utilization、communication overhead 和 throughput。论文的 `latency/traffic/congestion/throughput/energy` 多指标组合比单一 L3 miss rate 更能解释系统级退化（Tab. 1，Fig. 5）。
- chiplet 数量增加后，通信代价可能出现非线性放大。即使 2-chiplet 或 4-chiplet 下 naive mapping 尚可运行，8-chiplet 或 16-chiplet 下也可能进入拥塞区（Fig. 5d，Tab. A3）。
- 对 DNN/tensor computation 等 structured communication workload，目标不一定是全局最优映射。论文认为 `Q>=70%-80%` 已能捕获大量收益，适合用 heuristic 或 learning-based mapper 先获得中高质量分区（Sec. 3.6.2，Sec. 5.3）。

## 不可直接迁移点

- 不能从该论文推出当前 CPU attention 的 `acc-local-l3` 静态分组会有效。该论文优化的是 DNN task graph 的跨 chiplet communication edge cut；`acc-local-l3` 改变的是 CPU attention task 的领取范围和本地 L3 访问形态。
- 不能把 87.4% traffic reduction、8.75x throughput、10.3x energy efficiency 当作 vLLM CPU 推理可达收益。这些数字依赖 8-chiplet DNN accelerator、ResNet-50、论文网络模型和能耗模型。
- 不能把 `Qmin` 阈值直接用于 AMD EPYC CCD 数量。论文的 chiplet 是 NoI 节点；当前平台的 CCD/L3、NUMA/NPS、socket 互连和 DDR 访问路径不同。
- 论文没有覆盖 memory-bound graph/sorting、control-flow runtime scheduling 和 sparse irregular workload；当前 LLM 推理中的 decode/KV streaming 更接近持续读写和调度问题，不能直接套用 DNN layer task graph 结论。
- 对当前课题更可迁移的是“通信图和 edge cut 思路”，而不是论文的具体模拟参数。若要应用到 LLM serving，应先定位全链路中是否存在可通过 co-location 消除的大流量跨 chiplet 通信，例如跨 rank 数据交换、batch 编排、KV 搬迁或框架级队列通信。

## 证据

- 原始资料：`wiki/原始资料/papers/Effects_of_Poor_Workload_Partitioning_Chiplet.pdf`
- 关键锚点：摘要；Fig. 4-6；Tab. 1-2；Tab. A1-A3；Sec. 2.5；Sec. 3.1-3.9；Sec. 4.1-4.5；Sec. 5.1-5.4；Sec. 7
