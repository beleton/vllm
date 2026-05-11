# OLAP on Modern Chiplet-Based Processors 解读

## 说明
- 依据版本：PVLDB 2024 最终中稿版，卷 17 第 11 期，页码 3428–3441，doi: `10.14778/3681954.3682011`。
- 代码与数据：`https://github.com/Alessandro727/OLAP-on-Modern-Chiplet-Based-CPUs`。
- 以下内容均以本地 PDF 为准，未对照 arXiv 或其他版本。

## 问题
Chiplet CPU 在同一 socket 内存在分片 L3 cache，核间延迟最高可达 6x 差异（Fig. 1a），核间带宽也不均等（Fig. 4）。现有查询引擎的 worker 部署策略（WIM、WIN、WIC）均未显式考虑 chiplet 拓扑，在 chiplet CPU 上导致 CPU 利用不充分和性能非线性扩展（摘要，Sec. 1）。论文要回答的是：仅通过不修改查询引擎源码的外部 worker 部署策略，能否在 chiplet CPU 上获得显著性能提升。

## 背景：分布式 OLAP 查询引擎的计算模型

### 架构角色：Coordinator 与 Worker
分布式查询引擎的 compute 层由两类角色组成（Sec. 3.1）：
- **Coordinator**：接收 SQL 查询 → 解析 → 生成执行计划 → 优化 → 将查询拆分为可并行的 task → 下发给各 Worker。
- **Worker**：独立进程，持有数据库表的一部分数据分片（fragment）。对分配到的 query task 在其本地数据片段上执行 scan、filter、aggregation、join 等操作，然后将中间结果或最终结果通过网络返回给 Coordinator 或其他 Worker。

Worker 之间通过网络栈通信（TCP/RDMA），不是共享内存。同机多 Worker 部署时，Worker 间通信同样经过网络栈（Sec. 3.1, Sec. 3.2）。

### 部署粒度：一台机器上可启动几个 Worker
四种部署策略的核心区别是：一台机器上启动多少个 Worker 进程，以及每个 Worker 绑定到哪些硬件资源（Sec. 3.2, Fig. 6）。以 AMD EPYC Milan 双路（2 socket × 8 chiplet × 8 core = 128 core）为例：
- WIM：整机 1 个 Worker，管理全部 128 core 和 512 GB 内存。
- WIN：每 NUMA node 1 个 Worker（若 BIOS 设为一个 socket 一个 NUMA node，则共 2 个）。
- WICP：每 chiplet 1 个 Worker（共 16 个）。
- WIC：每 core 1 个 Worker（共 128 个）。

Worker 数量越多，每个 Worker 负责的数据分片越小，越容易装入本地 cache，但 Worker 间通信量也随之增大（Sec. 3.2, Fig. 6）。

### OLAP 查询的访存特征
OLAP 查询的访存模式与 chiplet 架构产生冲突的关键点在于：
1. **表扫描（TableScan）**：对整张表做顺序扫描。数据量大，扫描过程把数据逐批加载到执行线程所在 chiplet 的 L3 cache 中。当多个 chiplet 上的线程同时扫描同一张表的不同片段时，各自的数据进入各自的本地 L3。但如果后续操作（如 Repartition、join）需要跨线程访问数据，就会触发跨 chiplet cache coherence。
2. **重分区（Repartition）**：join 或聚合前，需按 key 重新分配数据到不同 Worker/线程。一个 chiplet 上的线程 A 扫描和分区后的数据，线程 B（在另一个 chiplet 上）需要消费。线程 B 访问该数据时，硬件 cache coherence 自动通过 Infinity Fabric 从 chiplet A 的 L3 拉取 64 字节 cache line。
3. **Hash Join**：两个表按 join key 做 hash 分区后匹配。build 端和 probe 端可能在不同 chiplet 处理，跨 chiplet 的数据交换频繁。

这些操作的共同点是：数据在多个 chiplet 的 L3 之间因 cache coherence 而反复移动，每次移动的粒度是 64 字节 cache line，产生大量细碎的 Infinity Fabric 流量。

## 观察：三种 Chiplet CPU 的架构差异

### 核间延迟与带宽的非均匀性
- **AMD EPYC Milan**：核间延迟差异最大。同 chiplet 内约 24 ns，跨 chiplet 平均 106 ns，最远 chiplet 间可达 125 ns 以上（Fig. 3a）。核间带宽也分层：chiplet 内约 12 GB/s，跨 chiplet 约 6 GB/s（Fig. 4a, Sec. 2.4）。内存控制器在中央 I/O die 上，所有 chiplet 经过 Infinity Fabric 访问。
- **Intel Sapphire Rapids**（4-tile XCC 版本）：核间延迟整体均匀，平均 59 ns。核间带宽 10.5–12.5 GB/s。使用 EMIB 水平互联 + Foveros 垂直堆叠，减少了 chiplet 间通信代价（Fig. 3b, Fig. 4b, Sec. 2.1）。
- **ARM Graviton 3**：64 个 core 全部在一个 compute chiplet 上，核间延迟一致低于 59 ns（Fig. 3c, Sec. 2.1）。WIM/WIN/WICP 在此平台上实质相同。

### 聚合内存带宽与 chiplet 部署
STREAM benchmark 在三款 chiplet 平台上均显示：`1 process per chiplet` 部署能获得更高聚合带宽。AMD Milan 上约 7 GB/s，优于 `1 process per NUMA node` 的 3.5 GB/s（Fig. 5a, Sec. 2.5）。作为对照，单芯片架构的 Intel Xeon Gold 在不同部署策略间无聚合带宽差异（Fig. 5d），说明 chiplet 分片是造成带宽差异的根因。

## 方法：四种 Worker 部署策略与 WICP 的机制原理

### 四种策略定义（Sec. 3.2, Fig. 6）
- **WIM（Worker per Machine）**：整机一个 worker，所有 thread 可访问所有 chiplet 的 L3 和全部主存。Presto、SparkSQL 的默认部署。
- **WIN（Worker per NUMA）**：每 NUMA node 一个 worker，通过 Membind 强制 worker 只使用本地内存。SingleStore 的 NUMA-aware 配置。
- **WICP（Worker per Chiplet）**：每 chiplet 一个独立 worker，worker 只能使用对应 chiplet 的 core 和本地 L3。论文首次提出。
- **WIC（Worker per Core）**：每 core 一个 worker。Greenplum 和 H-Store 采用。

### WICP 为何有效：切断跨 chiplet 的 cache coherence 路径

WIN 和 WICP 使用的线程总数相同（以 Milan 单 socket 64 core 为例，都跑满 64 core）。区别在于 WIN 把 64 个线程放在 1 个 Worker 进程内，WICP 把 64 个线程拆成 8 个独立 Worker 进程（每 chiplet 8 线程）。

**WIN 下跨 chiplet 访问是硬件 cache coherence 自动触发的**。WIN 的 64 个线程共享同一个虚拟地址空间。线程 A（chiplet 0）扫描的数据加载到 chiplet 0 的 L3。线程 B（chiplet 3）做 Repartition 时访问同一批数据——它只是正常读写共享内存中的地址，不感知最新副本在哪个 chiplet 的 L3 里。硬件 cache coherence 协议自动通过 Infinity Fabric 以 64 字节 cache line 粒度将数据从 chiplet 0 的 L3 拉到 chiplet 3。64 个线程在 8 个 chiplet 上交叉访问共享内存，每秒产生数百万次这种细碎 cache line 拉取请求，Infinity Fabric 被这些往返消息塞满。WIN 的 Membind 策略锁定了主存分配在本地 NUMA node，但无法阻止 L3 级别的跨 chiplet cache coherence 流量。

**WICP 从根本上切断了跨 chiplet 的 cache coherence 路径**。8 个 chiplet 上分别启动 8 个独立 Worker 进程，每个进程有独立的虚拟地址空间。chiplet 0 上 Worker 进程的数据只进入 chiplet 0 的 L3，chiplet 3 上 Worker 进程的地址空间映射不到 chiplet 0 L3 里的数据。不同进程之间不存在 cache coherence 关系，因此不再有跨 chiplet 的 cache line snoop/probe/invalidate 流量。Worker 之间需要交换数据时走网络栈（同机 TCP loopback），传输的是 MB 级连续 buffer，不是数百万次 64 字节 cache line 拉取。物理上仍然经过 Infinity Fabric，但流量从碎片化的 coherence 风暴变成了受控的大块数据搬运。

**TCP 比 IF 快不是因为 TCP 协议更快，而是因为 TCP 承载的是已经消除了 coherence 风暴之后的批量传输**。WIN 下 IF 同时承载计算访存和跨 chiplet cache coherence 流量，后者把通道碎片化。WICP 下 IF 只承载网络包的批量搬运，没有 coherence 争抢。

### WICP_Local 与 WICP_Mixed（Sec. 4.4）
论文根据工作集大小与 L3 容量的关系，将 WICP 进一步区分为：
- **WICP_Local**：worker 使用单个 chiplet 上的全部 core。工作集 < 单 chiplet L3（32 MB）时最优，所有 cache 命中都是本地 chiplet L3（~24 ns）。
- **WICP_Mixed**：worker 使用相同数量的 core 但均匀分布在多个 chiplet 上。工作集在 32 MB 到 256 MB（聚合 L3）之间时更优，以跨 chiplet 的 cache 访问代价换取更大的可用总容量。

STREAM benchmark 在单 socket Milan 上验证了这一边界（Fig. 10）：数组 < 32 MB 时 WICP_Local 带宽更高；超过 32 MB 后 WICP_Local 带宽骤降，WICP_Mixed 保持平稳；超过 256 MB（聚合 L3）后两者均回退到 DRAM，性能趋同。

## 实验设置（Sec. 4.1）

### 硬件平台
- 主实验：双路 AMD EPYC Milan 7713（64 core/socket, 512 GB RAM, 8 chiplet/socket）；双路 Intel Sapphire Rapids Xeon Platinum 8480+（56 core/socket, 512 GB RAM, 4 tile/socket）；单路 ARM Graviton 3（64 core, 128 GB RAM）。
- 微架构测量额外使用：单路 AMD EPYC Milan 7713；单路 Intel Sapphire Rapids Xeon Platinum 8488C（48 core）；单路 ARM Graviton 3；单路 Intel Xeon Gold 第三代（24 core 单芯片架构，作为对照）。

### 查询引擎与 Workload
- 引擎：Presto、SingleStore、SparkSQL。三者的默认部署均为 WIM，论文不修改引擎源码。
- Workload：TPC-H（22 条查询，SF100 = 100 GB）、TPC-DS（99 条查询）、JCC-H（22 条查询，在 TPC-H 基础上注入 join-crossing correlation 和 skew），共 143 条查询，覆盖 ad-hoc 分析和决策支持两类负载。
- 所有结果取 5 次运行平均值。度量指标是查询执行时间，以 speedup over WIM 的形式呈现。

### Baseline
总 baseline 是 WIM（所有三个引擎的默认部署）。子实验中另有 WIN 和 WIC 作为比较对象。

## 结果与解释

### 各平台上的 WICP 收益（Fig. 7, Sec. 4.2）
TPC-H SF100 geometric mean speedup：
- AMD Milan：SparkSQL 3.41x，SingleStore 2.64x，Presto 2.61x
- Intel Sapphire Rapids：SparkSQL 6.97x，SingleStore 3.64x，Presto 2.70x
- ARM Graviton 3：SparkSQL 1.36x，Presto 无改善（0.84x–1.0x），因为 Graviton 3 只有一个 compute chiplet

论文将 SparkSQL 的高收益归因于：WIM 下数据在线程间分配严重不均 → LLC miss rate 高 → 远端 NUMA 读取频繁 → 互联拥塞时间平均占查询执行时间的 34%。WICP 通过均分数据和强制本地内存访问缓解了这些问题（Sec. 4.2）。

### 逐查询分析（Fig. 8, Sec. 4.2）
AMD Milan 上 TPC-H 各查询逐一分析：
- **SingleStore**：WICP 下为部分查询生成了与 WIM 不同的查询计划（WIM 用 nested loop join，WICP/WIN/WIC 用 Bloom filter + hash join）。Q7（23.9x）、Q13（18.1x）、Q22（21.2x）受益最大，Q13 的 Orders 表扫描从 5.36s 降至 0.4s，Repartition 从 5.06s 降至 0.25s。
- **Presto**：所有策略使用相同查询计划，但 worker 数量增加使数据 fragment 更小更易装入 L3。涉及 Orders 表的 ScanFilter 操作加速最明显（Q5 88%、Q18 96% 时间花在 Orders ScanFilter）。涉及 Lineitem 表的查询（Q1、Q6、Q14、Q19、Q20）speedup 低于 2x。
- **SparkSQL**：所有查询均获加速（3.51x–5.15x），依赖 SortMerge Join，WIM 下数据跨 chiplet 不均分配导致 join 阶段互联拥塞严重。

### 多核可扩展性（Fig. 9, Sec. 4.3）
AMD Milan 双路 128 core：WICP 扩展性最优。所有引擎的 WICP 在 64 core 内接近线性扩展，128 core 时 speedup 分别达 71x（Presto）、55x（SingleStore）、31x（SparkSQL，均归一化到 1 core）。WIM 仅扩展到约 16 core（2 chiplet），之后平台化。WIN 在 64 core 前良好，64 core 后与 WICP 差距拉大。

### L3 容量与策略选择（Fig. 10, Fig. 11, Tab. 2, Sec. 4.4）
- SingleStore：WICP_Local（3.32x） > WICP_Mixed（1.85x），因其数据传输量在单 chiplet L3 容量以内。
- Presto 和 SparkSQL：数据传输量超出聚合 L3，两种策略差异缩小（WICP_Mixed 略优）。
- 结论：WICP_Local vs WICP_Mixed 的选择由工作集大小相对 L3 容量决定，不是一刀切的 "chiplet 粒度最好"。

### Worker 数量敏感性（Fig. 12, Sec. 4.5.1）
双路 Intel Sapphire Rapids（共 8 chiplet）：speedup 随 worker 数增加而上升，在 8 worker（= chiplet 数）时达峰值。超过 8 后停止增长，28 后下降。最优 worker 数 = chiplet 数。

### 数据 Skew 的影响（Fig. 14, Sec. 4.5.3）
JCC-H（带 skew）实验：WICP geometric mean speedup 从 TPC-H 的 3.40x 降至 2.31x。论文解释：skew 给 WIM 带来了隐式局部性（热数据集中在少数位置），同时 skew 导致 WICP 内某些 chiplet 负载过重、本地 cache 溢出后回退到主存。

## 分析
- chiplet CPU 上 worker 部署粒度比 NUMA 粒度更重要。WIN（NUMA 级）无法解决同 NUMA domain 内 chiplet 间的 cache 分片和通信差异（Fig. 9, Sec. 4.3），WICP 用更细的 chiplet 粒度覆盖了这一层。
- 部署策略的选择与工作集相对 L3 容量直接耦合：< 单 chiplet L3 → WICP_Local；介于单 chiplet L3 和聚合 L3 之间 → WICP_Mixed；> 聚合 L3 → 两者趋同（Sec. 4.4）。
- WICP 不需要修改引擎源码，仅靠外部 worker 绑定和数据放置即可实现收益（Sec. 4.1, Sec. 7）。

## 边界
- 研究对象是 OLAP 查询引擎（Presto、SingleStore、SparkSQL）的 worker 部署策略，不是 LLM 推理，不是 attention/kernel 层优化。
- WICP/Local/Mixed 针对分布式查询引擎的 worker-to-core 映射和数据 fragment 分配，不等价于 attention head、KV cache 或 tensor partition 的 chiplet 调度。
- 实验平台限于三款 chiplet CPU（Milan、Sapphire Rapids、Graviton 3），不覆盖 Genoa/Bergamo 或 heterogeneous chiplet 配置。
- 论文未直接测量 L3 miss latency、L3 slice source 等微架构计数器，也未测 vLLM 或大模型服务负载。

## 可迁移点
- "部署粒度不应停留在 NUMA 层级而应下到 chiplet 级" 这一原则对 chiplet CPU 上任何并行化部署策略都有参考价值（Sec. 3.2, Fig. 9）。
- 容量驱动的放置决策（工作集 vs 单 chiplet L3 vs 聚合 L3）可抽象为通用启发式（Sec. 4.4, Fig. 10）。
- WICP 用独立进程地址空间切断跨 chiplet cache coherence 的机制，对任何需要消除芯片内细碎 coherence 流量的场景均有参考意义。

## 不可直接迁移点
- OLAP 查询的 worker fragment 模型与 LLM attention 的 head 切分、sequence 切分、KV 共享在结构上不同，不能直接套用 WICP。
- 论文 speedup 数值（最高 7x、单查询 23.9x）依赖于 TPC-H 查询的特定表扫描/join/sort 操作，不能外推到 attention kernel 或模型推理延迟。
- ARM Graviton 3 单 compute chiplet 上 WICP 几乎无收益，说明 chiplet 数量是 WICP 有效的前提条件。

## 证据
- 原始资料：`wiki/原始资料/papers/OLAP on Modern Chiplet-Based Processors.pdf`
- 关键锚点：摘要；Fig. 1–14；Tab. 1–2；Sec. 2.1–2.5；Sec. 3.1–3.2；Sec. 4.1–4.5；Sec. 5；Sec. 7
