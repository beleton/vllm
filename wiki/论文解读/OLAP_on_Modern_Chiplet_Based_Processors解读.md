# OLAP on Modern Chiplet-Based Processors 解读

## 说明
- 当前依据的原始资料是 `wiki/原始资料/papers/OLAP on Modern Chiplet-Based Processors.pdf`。
- 该 PDF 是 PVLDB 2024 最终中稿版，卷 17 第 11 期，页码 3428–3441，doi: `10.14778/3681954.3682011`。
- 论文代码与数据已公开在 `https://github.com/Alessandro727/OLAP-on-Modern-Chiplet-Based-CPUs`。
- 本文未额外对照 arXiv 或其他版本，以下内容均以这份本地 PDF 为准。

## 问题
- 论文要解决的是：chiplet CPU 在同一 socket 内存在分片 L3 cache、核间延迟最高可达 6x 差异（Fig. 1a）、核间带宽也不均等（Fig. 4），这些异构性超出了 NUMA 级远近内存访问的范畴。现有查询引擎的 worker 部署策略（WIM、WIN、WIC）均未显式考虑 chiplet 拓扑，在 chiplet CPU 上会导致 CPU 利用不充分和性能非线性扩展（摘要，Sec. 1）。

## 核心内容
- 论文在三款 chiplet CPU（AMD EPYC Milan、Intel Sapphire Rapids、ARM Graviton 3）上系统比较了四种 worker 部署策略对 OLAP 查询引擎性能的影响，其中 WICP（一个 worker 实例对应一个 chiplet）是论文首次提出的策略（Sec. 3.2）。
- 论文的核心主张是：不修改查询引擎源码，仅通过 chiplet 粒度的 worker 部署和数据放置，就能在 chiplet CPU 上获得明显性能提升。WICP 相对硬件无关的 WIM 部署最高可达 7x speedup，相对 NUMA-aware 的 WIN 部署最高可达 2x（摘要，Fig. 1b，Fig. 7）。
- 论文进一步给出了两条细化指南：工作集小于单 chiplet L3 容量时用 WICP_Local（只使用单个 chiplet 上的 core），工作集大于单 chiplet L3 但小于所有 chiplet 聚合 L3 时用 WICP_Mixed（每个 chiplet 各取一个 core）；工作集超过聚合 L3 时两者表现接近（Sec. 4.4，Fig. 10，Fig. 11）。

## 背景：分布式 OLAP 查询引擎的关键概念

### Coordinator 与 Worker
- 分布式查询引擎的 compute 层由两类角色组成：一个 **Coordinator**（协调器）和多个 **Worker**（工作实例）。Coordinator 负责接收 SQL 查询、解析、生成执行计划、优化查询、以及将查询拆分为可并行的 task 后下发给各个 Worker。每个 Worker 是独立进程，持有数据库表的一部分数据分片（fragment），对分配到的 query task 在其本地数据片段上执行 scan、filter、aggregation、join 等操作，然后将中间结果或最终结果通过网络返回给 Coordinator 或其他 Worker（Sec. 3.1）。
- Worker 之间通过网络栈通信，例如在做两表 join 时，两个持有不同表分片的 Worker 需要交换数据来完成 join。Worker 之间的通信路径不是通过共享内存，而是走 TCP/RDMA 等网络协议，因此在单机多 Worker 部署中，同机 Worker 间通信也要经过网络栈（Sec. 3.1，Sec. 3.2）。

### 部署粒度：一台机器上可以有几个 Worker
- 论文中四种部署策略（WIM / WIN / WICP / WIC）的核心区别就是：在一台机器上启动多少个 Worker 进程，以及每个 Worker 绑定到哪些硬件资源（core 集合、cache 范围、内存区域）。Worker 数量越多，每个 Worker 负责的数据分片越小，越容易装入本地 cache，但 Worker 间通信量也随之增大（Sec. 3.2，Fig. 6）。
- 以 AMD EPYC Milan 双路为例：该机器有 2 个 socket，每个 socket 含 8 个 chiplet，共 16 chiplet、128 core。WIM 在整机上只启动 1 个 Worker，这个 Worker 管理全部 128 core 和 512 GB 内存；WIN 在每 NUMA node 上启动 1 个 Worker（若 BIOS 设为一个 socket 一个 NUMA node，则共 2 个 Worker）；WICP 在每个 chiplet 上启动 1 个 Worker（共 16 个）；WIC 在每个 core 上启动 1 个 Worker（共 128 个）（Sec. 4.1，Sec. 3.2）。

### 跨 CCD 访问与核间通信发生的条件
- 论文中的 chiplet 即 AMD 术语下的 CCD（Core Complex Die），每个 chiplet 内有 8 个 core 和 32 MB 独占 L3。论文用"跨 chiplet 访问"或"核间通信"指代以下场景，这些场景是 WICP 试图消除或减少的对象（Sec. 2.1，Sec. 2.3，Sec. 2.4）：
- **跨 chiplet L3 cache 访问**。AMD Milan 的 Infinity Fabric 允许一个 chiplet 上的 core 访问另一个 chiplet 的 L3 cache，但延迟从 chiplet 内的 24 ns 上升到跨 chiplet 的平均 106 ns（Fig. 3a）。当 Worker 的数据分片散布在多个 chiplet 的 L3 中，或一个 core 要访问的数据被缓存在另一个 chiplet 的 L3 时，就会产生这种跨 chiplet cache 访问（Sec. 2.3）。
- **跨 chiplet 的数据移动（data shuffling）**。在 join、repartition、数据重分布等操作中，一个 Worker 处理后的中间结果需要传输给另一个 Worker。如果这两个 Worker 分别绑定到不同 chiplet，数据就要通过互联（Infinity Fabric / EMIB）跨 chiplet 传输。论文中 SparkSQL 在 WIM 下互联拥塞时间平均占查询执行时间的 34%，就是这种场景的直接后果（Sec. 4.2）。
- **远端 NUMA 内存访问**。当 Worker 访问的内存不在其所在 NUMA node 的本地内存控制器管辖范围内时，需要跨 socket 走远端 NUMA 访问。这比跨 chiplet L3 访问更昂贵。WIN 策略通过 Membind 消除此类访问，但 WICP 在 NUMA node 内多个 chiplet 场景下也间接受益于更均衡的内存访问分布（Sec. 3.2–(2)，Sec. 2.5）。
- **共享内存竞争（cache contention）**。多个 Worker 或线程争抢同一 chiplet 的 L3 容量，导致互相 evict 对方的数据。WIM 下 SparkSQL 数据分配不均实际上加剧了某些 chiplet L3 的争抢，而另一些 chiplet L3 则未被充分利用（Sec. 4.2）。

## 观察
### 三种 chiplet CPU 的架构差异
- AMD EPYC Milan 使用 8 个 7nm compute chiplet，通过一个 14nm 中央 I/O die 连接内存和 I/O；每个 chiplet 有 32 MB L3，socket 共计 256 MB；L3 cache 在 chiplet 内所有 core 间共享，核间通信经过 Infinity Fabric，延迟差异可达 10x（同一 chiplet 内平均 24 ns，跨 chiplet 平均 106 ns）。内存控制器在 I/O die 上，不直连 chiplet（Tab. 1，Fig. 3a，Sec. 2.1）。
- Intel Sapphire Rapids（XCC-tile 版本）采用 4-tile 设计，每个 tile 有 28.125 MB L3，socket 共计 112.5 MB；每 tile 直连两个 DDR5 内存通道。核间延迟更均匀，平均 59 ns。Intel 使用 EMIB 做 chiplet 间水平互联，Foveros 做垂直堆叠以缩短数据传输距离（Tab. 1，Fig. 3b，Sec. 2.1）。
- ARM Graviton 3 的 64 个 core 全部在一个 compute chiplet 上，每个 core 有独立 1 MB L3，共 64 MB；内存控制器和 PCIe 控制器与 compute chiplet 分离。核间通过 LIPINCON 互联，延迟一致低于 59 ns（Fig. 3c，Sec. 2.1）。由于 compute chiplet 只有一个，WIM/WIN/WICP 在这个平台上实质上是相同配置。

### 核间延迟与带宽的非均匀性
- AMD Milan 的核间延迟差异是其关键特征。同一 chiplet 内延迟约 24 ns，跨 chiplet 延迟平均 106 ns，最远 chiplet 间可达 125 ns 以上（Fig. 3a）。核间带宽同样分层：chiplet 内约 12 GB/s，跨 chiplet 约 6 GB/s（Fig. 4a，Sec. 2.4）。
- Intel Sapphire Rapids 的核间延迟整体均匀（平均 59 ns），核间带宽在 10.5–12.5 GB/s 之间波动。论文注意到一个反直觉现象：距离最远的 chiplet 间反而有略高的带宽，形成了热力图上反斜对角线的高亮区域，这与 Intel 的高带宽跨 die 直连通道有关（Fig. 4b，Sec. 2.4）。
- ARM Graviton 3 核间带宽在 7.2–8.6 GB/s，邻近 core 间通信带宽略高（Fig. 4c，Sec. 2.4）。

### 聚合内存带宽与 chiplet 感知部署
- 在用 STREAM benchmark 测量聚合内存带宽时，三个 chiplet 平台均显示以 `1 process per chiplet` 部署能获得更高带宽。AMD Milan 采用 `1 process per chiplet` 时聚合带宽约 7 GB/s，优于 `1 process per NUMA node` 的 3.5 GB/s 和 `1 process per machine` 的更低保值（Fig. 5a，Sec. 2.5）。
- Intel Sapphire Rapids 趋势一致，`1 process per chiplet` 聚合带宽约 2.7 GB/s（Fig. 5b，Sec. 2.5）。
- 作为对照，单芯片架构的 Intel Xeon Gold 在不同部署策略间无聚合带宽差异（Fig. 5d，Sec. 2.5）。

## 方法与系统设计
### 四种 worker 部署策略
- 论文定义了从粗到细四种部署策略，核心差异在于 worker 实例数量、每个 worker 可用的计算资源范围、以及数据分配策略（Sec. 3.2，Fig. 6）。
- **WIM（Worker per Machine）**：整机一个 worker，所有 thread 可访问所有 chiplet 的 L3 和全部主存。Presto、SparkSQL 的默认部署即属此类。优点是实现与硬件拓扑无关，整个 L3 cache 都可用；缺点是数据和线程的局部性难以保证，数据在各 chiplet L3 间分配不均时会拥塞互联（Sec. 3.2–(1)）。
- **WIN（Worker per NUMA）**：每个 NUMA node 一个 worker，通过 Membind 策略强制 worker 只使用本地内存。SingleStore 的 NUMA-aware 配置即属此类。优点是避免跨 NUMA 远端内存访问；缺点是同一 NUMA node 内的 chiplet 间通信仍然是均匀假设覆盖不到的粒度（Sec. 3.2–(2)）。
- **WICP（Worker per Chiplet）**：每个 chiplet 分配一个独立 worker，worker 只能使用对应 chiplet 的 core 和本地 L3。论文声称这是首次提出的策略，此前无已知系统采用。优点是最大化 chiplet 内局部性、消除跨 chiplet 通信并实现更均衡的数据分配；缺点是限制了单个 worker 可用的 cache 容量和内存带宽（Sec. 3.2–(3)）。
- **WIC（Worker per Core）**：每个 core 一个 worker，Greenplum 和 H-Store 采用此策略。优点是单线程性能好，避免了 core 间通过互联通信；缺点是 worker 数量激增导致网络通信量大增，且每个 worker 的 cache 分区更小（Sec. 3.2–(4)）。

### WICP_Local 与 WICP_Mixed 的区分
- 论文在 WICP 内部进一步区分为 WICP_Local（worker 使用单个 chiplet 上的全部 core）和 WICP_Mixed（worker 使用相同数量的 core 但均匀分布在多个 chiplet 上）。两者的核心差异是后者能用多 chiplet 聚合 L3 容量，但付出跨 chiplet 通信开销（Sec. 4.4）。
- STREAM benchmark 在单 socket AMD Milan 上对比这两种策略：WICP_Local 在数组大小低于单 chiplet L3 容量 32 MB 时带宽更高，超过 32 MB 后带宽骤降；WICP_Mixed 在 32 MB 到 256 MB（聚合 L3 容量）之间带宽更平稳，超过 256 MB 后两者均下降（Fig. 10，Sec. 4.4）。

### WIN 与 WICP 的机制差异：为什么相同线程数、不同效果

WIN 和 WICP 使用的线程总数相同——以 AMD EPYC Milan 单 socket 64 core 为例，两种策略都跑满全部 64 个 core。区别在于 WIN 把 64 个线程放在 1 个 Worker 进程内，WICP 把 64 个线程拆成 8 个独立 Worker 进程（每 chiplet 8 线程），每个 Worker 只访问自己 chiplet 本地的那份数据。

以论文中 TPC-H Q13 查询（对 `Orders` 表做全表扫描 + Repartition + join）为例说明这种区别的实际影响。该查询在 WIN 下 Orders 表扫描耗时 5.36 秒，Repartition 耗时 5.06 秒；WICP 下分别降至 0.4 秒和 0.25 秒（Sec. 4.2 逐查询分析中 SingleStore 的数据）。

**WIN 下的跨 chiplet 访问是硬件 cache coherence 自动触发的，不是软件主动选择的。** WIN 只启动 1 个 Worker 进程，其 64 个线程共享同一个虚拟地址空间。线程 A 跑在 chiplet 0 上扫描的数据被加载到 chiplet 0 的 L3（32 MB），过一会儿线程 B 跑在 chiplet 3 上做 Repartition 时也要访问同一批数据。线程 B 只是正常读写共享内存中的一个地址，并不感知这个地址的最新副本在哪个 chiplet 的 L3 里。硬件 cache coherence 协议自动通过 Infinity Fabric 把 chiplet 0 L3 中的那条 cache line（64 字节粒度）拽到 chiplet 3。线程 B 看到的就是一次普通访存，但延迟从 chiplet 内 24 ns 变成了跨 chiplet 106 ns（Fig. 3a），带宽从 12 GB/s 降到 6 GB/s（Fig. 4a）。64 个线程在 8 个 chiplet 上交叉访问共享内存，每秒产生数百万次这种 64 字节粒度的 cache line 拉取请求，Infinity Fabric 被这些细碎往返消息塞满。WIN 的 Membind 策略锁定了主存分配在本地 NUMA node，但无法阻止 L3 级别的跨 chiplet cache coherence 流量。

**WICP 从根本上切断了跨 chiplet 的 cache coherence 路径。** WICP 在 8 个 chiplet 上分别启动 8 个独立 Worker 进程，每个进程有独立的虚拟地址空间。chiplet 0 上 Worker 进程的数据只进入 chiplet 0 的 L3，chiplet 3 上 Worker 进程的地址空间根本映射不到 chiplet 0 L3 里的数据。硬件 cache coherence 的边界就是进程地址空间的边界——不同进程之间不存在 cache coherence 关系，因此不再有跨 chiplet 的 cache line snoop/probe/invalidate 流量。Worker 之间需要交换数据时走网络栈（同机 TCP loopback），传输的是 MB 级连续 buffer，不是数百万次 64 字节 cache line 拉取。这种批量传输物理上仍然经过 Infinity Fabric（因为 chiplet 间通信走 IF），但跑在上面的流量从碎片化的 coherence 风暴变成了受控的大块数据搬运。

**所以 TCP 比 IF 快不是因为 TCP 协议更快，而是因为 TCP 承载的是已经消除了 coherence 风暴之后的批量传输。** WIN 下 IF 同时承载正常的计算访存和跨 chiplet cache coherence 流量，后者把通道碎片化了。WICP 下 IF 只承载网络包的批量搬运，没有 coherence 争抢。

## 实验设置
### 硬件平台
- 主实验使用三款 chiplet 机器：双路 AMD EPYC Milan 7713（每路 64 core、512 GB RAM、8 chiplet）；双路 Intel Sapphire Rapids Xeon Platinum 8480+（每路 56 core、512 GB RAM、4 tile）；单路 ARM Graviton 3（64 core、128 GB RAM）。前两者均为双路双 NUMA domain 配置（Sec. 4.1）。
- 微架构测量（核间延迟、带宽、STREAM）额外使用了：单路 AMD EPYC Milan 7713；单路 Intel Sapphire Rapids Xeon Platinum 8488C（48 core）；单路 ARM Graviton 3；双路 AMD EPYC Milan（用于对比跨 NUMA 聚合带宽）；单路 Intel Xeon Gold 第三代（24 core 单芯片架构，作为对照）（Sec. 2.2）。

### 查询引擎与 Workload
- 查询引擎是 Presto、SingleStore、SparkSQL，三者的默认部署均为 WIM，论文不修改引擎源码（Sec. 4.1）。
- workload 是 TPC-H（22 条查询，SF100，即 100 GB 数据）、TPC-DS（99 条查询）、JCC-H（22 条查询，在 TPC-H 基础上注入了 join-crossing correlation 和 skew）。共 143 条查询，覆盖 ad-hoc 分析和决策支持两类负载（Sec. 4.1）。
- 通过 libnuma 库将 worker 绑定到特定 core/NUMA/chiplet 并强制数据放置。所有结果取 5 次运行平均值。度量指标是查询执行时间，结果以 speedup over WIM 的形式呈现（Sec. 4.1）。

### Baseline 说明
- 论文的总 baseline 是 WIM（Worker per Machine），因为 Presto、SingleStore、SparkSQL 均以此为默认部署。所有 speedup 数值均为相对于各引擎自身 WIM 部署的比值。子实验中另有 WIN 和 WIC 作为比较对象（Sec. 4.1）。

## 结果与解释

### WICP 在各平台上的收益
- 在 TPC-H SF100 的 geometric mean speedup 中，WICP 在所有三款引擎和三款 CPU 上均优于 WIN 和 WIC（Fig. 7，Sec. 4.2）。
- 各引擎在 AMD Milan 上的 WICP speedup：SparkSQL 3.41x，SingleStore 2.64x（原文 Fig. 7 中为 3.40x，但文字 Sec. 4.2 记为 2.64x），Presto 2.61x。在 Intel Sapphire Rapids 上：SparkSQL 6.97x，SingleStore 3.64x（原文文字记为 3.64x，图表读值约 2.7x），Presto 2.70x。在 ARM Graviton 3 上：SparkSQL 1.36x，Presto 無明显改善（0.84x–1.0x 范围内），因为 Graviton 3 只有一个 compute chiplet（Fig. 7，Sec. 4.2）。
- 论文把 SparkSQL 的 WICP 收益特别高归因于：SparkSQL 在 WIM 下数据在线程间分配严重不均，导致 LLC miss rate 高、远端 NUMA 读取频繁、互联拥塞时间平均占查询执行时间的 34%。WICP 通过均分数据和强制本地内存访问缓解了这些问题（Sec. 4.2）。
- WIC 对 Presto 反而有负面影响（AMD 上 0.84x，Intel 上 0.65x），原因是 Presto 的 Master 实例同时充当 Worker，在 WIC 下被限制为仅用一个 core，协调开销限制了整体吞吐（Sec. 4.2）。

### 逐查询分析
- 论文对 AMD Milan 上 TPC-H 各查询逐一做了详细分析（Fig. 8，Sec. 4.2）。
- SingleStore 在 WICP 下为部分查询生成了与 WIM 不同的查询计划：WIM 使用 nested loop join，WICP/WIN/WIC 使用 Bloom filter 和 hash join。后两者易于并行化和分段分发，减少了数据迁移量。Q7、Q9、Q13、Q15、Q16、Q20、Q22 受益最大，其中 Q7 的 WICP speedup 达 23.9x，Q13 达 18.1x，Q22 达 21.2x（原文图 8 读值）。Q13 的 Orders 表扫描从 WIM 下的 5.36 秒降至 WICP 下的 0.4 秒，Repartition 从 5.06 秒降至 0.25 秒（Sec. 4.2）。
- Presto 在所有策略下使用相同查询计划，但 worker 数量增加导致数据 fragment 更小更易装入 L3。ScanFilterTable 操作占 Presto 大多数查询时间，WICP 下该操作显著加速。Q5 和 Q18 分别有 88% 和 96% 的时间花在 Orders 表的 ScanFilter 上，因此加速最明显。涉及 Lineitem 表的查询（Q1、Q6、Q14、Q19、Q20）speedup 低于 2x，因为 Presto 对 Lineitem 表跳过了 scan/filter 预处理阶段（Sec. 4.2）。
- SparkSQL 在 WICP 下所有查询均获加速，speedup 在 3.51x 到 5.15x 之间。论文说明 SparkSQL 依赖 SortMerge Join，在 WIM 下数据跨 chiplet 不均分配导致 join 阶段互联拥塞严重（Sec. 4.2）。

### 多核可扩展性
- 在 AMD Milan 双路（共 128 core）上，WICP 的扩展性在三种引擎上均最优。所有引擎的 WICP 在 64 core 以内接近线性扩展，64 core 以后增速放缓但仍继续提升。在 128 core 时，WICP speedup 分别达到 71x（Presto）、55x（SingleStore）、31x（SparkSQL），对应已归一化到 1 core 基准（Fig. 9，Sec. 4.3）。
- WIM 仅能扩展到约 16 core（对应 2 个 chiplet），之后性能平台化。WIN 在 64 core 前扩展性良好，但 64 core 后与 WICP 的差距拉大：128 core 时 WICP 高出 WIN 约 2.15x（SparkSQL）、1.44x（Presto）、1.25x（SingleStore）（Fig. 9，Sec. 4.3）。

### L3 Cache 容量与放置策略选择
- 在 TPC-H SF50 实验中，WICP_Local 和 WICP_Mixed 对三种引擎的影响不同（Fig. 11，Sec. 4.4）。SingleStore 在 WICP_Local 下 speedup 3.32x，显著高于 WICP_Mixed 的 1.85x。Presto 也偏向 WICP_Local（2.61x vs 2.38x），但差距较小。SparkSQL 则 WICP_Mixed 略优（3.40x vs 3.32x）。
- 论文通过 Tab. 2 中各查询的峰值内存用量和带宽来解释：SingleStore 的数据传输量在单 chiplet L3 容量以内，因此 WICP_Local 大幅优于 WICP_Mixed；而 Presto 和 SparkSQL 的数据移动量超出聚合 L3 容量，主要时间花在主存访问上，两种策略差异缩小（Tab. 2，Sec. 4.4）。

### Worker 数量敏感性
- 在双路 Intel Sapphire Rapids（共 8 chiplet）上，WICP speedup 随 worker 数量增加而上升，在 8 worker 时达到峰值：Presto 3.62x、SingleStore 2.75x、SparkSQL 7.04x。超过 8 worker 后 speedup 停止增长，28 worker 后开始下降。8 恰好等于该平台的 chiplet 总数，论文据此说明最优 worker 数应等于 chiplet 数（Fig. 12，Sec. 4.5.1）。

### Query 多样性与 Data Skew
- 在 AMD Milan 上用 SingleStore 评估 TPC-DS 时，所有查询在 WICP 下均获得 speedup。扫描大表（如 inventory 表）的查询受益最大，因为 WICP 消除了 WIM 下的远端内存访问（Fig. 13，Sec. 4.5.2）。
- JCC-H（带 skew 的 TPC-H 变体）实验显示 WICP 的 geometric mean speedup 从 TPC-H 的 3.40x 降至 2.31x，WIN 从 2.68x 降至 1.72x（Fig. 14，Sec. 4.5.3）。论文解释：数据 skew 本身给 WIM 带来了一定的隐式局部性（热数据集中在少数位置），同时 skew 导致 WICP 内某些 chiplet 负载过重、本地 cache 溢出后回退到主存访问。缓解措施包括：在保持局部化处理的前提下动态调整 chiplet 资源分配，以及对通信流量进行重分布（Sec. 4.5.3）。

## 分析
- 从论文可直接提炼的事实是：chiplet CPU 上 worker 部署粒度比 NUMA 粒度更重要。WIN（NUMA 级）在 chiplet 数较多的平台上（如 AMD Milan 的单 NUMA node 内含 8 chiplet）无法解决同 NUMA domain 内 chiplet 间的 cache 分片和通信差异，而 WICP 用更细的粒度覆盖了这一层（Sec. 2.3，Sec. 4.3，Fig. 9）。
- 另一个直接结论是：部署策略的选择与工作集相对于 cache 容量的大小直接耦合。WICP_Local 适合工作集 < 单 chiplet L3，WICP_Mixed 适合工作集介于单 chiplet L3 和聚合 L3 之间，大于聚合 L3 后两者无差别。这不是一刀切的"chiplet 粒度最好"，而是一个容量驱动的决策（Sec. 4.4，Fig. 10，Fig. 11）。
- 论文还表明 WICP 不需要修改引擎源码，仅靠外部 worker 绑定和数据放置就能实现上述收益。论文同时指出引擎内部的 chiplet-aware 优化可能进一步改善性能，但不在本文范围（Sec. 4.1，Sec. 7）。

## 边界
- 论文研究对象是 OLAP 查询引擎（Presto、SingleStore、SparkSQL）的 worker 部署策略，不是 LLM 推理、不是 attention/kernel 层优化，不是通用 HPC 或 ML 训练负载。
- 论文中的 WICP/Local/Mixed 是针对分布式查询引擎的 worker-to-core 映射和数据 fragment 分配设计的，不等价于 attention head、KV cache 或 tensor partition 的 chiplet 调度。
- 实验平台限于三款 chiplet CPU（AMD EPYC Milan、Intel Sapphire Rapids、ARM Graviton 3），不覆盖 AMD Genoa/Bergamo、Intel Granite Rapids 或更多 heterogeneous chiplet 配置（如 Intel 的 HBM 版本仅提及未实测）。
- 论文没有直接测量 L3 miss latency、L3 slice source、CCX/cache slice ID 等微架构计数器，也没有测 vLLM 或大模型服务负载。

## 可迁移点
- "部署粒度不应停留在 NUMA 层级，而应下到 chiplet 级"这一原则对 chiplet CPU 上任何并行化部署策略都有参考价值。当前 vLLM CPU attention kernel 中若将 worker/线程按 NUMA 做绑定，可能同样忽略了同一 NUMA node 内 chiplet 间 L3 分片和互联差异（Sec. 3.2，Fig. 9）。
- 容量驱动的线程放置决策（工作集 vs 单 chiplet L3 vs 聚合 L3）可直接抽象为通用启发式：在 chiplet CPU 上做数据分区时，先估算工作集大小相对本地 L3 的比例，再决定是否推广到跨 chiplet 共享更大聚合缓存（Sec. 4.4，Fig. 10）。
- WICP 的"networking 取代共享内存通信"设计在 worker 实例间引入了网络栈开销，但当 worker 数增加时，多机场景下 WICP 与 WIM 的通信开销差距缩小（Sec. 5 讨论）。这对设计跨机推理部署策略有参考意义。

## 不可直接迁移点
- 论文的 WICP/Local/Mixed 策略是面向查询引擎 worker 的，worker 之间通过 SQL 查询计划协调，数据交换以 fragment shuffle/join 为单位。LLM 推理中 attention 计算的并行模式（head 切分、sequence 切分、KV 共享）与查询引擎的 worker fragment 模型结构不同，不能直接套用。
- 论文中 speedup 数值（最高 7x、23.9x 等）依赖 TPC-H 查询的特定表扫描/join/sort 操作，不是通用计算任务的收益，不能外推到 attention kernel 或模型推理延迟。
- 论文对 ARM Graviton 3 的实验结论显示单 compute chiplet 架构下 WICP 几乎无收益，这意味着当前单 chiplet 或小 chiplet 数平台上 WICP 的价值需要重新评估。

## 证据
- 原始资料：`wiki/原始资料/papers/OLAP on Modern Chiplet-Based Processors.pdf`
- 关键锚点：摘要；`Fig. 1–14`；`Tab. 1–2`；`Sec. 2.1–2.5`；`Sec. 3.1–3.2`；`Sec. 4.1–4.5`；`Sec. 5`；`Sec. 7`
