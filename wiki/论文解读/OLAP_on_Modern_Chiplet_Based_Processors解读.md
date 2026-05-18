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

Worker instance 不是 CPU thread。一个 Worker 是查询引擎的执行实例或进程边界，内部通常有线程池、task queue、operator pipeline 和局部执行状态。一个 Worker 被分配多个 core 时，可以同时运行多个查询 task。WIM 中单个 Worker 可使用整机全部 core，所有线程共享同一进程地址空间；WIN 中每个 Worker 绑定一个 NUMA node，线程只能在该 NUMA node 内迁移；WICP 中每个 Worker 绑定一个 chiplet，线程只使用该 chiplet 的 core；WIC 中每个 Worker 绑定一个 core（Sec. 3.2）。

### Worker 内部的多线程并行

OLAP 查询在执行前会被拆成多个层级（以 Presto 为例）：

```
Query
  └── Stage × N                  ← Coordinator 将查询分为树状依赖的阶段
        └── Task × N             ← 每个 Stage 拆分为多个 Task，分发到各 Worker
              └── Driver × N     ← 最底层并行单元，一个 Driver 处理一个 Split
                    └── Operator pipeline
                          └── TableScan → Filter → HashBuild → Probe → ...
                                └── 以 Page (≤1MB, ≤16K行) 为处理单位
```

- **Stage**：Coordinator 的概念，对应查询计划的逻辑阶段。Source Stage 读取数据源，Fixed Stage 做分布式聚合，Single Stage 汇总结果。
- **Task**：Worker 上执行的基本单位。一个 Stage 被拆为多个 Task，分发给持有对应 fragment 的 Worker。
- **Driver**：最小的并行执行单元。一个 Task 包含一个或多个 Driver，每个 Driver 驱动一条 Operator 流水线。一个 Driver 处理一个 Split。
- **Split**：数据源的一个子集（表的一段 block/offset）。Coordinator 的 Connector 将表切分为 Split 列表，分发给 Worker。Worker 的线程池从 Split 队列中取 Split → 创建 Driver → 执行。
- **Operator**：具体的操作符（TableScan、Filter、HashBuilderOperator、LookupJoinOperator 等），以 Page 为最小单位读写数据。多个 Operator 通过 Driver 连成流水线。

**Presto 的两个关键并发参数**：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `task.max-worker-threads` | CPU 核数 × 2 | 每个 Worker 最多同时运行的线程数（处理 Split 的线程） |
| `task.concurrency` | 16 | JOIN / Aggregation / Exchange 算子被拆分为多少路并行（hash table 的并行 partition 数） |

例如 `task.concurrency=16` 时，JOIN 会生成 16 个 `HashBuilderOperator` 同时建表，16 路 `LookupJoinOperator` 同时 probe。`task.max-worker-threads=64` 时，Worker 内部最多 64 个线程同时跑 Driver。

因此 `1 Worker + 64 cores` 可以是 64 个线程并行处理多个 Split，不等价于单线程执行。论文中的 WIM、WIN、WICP、WIC 比较的是 Worker 实例数量和绑定范围，不是”单线程 vs 多线程”。

不同引擎的术语不同，但结构相似：
- SparkSQL：Stage → Task → Executor 线程
- SingleStore：Query → Leaf → 线程池并行处理分区
- Greenplum：Query → Slice → Gang（进程池）→ 并行处理 Segment 上的数据

Worker 内部并行带来两个后果。第一，单个 Worker 已经可以利用多个 core，提高 Worker 数量不一定增加总并行度。第二，Worker 边界通常也是局部状态边界 —— Split 队列、buffer、hash table、Bloom filter、scan 状态、内存分配和调度策略围绕 Worker 组织。这些局部状态所在的 JVM 堆和 L3 cache 都与 Worker 的物理位置绑定。WICP 的核心作用是改变这些局部状态与 chiplet 拓扑的对应关系。

**Split 分配的关键性质**：Worker 内部是多线程从同一个 Split 队列中抢任务。线程和 Split 之间没有预先绑定的映射关系 —— 线程 0 可以处理任意 Split。这意味着即使通过 `numactl` 将线程 0 绑定到 chiplet 0，线程 0 取的 Split 对应的数据可能落在 NUMA node 1（另一 socket）的 DRAM 上。Presto 的 Split 分配逻辑不感知数据物理位置。只有通过多 Worker 将 fragment 和数据物理分配到对应的 NUMA node 后，每个 Worker 内的 Split 队列才天然只包含本地数据。

Worker 之间通过网络栈通信，不共享进程地址空间。同机多 Worker 部署时，Worker 间通信同样经过网络栈；论文提到 RDMA/InfiniBand 可降低网络瓶颈，但常见系统通常仍尽量减少通信（Sec. 3.1）。通信主要发生在两类场景：

1. **跨 Worker 数据交换**：多表 join、shuffle/repartition 等算子需要按 key 重新分布中间数据，一个 Worker 产生的数据需要发送给负责目标分区的其他 Worker。
2. **结果汇聚**：Worker 将局部结果或最终结果发回 Coordinator。

只访问本地 fragment 的 scan、filter、局部 aggregation 可在 Worker 内完成，不必触发 Worker 间通信。Coordinator 的 cost-based query plan 会考虑数据分布并尽量降低通信开销。

### 数据分区与重分区
数据分区（partitioning）指查询引擎按规则决定哪些数据由哪些 Worker 负责。常见规则包括按 key 做 hash 分区、按 key 范围分区、按文件或 block 切分。Worker 通常只直接处理自己的数据分片。

同样分布指两个输入已经按同一个 join key 和同一套分区规则放到 Worker 上。例如 `orders` 和 `customer` 都按 `custkey` 做 hash 分区，且 Worker 数和 hash 规则相同，则相同 `custkey` 的行会落到同一个 Worker。该 Worker 可以本地完成 join。

重分区（Repartition）指把已有中间数据按新的 key 和分区规则重新分布。若 `orders` 当前按 `orderkey` 分区，而查询需要按 `custkey` 与 `customer` join，查询引擎需要把 `orders` 的中间结果按 `custkey` 重新发送到目标 Worker，使相同 `custkey` 的 `orders` 行和 `customer` 行落在同一个 Worker。该过程也常称为 shuffle 或 exchange。

Repartition 会产生 Worker 间数据移动。Worker 数量越多、分区越细，单个 Worker 的本地数据可能更容易进入 cache，但跨 Worker 交换和结果汇聚也会增加。OLAP 论文中的 WICP 收益与这类数据移动、数据局部性和 worker 部署粒度直接相关。

### 部署粒度：一台机器上可启动几个 Worker
四种部署策略的核心区别是：一台机器上启动多少个 Worker 进程，以及每个 Worker 绑定到哪些硬件资源（Sec. 3.2, Fig. 6）。以 AMD EPYC Milan 双路（2 socket × 8 chiplet × 8 core = 128 core）为例：
- WIM：整机 1 个 Worker，管理全部 128 core 和 512 GB 内存。
- WIN：每 NUMA node 1 个 Worker（若 BIOS 设为一个 socket 一个 NUMA node，则共 2 个）。
- WICP：每 chiplet 1 个 Worker（共 16 个）。
- WIC：每 core 1 个 Worker（共 128 个）。

Worker 数量越多，每个 Worker 负责的数据分片越小，越容易装入本地 cache，但 Worker 间通信量也随之增大（Sec. 3.2, Fig. 6）。

### OLAP 查询的访存特征
OLAP 查询的访存模式与 chiplet 架构产生冲突的关键点在于：
1. **表扫描（TableScan）**：对整张表做顺序扫描。数据量大，扫描过程依赖 L3 cache 与主存带宽。WIM 下数据和线程分布不均时，容易产生较高 LLC miss rate、远端 NUMA 读取和互联拥塞。
2. **重分区（Repartition）**：join 或聚合前，查询引擎按 join key 或 group key 重新分布中间数据，使相同 key 的数据落到同一个 Worker。该阶段会产生 Worker 间数据移动和 shuffle，论文将其列为 WICP 能显著改善的操作之一。
3. **Hash Join**：两个表按 join key 做 hash 分区后匹配。SingleStore 在 WICP/WIN/WIC 下可选择 Bloom filter 与 hash join，减少 nested-loop join 带来的数据传输量和执行时间。

这些操作的共同点是：数据扫描、shuffle 和 join 对数据放置、cache locality 与互联压力敏感。WICP 通过 chiplet 粒度的 worker 部署改善局部性和数据分布。

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

### WICP 为何有效：worker 边界与 chiplet 拓扑对齐

WIN 和 WICP 可使用相同的总线程数。以 Milan 单 socket 64 core 为例，WIN 可让 1 个 Worker 使用 64 个线程，WICP 可让 8 个 Worker 各使用 8 个线程。二者的总并行 core 数相同，差别在于 Worker 边界、局部状态和数据分片的粒度不同。

**WIM 的问题是 Worker 边界过粗**。WIM 中一个 Worker 覆盖整机，线程可在多个 chiplet 上运行，task queue、buffer、hash table、scan 状态和中间结果都在同一进程地址空间内共享。若某个数据结构由 chiplet A 上的线程创建，却被 chiplet B/C/D 上的线程频繁访问，硬件上会表现为跨 chiplet cache coherence、远端 L3 或主存访问。引擎层面仍是 Worker 内部共享内存访问，但物理路径已经跨越 chiplet。

**WIN 的问题是只约束到 NUMA 层级**。WIN 依赖 Membind 将 Worker 内存限制在本地 NUMA node，并限制线程只在该 NUMA node 内迁移。Milan 的一个 NUMA domain 内仍包含多个 chiplet，chiplet 内外的核间延迟、带宽和 L3 访问代价不同。WIN 可以减少跨 NUMA 访问，但不能消除同 NUMA 内部的跨 chiplet 访问。

**WIC 的问题是 Worker 边界过细**。每 core 一个 Worker 能获得强局部性，但 Worker 数过多会增加 exchange、shuffle、结果汇聚、运行时元数据和协调开销。Presto 在 WIC 下变慢的原因之一是 coordinator-worker 被限制在单 core，aggregation 等阶段受限。

**WICP 位于 WIM/WIN 与 WIC 之间**。每个 chiplet 对应一个 Worker，Worker 只使用该 chiplet 的 core。Worker 私有的 task queue、局部 buffer、局部 hash table、Bloom filter、scan 状态和中间结果更可能由同一 chiplet 内的线程创建和消费。结合 first-touch 或 membind，内存页更容易落在对应 NUMA 域内；结合 chiplet 绑核，L3 cache 的 producer-consumer 更容易留在本地 chiplet。论文将收益归因于更均衡的数据分布、更好的 cache locality、更少的 inter-chiplet communication 与更低的 interconnect pressure。

WICP 不是“通信一定更便宜”。WIM 的跨 chiplet 通信主要体现为 Worker 内部共享内存访问、cache coherence 流量、远端 L3/主存访问；WICP 的跨 chiplet 通信主要体现为 Worker 间 exchange、shuffle buffer 或本机网络栈通信。WICP 的收益条件是：减少的非结构化跨 chiplet 共享内存访问、局部 scan/hash/cache 行为改善、查询计划变化带来的算子收益，大于新增的 Worker 间通信开销。

### 多 Worker 对并行算子的影响

多 Worker 不会让所有算子天然更快。论文中 WICP 对 scan、repartition、hash join、Bloom filter 的收益有明确前提。

- **Scan/Filter**：每个 Worker 扫描自己的 fragment。若 fragment 与 Worker 的线程、内存页和 L3 拓扑对齐，scan/filter 能减少远端访问和 LLC miss。Presto 中涉及 Orders 表 ScanFilter 的查询收益明显。
- **Repartition/Shuffle**：该算子会增加 Worker 间数据移动，不是天然收益来源。WICP 的收益来自将 WIM 下无结构的跨 chiplet 共享内存访问转化为更规则的分区交换，并让交换前后的局部处理更快。
- **Hash Join**：多 Worker 可将 build/probe 分布化，每个 Worker 维护更小的局部 hash table，局部状态更容易落入 cache。若 join key 需要大量重分布，exchange 成本可能抵消收益。
- **Bloom Filter**：Bloom filter 是紧凑的 key 过滤结构，适合在 Worker 间传播，用较小结构提前过滤不可能匹配的行。SingleStore 部分查询在 WICP/WIN/WIC 下选择 Bloom filter + hash join，减少 nested loop join 的数据传输和执行时间。

因此，WICP 的有效性不是由 Worker 数量本身保证，而是由“Worker 边界承载局部状态”这一执行系统事实决定。Worker 切到 chiplet 粒度后，局部状态变小，生产者和消费者更局部，调度和数据分片更稳定地映射到同一组 core。

### WICP_Local 与 WICP_Mixed（Sec. 4.4）
论文根据工作集大小与 L3 容量的关系，将 WICP 进一步区分为：
- **WICP_Local**：worker 使用单个 chiplet 上的全部 core。工作集 < 单 chiplet L3（32 MB）时最优，数据访问主要落在本地 L3。
- **WICP_Mixed**：worker 使用相同数量的 core 但均匀分布在多个 chiplet 上。工作集在 32 MB 到 256 MB（聚合 L3）之间时更优，以跨 chiplet 的 cache 访问代价换取更大的可用总容量。

STREAM benchmark 在单 socket Milan 上验证了这一边界（Fig. 10）：数组 0.8 MB–1536 MB 范围内，< 32 MB 时 WICP_Local 带宽更高；超过 32 MB 后 WICP_Local 带宽下降，WICP_Mixed 因可利用 256 MB 聚合 L3 而更平稳；超过聚合 L3 后两者差异缩小。

WICP 的收益来自部署粒度、数据分布、cache locality、互联压力和查询计划变化的共同作用。WICP_Local 与 WICP_Mixed 的差异主要由工作集相对单 chiplet L3 与聚合 L3 的大小决定；当访问数据量超过聚合 L3 后，线程到 core 的放置影响下降。

### WIM / WIN / WICP 的深层机制对比

上面描述了每个策略解决了什么问题。本节从机制层面解释 WICP 为什么优于 WIN、为什么仅靠线程绑核无法在 WIM/WIN 下复现 WICP 的效果。

#### 一、三个层面的局部性：DRAM、L3 cache、线程-Split 映射

WIM、WIN、WICP 在局部性保证上有三个不同层级：

| | WIM | WIN | WICP |
|---|---|---|---|
| **DRAM 局部性** | 无保证（First-Touch 随机落点） | socket 级（membind） | chiplet 级（membind） |
| **L3 cache 局部性** | 无保证 | 无保证（同一 socket 内仍有多个 chiplet） | chiplet 级 |
| **线程-Split 映射** | 随机（N 线程抢同一 Split 队列，Split 位置与线程所在 chiplet 不对应） | 随机（同 socket 内 N 线程抢同一 Split 队列） | 良好（每 Worker 的线程只处理自己的 fragment，数据和线程在同一 chiplet） |

WIN 通过 `membind` 解决了跨 socket DRAM 访问，但它触及不到 socket 内部的 chiplet 差异。AMD Milan 的一个 socket 内含 8 个 chiplet（每 chiplet 8 core + 32MB L3），所有 chiplet 共享同一个 I/O Die 上的内存控制器。`membind` 保证了 DRAM 访问在 socket 本地，但无法保证线程在哪个 chiplet 上运行、数据在哪个 chiplet 的 L3 中。WIN 不能消除同一 NUMA 域内的跨 chiplet cache coherence 乱序访问问题。

#### 二、WIN 中的跨 chiplet cache coherence 污染

**一个常见但错误的直觉是**：只读共享数据会被缓存一致性协议复制到各 chiplet 的本地 L3（Shared 态），后续只读访问就是本地的，不应有跨 chiplet 开销。这忽略了 OLAP 查询的流水线写入模式。

Presto 的 join 是流水线执行的：

```
TableScan → Filter → HashBuilderOperator (写 hash table)
                       → LookupJoinOperator (读 hash table, probe)
```

同一 Task 的不同 Pipeline 在不同线程上交替执行。**Build 和 Probe 在同一 JVM 堆上交错进行**：

1. HashBuilderOperator 写 hash table —— 每一笔写入修改一个 cache line，所有其他 chiplet 上该 cache line 的 Shared 拷贝被 invalidate
2. LookupJoinOperator 读 hash table —— 被 invalidate 的 chiplet 需要跨 chiplet 重新加载该 cache line
3. 页面数据从 TableScan 出来，被 Filter 处理，再传给 HashBuilder——每一步的输出写入都会污染下游 chiplet 的缓存

更根本的是 **Page 流的 producer-consumer 链**：TableScan 线程（chiplet A）产生输出 Page → Filter 线程（chiplet B）消费并修改 → HashBuilder 线程（chiplet C）写入 hash table → Probe 线程（chiplet D）读取。在 WIN 下，每一步的写入都会 invalidate 所有其他 chiplet 上的缓存拷贝，下一步的读取又触发重新加载。这种无结构的跨 chiplet cache coherence 贯穿整个查询执行过程，没有批量，无法预取掩盖。

**WICP 通过 Worker 边界提供了明确的阶段隔离**：

```
Worker 0 (chiplet 0):
  [TableScan → Filter → HashBuild → Probe] 全部在 chiplet 0 的 core 和 L3 内完成
  ↑ 完全本地，无跨 chiplet cache coherence

Worker 1 (chiplet 1):
  [完全同理，在 chiplet 1 内完成]
```

在同一个 Worker 内部，所有线程绑定在同一个 chiplet 上，所有的读/写/L3 访问都在 24ns 内完成。如果需要跨 Worker 交换数据，数据经过 exchange 阶段被显式批量搬运一次到目标 chiplet（通过网络栈 DMA），搬运完成后下一阶段的全部访问又是本地的。

#### 三、隐式 cache coherence vs 显式 bulk transfer

WIN 和 WICP 中需要跨 chiplet 移动的数据量在物理上相同（都来自同一张表），但移动的方式决定了下游访问的代价：

**WIN（隐式 cache coherence）**：
- 粒度：64B cache line
- 频率：每次随机 probe 都可能触发
- 引擎感知：不感知（OS/硬件级）
- 传输后：数据不会被"搬"到本地 —— hash table probe 所在 chiplet 只是临时持有该 line 的 Shared 拷贝，下一次写入立即 invalidate

**WICP（显式 exchange）**：
- 粒度：MB 级 Page / buffer
- 频率：只在 exchange operator 边界触发一次
- 引擎感知：完全感知（Coordinator 的 exchange operator）
- 传输后：数据被永久搬到了目标 chiplet 的 L3/DRAM 中，后续所有访问都是纯本地的

以 TPC-H Q3 的 hash join 为例：`orders` 的 hash table 约 10GB，`lineitem` 的 probe 约 1500 万次：

- WIN 下，如果 hash table 由 chiplet A 创建但被 chiplet B/C/D 的线程 probe，即使只读部分能留在本地 L3，L3 容量（32MB）远小于 hash table（10GB），命中率极低。更致命的是 Build 阶段的写入持续 invalidate 其他 chiplet 的 Shared 拷贝，probe 阶段的随机访问模式下的 cache line 几乎不可能稳定命中。跨 chiplet cache coherence 的 106ns 延迟乘以百万次 probe 累计为秒级开销。

- WICP 下，exchange 阶段批量搬 3GB 数据到目标 chiplet（通过 Infinity Fabric DMA，耗时约 0.3-0.5s），搬完后 1500 万次 probe 全部在同 chiplet 的 24ns 内完成（约 0.36s）。总计约 0.7-0.9s vs WIN 的数秒。

**关键不是 106ns vs 24ns 的单一访问延迟对比，而是 (1500万 × 106ns) vs (一次 3GB 批量 DMA + 1500万 × 24ns) 的总开销对比。**

#### 四、为什么单 Worker 内通过线程绑核无法复现 WICP

一个自然的问题是：既然 `numactl` 可以绑线程到 core，为什么不在 1 个 Worker 内把线程 0-7 绑到 chiplet 0、线程 8-15 绑到 chiplet 1，从而在 WIM 下复现 WICP 的效果？

**原因一：Split 分配无 NUMA 感知**

Presto Worker 内部有一个统一的 Split 队列。线程池中的所有线程从同一个队列中抢 Split。即使你把线程 0-7 绑到了 chiplet 0，它们抓到的 Split 可能是 `orders` 文件在磁盘/内存中的任意一段 —— 这段数据的物理位置可能由 First-Touch 决定了在 NUMA node 1（socket 1）的 DRAM 上。Presto 的 Connector 生成的 Split 没有"所属 chiplet"标签，Split 调度器也不感知 NUMA 位置。

**原因二：JVM 堆的单一 NUMA 落点**

Presto 是 JVM 进程，JVM 堆是一个整块从 OS 申请的内存。Linux 的 First-Touch 策略意味着整个堆通常落在一个 NUMA node 的 DRAM 上。即使你通过线程绑核尝试分离计算位置，hash table、buffer、中间 Page 仍然全部在同一个 NUMA node 的 DRAM 上。不同 chiplet 上的线程访问这些数据，至少跨 chiplet DRAM 访问无法避免。多 Worker 方案通过独立的 JVM 进程和 `membind` 将每个 Worker 的内存强制放在对应的 NUMA node，解决了堆粘连问题。

**原因三：引擎层面的 Fragment 与执行计划差异**

Coordinator 看到 1 个 Worker 时，将全部数据作为 1 个 fragment 分配，执行计划中不包含 exchange operator。Join 使用完全本地的执行策略（如 broadcast join 或单节点 hash join），引擎层的逻辑不区分"这部分数据属于 chiplet 0、那部分属于 chiplet 1"。Coordinator 看到 N 个 Worker 时，将数据按 key hash 分为 N 个 fragment，在 join 前插入 exchange operator 重新分布数据，此后每个 Worker 全是本地操作。Worker 数与 fragment 数和执行计划的结构耦合，不可通过纯线程绑核模拟。

**简言之**：WICP 的收益不是来自"更好的线程绑定"，而是来自"将 Worker 边界对齐到 chiplet 边界 → 改变了引擎层面的数据分片、执行计划和内存策略"。改引擎源码可以在单 Worker 内近似 WICP 的效果，但论文的核心前提是"不改引擎源码"——在这个前提下，Worker 数是唯一能同时改变 fragment 粒度、执行计划和 OS 内存策略的杠杆。

#### 五、WICP 的收益 != "消除通信"

一件容易混淆的事：WICP 将"隐式跨 chiplet cache coherence"替换为"显式跨 Worker 网络通信"。物理上两者都经过 Infinity Fabric。WICP 的收益不是消除了物理通信，而是改变了通信的模型：

1. **隐含 → 显式**：cache coherence 是硬件隐式的，引擎不知；exchange 是引擎感知的，可以攒成批量 buffer 传输
2. **细粒度 → 粗粒度**：64B cache line vs MB 级 buffer —— 大批量 DMA 传输的效率远高于逐行 cache coherence 协议
3. **反复 → 一次性**：cache coherence 随着流水线读写交替反复触发；exchange 只在算子边界触发一次，之后全部本地
4. **污染 → 隔离**：WIN 中写入 invalidate 所有 chiplet 的拷贝，全流程被污染；WICP 中每阶段内完全隔离

WICP 不是"通信免费的"，它需要额外跨 Worker exchange。但在 chiplet CPU 上，非结构化的跨 chiplet cache coherence 带来的代价远大于结构化 exchange，因此 WICP 的净收益为正。

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

### 硬件微基准：chiplet 差异是否足以影响 OLAP（Fig. 3–5, Sec. 2.3–2.5）
实验目的：确认 chiplet CPU 是否真的存在足以影响查询执行的核间延迟、核间带宽和聚合内存带宽差异。

设置：论文在 AMD EPYC Milan、Intel Sapphire Rapids、ARM Graviton 3 和单芯片 Intel Xeon Gold 上运行 core-to-core latency、core-to-core bandwidth 与 STREAM 聚合带宽测试。Xeon Gold 作为单芯片对照。

现象：
- AMD Milan 差异最大。同 chiplet 核间延迟约 24 ns，同 socket 平均约 106 ns，最远路径可高出一个数量级；chiplet 内核间带宽约 12 GB/s，跨 chiplet 约 6 GB/s。
- Intel Sapphire Rapids 的核间延迟和带宽更均匀，平均延迟约 59 ns，核间带宽约 10.5–12.5 GB/s。
- ARM Graviton 3 的计算核心集中在一个 compute chiplet 上，标准 WIM/WIN/WICP 的拓扑差异有限。
- STREAM 中 `1 process per chiplet` 在 chiplet 平台上能提高聚合带宽；单芯片 Xeon Gold 不随部署策略变化。

解释：chiplet 平台的 L3 和互联不是均匀资源。线程和数据若跨 chiplet 随机分布，查询执行会暴露为高 LLC miss、远端访问、互联拥塞或带宽下降。该微基准给 WICP 的必要性提供前提：worker 部署粒度必须低于或等于 NUMA 内的 chiplet 差异。

### 部署策略整体对比（Fig. 7, Sec. 4.2）
实验目的：比较 WIM、WIN、WICP、WIC 在真实 OLAP 查询上的整体效果。

设置：TPC-H SF100，Presto、SingleStore、SparkSQL，三类 chiplet 平台。WIM 为默认 baseline，报告 WIN/WICP/WIC 相对 WIM 的 geometric mean speedup。

主要结果：
- AMD Milan：Presto 约 2.6x，SingleStore 约 3.4x，SparkSQL 约 3.4x。
- Intel Sapphire Rapids：Presto 约 3.6x，SingleStore 约 2.7x，SparkSQL 约 6.97x。
- ARM Graviton 3：WIM/WIN/WICP 的标准拓扑差异有限；modified WICP 后 SparkSQL 约 1.36x，Presto 约 1.2x。

解释：
- WICP 在 AMD 和 Intel 上稳定优于 WIN/WIM，说明仅按 NUMA 放置不够。即使 Sapphire Rapids 的核间通信更均匀，worker-to-topology 映射仍影响查询性能。
- SparkSQL 收益最高。论文将其归因于 WIM 下数据在线程间分配不均，导致 LLC miss rate 高、远端 NUMA read 高、interconnect congestion 明显；WICP 通过更均匀的数据分布和本地内存约束缓解该问题。
- WIC 不稳定。Presto 在 WIC 下退化，原因是 coordinator-worker 也被限制在单 core，aggregation 等协调阶段受限。该结果说明“更细粒度”不是单调收益。

机制含义：WICP 的提升来自局部状态和数据分布改善，不是来自更多 core。总 core 数相同的情况下，WICP 改变的是 Worker 边界、fragment 粒度和本地状态大小。

### 逐查询分析：算子层面的收益来源（Fig. 8, Sec. 4.2）
实验目的：解释哪些查询和算子贡献了 WICP 的加速。

设置：AMD EPYC Milan，TPC-H SF100，逐查询比较 Presto、SingleStore、SparkSQL 的 WIN/WICP/WIC speedup。

SingleStore：
- WICP 下部分查询生成了不同于 WIM 的查询计划。WIM 使用 nested loop join；WICP/WIN/WIC 使用 Bloom filter + hash join。
- Bloom filter 是紧凑的 key 过滤结构，可在 join 前丢弃不可能匹配的行。Hash join 先按 join key 构建 hash table，再用另一侧输入 probe。二者比 nested loop join 更容易分布式并行，也能减少后续 repartition 和数据传输。
- Q7、Q13、Q22 收益最大，分别达到 23.9x、18.1x、21.2x。Q13 的 Orders 表扫描从 5.36s 降至 0.4s，Repartition 从 5.06s 降至 0.25s。

Presto：
- 各策略使用相同查询计划，收益主要来自 fragment 数量和大小变化。Worker 数增加后，数据 fragment 更小，更容易进入 L3。
- ScanFilterTable 占主导的查询收益明显。Q5 和 Q18 分别有 88% 和 96% 的时间花在 Orders 表 ScanFilter，因此 WICP 提升更高。
- 涉及 Lineitem 的 Q1、Q6、Q14、Q19、Q20 低于 2x。论文解释为 Presto 对 Lineitem 的 scan/filter 处理不同，未出现与 Orders 相同的加速模式。

SparkSQL：
- WICP 下所有 TPC-H 查询均加速，范围约 3.51x–5.15x。
- SparkSQL 依赖 SortMerge Join。WIM 下数据跨 chiplet 不均，join 阶段产生较强互联拥塞；论文观测 interconnect congestion 平均占查询执行时间 34%。WICP 强制本地内存和更均衡的数据分布，缓解 join 阶段瓶颈。

机制含义：scan/filter 受益于 fragment 与 L3 的对齐；hash join 受益于更小局部 hash table；Bloom filter 受益于跨 Worker 传播小结构以减少大数据传输；repartition 本身有通信代价，但 WICP 可使通信前后的局部处理更快。

### 多核可扩展性（Fig. 9, Sec. 4.3）
实验目的：验证 WICP 是否只是单点加速，还是能改善随 core 数增加的扩展性。

设置：AMD Milan 双路 128 core，TPC-H，逐步增加 core 数，比较 WIM、WIN、WICP。结果归一化到 1 core。

现象：
- WICP 在 64 core 内接近线性扩展，128 core 时达到 71x（Presto）、55x（SingleStore）、31x（SparkSQL）。
- WIM 约扩展到 16 core 后平台化，对应约 2 个 chiplet 的核心数。
- WIN 在 64 core 前表现较好，但 64 core 后扩展性低于 WICP。

解释：WIM 的单 Worker 内部调度和数据分布无法有效覆盖更多 chiplet，core 数增加后跨 chiplet 访问和资源竞争抵消并行收益。WIN 能处理跨 NUMA 访问，但不能处理 NUMA 内 chiplet 差异。WICP 按 chiplet 拆分 Worker，让每个 Worker 维护较小局部状态，并减少跨 chiplet 共享状态访问，因此扩展性更好。

### L3 容量与 Local/Mixed 选择（Fig. 10, Fig. 11, Tab. 2, Sec. 4.4）
实验目的：验证“只用本地 chiplet L3”与“利用多个 chiplet 的聚合 L3”之间的容量权衡。

设置：单 socket AMD Milan。STREAM 中用 8 个 core 测试 0.8 MB–1536 MB 数组；TPC-H SF50 中比较 WICP_Local 和 WICP_Mixed。

STREAM 现象：
- 数据小于单 chiplet L3（32 MB）时，WICP_Local 带宽更高，因为访问主要落在本地 L3，避免跨 chiplet 通信。
- 数据超过 32 MB 后，WICP_Local 带宽下降，因为单 chiplet L3 容量不足，更多访问进入主存。
- WICP_Mixed 可利用多个 chiplet 的聚合 L3（Milan 单 socket 256 MB），在 32 MB–256 MB 区间更平稳。超过聚合 L3 后，两者差异缩小。

TPC-H 现象：
- SingleStore：WICP_Local 3.32x，WICP_Mixed 1.85x。其数据移动量较小，Local 的本地带宽优势更明显。
- Presto：WICP_Local 2.61x，WICP_Mixed 2.38x。差距较小。
- SparkSQL：WICP_Mixed 3.40x，WICP_Local 3.32x。SparkSQL 的数据移动更大，利用聚合 L3 的收益抵消部分跨 chiplet 代价。

Tab. 2 解释：SingleStore 的关键查询峰值内存多在单 chiplet L3 容量附近或以下；Presto 和 SparkSQL 的关键查询常超过单 chiplet L3，甚至超过聚合 L3，导致大部分时间进入主存路径，Local/Mixed 差异下降。

机制含义：WICP 不是固定选择本地化。工作集 < 单 chiplet L3 时应选 Local；单 chiplet L3 < 工作集 < 聚合 L3 时可选 Mixed；工作集超过聚合 L3 后，线程放置影响减弱。

### Worker 数量敏感性（Fig. 12, Sec. 4.5.1）
实验目的：验证最优 Worker 数是否等于 chiplet 数。

设置：双路 Intel Sapphire Rapids，共 112 core、8 个 chiplet/tile。改变 Worker 数，每个 Worker 分配 `112 / Worker数` 个 core；尽量将同一 Worker 的 core 映射到同一 chiplet 和 NUMA domain。

现象：
- Presto、SingleStore、SparkSQL 的 speedup 随 Worker 数增加到 8 而上升，分别达到 3.62x、2.75x、7.04x。
- 8 个 Worker 后收益平台化，28 个 Worker 后开始下降。
- 8 正好等于该机器 chiplet/tile 数。

解释：Worker 少于 chiplet 数时，每个 Worker 覆盖多个 chiplet，仍存在 WIM/WIN 式跨 chiplet 共享状态和数据分布问题。Worker 多于 chiplet 数时，每个 Worker 可用 core 和局部状态变小，exchange、调度和协调开销上升。chiplet 数量给出了该平台上的计算粒度与通信粒度平衡点。

### 查询多样性（Fig. 13, Sec. 4.5.2）
实验目的：确认 WICP 是否只对 TPC-H 的少数查询有效。

设置：AMD EPYC Milan，TPC-DS。正文描述对每条查询测量 WICP 和 WIN 相对 WIM 的 speedup。

现象：论文报告所有查询在 WICP 和 WIN 下均有 speedup。Q1、Q21、Q22、Q37、Q39、Q81、Q83、Q84、Q91 的收益最高。

解释：这些查询大部分执行时间花在大表扫描，例如 inventory 表。WIM 下这些扫描会触发昂贵的远端内存访问；WICP 将 scan 工作和局部状态限制在 chiplet 粒度后，减少跨 chiplet 访问。查询多样性没有消除 WICP 收益，说明该方法主要针对大表扫描和数据移动这类通用 OLAP 结构，而不是 TPC-H 特定查询。

### 数据 Skew 的影响（Fig. 14, Sec. 4.5.3）
实验目的：评估非均匀数据分布是否削弱 WICP。

设置：JCC-H 在 TPC-H 基础上加入 join-crossing correlations 和 data skew（数据倾斜）。论文使用 SingleStore 在 AMD EPYC Milan 上比较 WICP、WIN 相对 WIM 的 speedup。

现象：JCC-H 下 WICP geometric mean speedup 为 2.31x，WIN 为 1.72x；作为对照，TPC-H 下 WICP 为 3.40x，WIN 为 2.68x。WICP 仍优于 WIN，但相对 WIM 的收益低于无 skew 的 TPC-H。

解释：
- WIM baseline 变强。skew 让热点数据集中，重复访问热点时形成隐式局部性，WIM 不再完全表现为均匀大范围扫描。
- WIN 出现资源竞争。热点数据集中到特定 Worker、NUMA node 或内存路径，导致 resource contention。
- WICP 受负载不均和本地 cache 容量限制影响。热点分区可能集中到少数 chiplet，造成 workload imbalance；若热点或中间数据超过单 chiplet L3，访问会回退到主存路径。

机制含义：WICP 依赖数据分布相对均衡和工作集能有效利用本地 cache。skew 不会否定 WICP，但要求 skew-aware resource allocation（感知数据倾斜的资源分配）：根据数据分布动态调整每个 chiplet 的资源，重分布通信流量，并通过运行时监控观察 Worker/chiplet 的负载与性能指标。

## 分析
- chiplet CPU 上 worker 部署粒度比 NUMA 粒度更重要。WIN（NUMA 级）无法解决同 NUMA domain 内 chiplet 间的 cache 分片和通信差异（Fig. 9, Sec. 4.3），WICP 用更细的 chiplet 粒度覆盖了这一层。
- 部署策略的选择与工作集相对 L3 容量直接耦合：< 单 chiplet L3 → WICP_Local；介于单 chiplet L3 和聚合 L3 之间 → WICP_Mixed；> 聚合 L3 → 两者趋同（Sec. 4.4）。
- WICP 不需要修改引擎源码，仅靠外部 worker 绑定和数据放置即可实现收益（Sec. 4.1, Sec. 7）。

## 边界
- 研究对象是 OLAP 查询引擎（Presto、SingleStore、SparkSQL）的 worker 部署策略，不是 LLM 推理，不是 attention/kernel 层优化。
- WICP/Local/Mixed 针对分布式查询引擎的 worker-to-core 映射和数据 fragment 分配，不等价于 attention head、KV cache 或 tensor partition 的 chiplet 调度。
- 实验平台限于三款 chiplet CPU（Milan、Sapphire Rapids、Graviton 3），不覆盖 Genoa/Bergamo 或 heterogeneous chiplet 配置。

## 可迁移点
- "部署粒度不应停留在 NUMA 层级而应下到 chiplet 级" 这一原则对 chiplet CPU 上任何并行化部署策略都有参考价值（Sec. 3.2, Fig. 9）。
- 容量驱动的放置决策（工作集 vs 单 chiplet L3 vs 聚合 L3）可抽象为通用启发式（Sec. 4.4, Fig. 10）。
- WICP 通过 chiplet 粒度 worker 放置、局部 cache/内存使用和均衡数据分布降低互联压力，对需要减少跨 chiplet 数据移动的场景有参考意义。

## 不可直接迁移点
- OLAP 查询的 worker fragment 模型与 LLM attention 的 head 切分、sequence 切分、KV 共享在结构上不同，不能直接套用 WICP。
- 论文 speedup 数值（最高 7x、单查询 23.9x）依赖于 TPC-H 查询的特定表扫描/join/sort 操作，不能外推到 attention kernel 或模型推理延迟。
- ARM Graviton 3 的计算核心集中在单个 compute chiplet 上，标准 WICP 与 WIM/WIN 的硬件拓扑差异有限；论文使用的 modified WICP 不能直接外推到多 compute chiplet 场景。

## 证据
- 原始资料：`wiki/原始资料/papers/OLAP on Modern Chiplet-Based Processors.pdf`
- 关键锚点：摘要；Fig. 1–14；Tab. 1–2；Sec. 2.1–2.5；Sec. 3.1–3.2；Sec. 4.1–4.5；Sec. 5；Sec. 7
