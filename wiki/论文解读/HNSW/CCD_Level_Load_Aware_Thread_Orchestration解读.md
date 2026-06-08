# CCD-Level and Load-Aware Thread Orchestration for In-Memory Vector ANNS on Multi-Core CPUs 解读

**概述**：论文面向小红书生产环境中的内存型向量检索服务，提出一个 CCD 级、负载感知的线程编排框架。核心思想是让同一 HNSW 表或 IVF cluster 的热点工作集尽量驻留在同一个 CCD 的本地 L3 中，同时避免多个热点挤在同一个 CCD 的 L3 中相互驱逐。框架不做索引结构修改，而是在任务提交、任务到 CCD 的映射、动态重映射和 work stealing 四个环节感知 CCD 拓扑和负载。

## 论文基本信息

- 论文：`CCD-Level and Load-Aware Thread Orchestration for In-Memory Vector ANNS on Multi-Core CPUs`
- 作者：Yuchen Huang、Baiteng Ma、Yiping Sun、Yang Shi、Xiao Chen、Xiaocheng Zhong、Zhiyong Wang、Yao Hu、Chuliang Weng
- 机构：East China Normal University、Xiaohongshu Inc. (RedNote)
- 版本：arXiv:2605.10090v1，2026-05-11
- 研究对象：生产环境内存型 ANNS（Approximate Nearest Neighbor Search）服务中的 HNSW 和 IVF
- 原始资料：`wiki/原始资料/papers/HNSW/CCD-Level and Load-Aware Thread Orchestration.pdf`

## 背景

小红书将图像、文本、视频等非结构化数据嵌入为高维向量，通过近似最近邻检索返回相似对象。其搜索、推荐和广告业务部署了数千个百万级向量表，面向数亿月活用户，需在毫秒级延迟下支撑千万级 QPS（Sec. I）。

论文涉及的两种生产内存型索引：

**HNSW**（Hierarchical Navigable Small World）：图索引。构建时每个向量随机分配一个最大层级（由几何分布采样，参数 `m_L` 控制），该向量被插入到该层及以下所有层级（包括第 0 层）。每层中每个点连接最多 M 个近邻。高层稀疏，起"高速公路"作用；第 0 层稠密，包含全部向量。查询从最高层入口点出发，逐层贪心下降到第 0 层，然后在第 0 层以 `efSearch` 控制候选队列大小做 best-first search，返回最终 top-k。单次查询在单核上即可达到毫秒级延迟，但构建和增量更新成本高（几分钟到几小时），不适合高新鲜度场景（Fig. 2a；Sec. II-A）。

**IVF**（Inverted File）：聚类索引。用 k-means 将向量分到 `nlist` 个 cluster，每个 cluster 对应一个倒排列表。查询时先计算查询向量到所有聚类中心的距离，选最近的 `nprobe` 个 cluster，然后扫描这些 cluster 内的全部候选向量，排序后返回 top-k。IVF 构建快（秒到分钟级）、插入快（微秒级），适合高新鲜度业务，但单次查询需要扫描大量候选向量，必须用多线程并行扫描多个 list 才能满足毫秒级延迟（Fig. 2b；Sec. II-A）。

两类索引的并行模式不同：

- HNSW 单查询延迟已经很低，生产中把多个 HNSW 表共置在同一节点上，每个查询由一个 core 独立执行，通过多表、多请求并行提升吞吐。这是 **inter-query parallelism**（查询间并行）（Fig. 4a；Sec. III-A）。
- IVF 单查询的计算量大，生产中将一个查询拆成多个 per-list scan 任务，在多个 core 上并行扫描。这是 **intra-query parallelism**（查询内并行）（Fig. 4b；Sec. III-A）。

**CCD 架构背景**：现代 AMD EPYC 等服务器 CPU 通过 multi-chiplet 设计扩展核数。一个 CCD（Core Complex Die）封装多个 core 和一片本地 L3 cache。同一 CCD 内的 core 共享该 CCD 的本地 L3；不同 CCD 的 L3 不是全局共享的——一个 CCD 的 core 访问另一个 CCD 的 L3 延迟更高，需要经过 Infinity Fabric（AMD 的片间互联总线，跨 CCD 数据传输需经此总线，延迟显著高于 CCD 内 L3 命中）。这与传统 monolithic 多核 CPU 的单一全局 L3 有根本区别（Fig. 3；Sec. II-C）。论文实验用的两代 AMD EPYC 平台每 core 提供 4 MB L3，按 CCD 聚合为 32 MB/CCD（Genoa, 8 core/CCD）或 16 MB/CCD（Rome, 4 core/CCD）（Table I；Sec. III-D）。

## 研究问题
论文研究的问题不是修改 HNSW 或 IVF 的索引结构与搜索算法，而是：在多 CCD CPU 上，如何将向量检索任务调度到合适的 core/CCD，使生产 workload 中的热点工作集稳定驻留在某个 CCD 的本地 L3 中，同时保持各 CCD 负载均衡。

作者观察到三个关键现象：

1. 短时间窗口内，相同向量表的查询会重复访问重叠的 HNSW 节点子集或 IVF cluster 子集，存在可利用的时间局部性（Sec. I；Fig. 6a-b）。
2. CCD 之间 L3 独立。不合适的 CCD-请求分配会导致 chiplet 级 cache pollution——多个热点工作集被路由到同一 CCD，在有限的本地 L3 中相互驱逐，破坏缓存亲和性（Sec. I；Sec. VI-A）。
3. 不同 HNSW 表或 IVF cluster 的单次搜索开销差异大。普通 work stealing 可缓解负载不均，但跨 CCD stealing 会让任务失去原 CCD 已缓存的 L3 数据，降低缓存复用率（Sec. I；Fig. 8；Fig. 13）。

论文的目标是在不重写索引代码的前提下，统一支持 HNSW 的 inter-query 并行和 IVF 的 intra-query 并行，并把 CCD 本地 L3 作为调度器显式管理的关键资源（Sec. IV）。

## 动机观察

### 核数扩展效率不足

论文在 AMD 4th Gen EPYC 96-core（12 CCD）平台上，用生产 trace 测试现有调度方式的扩展性（Sec. III-B）：

- **HNSW**：60 个 HNSW 表共置，每表 1M-10M 向量。按 CCD 粒度从 16 core 扩展到 96 core（每次启用一个完整 CCD），朴素 round-robin inter-query dispatch 获得约 2× 到 9.9× 的 speedup。但 96-core 配置只达到理想吞吐的约 82%（理想吞吐 = CCD 数 × 单个 CCD 的吞吐），存在明显 gap（Fig. 5a）。
- **IVF**：15 个 IVF 表，FAISS 风格的 OpenMP intra-query dispatch（`#pragma omp for schedule(dynamic)`，线程从共享任务池拉取 per-list scan）。从 2 个 CCD 扩到 12 个 CCD，speedup 只有约 1.6× 到 2.8×；32 core 以后增益趋于平坦（Fig. 5b）。

结论：增加 core 数量没有自动转化为线性吞吐提升。瓶颈与 CCD 间独立的 L3 结构、跨 CCD 访问导致的工作集驱逐、以及调度不感知负载倾斜有关。

### 访问热点与流量倾斜

论文用约一分钟短窗口的生产日志分析访问模式（Sec. III-C）：

- **访问频次集中**：对 3 个 HNSW 表和 3 个 IVF 表统计被访问节点/cluster 的累积分布，CDF 在低访问频次区域快速攀升——少量节点/cluster 承担大部分访问，呈 heavy-tailed / Zipf-like 分布（Fig. 6a-b）。
- **Memory traffic 倾斜**：对一个共置约 60 个 HNSW 表的节点统计各表 memory traffic（总读写字节量），以及对一个 2M 向量、8192 cluster 的 IVF 表统计各 cluster memory traffic。两项都存在极强偏斜：单项流量从几十 MB 到数千 MB，相差一到数个数量级（Fig. 6c-d）。
- **热点动态变化**：Fig. 7 用固定采样窗口展示 15 个向量表的归一化 memory traffic 随时间的波动。热点和冷点的角色会随请求量动态转换，这意味着调度映射需要随时间自适应调整。
- **单任务开销偏斜**：单核测得的每 HNSW 表或每 IVF cluster 平均处理时间呈长尾分布。这意味着即使请求数均匀分发（如 round-robin），实际执行负载也会不均衡（Fig. 8）。

### L3 机会与面临的工程挑战

两代 AMD CCD CPU 每 core 对应 4 MB L3，CCD 级聚合后为 32 MB（Genoa）或 16 MB（Rome）本地 L3（Table I；Sec. III-D）。若短窗口内重复访问的 HNSW 节点或 IVF list 能驻留在某个 CCD 的本地 L3 中，查询可更多命中 L3 而非 DRAM，降低延迟和内存带宽压力。

但要在工业环境中将局部性转化为实际性能提升，需要同时解决三个工程问题：**兼容性**（drop-in 框架，不改索引代码）、**CCD 友好的任务分发与动态适配**、**感知硬件拓扑的负载均衡**（Sec. III-D）。

## 系统设计

论文框架位于向量索引和 CCD 处理器之间，包含三个协作组件（Fig. 9）：

1. **CCD-Level Task Submission**：统一任务提交接口。`submit(search_functor, query, Mapping_ID)` 将 HNSW 表搜索和 IVF list scan 抽象为同一种 task。
2. **Task Dispatcher & Workload Monitor**：维护 `Mapping ID → CCD` 的映射表，通过短窗口 memory traffic 统计在线估计各 item 的冷热程度，周期性调用 Algorithm 1 更新映射。
3. **Thread Orchestration with CCD-Affinity Task Stealing**：线程一对一 pin 到 core，每个 core 维护环形任务队列。stealing 按本 core → 同 CCD → 跨 CCD 三级优先级执行。

### 统一任务提交接口

接口形式：

```text
submit(search_functor, query, Mapping_ID)
```

- `search_functor`：可插拔的搜索函数。对 HNSW 是整表搜索，对 IVF 是单个 cluster list 的 flat scan。对运行时透明。
- `query`：封装原始查询向量、top-k、过滤条件和客户端信息。
- `Mapping_ID`：调度标识。HNSW 中为 table id；IVF 中为 table-cluster id（即某表下的某个 cluster）。

调度器调用 `pickCcd(id)` 查映射表决定目标 CCD，并调用 `adaCcd(fn, id)` 将任务包装为：执行搜索 → 完成后向 workload monitor 报告本次搜索的统计量（HNSW 的 touched nodes 数、IVF 的 scanned vectors 数），用于后续动态更新映射（Fig. 10；Sec. V-A）。

### HNSW 与 IVF 集成

**HNSW inter-query 集成**：每个 HNSW 请求对应一个 task，`Mapping_ID` 为 table id。调度器按 table id 将任务路由到某个 CCD 的 core 队列。搜索函数在该 CCD 的一个 core 上完成整次 HNSW 查询并直接返回最终 top-k 给客户端。Inter-query 并行来自多个表的大量任务在不同 core 上同时执行（Sec. V-B）。

**IVF intra-query 集成**：一个 IVF 查询先找到 `nprobe` 个最近 cluster，然后拆成多个 per-list scan task。每个 task 共享同一个 query，但携带不同的 table-cluster id 作为 `Mapping_ID`。每个 list scan 返回该 cluster 内的局部 top-k，最后做 k-way merge 得到全局 top-k 返回客户端。关键点是 per-list scan 和 HNSW 表搜索共用同一套 `submit()` 抽象，CCD 选择和监控适配都委托给调度器（Sec. V-B）。

### CCD 映射策略

调度器维护 `Mapping_ID → CCD` 的映射表，`pickCcd(id)` 查表决定任务去向。映射的目标有两层优先级（Sec. VI-A）：

1. **Sticky mapping（粘性映射）**：同一 HNSW 表或 IVF cluster 的后续任务尽量路由到同一个 CCD，使该 CCD 的 L3 保留相关热点工作集。这利用了短窗口内对相同数据对象的重复访问。
2. **Avoid hot-hot co-location**：避免多个热点表/cluster 被映射到同一 CCD。两个热点共享一个 CCD 的 L3 会相互驱逐对方的工作集（cache pollution）；而热项与冷项配对时，冷项请求少、L3 需求小，热项工作集更容易驻留（Fig. 11）。

### Memory Traffic 在线估计

HNSW 和 IVF 本身没有直接暴露"本次查询消耗多少内存流量"的接口。论文用低开销近似估计，公式基于搜索过程中实际访问的节点/向量数和已知数据结构大小（Sec. VI-B）。

设向量维度 `D`，每元素字节数 `s_v`（FP32 为 4），向量 payload 大小 `B_v = D * s_v`，ID 宽度 `s_id`（通常 UINT32 为 4 字节）。

**HNSW**：一次查询访问 `N` 个节点（从搜索过程的 visited list 获得精确计数），每个节点需读取向量（`B_v` 字节）和邻接表（最大出度 `M`，每条边一个 `s_id`）：

```text
T_HNSW ≈ N * (B_v + M * s_id) + δ_meta
```

`δ_meta` 是 top-k heap 和 visited list 的元数据读写。论文称其占比通常 < 1%，可忽略（Eq. 1）。

**IVF**：对 list `L_i`，若扫描 `S_i = |L_i|` 个向量（即该 list 的完整长度），仅计算向量读取量：

```text
T_IVF(L_i) ≈ S_i * B_v
```

IVF 不涉及邻接表遍历，因此估计比 HNSW 更简单（Eq. 2）。

这些估计使 workload monitor 能为每个 HNSW table 或 IVF cluster 维护短窗口内的累计流量，作为动态映射的输入。

### Hot-Cold 配对映射算法

Algorithm 1 的目标是：给定 `n` 个 item（HNSW 表或 IVF cluster）的估计流量 `T[1..n]` 和 `m` 个 CCD，输出一个映射，使得每个 CCD 的总流量接近均值 `μ = ΣT/m`，同时将 hot item 与 cold item 配对放置。

流程：

1. 按流量从高到低排序 item，`A[1..n]` 为排序后数组。维护双指针 `i=1`（当前最热）、`j=n`（当前最冷）。
2. 维护每个 CCD 的当前负载 `L[1..m]`，初始为 0。
3. 每轮：选择当前最轻载 CCD `r*`，取当前最热 item `A[i]`（`i` 前进），计算该 CCD 距目标 `μ` 的剩余容量 `cap = max{0, μ - L[r*] - T[A[i]]}`。
4. 若剩余容量能容纳当前最冷 item `A[j]`，则将该冷项与热项一起放入 `r*`，`j` 后退；否则仅放入热项。
5. 重复直到所有 item 完成映射。

该算法是贪心启发式，不保证全局最优，但计算开销低，适合在线重映射（Algorithm 1）。

### Snapshot 动态切换

热点分布随时间变化，映射需要动态更新。论文使用窗口化重映射 + 快照切换（snapshot swap）机制（Fig. 12；Sec. VI-B）：

1. 每个适配窗口（adaptation interval）聚合所有 HNSW table 或 IVF cluster 的估计流量，作为 Algorithm 1 的输入。window 长度需覆盖足够多样本使流量估计稳定，同时足够短以响应热点变化。论文在压力实验中设置 window 为 10 秒（Fig. 20）。
2. Workload monitor 在后台根据窗口统计调用 Algorithm 1 构造 `next-map`。
3. Dispatcher 当前使用 `current-map` 处理到达的请求。
4. `next-map` 就绪后原子替换 `current-map`。**新提交的任务**使用新 snapshot；**已在执行的任务**继续用旧 snapshot 直到完成。

该设计的要点：映射切换不中断在途任务，不引入锁争用——任务在提交时绑定某个 snapshot，执行期间不受重映射影响。10 秒 window 下重映射开销对服务延迟无明显扰动（Fig. 20）。

### CCD-Aware Work Stealing

Work stealing 的经典目标是避免 core 空闲：空闲 core 从其他 core 队列中窃取任务。但在 CCD 架构中，跨 CCD stealing 意味着任务迁移到另一个 L3 域——原 CCD L3 中已缓存的工作集在新 CCD 中不可用，需要重新从 DRAM 加载。相反，同 CCD stealing 时任务仍在同一个 L3 域内执行，缓存的工作集可以继续复用（Fig. 13；Sec. VII-A）。

Algorithm 2 将 stealing 分为三级优先级（Sec. VII-B）：

1. 从本 core 本地队列取任务（`PopLocal`）。
2. 本地队列空时，从同 CCD 的其他 core 集合 `S_in(i)` 窃取。
3. 同 CCD stealing 也失败后，才从其他 CCD 的 core 集合 `S_cross(i)` 窃取。

框架初始化时读取 CPU 拓扑（通过读取 `/sys/devices/system/cpu/` 等），为每个 core `i` 构造同 CCD core 集合 `S_in(i)` 和跨 CCD core 集合 `S_cross(i)`。线程一对一 pin 到 core，每个 core 维护一个环形任务队列。`TrySteal` 在目标集合内随机选择 victim（Algorithm 2）。

关键设计意图：不是取消 work stealing，而是把 stealing 限制在更可能保留本地 L3 工作集的同 CCD 范围内；跨 CCD stealing 作为保底手段，防止整 CCD 空闲和长期不均衡。

## 实验设置

### 硬件与软件

论文使用两代 AMD CCD 架构 CPU（Table I；Sec. VIII-A）：

| 平台 | Core 数 | CCD 数 | Core/CCD | 内存 | L3/CCD | L3/core | 主频 |
|---|---|---|---|---|---|---|---|
| AMD 4th Gen EPYC 9654 (Genoa) | 96 | 12 | 8 | 576 GB DDR5 | 32 MB | 4 MB | 3.5 GHz |
| AMD 2nd Gen EPYC 7K62 (Rome) | 48 | 12 | 4 | 512 GB DDR4 | 16 MB | 4 MB | 2.6 GHz |

系统为 CentOS 8 + Linux kernel 5.10。实现为 C++，GCC 11.4，`-O3`。HNSW 和 IVF 参考 hnswlib 与 FAISS 实现，距离计算使用 AVX 指令加速（Sec. VIII-A）。

### Workload

所有数据集和查询来自生产推荐和广告服务（Sec. VIII-A）：

- HNSW inter-query：60 个表，每表 1M-10M 行。
- IVF intra-query：15 个表，每表 10K-15M 行。
- 向量维度：64-256。
- top-k：100-500。
- 距离度量：L2。
- HNSW 参数：`M=32`，`efConstruction=500`，`efSearch` 调至 recall 达 99%。
- IVF 参数：`nlist=128-8192`（按表规模调整，使构建时间在分钟级）；`nprobe` 调至 recall 达 95%。
- 查询日志：约 100 万 HNSW 请求，约 110 万 IVF 请求。

### Baseline

论文比较三种线程编排方案（Sec. VIII-A）：

| 名称 | HNSW 实现 | IVF 实现 | 说明 |
|---|---|---|---|
| V0 (RR) | 朴素 round-robin 分发 | FAISS 默认 OpenMP intra-query (`schedule(dynamic)`) | 不感知 CCD，无 work stealing |
| V1 (bthread) | 基于 bthread（百度 M:N 用户态线程库）的 work stealing | 同左 | 有 work stealing 但不感知 CCD 拓扑，已在生产部署 |
| V2 (CCD) | 论文提出的 CCD 级负载感知框架 | 同左 | CCD 映射 + 动态重映射 + CCD-aware stealing |

## 结果与解释

### 饱和吞吐

实验按 CCD 粒度增加可用 core（每次启用一个完整 CCD 的所有 core），测量各方案的饱和吞吐（Sec. VIII-B）。

**HNSW**（Fig. 14）：V2 在两代 CPU 上都优于 V1 和 V0，且随 CCD 数增加更接近线性扩展。96-core Genoa 上 V2 超过 100 KQPS；48-core Rome 上 V2 约 50 KQPS。V2 在全部 CCD 配置上都保持对 V1/V0 的明显优势。

**IVF**（Fig. 15）：V2 同样扩展最好。96-core Genoa 上 V2 达 35 KQPS，V1 为 25 KQPS，V0 (FAISS) 为 10 KQPS。普通 OpenMP 并行在 CCD 数增多时增益衰减严重——仅靠多线程平分 list scan 工作，不感知 CCD L3 边界，导致大量跨 CCD 的 DRAM 访问。V2 通过 CCD 亲和映射和同 CCD 优先 stealing 同时提升了单 CCD 利用率和多 CCD 扩展效率。

### P50 与 P999 延迟

96-core Genoa 平台上测量 P50 和 P999 搜索延迟。HNSW 展示 60 个表中的 15 个（按 V0 性能排序后每 4 个取 1 个）；IVF 展示全部 15 个表（Sec. VIII-B）。

V2 在 HNSW 和 IVF 上都获得最低 P50 和 P999 延迟（Fig. 16-17）。V1 相比 V0 的收益主要来自 work stealing 缓解排队热点；V2 在此基础上通过两个机制进一步降低延迟：
- **Cache-friendly dispatch**：同一表/cluster 的任务集中调度到一个 CCD，使更多查询处理发生在 L3 命中路径而非 DRAM 访问路径。
- **减少跨 CCD stealing**：任务即使在 stealing 时也优先留在同 CCD，避免缓存工作集被丢弃。

### L3 Cache Miss Rate

使用 AMD uProf 在 Genoa 96-core、12-CCD 平台上采集端到端 search workload 的 L3 cache miss rate（Sec. VIII-C）。

**HNSW**（Fig. 18a）：V2 在全部 12 个 CCD 上的 L3 miss rate 均低于 V1 和 V0。V1 与 V0 的 miss rate 接近——仅引入不感知 CCD 的 work stealing 并不能改善本地 L3 复用。

**IVF**（Fig. 18b）：V1 相比 FAISS OpenMP baseline 已有小幅改善；V2 依靠 hot-cold colocation 和 CCD 优先 stealing 取得最低 miss rate。

该指标的物理含义：L3 miss 意味着 core 需要访问 DRAM（经 Infinity Fabric 跨 CCD 访问或本地内存控制器），延迟远高于 L3 命中（~10 ns vs ~100 ns 量级）。V2 降低 miss rate 直接减少了长延迟内存访问。

### CPU Stall

CPU stall 指 core 不能退休指令的周期，原因包括等待长延迟数据访问、指令取数阻塞、线程切换等。stall 越低，CPU 有效利用率越高（Sec. VIII-C）。

**HNSW**（Fig. 19a）：V0 与 V1 stall 接近（~相同量级），V2 明显更低。论文归因于 cache affinity 提升后，长延迟内存访问减少。
**IVF**（Fig. 19a）：V1 通过 stealing 已降低 stall（相对 FAISS baseline），V2 在此基础上继续降低。

### Cross-CCD Stealing 比例

V2 明确压低了跨 CCD stealing 的比例（Fig. 19b；Sec. VIII-C）：

| 场景 | V1 cross-CCD stealing | V2 cross-CCD stealing |
|---|---|---|
| HNSW | ~75% | < 10% |
| IVF | > 80% | ~5% |

在没有拓扑感知的 V1 中，约 3/4 到 4/5 的 stealing 都是跨 CCD 的——每次跨 CCD stealing 都意味着任务丢失了原 CCD L3 中已预热的工作集。V2 通过三级优先级将绝大多数 stealing 限制在同 CCD 内，仅在整 CCD 空闲等极端情况下才触发跨 CCD stealing。

### 生产式压力受限运行

在 production-style pressure-limited 设定下比较 V1 和 V2，连续运行 1000 秒，每秒聚合当秒到达请求并记录平均响应延迟（Sec. VIII-D）。

HNSW 和 IVF 上 V2 都比 V1 有更低平均延迟和更小波动。HNSW 中 V1 出现间歇性延迟 spike，V2 稳定；IVF 中二者都较稳定，但 V2 全线低于 V1。该实验中动态重映射窗口设为 10 秒，重映射未对服务延迟造成可见影响（Fig. 20）。

## 核心贡献

论文的核心贡献是把"任务访问的数据属于哪个热点"与"该热点应驻留在哪个 CCD L3"建立显式关联。传统线程池只关心 core 空闲与否，V2 额外追踪三个维度：

1. **Mapping_ID**：标识任务访问哪个 HNSW 表或 IVF cluster。
2. **短窗口 memory traffic 估计**：刻画每个 Mapping_ID 当前的冷热程度，驱动映射决策。
3. **CCD 拓扑**：记录哪些 core 共享同一片本地 L3。

三者结合产生三个效果：

- **Sticky mapping**：相同 Mapping_ID 的任务重复发到同一 CCD → 该 CCD L3 逐步积累对应的工作集 → L3 命中率提升。
- **Hot-cold 配对**：避免两个热项共享 CCD L3 → 每个 CCD 最多承载一个热项的工作集，冷项占用 L3 极少 → 热项工作集更完整驻留。
- **同 CCD 优先 stealing**：需要 stealing 解决负载不均时，优先在同 CCD 内偷 → 被偷任务的工作集仍在同一 L3 → 缓存不浪费。

论文的收益不来自减少 HNSW 访问节点数或 IVF 扫描向量数，而是将相同的搜索工作量放在更合适的 CPU 拓扑位置执行，提升本地 L3 命中率、降低 stall 和尾延迟。

## 边界与注意事项

1. **数据集不公开**：workload 来自小红书生产服务（搜索、推荐、广告），表规模（1M-15M 行）、维度（64-256）和查询量（百万级）给出了范围，但外部无法完全复现实验（Sec. VIII-A）。
2. **细粒度分析不足**：论文报告了端到端 L3 miss rate、CPU stall 和 cross-CCD stealing ratio，但未提供更细粒度的 AMD IBS（Instruction-Based Sampling）load latency source、DRAM row buffer hit/miss、Infinity Fabric 链路流量等分析。无法区分本地 DRAM 访问和跨 CCD 远程访问的各自贡献（Fig. 18-19）。
3. **热点稳定性假设**：V2 依赖短窗口内存在稳定的热点。若查询分布接近均匀，或热点变化快于重映射窗口，sticky mapping 和 hot-cold 配对的收益会下降。论文用 Fig. 7 和 Fig. 20 说明其 workload 下动态重映射有效，但不能外推到所有分布。
4. **优化范围**：V2 优化的是线程编排，不改变 HNSW 图布局、节点重排、向量压缩或距离计算算法。与 graph reordering、PQ/SQ、prefetch 等技术是正交关系，可叠加使用。
5. **映射算法的最优性**：Algorithm 1 是贪心启发式，不保证全局最优映射。它优先满足低开销和在线适配要求，而非映射质量的理论最优。
6. **并行粒度假定**：论文对 HNSW 假定单查询单 core 的 inter-query 并行，对 IVF 假定 per-list scan 的 intra-query 并行。若实际部署使用不同的并行粒度（如单 HNSW 查询跨多 core 并行），适用粒度需要重新定义。
7. **硬件差异**：论文实验硬件为 AMD 4th Gen EPYC 9654（8 core/CCD, 32 MB L3/CCD）和 2nd Gen EPYC 7K62（4 core/CCD, 16 MB L3/CCD）。在不同 CCD 拓扑（如 Turin 的 16 core/CCD）或双路 NUMA 条件下，收益幅度不能直接外推。

## 从本文可迁移的设计经验

### 可迁移的设计要素

- **Baseline 选择**：CCD-aware 调度的实验至少应对比三类 baseline：round-robin、普通 work stealing、CCD-aware/load-aware 调度。仅对比 round-robin 不足以分离 CCD 机制的独立价值。
- **`Mapping_ID` 抽象**：可迁移到其他粒度的 chiplet-aware 调度实验——HNSW 表、查询簇、入口点区域、graph partition 或热点节点集合都可作为候选 mapping 粒度。
- **实验指标体系**：应覆盖 P50/P99/P999 延迟、饱和吞吐、L3 miss rate（按 CCD 分解）、CPU stall、跨 CCD stealing/迁移比例。建议补充 AMD IBS load latency 和 L3 miss source（local vs. remote），用于区分本地 L3 容量收益和跨 CCD 远程访问收益。
- **Workload characterization 前置**：优化前应证明短窗口内访问热点存在、memory traffic 倾斜存在、单任务成本偏斜存在，否则 CCD-aware 调度缺乏触发条件。论文 Fig. 6-8 提供了可参考的证明方式。

### 不能直接迁移的假设

- 论文的 HNSW 场景是多表共置、每查询单 core 执行。若当前研究的是单表内相似 query 的 CCD 内并行，需要重新定义热点和复用粒度。
- 论文未展示节点级 cache-line 重叠或同一 cache line 被多 core 同时请求的证据，不能直接支撑"相似 query 在 CCD 内共享大量 cache line"的更细粒度结论。
- 论文未对比 graph reordering、prefetch 或向量压缩带来的通用 cache locality 收益。若提出 chiplet-aware HNSW，需将这些通用优化作为 baseline 或控制变量。
- 论文硬件是 Genoa（8 core/CCD）和 Rome（4 core/CCD）。若当前平台为 Turin（16 core/CCD, 每 CCD 约 32 MB L3），CCD 内 core 数更多、L3/core 比例不同、双路 NUMA 拓扑不同，收益幅度需要独立评估。
