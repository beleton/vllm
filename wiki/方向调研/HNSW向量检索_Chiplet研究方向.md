# HNSW 向量检索的 Chiplet 研究方向

## 结论

`HNSW`（Hierarchical Navigable Small World，层次化可导航小世界图）是当前最适合作为 chiplet 后续主线的场景。它不是矩阵计算，而是在内存中的近邻图上做随机遍历。每个 query 会访问一串图节点、邻接表和向量数据，访问路径由 query 内容决定，天然存在热节点、路径重叠、冷图随机访问和布局可控性。

推荐方向是：

```text
per-CCD navigator replica
  + trace-guided cold graph partition/layout
  + optional remote expansion
  + hot-path query grouping
```

四项含义：

- `per-CCD navigator replica`：每个 `CCD` 保存一份小型“导航图”副本，让 query 的开头搜索阶段尽量命中本地 L3。
- `trace-guided cold graph partition/layout`：根据真实 query 访问轨迹重排第 0 层大图，把经常连续访问的节点放到同一 `CCD` 或相邻内存页。
- `optional remote expansion`：跨分区访问很重时，把 query state 发给远端 owner worker 扩展，而不是让本地线程反复拉远端图节点 cache line。
- `hot-path query grouping`：把访问路径相近的 query 短窗口聚合到同一 `CCD` 执行，放大热节点和邻接表复用。

其中前两项最适合作为论文核心，后两项作为扩展和 tail latency 优化。该方向不依赖 TiNA 的“远端 L3 快于 DRAM”假设，只利用 AMD Zen 上“本地 CCD L3 明显快于 DRAM”的事实。

## HNSW 基础

### 向量检索问题

向量检索的输入是一批数据向量和一个 query 向量。系统需要返回距离 query 最近的 top-k 数据向量。距离可以是 L2 distance、inner product 或 cosine similarity。

精确检索会扫描所有向量：

```text
for each vector x in database:
  compute distance(query, x)
return top-k smallest distance
```

这种方法简单，但数据量大时会变成全表扫描，瓶颈通常是 SIMD 计算和 DRAM bandwidth。`HNSW` 属于 `ANN`（Approximate Nearest Neighbor，近似最近邻）索引。它牺牲少量 recall，用少得多的节点访问换取更低延迟。

### HNSW 的核心数据结构

HNSW 把向量组织成多层图：

```text
Layer 3:      A
              |
Layer 2:   A ---- B
            \    /
Layer 1:  A -- C -- D
          |    |    |
Layer 0:  大量节点，每个节点连接若干近邻
```

每个图节点对应一个数据向量。每个节点保存：

- `vector/code`：原始向量或压缩向量码，用于计算距离。
- `neighbor list`：该节点在每一层的邻居 ID 列表。
- `level`：该节点出现到最高哪一层。
- `metadata`：节点 ID、偏移、删除标记等索引元数据。

HNSW 的层次结构有两个特点：

- 上层节点少，边较长，用于快速跳到 query 附近区域。
- 第 0 层节点最多，保存主要近邻关系，用于精细搜索。

可以把 HNSW 理解为“多层道路系统”：上层像高速路，用少量节点快速接近目标区域；第 0 层像街区道路，在局部范围内仔细找最近邻。

### 查询流程

HNSW 查询包含两个阶段：上层贪婪导航和第 0 层 best-first 搜索。

#### 阶段一：上层贪婪导航

查询从全局 `entry point` 开始。`entry point` 是索引中一个高层节点。算法从最高层向下走，每层只保留当前找到的最近节点。

单层贪婪过程：

```text
current = entry_point
repeat:
  read current.neighbor_list at this layer
  compute distance(query, each neighbor)
  if some neighbor is closer than current:
      current = closest neighbor
  else:
      stop this layer
go to next lower layer using current as entry
```

该阶段的目标不是直接返回答案，而是找到第 0 层搜索的起点。上层节点数量少，很多 query 会反复访问同一个 entry point、上层 hub 和相近路径，因此上层结构是天然热集。

#### 阶段二：第 0 层 best-first 搜索

到第 0 层后，HNSW 不再只保留一个当前节点，而是维护两个集合：

- `candidate heap`：待扩展候选节点，优先扩展当前看起来离 query 最近的节点。
- `result heap`：当前找到的最好结果集合，大小由 `efSearch` 控制。

第 0 层搜索过程：

```text
candidate_heap = {entry_from_upper_layer}
result_heap = {entry_from_upper_layer}

while candidate_heap is not empty:
  u = pop closest candidate
  if u is already worse than the worst node in result_heap:
      break

  read u.neighbor_list at layer 0
  for each neighbor v:
      if v not visited:
          read v.vector/code
          d = distance(query, v)
          update candidate_heap and result_heap

return top-k from result_heap
```

`efSearch` 控制搜索宽度。`efSearch` 越大，访问节点越多，recall 通常越高，延迟也越高。HNSW 的优化必须保持 `recall@k`，不能只减少访问节点数。

### 一次节点扩展的内存访问

第 0 层扩展一个节点 `u` 时，CPU 通常要访问：

```text
u metadata
u layer-0 neighbor list
visited bitmap/hash
for each neighbor v:
  v metadata
  v vector/code
  candidate/result heap
```

这些访问具有不规则性：

- `neighbor list` 按图边跳转，不是连续数组扫描。
- `vector/code` 的位置由邻居 ID 决定，query 之间路径不同。
- `visited` 结构和 heap 会产生额外随机访问。
- 高维向量会增加距离计算占比，低维或压缩向量更容易暴露内存瓶颈。

对 chiplet CPU 而言，关键问题不是“HNSW 有图，所以一定有收益”，而是这些节点扩展是否频繁访问本地 `CCD` 外的数据，并且能否通过复制、布局、路由或分组把访问留在本地 L3/本地内存路径。

## CPU 使用形态

HNSW 在 CPU 上运行是常见做法，不是为了 chiplet 课题临时构造的 workload。

- hnswlib 是 C++ 实现并提供 Python binding，默认使用 CPU 内存中的 HNSW 图索引。
- Faiss 提供 `IndexHNSWFlat` 等 HNSW 索引；Faiss 的 GPU 支持并不覆盖所有索引类型，CPU 索引仍是常用路径。
- Milvus、Qdrant、Weaviate 等向量数据库长期把 HNSW 作为内存 ANN 索引使用，常见部署形态是多核 CPU 承载在线检索。
- RAG serving 中，向量检索服务经常与 LLM 推理解耦。LLM 可运行在 GPU 或 CPU，HNSW 检索服务仍可独立运行在 CPU 内存索引上。

该结论只说明 HNSW-on-CPU 是真实系统路径，不说明所有向量检索都应在 CPU 上完成。GPU brute-force、GPU IVF/PQ、CPU/GPU cooperative ANNS、DiskANN 等方案在不同规模和成本约束下同样常见。本文选择 HNSW 的原因是其内存图遍历、热导航层、query-dependent path 和布局可控性与 Zen chiplet 的本地 L3 机制更匹配。

## Chiplet 可优化性

### 与 attention 的差异

CPU attention 尝试中，`acc-local-l3` 没有形成稳定收益。原因是 attention tile 复用主要发生在 L2 或寄存器/线程局部范围内，跨 `CCD` demand fill 占比低，静态限制 `kv_head` 反而降低了补空能力。

HNSW 的不同点：

- 访问对象是图节点和邻接表，不是规则 tile。
- query 路径不固定，调度器有机会按路径相似性分组。
- 上层结构小且热，适合复制到每个 `CCD`。
- 第 0 层大图不适合完整复制，但适合按访问轨迹重排和分区。
- 在线查询主要读索引，副本一致性成本低于读写混合数据结构。

### 条件一：跨域开销可能进入关键路径

HNSW 的第 0 层扩展是随机访存，容易出现高 L3 miss 和 TLB miss。已有 graph reordering 工作报告仅改变 HNSW 节点布局即可降低查询时间，说明图布局和缓存局部性影响真实 runtime。Faiss 官方也提示 Faiss 自身不感知 NUMA，大页和内存分配会影响随机访问索引。

该事实不能直接推出跨 `CCD` 开销一定显著。必须通过 `AMDuProfPcm/IBS/perf c2c` 实测：

- `L3 miss/query`
- `another CCD/CCX same node` 来源占比
- `local memory` 与 `remote memory` 来源
- IBS load latency 分布
- `DTLB miss/query`

### 条件二：放置变量可控

HNSW 的可控变量比 attention 更明确：

- 图节点编号和内存布局可重排。
- 上层节点和热点 hub 可复制。
- 第 0 层可按 cluster、query trace 或 edge cut 分区。
- query 可按 coarse centroid、entry path 或前几跳节点路由。
- `efSearch` 可调，影响搜索范围和候选扩展数量。
- 在线索引只读，副本一致性成本低。

### 条件三：代价可摊销

复制完整 HNSW 不现实，索引远超 L3。可行的是复制小型热导航层：

- `L > 0` 的上层节点。
- top-N 热点第 0 层 hub 的邻接表。
- entry point 和 per-cluster entry points。
- 必要时复制压缩向量码，而不是完整 fp32 向量。

建议每 `CCD` 热副本预算先设为 `4-8 MiB`，避免挤占单 `CCD` 约 `32 MiB` L3。

## 四个设计点的位置

HNSW 查询路径可简化为：

```text
query
  -> upper-layer greedy search
  -> layer-0 entry node
  -> layer-0 best-first expansion
  -> top-k result
```

四个设计点分别作用在不同位置：

| 设计点 | 作用位置 | 改变内容 | 目标 |
| --- | --- | --- | --- |
| per-CCD navigator replica | upper-layer greedy search | 复制上层节点、entry point、热点 hub | 开头阶段本地 L3 命中 |
| trace-guided cold graph partition/layout | layer-0 best-first expansion | 重排和分区第 0 层节点、邻接表、向量码 | 减少随机跨域访问 |
| optional remote expansion | 跨分区扩展 | 把 query state 发给数据 owner worker | 用显式远端执行替代隐式远端 cache line 拉取 |
| hot-path query grouping | query 入队和调度 | 路径相近 query 放到同一 `CCD` queue | 放大热路径复用 |

## 研究点一：Per-CCD Navigator Replica

### 含义

`navigator` 指 HNSW 中负责“导航”的小型热结构，包括：

- 全局 entry point。
- 第 1 层及以上的节点和邻接表。
- 访问频率最高的少量第 0 层 hub。
- 必要的向量码、节点 ID、层级和偏移元数据。

`per-CCD navigator replica` 指每个 `CCD` 都保存一份 navigator 副本。运行在某个 `CCD` 上的 worker 处理 query 时，先访问本地副本完成上层导航。

```text
CCD 0 worker -> navigator replica on CCD 0 -> layer-0 entry
CCD 1 worker -> navigator replica on CCD 1 -> layer-0 entry
...
CCD 7 worker -> navigator replica on CCD 7 -> layer-0 entry
```

### 为什么可能有效

HNSW 上层节点少，但访问频率高。若所有 `CCD` 共享一份上层结构，多个 worker 会反复把同一批 cache lines 拉到不同 `CCD`。在 Zen 上，远端 `CCD` L3 不比 DRAM 稳定更快，因此不应把远端 cache 当容量池。更合理的做法是把小型只读热结构复制到每个本地 `CCD`。

该设计只复制热导航结构，不复制完整索引。完整索引通常远超单 `CCD` L3，复制会浪费内存并挤压缓存。

### 查询路径变化

baseline：

```text
query on CCD i
  -> shared entry point / upper layers
  -> possible remote cache/memory accesses
  -> layer-0 search
```

优化后：

```text
query on CCD i
  -> local navigator replica
  -> local upper-layer greedy search
  -> candidate entry in layer 0
  -> layer-0 search
```

### 设计变量

| 变量 | 说明 |
| --- | --- |
| 复制层级 | 全部 `L>0`，或 `L>=2` 加热点 `L1` |
| 热点阈值 | 按 sampled query 访问频率选 top-N 第 0 层节点 |
| 副本内容 | 邻接表、向量码、节点元数据、entry point |
| 副本预算 | 每 `CCD` `4/8/16 MiB` sweep |
| 路由策略 | query 固定本地导航，底层可本地优先或全局扩展 |

### 实验方案

1. 用 FAISS 或 hnswlib 建立 baseline。
2. 采样 query trace，统计节点访问频率和层级访问占比。
3. 构造 navigator replica 数据结构，保持搜索结果逻辑不变。
4. 对比 baseline、only thread binding、upper-layer replica、upper-layer + hot hub replica。
5. 指标包括 `QPS`、`p50/p95/p99`、`recall@k`、`L3 miss/query`、`another CCD` 来源占比和副本内存开销。

### 预期优化空间

合理预期是低维、内存受限、batch 查询下 `p99` 有 `10%-30%` 改善；若已有 graph reordering、prefetch 和大页，增量可能降到 `3%-15%`。高维 fp32 距离计算主导时收益可能低于 `5%`。

### 风险

- 上层导航只占总时延很小，复制收益有限。
- 热集超过 `CCD` L3 预算后，复制会挤占第 0 层访问。
- query 分布漂移会使热点副本失效。
- 在线 insert/update 会引入副本维护成本。

## 研究点二：Trace-Guided Cold Graph Partition/Layout

### 含义

`cold graph` 指 HNSW 第 0 层的大图。它包含绝大多数节点、邻接表和向量码，不能完整复制到每个 `CCD`。`trace-guided layout` 指先记录真实 query 访问过哪些节点，再按访问轨迹重排和分区第 0 层图。

trace 示例：

```text
query 1:  A -> C -> F -> K -> M
query 2:  A -> C -> G -> K -> N
query 3:  B -> D -> H -> P
```

从 trace 可得到节点转移频率：

```text
A-C high
C-F medium
C-G medium
F-K high
G-K high
B-D low
```

布局目标是让经常连续访问的节点在内存中更近，尽量属于同一个 `CCD` owner。

```text
sampled query trace
  -> visited node sequence
  -> transition weight T(u, v)
  -> partition nodes to CCD groups
  -> reorder nodes within group
  -> first-touch pages by owner group
```

### 为什么可能有效

HNSW 第 0 层搜索的主要成本来自节点扩展。若访问路径频繁跨 `CCD`，线程会不断等待远端 cache line 或 DRAM 返回。普通 graph reordering 已证明节点布局会影响 HNSW 查询时间，但普通线性重排不考虑 `CCD` 归属、NPS/NUMA first-touch 和 per-CCD queue。

chiplet-aware layout 的目标不是只让节点编号连续，而是让“同一 query 常连续访问的节点”落在相同拓扑域。

### 与普通 graph reordering 的区别

| 维度 | 普通 graph reordering | Chiplet-aware trace-guided layout |
| --- | --- | --- |
| 目标 | 改善 cache line / page locality | 改善本地 `CCD` L3 和 NUMA locality |
| 输入 | 图边、节点度、访问频率 | 图边 + query trace + 热点 + 分区负载 |
| 输出 | 单一节点顺序 | 节点顺序 + `CCD` owner + first-touch 策略 |
| 调度 | 通常不改 query 执行位置 | query 可路由到 owner queue |
| 评价 | 查询时间、cache miss | p99、recall、跨 `CCD` 来源占比、本地转移比例 |

### 设计变量

| 变量 | 说明 |
| --- | --- |
| 分区依据 | 向量 cluster、HNSW 边、query transition trace、节点热度 |
| 分区数量 | `CCD` 数、NPS domain 数、socket 数 |
| 边界策略 | 保留跨分区边，不删边；可复制少量边界 hub |
| 页放置 | `numactl/mempolicy/first-touch` |
| 重排粒度 | node、neighbor list、compressed code block |
| 负载约束 | 避免所有 hot nodes 被放到同一 `CCD` |

### 实验方案

1. baseline：原始 HNSW。
2. baseline2：普通 graph reordering。
3. 方案：trace-guided reorder + per-CCD partition。
4. 方案增强：加入 hot replica。
5. 对不同 query 分布测试 train/test 同分布、hotspot shift、uniform、Zipf。
6. 指标包括本地转移比例、跨分区边访问比例、`recall@k`、p99、`L3 source latency`。

### 预期优化空间

相对未优化 HNSW，若 baseline 没有重排和大页，可能有 `10%-30%` p99 改善。相对 graph reordering baseline，合理目标是 `3%-15%` 增量。

### 风险

- sampled trace 不稳定。
- 分区导致热门 `CCD` 过载。
- first-touch 无法稳定控制到单个 `CCD`，可能只能控制 socket/NPS。
- 跨分区边过多时，布局收益被远端访问抵消。

## 研究点三：Optional Remote Expansion

### 含义

HNSW 分区后，query 在本地 `CCD` 搜索时可能遇到远端分区节点。baseline 做法是本地线程直接读取远端节点的邻接表和向量码。`remote expansion` 的做法是把小的 query state 发给远端 owner worker，让远端 worker 在本地数据上扩展，再返回少量候选。

```text
local worker on CCD i
  -> sees boundary node owned by CCD j
  -> send(query_vector, heap_threshold, candidate_ids) to CCD j
  -> remote worker expands local graph around that node
  -> return top-m candidates
  -> local worker merges candidates into heap
```

### 为什么只是 optional

该方向风险高。AMD Zen 上跨 `CCD` 访问约百纳秒量级，软件队列、同步和 heap merge 的成本可能更高。只有当一次远端扩展会触碰大量邻接表、向量码和候选节点时，移动 query state 才可能优于硬件直接拉 cache line。

### 适用条件

- query 经常连续访问同一远端分区的一批节点。
- 远端扩展请求可 batch 化。
- owner worker 队列等待低。
- 返回候选数量小，heap merge 成本可控。
- recall 不因远端扩展截断而下降。

### 验证方案

- 只对跨分区访问密集的 query 启用。
- 按 batch 合并远端扩展请求。
- 比较 direct remote load、remote expansion、remote expansion + batching。
- 记录消息数、队列等待、heap merge 开销、p99 和 recall。

### 风险

- 软件消息开销超过硬件远端访问成本。
- owner worker 形成热点队列。
- 异步返回会改变搜索顺序，可能影响 recall 或需要更大 `efSearch`。
- 实现复杂度高，不适合作为第一阶段核心。

## 研究点四：Hot-Path Query Grouping

### 含义

不同 query 的 HNSW 搜索路径可能重叠。例如多个 query 都从同一个上层 hub 进入同一片第 0 层区域。`hot-path query grouping` 指在服务端短时间窗口内识别路径相近的 query，并把它们放到同一 `CCD` queue 里连续执行。

```text
incoming queries
  -> compute lightweight signature
  -> group by signature for 10-100 us
  -> dispatch group to same CCD queue
  -> reuse navigator / hot hub / nearby layer-0 nodes
```

signature 可以是：

- coarse centroid ID。
- 上层 entry path。
- 前几跳访问到的 hot hub。
- 预测的第 0 层 owner partition。

### 为什么可能有效

单个 query 的路径不一定能完全本地化，但多个相似 query 在短窗口内执行时，会重复访问相同 entry point、上层节点、热点 hub 和局部邻接表。把它们分散到不同 `CCD` 会造成重复 cache miss；放到同一 `CCD` 有机会把这些结构留在本地 L3。

### 设计变量

| 变量 | 说明 |
| --- | --- |
| grouping window | `10-100 us`，窗口越大复用越强但排队越高 |
| signature | centroid、entry point、top-layer path、visited hot hub |
| 排队策略 | strict grouping、bounded grouping、fallback FCFS |
| 负载均衡 | 热点 queue 溢出后 spill 到其他 `CCD` |
| SLO 约束 | p99 不允许因等待分组而恶化 |

### 适用场景

适合 batch RAG、企业知识库问答、离线评测服务、低延迟要求不极端的高并发检索。不适合单 query latency 或极低 QPS。

### 风险

- 等待成 batch 会增加单请求延迟。
- 热门 signature 会形成单 `CCD` 过载。
- grouping 预测错误时，局部性收益不稳定。
- 若 query 分布近似 uniform，路径重叠可能不足。

## 最小实现路径

### 阶段一：证明 HNSW 有 chiplet 相关瓶颈

1. 选择 `hnswlib` 或 FAISS HNSW，建立只读查询微基准。
2. 固定线程到 `CCD` 或 NPS domain，控制 query 并发。
3. 采集 `QPS`、`p50/p95/p99`、`recall@k`。
4. 用 `AMDuProfPcm/IBS/perf` 采集 `L3 miss/query`、`another CCD` 来源占比、load latency。
5. 做 CAT 实验验证 L3 容量敏感性。

判据：若 L3 容量限制和跨 `CCD` 来源占比都不敏感，该方向停止。

### 阶段二：实现 navigator replica counterfactual

1. 采样 query trace，统计各层节点访问频率。
2. 复制 `L>0` 节点和 top-N 热点 hub。
3. 每个 `CCD` worker 优先访问本地 navigator。
4. 对比 only binding、upper-layer replica、upper-layer + hot hub replica。

判据：若热副本命中率高但 runtime/p99 不变，说明上层导航不是瓶颈。

### 阶段三：实现 trace-guided layout

1. 根据 query trace 构造节点转移权重。
2. 生成普通 graph reordering baseline。
3. 生成 chiplet-aware partition + reorder。
4. 用 first-touch 或 NUMA policy 固定分区页归属。
5. 比较原始 HNSW、普通重排、chiplet-aware 重排。

判据：chiplet-aware layout 的增量必须相对普通 graph reordering 仍成立。

### 阶段四：评估扩展项

1. query grouping：验证路径重叠、排队代价和 p99。
2. remote expansion：只在跨分区访问密集且可 batch 的场景尝试。

## 判停条件

- CAT 限制 L3 后性能变化低于 `3%`，且 `another CCD` 来源占比低。
- 热副本命中率高但 runtime/p99 不变。
- graph reordering baseline 已吃掉几乎全部收益。
- query 分布轻微变化即导致布局退化。
- `recall@k` 下降或需要显著增大 `efSearch` 才能保持 recall。
- query grouping 带来的排队延迟超过缓存复用收益。
- remote expansion 的消息和同步成本超过 direct remote load。

## 参考资料

- HNSW 原论文：https://arxiv.org/abs/1603.09320
- Graph Reordering for Cache-Efficient Near Neighbor Search：https://arxiv.org/abs/2104.03221
- Faiss indexes：https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- Faiss performance tips：https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster
- hnswlib：https://github.com/nmslib/hnswlib
- d-HNSW：https://arxiv.org/abs/2505.11783
- DiskANN：https://github.com/microsoft/DiskANN
