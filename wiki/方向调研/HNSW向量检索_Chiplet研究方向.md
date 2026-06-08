# HNSW 向量检索的 Chiplet 研究方向

## 结论

`HNSW`（Hierarchical Navigable Small World，层次化可导航小世界图）适合作为 Chiplet CPU 上的候选研究对象，但不能直接假设存在 chiplet-aware 优化空间。当前可靠路径是先把 hnswlib 的查询访存问题、batch 并行方式和共享状态分开，再用硬件计数证明瓶颈来源。

本文保留三个主要研究点（均为 chiplet 特有问题）：

1. **同一 CCD 内相似 query 并行调度缺失**：面向分片 L3 的主要 chiplet 研究点。
2. **多 CCD 执行中的 L3 容量复用与线程放置失配**：研究集中执行、分散执行和每 CCD 分组执行的边界。
3. **查询辅助共享状态的跨 CCD 同步干扰**：排除 hnswlib runtime 同步对硬件归因的污染。

第 0 层图遍历的节点块布局低局部性（原研究点一）可行度低，不作为主线。原因：
- 该问题是通用 cache locality 问题，不涉及 CCD/分片 L3 等 chiplet 特有机制，Graph Reordering 等论文已有成熟结论（Gorder/RCM 可降低 cache/TLB miss，加速 10%-40%）。
- 文档此前称其为"所有 chiplet-aware 方案必须超过的强 baseline"，但 chiplet 方案关注的是跨 CCD 调度与 L3 复用，普通 graph reordering 改善的是单机通用局部性，二者解决的问题维度不同，不存在"必须超过"的关系。
- 降级为实验前置的 baseline 构建步骤（RCM reorder 作为通用优化对照），不作为独立研究贡献。

`Per-CCD navigator replica`、`remote expansion`、先验 `CCD owner/partition` 不作为主线。它们缺少 hnswlib 代码路径中的直接问题基础，且容易破坏普通 graph reordering 已经能改善的全局局部性。

## hnswlib 查询事实

### 查询并行粒度

`/home/zjj/hnswlib/python_bindings/bindings.cpp` 的 `knnQuery_return_numpy(...)` 通过 `ParallelFor(...)` 做 batch row 级并行。输入矩阵的一行就是一个 query。

```text
knn_query(query_matrix, num_threads)
  -> ParallelFor(0, rows, num_threads)
       -> worker 领取一个 row
       -> 对该 row 执行一次完整 searchKnn(...)
       -> 写 labels[row, :] 和 distances[row, :]
```

单个 `searchKnn(...)` 内部串行执行，不会把一个 query 拆给多个线程：

```text
searchKnn(query)
  -> 上层 greedy search
  -> 第 0 层 searchBaseLayerST
  -> internal id 转 external label
```

一个线程同一时刻只计算一个 query。多线程查询的并行来自多个线程同时计算不同 query。

### internal id 与 external label

`internal id` 是 hnswlib 内部节点编号，类型是 `tableint`，通常是 `0..N-1`。HNSW 图的邻接表保存 internal id。搜索过程中，候选堆、结果堆、visited 标记和邻接表跳转都使用 internal id。

`external label` 是用户传入和查询返回的 ID，类型是 `labeltype`。Python 中常见写法如下：

```python
index.add_items(data, ids=np.arange(n))
```

这里的 `ids` 是 external label。hnswlib 用 `label_lookup_` 保存：

```text
external label -> internal id
```

查询返回前，hnswlib 用 `getExternalLabel(internal_id)` 把内部节点编号转换为用户可见 ID。

### 第 0 层节点块布局

hnswlib 把第 0 层数据放在连续内存 `data_level0_memory_` 中。每个 internal id 对应固定大小的节点块：

```text
data_level0_memory_
  [internal id 0 的节点块]
  [internal id 1 的节点块]
  [internal id 2 的节点块]
  ...
```

每个节点块包含：

```text
| level-0 neighbor list | vector data | external label |
```

地址由 internal id 线性计算：

```text
get_linklist0(id)       -> data_level0_memory_ + id * size_data_per_element_ + offsetLevel0_
getDataByInternalId(id) -> data_level0_memory_ + id * size_data_per_element_ + offsetData_
getExternalLabel(id)    -> data_level0_memory_ + id * size_data_per_element_ + label_offset_
```

因此 internal id 同时决定图节点的逻辑编号和第 0 层节点块的内存位置。改变 internal id 排列会改变邻接表、向量和 label 的物理布局。

在 `dim=128, M=16` 时，单个第 0 层节点块约为：

```text
level-0 links = 2*M*4B + 4B = 132B
vector       = 128*4B       = 512B
label        = 8B           = 8B
total        = 652B/node
```

`50K / 200K / 1M` 节点的第 0 层主数据约为 `32.6 MiB / 130 MiB / 652 MiB`。本机每个 CCD 约 32 MiB L3，因此这三个规模分别覆盖接近单 CCD L3、超过单 CCD L3、超过单 socket 聚合 L3 的场景。

### 查询临时状态与共享状态

每个 query 独立维护：

- `candidate_set`：第 0 层待扩展候选堆。
- `top_candidates`：当前结果候选堆。
- `VisitedList`：当前 query 已访问节点标记。

这些状态不能在 query 之间共享。即使两个 query 很相似，它们的距离值、候选堆顺序和停止条件也不同。

多线程查询共享的主要数据是只读索引：

- `data_level0_memory_`
- `linkLists_`
- `element_levels_`
- `enterpoint_node_`
- `maxlevel_`

多线程查询还共享少量 runtime 状态：

- `ParallelFor` 的 `std::atomic<size_t> current`，用于领取 row。
- `VisitedListPool::poolguard`，用于分配和回收 `VisitedList`。
- `metric_hops` 和 `metric_distance_computations`，在开启 `collect_metrics` 的路径中是 atomic 写入。

分析 chiplet 瓶颈时必须区分两类现象：一类是只读索引随机访问导致的 cache/TLB/DRAM 问题，另一类是 row 调度、visited list 池和 metric 计数器导致的同步问题。

## 论文事实与边界

### graph reordering 的已知结论

`Graph Reordering for Cache-Efficient Near Neighbor Search` 已证明，HNSW 类图近邻搜索对节点内存布局敏感。`Graph reordering`（图重排序）在不改变图结构、搜索算法和 recall 的前提下重新分配节点编号，使搜索过程中常连续访问或共享邻域的节点在内存中更接近。

该论文报告：

- `Gorder` 和 `RCM` 可降低 L1/L2/L3/TLB miss。
- 大规模 embedding 数据集上查询加速约 `10%-40%`。
- SIFT100M 上 P99 latency 中，RCM 改善约 `17%`，Gorder 改善约 `30%`。
- HNSW 上层初始化占比约 `2.9% ± 1.9%`，第 0 层 beam search 主导查询时间。

论文平台是共享 L3 的 Intel Xeon，不是 AMD Zen chiplet CPU。该结论只能作为普通 cache-friendly baseline，不能直接写成 chiplet 贡献。

### HNSW 内存层次相关论文的迁移边界

HM-ANN 面向 DRAM + Optane PMM，d-HNSW 面向 RDMA 解耦内存。二者说明 HNSW 对内存层次和访问粒度敏感，但瓶颈分别是 PMM 慢内存和 RDMA round trip，不是单机 CCD/L3。

这些论文可借鉴的方法是：

- 区分热结构和冷结构。
- 用访问粒度和数据来源解释性能。
- 通过 counterfactual 实验证明放置策略的必要性。

不能迁移其 speedup、数据结构设计或远端访问假设到当前 Zen chiplet 平台。

### recall 约束

HNSW recall 受 `M`、`efConstruction`、`efSearch`、数据内在维度、插入顺序和图结构影响。任何重排 internal id、改变构建顺序、修改图边或改动查询调度的实验都必须报告 `recall@k`。只看 QPS 或 p99 会把图质量变化误判为硬件优化收益。

## 研究点一：第 0 层图遍历的节点块布局低局部性（可行度低，不作为主线）

### 问题

hnswlib 第 0 层搜索沿邻接表做 best-first 图遍历。每扩展一个节点，代码会读取该节点的第 0 层邻接表，并对邻居节点执行 `getDataByInternalId(candidate_id)` 读取向量计算距离。

该访问路径由图边和 query 共同决定，不是顺序扫描：

```text
current_node -> level-0 neighbor list
             -> candidate_id
             -> data_level0_memory_ + candidate_id * size_data_per_element_
```

如果连续扩展或连续检查的节点 internal id 相距很远，硬件看到的是分散 cache line 和分散 page 访问。对 `dim=128` 的 float 向量，一个节点向量约 512B，加上邻接表约 652B。第 0 层搜索访问几百到几千个节点时，访问 footprint 很容易超过 L2，并对 L3、DTLB 和 DRAM 产生压力。

该问题在 hnswlib 代码中有直接对应物：

- `data_level0_memory_` 按 internal id 线性存储。
- `get_linklist0(id)`、`getDataByInternalId(id)` 和 `getExternalLabel(id)` 都由 internal id 计算地址。
- `searchBaseLayerST(...)` 按邻接表中的 internal id 跳转。

该问题是通用 cache locality 问题，不是 chiplet 特有问题。Graph Reordering 等论文已有成熟结论，增量贡献有限。chiplet 方案关注的是跨 CCD 调度与 L3 复用，与此问题解决的维度不同，不存在"必须超过"的关系。

### 问题验证

验证目标是证明“连续访问节点的内存距离过大”会导致可观的 cache/TLB/DRAM 代价。

实验对照：

```text
original layout
random reorder
RCM reorder
trace-order reorder
```

`random reorder` 是负对照，用于排除“只要重排就变快”的错误解释。`RCM reorder` 基于第 0 层图边降低连接节点编号跨度。`trace-order reorder` 基于采样 query 的第 0 层访问序列，把同一 query 中连续访问的节点放近。

验证指标：

- `recall@k`
- QPS、p50、p95、p99
- L1/L2/L3 miss
- DTLB miss
- IBS load latency 或 AMDuProf demand fill 来源
- 每 query 访问节点数、距离计算次数

成立判据：

- RCM 或 trace-order 相对 original 有稳定收益。
- random reorder 不应产生同等收益。
- `recall@k` 不下降。
- cache/TLB/IBS 指标下降能解释延迟下降。

判停条件：

- RCM 和 trace-order 对 hnswlib 无稳定收益。
- 收益只出现在 original，对强 baseline 或不同数据规模不稳定。
- `recall@k` 变化解释了主要性能差异。

### 方法设计

最小实现只改变 internal id，不改变图结构、距离函数和搜索算法。

需要同步改写：

- `data_level0_memory_`：按新 internal id 重排节点块。
- 第 0 层邻接表：neighbor internal id 替换为新 id。
- `linkLists_`：高层邻接表中的 internal id 替换为新 id。
- `element_levels_`：按新 id 重排。
- `enterpoint_node_`：替换为新 id。
- `label_lookup_`：external label 映射到新 internal id。

`trace-order reorder` 的最小流程：

```text
1. 对采样 query 记录 searchBaseLayerST 的扩展序列。
2. 按 trace 出现顺序追加未出现节点。
3. trace 未覆盖节点按原 internal id 追加。
4. 生成 old_id -> new_id 映射。
5. 重写节点块和所有邻接表 internal id。
```

该方法输出通用 cache-friendly 重排结果，可作为实验对照，但不构成 chiplet 研究的强 baseline。

## 研究点二：同一 CCD 内相似 query 并行调度缺失

### 问题

hnswlib 的 batch 查询把每个 query row 作为独立任务动态分发。`ParallelFor` 只维护一个全局 row counter，不理解 query 相似性，也不理解 CCD/L3 拓扑。

在本机上，一个 CCD 内 16 个物理核心共享约 32 MiB L3。若 16 个核心同时执行相似 query，这些 query 可能访问相同或高度重叠的 HNSW 节点。相同节点的邻接表、向量和 label 都在只读索引内，可以被同一 CCD 的 L3 缓存复用。

该问题成立的机制是并行复用，不是单线程顺序复用：

```text
CCD 0 上 16 个核心
  core 0 执行 query A
  core 1 执行 query B
  ...
  core 15 执行 query P

若 A..P 的 HNSW 访问路径重叠：
  多个核心在相近时间读取相同节点块
  共享 L3 可保留这些只读 cache line
  后续核心读取相同节点时减少 L3 miss / DRAM fill
```

默认 `ParallelFor` 的问题是缺少这种共调度约束。若 batch 中 query 顺序随机，或者线程跨 CCD 分散，相似 query 可能被分配到不同 CCD 或在时间上错开执行。共享 L3 的复用窗口缩短，节点块需要在不同 CCD 的 L3 中重复填充。

该研究点不能假设共享候选堆或共享距离计算结果。hnswlib 中 `candidate_set`、`top_candidates` 和 `VisitedList` 都是 per-query 状态；不同 query 的距离值也不同。可复用的只有只读索引 cache line。

优化空间取决于三个条件：

- 相似 query 的 HNSW 访问节点重叠率足够高。
- 查询性能受 L3/DRAM/DTLB 影响，而不是完全受距离计算吞吐限制。
- 相似 query 的并行执行时间窗口足够接近，cache line 仍驻留在同一 CCD 的 L3 中。

### 问题验证

第一步验证 query 访问重叠。

对采样 query 记录：

```text
upper-layer path
level-0 expanded node ids
level-0 distance-computed node ids
```

计算相似 query 与随机 query 的访问重叠率：

```text
overlap(qi, qj) = |visited_i ∩ visited_j| / |visited_i ∪ visited_j|
```

query 相似性可用以下 signature（签名，即用于分组的轻量特征）近似：

- 向量 coarse centroid id。
- 上层 greedy path 的前几个 internal id。
- 第 0 层前若干个扩展节点的 MinHash/sketch。
- 采样阶段得到的访问集合 sketch。

第二步验证同一 CCD 并行共调度收益。

构造三组 batch：

```text
grouped batch   : 相似 query 连续排列
shuffled batch  : 相同 query 集随机打散
anti-group batch: 尽量让相邻 query 不相似
```

固定 16 个线程绑到同一 CCD：

```text
num_threads=16
physcpubind=<同一 CCD 的 16 个物理核>
```

对比 grouped、shuffled、anti-group 的 QPS、p99、L3 miss、DTLB miss、demand fill 来源和 IBS load latency。

第三步验证 chiplet 拓扑相关性。

在相同 query 集上比较：

```text
单 CCD 16 线程 grouped
单 CCD 16 线程 shuffled
多 CCD 每 CCD 16 线程 per-CCD grouped
多 CCD 每 CCD 16 线程 global shuffled
```

成立判据：

- 相似 query 的访问重叠率显著高于随机 query。
- grouped 在同一 CCD 16 线程下稳定优于 shuffled。
- grouped 的收益伴随 L3 miss、DTLB miss 或 demand fill 下降。
- per-CCD grouped 在多 CCD 下仍能保持收益。

判停条件：

- 相似 query 的访问重叠率不高。
- grouped 只改变负载均衡，不改变 L3/DTLB/IBS 指标。
- grouped 的收益小于排序、分桶和调度开销。
- grouped 只在单次 trace 上有效，对新 query 集不稳定。

### 方法设计

最小方法不改 hnswlib 内部搜索算法，只改 batch 输入顺序：

```text
1. 为每个 query 计算 signature。
2. 按 signature 排序或分桶。
3. 调用 knn_query(grouped_queries, num_threads=16)。
4. 查询结束后按原 query 顺序还原输出。
```

该方法依赖 `ParallelFor` 的动态领取顺序。相邻 row 会在启动阶段被同一组 worker 领取，但长尾 query 可能打乱后续执行窗口。

增强方法是在 binding 或 C++ benchmark 中实现 per-CCD queue：

```text
1. 每个 CCD 维护一个 query queue。
2. 每个 queue 内按 signature 分组。
3. 该 CCD 的 16 个 worker 只从本 CCD queue 领取任务。
4. 每次领取一个 chunk，而不是单 row。
```

chunk 粒度用于保证同一 CCD 内 16 个核心在相近时间处理同一组相似 query，并降低全局 atomic row counter 的干扰。

该研究点的 chiplet 特征来自“把相似 query 的并行执行窗口对齐到同一个共享 L3 域”。它不是普通 query 排序；普通排序只改变全局顺序，per-CCD grouped queue 明确使用 CCD/L3 拓扑作为调度域。

## 研究点三：多 CCD 执行中的 L3 容量复用与线程放置失配

### 问题

Chiplet CPU 上存在集中执行和分散执行的权衡。集中到一个 CCD 可获得低延迟共享 L3 和更强的同组 query 复用，但只有约 32 MiB L3。分散到多个 CCD 可获得更大的聚合 L3 和更多核心，但相同只读节点块可能在多个 CCD 的 L3 中重复缓存。

hnswlib 默认查询调度不控制这个权衡：

- `ParallelFor` 不知道 worker 运行在哪个 CCD。
- OS 调度可能迁移线程，除非显式绑核。
- 动态 row counter 不区分每个 CCD 的局部队列。
- 所有 CCD 读取同一个只读索引，没有 per-CCD 工作集规划。

当 HNSW 访问存在热点子图或相似 query 同时运行时，多 CCD 可能出现两类代价：

1. **重复缓存容量压力**：多个 CCD 的 L3 同时缓存相同热点节点块，单 CCD 本地有效容量不足，聚合 L3 没有完全转化为有效容量。
2. **线程放置失配**：相似 query 被分散到不同 CCD，同一 CCD 内无法复用共享 L3；不相似 query 被放在同一 CCD，互相挤占本地 L3。

该问题不是“query 在不同 CCD 之间转移”。真实执行路径是线程在哪个 core 上运行，就由该 core 拉取所需节点数据。问题来自线程放置、query 分组和分片 L3 的容量/复用关系。

### 问题验证

实验以 RCM reorder 作为通用优化对照，避免把普通布局低局部性误写成 chiplet 问题。

固定线程数，不同拓扑放置：

```text
16 threads on 1 CCD
16 threads spread across 16 CCDs
128 threads on 8 CCDs of one socket
256 threads on 16 CCDs of two sockets
```

固定拓扑，不同 L3 容量：

```text
resctrl CAT: full ways, 1/2 ways, 1/4 ways, 1/8 ways
```

固定拓扑，不同工作集：

```text
50K nodes  -> 接近单 CCD L3
200K nodes -> 超过单 CCD L3，低于单 socket 聚合 L3
1M nodes   -> 超过单 socket 聚合 L3
```

固定 query 集，不同调度：

```text
global shuffled
global grouped
per-CCD grouped
per-CCD random
```

主指标：

- QPS、p95、p99
- IPC/CPI
- L3 access/miss
- Demand DC Fills From Local L3 or different L2 in same CCX
- Demand DC Fills From another CCX in same node
- Demand DC Fills From Local Memory or I/O
- DTLB miss
- IBS load latency

成立判据：

- 性能随 CCD 放置策略稳定变化，且差异不能由线程数、频率或 NUMA 内存位置解释。
- CAT 缩小本地 L3 后性能明显退化，说明 L3 容量或驻留有实际影响。
- per-CCD grouped 相对 global shuffled 的收益伴随 L3/DTLB/demand fill 指标改善。
- 不同规模下表现符合容量边界：接近单 CCD L3 的数据更受集中复用影响，远超聚合 L3 的数据更受 DRAM/DTLB 影响。

判停条件：

- 同线程数不同 CCD 放置无稳定差异。
- CAT 限制 L3 ways 后性能变化低于噪声。
- 指标显示主要代价来自本地内存流式 miss，且不随 CCD 放置改变。
- per-CCD grouped 不优于普通 grouped。

### 方法设计

方法设计以调度为主，不先改图结构。

保守策略：

```text
小工作集或高 query 重叠:
  集中到较少 CCD
  每 CCD 内做相似 query grouped

大工作集或低 query 重叠:
  分散到更多 CCD
  用 per-CCD queue 避免全局 row counter
```

自适应策略：

```text
1. 采样一小段 query。
2. 估计 query signature 分布和访问重叠率。
3. 根据索引规模、efSearch、线程数和 CAT/L3 敏感性选择 CCD 数。
4. 每个 CCD 建立局部 queue。
5. queue 内按 signature 分组，queue 间做负载均衡。
```

该方向的贡献边界必须清晰：若收益只来自普通 query sorting，则不是 chiplet 贡献；只有 per-CCD 调度域、CCD 数选择或 L3 容量曲线带来额外收益，才能写成 chiplet-aware 优化。

## 研究点四：查询辅助共享状态的跨 CCD 同步干扰

### 问题

hnswlib 查询路径存在少量共享写和锁。它们不属于 HNSW 图遍历本身，但在多线程跨 CCD 执行时可能造成 cache line bouncing 或锁竞争，污染对索引访存问题的判断。

具体共享状态包括：

- `ParallelFor` 的 `std::atomic<size_t> current`：每处理一个 row 执行一次 `fetch_add(1)`。
- `VisitedListPool::poolguard`：每个 query 在进入和退出 `searchBaseLayerST(...)` 时各持有一次 mutex。
- `metric_hops` 和 `metric_distance_computations`：开启 `collect_metrics` 时在扩展节点处执行 atomic 写入。

这些共享状态的作用：

- row counter 用于动态负载均衡。
- `VisitedListPool` 避免每个 query 反复分配 visited array。
- metric atomic 用于统计访问 hop 和距离计算数。

它们与第 0 层节点块布局无关。若这些共享状态在多 CCD 下成为瓶颈，直接做 graph layout 或 query grouping 会误判收益来源。

### 问题验证

构造三个 counterfactual：

```text
chunk scheduler:
  每次领取一段 row，减少 fetch_add 次数

thread-local VisitedList:
  每个 worker 固定持有 visited array，不进入全局 pool

metric-off / thread-local metric:
  关闭全局 atomic metric，或先写线程本地计数再汇总
```

实验矩阵：

```text
baseline hnswlib
chunk scheduler
thread-local VisitedList
metric-off
三者叠加

1 CCD
多 CCD
低 efSearch
高 efSearch
小 batch
大 batch
```

成立判据：

- 消除共享状态后，多 CCD 扩展性明显改善。
- 改善不伴随 HNSW 节点访问的 L3/DTLB/IBS 指标变化。
- perf lock、atomic 相关事件或热点栈显示同步路径占比下降。

判停条件：

- 共享状态 counterfactual 对 QPS/p99 无稳定影响。
- 性能差异主要随 HNSW 访问节点数和 L3/DTLB 指标变化。
- metric 路径默认未开启，atomic metric 不在实际查询路径中出现。

### 方法设计

该研究点优先作为实验清洁步骤。

可落地修改：

- 把 `ParallelFor` 从单 row 动态领取改为 chunk 动态领取。
- 在 worker 创建时分配 thread-local `VisitedList`。
- 默认关闭全局 metric atomic，或使用 thread-local counter 后汇总。
- 在 per-CCD grouped queue 中复用 chunk scheduler，减少全局共享写。

若该方向收益显著，可作为 hnswlib 多线程实现优化。若论文主线是 chiplet-aware HNSW，应把它作为消除干扰的前置条件，而不是替代研究点一和研究点二。

## 不建议作为主线的方向

### Per-CCD navigator replica

上层导航占比通常很小。Graph Reordering 论文报告 HNSW 层次初始化约占查询时间 `2.9% ± 1.9%`。hnswlib 的高层邻接表在 `linkLists_`，但距离计算仍读取 `data_level0_memory_` 中的向量。只复制高层 navigator 不能避免第 0 层向量访问。

只读 cache line 可由硬件在不同 CCD 的 L3 中保存副本。没有函数级 PMU/IBS 证据前，软件显式复制上层结构缺少必要性。

### Remote expansion

hnswlib 单 query 的候选堆、结果堆和 visited state 是线程本地串行状态。把节点扩展任务发给远端 CCD 需要拆分状态、同步候选、合并 heap，并维护 visited 一致性。当前平台跨 CCD cache 路径延迟接近本地内存量级，软件消息和同步成本大概率高于直接由当前 core 拉取数据。

### 先验 CCD owner / partition

HNSW 节点没有天然 CCD owner。先验给节点分配 owner 是人工概念，代码中没有对应状态。若按 owner 限制访问，可能破坏普通 graph reordering 的全局局部性，也可能造成负载倾斜。

只有研究点二证明 per-CCD query 调度后仍存在明确 L3/CCD 剩余瓶颈，才可把 CCD partition 作为 counterfactual，而不是起点。

## 最小工作计划

1. 在 hnswlib 中增加 trace 采样，记录上层路径、第 0 层扩展节点和距离计算节点。
2. 实现 random、RCM、trace-order 三种 internal id 重排。
3. 验证重排前后 `recall@k` 一致。
4. 在 `50K/200K/1M` 节点规模上测 RCM 重排收益，作为通用优化对照。
5. 基于 trace 计算 query 访问重叠率，构造 grouped、shuffled、anti-group batch。
6. 在单 CCD 16 线程下验证相似 query 并行共调度收益。
7. 扩展到 per-CCD grouped queue，比较不同 CCD 数、CAT 容量和工作集规模。
8. 加入 chunk scheduler、thread-local VisitedList、metric-off counterfactual，排除 runtime 同步干扰。

## 总判停条件

- 普通 graph reordering 无稳定收益，且 L3/DTLB/IBS 指标不支持布局问题。
- 相似 query 的 HNSW 访问重叠率不高。
- grouped batch 不优于 shuffled batch，或收益不能由 L3/DTLB/IBS 指标解释。
- per-CCD grouped 不优于普通 grouped。
- CAT 限制 L3 后性能变化低于噪声。
- 多 CCD 放置差异主要来自 NUMA 内存、频率或负载均衡，而不是 L3/CCD 指标。
- 任一优化导致 `recall@k` 下降，或需要提高 `efSearch` 才能维持 recall。

## 参考资料

- `wiki/源码分析/hnsw/hnswlib_HNSW并行计算过程_线程任务与图遍历.md`
- `wiki/实验方案/HNSW_hnswlib_Chiplet敏感性实验方案.md`
- `wiki/论文解读/Graph_Reordering_for_Cache-Efficient_Near_Neighbor_Search解读.md`
- `wiki/论文解读/HNSW原论文解读.md`
- `wiki/论文解读/Graph-Based_Vector_Search实验评估解读.md`
- `wiki/论文解读/HM-ANN解读.md`
- `wiki/论文解读/d-HNSW解读.md`
- `wiki/论文解读/The_Impacts_of_Data_Ordering_and_Intrinsic_Dimensionality_on_Recall_in_Hierarchical_Navigable_Small_Worlds解读.md`
- `wiki/术语与背景/Chiplet与硬件背景.md`
- `/home/zjj/hnswlib/hnswlib/hnswalg.h`
- `/home/zjj/hnswlib/python_bindings/bindings.cpp`
