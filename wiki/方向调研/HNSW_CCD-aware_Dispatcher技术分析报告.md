# HNSW CCD-aware Dispatcher 技术分析报告

## 结论

当前 HNSW/Chiplet 方向的关键事实已经成立：相似 query 在同一 CCD 内并发执行时，会提高本地 cache 命中并降低本地 DRAM 访问。Lab1-Lab3 的模拟 topic workload 显示，`similar` 相对 `different` 的收益随 query 相似度上升而增强；Lab4 的真实 MS MARCO + Qwen3 embedding trace 显示，按 query-query cosine oracle 分组后，16 个并发 query 的 HNSW 路径重叠显著提高。

尚未成立的是“在线 dispatcher 可以用足够低的成本稳定找到这些 query”。完整 query-query cosine kNN、全 batch 贪心分组、访问路径 sketch 等机制都不能作为毫秒级 HNSW 前处理。Lab4 的 `cosine-oracle` 是收益上界，不是可部署 dispatcher：它将 search-only `avg_ms_per_query` 从 `7.168 ms` 降到 `6.483 ms`，但 dispatcher 本身消耗 `5905 us/query`，端到端吞吐从 `1796.5 QPS` 降到 `407.84 QPS`。

适合当前系统的 dispatcher 应采用极低成本签名和局部队列，不做全 batch 两两比较。首选路线是“轻量 query signature + per-CCD sticky queue + 负载感知溢出”：用 SimHash、低维 PCA 后的粗 centroid、少量随机投影或离线训练的 coarse code 给 query 生成短签名，再按签名把 query 放入固定 CCD 队列。每个 CCD 内只在小窗口内凑 cohort，超时立即发出，避免为分组引入可见排队。

## 当前实验依据

平台背景：`2 x AMD EPYC 9745 128-Core`，每个 CPU 8 个 CCD，每个 CCD 16 个物理核共享约 32 MiB 本地 L3。HNSW 查询采用 hnswlib row 级并行，单个 `searchKnn` 内部串行；多线程来自不同 query 同时执行。

Lab1 使用 10M、1024 维 cosine、`M=32`、`efSearch=512` 的 topic 数据集，在 16 线程同 CCD 条件下得到：

| workload | avg ms/query | 相对 different |
|---|---:|---:|
| similar | 1.755 | 0.534x |
| same | 2.394 | 0.729x |
| different | 3.285 | 1.000x |

AMDuProfCLI 显示差异主要来自 `InnerProductSIMD16ExtAVX512` 的向量读取路径。`different` 的 Local Cache hit 很低，Local DRAM hit 高；`similar/same` 把大量 L1 miss 从 DRAM 路径转移到本地 cache 路径。Peer/Remote cache 与 Remote DRAM 接近 0，因此当前收益不应表述为“减少跨 CCD 访问”，而应表述为“同 CCD 本地 cache 复用只读 HNSW 节点向量”。

Lab3 在多 CCD 场景复现实验2，`similar` 在所有已测 alpha 下均快于 `different`。`alpha=0.998` 时，`similar avg_ms=1.552697`，`different avg_ms=6.931814`，`similar/different=0.223996`。Local L3 hit 随 alpha 上升，Local DRAM hit 与 L1 miss 平均延迟下降。

Lab4 使用真实 MS MARCO + Qwen3-Embedding-0.6B：

| dispatcher | search-only avg ms/query | dispatch us/query | 端到端 QPS | round cosine mean | path pairwise Jaccard mean |
|---|---:|---:|---:|---:|---:|
| random | 7.168 | 0 | 1796.50 | 0.099 | 0.00161 |
| cosine-oracle | 6.483 | 5905.51 | 407.84 | 0.401 | 0.08663 |

Oracle 分组把路径重叠显著提高：`pairwise_overlap_mean_mean` 从 `35.45` 增至 `1400.12`，`shared_node_fraction_mean` 从 `0.0142` 增至 `0.2855`。这证明真实 query 中存在可利用的相似性，但也证明精确 oracle 前处理成本远超收益。

当前 `simhash-window` 的实验结果不能直接否定 SimHash 路线。它的签名计算仅 `0.691 us/query`，但当前 order 构造消耗 `233 us/query`，且 16-bit exact signature 在 4096 窗口内过稀，`same_signature_cohort_ratio=0`。问题主要在分桶策略和实现复杂度，不在签名计算本身。

## 相关技术路线

### 动态 query reordering

DEXA 2016 的 Dynamic Query Reordering 证明，相似 query 连续执行可以提高相似检索吞吐。其方法在缓冲区内按 pivot-based cluster 选择查询，使后续查询复用 M-Index 的分区缓存。该工作适合作为“请求级 locality scheduling”的先例，但不适合直接迁移到毫秒级 HNSW：实验对象是 HDD 上的 M-Index，允许秒级到分钟级等待，中位等待时间可达几十到数百秒。

可迁移内容是有限窗口、超时、相似簇连续执行和吞吐/延迟权衡。不可迁移内容是等待预算和磁盘分区缓存收益。

### Query-aware vector cache

QVCache 与 CES 属于查询级缓存。它们利用语义相近 query 的近邻集合重叠，在缓存命中时绕过后端检索。QVCache 通过 mini-index、空间阈值和后端反馈判断缓存结果是否可信；CES 利用 shared-neighbor 信息估计 top-k。

这类机制说明 query 之间存在时间-语义局部性，也提供了滑动窗口、扰动 query、区域阈值等 workload 建模方法。但它们会改变后端执行次数，通常伴随 recall 取舍，不适合作为“所有 query 仍执行 HNSW，只改变 CCD 放置”的直接方案。

### IVF 粗路由与 coarse quantizer

IVF 系统在正式扫描前会计算 query 到 coarse centroids 的距离，选择 `nprobe` 个 list。该步骤是 ANN 系统中最典型的正式检索前粗路由机制。它的作用是减少扫描范围，不是为 HNSW 做线程调度，但可转化为 dispatcher signature：query 的 coarse centroid id 可作为 CCD routing key。

成本取决于 centroid 数、维度和是否降维。直接对 1024 维 query 与 1024 个 centroid 做精确内积过重；对 16-64 维 PCA 投影或 64-128 个 coarse centroid 做计算，才可能进入微秒到十微秒级预算。

### Pyramid 分布式粗路由

Pyramid 将 HNSW 扩展到分布式相似检索。它离线构建 `meta-HNSW`，用 `meta-HNSW` 第 0 层图做均衡划分，把相似数据放入同一个 `sub-HNSW`；查询时先搜索 `meta-HNSW`，根据 top-K meta 近邻所在分区，只访问少量 `sub-HNSW`。该方法本质是 per-query 粗路由，不是 batch 内相似 query 分组。

Pyramid 对当前问题的参考价值是：它提供了“正式 HNSW 检索前先做区域判断”的系统先例，并明确展示访问分区数与 precision、吞吐、尾延迟之间的权衡。其 `meta-HNSW=10000` 与 `meta-HNSW=100000` 的查询开销分别约为 `0.06 ms` 和 `0.18 ms`，对分布式毫秒级查询可接受，但对当前希望接近微秒级的 dispatcher 偏高。

Pyramid 不适合直接作为当前 CCD dispatcher。它需要把单个全局 HNSW 改成多个 `sub-HNSW`，会改变索引结构、查询 fanout 和 recall-latency 曲线；其收益来自减少跨机器 fanout，不是同一 CCD 内 L3 复用。只有当研究方向改成“每个 CCD 持有 region-HNSW / sub-HNSW，query 路由到少数 CCD”时，Pyramid 才应作为主要相关工作。

### MemANNS batch 任务调度

MemANNS 面向 IVFPQ 和 UPMEM PIM 硬件。在线阶段先对一个 query batch 做 IVF cluster filtering，得到每个 query 的 `nprobe` 个 cluster，再把 `query-cluster` 任务分配给持有对应 cluster 副本的 DPU。其调度复杂度为 `O(|Q| * nprobe)`，论文实验每批处理 1000 个 query。

MemANNS 对当前问题的主要价值不是 IVFPQ 或 PIM 硬件，而是 batch 负载模型和受约束任务分配。它用 `cluster workload = cluster_size * query_frequency` 建模热点，并对高工作量 cluster 建立多个副本；在线调度时先处理无副本选择的任务，再把多副本任务分配给当前负载较低的 DPU。该思想可迁移为：

```text
signature workload = recent_frequency(signature) * measured_search_cost(signature)
hot signature -> 多个 home CCD
cold signature -> sticky 到单个 CCD 或与 hot signature 共置
timeout / overload -> spill 到非 home CCD
```

不可直接迁移的边界也很明确。MemANNS 的任务粒度是 IVFPQ 的 `query-cluster`，HNSW 没有天然的 `nprobe` cluster 任务；DPU 私有 MRAM 是显式放置的本地存储，而 CCD L3 是硬件自动管理缓存。若要借鉴 MemANNS，需要先把 HNSW query 映射到轻量 signature 或 region，再用其负载模型做 CCD 队列调度，而不是直接套用 DPU 数据放置算法。

### LSH、SimHash 与随机投影

Random hyperplane LSH/SimHash 用少量随机超平面的符号位形成二进制签名。对 cosine 空间，签名 Hamming 距离与向量夹角有统计关系。在线成本为 `O(d * bits)`，适合高维 query 的极低成本分类。

该路线适合当前 dispatcher 的原因：

- 不需要访问 HNSW 图，不执行完整搜索。
- 签名计算可批量矩阵乘，Lab4 已测得 16-bit 签名约 `0.6-0.7 us/query`。
- 签名可用于 per-CCD sticky mapping、窗口分桶和多探测邻近桶。

主要风险是签名与 HNSW 路径重叠的相关性不足。当前 16-bit exact bucket 在真实 query 上过稀，应改用 8-12 bit prefix、多 probe Hamming 邻桶或 group size 小于 16 的 cohort，而不是要求 16 个 query 完全同签名。

### HNSW 上层入口 dispatcher

HNSW 标准查询本来先执行上层 greedy search，再用得到的节点作为第 0 层 `searchBaseLayer` 的入口。该中间结果可作为 HNSW 原生 routing key：

```text
query q
  -> 执行 HNSW 上层 greedy search
  -> 得到第 0 层 entry point ep(q)
  -> 按 ep(q) / entry region 分发到 CCD queue
  -> CCD worker 从 ep(q) 继续完整第 0 层搜索
```

该路线不做文档分片，不限制第 0 层搜索范围，不改变 HNSW 图结构和 recall。所有 CCD 仍访问同一个全局 HNSW 索引；CCD 只是执行域。收益目标是让相同或相近 `ep(q)` 的 query 在同一 CCD 内相邻执行，从而复用上层落点附近和第 0 层早期扩展路径中的只读节点向量。

该路线成立的前提是上层 search 结果必须被复用。若 dispatcher 先执行一次上层 search，而 worker 又调用标准 `searchKnn` 从头执行，上层计算就变成额外开销。实现上应拆出两个接口：

```text
search_upper_layers(q) -> ep
search_l0_from_entry(q, ep) -> top-k
```

离线阶段不应按 entry id 数量均分给 CCD，而应按采样 query 统计构造 entry 到 CCD 的映射：

```text
entry_workload(ep) = query_frequency(ep) * avg_l0_search_cost(ep)
```

低频 entry 可 sticky 到单个 CCD；高频 entry 应分配多个 home CCD；过载或超时 query 允许 spill。主要风险是 `ep(q)` 可能不能充分预测第 0 层真实访问路径，或 entry 分布过度倾斜导致负载不均。该路线下一步应先做 trace 验证：比较 same-entry、same-entry-region、cosine-oracle、path-oracle 与 random 的第 0 层 visited node overlap 和 search-only latency。

### CCD-aware ANNS 线程编排

2026 年 CCD-Level and Load-Aware Thread Orchestration 直接面向 AMD CCD、HNSW/IVF 与生产向量检索。其核心是 `Mapping_ID -> CCD`：同一 HNSW 表或 IVF cluster 的任务粘到同一 CCD，并用 hot-cold 配对和 CCD-aware stealing 平衡负载。

该工作是最强相关工作，但其 HNSW 场景是多表共置，`Mapping_ID` 是 table id。当前研究是单 HNSW 表内的 query 相似性调度，需要把 `Mapping_ID` 改成 query signature 或 coarse region id，并重新验证负载均衡和 cache 复用。

## 技术路线对比

| 路线 | 在线成本 | 是否改变 HNSW 结果 | 可用于 CCD dispatcher | 主要问题 |
|---|---:|---|---|---|
| 全 batch query-query exact cosine | 极高，Lab4 为 `5905 us/query` | 不改变 | 只适合 oracle | 成本远超收益 |
| Dynamic query reordering | 取决于缓冲区与聚类 | 不改变 | 概念可用 | 原论文允许长等待，不适合严格低延迟 |
| QVCache/CES | 命中判断和缓存搜索通常为亚毫秒到毫秒 | 会绕过后端或影响 recall | 不适合作为本研究主线 | 优化对象变成应用级缓存 |
| IVF coarse centroid | 低到中，取决于 centroid 数和维度 | 不改变 | 适合做 signature | 需离线训练和低维化 |
| Pyramid meta-HNSW routing | `0.06-0.18 ms/query` 量级 | 若改成 sub-HNSW 会改变 recall-latency 曲线 | 只适合分区索引架构 | 解决跨机器 fanout，不解决 query batch L3 复用 |
| MemANNS batch scheduling | `O(|Q| * nprobe)`，依赖已有 cluster filtering | 不改变 IVFPQ 语义，不适用于原生 HNSW | 可借鉴负载模型和热点复制 | 对象是 IVFPQ+DPU，不是 HNSW+CCD |
| SimHash/随机投影 | 极低，Lab4 签名约 `0.6-0.7 us/query` | 不改变 | 适合作为第一版 | 需要验证与路径重叠相关性 |
| HNSW 上层入口 dispatcher | 近零额外计算，前提是复用上层结果 | 不改变 | 当前最值得验证的 HNSW-specific 路线 | 需要拆分 search 接口，并验证 ep 是否预测 L0 路径 |
| CCD-aware sticky mapping | 极低，查表和队列操作 | 不改变 | 必需组件 | 需要处理热点与负载均衡 |

结论：不存在一个可以直接照搬、且已证明适用于毫秒级 HNSW 的微秒级 query dispatcher。可用技术应拆成两个部分：用极低成本 signature 做近似分类，用 CCD-aware queueing 做拓扑放置。真正的研究贡献在于证明这两个部分在当前 HNSW 访问路径上足够相关、成本足够低、收益能覆盖负载不均和排队。

## 推荐设计

### 总体结构

```text
query batch / request stream
    -> HNSW upper-entry or lightweight signature
    -> entry/signature-to-CCD mapper
    -> per-CCD entry/signature buckets
    -> cohort flush / timeout
    -> CCD-pinned workers
    -> result reorder
```

dispatcher 不改变 HNSW 图结构、`efSearch`、`k`、距离函数和 recall。它只改变 query 到 CCD worker group 的映射和同一 CCD 内的执行时间窗口。

### Signature

第一版建议采用 8-12 bit SimHash prefix，保留 16-bit/32-bit 完整签名作为 tie-breaker 和统计字段。原因是 16-bit exact bucket 在 65K 真实 query 中产生 29,321 个唯一签名，窗口内过稀，难以凑齐 16 个同签名 query。8-12 bit prefix 可提高桶密度，multi-probe 可把 Hamming 距离 1 的邻桶纳入同一 cohort。

备选是低维 coarse centroid：

```text
q -> PCA/random projection 到 32/64 维
  -> 与 64/128 个 centroid 做内积
  -> centroid id 或 top-2 centroid id 作为 signature
```

该方案的签名质量可能高于 SimHash，但需要离线训练和额外矩阵乘。若 SimHash 与路径重叠相关性不足，应切到该路线。

不建议第一版使用 HNSW 上层路径作为在线 signature。它可用于离线标注和评估：验证 SimHash/centroid signature 是否预测第 0 层访问重叠。

### CCD 映射

每个 signature 维护一个 preferred CCD。默认使用一致性哈希或离线 profile 得到的 `signature -> CCD` 表。运行时维护三类轻量统计：

- 每个 CCD 的队列长度和最近服务时间。
- 每个 signature 的到达率和平均 query 时间。
- 每个 CCD 上各 signature 的近期占用比例。

路由规则：

```text
若 preferred CCD 未过载:
  放入 preferred CCD
若 preferred CCD 过载且 query 未等待:
  放入该 signature 的 secondary CCD
若触发超时或空闲 CCD 较多:
  允许跨 CCD spill
```

热点 signature 不应强行固定到单个 CCD。可按到达率为热点 signature 分配 2-4 个 home CCD，避免单 CCD 排队；冷 signature 可与热点同 CCD，形成 hot-cold co-location，减少热点之间互相驱逐。

### CCD 内 cohort

每个 CCD 维护多个 signature bucket。worker 以 cohort 为单位领取 query，而不是全局 atomic row counter 单行领取。

推荐参数范围：

| 参数 | 初始值 | 作用 |
|---|---:|---|
| signature bits | 8-12 prefix | 控制桶密度 |
| cohort size | 4 或 8 | 避免强制 16 个完全相似 query |
| max wait | 0-50 us | 控制排队开销 |
| dispatch window | 256-4096 query | batch 场景下限制排序范围 |
| spill threshold | 队列时间或队列深度 | 负载均衡 |

完全相同 query 不一定是最优目标。Lab1/Lab2 中 `alpha=1` 或 `same` 存在性能回升和同步访问压力。因此 cohort 目标应是“足够路径重叠”，不是“所有线程完全相同访问序列”。

### 前处理预算

若单 query HNSW 为 `1-3 ms`，dispatcher 应控制在 `5-30 us/query` 以内；若当前真实 MS MARCO 实验约 `7 ms/query`，也应优先控制在 `50 us/query` 以内。超过 `100 us/query` 的前处理只有在 search-only 收益非常稳定且显著时才有意义。

预算分配建议：

| 组件 | 目标成本 |
|---|---:|
| signature 计算 | `< 2-5 us/query` |
| 映射查表与队列入队 | `< 1 us/query` |
| cohort 选择 | `< 1-5 us/query` |
| 重排/结果映射 | `< 1 us/query`，优先传 row index，避免复制 4 KiB query |
| 总前处理 | `< 10 us/query` 为第一目标，`< 30 us/query` 为可接受上限 |

Lab4 当前 `simhash-window` 的签名成本满足预算，但 order 构造不满足；应改成流式桶队列，而不是在窗口内反复扫描大字典和构造全 batch order。

## 负载均衡

只追求 locality 会导致热点 signature 堵塞在少数 CCD。dispatcher 需要把负载均衡作为一等约束。

推荐采用三级策略：

1. **Sticky routing**：signature 默认进入 home CCD，保持本地 L3 热集。
2. **Limited replication**：高频 signature 根据到达率分配多个 home CCD；同一 signature 不扩散到所有 CCD，避免每个 CCD 都重新缓存同一热点。
3. **Timeout spill**：query 等待超过预算时，允许送往低负载 CCD；spill 事件计数作为 locality 损失指标。

worker stealing 顺序应为：

```text
本 core 队列
  -> 同 CCD 队列
  -> 同 socket 其他 CCD
  -> 跨 socket CCD
```

跨 CCD stealing 不能禁止，否则尾延迟会被热点 signature 拉高。它应作为 SLO 保护路径，并在报告中单独统计比例。

## 实验路线

### 阶段一：signature 与路径重叠相关性

在 Lab4 trace 上评估：

- HNSW 上层 search 得到的第 0 层入口 `ep(q)` 是否预测 L0 visited node overlap；
- same-entry、same-entry-region、cosine-oracle、path-oracle 与 random 的路径重叠差异；
- query cosine、SimHash Hamming、coarse centroid 是否预测 HNSW node overlap；
- signature 相同、Hamming 距离 1、同 centroid、随机 query 的 `pairwise_jaccard_mean`；
- 不同 signature bits、PCA 维度、centroid 数下的桶密度和路径重叠。

通过条件：

```text
signature bucket 内 path overlap 明显高于随机；
bucket 足够密，能在 0-50 us 或给定 batch window 内形成 cohort；
signature 计算和入队成本在预算内。
```

### 阶段二：单 CCD dispatcher

对比：

- random；
- cosine-oracle；
- upper-entry dispatcher；
- simhash-prefix；
- centroid-signature；
- global grouped；
- per-CCD bucket grouped。

指标：

- search-only `avg_ms_per_query`；
- dispatcher `us/query`；
- end-to-end QPS；
- P50/P95/P99；
- path overlap；
- Local Cache hit、Local DRAM hit、L1 miss latency；
- result reorder 成本。

通过条件：

```text
end-to-end 指标优于 random；
收益来自 Local Cache/Local DRAM 指标变化；
dispatcher 成本低于 search-only 收益至少一个数量级。
```

### 阶段三：多 CCD 与负载均衡

对比：

- round-robin；
- global grouped；
- per-CCD sticky grouped；
- per-CCD grouped + spill；
- CCD-aware stealing。

需要报告：

- 每 CCD QPS、队列时间和空闲时间；
- signature 到 CCD 的分布；
- spill ratio；
- cross-CCD stealing ratio；
- 每 CCD Local Cache/DRAM 指标；
- P99/P999。

通过条件：

```text
per-CCD grouped 相对 global grouped 有增量；
负载不均未吞噬 locality 收益；
跨 CCD spill 控制尾延迟但不把热点扩散到所有 CCD。
```

## 风险与判停条件

1. **签名与路径重叠弱相关**：若 SimHash/centroid bucket 内路径重叠接近随机，dispatcher 不应继续优化调度实现，应换 signature 或停止该方向。
2. **前处理成本过高**：任何需要全 batch query-query kNN、全局贪心、不可复用路径预搜索或大规模字典扫描的方案都不适合作为在线 dispatcher。HNSW 上层入口 dispatcher 只有在上层结果被第 0 层搜索复用时才满足该约束。
3. **收益只在 oracle 中出现**：若只有 `cosine-oracle` 能提升 search-only，而低成本 signature 无法带来端到端收益，论文贡献会退化为 workload 现象，不是系统设计。
4. **负载均衡抵消收益**：若热点 signature 导致单 CCD 排队，或 spill 后 local cache 复用消失，需要把研究转向更粗粒度 workload 或其他应用。
5. **收益来自普通排序而非 CCD 拓扑**：若 global grouped 与 per-CCD grouped 等价，不能写成 chiplet-aware dispatcher，只能写成 query ordering。

## 最终建议

当前不应继续投入 `cosine-oracle` 作为实现路线。它只保留为上界和分析工具。

下一步应优先验证一个 HNSW-specific dispatcher，并保留两个外部 signature 方案作为后备：

1. **HNSW upper-entry dispatcher**：先拆分 HNSW 查询，复用上层 greedy search 产生的第 0 层入口 `ep(q)`，按 `ep(q)` 或 entry region 分发到 per-CCD queue。该路线不做文档分片，不限制第 0 层搜索范围，目标是用 HNSW 原生中间结果获得近零额外计算的 routing key。下一步先做 trace 验证，而不是直接实现完整 dispatcher。
2. **SimHash-prefix dispatcher**：8-12 bit prefix、Hamming multi-probe、cohort size 4/8、per-CCD FIFO bucket、超时 flush。若 upper-entry 与 L0 path overlap 相关性不足，再使用该路线。目标是把签名计算保留在微秒级，把 order 构造从当前 `233 us/query` 降到 `1-5 us/query`。
3. **Low-dimensional centroid dispatcher**：离线用采样 query 或 passage/query 混合训练 PCA + 64/128 centroid，在线计算 centroid id 作为 signature。若 SimHash 过稀或与路径重叠相关性弱，再使用该路线。目标是提高 signature 与 HNSW 路径重叠的相关性，成本控制在十微秒级以内。

报告与论文表述应保持边界清晰：核心贡献不是“query 相似度计算”，而是“在分片 L3/CCD 体系上，用极低成本 query signature 把可能共享 HNSW 节点向量的 query 共调度到同一个本地 L3 域，并在负载均衡约束下保持端到端收益”。

## 参考资料

- `/home/zjj/HNSW/设计/HNSW设计.md`
- `/home/zjj/HNSW/实验结果分析/实验1结果分析.md`
- `/home/zjj/HNSW/实验结果分析/实验2总结.md`
- `/home/zjj/HNSW/实验结果分析/实验3总结.md`
- `/home/zjj/HNSW/实验/实验4.md`
- `/home/zjj/HNSW/Lab4/timing/4.1_1ccd/*.json`
- `/home/zjj/HNSW/Lab4/overlap/4.1_1ccd/*/measure_0_summary.json`
- `wiki/论文解读/HNSW/HNSW论文总览.md`
- `wiki/论文解读/HNSW/Pyramid_Distributed_Similarity_Search解读.md`
- `wiki/论文解读/HNSW/MemANNS解读.md`
- `sources/search_20260603_hnsw_dispatcher_low_cost_routing.md`
