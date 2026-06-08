# HNSW 路径 trace 记录与重叠指标解析

## 范围

本文说明当前 `/home/zjj/hnswlib` 与 `/home/zjj/HNSW` 实验代码中的 HNSW 路径 trace 口径，覆盖：

- `knn_query_fixed_thread_rounds(..., trace_path=True)` 如何记录每个 query 的访问路径。
- `trace_offsets`、`trace_node_ids`、`trace_round_ids`、`trace_thread_ids`、`trace_query_rows` 的含义。
- `/home/zjj/HNSW/scripts/analyze_hnsw_path_trace.py` 如何把 trace 解析为 round 级路径重叠指标。
- CSV 与 JSON 汇总字段的含义和使用边界。

本文只覆盖当前修改版 hnswlib 的 trace 路径，不覆盖 upstream hnswlib 的默认接口。

## 核心结论

- trace 记录的是 HNSW 查询过程中进入距离计算的 internal node id 序列。
- `trace_node_ids` 是原始记录序列，允许同一个 node id 在同一个 query 中出现多次。
- 解析脚本计算路径重叠时，会先把每个 query 的 `trace_node_ids` 转成 `set`，即按 query 内部去重后的节点集合计算重叠。
- `distance_call_count` 使用原始序列长度，反映距离计算调用数量；`unique_nodes_*` 使用去重集合大小，反映单 query 实际覆盖的节点规模。
- `common_node_count`、`shared_node_count`、`pairwise_overlap_*` 和 `pairwise_jaccard_*` 都是同一 fixed round 内多个 query 的访问集合重叠指标。
- `*_mean` 结尾的 JSON 字段是先得到每个 round 的指标，再对所有 round 求平均。

## trace 入口

Lab1 与 Lab2 的实验入口通过 `--trace-path` 启用路径记录：

```text
lab2.py --scheduler fixed-rounds --trace-path
  -> run_fixed_thread_rounds(...)
  -> index.knn_query_fixed_thread_rounds(..., trace_path=True)
  -> hnswlib Python binding
  -> HierarchicalNSW::searchKnnWithTrace(...)
```

`lab2.py` 保存 trace 的字段如下：

```python
trace_offsets, trace_node_ids, trace_round_ids, trace_thread_ids, trace_query_rows = trace_result[4:9]

np.savez_compressed(
    out_path,
    labels=labels,
    distances=distances,
    round_times=round_times,
    search_times=search_times,
    trace_offsets=trace_offsets,
    trace_node_ids=trace_node_ids,
    trace_round_ids=trace_round_ids,
    trace_thread_ids=trace_thread_ids,
    trace_query_rows=trace_query_rows,
)
```

`labels`、`distances`、`round_times`、`search_times` 是查询结果和时间字段。路径重叠解析只使用 `trace_offsets`、`trace_node_ids`、`trace_round_ids`、`trace_thread_ids`、`trace_query_rows`。

## fixed-rounds 调度关系

`knn_query_fixed_thread_rounds(...)` 使用固定轮次同步。每个 round 中，每个 worker 线程处理一个 query。

当 `same_query=False` 时：

```text
rounds      = rows / num_threads
output_rows = rows
query_row   = round * num_threads + threadId
result_row  = query_row
```

含义：

- 一个 `round` 包含 `num_threads` 个 query。
- `threadId` 是该 round 内执行 query 的固定线程编号。
- `query_row` 是输入 query 矩阵中的行号。
- `result_row` 是输出数组和 trace 数组中的行号。

Lab2 的 `similar` 与 `different` workload 主要依赖 fixed round 结构表达“同一轮 16 个 query 是否来自相同 topic 或不同 topic”。

## Python binding 中的记录方式

`python_bindings/bindings.cpp` 在每个 query 执行前为当前 `result_row` 准备一个 `trace_node_ids` 向量：

```cpp
std::vector<hnswlib::tableint>* trace_node_ids = nullptr;
if (trace_path) {
    trace_node_ids = &trace_nodes_per_result[result_row];
    trace_round_ids[result_row] = static_cast<uint64_t>(round);
    trace_thread_ids[result_row] = static_cast<uint64_t>(threadId);
    trace_query_rows[result_row] = static_cast<uint64_t>(query_row);
}

std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result =
    appr_alg->searchKnnWithTrace(query_data, k, nullptr, trace_node_ids);
```

每个 `result_row` 对应一个 query 的完整 HNSW 搜索。该 query 的路径节点临时存放在：

```text
trace_nodes_per_result[result_row]
```

所有 query 完成后，binding 把二维的 `vector<vector<tableint>>` 压平成 NumPy 数组：

```text
trace_offsets[row]     = 当前 query 在 trace_node_ids 中的起始位置
trace_offsets[row + 1] = 当前 query 在 trace_node_ids 中的结束位置
trace_node_ids         = 所有 query 的 trace 序列拼接
```

因此，第 `row` 个 query 的原始访问序列为：

```python
begin = trace_offsets[row]
end = trace_offsets[row + 1]
nodes = trace_node_ids[begin:end]
```

## HNSW 内部记录点

### 入口节点

`searchKnnWithTrace(...)` 开始时记录全局入口节点 `enterpoint_node_`，随后立即计算 query 到入口节点的距离：

```cpp
tableint currObj = enterpoint_node_;
if (trace_node_ids) {
    trace_node_ids->push_back(enterpoint_node_);
}
dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
```

该记录表示入口节点参与了距离计算。

### 上层 greedy search

从 `maxlevel_` 到第 1 层，HNSW 执行贪婪搜索。当前层的每个邻居候选 `cand` 在计算距离前被记录：

```cpp
for (int i = 0; i < size; i++) {
    tableint cand = datal[i];
    if (trace_node_ids) {
        trace_node_ids->push_back(cand);
    }
    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

    if (d < curdist) {
        curdist = d;
        currObj = cand;
        changed = true;
    }
}
```

上层 greedy search 没有使用第 0 层的 `visited_array` 去重机制。同一个 internal node id 可能通过不同邻接表再次出现，并被再次记录。

### 第 0 层入口

进入第 0 层 `searchBaseLayerST(...)` 后，函数会先记录入口 `ep_id`，并计算该节点距离：

```cpp
if (trace_node_ids) {
    trace_node_ids->push_back(ep_id);
}
dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
```

`ep_id` 通常是上层 greedy search 结束时的 `currObj`。如果该节点在上层已经计算过距离，trace 中会再次出现同一个 node id。

### 第 0 层候选节点

第 0 层搜索使用 `visited_array` 记录当前 query 已访问节点。只有未访问过的 `candidate_id` 才会被记录并计算距离：

```cpp
if (!(visited_array[candidate_id] == visited_array_tag)) {
    visited_array[candidate_id] = visited_array_tag;

    char *currObj1 = getDataByInternalId(candidate_id);
    if (trace_node_ids) {
        trace_node_ids->push_back(candidate_id);
    }
    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
    ...
}
```

因此，在第 0 层内部，同一个 query 对同一个 `candidate_id` 通常只记录一次。重复主要来自：

- 入口节点先在上层被计算，又作为第 0 层入口被计算。
- 上层 greedy search 中同一节点通过不同路径再次作为候选出现。
- 同一 internal node id 同时存在于多个 HNSW 层，跨层距离计算会重复记录。

## trace 字段含义

| 字段 | 维度 | 含义 |
| --- | --- | --- |
| `trace_offsets` | `output_rows + 1` | 每个 query 在 `trace_node_ids` 中的起止偏移 |
| `trace_node_ids` | `trace_total` | 所有 query 的原始距离计算节点序列拼接 |
| `trace_round_ids` | `output_rows` | 每个 query 所属 fixed round |
| `trace_thread_ids` | `output_rows` | 每个 query 的执行线程编号 |
| `trace_query_rows` | `output_rows` | 每个 query 在输入 query 矩阵中的行号 |

`trace_node_ids` 中存的是 hnswlib internal id。当前 Lab1/Lab2 构建索引时使用：

```python
ids = np.arange(num_elements)
```

因此当前实验中 internal id 与 external label 可直接对应。若以后改用非连续 external ids，该等价不再成立。

## 解析入口

Lab2 批量解析命令：

```bash
python /home/zjj/HNSW/scripts/analyze_hnsw_path_trace.py parse-lab2 \
    --root /home/zjj/HNSW/Lab2/topic_alpha_query \
    --jobs 16
```

脚本扫描：

```text
<root>/trace/alpha_*/*/measure_*.npz
```

并输出：

```text
<root>/overlap/alpha_<alpha>/<mode>/measure_<n>_round_overlap.csv
<root>/overlap/alpha_<alpha>/<mode>/measure_<n>_summary.json
```

其中：

- CSV 是每个 fixed round 一行。
- JSON 是对 CSV 中所有 round 的数值字段求平均。
- `metadata` 会把 `alpha`、`mode`、`measure` 写入 JSON。

## query 级节点集合

解析脚本先从压平数组恢复每个 query 的节点集合：

```python
def _query_node_sets(trace_offsets, trace_node_ids):
    sets = []
    lengths = []
    for row in range(len(trace_offsets) - 1):
        begin = int(trace_offsets[row])
        end = int(trace_offsets[row + 1])
        nodes = trace_node_ids[begin:end]
        sets.append(set(_as_int_list(nodes)))
        lengths.append(end - begin)
    return sets, lengths
```

这里同时保留两个口径：

- `lengths[row] = end - begin`：原始 trace 序列长度，包含重复记录。
- `sets[row] = set(nodes)`：单个 query 内部去重后的节点集合。

`distance_call_count` 使用 `lengths`，路径重叠指标使用 `sets`。

## round 级分组

脚本按 `trace_round_ids` 把 query 分组：

```python
round_to_rows = defaultdict(list)
for row, round_id in enumerate(trace_round_ids):
    round_to_rows[int(round_id)].append(row)
```

每个 round 内的 query 集合为：

```python
rows = round_to_rows[round_id]
node_sets = [all_sets[row] for row in rows]
```

后续所有重叠指标都在这个 `node_sets` 列表上计算。

## round 级指标

### query 标识字段

| 字段 | 含义 |
| --- | --- |
| `round_id` | fixed round 编号 |
| `query_count` | 该 round 内 query 数量，通常等于线程数 |
| `query_rows` | 该 round 内 query 的输入行号，以分号拼接 |
| `thread_ids` | 该 round 内执行线程编号，以分号拼接 |

### 搜索规模字段

| 字段 | 计算方式 | 含义 |
| --- | --- | --- |
| `distance_call_count` | `sum(call_lengths[row] for row in rows)` | 该 round 内所有 query 原始 trace 长度总和，近似表示距离计算调用次数 |
| `unique_nodes_min` | `min(len(nodes) for nodes in node_sets)` | 该 round 内单 query 去重访问节点数的最小值 |
| `unique_nodes_mean` | `mean(len(nodes) for nodes in node_sets)` | 该 round 内单 query 去重访问节点数的平均值 |
| `unique_nodes_max` | `max(len(nodes) for nodes in node_sets)` | 该 round 内单 query 去重访问节点数的最大值 |

`unique_nodes_*` 中的 “unique” 指单个 query 的访问序列去重，不表示该节点只被一个 query 访问。

### 集合级共享字段

设同一 round 内第 \(i\) 个 query 的去重节点集合为 \(S_i\)。

| 字段 | 计算方式 | 含义 |
| --- | --- | --- |
| `union_node_count` | \(\left|\bigcup_i S_i\right|\) | 该 round 内所有 query 形成的总节点工作集大小 |
| `common_node_count` | \(\left|\bigcap_i S_i\right|\) | 被该 round 内所有 query 都访问到的节点数量 |
| `common_node_fraction` | `common_node_count / union_node_count` | 全体共同节点在总工作集中的比例 |
| `shared_node_count` | `count(node where occurrence_round >= 2)` | 被该 round 内至少 2 个 query 访问到的节点数量 |
| `shared_node_fraction` | `shared_node_count / union_node_count` | 至少两 query 共享节点在总工作集中的比例 |

`common_node_count` 是最严格的共享口径。只要某个节点没有被 round 内任意一个 query 访问到，就不计入 common。

`shared_node_count` 是较宽的共享口径。节点只要被两个或更多 query 访问到，就计入 shared。

只被一个 query 访问到的节点数可由以下公式得到：

```text
private_node_count = union_node_count - shared_node_count
```

### 两两重叠字段

脚本枚举同一 round 内所有 query pair：

```python
for i in range(len(node_sets)):
    for j in range(i + 1, len(node_sets)):
        inter = len(node_sets[i] & node_sets[j])
        union = len(node_sets[i] | node_sets[j])
        overlaps.append(inter)
        jaccards.append(inter / union if union else 0.0)
```

| 字段 | 计算方式 | 含义 |
| --- | --- | --- |
| `pairwise_overlap_min` | `min(|S_i ∩ S_j|)` | 该 round 内最不重叠 query pair 的绝对交集规模 |
| `pairwise_overlap_mean` | `mean(|S_i ∩ S_j|)` | 该 round 内典型 query pair 的绝对交集规模 |
| `pairwise_overlap_max` | `max(|S_i ∩ S_j|)` | 该 round 内最重叠 query pair 的绝对交集规模 |
| `pairwise_jaccard_min` | `min(|S_i ∩ S_j| / |S_i ∪ S_j|)` | 该 round 内最不相似 query pair 的相对重叠比例 |
| `pairwise_jaccard_mean` | `mean(|S_i ∩ S_j| / |S_i ∪ S_j|)` | 该 round 内典型 query pair 的相对重叠比例 |
| `pairwise_jaccard_max` | `max(|S_i ∩ S_j| / |S_i ∪ S_j|)` | 该 round 内最相似 query pair 的相对重叠比例 |

当一个 round 中 query 数量小于 2 时，pairwise 指标返回 0。Lab2 fixed-rounds 正常情况下每轮有 16 个 query，不触发该退化口径。

## summary JSON 字段

`summarize_rows(rows)` 对 CSV 中的数值字段求均值：

```python
summary = {"round_count": len(rows)}
for key in numeric_keys:
    values = [row[key] for row in rows]
    summary[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
```

因此 JSON 中：

```text
distance_call_count_mean
unique_nodes_mean_mean
union_node_count_mean
pairwise_jaccard_mean_mean
```

含义分别是：

- `distance_call_count_mean`：每个 round 的 `distance_call_count` 再跨 round 求平均。
- `unique_nodes_mean_mean`：每个 round 内单 query 唯一节点数的平均值，再跨 round 求平均。
- `union_node_count_mean`：每个 round 的总节点工作集大小，再跨 round 求平均。
- `pairwise_jaccard_mean_mean`：每个 round 内 query pair Jaccard 的平均值，再跨 round 求平均。

字段中出现两次 `mean` 时，第一个 `mean` 来自 round 内统计，第二个 `mean` 来自跨 round 汇总。例如：

```text
pairwise_overlap_mean_mean
= mean_over_rounds(mean_over_query_pairs(|S_i ∩ S_j|))
```

## 指标解读

### 路径规模

`distance_call_count_mean` 和 `unique_nodes_mean_mean` 用于判断搜索本身的工作量：

- `distance_call_count_mean` 高，说明每轮距离计算调用多。
- `unique_nodes_mean_mean` 高，说明单个 query 覆盖的去重节点多。
- 二者差距反映重复距离计算数量，但当前重复主要来自跨层和上层 greedy search，不应直接解释为第 0 层重复访问。

### round 工作集

`union_node_count_mean` 表示同一 round 内所有 query 合起来覆盖多少不同节点。它接近该 round 形成的 HNSW 节点工作集大小。

在 fixed round 中，若 `union_node_count_mean` 远大于单 query 的 `unique_nodes_mean_mean`，说明同轮多个 query 的路径分散。若二者接近，说明多个 query 访问集合高度重叠。

### 强共享路径

`common_node_count_mean` 和 `common_node_fraction_mean` 只统计被所有 query 共同访问的节点。该指标适合判断是否存在稳定的全体共享入口路径或公共热点路径。

该口径很严格。对于 16 query 一轮，只要某个节点缺席任意一个 query，就不会计入 common。

### 弱共享路径

`shared_node_count_mean` 和 `shared_node_fraction_mean` 统计至少被两个 query 访问的节点。该指标适合衡量 cache 复用潜力：

- `shared_node_fraction_mean` 高，说明总工作集中有较大比例节点可能被同轮不同 query 复用。
- `shared_node_fraction_mean` 低，说明绝大多数节点只被单个 query 访问，同轮共享访问弱。

该指标只说明节点 id 重叠，不直接等价于 cache hit。cache hit 还受访问时间间隔、节点数据布局、cache 容量、替换策略和硬件预取影响。

### 两两路径相似度

`pairwise_overlap_mean_mean` 是绝对交集规模，适合回答“典型两个 query 共享多少个节点”。

`pairwise_jaccard_mean_mean` 是相对重叠比例，适合跨不同搜索规模比较路径相似度：

```text
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

当两个 workload 的单 query 搜索规模不同，仅比较 `pairwise_overlap_mean_mean` 可能受路径长度影响。此时需要同时看 `pairwise_jaccard_mean_mean`。

## 与 cache 复用分析的关系

这些指标用于描述同一 round 内 query 访问 HNSW 节点的集合重叠：

- 路径重叠高，说明不同 query 访问同一批 HNSW 节点的概率高，具备 cache 复用前提。
- 路径重叠低，说明不同 query 的节点工作集分散，依赖同轮 query co-location 获得 cache 复用的空间较小。

这些指标不能单独证明硬件 cache 复用已经发生。需要结合：

- `avg_ms_per_query`
- `InnerProductSIMD16ExtAVX512` 的 Local Cache hit / Local DRAM hit
- L3 miss 或 L3 occupancy
- 同一 `alpha, mode` 下的 timing 与 PMU 对齐结果

## 口径限制

- trace 采集会增加开销，不用于报告查询延迟；正式测速应关闭 `--trace-path`。
- trace 只记录进入距离计算的 node id，不记录邻接表读取、候选堆操作、visited bitmap 访问和数据 cache line 地址。
- `trace_node_ids` 记录 internal id，不记录节点向量的 cache line 粒度访问。
- `shared_node_fraction` 衡量节点集合共享，不等价于 cache-line 共享。
- 当前解析按 fixed round 分组，适用于 `knn_query_fixed_thread_rounds` 生成的同步 round；普通 `knn_query` 的动态调度结果不适合直接套用该 round 口径。
