# IVF 批量检索的 Chiplet 研究方向

## 结论

`IVF`（Inverted File Index，倒排文件索引）只有在 batch 场景下才适合作为 chiplet 研究方向。单 query IVF 的倒排表扫描接近流式读，缺少 L3 复用；多个 query 若共享 coarse cluster 或 inverted list，才有机会通过 query grouping 和 per-CCD list ownership 放大本地 L3/NUMA 局部性。

推荐研究点是 **cluster-aware query grouping + per-CCD inverted-list ownership**。它可作为 HNSW 主线的并列子方向，适合 RAG serving 的 batch 检索阶段。

## IVF 检索流程

IVF 通常包含两步：

1. 用 coarse quantizer 找到 query 最近的 `nprobe` 个 cluster/list。
2. 扫描这些 inverted lists 中的向量或压缩码，计算距离并返回 top-k。

关键参数：

| 参数 | 含义 |
| --- | --- |
| `nlist` | coarse cluster / inverted list 数量 |
| `nprobe` | 每个 query 扫描的 list 数 |
| `max_codes` | 最多扫描的向量码数量 |
| batch size | 同时处理的 query 数 |

## Chiplet 可优化性

### 单 query 不适合

单 query IVF 的 list 内扫描通常是顺序访问。每个向量码或向量元素被读一次，L3 主要作为 DRAM 到 SIMD 的通道。即使将线程固定到某个 `CCD`，也只是改变流式扫描路径，难以形成稳定收益。

### batch 场景可能适合

batch query 中，不同 query 可能命中相同或相近的 inverted lists。若这些 query 被分散到不同 `CCD`，相同 list 会在多个 L3 中重复加载；若把共享 list 的 query 放到同一 `CCD`，list 的向量码、PQ code、id array 和局部 top-k buffer 可能获得短时间复用。

该模式与 GPU attention 的 `ACC` 类似：

```text
GPU attention: 共享同一组 K/V 的 workgroups -> 同一拓扑域
IVF batch:     共享同一批 inverted lists 的 queries -> 同一 CCD queue
```

## 研究点：Cluster-Aware Query Grouping

### 思路

先执行 coarse quantizer，得到每个 query 的 top-`nprobe` lists。调度器根据 list overlap 把 query 分组，再将组路由到对应 `CCD` queue。

```text
query batch
  -> coarse quantizer
  -> list signature = top-nprobe list ids
  -> group by overlap / home list
  -> execute group on owner CCD
  -> merge results
```

### 设计变量

| 变量 | 说明 |
| --- | --- |
| signature | top-1 centroid、top-nprobe set、weighted list vector |
| grouping window | `10-100 us` 或固定 batch |
| owner policy | list hash 到 `CCD`、热 list 独占、热 list 多副本 |
| stealing | owner queue 过载时是否允许其他 `CCD` 扫描 |
| list layout | 每个 owner 的 list 连续存放，first-touch 到 owner NUMA/NPS |
| list replica | 高频 list 复制到多个 `CCD` |

## 研究点：Per-CCD Inverted-List Ownership

### 思路

每个 inverted list 有 home `CCD`。list 的向量码、id array、PQ residual、metadata 由 home `CCD` 的线程优先访问。query 被路由到覆盖其主要 list 的 owner。

对于命中多个 owner 的 query，有三种策略：

1. **query-to-primary**：选择权重最高的 list owner，本线程远端扫描其他 list。
2. **split execution**：多个 owner 各自扫描本地 list，返回局部 top-m，再合并。
3. **hot-list replica**：热门 list 多副本，减少跨 owner 扫描。

### 与 HNSW 的差异

IVF 的数据单元是 list，边界稳定；HNSW 的数据单元是图节点和 query path，边界动态。IVF 更容易做 ownership，但也更容易退化成流式 DRAM 扫描。它的收益依赖 batch overlap，而不是图路径局部性。

## 理论收益模型

设一个 batch 中有 `B` 个 query，每个 query 扫描 `nprobe` 个 lists。若不分组，同一 list 被 `r` 个 query 命中但分散在 `g` 个 `CCD` 上，则该 list 的数据可能被加载 `g` 次。分组后若落在同一 `CCD`，加载次数接近 1 次。

收益上界约束：

```text
saved_bytes ~= sum_hot_lists (replicated_loads_before - loads_after) * list_bytes
```

实际收益还要扣除：

- coarse quantizer 和 grouping 开销
- queue 等待
- list owner 负载不均
- 结果合并开销
- 热 list 复制内存成本

## 实验方案

### 数据集

- SIFT / GIST / DEEP 子集
- 企业文档 embedding
- 可控合成数据：调整 cluster skew 与 overlap

### Baseline

| baseline | 含义 |
| --- | --- |
| FCFS batch | 按到达顺序处理 |
| random thread placement | 默认线程池 |
| hash partition by query id | 与 locality 无关的分组 |
| cluster grouping | 按 centroid/list overlap 分组 |
| per-CCD ownership | list owner + query routing |

### 指标

- `QPS`
- `p50/p95/p99`
- `recall@k`
- grouping wait time
- list overlap ratio
- `L3 miss/query`
- `another CCD` 来源占比
- owner queue imbalance

### 变量 sweep

- batch size：`1/4/8/16/32/64`
- `nprobe`
- `nlist`
- grouping window
- Zipf skew
- list replica budget
- `NPS1/NPS4`

## 预期优化空间

高 overlap、低维/PQ code、in-memory IVF、batch 检索下，合理预期 p99 改善 `5%-20%`。如果 query overlap 很低，或 list 扫描完全由 DRAM bandwidth 主导，收益会低于 `5%`。

## 风险

- 等待分组增加 tail latency。
- 热门 cluster 形成单 `CCD` 队列热点。
- 若 `nprobe` 大，query 涉及多个 owner，split execution 合并成本上升。
- 若 list 体积远超 L3，分组只能减少重复 DRAM 读取，不能保证 L3 命中。
- 与 FAISS 现有 big-batch IVF 优化的增量需要单独证明。

## 判停条件

- batch 内 list overlap 低。
- cluster grouping 后 p99 上升。
- owner imbalance 明显高于 locality 收益。
- `L3 miss/query` 和 `another CCD` 指标改善不能转化为 wall time。

## 参考资料

- Faiss indexes：https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- Faiss performance tips：https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster
- CaGR-RAG：https://arxiv.org/abs/2505.01164
- RAGO：https://arxiv.org/abs/2503.14649
