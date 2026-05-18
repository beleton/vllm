# LSM 与 HTAP 的 Chiplet 研究方向

## 结论

`LSM/HTAP` 适合作为数据库系统备选方向。它的优势是阶段边界清晰：flush、compaction、merge、scan、join、delta merge 的工作集大小和复用距离不同，可迁移 sorting 与 OLAP 的 `Local/Mixed` 方法线。它的风险是工程面较大，且离当前 RAG/LLM 应用链路较远。

推荐研究点是 **phase-aware Local/Mixed 切换**：小工作集和短状态阶段收紧到单 `CCD`，中等工作集扩散到多个 `CCD` 利用聚合 L3，大流式阶段转向 DRAM bandwidth 与 NUMA locality。

## 场景

`LSM`（Log-Structured Merge-tree，日志结构合并树）用于 RocksDB、Pebble 等存储引擎。在线写入进入 `MemTable`，随后 flush 为 sorted run，再通过 compaction 合并。

`HTAP`（Hybrid Transactional/Analytical Processing，混合事务/分析处理）同时服务事务更新和分析查询，常见结构包括 delta/main 分层、冷热分区、增量视图和后台 merge。

两类系统都包含多阶段数据搬运和排序/合并过程，天然适合按阶段改变拓扑策略。

## Chiplet 可优化性

### 阶段边界清晰

| 阶段 | 访存特征 | 推荐策略 |
| --- | --- | --- |
| `MemTable` 写入 | 小对象、热点索引、锁/跳表/哈希表 | `Local` |
| flush | 中小规模排序、编码、Bloom/filter build | `Local` 或小范围 `Mixed` |
| small compaction | 多个小 run 合并，工作集可能接近 32 MiB | `Local` |
| large compaction | 多路 merge，输入远超 L3 | `Mixed` 或 NUMA bandwidth 优先 |
| range scan | 流式读，复用弱 | DRAM/NUMA 优先 |
| HTAP delta merge | 热 delta + 冷 main 合并 | 按 delta/main 工作集切换 |
| analytical join/agg | 局部 hash table 或中间状态 | `Local/Mixed` 按容量判断 |

### 与已有论文的继承关系

- sorting 证明多轮 pass 和分区缓冲对 `CCD` L3 敏感。
- OLAP 证明 worker 边界与 chiplet 对齐能减少隐式跨 chiplet 共享。
- P-MOSS 证明主内存索引可用分片、路由和 PMU 反馈做空间调度。

`LSM/HTAP` 的新点是：同一个系统中同时存在多个阶段，最优策略不是固定 `Local` 或固定 `Mixed`，而是按阶段切换。

## 研究点一：Compaction Local/Mixed 切换

### 思路

对 compaction task 估算输入 run、输出 buffer、Bloom/filter、iterator state 和 merge heap 的活跃工作集。若活跃集小于单 `CCD` L3，绑定到单 `CCD`；若介于单 `CCD` 与聚合 L3 之间，分散到多个 `CCD`；若远超聚合 L3，优先 DRAM bandwidth 和 NUMA locality。

### 设计变量

| 变量 | 说明 |
| --- | --- |
| task size | 输入 run 总量、输出 SST 大小 |
| active set | merge heap、block cache、filter/index block、write buffer |
| core placement | 单 `CCD`、多 `CCD`、socket、NPS domain |
| memory placement | first-touch、membind、interleave |
| scheduling | foreground read/write 与 background compaction 是否隔离 |

### 实验方案

- 微基准：独立 compaction/merge task。
- 系统基准：RocksDB `db_bench`。
- baseline：默认调度、socket 绑核、NUMA interleave、固定 all-core。
- 方案：按 compaction size 和阶段切换 `Local/Mixed`。
- 指标：
  - compaction throughput
  - foreground read/write p99
  - write stall time
  - `L3 miss/MB`
  - DRAM bandwidth
  - `another CCD` 来源占比

## 研究点二：HTAP Delta Merge 拓扑感知

### 思路

HTAP 中热 delta 和冷 main 的访问模式不同。delta 较小、更新频繁，适合局部化；main 较大、分析 scan 多，适合利用更大并行和带宽。delta merge 是两个层次相遇的阶段，适合动态切换。

```text
transactional delta: Local
analytical scan: Mixed / bandwidth
delta merge: size-aware Local/Mixed
```

### 设计变量

- delta 大小
- main partition 大小
- merge batch
- 分区热度
- 是否复制 hot metadata
- scan 与 update 是否隔离到不同 `CCD`

### 实验方案

- 构造小型 HTAP 微基准：point update + scan + periodic merge。
- 对比默认线程池、NUMA-aware、CCD-aware phase switching。
- 指标：transaction p99、scan throughput、merge stall、L3/DRAM 指标。

## 研究点三：PMU 反馈调度

### 思路

不使用 P-MOSS 的离线 `Decision Transformer`，先做轻量阈值控制：

- `L3 miss/op` 高且 `another CCD` 高：收紧到 `Local` 或复制热结构。
- `Local` 下 L3 miss 高、DRAM bandwidth 未满：切到 `Mixed`。
- owner queue 过载：允许 bounded stealing。
- scan/compaction 大任务：转 bandwidth 优先。

### 价值

该方向可把数据库系统的 phase metadata 与硬件计数器结合，形成比静态规则更强的自适应策略。它也能作为 HNSW/IVF 的通用调度器基础。

## 预期优化空间

对 compaction/merge 微基准，若工作集处于 `32-256 MiB` 边界，可能出现 `10%-30%` 的阶段级改善。端到端 RocksDB/HTAP 受前台请求、I/O、压缩和锁影响，整体收益可能降到 `3%-15%`。若瓶颈在 SSD 或压缩算法，chiplet 放置收益很低。

## 风险

- `LSM` 大量路径受存储 I/O、压缩、block cache、锁和 write stall 共同影响，归因难。
- background compaction 改善不一定转化为 foreground p99。
- `CCD` 级内存放置不可直接控制，只能做 `NPS/socket` 级页面放置和 `CCD` 级绑核。
- 系统工程量明显高于 HNSW 微基准。

## 判停条件

- compaction 微基准对 CAT/L3 容量不敏感。
- 默认 RocksDB 已被 I/O 或压缩主导。
- `Local/Mixed` 改善阶段指标但前台 p99 不变。
- 调度引入额外排队或 CPU under-utilization。

## 参考资料

- P-MOSS：https://arxiv.org/abs/2411.02933
- OLAP on Modern Chiplet-Based Processors：https://www.vldb.org/pvldb/vol17/p3428-fogli.pdf
- Optimizing Sorting for Chiplet-Based CPUs：https://vldb.org/workshops/2024/proceedings/ADMS/ADMS24_03.pdf
- AHA-Tree：https://arxiv.org/abs/2406.08746
- Quake：https://arxiv.org/abs/2506.03437
- Bandwidth-Aware Page Placement：https://arxiv.org/abs/2003.03304
