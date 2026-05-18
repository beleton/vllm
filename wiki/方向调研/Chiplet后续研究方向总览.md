# Chiplet 后续研究方向总览

生成时间：2026-05-14

## 目的

当前 `CPU attention + acc-local-l3` 已被实验排除。后续研究需要重新选择应用场景，要求满足三个条件：

1. 跨 `CCD/chiplet` 访问或共享资源争用进入关键路径。
2. 数据、请求、任务或共享对象存在可控放置变量。
3. 优化收益可通过复制、分片、路由、迁移、隔离或阶段切换获得，且代价可摊销。

本文按上述条件筛选 attention 之外的候选方向，并给出优先级。

## 结论

最适合作为下一主线的是 **RAG/向量检索中的 HNSW 图检索**。该方向同时具备只读索引、层次化热点结构、随机图访问、请求路径重叠、可控图布局和可测 `L3/NUMA` 指标。它与已有 `CHARM` 图计算、`P-MOSS` 主内存索引调度、`OLAP WICP`、sorting `Local/Mixed` 方法线相关，但不直接照搬任何一篇：优化对象变成 `ANN`（Approximate Nearest Neighbor，近似最近邻）检索中的 `HNSW` 层次图和 query-dependent search path。

第二优先级是 **IVF batch 检索的 query grouping 与倒排表放置**。它依赖批量请求中共享 coarse cluster 或 inverted list，适合服务端 batch/p99 场景，不适合作为单 query latency 主线。

第三优先级是 **LSM/HTAP 的 phase-aware Local/Mixed 切换**。该方向和 sorting/OLAP 继承关系清楚，但工程面更偏数据库系统，离当前 RAG/LLM 应用链路较远。

网络 I/O、RPC、KV serving 和 Agent tool sandbox 都有可做空间，但更适合作为系统型备选。网络方向不能再以 TiNA 的 Remote-tier tradeoff 作为 AMD Zen 动机；Zen 上远端 `CCD` L3/cache 路径接近本地 DRAM，远端 cache 容量套利不成立。可保留的是 queue/worker/mempool 分区、RPC tail 隔离和本地热元数据复制。

除向量检索、ToolSandbox、网络 I/O 外，新增值得关注的 Zen 场景包括推荐系统 embedding lookup、GNN neighbor sampling、HPC/CFD/FEA/stencil、EDA/RTL/离散事件仿真、genomics k-mer counting。它们共同依赖本地 `CCD` L3 热集复用，而不是远端 L3 快于 DRAM。

CPU 使用基础：

- `HNSW`：CPU 内存索引是常见部署路径。hnswlib、Faiss CPU HNSW 和多种向量数据库均支持 CPU 上的 HNSW 检索。该方向不是只为 chiplet 构造的 synthetic workload。
- `Embedding lookup`：推荐系统的大 embedding table 查表在 CPU 或 CPU+GPU 混合系统中常见。原因是表规模大、访问稀疏、容量压力高，CPU 内存容量和多线程 embedding bag kernel 仍有实际价值。
- `GNN sampling`：常见形态是 CPU 负责 neighbor sampling、feature loading、mini-batch 构造，GPU 负责 GNN layer compute。CPU chiplet 优化主要作用在采样和数据管线，不应默认声称纯 CPU GNN 训练是主流。

## 优先级

| 优先级 | 方向 | 推荐结论 |
| --- | --- | --- |
| P0 | `HNSW` 图检索的热层复制、冷图分区与 query trace 布局 | 主线 |
| P1 | `IVF` batch query grouping 与 per-CCD inverted-list ownership | batch 检索子方向 |
| P2 | `LSM/HTAP` phase-aware `Local/Mixed` 切换 | 数据库系统备选 |
| P3 | 推荐系统 embedding lookup 的 hot row 复制与 request grouping | AI serving 新候选 |
| P4 | GNN neighbor sampling 与图特征缓存 | 图/AI 数据管线新候选 |
| P5 | HPC/CFD/FEA/stencil 的 CCD 级块调度 | HPC 备选 |
| P6 | EDA/RTL/离散事件仿真的 process/event queue 分片 | 系统仿真备选 |
| P7 | 网络 I/O / RPC / KV serving 的 queue-worker-mempool 分区与 tail spill | 系统备选，不含 TiNA Remote-tier |
| P8 | Agent tool sandbox 的共享 OS 路径分片与 session/page-cache 亲和 | 多租户系统备选 |
| 排除 | `Flat` 向量检索、单 query IVF、LLM attention/GEMM/decode | 不作为 chiplet 主线 |

## 筛选框架

### 可优化性判据

| 判据 | 必须回答的问题 | 失败判定 |
| --- | --- | --- |
| 开销归因 | `another CCD/CCX`、远端 NUMA 页、共享 cache line、锁争用、L3 容量 miss 中哪一项进入关键路径 | 只能作为现象观察 |
| 放置可控 | 是否能控制线程、数据页、索引分区、请求队列、worker pool 或共享 OS 对象的位置 | 不能形成 chiplet-aware 优化 |
| 替代路径 | 是否能用本地复制、分片、远端执行、请求分组、显式 exchange 或隔离替代隐式跨域访问 | 无可实施机制 |
| 代价可摊销 | 复制、路由、排队、负载不均、消息传递和一致性维护是否小于节省的 stall | 指标变好但端到端变慢 |
| 语义约束 | `recall`、查询正确性、RAG 质量、tail SLO、隔离语义是否保持 | 不能作为有效系统优化 |

### 最小验证流程

1. 建立未优化 baseline：吞吐、`p50/p95/p99`、`recall` 或业务质量。
2. 用 `AMDuProfPcm/IBS/perf c2c/resctrl` 定位跨域访问、L3 容量敏感性和共享对象热点。
3. 做人工 counterfactual：手动复制热区、固定分区 first-touch、强制 query 绑定、限制 L3 容量、分片 OS 路径。
4. 若 counterfactual 对 runtime 或 tail latency 无改善，停止该方向。
5. 若 counterfactual 有效，再做自动调度、布局或反馈控制。

## 主线一：HNSW 图检索

`HNSW`（Hierarchical Navigable Small World，层次化可导航小世界图）是 RAG 向量检索常用索引。查询从上层入口点开始，逐层贪婪下降，再在第 0 层做 best-first 搜索。它具备四个适合 chiplet 优化的特征：

- 上层节点少、几乎所有 query 都会访问，适合 per-CCD 复制。
- 第 0 层节点多、访问不规则，适合按 query trace 或 graph edge cut 做分区和重排。
- 索引查询阶段基本只读，复制热节点不引入写失效。
- 多 query batch 中路径可能重叠，存在请求调度放大 L3 复用的空间。

推荐研究点：

1. **Replicated Navigator**：每个 `CCD` 复制上层节点、entry points、热点第 0 层 hub 的邻接表和压缩向量码。
2. **Trace-Guided Cold Graph Layout**：按 sampled query path、向量 cluster 和图边转移概率重排第 0 层节点，并把分区 first-touch 到固定拓扑域。
3. **Remote Expansion**：跨分区扩展时移动 query state 和候选堆摘要，而不是让本线程拉远端图节点 cache line。
4. **Hot-path Query Grouping**：用 entry path、coarse centroid 或前几跳节点作为 locality signature，将路径相近的 query 短窗口聚合到同一 `CCD`。

独立文档：[HNSW向量检索_Chiplet研究方向.md](./HNSW向量检索_Chiplet研究方向.md)。

## 主线二：IVF 批量检索

`IVF`（Inverted File Index，倒排文件索引）先用 coarse quantizer 找到若干 `list/cluster`，再扫描选中的倒排表。单 query 的 list 扫描多为流式访问，chiplet 空间有限；batch 场景下，多个 query 可能共享相同 cluster，才出现 L3 复用和请求分组空间。

推荐研究点：

- 将 coarse centroid 或 top-`nprobe` list 集合作为 locality signature。
- 每个 `CCD` 负责一组 inverted lists，优先处理命中本组 list 的 query。
- 热门 list 可复制，冷 list 分片；负载过高时做 bounded stealing。
- 对比 `FCFS`、普通 batch、cluster grouping、per-CCD ownership。

独立文档：[IVF批量检索_Chiplet研究方向.md](./IVF批量检索_Chiplet研究方向.md)。

## 备选一：LSM/HTAP 阶段切换

`LSM`（Log-Structured Merge-tree，日志结构合并树）和 `HTAP`（Hybrid Transactional/Analytical Processing，混合事务/分析处理）含有多类阶段：小范围写入、flush、compaction、delta merge、scan、join。不同阶段的工作集大小和复用距离不同，适合把 sorting 与 OLAP 的 `Local/Mixed` 规则迁移过来。

推荐研究点：

- `MemTable` flush、small compaction、Bloom/filter build 走 `Local`。
- 大 compaction、多路 merge、长 scan、delta merge 走 `Mixed`。
- 用 `PMU + queue depth + phase metadata` 做轻量在线切换。

独立文档：[LSM_HTAP_Chiplet研究方向.md](./LSM_HTAP_Chiplet研究方向.md)。

## 新候选一：推荐系统 Embedding Lookup

推荐系统的 embedding table lookup 是随机访存、读多、热点分布明显的 AI serving 场景。它不依赖远端 L3 快于 DRAM，而是依赖本地热 row 复制和 batch 内 request grouping。

CPU 上做 embedding lookup 是常见系统形态，尤其是 DLRM、广告推荐、排序/召回模型中的大表 sparse feature 查表。GPU 适合 dense compute 和部分 embedding 加速，但大表容量、稀疏访问、特征预处理、CPU/GPU 数据搬移会让 CPU 或 CPU+GPU 混合执行长期存在。Chiplet 研究应聚焦 CPU embedding bag / table batched embedding 的访存路径，而不是把结论泛化到所有推荐模型训练。

推荐研究点：

- hot embedding rows per-CCD replica。
- cold embedding table 按 ID range/hash 分片。
- batch 内按 table id、hot ID 或 feature field 分组。
- 写少读多 serving 优先；训练或高频更新场景需单独处理一致性。

独立文档：[Zen平台_TiNA适用性复核与新增场景.md](./Zen平台_TiNA适用性复核与新增场景.md)。

## 新候选二：GNN Neighbor Sampling 与图特征缓存

GNN neighbor sampling、feature gather 与 HNSW 类似，都有图结构访问、热节点、邻域路径重叠和特征行复用。区别是 GNN 需要保持训练/推理精度，且很多部署中 CPU 只负责采样和数据加载。

GNN 中 CPU 的常见角色是采样器和数据加载器。DGL、PyG、GraphBolt 等框架都提供 neighbor sampling / mini-batch loading 管线，实际训练常把采样和 feature gather 放在 CPU 侧，把 GNN forward/backward 放在 GPU 侧。若采用纯 CPU GNN 推理或训练，场景也存在，但不应作为默认主流假设。Chiplet 优化的主要目标是降低 CPU sampling、adjacency access、feature fetch 和 batch 构造的 p99 与 GPU wait time。

推荐研究点：

- seed locality grouping。
- adjacency/feature hot cache per-CCD。
- cold graph partition first-touch 到拓扑域。
- 同一 mini-batch 内高重叠 seed 聚合到同一 `CCD`。

独立文档：[Zen平台_TiNA适用性复核与新增场景.md](./Zen平台_TiNA适用性复核与新增场景.md)。

## 新候选三：HPC/CFD/FEA/Stencil

AMD 3D V-Cache 相关资料显示 CFD、FEA、stencil、部分 sparse/HPC workload 对 L3 容量敏感。该方向偏 HPC，但最符合 Zen 上“本地 L3 命中替代 DRAM”的硬件事实。

推荐研究点：

- mesh/block/stencil tile per-CCD working-set control。
- halo exchange 显式批量化。
- 小 block 走 `Local`，中等 block 走 `Mixed`，超大 block 走 DRAM bandwidth 优先。

独立文档：[Zen平台_TiNA适用性复核与新增场景.md](./Zen平台_TiNA适用性复核与新增场景.md)。

## 备选二：网络 I/O / RPC / KV Serving

网络 I/O 场景的有效性来自 queue、worker、mempool、RSS、RPC request class 与 tail latency 的组合。对 AMD Zen，`TiNA` 的远端 DCA/LLC 容量套利不成立；该方向不应依赖 Remote-tier 策略。

推荐先做纯软件原型：

- `RX/TX queue -> polling core -> worker group -> mempool` 拓扑绑定。
- 小/大 RPC 或 get/scan/set 分队列，降低 head-of-line blocking。
- tail-latency 驱动的 bounded spill/steal。
- KV 热元数据复制、冷 value 分片。

独立文档：[网络IO_RPC_KV_Chiplet研究方向.md](./网络IO_RPC_KV_Chiplet研究方向.md)。

## 备选三：Agent Tool Sandbox

Agent 场景的 chiplet 优化对象不是 sandbox 私有堆/栈，而是共享 OS 路径和混部资源隔离：`VFS/overlayfs/page cache/dentry/inode/cgroup/IPC/log queue`、session page-cache reuse、tool 与 vLLM 并发时的 `TTFT/decode p99` 干扰。当前未发现直接把 agent tool execution 映射到多 chiplet CPU 拓扑的论文，因此创新空间存在，但实验不确定性高。

推荐研究点：

- 每个拓扑域独立 sandbox pool、cgroup subtree、tmp/log/workdir、IPC acceptor。
- 同一 repo/session/tool-class 固定 home domain。
- tool 与 vLLM 混部时用 `cpuset + resctrl CAT/MBA + PSI` 做 tail 隔离。

独立文档：[AgentToolSandbox_Chiplet研究方向.md](./AgentToolSandbox_Chiplet研究方向.md)。

## 不建议方向

### LLM Attention / GEMM

已有实验表明当前 CPU attention 的跨 `CCD` demand fill 占比低，CAT 限制 L3 容量后性能变化很小。attention tile 复用主要在 L2 内完成，L3 只是通道。GEMM 权重多为流式读，激活要么小到 L2，要么大到 DRAM。继续沿 attention 内部 `kv_head` 或 tile 做静态 L3 局部化不具备论文主线价值。

### Flat 向量检索

Flat 检索对全量向量做顺序扫描，主要瓶颈是 SIMD 和 DRAM bandwidth。每个向量通常只读一次，不存在 L3 级短时间复用。除非做 batch GEMM 式计算优化，否则不适合作为 chiplet 局部性方向。

### 单 query IVF

单 query IVF 只扫描少数倒排表，但 list 内仍近似流式扫描。没有 batch overlap 时，chiplet 调度缺少共享工作集。

## 推荐推进顺序

1. **HNSW 最小验证**：用 FAISS/hnswlib 建立未优化 baseline，采集 `L3 miss/op`、`another CCD share`、`IBS load latency`、`recall/p99`。
2. **HNSW counterfactual**：复制上层节点与热点 hub，固定线程到 `CCD`，对比热复制 on/off。
3. **HNSW layout**：按 sampled query trace 重排第 0 层，加入 per-CCD 分区和 first-touch。
4. **IVF batch**：在检索服务场景验证 cluster overlap、grouping window 和 tail latency。
5. **根据测量结果决定是否扩展到 LSM/HTAP 或网络/RPC**。
6. 若希望继续保持 AI serving 语境，优先补测 embedding lookup 或 GNN sampling，而不是 TiNA 式网络 buffer tiering。

## 参考资料

- `HNSW`：https://arxiv.org/abs/1603.09320
- `Graph Reordering for Cache-Efficient Near Neighbor Search`：https://arxiv.org/abs/2104.03221
- `Faiss indexes`：https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- `Faiss performance tips`：https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster
- `CaGR-RAG`：https://arxiv.org/abs/2505.01164
- `DiskANN`：https://github.com/microsoft/DiskANN
- `P-MOSS`：https://arxiv.org/abs/2411.02933
- `OLAP on Modern Chiplet-Based Processors`：https://www.vldb.org/pvldb/vol17/p3428-fogli.pdf
- `Optimizing Sorting for Chiplet-Based CPUs`：https://vldb.org/workshops/2024/proceedings/ADMS/ADMS24_03.pdf
- `TiNA`：https://saksham.web.illinois.edu/assets/pdf/tina.pdf
- `AgentCgroup`：https://arxiv.org/abs/2602.09345
