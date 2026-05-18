# Zen 平台 TiNA 适用性复核与新增场景

生成时间：2026-05-14

## 结论

`TiNA` 的核心策略不适合直接迁移到 AMD Zen 架构 CPU。原因不是“软件实现困难”这一层，而是硬件收益模型不成立：

- Intel Sapphire Rapids 上，远端 chiplet LLC 访问仍明显快于 DRAM。TiNA 在 local DCA ways 溢出后，把 packet 放到远端 DCA ways，收益来自“远端 LLC hit 替代 DRAM access”。
- AMD Zen EPYC 上，跨 `CCD` 远端 L3/cache 路径与本地 DRAM 延迟接近，约 `110-150 ns` vs `~120 ns`。远端 L3 不能稳定提供比 DRAM 更低的延迟。
- 因此，TiNA 的 Remote-tier 容量套利在 Zen 上被拓扑抹平。把数据主动放到远端 `CCD` cache 中，通常没有比让它走本地 DRAM 更好的延迟收益。

受 TiNA 启发的方向需要重判：

| 方向 | Zen 上结论 |
| --- | --- |
| Remote-tier DCA / active-buffer overflow tiering | 排除或降为 Intel-SPR 特定方向 |
| 本地 queue/worker/mempool 分区 | 保留，但收益来自减少共享队列、core migration、mempool 远端页和 tail interference，不来自远端 LLC 容量 |
| RPC/KV size-aware queueing | 保留，属于 tail-latency 调度，不依赖 TiNA tradeoff |
| bounded spill/steal | 保留，但 spill 目标是负载均衡和 SLO 保护，不是把数据放进远端 L3 |
| KV 热元数据复制 | 保留，前提是复制到每个本地 `CCD`，不是依赖远端 cache |

后续 Zen 主线应优先寻找 **本地 L3 热集复制、本地分片路由、阶段工作集控制、共享状态隔离**，而不是利用远端 cache 做容量扩展。

## TiNA 在 Zen 上失效的机制

TiNA 在 Intel SPR 上成立的关键路径是：

```text
local DCA ways 溢出
  -> packet 若留在 local tier，会发生 DMA leak/bloat
  -> packet 若写入 remote DCA ways，后续 CPU 远端 LLC hit
  -> remote LLC hit 仍快于 DRAM
  -> tail latency 下降
```

Zen EPYC 的关键路径不同：

```text
local L3 溢出
  -> 远端 CCD L3/cache 路径约等于本地 DRAM
  -> remote cache hit 不再显著快于 DRAM
  -> 主动使用 remote cache 容量没有明确净收益
```

这意味着 TiNA 的“本地低延迟 vs 远端容量”模型，在 Zen 上不能照搬为“本地 DCA ways vs 远端 DCA ways”。Zen 上仍有“本地 L3 vs DRAM”的强差异，但没有“远端 L3 明显优于 DRAM”的稳定差异。

## Zen 上保留的网络/RPC方向

### 拓扑感知 queue/worker/mempool

保留。优化目标改为：

- 避免同一 RX/TX queue 被多个拓扑域争用。
- 避免 polling core 和 worker 迁移导致 L1/L2/L3 冷启动。
- 避免 mempool 页与处理线程跨 socket/NPS 错配。
- 避免小请求与大请求混跑导致 head-of-line blocking。

不再描述为“近似 TiNA 的 local/remote tiering”。

### RPC/KV size-aware queueing

保留。该方向与 Minos、Shenango、Caladan、Arachne 更相关，核心是把小请求 tail latency 从长请求、scan、background task 中隔离出来。chiplet 只提供拓扑边界，不提供 TiNA 式 remote cache 容量收益。

### tail-latency bounded spill

保留，但语义是：

- 本地队列未过载时保持局部执行。
- 本地队列过载时为了 SLO 把新请求 spill 到远端组。
- spill 是负载均衡代价，不是局部性优化。

### active-buffer-aware overflow tiering

降级。除非确认 AMD 平台存在可控、低延迟、远端 cache 明显快于 DRAM 的 NIC-to-cache 路径，否则不应作为 Zen 论文方向。

## 新增候选场景

以下场景不依赖“远端 L3 快于 DRAM”。它们依赖的是“本地 L3 明显快于 DRAM”，更符合 Zen 平台。

## 场景一：推荐系统 Embedding Lookup

推荐系统的 DLRM / embedding bag 阶段通常由大规模 embedding table lookup 主导。公开资料和论文均将 embedding table 描述为超大、随机访问、memory-bound。AMD CPU 相关工作也表明 embedding bag 在 EPYC 上仍有明显并行优化空间。

CPU 使用形态：

- 推荐系统大表 embedding lookup 在 CPU 或 CPU+GPU 混合系统中常见。CPU 侧价值来自大内存容量、多线程稀疏查表、特征预处理和 serving 系统集成。
- PyTorch `EmbeddingBag`、TorchRec/FBGEMM 的 embedding bag / table batched embedding operator 都体现了 CPU 与 GPU 并存的推荐系统执行路径。
- 该场景不等价于“推荐模型全流程在 CPU 上执行”。dense MLP、ranking tower 或训练反向传播可在 GPU 上执行；chiplet 优化对象是 CPU 侧 sparse lookup、batching、cache locality 和 table placement。

推荐方向：**hot embedding row per-CCD replica + cold table shard + request grouping**。

- 热 embedding rows 或 hot IDs 复制到每个 `CCD` 本地工作集。
- 冷 embedding table 按 ID range/hash 分片。
- batch 内按 hot ID/table id 分组，减少同一热 row 被多个 `CCD` 重复从 DRAM 拉取。
- 对写少读多的 serving，副本维护成本可控；对训练或高频更新，成本显著上升。

风险：

- embedding lookup 可能缺少足够时间局部性，热度分布不稳定。
- 高维 embedding row 体积大，复制预算容易超过单 `CCD` L3。
- 如果工作负载已由 DRAM bandwidth 饱和主导，分片/复制只能降低 p99，难以提高均值吞吐。

参考资料：

- NVIDIA 推荐系统性能指南：https://docs.nvidia.com/deeplearning/performance/recsys-best-practices/index.html
- PyTorch EmbeddingBag：https://docs.pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html
- TorchRec：https://github.com/meta-pytorch/torchrec
- FBGEMM：https://github.com/pytorch/FBGEMM
- Optimizing CPU Performance for Recommendation Systems At-Scale：https://pure.psu.edu/en/publications/optimizing-cpu-performance-for-recommendation-systems-at-scale/
- Parallelization Strategies for DLRM Embedding Bag Operator on AMD CPUs：https://pure.psu.edu/en/publications/parallelization-strategies-for-dlrm-embedding-bag-operator-on-amd
- RecMG：https://arxiv.org/abs/2511.08568

## 场景二：GNN Neighbor Sampling 与图特征缓存

GNN 训练/推理中的 neighbor sampling 和 feature gather 具有图遍历、邻接表访问、节点 feature 访问和 batch 路径重叠。已有 GNN 系统大量使用 graph partitioning、feature cache、neighbor cache、cooperative minibatching，说明“采样路径重叠 + 热节点缓存”是可优化对象。

CPU 使用形态：

- 现代 GNN 训练常采用 mini-batch neighbor sampling。DGL、PyG 和 GraphBolt 都提供采样与数据加载接口，CPU 侧负责 seed batching、邻居采样、subgraph construction 和 feature fetch 是常见路径。
- GNN layer compute 常在 GPU 上执行。CPU chiplet 优化的端到端收益需要用 GPU wait time、sampling throughput、mini-batch ready latency 证明，不能只看采样微基准。
- 纯 CPU GNN 推理/训练可作为可控实验环境，但论文动机更适合表述为“CPU-side graph sampling/data pipeline optimization for GNN”，而不是“CPU GNN 训练主流优化”。

推荐方向：**seed locality grouping + adjacency/feature hot cache per-CCD**。

- 以 seed nodes、community、partition id 作为 locality signature。
- 同一 batch 内邻域重叠高的 seeds 放到同一 `CCD`。
- 热节点邻接表和 feature rows 做 per-CCD 副本。
- 冷图按 partition first-touch 到对应 NUMA/NPS 域。

风险：

- 如果训练主要在 GPU，CPU chiplet 优化只影响采样/数据加载，不一定主导端到端。
- full-batch GNN 与 sampling-based GNN 的瓶颈不同。
- 精度约束比 HNSW 更复杂。

参考资料：

- DGL neighbor sampling 文档：https://www.dgl.ai/dgl_docs/en/2.3.x/_modules/dgl/sampling/neighbor.html
- DGL GraphBolt 文档：https://www.dgl.ai/dgl_docs/guide/minibatch.html
- PyG NeighborLoader 文档：https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html
- PaGraph：https://paperswithcode.com/paper/pagraph-scaling-gnn-training-on-large-graphs
- GNNLab：https://colab.ws/articles/10.1145%2F3492321.3519557
- DCI GNN inference cache：https://arxiv.org/abs/2503.01281
- FreshGNN：https://arxiv.org/abs/2301.07482

## 场景三：HPC/CFD/FEA/Stencil 的 CCD 级块调度

AMD 3D V-Cache 面向 CFD、FEA、EDA 等技术计算负载已有公开资料。部分 CFD/OpenFOAM、Ansys、EDA workload 对 L3 容量敏感。该事实说明 Zen 平台上存在一批“本地 L3 命中替代 DRAM”的真实场景。

推荐方向：**mesh/block per-CCD working-set control**。

- 将网格块、cell block、stencil tile、sparse row block 控制在单 `CCD` L3 或少数 `CCD` 聚合范围内。
- halo exchange 显式批量化，避免细粒度跨 `CCD` 随机读取。
- 小 block 使用 `Local`，中等 block 使用 `Mixed`，超大 block 走 DRAM bandwidth 优先。

风险：

- 需要进入具体 HPC 应用或 proxy-app，工程成本较高。
- 很多 stencil/CFD 已有成熟 NUMA/MPI/OpenMP 优化，chiplet 增量需要与现有 blocking 对比。
- 若数据集远超 L3 且复用距离大，收益可能转为内存带宽问题。

参考资料：

- Evaluating the impact of L3 cache size of AMD EPYC CPUs on CFD：https://arxiv.org/abs/2505.17934
- AMD 3D V-Cache HPC 技术计算资料：https://www.amd.com/en/blogs/2024/amd-epyc-cpus-deliver-breakthrough-performance-wi.html
- Ansys CFX AMD 3D V-Cache performance brief：https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/performance-briefs/amd-epyc-9004x-pb-ansys-competitve-hbm.pdf

## 场景四：EDA / RTL / 离散事件仿真

EDA functional simulation、RTL simulation 和 parallel discrete-event simulation 具有事件队列、process/entity 状态、时间戳优先级队列、频繁小对象访问和局部事件传播。AMD 与 Synopsys 资料显示 VCS 等 EDA 仿真对 3D V-Cache 有明显收益；并行离散事件仿真文献也长期关注 event locality、NUMA locality 和优先队列争用。

推荐方向：**LP/process-to-CCD mapping + local event queue batching**。

- 将强通信/频繁事件传播的 logical processes 或 RTL processes 映射到同一 `CCD`。
- 每 `CCD` 本地 event queue，跨 `CCD` event 通过显式 batch 交换。
- hot module/process state 保留本地，减少共享优先队列 cache line bounce。
- 对全局时间同步使用层次化 barrier 或 bounded lookahead。

风险：

- 商用 EDA 工具不可改，可能只能做外部调度或开源模拟器验证。
- correctness 和 determinism 要求高。
- 全局时间同步可能掩盖 locality 收益。

参考资料：

- Synopsys AMD EPYC EDA workload：https://www.synopsys.com/blogs/chip-design/eda-workloads-amd-processors.html
- AMD EPYC 9004X Synopsys VCS performance brief：https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/performance-briefs/amd-epyc-9004x-pb-synopsys-vcs.pdf
- SmartPQ NUMA priority queue：https://arxiv.org/abs/2406.06900
- PARSIR NUMA-aware PDES：https://arxiv.org/abs/2410.00644

## 场景五：Genomics k-mer Counting

k-mer counting 使用 hash table、Bloom filter、sort/compact、minimizer partition 等结构。近期工作通过 minimizer-based hashing 利用 k-mer locality，说明生物信息学中存在可利用的序列局部性和 hash locality。

推荐方向：**minimizer bucket per-CCD partition + hot minimizer cache**。

- 按 minimizer 或 hash bucket 将 k-mers 分配到 `CCD`。
- hot minimizer bucket 的计数结构本地化或复制。
- 跨 `CCD` 合并在 batch 边界显式完成。
- 对 counting/sorting 阶段使用 `Local/Mixed` 切换。

风险：

- 许多 k-mer 工具瓶颈在 I/O、压缩或磁盘 spill。
- k-mer 数据量通常远超 L3，只有 bucket/metadata 层可能适合本地 L3。
- 需要与已有 minimizer 分区和 cache-efficient hashing 比较。

参考资料：

- k ache-hash：https://arxiv.org/abs/2404.16516
- KMC：https://academic.oup.com/bioinformatics/article/33/17/2759/3796399
- Jellyfish：https://academic.oup.com/bioinformatics/article/27/6/764/234905

## 新增场景优先级

| 优先级 | 场景 | 推荐结论 |
| --- | --- | --- |
| A | 推荐系统 embedding lookup | 可作为 HNSW 之外的 AI serving 主线候选 |
| A | GNN neighbor sampling / feature cache | 与 HNSW 同属图/feature 热集复制，值得进一步查 |
| B | HPC/CFD/FEA/stencil | Zen 证据强，但偏 HPC，离 LLM/RAG 应用链路远 |
| B | EDA/RTL/离散事件仿真 | 3D V-Cache 证据强，系统可改性风险高 |
| C | Genomics k-mer counting | 可做，但应用域较窄 |

## 对现有方向总览的修订结论

网络/RPC 方向仍可作为系统备选，但不应再以 TiNA 的 Remote-tier tradeoff 作为主要动机。Zen 平台更合理的主线是：

1. HNSW / GNN / embedding / k-mer：本地热结构复制。
2. LSM / HTAP / CFD / stencil：阶段工作集控制。
3. EDA / PDES / Agent / RPC：共享状态分片与显式批量通信。

这些方向共同遵守同一条原则：**只把本地 `CCD` L3 当作低延迟资源；不要把远端 `CCD` L3 当作比 DRAM 更优的容量池。**
