# Pyramid: A General Framework for Distributed Similarity Search 解读

**一句话概述**：Pyramid 将单机 HNSW 扩展为分布式相似性搜索框架。它先用抽样数据构建一个小型 `meta-HNSW`，再用 `meta-HNSW` 的底层图做均衡图划分，把相似数据放到同一个子数据集；查询时先搜索 `meta-HNSW`，只访问少量可能包含近邻的 `sub-HNSW`，从而避免朴素分布式 HNSW 每个查询都打到所有机器（Algorithm 3-4；Fig. 2）。

## 论文基本信息

- 论文：`Pyramid: A General Framework for Distributed Similarity Search` 
- 会议：2019 IEEE International Conference on Big Data (IEEE BigData 2019)
- 作者：Shiyuan Deng、Xiao Yan、Kelvin K.W. Ng、Chenyu Jiang、James Cheng
- 机构：The Chinese University of Hong Kong
- 研究对象：大规模分布式相似性搜索，底层索引为 HNSW
- 支持相似度：欧氏距离、角距离、内积搜索（MIPS，Maximum Inner Product Search，最大内积搜索）
- 原始资料：`wiki/原始资料/papers/HNSW/Pyramid: A General Framework for DistributedSimilarity Search.pdf`

## 背景

相似性搜索的输入是查询向量 `q` 和数据集 `X`，目标是返回与 `q` 最相似的 `k` 个数据项。常见相似度包括欧氏距离、角距离和内积。在线推荐、广告和图像检索中，数据规模可达到数亿到十亿级，系统同时需要较高结果质量、较高吞吐和毫秒级延迟（Sec. I）。

**HNSW**（Hierarchical Navigable Small World，层次化可导航小世界图）是一类图近邻索引。它把数据点组织成多层近邻图，上层稀疏、用于长距离导航，底层包含全部数据、用于精细搜索。查询从顶层入口点开始逐层下降，到第 0 层后用较大的候选队列做近似近邻搜索（Algorithm 1；Fig. 1）。HNSW 在单机近似近邻搜索中有较好的 recall-time 性能，但完整保存原始向量和图结构会消耗大量内存。

单机系统可用向量量化压缩数据，例如 PQ、OPQ。论文指出压缩会引入量化误差，可能改变相似度排序；文中给出的例子是 FAISS 使用 8 个 OPQ codebook、探测 222 个 item 做 top-10 欧氏近邻搜索时，precision 只有 25.15%（Sec. I）。Pyramid 的目标不是用压缩降低单机内存，而是把未压缩数据分布到多台机器上，保持结果质量。

论文使用 **precision** 衡量 top-k 质量：若返回的 `k` 个结果中有 `k'` 个属于真实 top-k，则 precision 为 `k'/k`。该定义比只要求命中真实 top-1 的 Recall@k 更严格（Sec. V-A）。

## 研究问题

Pyramid 解决的是分布式 HNSW 的查询扩展问题。

朴素方案 `HNSW-naive` 将数据随机分到多台机器，每台机器独立建一个 HNSW。每个查询都广播到所有机器，各机器返回局部 top-k，协调者再合并结果。该方案能容纳更大数据集，但每个查询仍会触发所有 worker 的计算，吞吐随机器数增加受限（Sec. III）。

Pyramid 的问题定义可拆成四点：

1. 数据量超过单机内存时，如何用 HNSW 支持未压缩数据搜索。
2. 如何让每个查询只访问少量机器，同时保持接近全局搜索的结果质量。
3. 如何同时支持欧氏距离、角距离和内积搜索。
4. 如何在分布式环境中处理 straggler（慢节点）和 worker failure（节点故障）（Sec. I；Sec. IV-B）。

## 核心思想

Pyramid 用一个小型 `meta-HNSW` 同时承担两类工作：

- **建索引时划分数据**：`meta-HNSW` 的底层图反映抽样数据的近邻结构。Pyramid 对该底层图做均衡图划分，使每个分区权重接近，并尽量减少跨分区边。跨分区边少表示分区内部数据更相似（Algorithm 3）。
- **查询时选择 worker**：查询先在 `meta-HNSW` 中找 top-K 近邻。若这些近邻落在某些图分区中，Pyramid 只把查询发送到这些分区对应的 `sub-HNSW`。参数 `K` 称为 branching factor，控制一个查询可能访问多少子索引（Algorithm 4）。

论文把 `meta-HNSW` 类比成多个 `sub-HNSW` 的共享上层。HNSW 上层用于快速接近查询所在区域，Pyramid 的 `meta-HNSW` 则用于快速判断哪些机器上的子数据集可能包含结果（Fig. 2）。

## 方法与系统设计

### 索引构建

Pyramid 的欧氏距离和角距离索引构建流程如下（Algorithm 3）：

1. 从原始数据集 `X` 随机抽取 `n'` 个样本，形成 `X'`。
2. 对 `X'` 做 `k-means`，得到 `m` 个中心点。
3. 用这些中心点构建 `meta-HNSW`。
4. 对 `meta-HNSW` 第 0 层图做 `w` 路均衡图划分。`w` 也是 `sub-HNSW` 数量。
5. 对原始数据集中的每个 item `x`，在 `meta-HNSW` 中找最相似中心点 `n(x)`，并根据 `n(x)` 所属分区把 `x` 放入对应子数据集。
6. 每个子数据集独立构建一个 `sub-HNSW`。

均衡图划分的权重来自 `k-means` 中心覆盖的样本数。该设计假设各 item 被查询命中的概率相近时，子数据集大小接近即可平衡负载。若有样本查询可反映热点数据，论文也提出可把中心点权重改成该中心点出现在样本查询 top-k 结果中的频率，用于负载均衡（Sec. III-A）。

论文没有直接随机抽 `m` 个 item 作为 `meta-HNSW` 节点，而是先抽更大的 `X'` 再做 `k-means`。原因是 `m` 可能较小，直接抽样不能稳定反映整体分布；`k-means` 中心更能作为数据分布的代表点（Sec. III-A）。

### 分布式构建流程

分布式构建按 worker 协作执行：

1. 每个 worker 从分布式文件系统读入本地数据片段。
2. 每个 worker 按全局数据量和本地数据量抽样。
3. worker 协同完成分布式 `k-means`。
4. 某个 worker 保存中心点，构建 `meta-HNSW` 并完成图划分。
5. `meta-HNSW` 和分区到 worker 的映射广播给所有 worker。
6. 每个 worker 用 `meta-HNSW` 判断本地 item 的目标子数据集，并通过 shuffle 发往目标 worker。
7. shuffle 完成后，各 worker 构建本地 `sub-HNSW`（Sec. III-A）。

该流程把昂贵的数据重分布放到离线构建阶段，用在线查询阶段的更少访问数换取吞吐。

### 查询处理

Pyramid 的查询流程如下（Algorithm 4）：

1. 协调者接收查询 `q`。
2. 协调者在 `meta-HNSW` 中找 top-K 近邻。
3. 对 `meta-HNSW` 第 0 层每个图分区，若该分区包含 top-K 中任一节点，则激活该分区对应的 `sub-HNSW`。
4. 被激活的 worker 在本地 `sub-HNSW` 中搜索 top-k。
5. 协调者收集各 worker 的局部 top-k，并按相似度合并出最终 top-k。

`sub-HNSW` 不是独立的网络节点，而是 executor/worker 机器内存中的本地 HNSW 索引。查询路径是 coordinator 先搜索 `meta-HNSW`，再把请求发送到持有目标 `sub-HNSW` 的 executor；不存在“先经过 `sub-HNSW` 再转发到机器”的额外阶段。论文实现中每个 `sub-HNSW` 可对应一个 Kafka topic，请求投递到 topic 后由订阅该 topic、持有该索引副本的 executor 执行（Sec. IV-A；Sec. IV-B）。

`K` 增大时，访问更多子索引，precision 通常上升，但每个查询的计算量、网络响应数和尾延迟也会上升（Fig. 5-8）。

### 欧氏距离与角距离支持

欧氏距离可直接用 `s(q,x) = -||q-x||` 作为相似度。角距离可通过归一化转成欧氏距离问题：构建索引前把 item 归一化为单位向量，查询时也把 query 归一化。单位向量之间的角相似与欧氏距离单调相关，因此 Algorithm 3-4 可直接用于角距离搜索（Sec. III-C）。

### MIPS 支持

MIPS 的难点是高范数向量更容易成为结果。论文在 ImageNet 约 200 万个 150 维描述符上做线性扫描统计，1000 个查询的 top-10 MIPS 结果中，范数排名前 5% 的 item 占结果集 93.1%（Fig. 3）。这会导致两个问题：

- 若按原始内积 HNSW 图划分，很多高范数 item 会聚在一个“大范数分区”，使对应子数据集过大。
- 查询 `meta-HNSW` 时，大量查询都会访问该大范数分区，导致对应 worker 成为热点或 straggler（Sec. III-C）。

Pyramid 对 MIPS 使用 Algorithm 5：

1. 抽样后先把样本归一化为单位向量。
2. 用 spherical k-means（球面 k-means，用角方向聚类）得到中心。
3. 用中心构建 `meta-HNSW` 并划分底层图，使分区代表方向相近的数据，而不是范数大的数据。
4. 原始 item 按其在 `meta-HNSW` 中的 MIPS 方向归属分配到子数据集。
5. 对每个 `meta-HNSW` 分区中的向量，再找原始数据集中的 top-r MIPS item，将这些高范数候选额外加入对应子数据集。

第 5 步允许少量 item 出现在多个子数据集中，目标是减少查询需要访问的分区数。论文称 `m*r` 只需占数据集规模的几个百分点即可获得较好效果，top-r MIPS 可用 LSH 方法近似完成（Sec. III-C）。

### 系统架构

Pyramid 包含三类主要组件（Fig. 4）：

- **Coordinator**：接收上游查询，搜索 `meta-HNSW`，决定访问哪些 `sub-HNSW`，收集局部结果并合并。
- **Executor**：持有一个或多个 `sub-HNSW`，执行本地 HNSW 搜索并返回局部结果。
- **Broker**：通过 Kafka 传递协调者到执行者的查询请求。Zookeeper 用于监控实例状态。

Coordinator 是可多实例部署的角色，不是固定单台中心机器。论文描述中，一个查询到来时会被分配给一个随机 worker 作为 coordinator；系统架构也假设上游应用通过 hashing 等机制把负载均匀分发给多个 coordinator。每个查询只由其中一个 coordinator 负责 `meta-HNSW` 路由、请求下发和结果合并（Sec. III-B；Sec. IV-A；Sec. IV-B）。

论文提供三个 API 类：

- `Coordinator`：提供同步 `execute(query, para)` 和异步 `execute_async(query, para, callback)` 查询接口，`para` 包含 branching factor `K` 和返回近邻数 `k`。
- `Executor`：加载指定 `sub-HNSW`，通过 `start(para)` 开始处理请求，`para` 包含底层图搜索因子等查询参数。
- `GraphConstructor`：离线构建 `meta-HNSW` 和 `sub-HNSW`，`refresh()` 可在数据更新后重建并通知查询组件（Sec. IV-A）。

### Straggler 与故障处理

Pyramid 用复制和 Kafka 分组处理慢 executor。每个 `sub-HNSW` 对应一个 Kafka topic，持有该 `sub-HNSW` 副本的 executor 组成一个 consumer group。Kafka 会周期性重平衡消息队列，慢 executor 会收到更少请求，其他副本 executor 接管更多请求（Sec. IV-B）。

故障分两类：

- **Coordinator failure**：Pyramid 不恢复正在处理的查询，依赖上游应用超时后重试其他 coordinator。
- **Executor failure**：若同一 `sub-HNSW` 有多个副本，Kafka 会把请求发给其他 executor。Zookeeper 记录实例锁，Master 发现锁释放后在可用机器上重启实例。Master 也有热备，避免单点故障（Sec. IV-B）。

论文明确没有处理 straggling coordinator，理由是 coordinator 负载较轻，且上游应用可均匀分发请求（Sec. IV-B）。

## 实验设置

### 指标

实验主要报告三类指标（Sec. V-A）：

- **Precision**：返回 top-k 中属于真实 top-k 的比例。
- **Throughput**：系统每秒处理的查询数。
- **Latency**：从系统收到查询到返回结果的时间。论文报告 90th percentile latency，而不是平均延迟。

除特别说明外，实验搜索 top-10 近邻，precision 是 10000 个随机查询的平均值。`meta-HNSW` 和 `sub-HNSW` 参数按 HNSW 原论文推荐设置，未细调：底层最大出度 32，其他层最大出度 16，底层 graph walk 的 search factor `l=100`（Sec. V-A）。

### 数据集

| 数据集 | 数据量 | 维度 | 大小 | 用途 |
|---|---:|---:|---:|---|
| Deep500M | 500M | 96 | 192 GB | 欧氏近邻 |
| SIFT500M | 500M | 128 | 256 GB | 欧氏近邻 |
| Tiny10M | 10M | 384 | 15.4 GB | MIPS |

Deep500M 来自 Deep1B，SIFT500M 来自 SIFT1B，Tiny10M 来自 Tiny80M。Deep500M 和 SIFT500M 的向量范数相近，因此 MIPS 与欧氏近邻差异不明显；Tiny10M 的范数分布更宽，更适合测试 MIPS（Table I；Sec. V-A）。

### 平台

实验使用 10 台机器，通过 10 Gbps Ethernet 互联。每台机器有两个 8 核 Intel Xeon E5-2620v4 2.1 GHz 处理器和 48 GB RAM，系统为 CentOS 7.2。因为集群有 10 台机器，实验使用 10 个 `sub-HNSW`（Sec. V-A）。

## 结果与解释

### 参数影响

论文调节两个 `meta-HNSW` 参数：`meta-HNSW` 大小 `m ∈ {1000, 10000, 100000}`，以及 branching factor `K ∈ {1, 5, 10, 20, 50, 100}`（Sec. V-B）。

**访问率**：访问率定义为每个查询访问的 `sub-HNSW` 比例。相同 `meta-HNSW` 大小时，`K` 越大，访问率越高。相同 `K` 下，`meta-HNSW` 越大，访问率越低，因为更细粒度的 `meta-HNSW` 划分会让 top-K meta 近邻集中在更少分区中（Fig. 5）。

**Precision**：precision 随 `K` 增大先快速上升再趋于稳定。相同 `K` 下，较小 `meta-HNSW` 的 precision 更高，因为它通常访问更多 `sub-HNSW`。即使 `K=1`，只访问一个 `sub-HNSW` 时，Deep500M 和 SIFT500M 的 precision 都超过 65%（Fig. 6）。

**Throughput**：throughput 随 `K` 增大持续下降，因为每个查询访问更多 `sub-HNSW`，计算和通信负载更重。`meta-HNSW=100000` 不总是优于 `meta-HNSW=10000`，因为搜索更大的 `meta-HNSW` 更慢；论文测得 `meta-HNSW=10000` 每查询约 0.06 ms，`meta-HNSW=100000` 每查询约 0.18 ms（Fig. 7；Sec. V-B）。

**90th percentile latency**：尾延迟随 `K` 增大而上升。原因是 coordinator 需要等待更多 executor 返回，查询延迟受最慢 executor 影响。`meta-HNSW` 大小的影响不单调：更大 `meta-HNSW` 降低访问率，但也增加 `meta-HNSW` 搜索时间（Fig. 8）。

论文总结：在 Deep500M 和 SIFT500M 上，Pyramid 可达到 90% 以上 precision、超过 100000 queries/s 吞吐，并保持 2-3 ms 延迟。后续实验固定 `meta-HNSW=10000`（Sec. V-B）。

### 与 HNSW-naive 和 FLANN 对比

论文比较 Pyramid、HNSW-naive 和 FLANN。HNSW-naive 与 Pyramid 使用相同 `sub-HNSW` 最大出度配置，并调节查询参数使二者 precision 约为 90%。FLANN 难以达到 90% precision，因此按其推荐设置报告（Sec. V-C）。

Fig. 9 的主要结果如下：

| 数据集 | 系统 | Throughput | Precision |
|---|---|---:|---:|
| Deep500M | Pyramid | 105319 q/s | 0.894 |
| Deep500M | HNSW-naive | 42827 q/s | 0.907 |
| Deep500M | FLANN | 1034 q/s | 0.621 |
| SIFT500M | Pyramid | 80106 q/s | 0.892 |
| SIFT500M | HNSW-naive | 33596 q/s | 0.905 |
| SIFT500M | FLANN | 752 q/s | 0.738 |

Pyramid 在相近 precision 下吞吐超过 HNSW-naive 的 2 倍，原因是每个查询只访问部分 `sub-HNSW`。Pyramid 和 HNSW-naive 都显著优于 FLANN，论文将其归因于 HNSW 相比 FLANN 所用 KD-tree 分布式方案有更好的算法性能（Fig. 9；Sec. V-C）。

该对比主要证明吞吐收益，不证明单个查询的端到端时间一定低于 HNSW-naive。论文在参数实验中报告 Pyramid 的 90th percentile latency，并说明 `K` 增大会提高尾延迟；但 Fig. 9 的 Pyramid 与 HNSW-naive 主对比只给出 throughput 和 precision，没有给出相同 precision 下两者的逐查询 latency 对照（Fig. 8-9；Sec. V-B；Sec. V-C）。

### 构建开销

Pyramid 的查询收益来自更昂贵的离线构建。Deep500M 上，Pyramid 用 10 台机器构建索引约 162 分钟，其中 `meta-HNSW` 构建 31 分钟、数据集划分 87 分钟、`sub-HNSW` 构建 44 分钟。HNSW-naive 总构建时间为 53 分钟，其中数据划分 14 分钟、`sub-HNSW` 构建 39 分钟。FLANN 构建仅 38 秒（Sec. V-C）。

论文认为 Pyramid 适合数据更新不频繁、在线查询性能关键的场景。若业务需要高频更新，Pyramid 的重划分和重建成本会成为限制（Sec. V-C）。

### MIPS

Tiny10M MIPS 实验以 HNSW-naive 为基线。HNSW-naive precision 为 99.7%，吞吐为 12732 q/s。Pyramid 通过低访问率获得更高吞吐；在 `K=1` 时只访问 10 台机器中的 1 个 `sub-HNSW`，precision 仍达到 96.98%（Fig. 10；Sec. V-C）。

MIPS 实验中 replication factor `r=300`，所有 `sub-HNSW` 合计存储 10060599 个 item，只比原始 Tiny10M 多 0.6%。该结果说明少量跨分区复制可显著降低 MIPS 查询访问率，内存开销较小（Fig. 10；Sec. V-C）。

### 扩展性

SIFT100M 上，论文比较 10 台机器和 5 台机器，并调节查询参数使两种配置达到相同 precision。10 台机器相对 5 台机器的吞吐提升为：

- 80% precision：1.78 倍
- 90% precision：1.59 倍

论文认为未达到线性扩展与 HNSW 搜索复杂度有关。机器少时，每个 `sub-HNSW` 更大，但访问的 `sub-HNSW` 数更少；由于 HNSW 复杂度随数据规模约为 `O(log n)`，减少访问子索引数量的影响可能占主导（Fig. 11；Sec. V-D）。

### Straggler 与故障

Straggler 实验在 SIFT100M 上进行。每个 `sub-HNSW` 有两个副本，分布在不同机器；每台机器托管两个不同 `sub-HNSW`。系统运行在峰值吞吐 70%，并用 CPU-limit 限制其中一台机器 CPU share（Sec. V-D）。

当 CPU share 高于 30% 时，访问受限机器上 `sub-HNSW` 的查询吞吐没有显著变化。原因是对应副本在其他机器上，Kafka 会把部分请求转移过去；其他机器在 70% 峰值负载下还有空闲资源。CPU share 降到 10% 时，吞吐明显下降，因为过多请求被转移，其他机器资源不足（Fig. 12）。

故障实验显示：约第 300 秒杀掉一台机器时吞吐下降；约第 500 秒该机器重新加入时，因为 Kafka 需要重平衡队列，吞吐再次下降；约第 600 秒重平衡结束后，吞吐恢复到故障前水平（Fig. 13）。

## 分析

Pyramid 的主要价值是把 HNSW 的“上层导航、底层精搜”思想扩展到分布式场景。`meta-HNSW` 负责粗粒度路由，`sub-HNSW` 负责局部搜索。它没有改变单个 `sub-HNSW` 的搜索过程，而是改变了查询被发送到哪些机器。

该设计的关键假设是：相似数据被划到同一子数据集后，查询的近邻也会集中在少量子数据集中。Fig. 5-6 的参数实验支持这一点：`K=1` 只访问一个 `sub-HNSW` 时 precision 已超过 65%，提高 `K` 后 precision 可超过 90%。该现象说明 `meta-HNSW` 路由能较好捕获数据空间局部性。

Pyramid 的吞吐收益来自减少每个查询的 worker 参与数，而不是加快单个 worker 上的 HNSW 搜索。HNSW-naive 中一个查询会并行打到所有 worker，单次查询 wall-clock 时间更接近“网络分发 + 所有 worker 本地搜索耗时的最大值 + 汇总合并”，不是所有 worker 搜索时间相加。Pyramid 增加了 `meta-HNSW` 路由步骤，但减少了被访问 executor 数量，因此高并发下系统总工作量更低、吞吐更高；在低负载或单个查询本地搜索很快时，单 query latency 不一定低于 HNSW-naive。它的构建成本、数据 shuffle、跨机器复制和查询聚合逻辑都比 HNSW-naive 更复杂，因此更适合读多写少的在线服务。

## 边界

1. 实验平台是 10 台双路 Intel Xeon E5-2620v4 机器，通过 10 Gbps Ethernet 连接；论文没有讨论单机多 CCD、NUMA、每 CCD L3、线程绑定或跨 CCD 访问（Sec. V-A）。
2. Pyramid 的查询优化粒度是机器或 `sub-HNSW`，不是 CPU core、CCD 或 cache line。不能直接把论文结果解释为 chiplet locality 收益。
3. 系统依赖离线构建和数据重分布。Deep500M 上 Pyramid 构建 162 分钟，高于 HNSW-naive 的 53 分钟；数据频繁更新时成本较高（Sec. V-C）。
4. Straggler 缓解依赖 `sub-HNSW` 副本和 Kafka 队列重平衡。若系统接近满负载，副本机器没有空闲资源，缓解能力会下降（Fig. 12）。
5. 论文不处理 coordinator 的慢节点问题，coordinator failure 也依赖上游重试（Sec. IV-B）。
6. MIPS 方案使用额外复制减少访问率，Tiny10M 上开销为 0.6%；该开销在其他数据分布、`r` 参数或更高 precision 目标下可能变化（Fig. 10）。
7. 论文没有证明 Pyramid 对单个查询的端到端时延一定优于 HNSW-naive；其核心实验结论是通过减少每个查询访问的 `sub-HNSW` 数量提升系统吞吐。

## 与当前 Chiplet/HNSW 研究的关系

### 可迁移点

- `meta-HNSW` 本质上是一个查询路由器。当前若研究单机多 CCD HNSW，可借鉴“先粗粒度判断可能命中的局部区域，再只激活部分执行资源”的思想。
- Pyramid 的参数权衡清晰：访问更多分区会提高质量，但降低吞吐并增加尾延迟。CCD 级 HNSW 调度也需要同时报告 recall、吞吐、平均延迟和尾延迟。
- MIPS 部分说明数据范数或热点分布可能破坏普通图划分的负载均衡。当前分析 HNSW 热点时不能只看图边或空间近邻，还要看查询分布和向量属性。
- Straggler 实验提供了一个服务级对照：副本可以提升鲁棒性，但只有在其他执行节点有空闲资源时有效。

### 不可直接迁移点

- Pyramid 的优化目标是减少跨机器查询扇出，不是提高单机 cache 命中率。
- 论文没有硬件计数器、L3 miss、DRAM 访问来源、线程调度或跨 CCD latency 数据。
- `sub-HNSW` 与 worker 一一映射，不涉及同一 HNSW 内多个线程如何共享局部 L3。
- 对当前 AMD EPYC CCD 平台，Pyramid 只能作为“路由与分区”相关工作，不能作为 chiplet-aware thread orchestration 的直接证据。
