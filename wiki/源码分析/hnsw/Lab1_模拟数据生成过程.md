# Lab1 模拟数据生成过程

## 范围

本文说明 `/home/zjj/HNSW/Lab1/lab1.py` 当前实验入口如何生成 HNSW 索引数据和查询数据，重点覆盖 `topic` 模式下 `topic-alpha-doc`、`topic-alpha-query` 和 `topic-anisotropy-strength` 的作用。

实际向量生成逻辑位于 `/home/zjj/HNSW/scripts/data_gen.py`。`lab1.py` 负责解析参数、确定 topic 数量、调用生成函数、构造 `same` / `similar` / `different` 三类查询集。

## 参数入口

`lab1.py` 中与模拟数据直接相关的参数如下：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--space` | `l2` | hnswlib 向量距离类型：`l2`、`ip`、`cosine` |
| `--dim` | `768` | 向量维度 |
| `--num-elements` | 必填 | 索引向量数量 |
| `--query-count` | `262144` | 基础查询数量 |
| `--threads` | `16` | fixed-rounds 调度下每轮并发查询数 |
| `--data-mode` | `uniform` | 索引数据分布：`uniform`、`clustered`、`realistic`、`topic` |
| `--query-mode` | `exact` | 查询分布：`exact`、`noisy`、`topic` |
| `--num-clusters` | `None` | cluster/topic 数量 |
| `--query-noise-std` | `0.3` | `noisy` 查询的高斯噪声标准差 |
| `--topic-alpha-doc` | `0.85` | topic center 与索引向量的 cosine similarity |
| `--topic-alpha-query` | `0.75` | topic center 与查询向量的 cosine similarity |
| `--topic-anisotropy-strength` | `0.2` | topic center 共享主方向强度 |

`num_clusters` 的有效值由 `resolve_query_num_clusters(args)` 决定：

$$
C =
\begin{cases}
\texttt{num\_clusters}, & \texttt{num\_clusters} \ne \texttt{None} \\
\max(1, \lfloor N / 1000 \rfloor), & \texttt{data\_mode} \in \{\texttt{clustered}, \texttt{realistic}, \texttt{topic}\} \text{ 或 } \texttt{query\_mode}=\texttt{topic} \\
\texttt{None}, & \texttt{data\_mode}=\texttt{uniform} \text{ 且 } \texttt{query\_mode}\ne\texttt{topic}
\end{cases}
$$

其中 \(N\) 是 `num_elements`，\(C\) 是 cluster/topic 数量。

## 总体流程

`lab1.py` 的数据流程如下：

```text
parse_args()
  -> resolve_query_num_clusters(args)
  -> resolve_benchmark_labels(args)
  -> build-index:
       make_data(num_elements, dim, seed, ...)
       index.add_items(data, ids, ...)
       index.save_index(index_path)
     load-index:
       index.load_index(index_path, ...)
       非 topic query:
         make_data(ref_size, dim, seed, ...)
       topic query:
         不生成 reference data
  -> prepare_query_sets(args, data, C, labels)
  -> benchmark(query_sets[label], label)
  -> output_json
```

构建索引时，`make_data(...)` 返回：

$$
\texttt{data} \in \mathbb{R}^{N \times d}, \quad
\texttt{ids} = [0, 1, \dots, N-1]
$$

`ids` 类型为 `uint64`，`data` 转为 C-contiguous 的 `float32` 数组。

加载已有索引时，若 `query_mode != "topic"`，脚本会额外生成一批 reference data（参考数据）用于构造 `exact` 或 `noisy` 查询：

$$
N_{\text{ref}} = \max(2Q, 50000)
$$

其中 \(Q\) 是 `query_count`。若 `query_mode == "topic"`，查询直接从共享 latent topic model（潜在主题模型）生成，不需要 reference data。

## 向量距离计算

`lab1.py` 通过：

```python
index = hnswlib.Index(space=args.space, dim=args.dim)
```

把 `--space` 传给 hnswlib。hnswlib 在 Python binding 中根据 `space` 选择距离函数：

- `l2`：使用 `L2Space`
- `ip`：使用 `InnerProductSpace`
- `cosine`：使用 `InnerProductSpace`，并在 `add_items(...)` 和 `knn_query(...)` / `knn_query_fixed_thread_rounds(...)` 入口对向量做 L2 归一化

HNSW 搜索路径中的上层 greedy search 和第 0 层 best-first search 都通过同一个 `fstdistfunc_` 计算 query 与候选节点向量的距离。候选比较采用“距离越小越近”的规则。

### l2

`space="l2"` 计算 squared L2 distance（平方欧氏距离），不取平方根：

$$
d_{\text{l2}}(q, x) = \sum_{j=1}^{d}(q_j - x_j)^2
$$

代码中的基础函数是 `L2Sqr(...)`。若编译和运行平台支持 SSE / AVX / AVX512，`L2Space` 会按维度选择 SIMD 版本或 residual 版本；SIMD 版本只改变计算实现，不改变数学定义。

### ip

`space="ip"` 先计算 inner product（内积）：

$$
\operatorname{ip}(q, x) = \sum_{j=1}^{d} q_j x_j
$$

hnswlib 返回的距离不是内积本身，而是：

$$
d_{\text{ip}}(q, x) = 1 - \operatorname{ip}(q, x)
$$

该定义把“内积越大越相似”转换为“距离越小越近”。代码中的基础函数是 `InnerProductDistance(...)`。SIMD 版本同样只改变点乘的实现方式，不改变距离定义。

### cosine

`space="cosine"` 在 binding 层仍使用 `InnerProductSpace`，但会先对索引向量和查询向量做 L2 归一化。对任意输入向量 \(v\)，归一化为：

$$
\hat v = \frac{v}{\lVert v \rVert_2 + 10^{-30}}
$$

随后距离仍按 inner product distance 计算：

$$
d_{\text{cosine}}(q, x)
= 1 - \hat q^\top \hat x
= 1 - \frac{q^\top x}{\lVert q \rVert_2 \lVert x \rVert_2}
$$

`cosine` 模式下，`add_items(...)` 存入 HNSW 的是归一化后的索引向量；查询时先归一化 query，再执行 HNSW 图遍历。若输入向量本身已经单位化，归一化不会改变其方向。

### topic 数据与距离类型的关系

`topic` 模式生成的 document 和 query 向量在 `_vectors_from_topics(...)` 末尾已经做行归一化：

$$
\lVert x \rVert_2 \approx 1,\quad \lVert q \rVert_2 \approx 1
$$

因此在 `data_mode="topic"` 且 `query_mode="topic"` 下：

- `space="ip"` 的距离近似为 \(1-q^\top x\)
- `space="cosine"` 的距离同样近似为 \(1-q^\top x\)，但 hnswlib 会在入口再次归一化
- `space="l2"` 的距离为 \(\lVert q-x \rVert_2^2\)

对于单位向量，squared L2 distance 与 inner product distance 单调等价：

$$
\lVert q-x \rVert_2^2
= \lVert q \rVert_2^2 + \lVert x \rVert_2^2 - 2q^\top x
\approx 2 - 2q^\top x
= 2d_{\text{ip}}(q, x)
$$

该等价只说明候选排序方向一致，不说明实际返回的 distance 数值相同。

## 索引数据生成

### uniform

`data_mode="uniform"` 时，每个维度独立采样：

$$
x_{i,j} \sim U(0, 1)
$$

其中 \(i \in [0, N)\)，\(j \in [0, d)\)。

### clustered

`data_mode="clustered"` 时，先生成 \(C\) 个 cluster centroid。原始 centroid 从标准正态分布采样并归一化到半径 \(\sqrt d\)：

$$
g_c \sim \mathcal{N}(0, I_d)
$$

$$
\mu_c = \sqrt d \cdot \frac{g_c}{\lVert g_c \rVert_2}
$$

cluster 权重来自 log-normal 分布：

$$
w_c \sim \operatorname{LogNormal}(0, 0.6)
$$

$$
p_c = \frac{w_c}{\sum_{r=1}^{C} w_r}
$$

每个 cluster 的样本数来自多项分布：

$$
(n_1, \dots, n_C) \sim \operatorname{Multinomial}(N, p)
$$

每个 cluster 有独立 spread：

$$
\sigma_c = 0.05 + 0.4 u_c,\quad u_c \sim U(0, 1)
$$

cluster \(c\) 中的向量为：

$$
x = \mu_c + \sigma_c \epsilon,\quad \epsilon \sim \mathcal{N}(0, I_d)
$$

所有向量生成后会打乱顺序，避免插入顺序与 cluster id 相关。

### realistic

`data_mode="realistic"` 先执行 `clustered` 生成，再调用 `_add_anisotropy(..., strength=0.5)` 添加主方向偏置。该 `strength=0.5` 是 `realistic` 模式内部固定值，不使用 `topic-anisotropy-strength`。

令共享主方向为：

$$
u = \frac{g}{\lVert g \rVert_2},\quad g \sim \mathcal{N}(0, I_d)
$$

代码中：

$$
\operatorname{scale}=1-0.6s
$$

其中 \(s=0.5\)。每个向量 \(x\) 被改写为：

$$
x' = \operatorname{scale}\cdot x + s (x^\top u) u
$$

该模式模拟 embedding 中向共享方向收缩的趋势，但不强制归一化。

### topic

`data_mode="topic"` 使用 latent topic model。该模式先生成 topic center 和 topic 权重，再按 topic 生成索引向量。

#### topic center

原始 topic center 从标准正态分布采样并归一化：

$$
g_c \sim \mathcal{N}(0, I_d)
$$

$$
\bar c_c = \frac{g_c}{\lVert g_c \rVert_2}
$$

若 `topic-anisotropy-strength` 为 \(s>0\)，代码额外生成共享主方向：

$$
u = \frac{g_u}{\lVert g_u \rVert_2},\quad g_u \sim \mathcal{N}(0, I_d)
$$

最终 topic center 为：

$$
c_c =
\frac{(1-s)\bar c_c + s u}
{\lVert (1-s)\bar c_c + s u \rVert_2}
$$

参数范围由代码检查：

$$
0 \le s < 1
$$

`topic-anisotropy-strength` 越大，不同 topic center 越共享同一个主方向。\(s=0\) 时，各 topic center 是单位球面上的随机方向。

#### topic 权重

topic 权重同样来自 log-normal 分布：

$$
w_c \sim \operatorname{LogNormal}(0, 0.6)
$$

$$
p_c = \frac{w_c}{\sum_{r=1}^{C} w_r}
$$

索引向量数量按 \(p\) 分配：

$$
(n_1, \dots, n_C) \sim \operatorname{Multinomial}(N, p)
$$

#### topic document 向量

`topic-alpha-doc` 记为 \(\alpha_d\)。代码要求：

$$
0 \le \alpha_d \le 1
$$

对 topic \(c\) 的每个 document，先采样噪声：

$$
\epsilon \sim \mathcal{N}(0, I_d)
$$

移除噪声在 topic center 上的投影：

$$
z = \epsilon - (\epsilon^\top c_c)c_c
$$

归一化：

$$
\hat z = \frac{z}{\lVert z \rVert_2}
$$

令：

$$
\beta_d = \sqrt{1-\alpha_d^2}
$$

document 向量为：

$$
x =
\frac{\alpha_d c_c + \beta_d \hat z}
{\lVert \alpha_d c_c + \beta_d \hat z \rVert_2}
$$

由于 \(\hat z\) 被投影到 \(c_c\) 的正交子空间，理想情况下：

$$
x^\top c_c \approx \alpha_d
$$

默认 \(\alpha_d=0.85\)，表示索引向量与所属 topic center 方向较接近，但不是完全重合。

## 查询数据生成

### exact

`query_mode="exact"` 时，从 reference data 中随机抽样：

$$
i_q \sim \operatorname{Uniform}\{0, \dots, N_{\text{ref}}-1\}
$$

$$
q = x_{i_q}
$$

该模式生成与索引/reference 分布完全一致的已有向量。

### noisy

`query_mode="noisy"` 时，先抽样 reference data，再加入高斯噪声：

$$
i_q \sim \operatorname{Uniform}\{0, \dots, N_{\text{ref}}-1\}
$$

$$
\eta \sim \mathcal{N}(0, \sigma_q^2 I_d)
$$

$$
q = x_{i_q} + \eta
$$

其中 \(\sigma_q=\texttt{query\_noise\_std}\)。代码不对 noisy query 做归一化。

### topic

`query_mode="topic"` 时，查询向量从与 `data_mode="topic"` 相同的 latent topic model 生成。只要 `seed`、`dim`、`num_clusters` 和 `topic-anisotropy-strength` 相同，query 使用的 topic center 与 topic index data 使用的 center 一致。

`topic-alpha-query` 记为 \(\alpha_q\)。代码要求：

$$
0 \le \alpha_q \le 1
$$

给定查询所属 topic \(t\)，查询向量使用与 document 相同的正交噪声构造：

$$
\epsilon \sim \mathcal{N}(0, I_d)
$$

$$
z = \epsilon - (\epsilon^\top c_t)c_t
$$

$$
\hat z = \frac{z}{\lVert z \rVert_2}
$$

$$
\beta_q = \sqrt{1-\alpha_q^2}
$$

$$
q =
\frac{\alpha_q c_t + \beta_q \hat z}
{\lVert \alpha_q c_t + \beta_q \hat z \rVert_2}
$$

理想情况下：

$$
q^\top c_t \approx \alpha_q
$$

默认 \(\alpha_q=0.75\)，表示 query 与 topic center 相关，但比默认 document 更分散。

同一 topic 下 document 与 query 的内积期望近似为：

$$
\mathbb{E}[x^\top q \mid t_x=t_q] \approx \alpha_d \alpha_q
$$

默认参数下：

$$
\alpha_d \alpha_q = 0.85 \times 0.75 = 0.6375
$$

该值只是同 topic 的方向相关性近似，不等价于 HNSW 搜索距离的完整分布；不同 topic 的相似度还受 topic center 之间夹角和 `topic-anisotropy-strength` 影响。

## same、similar 与 different 查询集

`lab1.py` 通过 `resolve_benchmark_labels(args)` 决定运行哪些 workload：

- `--mode all`：运行 `same`、`similar`、`different`
- `--mode same`：只运行 `same`
- `--mode similar`：只运行 `similar`
- `--mode different`：只运行 `different`

`similar` 只允许在 `query_mode="topic"` 下使用。

### topic query pool

当 `query_mode="topic"` 时，`prepare_query_sets(...)` 调用 `make_topic_query_pool(...)`。它先生成一份按 round 分组的 `similar` query，再生成同一批 query 的随机打乱视图作为 `different`。

设并发线程数为 \(T\)，基础查询数为 \(Q\)，要求：

$$
Q \bmod T = 0
$$

round 数量为：

$$
R = Q / T
$$

`similar` 的 topic 分配规则为：每个 round 采样一个 topic，同一 round 内 \(T\) 个 query 使用相同 topic：

$$
t_r \sim \operatorname{Categorical}(p)
$$

$$
topic(rT), topic(rT+1), \dots, topic(rT+T-1) = t_r
$$

每个 query 的噪声独立，因此同 round query 属于同一 topic，但向量不完全相同。

`different` 是 `similar` 中同一批 query 的随机排列：

$$
Q_{\text{different}} = Q_{\text{similar}}[\pi]
$$

其中 \(\pi\) 由 `seed + 4000` 产生。`different` 不重新采样 query，只改变同一批 query 的顺序，降低 fixed-rounds 同轮查询属于同一 topic 的概率。

### same

topic 模式下的 `same` 由 `different` pool 构造：

```python
query_sets["same"] = np.repeat(pool["different"], args.threads, axis=0)
```

即每个基础 query 连续重复 \(T\) 次：

$$
q_{iT}, q_{iT+1}, \dots, q_{iT+T-1} = \tilde q_i
$$

因此 fixed-rounds 调度下，同一 round 的 \(T\) 个线程执行完全相同的 query 向量。

topic 模式下 `same` 不使用 C++ 侧 `same_query=True` 快捷路径。`use_cpp_same_query(args, "same")` 在 `query_mode="topic"` 时返回 `False`，避免绕过 Python 侧已经构造好的 repeated query set。

### fixed-rounds 下的扩展

当 `scheduler="fixed-rounds"` 且 `query_mode="topic"` 时，`prepare_query_sets(...)` 会对 `similar` 和 `different` 执行：

```python
query_sets[label] = np.tile(query_sets[label], (args.threads, 1))
```

因此：

- `similar` 行数从 \(Q\) 扩展到 \(QT\)
- `different` 行数从 \(Q\) 扩展到 \(QT\)
- `same` 在 `np.repeat(..., T)` 后已经是 \(QT\) 行，不再额外 tile

该处理使三个 workload 在 fixed-rounds 路径下执行相同数量的 query row。

## 随机种子

相关随机数生成器使用不同 seed 偏移：

| 位置 | seed |
| --- | --- |
| `uniform` / `clustered` / `realistic` 基础数据 | `seed` |
| topic model 的 center 与权重 | `seed + 3000` |
| topic document 数量与 document 噪声 | `seed + 2000` |
| topic query 的 topic 抽样与 query 噪声 | `seed + 1000` |
| topic query pool 的 `different` shuffle | `seed + 4000` |
| `exact` / `noisy` query 抽样 | `seed + 1000` |

因此 topic document 和 topic query 共享同一组 latent topic center 与 topic 权重，但 document 向量和 query 向量独立采样。

## 参数含义

### `topic-alpha-doc`

`topic-alpha-doc` 控制索引向量贴近所属 topic center 的程度：

$$
x = \operatorname{normalize}(\alpha_d c + \sqrt{1-\alpha_d^2} \hat z)
$$

\(\alpha_d\) 越大，同一 topic 的索引向量越集中，HNSW 中相同 topic 的近邻关系越强。默认值 `0.85` 对应较强 topic 聚集。

### `topic-alpha-query`

`topic-alpha-query` 控制查询向量贴近所属 topic center 的程度：

$$
q = \operatorname{normalize}(\alpha_q c + \sqrt{1-\alpha_q^2} \hat z)
$$

\(\alpha_q\) 越大，query 越接近 topic center；\(\alpha_q\) 越小，query 在 topic 内越分散。默认值 `0.75` 使 query 与 topic 相关，但弱于 document 的默认聚集强度。

### `topic-anisotropy-strength`

`topic-anisotropy-strength` 控制 topic center 是否共享主方向：

$$
c = \operatorname{normalize}((1-s)\bar c + su)
$$

\(s=0\) 时，topic center 只来自随机单位方向。\(s>0\) 时，所有 topic center 都混入同一个 \(u\)，topic 分布出现共同方向偏置。默认值 `0.2` 表示保留 topic 差异，同时加入弱主方向偏置。

该参数只传给 `topic` 数据和 `topic` 查询。`realistic` 模式的 `_add_anisotropy(..., strength=0.5)` 使用独立固定强度。

## 当前实验语义

在 `data_mode="topic"` 且 `query_mode="topic"` 的实验中：

- 索引向量和查询向量不是互相抽样得到的同一批向量。
- 二者共享 latent topic center 和 topic 权重。
- `topic-alpha-doc` 决定索引向量围绕 topic center 的集中程度。
- `topic-alpha-query` 决定查询向量围绕 topic center 的集中程度。
- `topic-anisotropy-strength` 决定不同 topic center 是否朝共同方向收缩。
- `similar` 让同一 fixed round 内的线程查询同一 topic 的不同 query。
- `same` 让同一 fixed round 内的线程查询完全相同的 query。
- `different` 使用与 `similar` 相同的 query multiset，但打乱顺序，作为降低同轮 topic 重合的对照。
