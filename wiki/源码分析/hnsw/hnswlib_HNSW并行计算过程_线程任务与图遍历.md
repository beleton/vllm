# hnswlib HNSW 并行计算过程：线程、任务与图遍历

## 范围

- 只覆盖 `/home/zjj/hnswlib` 当前代码中的 HNSW 主路径：
  - `python_bindings/bindings.cpp`
  - `hnswlib/hnswalg.h`
  - `hnswlib/visited_list_pool.h`
  - `hnswlib/space_l2.h`
  - `hnswlib/space_ip.h`
- 重点分析：
  - `add_items(...)` 的批量构建并行方式
  - `knn_query(...)` 的批量查询并行方式
  - `searchKnn(...)`、`searchBaseLayerST(...)` 和 `addPoint(...)` 的单任务内部流程
  - 线程之间是否通信、共享哪些状态、在哪些位置加锁
- 不覆盖：
  - `BruteforceSearch`
  - C++ 示例中的自定义 `ParallelFor`
  - pickle、save/load、metadata 导出路径
  - HNSW 论文中的算法变体

## 核心结论

- hnswlib 的 Python binding 通过 `ParallelFor(...)` 做 batch 级并行，任务粒度是输入矩阵的一行 `row`。
- 一个查询任务等价于：

```text
一个 query row
+ 一次完整 searchKnn(...)
+ 输出 labels[row, :] 和 distances[row, :]
```

- 单个 `searchKnn(...)` 内部没有多线程拆分。上层贪婪导航和第 0 层 best-first 搜索都在当前线程内串行执行。
- 批量查询中的线程之间不交换候选集、不做 reduction、不共享单个 query 的中间状态。
- 查询路径主要共享只读索引数据：
  - `data_level0_memory_`
  - `linkLists_`
  - `element_levels_`
  - `enterpoint_node_`
  - `maxlevel_`
- 查询路径仍有少量共享同步：
  - `ParallelFor` 用全局原子计数器分发 row
  - `VisitedListPool` 用 `poolguard` 分配和回收 visited bitmap
  - metric 计数器是原子变量
  - 若使用 Python filter，会从 C++ 线程回调 Python 函数，README 建议 `num_threads=1`
- 批量插入也是 row 级并行，但单个 `addPoint(...)` 会修改共享图。插入线程之间通过 label 锁、label lookup 锁、节点邻接表锁、deleted set 锁和全局入口锁同步。
- 查询与插入不应并发。README 明确说明 `knn_query` 可与其他 `knn_query` 并发，但不能与 `add_items` 并发；`add_items` 可与其他 `add_items` 并发，但不能与 `knn_query` 并发。
- hnswlib 没有按 `ef`、候选队列、图层、邻居列表或向量维度把单个 query 拆给多个线程。提升 `num_threads` 只增加同时处理的 query/insert row 数量。

## 1. 调用链

### 1.1 Index 创建与 init_index

`hnswlib.Index(space, dim)` 只创建 Python binding wrapper 和距离空间对象，不分配 HNSW 图结构：

```text
hnswlib.Index(space, dim)
  -> pybind11: py::init<const std::string &, const int>()
  -> Index<float>::Index(space_name, dim)
       -> l2:     new hnswlib::L2Space(dim)
       -> ip:     new hnswlib::InnerProductSpace(dim)
       -> cosine: new hnswlib::InnerProductSpace(dim), normalize = true
       -> appr_alg = NULL
       -> index_inited = false
       -> default_ef = 10
```

`cosine` 底层使用 inner product 距离，但在 `add_items(...)` 和 `knn_query(...)` 入口归一化向量。

真正的 HNSW 索引对象在 `init_index(...)` 时创建：

```text
hnswlib.Index.init_index(max_elements, M, ef_construction, random_seed, allow_replace_deleted)
  -> Index<float>::init_new_index(...)
       -> if appr_alg already exists: throw
       -> cur_l = 0
       -> appr_alg = new hnswlib::HierarchicalNSW<float>(
            l2space, maxElements, M, efConstruction, random_seed, allow_replace_deleted)
       -> index_inited = true
       -> ep_added = false
       -> appr_alg->ef_ = default_ef
```

`HierarchicalNSW` 构造函数完成空索引初始化：

```text
max_elements_ = max_elements
data_size_ = s->get_data_size()
fstdistfunc_ = s->get_dist_func()
dist_func_param_ = s->get_dist_func_param()

M_ = M
maxM_ = M
maxM0_ = 2 * M
ef_construction_ = max(ef_construction, M)

size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint)
size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype)
offsetLevel0_ = 0
offsetData_ = size_links_level0_
label_offset_ = size_links_level0_ + data_size_

data_level0_memory_ = malloc(max_elements_ * size_data_per_element_)
linkLists_ = malloc(sizeof(void*) * max_elements_)
link_list_locks_ = vector<mutex>(max_elements)
element_levels_ = vector<int>(max_elements)

cur_element_count = 0
enterpoint_node_ = -1
maxlevel_ = -1
```

此时只是预分配容量和初始化元数据，还没有任何节点或边。节点、层数和邻接表要到 `add_items(...) -> addPoint(...)` 时才生成。

### 1.2 查询路径

```text
hnswlib.Index.knn_query(data, k, num_threads, filter)
  -> Index<float>::knnQuery_return_numpy(...)
       -> 解析输入 rows/features
       -> rows <= num_threads * 4 时退化为单线程
       -> 释放 Python GIL
       -> ParallelFor(0, rows, num_threads, row_task)
            -> 每个线程动态领取一个 row
            -> appr_alg->searchKnn(items.data(row), k, filter)
                 -> 上层从 maxlevel_ 到 1 做 greedy search
                 -> 第 0 层 searchBaseLayerST(...)
                 -> 裁剪 top_candidates 到 k
                 -> internal id 转 external label
            -> 写 data_numpy_l[row * k + i]
            -> 写 data_numpy_d[row * k + i]
       -> 返回 numpy labels/distances
```

### 1.3 构建路径

```text
hnswlib.Index.add_items(data, ids, num_threads, replace_deleted)
  -> Index<float>::addItems(...)
       -> 解析输入 rows/features/ids
       -> rows <= num_threads * 4 时退化为单线程
       -> 若 index 还没有 entry point，先串行插入第 0 个元素
       -> 释放 Python GIL
       -> ParallelFor(start, rows, num_threads, row_task)
            -> 每个线程动态领取一个 row
            -> appr_alg->addPoint(items.data(row), label, replace_deleted)
                 -> label 级互斥
                 -> 分配或复用 internal id
                 -> 分配层数和节点存储
                 -> 上层 greedy search 找入口
                 -> 各层 searchBaseLayer(...)
                 -> mutuallyConnectNewElement(...) 改写双向连接
       -> cur_l += rows
```

## 2. 概念表

| 名称 | 层次 | 含义 |
| --- | --- | --- |
| `num_threads` | binding 参数 | `add_items` 或 `knn_query` 使用的 C++ worker 线程数 |
| `ParallelFor` | batch 调度层 | 用 `std::atomic<size_t> current` 动态分发 row |
| `row` | runtime task 层 | 输入矩阵中的一行；查询时是一个 query，构建时是一个待插入向量 |
| `threadId` | worker 标号 | `ParallelFor` 创建线程时分配的固定编号，用于选择线程本地归一化缓冲区 |
| `data_point` | 单点插入参数 | 指向一条待插入向量的原始内存指针，float 索引中实际是 `float*` |
| `label` | 用户可见 id | `add_items(..., ids=...)` 传入的 external label；查询结果返回的是它 |
| `internal id` / `tableint` | 图节点编号 | hnswlib 内部分配的连续节点编号，邻接表中存的是 internal id |
| `searchKnn` | 单 query 计算层 | 完整 HNSW 查询流程，当前实现为单线程串行 |
| `candidate_set` | 第 0 层搜索状态 | 待扩展节点候选堆，保存 `(-distance, internal_id)` |
| `top_candidates` | 第 0 层搜索状态 | 当前结果候选堆，保存 `(distance, internal_id)` |
| `M_` / `maxM_` / `maxM0_` | 图度数参数 | 高层最多 `M` 个邻居；第 0 层容量是 `2M` |
| `VisitedList` | 单次搜索临时状态 | 记录当前 query 已访问节点，避免重复扩展 |
| `link_list_locks_` | 图结构同步层 | 每个 internal id 一个锁，用于插入/更新时保护邻接表 |
| `label_lookup_lock` | label 映射同步层 | 保护 external label 到 internal id 的哈希表 |
| `global` | 入口同步层 | 插入高层节点时保护 `enterpoint_node_` 和 `maxlevel_` 更新 |

## 3. batch 任务划分

### 3.1 `ParallelFor`

`ParallelFor(start, end, numThreads, fn)` 是 hnswlib Python binding 的统一并行入口。它不是 OpenMP，也不是长期存在的线程池，而是在一次 `add_items(...)` 或 `knn_query(...)` 调用内直接创建 `std::thread`，调用结束前 `join()` 回收。

多线程模式下，它只维护一个共享原子计数器：

```cpp
std::atomic<size_t> current(start);
...
while (true) {
    size_t id = current.fetch_add(1);
    if (id >= end) {
        break;
    }
    fn(id, threadId);
}
```

任务空间是连续整数区间 `[start, end)`。对 HNSW binding 来说：

- `id` 就是输入矩阵的 `row`
- 每次 `fetch_add(1)` 领取一个 row
- 不预先给线程静态切块
- 某个 row 搜索较慢时，其他线程可以继续领取后续 row
- `threadId` 是 worker 创建时分配的固定编号，不代表固定 row 区间
- 同一个 worker 可能处理多个不连续的 row，具体取决于各 row 的搜索耗时和原子领取时序
- row 的领取顺序由 `fetch_add(1)` 保证全局唯一递增，但不同线程完成 row 的顺序不固定

以 `knn_query(queries, k, num_threads=16)` 且 `queries.shape[0] == 65536` 为例，`ParallelFor(0, 65536, 16, row_task)` 创建 16 个 worker。16 个 worker 共享一个 `current` 计数器，初值为 0。某个 worker 领取到 `row=0` 后处理 `queries[0]`，另一个 worker 领取到 `row=1` 后处理 `queries[1]`。worker 完成当前 row 后再次执行 `fetch_add(1)` 领取下一个未处理 row，直到计数器达到 65536。代码不保证第 0 个线程处理 `[0, 4096)`，第 1 个线程处理 `[4096, 8192)` 这类静态分段。

异常处理通过共享的 `lastException` 和 `lastExceptMutex` 完成。任一线程捕获异常后设置 `current = end`，其余线程停止领取新任务，join 后由主线程重新抛出异常。

### 3.2 小 batch 退化

`addItems(...)` 和 `knnQuery_return_numpy(...)` 都包含相同规则：

```cpp
if (rows <= num_threads * 4) {
    num_threads = 1;
}
```

含义：

- 当平均每个线程不到约 4 个 row 时，不创建多线程
- 小 batch 下没有并行搜索或并行插入
- `num_threads` 不是单个 query 内的线程数，只对 batch 行数足够大时生效

### 3.3 cosine 归一化缓冲区

当 `space_name == "cosine"` 时，binding 使用 inner product 距离并在入口处归一化向量。多线程路径中会分配：

```cpp
std::vector<float> norm_array(num_threads * dim);
size_t start_idx = threadId * dim;
normalize_vector(items.data(row), norm_array.data() + start_idx);
```

每个 worker 使用自己的 `[threadId * dim, (threadId + 1) * dim)` 缓冲区。不同线程不会写同一段归一化缓冲区。

## 4. 查询路径

### 4.1 查询 task 单元

`knn_query(data, k, num_threads)` 的 runtime task 是单个 row：

```text
task(row):
  query = data[row]
  result = searchKnn(query, k)
  for i in k-1..0:
    labels[row, i] = result.top().label
    distances[row, i] = result.top().distance
```

每个 row 的输出区间固定为：

```text
labels[row * k, row * k + k)
distances[row * k, row * k + k)
```

因此输出数组写入没有跨线程冲突。线程之间不需要归并结果。

动态领取只影响 batch 内 row 到 worker 的分配，不改变单个 row 的计算内容。某个线程一旦领取 `row`，该 row 的完整 `searchKnn(...)`、第 0 层候选堆、visited 标记和结果写回都在该线程内完成。其他线程不会参与这个 row 的图遍历，也不会读取或合并这个 row 的候选队列。

### 4.2 `searchKnn` 上层导航

`searchKnn(...)` 首先从全局入口点开始：

```cpp
tableint currObj = enterpoint_node_;
dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
```

随后从 `maxlevel_` 逐层下降到第 1 层。每一层执行贪婪搜索：

```text
while changed:
  changed = false
  读取 currObj 在当前 level 的邻接表
  for cand in neighbors(currObj, level):
    d = distance(query, vector(cand))
    if d < curdist:
      currObj = cand
      curdist = d
      changed = true
```

该阶段只保留一个当前节点 `currObj`。如果某个邻居更近，就移动到该邻居并继续扫描新节点的邻接表；如果本轮没有更近邻居，就下降一层。

上层导航的并行属性：

- 单个 query 内串行执行
- 不维护共享候选队列
- 不加 `link_list_locks_`
- 直接读取 `linkLists_[internal_id]` 中的高层邻接表
- 距离计算调用 `fstdistfunc_`，具体实现来自 `L2Space` 或 `InnerProductSpace`

### 4.3 第 0 层 best-first 搜索

上层导航得到第 0 层入口 `currObj` 后，`searchKnn(...)` 调用：

```cpp
searchBaseLayerST(currObj, query_data, std::max(ef_, k), isIdAllowed)
```

`ef_` 是查询阶段的搜索宽度。实际传入宽度是 `max(ef_, k)`。

`searchBaseLayerST(...)` 使用三个核心状态：

- `VisitedList`：当前 query 的已访问标记
- `candidate_set`：待扩展候选堆
- `top_candidates`：当前结果候选堆

初始化：

```text
dist = distance(query, entry_point)
lowerBound = dist
top_candidates = {(dist, entry_point)}
candidate_set = {(-dist, entry_point)}
visited[entry_point] = current_tag
```

主循环：

```text
while candidate_set not empty:
  current = candidate_set.top()
  candidate_dist = -current.first

  if stop_condition(candidate_dist, lowerBound):
    break

  candidate_set.pop()
  neighbors = level0_neighbors(current_node)

  for candidate_id in neighbors:
    if visited[candidate_id]:
      continue
    visited[candidate_id] = true

    dist = distance(query, vector(candidate_id))
    if top_candidates.size < ef or dist < lowerBound:
      candidate_set.push((-dist, candidate_id))
      if allowed(candidate_id):
        top_candidates.push((dist, candidate_id))
      while top_candidates.size > ef:
        top_candidates.pop()
      lowerBound = top_candidates.top().distance
```

停止条件分两种：

- `bare_bone_search == true`：没有删除节点且没有 filter，只判断 `candidate_dist > lowerBound`
- `bare_bone_search == false`：需要检查 deleted/filter；无自定义 stop condition 时，还要求 `top_candidates.size() == ef`

查询完成后，`searchKnn(...)` 将 `top_candidates` 裁剪到 `k` 个，再把 internal id 转成 external label。

### 4.4 单 query 内部切分层次

单个 query 的计算层次是：

```text
query row
  -> upper-layer greedy search
       -> level
          -> greedy iteration
             -> neighbor scan
                -> distance(query, candidate)
  -> level-0 best-first search
       -> candidate pop
          -> neighbor scan
             -> distance(query, candidate)
             -> heap update
  -> result materialization
```

没有以下切分：

- 没有按图层并行
- 没有按候选堆并行
- 没有按邻居列表并行
- 没有按向量维度并行给多个线程
- 没有 batch 内 query 之间的共享候选或共享 visited 状态

### 4.5 距离计算

距离函数由 `SpaceInterface` 提供：

- `l2` 使用平方 L2 距离
- `ip` 使用 `1.0f - inner_product`
- `cosine` 在 binding 层先归一化，再使用 inner product 距离

`space_l2.h` 和 `space_ip.h` 中包含 SSE/AVX/AVX512 版本。SIMD 只发生在单线程内部的距离函数中。它不改变 HNSW 的线程任务划分。

## 5. 查询线程通信与共享状态

### 5.1 查询线程之间不共享的状态

每个 query task 独占：

- 当前 query 指针
- `searchKnn(...)` 局部变量
- 上层 `currObj/curdist`
- 第 0 层 `candidate_set`
- 第 0 层 `top_candidates`
- 当前搜索借出的 `VisitedList`
- 输出数组中自己的 row 区间

这些状态不需要线程间通信。

### 5.2 查询线程共享但基本只读的状态

所有查询线程共享同一个 `HierarchicalNSW` 对象。查询时主要读取：

- `enterpoint_node_`
- `maxlevel_`
- `data_level0_memory_`
- `linkLists_`
- `element_levels_`
- `size_data_per_element_`
- `offsetData_`
- `offsetLevel0_`
- `label_offset_`
- `fstdistfunc_`
- `dist_func_param_`

只读共享意味着多个线程可能访问同一批热节点、邻接表和向量，但代码没有显式把这些访问按 CCD、NUMA 或 cache locality 分组。

### 5.3 查询路径上的同步点

| 位置 | 同步对象 | 作用 | 对查询计算的影响 |
| --- | --- | --- | --- |
| row 调度 | `std::atomic<size_t> current` | 动态领取 batch row | 每个 row 领取一次 |
| 异常传播 | `lastExceptMutex` | 记录最后一个异常 | 仅异常路径 |
| visited list 分配 | `VisitedListPool::poolguard` | 借出/归还 `VisitedList` | 每次第 0 层搜索借还各一次 |
| metrics | `metric_hops`、`metric_distance_computations` | 统计 hop 和距离计算数 | 原子加；当前 README 说明默认不聚合统计以加速多线程搜索 |
| Python filter | Python 回调路径 | 判断 label 是否允许返回 | README 建议 filter 场景 `num_threads=1` |

查询主循环中没有锁住邻接表。`searchBaseLayerST(...)` 直接读取 level 0 邻接表；上层 greedy search 也直接读取高层邻接表。因此查询路径依赖“不与插入并发”的使用约束。

## 6. 构建路径

### 6.1 构建 task 单元

`add_items(data, ids, num_threads, replace_deleted)` 的 runtime task 也是单个 row：

```text
task(row):
  label = ids[row] 或 cur_l + row
  vector = data[row] 或 normalized(data[row])
  addPoint(vector, label, replace_deleted)
```

其中 `vector` 作为 `const void *data_point` 传入 C++ core。对 `Index<float>` 来说，它实际指向一条连续的 `float[dim]`。`label` 是用户传入的 external id；如果没有传 `ids`，binding 使用 `cur_l + row` 作为默认 label。

批量构建与批量查询使用同一个 `ParallelFor` 动态分发机制。

### 6.2 首个节点串行插入

如果 index 刚初始化，还没有 entry point，binding 会先串行插入第一个 row：

```cpp
if (!ep_added) {
    appr_alg->addPoint(items.data(0), id, replace_deleted);
    start = 1;
    ep_added = true;
}
```

这样后续并行插入时已有合法的 `enterpoint_node_`。

### 6.3 `addPoint(data_point, label, replace_deleted)`

外层 `addPoint(...)` 先按 label 加锁：

```cpp
std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
```

`label_op_locks_` 有 65536 个 mutex，通过 label 低位映射。它的作用是避免相同 label 的并发插入、删除、更新互相交错。

如果 `replace_deleted == true`，还会使用 `deleted_elements_lock` 从 `deleted_elements` 中取一个已删除 internal id。否则进入内部插入：

```cpp
addPoint(data_point, label, -1)
```

### 6.4 internal id 分配与 label 映射

内部 `addPoint(data_point, label, level)` 先锁住 `label_lookup_lock`：

```cpp
std::unique_lock<std::mutex> lock_table(label_lookup_lock);
```

该锁保护：

- 检查 label 是否已存在
- 更新已有 label 时走 `updatePoint(...)`
- 检查 `cur_element_count >= max_elements_`
- 分配新的 `cur_c = cur_element_count`
- `cur_element_count++`
- `label_lookup_[label] = cur_c`

`cur_element_count` 是原子变量，但 label 映射和容量检查仍在 `label_lookup_lock` 临界区内完成。

### 6.5 新节点初始化与入口锁

分配 internal id 后，线程锁住新节点自己的邻接表锁：

```cpp
std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
```

随后随机生成层数 `curlevel`，写入：

- `element_levels_[cur_c]`
- `data_level0_memory_` 中的 label、向量和第 0 层邻接表区域
- `linkLists_[cur_c]` 中的高层邻接表区域

每个新增节点都会随机生成一次层数：

```cpp
int curlevel = getRandomLevel(mult_);
```

正常 `add_items(...)` 不显式传 `level`，因此 `curlevel` 由随机数决定。`mult_ = 1 / log(M_)`，层数服从指数衰减分布，近似可以理解为 `P(level >= L) ~= M^{-L}`。如果某次随机出比当前 `maxlevel_` 更高的层，该节点会成为新的 `enterpoint_node_`，并把 `maxlevel_` 提高到该层。第一个节点也可以随机出高层；此时高层只有这个节点，邻接表为空。

接着进入全局入口锁：

```cpp
std::unique_lock<std::mutex> templock(global);
int maxlevelcopy = maxlevel_;
if (curlevel <= maxlevelcopy)
    templock.unlock();
```

含义：

- 如果新节点层数不超过当前最高层，复制 `maxlevel_` 和 `enterpoint_node_` 后释放全局锁
- 如果新节点层数高于当前最高层，保留全局锁直到末尾更新 `enterpoint_node_` 和 `maxlevel_`

全局锁只保护入口和最高层元数据，不包住普通插入的完整搜索和连边流程。

### 6.6 插入时的上层导航与第 0 层/低层搜索

若图非空，插入线程先从 `enterpoint_node_` 出发。

当 `curlevel < maxlevelcopy` 时，在高于新节点层数的层上执行 greedy search：

```text
for level = maxlevelcopy down to curlevel + 1:
  while changed:
    读取 currObj 在该 level 的邻接表
    找到更接近 data_point 的邻居则移动 currObj
```

该过程在读取每个 `currObj` 邻接表时会短暂锁住：

```cpp
std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
```

之后从 `min(curlevel, maxlevelcopy)` 到第 0 层逐层执行：

```text
top_candidates = searchBaseLayer(currObj, data_point, level)
currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false)
```

`searchBaseLayer(...)` 与查询用的 `searchBaseLayerST(...)` 类似，但用于构建，宽度固定为 `ef_construction_`。它在扩展每个节点时锁住该节点的邻接表：

```cpp
std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);
```

### 6.7 双向连边

`mutuallyConnectNewElement(...)` 完成两类写入。

第一类是写新节点 `cur_c` 的邻接表：

```text
selectedNeighbors = heuristic(top_candidates, M_)
ll_cur = linklist(cur_c, level)
setListCount(ll_cur, selectedNeighbors.size())
ll_cur[i] = selectedNeighbors[i]
```

普通插入时，外层 `addPoint(...)` 已经持有 `link_list_locks_[cur_c]`，所以这里不再重复加锁。更新路径 `isUpdate == true` 才会在函数内部锁 `cur_c`。

第二类是改写每个被选邻居的反向连接：

```text
for neighbor in selectedNeighbors:
  lock(link_list_locks_[neighbor])
  if neighbor adjacency has room:
    append cur_c
  else:
    candidates = old_neighbors + cur_c
    heuristic(candidates, Mcurmax)
    rewrite neighbor adjacency
```

这是并行插入中最主要的共享写路径。多个插入线程如果同时连接到同一个热点节点，会在该节点的 `link_list_locks_[neighbor]` 上串行化。

### 6.8 更新已有 label

如果 label 已存在，内部 `addPoint(...)` 不创建新节点，而是执行 `updatePoint(...)`：

```text
复制新 vector 到 existing internal id
按层收集一跳和二跳邻居
对相关邻居重算候选连接
锁住每个受影响邻居的 link_list_locks_[neigh] 并改写邻接表
repairConnectionsForUpdate(...)
```

更新路径比普通新插入更重，因为它会修复已有节点附近的连接。

## 7. 构建线程通信与共享状态

### 7.1 构建线程共享并写入的状态

构建线程会并发读写：

- `cur_element_count`
- `label_lookup_`
- `data_level0_memory_`
- `linkLists_`
- `element_levels_`
- `enterpoint_node_`
- `maxlevel_`
- 每个节点的邻接表
- `deleted_elements`

因此构建线程之间存在真实同步和等待。

### 7.2 构建路径锁表

| 锁 | 粒度 | 使用位置 | 作用 |
| --- | --- | --- | --- |
| `label_op_locks_[label & 65535]` | label hash | 外层 `addPoint`、delete/unmark/getData | 串行化同 label 操作 |
| `label_lookup_lock` | 全局 label map | internal id 分配、label 查询、label 替换 | 保护 `label_lookup_` 和分配流程 |
| `deleted_elements_lock` | 全局 deleted set | replace deleted、mark/unmark deleted | 保护可复用 internal id 集合 |
| `global` | 全局入口 | 插入高层节点、pickle/setAnnData | 保护 `enterpoint_node_`、`maxlevel_` |
| `link_list_locks_[id]` | 单节点邻接表 | 构建搜索、连边、更新 | 保护节点邻接表读写 |

### 7.3 插入线程之间的通信性质

代码中没有消息队列、barrier 或显式任务依赖图。插入线程之间通过共享图结构和锁隐式协调：

- internal id 分配通过 `label_lookup_lock` 串行化
- 高层入口更新通过 `global` 串行化
- 热点节点邻接表更新通过 `link_list_locks_[id]` 串行化
- 删除复用通过 `deleted_elements_lock` 串行化

因此构建路径不是 embarrassingly parallel。不同 row 的插入顺序会受动态调度和锁等待影响；HNSW 图结构本身也与插入顺序有关。

### 7.4 并行插入的语义边界

`add_items(..., num_threads > 1)` 的线程安全含义是：多个 `addPoint(...)` 可以并发修改同一个 `HierarchicalNSW` 对象而不破坏内存结构。它不保证并行构建得到的图与单线程固定顺序构建完全相同。

原因是插入新点时要从当前图的 `enterpoint_node_` 出发搜索候选邻居。两个线程同时插入新节点 A 和 B 时，可能出现：

```text
线程 1 插入 A：搜索时 B 还没有完成连边，因此不可达
线程 2 插入 B：搜索时 A 还没有完成连边，因此不可达
```

这种情况下 A 和 B 即使彼此很近，也不一定在本次插入中直接建立边。代码保证的是：

- internal id 分配不会冲突
- 同一个 label 的操作不会交错
- 同一个节点邻接表不会被多个线程同时写坏
- `enterpoint_node_` 和 `maxlevel_` 更新受 `global` 保护

但 HNSW 图是近似图，不要求所有近邻都直接连边。A 和 B 仍可能通过共同邻居间接可达；后续插入的新点也可能改善局部连接。如果实验需要固定插入顺序和更稳定的图结构，应使用 `num_threads=1` 构建。

## 8. 索引内存布局与访存路径

### 8.1 第 0 层

第 0 层存储在连续大块 `data_level0_memory_` 中。每个元素占用 `size_data_per_element_` 字节：

```text
[level0 link list][vector data][external label]
```

相关 offset：

- `offsetLevel0_ = 0`
- `offsetData_ = size_links_level0_`
- `label_offset_ = size_links_level0_ + data_size_`

访问 internal id 的向量：

```cpp
data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_
```

访问 internal id 的第 0 层邻接表：

```cpp
data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_
```

第 0 层邻接表本身是定长结构：

```text
byte offset
0      2      3      4
|------|------|------|--------------------------------|
|count |del   |pad   | neighbor internal ids           |
|u16   |flags |      | tableint[maxM0_]                |
|------|------|------|--------------------------------|
```

对应容量：

```text
size_links_level0_ = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint)
maxM0_ = 2 * M
```

`getListCount(...)` 读取 header 前 2 字节作为邻居数量；删除标记写在 header 偏移 2 的字节上。邻接表中的邻居不是 external label，而是 internal id。

因此 `M` 在 hnswlib 中不是所有层的硬上限：

```text
level 0:  每个节点邻接表容量为 2M
level >= 1: 每个节点每层邻接表容量为 M
```

新节点主动选择邻居时通常先用 heuristic 选最多 `M` 个；但由于后续其他节点可能向它追加反向边，第 0 层实际度数可以增长到 `2M`。

正常新增节点时，`data_level0_memory_` 按 internal id 连续排列：

```text
slot 0 -> internal id 0
slot 1 -> internal id 1
slot 2 -> internal id 2
...
```

单线程构建时 internal id 与插入顺序一致。多线程构建时 internal id 按线程进入 `label_lookup_lock` 临界区的实际顺序分配，不保证等于 numpy row 顺序。若 label 已存在则更新原 slot；若启用 replace deleted，则可能复用已删除节点的 slot。

### 8.2 高层

高层邻接表存储在 `linkLists_[internal_id]` 中。只有 `element_levels_[internal_id] > 0` 的节点才有高层内存。

访问某节点某高层：

```cpp
linkLists_[internal_id] + (level - 1) * size_links_per_element_
```

`linkLists_` 本身是按 internal id 连续排列的指针数组：

```text
linkLists_[0]
linkLists_[1]
linkLists_[2]
...
```

但每个 `linkLists_[i]` 指向的高层邻接表内存是在节点插入时按需 `malloc` 的：

```text
linkLists_[i] == nullptr
    -> 节点 i 只有 level 0

linkLists_[i] != nullptr
    -> 节点 i 有 level 1..element_levels_[i] 的邻接表
```

单个节点自己的高层内存内部是连续的：

```text
linkLists_[i]:
  [level 1 link list][level 2 link list][level 3 link list]...
```

不同节点的高层内存不保证物理连续，也不保证按 internal id 地址递增。这与 level 0 的连续大数组不同。

每个节点存在于：

```text
level 0 .. element_levels_[internal_id]
```

全局图层数由当前所有节点的最大随机层数控制：

```text
maxlevel_ = max(element_levels_[i])
```

如果 `maxlevel_ == 5`，索引包含 level 0 到 level 5 共 6 层；但绝大多数节点仍只在 level 0。高层节点数量随层数指数下降。

### 8.3 单次节点扩展的访存序列

第 0 层扩展一个节点时，典型访存序列是：

```text
读取 current_node 的邻接表
读取邻居 candidate_id
检查 visited_array[candidate_id]
读取 candidate_id 的 vector data
执行 distance(query, vector)
按距离更新 candidate_set/top_candidates
```

这些访问由 query 路径决定，邻居 id 通常不连续。当前 hnswlib 代码只在 `USE_SSE` 下插入少量 `_mm_prefetch`，没有按 CCD、NUMA node、cache set 或 first-touch 做布局控制。

## 9. 与 CPU attention 文档中 workitem/tile 模型的差异

| 维度 | vLLM CPU attention | hnswlib HNSW |
| --- | --- | --- |
| batch 内任务 | `kv_head_idx + thread_offset + workitem[]` | 输入矩阵的单个 `row` |
| 单请求内部并行 | 有，按 head/workitem/tile/micro-iter 切分 | 无，单 query 串行图遍历 |
| 运行时任务领取 | attention/reduction 全局 counter | `ParallelFor` 原子 row counter |
| 线程间 reduction | split-KV 时有 reduction task | 查询无 reduction |
| 主要共享数据 | KV cache、metadata、输出 buffer | HNSW 图、向量、邻接表 |
| 查询线程通信 | attention task/reduction 通过输出中间结果关联 | 不交换候选集；仅共享 visited pool 和调度 counter |
| 构建/更新 | 不在 attention kernel 范围内 | 是 `add_items` 主路径，会大量加锁改图 |

HNSW 的关键区别是：`num_threads` 提升的是 batch 级吞吐，而不是单 query latency 的内部并行度。单 query latency 主要由访问路径长度、`ef_`、距离函数成本、图布局和 cache/DRAM 行为决定。

## 10. 对 chiplet 分析的直接含义

- 当前 hnswlib baseline 不做 CCD 感知调度。`ParallelFor` 只按原子计数器动态分发 row，不考虑 query 会访问哪些图区域。
- 查询线程之间没有候选集通信，因此 per-query path tracing 比线程通信分析更关键。
- batch 查询中可能存在热节点复用，但代码没有把路径相近的 query 放到同一线程或同一 CCD。
- 第 0 层节点扩展是主要随机访存路径，访问内容包括邻接表、visited bitmap 和向量数据。
- 上层节点数量较少、访问频率高，适合作为 per-CCD navigator replica 的候选对象；该结论来自代码访问路径，不等价于已经证明有性能收益。
- 构建路径有大量图写锁，不适合作为第一阶段 chiplet 查询 locality baseline。只读 `knn_query` 更适合先用于 PMU 和路径 tracing。
- 若后续实现 chiplet-aware HNSW，最小侵入点可能在 binding 层 batch 调度和索引内存布局；要改变单 query 内部远端扩展，则需要改写 `searchBaseLayerST(...)` 的候选扩展流程。
