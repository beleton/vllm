# CPU attention 并行计算过程：线程、task、workitem 与 tile

## 范围
- 只覆盖 legacy CPU attention 路径：
  - `csrc/cpu/cpu_attn.cpp`
  - `csrc/cpu/cpu_attn_impl.hpp`
- 不讨论：
  - `csrc/cpu/cpu_attn_acc_locality.cpp`
  - `acc-local-l3` subgroup 调度
  - NUMA / TP 放置策略

## 核心结论
- `schedule()` 负责生成 `AttentionMetadata`、`AttentionWorkItemGroup[]`、`ReductionWorkItemGroup[]`；`AttentionMainLoop::operator()` 负责真正执行 attention。
- runtime attention task 总数是 `actual_kv_head_num * effective_thread_num`，不是 `workitem` 数。
- 一个 runtime attention task 由三部分确定：
  - 一个 `kv_head_idx`
  - 一个逻辑线程桶 `thread_offset`
  - 该桶里的一整段 `AttentionWorkItemGroup[]`
- `AttentionWorkItemGroup` 只描述 request-local 的 `q token` 片段和一段逻辑 `KV` 区间，不带 head 维信息。
- `q_head` 在 runtime task 映射阶段由 `kv_head_idx` 固定；同一个 `workitem` 会在不同 `kv_head_idx` 下各执行一遍。
- 切分层次是：
  - `task -> workitem -> q_tile -> kv_tile -> q micro-iter`
- `q micro-iter` 已是本文里的最后一层切分；每次迭代的循环体对应一次 `execute_attention(...)` 调用。
- `max_num_q_per_iter` 是当前 `attn_impl` kernel 的并行 `q head` 容量上限；scheduler 在 `get_scheduler_metadata(...)` 中从 `attn_impl::MaxQHeadNumPerIteration` 读取它，运行时主循环再以 `max_q_head_num_per_iter` 使用它；`max_q_token_num_per_iter = max_q_head_num_per_iter / actual_q_heads_per_kv` 控制最内层 `q token` 步长。

## 1. 调用链

```text
CPUAttentionMetadataBuilder.build(...)
  -> ops.cpu_attn_get_scheduler_metadata(...)
       -> get_scheduler_metadata(...)
            -> AttentionScheduler::schedule(...)
                 -> 生成 AttentionMetadata
                 -> 生成 AttentionWorkItemGroup[]
                 -> 生成 ReductionWorkItemGroup[]

CPUAttentionBackendImpl.forward(...)
  -> ops.cpu_attention_with_kv_cache(...)
       -> cpu_attention_with_kv_cache(...)
            -> AttentionMainLoop<attn_impl>::operator()(...)
                 -> OpenMP 线程启动
                 -> 动态抢 runtime task
                 -> 执行 attention 或 reduction
```

- `schedule()` 只做调度拆分，不做 kernel 主计算。
- `operator()` 只消费 metadata，不再重新决定 `workitem` 划分。

## 2. 概念表

| 名称 | 层次 | 含义 |
| --- | --- | --- |
| `thread_num` | OpenMP 真实线程层 | 本次 kernel 实际启动的 OMP 线程数 |
| `effective_thread_num` | metadata 层 | 分到非空 `workitem` 桶的逻辑线程数 |
| `task_idx` | runtime task 层 | 线程从全局计数器抢到的任务编号 |
| `AttentionWorkItemGroup` | workitem 层 | request-local 的 q 片段和一段 KV 区间 |
| `default_q_tile_token_num` | q tile 层 | 一个 `workitem` 在运行时继续切成多大的 q 子块 |
| `kv_tile` | KV tile 层 | 当前 `q_tile` 对应的可见 KV 区间再切出的较小块 |
| `max_q_token_num_per_iter` | q micro-iter 层 | 当前 `kv_tile` 内最内层一次处理多少个 q token |
| `max_num_q_per_iter` | 寄存器容量层 | 当前 `attn_impl` kernel 一次最内层 attention 调用最多并行承载的 `q heads` 数；scheduler 在 `get_scheduler_metadata(...)` 中从 `attn_impl::MaxQHeadNumPerIteration` 读取 |

## 3. scheduler 输出

### 3.1 `AttentionWorkItemGroup`

`AttentionWorkItemGroup` 字段只有：
- `req_id`
- `q_token_id_start`
- `q_token_num`
- `kv_split_pos_start`
- `kv_split_pos_end`
- `total_kv_len`
- `split_id`
- `local_split_id`

一个 `workitem` 的语义是：

```text
某个 request
+ 一段连续 q token
+ 一段逻辑 KV token 区间 [kv_split_pos_start, kv_split_pos_end)
+ 若 split-KV，则再带 split_id / local_split_id
```

它没有 head 字段，不是按 head 切出来的对象。

### 3.2 KV 工作量分组

`schedule()` 先按最小 q 片段扫描 request：

```cpp
for (int32_t token_id = 0; token_id < q_token_num;
     token_id += max_num_q_token_per_iter)
```

对每个最小 q 片段，先求当前可见 `KV` 区间，再按 `kv_block_alignment` 对齐：

```cpp
curr_kv_len = aligned_kv_tile_pos_right - aligned_kv_tile_pos_left;
```

再把相邻 q 片段并入当前 `curr_workitem`：

```cpp
curr_workitem.q_token_num += q_tile_token_num;
curr_workitem.total_kv_len += curr_kv_len;
```

写回 `workitem` 的主要触发条件：
- 当前逻辑桶剩余预算过短，且当前桶已有工作
- 当前累计 q 片段走到 `default_tile_token_num` 边界，且当前桶已有工作
- 进入 split-KV 路径
- 单个 request 扫描结束

结果：
- `workitem_group.q_token_num` 通常不均衡
- 被尽量拉齐的是 `total_kv_len`
- `causal prefill` 下，后部 q token 可见前缀更长，单个 q 片段对应的 `curr_kv_len` 更大，后部 `workitem` 往往包含更少的 q token
- `default_tile_token_num` 是 scheduler 在换桶时参考的 q 片段边界，不是 `workitem.q_token_num` 的硬上限
- 多线程长 prefill 下，长 request 通常会被拆成多个 `workitem`
- 源码没有禁止单个 `workitem` 覆盖很长的 q 片段；`effective_thread_num = 1`，或剩余工作集中到最后一个逻辑桶时，单个 `workitem` 可以覆盖整个 request

### 3.3 逻辑线程桶与 `effective_thread_num`

`schedule()` 里先构造：

```cpp
std::vector<int32_t> workitem_num_per_thread(thread_num, 0);
```

这里的 `thread` 指逻辑线程桶，不是运行时真实 OMP 线程。每次写回一个 `workitem`，就给当前桶计数：

```cpp
++workitem_num_per_thread[curr_thread_id];
```

之后从前往后扫描第一个空桶，得到 `effective_thread_num`。再把 `workitem_num_per_thread[]` 转成前缀和 `cu_workitem_num_per_thread[]`，供运行时按 `thread_offset` 取出该桶对应的一整段 `workitems`。

要点：
- `thread_num`：真实 OMP 线程数
- `workitem_num_per_thread[i]`：第 `i` 个逻辑桶里有多少个 `workitem`
- `effective_thread_num`：前面连续非空的逻辑桶数

request 和逻辑线程桶不是一一对应关系：
- 一个 request 可以跨多个桶
- 一个桶也可以同时接住上一个 request 的尾段和下一个 request 的开头

## 4. runtime task 映射

### 4.1 任务编号空间

运行时线程从统一的全局计数器抢任务：

```cpp
int64_t task_idx = metadata.acquire_counter();
```

其中：

```cpp
const int32_t actual_kv_head_num = use_gqa ? kv_head_num : q_head_num;
```

任务空间分成两段：

```cpp
const int32_t workitem_groups_counter_num =
    actual_kv_head_num * effective_thread_num;
const int32_t reduction_items_counter_num =
    actual_kv_head_num * reduction_item_num;
const int32_t total_counter_num =
    workitem_groups_counter_num + reduction_items_counter_num;
```

- `0 <= task_idx < workitem_groups_counter_num`：attention task
- `workitem_groups_counter_num <= task_idx < total_counter_num`：reduction task

### 4.2 attention task 单元

attention task 的映射是：

```cpp
const int32_t kv_head_idx = task_idx / effective_thread_num;
const int32_t thread_offset = task_idx % effective_thread_num;
```

随后用 `thread_offset` 取出该逻辑桶对应的一整段 `workitems`：

```cpp
AttentionWorkItemGroup* const curr_workitem_groups =
    workitem_groups + cu_workitem_num_per_thread[thread_offset];
const int32_t curr_workitem_groups_num =
    cu_workitem_num_per_thread[thread_offset + 1] -
    cu_workitem_num_per_thread[thread_offset];
```

一个 attention task 的定义是：

```text
一个 kv_head_idx
+ 一个 thread_offset
+ 该 thread_offset 桶里的一整段 AttentionWorkItemGroup[]
```

attention task 总数：

```text
actual_kv_head_num * effective_thread_num
```

## 5. attention task 内部执行链

单个 attention task 的计算顺序可概括为：

```text
for workitem
  for q_tile
    copy 当前 q_tile -> q_buffer
    for kv_tile
      for q micro-iter
        从 q_buffer 取当前 micro-iter 的子块
        execute_attention(...)
```

`execute_attention(...)` 位于最内层 `q micro-iter`。它处理的是“当前 q micro-iter × 当前 kv_tile”，主要包含三步：
- 计算 `Q @ K^T`，写入 `logits_buffer`
- 对 logits 做 softcap、mask、softmax，更新 `max_buffer/sum_buffer`
- 计算 `P @ V`，把结果累加到 `partial_q_buffer`

### 5.1 固定的 `kv_head` 与 `q_head`

一个 attention task 先固定：
- 一个 `kv_head_idx`
- 一个逻辑线程桶 `thread_offset`
- 当前 task 对应的一组 `q heads`

运行时先计算：

```cpp
const int32_t q_head_num = input->num_heads;
const int32_t kv_head_num = input->num_kv_heads;
const int32_t q_heads_per_kv = q_head_num / kv_head_num;
const bool use_gqa = (max_q_head_num_per_iter % q_heads_per_kv == 0);
const int32_t actual_q_heads_per_kv = use_gqa ? q_heads_per_kv : 1;
const int32_t q_head_start_idx = kv_head_idx * actual_q_heads_per_kv;
```

本次 task 处理的全局 `q head` 区间：

```text
[q_head_start_idx, q_head_start_idx + actual_q_heads_per_kv)
```

`workitem` 本身不带 head 维信息，只决定：
- 哪个 request
- 哪段 q token
- 哪段 KV 区间

同一个 `workitem` 会在不同 `kv_head_idx` 下被重复消费；变化的是当前 task 固定的 q head 段，不是 `workitem` 本身。

### 5.2 `workitem -> q_tile`

一个 task 会顺序处理自己桶里的多个 `workitems`。对每个 `workitem`，先取：

```cpp
const int32_t q_token_num = current_workitem_group->q_token_num;
```

再按 q token 维切成多个 `q_tile`：

```cpp
for (int32_t q_token_offset = 0; q_token_offset < q_token_num;
     q_token_offset += default_q_tile_token_num) {
  const int32_t actual_q_token_num =
      std::min(default_q_tile_token_num, q_token_num - q_token_offset);
```

关系是：

```text
一个 workitem 的 q_token_num
  -> 被切成一个或多个 q_tile
  -> 每个 q_tile 的 token 数是 actual_q_token_num
```

`q_tile` 不是“某个单独 q_head 下面的 tile”，而是“当前 task 固定的整组 q heads”在一段 q token 上的子块。当前 `q_tile` 的逻辑 shape 是：

```text
[actual_q_token_num, actual_q_heads_per_kv, head_dim]
```

不是逐个 head 完整算完再切下一个 head。只有 `actual_q_heads_per_kv = 1` 时，才退化成单 head 情况。

`default_q_tile_token_num` 约束的是这里的运行时 `q_tile` 步长，不是 scheduler 阶段 `workitem.q_token_num` 的上限。一个很长的 `workitem` 进入 runtime 后，仍会继续被切成多个 `q_tile`。

它也不是最内层一次 `execute_attention(...)` 能处理的 q token 数。`default_q_tile_token_num` 是外层 `q_tile` 的默认步长；最内层单次容量由 `max_q_token_num_per_iter` 决定。

### 5.3 `q_tile -> q_buffer`

线程先把 query 和 output 的起点偏到当前这组 q heads：

```cpp
query_t* const q_tile_ptr =
    reinterpret_cast<query_t*>(input->query) +
    q_token_start_idx * q_token_num_stride +
    q_head_start_idx * q_head_num_stride;

size_t output_buffer_offset =
    q_token_start_idx * q_head_num * head_dim +
    q_head_start_idx * head_dim;
```

然后在 `q_tile` 循环内部做两件事：
- `buffer_manager.update(...)`
- `copy_q_heads_tile(...)`

即：

```text
for q_tile
  copy 当前 q_tile -> q_buffer
```

不是在 `q_tile` 循环外一次把整个 `workitem` 的 q token 全部拷进 `q_buffer`。如果一个 `workitem` 被切成多个 `q_tile`，就会发生多次拷贝；每次只拷当前这一个 `q_tile`，并覆盖前一次 `q_tile` 留在 `q_buffer` 里的内容。

`copy_q_heads_tile(...)` 拷贝后的逻辑 shape 是：

```text
[actual_q_token_num, actual_q_heads_per_kv, head_dim]
```

### 5.4 线程私有 scratchpad 与 attention buffers

进入并行区前，`schedule()` 先计算：
- `attention_scratchpad_size_per_thread`
- `reduction_scratchpad_size_per_kv_head`

再按本次 metadata 申请或扩容全局 scratchpad：

```cpp
scratchpad_size =
    attention_scratchpad_size_per_thread * thread_num +
    reduction_scratchpad_size_per_kv_head * actual_kv_head_num;
```

进入并行区后，每个线程只创建一次：

```cpp
AttentionScratchPad buffer_manager(thread_id, metadata, scratchpad_ptr);
```

其中：
- `thread_scratchpad_ptr = scratchpad_ptr + thread_id * attention_scratchpad_size_per_thread`
- `reduction_scratchpad_ptr` 指向所有线程私有切片之后的公共 reduction 区

attention buffers 是线程私有的，布局是：

```text
thread_scratchpad_ptr
  -> [q_buffer][logits_buffer][partial_output_buffer][max_buffer][sum_buffer]
```

各子区含义：
- `q_buffer`：当前 `q_tile` 的 Q staging 区
- `logits_buffer`：当前 `Q @ K^T` 的 logits
- `partial_output_buffer`：当前 `P @ V` 的部分输出
- `max_buffer / sum_buffer`：softmax 过程中的按 q-head 统计量

`q_buffer` 是线程私有 attention 切片里的第一个子区，起始偏移恒为 `0`。

### 5.5 attention buffer 大小

当前 `q_tile` 先展平为：

```text
q_head_tile_size = actual_q_token_num * actual_q_heads_per_kv
```

再按最内层 head 容量对齐为：

```text
rounded_q_head_tile_size =
  ceil_div(q_head_tile_size, max_q_head_num_per_iter) *
  max_q_head_num_per_iter
```

`AttentionScratchPad::update(...)` 中 attention buffers 的大小公式为：

```text
q_buffer_size =
  round_to_64(q_head_tile_size * head_dim * q_buffer_elem_size)

logits_buffer_size =
  round_to_64(max_num_q_per_iter * kv_tile_size * logits_buffer_elem_size)

partial_output_buffer_size =
  round_to_64(q_head_tile_size * head_dim * output_buffer_elem_size)

max_buffer_size =
  round_to_64(q_head_tile_size * sizeof(float))

sum_buffer_size =
  round_to_64(q_head_tile_size * sizeof(float))
```

传给 `buffer_manager.update(...)` 的容量参数是 `rounded_q_head_tile_size`；实际写入 `q_buffer` 的逻辑 shape 仍是
`[actual_q_token_num, actual_q_heads_per_kv, head_dim]`。

### 5.6 `q_tile` 与 `kv_tile` 大小

调度器先按可用 `L2` 容量估一个默认 tile 大小：

```text
default_tile_size =
  floor_to_multiple(
    cache_size /
    (head_dim * (q_buffer_elem_size + 2 * elem_size +
                 output_buffer_elem_size) +
     max_num_q_per_iter * logits_buffer_elem_size),
    max_num_q_per_iter)
```

这里的 `default_tile_size` 单位是 `q head` 数。运行时把它换成外层 `q_tile` 的 token 步长：

```text
default_q_tile_token_num =
  default_tile_size / actual_q_heads_per_kv
```

`default_q_tile_token_num` 在进入具体 `workitem -> q_tile -> kv_tile` 计算前就能确定。它由当前 ISA 的 `max_q_head_num_per_iter`、`head_dim`、数据类型大小、当前 task 的 `actual_q_heads_per_kv` 与可用 `L2` 容量共同决定，表示当前 task 的外层 `q_tile` 默认步长。

当前外层 `q_tile` 的 token 数为：

```text
actual_q_token_num =
  min(default_q_tile_token_num, 当前 workitem 剩余 q_token_num)
```

所以运行时顺序是：
- 先由 `L2` 预算得到 `default_q_tile_token_num`
- 再结合当前 `workitem` 剩余 q token 得到本轮 `actual_q_token_num`
- 再根据当前 `q_tile` 的大小反推 `kv_tile_size`

已知当前 `q_tile` 后，再反推本轮 `kv_tile_size`：

```text
q_head_tile_size = actual_q_token_num * actual_q_heads_per_kv
rounded_q_head_tile_size =
  ceil_div(q_head_tile_size, max_q_head_num_per_iter) *
  max_q_head_num_per_iter
```

当前 `q_tile` 是否只需要一次 `q micro-iter`，源码写法是：

```text
rounded_q_head_tile_size <= max_q_head_num_per_iter
```

它与下面这个 token 判据等价：

```text
actual_q_token_num <= max_q_token_num_per_iter
```

源码保留 `q head` 口径，是因为后续 `kv_tile_size` 计算和 scratchpad 容量分配都使用 `rounded_q_head_tile_size`。

若当前 `q_tile` 只需要一次 `q micro-iter`，`kv_tile_size` 按：

```text
(cache_size -
 q_tile_size * head_dim *
 (q_buffer_elem_size + output_buffer_elem_size)) /
(logits_buffer_elem_size * max_num_q_per_iter)
```

否则按：

```text
(cache_size -
 q_tile_size * head_dim *
 (q_buffer_elem_size + output_buffer_elem_size)) /
(logits_buffer_elem_size * max_num_q_per_iter +
 2 * head_dim * elem_size)
```

两者差别在于 K/V 是否需要被同一个 q_tile 反复复用：
- 单次 q micro-iter 时，同一个 kv_tile 的 K/V 只会被扫一遍，不需要把 K/V 的驻留成本算进 cache 预算，所以公式里没有 2 *
  head_dim * elem_size
- 多次 q micro-iter 时，同一个 kv_tile 会被反复用于多个 q 子批次，公式就把 K/V 的 cache 成本算进去

最后再向下对齐到 `blocksize_alignment` 的倍数。

### 5.7 `q_tile -> kv_tile`

每个 `q_tile` 会先求自己的可见 KV 区间，再按 `kv_tile_size` 切成多个 `kv_tile`：

```cpp
for (int32_t kv_tile_pos = rounded_kv_tile_start_pos;
     kv_tile_pos < rounded_kv_tile_end_pos;
     kv_tile_pos += kv_tile_size) {
```

`kv_tile` 没有独立的 K/V scratchpad buffer。代码只固定：
- `curr_k_cache`
- `curr_v_cache`
- 当前 `kv_tile` 的位置区间

然后 `execute_attention(...)` 直接从 KV cache 中读取这段 `kv_tile`。也就是说：
- `q_tile` 有显式 `q_buffer`
- `kv_tile` 没有对应的 `k_buffer/v_buffer`
- `logits_buffer` 的容量按 `kv_tile_size` 预留，用来放当前 `Q @ K^T`

### 5.8 `kv_tile -> q micro-iter`

在当前 `kv_tile` 内，代码再按 `max_q_token_num_per_iter` 切成最内层 `q micro-iter`：

```cpp
for (int32_t q_head_tile_token_offset = 0;
     q_head_tile_token_offset < actual_q_token_num;
     q_head_tile_token_offset += max_q_token_num_per_iter) {
```

每个 `q micro-iter` 会从当前 `q_buffer/partial_output_buffer/max_buffer/sum_buffer` 中取一个子视图：

```cpp
q_buffer + q_tile_head_offset * head_dim
partial_q_buffer + q_tile_head_offset * head_dim
max_buffer + q_tile_head_offset
sum_buffer + q_tile_head_offset
```

这里：

```text
q_tile_head_num = q_tile_token_num * actual_q_heads_per_kv
```

所以一次 `execute_attention(...)` 处理的是“若干 q token × 当前 task 固定的整组 q heads”，不是逐个 head 独立调用。

当前常见 `VEC + GQA=8` 场景下，最内层通常是：

```text
1 token × 8 q heads
```

### 5.9 reduction buffers

reduction 区按 `kv_head_idx` 切片，子区布局是：

```text
[reduce_flag_buffer][reduce_output_buffer][reduce_max_buffer][reduce_sum_buffer]
```

大小公式为：

```text
reduce_flag_buffer_size =
  round_to_64(total_split_num * sizeof(bool))

reduce_output_buffer_size =
  round_to_64(total_split_num * q_head_tile_size * head_dim *
              output_buffer_elem_size)

reduce_max_buffer_size =
  round_to_64(total_split_num * q_head_tile_size * sizeof(float))

reduce_sum_buffer_size =
  round_to_64(total_split_num * q_head_tile_size * sizeof(float))
```

只在 split-KV / reduction 路径里使用，不参与普通 attention task 的主计算链。

## 6. KV 可见区间裁剪

运行时先在 `q_tile` 层裁一次可见 KV 区间：

```cpp
const auto [kv_tile_start_pos, kv_tile_end_pos] =
    AttentionScheduler::calcu_kv_tile_pos(...);
```

再对齐到 `blocksize_alignment`：

```cpp
const auto [rounded_kv_tile_start_pos, rounded_kv_tile_end_pos] =
    AttentionScheduler::align_kv_tile_pos(...);
```

进入最内层 `q micro-iter` 后，再根据当前 q 子块裁一次：

```cpp
const auto [actual_kv_tile_pos_left, actual_kv_tile_pos_right] =
    AttentionScheduler::calcu_kv_tile_pos(...);
```

结论：
- 大段不可见 KV 不会进入当前计算
- 真正进入 kernel 的是“对齐后的可见区间”
- 只有对齐额外带进来的少量位置会在 `apply_mask(...)` 中被置成无效

`schedule()` 中的 `total_kv_len` 和 `workitem.total_kv_len` 也按同一口径累计，`workitem` 划分依据是“对齐后的可见 KV 工作量”。

## 7. split-KV 额外对象

当 `enable_kv_split` 打开且尾段 q token 满足条件时，scheduler 会把同一段 q token 对应的 KV 区间切成多个片段：
- 为每个 KV 片段生成一个 `AttentionWorkItemGroup`
- 额外生成一个对应的 `ReductionWorkItemGroup`

attention 阶段先产出多个 partial outputs，reduction 阶段再归并。

## 8. 示例：Qwen3-30B-A3B，单 request，`q_token_num=1024`

假设：
- 只有 1 个 request
- `query = [1024, 32, 128]`
- `kv = [1024, 4, 128]`
- `dtype = bf16`
- 当前 ISA 走 `VEC`
- 当前机器每核 `L2 = 1 MiB`，代码只用其中一半，`available_cache_size = 512 KiB`

则：
- `MaxQHeadNumPerIteration = 8`
- `q_heads_per_kv = 32 / 4 = 8`
- `actual_q_heads_per_kv = 8`
- `max_q_token_num_per_iter = 8 / 8 = 1`
- `q_buffer_elem_size = 4`
- `elem_size = 2`
- `logits_buffer_elem_size = 4`
- `output_buffer_elem_size = 4`

先算外层默认 `q_tile` 步长：

```text
default_tile_size =
  floor_to_multiple(
    524288 /
    (128 * (4 + 2 * 2 + 4) + 8 * 4),
    8)
  = floor_to_multiple(334, 8)
  = 328

default_q_tile_token_num = 328 / 8 = 41
```

这里的 `default_q_tile_token_num = 41` 在进入具体 `q_tile` 循环前就能确定；它是外层 `q_tile` 默认步长，不是最内层单次 `execute_attention(...)` 的 q token 数。最内层仍由 `max_q_token_num_per_iter = 1` 决定。

一个完整外层 `q_tile` 的大小是：

```text
actual_q_token_num = 41
q_head_tile_size = 41 * 8 = 328
rounded_q_head_tile_size = 328
```

由于 `328 > 8`，当前 `q_tile` 需要多次 `q micro-iter`，所以 `kv_tile_size` 走多轮公式：

```text
kv_tile_size =
  floor_to_multiple(
    (524288 - 328 * 128 * (4 + 4)) /
    (4 * 8 + 2 * 128 * 2),
    32)
  = floor_to_multiple(346, 32)
  = 320
```

若最终 `effective_thread_num = 8`，则：
- attention task 总数 = `4 * 8 = 32`
- `kv_head_idx = 0` 的 task 固定全局 `q heads [0, 8)`
- `kv_head_idx = 1` 的 task 固定全局 `q heads [8, 16)`
- `kv_head_idx = 2` 的 task 固定全局 `q heads [16, 24)`
- `kv_head_idx = 3` 的 task 固定全局 `q heads [24, 32)`

`q_token_num = 1024` 时，若某个 `workitem` 恰好覆盖整段 q token，runtime 内部仍会继续切成：

```text
24 个 q_tile，每个 41 token
+ 1 个尾部 q_tile，40 token
```

对应尾部 `q_tile`：

```text
q_head_tile_size = 40 * 8 = 320
rounded_q_head_tile_size = 320
kv_tile_size =
  floor_to_multiple(
    (524288 - 320 * 128 * (4 + 4)) /
    (4 * 8 + 2 * 128 * 2),
    32)
  = floor_to_multiple(361, 32)
  = 352
```

最内层 `q micro-iter` 仍是：

```text
1 token × 8 q heads
```

在 `causal prefill` 下，当前 `q_tile` 的可见 KV 上界等于自己的结束位置，再向上对齐到 `blocksize_alignment = 32`。因此不同 `q_tile` 的实际 `kv_tile` 个数还要继续看自己的可见 KV 范围。

第 1 个 `q_tile`：

```text
q 位置 = [0, 41)
可见 KV = [0, 41)
对齐后 = [0, 64)
kv_tile_size = 320
=> 只有 1 个 kv_tile: [0, 64)
```

第 9 个 `q_tile`：

```text
q 位置 = [328, 369)
可见 KV = [0, 369)
对齐后 = [0, 384)
kv_tile_size = 320
=> 2 个 kv_tile: [0, 320), [320, 384)
```

最后 1 个尾部 `q_tile`：

```text
q 位置 = [984, 1024)
可见 KV = [0, 1024)
对齐后 = [0, 1024)
kv_tile_size = 352
=> 3 个 kv_tile: [0, 352), [352, 704), [704, 1024)
```

`1024` token request 不等于单个 `workitem`。源码里的实际关系是：
- scheduler 先按 `1 token` 的最小 q 片段扫描
- 再按累计 `total_kv_len`、逻辑桶剩余预算、`default_tile_token_num` 边界和 split-KV 条件，把这些最小 q 片段聚合成 `workitem`
- 多线程长 prefill 下，`1024` token request 通常会被拆成多个 `workitem`
- `default_q_tile_token_num = 41` 约束的是 runtime 的 `q_tile` 步长，不是 `workitem` 大小
- 若 `effective_thread_num = 1`，或剩余工作最终都落到最后一个逻辑桶，单个 `workitem` 仍可能覆盖整个 `1024` token request

单个 task 内部继续切分后，最内层一次 `execute_attention(...)` 更接近：

```text
Q_micro = [1, 8, 128]
K_tile  = [kv_tile_token_num, 128]
V_tile  = [kv_tile_token_num, 128]
```

它不是一次把整段 `q=[1024,32,128]` 全算完，而是固定一组 q heads 后，在自己的 `workitem -> q_tile -> kv_tile` 范围内反复推进最小 `q micro-iter`。

## 9. 关键变量

- `effective_thread_num`
- `workitem_groups_counter_num`
- `kv_head_idx`
- `thread_offset`
- `q_head_start_idx`
- `curr_workitem_groups_num`
- `default_q_tile_token_num`
- `max_q_token_num_per_iter`
- `q_tile_token_num`
- `actual_kv_tile_pos_left / right`

## 10. 关键源码位置

- `csrc/cpu/cpu_attn.cpp`
- `csrc/cpu/cpu_attn_impl.hpp::AttentionScheduler::schedule`
- `csrc/cpu/cpu_attn_impl.hpp::AttentionMainLoop::operator()`
- `csrc/cpu/cpu_attn_vec.hpp::copy_q_heads_tile`
