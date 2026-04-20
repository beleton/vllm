# CPU attention 并行计算过程：线程、task、workitem 与 tile

## 说明
- 范围只覆盖 legacy CPU attention 路径：
  - `csrc/cpu/cpu_attn.cpp`
  - `csrc/cpu/cpu_attn_impl.hpp`
- 本页不讨论：
  - `csrc/cpu/cpu_attn_acc_locality.cpp`
  - `acc-local-l3` subgroup 路径
  - NUMA / TP 放置优劣

## 问题
- 当前 `vLLM v1 CPU backend` 的 `cpu_attention_with_kv_cache -> AttentionMainLoop::operator()`，到底是怎样把一次 attention 计算拆成线程、runtime task、workitem 和 tile 的。
- `max_num_q_per_iter`、`max_q_token_num_per_iter`、`default_q_tile_token_num` 这几个量分别控制哪一层切分。

## 结论
- 真正参与 kernel 的是 OpenMP 真实线程，数量等于 `thread_num`；但 metadata 里还有一层独立的逻辑线程桶，非空桶数是 `effective_thread_num`。
- runtime attention task 总数不是 `workitem` 数，而是：
  - `actual_kv_head_num * effective_thread_num`
- 一个 runtime attention task 不是“一个 request”也不是“一个 workitem”，更准确地说是：
  - 固定一个 `kv_head_idx`
  - 固定一个 `thread_offset`
  - 再消费这个 `thread_offset` 桶里的一整段 `AttentionWorkItemGroup[]`
- 一个 `AttentionWorkItemGroup` 更接近：
  - 某个 `request`
  - 一段连续 `q token`
  - 一段逻辑 `KV` 区间
  - 若开启 split-KV，再带 `split_id / local_split_id`
- runtime 真正的执行层次是：
  - `task -> workitem -> q_tile -> kv_tile -> q micro-iter`
- `max_num_q_per_iter` 不是“线程每次处理多少个 q token”，而是：
  - 一次最内层 attention 调用最多容纳多少个 `q heads`
- 真正控制最内层 q token 切分步长的是：
  - `max_q_token_num_per_iter = max_q_head_num_per_iter / actual_q_heads_per_kv`
- 因此，对当前常见的 `VEC + GQA(q_heads_per_kv=8)` 场景，确实会出现：
  - `max_q_head_num_per_iter = 8`
  - `max_q_token_num_per_iter = 1`
  - 即一个线程在固定 `kv_tile` 内，会把 `q_tile` 再切成“每次 1 个 token”的 q micro-iter
- 但这不是普遍恒等式；如果 `actual_q_heads_per_kv` 变小，`max_q_token_num_per_iter` 也可能大于 1。

## 1. 调用顺序：谁先生成任务，谁后执行任务

```text
CPUAttentionMetadataBuilder.build(...)
  -> ops.cpu_attn_get_scheduler_metadata(...)
       -> csrc/cpu/cpu_attn.cpp::get_scheduler_metadata(...)
            -> AttentionScheduler::schedule(...)
                 -> 生成并填充 AttentionMetadata
                 -> 生成 AttentionWorkItemGroup[]
                 -> 生成 ReductionWorkItemGroup[]
                 -> 计算 effective_thread_num / prefix sum / scratchpad 大小
                 -> 返回 scheduler_metadata

CPUAttentionBackendImpl.forward(...)
  -> ops.cpu_attn_reshape_and_cache(...)
  -> ops.cpu_attention_with_kv_cache(...)
       -> csrc/cpu/cpu_attn.cpp::cpu_attention_with_kv_cache(...)
            -> AttentionMainLoop<attn_impl>::operator()(...)
                 -> OpenMP 线程启动
                 -> 动态抢 runtime task
                 -> 遍历 workitem
                 -> 切 q_tile / kv_tile / q micro-iter
                 -> execute_attention(...)
                 -> final_output(...) 或 partial_output(...)
                 -> reduce_splits(...)
```

- `schedule()` 负责生成调度 metadata。
- `operator()` 才负责真正执行 attention / reduction。

## 2. 先把 8 个概念分清

| 名称 | 所在层次 | 最短定义 |
| --- | --- | --- |
| `thread_num` | OpenMP 真实线程层 | 这次 kernel 真正拉起多少个 OMP 线程 |
| `effective_thread_num` | 调度元数据层 | 实际分到非空 workitem 桶的逻辑线程数 |
| `task_idx` | runtime task 层 | 线程运行时从全局 atomic counter 抢到的任务编号 |
| `AttentionWorkItemGroup` | workitem 层 | request-local 的 q 片段 + 一段逻辑 KV 区间 |
| `default_q_tile_token_num` | q tile 层 | 一个 workitem 在运行时继续切出的较大 q 子块大小 |
| `kv_tile` | kernel tile 层 | 当前 q_tile 对应的 KV 访问区间再切出的较小块 |
| `max_q_token_num_per_iter` | q micro-iter 层 | 当前 `kv_tile` 内最内层一次处理多少个 q token |
| `max_num_q_per_iter` | 寄存器容量 / 模板常量层 | 一次最内层 attention 调用最多容纳多少个 q heads |

## 3. `max_num_q_per_iter`、`max_q_token_num_per_iter`、`default_q_tile_token_num` 的关系

### 3.1 `max_num_q_per_iter`
- 在 scheduler 输入里，它的注释就是：

```cpp
int32_t max_num_q_per_iter;  // max Q head num can be hold in registers
```

- 这表示当前 ISA 实现一次最内层 attention 调用，最多能容纳多少个 q heads。
- 它来自具体 `attn_impl` 的静态常量：
  - `attn_impl::MaxQHeadNumPerIteration`
- 对当前机器如果走 `VEC`，这个值是 `8`。

### 3.2 `max_q_token_num_per_iter`
- 运行时在 `AttentionMainLoop::operator()` 里计算：

```cpp
const int32_t max_q_token_num_per_iter =
    max_q_head_num_per_iter / actual_q_heads_per_kv;
```

- 它决定最内层 q micro-iter 每次跨多少个 q token。
- 最内层循环就是：

```cpp
for (int32_t q_head_tile_token_offset = 0;
     q_head_tile_token_offset < actual_q_token_num;
     q_head_tile_token_offset += max_q_token_num_per_iter) {
  ...
}
```

- 所以你问的“是不是线程在计算时把 q 再按 `max_q_token_num_per_iter` 切成 micro-iter”，答案是：
  - 是
  - 但要更精确地说，是线程在固定 `workitem -> q_tile -> kv_tile` 后，再按 `max_q_token_num_per_iter` 切成最内层 q micro-iter
- `max_num_q_per_iter` 本身不是 token 维切分步长，它是 q-head 容量；token 维步长是从它推导出来的 `max_q_token_num_per_iter`。

### 3.3 `default_q_tile_token_num`
- 这是更外层的 q tile 大小：

```cpp
const int32_t default_q_tile_token_num =
    default_tile_size / actual_q_heads_per_kv;
```

- 所以层次是：
  - `workitem` 先按 `default_q_tile_token_num` 切成若干 `q_tile`
  - 每个 `q_tile` 再按 `max_q_token_num_per_iter` 切成最内层 q micro-iter

## 4. 有多少线程真正参与计算

### 4.1 真实线程数：`thread_num`

- `AttentionMainLoop::operator()` 一开始直接取 `omp_get_max_threads()`，然后进入：

```cpp
#pragma omp parallel for schedule(static, 1)
for (int thread_id = 0; thread_id < thread_num; ++thread_id) { ... }
```

- 所以真正被拉起的执行者就是这 `thread_num` 个 OMP 线程。

### 4.2 为什么还要有 `effective_thread_num`

- `schedule()` 生成 workitems 时，并不是“有多少线程就给多少 workitem”，而是先按总 KV 工作量构造逻辑线程桶。
- `workitem_num_per_thread` 的含义是：
  - scheduler 阶段第 `i` 个逻辑线程桶最终被分到了多少个 `AttentionWorkItemGroup`
- 它是在 `schedule()` 里先初始化成长度为 `thread_num` 的全 0 数组：

```cpp
std::vector<int32_t> workitem_num_per_thread(thread_num, 0);
```

- 后面每当 scheduler 决定“把当前 `curr_workitem` 正式写回到 `workitems[]`，并归到当前 `curr_thread_id` 这个逻辑桶”时，就会同步执行：

```cpp
++workitem_num_per_thread[curr_thread_id];
```

- 所以它不是运行时统计出来的值，而是 scheduler 在构造 `workitems[]` 过程中，一边分桶一边累计得到的计数器。
- 它的来源可以概括成：
  - scheduler 先维护当前逻辑桶 `curr_thread_id`
  - 再根据 `remaining_kv_len`、split-KV 条件和 q tile 边界决定是否把 `curr_workitem` 写回
  - 每次写回一个 workitem，就把这个桶的计数加 1
- 因此 `workitem_num_per_thread[i]` 表示的不是“真实 OMP 线程 i 跑了多少 workitem”，而是“逻辑线程桶 i 在 metadata 里拥有多少个 workitem”。
- 最后通过 `workitem_num_per_thread[]` 扫出真正非空的桶数：

```cpp
int32_t effective_thread_num = 0;
for (; effective_thread_num < thread_num; ++effective_thread_num) {
  if (workitem_num_per_thread[effective_thread_num] == 0) {
    break;
  }
}
```

- 之所以可以这样扫描，是因为 scheduler 只会从 `curr_thread_id = 0` 开始顺序往后分桶，不会跳着给后面的桶塞 workitem。
- 所以一旦遇到第一个 `workitem_num_per_thread[i] == 0`，后面那些桶也都还是空桶。
- 扫描出 `effective_thread_num` 后，代码又把 `workitem_num_per_thread[]` 转成前缀和 `cu_workitem_num_per_thread[]`，供运行时根据 `thread_offset` 取出该桶对应的一整段 workitems：

```cpp
std::memcpy(metadata_ptr->cu_workitem_num_per_thread + 1,
            workitem_num_per_thread.data(),
            workitem_num_per_thread.size() * sizeof(int32_t));
for (int32_t i = 1; i <= thread_num; ++i) {
  metadata_ptr->cu_workitem_num_per_thread[i] +=
      metadata_ptr->cu_workitem_num_per_thread[i - 1];
}
```

- 所以：
  - `thread_num` 是真实 OMP 线程数
  - `workitem_num_per_thread` 是 scheduler 为每个逻辑线程桶累计出的 workitem 数
  - `effective_thread_num` 是 metadata 中真正分到非空 workitem 的逻辑桶数
- runtime summary 里虽然打印成：

```text
thread N: workitem_group_range=[begin,end)
```

  但这里的 `thread N` 实际是 `cu_workitem_num_per_thread[]` 的下标，也就是“逻辑线程桶 N”的编号；不应理解成固定的真实 OMP 线程 `N`。

### 4.3 哪些线程“真正参与了计算”

- 所有 `thread_num` 个线程都会被拉起，并完成初始化、scratchpad 绑定、可能的 reduction flag 清零。
- 进入主循环后，线程通过 `metadata.acquire_counter()` 动态抢 task；如果 task 数少于线程数，后面的线程可能很快退出。
- 因此，当 `effective_thread_num < thread_num` 时，不能把它理解成“只有前 `effective_thread_num` 个真实线程会参与计算”。
- 更准确地说：
  - `effective_thread_num` 只限制 metadata 里有多少个非空逻辑桶
  - 所有 `thread_num` 个真实 OMP 线程仍会启动并进入抢任务循环
  - 哪些真实线程真正做到 attention / reduction 主体计算，取决于它们是否在运行时抢到了 `task_idx < total_counter_num` 的任务
- 所以实践上可能出现三种情况：
  - `thread_num` 个线程都启动，但只有一部分线程真正执行主体计算
  - 参与主体计算的真实线程数大于 `effective_thread_num`
  - 某些线程只做了初始化和很少量控制流，就因为没抢到任务而退出
- 因此更准确地说：
  - 参与 kernel 启动的线程数 = `thread_num`
  - metadata 中真正有非空任务桶的数量 = `effective_thread_num`
  - 真正做了多少 attention / reduction 计算，还取决于运行时 task 是否足够多

## 5. attention task 总数为什么和 `effective_thread_num` 有关

- 运行期 attention task 的编号空间定义为：

```cpp
const int32_t workitem_groups_counter_num =
    actual_kv_head_num * effective_thread_num;
```

- 所以 attention task 总数不是 `workitem_group_num`，而是：

```text
attention task 总数 = actual_kv_head_num * effective_thread_num
```

- 更直白地说，运行期不是直接把 `workitems[]` 一个个拿出来当 task；而是：
  1. scheduler 先把 `workitems[]` 分到 `effective_thread_num` 个逻辑线程桶里
  2. 运行时再把“每个逻辑桶”沿 `kv_head` 维再展开一遍
- 所以 attention task 的基本单元不是“一个 workitem”，而是：

```text
一个 kv_head_idx
+ 一个逻辑线程桶(thread_offset)
+ 这个桶里的一整段 workitems
```

- 这表示 task 是先按 `kv_head` 展开，再按逻辑线程桶展开，而不是“一个 workitem 一个 task”。

## 6. `runtime task`、`attention task` 和 `reduction task` 的关系

- 线程在主循环里真正拿到的是统一编号的 `runtime task`：

```cpp
int64_t task_idx = metadata.acquire_counter();
```

- 代码先构造两段编号空间：

```cpp
const int32_t workitem_groups_counter_num =
    actual_kv_head_num * effective_thread_num;
const int32_t reduction_items_counter_num =
    actual_kv_head_num * reduction_item_num;
const int32_t total_counter_num =
    workitem_groups_counter_num + reduction_items_counter_num;
```

- 然后线程通过 `task_idx` 落在哪个范围内，判断自己拿到的是哪一类 runtime task：
  - `0 <= task_idx < workitem_groups_counter_num`
    - 这是 `attention task`
  - `workitem_groups_counter_num <= task_idx < total_counter_num`
    - 这是 `reduction task`

- 所以关系是：

```text
runtime task
= attention task + reduction task
```

- 更准确地说：
  - `runtime task` 是总类，表示线程运行时从全局计数器抢到的任务编号
  - `attention task` 是 runtime task 空间中的前半段，负责真正跑 `workitem -> q_tile -> kv_tile -> execute_attention(...)`
  - `reduction task` 是 runtime task 空间中的后半段，只在 split-KV 场景下出现，负责归并多个 partial outputs

- 因此，线程确实是通过 `task_idx` 的范围来判断：
  - 这次拿到的是 `attention task`
  - 还是 `reduction task`

## 7. 一个 runtime attention task 到底对应什么

- 线程抢到 `task_idx` 后，会先映射成：

```cpp
const int32_t kv_head_idx = task_idx / effective_thread_num;
const int32_t thread_offset = task_idx % effective_thread_num;
```

- 再通过前缀和定位这个逻辑线程桶对应的一整段 `AttentionWorkItemGroup[]`：

```cpp
AttentionWorkItemGroup* const curr_workitem_groups =
    workitem_groups + cu_workitem_num_per_thread[thread_offset];
const int32_t curr_workitem_groups_num =
    cu_workitem_num_per_thread[thread_offset + 1] -
    cu_workitem_num_per_thread[thread_offset];
```

- 所以一个 runtime attention task 可以精确理解成：

```text
固定 kv_head_idx
+ 固定 thread_offset
+ 这个 thread_offset 桶里的一整段 workitems
```

### 7.1 `batch > 1` 时，一个 `thread_offset` 桶可以同时挂多个请求

- `thread_offset` 桶和 `request` 不是一一对应关系；同一个桶里可以连续挂多个 `req_id` 的 `workitem_group`。
- runtime 侧只按 `thread_offset` 对应的前缀和区间取出 `curr_workitem_groups`，随后顺序遍历这段连续的 `AttentionWorkItemGroup[]`。
- scheduler 侧 `curr_thread_id` 和 `remaining_kv_len` 在 request 之间不会重置，所以一个逻辑桶可以接住上一个 request 的尾段，也可以继续接下一个 request 的开头。
- 因此，运行时任务 `(kv_head_idx=k, thread_offset=7)` 的实际含义是：

```text
固定 kv_head_idx = k
+ 依次处理 thread_offset=7 这个桶里的所有 workitem_group
+ 每个 workitem_group 再各自读取自己的 req_id / q / KV 区间
```

- 这说明同一个 attention task 确实可能跨多个请求访问数据；但它在整个任务内访问的仍是同一个 `kv_head_idx`，不会在同一个 task 里切换到别的 `kv_head`。
- 因此更准确地说：
  - `(kv_head_idx=0, thread_offset=7)` 这种 task，会顺序读取“请求 1 的 `kv_head 0`”和“请求 2 的 `kv_head 0`”
  - 不会在同一个 task 里同时读取“请求 1 的 `kv_head 0`”和“请求 2 的 `kv_head 1`”
- 这也是 `balanced` 更容易打散 locality 的一个直接来源：
  - 外层任务先按 `kv_head_idx × thread_offset` 展开
  - 而单个 `thread_offset` 桶内部又可能混有多个请求的 `workitem_group`
  - 因此同一个 `kv_head_idx` 任务会顺序扫过多个请求各自的 `K/V` 区间

## 8. 一个 workitem 到底是什么

`AttentionWorkItemGroup` 里有：

- `req_id`
- `q_token_id_start`
- `q_token_num`
- `kv_split_pos_start`
- `kv_split_pos_end`
- `total_kv_len`
- `split_id`
- `local_split_id`

所以一个 workitem 更接近：

```text
某个 request
+ 一段连续 q token
+ 一段逻辑 KV token 区间 [kv_split_pos_start, kv_split_pos_end)
+ 若 split-KV，则再带 split_id / local_split_id
```

注意两点：

- workitem 没有 head 维字段，所以它不是按 head 切出来的对象
- head 维是在 runtime task 里通过 `kv_head_idx` 固定下来的

## 9. 一个线程执行 1 个 attention task 时，到底在做哪部分计算

线程拿到 attention task 后，执行层次是：

```text
for 每个 workitem
  for q_tile
    计算当前 q_tile 的可见 KV 区间
    for kv_tile
      for q micro-iter
        execute_attention(...)
    写回 final_output(...) 或 partial_output(...)
```

所以线程执行 1 个 task 时，做的不是“一个完整 request”，也不是“一个完整 q_tile”，而是：

- 固定一个 `kv_head_idx`
- 跑该 task 对应桶里的所有 workitems
- 每个 workitem 再逐层切成更小 tile

## 10. 一个完整 `Q @ K^T -> softmax -> P @ V` 怎么被拆成并行任务

### 10.1 第 1 层：scheduler 先按 request 内 q token 扫描

- scheduler 先按：

```cpp
for (int32_t token_id = 0; token_id < q_token_num;
     token_id += max_num_q_token_per_iter)
```

  扫 q token。
- 这里用的是 `max_num_q_token_per_iter`，它和运行期的 `max_q_token_num_per_iter` 是同一含义，只是一个在 scheduler 中命名成 `num`，一个在 mainloop 中命名成 `token_num`。
- 每次扫描得到一个最小 q 片段，再统计这段 q 在当前可见窗口下对应多少 KV 工作量。

### 10.2 第 2 层：把这些最小 q 片段聚成 workitem

- scheduler 用 `curr_workitem.q_token_num += q_tile_token_num` 和 `curr_workitem.total_kv_len += curr_kv_len` 把相邻最小 q 片段并入当前 `curr_workitem`。
- `curr_workitem` 写回 `workitems[]` 的触发点有 4 类：
  - 当前桶剩余预算过短，且当前桶已经承接了工作；这时会先写回当前非空 `curr_workitem`，再切到下一个逻辑桶
  - 当前累计 q 片段正好走到 `default_tile_token_num` 边界，且当前桶已经承接了工作；这时会先写回当前非空 `curr_workitem`，再切桶
  - 进入 split-KV 路径；已有的 `curr_workitem` 会先写回，split 出来的 partial-KV workitem 也会立即写回
  - 单个 request 扫描结束后，如果 `curr_workitem.total_kv_len > 0`，会在 request 尾部写回
- 因此，一个 workitem 可能包含多个最小 q 片段；写回时机由预算、tile 边界、split-KV 和 request 结束共同决定。

### 10.3 第 3 层：把 workitems 塞进逻辑线程桶

- 每个逻辑线程桶受 `remaining_kv_len` 预算约束。
- 这里的 `remaining_kv_len` 不是“这个桶最多能存多少真实字节”，而是：
  - 当前这个逻辑桶还想再承接多少 `KV` 工作量
- 这里的 `KV` 工作量在 scheduler 里最直接对应：

```cpp
curr_kv_len = aligned_kv_tile_pos_right - aligned_kv_tile_pos_left;
```

  也就是“当前这段 q 对应的、按对齐后的可见 `KV token` 数”。

### 10.3.1 `total_kv_len` 是怎么得到的

- scheduler 会先扫一遍所有最小 q 片段，并累计全局 `total_kv_len`：

```cpp
const auto [kv_tile_pos_left, kv_tile_pos_right] = calcu_kv_tile_pos(...);
const auto [aligned_kv_tile_pos_left, aligned_kv_tile_pos_right] =
    align_kv_tile_pos(...);
int32_t curr_kv_len =
    aligned_kv_tile_pos_right - aligned_kv_tile_pos_left;
total_kv_len += curr_kv_len;
```

- `AttentionWorkItemGroup.total_kv_len` 也是同一套口径；某个 q 片段被并入当前 workitem 时，会执行：

```cpp
curr_workitem.total_kv_len += curr_kv_len;
```

- 当前 `causal prefill + window_size=-1` 下，scheduler 输入会设成：
  - `left_sliding_window_size = -1`
  - `right_sliding_window_size = 0`
- 这时每个 q 片段的可见 `KV` 右边界跟着 `q_tile_pos_right` 走；再经过 `kv_block_alignment` 对齐后，越靠后的 q 片段，`curr_kv_len` 越大。

### 10.3.2 为什么 `workitem_group.q_token_num` 不均衡

- scheduler 不是按固定 `q token` 数切 workitem，而是尽量让每个 workitem 和逻辑线程桶承接的 `total_kv_len` 接近预算。
- `causal prefill` 里，前面的 q token 可见前缀短，单个 q 片段的 `curr_kv_len` 小；后面的 q token 可见前缀长，单个 q 片段的 `curr_kv_len` 大。
- 所以前面的 workitem 可以装更多 q token，后面的 workitem 只能装更少 q token。
- `workitem_group.q_token_num` 不均衡是 scheduler 按 `KV` 工作量分组的直接结果。

### 10.3.3 `workitem_group` 按“可见 KV 区间的对齐长度”累计

- scheduler 先算当前 q 片段的精确可见 `KV` 区间：

```cpp
const auto [kv_tile_pos_left, kv_tile_pos_right] = calcu_kv_tile_pos(
    kv_start_pos, kv_end_pos, q_tile_pos_left, q_tile_pos_right,
    left_sliding_window_size, right_sliding_window_size);
```

- 再把这个区间按 `kv_block_alignment` 向下/向上对齐：

```cpp
const auto [aligned_kv_tile_pos_left, aligned_kv_tile_pos_right] =
    align_kv_tile_pos(kv_tile_pos_left, kv_tile_pos_right, kv_len_alignment);
curr_kv_len = aligned_kv_tile_pos_right - aligned_kv_tile_pos_left;
```

- 因此 `workitem_group.total_kv_len` 累加的不是“精确可见 KV 长度”，而是“对齐后的可见 KV 长度”。
- 当前 `causal prefill + window_size=-1` 下：
  - 精确可见区间是 `[0, q_right_pos)`
  - 对齐后区间是 `[0, align_up(q_right_pos, kv_block_alignment))`
  - 所以 `curr_kv_len = align_up(q_right_pos, kv_block_alignment)`
- 例如在 `VEC` 路径下，`BlockSizeAlignment = 32`；若某组输入满足 `q_heads_per_kv = 2`，则最小 q 片段大小是 `4`，单个 q 片段的 `curr_kv_len` 就会按 `align_up(q_right_pos, 32)` 台阶式增长。

### 10.3.4 runtime 实际读取的是“对齐后的可见 KV 区间”

- 运行期先在 `q_tile` 层裁一次可见 `KV` 区间，再做一次对齐：

```cpp
const auto [kv_tile_start_pos, kv_tile_end_pos] = calcu_kv_tile_pos(...);
const auto [rounded_kv_tile_start_pos, rounded_kv_tile_end_pos] =
    align_kv_tile_pos(kv_tile_start_pos, kv_tile_end_pos, blocksize_alignment);
```

- 外层 `kv_tile` 循环只遍历这段对齐后的区间：

```cpp
for (int32_t kv_tile_pos = rounded_kv_tile_start_pos;
     kv_tile_pos < rounded_kv_tile_end_pos;
     kv_tile_pos += kv_tile_size) { ... }
```

- 进入最内层 q micro-iter 后，代码会再按当前 q token 的真实可见范围裁一次，并再次对齐；最终传给 `execute_attention(...)` 的，就是这段“对齐后的实际可见区间”：

```cpp
const auto [actual_kv_tile_pos_left, actual_kv_tile_pos_right] =
    calcu_kv_tile_pos(...);
const auto [aligned_actual_kv_tile_pos_left, aligned_actual_kv_tile_pos_right] =
    align_kv_tile_pos(...);
attn_impl.execute_attention(...,
    aligned_actual_kv_tile_pos_left,
    aligned_actual_kv_tile_pos_right,
    actual_kv_token_num, ...);
```

- 因此：
  - 不会把完整 `kv_len` 都读一遍
  - 会读取“对齐后的可见区间”
  - 只会把对齐额外带进来的那一小段不可见位置一并读入当前 `kv_tile`
- 随后 `apply_mask(...)` 会把这部分额外位置对应的 `logits` 设成 `-inf`，使其在 `softmax` 后权重为 `0`。
- 这些被 mask 掉的位置仍占当前 `kv_tile` 的临时槽位，但当前 `kv_tile` 之外的大段不可见 `KV` 根本不会进入这次计算。

- scheduler 先给每个逻辑桶一个目标预算：

```cpp
const int64_t kv_len_per_thread = ...;
int64_t remaining_kv_len = kv_len_per_thread;
```

- 这里的 `kv_len_per_thread` 不是预先写死的固定值，而是这一次 `schedule()` 根据当前输入动态算出来的。
- 它会随这次 attention 的实际工作负载变化，至少受这些量影响：
  - 当前 batch 各 request 的可见 `KV` 长度总和 `total_kv_len`
  - `thread_num`
  - `kv_len_alignment`
  - 当前是按 `num_heads_kv` 还是 `num_heads_q` 展开
- 所以它更准确的含义是：
  - “这一次调度里，每个逻辑线程桶希望承接的目标 `KV` 工作量”

- 每往当前桶里吸收一段 q / KV 工作，就会做：

```cpp
remaining_kv_len -= curr_kv_len;
```

- 所以它控制的是负载均衡目标，不是硬容量上限。
- 当前桶塞不下时，切到下一个桶。
- 所以一个桶里放多少个 workitems 没有固定值，主要取决于这些 workitems 累积的 KV 工作量是否接近 `kv_len_per_thread`。
- 这里还要分两种情况：
  - 如果当前 q 片段满足 split-KV 条件，scheduler 可以把这段 q 对应的 KV 区间切开，分别放进不同桶
  - 如果当前 q 片段不满足 split-KV 条件，scheduler 不会把这段 q 对应的 KV 区间切开
- 所以当“某一个 q 片段对应的那段可见 KV 太长，超过当前桶预算”时，可能出现：
  - 当前桶在预算意义上已经“放不下”这段任务
  - 但如果这段 q 当前不允许 split-KV，scheduler 仍会把这段完整工作量记到当前桶里
  - 随后让 `remaining_kv_len` 变成负数，并停止继续向这个桶接更多 workitems
- 也就是说：
  - `remaining_kv_len` 是软预算
  - 会出现“超预算但仍承接该任务”的情况
  - 不会因为单个 q 片段的可见 KV 太长就导致调度失败
- 因此，同一个 request 的 q 可以跨多个桶；而同一个 q 片段对应的 KV 是否会跨桶，取决于 split-KV 条件是否成立。

### 10.3.5 request 和逻辑线程桶不是一一对应

- scheduler 进入 `for (req_id ...)` 之前就先初始化：

```cpp
int32_t curr_thread_id = 0;
int64_t remaining_kv_len = kv_len_per_thread;
```

- 这两个状态在 request 之间不会重置；只有在当前桶预算不够，或进入 split-KV 分支时，代码才会：

```cpp
++curr_thread_id;
remaining_kv_len = kv_len_per_thread;
```

- 因此：
  - 一个 request 可以跨多个逻辑线程桶
  - 一个逻辑线程桶也可以同时装“上一个 request 的尾巴”和“下一个 request 的开头”
  - request 边界不是桶边界

### 10.4 第 4 层：运行期按 `kv_head_idx × 逻辑桶` 展开成 attention task

- 这一步就是前面第 6 节的 `kv_head_idx / thread_offset` 映射。
- 同一组 workitems 会在不同 `kv_head_idx` 下各跑一遍。

### 10.5 第 5 层：attention task 内再切 `q_tile -> kv_tile -> q micro-iter`

- workitem 先按 `default_q_tile_token_num` 切成 `q_tile`
- 再为当前 `q_tile` 计算可见 `KV` 区间
- 这个 `KV` 区间继续切成多个 `kv_tile`
- 每个 `kv_tile` 内，再按 `max_q_token_num_per_iter` 切最内层 q micro-iter

## 11. CPU attention 不是“先把完整 KV 全算一遍再靠 mask 扔掉”

- 更准确的说法是：
  - 先用 `calcu_kv_tile_pos(...)` 做可见 KV 的粗裁剪
  - 再按 tile 继续裁剪
  - 最后只对因 block 对齐额外带进来的少量位置做 `apply_mask(...)`

### 11.1 q_tile 层的可见 KV 粗裁剪

```cpp
const auto [kv_tile_start_pos, kv_tile_end_pos] =
    AttentionScheduler::calcu_kv_tile_pos(
        kv_start_pos, kv_end_pos, q_tile_start_pos, q_tile_end_pos,
        sliding_window_left, sliding_window_right);
```

### 11.2 q micro-iter 层再裁一次

```cpp
const auto [actual_kv_tile_pos_left, actual_kv_tile_pos_right] =
    AttentionScheduler::calcu_kv_tile_pos(
        kv_tile_pos_left, kv_tile_pos_right, q_tile_pos_left, q_tile_pos_right,
        sliding_window_left, sliding_window_right);
```

### 11.3 为什么还需要 `apply_mask(...)`

- 因为 kernel 会把 KV 范围按 `BlockSizeAlignment` 对齐：

```cpp
const auto [aligned_actual_kv_tile_pos_left, aligned_actual_kv_tile_pos_right] =
    AttentionScheduler::align_kv_tile_pos(...);
```

- 所以少量本不该参与的 token 可能因为对齐被带进 GEMM。
- 这些尾巴会在 `apply_mask(...)` 中被置成无效。

所以最准确的理解是：

- 大范围无效区不会进入主计算循环
- 少量对齐带来的额外位置可能先参与 QK
- 然后再被 mask 成 `-inf`

## 12. 例子：`q=(1024,16,128), kv=(1024,2,128)` 时到底怎么拆

### 12.1 固定条件

假设：

- 只有 1 个 request
- 本地 `q = [1024, 16, 128]`
- 本地 `kv = [1024, 2, 128]`
- 当前走 `VEC`

那么：

- `MaxQHeadNumPerIteration = 8`
- `q_heads_per_kv = 16 / 2 = 8`
- `actual_q_heads_per_kv = 8`
- `max_q_token_num_per_iter = 8 / 8 = 1`

这就意味着：

- 最内层一次 `execute_attention(...)` 只处理 `1 token × 8 q heads`

### 12.2 scheduler 先生成多少 workitems

- scheduler 先按 `max_num_q_token_per_iter = 1` 逐 token 扫
- 但写回 workitem 时不会把 `1024` 个 q token 平均切成若干等长 workitem，而是按累计 `total_kv_len` 是否接近当前逻辑桶预算来决定何时切分
- 所以同一个 request 被切出来的各个 workitem，`q_token_num` 往往并不相同
- 对这个例子，更接近的理解是：
  - scheduler 逐 token 扫描
  - 把相邻 q 片段不断并入当前 workitem
  - 当累计的对齐后可见 `KV` 工作量接近预算，或碰到 tile 边界时，再写回一个 workitem
- 因此运行期常见会看到：
  - 一个 workitem 覆盖很多 q token
  - 但不同 workitem 覆盖的 q 范围不等长
- 这一点与第 `10.3.2` 节一致：被尽量拉齐的是 `total_kv_len`，不是 `q_token_num`

### 12.3 这些 workitems 怎么进入 attention task

更准确地说，顺序是：

- scheduler 先把 workitems 分到 `thread_num` 个逻辑桶里，并统计每个桶的 `workitem_num_per_thread[i]`
- 然后从前往后找第一个空桶；前面连续非空的桶数才记为 `effective_thread_num`

如果最后 `effective_thread_num = 8`，表示：

- 前 8 个逻辑桶非空
- 从第 8 号桶之后开始出现空桶
- 运行期只会按这 8 个非空桶去展开 attention task

接着再按 `kv_head_idx` 展开。

所以 attention task 总数是：

```text
2 * 8 = 16
```

### 12.4 单个 attention task 里又怎么往下切

假设某个 task 固定了：

```text
kv_head = 0
thread_offset = 3
```

并且这个桶里有若干 workitems。

那么线程会：

1. 先固定 `kv_head=0` 对应的一组 q heads
2. 顺序遍历该桶里的 workitems
3. 每个 workitem 再切成若干 `q_tile`
4. 每个 `q_tile` 再切若干 `kv_tile`
5. 每个 `kv_tile` 里，再按 `max_q_token_num_per_iter=1` 扫 1 token 的 q micro-iter

### 12.5 最内层一次真正执行的 shape

当前 `VEC` 下，最内层一次 `execute_attention(...)` 更接近：

- `Q_micro = [1, 8, 128]`
- `K_tile = [kv_tile_token_num, 128]`
- `V_tile = [kv_tile_token_num, 128]`

也就是说：

- 不是一次把 `[40, 16, 128]` 全部算完
- 而是同一个线程在自己的 task 里，一次次拿最小的 q micro-iter 去扫当前 `kv_tile`

## 13. 开启 `kv_split` 时，decode 是怎么拆的

### 13.1 什么时候会真的 split KV

- scheduler 里先设置：

```cpp
const int32_t split_kv_q_token_num_threshold =
    input.enable_kv_split ? 1 : 0;
```

- 真正进入 split KV 分支，还要满足尾段 q token 等条件。
- decode 常见 `q_token_num=1`，所以这组条件最容易成立。

### 13.2 split 后会生成什么

- 会生成多个 `AttentionWorkItemGroup`，它们共享同一个 q 片段，但各自只负责不同 KV 区间
- 同时还会生成对应的 `ReductionWorkItemGroup`

所以 split-KV 的实质是：

- attention 阶段先算多个 partial outputs
- reduction 阶段再把这些 partial outputs 归并起来

## 14. 边界

- 本页回答的是当前 legacy CPU attention 的并行拆分机制。
- 不直接回答：
  - `acc-local-l3` 的 subgroup 调度
  - `NPS/TP` 放置优劣
  - 服务侧端到端吞吐

## 15. 继续读代码时最值得看的变量

如果你在 `AttentionMainLoop::operator()` 里单步，最值得盯住的是：

- `effective_thread_num`
- `workitem_groups_counter_num`
- `kv_head_idx`
- `thread_offset`
- `curr_workitem_groups_num`
- `default_q_tile_token_num`
- `max_q_token_num_per_iter`
- `q_tile_token_num`
- `actual_kv_tile_pos_left / right`

## 16. 关键源码位置

- `csrc/cpu/cpu_attn.cpp:100-162`
- `csrc/cpu/cpu_attn.cpp:225-292`
- `csrc/cpu/cpu_attn_impl.hpp:455-760`
- `csrc/cpu/cpu_attn_impl.hpp:773-856`
- `csrc/cpu/cpu_attn_impl.hpp:1438-1935`
