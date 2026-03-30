# 10. CPU attention 并行计算过程：线程、task、workitem 与 tile

> 更新时间：2026-03-25 23:13 +0800  
> 范围：`vLLM v1 CPU backend` 的 `cpu_attention_with_kv_cache -> AttentionMainLoop::operator()`。  
> 目标：只回答“并行计算到底怎么拆”。不展开 NUMA/TP 通信优化，不重复讲大而全背景。  
> 平台：`2 x AMD EPYC 9745 128-Core Processor`。  
> 假设：除明确绑定源码位置的事实外，文中带“假设”的数字例子都基于当前常见分析口径：`Qwen3-30B-A3B + TP=2 + VEC kernel`。

## 概述
- 真正参与 CPU attention 计算的是 `AttentionMainLoop::operator()` 里拉起的 **真实 OpenMP 线程**；但调度元数据里还有一层独立的 **逻辑线程桶**，两者不是一回事。
- 线程运行时不是直接拿到一个 workitem，而是先通过 `metadata.acquire_counter()` 动态抢一个 **runtime task**；这个 task 通常对应“某个 `kv_head_idx` + 某个逻辑线程桶”。
- 一个 runtime task 下面通常会顺序执行多个 workitems；每个 workitem 在同一个线程里又会继续切成 `q_tile -> kv_tile -> q micro-iter`，最后才真正调用一次次 `execute_attention(...)`。

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

- 前半段 `schedule()` 负责 **生成调度 metadata**，包括 `workitem/reduction item`、`effective_thread_num`、前缀和和 scratchpad 大小；后半段 `operator()` 才负责 **真正执行 attention/reduction 计算**。证据：`csrc/cpu/cpu_attn.cpp:100-162`、`csrc/cpu/cpu_attn_impl.hpp:600-662`、`csrc/cpu/cpu_attn.cpp:225-291`。

## 2. 先把 7 个概念分清

| 名称 | 所在层次 | 最短定义 |
| --- | --- | --- |
| `thread_num` | OpenMP 真实线程层 | 这次 kernel 真正拉起多少个 OMP 线程 |
| `effective_thread_num` | 调度元数据层 | 实际分到 workitem 的非空逻辑线程桶数量 |
| `task_idx` | runtime task 层 | 线程运行时从全局 atomic counter 抢到的任务编号 |
| `AttentionWorkItemGroup` | workitem 层 | request-local 的 q 片段 + 一段逻辑 KV 区间 |
| `q_tile` | kernel tile 层 | 一个 workitem 在当前线程里继续切出的较大 q 子块 |
| `kv_tile` | kernel tile 层 | 当前 q_tile 对应的 KV 访问区间再切出的较小块 |
| `q micro-iter` | 最内层循环 | 当前 `VEC` 下通常是 `1 token × q_heads_per_kv` 的最小 q 子块 |

## 3. 有多少线程真正参与计算

### 3.1 真实线程数：`thread_num`

- `AttentionMainLoop::operator()` 一开始直接取 `omp_get_max_threads()`，然后进入：

```cpp
#pragma omp parallel for schedule(static, 1)
for (int thread_id = 0; thread_id < thread_num; ++thread_id) { ... }
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1340-1346`。

- 这说明真正被拉起的执行者就是这 `thread_num` 个 OMP 线程。

### 3.2 为什么还要有 `effective_thread_num`

- `schedule()` 生成 workitems 时，并不是简单地“有多少线程就给多少 workitem”，而是先用
  - `thread_num`
  - `total_kv_len`
  - `kv_len_per_thread`
  去构造一组 **逻辑线程桶**。证据：`csrc/cpu/cpu_attn_impl.hpp:384-444`。
- 最后它扫描 `workitem_num_per_thread[]`，找到真正非空的桶数：

```cpp
int32_t effective_thread_num = 0;
for (; effective_thread_num < thread_num; ++effective_thread_num) {
  if (workitem_num_per_thread[effective_thread_num] == 0) {
    break;
  }
}
```

证据：`csrc/cpu/cpu_attn_impl.hpp:620-634`。

- 因此：
  - `thread_num`：真实线程数
  - `effective_thread_num`：真正分到 workitem 的逻辑桶数

### 3.3 哪些线程“真正参与了计算”

- 所有 `thread_num` 个线程都会被拉起，并完成初始化、scratchpad 绑定、可能的 reduction flag 清零。证据：`csrc/cpu/cpu_attn_impl.hpp:1381-1440`。
- 进入主循环后，线程通过 `metadata.acquire_counter()` 动态抢任务；如果任务数少于线程数，后面的线程可能几乎不算 attention 主体，直接很快退出。证据：`csrc/cpu/cpu_attn_impl.hpp:1442-1448`。
- 所以更准确地说：
  - **参与 kernel 启动的线程数** = `thread_num`
  - **metadata 中真正有非空任务桶的数量** = `effective_thread_num`
  - **真正做了多少 attention/reduction 计算** 还取决于运行时 task 是否足够多

## 4. attention task 总数为什么和 `effective_thread_num` 有关

- 在运行期，attention task 的编号空间定义为：

```cpp
const int32_t workitem_groups_counter_num =
    actual_kv_head_num * effective_thread_num;
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1424-1425`。

- 所以 attention task 总数不是 `workitem_group_num`，而是：

```text
attention task 总数 = actual_kv_head_num * effective_thread_num
```

- 这意味着 task 是先按 `kv_head` 展开，再按逻辑线程桶展开，而不是“一个 workitem 一个 task”。

## 5. 一个 runtime task 到底对应什么

- 线程抢到 `task_idx` 后，先做这两个映射：

```cpp
const int32_t kv_head_idx = task_idx / effective_thread_num;
const int32_t thread_offset = task_idx % effective_thread_num;
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1451-1455`。

- 然后通过 `cu_workitem_num_per_thread` 前缀和，找到这个逻辑线程桶对应的一整段 `AttentionWorkItemGroup[]`：

```cpp
AttentionWorkItemGroup* const curr_workitem_groups =
    workitem_groups + cu_workitem_num_per_thread[thread_offset];
const int32_t curr_workitem_groups_num =
    cu_workitem_num_per_thread[thread_offset + 1] -
    cu_workitem_num_per_thread[thread_offset];
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1456-1460`。

- 所以一个 runtime attention task 可以精确理解成：

```text
固定 kv_head_idx
+ 固定 thread_offset
+ 这个 thread_offset 桶里的一整段 workitems
```

- 这也是为什么：
  - task 不等于 workitem
  - 一个 task 可能包含多个 workitems

## 6. 一个 workitem 到底是什么

`AttentionWorkItemGroup` 结构体里有：

- `req_id`
- `q_token_id_start`
- `q_token_num`
- `kv_split_pos_start`
- `kv_split_pos_end`
- `total_kv_len`
- `split_id`
- `local_split_id`

定义见：`csrc/cpu/cpu_attn_impl.hpp:20-30`。

所以一个 workitem 更接近：

```text
某个 request
+ 一段连续 q token
+ 一段逻辑 KV token 区间 [kv_split_pos_start, kv_split_pos_end)
+ 若 split-KV，则再带 `split_id / local_split_id`
```

注意两点：

- workitem **没有** head 维字段，所以它不是“按 head 切出来”的对象。
- head 维是在 runtime task 里通过 `kv_head_idx` 固定下来的。

## 7. 一个线程执行 1 个 task 时，到底在做哪部分计算

线程拿到 attention task 后，执行顺序是：

```text
for 每个 workitem
  for q_tile
    计算当前 q_tile 的可见 KV 区间
    for kv_tile
      for q micro-iter
        execute_attention(...)
    写回 final_output(...) 或 partial_output(...)
```

对应源码：

- 遍历 workitems：`csrc/cpu/cpu_attn_impl.hpp:1464-1488`
- 按 q tile 循环：`csrc/cpu/cpu_attn_impl.hpp:1490-1525`
- 计算当前 q_tile 的可见 KV 范围：`csrc/cpu/cpu_attn_impl.hpp:1527-1538`
- 按 kv_tile 循环：`csrc/cpu/cpu_attn_impl.hpp:1631-1636`
- 按 q micro-iter 循环：`csrc/cpu/cpu_attn_impl.hpp:1637-1658`
- 真正执行 `QK -> softmax -> PV`：`csrc/cpu/cpu_attn_impl.hpp:1717-1727`

所以，线程执行 1 个 task 时，做的不是“一个完整 request”，也不是“一个完整 q tile”，而是：

- 固定一个 `kv_head_idx`
- 跑该 task 对应桶里的所有 workitems
- 每个 workitem 再逐层切成更小 tile；且 `final_output(...) / partial_output(...)` 的写回粒度是 **每个 q_tile 一次**，不是“整个 workitem 只写回一次”

## 8. 一个完整 `Q @ K^T -> softmax -> P @ V` 怎么被拆成并行任务

### 8.1 第 1 层：先按 request 内的 q token 聚 workitem

- scheduler 先在单个 request 内按 `token_id += max_num_q_token_per_iter` 扫 q token。证据：`csrc/cpu/cpu_attn_impl.hpp:460-463`。
- 然后把若干相邻 q token 聚成一个较大的 `curr_workitem`。证据：`csrc/cpu/cpu_attn_impl.hpp:479-480`、`csrc/cpu/cpu_attn_impl.hpp:546-547`。

### 8.2 第 2 层：把这些 workitems 塞进逻辑线程桶

- 关键约束是每个桶的 `remaining_kv_len` 预算。证据：`csrc/cpu/cpu_attn_impl.hpp:447-449`。
- 当前桶塞不下时，就切到下一个桶。证据：`csrc/cpu/cpu_attn_impl.hpp:505-523`、`csrc/cpu/cpu_attn_impl.hpp:517-519`。

所以一个桶放多少个 workitems，没有固定值；主要看这些 workitems 累积的 KV 工作量有没有接近 `kv_len_per_thread`。

### 8.3 第 3 层：运行期按 `kv_head_idx × 逻辑桶` 展开成 task

- 这一步就是上面第 5 节的 `kv_head_idx / thread_offset` 映射。
- 同一组 workitems 会在不同 `kv_head_idx` 下各跑一遍。

### 8.4 第 4 层：task 内再切 tile

- workitem 先按 q token 继续切成 `q_tile`，大小大致受 `default_q_tile_token_num` 控制。证据：`csrc/cpu/cpu_attn_impl.hpp:1490-1504`。
- 当前 q_tile 的可见 KV 区间再被切成多个 `kv_tile`。证据：`csrc/cpu/cpu_attn_impl.hpp:1531-1538`、`csrc/cpu/cpu_attn_impl.hpp:1631-1636`。
- 每个 `kv_tile` 内，再按 `max_q_token_num_per_iter` 切最小 q micro-iter。证据：`csrc/cpu/cpu_attn_impl.hpp:1637-1648`。

### 8.5 CPU attention 不是“先和完整 KV 都乘一遍，再全靠 mask 扔掉”

- 很多框架的高层实现会让人形成一种直觉：先让 `Q` 和完整 `K` 做乘法，然后再通过 causal mask 把右上角无效区域置成 `-inf`。但对当前这个 CPU attention kernel，更准确的说法是：
  - **先用 `calcu_kv_tile_pos(...)` 做可见 KV 的粗裁剪**
  - **再按 tile 继续裁剪**
  - **最后只对因 block 对齐额外带进来的少量位置做 `apply_mask(...)`**
- 第一次粗裁剪发生在 q tile 层：

```cpp
const auto [kv_tile_start_pos, kv_tile_end_pos] =
    AttentionScheduler::calcu_kv_tile_pos(
        kv_start_pos, kv_end_pos, q_tile_start_pos, q_tile_end_pos,
        sliding_window_left, sliding_window_right);
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1531-1538`。

- 第二次裁剪发生在更小的 q micro-iter 层：

```cpp
const auto [actual_kv_tile_pos_left, actual_kv_tile_pos_right] =
    AttentionScheduler::calcu_kv_tile_pos(
        kv_tile_pos_left, kv_tile_pos_right, q_tile_pos_left, q_tile_pos_right,
        sliding_window_left, sliding_window_right);
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1651-1669`。

- `calcu_kv_tile_pos(...)` 本身的逻辑很直接：对 causal 场景，`sliding_window_right = 0`，因此会把 `kv_right_pos` 收紧到 `q_right_pos`。证据：`csrc/cpu/cpu_attn_impl.hpp:678-689`。
- 因此，对 prefill 里的某段 `q[360:400)`，这层 CPU kernel 不是认真去算完整的 `kv[0:1024)`；外层通常会先把可见 KV 粗裁成接近 `[0:400)`，然后线程只在这个范围内继续切 `kv_tile`。
- 但因为 kernel 会把 KV 范围按 `BlockSizeAlignment` 向上对齐，例如从 `[0:400)` 对齐成 `[0:416)`，所以仍可能让少量本不该参与的 token 落进 GEMM。真正把这些尾巴置成无效的，是后面的 `apply_mask(...)`。证据：`csrc/cpu/cpu_attn_impl.hpp:691-697`、`csrc/cpu/cpu_attn_impl.hpp:1013-1015`、`csrc/cpu/cpu_attn_impl.hpp:1095-1144`。
- 所以最准确的理解是：
  - 大范围无效区不会进入主计算循环
  - 少量对齐带来的额外位置可能会先参与 QK，但随后被 mask 成 `-inf`
  - 当前 CPU kernel 既做“裁剪”，也做“mask”，不是二选一

## 9. 例子：`q=(1024,16,128), kv=(1024,2,128)` 时到底怎么拆

### 9.1 固定条件

假设：

- 只有 1 个 request
- 本地 `q = [1024, 16, 128]`
- 本地 `kv = [1024, 2, 128]`
- 当前走 `VEC`
- 所以 `MaxQHeadNumPerIteration = 8`
- `q_heads_per_kv = 16 / 2 = 8`
- `max_q_token_num_per_iter = 8 / 8 = 1`
- `default_q_tile_token_num ≈ 40`

证据：`csrc/cpu/cpu_attn_vec.hpp:126-131`、`csrc/cpu/cpu_attn_impl.hpp:1355-1364`。

### 9.2 scheduler 先生成多少 workitems

- 因为 `default_q_tile_token_num ≈ 40`，上层常见会先把 1024 个 q token 聚成大约 26 个 workitems：

```text
workitem0  = q[0:40)
workitem1  = q[40:80)
workitem2  = q[80:120)
...
workitem25 = q[1000:1024)
```

- 这里 workitem 数量大约是：

```text
ceil(1024 / 40) = 26
```

- 注意：这一步 **不会因为 2 个 `kv_head` 而把 workitem 数直接乘 2**。head 维展开发生在后面的 task 层，不在 workitem 层。

### 9.3 这些 workitems 怎么进入 task

假设最后 `effective_thread_num = 8`，那么：

- 26 个 workitems 会先被分进 8 个逻辑桶
- 可能类似：

```text
bucket0: workitem0, workitem1, workitem2
bucket1: workitem3, workitem4, workitem5
bucket2: workitem6, workitem7, workitem8
...
bucket7: workitem21, ..., workitem25
```

然后再按 `kv_head_idx` 展开：

```text
task(kv_head=0, bucket0)
task(kv_head=0, bucket1)
...
task(kv_head=1, bucket0)
task(kv_head=1, bucket1)
...
```

所以：

- workitems 约 26 个
- attention task 总数约 `2 * 8 = 16`

这也说明：

- task 数可以小于 workitem 数
- 因为一个 task 下面可以包含多个 workitems

### 9.4 线程拿到 1 个 task 后到底遍历多少 token

假设某个线程抢到：

```text
task = (kv_head=0, bucket3)
```

并且 `bucket3` 里有：

```text
workitem9  = q[360:400)
workitem10 = q[400:440)
workitem11 = q[440:480)
```

那么这个线程在这个 task 里会顺序做：

1. 先算 `kv_head=0` 对应的 8 个 q heads
2. 跑 `workitem9`
3. 跑 `workitem10`
4. 跑 `workitem11`

所以这个 task 在 q token 维上，一共覆盖：

```text
40 + 40 + 40 = 120 个 q tokens
```

但这 120 个 token 不是一次 GEMM 做完，而是继续往下切。

### 9.5 单个 workitem 里又怎么切

以 `workitem9 = q[360:400)` 为例：

- 上层 q 片段可以理解成 `[40, 16, 128]`
- 进入 `kv_head=0` 的 task 后，真正处理的是 `[40, 8, 128]`

然后：

1. 如果当前 `default_q_tile_token_num≈40`，那这个 workitem 常常只对应 1 个 `q_tile`
2. 对这个 `q_tile`，先根据 causal 约束把可见 KV 区间粗裁成接近 `[0:400)`，而不是直接看完整 `[0:1024)`
3. 再把这段可见 KV 区间切成多个 `kv_tile`
4. 每个 `kv_tile` 里，再按 `max_q_token_num_per_iter=1` 做 40 次 q micro-iter

也就是：

```text
workitem9
  -> q_tile = [40, 8, 128]
       -> visible kv range ≈ [0:400)，对齐后可能接近 [0:416)
       -> kv_tile0
            -> 40 次 q micro-iter，每次 [1, 8, 128]
       -> kv_tile1
            -> 40 次 q micro-iter，每次 [1, 8, 128]
       -> ...
```

- 若某个 `kv_tile` 因对齐带入了少量超过当前 q 位置的 token，这些位置会在 `apply_mask(...)` 里被置成无效；所以这里不是“完整算 1024，再全靠 mask 扔掉”，而是“先裁剪到可见区，再对齐，再 mask 掉少量额外尾巴”。

### 9.6 最内层一次真正执行的 shape

当前 `VEC` 下，最内层一次 `execute_attention(...)` 更接近：

- `Q_micro = [1, 8, 128]`
- `K_tile = [kv_tile_token_num, 128]`
- `V_tile = [kv_tile_token_num, 128]`

也就是说：

- 不是一次把 `[40,16,128]` 全部算完
- 而是同一个线程在自己的 task 里，一次次拿最小的 q micro-iter 去扫当前 `kv_tile`

## 10. 例子：开启 `kv_split` 时，decode 是怎么拆的

### 10.1 先看它什么时候会真的 split KV

- `schedule()` 里先把 `split_kv_q_token_num_threshold` 设成：

```cpp
const int32_t split_kv_q_token_num_threshold =
    input.enable_kv_split ? 1 : 0;
```

证据：`csrc/cpu/cpu_attn_impl.hpp:401-403`。

- 真正进入 `split kv` 分支，还要同时满足：
  - 当前 `enable_kv_split=true`
  - 当前这段 q 是尾段
  - `q_tile_token_num <= split_kv_q_token_num_threshold`
- 对 decode，常见就是 `q_token_num=1`，所以这组条件最容易成立。证据：`csrc/cpu/cpu_attn_impl.hpp:527-581`。

### 10.2 固定一个 decode 例子

假设：

- 只有 1 个 request
- `q = [1, 16, 128]`
- `kv = [4096, 2, 128]`
- 当前走 `VEC`
- `enable_kv_split = true`
- `q_heads_per_kv = 16 / 2 = 8`
- `max_num_q_token_per_iter = 8 / 8 = 1`
- 假设本次调度后 `effective_thread_num = 4`
- 假设按当前 `kv_len_per_thread` 预算，4096 个可见 KV token 被切成 4 段，每段约 1024 token

其中最后两条是为了讲清执行过程做的 **假设**；真实 split 数量取决于 `thread_num`、`kv_len_per_thread`、对齐粒度和可见 KV 长度。相关计算位置：`csrc/cpu/cpu_attn_impl.hpp:435-439`、`csrc/cpu/cpu_attn_impl.hpp:447-581`。

### 10.3 scheduler 生成出来的 workitems 长什么样

- 这个 decode 请求只有 1 个 q token，所以所有 split workitem 都对应同一个 q 片段：

```text
q[0:1)
```

- 但它们各自只负责不同的 KV 区间。按上面的假设，可能得到：

```text
workitem0: req=0, q[0:1), kv[0:1024),    split_id=0, local_split_id=0
workitem1: req=0, q[0:1), kv[1024:2048), split_id=1, local_split_id=1
workitem2: req=0, q[0:1), kv[2048:3072), split_id=2, local_split_id=2
workitem3: req=0, q[0:1), kv[3072:4096), split_id=3, local_split_id=3
```

- 同时还会生成 1 个 `ReductionWorkItemGroup`：

```text
reduction_item0: req=0, q[0:1), split_start_id=0, split_num=4
```

- 这正对应源码里的两类对象：
  - split attention workitem：`AttentionWorkItemGroup`
  - split 归并 workitem：`ReductionWorkItemGroup`

证据：`csrc/cpu/cpu_attn_impl.hpp:20-30`、`csrc/cpu/cpu_attn_impl.hpp:59-76`、`csrc/cpu/cpu_attn_impl.hpp:553-581`。

### 10.4 然后会展开成多少个 runtime task

- attention task 总数：

```text
actual_kv_head_num * effective_thread_num = 2 * 4 = 8
```

- reduction task 总数：

```text
actual_kv_head_num * reduction_item_num = 2 * 1 = 2
```

- 所以这个例子里总 runtime task 数是：

```text
8 个 attention task + 2 个 reduction task = 10
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1420-1429`。

### 10.5 attention task 的视角怎么理解

线程先抢到 `task_idx`，再映射成：

```cpp
const int32_t kv_head_idx = task_idx / effective_thread_num;
const int32_t thread_offset = task_idx % effective_thread_num;
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1451-1455`。

所以在这个例子里，attention task 可以理解成：

```text
task0: kv_head=0, bucket0 -> 跑 workitem0
task1: kv_head=0, bucket1 -> 跑 workitem1
task2: kv_head=0, bucket2 -> 跑 workitem2
task3: kv_head=0, bucket3 -> 跑 workitem3
task4: kv_head=1, bucket0 -> 跑 workitem0
task5: kv_head=1, bucket1 -> 跑 workitem1
task6: kv_head=1, bucket2 -> 跑 workitem2
task7: kv_head=1, bucket3 -> 跑 workitem3
```

也就是说：

- 同一个 `q[0:1)` 会被重复执行 2 轮
- 每一轮固定 1 个 `kv_head`
- 每一轮里再按不同 `kv_span` 分成多个 split task

### 10.6 单个线程拿到 1 个 split task 后，实际算什么

假设某个线程抢到：

```text
task1 = (kv_head=0, bucket1)
```

那它这次不是算完整的：

```text
q[0:1), head[0:8), kv[0:4096)
```

而是只算：

```text
q[0:1), head[0:8), kv[1024:2048)
```

在线程内部，这个 split task 仍会继续切成更小 tile：

```text
split task
  -> q_tile = [1, 8, 128]
  -> kv_tile0
       -> q micro-iter = [1, 8, 128]
  -> kv_tile1
       -> q micro-iter = [1, 8, 128]
  -> ...
```

- 因为 decode 这里只有 1 个 q token，所以这里的 `q_tile` 和 `q micro-iter` 通常都只覆盖这 1 个 token。
- 真正变化的是它扫过的 `kv_tile`，但这些 `kv_tile` 都只落在当前 split 的 `kv_span` 内。证据：`csrc/cpu/cpu_attn_impl.hpp:1472-1474`、`csrc/cpu/cpu_attn_impl.hpp:1527-1538`、`csrc/cpu/cpu_attn_impl.hpp:1631-1658`。

### 10.7 split task 算完后，先写 partial，不直接写最终 output

- 非 split 情况下，线程会直接走 `final_output(...)`。
- split 情况下，线程会走 `partial_output(...)`，把当前 split 的 `partial output`、`local max`、`local sum` 和 `completion flag` 写入 reduction buffer。

对应代码：

```cpp
if (curr_spilt_id == -1) {
  final_output(...);
} else {
  partial_output(..., split_flag_buffer);
}
```

证据：`csrc/cpu/cpu_attn_impl.hpp:1742-1760`、`csrc/cpu/cpu_attn_impl.hpp:1903-1934`。

所以这个 decode 例子中，前 8 个 attention task 做完后，得到的是：

```text
kv_head=0: split0, split1, split2, split3 的 partial
kv_head=1: split0, split1, split2, split3 的 partial
```

而不是最终 `output[0, 16, 128]`。

### 10.8 reduction task 再把多个 split 合成最终结果

- 当 `task_idx >= workitem_groups_counter_num` 时，线程开始执行 reduction task。证据：`csrc/cpu/cpu_attn_impl.hpp:1765-1808`。
- 对上面的例子，reduction task 可以理解成：

```text
reduction task 0: kv_head=0, reduce split0..split3 -> output token0 的前 8 个 heads
reduction task 1: kv_head=1, reduce split0..split3 -> output token0 的后 8 个 heads
```

- `reduce_splits(...)` 会先等待所有 split flag 就绪，再把多个 split 的 softmax 统计量重新归一，最后再 `final_output(...)` 写入真正输出。证据：`csrc/cpu/cpu_attn_impl.hpp:1802-1808`、`csrc/cpu/cpu_attn_impl.hpp:1817-1899`。

所以最终链路就是：

```text
split attention task
  -> partial_output(split0)
  -> partial_output(split1)
  -> partial_output(split2)
  -> partial_output(split3)
  -> reduce_splits(split0..split3)
  -> final_output
```

### 10.9 用一句话概括 decode + kv_split 的并行方式

- 不开 `kv_split` 时，更像是：

```text
固定 kv_head，1 个 task 直接扫完整可见 KV
```

- 开 `kv_split` 时，更像是：

```text
固定 kv_head，把同一个 q token 对应的可见 KV 区间切成多个 split
多个线程分别算各自 split 的 partial
最后再起 reduction task 做一次归并
```

这也是 decode 场景里 `kv_split` 最核心的并行化方向：**q token 很少，没法再沿 q 维展开太多并行度，于是转而沿 KV token 维把同一个 q token 的计算拆成多个 split。**

## 11. 最后只记这 5 句话就够了

1. 真正的执行者是 OpenMP 真实线程；逻辑线程桶只是 metadata 里的分桶结果。  
2. `effective_thread_num` 是非空逻辑桶数量，不是实际线程数。  
3. attention task = `kv_head_idx × 逻辑线程桶`，不是一个 workitem。  
4. workitem 主要描述 request 内的一段 q token 和一段逻辑 KV 区间；它不直接按 head 展开。  
5. 同一个线程拿到一个 task 后，会顺序执行该 task 下的多个 workitems；每个 workitem 又会继续切成 `q_tile -> kv_tile -> q micro-iter`。

## 12. 结合 2026-03-23 PCM 结果看 prefill 的可优化点

### 12.1 数据口径

- 原始 PCM 文件：
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B16_I1024_O1/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-27-32/report-cumulative.csv`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B16_I512_O1/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-59-21/report-cumulative.csv`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B16_I128_O1/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_21-06-25/report-cumulative.csv`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/decode_B16_I1_O1024/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-48-58/report-cumulative.csv`
- 实验记录：`info/实验记录/实验2.md`
- `PCm_res` 当前只有 `prefill_B16_I128/O1`、`prefill_B16_I512/O1`、`prefill_B16_I1024/O1` 和 `decode_B16_I1/O64~O1024`。如果要判断 `I1024/O1024` 的同批次结果，当前无数据。

### 12.2 先看 `I1024/O1` 与 `I1/O1024`

| case | L3 Miss % | Ave L3 Miss Latency | same-node another CCX | Remote miss latency | Remote DRAM Reads % | All Demand DC Fills | Total Mem Bw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill `B16/I1024/O1` | 49.43 | 180.12 ns | 2.91% | 2.09% | 0.03% | 2.18 | 328.64 GB/s |
| decode `B16/I1/O1024` | 96.51 | 326.57 ns | 1.51% | 0.07% | 0.03% | 5.59 | 350.99 GB/s |

- 这些值来自上面的 `report-cumulative.csv` 总值列。
- `info/实验记录/实验2.md` 中的 L3 miss 率与 miss latency 方向一致，但 `Total Mem Bw` 记录为 prefill `323.13 GB/s`、decode `456.20 GB/s`，与当前 raw csv 的 `328.64/350.99 GB/s` 不完全一致。这里先做一个假设：两者不是同一次整理口径，因此本文不把带宽作为主证据。
- 从当前 raw csv 看，`prefill_B16_I1024_O1` 的问题不像“远端 DRAM 明显更高”，因为 `Remote DRAM Reads %` 两组都只有 `0.03%`，`Remote miss latency %` 也只是 `2.09%`。
- 但 prefill 的 `same-node another CCX %` 比 decode 高，说明它相对更像“同 socket 内跨 CCX/L3 的局部性损失”，而不是典型的跨 socket 远端内存主导。

### 12.3 再看 prefill 长度变化是不是单调恶化

- `prefill_B16_I128_O1`：`L3 Miss %=8.82`，`Ave L3 Miss Latency=287.01 ns`，`Remote miss latency %=4.67`，`same-node another CCX %=25.42`
- `prefill_B16_I512_O1`：`L3 Miss %=14.55`，`Ave L3 Miss Latency=177.99 ns`，`Remote miss latency %=5.38`，`same-node another CCX %=11.65`
- `prefill_B16_I1024_O1`：`L3 Miss %=49.43`，`Ave L3 Miss Latency=180.12 ns`，`Remote miss latency %=2.09`，`same-node another CCX %=2.91`

这说明两件事：

1. 序列变长时，prefill 的 L3 miss 率会明显上升。  
2. 但 miss 来源并没有单调向“更远的 NUMA/chiplet”恶化，所以不能简单写成“输入越长，跨 chiplet/远端访存一定越严重”。

### 12.4 为什么 prefill 仍然值得做局部性优化

- scheduler 先根据总可见 KV 计算 `kv_len_per_thread`，再按 `remaining_kv_len` 把 q tile/workitem 塞进不同逻辑线程桶，它的主目标是负载均衡，不是把共享 KV 的 workitem 聚到同一局部域。证据：`csrc/cpu/cpu_attn_impl.hpp:436-589`。
- 运行时 attention task 按 `actual_kv_head_num * effective_thread_num` 展开，所以同一个 `kv_head` 会对应多个不同线程桶的 task。证据：`csrc/cpu/cpu_attn_impl.hpp:1424-1459`。
- 每个 task 真正执行时，都会按 `curr_kv_head_idx` 从全局 `key_cache/value_cache` 取 KV。证据：`csrc/cpu/cpu_attn_impl.hpp:1556-1561`。
- causal prefill 下，`q_tile` 的可见 KV 由 `calcu_kv_tile_pos(...)` 动态裁剪；后面的 q tile 会看到更长的前缀，因此不同 workitem/q tile 的 `kv_range` 高度重叠。证据：`csrc/cpu/cpu_attn_impl.hpp:450-468`、`csrc/cpu/cpu_attn_impl.hpp:1527-1538`、`csrc/cpu/cpu_attn_impl.hpp:1631-1727`。

所以，prefill 虽然不像论文里 GPU workgroup 那样容易定义出很整齐的 ACC，但源码上确实存在一种可优化模式：**同一个 `kv_head` 的多个 task，可能在不同线程上重复读取高度重叠的 KV 前缀。**

### 12.4.1 更严谨的表述

如果要把这段优化逻辑压缩成更适合汇报的表述，我认为下面这版更准确：

- **Prefill 更值得优化。** 因为 prefill 阶段 `q` 很长，会被切成多个 `q_tile/workitem`；不同线程虽然在算不同的 `q` 行，但只要它们落在同一个 `kv_head` 上，就可能重复读取高度重叠的 KV 前缀。
- **Decode 的重复访问形态不同。** decode 开启 `kv_split` 后，不同线程更多是在处理同一个 `q` 对应的不同 KV split，因此真正重复的主要是很小的一行 `q` 和 reduction 相关数据，而不是大块重叠的 KV 前缀。
- **所以 chiplet/CCD 调度优化应优先打在 prefill。** 目标不是单纯减少线程数，而是让同 `kv_head`、高 prefix-overlap 的 prefill task 更稳定地落在同一 L3/CCD 局部域。

对应到源码，可把这 3 句话直接绑定到：

- prefill 的 q tile/workitem 切分：`csrc/cpu/cpu_attn_impl.hpp:450-468`
- 逻辑线程桶与 runtime task 展开：`csrc/cpu/cpu_attn_impl.hpp:436-589`、`csrc/cpu/cpu_attn_impl.hpp:1424-1459`
- 同一 `kv_head` 从全局 KV cache 取数：`csrc/cpu/cpu_attn_impl.hpp:1556-1561`
- decode 的 `kv_split` 与 partial/reduction：`csrc/cpu/cpu_attn_impl.hpp:553-589`、`csrc/cpu/cpu_attn_impl.hpp:1741-1760`

### 12.5 结论

- 有可优化空间，但更像 `workitem/task` 放置问题，即尽量把同 `kv_head`、高 `kv_range` 重叠的 task 固定在同一 CCD/CCX/L3 域，而不是直接把问题归因为远端 DRAM。
- 如果借 `/home/zjj/chiplet/papers/GPU_Attention_NUMA_Optimization.pdf` 的思路，CPU prefill 更适合退化成“按 `kv_head + prefix overlap` 聚类并绑到 chiplet/CCD”，不适合直接照搬 GPU 的 workgroup/ACC 映射。
- 仅凭当前数据，还不足以证明收益一定大，因为 `Remote DRAM Reads %` 没有拉开，`Remote miss latency %` 也不是主导项。

### 12.6 下一步实验建议

1. 固定 `TP`、`NPS`、OMP 线程数和 batch，单独比较“跨 CCD 分散绑核”与“按 CCD 聚拢绑核”的 prefill `B16/I1024/O1`。  
2. 在同一批次同时采 `CPI`、`Remote DRAM Reads %`、`Total Mem Bw` 和 `hotspots/IBS`，避免只靠单份 PCM 下结论。  
3. 如果后续改 scheduler，优先验证“同 `kv_head` 的线程桶稳定落在同一 L3/CCD”是否能降低 `same-node another CCX %`。  
