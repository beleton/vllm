# CPU attention acc-local-l3 新 kernel 实现说明

## 结论

`acc-local-l3` 没有重写 attention 数学 kernel。当前实现仍复用 legacy CPU attention 的 `AttentionImpl<ISA, scalar_t, head_dim>`、workitem 生成逻辑、tile 计算、softmax、partial output 和 split-KV reduction。

新增内容集中在三处：

- 在 legacy metadata 前增加 `cpu_attention_acc_locality::AttentionMetadata` 外层头。
- 按 `ThreadLocalityManager` 生成 `subgroup`，`subgroup` 对应运行时可见的线程局部性分组。
- 对每个 `kv_head` 指定起始 `subgroup` 和 `group_span` 覆盖范围，只允许覆盖范围内的线程领取该 `kv_head` 的 attention / reduction 任务。

当前 runtime 的任务领取方式是 `local-dynamic`：每个 `kv_head` 有独立的 attention 原子计数器和 reduction 原子计数器。覆盖到同一个 `kv_head` 的多个 `subgroup` 共享这两个计数器，动态领取 legacy logical slot 和 reduction item。`group_span>1` 不会让多个 `subgroup` 重复执行同一批 legacy slot。

`attention_task_num` 和 `reduction_task_num` 当前按 legacy 总任务数统计：

- `attention_task_num = actual_kv_head_num * legacy_effective_thread_num`
- `legacy_attention_task_num = actual_kv_head_num * legacy_effective_thread_num`
- `reduction_task_num = actual_kv_head_num * reduction_item_num`

因此，当前 `acc-local-l3` 的差异不是减少任务总数，而是限制每个 `kv_head` 的可领取线程范围。

## Python 侧入口

入口文件是 `vllm/v1/attention/backends/cpu_attn.py`。

`_resolve_cpu_attn_locality_mode()` 只接受两种模式：

- `balanced`
- `acc-local-l3`

不支持的 `VLLM_CPU_ATTN_LOCALITY_MODE` 会回退到 `balanced`。

`_resolve_cpu_attn_locality_group_span()` 从 `VLLM_CPU_ATTN_LOCALITY_GROUP_SPAN` 读取 `group_span`，默认值为 `1`。非法值或小于 `1` 的值会回退到 `1`。

`_get_cpu_attn_ops()` 按 mode 选择 op：

- `balanced` 使用 `ops.cpu_attn_get_scheduler_metadata` 和 `ops.cpu_attention_with_kv_cache`
- `acc-local-l3` 使用 `ops.cpu_attn_get_scheduler_metadata_acc_locality` 和 `ops.cpu_attention_with_kv_cache_acc_locality`

`CPUAttentionMetadataBuilder.build()` 只在 `locality_mode == "acc-local-l3"` 时把 `group_span` 传给 scheduler op。`CPUAttentionBackendImpl.forward()` 根据 `attn_metadata.locality_mode` 再次选择执行 op。

## C++ 入口

`csrc/cpu/torch_bindings.cpp` 导出三组 acc-locality 接口：

- `get_scheduler_metadata_acc_locality(...)`
- `cpu_attention_with_kv_cache_acc_locality(...)`
- `inspect_cpu_attn_acc_locality_metadata(...)`

`vllm/_custom_ops.py` 提供 Python wrapper：

- `cpu_attn_get_scheduler_metadata_acc_locality(...)`
- `cpu_attention_with_kv_cache_acc_locality(...)`

`csrc/cpu/cpu_attn_acc_locality.cpp` 负责入口参数检查、`dtype / head_dim / isa` 模板分发、`AttentionInput` 组装和 mainloop 调用。真正的 acc-locality metadata 与 runtime 逻辑在 `csrc/cpu/cpu_attn_acc_locality_impl.hpp`。

## metadata 构造

### 模板分发

`get_scheduler_metadata_acc_locality(...)` 先把 `isa_hint` 解析成 `cpu_attention::ISA`，再按 `dtype / head_dim / isa` 分发到具体 `attn_impl`，读取 `attn_impl::MaxQHeadNumPerIteration` 后调用 `cpu_attention_acc_locality::build_scheduler_metadata(...)`。

`head_dim` 支持范围与 legacy 入口一致：

- `32 / 64 / 80 / 96 / 112 / 128 / 160 / 192 / 224 / 256`

### 外层 metadata

外层结构是 `cpu_attention_acc_locality::AttentionMetadata`。关键字段包括：

- `magic` / `version`：识别 acc-locality metadata。
- `thread_num`：构造 metadata 时的 OpenMP 最大线程数。
- `subgroup_num`：线程局部性分组数量。
- `group_span`：单个 `kv_head` 覆盖的连续 `subgroup` 数。
- `legacy_thread_num`：嵌入的 legacy metadata 线程数。
- `legacy_effective_thread_num`：legacy scheduler 中有 workitem 的 logical slot 数。
- `actual_kv_head_num`：runtime 外层调度使用的 head 数。
- `reduction_item_num`：legacy reduction item 数量。
- `attention_task_num` / `legacy_attention_task_num` / `reduction_task_num`：当前任务统计值。
- `max_subgroup_thread_num`：最大 subgroup 线程数。
- `subgroup_thread_num[]`：每个 subgroup 的线程数。
- `thread_to_group_id[]`：OpenMP thread 到 subgroup 的映射。
- `thread_to_local_offset[]`：线程在 subgroup 内的局部序号。
- `kv_head_to_subgroup[]`：每个 `kv_head` 的起始 subgroup。
- `legacy_metadata_offset` / `legacy_metadata_size`：嵌入 legacy metadata blob 的位置和大小。

`actual_kv_head_num` 复用 legacy/GQA 判定：

```cpp
q_heads_per_kv = num_heads_q / num_heads_kv;
use_gqa = (max_num_q_per_iter % q_heads_per_kv == 0);
actual_kv_head_num = use_gqa ? num_heads_kv : num_heads_q;
```

### legacy metadata 嵌入

`build_scheduler_metadata(...)` 会先调用全局 legacy `::get_scheduler_metadata(...)`，得到完整的 legacy metadata tensor。

随后新建一个 CPU `int8` tensor：

- 开头放 `cpu_attention_acc_locality::AttentionMetadata`。
- legacy metadata blob 放在 64B 对齐后的 offset。
- 通过 `memcpy` 把 legacy metadata 整块复制进去。

legacy metadata 内部包含裸指针：

- `workitem_groups_ptr`
- `reduction_items_ptr`

复制后代码会按新地址重新绑定这两个指针，并调用 `legacy_metadata->reset_counter()`。当前 acc-locality runtime 不使用 legacy 的全局 counter 做任务分发，但仍保留并修复完整 legacy payload。

### subgroup 映射

`init_thread_locality_mapping(...)` 从 `cpu_utils::ThreadLocalityManager::get_thread_locality_manager()->get_groups()` 读取线程局部性分组。

每个非空 group 会生成一个 `subgroup`，并写入：

- `subgroup_thread_num[subgroup_id]`
- `thread_to_group_id[thread_id]`
- `thread_to_local_offset[thread_id]`
- `max_subgroup_thread_num`

如果 group 为空、无效，或不能覆盖全部 OpenMP 线程，则回退到默认映射：

- `subgroup_num = 1`
- 所有线程属于 subgroup 0
- `thread_to_local_offset[thread_id] = thread_id`

### group_span 与 kv_head 覆盖范围

`group_span` 会被 `normalize_group_span(group_span, subgroup_num)` 归一化到 `[1, subgroup_num]`。

每个 `kv_head` 的起始 subgroup 为：

```cpp
start_subgroup = (kv_head_idx * group_span) % subgroup_num;
kv_head_to_subgroup[kv_head_idx] = start_subgroup;
```

`kv_head_covers_subgroup(...)` 通过环形距离判断某个 subgroup 是否被该 `kv_head` 覆盖：

```cpp
subgroup_delta = (subgroup_id - start_subgroup + subgroup_num) % subgroup_num;
covered = subgroup_delta < group_span;
```

`group_span=1` 时，一个 `kv_head` 覆盖一个 subgroup。`group_span>1` 时，一个 `kv_head` 覆盖从 `start_subgroup` 开始的多个连续 subgroup，超过 `subgroup_num` 时按取模回绕。

## runtime 主循环

### 入口分发

`cpu_attention_with_kv_cache_acc_locality(...)` 会：

- 检查 query / KV cache 维度。
- 把 `scheduler_metadata.data_ptr()` 解释为 `cpu_attention_acc_locality::AttentionMetadata*`。
- 校验 `metadata->magic == kAccLocalityMetadataMagic`。
- 组装 `cpu_attention_acc_locality::AttentionInput`。
- 按 `query.scalar_type()`、`query.size(2)` 和 `metadata->legacy_metadata()->isa` 分发到具体 `attn_impl`。
- 实例化 `cpu_attention_acc_locality::AttentionMainLoop<attn_impl>` 并调用 `mainloop(&input)`。

### 与 legacy 共享的执行逻辑

`cpu_attention_acc_locality::AttentionMainLoop<attention_impl_t>` 继承自 `cpu_attention::AttentionMainLoop<attention_impl_t>`。

每个 OpenMP 线程进入并行区后，仍执行和 legacy 基本一致的公共初始化：

- 计算 `q_heads_per_kv`、`use_gqa`、`actual_kv_head_num`、`actual_q_heads_per_kv`。
- 初始化 `AttentionScratchPad`。
- 在 split-KV 存在时清空 reduction flag。
- 通过 `cpu_utils::get_available_l2_size()` 和 `AttentionScheduler::calcu_default_tile_size(...)` 计算默认 tile 大小。
- 读取 legacy `workitem_groups_ptr`、`cu_workitem_num_per_thread`、`reduction_items_ptr`、`effective_thread_num`、`reduction_item_num`。

attention 计算主体仍调用：

- `attn_impl.copy_q_heads_tile(...)`
- `attn_impl.template execute_attention<Attention>(...)`
- 基类 `final_output(...)`
- 基类 `partial_output(...)`

reduction 阶段仍调用基类：

- `reduce_splits(...)`
- `final_output(...)`

### 新增的 local-dynamic 调度

进入 `operator()` 后，acc-locality runtime 会创建两组本地计数器：

```cpp
std::vector<std::atomic<int32_t>> attention_task_counters(actual_kv_head_num);
std::vector<std::atomic<int32_t>> reduction_task_counters(actual_kv_head_num);
```

这两组计数器每个 `kv_head` 各一个，初始值为 `0`。

每个 OpenMP 线程读取自己的 subgroup：

```cpp
subgroup_id = metadata.thread_to_group_id[thread_id];
subgroup_thread_num = metadata.subgroup_thread_num[subgroup_id];
```

非法 subgroup 或空 subgroup 直接跳过。

attention 阶段执行流程：

```cpp
for each kv_head_idx:
  if (!kv_head_covers_subgroup(metadata, kv_head_idx, subgroup_id)):
    continue
  for (;;) {
    legacy_thread_offset = attention_task_counters[kv_head_idx].fetch_add(1);
    if (legacy_thread_offset >= effective_thread_num):
      break
    run_attention_for_legacy_thread(kv_head_idx, legacy_thread_offset);
  }
```

reduction 阶段执行流程：

```cpp
for each kv_head_idx:
  if (!kv_head_covers_subgroup(metadata, kv_head_idx, subgroup_id)):
    continue
  for (;;) {
    item_offset = reduction_task_counters[kv_head_idx].fetch_add(1);
    if (item_offset >= reduction_item_num):
      break
    run_reduction_for_item(kv_head_idx, item_offset);
  }
```

因此，当前实现不是按 `local_offset` 做静态条带式分片。`thread_to_local_offset[]` 仍保存在 metadata 中，但当前 runtime 的 attention / reduction 任务领取由每个 `kv_head` 的原子计数器决定。

覆盖到同一个 `kv_head` 的所有线程共享同一组计数器。`group_span>1` 时，多个 covered subgroup 组成联合线程池，共同领取该 `kv_head` 的 legacy logical slot 和 reduction item；同一个 `legacy_thread_offset` 或 `item_offset` 只会被领取一次。

### legacy slot 的使用方式

`run_attention_for_legacy_thread(kv_head_idx, legacy_thread_offset)` 用 legacy prefix sum 找到该 logical slot 对应的 workitem 范围：

```cpp
curr_workitem_groups =
    workitem_groups + cu_workitem_num_per_thread[legacy_thread_offset];
curr_workitem_groups_num =
    cu_workitem_num_per_thread[legacy_thread_offset + 1]
  - cu_workitem_num_per_thread[legacy_thread_offset];
```

后续逻辑逐个执行该 slot 下的 `AttentionWorkItemGroup`。每个 workitem 仍携带：

- `req_id`
- `q_token_id_start`
- `q_token_num`
- `kv_split_pos_start`
- `kv_split_pos_end`
- `split_id`
- `local_split_id`

`acc-local-l3` 没有重排 legacy workitem，也没有改变单个 workitem 内的 tile、mask、softmax 和写回逻辑。

### reduction item 的使用方式

`run_reduction_for_item(kv_head_idx, item_offset)` 直接从 legacy reduction item 数组读取：

```cpp
curr_workitem_group = reduction_items + item_offset;
```

随后按 `kv_head_idx` 和 `actual_q_heads_per_kv` 计算输出 head 范围，调用基类 `reduce_splits(...)` 和 `final_output(...)`。`acc-local-l3` 对 reduction 的限制同样只体现在“哪些线程可以领取哪个 `kv_head` 的 item”。

## 与 balanced 的差异

`balanced` runtime 使用 legacy metadata 内的单个全局 atomic counter：

```cpp
task_idx = metadata.acquire_counter();
```

全局任务空间为：

- attention：`actual_kv_head_num * effective_thread_num`
- reduction：`actual_kv_head_num * reduction_item_num`

所有 OpenMP 线程都从同一个全局 counter 抢任务，`task_idx` 再映射为 `kv_head_idx + thread_offset` 或 `kv_head_idx + item_offset`。

`acc-local-l3` runtime 不使用这个全局 counter。它先按 `kv_head_covers_subgroup(...)` 限定可领取线程集合，再由 covered 线程通过该 `kv_head` 自己的计数器领取任务。

两者的核心差异：

- `balanced`：全线程池全局动态抢任务。
- `acc-local-l3`：先按 `kv_head -> subgroup span` 限制线程范围，再在该局部范围内动态抢任务。

两者共享：

- legacy scheduler 生成的 workitem / reduction item。
- `AttentionImpl<ISA, scalar_t, head_dim>` 数学实现。
- tile 计算、softmax、partial output、split-KV reduction 的核心代码。

## group_span 当前语义

`group_span` 只改变某个 `kv_head` 覆盖的 subgroup 数量。

- `group_span=1`：每个 `kv_head` 只允许一个 subgroup 内的线程领取任务。
- `group_span>1`：每个 `kv_head` 允许多个连续 subgroup 内的线程共同领取任务。

当前实现已经用每个 `kv_head` 的共享计数器保证唯一领取。`group_span>1` 不是把同一批 legacy slot 分别复制给多个 subgroup 执行。

`build_runtime_summary(...)` 中的 `legacy_slot_pool=[...]` 和 `reduction_item_pool=[...]` 表示该 `kv_head` 的完整可领取池。实际由哪个线程领取，需要看 runtime trace 中的 `thread_id`、`subgroup_id`、`kv_head_idx`、`legacy_thread_offset`。

## 调试接口

metadata 可通过 `torch.ops._C_utils.inspect_cpu_attn_acc_locality_metadata(...)` 查看。JSON 输出包含：

- `thread_num`
- `subgroup_num`
- `group_span`
- `legacy_thread_num`
- `legacy_effective_thread_num`
- `actual_kv_head_num`
- `attention_task_num`
- `legacy_attention_task_num`
- `reduction_task_num`
- `subgroup_thread_num`
- `kv_head_to_subgroup`

runtime summary 由 `cpu_attention::should_log_runtime_summary()` 控制：

- 如果设置了 `VLLM_CPU_ATTN_DEBUG`，按它判断是否打印。
- 如果未设置 `VLLM_CPU_ATTN_DEBUG`，再读取 `VLLM_CPU_ATTN_ACC_LOCALITY_DEBUG`。

summary 中会打印 `scheduler_mode=local-dynamic`、每个 `kv_head` 覆盖的 subgroup、每个 subgroup 的线程列表和可见 `kv_head`。

runtime trace 由 `VLLM_CPU_ATTN_TRACE` 控制，可用 `VLLM_CPU_ATTN_TRACE_RANK` 给输出加 rank 前缀。`acc-local-l3` trace 行包含：

- `mode=acc-local-l3`
- `thread_id`
- `subgroup_id`
- `legacy_thread_offset`
- `kv_head_idx`
- `workitem_group_idx`
- `req_id`
- `q_token_id_start`
- `q_token_num`
- `kv_split_pos_start`
- `kv_split_pos_end`
- `split_id`
- `local_split_id`

## 当前实现边界

已实现：

- 单独的 `acc-local-l3` scheduler op 与 attention op。
- 外层 acc-locality metadata header。
- legacy metadata blob 嵌入与裸指针修复。
- 基于 `ThreadLocalityManager` 的 subgroup 映射，异常时回退为单 subgroup。
- `kv_head -> start_subgroup` 和 `group_span` 覆盖范围。
- 每个 `kv_head` 的 local-dynamic attention / reduction 计数器。
- 多 subgroup 覆盖同一 `kv_head` 时的唯一任务领取。
- metadata inspect、runtime summary、runtime trace。

未实现：

- 根据 workload 自动选择 `balanced` 或 `acc-local-l3`。
- 更复杂的 `kv_head -> subgroup` 映射策略。
- 独立于 legacy scheduler 的 workitem 生成算法。
- 针对 L2/L3 容量重新设计的 tile 策略。

## 一句话概括

`acc-local-l3` 保留 legacy CPU attention 的 workitem、reduction item 和数学 kernel，在外层增加 `L3/CCX subgroup` 约束；每个 `kv_head` 只允许覆盖范围内的线程通过本 `kv_head` 的局部动态计数器领取任务，从而把共享 `K/V` 工作集的执行范围限制在指定的局部线程子池内。
