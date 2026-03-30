# 08. CPU attention 计算切分、数据放置与 Chiplet 优化借鉴

> 更新时间：2026-03-24 18:49 +0800  
> 范围：`vLLM v1 CPU backend`；重点看 `decoder attention + paged KV cache + TP + NUMA + OpenMP`。  
> 对照论文：`info/papers/GPU_Attention_NUMA_Optimization.pdf`。  
> 当前机器：`2 x AMD EPYC 9745 128-Core Processor`。  
> 证据类型：静态源码、论文 PDF、已有日志 `test_results/qwen3_tp2_init_2026-03-18_2338.log`。  
> 假设：除明确绑定日志/实验结果的条目外，本文所有“性能收益/瓶颈”判断都只是源码推断，不是已跑实验结论。

## 术语速记

- `paged KV cache`：把 KV cache 按固定大小的 page/block 管理；单个 request 的历史 KV 在逻辑上连续，但物理上通常要靠 `block_table` 跳转。
- `GQA`：grouped-query attention，多个 query heads 共享同一组 KV heads。
- `SDPA`：scaled dot-product attention；文中 `sdpa_prefill` 指把 prefill token 单独送去通用 SDPA 实现，而不是走 CPU 自定义 attention kernel。
- `DCP/PCP`：decode context parallel / prefill context parallel；开启后 `slot_mapping` 不再是简单连续布局，而会按 context-parallel rank 交错。
- `workitem`：CPU attention kernel 内部的调度任务，可粗看成“某个 request 的一段 q token + 一个 `kv_head` + 一段 KV 范围”。
- `scratchpad`：kernel 执行时用的临时缓冲区；不是持久 KV cache。
- `ACC/XCD/swizzle`：论文术语；ACC 是共享同一组 K/V 的计算簇，XCD 是 MI300X 上的局部执行/缓存域，swizzle 是把 workgroup 重新映射到这些硬件域。

## 结论

### 事实
- 对当前 x86 CPU backend，vLLM 的 attention 不是“prefill 一套 kernel、decode 一套 kernel”。`CPUModelRunner` 直接继承 `GPUModelRunner`，x86 默认 `use_sdpa_prefill=False`，所以 prefill / decode / mixed batch 通常都进入同一个 `cpu_attention_with_kv_cache` 主算子；区别主要来自 `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping`、`scheduler_metadata` 的内容。证据：`vllm/v1/worker/cpu_model_runner.py`、`vllm/v1/attention/backends/cpu_attn.py`、`vllm/v1/worker/gpu_model_runner.py`。
- CPU attention 的最小计算单元更接近 `(req_id, q_token slice, kv span/split, kv_head_idx)`，不是“一个 request”也不是“一个 token”。`cpu_attn_get_scheduler_metadata` 会先生成 `AttentionWorkItemGroup` 和 `ReductionWorkItemGroup`，运行时再由 OpenMP 线程消费。证据：`csrc/cpu/cpu_attn.cpp`、`csrc/cpu/cpu_attn_impl.hpp`。
- 历史 KV 的读取 locality 主要由 `block_table -> physical_block_idx -> key_cache/value_cache` 决定；`slot_mapping` 只负责把“本轮新生成的 K/V”写到 KV cache。也就是说，分析 decode 的 `Remote DRAM Reads %` 时，重点应盯 `block_table` 和 KV 页放置，而不是 `slot_mapping`。证据：`vllm/v1/worker/block_table.py`、`csrc/cpu/cpu_attn.cpp`、`csrc/cpu/cpu_attn_impl.hpp`。
- 当前 CPU kernel 已经有“部分 head-first”特征：运行时 task 先按 `kv_head_idx` 展开，GQA 场景下一组 `q heads` 会共享同一个 `kv head`；但它还不是论文那种 NUMA-aware head-first，因为 1) KV 页没有按 head/NUMA 显式分配；2) 实际线程通过全局 atomic 动态抢 task，不是稳定的 `chiplet/thread cluster -> head group` 映射。证据：`csrc/cpu/cpu_attn_impl.hpp`。
- 数据放置不是“建议”，而是 CPU backend 真实会做的事。`CPUWorker.init_device()` 会在 worker 初始化早期调用 `init_cpu_threads_env()`：单 node 用 `numa_set_membind`，多 node 用 `numa_set_interleave_mask`，还会迁移已有页并对每个 OMP thread 调 `sched_setaffinity`。证据：`vllm/v1/worker/cpu_worker.py`、`csrc/cpu/utils.cpp`。
- TP 会先把 attention head 切到 rank 级：每个 TP rank 只保留自己的 `num_heads / num_kv_heads`，本 rank 只算本地 head 对应的 attention；attention 之后 `o_proj` 才做 `all_reduce`。当前这台机器、当前日志里 TP 数据面实际走 `gloo`，不是 CPU SHM collective。证据：`vllm/model_executor/models/qwen3_moe.py`、`vllm/model_executor/layers/linear.py`、`vllm/distributed/device_communicators/cpu_communicator.py`、`test_results/qwen3_tp2_init_2026-03-18_2338.log`。

### 假设
- 对 Chiplet CPU，最值得先优化的不是 QK/PV 数学公式，而是“KV 页放置 + `kv_head`/thread locality + mixed batch 组织”。原因是当前 CPU attention 的读路径天然带有 `block_table` 间接寻址、分页 KV、长上下文流式读取，以及 NUMA 内存策略。证据链：`vllm/v1/worker/block_table.py`、`csrc/cpu/cpu_attn_impl.hpp`、`csrc/cpu/utils.cpp`。
- decode 比 prefill 更像论文思路的主要受益者，因为 CPU kernel 的 split-KV 只在 `q_tile_token_num <= 1` 的尾 tile 上启用；这使单 token query 或尾部单 token tile 更容易被拆成“多个线程/多个 split 共享同一组 KV 页”的形态。证据：`csrc/cpu/cpu_attn_impl.hpp`。
- 如果后续优化真的有效，最先下降的指标更可能是 `Remote DRAM Reads %` 和 `Ave L3 Miss Latency`，随后才是 `CPI`；`Total Mem Bw` 可能下降，也可能保持接近但做了更多有效工作。这一条目前只是待验证假设。

### 建议
- 先把 attention 看成“`request flatten + paged KV read + kv_head/workitem split + reduction`”问题，不要先把它看成纯 GEMM 问题。
- 若要借鉴论文，优先研究“共享同一组 K/V 的 decode workitem 是否能稳定地待在同一 NUMA node / 同一 TP rank / 同一线程簇”，而不是直接移植 GPU 的 swizzle 代码。
- 后续实验优先级建议是：`decode-only 微基准 -> mixed batch 微基准 -> TP+NUMA 端到端复验`。

## 1. 从 scheduler 到 CPU attention kernel 的完整链路

```text
Scheduler.schedule()
  -> SchedulerOutput.num_scheduled_tokens
  -> CPUModelRunner._update_states(...)
  -> GPUModelRunner._prepare_inputs(...)
  -> GPUModelRunner._build_attention_metadata(...)
  -> CPUAttentionMetadataBuilder.build(...)
       -> ops.cpu_attn_get_scheduler_metadata(...)
  -> CPUAttentionBackendImpl.forward(...)
       -> ops.cpu_attn_reshape_and_cache(...)
       -> ops.cpu_attention_with_kv_cache(...)
```

### 1.1 scheduler 并不区分“prefill phase / decode phase”
- `schedule()` 注释的原意是：调度器不给 request 打上独立的 “prefill phase / decode phase” 标签，它只维护两个计数器。`num_computed_tokens` 表示这个 request 已经算完了多少 token；`num_tokens_with_spec` 表示这个 request 目前总共需要覆盖到多少 token。证据：`vllm/v1/core/sched/scheduler.py:313-323`。
- `num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)`。也就是 `prompt token + 已确认的输出 token + speculative draft token`。如果没开 speculative decoding，`spec_token_ids` 通常为空，那么它就基本等于“这个 request 当前总共有多少 token 需要被 attention 覆盖”。证据：`vllm/v1/core/sched/scheduler.py:316-323`、`vllm/v1/request.py:220-222`。
- 直观地说，调度器每轮只是在补差值 `num_tokens_with_spec - num_computed_tokens`。例如某个 request 有 100 个 prompt token、8 个已确认输出 token、4 个 speculative token，那么 `num_tokens_with_spec=112`；若当前 `num_computed_tokens=108`，调度器只会关心“还差 4 个 token 要算完”，不会额外维护“现在已经切到 decode phase”这样的状态。
- 所以“prefill / decode”在 vLLM 里更像是 **某个 request 本轮补了多少 query token、这些 token 落在序列的哪个区间**，而不是调度器切换到了另一套 attention 实现。
- 这也解释了为什么同一批里可以天然出现 mixed batch：有的 request `q_len=1`，有的 `q_len>>1`，但它们都来自同一个 `SchedulerOutput.num_scheduled_tokens`。

### 1.2 CPUModelRunner 复用 GPUModelRunner 的 batch/metadata 准备逻辑
- `CPUModelRunner` 只是把原本 `CpuGpuBuffer` 里的 `gpu` 指针替换成 `cpu` 指针，本质没有重写 `_update_states()`、`_prepare_inputs()`、`_build_attention_metadata()` 这一整套逻辑。证据：`vllm/v1/worker/cpu_model_runner.py:18-53`。
- 因此 CPU attention 的上游输入整理，仍然要看 `vllm/v1/worker/gpu_model_runner.py`。

### 1.3 `_prepare_inputs()` 做了什么
- 它先把 request 维的调度结果摊平成 token 维连续视图：
  - `req_indices = repeat(request_index, num_scheduled_tokens)`
  - `positions = num_computed_tokens + token_arange_within_req`
  - `token_indices = positions + req_idx * max_model_len`
  证据：`vllm/v1/worker/gpu_model_runner.py:1432-1479`。
- 随后它构造 attention 真正依赖的几个核心张量：
  - `slot_mapping`：本轮每个 token 写 KV cache 的物理槽位。
  - `query_start_loc`：flatten 后每个 request 的 query token 起止偏移。
  - `seq_lens`：本轮执行后每个 request 的总可见 KV 长度。证据：`vllm/v1/worker/gpu_model_runner.py:1527-1545`。
- 这一步只是在整理数据，不做 attention 计算。

### 1.4 CPU metadata builder 做了什么
- `CPUAttentionMetadataBuilder.build()` 接收 `CommonAttentionMetadata`，然后调用 `ops.cpu_attn_get_scheduler_metadata(...)` 生成 kernel 内部调度元数据。证据：`vllm/v1/attention/backends/cpu_attn.py:143-207`。
- 当前 x86 / ARM 默认 `_CPU_ARCH_PREFER_MIXED_BATCH`，因此 `reorder_batch_threshold=None`、`use_sdpa_prefill=False`；也就是当前 AMD EPYC 场景下，默认不会把 prefill 单独送到 SDPA。证据：`vllm/v1/attention/backends/cpu_attn.py:28`、`vllm/v1/attention/backends/cpu_attn.py:115-124`。

### 1.4.1 如果只看 `scheduler_metadata`，代码调用顺序是什么

```text
GPUModelRunner._build_attention_metadata(...)
  -> CPUAttentionMetadataBuilder.build(...)
       -> ops.cpu_attn_get_scheduler_metadata(...)
            -> csrc/cpu/cpu_attn.cpp::get_scheduler_metadata(...)
                 -> AttentionScheduler::schedule(...)
                      -> 读取 query_start_loc / seq_lens
                      -> 计算 q_head_per_kv / default_tile_token_num / kv_len_per_thread
                      -> 生成 AttentionWorkItemGroup[]
                      -> 生成 ReductionWorkItemGroup[]
                      -> 回填 AttentionMetadata
                      -> 返回 scheduler_metadata tensor
```

- 所以后文凡是出现“调度器先读 `query_start_loc` 和 `seq_lens`”这类表述，默认指的是 `AttentionScheduler::schedule(...)` 这一步；它发生在 `CPUAttentionMetadataBuilder.build()` 期间，而不是在 `AttentionMainLoop` 里。证据：`vllm/v1/attention/backends/cpu_attn.py:143-207`、`csrc/cpu/cpu_attn.cpp:104-140`、`csrc/cpu/cpu_attn_impl.hpp:382-662`。
- `schedule(...)` 生成完的结果不会立刻做 attention 计算；它只是把“这一轮该怎么切 task、task 之间怎么 reduction、每线程需要多大 scratchpad”打包进 `scheduler_metadata`，留给后面的 `cpu_attention_with_kv_cache(...)` 消费。证据：`csrc/cpu/cpu_attn.cpp:229-267`。

### 1.5 CPUAttentionBackendImpl.forward 的两段主工作
- 第 1 段：`cpu_attn_reshape_and_cache(...)`，把本轮新生成的 K/V 写入 paged KV cache。证据：`vllm/v1/attention/backends/cpu_attn.py:321-333`。
- 第 2 段：`cpu_attention_with_kv_cache(...)`，读取历史 KV + 当前 query，完成 QK / mask / softmax / PV / reduction / final output。证据：`vllm/v1/attention/backends/cpu_attn.py:348-364`。

### 1.5.0 如果只看 attention 计算主路径，代码调用顺序是什么

```text
CPUAttentionBackendImpl.forward(...)
  -> ops.cpu_attn_reshape_and_cache(...)
       -> csrc/cpu/cpu_attn.cpp::cpu_attn_reshape_and_cache(...)
            -> attn_impl::reshape_and_cache(...)
  -> ops.cpu_attention_with_kv_cache(...)
       -> csrc/cpu/cpu_attn.cpp::cpu_attention_with_kv_cache(...)
            -> AttentionMainLoop<attn_impl>::operator()(...)
                 -> 初始化 scratchpad / 读取 scheduler_metadata
                 -> metadata.acquire_counter() 动态抢 attention task 或 reduction task
                 -> attention task:
                      -> copy_q_heads_tile(...)
                      -> execute_attention<Attention>(...)
                           -> QK gemm
                           -> apply_softcap(...)
                           -> apply_alibi_slopes(...)
                           -> apply_mask(...)
                           -> apply_softmax(...)
                           -> PV gemm
                      -> final_output(...) 或 partial_output(...)
                 -> reduction task:
                      -> reduce_splits(...)
                      -> final_output(...)
```

- 这条调用链里，`cpu_attn_reshape_and_cache(...)` 和 `cpu_attention_with_kv_cache(...)` 是前后串行的；前者先把本轮新 K/V 写进 paged KV cache，后者再把新旧 KV 一起当作历史 cache 读取。证据：`vllm/v1/attention/backends/cpu_attn.py:321-364`。
- 因而后文凡是出现 `AttentionMainLoop::operator()`、`copy_q_heads_tile()`、`execute_attention()`、`reduce_splits()` 等函数，如果没有额外说明，默认都处在这条主计算链里，而不是 metadata 生成链里。证据：`csrc/cpu/cpu_attn.cpp:229-267`、`csrc/cpu/cpu_attn_impl.hpp:1339-1971`。

### 1.5.1 `cpu_attn_reshape_and_cache` 的入口其实很薄
- `CPUAttentionBackendImpl.forward()` 先把 `kv_cache` 拆成 `key_cache, value_cache = kv_cache.unbind(0)`，随后在 decoder / cross-attention 路径里，只要本轮 `key/value` 不为空，就先调用 `ops.cpu_attn_reshape_and_cache(...)`，再进入后面的 attention 主算子。证据：`vllm/v1/attention/backends/cpu_attn.py:316-333`。
- Python 层 `cpu_attn_reshape_and_cache()` 本身只是 `torch.ops._C.cpu_attn_reshape_and_cache(...)` 的薄封装，没有额外逻辑。证据：`vllm/_custom_ops.py:2988-3003`。
- C++ 入口 `cpu_attn_reshape_and_cache(...)` 也主要只做三件事：1) 校验 `key/value/key_cache/value_cache` 的维度与 stride；2) 解析 `token_num/head_num/head_dim/block_size`；3) 依据 `dtype + head_dim + isa` 分发到 `attn_impl::reshape_and_cache(...)`。证据：`csrc/cpu/cpu_attn.cpp:143-201`。
- 重要点：这个函数完全不看 `query_start_loc`、`seq_lens`、`block_table`、`scheduler_metadata`。它不是“attention 调度器”的一部分，而是一个纯写路径 pack kernel；它唯一的地址输入就是 `slot_mapping`。证据：`csrc/cpu/cpu_attn.cpp:143-201`。

### 1.5.2 `cpu_attn_reshape_and_cache` 实际写了什么
- 真正的实现是 `attn_impl::reshape_and_cache(...)`。它在 `token_idx` 和 `head_idx` 两个维度上做 `#pragma omp parallel for collapse(2)`，所以当前轮每个 token、每个 KV head 的写入都可以并行展开。证据：`csrc/cpu/cpu_attn_vec.hpp:197-208`。
- 每个 token 先取 `pos = slot_mapping[token_idx]`。若 `pos < 0` 就直接跳过；这对应“这个 token 不归当前 rank / 当前本地 cache 写”的情况。证据：`csrc/cpu/cpu_attn_vec.hpp:209-213`。
- 然后把这个一维 slot 反解成 `block_idx = pos / block_size` 和 `block_offset = pos % block_size`，也就是“落到哪个物理 KV block、该 block 内第几个 token 槽位”。证据：`csrc/cpu/cpu_attn_vec.hpp:215-216`。
- K 的写入不是原样 memcpy，而是列主序。一个block的逻辑形状是`[num_kv_heads, block_size, head_dim]`，在存储一个kv_head的K时，会按照`[head_dim, block_size]`来存储，方便后续计算 `Q @ K^T` 读出 `[head_dim, token_group]`时连续的内存。如果是行主序存储，则读取元素时内存是不连续的。证据：`csrc/cpu/cpu_attn_vec.hpp:217-229`、`csrc/cpu/cpu_attn_impl.hpp:939-960`。

- V 的写入则是行主序：起点是 `block_offset * head_dim`，随后直接 `memcpy(head_dim)`。这对应后续 `P @ V` 想读的 `[token_group, head_dim]` 布局。证据：`csrc/cpu/cpu_attn_vec.hpp:231-240`、`csrc/cpu/cpu_attn_impl.hpp:1053-1073`。
- `slot_mapping` 本身又不是凭空生成的。非 DCP/PCP 下它直接来自 `block_numbers * block_size + block_offsets`，而这里的 `block_numbers` 就取自 request 当前的 `block_table`。所以“本轮新写入的 token”与“后续读路径要访问的 physical block”本来就是同一组页。证据：`vllm/v1/worker/block_table.py:181-191`。

### 1.5.3 写完之后，后续调用怎样把它立刻消费掉
- `CPUAttentionMetadataBuilder.build()` 先基于 `query_start_loc / seq_lens / block_table` 生成 `scheduler_metadata`；其中 `seq_lens = num_computed_tokens + num_scheduled_tokens`，也就是已经把“本轮新增 token”计入本次 attention 的可见 KV 长度。证据：`vllm/v1/attention/backends/cpu_attn.py:177-205`、`vllm/v1/worker/gpu_model_runner.py:1539-1541`。
- 随后 `forward()` 直接把同一个 `key_cache/value_cache` 指针传给 `cpu_attention_with_kv_cache(...)`；中间没有任何额外重排，也没有“当前 token 单独走一条旁路”。证据：`vllm/v1/attention/backends/cpu_attn.py:326-364`。
- C++ 侧 `cpu_attention_with_kv_cache(...)` 只是把这些 tensor 指针塞进 `AttentionInput`，再把 `scheduler_metadata.data_ptr()` 解释成 `AttentionMetadata*`。因此 `reshape_and_cache` 刚写进去的数据，会被同一个 `forward()` 紧接着读到。证据：`csrc/cpu/cpu_attn.cpp:203-240`。
- 进入主循环后，kernel 先按 `req_id / q_token slice / kv_head_idx` 展开 workitem，再取 `curr_block_table = block_table + current_group_idx * block_table_stride`。真正读 K/V 时，会先拿 `physical_block_idx = block_table[block_idx]`，再从 `key_cache/value_cache` 里取对应 block 指针做 `QK` 与 `PV`。证据：`csrc/cpu/cpu_attn_impl.hpp:1451-1570`、`csrc/cpu/cpu_attn_impl.hpp:939-960`、`csrc/cpu/cpu_attn_impl.hpp:1053-1073`。
- 这意味着一个关键事实：对 decoder attention 而言，“历史 KV”与“本轮刚写入的 KV”在读路径上没有本质区别。只要 `slot_mapping` 先把新 token 写进了对应 physical block，后面的 `cpu_attention_with_kv_cache()` 就统一通过 `block_table + seq_lens` 把它们当作 paged KV cache 读取。证据：`vllm/v1/worker/block_table.py:181-191`、`vllm/v1/worker/gpu_model_runner.py:1539-1541`、`csrc/cpu/cpu_attn_impl.hpp:1451-1728`。

### 1.5.4 测试和微基准里也复用了这条最小链路
- `tests/kernels/attention/test_cpu_attn.py` 的最小验证顺序就是：先 `cpu_attn_reshape_and_cache(...)` 把原始 K/V pack 成 CPU cache 布局，再调 `cpu_attn_get_scheduler_metadata(...)`，最后调 `cpu_attention_with_kv_cache(...)`。证据：`tests/kernels/attention/test_cpu_attn.py:252-320`。
- `benchmarks/kernels/cpu/benchmark_cpu_attn.py` 也是同一顺序，因此它很适合专门观察“写 cache 之后，attention kernel 如何读这些页”的纯算子行为。证据：`benchmarks/kernels/cpu/benchmark_cpu_attn.py:118-164`。

### 1.5.5 `cpu_attention_with_kv_cache` 的代码作用到底是什么
- `cpu_attention_with_kv_cache(...)` 是当前 x86 CPU decoder attention 的主计算入口。和前面的 `cpu_attn_reshape_and_cache(...)` 不同，它不是单纯 pack 数据，而是把 `query + key_cache + value_cache + query_start_loc + seq_lens + block_table + scheduler_metadata` 真正送进 attention 主循环，生成最终 `output`。证据：`vllm/v1/attention/backends/cpu_attn.py:348-364`、`csrc/cpu/cpu_attn.cpp:203-267`。
- C++ 入口里，它先把这些 tensor 指针和 stride 组装成 `AttentionInput`，包括 `query/query_start_loc/seq_lens/block_table/key_cache/value_cache/output`，以及 `scale/causal/sliding_window/alibi/softcap`。这一步只是把 Python/Torch tensor 语义转成 kernel 可直接消费的扁平指针结构。证据：`csrc/cpu/cpu_attn.cpp:224-257`。
- 随后它按 `dtype + head_dim + isa` 分发到 `AttentionMainLoop<attn_impl>`。真正的 QK/PV 循环不写在这个入口里，而是在 `AttentionMainLoop` 内完成。证据：`csrc/cpu/cpu_attn.cpp:258-267`。
- `AttentionMainLoop` 启动后，会先校验线程数、为每个线程准备 `scratchpad`，再从 `metadata` 里取出 workitem / reduction item 和前缀和信息。后续通过 `metadata.acquire_counter()` 动态分发 task：前半段是 attention task，后半段是 reduction task。证据：`csrc/cpu/cpu_attn_impl.hpp:1339-1815`。
- 职责边界也很明确：它不负责生成 `scheduler_metadata`，也不负责决定新 K/V 写到哪；它只消费已经准备好的 metadata 和 paged KV cache，完成本轮 attention 计算并写 `output`。证据：`vllm/v1/attention/backends/cpu_attn.py:326-364`、`csrc/cpu/cpu_attn.cpp:224-267`。

### 1.5.6 `cpu_attention_with_kv_cache` 返回后，结果怎么继续往后走
- 在统一 attention 层里，`Attention.forward()` 会先把 `query/key/value` reshape 成 `[num_tokens, num_heads, head_dim]` / `[num_tokens, num_kv_heads, head_dim]`，再通过 `torch.ops.vllm.unified_attention_with_output(...)` 进入 backend。这个统一 op 最终调用的是 `self.impl.forward(...)`；对 CPU backend 来说，就是 `CPUAttentionBackendImpl.forward(...)`，也就是前面包含 `cpu_attn_reshape_and_cache(...)` 和 `cpu_attention_with_kv_cache(...)` 的那条路径。证据：`vllm/attention/layer.py:383-423`、`vllm/attention/layer.py:868-915`、`vllm/v1/attention/backends/cpu_attn.py:265-366`。
- 当 `cpu_attention_with_kv_cache(...)` 把 `output[:num_actual_tokens]` 写满后，`CPUAttentionBackendImpl.forward()` 直接 `return output`；`Attention.forward()` 再把这个三维 tensor `view(-1, hidden_size)` 返回给模型层。证据：`vllm/v1/attention/backends/cpu_attn.py:348-366`、`vllm/attention/layer.py:381-423`。
- 对当前重点模型 `Qwen3MoeAttention`，返回值马上被接到 `attn_output = self.attn(q, k, v)`，随后进入 `output, _ = self.o_proj(attn_output)`。也就是说，`cpu_attention_with_kv_cache` 产出的不是最终 layer 输出，而是 attention context；后面还要过一次 `o_proj`。证据：`vllm/model_executor/models/qwen3_moe.py:333-351`。
- `o_proj` 是 `RowParallelLinear`。在 `RowParallelLinear.forward()` 里，若 `tp_size > 1` 且 `reduce_results=True`，就会对各个 TP rank 的 `output_parallel` 做 `tensor_model_parallel_all_reduce`。这也是为什么说 attention 本身先是 rank 内本地算子，而真正把各 rank attention 结果汇总起来的是 attention 之后的 `o_proj`。证据：`vllm/model_executor/layers/linear.py:1442-1464`、`vllm/model_executor/models/qwen3_moe.py:300-305`。
- 在 `Qwen3MoeDecoderLayer.forward()` 里，`self.self_attn(...)` 返回的结果接着进入 `post_attention_layernorm`，再进 `MLP/MoE`。所以从 decoder layer 视角看，链路可以简化成：

```text
cpu_attn_reshape_and_cache
  -> cpu_attention_with_kv_cache
  -> Attention.forward return
  -> Qwen3MoeAttention.o_proj
  -> TP all_reduce（若 TP>1）
  -> post_attention_layernorm
  -> MLP / MoE
```

- 因而若后续要把 attention kernel 的性能现象和端到端 `CPI / Total Mem Bw / Remote DRAM Reads %` 对齐，必须记住：`cpu_attention_with_kv_cache` 结束后并没有离开 decoder layer，后面至少还有 `o_proj` 和可能的 TP `all_reduce`，它们会继续贡献一部分 stall 与带宽。证据：`vllm/model_executor/models/qwen3_moe.py:406-426`、`vllm/model_executor/layers/linear.py:1442-1464`。

## 2. attention 关键数据结构：它们各自到底表示什么

| 名称 | 来源 | 什么时候用 | 物理含义 |
| --- | --- | --- | --- |
| `block_ids` | scheduler / KV manager | request 状态更新时 | 一个 request 当前已经拿到的 KV page ID 列表 |
| `block_table` | `InputBatch.block_table` | 读历史 KV 时 | `request -> physical block` 映射表 |
| `slot_mapping` | `_prepare_inputs()` | 写新 K/V 时 | `token -> flat KV slot` 映射 |
| `query_start_loc` | `_prepare_inputs()` | attention 读 query 时 | flatten token 视图中每个 request 的 query 边界 |
| `seq_lens` | `_prepare_inputs()` | attention 计算可见长度时 | 每个 request 当前可见的总 KV 长度 |
| `scheduler_metadata` | `cpu_attn_get_scheduler_metadata()` | kernel 内部 | workitem 列表、split 信息、scratchpad 大小、前缀和等 |

### 2.1 `block_table`：历史 KV 的真正索引表
- 在 `_build_attention_metadata()` 里，CPU backend 最终把 `self.input_batch.block_table[kv_cache_gid].get_device_tensor(num_reqs_padded)` 塞进 `CommonAttentionMetadata.block_table_tensor`。证据：`vllm/v1/worker/gpu_model_runner.py:1680-1718`。
- 之后在 kernel 内，QK / PV 读取都会先拿 `curr_block_table = block_table + current_group_idx * block_table_stride`，再按 `block_idx -> physical_block_idx` 去找真实的 KV page。证据：`csrc/cpu/cpu_attn_impl.hpp:1569-1570`、`csrc/cpu/cpu_attn_impl.hpp:941-945`、`csrc/cpu/cpu_attn_impl.hpp:1055-1059`。
- 这意味着：**历史 KV 的 locality，本质取决于物理 block 是怎样被分配、这些 block 落在哪个 NUMA node、以及当前线程在哪个 node 上读它们。**

### 2.2 `slot_mapping`：只管写，不管读
- 非 DCP/PCP 情况下，`slot = block_number * block_size + position % block_size`，就是最普通的“block 内顺序写入”。证据：`vllm/v1/worker/block_table.py:181-191`。
- DCP/PCP 开启时，会变成交错布局；不属于当前 context-parallel rank 的 token 会被写成 `-1`，表示本 rank 不负责把它写进本地 KV cache。证据：`vllm/v1/worker/block_table.py:142-179`。
- `cpu_attn_reshape_and_cache()` 接收到的正是这个 `slot_mapping`，然后把每个 token 的 K/V 写进相应 `block_idx / block_offset`。证据：`csrc/cpu/cpu_attn.cpp:143-201`、`csrc/cpu/cpu_attn_vec.hpp:196-244`。
- 重要结论：`slot_mapping` 影响“本轮写进去的位置”，但 decode 的主要成本通常在“反复读取历史 KV”，因此更该盯 `block_table` 和物理页放置。

### 2.3 `query_start_loc` 和 `seq_lens`
- `query_start_loc` 是 flatten 后 request 边界。例如 `[2,5,3]` 个 query token 会变成前缀和 `[0,2,7,10]`。证据：`vllm/v1/worker/gpu_model_runner.py:1531-1537`。
- `seq_lens = num_computed_tokens + num_scheduled_tokens`，表示本轮 attention 可以看到的 KV 长度。证据：`vllm/v1/worker/gpu_model_runner.py:1539-1545`。
- 这两个数组共同决定：某个 request 本轮是 decode 风格（通常 `q_len=1`）还是 prefill 风格（`q_len>1`）。

### 2.4 `scheduler_metadata`
- `scheduler_metadata` 不是 Python 字典，也不是只放几个计数器的轻量对象；它是 `cpu_attn_get_scheduler_metadata(...)` 返回的一块 `torch.int8` 张量，内存里依次放着：
  - `AttentionMetadata`
  - `AttentionWorkItemGroup[]`
  - `ReductionWorkItemGroup[]`
  证据：`csrc/cpu/cpu_attn.cpp:99-140`、`csrc/cpu/cpu_attn_impl.hpp:600-618`。
- `cpu_attention_with_kv_cache(...)` 运行时不会重新解释这些调度规则，而是直接把 `scheduler_metadata.data_ptr()` 强转成 `AttentionMetadata*` 来消费。证据：`csrc/cpu/cpu_attn.cpp:224-226`。

#### 2.4.1 `scheduler_metadata` 是怎么生成的
- 第 1 步：`CPUAttentionMetadataBuilder.build()` 把 `num_reqs / num_heads / num_kv_heads / head_dim / seq_lens / query_start_loc / sliding_window / isa / enable_kv_split` 传给 `ops.cpu_attn_get_scheduler_metadata(...)`。这一步只依赖长度和调度信息，不直接读 `key_cache/value_cache`。证据：`vllm/v1/attention/backends/cpu_attn.py:177-205`、`csrc/cpu/cpu_attn.cpp:99-140`。
- 第 2 步：`AttentionScheduler::schedule()` 先推导本轮的基础切分参数：
  - `q_head_per_kv` / `use_gqa`
  - `min_split_kv_len`
  - `max_num_q_token_per_iter`
  - `default_tile_size`
  - `default_tile_token_num`
  - `split_kv_q_token_num_threshold`
  证据：`csrc/cpu/cpu_attn_impl.hpp:382-406`。
- 第 3 步：scheduler 先做一次全局统计。它遍历每个 request，用 `query_start_loc` 求 `q_token_num`，用 `seq_lens` 求本轮可见 `seq_len`，再结合 causal/sliding-window 算每个 q 微步长对应的有效 `kv_len`，最后累加成 `total_kv_len`。证据：`csrc/cpu/cpu_attn_impl.hpp:408-435`。
- 第 4 步：把总工作量折算成“每个逻辑线程桶大约承担多少 KV 长度”：
  - `kv_len_per_thread = ceil(total_kv_len / thread_num, alignment) * (num_kv_heads 或 num_q_heads)`
  - 这里乘 `num_kv_heads`（GQA）或 `num_q_heads`（fallback MHA），意思是后续任务空间还会再按 `kv_head` 展开；它不是简单地“每线程分几个 request”。证据：`csrc/cpu/cpu_attn_impl.hpp:436-444`。
- 第 5 步：第二次遍历 request，真正生成 `AttentionWorkItemGroup` / `ReductionWorkItemGroup`。核心逻辑是：
  - 先在 request 内按 `max_num_q_token_per_iter` 向前扫，得到细粒度 q 微步长。证据：`csrc/cpu/cpu_attn_impl.hpp:450-474`。
  - 若当前 `curr_kv_len` 还能塞进当前逻辑线程桶的剩余预算 `remaining_kv_len`，就把这段 q/KV 合并进当前 `curr_workitem`。证据：`csrc/cpu/cpu_attn_impl.hpp:476-503`。
  - 若当前线程桶剩余预算太小，且已经装过 workitem，就切到下一个逻辑线程桶。证据：`csrc/cpu/cpu_attn_impl.hpp:505-523`。
  - 只有“尾部 q tile 且 `q_tile_token_num <= split_kv_q_token_num_threshold`”时才允许 split-KV；否则这段 KV 不会被拆成多个 split。证据：`csrc/cpu/cpu_attn_impl.hpp:525-551`。
  - 一旦真的 split-KV，scheduler 会在第一次 split 时创建一个 `ReductionWorkItemGroup(req_id, token_id, q_tile_token_num, cum_split_num)`，随后为每个 split 生成一个 `AttentionWorkItemGroup`，并给它们分配连续的 `split_id/local_split_id`。证据：`csrc/cpu/cpu_attn_impl.hpp:553-590`。
- 第 6 步：把 workitem 数组和 reduction 数组拷进 metadata tensor，补齐前缀和与 scratchpad 大小：
  - `effective_thread_num`
  - `cu_workitem_num_per_thread`
  - `attention_scratchpad_size_per_thread`
  - `reduction_scratchpad_size_per_kv_head`
  然后按这些大小统一 `realloc` scratchpad。证据：`csrc/cpu/cpu_attn_impl.hpp:600-662`。

#### 2.4.2 `scheduler_metadata` 里每个信息是什么意思
- `workitem_group_num`：attention 主任务总数，也就是 `AttentionWorkItemGroup[]` 的长度。每个元素描述“一段 request-local q token + 一段 kv span/split”。证据：`csrc/cpu/cpu_attn_impl.hpp:20-57`、`csrc/cpu/cpu_attn_impl.hpp:92`。
- `reduction_item_num`：需要后续 reduction 的 q tile 组数，也就是 `ReductionWorkItemGroup[]` 的长度；只有发生 split-KV 时才会非零。证据：`csrc/cpu/cpu_attn_impl.hpp:59-86`、`csrc/cpu/cpu_attn_impl.hpp:93`。
- `reduction_split_num`：这一轮所有 split-KV workitem 的总 split 数；它决定 reduction scratchpad 里 flag / partial output / max / sum 要预留多少槽位。证据：`csrc/cpu/cpu_attn_impl.hpp:94`、`csrc/cpu/cpu_attn_impl.hpp:650-654`。
- `thread_num`：生成 metadata 时看到的 `omp_get_max_threads()`；运行时 `AttentionMainLoop` 会再次校验当前线程数必须一致。证据：`csrc/cpu/cpu_attn_impl.hpp:113`、`csrc/cpu/cpu_attn_impl.hpp:1340-1341`。
- `effective_thread_num`：真正分到非零 workitem 的逻辑线程桶数量；后面的任务编号空间是按它而不是按 `thread_num` 展开。证据：`csrc/cpu/cpu_attn_impl.hpp:620-634`、`csrc/cpu/cpu_attn_impl.hpp:1420-1429`。
- `split_kv_q_token_num_threshold`：允许 split-KV 的 q tile 阈值。当前默认 `enable_kv_split=True` 时它等于 `1`，所以更接近“decode/尾部单 token tile 可 split，长 prefill tile 不可 split”。证据：`csrc/cpu/cpu_attn_impl.hpp:402-403`、`csrc/cpu/cpu_attn_impl.hpp:525-551`。
- `attention_scratchpad_size_per_thread`：每个真实 OMP 线程的 thread-local scratchpad 大小，里面装 Q tile、QK logits、partial output、softmax max/sum。证据：`csrc/cpu/cpu_attn_impl.hpp:98`、`csrc/cpu/cpu_attn_impl.hpp:180-185`、`csrc/cpu/cpu_attn_impl.hpp:647-648`。
- `reduction_scratchpad_size_per_kv_head`：每个 `kv_head` 的共享 reduction scratchpad 大小，里面装 split flags、split partial outputs、split max/sum。证据：`csrc/cpu/cpu_attn_impl.hpp:99`、`csrc/cpu/cpu_attn_impl.hpp:186-189`、`csrc/cpu/cpu_attn_impl.hpp:653-654`。
- `workitem_groups_ptr` / `reduction_items_ptr`：指向 metadata tensor 内嵌数组的起始地址；运行时不再复制。证据：`csrc/cpu/cpu_attn_impl.hpp:100-101`、`csrc/cpu/cpu_attn_impl.hpp:118-123`。
- `cu_workitem_num_per_thread`：逻辑线程桶到 workitem 子数组的前缀和。运行时 `task_idx` 先映射成 `thread_offset`，再通过这个前缀和找到“这个逻辑线程桶对应哪一段 `AttentionWorkItemGroup[]`”。证据：`csrc/cpu/cpu_attn_impl.hpp:102-103`、`csrc/cpu/cpu_attn_impl.hpp:627-633`、`csrc/cpu/cpu_attn_impl.hpp:1454-1460`。

#### 2.4.3 它在运行期怎么被消费
- `AttentionMainLoop` 启动后，先从 `metadata` 里取：
  - `workitem_groups_ptr`
  - `reduction_items_ptr`
  - `cu_workitem_num_per_thread`
  - `effective_thread_num`
  - `reduction_item_num`
  - `split_kv_q_token_num_threshold`
  然后据此构造整个任务编号空间。证据：`csrc/cpu/cpu_attn_impl.hpp:1413-1429`。
- 任务空间的前半段是 attention tasks，后半段是 reduction tasks：
  - `workitem_groups_counter_num = actual_kv_head_num * effective_thread_num`
  - `reduction_items_counter_num = actual_kv_head_num * reduction_item_num`
  - `total_counter_num = 两者相加`
  证据：`csrc/cpu/cpu_attn_impl.hpp:1424-1429`。
- 真实 OMP 线程通过 `metadata.acquire_counter()` 全局 atomic 抢任务，不是固定线程只跑固定 request。抢到 attention task 后，再按 `kv_head_idx + thread_offset` 映射到一段 `AttentionWorkItemGroup[]`；抢到 reduction task 后，再映射到一个 `ReductionWorkItemGroup`。证据：`csrc/cpu/cpu_attn_impl.hpp:132-133`、`csrc/cpu/cpu_attn_impl.hpp:1443-1458`、`csrc/cpu/cpu_attn_impl.hpp:1765-1804`。
- 因而最准确的理解是：`scheduler_metadata` 描述的是“这一轮 CPU attention 的内部任务图 + scratchpad 配置 + task index 映射规则”，而不是“哪一个 request 由哪一个真实线程静态执行”。

## 3. Q / K / V 在 CPU attention 里的逻辑形状与物理布局

### 3.1 Q 的入口 shape：先按 batch flatten，但语义上仍按 request 处理
- `cpu_attention_with_kv_cache()` 的 `query` 输入 shape 是 `[num_tokens, num_heads, head_size]`，`output` 也是同 shape。这里的 `num_tokens` 不是单个 request 的长度，而是 **本轮 batch 所有 query token 扁平化后的总数**。证据：`csrc/cpu/cpu_attn.cpp:203-220`。
- `query_start_loc` 负责把这个扁平 tensor 重新切回 request 边界。例如 3 个 request 的 `q_len=[2,5,3]`，就会有 `query_start_loc=[0,2,7,10]`。证据：`vllm/v1/worker/gpu_model_runner.py:1531-1537`。
- 重要点：**物理存储是跨 request flatten 的，但 scheduler 切分时不会把两个 request 的 Q 混成一个 q tile**。它总是先固定 `req_id`，再在这个 request 内部沿 q token 维切。证据：`csrc/cpu/cpu_attn_impl.hpp:20-57`、`csrc/cpu/cpu_attn_impl.hpp:446-475`。
- 对当前 `Qwen3-30B-A3B + TP=2`，每个 TP rank 只看本 rank 的局部 head，因此本 rank 的 Q 为 `[num_tokens, 16, 128]`；不是全模型总 head。证据：`vllm/model_executor/models/qwen3_moe.py`、`csrc/cpu/cpu_attn_impl.hpp`。

### 3.2 K / V 的逻辑 shape：paged KV cache 中每个 rank 一份
- CPU backend 把单层单 rank 的 KV cache shape 定义为：`(2, num_blocks, num_kv_heads, block_size, head_size)`。拆开看就是：
  - `key_cache.shape = [num_blocks, num_kv_heads, block_size, head_size]`
  - `value_cache.shape = [num_blocks, num_kv_heads, block_size, head_size]`
  证据：`vllm/v1/attention/backends/cpu_attn.py:71-79`、`csrc/cpu/cpu_attn.cpp:203-208`。
- 这里的 4 个维度分别表示：
  - `num_blocks`：物理 KV page 数
  - `num_kv_heads`：当前 TP rank 本地的 KV head 数
  - `block_size`：一个 page 里连续 token 数，CPU 默认一般是 `128`
  - `head_size`：每个 head 的 hidden dim
- 因而单层单 rank 的 KV cache 总字节数可以直接由
  `2 * num_blocks * num_kv_heads * block_size * head_size * dtype_size`
  推出。证据：`vllm/platforms/cpu.py:187-189`。
- 对一个具体 request 来说，历史 K/V 并不是“天然连续”地放着；kernel 会通过 `block_table[req_id]` 找到这个 request 当前每个逻辑 block 对应的 `physical_block_idx`，再据此跳到真实的 KV page。证据：`csrc/cpu/cpu_attn_impl.hpp:939-945`、`csrc/cpu/cpu_attn_impl.hpp:1053-1059`。
- `slot_mapping` 只影响“本轮新 token 写入哪个物理 slot”，而 decode 读历史 KV 主要依赖 `block_table`。证据：`vllm/v1/worker/block_table.py:181-191`、`csrc/cpu/cpu_attn.cpp:143-201`。

### 3.3 计算时真正参与 GEMM 的 Q / K / V 是什么 shape
- 对某个 attention task，kernel 会先固定：
  - 一个 `req_id`
  - 一个 `kv_head_idx`
  - 该 request 内的一段 `q token slice`
  - 该 slice 当前可见的一段 `kv token span`
- 在这组固定索引下，Q/K/V 的**逻辑形状**更适合写成：
  - `Q_tile = [q_tile_token_num, q_heads_per_kv, head_dim]`
  - `K_tile = [kv_tile_token_num, head_dim]`
  - `V_tile = [kv_tile_token_num, head_dim]`
  证据：`csrc/cpu/cpu_attn_impl.hpp:1577-1584`、`csrc/cpu/cpu_attn_impl.hpp:1531-1538`。
- 做 QK 乘法前，Q 会被临时视成二维矩阵：
  - `Q_tile_2d = [q_tile_head_num, head_dim]`
  - 其中 `q_tile_head_num = q_tile_token_num * q_heads_per_kv`
- K 在逻辑上是 `[kv_tile_token_num, head_dim]`，但为了做 `Q @ K^T`，读取时等价于用：
  - `K_tile_T = [head_dim, kv_tile_token_num]`
- 所以 QK 阶段真正的矩阵乘法是：
  - `logits_tile = [q_tile_head_num, head_dim] @ [head_dim, kv_tile_token_num]`
  - 输出 `logits_tile.shape = [q_tile_head_num, kv_tile_token_num]`
  证据：`csrc/cpu/cpu_attn_impl.hpp:951-960`。
- softmax 之后，PV 阶段再做：
  - `output_tile = [q_tile_head_num, kv_tile_token_num] @ [kv_tile_token_num, head_dim]`
  - 输出 `output_tile.shape = [q_tile_head_num, head_dim]`
  证据：`csrc/cpu/cpu_attn_impl.hpp:1067-1072`。
- 所以要牢记：**Q 的主切分维度是 request 内的 q token 维；K/V 的主切分维度是 kv token 维；`head_dim` 通常不在 scheduler 层切，而是在 ISA kernel 内按向量宽度消费。**

### 3.4 K 和 V 的物理存储布局并不对称
- `reshape_and_cache()` 会把 K 写成列主序，把 V 写成行主序。证据：`csrc/cpu/cpu_attn_vec.hpp:196-244`。
- 这样设计不是随意 pack，而是为了匹配上面的两段 GEMM：
  - QK 阶段想要高效读出 `[head_dim, BlockSizeAlignment]` 方向的 K tile，因此 K 更接近“转置后友好”的布局。证据：`csrc/cpu/cpu_attn_impl.hpp:931-980`。
  - PV 阶段想要高效读出 `[block_size, HeadDimAlignment]` 方向的 V tile，因此 V 更接近 row-major。证据：`csrc/cpu/cpu_attn_impl.hpp:1038-1088`。

### 3.5 KV block 的容量与 rank 间对齐
- CPU 平台默认 KV 空间不是整机 50%，而是 `总内存 / NUMA 节点数 * 0.5`。证据：`vllm/platforms/cpu.py:146-165`。
- `num_blocks = available_memory // page_size // num_layers`。证据：`vllm/v1/core/kv_cache_utils.py:829-844`。
- 多 rank 时，所有 rank 最终会收缩到最小 `num_blocks`。证据：`vllm/v1/core/kv_cache_utils.py:1550-1565`。
- 这说明 KV cache 从一开始就是“每 rank 单独算容量，但最后按最小 rank 对齐”。如果某个 rank 因 NUMA 可见内存更少而缩小，所有 rank 都会一起缩小。

## 4. prefill / decode / mixed batch 在 CPU 上到底哪里不同

### 4.1 当前 x86 默认是 mixed batch
- `CPUAttentionMetadataBuilder` 在 x86 / ARM 上默认保留 mixed batch，不重排，也不启 `sdpa_prefill`。证据：`vllm/v1/attention/backends/cpu_attn.py:28`、`vllm/v1/attention/backends/cpu_attn.py:115-124`。
- 因此你当前这台 AMD EPYC 机器上，prefill / decode / mixed 的差别主要不是“走不同 Python 函数”，而是 **同一个 kernel 收到的 metadata 不同**。

### 4.2 三种 batch 在 kernel 里的差别

| 场景 | 典型 `q_len` | 当前 x86 路径 | kernel 内主要变化 | 对 locality 的影响 |
| --- | --- | --- | --- | --- |
| decode | 1 | `cpu_attention_with_kv_cache` | `q_tile_token_num` 很小，容易触发 split-KV | 共享同一组 KV 页的并发读最强 |
| prefill | `>1` 且可很长 | 同上 | Q tile 更大，split-KV 很少触发 | 算术密度更高，但 KV 仍可能流式读取 |
| mixed | 有的 1，有的 >1 | 同上 | 同一 metadata 同时存在多种 `q_len/seq_len` | 最容易把 locality 和负载均衡混在一起 |

### 4.3 为什么 decode 更像论文思路的受益者
- 调度器里 `split_kv_q_token_num_threshold = enable_kv_split ? 1 : 0`。证据：`csrc/cpu/cpu_attn_impl.hpp:402-403`。
- 真正允许切 KV 的条件是：只有尾部 q tile，且 `q_tile_token_num <= 1`。证据：`csrc/cpu/cpu_attn_impl.hpp:525-551`。
- 所以长 prefill 默认并不会大规模走 split-KV；decode 或尾部单 token tile 才更容易出现“多个 split 共享同一组 KV 页”的形态。
- 这和论文里的 ACC 很接近。ACC 可以简单理解成“共享同一组 K/V 的计算簇”；在 GPU 里它对应一组 workgroup，在这里更接近一组共享 KV 页的 workitem split。

### 4.4 测试里现成的三组 shape 模板
- decode batch：`[(1,213), (1,1), (1,312), (1,7), (1,7812)]`。证据：`tests/kernels/attention/test_cpu_attn.py:34-38`。
- prefill batch：`[(2345,2345), (5,5), (3,16), (134,5131)]`。证据同上。
- mixed batch：`[(992,2456), (1,1234), (98,1145), (1,4162), (2345,2345)]`。证据同上。
- 这三组 shape 很适合作为后续 decode / prefill / mixed 的受控微基准模板。

## 5. CPU kernel 内部到底按什么维度切分

### 5.1 先说结论：切分不是一次完成，而是 4 层叠加

| 层次 | 先固定什么 | 主要切哪一维 | 产物 |
| --- | --- | --- | --- |
| request 层 | `req_id` | 不跨 request | 单个 request 的 q / kv 范围 |
| scheduler Q 层 | `req_id`、可见 KV 范围 | request 内的 `q token` 维 | `AttentionWorkItemGroup` |
| task 展开层 | `req_id`、`q token slice` | `kv_head` 维 | attention task |
| kernel 内层 | `req_id`、`kv_head_idx`、`q workitem` | `kv token` 维，必要时再切更小 q micro-tile | `kv_tile` / split / reduction |

- 所以不要把“q_tile”“workitem”“kv_tile”当成同一层东西：
  - `q_tile` 常指当前这段 request-local 的 q token 切片
  - `workitem` 是 scheduler 聚出来、准备交给线程消费的较大任务
  - `kv_tile` 是运行时为了放进 L2/scratchpad 再切出来的一段 KV

### 5.1.1 `max_num_q_per_iter` 是怎么确定的
- `max_num_q_per_iter` 不是按 `q_len`、`seq_lens` 或 batch 大小动态算出来的；它来自当前被 dispatch 到的 attention ISA 实现的编译期常量 `attn_impl::MaxQHeadNumPerIteration`。证据：`csrc/cpu/cpu_attn.cpp:150-159`。
- 对当前这台机器、当前分析上下文，假设：实际走的是 `VEC` kernel；那么 `MaxQHeadNumPerIteration = 8`。证据：`csrc/cpu/cpu_attn_vec.hpp:126-131`。
- 进入 `AttentionScheduler::schedule()` 后，这个值先被读成 `max_num_q_per_iter = input.max_num_q_per_iter`，随后再结合
  - `q_head_per_kv = num_heads_q / num_heads_kv`
  - `max_num_q_token_per_iter = max_num_q_per_iter / q_head_per_kv`
  一起决定“最内层一次最多处理多少个 q heads / q tokens”。证据：`csrc/cpu/cpu_attn_impl.hpp:386-397`。
- 要注意名字里的歧义：
  - `max_num_q_per_iter` 说的是 **Q head 数**
  - `max_num_q_token_per_iter` 才是折算后的 **Q token 数**
- 对当前重点模型 `Qwen3-30B-A3B + TP=2`，每个 TP rank 有 `num_heads_q=16`、`num_heads_kv=2`，所以：
  - `q_head_per_kv = 16 / 2 = 8`
  - 当前若走 `VEC`，则 `max_num_q_per_iter = 8`
  - 因而 `max_num_q_token_per_iter = 8 / 8 = 1`

### 5.2 第 1 层：先按 request 切，绝不跨 request
- 调度器先读 `query_start_loc` 和 `seq_lens`，然后最外层就是 `for (req_id = 0; req_id < num_reqs; ++req_id)`。证据：`csrc/cpu/cpu_attn_impl.hpp:446-457`。
- 因此 q 切分语义一定是：
  - `req0` 内的 `q[a:b)`
  - `req1` 内的 `q[c:d)`
  但不会出现一个 q tile 同时跨 `req0` 和 `req1`。
- `AttentionWorkItemGroup` 里显式保存 `req_id`、`q_token_id_start`、`q_token_num`，也说明 task 是 request-local 的。证据：`csrc/cpu/cpu_attn_impl.hpp:20-57`。

### 5.3 第 2 层：在 request 内沿 q token 维扫描，再聚成 workitem
- 调度器先推导：
  - `q_head_per_kv`
  - `max_num_q_token_per_iter`
  - `default_tile_size`
  - `default_tile_token_num`
  - `split_kv_q_token_num_threshold`
  - `kv_len_per_thread`
  证据：`csrc/cpu/cpu_attn_impl.hpp:382-445`、`csrc/cpu/cpu_attn_impl.hpp:699-749`。
- 然后它在**单个 request 内**按
  `for (token_id += max_num_q_token_per_iter)`
  前进，先以很细的 q 步长扫过去。证据：`csrc/cpu/cpu_attn_impl.hpp:460-463`。
- 但 scheduler 并不会每扫到一次就立刻生成一个最终 task；它会把若干个相邻 q token 聚成一个较大的 `AttentionWorkItemGroup`，大小大致受 `default_tile_token_num` 控制。证据：`csrc/cpu/cpu_attn_impl.hpp:531-549`。
- 这里“在 request 内的 q token 维上切分”可以直接按 shape 理解：如果某个 prefill request 的本地 Q 逻辑 shape 是 `[1024, 16, 128]`，那么这一层更接近先把第 0 维 `1024` 个 token 按连续区间切成
  - `q[0:41, 16, 128]`
  - `q[41:82, 16, 128]`
  - `q[82:123, 16, 128]`
  - ...
  也就是先得到“若干段连续 token 片段”，再把这些片段聚成 workitem。
- 但要注意：上面这个 `[41, 16, 128]` 只是站在“request 内 q token 切分”这一层的直观写法。后续一旦进入具体的 `kv_head` task，同一段 q token 还会再按 GQA 视角变成 `[41, q_heads_per_kv, 128]`；对当前 `Qwen3-30B-A3B + TP=2`，就是 `[41, 8, 128]`。再往最内层走，当前 `VEC` 下通常又会按 `1 token` 的 q micro-iter 推进。证据：`csrc/cpu/cpu_attn_impl.hpp:1500-1504`、`csrc/cpu/cpu_attn_impl.hpp:1577-1584`、`csrc/cpu/cpu_attn_impl.hpp:1637-1648`。

### 5.4 第 3 层：task 空间再按 `kv_head` 展开
- 进入主循环后，task 空间先按 `kv_head_idx` 展开，再按虚拟 `thread_offset` 展开。证据：`csrc/cpu/cpu_attn_impl.hpp:1451-1463`。
- 也就是说，一个 attention task 可以粗略理解成：
  `固定 req_id + 固定 q token slice + 固定 kv_head_idx + 一段 kv span`
- 对 GQA，多个 `q heads` 会共享同一个 `kv head`，所以 task 里真正处理的是“一组 q heads 对一个 kv head 的计算”。证据：`csrc/cpu/cpu_attn_impl.hpp:1360-1364`、`csrc/cpu/cpu_attn_impl.hpp:1540-1543`。
- 这一步相当于把原始的 `Q.shape=[token,head,dim]` 拆成了“token 片段 × kv_head group”两个正交方向。
- 这里的 `kv span` 指的是一个 **逻辑 KV token 区间**，更准确可写成 `[kv_start_pos, kv_end_pos)`。它来自 `AttentionWorkItemGroup` 里的
  - `kv_split_pos_start`
  - `kv_split_pos_end`
  运行时线程拿到 workitem 后，也正是先读出这两个值。证据：`csrc/cpu/cpu_attn_impl.hpp:20-30`、`csrc/cpu/cpu_attn_impl.hpp:1470-1477`。
- 所以 `kv span` 说的是“这个 workitem / task 负责看哪一段历史 KV token 范围”，不是某段物理内存地址；真正读物理页时，后面还要结合 `block_table -> physical_block_idx` 再跳到对应的 paged KV block。
- `kv span` 和后面的 `kv_tile` 也不是同一层：
  - `kv span`：task/workitem 级的较大逻辑区间
  - `kv_tile`：线程在执行这个 workitem 时，为了放进 L2/scratchpad，再把这段 `kv span` 继续切成的小块
  所以通常是：**一个 `kv span` 下面包含多个 `kv_tile`**。

### 5.4.1 OMP 线程、task、workitem、tile 到底是什么关系
- `#pragma omp parallel for schedule(static, 1)` 这一层只是先拉起真实的 OMP 线程，并给每个线程一个固定 `thread_id`，以及一块自己的 thread-local scratchpad；真正的 attention 计算任务不是在这个 OpenMP `for` 上静态切好的，而是线程进入主循环后通过 `metadata.acquire_counter()` 动态抢到的。证据：`csrc/cpu/cpu_attn_impl.hpp:1345-1385`、`csrc/cpu/cpu_attn_impl.hpp:1442-1445`。
- 因此可以把运行时层级关系写成：

```text
真实 OMP 线程
  -> 抢到一个 runtime task（task_idx）
       -> 这个 task 对应某个 kv_head_idx + 某个 thread_offset
       -> 这个 task 下面要跑一段 AttentionWorkItemGroup[]
            -> 每个 workitem 再切成若干 q_tile
                 -> 每个 q_tile 再切成若干 kv_tile
                      -> 每个 kv_tile 里再按 q micro-iter 调 execute_attention(...)
```

- 这里最容易混淆的点是：**runtime task 不等于单个 workitem**。代码里线程抢到一个 `task_idx` 后，会先算出
  - `kv_head_idx = task_idx / effective_thread_num`
  - `thread_offset = task_idx % effective_thread_num`
  然后再通过 `cu_workitem_num_per_thread` 找到“这个逻辑线程桶对应的一整段 `AttentionWorkItemGroup[]`”，并把这段 workitems 全部跑完。证据：`csrc/cpu/cpu_attn_impl.hpp:1451-1460`。
- 所以更准确地说：一个真实线程抢到一个 runtime task 后，通常会在这个 task 里连续执行多个 workitem；而每个 workitem 在这个线程内部又会继续被切成多个 tile 来算。证据：`csrc/cpu/cpu_attn_impl.hpp:1464-1525`、`csrc/cpu/cpu_attn_impl.hpp:1630-1728`。
- 对当前 `Qwen3-30B-A3B + TP=2 + VEC` 的常见情况，可以把最内层层级近似记成：
  - 线程先拿到一个 `kv_head_idx`
  - 对这个 `kv_head_idx` 下的一组 workitems 逐个处理
  - 每个 workitem 里，先按 `default_q_tile_token_num` 取一段 q tile
  - 再把这段 q tile 对应的 KV 范围切成多个 `kv_tile`
  - 对每个 `kv_tile`，再按 `max_num_q_token_per_iter=1` 做最小 q micro-iter
- 因而如果只问“一个 task 是不是分给一个 OpenMP 线程，而这个 task 会不会再切成多个 tile 由同一个线程计算”，答案是：**是的，近似可以这样理解；但要记住这里的 task 指的是 runtime `task_idx`，它下面通常还包含多个 workitems，而不是单个 workitem。**

### 5.5 第 4 层：运行时再按 KV token 维切成 `kv_tile`
- 对每个 workitem，kernel 会重新根据当前 q tile 大小和 L2 容量算 `kv_tile_size`。证据：`csrc/cpu/cpu_attn_impl.hpp:1498-1515`。
- 然后内层按
  `for (kv_tile_pos += kv_tile_size)`
  前进，也就是把当前可见 KV 范围切成多段 `kv_tile`。证据：`csrc/cpu/cpu_attn_impl.hpp:1630-1636`。
- 所以 K/V 的主运行时切分维度，是 **kv token 维**，而不是 `head_dim` 维。
- `head_dim` 更像 GEMM 的列维/特征维，通常保持完整，由底层 ISA kernel 向量化消费；文档里如果把它说成“scheduler 在 head_dim 上切块”，是不准确的。

### 5.6 `Qwen3-30B-A3B + TP=2` 下，Q/K/V 在计算时到底变成什么 shape
- 下面统一按每个 TP rank 的本地 attention shape 来看：`num_query_heads=16`、`num_kv_heads=2`、`head_dim=128`。来源：`/models/Qwen3-30B-A3B/config.json` 与 `vllm/model_executor/models/qwen3_moe.py`。
- 对这组参数：
  - `q_head_per_kv = 16 / 2 = 8`
  - 当前 x86 常走 `vec` kernel，`MaxQHeadNumPerIteration = 8`
  - 所以 `max_num_q_token_per_iter = 8 / 8 = 1`
  证据：`csrc/cpu/cpu_attn_impl.hpp:388-396`、`csrc/cpu/cpu_attn_vec.hpp`。
- 这意味着最底层 q micro-tile 的逻辑 shape 是：
  - `Q_micro = [1, 8, 128]`
  - 展平成二维就是 `[8, 128]`
- 对应的某段 K/V tile 逻辑 shape 是：
  - `K_tile = [kv_tile_token_num, 128]`
  - `V_tile = [kv_tile_token_num, 128]`
- 真正矩阵乘法时：
  - `QK: [8, 128] @ [128, kv_tile_token_num] -> [8, kv_tile_token_num]`
  - `PV: [8, kv_tile_token_num] @ [kv_tile_token_num, 128] -> [8, 128]`
- 所以对这台机器当前常见配置，最小底层步长可以直接记成：
  - **`1 个 q token × 1 个 kv_head × 8 个 q_heads × 一段 kv tokens`**

### 5.7 例子 A：decode 场景 `q_len=1, kv_len=4096`
- scheduler 输入可以写成：
  - `query_start_loc = [0, 1]`
  - `seq_lens = [4096]`
  - `q_token_num = 1`
  - `q_start_pos = seq_len - q_token_num = 4095`
  证据：`csrc/cpu/cpu_attn_impl.hpp:411-425`、`csrc/cpu/cpu_attn_impl.hpp:451-468`。
- 因为 `max_num_q_token_per_iter = 1`，所以这个 request 只有一个最细 q micro-tile：
  - `Q_micro(logical) = [1, 8, 128]`
  - `Q_micro_2d = [8, 128]`
- 若 **关闭 `kv_split`**：
  - scheduler 更像只给出 1 个 workitem：`(req0, q[0:1), kv[0:4096))`
  - 运行时再把 `kv[0:4096)` 切成若干 `kv_tile`
- 若 **开启 `kv_split`**：
  - 同一个 `(req_id, q[0:1), kv_head_idx)` 会被沿 KV token 维拆成多个 split
  - 每个 split 都各自做 `QK/PV`
  - 最后再做 reduction 合并
  证据：`csrc/cpu/cpu_attn_impl.hpp:402-403`、`csrc/cpu/cpu_attn_impl.hpp:525-579`。
- 所以 decode 的本质是：
  - Q 侧几乎不切
  - 主要沿 KV token 维切
  - 多个 split 共享同一个 request、同一个 q token、同一个 kv_head

### 5.8 例子 B：prefill 场景 `q_len=1024, kv_len=1024`
- 对这组参数，按 `dtype=half`、`cache_size=512 KiB` 代入 `calcu_default_tile_size(...)`，常见会得到：
  - `default_tile_size ≈ 328`
  - `default_tile_token_num = 328 / 8 = 41`
  证据：`csrc/cpu/cpu_attn_impl.hpp:397-403`、`csrc/cpu/cpu_attn_impl.hpp:699-724`。
- 所以 scheduler 更像先在 request 内沿 q token 维聚出：

```text
req0
  -> workitem0:  q[0:41),    kv[0:1024)
  -> workitem1:  q[41:82),   kv[0:1024)
  -> ...
  -> workitem24: q[984:1024), kv[0:1024)
```

- 对其中一个 `q[0:41)` workitem：
  - `Q_tile(logical) = [41, 8, 128]`
  - `Q_tile_2d = [328, 128]`
- 然后运行时再把这个 workitem 的可见 KV 范围切成若干 `kv_tile`，例如：

```text
workitem0 = q[0:41), kv[0:1024)
  -> kv_tile0: Q=[41,8,128], K/V=[320,128]
  -> kv_tile1: Q=[41,8,128], K/V=[320,128]
  -> kv_tile2: Q=[41,8,128], K/V=[320,128]
  -> kv_tile3: Q=[41,8,128], K/V=[64,128]
```

- 但要注意：虽然 workitem 已经是 `41 token`，最里面真正送进 QK/PV 的仍是更小的 q micro-tile。对这组 GQA 参数，它仍然按 `1 token` 一步步推进。证据：`csrc/cpu/cpu_attn_impl.hpp:1637-1658`。
- 所以 prefill 的本质是：
  - 先沿 request 内的 q token 维聚 workitem
  - 再沿 KV token 维切 `kv_tile`
  - 最内层仍以小 q micro-tile 驱动 GEMM

### 5.9 把 decode 和 prefill 并排看

```text
decode:  q_len=1, kv_len=4096
  Q 的主要形态：Q_micro = [1, 8, 128]
  主要切分方向：沿 KV token 维切 split / kv_tile

prefill: q_len=1024, kv_len=1024
  Q 的主要形态：先聚成 Q_tile = [41, 8, 128] 左右的 workitem
  主要切分方向：先沿 request 内 q token 维聚包，再在 workitem 内沿 KV token 维切 tile
```

- 这也解释了为什么当前源码下，`kv_split` 对 decode 更敏感、对长 prefill 往往不那么敏感：decode 的 Q 天然只有 1 个 token，而 prefill 的主体工作已经先被“Q 聚包”吃掉了。证据：`csrc/cpu/cpu_attn_impl.hpp:402-403`、`csrc/cpu/cpu_attn_impl.hpp:525-551`。
- 如果要给导师一句话总结，可以直接说：**对 `Qwen3-30B-A3B + TP=2`，Q 的入口 shape 是 `[token,16,128]`，K/V cache 的入口 shape 是 `[block,2,128,128]`；运行时先按 request 内 q token 聚成 workitem，再固定 `kv_head`，最后沿 KV token 维切成 tile 做 `QK/PV`。**

## 6. 单个 attention task 在 kernel 内到底做了什么

### 6.1 scratchpad 布局
- `scratchpad` 就是 kernel 在执行期间使用的临时缓冲区。每个线程有自己的 thread scratchpad；若用了 split-KV，还会有按 `kv_head` 切分的 reduction scratchpad。证据：`csrc/cpu/cpu_attn_impl.hpp:180-353`。
- thread scratchpad 里主要有：
  - `q_buffer`
  - `logits_buffer`
  - `partial_output_buffer`
  - `max_buffer`
  - `sum_buffer`
- reduction scratchpad 里主要有：
  - `flag`
  - `split_output_buffer`
  - `split_max_buffer`
  - `split_sum_buffer`

### 6.1.1 从 `AttentionMainLoop` 往下看，单个 task 的内部调用顺序

```text
AttentionMainLoop::operator()
  -> 从 scheduler_metadata 取出 workitem / reduction item / prefix sum
  -> 若抢到的是 attention task
       -> 解析 req_id / q_token slice / kv span / kv_head_idx
       -> copy_q_heads_tile(...)
       -> 对每个 kv_tile:
            -> execute_attention<Attention>(...)
                 -> QK
                 -> softcap / alibi / mask
                 -> online softmax
                 -> PV
       -> 若未 split: final_output(...)
       -> 若已 split: partial_output(...)
  -> 若抢到的是 reduction task
       -> reduce_splits(...)
       -> final_output(...)
```

- 所以后文凡是说“进入主循环后先做什么”，默认指的是 `AttentionMainLoop::operator()` 这一层；凡是说“QK/PV 在哪里做”，默认指的是 `execute_attention<Attention>(...)` 这一层。证据：`csrc/cpu/cpu_attn_impl.hpp:1339-1815`、`csrc/cpu/cpu_attn_impl.hpp:875-1093`。

### 6.2 先复制并缩放 Q
- `copy_q_heads_tile()` 会把 query tile 拷到 `q_buffer`，并转成 `fp32`，同时乘上 `scale`。证据：`csrc/cpu/cpu_attn_vec.hpp:164-194`、`csrc/cpu/cpu_attn_impl.hpp:1577-1584`。
- 若是 GQA，一个 `kv_head` 下的多路 `q heads` 会一起放在这个 tile 里。
- 这时 `q_buffer` 的逻辑 shape 是 `[actual_q_token_num, actual_q_heads_per_kv, head_dim]`；进入 GEMM 时再视成二维 `[actual_q_token_num * actual_q_heads_per_kv, head_dim]`。证据：`csrc/cpu/cpu_attn_impl.hpp:1577-1584`。

### 6.3 用 `block_table` 找到物理 K/V block
- 对每个 KV tile，kernel 先通过 `block_table[block_idx]` 取出 `physical_block_idx`，再得到实际的 `k_cache_block_ptr` / `v_cache_block_ptr`。证据：`csrc/cpu/cpu_attn_impl.hpp:939-945`、`csrc/cpu/cpu_attn_impl.hpp:1053-1059`。
- 这一步就是 paged KV 的真正物理跳转点。

### 6.4 QK -> softcap / alibi / mask -> online softmax -> PV
- QK：`tile_gemm<AttentionGemmPhase::QK>`。证据：`csrc/cpu/cpu_attn_impl.hpp:931-980`。
- QK 这一段更具体地看，是：
  - `Q_tile_2d = [q_tile_head_num, head_dim]`
  - `K_tile_T = [head_dim, kv_tile_token_num]`
  - `logits_tile = [q_tile_head_num, kv_tile_token_num]`
  证据：`csrc/cpu/cpu_attn_impl.hpp:951-960`。
- softcap：`apply_softcap(...)`，可以把它理解成“对 attention logits 做额外压缩/封顶”，避免 logits 过大。证据：`csrc/cpu/cpu_attn_impl.hpp:997-1002`、`csrc/cpu/cpu_attn_impl.hpp:1256-1265`。
- alibi：`apply_alibi_slopes(...)`，也就是给不同相对位置加一项线性偏置。证据：`csrc/cpu/cpu_attn_impl.hpp:1004-1011`。
- mask：`apply_mask(...)`。证据：`csrc/cpu/cpu_attn_impl.hpp:1013-1015`、`csrc/cpu/cpu_attn_impl.hpp:1095-1144`。
- online softmax：`apply_softmax(...)`，维护 `max_buffer` 和 `sum_buffer`，必要时对已有 partial output 做 rescale。证据：`csrc/cpu/cpu_attn_impl.hpp:1024-1026`、`csrc/cpu/cpu_attn_impl.hpp:1146-1254`。
- PV：`tile_gemm<AttentionGemmPhase::PV>`。证据：`csrc/cpu/cpu_attn_impl.hpp:1038-1088`。
- PV 更具体地看，是：
  - `prob_tile = [q_tile_head_num, kv_tile_token_num]`
  - `V_tile = [kv_tile_token_num, head_dim]`
  - `output_tile = [q_tile_head_num, head_dim]`
  证据：`csrc/cpu/cpu_attn_impl.hpp:1067-1072`。

### 6.5 split-KV 的写回和归约
- 若当前 workitem 没 split，直接 `final_output(...)` 写最终输出。证据：`csrc/cpu/cpu_attn_impl.hpp:1732-1740`、`csrc/cpu/cpu_attn_impl.hpp:1929-1971`。
- 若 split 了，就先 `partial_output(...)` 把局部结果、局部 `max/sum` 写到 reduction scratchpad，并置 flag。证据：`csrc/cpu/cpu_attn_impl.hpp:1741-1761`、`csrc/cpu/cpu_attn_impl.hpp:1903-1927`。
- reduction task 会等待所有 flag，就地做 rescale + merge，再调用 `final_output(...)`。证据：`csrc/cpu/cpu_attn_impl.hpp:1765-1810`、`csrc/cpu/cpu_attn_impl.hpp:1817-1901`。

## 7. NUMA 与线程绑定：CPU 上的数据放置边界在哪里

### 7.1 auto bind 的实际行为
- `CPUWorker.init_device()` 中，若 `VLLM_CPU_OMP_THREADS_BIND=auto`，x86 会对每个 physical core 只选 1 个 SMT 线程，并把 `local_rank` 映射到 `allowed_numa_nodes[local_rank]`。证据：`vllm/v1/worker/cpu_worker.py:54-107`、`vllm/v1/worker/cpu_worker.py:126-187`。
- 因此在 auto bind 下，一个 worker 的 OMP 线程集合默认就是“一个 NUMA node 里的若干 core”。

### 7.2 内存策略在 C++ 里真的落地
- `init_cpu_threads_env()` 会先把 `cpu_ids` 解析成 cpuset，再反查这些 CPU 属于哪些 NUMA node。证据：`csrc/cpu/utils.cpp:25-66`。
- 只有一个 node 时：`numa_set_membind(mask)`。证据：`csrc/cpu/utils.cpp:96-109`。
- 多个 node 时：`numa_set_interleave_mask(mask)`。证据：`csrc/cpu/utils.cpp:82-95`。
- 同时它还会：
  - `numa_migrate_pages` 迁移已有页；
  - `omp_set_num_threads()` / `torch::set_num_threads()`；
  - 对每个 OMP thread 调 `sched_setaffinity()`。证据：`csrc/cpu/utils.cpp:71-79`、`csrc/cpu/utils.cpp:124-165`。

### 7.3 对 KV cache/权重/临时 buffer 的含义
- 事实：上述 NUMA policy 在 worker 初始化早期就设置了，而且发生在 `CPUModelRunner` 构造前。证据：`vllm/v1/worker/cpu_worker.py:54-107`。
- 假设：worker 后续的大多数 CPU 内存分配，包括 KV cache 页、scratchpad、模型权重，都会继承该进程当前的 NUMA policy。因此“单 node membind”与“多 node interleave”会直接改变 attention 访问到的物理内存分布。
- 这也是为什么我认为 **手工把一个 worker 绑到多个 node** 时，要非常谨慎：源码会显式进入 interleave，而不是“自动 local-first”。

### 7.4 NPS 为什么会直接影响 attention locality
- 代码本身不认识“NPS”这个词，但它认识“OS 暴露出来有几个 NUMA node”。
- `CpuPlatform.get_device_total_memory()` 默认按 `总内存 / NUMA 节点数 * 0.5` 估 KV 空间。证据：`vllm/platforms/cpu.py:146-165`。
- `CPUWorker._get_autobind_cpu_ids()` 又要求 `allowed_numa_nodes >= world_size`，并且 `local_rank -> selected_numa_node` 一一映射。证据：`vllm/v1/worker/cpu_worker.py:140-187`。
- 假设：NPS 改变后，如果 OS 暴露的 node 数变多，auto bind 的粒度会更细，默认 KV 容量估算也会跟着变化；因此 NPS 不只是 BIOS 背景变量，它会实打实影响 worker 放置和 KV 预算。

## 8. TP 如何改变 attention 的计算边界和数据边界

### 8.1 attention head 先在 TP rank 级切开
- `Qwen3MoeAttention` 里：
  - `self.num_heads = total_num_heads // tp_size`
  - `self.num_kv_heads = max(1, total_num_kv_heads // tp_size)`，若 KV head 比 TP 小则复制
  - `qkv_proj = QKVParallelLinear(...)`
  - `o_proj = RowParallelLinear(...)`
  证据：`vllm/model_executor/models/qwen3_moe.py:269-306`。
- 这意味着每个 TP rank 的 attention 只看本 rank 的那部分 head / KV head。

### 8.2 attention 后才回到 TP 通信路径
- `ColumnParallelLinear.forward()` 只有 `gather_output=True` 才 `all_gather`；默认 `qkv_proj` 不会立刻 all-gather。证据：`vllm/model_executor/layers/linear.py:595-614`。
- `RowParallelLinear.forward()` 在 `reduce_results=True` 且 `tp_size>1` 时做 `tensor_model_parallel_all_reduce`。证据：`vllm/model_executor/layers/linear.py:1442-1464`。
- 所以 attention 本身更像“rank 内本地算子”，attention 输出经 `o_proj` 才汇总到 TP group。

### 8.3 当前环境的 TP 数据面后端
- 当前仓库 CPU 通信器优先用 `torch.distributed`；只有编译出 `torch.ops._C.init_shm_manager` 且 group 名符合条件时才切到 `_CPUSHMDistributed`。证据：`vllm/distributed/device_communicators/cpu_communicator.py:20-40`、`vllm/distributed/device_communicators/cpu_communicator.py:158-213`。
- `2026-03-18 23:38 +0800` 的日志显示当前 TP=2 实跑 backend 是 `gloo`。证据：`test_results/qwen3_tp2_init_2026-03-18_2338.log:148`。

### 8.4 对 Chiplet/NUMA 优化的直接含义
- 若 TP rank 与 NUMA node 一一对应，那么“attention 读本地 KV + 本地 head”这件事就更容易成立。
- 若 TP rank 覆盖多个 NUMA node，或者 worker 本身走 interleave，那么即使 head 已经在 rank 级切开，rank 内 attention 仍可能跨 node 读页。
- 所以 TP 并没有消除数据放置问题，它只是把问题的边界缩到了“每个 rank 内部”。

## 9. 与 GPU 论文的逐项对照：哪些能迁移，哪些不能

### 9.1 论文真正做了什么
- 论文核心观点：在 disaggregated / chiplet GPU 上，attention 的多个计算块如果共享同一组 K/V，就应尽量被调到同一 NUMA 域 / 同一 XCD，以提高缓存复用。这里的 XCD 可以简单看成 MI300X 上一个相对局部的执行/缓存域。证据：`info/papers/GPU_Attention_NUMA_Optimization.pdf` 第 1 页摘要、第 7 页 Section 3.3。
- 论文提出 `Swizzled Head-first Mapping`：先按 head-first 组织，再通过 workgroup ID 重映射，把共享 K/V 的 workgroup 尽量限制在同一 XCD。证据：同 PDF 第 7 页 Figure 10/11。
- 论文给出的收益是：在 MI300X 上 attention 可达最高约 50% 性能提升，并维持 80-97% 的 L2 hit rate。证据：同 PDF 第 1 页摘要、第 8-10 页 Figure 12-16 与结论。

### 9.2 CPU 上能直接对应的对象是什么

| 论文概念 | GPU 含义 | 我认为在 CPU vLLM 上最接近的对象 |
| --- | --- | --- |
| ACC | 共享同一组 K/V 的 workgroup 集合 | 同一 `TP rank` 内，共享同一组 `KV head + block_table 页集合` 的 workitem 集合 |
| Head-first | 先把同一 head 的块跑完 | 主循环 task 先按 `kv_head_idx` 展开，再处理该 head 的 workitem/reduction |
| XCD-locality | 同一 XCD 服务同一 ACC | 同一 NUMA node / 同一 thread cluster 服务同一 `kv_head` 热点工作集 |
| Swizzle | 改变 workgroup ID 到硬件域的映射 | 改变 `task -> thread cluster / NUMA domain` 的映射 |

### 9.3 已经“部分像”论文的地方
- CPU kernel 不是完全 block-first。它在运行时会先固定 `kv_head_idx`，并在 GQA 下让多个 `q heads` 共用同一 `kv head`。证据：`csrc/cpu/cpu_attn_impl.hpp:1451-1463`、`csrc/cpu/cpu_attn_impl.hpp:1540-1543`。
- 这说明 vLLM CPU attention 并不缺“共享 K/V 的计算集合”这个概念，它缺的是把这个集合和 NUMA 域稳定对齐。

### 9.4 不能直接照搬的地方
- GPU 论文的调度对象是 workgroup / XCD；CPU vLLM 的调度对象是“TP 多进程 + 进程内 OpenMP 线程 + kernel workitem”。
- GPU 论文主要追 L2/XCD locality；CPU 上更可能先暴露的是 `Remote DRAM Reads %`、`Ave L3 Miss Latency`、页放置和跨 CCD/跨 node 访存。
- GPU 的 swizzle 本质是 launch mapping；CPU 这里更像“KV 页分配 + 线程簇映射 + batch 组织”三件事的组合，不能照着 GPU kernel 的 grid 映射方式生搬。

### 9.5 我认为最值得试的 4 条 CPU 优化方向

### 方向 A：KV page local-first placement
- 思路：不是只靠进程级 `membind/interleave`，而是让 attention 的 KV page 分配更接近“每个 worker / 每个 node 自己的 page pool”。
- 预期最敏感指标：`Remote DRAM Reads %`、`Ave L3 Miss Latency`。
- 更可能受益的场景：长 context decode、mixed batch 中 decode 占比较高时。
- 当前证据基础：`block_table` 决定了历史 KV 的真实物理 block，读路径完全依赖它。证据：`vllm/v1/worker/block_table.py`、`csrc/cpu/cpu_attn_impl.hpp`。

### 方向 B：把 `kv_head` 固定映射到 thread cluster / chiplet
- 思路：当前 task 由全局 atomic 动态领取，会打散 `kv_head -> physical thread` 的稳定关系；可尝试做“静态 head-cluster 分配”或“NUMA-aware head-first dispatch”。
- 预期最敏感指标：decode-only 下的 `CPI`、`Ave L3 Miss Latency`、`Remote DRAM Reads %`。
- 当前证据基础：task 先按 `kv_head_idx` 展开，但线程领取是动态的。证据：`csrc/cpu/cpu_attn_impl.hpp:1442-1463`。

### 方向 C：mixed batch 隔离或 decode-first NUMA-aware batching
- 思路：x86 当前默认保留 mixed batch，这对吞吐友好，但不一定对 locality 最友好。可以尝试把 decode-heavy 子集优先聚到更稳定的 thread/node 上，或把长 prefill 与 decode 分离比较。
- 预期最敏感指标：mixed batch 的 `CPI` 波动、`Total Mem Bw`、`Remote DRAM Reads %`。
- 当前证据基础：x86 默认 mixed，不启 `sdpa_prefill`。证据：`vllm/v1/attention/backends/cpu_attn.py`。

### 方向 D：TP rank 与 NUMA node / NPS 对齐
- 思路：尽量让 `TP rank == local NUMA domain`，避免一个 attention rank 自己跨多个 node 读 KV。
- 预期最敏感指标：`Remote DRAM Reads %`，其次是 TP 后段 all-reduce 前后的 stall。
- 当前证据基础：`CPUWorker` auto bind 是 `local_rank -> selected_numa_node`；TP attention 先在 rank 内算。证据：`vllm/v1/worker/cpu_worker.py`、`vllm/model_executor/models/qwen3_moe.py`。

## 10. 导师可能追问的问题

### Q1：vLLM 在 CPU 上有没有“专门的 prefill attention kernel”？
- 对你当前这台 x86 机器，通常没有。prefill / decode / mixed 大多都走 `cpu_attention_with_kv_cache`，只是 metadata 不一样。证据：`vllm/v1/attention/backends/cpu_attn.py`。
- 只有非 `_CPU_ARCH_PREFER_MIXED_BATCH` 的架构，才会把 prefill 单独送 SDPA。证据同上。

### Q2：`slot_mapping` 和 `block_table` 到底谁更重要？
- 写路径看 `slot_mapping`，读路径看 `block_table`。
- 若问题是“本轮新 token 的 K/V 写到哪”，看 `slot_mapping`；若问题是“decode 为什么会远端读内存”，先看 `block_table` 和物理页放置。

### Q3：为什么说 decode 更值得先研究 Chiplet/NUMA 优化？
- 因为 split-KV 只对 `q_tile_token_num <= 1` 的尾 tile 开放，decode 更容易形成“多个 workitem 共享同一组 KV 页”的形态。证据：`csrc/cpu/cpu_attn_impl.hpp:402-403`、`csrc/cpu/cpu_attn_impl.hpp:525-551`。

### Q4：当前 CPU kernel 已经是 head-first 了吗？
- 只能说“部分是”。
- 是的部分：task 空间先按 `kv_head_idx` 展开，GQA 也是围绕共享 KV head 来组织。
- 不是的部分：KV 页没有按 head 放置；实际物理线程通过 atomic 动态领取 task，没有稳定的 `head -> chiplet` 映射。
- 所以它还不是论文那种 NUMA-aware head-first。

### Q5：TP 打开以后，attention 前后哪些地方会通信？
- `qkv_proj` 默认不会立即 all-gather；attention 先在 rank 内做。
- `o_proj` 是 `RowParallelLinear`，默认会 `all_reduce`。证据：`vllm/model_executor/layers/linear.py`、`vllm/model_executor/models/qwen3_moe.py`。

### Q6：当前 CPU attention 更像算力瓶颈还是内存瓶颈？
- 假设：对长 context decode，更像“KV 读路径 + NUMA/locality”问题，而不是纯算力上限问题。
- 证据不是现成实验，而是源码链路：读取必须经过 `block_table -> physical_block_idx -> paged KV`，且 tile/scratchpad 设计明显围绕 cache 大小。证据：`csrc/cpu/cpu_attn_impl.hpp`、`csrc/cpu/utils.hpp`。
- 但这条必须靠后续 `CPI / Total Mem Bw / Ave L3 Miss Latency / Remote DRAM Reads %` 实测确认，当前不能写成结论。

### Q7：NPS 变化为什么会影响 attention，不只是影响 OS 拓扑显示？
- 假设：因为 CPU backend 直接按“可见 NUMA node 数”来做 auto bind 和默认 KV 空间估算，所以 NPS 改变后，worker 绑核边界、每 rank 的默认 KV 预算、甚至 TP 是否能一 rank 一 node 都会变。证据：`vllm/platforms/cpu.py:146-165`、`vllm/v1/worker/cpu_worker.py:140-187`。

## 11. 我建议的下一步实验

### 11.1 先做 3 组 attention 微基准
- decode-only：直接用 `tests/kernels/attention/test_cpu_attn.py` 里的 decode batch 模板。
- prefill-only：同文件里的 prefill batch 模板。
- mixed：同文件里的 mixed batch 模板。
- 基准入口：`benchmarks/kernels/cpu/benchmark_cpu_attn.py`。证据：`benchmarks/kernels/cpu/benchmark_cpu_attn.py`。

### 11.2 需要补的打点
- Python 层：`CPUAttentionMetadataBuilder.build()`，记录 `seq_lens`、`query_start_loc`、`use_sdpa_prefill`、`num_decode_tokens`。位置：`vllm/v1/attention/backends/cpu_attn.py`。
- C++ 调度层：`cpu_attn_get_scheduler_metadata()`，记录 `workitem_group_num`、`reduction_item_num`、`reduction_split_num`、`effective_thread_num`、`attention_scratchpad_size_per_thread`。位置：`csrc/cpu/cpu_attn_impl.hpp`。
- C++ 运行层：采样记录 `thread_id -> task_idx -> kv_head_idx -> req_id -> split_id`，验证是否存在“共享 KV 的 task 被多个 chiplet 打散执行”。位置：`csrc/cpu/cpu_attn_impl.hpp`。

### 11.3 建议和 uProf/PCM 对齐的指标
- `CPI`
- `Total Mem Bw`
- `Ave L3 Miss Latency`
- `Remote DRAM Reads %`
- 如果能补：`Local DRAM Reads` 或 node-local/remote 细分来源

### 11.4 建议的结果目录组织
- 建议用绝对时间戳，例如：`outputs/cpu_attn_numa/2026-03-19_1200_decode_localbind/`。
- 同目录至少保存：
  - attention metadata 日志
  - worker 绑核日志
  - uProf/PCM 指标输出
  - 运行命令与环境变量快照

## 12. 最后给老师汇报时可以用的一句话版本
- 当前 vLLM 在 x86 CPU 上的 attention，本质是“同一个 paged-KV kernel 处理 prefill、decode 和 mixed batch，内部再按 `kv_head + q tile + KV split` 切成 workitem”；它已经有部分 head-first 特征，但真正缺的是 `KV 页放置` 和 `head/workitem 到 NUMA chiplet 的稳定映射`。
- 所以 `GPU_Attention_NUMA_Optimization.pdf` 的思路可以借鉴，但借鉴的不是 GPU swizzle 代码本身，而是“让共享同一组 K/V 的计算单元和数据页尽量待在同一 NUMA 域”。
- 对 CPU 来说，这更像 `KV allocator + thread-cluster scheduling + mixed batch 组织` 的联合优化问题；最先该验证的目标场景是长 context decode。

## 证据索引
- CPU attention Python 入口：`vllm/v1/attention/backends/cpu_attn.py`
- batch 与 metadata 准备：`vllm/v1/worker/cpu_model_runner.py`、`vllm/v1/worker/gpu_model_runner.py`
- block table / slot mapping：`vllm/v1/worker/block_table.py`
- scheduler：`vllm/v1/core/sched/scheduler.py`
- KV cache 容量与 rank 对齐：`vllm/v1/core/kv_cache_utils.py`
- CPU platform / KV 默认空间 / block_size / 可见 NUMA node：`vllm/platforms/cpu.py`
- worker NUMA/绑核：`vllm/v1/worker/cpu_worker.py`、`csrc/cpu/utils.cpp`、`csrc/cpu/utils.hpp`
- CPU attention C++ 入口：`csrc/cpu/cpu_attn.cpp`
- CPU attention scheduler / main loop：`csrc/cpu/cpu_attn_impl.hpp`
- K/V pack 布局：`csrc/cpu/cpu_attn_vec.hpp`
- attention 微基准：`benchmarks/kernels/cpu/benchmark_cpu_attn.py`
- attention 测试 shape 模板：`tests/kernels/attention/test_cpu_attn.py`
- batch split 测试：`tests/v1/attention/test_attention_splitting.py`
- Qwen3 MoE attention TP 切分：`vllm/model_executor/models/qwen3_moe.py`
- TP linear 通信边界：`vllm/model_executor/layers/linear.py`
- 当前 TP backend 证据：`test_results/qwen3_tp2_init_2026-03-18_2338.log`
- 对照论文：`info/papers/GPU_Attention_NUMA_Optimization.pdf`
