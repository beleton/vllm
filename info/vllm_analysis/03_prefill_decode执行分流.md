# 03. prefill / decode 执行分流

> 更新时间：2026-03-18  
> 范围：只看 `CPUModelRunner -> CPUAttentionBackendImpl -> cpu_attention_with_kv_cache` 这条 CPU attention 主路径。  
> 假设：当前分析对象是 decoder-only / encoder-decoder 模型的 CPU 推理；`CPUModelRunner` 复用 `GPUModelRunner` 的批处理与 metadata 准备逻辑。

## 结论

### 事实
- `CPUModelRunner` 没有重写整套 prefill / decode 执行框架；CPU 路径仍复用 `vllm/v1/worker/gpu_model_runner.py` 的 `_update_states`、`_may_reorder_batch`、`_prepare_inputs`、`_build_attention_metadata`、`execute_model`。
- 对 CPU decoder attention 来说，真正的计算入口不是 `_may_reorder_batch` 或 `_prepare_inputs`，而是 `vllm/v1/attention/backends/cpu_attn.py::CPUAttentionBackendImpl.forward` 里的 `ops.cpu_attention_with_kv_cache(...)`。
- `_may_reorder_batch` 的作用只是“按 attention backend 的要求重排 batch 内请求顺序”；它不做 attention 计算，也不改变 scheduler 已分配的 token 数。
- `_prepare_inputs` 的作用是把本轮 `scheduler_output.num_scheduled_tokens` 展平为连续的 token 视图，并准备 `input_ids`、`positions`、`slot_mapping`、`query_start_loc`、`seq_lens` 等 attention 所需输入；它本质上是数据整理，不是 attention 算子。
- 在 x86 CPU backend 上，源码默认偏向保留 mixed batch：`vllm/v1/attention/backends/cpu_attn.py` 中 `_CPU_ARCH_PREFER_MIXED_BATCH = (X86, ARM)`，因此 `reorder_batch_threshold` 默认是 `None`，`use_sdpa_prefill=False`。
- 因而在你当前 AMD EPYC x86 场景里，decoder attention 的 prefill 和 decode 通常都会进入同一个 CPU 自定义算子 `cpu_attention_with_kv_cache`；差异主要体现在 `query_start_loc`、`seq_lens`、`slot_mapping`、`block_table`、`scheduler_metadata` 的内容，而不是 Python 层换了另一条完全不同的执行函数。
- `cpu_attention_with_kv_cache` 内部不是“直接一把算完”，而是先由 `cpu_attn_get_scheduler_metadata` 预先生成 CPU 侧调度元数据，再按 `dtype / head_dim / ISA` 分发到 C++ 模板主循环，执行 `QK -> mask / alibi / softcap / softmax -> PV -> 必要时 split-KV reduction -> 写回 output`。
- CPU attention 内部对 prefill / decode 的真正差异，更多来自“query token 数是否很小”和“KV 长度是否很长”：源码里 `enable_kv_split=True` 时，只有 `q_tile_token_num <= 1` 的短 query tile 才允许做 split-KV，这本质上更偏向 decode，而不是长 prefill。

### 推断
- 对当前 x86 CPU 路径，prefill / decode 更像是“同一 attention 主算子 + 不同 metadata / 不同访存形态”，而不是“两套完全不同的 CPU attention kernel”。
- 如果后续观察到 decode 比 prefill 更容易吃满线程，很可能不是 Python 调度层原因，而是 `cpu_attention_with_kv_cache` 内部的 split-KV 与 workitem 切分策略在起作用。
- 如果出现 L3 miss / Remote DRAM Reads 偏高，优先应怀疑 `block_table` 指向的 paged KV cache 访问、长上下文 KV 流式读取、以及 mixed batch 下不同请求 `seq_len` 差异，而不是 `_may_reorder_batch` 本身。

### 建议
- 分析 CPU attention 性能时，把关注点放在 `cpu_attention_with_kv_cache`、`cpu_attn_get_scheduler_metadata`、`block_table / slot_mapping / seq_lens / query_start_loc`，不要先把精力放在 `_may_reorder_batch` 上。
- 若要验证“prefill 与 decode 在 CPU 上是否真的共用同一算子”，直接在 `CPUAttentionBackendImpl.forward` 附近打印 `use_sdpa_prefill`、`num_decode_tokens`、`num_actual_tokens`，比只看 scheduler 层更直接。
- 若要解释 decode 与 prefill 的性能差异，建议同时抓：`scheduler_output.num_scheduled_tokens`、`query_start_loc`、`seq_lens`、`scheduler_metadata` 中的 split / reduction 数量，再和 `CPI`、`Total Mem Bw`、`Ave L3 Miss Latency`、`Remote DRAM Reads %` 对齐。

## 代码链路总览

```text
scheduler.schedule()
  -> SchedulerOutput.num_scheduled_tokens
  -> GPUModelRunner._update_states(...)
  -> GPUModelRunner._may_reorder_batch(...)
  -> GPUModelRunner._prepare_inputs(...)
  -> GPUModelRunner._build_attention_metadata(...)
  -> CPUAttentionMetadataBuilder.build(...)
      -> ops.cpu_attn_get_scheduler_metadata(...)
  -> CPUAttentionBackendImpl.forward(...)
      -> ops.cpu_attn_reshape_and_cache(...)
      -> ops.cpu_attention_with_kv_cache(...)
```

## 1. runner 层几个函数各自做什么

### 1.1 `_may_reorder_batch`

### 事实
- 位置：`vllm/v1/worker/gpu_model_runner.py::_may_reorder_batch`
- 调用时机：`_update_states(...)` 末尾，在 `self.input_batch.condense()` 之后、`refresh_metadata()` 之前。
- 生效条件：只有 `self.reorder_batch_threshold is not None` 时才会调用 `reorder_batch_to_split_decodes_and_prefills(...)`。
- 真正重排规则在 `vllm/v1/attention/backends/utils.py::reorder_batch_to_split_decodes_and_prefills`：目标顺序是 `decode -> extend -> prefill`。
- 分类依据：
  - `prefill`: `num_computed_tokens == 0`
  - `decode`: `num_scheduled_tokens <= threshold` 且非 `prefill`
  - `extend`: `num_scheduled_tokens > threshold` 且非 `prefill`

### 作用
- 它只是给 attention backend 一个“重排队形”的机会，让后端更容易把 memory-bound 的 decode 和 compute-bound 的 prefill 区分开。
- 它不参与 token 选择，不改变 `scheduler_output.total_num_scheduled_tokens`，也不改变每个请求已经被 scheduler 决定好的 `num_scheduled_tokens`。
- 它的本质是“batch 内索引重排”，不是 attention 计算，也不是 scheduler 再调度。

### 对当前 x86 CPU 的意义
- `vllm/v1/attention/backends/cpu_attn.py` 里，x86 / ARM 被放进 `_CPU_ARCH_PREFER_MIXED_BATCH`。
- 所以在 AMD EPYC x86 上，`reorder_batch_threshold` 默认通常是 `None`，`_may_reorder_batch()` 大概率不发生实际重排。
- 因此对你当前 CPU attention 分析，`_may_reorder_batch` 更适合作为“确认是否发生 batch 重排”的观察点，而不是主要性能瓶颈点。

### 1.2 `_prepare_inputs`

### 事实
- 位置：`vllm/v1/worker/gpu_model_runner.py::_prepare_inputs`
- 输入：`scheduler_output` 和按当前 batch 顺序排列的 `num_scheduled_tokens` 数组。
- 输出：`logits_indices` 与 `spec_decode_metadata`；同时就地填充 runner 内部的输入缓冲和 attention 元数据缓冲。

### 作用
- 它把“按请求组织”的调度结果，转换成“按 token 连续排布”的本轮执行输入。
- 关键工作包括：
  - 用 `num_scheduled_tokens` 生成每个 token 对应的 `req_indices`；
  - 用 `num_computed_tokens + arange` 生成本轮 token 的 `positions`；
  - 从 `token_ids_cpu` 抽取本轮 `input_ids`；
  - 计算 `slot_mapping`，把逻辑 token 位置映射到 paged KV cache 的物理 slot；
  - 构造 `query_start_loc`，描述每个 request 在扁平 token 缓冲中的起止位置；
  - 构造 `seq_lens`，描述每个 request 本轮执行后总上下文长度；
  - 记录 `discard_request_mask`，标记 chunked prefill 这类“本轮不该真正采样输出 token”的请求。

### 为什么它重要
- attention kernel 不关心“用户原来提交了几个请求对象”，它需要的是一组连续的 token buffer 和配套 metadata。
- `_prepare_inputs` 正是在做这层“从 scheduler 视角到 kernel 视角”的数据转译。
- `cpu_attention_with_kv_cache` 后面看到的 `query_start_loc`、`seq_lens`、`slot_mapping`、`block_table`，本质上都依赖这里准备好的结果。

### 它不做什么
- 它不决定本轮调度多少 token；那个决定已经在 `scheduler_output.num_scheduled_tokens` 里。
- 它不执行 attention 乘法，也不直接访问 KV cache 做 QK / PV。
- 所以它是 attention 的“入场准备函数”，不是 attention 本体。

## 2. CPU attention 的 Python 入口

### 事实
- 位置：`vllm/v1/attention/backends/cpu_attn.py::CPUAttentionBackendImpl.forward`
- encoder-only / encoder attention：直接走 `_run_sdpa_forward(...)`。
- decoder / encoder-decoder attention：
  1. 先把本轮 `key/value` 写入 paged KV cache：`ops.cpu_attn_reshape_and_cache(...)`
  2. 再调用 `ops.cpu_attention_with_kv_cache(...)`
- 若 `use_sdpa_prefill=True`，则 prefill token 先走 SDPA，decode token 再走 `cpu_attention_with_kv_cache`。
- 但在 x86 CPU 默认路径上，`use_sdpa_prefill=False`，所以 mixed batch 中的 decoder attention 一般整体走 `cpu_attention_with_kv_cache`。

### 作用分工
- `cpu_attn_reshape_and_cache`：把本轮新生成的 `K/V` 按 `slot_mapping` 写入 paged KV cache。
- `cpu_attention_with_kv_cache`：给定 query 和已经写好的 KV cache，真正完成注意力计算。

## 3. `cpu_attention_with_kv_cache` 深入分析

### 3.1 Python 层传入了什么

### 事实
在 `CPUAttentionBackendImpl.forward` 中，传给 `ops.cpu_attention_with_kv_cache(...)` 的核心参数是：
- `query[:num_actual_tokens]`
- `key_cache`, `value_cache`
- `output[:num_actual_tokens]`
- `query_start_loc`
- `seq_lens`
- `block_table`
- `scheduler_metadata`
- `causal`、`scale`、`alibi_slopes`、`sliding_window`、`softcap`

### 含义
- `query_start_loc`：扁平 token buffer 中，每个 request 的 query 起止位置。
- `seq_lens`：每个 request 当前总上下文长度。
- `block_table`：逻辑 block 到 KV cache 物理 block 的映射。
- `scheduler_metadata`：CPU attention 内部线程切分、tile 划分、split-KV reduction 计划。

这说明 `cpu_attention_with_kv_cache` 不是只拿 `Q/K/V` 做一次普通 GEMM；它依赖一整套“paged KV + per-request metadata + CPU 线程调度计划”。

### 3.2 `scheduler_metadata` 是怎么来的

### 事实
- 构建位置：`vllm/v1/attention/backends/cpu_attn.py::CPUAttentionMetadataBuilder.build`
- 调用：`ops.cpu_attn_get_scheduler_metadata(...)`
- 输入信息包括：`num_reqs`、`num_heads`、`num_kv_heads`、`head_dim`、`seq_lens`、`query_start_loc`、`causal`、`sliding_window_size`、`isa`、`enable_kv_split=True`

### 作用
`cpu_attn_get_scheduler_metadata` 不做 attention 数值计算，它做的是 CPU 侧“算子内调度”：
- 根据 `seq_lens` / `query_start_loc` 算每个 request 有多少 query token、每个 token 需要访问多长的 KV；
- 根据 ISA、head_dim、KV block 对齐要求、L2 cache 容量，推导 tile 大小；
- 把工作切成 `AttentionWorkItemGroup` 和 `ReductionWorkItemGroup`；
- 预先规划每个线程负责哪些 workitem，以及是否需要 split-KV 和后续 reduction；
- 把这些信息打包成一块 CPU tensor，作为 `scheduler_metadata` 传给 `cpu_attention_with_kv_cache`。

### 关键观察
- `split_kv_q_token_num_threshold = input.enable_kv_split ? 1 : 0`。
- 后续调度里，只有 `q_tile_token_num <= threshold` 的短 query tile 才允许做 split-KV。
- 也就是说，CPU attention 对 decode 更友好的一点，不一定来自 Python 层显式分流，而是来自算子内部：单 token query 更容易被沿 KV 维度拆分给多个线程并行处理；长 prefill query 不会这样拆。

### 3.3 C++ 绑定层做了什么

### 事实
- 绑定入口：`csrc/cpu/torch_bindings.cpp`
- 实现入口：`csrc/cpu/cpu_attn.cpp::cpu_attention_with_kv_cache`

### 作用
- 检查 tensor 维度和 stride；
- 把 PyTorch tensor 指针整理进 `AttentionInput`；
- 保存 `query_start_loc / seq_lens / block_table / scheduler_metadata` 的裸指针；
- 按 `query.scalar_type()`、`head_dim`、`input.metadata->isa` 分派到具体模板实现；
- 最终进入 `cpu_attention::AttentionMainLoop<attn_impl>`。

### 这层的意义
- 它本身不做主要数值计算，但决定了本次 attention 走哪套 ISA 模板：`AMX / VEC / VEC16 / NEON`。
- 所以如果你分析不同 CPU 架构、不同 dtype 的性能差异，这一层是 ISA 分叉点。

### 3.4 主循环到底怎么算

### 事实
`AttentionMainLoop` 的大体流程是：
1. OpenMP 拉起线程；
2. 每个线程从 `metadata.counter` 原子领取 task；
3. task 分两类：
   - attention task：处理某个 `kv_head` 上的一组 workitem；
   - reduction task：对 split-KV 的多个 partial result 做归并；
4. 对每个 workitem：
   - 根据 `query_start_loc` / `seq_lens` 算当前 request 的 query 区间和 KV 区间；
   - 根据 `block_table` 找到 paged KV cache 中真实的物理 block；
   - 把 query tile 拷到本线程 scratchpad；
   - 做 `Q @ K`；
   - 应用 `mask / sliding window / ALiBi / softcap / softmax`；
   - 再做 `P @ V`；
   - 若没有 split-KV，直接写回最终 output；若有 split-KV，先写 partial output，后续再 reduction。

### 细节拆开看
- `QK` 阶段：
  - 先根据 `block_table[block_idx]` 找到物理 `K` block；
  - 按 `blocksize_alignment` 对齐后分块做 tile GEMM；
  - 结果写到 `logits_buffer`。
- `mask / bias / softmax` 阶段：
  - `apply_mask` 根据 causal / sliding window 把非法位置置为 `-inf`；
  - `apply_alibi_slopes` 给每个 head 加相对位置 bias；
  - `apply_softcap` 在需要时做 logits soft cap；
  - `apply_softmax` 用 online softmax 形式维护 `max_buffer` 和 `sum_buffer`。
- `PV` 阶段：
  - 把 softmax 后的概率和 `V` cache 做 tile GEMM；
  - partial output 暂存在 scratchpad 的 `partial_q_buffer`。
- `split-KV reduction`：
  - 如果同一个 query tile 的 KV 被拆到多个 split，先分别写 `partial_output / partial_max / partial_sum`；
  - `reduce_splits(...)` 再按 softmax 规则重标定并合并；
  - `final_output(...)` 最终再除以 `sum_buffer`，写回输出 tensor。

### 3.5 为什么它对 CPU 性能很关键

### 事实
- 这个算子不是连续读取完整 `K/V` 矩阵，而是通过 `block_table` 访问 paged KV cache。
- 每个 workitem 的 tile 大小与切分方式显式受 L2 cache 容量、ISA、head_dim、KV block alignment 影响。
- split-KV 需要额外 partial buffer、flag、reduction，同样会增加 scratchpad 和同步开销。

### 推断
- 长上下文 decode 更像“少量 query token + 很长 KV 读取”，更容易受内存带宽、L3 miss、远端 NUMA 访问影响。
- 长 prefill 更像“query 侧工作量更大、tile 内计算更饱满”，但未必能像 decode 一样享受到 split-KV 带来的线程并行。
- mixed batch 时，不同 request 的 `seq_len`、`q_token_num`、`block_table` 稀疏性不同，会直接影响 workitem 划分均衡性和缓存行为。

## 4. 对当前 AMD EPYC x86 场景的具体理解

### 事实
- x86 CPU backend 默认 `use_sdpa_prefill=False`，`reorder_batch_threshold=None`。
- 因此在默认 CPU serve 路径里，`_may_reorder_batch()` 大概率不改变 batch 顺序。
- 对 decoder attention，prefill 和 decode 通常一起走 `cpu_attention_with_kv_cache`。

### 推断
- 你看到的 prefill / decode 差异，更应从以下几点解释：
  - `scheduler_output.num_scheduled_tokens` 导致的 query 长度差异；
  - `query_start_loc` / `seq_lens` 导致的 attention tile 形状差异；
  - decode 是否触发 split-KV；
  - `block_table` 指向的 paged KV 物理块分布是否跨 NUMA / 跨 chiplet 不友好。

## 5. 适合继续加的日志点

1. `CPUAttentionBackendImpl.forward`
   - 打 `use_sdpa_prefill`、`num_decode_tokens`、`num_actual_tokens`
2. `CPUAttentionMetadataBuilder.build`
   - 打 `seq_lens`、`query_start_loc`、`use_sdpa_prefill`
3. `cpu_attn_get_scheduler_metadata`
   - 打 `workitem_group_num`、`reduction_item_num`、`reduction_split_num`
4. `cpu_attention_with_kv_cache`
   - 打 `isa`、`num_tokens`、`num_heads`、`num_kv_heads`、`block_size`

这样拿到的日志，才可以直接对应 CPU attention 的线程划分与访存形态。
