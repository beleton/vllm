# prefill / decode 执行分流

## 问题
- 在当前 CPU attention 路径里，prefill 和 decode 是否走两套不同算子；如果不是，它们的主要差异来自哪里。

## 结论
- `CPUModelRunner` 没有重写整套 prefill / decode 执行框架；CPU 路径仍复用 `vllm/v1/worker/gpu_model_runner.py` 的 `_update_states`、`_may_reorder_batch`、`_prepare_inputs`、`_build_attention_metadata`、`execute_model`。
- 对 CPU decoder attention，真正的计算入口是 `vllm/v1/attention/backends/cpu_attn.py::CPUAttentionBackendImpl.forward` 中的 `ops.cpu_attention_with_kv_cache(...)`。
- `_may_reorder_batch` 只是按 attention backend 的要求重排 batch 内请求顺序，不做 attention 计算，也不改变 scheduler 已分配的 token 数。
- `_prepare_inputs` 负责把 `scheduler_output.num_scheduled_tokens` 展平为连续 token 视图，并准备 `input_ids`、`positions`、`slot_mapping`、`query_start_loc`、`seq_lens` 等 attention 输入；它本质上是数据整理，不是 attention 算子。
- 在 x86 CPU backend 默认路径上，`_CPU_ARCH_PREFER_MIXED_BATCH = (X86, ARM)`，因此 `reorder_batch_threshold` 默认通常是 `None`，`use_sdpa_prefill=False`。
- 因而在当前 AMD EPYC x86 场景里，decoder attention 的 prefill 和 decode 通常都会进入同一个 CPU 自定义算子 `cpu_attention_with_kv_cache`；差异主要体现在 `query_start_loc`、`seq_lens`、`slot_mapping`、`block_table`、`scheduler_metadata` 的内容。
- CPU attention 内部对 prefill / decode 的真正差异，更多来自“query token 数是否很小”和“KV 长度是否很长”：源码里 `enable_kv_split=True` 时，只有 `q_tile_token_num <= 1` 的短 query tile 才允许做 split-KV，这本质上更偏向 decode。

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

## `_may_reorder_batch`
- 生效条件：只有 `self.reorder_batch_threshold is not None` 时才会重排
- 真正重排规则在 `vllm/v1/attention/backends/utils.py::reorder_batch_to_split_decodes_and_prefills`
- 目标顺序是：`decode -> extend -> prefill`
- 对当前 x86 CPU 默认路径，这一重排通常不发生

## `_prepare_inputs`
- 关键工作包括：
  - 用 `num_scheduled_tokens` 生成每个 token 对应的 `req_indices`
  - 生成本轮 token 的 `positions`
  - 抽取本轮 `input_ids`
  - 计算 `slot_mapping`
  - 构造 `query_start_loc`
  - 构造 `seq_lens`
- 它是从 scheduler 视角到 kernel 视角的数据转译，不是 attention 本体

## `CPUAttentionBackendImpl.forward`
- decoder / encoder-decoder attention 的主流程：
  1. `ops.cpu_attn_reshape_and_cache(...)`
  2. `ops.cpu_attention_with_kv_cache(...)`
- x86 CPU 默认 `use_sdpa_prefill=False`，所以 mixed batch 中的 decoder attention 一般整体走 `cpu_attention_with_kv_cache`

## `scheduler_metadata`
- 构建位置：`CPUAttentionMetadataBuilder.build`
- 调用：`ops.cpu_attn_get_scheduler_metadata(...)`
- 它负责：
  - 根据 `seq_lens` / `query_start_loc` 推导 query 和 KV 长度
  - 根据 ISA、`head_dim`、KV block 对齐要求、L2 cache 容量推导 tile 大小
  - 生成 `AttentionWorkItemGroup` / `ReductionWorkItemGroup`
  - 规划 split-KV 与 reduction

## 分析
- 对当前 x86 CPU 路径，prefill / decode 更像“同一 attention 主算子 + 不同 metadata / 不同访存形态”，而不是“两套完全不同的 CPU attention kernel”。
- decode 更容易表现出 split-KV 和长 KV 读取带来的特征，因为只有 `q_tile_token_num <= 1` 的短 query tile 才允许做 split-KV。
- 若出现 `L3 miss` 或 `Remote DRAM Reads` 偏高，优先该看的是 `block_table` 指向的 paged KV cache 访问、长上下文 KV 流式读取，以及 mixed batch 下不同请求 `seq_len` 差异，而不是 `_may_reorder_batch` 本身。

## 证据
- 关键源码路径：
  - `vllm/v1/worker/gpu_model_runner.py`
  - `vllm/v1/attention/backends/cpu_attn.py`
  - `vllm/v1/attention/backends/utils.py`
  - `csrc/cpu/torch_bindings.cpp`
  - `csrc/cpu/cpu_attn.cpp`
  - `csrc/cpu/cpu_attn_impl.hpp`

## 边界
- 本页只看 `CPUModelRunner -> CPUAttentionBackendImpl -> cpu_attention_with_kv_cache` 这条 CPU attention 主路径。
- 它不展开 encoder-only / encoder attention 的 SDPA 分支，也不展开更细的 tile/workitem/runtime task 公式。

## 后续验证点
- 若要继续解释 prefill / decode 差异，优先抓：
  - `scheduler_output.num_scheduled_tokens`
  - `query_start_loc`
  - `seq_lens`
  - `scheduler_metadata` 中的 split / reduction 数量
- 若要验证“prefill 与 decode 在 CPU 上是否真的共用同一算子”，直接在 `CPUAttentionBackendImpl.forward` 附近打印 `use_sdpa_prefill`、`num_decode_tokens`、`num_actual_tokens`

