# benchmarks 与 tests 入口筛选

## 问题
- 当前仓库里哪些 `benchmarks / tests` 入口最适合支撑两条主线：
  - 解释高 `L3 Miss` 更偏权重流式访问还是 `KV/工作集` 过大
  - 在 Chiplet CPU 上研究 attention / MoE 算子的优化空间

## 结论
- `benchmarks/benchmark_serving.py`、`benchmarks/benchmark_latency.py`、`benchmarks/benchmark_throughput.py` 都只是废弃提示，真正实现已经移到 `vllm/benchmarks/serve.py`、`vllm/benchmarks/latency.py`、`vllm/benchmarks/throughput.py`，CLI 包装在 `vllm/entrypoints/cli/benchmark/*.py`。
- 对“高 `L3 Miss` 来源”主线，`vllm bench serve` 仍是最合适的端到端主入口。
- 对 Chiplet CPU attention / MoE 优化，`benchmarks/kernels/cpu/benchmark_cpu_attn.py` 和 `benchmarks/kernels/cpu/benchmark_cpu_fused_moe.py` 更合适，因为它们直接进入 `cpu_attention_with_kv_cache` / `cpu_fused_moe`，可独立控制 `q_len`、`kv_len`、`batch_size`、`topk`、`isa` 等变量。
- `vllm bench latency` / `vllm bench throughput` 更适合做“去掉 HTTP/API server 干扰”的对照组，不适合作为当前主线的替代品。
- 当前 x86 CPU backend 默认偏向保留 mixed batch，而不是在 CPU attention 路径里显式拆成“prefill 走 SDPA / decode 走 KV attention”；因此 phase 级判断不能简单理解为两套独立 kernel。

## 值得优先使用的入口

| 类别 | 文件 | 作用 | 是否比 `bench serve` 更好 |
| --- | --- | --- | --- |
| 端到端主入口 | `vllm/benchmarks/serve.py` | 在线请求压测；支持 `--request-rate`、`--max-concurrency`、`--save-detailed`；结果里有 `ttft` / `itl` / `tpot` | 否 |
| CLI 包装 | `vllm/entrypoints/cli/benchmark/serve.py` | 证明 `vllm bench serve` 只是调用 `vllm/benchmarks/serve.py` | 否 |
| CLI 用法样例 | `tests/benchmarks/test_serve_cli.py` | 最小可运行参数组合 | 否 |
| 服务端 phase 指标 | `tests/entrypoints/instrumentator/test_metrics.py` | 证明 `/metrics` 上存在 `prefill/decode` 时间与 prompt/generation token 指标 | 不是替代，是 `bench serve` 的强补充 |
| KV residency 指标 | `tests/v1/core/test_kv_cache_metrics.py`、`vllm/v1/core/kv_cache_metrics.py`、`vllm/engine/arg_utils.py` | 已有 `KV` block lifetime / idle / reuse gap 采样链路和 CLI 参数 | 不是替代，适合第二阶段验证 |
| attention 微基准 | `benchmarks/kernels/cpu/benchmark_cpu_attn.py` | 直接跑 `cpu_attn_reshape_and_cache` / `cpu_attention_with_kv_cache`，适合 prefill/decode/mixed 受控实验 | 是，但仅对 attention 算子优化更好 |
| attention shape 模板 | `tests/kernels/attention/test_cpu_attn.py` | 已给出 decode batch / prefill batch / mixed batch 的代表性 shape | 不是替代，是参数模板 |
| attention 分相逻辑 | `tests/v1/attention/test_attention_splitting.py`、`tests/v1/attention/test_attention_backends.py`、`vllm/v1/attention/backends/utils.py` | 给出 decode/prefill 的判定与 batch 组织方法 | 不是替代，是 synthetic case 参考 |
| CPU backend 分流证据 | `vllm/v1/attention/backends/cpu_attn.py` | 说明 x86 默认不拆 `prefill/decode` 子路径，避免误把端到端 phase 当作两套 attention kernel | 否 |
| MoE 微基准 | `benchmarks/kernels/cpu/benchmark_cpu_fused_moe.py` | 直接跑 `cpu_prepack_moe_weight` / `cpu_fused_moe`，适合研究专家权重流式访问与绑核策略 | 是，但仅对 MoE 算子优化更好 |
| MoE 正确性模板 | `tests/kernels/moe/test_cpu_fused_moe.py` | 给出 `expert_num`、`hidden_size`、`intermediate_size`、`topk` 的有效组合 | 不是替代，是参数模板 |
| 理论字节/FLOPs 对照 | `tests/v1/metrics/test_perf_metrics.py`、`vllm/v1/metrics/perf.py` | 可按 prefill/decode 上下文估算 attention 读 KV 字节与 FFN/MoE 读权重字节 | 不是替代，适合实验前做理论排序 |

## 可以做对照，但不建议替代当前主线

### `vllm bench latency`
- 直接构造 `LLM`，用单批 `llm.generate(...)` 跑固定 `input_len/output_len`
- 适合做“单批、低噪声、去掉 HTTP”的控制实验
- 不适合替代当前 `bench serve` 的分相主线

### `vllm bench throughput`
- 直接走 `LLM` 或 `AsyncEngine`
- 适合做“去掉 API 层”的离线吞吐对照
- 但请求到达模型、并发排队、HTTP 序列化/反序列化都被改掉了

## 当前不值得投入的入口

### 废弃入口
- `benchmarks/benchmark_serving.py`
- `benchmarks/benchmark_latency.py`
- `benchmarks/benchmark_throughput.py`

这三个文件都只打印 “DEPRECATED: moved to the vLLM CLI”。

### GPU 专用 kernel benchmark / tests
- 例如：
  - `benchmarks/kernels/benchmark_paged_attention.py`
  - `benchmarks/kernels/benchmark_moe.py`
  - `tests/kernels/attention/test_triton_*`
  - `tests/kernels/moe/test_cutlass_*`

它们主要覆盖 CUDA / Triton / Cutlass 路径，对当前 EPYC CPU + NUMA + Chiplet 研究帮助很小。

### `benchmark_block_pool.py` / `benchmark_prefix_block_hash.py`
- 更偏 KV block 分配器和 prefix hash 算法本身
- 不是当前高 `L3 Miss` 根因分析的主矛盾

## 和当前两个研究目标的对应关系

### 目标 A：高 `L3 Miss` 来源判定
- 主入口：`vllm/benchmarks/serve.py`
- 强补充：
  - `tests/entrypoints/instrumentator/test_metrics.py` 对应的 `/metrics`
  - `vllm/v1/core/kv_cache_metrics.py` 对应的 `--kv-cache-metrics`
- 不建议替代：
  - `vllm/benchmarks/latency.py`
  - `vllm/benchmarks/throughput.py`

### 目标 B：Chiplet CPU 上 attention / MoE 优化空间
- attention：
  - `benchmarks/kernels/cpu/benchmark_cpu_attn.py`
  - `tests/kernels/attention/test_cpu_attn.py`
  - `tests/v1/attention/test_attention_splitting.py`
- MoE：
  - `benchmarks/kernels/cpu/benchmark_cpu_fused_moe.py`
  - `tests/kernels/moe/test_cpu_fused_moe.py`
- 理论建模：
  - `tests/v1/metrics/test_perf_metrics.py`
  - `vllm/v1/metrics/perf.py`

## 分析
- 若目标是判断“高 miss 更像权重流式访问还是 KV 工作集过大”，单看 `bench serve` 输出还不够；更稳妥的是同步记录：
  - `vllm:request_prefill_time_seconds`
  - `vllm:request_decode_time_seconds`
  - `vllm:request_prefill_kv_computed_tokens`
- 若目标是找 attention / MoE 的 Chiplet 优化空间，先在 CPU kernel 微基准里筛候选策略，再回到端到端复验，更省实验成本。

## 建议
1. 保持当前 `Qwen3-30B-A3B + vllm bench serve` 分相方案不变。
2. 在同一轮 `bench serve` 期间同步抓 `/metrics`。
3. 另起 attention 微基准，把 shape 映射成 decode / prefill / mixed 三类受控 case。
4. 对 MoE 先用 `benchmark_cpu_fused_moe.py` 做绑核 / NUMA / CCX 策略筛选，再回到端到端复验。

