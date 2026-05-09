# CPU attention profiling指标介绍

## 范围

- 适用 `VLLM_CPU_ATTN_PROFILE=1` 的 `profile.json`。
- 原始数据主要在 `profile_summary` 与 `rank_results[*].profile`。
- `tools/p3_attn_only/pcm_compare.py` 会把其中一部分整理到 `summary.csv`、`compare_summary.csv`、`summary.md`。

## 计时窗口

- `Scheduler ...` 指标来自 `prepare_attention_run()` 阶段的 metadata 构建时间，不计入 `slowest rank mean (ms)`。
- `Runtime ...` 指标在 warmup 后单独 reset，只统计正式迭代，不包含 warmup。
- `Scheduler ...`、`Attention Task ...`、`Execute Attention ...` 这类平均值来自顶层 `profile_summary`，是所有 rank 合并后的总量与总次数。
- `Slowest Rank ... Effective Threads` 来自最慢 rank 的 `rank_results[*].profile` 与该 rank 的 `elapsed_ms`，是尾部 rank 口径。

## summary.csv 中 profile 行的含义

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| Scheduler Metadata Avg (ns) | 每次调用 scheduler 时，构建最终 scheduler metadata 的平均耗时。 | `balanced` 下就是完整 scheduler 时间；`acc-local-l3` 下是总 scheduler 时间。 |
| Legacy Scheduler Metadata Avg (ns) | `acc-local-l3` 中 legacy metadata 构建阶段的平均耗时。 | `balanced` 没有这项。 |
| Locality Metadata Build Avg (ns) | `acc-local-l3` 中在 legacy metadata 基础上补 locality 映射信息的平均耗时。 | `acc-local-l3` 下通常满足 `Legacy + Locality Build ≈ Scheduler Metadata Avg`。 |
| Attention Task Body Avg (ns) | 每个 attention task 的平均总耗时。 | 包含 task 内部的 tile 循环、partial/final output 写回等，不只算 `execute_attention`。 |
| Execute Attention Avg (ns) | 每次 `execute_attention(...)` 调用的平均耗时。 | 更接近真正的 attention 计算时间。它与 `Attention Task Body Avg (ns)` 的分母不同，不能直接做差。 |
| Attention Task Non-execute (%) | `attention_task_body_ns` 中不属于 `execute_attention_ns` 的比例。 | 偏高表示 task body 内部的非计算部分占比更大。 |
| Slowest Rank Attention Effective Threads | 最慢 rank 上 `attention_task_body_ns / elapsed_ns`。 | 表示正式计时窗口内，attention task body 的等效平均并行度。不是实际线程数，而是累计忙碌时间折算出的平均并行度。 |
| Slowest Rank Execute Effective Threads | 最慢 rank 上 `execute_attention_ns / elapsed_ns`。 | 表示正式计时窗口内，真正落在 `execute_attention` 里的等效平均并行度。 |
| Scheduler Metadata Avg Delta (%) | `(acc-local-l3 - balanced) / balanced * 100%`。 | 正值表示 `acc-local-l3` 的 scheduler 开销更高。 |
| Attention Task Body Avg Delta (%) | `(acc-local-l3 - balanced) / balanced * 100%`。 | 正值表示单个 attention task 总体更慢。 |
| Execute Attention Avg Delta (%) | `(acc-local-l3 - balanced) / balanced * 100%`。 | 正值表示单次 `execute_attention` 更慢。 |
| Slowest Rank Attention Effective Threads Delta (%) | `(acc-local-l3 - balanced) / balanced * 100%`。 | 负值表示 `acc-local-l3` 的有效并行度更低。 |
| Slowest Rank Execute Effective Threads Delta (%) | `(acc-local-l3 - balanced) / balanced * 100%`。 | 负值表示 `acc-local-l3` 在真正计算阶段的有效并行度更低。 |

## 读取注意事项

- `-` 表示当前 shape 没有 `profile.json`，或该 mode 不提供这个字段。
- `Legacy Scheduler Metadata Avg (ns)`、`Locality Metadata Build Avg (ns)` 只会出现在 `acc-local-l3`。
- 若 `Execute Attention Avg (ns)` 变化很小，但 `Slowest Rank ... Effective Threads` 明显下降，主信号更偏向并行度下降，而不是单次 attention 计算本身变慢。
