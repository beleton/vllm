# 研究简报（vLLM CPU / NPS-TP 根因分析）

## 1. 研究问题定义

核心问题：

- 在双路 AMD EPYC 9745 平台上，为什么 `NPS2_TP4` 优于 `NPS1_TP2`，但 `NPS4_TP8` 反而明显退化？

当前默认基线：

- 模型：`DeepSeek-R1-Distill-Llama-8B`
- 关键对比点：`concurrency=16`
- 主要证据源：
  - `test_results/DeepSeek-R1-Distill-Llama-8B/benchmark_latest_summary.csv`
  - `test_results/DeepSeek-R1-Distill-Llama-8B/pcm_cumulative_system_by_nps.csv`
  - `test_results/DeepSeek-R1-Distill-Llama-8B/pcm_l3_metrics_conc16_system.csv`

## 2. 当前结论快照（并发 16）

| 配置 | Avg TTFT(s) | Avg TPOT(s) | Output tok/s | Mem BW(GB/s) | CPI | Ave L3 Miss Latency(ns) | Remote DRAM Reads % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NPS1_TP2` | 5.5891 | 0.0755 | 197.5150 | 271.03 | 1.73 | 347.45 | 0.03 |
| `NPS2_TP4` | 4.8408 | 0.0717 | 209.4394 | 302.40 | 1.72 | 388.43 | 0.12 |
| `NPS4_TP8` | 10.2484 | 0.0945 | 153.2066 | 231.04 | 2.37 | 441.73 | 0.43 |

快照结论：

- `NPS2_TP4` 在吞吐与时延上均优于 `NPS1_TP2`。
- `NPS4_TP8` 出现显著退化，且伴随 `CPI` 上升、`L3 miss latency` 上升、`Remote DRAM Reads %` 上升。

## 3. 主要假设列表

- `H1`：`NPS4_TP8` 下跨 Chiplet/远端访存占比升高，导致 L3 miss 延迟和 CPI 恶化，最终拉低 token 吞吐。
- `H2`：`TP=8` 在该模型与该平台上的进程间通信/同步开销超过计算并行收益，导致收益反转。
- `H3`：当前实验多使用 `bfloat16`，与外部结论（提到 `float16`）口径不一致，导致复现差异。

## 4. 与助手协作约定

后续所有分析默认使用“科研四段式”输出：

1. 结论
2. 证据（必须带路径）
3. 机制解释（源码执行路径 + NUMA/Chiplet）
4. 下一步实验（命令级）

参考入口：

- [vllm实验步骤.md](./vllm实验步骤.md)
