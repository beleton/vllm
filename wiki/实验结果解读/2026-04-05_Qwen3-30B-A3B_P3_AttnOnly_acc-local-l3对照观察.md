# 2026-04-05 Qwen3-30B-A3B P3_AttnOnly acc-local-l3 对照观察

> 本文已按 `2026-04-20` 当前代码与最新结果重写。
> 结果文件：`test_results/P3_AttnOnly/Qwen3-30B-A3B/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1/summary.csv`
> 当前对比口径：`balanced` vs `acc-local-l3`
> 当前实验参数：`NPS1_TP2 / prefill-like / global-fixed / batch=1 / qhead=32 / kvhead=16 / group_span=1 / q=kv in {64,128,256,512,1024,2048,4096}`

## 问题

- 在当前 `batch=1 + kvhead=16 + span=1` 口径下，`acc-local-l3` 相对 `balanced` 是否已经带来稳定收益。

## Dry-run latency

| q_len | kv_len | balanced slowest rank mean (ms) | acc-local-l3 slowest rank mean (ms) | acc 相对 balanced 变化 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 0.026344 | 0.028935 | +9.84% |
| 128 | 128 | 0.028220 | 0.029001 | +2.77% |
| 256 | 256 | 0.056132 | 0.053291 | -5.06% |
| 512 | 512 | 0.118892 | 0.118152 | -0.62% |
| 1024 | 1024 | 0.376292 | 0.364960 | -3.01% |
| 2048 | 2048 | 1.432343 | 1.864542 | +30.17% |
| 4096 | 4096 | 6.738855 | 7.844426 | +16.41% |

## PCM 对照

- `q64/q128`：
  - `acc-local-l3` 更慢。
  - `L3 Access (pti)` 更低：`1.32 -> 1.19`、`2.00 -> 1.58`
  - `L3 Miss (pti)` 更低：`1.20 -> 1.13`、`1.64 -> 1.53`
  - `Ave L3 Miss Latency (ns)` 更低：`222.92 -> 169.65`、`224.02 -> 192.60`

- `q256/q512/q1024`：
  - `acc-local-l3` 分别快 `5.06% / 0.62% / 3.01%`。
  - `L3 Access (pti)` 明显更低：`2.67 -> 1.37`、`3.36 -> 1.20`、`2.91 -> 1.08`
  - `L3 Miss (pti)` 在 `q256` 更低，在 `q512` 持平，在 `q1024` 略高：`1.31 -> 1.29`、`0.81 -> 0.81`、`0.45 -> 0.47`
  - `IPC (Sys + User)` 略高：`1.65 -> 1.69`、`2.25 -> 2.29`、`2.62 -> 2.64`

- `q2048/q4096`：
  - `acc-local-l3` 明显更慢：`+30.17% / +16.41%`
  - `L3 Access (pti)` 仍更低：`2.04 -> 1.69`、`1.30 -> 1.24`
  - `L3 Miss (pti)` 仍更低：`0.31 -> 0.25`、`0.35 -> 0.30`
  - `IPC (Sys + User)` 明显更低：`2.67 -> 2.23`、`1.94 -> 1.77`
  - `Ave L3 Miss Latency (ns)` 没有变差：`198.09 -> 203.93`、`167.34 -> 161.54`

## 分析

- 当前 sweep 下，`acc-local-l3` 没有形成稳定收益。
- 收益区间只出现在 `q256/q512/q1024`，其中 `q512` 的优势很小，基本接近持平。
- `q64/q128` 与 `q2048/q4096` 上，`acc-local-l3` 都慢于 `balanced`。
- `q2048/q4096` 的慢点没有伴随更高的 `L3 Access (pti)` 或 `L3 Miss (pti)`；这份 summary 只能支持“慢点与更低 IPC 同时出现”，不能支持“慢点来自更差的 L3 locality”。
- 当前这组数据更接近这样的事实：`acc-local-l3` 在一部分中等长度上把 locality 改善转成了 latency 收益，但在更短和更长的长度点上，这种收益都不稳定。

## 容量前提

- 当前代码路径里，`causal prefill` 的后续 `q_tile` 会继续访问更长的 `KV` 前缀；`KV` 数据在 `execute_attention()` 中直接从 `key_cache/value_cache` 读取，同一前缀会被后续 `q_tile` 重访。
- 对当前 `head_dim=128`、`bf16` 口径，单个 `kv_head` 的 `K+V` 容量是 `512 B / token`。同一 `CCD` 若同时需要复用 `n` 个 `kv_head` 的长前缀，可复用 `KV` 容量压力可近似写成 `n * 512 B / token * kv_len`。
- 若构造数据使 `acc-local-l3` 在同一 `CCD` 上主要复用 `1` 个 `kv_head`，而 `balanced` 在同一 `CCD` 上复用多个 `kv_head`，则在“单个 `kv_head` 前缀仍可落在本地 `L3`、多个 `kv_head` 聚合前缀已超过本地 `L3`”的长度区间里，`balanced` 更容易在后续 `q_tile` 重访旧前缀时发生本地 `L3` 容量不足，进而增加跨 `CCD` 或内存供数；`acc-local-l3` 理论上更有机会减少这类访问。
- 这一段只代表当前代码与容量模型支持的理论预期，不代表本页已有实验已经验证该预期。

## 与前一版结论的区别

- 前一版文档记录的是另一组实验：`batch=16 / kvhead=4 / span4`。
- 前一版结论是：
  - `acc-local-l3_span4` 在 `q64/q128/q256` 上优于 `balanced_span4`
  - 到 `q512` 时 locality 指标更好，但 latency 仍落后
- 当前这组实验是：`batch=1 / kvhead=16 / span1`
- 当前结论变为：
  - `q64/q128` 没有转成收益，反而慢于 `balanced`
  - `q256/q512/q1024` 才出现收益
  - `q2048/q4096` 又出现明显回退

因此，旧文档中“较短 prefill 已经转正、只在 `q512` 落后”的结论，不适用于当前口径。两组结果唯一一致的部分是：`acc-local-l3` 往往能压低一部分 locality 指标，但这种改善并不会稳定转成端到端 latency 收益。

## Idle baseline

- idle baseline 命令是 `AMDuProfPcm profile -m ipc,l3,dc -a -s -d 40 -I 200 -- sleep 50`，没有 `start-delay`，有效采样窗是 `40s`。
- 当前这组 `batch=1` attention PCM 命令使用了两种配置：`--start-delay 30000 -d 90` 和 `--start-delay 20000 -d 80`。实验步骤文档已明确 `-d` 从 profiler 启动开始计时并包含 `start-delay`，因此这两类 case 的有效采样窗都应按 `60s` 计算。
- idle 与 attention 的 L3 压力对照，应优先看 raw count 或按有效采样窗折算后的 `/s`，不能只看 `pti`。

| case | 有效采样窗 | L3 Access/s | L3 Miss/s |
| --- | ---: | ---: | ---: |
| idle | 40s | 11.46M | 9.30M |
| q2048 balanced | 60s | 3412.29M | 348.69M |
| q2048 acc-local-l3 | 60s | 2413.68M | 272.74M |
| q4096 balanced | 60s | 1616.09M | 182.99M |
| q4096 acc-local-l3 | 60s | 1412.21M | 126.42M |

- idle 的 `L3 Miss (pti)` 与 workload 接近，不能推出 attention 期间没有发生很多 `L3 Miss`。
- 按 raw count 和 `/s` 看，attention 期间的 `L3 Access` 与 `L3 Miss` 都显著高于 idle。
- idle 报告里的系统 `Utilization` 仍有 `8.85%`，它是背景活动基线，不是零活动基线。

## 结论

- 在当前 `batch=1 / kvhead=16 / span1` 口径下，`acc-local-l3` 没有稳定优于 `balanced`。
- 当前最好的区间是 `q256/q512/q1024`，但收益幅度有限。
- 当前最大的退化点是 `q2048/q4096`。
- 仅凭这份 `summary.csv`，不能把 `q2048/q4096` 的退化归因到更差的 L3 locality；能直接看到的是，这两个长度点上 `acc-local-l3` 更慢，同时 `IPC` 更低，而 `L3 Access/Miss` 没有更高。

## 边界

- 当前只覆盖 `prefill-like`
- 当前只覆盖 `NPS1_TP2`
- 当前只覆盖 `global-fixed`
- 当前只覆盖 `batch=1`
- 当前只覆盖 `qhead=32 / kvhead=16`
- 当前只覆盖 `group_span=1`
- 当前只覆盖 `q=kv in {64, 128, 256, 512, 1024, 2048, 4096}`
