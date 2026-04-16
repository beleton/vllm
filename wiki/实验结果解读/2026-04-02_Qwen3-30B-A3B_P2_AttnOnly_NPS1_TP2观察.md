# 2026-04-02 Qwen3-30B-A3B P2_AttnOnly NPS1_TP2 观察

> 来源汇总文档：`test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md`

## 问题
- `P2_AttnOnly/NPS1_TP2` 首轮 attention-only 结果，是否已经支持把“按共享 K/V 工作集做拓扑感知放置”当成 prefill 的候选优化方向。

## 当前覆盖范围
- `NPS1_TP2`
- `global-fixed`
- `batch=16`
- `prefill-like q=kv=128/256/512/1024/2048`
- `decode-like q=1, kv=256/512/1024/2048`

## Prefill-like 全量结果

| q_len | kv_len | slowest rank mean (ms) | L3 Access (pti) | L3 Miss (pti) | L3 Miss % | Ave L3 Miss Latency (ns) | same-node another CCX % | local memory / I/O % | remote memory / I/O % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 0.1927 | 4.51 | 1.59 | 35.20 | 432.37 | 99.13 | 0.80 | 0.03 |
| 256 | 256 | 0.4480 | 3.19 | 1.27 | 39.86 | 199.23 | 94.00 | 5.83 | 0.07 |
| 512 | 512 | 1.4921 | 1.90 | 0.70 | 37.19 | 155.80 | 61.15 | 38.53 | 0.19 |
| 1024 | 1024 | 5.5339 | 1.29 | 0.50 | 38.97 | 148.15 | 30.70 | 68.80 | 0.31 |
| 2048 | 2048 | 21.1150 | 1.27 | 0.32 | 25.37 | 135.44 | 21.95 | 77.32 | 0.44 |

## Decode-like 全量结果

| q_len | kv_len | slowest rank mean (ms) | L3 Access (pti) | L3 Miss (pti) | L3 Miss % | Ave L3 Miss Latency (ns) | same-node another CCX % | local memory / I/O % | remote memory / I/O % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 0.0417 | 15.44 | 5.70 | 36.94 | 580.66 | 99.16 | 0.75 | 0.03 |
| 1 | 512 | 0.0446 | 19.89 | 4.22 | 21.24 | 592.69 | 98.94 | 0.98 | 0.03 |
| 1 | 1024 | 0.0496 | 24.12 | 2.82 | 11.71 | 342.73 | 97.48 | 2.30 | 0.08 |
| 1 | 2048 | 0.0707 | 24.21 | 3.99 | 16.49 | 161.62 | 45.48 | 54.24 | 0.30 |

## 观察
- 在 `prefill-like/global-fixed/batch_16` 下：
  - `q=128/256/512` 时，`same-node another CCX %=99.13/94.00/61.15`
  - 同时 `Ave L3 Miss Latency=432.37/199.23/155.80 ns`
- 同一批结果里：
  - `q=1024/2048` 时，`same-node another CCX %=30.70/21.95`
  - `local memory / I/O %=68.80/77.32`
- 在 `decode-like/global-fixed/batch_16` 下：
  - `kv=256/512/1024` 时，`same-node another CCX` 仍在 `97%+`
  - 到 `kv=2048` 时，`same-node another CCX %=45.48`，`local memory / I/O %=54.24`

## 分析
- 这批 attention-only 结果已经支持一个阶段性判断：短到中等长度的 prefill 确实存在明显跨 CCX 路径占比，因此“把共享 `K/V` 工作集尽量收进同一局部域”是值得继续验证的候选方向。
- 但它同样说明，这条方向并不是对所有长度都同样有效。随着 prefill 长度增加，主导来源已经从 `another CCX` 向 `local memory / I/O` 切换；这时只靠收紧 `CCX/L3` 未必还能打中主要瓶颈。
- decode-like 结果也提示：来源切换并不只发生在 prefill；但目前 decode 侧还不能仅凭这批 attention-only 数据就推出“当前最值得优先改的是 decode locality”。
- 就长度转折看：
  - prefill 侧最值得补更密点位的是 `q=512 -> 1024`
  - decode 侧最值得补更密点位的是 `kv=1024 -> 2048`

## 证据
- 汇总表：
  - [../../test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md](../../test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md)
- 迁移关联页面：
  - [../实验方案/Qwen3-30B-A3B_attention-only_TP实验步骤.md](../实验方案/Qwen3-30B-A3B_attention-only_TP实验步骤.md)
  - [./2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md](./2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md)
  - [../源码分析/CPU_attention_acc_locality新kernel实现说明.md](../源码分析/CPU_attention_acc_locality新kernel实现说明.md)

## 结论
- 当前可以写成：
  - `NPS1_TP2` 的首轮结果支持把“按共享 `K/V` 工作集做拓扑感知放置”当作 prefill 的候选优化方向
  - `q=512 -> 1024` 是当前最值得补更密点位的转折区间
- 当前不能写成：
  - `acc-local-l3` 已经被现有数据证明可稳定实现
  - 现有结果已证明它能跨 `NPS/TP` 泛化

## 边界
- 还没有：
  - `NPS2_TP4 / NPS4_TP8`
  - `per-rank-fixed`
  - 更低或更高 batch

## 下一步
- 先补 `NPS2_TP4 / NPS4_TP8` 的 `global-fixed`
- 再补 `per-rank-fixed`
- 在 `prefill q=512 -> 1024`、`decode kv=1024 -> 2048` 之间补更密长度点
