# 2026-05-03 Qwen3-30B-A3B P4_AttnOnly L3驻留实验一分析

结果文件：`test_results/P4_AttnOnly_L3Residency/Qwen3-30B-A3B/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1/span1/l3_task_metrics.csv`

实验口径：`NPS1_TP2 / prefill-like / global-fixed / batch=1 / qhead=32 / kvhead=16 / balanced vs acc-local-l3 / q=kv in {256,512,1024,2048,4096,8192,16384,32768,65536}`

## 结论

- 当前 `cpu attention` 上，优化跨 `CCD` 访问不是主要瓶颈。
- `acc-local-l3` 在 `q8192+` 已明显降低 `L3 Miss / attention task`，但端到端 runtime 仍慢 `8%~11%`，说明“减少 L3 miss 数量”没有转成关键路径收益。
- 标准 `PCM` 显示，`Demand DC Fills From another CCX in same node` 全程都很小，`q2048+` 仅约 `0.00~0.07 pti`，占 `All Demand DC Fills` 的比例约 `0%~0.8%`；主体 demand fill 一直是本地 `L2`。
- 因此，若 `acc` 的核心假设是“跨 `CCD` 访问是主要瓶颈，只要把 `kv_head` 收进本地 `L3` 就会稳定提速”，这组数据不支持该假设。

## 关键现象

### runtime 与 `L3 Miss / task`

| q_len | balanced runtime (ms) | acc runtime (ms) | acc变化 | balanced `L3 Miss / task` | acc `L3 Miss / task` | acc变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 118.221 | 113.711 | -3.81% | 819.25 | 922.88 | +12.65% |
| 512 | 245.248 | 249.541 | +1.75% | 1144.81 | 1299.98 | +13.55% |
| 1024 | 748.119 | 746.779 | -0.18% | 2146.34 | 2488.74 | +15.95% |
| 2048 | 850.732 | 1128.438 | +32.64% | 3833.88 | 4327.09 | +12.86% |
| 4096 | 4003.734 | 4752.395 | +18.70% | 10254.83 | 9464.04 | -7.71% |
| 8192 | 17857.676 | 19892.239 | +11.39% | 51426.87 | 24137.74 | -53.06% |
| 16384 | 30470.914 | 33447.592 | +9.77% | 268554.52 | 153566.39 | -42.82% |
| 32768 | 124768.187 | 136111.839 | +9.09% | 2127041.35 | 1539152.42 | -27.64% |
| 65536 | 506282.569 | 547830.065 | +8.21% | 10923086.67 | 9045469.04 | -17.19% |

- `q256~q2048`：`acc-local-l3` 没有压低 `L3 Miss / task`，`q2048` 还同时出现最大退化。
- `q4096`：`L3 Miss / task` 开始下降，但 runtime 仍更慢。
- `q8192+`：`L3 Miss / task` 已显著下降，但 runtime 继续稳定落后。

这意味着 `acc-local-l3` 在长序列上确实改变了 cache 行为，但这个变化没有落到最终瓶颈。

### 跨 `CCD` demand fill 占比

| q_len | mode | IPC | All Demand DC Fills (pti) | Local L2 (pti) | another CCX same node (pti) | remote share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | balanced | 2.72 | 8.62 | 8.32 | 0.07 | 0.81% |
| 2048 | acc-local-l3 | 2.26 | 7.90 | 7.64 | 0.05 | 0.63% |
| 4096 | balanced | 2.43 | 7.88 | 7.68 | 0.03 | 0.38% |
| 4096 | acc-local-l3 | 2.19 | 7.35 | 7.16 | 0.03 | 0.41% |
| 8192 | balanced | 2.26 | 7.25 | 7.10 | 0.01 | 0.14% |
| 8192 | acc-local-l3 | 2.13 | 6.85 | 6.70 | 0.01 | 0.15% |
| 16384 | balanced | 2.19 | 6.80 | 6.68 | 0.01 | 0.15% |
| 16384 | acc-local-l3 | 2.10 | 6.67 | 6.55 | 0.01 | 0.15% |

- `q2048+` 的 demand fill 主体始终是 `Local L2`。
- 跨 `CCD` demand fill 份额从未接近主导项。
- `acc-local-l3` 把这部分再压低，也只是在很小的底数上做微调。

### IPC 变化

- `q2048`：`IPC 2.72 -> 2.26`
- `q4096`：`IPC 2.43 -> 2.19`
- `q8192`：`IPC 2.26 -> 2.13`
- `q16384`：`IPC 2.19 -> 2.10`
- `q65536`：`IPC 2.09 -> 1.94`

长序列区间里，`acc-local-l3` 更慢几乎总伴随更低 `IPC`。当前更像是：

- 任务绑定后削弱了 `balanced` 的全局补空能力；
- 不同 `kv_head/subgroup` 的完成时间更不均衡；
- 或共享执行资源、调度效率下降盖过了 locality 收益。

仅凭这组数据，不能把慢点归因为 `L3` 不够本地，反而更接近“locality 变好，但执行效率变差”。

## 判断

- 对当前 `NPS1_TP2 + prefill-like + batch=1 + kvhead=16`，跨 `CCD` 访问不是 `cpu attention` 的主瓶颈。
- `acc-local-l3` 的 `L3` 局部优化不是完全无效；它在 `q4096+` 已经能压低 `L3 Miss / task`。
- 但如果目标是提升端到端 attention latency，这个优化在当前口径下没有证明自身有意义，因为它没有稳定转成性能收益。
- 若后续只围绕“继续压跨 `CCD` 访问”投入实现复杂度，预期收益很有限。

## 后续重点

- 优先继续看 `resctrl llc_occupancy` 和 `CAT` 结果，确认 `acc-local-l3` 是否真的形成了更高本地 `L3` 驻留，以及这种驻留是否值得。
- 若继续优化 `acc`，优先检查 `subgroup` 固定绑定后的负载均衡、抢任务补空能力、以及 `IPC` 下降原因，而不是先假设跨 `CCD` 访问仍是主矛盾。

## 适用边界

- 当前结论只绑定 `Qwen3-30B-A3B`
- 当前结论只绑定 `NPS1_TP2`
- 当前结论只绑定 `prefill-like`
- 当前结论只绑定 `global-fixed / batch=1 / qhead=32 / kvhead=16 / span=1`
- 当前结论不能直接外推到 `decode-like`、更大 batch、其他 `NPS/TP` 或真实服务全链路
