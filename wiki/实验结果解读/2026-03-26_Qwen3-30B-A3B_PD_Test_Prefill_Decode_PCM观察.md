# 2026-03-26 Qwen3-30B-A3B PD_Test Prefill/Decode PCM 观察

> 来源汇总文档：`test_results/PD_Test/Qwen3-30B-A3B/PCm_res/2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md`

## 问题
- `PD_Test` 的分相 PCM 数据是否已经支持“decode 的 L3 压力显著高于 prefill”这一判断。

## 来源与边界
- 当前 `PCm_res` 可见 8 组 session
- 全部目录名都绑定：
  - `B=16`
  - `metric2_l3_dc_l2_memory`
- 本页所有指标都取自各自 `report-cumulative.csv` 的 `System (Aggregated)` 列
- 本页没有对 `report.json` / `report-timeseries.csv` 再做 time-filter 稳定窗裁剪，因此都是 whole-run cumulative，不是稳定窗均值
- 当前工作区未找到对应 `bench_res/result.json`，因此这里没有调度元数据可与 PCM 同步对齐

## 原始目标对照：`prefill B16/I1024/O1` vs `decode B16/I1/O1024`

| 指标 | prefill B16/I1024/O1 | decode B16/I1/O1024 | 对比 |
| --- | ---: | ---: | ---: |
| IPC | 2.26 | 0.61 | decode 为 0.27x |
| CPI | 0.44 | 1.64 | decode 为 3.73x |
| L3 Miss % | 49.43 | 96.51 | +47.08 pp |
| Ave L3 Miss Latency (ns) | 180.12 | 326.57 | +146.45 ns |
| Remote DRAM Reads % | 0.03 | 0.03 | 基本一致 |
| Total Mem Bw (GB/s) | 328.64 | 350.99 | +22.35 |
| Total Mem RdBw (GB/s) | 296.41 | 347.75 | +51.34 |
| Total Mem WrBw (GB/s) | 32.23 | 3.23 | decode 为 0.10x |

从这两组 cumulative 值看：
- `decode` 的 `CPI`、`L3 Miss %`、`Ave L3 Miss Latency` 都显著高于 `prefill`
- `decode` 明显更偏读密集
- 两组 `Remote DRAM Reads %` 都只有 `0.03`，因此当前结果不支持直接写成“远端 DRAM 读占比明显抬升”

## 全量索引表

| phase | B | I | O | IPC | CPI | L3 Miss % | Ave L3 Miss Latency(ns) | Remote DRAM Reads % | Total Mem Bw(GB/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 16 | 1 | 64 | 0.64 | 1.57 | 96.34 | 327.33 | 0.03 | 423.00 |
| decode | 16 | 1 | 128 | 0.65 | 1.53 | 96.74 | 326.19 | 0.03 | 433.84 |
| decode | 16 | 1 | 256 | 0.62 | 1.62 | 96.54 | 327.73 | 0.03 | 483.82 |
| decode | 16 | 1 | 512 | 0.61 | 1.65 | 96.35 | 328.00 | 0.03 | 459.79 |
| decode | 16 | 1 | 1024 | 0.61 | 1.64 | 96.51 | 326.57 | 0.03 | 350.99 |
| prefill | 16 | 128 | 1 | 2.06 | 0.48 | 8.82 | 287.01 | 0.03 | 93.99 |
| prefill | 16 | 512 | 1 | 2.26 | 0.44 | 14.55 | 177.99 | 0.03 | 142.69 |
| prefill | 16 | 1024 | 1 | 2.26 | 0.44 | 49.43 | 180.12 | 0.03 | 328.64 |

## Prefill 输入长度扫描

| I | O | IPC | L3 Miss % | L3 Access (pti) | same-node another CCX % | local memory / I/O % | remote memory / I/O % | Remote DRAM Reads % | Total Mem Bw | RdBw | WrBw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 2.06 | 8.82 | 23.70 | 25.42 | 70.00 | 4.67 | 0.03 | 93.99 | 81.96 | 12.03 |
| 512 | 1 | 2.26 | 14.55 | 24.48 | 11.65 | 83.16 | 5.38 | 0.03 | 142.69 | 118.31 | 24.38 |
| 1024 | 1 | 2.26 | 49.43 | 23.96 | 2.91 | 95.27 | 2.09 | 0.03 | 328.64 | 296.41 | 32.23 |

这 3 组 prefill 数据说明：
- `input_len` 从 `128 -> 512 -> 1024` 时，`Total Mem Bw` 与 `L3 Miss %` 都明显上升
- `L3 Access (pti)` 只落在 `23.70 ~ 24.48`，没有同步大幅抬升
- `prefill_B16_I128_O1` 是值得单独警惕的异常点：虽然 `L3 Miss %=8.82`，但 `Ave L3 Miss Latency=287.01 ns`，同时 `same-node another CCX %=25.42`
- `prefill_B16_I1024_O1` 的写带宽最高，但当前页不把它直接因果归到某个具体写路径

## Decode 输出长度扫描

| I | O | IPC | L3 Miss % | L3 Access (pti) | same-node another CCX % | local memory / I/O % | remote memory / I/O % | Remote DRAM Reads % | Total Mem Bw | RdBw | WrBw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0.64 | 96.34 | 17.04 | 1.35 | 98.85 | 0.08 | 0.03 | 423.00 | 419.29 | 3.72 |
| 1 | 128 | 0.65 | 96.74 | 17.46 | 1.40 | 98.81 | 0.07 | 0.03 | 433.84 | 430.06 | 3.78 |
| 1 | 256 | 0.62 | 96.54 | 17.55 | 1.46 | 98.74 | 0.07 | 0.03 | 483.82 | 480.02 | 3.79 |
| 1 | 512 | 0.61 | 96.35 | 17.51 | 1.48 | 98.73 | 0.07 | 0.03 | 459.79 | 455.94 | 3.85 |
| 1 | 1024 | 0.61 | 96.51 | 17.65 | 1.51 | 98.70 | 0.07 | 0.03 | 350.99 | 347.75 | 3.23 |

这 5 组 decode 数据说明：
- `L3 Miss %` 始终非常高，范围只有 `96.34% ~ 96.74%`
- `Ave L3 Miss Latency` 也非常稳定，范围 `326.19 ~ 328.00 ns`
- `Remote DRAM Reads %` 在 5 组 decode 里都固定为 `0.03`
- decode 全部是明显读密集：`WrBw` 只在 `3.23 ~ 3.85 GB/s`，而 `RdBw` 在 `347.75 ~ 480.02 GB/s`

## 分析
- 这批数据已经足够支持一个阶段性判断：在当前 whole-run cumulative 口径下，decode 比 prefill 更容易表现出持续高 `L3 Miss` 和高 `Ave L3 Miss Latency`。
- 但这还不足以推出“decode 的主要瓶颈已经被定位到跨 CCX 或远端 DRAM”。原因是两组对照里的 `Remote DRAM Reads %` 都只有 `0.03`，而且 decode sweep 里 `local memory / I/O %` 始终接近主导。
- 对 prefill，更合理的理解是“随着输入长度增长，来源分布在变化”，而不是简单地把它看成缩小版 decode。
- 当前结果还不能直接回答“`L3 Miss` 的来源并不主要是权重流式读取”，因为这里只给出了 phase 级累计指标，没有把 miss 归到具体函数、数据结构或访存类型。

## 证据
- 汇总表：
  - [../../test_results/PD_Test/Qwen3-30B-A3B/PCm_res/2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md](../../test_results/PD_Test/Qwen3-30B-A3B/PCm_res/2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md)
- 迁移关联页面：
  - [../实验方案/Qwen3-30B-A3B_attention-only_TP实验步骤.md](../实验方案/Qwen3-30B-A3B_attention-only_TP实验步骤.md)
  - [../资料总览/当前研究主线.md](../资料总览/当前研究主线.md)
  - [../资料总览/Chiplet架构下L3指标关注重点与高L3Miss归因边界.md](../资料总览/Chiplet架构下L3指标关注重点与高L3Miss归因边界.md)

## 结论
- 当前 `PD_Test` 的 cumulative PCM 结果支持写成：
  - decode 的 L3 压力显著高于 prefill
  - prefill 内部不同长度的访存特征差异较大
- 不支持直接写成：
  - decode 已被证明主要受远端 DRAM 驱动
  - prefill 与 decode 只是同一条趋势上的两个长度点

## 边界
- 当前结果只绑定：
  - `Qwen3-30B-A3B`
  - `B=16`
  - `metric2_l3_dc_l2_memory`
  - whole-run cumulative 口径
- 若要做更细归因，仍需要继续走稳定窗 time-filter 或函数级 profile

## 下一步
- 把 `Prefill/Decode` 的观察和 `P2_AttnOnly` 的 attention-only 结果对齐，区分“全链路分相现象”和“attention kernel 层局部性现象”
- 对 decode 继续补 `kv=1024 -> 2048` 之间更密点位

