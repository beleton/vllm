# AMDuProf 指标与使用

当前项目最常用的 AMDuProfCLI / AMDuProfPcm 指标语义和常见坑。当前机器：AMD EPYC 9745，family `0x1a`。

## 解释规则

- **PTI**：分母是退休指令，不是 dispatch/issue 指令。
- **DC Fills**：回填到 L1D 的 cache line fill 次数，不是 load/store 指令数。`All/Demand/HwPf/SwPf` 是按触发类型的子集。`From XXX` 表示回填的响应源。
- **L3 Miss Latency From XXX (%)**：延迟占比，不是 miss 次数占比。
- `report-cumulative.csv` 主要读取 `System (Aggregated)` 列。

## 通用指标

| 指标 | 含义 |
| --- | --- |
| CPI (Sys + User) | 总体每条指令平均周期数，越低越好。 |
| IPC (Sys + User) | 总体每周期退休指令数，越高越好。 |
| Eff Freq (MHz) | 有效工作频率。 |
| System time (%) | 内核态时间占比，过高可能说明系统调用/调度开销大。 |
| Utilization (%) | CPU 利用率。 |

## 内存互连指标

| 指标 | 含义 |
| --- | --- |
| Total Mem Bw (GB/s) | 总内存带宽（DF 统计）。 |
| Total Mem RdBw / WrBw (GB/s) | 总内存读/写带宽。 |
| Local Inbound Read Data Bytes (GB/s) | 从本地内存或互连读入的带宽。 |
| Remote Inbound Read Data Bytes (GB/s) | 从远端节点读入的带宽，越低越好。 |
| Remote DRAM Reads % | 远端 DRAM 读占比，越低越好。 |

## L3 与数据来源（核心）

`DC Fills` 统计回填到 L1D 的 cache line fill。`Demand DC Fills From XXX` 可近似理解为 demand load 在 L1D miss 后最终从 XXX 返回。

| 指标 | 含义 |
| --- | --- |
| All DC Fills (pti) | 所有 data-cache fill（含 demand、HW prefetch、SW prefetch）。 |
| All Demand DC Fills (pti) | 仅 demand load 触发的 fill。 |
| L3 Access (pti) | 每千条指令的 L3 访问数。 |
| L3 Miss % / L3 Miss (pti) | L3 miss 比例和每千条指令 miss 数。 |
| Ave L3 Miss Latency (ns) | L3 miss 平均延迟。 |
| Demand DC Fills From Local L2 (pti) | demand fill 响应源为本地 L2。越高越好。 |
| Demand DC Fills From Local L3 or different L2 in same CCX (pti) | demand fill 响应源为同 CCX 的 L3/其他核 L2。 |
| Demand DC Fills From another CCX in same node (pti) | demand fill 响应源为同节点其他 CCX。越低越好。 |
| Demand DC Fills From Local Memory or I/O (pti) | demand fill 响应源为本地内存/IO。 |
| Demand DC Fills From Remote memory or I/O (pti) | demand fill 响应源为远端内存/IO。越低越好。 |
| DC Fills From XXX (pti) | 同上但包含所有触发类型（All = Demand + HwPf + SwPf）。 |
| L3 Miss Latency From another CCX in same node (%) | 跨 CCX 路径在总 miss 延迟中的代价占比，不表示访问量占比。 |
| L3 Miss Latency From Local Memory or I/O (%) | 本地内存路径在总 miss 延迟中的占比。 |
| L3 Miss Latency From Remote Memory or I/O (%) | 远端内存路径在总 miss 延迟中的占比。 |

## 跨 CCX 分析口径

研究跨 CCX 访问行为时，优先看访问量，不优先看延迟占比：

1. 主指标：**Demand DC Fills From another CCX in same node (pti)**——直接描述 demand 访问有多少次从同节点其他 CCX 回填。
2. 对照指标：`Demand DC Fills From Local L3 or different L2 in same CCX (pti)`、`Demand DC Fills From Remote memory or I/O (pti)`——判断访问是停在本地 CCX、跨 CCX、还是外溢到更远层级。
3. 代价指标：`L3 Miss Latency From another CCX in same node (%)` + `Ave L3 Miss Latency (ns)`——判断跨 CCX 访问是否真的拖慢执行。

判断"跨 CCX 行为是否变多"看 demand fill 量；判断"是否拖慢执行"再结合延迟占比。

## L2 与预取

| 指标 | 含义 |
| --- | --- |
| L2 Access (pti) / L2 Hit (pti) / L2 Miss (pti) | L2 访问/命中/未命中。 |
| HwPf DC Fills From XXX (pti) | 硬件预取触发的 fill，可按来源拆分。用于判断预取行为是否拉来过多远端数据。 |
| SwPf DC Fills From XXX (pti) | 软件预取触发的 fill，同上。 |

## Topdown

| 指标 | 含义 |
| --- | --- |
| Retiring | 完成有用工作的 slot 占比，越高越好。 |
| Frontend_Bound | 前端供给不足造成的停顿占比。 |
| Backend_Bound.Memory | 访存造成的后端停顿占比。 |
| Bad_Speculation | 错误推测浪费的 slot 占比。 |

## 常用命令

```bash
# 系统信息
AMDuProfCLI info --system

# 列出可用事件
AMDuProfCLI info --list pmu-events

# PCM profile（L3 指标）
AMDuProfPcm profile -m l3 -a -d 140 -I 100 -O <output_dir> -- <workload>

# IBS L3-miss 采样
AMDuProfCLI collect -e event=ibs-op,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp --start-delay <s> --duration <s> \
  --output-dir <dir> -- <workload>
```

## 常见坑

- 从 PDF 复制命令时容易混入 Unicode 横杠，命令里应统一使用 ASCII `-`。
- `L3 Miss Latency From XXX (%)` 是延迟占比，不要直接解释成"该来源的 miss 次数占比"。
- `--pid/--tid` 只适用于 Core metrics，L3 source latency 只能按硬件域采集，不能按单个线程归属。
- 列事件写法是 `AMDuProfCLI info --list pmu-events`。
