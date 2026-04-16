# AMDuProfPcm指标介绍

## 1. 读取说明

- `PCm_res/*/report-cumulative.csv` 主要读取 `System (Aggregated)` 列，表示整个系统聚合后的计数器结果。
- 单位说明：`%` 为占比，`GB/s` 为带宽，`MHz` 为有效频率，`ns` 为纳秒，`pti` 为每千条退休指令。
- Topdown 指标按 slot 百分比报告，`Retiring` 越高越好，`Frontend/Backend/Bad_Speculation` 越低越好。

## 2. AMDuProfPcm 通用 CPU 指标

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| CPI (Sys + User) | 总体每条指令平均消耗的周期数。 | 越低越好，与 IPC 互为倒数关系。 |
| Eff Freq (MHz) | 有效工作频率。 | 越高通常越有利，但也要结合 IPC 一起看。 |
| Giga Instructions Per Sec | 每秒退休的十亿条指令数。 | 越高说明单位时间完成的指令更多。 |
| IPC (Sys + User) | 总体每周期退休指令数。 | 越高越好，是核心执行效率的直接指标。 |
| Locked Instructions (pti) | 每千条退休指令中的锁前缀指令数。 | 偏高可能意味着同步争用或原子操作较多。 |
| Retired Branches (pti) | 每千条退休指令中的分支数。 | 用于辅助理解控制流复杂度。 |
| Retired Branches Mispredicted (pti) | 每千条退休指令中的分支预测失败数。 | 越低越好。 |
| System instructions (%) | 退休指令中属于内核态的比例。 | 过高意味着内核路径占比偏大。 |
| System time (%) | 内核态时间占比。 | 过高可能说明系统调用、调度或驱动开销较大。 |
| User instructions (%) | 退休指令中属于用户态的比例。 | 越高说明主要在执行用户态代码。 |
| User time (%) | 用户态时间占比。 | 越高通常说明时间主要花在应用计算上。 |
| Utilization (%) | CPU 利用率。 | 越高说明 CPU 更忙，但并不必然代表更高效率。 |

## 3. Batch 1: 内存互连 + L1/TLB/miss

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| CPI (Sys) | 内核态 CPI。 | 越低越好。 |
| CPI (User) | 用户态 CPI。 | 越低越好。 |
| DC Access (pti) | 每千条退休指令对应的数据缓存访问数。 | 用于观察访存强度。 |
| IC Miss (pti) | 每千条退休指令中的指令缓存 miss 数。 | 越低越好。 |
| IPC (Sys) | 内核态 IPC。 | 用于分辨系统路径效率。 |
| IPC (User) | 用户态 IPC。 | 更接近模型推理主路径效率。 |
| L1 DC Miss (pti) | L1 数据缓存 miss。 | 越低越好。 |
| L1 DTLB Miss (pti) | L1 数据 TLB miss。 | 越低越好。 |
| L1 IC Miss (pti) | L1 指令缓存 miss。 | 越低越好。 |
| L1 ITLB Miss (pti) | L1 指令 TLB miss。 | 越低越好。 |
| L2 Code Read Miss (pti) | L2 指令读取 miss。 | 越低越好。 |
| L2 DTLB Miss (pti) | L2 数据 TLB miss。 | 越低越好。 |
| L2 Data Read Miss (pti) | L2 数据读 miss。 | 越低越好。 |
| L2 ITLB Miss (pti) | L2 指令 TLB miss。 | 越低越好。 |
| Local DRAM Write Data Bytes(GB/s) | 写向本地 DRAM 的带宽。 | 越高表示更多写流量落在本地 NUMA 域。 |
| Local Inbound Read Data Bytes(GB/s) | 从本地内存或本地互连读入的数据带宽。 | 越高说明更多读流量在本地完成。 |
| Op Cache Fetch Miss Ratio | 操作缓存取指 miss 比率。 | 越低越好，偏高会拖慢前端取指。 |
| Remote DRAM Write Data Bytes (GB/s) | 写向远端 DRAM 的带宽。 | 越低越好，偏高代表跨 NUMA 写流量较多。 |
| Remote Inbound Read Data Bytes(GB/s) | 从远端节点读入的数据带宽。 | 越低越好，偏高说明跨 NUMA 读流量较多。 |
| Total Mem Bw (GB/s) | DF 统计的总内存带宽。 | 越高说明内存子系统搬运的数据更多。 |
| Total Mem RdBw (GB/s) | 总内存读带宽。 | 越高说明读流量更大。 |
| Total Mem WrBw (GB/s) | 总内存写带宽。 | 越高说明写流量更大。 |

- `CPI (Sys/User)`、`IPC (Sys/User)` 这类通用执行效率指标统一按 Batch 1 理解；后续 batch 若同样采到这些列，不再重复说明。

## Batch 2: L3 + 数据来源

- `DC Fills` 统计的是回填到 data cache（L1D）的 cache line fill 次数，不按 load/store 指令条数计数。
- `All DC Fills` 对应 `any_dc_fills_by_data_source.*`，覆盖 demand load、硬件预取、软件预取等所有触发类型；`Demand/HwPf/SwPf` 是按触发类型拆开的子集。
- `From XXX` 表示这次 fill 的响应源/返回源是 `XXX`；对 `Demand DC Fills From XXX`，可近似理解为 demand load 在 L1D miss 后最终从 `XXX` 返回并填入 L1D。
- `L3 source latency` 的底层公式、原始事件导出方式与 `same-CCX local-cache hit latency (IBS)` 口径见 [访存延迟测量.md](./访存延迟测量.md)。

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| All DC Fills (pti) | 所有 data-cache fill 次数，包含 demand load、硬件预取、软件预取等触发的回填。 | 越高说明回填活动更多。 |
| All Demand DC Fills (pti) | 仅 demand load 触发并回填到 L1D 的 fill 次数。 | 越高表示真实按需访问的回填更多。 |
| Ave L3 Miss Latency (ns) | L3 miss 平均访问延迟。 | 越低越好。 |
| DC Fills From Local L2 (pti) | 所有 data-cache fill 中，响应源为本地 L2 的部分。 | 越高说明更多回填可在最近一级找到。 |
| DC Fills From Local L3 or different L2 in same CCX (pti) | 所有 data-cache fill 中，响应源为同一 CCX 内 L3 或其他核 L2 的部分。 | 越高通常比跨 CCX/跨 NUMA 更好。 |
| DC Fills From Local Memory or I/O (pti) | 所有 data-cache fill 中，响应源为本地 DRAM 或本地 MMIO 的部分。 | 偏高说明更多回填下探到本地内存/I-O。 |
| DC Fills From Remote Memory or I/O (pti) | 所有 data-cache fill 中，响应源为远端 DRAM 或远端 MMIO 的部分。 | 越低越好。 |
| DC Fills From another CCX in remote node (pti) | 所有 data-cache fill 中，响应源为远端 NUMA 节点其他 CCX cache 的部分。 | 越低越好。 |
| DC Fills From another CCX in same node (pti) | 所有 data-cache fill 中，响应源为同一 NUMA 节点其他 CCX cache 的部分。 | 偏高说明跨 CCX 流量增加。 |
| Demand DC Fills From Local L2 (pti) | demand load 触发并回填到 L1D 的 fill 中，响应源为本地 L2 的部分。 | 越高说明 demand 数据更常在近端找到。 |
| Demand DC Fills From Local L3 or different L2 in same CCX (pti) | demand load 触发并回填到 L1D 的 fill 中，响应源为同 CCX 的 L3 或其他核 L2 的部分。 | 越高通常更好。 |
| Demand DC Fills From Local Memory or I/O (pti) | demand load 触发并回填到 L1D 的 fill 中，响应源为本地内存或 I/O 的部分。 | 偏高说明 demand miss 常落到内存。 |
| Demand DC Fills From Remote memory or I/O (pti) | demand load 触发并回填到 L1D 的 fill 中，响应源为远端内存或 I/O 的部分。 | 越低越好。 |
| Demand DC Fills From another CCX in remote node (pti) | demand load 触发并回填到 L1D 的 fill 中，响应源为远端节点其他 CCX cache 的部分。 | 越低越好。 |
| Demand DC Fills From another CCX in same node (pti) | demand load 触发并回填到 L1D 的 fill 中，响应源为同节点其他 CCX cache 的部分。 | 越低越好。 |
| L3 Access | L3 访问次数。 | 越高表示工作集更依赖 L3。 |
| L3 Access (pti) | 每千条退休指令的 L3 访问数。 | 反映单位指令的 L3 压力。 |
| L3 Hit % | L3 hit 比例。 | 越高越好。 |
| L3 Miss | L3 miss 次数。 | 越低越好。 |
| L3 Miss % | L3 miss 比例。 | 越低越好。 |
| L3 Miss (pti) | 每千条退休指令的 L3 miss 数。 | 越低越好。 |
| L3 Miss Latency From Local Extension Memory (CXL) (%) | L3 miss 延迟来源中，本地扩展内存/CXL 的占比。 | 若接近 0，可认为基本未使用该路径。 |
| L3 Miss Latency From Local Memory or I/O (%) | L3 miss 延迟来源中，本地内存或 I/O 的占比。 | 偏高说明 miss 主要落在本地内存路径。 |
| L3 Miss Latency From Remote Extension Memory (CXL) (%) | L3 miss 延迟来源中，远端扩展内存/CXL 的占比。 | 越低越好。 |
| L3 Miss Latency From Remote Memory or I/O (%) | L3 miss 延迟来源中，远端内存或 I/O 的占比。 | 越低越好。 |
| L3 Miss Latency From another CCX in remote node (%) | L3 miss 延迟来源中，远端节点其他 CCX 的占比。 | 越低越好。 |
| L3 Miss Latency From another CCX in same node (%) | L3 miss 延迟来源中，同节点其他 CCX 的占比。 | 偏高说明跨 CCX 路径占比增加。 |
| Remote DRAM Reads % | 远端 DRAM 读占比。 | 越低越好。 |

## Batch 3: L2 + 预取

- `HwPf/SwPf DC Fills From XXX` 仅统计由硬件预取或软件预取触发、并回填到 L1D 的 fill；`From XXX` 同样表示该次 fill 的响应源。

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| HwPf DC Fills From CCX Cache in remote node (pti) | 硬件预取触发并回填到 L1D 的 fill 中，响应源为远端节点其他 CCX cache 的部分。 | 可用于判断硬件预取是否把数据留在近端层级，以及是否拉来了过多远端数据。 |
| HwPf DC Fills From Cache of another CCX in local node (pti) | 硬件预取触发并回填到 L1D 的 fill 中，响应源为同节点其他 CCX cache 的部分。 | 可用于判断硬件预取是否把数据留在近端层级，以及是否拉来了过多远端数据。 |
| HwPf DC Fills From DRAM or IO connected in local node (pti) | 硬件预取触发并回填到 L1D 的 fill 中，响应源为本地 DRAM 或 MMIO 的部分。 | 可用于判断硬件预取是否把数据留在近端层级，以及是否拉来了过多远端数据。 |
| HwPf DC Fills From DRAM or IO connected in remote node (pti) | 硬件预取触发并回填到 L1D 的 fill 中，响应源为远端 DRAM 或 MMIO 的部分。 | 可用于判断硬件预取是否把数据留在近端层级，以及是否拉来了过多远端数据。 |
| HwPf DC Fills From L2 (pti) | 硬件预取触发并回填到 L1D 的 fill 中，响应源为本地 L2 的部分。 | 可用于判断硬件预取是否把数据留在近端层级，以及是否拉来了过多远端数据。 |
| HwPf DC Fills From L3 or different L2 in same CCX (pti) | 硬件预取触发并回填到 L1D 的 fill 中，响应源为同 CCX 的 L3 或其他核 L2 的部分。 | 可用于判断硬件预取是否把数据留在近端层级，以及是否拉来了过多远端数据。 |
| L2 Access (pti) | 每千条退休指令的 L2 访问数。 | 反映 L2 工作压力。 |
| L2 Access from DC Miss (pti) | 由数据缓存 miss 触发的 L2 访问。 | 越低越好。 |
| L2 Access from IC Miss (pti) | 由指令缓存 miss 触发的 L2 访问。 | 越低越好。 |
| L2 Access from L2 HWPF (pti) | 由 L2 硬件预取器发起的 L2 访问总数，包含这些预取请求在 L2 命中以及在 L2 未命中的两部分。 | 用于观察 L2 硬件预取器的活跃度。 |
| L2 Hit (pti) | 每千条退休指令的 L2 hit 数。 | 越高说明更多访问被 L2 消化。 |
| L2 Hit from DC Miss (pti) | 由数据缓存 miss 触发、最终在 L2 命中的次数。 | 越高通常更好。 |
| L2 Hit from IC Miss (pti) | 由指令缓存 miss 触发、最终在 L2 命中的次数。 | 越高通常更好。 |
| L2 Hit from L2 HWPF (pti) | 由 L2 硬件预取器发起、并在 L2 就命中的访问数。 | 偏高说明预取器发起的很多访问可直接在 L2 满足。 |
| L2 Miss (pti) | 每千条退休指令的 L2 miss 数。 | 越低越好。 |
| L2 Miss from DC Miss (pti) | 由数据缓存 miss 进一步导致的 L2 miss。 | 越低越好。 |
| L2 Miss from IC Miss (pti) | 由指令缓存 miss 进一步导致的 L2 miss。 | 越低越好。 |
| L2 Miss from L2 HWPF (pti) | 由 L2 硬件预取器发起、但在 L2 未命中的访问数；其中既包括随后在 L3 命中的情况，也包括 L3 继续未命中的情况。 | 偏高说明预取器发起的很多访问需要继续下探到更低层级。 |
| SwPf DC Fills From CCX Cache in remote node (pti) | 软件预取触发并回填到 L1D 的 fill 中，响应源为远端节点其他 CCX cache 的部分。 | 越多回填发生在本地 L2/L3 或本地节点，通常越有利；远端来源越低越好。 |
| SwPf DC Fills From Cache of another CCX in local node (pti) | 软件预取触发并回填到 L1D 的 fill 中，响应源为同节点其他 CCX cache 的部分。 | 越多回填发生在本地 L2/L3 或本地节点，通常越有利；远端来源越低越好。 |
| SwPf DC Fills From DRAM or IO connected in local node (pti) | 软件预取触发并回填到 L1D 的 fill 中，响应源为本地 DRAM 或 MMIO 的部分。 | 越多回填发生在本地 L2/L3 或本地节点，通常越有利；远端来源越低越好。 |
| SwPf DC Fills From DRAM or IO connected in remote node (pti) | 软件预取触发并回填到 L1D 的 fill 中，响应源为远端 DRAM 或 MMIO 的部分。 | 越多回填发生在本地 L2/L3 或本地节点，通常越有利；远端来源越低越好。 |
| SwPf DC Fills From L2 (pti) | 软件预取触发并回填到 L1D 的 fill 中，响应源为本地 L2 的部分。 | 越多回填发生在本地 L2/L3 或本地节点，通常越有利；远端来源越低越好。 |
| SwPf DC Fills From L3 or different L2 in same CCX (pti) | 软件预取触发并回填到 L1D 的 fill 中，响应源为同 CCX 的 L3 或其他核 L2 的部分。 | 越多回填发生在本地 L2/L3 或本地节点，通常越有利；远端来源越低越好。 |

## Batch 4: Topdown

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| Backend_Bound | 后端执行或访存受限导致的停顿占比。 | 越低越好。 |
| Backend_Bound.CPU | 后端受限中由执行单元/调度造成的部分。 | 越低越好。 |
| Backend_Bound.Memory | 后端受限中由访存造成的部分。 | 越低越好。 |
| Bad_Speculation | 错误推测导致浪费的 slot 占比。 | 越低越好。 |
| Bad_Speculation.Mispredicts | 错误推测中由分支预测失败造成的部分。 | 越低越好。 |
| Bad_Speculation.Pipeline_Restarts | 错误推测中由流水线重启造成的部分。 | 越低越好。 |
| Frontend_Bound | 由于前端供给不足造成的停顿占比。 | 越低越好。 |
| Frontend_Bound.BW | 前端停顿中由带宽不足导致的部分。 | 越低越好。 |
| Frontend_Bound.Latency | 前端停顿中由延迟导致的部分。 | 越低越好。 |
| Retiring | 真正完成有用工作的 slot 占比。 | 越高越好。 |
| Retiring.Fastpath | Retiring 中走快路径退休的部分。 | 越高通常越好。 |
| Retiring.Microcode | Retiring 中依赖微码辅助的部分。 | 偏高可能说明复杂指令或慢路径更多。 |
| SMT_Disp_contention | SMT 双线程共享前端/后端资源造成的 dispatch 争用。 | 越低越好。 |
| Total_Dispatch_Slots | Topdown 分析中的总 dispatch slot 数。 | 是 Topdown 百分比的分母，不直接比较高低。 |

## Batch 5: UMC

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| Total Est Mem Bw (GB/s) | UMC 估算的总内存带宽。 | 越高说明内存控制器侧流量更大。 |
| Total Est Mem RdBw (GB/s) | UMC 估算的总内存读带宽。 | 越高说明读流量更大。 |
| Total Est Mem WrBw (GB/s) | UMC 估算的总内存写带宽。 | 越高说明写流量更大。 |

## 5. 备注

- `batch6_roofline` 的输出是 `report.json/html`，不属于本次 `report-cumulative.csv` 主分析范围。
