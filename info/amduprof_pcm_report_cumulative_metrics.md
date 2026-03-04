# AMDuProfPcm `report-cumulative.csv` 指标释义（逐项解释）

本文用于解释下面这份 **AMDuProfPcm（AMDuProfPcm-Multi）累计报告**里出现的各个指标含义（按指标名逐项说明）：

- 源文件：`/home/zjj/vllm/test_results/Qwen1.5-MoE-A2.7B/NPS1_TP2/PCm_res/AMDuProfPcm-Multi_Feb-05-2026_21-35-29/report-cumulative.csv`
- Profile Time：`2026/02/05 21:35:29`（见 CSV Header）
- CPU：`AMD EPYC 9745 128-Core Processor`，2 Socket / 256 Core / 512 Thread（见 CSV Header）

> 说明：该 CSV 是 “cumulative/累计” 口径导出；**同一个指标**会在 System / Package(Socket) / CCX / Core 等不同粒度上给出汇总值。本文重点解释“指标本身的含义”，不展开具体数值的好坏阈值。

---

## 1. 如何读这份 CSV（行/列/分区）

这份 `report-cumulative.csv` 大体分为三段：

1. **Header（环境/拓扑/采样配置）**：CPU/OS 信息、Socket/CCX/Core 拓扑、采样间隔、启动命令、缩写解释、告警等。
2. **CORE METRICS**：以 Core 事件计数器为主的指标（行=指标，列=System/Socket/CCX/Core）。
3. **L3 METRICS / DF METRICS**：
   - `L3 METRICS`：L3(LLC) 访问、miss、miss latency 及其来源分解（列到 CCX）。
   - `DF METRICS`：Data Fabric/内存控制器相关的带宽指标（列到 Package/内存通道）。

**列（组件粒度）常见含义：**

- `System (Aggregated)`：系统维度汇总（通常跨所有 socket/core 聚合）。
- `Package-0 / Package-1` 或 `Package (Aggregated)-0/-1`：按 **Socket**（CPU 封装）聚合的指标。
- `CCX-*` / `CCX (Aggregated)-*`：按 **CCX（Core Complex）** 聚合的指标。
- `Core-*`：按 **逻辑 CPU（硬件线程）** 聚合的指标。此机器开启 SMT，因此你会看到 `Core-0` 与 `Core-256` 这种“成对”的编号（通常是同一物理核的两个 SMT 线程）。
- `Mem Ch-A ... Mem Ch-L`：按 **内存通道**（channel）统计的带宽指标（每个 socket 一组通道；某些通道可能为 0/空，取决于硬件/BIOS/装条）。

---

## 2. 缩写/单位与通用口径（先读这个很关键）

CSV Header 已给出部分缩写（原文）：

- `IPC`: Instructions Per CPU Cycle（每周期退役指令数，Instructions Per Cycle）
- `CPI`: CPU Cycles Per Instructions（每条指令消耗的周期数，Cycles Per Instruction）
- `pti`: Per Thousand Instructions（**每千条退役指令**的事件次数；类似 “MPKI/TPKI”）
- `Topdown metrics are reported in "% slots"`：Top-down（流水线瓶颈）类指标用 **dispatch slots 的百分比** 表示

本文补充几条常用理解方式：

- **`pti` 的直观公式**：`pti = (事件计数 / 退役指令数) * 1000`  
  好处是：不同核心/不同时间窗口、不同频率下，`pti` 更适合横向比较（对“执行了多少指令”做了归一化）。
- **`IPC` 与 `CPI`**（同一口径下）通常互为倒数：`CPI ≈ 1 / IPC`。由于采样/多路复用/四舍五入等因素，报表中可能不会严格相等。
- **百分比（`%`）与 ratio（`ratio`）**：  
  `ratio` 是无量纲比值；`%` 是比值乘以 100 的百分数展示。
- **Top-down 的 “% slots”**：  
  “dispatch slots” 可以理解为 **前端每周期可向后端“投递/派发”的微操作/操作（ops）名额**。Top-down 指标用“这些 slots 被谁占用/浪费”来分解瓶颈。
- **L1 cache miss 统计口径提示（来自 CSV 警告）**：  
  L1 miss 不一定统计“所有导致 miss 的访问”，例如当某条 cache line 已经因为更早的 miss 在途（outstanding request）被请求了，后续访问可能不会再次计为 miss。用这些指标做精确“miss 次数”推导时要谨慎。

---

## 3. 术语：Local/Remote、Node、Fill、Demand/Prefetch

理解 DC/L2/L3/NUMA 相关指标时，下面这些词反复出现：

- **Local / Remote**：相对某个 core/CCX 所在的 **NUMA node** 来说：
  - *Local*：同一 NUMA node 内（通常=同一 socket，具体取决于 NUMA 配置）。
  - *Remote*：跨 NUMA node（常见是跨 socket）。
- **“another CCX in same node / remote node”**：
  - *same node*：同一 NUMA node 内的其它 CCX。
  - *remote node*：其它 NUMA node 上的 CCX。
- **Fill（填充）**：一次 fill 通常表示把一条 cache line **装入（填入）上层 cache**（比如 L1D）。Fill 可能由 demand miss 触发，也可能由 prefetch 触发。
- **Demand vs Prefetch**：
  - *Demand*：由真实 load/store 访问触发的 miss/fill。
  - *SwPf*：software prefetch（软件预取指令）触发的 fill。
  - *HwPf*：hardware prefetch（硬件预取器）触发的 fill。

---

## 4. CORE METRICS（核心相关指标）

### 4.1 CPU 利用率与内核/用户态占比

- `Utilization (%)`（%）  
  CPU 处于 **C0（active）电源状态**的时间占比。核心空闲时可能进入更低功耗的 idle state（C1/C2/…），此指标反映“有多少时间核心是活跃的”。

- `System time (%)`（%）  
  CPU 时间中处于 **内核态（kernel / system）**的占比（OS 统计）。

- `User time (%)`（%）  
  CPU 时间中处于 **用户态（user）**的占比（OS 统计）。

- `System instructions (%)`（%）  
  **退役指令（retired instructions）**中在内核态执行的占比。

- `User instructions (%)`（%）  
  退役指令中在用户态执行的占比。

> 提示：`System time (%)`/`User time (%)` 与 `System instructions (%)`/`User instructions (%)` 不一定一致，因为用户态/内核态的 IPC、stall 情况可能差异很大。

### 4.2 频率、IPC/CPI、指令吞吐

- `Eff Freq (MHz)`（MHz）  
  Effective Frequency：采样窗口内的 **平均有效核心频率**（不是 P0 标称频率；会受到 boost、idle、调度等影响）。

- `IPC (Sys + User)`（ratio）  
  平均每 CPU cycle 退役的指令数（user+kernel 合计）。IPC 越低，通常表示更多周期消耗在 cache miss、分支误预测、流水线停顿、内存/I/O 等瓶颈上。

- `IPC (Sys)`（ratio）  
  仅内核态下的 IPC。

- `IPC (User)`（ratio）  
  仅用户态下的 IPC。

- `CPI (Sys + User)`（ratio）  
  平均每条退役指令消耗的 cycle 数（user+kernel 合计）。CPI 越高，通常表示指令执行“更慢/更不高效”（更长的 stall）。

- `CPI (Sys)`（ratio）  
  仅内核态下的 CPI。

- `CPI (User)`（ratio）  
  仅用户态下的 CPI。

- `Giga Instructions Per Sec`（GI/Sec）  
  每秒退役指令数（十亿条/秒），反映整体指令吞吐。

### 4.3 锁、分支与前端（前端供给/取指）

- `Locked Instructions (pti)`（pti）  
  每千条退役指令中发生的 **locked/原子类指令**数量。偏高通常意味着：
  - 锁竞争/同步频繁（例如大量 mutex/spinlock/原子自旋）
  - cache line 在多核间来回迁移（cache-to-cache 交通增多）

- `Retired Branches (pti)`（pti）  
  每千条退役指令中的 **分支指令**数量（控制流密度）。

- `Retired Branches Mispredicted (pti)`（pti）  
  每千条退役指令中的 **分支误预测**数量。可结合 `Retired Branches (pti)` 推算误预测率（粗略）：  
  `mispredict_rate ≈ mispredicted_branches / branches`

- `Op Cache Fetch Miss Ratio`（ratio）  
  Op Cache（可理解为 AMD 前端的 “micro-op/op cache”）**取指/取 op 的 miss 比例**。  
  ratio 越高，表示 Op Cache 命中越差，前端更依赖 decode/指令 cache 路径，容易产生前端瓶颈。

- `IC Miss (pti)`（pti）  
  Instruction Cache miss（通常指 L1I/前端取指相关 miss）的强度：每千条退役指令对应的 IC miss 次数。偏高通常意味着：
  - 代码/指令工作集较大（I-cache 容量/关联度压力）
  - ITLB/I-cache miss 导致前端取指延迟

> 注：报表里同时存在 `IC Miss (pti)` 与 `L1 IC Miss (pti)`（见 4.9）。它们都反映取指 miss 压力，但来自不同指标组/事件口径，数值可能不同；建议在同一组内做横向对比。

### 4.4 L1D：访问与（All）Fill 及其来源分解

- `DC Access (pti)`（pti）  
  Data Cache access：每千条退役指令对应的 **L1D 访问次数**（负载/存储等引发的数据 cache 访问强度指标）。

- `All DC Fills (pti)`（pti）  
  每千条退役指令对应的 **L1D cache line fill 次数（Demand + Prefetch 合计）**。  
  fill 越多通常意味着 L1D miss/替换更频繁，或者 prefetch 更激进（不一定都是“坏事”，需要结合 demand/prefetch 拆分看）。

`All DC Fills (pti)` 的来源（source）拆分（同为 pti）：

- `DC Fills From Local L2 (pti)`  
  L1D miss 后由 **本核私有 L2** 供给并填充到 L1D（L2 hit）。

- `DC Fills From Local L3 or different L2 in same CCX (pti)`  
  由 **同一 CCX 内的共享 L3** 或 **同 CCX 其它 core 的 L2** 供给（CCX 内 cache-to-cache 或 L3 hit）。

- `DC Fills From another CCX in same node (pti)`  
  由 **同一 NUMA node（常见同 socket）内其它 CCX** 的 cache 供给（跨 CCX 的 cache-to-cache）。

- `DC Fills From Local Memory or I/O (pti)`  
  L3 也 miss 后由 **本地内存（DRAM）** 或本地 I/O（例如 DMA）供给。

- `DC Fills From another CCX in remote node (pti)`  
  由 **远端 NUMA node（常见另一 socket）上的 CCX cache** 供给（跨 socket 的 cache-to-cache）。

- `DC Fills From Remote Memory or I/O (pti)`  
  由 **远端 NUMA node 的 DRAM 或 I/O** 供给（跨 socket 远端内存访问）。

- `Remote DRAM Reads %`（%）  
  远端 DRAM read（通常指 LLC miss 后到 DRAM 的 demand read）占总内存 read 的比例（**百分比**；一般不包含 LLC prefetch）。  
  值越高通常表示 NUMA 远端访存更多（线程绑核/内存策略/页面归属不匹配、跨 socket 数据共享等）。

### 4.5 L1D：Demand Fill（真实访问触发）及其来源分解

- `All Demand DC Fills (pti)`（pti）  
  仅统计 **demand（真实 load/store）触发**的 L1D fills（不含 prefetch）。

来源拆分（同为 pti，含义与 4.4 对应项一致，但限定为 demand）：

- `Demand DC Fills From Local L2 (pti)`
- `Demand DC Fills From Local L3 or different L2 in same CCX (pti)`
- `Demand DC Fills From another CCX in same node (pti)`
- `Demand DC Fills From Local Memory or I/O (pti)`
- `Demand DC Fills From another CCX in remote node (pti)`
- `Demand DC Fills From Remote memory or I/O (pti)`

### 4.6 L1D：Software Prefetch（SwPf）Fill 及其来源分解

这些指标统计 **软件预取指令**带来的 L1D fills（pti）：

- `SwPf DC Fills From DRAM or IO connected in remote node (pti)`  
  软件预取从 **远端 node 的 DRAM/I/O** 拉取数据并填充到 L1D。

- `SwPf DC Fills From CCX Cache in remote node (pti)`  
  软件预取从 **远端 node 的 CCX cache（cache-to-cache）** 拉取数据。

- `SwPf DC Fills From DRAM or IO connected in local node (pti)`  
  软件预取从 **本地 node 的 DRAM/I/O** 拉取数据。

- `SwPf DC Fills From Cache of another CCX in local node (pti)`  
  软件预取从 **本地 node 的其它 CCX cache** 拉取数据。

- `SwPf DC Fills From L3 or different L2 in same CCX (pti)`  
  软件预取从 **同 CCX 的 L3/其它 L2** 拉取数据。

- `SwPf DC Fills From L2 (pti)`  
  软件预取命中 **本核 L2**，再填充到 L1D。

### 4.7 L1D：Hardware Prefetch（HwPf）Fill 及其来源分解

这些指标统计 **硬件预取器**带来的 L1D fills（pti）。各项含义与 4.6 完全平行，只是触发者从软件指令变为硬件预取逻辑：

- `HwPf DC Fills From DRAM or IO connected in remote node (pti)`
- `HwPf DC Fills From CCX Cache in remote node (pti)`
- `HwPf DC Fills From DRAM or IO connected in local node (pti)`
- `HwPf DC Fills From Cache of another CCX in local node (pti)`
- `HwPf DC Fills From L3 or different L2 in same CCX (pti)`
- `HwPf DC Fills From L2 (pti)`

### 4.8 L2：访问/命中/未命中及来源归因

- `L2 Access (pti)`（pti）  
  每千条退役指令的 **L2 访问**次数（所有原因合计）。

- `L2 Access from IC Miss (pti)`（pti）  
  由 **L1I/取指 miss** 触发的 L2 访问次数（每千指令）。

- `L2 Access from DC Miss (pti)`（pti）  
  由 **L1D miss** 触发的 L2 访问次数（每千指令）。

- `L2 Access from L2 HWPF (pti)`（pti）  
  由 **L2 硬件预取器**触发的 L2 访问次数（每千指令）。

- `L2 Miss (pti)`（pti）  
  每千条退役指令的 **L2 miss** 次数（需要进一步去 L3/内存/远端）。

- `L2 Miss from IC Miss (pti)`（pti）  
  取指相关访问导致的 L2 miss（每千指令）。

- `L2 Miss from DC Miss (pti)`（pti）  
  数据访问导致的 L2 miss（每千指令）。

- `L2 Miss from L2 HWPF (pti)`（pti）  
  L2 硬件预取导致的 L2 miss（每千指令）。

- `L2 Hit (pti)`（pti）  
  每千条退役指令的 **L2 hit** 次数。

- `L2 Hit from IC Miss (pti)`（pti）
- `L2 Hit from DC Miss (pti)`（pti）
- `L2 Hit from L2 HWPF (pti)`（pti）  
  分别表示由取指 miss / 数据 miss / L2 预取引发的访问在 L2 命中（每千指令）。

### 4.9 L1/L2 miss：更“直观”的 miss 强度指标

- `L1 DC Miss (pti)`（pti）  
  L1D miss 强度（每千指令）。注意 CSV warning：部分“已在途的 cache line”不会重复计 miss。

- `L2 Data Read Miss (pti)`（pti）  
  数据读取路径上的 L2 miss（每千指令）。可近似理解为“load 相关的 L2 miss 压力”。

- `L1 IC Miss (pti)`（pti）  
  L1I miss 强度（每千指令）（与 `IC Miss (pti)` 同属取指 miss 指标，但口径可能不同）。

- `L2 Code Read Miss (pti)`（pti）  
  取指/代码读取路径上的 L2 miss（每千指令）。可近似理解为“指令供给路径上的 L2 miss 压力”。

### 4.10 Top-down：流水线瓶颈（以 “% slots” 表示）

- `Total_Dispatch_Slots`（slots，计数）  
  dispatch slots 的**总量（累计计数）**：可把它理解为 Top-down 百分比指标的分母（总可用/观测到的派发名额）。  
  该值用于将不同类型的 slot 使用/浪费换算成百分比；**本身不是百分比**，数值通常非常大。

Top-down 一级分类（单位均为 `%`，口径是 `% slots`）：

- `SMT_Disp_contention`（%）  
  因 SMT 另一线程被选中而导致 **本线程未获得派发机会**的 slot 比例（SMT 竞争造成的“空转/损失”）。

- `Frontend_Bound`（%）  
  前端供给不足导致的空闲 slot 比例（取指/解码/Op Cache 带宽或 miss 等）。

- `Bad_Speculation`（%）  
  已派发但最终 **未能退役** 的 ops 占比（错误推测/冲刷），例如分支误预测或流水线重启。

- `Backend_Bound`（%）  
  后端资源/执行导致的空闲 slot 比例（执行单元忙、依赖、队列满、内存子系统 stall 等）。

- `Retiring`（%）  
  被真正用于并最终退役（retire）的 ops 占比（更接近“有效工作”）。

Top-down 二级分类（细分归因，单位 `%`）：

- `Frontend_Bound.Latency`：前端 **延迟型**瓶颈（例如 I-cache/ITLB miss）。
- `Frontend_Bound.BW`：前端 **带宽型**瓶颈（例如 decode 带宽、Op Cache fetch 带宽）。
- `Bad_Speculation.Mispredicts`：由 **分支误预测**导致的 flush 占比。
- `Bad_Speculation.Pipeline_Restarts`：由 **pipeline restart/resync** 导致的 flush 占比。
- `Backend_Bound.Memory`：后端 stall 中由 **内存子系统**（cache/memory）导致的部分。
- `Backend_Bound.CPU`：后端 stall 中由 **非内存原因**（执行资源/依赖等）导致的部分。
- `Retiring.Fastpath`：退役 ops 中属于 **fastpath** 的部分（常见简单指令路径）。
- `Retiring.Microcode`：退役 ops 中属于 **microcode** 的部分（复杂指令走微码引擎）。

> 经验性关系（用于 sanity check）：  
> `Frontend_Bound.Latency + Frontend_Bound.BW ≈ Frontend_Bound`  
> `Bad_Speculation.Mispredicts + Bad_Speculation.Pipeline_Restarts ≈ Bad_Speculation`  
> `Backend_Bound.Memory + Backend_Bound.CPU ≈ Backend_Bound`  
> `Retiring.Fastpath + Retiring.Microcode ≈ Retiring`  
> `SMT_Disp_contention + Frontend_Bound + Bad_Speculation + Backend_Bound + Retiring ≈ 100%`

---

## 5. L3 METRICS（LLC：访问/miss/延迟及来源）

> 这些指标通常按 CCX 汇总（因为 L3/LLC 是 CCX 级共享资源）。

- `L3 Access`（次，计数）  
  L3（LLC）访问总次数（累计计数）。访问包含 hit 与 miss。

- `L3 Miss`（次，计数）  
  L3（LLC）miss 总次数（累计计数）。每个 miss 通常意味着需要访问本地/远端内存或远端 cache 来供数。

- `L3 Access (pti)`（pti）  
  每千条退役指令的 L3 访问次数（对指令数做了归一化）。

- `L3 Miss (pti)`（pti）  
  每千条退役指令的 L3 miss 次数。

- `L3 Miss %`（%）  
  L3 miss rate：`L3 Miss / L3 Access * 100`。

- `L3 Hit %`（%）  
  L3 hit rate：`100 - L3 Miss %`（或 `L3 Hit / L3 Access * 100`）。

- `Ave L3 Miss Latency (ns)`（ns）  
  L3 miss 的平均延迟（纳秒）。该延迟取决于 miss 的供数来源（本地 DRAM、远端 DRAM、远端 cache、CXL 等）。

L3 miss latency 的来源分解（单位 `%`，通常加和约为 100%）：

- `L3 Miss Latency From Local Memory or I/O (%)`  
  来自 **本地内存/本地 I/O** 的 L3 miss 延迟占比。

- `L3 Miss Latency From Remote Memory or I/O (%)`  
  来自 **远端内存/远端 I/O** 的 L3 miss 延迟占比。

- `L3 Miss Latency From another CCX in same node (%)`  
  来自 **同 node 其它 CCX cache-to-cache** 供数的延迟占比。

- `L3 Miss Latency From another CCX in remote node (%)`  
  来自 **远端 node 其它 CCX cache-to-cache** 供数的延迟占比。

- `L3 Miss Latency From Local Extension Memory (CXL) (%)`  
  来自 **本地 CXL 扩展内存** 的延迟占比（无 CXL 时通常为 0）。

- `L3 Miss Latency From Remote Extension Memory (CXL) (%)`  
  来自 **远端 CXL 扩展内存** 的延迟占比（无 CXL 时通常为 0）。

---

## 6. DF METRICS（Data Fabric / 内存带宽）

这些指标主要描述 **内存带宽（GB/s）** 与 **本地/远端内存流量**。

- `Total Mem Bw (GB/s)`（GB/s）  
  总内存带宽（读+写）。通常满足：`Total Mem Bw ≈ Total Mem RdBw + Total Mem WrBw`。

- `Total Mem RdBw (GB/s)`（GB/s）  
  总内存读带宽（从内存向 CPU 返回的数据速率）。

- `Total Mem WrBw (GB/s)`（GB/s）  
  总内存写带宽（CPU 向内存写入的数据速率）。

- `Local DRAM Write Data Bytes(GB/s)`（GB/s）  
  写入到 **本地 DRAM** 的写数据带宽。

- `Remote DRAM Write Data Bytes (GB/s)`（GB/s）  
  写入到 **远端 DRAM** 的写数据带宽（跨 NUMA node 的远端写流量）。

### 6.1 每通道读/写带宽（Mem Ch-A ~ Mem Ch-L）

每个 `Mem Ch-<X> RdBw/WrBw (GB/s)` 表示某个 socket 上 **单条内存通道**的读/写带宽：

- `Mem Ch-A RdBw (GB/s)` / `Mem Ch-A WrBw (GB/s)`
- `Mem Ch-B RdBw (GB/s)` / `Mem Ch-B WrBw (GB/s)`
- `Mem Ch-C RdBw (GB/s)` / `Mem Ch-C WrBw (GB/s)`
- `Mem Ch-D RdBw (GB/s)` / `Mem Ch-D WrBw (GB/s)`
- `Mem Ch-E RdBw (GB/s)` / `Mem Ch-E WrBw (GB/s)`
- `Mem Ch-F RdBw (GB/s)` / `Mem Ch-F WrBw (GB/s)`
- `Mem Ch-G RdBw (GB/s)` / `Mem Ch-G WrBw (GB/s)`
- `Mem Ch-H RdBw (GB/s)` / `Mem Ch-H WrBw (GB/s)`
- `Mem Ch-I RdBw (GB/s)` / `Mem Ch-I WrBw (GB/s)`
- `Mem Ch-J RdBw (GB/s)` / `Mem Ch-J WrBw (GB/s)`
- `Mem Ch-K RdBw (GB/s)` / `Mem Ch-K WrBw (GB/s)`
- `Mem Ch-L RdBw (GB/s)` / `Mem Ch-L WrBw (GB/s)`

> 说明：  
> 1) 通道字母与物理通道的映射由平台决定；  
> 2) 若某些通道为 `0` 或空值，可能表示该通道未启用/未装条/不适用于该 CPU SKU 或 BIOS 配置。

### 6.2 本地/远端读数据回传带宽（Inbound Read）

- `Local Inbound Read Data Bytes(GB/s)`（GB/s）  
  从 **本地内存域** 回传给 CPU 的读数据带宽（本地供数）。

- `Remote Inbound Read Data Bytes(GB/s)`（GB/s）  
  从 **远端内存域** 回传给 CPU 的读数据带宽（远端供数，跨 node）。

---

## 7. 建议的使用方式（快速定位瓶颈时怎么搭配看）

如果你的目的不是“理解名词”，而是要快速做性能归因，下面是最常用的搭配思路：

1. 先看 Top-down：`Frontend_Bound / Bad_Speculation / Backend_Bound / Retiring (+ SMT_Disp_contention)`，判断瓶颈大类。
2. 若 `Backend_Bound.Memory` 高：重点看 `L3 Miss (pti)`、`Ave L3 Miss Latency (ns)`、`Total Mem RdBw (GB/s)`、`Remote DRAM Reads %`。
3. 若 `Frontend_Bound` 高：结合 `IC Miss (pti)`、`Op Cache Fetch Miss Ratio`、分支误预测相关指标。
4. 若出现明显 NUMA：结合 DC fill 的 Local/Remote 拆分与 DF 的 Local/Remote Inbound/Write 指标，排查绑核/内存策略。
