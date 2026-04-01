# 2026-03-31 Chiplet 架构下 `L3` 指标关注重点与高 `L3 Miss` 归因边界

生成时间：`2026-03-31 23:54:44 +0800`

## 0. 任务边界与证据底座

- 任务来源：`tasks.json` 任务 `28`。
- 直接依据文档：
  - `info/进展.md`
  - `info/实验记录/2026-03-30_下一步研究计划.md`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md`
  - `info/AMDuProf_context.md`
- 原始数据路径：
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B16_I1024_O1/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-27-32/report-cumulative.csv`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/decode_B16_I1_O1024/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-48-58/report-cumulative.csv`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B16_I512_O1/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-59-21/report-cumulative.csv`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B16_I128_O1/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_21-06-25/report-cumulative.csv`
- 源码证据：
  - `csrc/cpu/cpu_attn_impl.hpp:436-590`
  - `csrc/cpu/cpu_attn_impl.hpp:1420-1460`
  - `csrc/cpu/cpu_attn_impl.hpp:1527-1561`
  - `csrc/cpu/cpu_attn_impl.hpp:1631-1727`

本文只整理“当前该看哪些 `L3` 指标、怎样给高 `L3 Miss` 设归因边界、什么时候才值得进入优化”。不新增实验，不替代后续 `hotspots + IBS L3-miss`。

## 1. 结论先行

### 1.1 当前最该盯的不是单个 `L3 Miss %`

在这台 `2 x AMD EPYC 9745 128-Core Processor` 上，判断 chiplet/NUMA/L3 问题时，优先级应是：

1. `CPI`
2. `Ave L3 Miss Latency (ns)`
3. `Total Mem Bw`，必要时再拆 `Total Mem RdBw/WrBw`
4. `L3 Miss Latency From another CCX in same node (%)`
5. `Remote DRAM Reads %`
6. `L3 Miss %`
7. `L3 Access (pti)`

原因是：

- `L3 Miss %` 只能说明“miss 多不多”，不能说明 miss 代价来自哪里。证据：`info/AMDuProf_context.md` 已明确 `L3 Miss Latency From XXX` 是延迟占比，不是 miss 次数占比。
- `CPI + Ave L3 Miss Latency` 才能回答 miss 是否真的在拖慢执行。
- `same-node another CCX %` 与 `Remote DRAM Reads %` 才能把“普通容量/流式 miss”和“chiplet/NUMA 放大”分开。

### 1.2 现有 decode 高 `L3 Miss`，还不能写成 chiplet 远端访问主导

当前最强的 phase 级证据是：

- `decode_B16_I1_O1024`：`CPI=1.64`，`L3 Miss %=96.51`，`Ave L3 Miss Latency=326.57 ns`，`L3 Miss Latency From Local Memory or I/O=98.70%`，`same-node another CCX=1.51%`，`remote node another CCX=0.08%`，`Remote DRAM Reads %=0.03`。证据：`.../decode_B16_I1_O1024/...20-48-58/report-cumulative.csv:71,85,112,114,115,117,118,124-126`。
- `prefill_B16_I1024_O1`：`CPI=0.44`，`L3 Miss %=49.43`，`Ave L3 Miss Latency=180.12 ns`，`L3 Miss Latency From Local Memory or I/O=95.27%`，`same-node another CCX=2.91%`，`Remote DRAM Reads %=0.03`。证据：`.../prefill_B16_I1024_O1/...20-27-32/report-cumulative.csv:71,85,112,114,115,117,118,124-126`。

因此，当前更稳的写法是：

- `decode` 的高 `L3 Miss` 和高 miss latency 是真实现象。
- 但在现有 whole-run cumulative 口径下，它更像“本地内存路径上的高代价 miss / 容量或流式 miss”，不支持直接写成“跨 `CCX` / 远端 DRAM 是主因”。

这和 `info/实验记录/2026-03-30_下一步研究计划.md` 里的判断一致。

### 1.3 prefill 某些长度点可能更敏感于 `CCX` 局部性，但还不是稳定规律

现有 prefill 长度点里：

- `I128`：`L3 Miss %=8.82`，但 `Ave L3 Miss Latency=287.01 ns`，`same-node another CCX=25.42%`。证据：`.../prefill_B16_I128_O1/...21-06-25/report-cumulative.csv:85,110,112,114,115,117,118,124-126`。
- `I512`：`L3 Miss %=14.55`，`Ave L3 Miss Latency=177.99 ns`，`same-node another CCX=11.65%`。证据：`.../prefill_B16_I512_O1/...20-59-21/report-cumulative.csv:85,110,112,114,117,124`。
- `I1024`：`L3 Miss %=49.43`，`Ave L3 Miss Latency=180.12 ns`，`same-node another CCX=2.91%`。证据同上。

这说明：

- `L3 Miss %` 低，不代表 miss 代价低。
- prefill 长度变化时，miss 次数和 miss 来源占比可能朝相反方向动。
- 在没补齐 `I256`、没做稳定窗裁剪前，不能把 `I128 -> I512 -> I1024` 写成完整规律。

## 2. Chiplet 架构下的 `L3` 指标关注重点

### 2.1 一级指标：先判断“高 miss 是否真拖慢了程序”

- `CPI`
  - 用来判断 miss 是否已经转化成执行停顿。
  - 例子：`decode_B16_I1_O1024` 的 `CPI=1.64`，显著高于 `prefill_B16_I1024_O1` 的 `0.44`，所以 decode 的 miss 不是“计数器噪声”。证据：两组 `report-cumulative.csv:71`。
- `Ave L3 Miss Latency (ns)`
  - 用来判断单次 miss 的代价。
  - 例子：decode `326.57 ns` 对比 prefill `180.12 ns`。证据：两组 `report-cumulative.csv:114`。
- `Total Mem Bw`
  - 用来判断高 miss 是否伴随更高的数据搬运压力。
  - 当前 decode `350.99 GB/s`、prefill `328.64 GB/s`，两者都不低，因此不能把 decode 简单解释成“内存没跑起来”。证据：两组 `report-cumulative.csv:124-126`。

### 2.2 二级指标：再判断 miss 来自哪里

- `L3 Miss Latency From another CCX in same node (%)`
  - 这是当前最直接的 socket 内 chiplet/CCX 相关指标。
  - 若只改绑核/NUMA 放置，这个值系统性上升，同时 `CPI` 和 `Ave L3 Miss Latency` 也跟着升，才更像 chiplet 放大。
- `Remote DRAM Reads %`
  - 这是 `DC fills` 中远端内存回填占比，不等于 `L3 miss` 次数占比。证据：`info/AMDuProf_context.md`。
  - 若它不升，就不能把高 `L3 Miss` 直接写成“远端 DRAM 主导”。
- `L3 Miss Latency From Local Memory or I/O (%)`
  - 当这个值接近 `100%` 时，说明 miss 代价主要落在本地内存路径，而不是跨 `CCX` / 跨 node。

### 2.3 辅助指标：帮助识别“访问强度”和“形态”，但不单独下结论

- `L3 Miss %`
  - 必看，但不能单独做根因归因。
- `L3 Access (pti)`
  - 适合看单位指令的 `L3` 压力。
  - 当前 prefill `I128/I512/I1024` 的 `L3 Access (pti)` 只有 `23.70/24.48/23.96`，变化远小于 `L3 Miss %` 和 `Total Mem Bw`。证据：对应 `report-cumulative.csv:110`。
  - 当前 decode `O64~O1024` 里 `L3 Access (pti)` 也较平。证据：`test_results/PD_Test/Qwen3-30B-A3B/PCm_res/2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md`。

## 3. 高 `L3 Miss` 的归因判据

### 3.1 可写成“更像普通容量/流式 miss”的条件

同时满足下面几条时，更稳的写法是“高 miss 主要落在本地内存路径，当前不支持写成 chiplet 远端访问主导”：

1. `L3 Miss %` 高。
2. `Ave L3 Miss Latency` 高。
3. `CPI` 也高。
4. `L3 Miss Latency From Local Memory or I/O (%)` 接近主导。
5. `same-node another CCX %` 和 `Remote DRAM Reads %` 都低。

`decode_B16_I1_O1024` 当前就满足这组条件。证据见第 1.2 节的 raw csv 路径。

### 3.2 可写成“可能受 chiplet/CCX 局部性影响”的条件

同时满足下面几条时，才更适合往 chiplet/NUMA 方向解释：

1. 固定同一 workload。
2. 只改 `NPS`、`TP`、线程绑核或 NUMA 内存策略。
3. `same-node another CCX %`、`remote node another CCX %` 或 `Remote DRAM Reads %` 系统性上升。
4. `CPI`、`Ave L3 Miss Latency` 也同步变差。

也就是说，**必须先做“只改拓扑、不改 workload”的对照**。单看某一组 `prefill` 或 `decode` 的绝对值，不够。

### 3.3 可写成“可能存在 `CCX` 局部性敏感点，但仍需补证”的条件

如果出现：

- `L3 Miss %` 不高，
- 但 `Ave L3 Miss Latency` 高，
- 且 `same-node another CCX %` 明显偏高，

那么可以写成“可能存在 `CCX` 局部性敏感点”，但不能直接升格成稳定规律。

`prefill_B16_I128_O1` 当前符合这类现象。证据：`.../prefill_B16_I128_O1/...21-06-25/report-cumulative.csv:112,114,117`。

### 3.4 不能跳过函数级证据

即使 phase 级计数器已经显示高 `L3 Miss`，在拿到 `hotspots + IBS L3-miss` 前，仍不能把主因直接落到 attention、MoE 或 TP 通信。

当前只能说：

- attention 是合理怀疑对象，因为源码明确显示：
  - scheduler 先按 `kv_len_per_thread` 做近似负载均衡切分，而不是按共享 `KV` 工作集做聚类。证据：`csrc/cpu/cpu_attn_impl.hpp:436-590`。
  - 运行时 task 数被展开成 `actual_kv_head_num * effective_thread_num`。证据：`csrc/cpu/cpu_attn_impl.hpp:1420-1460`。
  - 同一 `kv_head` 的线程会从全局 `key_cache/value_cache` 读取，并在 causal prefill 下访问重叠 `KV` 前缀。证据：`csrc/cpu/cpu_attn_impl.hpp:1527-1561`、`1631-1727`。
- 但“attention 就是当前高 miss 主因”还不是事实结论，只能等任务 `8` 的函数级归因补齐。

## 4. 优化判断边界

### 4.1 什么时候才值得进入 chiplet/L3 感知优化

至少要同时具备：

1. phase 级指标确认存在真实性能损失：`CPI` 和 `Ave L3 Miss Latency` 抬升。
2. 拓扑来源指标确认和放置有关：`same-node another CCX %` 或 `Remote DRAM Reads %` 会随放置策略变化。
3. 函数级证据确认热点函数确实落在目标路径，例如 attention。

缺一项都不应直接进入 patch 设计。

### 4.2 当前不该越界写的结论

- 不能把 `decode` 的高 `L3 Miss` 写成“远端 DRAM 访问导致”。
- 不能把 `prefill I128` 的高 `same-node another CCX %` 写成“prefill 天然更依赖跨 CCX”。
- 不能把 attention 的源码局部性证据，直接写成“attention 局部性优化一定能提升端到端吞吐”。
- 不能把 whole-run cumulative 结果当成 steady-state 稳定窗结果。证据：`2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md` 已写明当前汇总未做 time-filter 裁剪。

### 4.3 当前最稳的下一步

按 `info/实验记录/2026-03-30_下一步研究计划.md`，优先级仍应是：

1. 先补 `decode_B16_I1_O1024` 和 `prefill_B16_I1024_O1` 的 `hotspots + IBS L3-miss`。
2. 再补 `prefill_B16_I256_O1`，把 prefill 转折区间补齐。
3. 只有函数级证据支持 attention 主导时，才进入 `NPS`、`TP`、绑核、NUMA 策略矩阵。

## 5. 一句话口径

当前在 chiplet CPU 上看 `L3`，最重要的是把 `CPI`、`Ave L3 Miss Latency`、`Total Mem Bw` 和 `same-node another CCX %`、`Remote DRAM Reads %` 联合起来看；仅凭高 `L3 Miss %`，还不能把问题写成 chiplet 远端访问主导，更不能跳过函数级证据直接进入优化。
