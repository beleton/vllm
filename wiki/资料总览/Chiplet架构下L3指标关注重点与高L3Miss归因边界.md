# Chiplet 架构下 L3 指标关注重点与高 L3 Miss 归因边界

## 问题
- 在当前 `2 x AMD EPYC 9745 128-Core Processor` 上，分析 `L3` 问题时最该先看哪些指标。
- 什么时候可以把高 `L3 Miss` 写成 chiplet / `CCX` / 远端访问问题，什么时候还不能。

## 证据底座
- 直接依据文档：
  - `wiki/论文解读/CHARM解读.md`
  - `wiki/术语与背景/AMDuProfPcm指标介绍.md`
  - `wiki/术语与背景/访存延迟测量.md`
  - `wiki/术语与背景/AMDuProf背景速记.md`
  - `wiki/实验结果解读/2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md`
  - `wiki/资料总览/当前研究主线.md`
- 原始数据路径：
  - `prefill_B16_I128_O1`
  - `prefill_B16_I512_O1`
  - `prefill_B16_I1024_O1`
  - `decode_B16_I1_O1024`
- 当前本文只使用 `report-cumulative.csv` 的 `System (Aggregated)`，因此所有判断都绑定 whole-run cumulative，不是 steady-state 稳定窗。
- 源码证据：
  - `csrc/cpu/cpu_attn_impl.hpp:436-590`
  - `csrc/cpu/cpu_attn_impl.hpp:1420-1460`
  - `csrc/cpu/cpu_attn_impl.hpp:1527-1561`
  - `csrc/cpu/cpu_attn_impl.hpp:1631-1727`

## 结论先行

### 当前最该补进主判据的是 `Demand DC Fills`
当前更合理的观察顺序应拆成三层：
1. 访问量/来源：`All Demand DC Fills (pti)`、`Demand DC Fills From Local L3 or different L2 in same CCX (pti)`、`Demand DC Fills From another CCX in same node (pti)`、`Demand DC Fills From Local Memory or I/O (pti)`、`Demand DC Fills From Remote memory or I/O (pti)`
2. 代价：`Ave L3 Miss Latency (ns)`、`L3 Miss Latency From another CCX in same node (%)`、`L3 Miss Latency From Local Memory or I/O (%)`、`L3 Miss Latency From Remote Memory or I/O (%)`、`CPI`、`IPC`
3. 外溢强度：`Remote DRAM Reads %`、`Total Mem Bw/RdBw`、`L3 Miss (pti)`、`L3 Miss %`、`L3 Access (pti)`

原因是：
- `CHARM` 的 worker 调整直接看 `cache fill event counter`；按当前指标语义，`Demand DC Fills From XXX` 比 `L3 Miss Latency From XXX` 更接近“访问到底从哪里回来”的口径。
- `L3 Miss Latency From XXX` 只表示延迟占比，不表示该来源的访问量占比。
- `L3 Miss %` 只能说明 miss 多不多，不能直接说明 miss 是在本地内存、同节点其他 `CCX` 还是更远路径上变贵。

### `All DC Fills`、`HwPf/SwPf DC Fills` 应作为辅证，不应替代 `Demand DC Fills`
- `All DC Fills` 混合了 demand load、硬件预取、软件预取。
- 如果怀疑是预取行为把数据拉到更远层级，再补看 `HwPf/SwPf DC Fills From XXX`。
- 如果目标是判断“真实按需访问”的局部性变化，主判据仍应优先放在 `Demand DC Fills From XXX`。

### 现有 `decode` 高 `L3 Miss`，还不能写成 chiplet 远端访问主导
- `decode_B16_I1_O1024`：
  - `CPI=1.64`
  - `All Demand DC Fills=5.59 pti`
  - `Demand same-CCX=0.42 pti`
  - `Demand same-node another CCX=0.11 pti`
  - `Demand local memory=1.85 pti`
  - `Demand remote memory=0.00 pti`
  - `L3 Miss %=96.51`
  - `Ave L3 Miss Latency=326.57 ns`
  - `L3 Miss Latency From Local Memory or I/O %=98.70`
  - `L3 Miss Latency From another CCX in same node %=1.51`
  - `Remote DRAM Reads %=0.03`

因此当前可写为：
- decode 的高 `L3 Miss` 和高 miss latency 是真实现象。
- 但在现有 whole-run cumulative 口径下，它更像“本地内存路径上的高代价 miss / 容量或流式 miss”，还不支持写成“跨 `CCX` 或远端 DRAM 主导”。

### `prefill` 某些长度点可能更敏感于 `CCX` 局部性，但还不是稳定规律
- `prefill_B16_I128_O1`：
  - `All Demand DC Fills=2.13 pti`
  - `Demand same-CCX=0.62 pti`
  - `Demand same-node another CCX=0.06 pti`
  - `Demand local memory=0.15 pti`
  - `L3 Miss %=8.82`
  - `Ave L3 Miss Latency=287.01 ns`
  - `L3 Miss Latency From another CCX in same node %=25.42`
- `prefill_B16_I512_O1`：
  - `All Demand DC Fills=2.11 pti`
  - `Demand same-CCX=0.55 pti`
  - `Demand same-node another CCX=0.02 pti`
  - `Demand local memory=0.19 pti`
  - `L3 Miss %=14.55`
  - `Ave L3 Miss Latency=177.99 ns`
  - `L3 Miss Latency From another CCX in same node %=11.65`
- `prefill_B16_I1024_O1`：
  - `All Demand DC Fills=2.18 pti`
  - `Demand same-CCX=0.57 pti`
  - `Demand same-node another CCX=0.01 pti`
  - `Demand local memory=0.22 pti`
  - `L3 Miss %=49.43`
  - `Ave L3 Miss Latency=180.12 ns`
  - `L3 Miss Latency From another CCX in same node %=2.91`

这三组事实表明：
- `L3 Miss %` 低，不代表 miss 代价低。
- `L3 Miss Latency From another CCX in same node (%)` 高，也不代表跨 `CCX` 访问量高。
- `prefill` 长度变化时，`L3 Miss %` 在升，但 `Demand DC Fills From another CCX in same node (pti)` 反而在降；因此 miss 次数和 `CCX` 来源证据不能混成一个结论。

## 当前样例

| workload | `All Demand DC Fills` | `Demand same-CCX` | `Demand same-node another CCX` | `Demand local memory` | `Demand remote memory` | `L3 Miss %` | `Ave L3 Miss Latency` | `same-node another CCX latency %` | `local memory latency %` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill_B16_I128_O1` | 2.13 | 0.62 | 0.06 | 0.15 | 0.00 | 8.82 | 287.01 ns | 25.42 | 70.00 |
| `prefill_B16_I512_O1` | 2.11 | 0.55 | 0.02 | 0.19 | 0.00 | 14.55 | 177.99 ns | 11.65 | 83.16 |
| `prefill_B16_I1024_O1` | 2.18 | 0.57 | 0.01 | 0.22 | 0.00 | 49.43 | 180.12 ns | 2.91 | 95.27 |
| `decode_B16_I1_O1024` | 5.59 | 0.42 | 0.11 | 1.85 | 0.00 | 96.51 | 326.57 ns | 1.51 | 98.70 |

表中可见：
- `prefill` 三个长度点的 `All Demand DC Fills` 很接近，范围只在 `2.11 ~ 2.18 pti`。
- `prefill` 三个长度点里，`Demand same-node another CCX` 从 `0.06 -> 0.02 -> 0.01 pti` 下降，但 `L3 Miss %` 从 `8.82 -> 14.55 -> 49.43` 上升。
- `decode` 的 demand 回填活动远高于 prefill，且 `Demand local memory=1.85 pti` 显著高于 `Demand same-node another CCX=0.11 pti`。

## 归因判据

### 先回答“访问走到哪里了”
优先看 `Demand DC Fills From XXX`：
1. 看 `Demand DC Fills From Local L3 or different L2 in same CCX (pti)`：判断 demand 访问是否还主要停留在本地 `CCX`。
2. 看 `Demand DC Fills From another CCX in same node (pti)`：判断 demand 访问是否更多落到同节点其他 `CCX`。
3. 看 `Demand DC Fills From Remote memory or I/O (pti)`：判断 demand 访问是否已经明显外溢到更远内存域。

如果这里只有延迟占比变化、但 `Demand DC Fills From XXX` 没有同步上升，就不能直接写成“该路径的访问量变多了”。

### 再回答“这些路径是否真的拖慢了执行”
再看：
1. `Ave L3 Miss Latency (ns)`
2. `L3 Miss Latency From XXX (%)`
3. `CPI / IPC`

只有当来源变化和代价变化同方向出现时，才能把“路径变化”和“性能变差”连起来。

### 更像普通容量/流式 miss 的条件
同时满足下面几条时，可写为“高 miss 主要落在本地内存路径，当前不支持写成 chiplet 远端访问主导”：
1. `L3 Miss %` 高
2. `Ave L3 Miss Latency` 高
3. `CPI` 也高
4. `Demand DC Fills From Local Memory or I/O (pti)` 明显高于 `Demand DC Fills From another CCX in same node (pti)` 与 `Demand DC Fills From Remote memory or I/O (pti)`
5. `L3 Miss Latency From Local Memory or I/O (%)` 接近主导
6. `Remote DRAM Reads %` 低

`decode_B16_I1_O1024` 满足这组条件。

### 更像 chiplet / `CCX` 局部性问题的条件
同时满足下面几条时，才更适合往 chiplet/NUMA 方向解释：
1. 固定同一 workload。
2. 只改 `NPS`、`TP`、线程绑核或 NUMA 内存策略。
3. `Demand DC Fills From another CCX in same node (pti)` 系统性上升，或者 `Demand DC Fills From Local L3 or different L2 in same CCX (pti)` 系统性下降。
4. `Ave L3 Miss Latency`、`CPI` 也同步变差。

需先做“只改拓扑、不改 workload”的对照，且不能只拿 `L3 Miss Latency From another CCX in same node (%)` 单独下结论。

### 更像远端内存 / 跨 node 问题的条件
同时满足下面几条时，才更适合写成“访问已外溢到更远层级”：
1. `Demand DC Fills From Remote memory or I/O (pti)` 上升。
2. `Remote DRAM Reads %` 上升。
3. `L3 Miss Latency From Remote Memory or I/O (%)` 或 `...another CCX in remote node (%)` 也上升。
4. `CPI`、`Ave L3 Miss Latency` 同步变差。

### 可能存在 `CCX` 局部性敏感点，但仍需补证
如果出现：
- `L3 Miss %` 不高
- 但 `Ave L3 Miss Latency` 高
- 且 `L3 Miss Latency From another CCX in same node (%)` 偏高
- 但 `Demand DC Fills From another CCX in same node (pti)` 并不高

则只能写为“可能存在 `CCX` 局部性敏感点”，不能直接升格成“跨 `CCX` 访问主导”。

`prefill_B16_I128_O1` 当前符合这类现象：它的 `same-node another CCX latency %=25.42`，但 `Demand same-node another CCX` 只有 `0.06 pti`，而 `Demand same-CCX` 仍有 `0.62 pti`。

## 不能跳过函数级证据
- 即使 phase 级计数器已经显示高 `L3 Miss`，在拿到 `hotspots + IBS L3-miss` 前，仍不能把主因直接落到 attention、MoE 或 TP 通信。
- 当前只能说 attention 是合理怀疑对象，因为源码已显示：
  - scheduler 先按 `kv_len_per_thread` 做近似负载均衡切分
  - 运行时 task 数被展开成 `actual_kv_head_num * effective_thread_num`
  - 同一 `kv_head` 的线程会看到重叠 `KV` 前缀

## 优化判断边界

### 什么时候才值得进入 chiplet/L3 感知优化
至少要同时具备：
1. phase 级指标确认存在真实性能损失：`CPI` 和 `Ave L3 Miss Latency` 抬升。
2. 来源指标确认和放置有关：`Demand DC Fills From another CCX in same node (pti)`、`Demand DC Fills From Remote memory or I/O (pti)` 或 `Demand DC Fills From Local L3 or different L2 in same CCX (pti)` 会随放置策略变化。
3. 函数级证据确认热点函数确实落在目标路径，例如 attention。

缺一项都不应直接进入 patch 设计。

### 当前不该越界写的结论
- 不能把 `decode` 的高 `L3 Miss` 写成“远端 DRAM 访问导致”。
- 不能把 `prefill I128` 的高 `same-node another CCX latency %` 写成“跨 `CCX` 访问量主导”，因为当前 `Demand same-node another CCX` 只有 `0.06 pti`。
- 不能把 `L3 Miss Latency From XXX (%)` 直接当成“`XXX` 来源访问量占比”。
- 不能把 attention 的源码局部性证据，直接写成“attention 局部性优化一定能提升端到端吞吐”。
- 不能把 whole-run cumulative 结果当成 steady-state 稳定窗结果。

## 下一步
1. 先补 `decode_B16_I1_O1024` 和 `prefill_B16_I1024_O1` 的 `hotspots + IBS L3-miss`。
2. 再补 `prefill_B16_I256_O1`，把 prefill 转折区间补齐。
3. 只有函数级证据支持 attention 主导时，才进入 `NPS`、`TP`、绑核、NUMA 策略矩阵。

## 相关页面
- [../论文解读/CHARM解读.md](../论文解读/CHARM解读.md)
- [../术语与背景/AMDuProf背景速记.md](../术语与背景/AMDuProf背景速记.md)
- [../实验结果解读/2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md](../实验结果解读/2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md)
- [../实验方案/Qwen3-30B-A3B_P0_AMDuProfCLI函数级归因实验步骤.md](../实验方案/Qwen3-30B-A3B_P0_AMDuProfCLI函数级归因实验步骤.md)
- [../术语与背景/访存延迟测量.md](../术语与背景/访存延迟测量.md)
- [./当前研究主线.md](./当前研究主线.md)
