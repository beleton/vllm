# AMDuProf 背景速记

## 范围
- 本页只整理当前项目里最常用的 `AMDuProfCLI / AMDuProfPcm` 指标语义和常见坑。

## 常用资料
- `/home/zjj/vllm/wiki/原始资料/AMDuProfPcm/AMDuProfPcm.pdf`
- `/home/zjj/vllm/wiki/原始资料/AMDuProfPcm/AMDuProfPcm_metrics.pdf`
- `/opt/AMDuProf_5.2-606/bin/AMDPerf/data/0x1a_0x1/core_metrics.json`
- `/opt/AMDuProf_5.2-606/bin/AMDPerf/data/0x1a_0x1/core_event.json`
- `/opt/AMDuProf_5.2-606/bin/AMDPerf/data/0x1a_0x1/l3_metrics.json`
- `/opt/AMDuProf_5.2-606/bin/Data/Config/0x1a_0x1.conf`
- `/home/zjj/vllm/wiki/术语与背景/AMDuProfPcm指标介绍.md`
- `/home/zjj/vllm/wiki/术语与背景/访存延迟测量.md`

## 解释规则
- 当前机器：AMD EPYC 9745，family `0x1a`，model `0x11`。
- `PTI` 的分母是退休指令，不是 dispatch / issue 指令。
- `DC Fills` 是回填到 `L1D` 的 cache line fill 次数，不是 load/store 指令数。
- `All DC Fills` 对应 `any_dc_fills_by_data_source.*`，`Demand/HwPf/SwPf` 是按触发类型拆开的子集。
- `DC Fills From XXX` 表示这次 `L1D` 回填的数据返回源是 `XXX`。
- `L2 Access/Hit/Miss from L2 HWPF` 看的是 `L2` 硬件预取器自己发起的请求。
- `L3 Miss Latency From XXX` 是延迟占比，不是 miss 次数占比。

## 常见坑
- 从 PDF 复制命令时，容易混入 Unicode 横杠；命令里应统一使用 ASCII `-`。
- 如果要列事件，示例写法是 `AMDuProfCLI info --list pmu-events`。
- 看到 `L3 Miss Latency From XXX` 时，不要直接把它解释成“该来源的 miss 次数占比”。

## 跨 CCX 分析口径
- 研究跨 `CCX` 访问行为时，优先看访问量，不优先看延迟占比。
- 主指标优先看 `Demand DC Fills From another CCX in same node (pti)`；它直接描述 demand 访问在 `L1D miss` 后有多少次最终从同节点其他 `CCX` 回填。
- 对照指标优先看 `Demand DC Fills From Local L3 or different L2 in same CCX (pti)`、`Demand DC Fills From Remote memory or I/O (pti)` 和 `Demand DC Fills From another CCX in remote node (pti)`，用于判断访问是停留在本地 `CCX`、落到同节点其他 `CCX`，还是继续外溢到更远层级。
- `L3 Miss Latency From another CCX in same node (%)` 只用于判断跨 `CCX` 路径在总 miss 延迟中的代价占比，不表示跨 `CCX` 访问量占比。
- 判断“跨 `CCX` 行为是否变多”时，看 `Demand DC Fills From another CCX in same node (pti)`；判断“这些跨 `CCX` 访问是否真的拖慢执行”时，再结合 `L3 Miss Latency From another CCX in same node (%)` 与 `Ave L3 Miss Latency (ns)`。
