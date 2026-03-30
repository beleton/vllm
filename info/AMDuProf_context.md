# AMDuProf 背景速记
- 机器：AMD EPYC 9745，family `0x1a`，model `0x11`；本机指标定义主要对应 `/opt/AMDuProf_5.2-606/bin/AMDPerf/data/0x1a_0x1/`。
- 常用本地资料：
  - `/home/zjj/vllm/info/AMDuProfPcm/AMDuProfPcm.pdf`
  - `/home/zjj/vllm/info/AMDuProfPcm/AMDuProfPcm_metrics.pdf`
  - `/opt/AMDuProf_5.2-606/bin/AMDPerf/data/0x1a_0x1/core_metrics.json`
  - `/opt/AMDuProf_5.2-606/bin/AMDPerf/data/0x1a_0x1/core_event.json`
  - `/opt/AMDuProf_5.2-606/bin/AMDPerf/data/0x1a_0x1/l3_metrics.json`
  - `/opt/AMDuProf_5.2-606/bin/Data/Config/0x1a_0x1.conf`
- 现有项目术语表：
  - `/home/zjj/vllm/test_results/Qwen3-30B-A3B_analysis/docs/Qwen3-30B-A3B_metrics_glossary.md`

- 解释规则：
  - `PTI` = 每千条退休指令，分母是 `OsUserInst` / `retired_instructions`，不是 dispatch/issue 指令。
  - `DC Fills` = 回填到 L1D 的 cache line fill 次数，不是 load/store 指令数。
  - `All DC Fills` 对应 `any_dc_fills_by_data_source.*`；`Demand/HwPf/SwPf` 是按触发类型拆分的子集。
  - `DC Fills From XXX` 表示这次 L1D 回填的数据返回源是 `XXX`。
  - `L2 Access/Hit/Miss from L2 HWPF` 统计的是 L2 硬件预取器自己发起的请求；其中 `Hit` 表示该预取请求在 L2 命中。
  - `L3 Miss Latency From XXX` 是“延迟占比”指标，不是“miss 次数占比”指标。

- 常见坑：
  - 从 PDF 复制命令时可能带 Unicode 横杠，命令里应使用 ASCII `-`，例如 `AMDuProfCLI info --list pmu-events`。
