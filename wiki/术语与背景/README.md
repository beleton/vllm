# 术语与背景

- [Chiplet与硬件背景.md](./Chiplet与硬件背景.md)：Chiplet/CCD/CCX/L3/NUMA 最小背景、本机延迟量级、AMD/Intel chiplet 与 AMX/L3 关系
- [AMDuProf指标与使用.md](./AMDuProf指标与使用.md)：AMDuProfPcm 常用指标语义、跨 CCX 分析口径与常见坑
- [访存延迟测量.md](./访存延迟测量.md)：PCM L3 source latency、IBS 来源延迟反推与工具联读规则
- [perf_mem按进程访存测量.md](./perf_mem按进程访存测量.md)：`perf mem` 按进程测量 data source 分布（跨 CCD 访问、L1/L2/L3 hit、本地/远端 DRAM）
- [CPU_attention_profiling指标介绍.md](./CPU_attention_profiling指标介绍.md)：`VLLM_CPU_ATTN_PROFILE=1` 的 profile.json、summary.csv 各字段含义
- [Evalscope指标介绍.md](./Evalscope指标介绍.md)：Evalscope 性能指标的含义与解读
- [CPU缓存与内存访问体系/](./CPU缓存与内存访问体系/)：**系统化技术文档**——从一次 Load/Store 的完整生命周期出发，覆盖 TLB、L1/L2/L3 缓存、DRAM、乱序执行访存、硬件预取与缓存一致性协议
