# info 文档索引（vLLM CPU 课题）

主入口：

- 仓库级协作协议：[AGENT.md](../AGENT.md)
- 当前任务清单：[current_tasks.md](./current_tasks.md)

## 目录索引表

| 文件 | 用途 |
| --- | --- |
| `current_tasks.md` | 当前任务清单 |
| `Chiplet背景知识.md` | Chiplet/NUMA/L3 分层背景与相关研究梳理 |
| `vllm实验步骤.md` | NPS/TP 相关实验步骤、结果与后续计划 |
| `vllm_cpu_nps_tp_root_cause_analysis.md` | NPS/TP 性能变化根因分析（源码路径 + 计数器闭环） |
| `vllm_serve_tp_numa_线程核心代码全流程详解.md` | 指定 `vllm serve` 命令的代码流程详解（重点：TP/NUMA/线程/核心） |
| `evalscope_perf.txt` | `evalscope perf` 命令参数说明 |
| `research_brief.md` | 当前研究问题、证据快照、假设与下一步实验 |
| `experiment_log_template.md` | 标准化实验记录模板 |
| `AMDuProfPcm/AMDuProfPcm.pdf` | AMD官方对`AMDuProfPcm`的说明文档，只包含linux部分 |
| `AMDuProfPcm/AMDuProfPcm_metrics.pdf` | AMD官方对`AMDuProfPcm`命令行参数的说明文档 |
| `AMD_PPR_57883/*` | 本机(AMD Zen5)的官方PPR文档 |


## 建议阅读顺序

1. [Chiplet背景知识.md](./Chiplet背景知识.md)
2. [vllm实验步骤.md](./vllm实验步骤.md)
3. [current_tasks.md](./current_tasks.md)
