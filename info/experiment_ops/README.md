# 实验操作文档索引

本目录集中存放“怎么做实验”的操作文档，避免与背景知识、研究结论、源码分析混在一起。

## 文件列表

| 文件 | 用途 |
| --- | --- |
| `vllm实验步骤.md` | NPS/TP 主线实验步骤、已有结果与下一步计划 |
| `DeepSeek-R1-Distill-Llama-8B分批实验命令.md` | DeepSeek-R1-Distill-Llama-8B 的 AMDuProfPcm 分批采样命令 |
| `Qwen3-30B-A3B_AMDuProfPcm分批实验命令.md` | Qwen3-30B-A3B 的 AMDuProfPcm 分批采样命令 |
| `TinyMoe_AMDuProfPcm分批实验命令.md` | TinyMoe 在 `evalscope perf` 场景下的分批实验命令 |
| `TinyMoe_AMDuProfPcm_vllm_bench实验命令.md` | TinyMoe 在 `vllm bench serve` 场景下的实验命令 |
| `experiment_log_template.md` | 标准化实验记录模板 |

## 使用建议

- 想了解背景、术语和机制：先回到上级目录查看 `info/README.md`
- 想直接复现实验：从 `vllm实验步骤.md` 开始，再进入对应模型命令文档
- 想沉淀一轮实验结果：复用 `experiment_log_template.md`
