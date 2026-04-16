# experiment_ops 索引
- `vllmNPS实验步骤.md`：记录 NPS/TP 主线实验配置、读数、结论与下一步计划，适合作为复现实验的起点。
- `DeepSeek-R1-Distill-Llama-8B分批实验命令.md`：提供针对 DeepSeek-R1-Distill-Llama-8B 的 NPS4_TP8 AMDuProfPcm batch1~6 命令及依赖配置。
- `Qwen3-30B-A3B_AMDuProfPcm分批实验命令.md`：列出 Qwen3-30B-A3B NPS1_TP2 的分批采样流程与常用输出路径。
- `Qwen3-30B-A3B_vllm_bench_Prefill_Decode实验步骤.md`：记录 Prefill/Decode 分相在 strict-batch 下的端到端命令与 PCM 对应时间点，帮助定位 Prefill vs Decode 的指标差异。
- `Qwen3-30B-A3B_P0_AMDuProfCLI函数级归因实验步骤.md`：用于 `P0` 阶段的 `hotspots + IBS L3-miss` 函数级归因，回答高 `L3 miss` 主要落在哪条调用链。
- `Qwen3-30B-A3B_IBS来源延迟采样实验步骤.md`：说明如何用 `AMDuProfCLI IBS OP` 采样并估算 `another CCX cache` 与 `local memory` 的平均访问延迟。
- `Qwen3-30B-A3B_attention-only_TP实验步骤.md`：用于 `P2` 阶段的 attention-only 多进程 TP 试验台，说明 `benchmark_cpu_attn_mp.py` 的用法、NUMA/OMP 绑定口径和可直接复制执行的 `AMDuProfPcm` 命令。
- `Qwen3-30B-A3B_strict-batch_attention调试.md`：说明在 vllm bench strict-batch 下调试 attention 的 gdb 断点、TP=1/TP=2 的 attach 流程与注意事项。
- `gdb_debug.md`：调试附录，列出 strict-batch、ps-附加与断点命令片段。
