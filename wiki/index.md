# 当前研究 wiki 总入口

## 范围
- 本库只服务当前 `vLLM CPU + Chiplet` 研究。
- `wiki/` 是后续主入口。
- 旧 `info/*.md` 已停止作为主入口；历史内容已收口到 `wiki/`，原始资料收口到 `wiki/原始资料/`。

## 关键入口
- [wiki 规范](./2026-04-08_当前研究wiki规范.md)
- [术语与背景](./术语与背景/README.md)
- [源码分析](./源码分析/README.md)
- [实验方案](./实验方案/README.md)
- [实验结果解读](./实验结果解读/README.md)
- [资料总览](./资料总览/README.md)
- [论文解读](./论文解读/README.md)
- [原始资料](./原始资料/README.md)

## 当前优先阅读
- [Chiplet背景知识](./术语与背景/Chiplet背景知识.md)
- [AMDuProf 背景速记](./术语与背景/AMDuProf背景速记.md)
- [vllm serve 启动到 server ready](./源码分析/vllm_serve启动到server_ready.md)
- [CPU attention 并行计算过程 线程任务 workitem 与 tile](./源码分析/CPU_attention并行计算过程_线程任务workitem与tile.md)
- [Qwen3-30B-A3B attention-only TP 实验步骤](./实验方案/Qwen3-30B-A3B_attention-only_TP实验步骤.md)
- [Qwen3-30B-A3B bfloat16 统一口径与重跑清单](./实验方案/Qwen3-30B-A3B_bfloat16统一口径与重跑清单.md)
- [Qwen3-30B-A3B attention-only TP gdbserver 调试](./实验方案/Qwen3-30B-A3B_attention-only_TP_gdbserver调试.md)
- [Qwen3-30B-A3B AMDuProfPcm 分批实验命令](./实验方案/Qwen3-30B-A3B_AMDuProfPcm分批实验命令.md)
- [Qwen3-30B-A3B vllm bench strict-batch Prefill Decode 实验步骤](./实验方案/Qwen3-30B-A3B_vllm_bench_Prefill_Decode实验步骤.md)
- [Qwen3-30B-A3B strict-batch attention 调试](./实验方案/Qwen3-30B-A3B_strict-batch_attention调试.md)
- [Qwen3-30B-A3B strict-batch gdb 调试附录](./实验方案/Qwen3-30B-A3B_strict-batch_gdb调试附录.md)
- [Chiplet 架构下 L3 指标关注重点与高 L3 Miss 归因边界](./资料总览/Chiplet架构下L3指标关注重点与高L3Miss归因边界.md)
- [benchmarks 与 tests 入口筛选](./资料总览/benchmarks与tests入口筛选.md)
- [2026-04-07 Qwen3-30B-A3B 注意力 KV 工作集与 32MiB L3 容量估算](./实验结果解读/2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md)
- [当前研究主线](./资料总览/当前研究主线.md)
- [相关工作检索清单](./资料总览/相关工作检索清单.md)
- [相关工作论文筛选与研究方向建议](./资料总览/相关工作论文筛选与研究方向建议.md)
- [AMD Intel Chiplet CPU 与 AMX L3 关系](./术语与背景/AMD_Intel_Chiplet_CPU与AMX_L3关系.md)
- [GPU Attention NUMA Effects 解读](./论文解读/GPU_Attention_NUMA_Optimization解读.md)
- [Optimizing Sorting for Chiplet-Based CPUs 解读](./论文解读/Optimizing_Sorting_for_Chiplet_Based_CPUs解读.md)

## 迁移说明
- 新文档优先写入 `wiki/`。
- 旧 `info/*.md` 的有效内容已经并入现有页面或转成新的归档页。
