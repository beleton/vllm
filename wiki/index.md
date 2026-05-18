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
- [Chiplet与硬件背景](./术语与背景/Chiplet与硬件背景.md)
- [AMDuProf 指标与使用](./术语与背景/AMDuProf指标与使用.md)
- [vllm serve 启动到 server ready](./源码分析/vllm_serve启动到server_ready.md)
- [CPU attention balanced 与 acc-local-l3 线程级任务划分简述](./源码分析/CPU_attention_balanced线程级任务划分简述.md)
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
- [Chiplet 后续研究方向总览](./方向调研/Chiplet后续研究方向总览.md)
- [HNSW 向量检索 Chiplet 研究方向](./方向调研/HNSW向量检索_Chiplet研究方向.md)
- [Zen 平台 TiNA 适用性复核与新增场景](./方向调研/Zen平台_TiNA适用性复核与新增场景.md)
- [网络 I/O、DPDK、RPC、KV Serving 在 Chiplet CPU 上的优化方向研究备忘](./方向调研/网络IO_DPDK_RPC_KV_Chiplet优化方向研究备忘.md)
- [2026-04-07 Qwen3-30B-A3B 注意力 KV 工作集与 32MiB L3 容量估算](./实验结果解读/2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md)
- [当前研究主线](./资料总览/当前研究主线.md)
- [ACC-local-L3 尝试分析](./资料总览/ACC-local-L3尝试分析.md)
- [相关工作与方向](./资料总览/相关工作与方向.md)
- [早期探索记录](./资料总览/早期探索记录.md)
- [GPU Attention NUMA Effects 解读](./论文解读/GPU_Attention_NUMA_Optimization解读.md)
- [Optimizing Sorting for Chiplet-Based CPUs 解读](./论文解读/Optimizing_Sorting_for_Chiplet_Based_CPUs解读.md)

## 迁移说明
- 新文档优先写入 `wiki/`。
- 旧 `info/*.md` 的有效内容已经并入现有页面或转成新的归档页。
