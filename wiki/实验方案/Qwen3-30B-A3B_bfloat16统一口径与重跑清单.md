# Qwen3-30B-A3B `bfloat16` 统一口径与重跑清单

## 目的
- 统一 `wiki/实验方案/` 下 `Qwen3-30B-A3B` 相关实验口径为 `bfloat16`。
- 明确哪些已有实验结果若要纳入统一口径对比，必须重跑。

## 当前文档口径
- 已统一为 `bfloat16` 的正式实验文档：
  - `Qwen3-30B-A3B_vllm_bench_Prefill_Decode实验步骤.md`
  - `Qwen3-30B-A3B_P0_AMDuProfCLI函数级归因实验步骤.md`
  - `Qwen3-30B-A3B_AMDuProfPcm分批实验命令.md`
  - `Qwen3-30B-A3B_attention-only_TP实验步骤.md`
- 已统一为 `bfloat16` 的调试/附录文档：
  - `Qwen3-30B-A3B_attention-only_TP_gdbserver调试.md`
  - `Qwen3-30B-A3B_strict-batch_gdb调试附录.md`

## 需要重跑的实验
- 若目标是让 `Qwen3-30B-A3B` 的正式实验结果全部纳入同一 `bfloat16` 口径，则下列历史 `float16` 实验需要重跑：
  - `Qwen3-30B-A3B_vllm_bench_Prefill_Decode实验步骤.md`
  - `Qwen3-30B-A3B_P0_AMDuProfCLI函数级归因实验步骤.md`
  - `Qwen3-30B-A3B_AMDuProfPcm分批实验命令.md`

## 不需要因统一口径而重跑的实验
- `Qwen3-30B-A3B_attention-only_TP实验步骤.md`
  - 该文档原本就是 `bfloat16` 口径。

## 不属于“正式实验结果需要补跑”的文档
- `../术语与背景/访存延迟测量.md`
- `Qwen3-30B-A3B_attention-only_TP_gdbserver调试.md`
- `Qwen3-30B-A3B_strict-batch_gdb调试附录.md`
- `Qwen3-30B-A3B_strict-batch_attention调试.md`

这些页面用于调试或附录说明。即使同步改成 `bfloat16` 示例，也不代表存在必须补齐的一组正式结果。

## 使用口径
- 后续新跑 `Qwen3-30B-A3B` 相关实验时，命令优先以本目录当前文档为准。
- 若引用历史结果，必须先确认该批结果的实际 `dtype`；不能把历史 `float16` 结果直接并入新的 `bfloat16` 对照表。
