# Qwen3-30B-A3B：P0 `Prefill/Decode` 函数级归因实验步骤

> 更新时间：2026-03-30 16:10 +0800  
> 适用机器：`2 x AMD EPYC 9745 128-Core Processor`  
> 目标模型：`/models/Qwen3-30B-A3B`  
> 结论先行：下一步应先做 `P0`，再决定是否继续做 attention 局部性实验或拓扑优化。

## 0. 为什么先做 `P0`

- 当前已经有 phase 级证据，说明 `decode_B16_I1_O1024` 的 `L3 Miss %` 与 `Ave L3 Miss Latency` 明显高于 `prefill_B16_I1024_O1`。证据目录：
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/decode_B16_I1_O1024/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-48-58`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B16_I1024_O1/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-27-32`
- 但这些结果只回答“哪一阶段 miss 更高”，还没有回答“miss 主要由哪个函数/调用链制造”。
- 因此，下一步最应该先做 `AMDuProfCLI hotspots + IBS L3-miss`，确认高 miss 主要落在：
  - `cpu_attention_with_kv_cache`
  - 还是 `cpu_fused_moe`
  - 还是 `linear / TP all_reduce / 通信等待`

## 1. 本实验要回答的问题

1. `decode_B16_I1_O1024` 的 CPU 时间主要压在哪些函数。
2. `decode_B16_I1_O1024` 的 `L3 miss` 样本主要落在哪些函数。
3. `prefill_B16_I1024_O1` 的对应热点函数和 miss 函数是什么。
4. `decode` 与 `prefill` 的差异是否主要出现在 attention 路径。

## 2. 前置条件

### 2.1 已验证可用的工具

- `AMDuProfCLI`：`/opt/AMDuProf_5.2-606/bin/AMDuProfCLI`
- `AMDuProfPcm`：`/opt/AMDuProf_5.2-606/bin/AMDuProfPcm`
- `vllm` 可执行文件：`/home/zjj/.conda/envs/vllm-cpu/bin/vllm`

### 2.2 `strict-batch` 口径

- `strict-batch` 必须在 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 下运行。证据：`vllm/benchmarks/strict_batch.py:230-235`
- `strict-batch` 当前只支持 `--n 1`。证据：`vllm/benchmarks/strict_batch.py:225-227`
- 实验后要检查输出 JSON 里的：
  - `avg_first_step_scheduled_requests`
  - `avg_first_step_scheduled_tokens`
- 这两个字段由 `strict_batch.py` 直接写出。证据：`vllm/benchmarks/strict_batch.py:338-346`

## 3. 一次性准备

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh

export VLLM_BIN=/home/zjj/.conda/envs/vllm-cpu/bin/vllm
export CLI_BIN=/opt/AMDuProf_5.2-606/bin/AMDuProfCLI
export MODEL=/models/Qwen3-30B-A3B
export BASE=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/CLI_res
export LD_LIBRARY_PATH=/home/zjj/.conda/envs/vllm-cpu/lib:${LD_LIBRARY_PATH}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CPU_KVCACHE_SPACE=32
```

## 4. 先跑 `decode_B16_I1_O1024`

### 4.1 `hotspots`

```bash
OUT=${BASE}/decode_B16_I1_O1024/hotspots
mkdir -p "${OUT}"

${CLI_BIN} collect \
  --config hotspots \
  -g \
  --call-graph-depth 64 \
  --output-dir "${OUT}" \
  env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
    VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
    ${VLLM_BIN} bench strict-batch \
      --model ${MODEL} \
      --dtype float16 \
      --block-size 128 \
      --tensor-parallel-size 2 \
      --max-model-len 8192 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 200000 \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --load-format dummy \
      --batch-size 16 \
      --input-len 1 \
      --output-len 1024 \
      --n 1 \
      --num-iters-warmup 2 \
      --num-rounds 10 \
      --disable-detokenize \
      --output-json ${OUT}/strict_batch_result.json
```

### 4.2 `IBS L3-miss`

```bash
OUT=${BASE}/decode_B16_I1_O1024/ibs_l3miss
mkdir -p "${OUT}"

${CLI_BIN} collect \
  -e event=ibs-op,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp \
  --output-dir "${OUT}" \
  env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
    VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
    ${VLLM_BIN} bench strict-batch \
      --model ${MODEL} \
      --dtype float16 \
      --block-size 128 \
      --tensor-parallel-size 2 \
      --max-model-len 8192 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 200000 \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --load-format dummy \
      --batch-size 16 \
      --input-len 1 \
      --output-len 1024 \
      --n 1 \
      --num-iters-warmup 2 \
      --num-rounds 10 \
      --disable-detokenize \
      --output-json ${OUT}/strict_batch_result.json
```

## 5. 再跑 `prefill_B16_I1024_O1`

### 5.1 `hotspots`

```bash
OUT=${BASE}/prefill_B16_I1024_O1/hotspots
mkdir -p "${OUT}"

${CLI_BIN} collect \
  --config hotspots \
  -g \
  --call-graph-depth 64 \
  --output-dir "${OUT}" \
  env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
    VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
    ${VLLM_BIN} bench strict-batch \
      --model ${MODEL} \
      --dtype float16 \
      --block-size 128 \
      --tensor-parallel-size 2 \
      --max-model-len 8192 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 200000 \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --load-format dummy \
      --batch-size 16 \
      --input-len 1024 \
      --output-len 1 \
      --n 1 \
      --num-iters-warmup 2 \
      --num-rounds 10 \
      --disable-detokenize \
      --output-json ${OUT}/strict_batch_result.json
```

### 5.2 `IBS L3-miss`

```bash
OUT=${BASE}/prefill_B16_I1024_O1/ibs_l3miss
mkdir -p "${OUT}"

${CLI_BIN} collect \
  -e event=ibs-op,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp \
  --output-dir "${OUT}" \
  env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
    VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
    ${VLLM_BIN} bench strict-batch \
      --model ${MODEL} \
      --dtype float16 \
      --block-size 128 \
      --tensor-parallel-size 2 \
      --max-model-len 8192 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 200000 \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --load-format dummy \
      --batch-size 16 \
      --input-len 1024 \
      --output-len 1 \
      --n 1 \
      --num-iters-warmup 2 \
      --num-rounds 10 \
      --disable-detokenize \
      --output-json ${OUT}/strict_batch_result.json
```

## 6. 导出报告

### 6.1 `decode hotspots` 报告

```bash
SESSION=$(ls -td ${BASE}/decode_B16_I1_O1024/hotspots/*/ | head -n1)

${CLI_BIN} report \
  -i "${SESSION}" \
  --category cpu \
  --view timer \
  --sort-by metric=total_cpu_time \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_timer.txt"
```

### 6.2 `decode IBS L3-miss` 报告

```bash
SESSION=$(ls -td ${BASE}/decode_B16_I1_O1024/ibs_l3miss/*/ | head -n1)

${CLI_BIN} report \
  -i "${SESSION}" \
  --category cpu \
  --view ibs_op_ld \
  --sort-by event=ibs-op \
  --detail \
  -g \
  --cutoff 20 \
  --show-percentage \
  --stdout | tee "${SESSION}/report_ibs_l3miss.txt"
```

### 6.3 `prefill hotspots` 报告

```bash
SESSION=$(ls -td ${BASE}/prefill_B16_I1024_O1/hotspots/*/ | head -n1)

${CLI_BIN} report \
  -i "${SESSION}" \
  --category cpu \
  --view timer \
  --sort-by metric=total_cpu_time \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_timer.txt"
```

### 6.4 `prefill IBS L3-miss` 报告

```bash
SESSION=$(ls -td ${BASE}/prefill_B16_I1024_O1/ibs_l3miss/*/ | head -n1)

${CLI_BIN} report \
  -i "${SESSION}" \
  --category cpu \
  --view ibs_op_ld \
  --sort-by event=ibs-op \
  --detail \
  -g \
  --cutoff 20 \
  --show-percentage \
  --stdout | tee "${SESSION}/report_ibs_l3miss.txt"
```

## 7. 结果自检

### 7.1 `strict-batch` 是否真的收满一轮

先看 `decode`：

```bash
jq '{avg_first_step_scheduled_requests, avg_first_step_scheduled_tokens}' \
  ${BASE}/decode_B16_I1_O1024/hotspots/strict_batch_result.json
```

再看 `prefill`：

```bash
jq '{avg_first_step_scheduled_requests, avg_first_step_scheduled_tokens}' \
  ${BASE}/prefill_B16_I1024_O1/hotspots/strict_batch_result.json
```

预期检查：

- `avg_first_step_scheduled_requests` 应为 `16`
- `decode_B16_I1_O1024` 的 `avg_first_step_scheduled_tokens` 应接近 `16`
- `prefill_B16_I1024_O1` 的 `avg_first_step_scheduled_tokens` 应接近 `16384`

若明显更小，先不要解释归因结果，优先检查：

- `--max-num-batched-tokens`
- `--max-num-seqs`
- 是否误开 chunked prefill

### 7.2 先看哪些函数

优先检查这些符号是否排到前面：

- attention：
  - `cpu_attention_with_kv_cache`
  - `cpu_attn_get_scheduler_metadata`
  - `csrc/cpu/cpu_attn_impl.hpp` 对应函数
- MoE / MLP：
  - `cpu_fused_moe`
  - `linear`
- TP / 通信：
  - `tensor_model_parallel_all_reduce`
  - `gloo`
  - `cpu_communicator`

## 8. 结果判读标准

- 如果 `decode hotspots` 和 `decode IBS L3-miss` 都主要指向 attention 路径，则下一步进入 attention 局部性实验是合理的。
- 如果 `decode` 的 CPU 时间在 attention，但 `L3 miss` 样本主要落在 `MoE/linear/通信`，则下一步不应直接把 attention 当主矛盾。
- 如果 `prefill` 与 `decode` 都显示 attention 不是主 miss 来源，则 `P2/P3` 的优先级应整体下调。

## 9. 可选：样本不足时的补救

- 若 `hotspots` 或 `IBS` 样本太少，优先把 `--num-rounds` 从 `10` 增加到 `20`。
- 若只想先做烟雾测试，可把 `--num-iters-warmup 0 --num-rounds 1` 跑通命令，再切回正式参数。
