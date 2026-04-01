# Qwen3-30B-A3B：`vllm bench strict-batch` 的 Prefill / Decode 分相实验步骤

## 0. 目的

- 模型：`/models/Qwen3-30B-A3B`
- 压测工具：`vllm bench strict-batch`
- 目标：先回答“高 `L3 Miss` 主要出现在 `prefill` 还是 `decode`”
- 本文适用于“尽量减少 HTTP / server 干扰，做 `prefill` / `decode` 分相”的受控实验；不再经过 `vllm serve` 的 OpenAI server / HTTP 路径

本次使用 `vllm bench strict-batch` 而不是 `vllm bench latency` 的原因：
- `事实`：`vllm/benchmarks/strict_batch.py` 每轮会先构造这一轮的全部 `request_id` 和 `dummy_prompts`，逐个执行 `engine.add_request(...)`
- `事实`：只有当这一轮 `batch_size` 个请求都完成入队后，脚本才会显式执行第一次 `engine.step()`
- `事实`：脚本还会记录 `waiting_before_first_step`、`first_step_scheduled_requests` 和 `first_step_scheduled_tokens`，可以直接把“首轮是否真收满 16 请求”写进 JSON 结果
- `事实`：如果 `max_num_batched_tokens`、`max_num_seqs` 或 `chunked prefill` 约束不足，scheduler 仍然可能主动拆分；这时拆分原因就是调度预算，而不再是“请求尚未全部送达”
- `推断`：因此，`strict-batch` 更适合当前这类“先验证 strict admission，再做 phase split + PCM”的实验目标

这轮实验要回答的核心问题：
1. `prefill_only` 是否已经有很高的 `L3 Miss %`
2. `decode_dominant` 是否明显更高
3. `decode_dominant` 下的高 miss 更像权重流式访问，还是更像 `KV / working set` 变大


---

## 4. 具体命令

sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh

- 对当前 `vllm bench strict-batch + TP worker` 场景，不要把 `-d` 当成“强制截断 workload”的主手段。
- 若 `AMDuProfPcm` 到时先结束了父进程，`vllm` 的 worker 可能还在走父进程死亡后的清理路径，常见日志是：
  - `Parent process exited, terminating worker`
  - `WorkerProc shutting down.`
  - `resource_tracker ... leaked shared_memory`
- 这时通常说明 `PCM` 报告已经生成，但 workload 退出不干净；更稳的做法是把 `-d` 设得大于这次校准 run 的自然运行时间，让 `strict-batch` 自己先结束。
- 如果只是为了看“启动到稳定”的时间，不要先靠缩短 `-d` 截断；更稳的是临时减少 `--num-rounds`，让校准 run 自然结束，再看 `report-timeseries.csv` 前段。
（挂住时先 kill -TERM <AMDuProfPcm_pid>，不要直接 kill -9；让它补齐 session.uprof 后，再跑 AMDuProfPcm hreport <session>。）

### 4.1 `prefill_only`：`metric2_l3_dc_l2_memory`
原来的分离式两段命令，保留作参考，不再作为主流程：
```bash
# METRIC=metric2_l3_dc_l2_memory
# BATCH_SIZE=16
# INPUT_LEN=128
# OUTPUT_LEN=1
# RESULT_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/bench_res/prefill_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
# RESULT_JSON=${RESULT_DIR}/result.json
# VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=32 \
# vllm bench strict-batch \
#   --model /models/Qwen3-30B-A3B \
#   --dtype float16 \
#   --block-size 128 \
#   --tensor-parallel-size 2 \
#   --max-model-len 8192 \
#   --max-num-seqs ${BATCH_SIZE} \
#   --max-num-batched-tokens 200000 \
#   --no-enable-chunked-prefill \
#   --no-enable-prefix-caching \
#   --load-format dummy \
#   --batch-size ${BATCH_SIZE} \
#   --input-len ${INPUT_LEN} \
#   --output-len ${OUTPUT_LEN} \
#   --num-iters-warmup 1 \
#   --num-rounds 10 \
#   --disable-detokenize \
#   --output-json ${RESULT_JSON}
#
# AMDuProfPcm profile \
#   -m ipc,l3,dc,l2,memory \
#   -a -I 200 -s \
#   -d 500 \
#   -O /home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
```

建议使用下面这组 launcher 版本：
```bash
METRIC=metric2_l3_dc_l2_memory
BATCH_SIZE=16
INPUT_LEN=1024
OUTPUT_LEN=1
RESULT_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/bench_res/prefill_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
RESULT_JSON=${RESULT_DIR}/result.json
PCM_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
LD_LIBRARY_PATH=/home/zjj/.conda/envs/vllm-cpu/lib:${LD_LIBRARY_PATH}
VLLM_BIN=/home/zjj/.conda/envs/vllm-cpu/bin/vllm
mkdir -p "${RESULT_DIR}" "${PCM_DIR}"

# 如果只是校准 steady-state 起点，建议先把 --num-rounds 临时降到 2~3，
# 让 strict-batch 自然结束；确认时间窗后再恢复正式轮数。
AMDuProfPcm profile \
  -m ipc,l3,dc,l2,memory \
  -a -I 300 -s \
  -d 500 \
  -O "${PCM_DIR}" \
  -- env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_CPU_KVCACHE_SPACE=32 \
    ${VLLM_BIN} bench strict-batch \
      --model /models/Qwen3-30B-A3B \
      --dtype float16 \
      --block-size 128 \
      --tensor-parallel-size 2 \
      --max-model-len 8192 \
      --max-num-seqs ${BATCH_SIZE} \
      --max-num-batched-tokens 200000 \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --load-format dummy \
      --batch-size ${BATCH_SIZE} \
      --input-len ${INPUT_LEN} \
      --output-len ${OUTPUT_LEN} \
      --num-iters-warmup 1 \
      --num-rounds 2 \
      --disable-detokenize \
      --output-json ${RESULT_JSON}
```

### 4.2 `decode_dominant`：`metric2_l3_dc_l2_memory`
原来的分离式两段命令，保留作参考，不再作为主流程：
```bash
METRIC=metric2_l3_dc_l2_memory
BATCH_SIZE=16
INPUT_LEN=1
OUTPUT_LEN=1024
RESULT_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/bench_res/decode_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
RESULT_JSON=${RESULT_DIR}/result.json
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=32 \
vllm bench strict-batch \
  --model /models/Qwen3-30B-A3B \
  --dtype float16 \
  --block-size 128 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-seqs ${BATCH_SIZE} \
  --max-num-batched-tokens 200000 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --load-format dummy \
  --batch-size ${BATCH_SIZE} \
  --input-len ${INPUT_LEN} \
  --output-len ${OUTPUT_LEN} \
  --num-iters-warmup 5 \
  --num-rounds 20 \
  --disable-detokenize \
  --output-json ${RESULT_JSON}

METRIC=metric2_l3_dc_l2_memory
BATCH_SIZE=16
INPUT_LEN=1
OUTPUT_LEN=1024
RESULT_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/bench_res/decode_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
RESULT_JSON=${RESULT_DIR}/result.json
VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_CPU_KVCACHE_SPACE=32 \
vllm bench latency \
  --model /models/Qwen3-30B-A3B \
  --dtype float16 \
  --block-size 128 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-seqs ${BATCH_SIZE} \
  --max-num-batched-tokens 200000 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --load-format dummy \
  --batch-size ${BATCH_SIZE} \
  --input-len ${INPUT_LEN} \
  --output-len ${OUTPUT_LEN} \
  --num-iters-warmup 1 \
  --num-iters 2 \
  --disable-detokenize \
  --output-json ${RESULT_JSON} \
  >"${RESULT_DIR}/latebcy_v1mp1.log" 2>&1
#
# AMDuProfPcm profile \
#   -m ipc,l3,dc,l2,memory \
#   -a -s \
#   -d 240 \
#   -O /home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/PCm_res/decode_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
```

建议使用下面这组 launcher 版本：
```bash
METRIC=metric2_l3_dc_l2_memory
BATCH_SIZE=16
INPUT_LEN=1
OUTPUT_LEN=1024
RESULT_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/bench_res/decode_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
RESULT_JSON=${RESULT_DIR}/result.json
PCM_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/PCm_res/decode_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
LD_LIBRARY_PATH=/home/zjj/.conda/envs/vllm-cpu/lib:${LD_LIBRARY_PATH}
VLLM_BIN=/home/zjj/.conda/envs/vllm-cpu/bin/vllm
mkdir -p "${RESULT_DIR}" "${PCM_DIR}"

# 如果只是校准 steady-state 起点，建议先把 --num-rounds 临时降到 2~3，
# 让 strict-batch 自然结束；确认时间窗后再恢复正式轮数。
AMDuProfPcm profile \
  -m ipc,l3,dc,l2,memory \
  -a -I 300 -s \
  -d 540 \
  -O "${PCM_DIR}" \
  -- env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_CPU_KVCACHE_SPACE=32 \
    ${VLLM_BIN} bench strict-batch \
      --model /models/Qwen3-30B-A3B \
      --dtype float16 \
      --block-size 128 \
      --tensor-parallel-size 2 \
      --max-model-len 8192 \
      --max-num-seqs ${BATCH_SIZE} \
      --max-num-batched-tokens 200000 \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --load-format dummy \
      --batch-size ${BATCH_SIZE} \
      --input-len ${INPUT_LEN} \
      --output-len ${OUTPUT_LEN} \
      --num-iters-warmup 1 \
      --num-rounds 2 \
      --disable-detokenize \
      --output-json ${RESULT_JSON} \
      >"${RESULT_DIR}/decode.log" 2>&1
```
