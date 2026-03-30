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

### 4.1 `prefill_only`：`metric2_l3_dc_l2_memory`
(RESULT_JSON目前不存在，但是PCM的结果是存在的)
```bash
METRIC=metric2_l3_dc_l2_memory
BATCH_SIZE=16
INPUT_LEN=128
OUTPUT_LEN=1
RESULT_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/bench_res/prefill_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
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
  --num-iters-warmup 2 \
  --num-rounds 10 \
  --disable-detokenize \
  --output-json ${RESULT_JSON}

METRIC=metric2_l3_dc_l2_memory
BATCH_SIZE=16
INPUT_LEN=128
OUTPUT_LEN=1
AMDuProfPcm profile \
  -m ipc,l3,dc,l2,memory \
  -a -I 200 -s \
  -d 160 \
  -O /home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
```

<!-- VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=32 \
vllm bench latency \
  --model /models/Qwen3-30B-A3B \
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
  --num-iters-warmup 5 \
  --num-iters 20 \
  --disable-detokenize \
  --enable-logging-iteration-details -->

### 4.2 `decode_dominant`：`metric2_l3_dc_l2_memory`
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
AMDuProfPcm profile \
  -m ipc,l3,dc,l2,memory \
  -a -s \
  -d 240 \
  -O /home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/PCm_res/decode_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
```
<!-- VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=32 \
vllm bench latency \
  --model /models/Qwen3-30B-A3B \
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
  --input-len 16 \
  --output-len 1024 \
  --num-iters-warmup 5 \
  --num-iters 20 \
  --disable-detokenize \
  --enable-logging-iteration-details -->