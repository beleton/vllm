# Qwen3-30B-A3B vllm bench strict-batch Prefill Decode 实验步骤

## 目的
- 用 `vllm bench strict-batch` 在尽量减少服务端和 HTTP 干扰的前提下，拆开看 `prefill` 与 `decode` 的 phase 级 `AMDuProfPcm` 指标。

## 前提
- 模型：`/models/Qwen3-30B-A3B`
- 工具：
  - `/opt/AMDuProf_5.2-606/bin/AMDuProfPcm`
  - `/home/zjj/.conda/envs/vllm-cpu/bin/vllm`
- 当前推荐口径：
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `VLLM_CPU_KVCACHE_SPACE=32`
  - `strict-batch`
  - `tensor-parallel-size 2`
  - `dtype bfloat16`

## 命令

### 1. 一次性准备

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh

export LD_LIBRARY_PATH=/home/zjj/.conda/envs/vllm-cpu/lib:${LD_LIBRARY_PATH}
export VLLM_BIN=/home/zjj/.conda/envs/vllm-cpu/bin/vllm
```

### 2. `prefill_only`

```bash
METRIC=metric2_l3_dc_l2_memory
BATCH_SIZE=16
INPUT_LEN=128
OUTPUT_LEN=1
RESULT_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/bench_res/prefill_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
RESULT_JSON=${RESULT_DIR}/result.json
PCM_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
mkdir -p "${RESULT_DIR}" "${PCM_DIR}"

AMDuProfPcm profile \
  -m ipc,l3,dc,l2,memory \
  -a -I 300 -s \
  --start-delay 300000 -d 480 \
  -O "${PCM_DIR}" \
  -- env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_CPU_KVCACHE_SPACE=32 \
    ${VLLM_BIN} bench strict-batch \
      --model /models/Qwen3-30B-A3B \
      --dtype bfloat16 \
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
      --num-rounds 300 \
      --disable-detokenize \
      --output-json ${RESULT_JSON}
```

### 3. `decode_dominant`

```bash
METRIC=metric2_l3_dc_l2_memory
BATCH_SIZE=16
INPUT_LEN=1
OUTPUT_LEN=1024
RESULT_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/bench_res/decode_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
RESULT_JSON=${RESULT_DIR}/result.json
PCM_DIR=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/PCm_res/decode_B${BATCH_SIZE}_I${INPUT_LEN}_O${OUTPUT_LEN}/${METRIC}
mkdir -p "${RESULT_DIR}" "${PCM_DIR}"

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
      --dtype bfloat16 \
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

## 输出
- `bench_res/.../result.json`
- `PCm_res/.../report.json`
- `PCm_res/.../report-cumulative.csv`
- `PCm_res/.../report-timeseries.csv`

## 注意事项
- 当前 `TP worker` 场景下，不要把 `-d` 当成强制截断 workload 的主手段；更稳的是让 `strict-batch` 自然结束。
- 若只是校准 steady-state 时间窗，优先临时减小 `--num-rounds`，不要先缩短 profiler 时间。
- 若 `AMDuProfPcm` 在报告已落盘后不退出，先发 `SIGTERM/SIGINT`，不要先 `kill -9`。

## 判读口径
- 优先联合看：
  - `CPI`
  - `Ave L3 Miss Latency (ns)`
  - `L3 Miss Latency From another CCX in same node (%)`
  - `L3 Miss Latency From Local Memory or I/O (%)`
  - `Remote DRAM Reads %`
- 相关结果页：
  - [../实验结果解读/2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md](../实验结果解读/2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md)
  - [../资料总览/Chiplet架构下L3指标关注重点与高L3Miss归因边界.md](../资料总览/Chiplet架构下L3指标关注重点与高L3Miss归因边界.md)
