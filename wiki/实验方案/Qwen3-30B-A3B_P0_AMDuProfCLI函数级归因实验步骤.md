# Qwen3-30B-A3B P0 AMDuProfCLI 函数级归因实验步骤

## 目的
- 在 phase 级 `PCM` 已确认 `decode/prefill` 差异后，再用 `hotspots + IBS L3-miss` 回答：
  - 高 `L3 miss` 主要落在哪条函数链路
  - 差异是否主要来自 attention

## 前提
- 工具：
  - `/opt/AMDuProf_5.2-606/bin/AMDuProfCLI`
  - `/home/zjj/.conda/envs/vllm-cpu/bin/vllm`
- 当前推荐口径：
  - `strict-batch`
  - `tensor-parallel-size 2`
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `VLLM_CPU_KVCACHE_SPACE=32`
  - `dtype bfloat16`

## 命令

### 1. 一次性准备

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
export DECODE_START_DELAY_SEC=300
export DECODE_DURATION_SEC=240
export PREFILL_START_DELAY_SEC=300
export PREFILL_DURATION_SEC=240
```

### 2. `hotspots` 采 `decode_B16_I1_O1024`

```bash
OUT=${BASE}/decode_B16_I1_O1024/hotspots
mkdir -p "${OUT}"
APP_LOG="${OUT}/strict_batch_stdout.log"

PYTHONUNBUFFERED=1 \
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
${VLLM_BIN} bench strict-batch \
  --model ${MODEL} \
  --dtype bfloat16 \
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
  --num-iters-warmup 1 \
  --num-rounds 20 \
  --disable-detokenize \
  --output-json ${OUT}/strict_batch_result.json \
>"${APP_LOG}" 2>&1 &
APP_PID=$!

while ! grep -q 'Worker_TP1 pid=' "${APP_LOG}"; do
  if ! kill -0 "${APP_PID}" 2>/dev/null; then
    wait "${APP_PID}"
    exit 1
  fi
  sleep 2
done

WORKER_PIDS=$(grep -o 'Worker_TP[0-9]\\+ pid=[0-9]\\+' "${APP_LOG}" \
  | sed 's/.*pid=//' | sort -u | paste -sd, -)

${CLI_BIN} collect \
  --config hotspots \
  -g \
  --call-graph-depth 64 \
  -p "${WORKER_PIDS}" \
  --start-delay ${DECODE_START_DELAY_SEC} \
  --duration ${DECODE_DURATION_SEC} \
  --output-dir "${OUT}"

wait "${APP_PID}"
```

### 3. `IBS L3-miss` 采 `decode_B16_I1_O1024`

```bash
OUT=${BASE}/decode_B16_I1_O1024/ibs_l3miss
mkdir -p "${OUT}"

${CLI_BIN} collect \
  -e event=ibs-op,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp \
  --start-delay ${DECODE_START_DELAY_SEC} \
  --duration ${DECODE_DURATION_SEC} \
  --output-dir "${OUT}" \
  env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
    VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
    ${VLLM_BIN} bench strict-batch \
      --model ${MODEL} \
      --dtype bfloat16 \
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
      --num-iters-warmup 1 \
      --num-rounds 20 \
      --disable-detokenize \
      --output-json ${OUT}/strict_batch_result.json
```

### 4. `prefill_B16_I1024_O1`
- `hotspots` 和 `IBS L3-miss` 的命令与上面相同，只把：
  - `OUT` 改为 `prefill_B16_I1024_O1/...`
  - `--input-len 1024`
  - `--output-len 1`
  - `--num-rounds 60`
  - `start-delay/duration` 改成 prefill 对应环境变量

## 输出
- `CLI_res/<case>/hotspots/*/`
- `CLI_res/<case>/ibs_l3miss/*/`
- 推荐导出的报告：
  - `report_timer.txt`
  - `report_ibs_l3miss_load.txt`
  - `report_ibs_l3miss_ls_overview.txt`
  - `report_ibs_l3miss_ld_lat.txt`

## 注意事项
- 当前 `TP=2 + Python launcher` 下，`hotspots` 不要直接走 launcher 模式；应先启动 workload，再对 worker `pid` attach。
- `IBS L3-miss` 当前仍可保留 launcher 模式。
- 没拿到函数级证据前，不能把 phase 级高 miss 直接落到 attention、MoE 或 TP 通信。

## 判读口径
- 若 `decode/prefill` 差异最终主要落在 attention 相关调用链，再继续做 locality patch 更稳妥。
- 相关页面：
  - [./Qwen3-30B-A3B_vllm_bench_Prefill_Decode实验步骤.md](./Qwen3-30B-A3B_vllm_bench_Prefill_Decode实验步骤.md)
  - [../实验结果解读/2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md](../实验结果解读/2026-03-26_Qwen3-30B-A3B_PD_Test_Prefill_Decode_PCM观察.md)
  - [../资料总览/Chiplet架构下L3指标关注重点与高L3Miss归因边界.md](../资料总览/Chiplet架构下L3指标关注重点与高L3Miss归因边界.md)
