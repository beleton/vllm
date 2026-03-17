# TinyMoe：AMDuProfPcm + vLLM Bench 实验命令

## 0. 目的

- 模型：`/models/TinyMoe`
- 压测工具：`vllm bench serve`
- 目标：观察 TinyMoe 推理过程中的 `L2/L3 miss`、预取和 `Topdown` 行为，作为 `Qwen3-30B-A3B` 的对照组
- 重点：TinyMoe 很小，建议缩短 `AMDuProfPcm --start-delay`，否则容易错过 `prefill -> decode` 的相位切换

---

## 1. 一次性前置

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

---

## 2. 启动服务

```bash
VLLM_CPU_KVCACHE_SPACE=8 \
vllm serve /models/TinyMoe \
  --served-model-name TinyMoe \
  --port 8123 \
  --dtype bfloat16 \
  --block-size 128 \
  --max-num-seqs 256 \
  --max_model_len 2048 \
  --tensor-parallel-size 1
```

说明：

- 先用 `TP1`，避免把跨 rank 通信和 `L3 miss` 混在一起。
- `TinyMoe` 比 `Qwen3-30B-A3B` 快很多，所以这里默认 `--start-delay 1000`。

---

## 3. Bench 基线命令模板

```bash
vllm bench serve \
  --base-url http://127.0.0.1:8123 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model TinyMoe \
  --tokenizer /models/TinyMoe \
  --dataset-name random \
  --input-len 1024 \
  --output-len 1024 \
  --random-prefix-len 0 \
  --max-concurrency 64 \
  --num-prompts 64 \
  --request-rate inf \
  --temperature 0 \
  --ignore-eos \
  --save-result \
  --save-detailed \
  --result-dir /home/zjj/vllm/test_results/TinyMoe/bench_res/balanced
```

说明：

- `--max-concurrency` 对应你之前的 `parallel`
- `--num-prompts` 对应你之前的 `number`
- `--save-detailed` 会保留 `start_times`、`ttfts`、`itls`，方便后续按 `TTFT` 做相位切分

---

## 4. 推荐三组负载

### 4.1 Balanced：`prompt=1024, output=1024`

用于复现你之前的大模型固定负载结构。

```bash
INPUT_LEN=1024
OUTPUT_LEN=1024
MAX_CONCURRENCY=64
NUM_PROMPTS=64
SCENARIO=balanced
```

### 4.2 Prefill-only：`prompt=1024, output=1`

用于看 `prefill` 是否本身就有很高的 `L3 miss`。

```bash
INPUT_LEN=1024
OUTPUT_LEN=1
MAX_CONCURRENCY=64
NUM_PROMPTS=64
SCENARIO=prefill_only
```

### 4.3 Decode-dominant：`prompt=16, output=1024`

用于看 `decode` 是否一开始就进入高 miss 模式。

```bash
INPUT_LEN=16
OUTPUT_LEN=1024
MAX_CONCURRENCY=64
NUM_PROMPTS=64
SCENARIO=decode_dominant
```

---

## 5. Batch-1：内存互连 + miss/TLB

```bash
AMDuProfPcm profile \
  -m ipc,memory,ccm_bw,l1,cache_miss,tlb \
  -a --start-delay 1000 -d 30 \
  -O /home/zjj/vllm/test_results/TinyMoe/PCm_res/${SCENARIO}/batch1_mem_ccm_l1_miss_tlb -- \
vllm bench serve \
  --base-url http://127.0.0.1:8123 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model TinyMoe \
  --tokenizer /models/TinyMoe \
  --dataset-name random \
  --input-len ${INPUT_LEN} \
  --output-len ${OUTPUT_LEN} \
  --random-prefix-len 0 \
  --max-concurrency ${MAX_CONCURRENCY} \
  --num-prompts ${NUM_PROMPTS} \
  --request-rate inf \
  --temperature 0 \
  --ignore-eos \
  --save-result \
  --save-detailed \
  --result-dir /home/zjj/vllm/test_results/TinyMoe/bench_res/${SCENARIO}/batch1_mem_ccm_l1_miss_tlb \
  --result-filename result.json
```

---

## 6. Batch-2：缓存来源（L3 + DC）

```bash
AMDuProfPcm profile \
  -m ipc,l3,dc \
  -a --start-delay 1000 -d 30 \
  -O /home/zjj/vllm/test_results/TinyMoe/PCm_res/${SCENARIO}/batch2_l3_dc -- \
vllm bench serve \
  --base-url http://127.0.0.1:8123 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model TinyMoe \
  --tokenizer /models/TinyMoe \
  --dataset-name random \
  --input-len ${INPUT_LEN} \
  --output-len ${OUTPUT_LEN} \
  --random-prefix-len 0 \
  --max-concurrency ${MAX_CONCURRENCY} \
  --num-prompts ${NUM_PROMPTS} \
  --request-rate inf \
  --temperature 0 \
  --ignore-eos \
  --save-result \
  --save-detailed \
  --result-dir /home/zjj/vllm/test_results/TinyMoe/bench_res/${SCENARIO}/batch2_l3_dc \
  --result-filename result.json
```

---

## 7. Batch-3：预取与 L2

```bash
AMDuProfPcm profile \
  -m ipc,l2,swpfdc,hwpfdc \
  -a --start-delay 1000 -d 30 \
  -O /home/zjj/vllm/test_results/TinyMoe/PCm_res/${SCENARIO}/batch3_l2_prefetch -- \
vllm bench serve \
  --base-url http://127.0.0.1:8123 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model TinyMoe \
  --tokenizer /models/TinyMoe \
  --dataset-name random \
  --input-len ${INPUT_LEN} \
  --output-len ${OUTPUT_LEN} \
  --random-prefix-len 0 \
  --max-concurrency ${MAX_CONCURRENCY} \
  --num-prompts ${NUM_PROMPTS} \
  --request-rate inf \
  --temperature 0 \
  --ignore-eos \
  --save-result \
  --save-detailed \
  --result-dir /home/zjj/vllm/test_results/TinyMoe/bench_res/${SCENARIO}/batch3_l2_prefetch \
  --result-filename result.json
```

---

## 8. Batch-4：Topdown

```bash
AMDuProfPcm profile \
  -m ipc,pipeline_util \
  -a --start-delay 1000 -d 30 \
  -O /home/zjj/vllm/test_results/TinyMoe/PCm_res/${SCENARIO}/batch4_topdown -- \
vllm bench serve \
  --base-url http://127.0.0.1:8123 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model TinyMoe \
  --tokenizer /models/TinyMoe \
  --dataset-name random \
  --input-len ${INPUT_LEN} \
  --output-len ${OUTPUT_LEN} \
  --random-prefix-len 0 \
  --max-concurrency ${MAX_CONCURRENCY} \
  --num-prompts ${NUM_PROMPTS} \
  --request-rate inf \
  --temperature 0 \
  --ignore-eos \
  --save-result \
  --save-detailed \
  --result-dir /home/zjj/vllm/test_results/TinyMoe/bench_res/${SCENARIO}/batch4_topdown \
  --result-filename result.json
```

---

## 9. 最小执行顺序建议

如果只是判断 miss 来源，先跑这三组：

1. `SCENARIO=prefill_only` 的 `batch2_l3_dc`
2. `SCENARIO=decode_dominant` 的 `batch2_l3_dc`
3. `SCENARIO=decode_dominant` 的 `batch3_l2_prefetch`

---

## 10. 如何解读

- 如果 `prefill_only` 的 `L3 Miss %` 很低，而 `decode_dominant` 很高，说明高 miss 主要来自 `decode`
- 如果 `decode_dominant` 下 `L3 Miss %` 很高，且 `Ave L3 Miss Latency` 主要仍是本地内存路径，同时 `HwPf DC Fills From DRAM or IO connected in local node` 很高，更像权重流式访问
- 如果随着 `output_len` 增长，后段 miss 还在持续上升，而不是很快进入平台，更像 `KV / working set` 变大
- 如果 `Remote DRAM Reads %`、`Remote Memory`、`another CCX in remote node` 这类指标抬升，才更像 NUMA / CCX 拓扑问题
