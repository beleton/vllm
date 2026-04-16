# DeepSeek-R1-Distill-Llama-8B NPS 实验步骤

## 目的
- 复现“修改 `NPS` 并匹配相应 `TP` 设置会影响 LLM 推理性能”的现象。

## 前提
- 目标模型：`/models/DeepSeek-R1-Distill-Llama-8B`
- 需要手工修改 BIOS：
  - `AMD CBS -> DF Common Options -> Memory Addressing -> NUMA nodes per socket`
- 工具：
  - `AMDuProfPcm`
  - `vllm serve`
  - `evalscope perf`

## 结果目录约定
- `NPS1_TP2`：`/home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS1_TP2`
- `NPS2_TP4`：`/home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS2_TP4`
- `NPS4_TP8`：`/home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8`

## 一次性前置

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

## `vllm serve` 命令

### `NPS1_TP2`

```bash
conda activate vllm-cpu
vllm serve /models/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name DeepSeek-R1-Distill-Llama-8B \
  --port 8122 \
  --dtype bfloat16 \
  --block-size 128 \
  --max-num-seqs 64 \
  --max_model_len 8192 \
  --tensor-parallel-size 2
```

### `NPS2_TP4`

```bash
conda activate vllm-cpu
vllm serve /models/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name DeepSeek-R1-Distill-Llama-8B \
  --port 8122 \
  --dtype bfloat16 \
  --block-size 128 \
  --max-num-seqs 64 \
  --max_model_len 8192 \
  --tensor-parallel-size 4
```

### `NPS4_TP8`

```bash
conda activate vllm-cpu
vllm serve /models/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name DeepSeek-R1-Distill-Llama-8B \
  --port 8122 \
  --dtype bfloat16 \
  --block-size 128 \
  --max-num-seqs 64 \
  --max_model_len 8192 \
  --tensor-parallel-size 8
```

## `evalscope perf` 命令

### `NPS1_TP2`

```bash
conda activate evalscope
evalscope perf \
  --model DeepSeek-R1-Distill-Llama-8B \
  --parallel 1 2 4 8 16 32 64 \
  --number 1 2 4 8 16 32 64 \
  --url http://0.0.0.0:8122/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 --max-prompt-length 1024 \
  --tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
  --outputs-dir /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS1_TP2/evalscope_res
```

### `NPS2_TP4`

```bash
conda activate evalscope
evalscope perf \
  --model DeepSeek-R1-Distill-Llama-8B \
  --parallel 1 2 4 8 16 32 64 \
  --number 1 2 4 8 16 32 64 \
  --url http://0.0.0.0:8122/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 --max-prompt-length 1024 \
  --tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
  --outputs-dir /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS2_TP4/evalscope_res
```

### `NPS4_TP8`

```bash
conda activate evalscope
evalscope perf \
  --model DeepSeek-R1-Distill-Llama-8B \
  --parallel 1 2 4 8 16 32 64 \
  --number 1 2 4 8 16 32 64 \
  --url http://0.0.0.0:8122/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 --max-prompt-length 1024 \
  --tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
  --outputs-dir /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res
```

## `AMDuProfPcm` 联动命令

原始 NPS 页只给了一条固定负载 `parallel=16 / number=16` 口径，下面按三个 `NPS/TP` 配置完整展开。

### `NPS1_TP2`

```bash
AMDuProfPcm profile \
  -m ipc,l1,l2,l3,swpfdc,hwpfdc,dc,pipeline_util,cache_miss,memory,ccm_bw \
  -a --start-delay 5000 -d 80 \
  -O /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS1_TP2/PCm_res -- \
  evalscope perf \
    --model DeepSeek-R1-Distill-Llama-8B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS1_TP2/evalscope_res/pcm_run
```

### `NPS2_TP4`

```bash
AMDuProfPcm profile \
  -m ipc,l1,l2,l3,swpfdc,hwpfdc,dc,pipeline_util,cache_miss,memory,ccm_bw \
  -a --start-delay 5000 -d 80 \
  -O /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS2_TP4/PCm_res -- \
  evalscope perf \
    --model DeepSeek-R1-Distill-Llama-8B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS2_TP4/evalscope_res/pcm_run
```

### `NPS4_TP8`

```bash
AMDuProfPcm profile \
  -m ipc,l1,l2,l3,swpfdc,hwpfdc,dc,pipeline_util,cache_miss,memory,ccm_bw \
  -a --start-delay 5000 -d 80 \
  -O /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/PCm_res -- \
  evalscope perf \
    --model DeepSeek-R1-Distill-Llama-8B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res/pcm_run
```

若要在 `NPS4_TP8` 下分六批采 `memory/l3/l2/topdown/umc/roofline`，见 [DeepSeek-R1-Distill-Llama-8B_AMDuProfPcm分批实验命令.md](./DeepSeek-R1-Distill-Llama-8B_AMDuProfPcm分批实验命令.md)。

## 历史观察
- 原始历史批次记录的是：
  - `NPS2_TP4` 相比 `NPS1_TP2` 有提升
  - `NPS4_TP8` 反而不如 `NPS1_TP2`

## 历史数据

| NPS | Avg TTFT(s) | Avg TPOT(s) | Tokens/s | 内存带宽(GB/s) | CPI |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.5891 | 0.0755 | 197.5150 | 271.03 | 1.73 |
| 2 | 4.8408 | 0.0717 | 209.4394 | 302.40 | 1.72 |
| 4 | 10.2484 | 0.0945 | 153.2066 | 231.04 | 2.37 |

| NPS | 内存带宽(GB/s) | 吞吐量 GFLOPs | 算术强度(FLOP/B) |
| --- | ---: | ---: | ---: |
| 1 | 284.4402 | 4769.1319 | 16.76 |
| 2 | 318.3504 | 3836.0272 | 12.04 |
| 4 | 235.5316 | 4015.8258 | 17.05 |

## 边界
- 这是较早的 `DeepSeek-R1-Distill-Llama-8B` 批次，不应直接拿来替代当前 `Qwen3-30B-A3B` 主线判断。
- 当前本地 `test_results` 已确认保留 `NPS1_TP2` 和 `NPS4_TP8` 的结果目录；本页里的 `NPS2_TP4` 路径是重跑时应使用的输出目录约定。
- 本页历史数据表来自原始实验记录页，不是本轮重新汇总生成。
- 本页保留价值主要是：
  - BIOS 与 `NPS` 修改入口
  - `evalscope + PCM` 的完整联动口径
  - “`NPS` 变化未必单调提升”的历史提醒
