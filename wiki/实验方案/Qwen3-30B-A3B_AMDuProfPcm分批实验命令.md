# Qwen3-30B-A3B `AMDuProfPcm` 分批实验命令

## 适用范围
- 模型：`Qwen3-30B-A3B`
- 拓扑：`NPS1_TP2`
- 固定负载：`--parallel 16 --number 16`
- 原始来源：`info/experiment_ops/Qwen3-30B-A3B_AMDuProfPcm分批实验命令.md`

## 说明
- 原始命令按固定格式组织：`AMDuProfPcm profile -m xxx -a --start-delay 10000 -d 60 -O ... -- evalscope perf ...`
- 原始页注明：每批建议跑 `3` 次取中位数。
- 原始页注明：把指标控制在 `6` 个以内有助于降低复用概率，但不保证完全无 multiplexing。

## 一次性前置

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

## `vllm serve` 命令

```bash
# 修改 NPS 后，需要同步修改 VLLM_CPU_KVCACHE_SPACE 与 --tensor-parallel-size
VLLM_CPU_KVCACHE_SPACE=32 \
vllm serve /models/Qwen3-30B-A3B \
    --served-model-name Qwen3-30B-A3B \
    --port 8122 \
    --dtype bfloat16 \
    --block-size 128 \
    --max-num-seqs 64 \
    --max_model_len 8192 \
    --load_format dummy \
    --no-enable-prefix-caching \
    --tensor-parallel-size 2
```

## `evalscope perf` 命令

带扫参输出目录版本：

```bash
# 修改 NPS 后，需要同步修改 --outputs-dir
evalscope perf \
    --model Qwen3-30B-A3B \
    --parallel 1 2 4 8 16 32 64 \
    --number 1 2 4 8 16 32 64 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/Qwen3-30B-A3B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/evalscope_res
```

固定负载版本：

```bash
evalscope perf \
    --model Qwen3-30B-A3B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/Qwen3-30B-A3B
```

## Batch-1：内存互连 + miss/TLB

```bash
# 修改 NPS 后，需要同步修改 -O 与 --outputs-dir
AMDuProfPcm profile \
    -m ipc,memory,ccm_bw,l1,cache_miss,tlb \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/PCm_res/batch1_mem_ccm_l1_miss_tlb -- \
evalscope perf \
    --model Qwen3-30B-A3B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/Qwen3-30B-A3B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/evalscope_res/batch1_mem_ccm_l1_miss_tlb
```

## Batch-2：缓存来源

```bash
AMDuProfPcm profile \
    -m ipc,l3,dc \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/PCm_res/batch2_l3_dc -- \
evalscope perf \
    --model Qwen3-30B-A3B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/Qwen3-30B-A3B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/evalscope_res/batch2_l3_dc
```

## Batch-3：预取与 L2

```bash
AMDuProfPcm profile \
    -m ipc,l2,swpfdc,hwpfdc \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/PCm_res/batch3_l2_prefetch -- \
evalscope perf \
    --model Qwen3-30B-A3B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/Qwen3-30B-A3B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/evalscope_res/batch3_l2_prefetch
```

## Batch-4：Topdown

```bash
AMDuProfPcm profile \
    -m ipc,pipeline_util \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/PCm_res/batch4_topdown -- \
evalscope perf \
    --model Qwen3-30B-A3B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/Qwen3-30B-A3B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/evalscope_res/batch4_topdown
```

## Batch-5：UMC

```bash
# UMC 需要 --msr 模式
AMDuProfPcm profile \
    -m ipc,umc \
    -a --msr --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/PCm_res/batch5_umc_msr -- \
evalscope perf \
    --model Qwen3-30B-A3B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/Qwen3-30B-A3B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/evalscope_res/batch5_umc_msr
```

## Batch-6：Roofline

```bash
AMDuProfPcm roofline \
    -a --start-delay 10000 -d 60 \
    -O /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/PCm_res/batch6_roofline -- \
evalscope perf \
    --model Qwen3-30B-A3B \
    --parallel 16 \
    --number 16 \
    --url http://0.0.0.0:8122/v1/chat/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --tokenizer-path /models/Qwen3-30B-A3B \
    --outputs-dir /home/zjj/vllm/test_results/NPS_Test/Qwen3-30B-A3B/NPS1_TP2/evalscope_res/batch6_roofline
```

## 执行顺序
1. `batch1_mem_ccm_l1_miss_tlb`
2. `batch2_l3_dc`
3. `batch3_l2_prefetch`
4. `batch4_topdown`
5. `batch5_umc_msr`
6. `batch6_roofline`
