# DeepSeek-R1-Distill-Llama-8B：AMDuProfPcm 分批实验命令（NPS4_TP8）
---

## 1. 一次性前置（每次开新机器/重启后先确认）

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

vllm命令：
``` bash
# 修改NPS后，修改VLLM_CPU_KVCACHE_SPACE, --tensor-parallel-size
VLLM_CPU_KVCACHE_SPACE=8 \
vllm serve /models/DeepSeek-R1-Distill-Llama-8B \
	--served-model-name DeepSeek-R1-Distill-Llama-8B \
	--port 8122 \
	--dtype float16 \
	--block-size 128 \
	--max-num-seqs 64 \
	--max_model_len 8192 \
	--load_format "dummy" \
	--no-enable-prefix-caching \
	--tensor-parallel-size 8

```

evalscope命令：
``` bash
# 修改NPS后，修改--outputs-dir
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
	--outputs-dir /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res
```
---

## 2. Batch-1：内存互连 + miss/TLB

```bash
# 修改NPS后，修改-O, --outputs-dir
AMDuProfPcm profile \
-m ipc,memory,ccm_bw,l1,cache_miss,tlb \
-a --start-delay 10000 -d 60 \
-O /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/PCm_res/batch1_mem_ccm_l1_miss_tlb  -- \
evalscope perf \
	--model DeepSeek-R1-Distill-Llama-8B \
	--parallel 16 \
	--number  16 \
	--url http://0.0.0.0:8122/v1/chat/completions \
	--api openai \
	--dataset random \
	--max-tokens 1024 --min-tokens 1024 \
	--prefix-length 0 \
	--min-prompt-length 1024 --max-prompt-length 1024 \
	--tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
	--outputs-dir  /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res/batch1_mem_ccm_l1_miss_tlb
```

---

## 3. Batch-2：缓存来源（L3 + DC）

```bash
AMDuProfPcm profile \
-m ipc,l3,dc \
-a --start-delay 10000 -d 60 \
-O /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/PCm_res/batch2_l3_dc  -- \
evalscope perf \
	--model DeepSeek-R1-Distill-Llama-8B \
	--parallel 16 \
	--number  16 \
	--url http://0.0.0.0:8122/v1/chat/completions \
	--api openai \
	--dataset random \
	--max-tokens 1024 --min-tokens 1024 \
	--prefix-length 0 \
	--min-prompt-length 1024 --max-prompt-length 1024 \
	--tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
	--outputs-dir  /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res/batch2_l3_dc
```

---

## 4. Batch-3：预取与 L2

```bash
AMDuProfPcm profile \
-m ipc,l2,swpfdc,hwpfdc \
-a --start-delay 10000 -d 60 \
-O /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/PCm_res/batch3_l2_prefetch  -- \
evalscope perf \
	--model DeepSeek-R1-Distill-Llama-8B \
	--parallel 16 \
	--number  16 \
	--url http://0.0.0.0:8122/v1/chat/completions \
	--api openai \
	--dataset random \
	--max-tokens 1024 --min-tokens 1024 \
	--prefix-length 0 \
	--min-prompt-length 1024 --max-prompt-length 1024 \
	--tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
	--outputs-dir  /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res/batch3_l2_prefetch
```

---

## 5. Batch-4：Topdown（单独采）

```bash
AMDuProfPcm profile \
-m ipc,pipeline_util \
-a --start-delay 10000 -d 60 \
-O /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/PCm_res/batch4_topdown  -- \
evalscope perf \
	--model DeepSeek-R1-Distill-Llama-8B \
	--parallel 16 \
	--number  16 \
	--url http://0.0.0.0:8122/v1/chat/completions \
	--api openai \
	--dataset random \
	--max-tokens 1024 --min-tokens 1024 \
	--prefix-length 0 \
	--min-prompt-length 1024 --max-prompt-length 1024 \
	--tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
	--outputs-dir  /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res/batch4_topdown
```

---

## 6. Batch-5：UMC（单独批次，MSR 模式）

```bash
# UMC 需要 --msr 模式
AMDuProfPcm profile \
-m ipc,umc \
-a --msr --start-delay 10000 -d 60 \
-O /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/PCm_res/batch5_umc_msr  -- \
evalscope perf \
	--model DeepSeek-R1-Distill-Llama-8B \
	--parallel 16 \
	--number  16 \
	--url http://0.0.0.0:8122/v1/chat/completions \
	--api openai \
	--dataset random \
	--max-tokens 1024 --min-tokens 1024 \
	--prefix-length 0 \
	--min-prompt-length 1024 --max-prompt-length 1024 \
	--tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
	--outputs-dir  /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res/batch5_umc_msr
```

---

## 7. Batch-6：Roofline（单独）

```bash
AMDuProfPcm roofline \
-a --start-delay 10000 -d 60 \
-O /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/PCm_res/batch6_roofline  -- \
evalscope perf \
	--model DeepSeek-R1-Distill-Llama-8B \
	--parallel 16 \
	--number  16 \
	--url http://0.0.0.0:8122/v1/chat/completions \
	--api openai \
	--dataset random \
	--max-tokens 1024 --min-tokens 1024 \
	--prefix-length 0 \
	--min-prompt-length 1024 --max-prompt-length 1024 \
	--tokenizer-path /models/DeepSeek-R1-Distill-Llama-8B \
	--outputs-dir  /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS4_TP8/evalscope_res/batch6_roofline
```

---

## 8. 执行顺序建议

1. `batch1_mem_ccm_l1_miss_tlb`
2. `batch2_l3_dc`
3. `batch3_l2_prefetch`
4. `batch4_topdown`
5. `batch5_umc_msr`
6. `batch6_roofline`
