
## 实验1
### 目的
浪潮的工程师测出在配有2个AMD的Zen5架构物理CPU的机器上，开启NPS并修改相应的TP设置，会影响LLM推理的性能，如NPS2_TP4比NPS1_TP2性能更好，NPS4_TP8比NPS2_TP4性能更好，本次实验目的是复现这一现象。

### 步骤
1. 修改NPS：`AMD CBS -> DF Common Options-> Memory Addressing -> NUMA nodes per socket`

2. 使用AMDuProfPcm前准备工作
```
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

3. 启动vllm，其中--tensor-parallel-size根据不同的NPS来指定
``` bash
vllm serve /models/DeepSeek-R1-Distill-Llama-8B \
	--served-model-name DeepSeek-R1-Distill-Llama-8B \
	--port 8122 \
	--dtype bfloat16 \
	--block-size 128 \
	--max-num-seqs 64 \
	--max_model_len 8192 \
	--tensor-parallel-size 8
```

4. evalscope命令发起推理请求，测试性能
``` bash
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
	--outputs-dir /home/zjj/vllm/test_results/NPS1_TP2/evalscope_res
```

5. 结合AMDuProfPcm工具，测量系统的一些性能指标
``` bash
AMDuProfPcm profile \
-m ipc,l1,l2,l3,swpfdc,hwpfdc,dc,pipeline_util,cache_miss,memory,ccm_bw \
-a --start-delay 5000 -d 80 \
-O /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS2_TP4/PCm_res  -- \
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
	--outputs-dir  /home/zjj/vllm/test_results/DeepSeek-R1-Distill-Llama-8B/NPS2_TP4/evalscope_res
```

### 数据
不同NPS配置下的推理吞吐

| NPS | Avg TTFT(s) | Avg TPOT(s) | Tokens/s | 内存带宽(GB/s) | CPI  |
| --- | ----------- | ----------- | -------- | ---------- | ---- |
| 1   | 5.5891      | 0.0755      | 197.515  | 271.03     | 1.73 |
| 2   | 4.8408      | 0.0717      | 209.4394 | 302.4      | 1.72 |
| 4   | 10.2484     | 0.0945      | 153.2066 | 231.04     | 2.37 |

| NPS | 内存带宽(GB/s) | 吞吐量GFLOPs | 算术强度(FLOP/B) |
| --- | ---------- | --------- | ------------ |
| 1   | 284.4402   | 4769.1319 | 16.76        |
| 2   | 318.3504   | 3836.0272 | 12.04        |
| 4   | 235.5316   | 4015.8258 | 17.05        |

### 结论
发现NPS2_TP4相比NPS1_TP2有提升，但NPS4_TP8反而不如NPS1_TP2，与浪潮的结论不相符。

### 下一步计划
1. 浪潮采用dtype为float16，下次实验采用dtype为float16
2. 测试不同大小的模型在不同NPS下的性能
3. 需要深入分析vllm的相关源代码，从代码层面找出性能变化的原因