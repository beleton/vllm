# 实验记录模板（vLLM CPU / NPS-TP）

用途：统一实验记录口径，确保可复现、可比较、可回溯。  
原则：每轮实验尽量只改一个关键变量（单变量原则）。

---

## 1. 实验元信息

- 实验日期：
- 实验人员：
- 机器信息（CPU/内存/BIOS/NPS）：
- OS / Kernel：
- vLLM commit / 分支：
- Python / PyTorch / oneDNN / IPEX 版本：
- 模型：
- 数据集与请求配置：
- 对照组配置：
- 本轮唯一变量：

配置参数：

- `NPS`：
- `TP`：
- `dtype`：
- `concurrency`：
- `prompt length`：
- `output length`：

---

## 2. 命令区

### 2.1 启动服务（vllm serve）

```bash
vllm serve <model_path> \
  --served-model-name <model_name> \
  --port <port> \
  --dtype <dtype> \
  --block-size 128 \
  --max-num-seqs 64 \
  --max_model_len 8192 \
  --tensor-parallel-size <tp>
```

### 2.2 负载压测（evalscope perf）

```bash
evalscope perf \
  --model <model_name> \
  --parallel <parallel_list> \
  --number <number_list> \
  --url http://0.0.0.0:<port>/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens <output_len> --min-tokens <output_len> \
  --prefix-length 0 \
  --min-prompt-length <prompt_len> --max-prompt-length <prompt_len> \
  --tokenizer-path <model_path> \
  --outputs-dir <evalscope_output_dir>
```

### 2.3 性能计数器采集（AMDuProfPcm）

```bash
AMDuProfPcm profile \
  -m ipc,l1,l2,l3,swpfdc,hwpfdc,dc,pipeline_util,cache_miss,memory,ccm_bw \
  -a --start-delay 5000 -d 80 \
  -O <pcm_output_dir> -- \
  evalscope perf <same_or_targeted_args>
```

### 2.4 关键环境变量（若使用）

```bash
export VLLM_CPU_OMP_THREADS_BIND=<auto_or_manual>
export VLLM_CPU_NUM_OF_RESERVED_CPU=<num>
```

---

## 3. 结果区（固定指标）

### 3.1 核心性能指标

| 配置 | Avg TTFT(s) | Avg TPOT(s) | Output tok/s | Total tok/s |
| --- | ---: | ---: | ---: | ---: |
| 对照组 |  |  |  |  |
| 实验组 |  |  |  |  |

### 3.2 系统与微架构指标

| 配置 | CPI | Total Mem Bw(GB/s) | Ave L3 Miss Latency(ns) | Remote DRAM Reads % |
| --- | ---: | ---: | ---: | ---: |
| 对照组 |  |  |  |  |
| 实验组 |  |  |  |  |

数据来源路径：

- evalscope summary：
- PCM cumulative CSV：
- 其它日志：

---

## 4. 观察与异常

- 是否复现预期现象（是/否）：
- 主要观察（3 条以内）：
1.
2.
3.

- 异常信息：
- 可能原因（区分事实/推断）：

---

## 5. 结论与下一步

### 5.1 本轮结论

- 结论 1：
- 结论 2：

### 5.2 下一轮实验（继续单变量）

- 下一轮唯一变量：
- 下一轮对照配置：
- 预计观察指标：

### 5.3 复现实验命令快照

```bash
# 在此粘贴最终可复现命令集合
```

---

## 6. 填写完成检查

- [ ] 已记录绝对日期与结果目录时间戳  
- [ ] 已记录完整命令而非口述  
- [ ] 结论均可追溯到具体数据路径  
- [ ] 明确区分事实与推断  
- [ ] 下一轮只改一个关键变量  
