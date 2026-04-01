# Qwen3-30B-A3B：attention 微基准实验步骤

> 更新时间：2026-03-23 23:27 +0800  
> 目的：在现有 `7.1/7.2` 端到端 `strict-batch` 分相结果之外，补一组直接打到 `cpu_attention_with_kv_cache` 的 attention kernel 微基准数据。  
> 入口：`benchmarks/kernels/cpu/benchmark_cpu_attn.py`  
> 当前机器：`2 x AMD EPYC 9745 128-Core Processor`  
> 假设：为了和 `/models/Qwen3-30B-A3B + TP=2` 的端到端实验对齐，这里采用 **每个 TP rank 的本地 attention shape**，即 `16` 个 query heads、`2` 个 KV heads、`head_dim=128`。这个 shape 来自 `/models/Qwen3-30B-A3B/config.json` 的 `32` 个 attention heads、`4` 个 KV heads，以及 `vllm/model_executor/models/qwen3_moe.py` 里的 TP 按 head 切分逻辑。

## 0. 这份文档回答什么问题

- `benchmark_cpu_attn.py` 直接调用 `cpu_attn_reshape_and_cache`、`cpu_attn_get_scheduler_metadata`、`cpu_attention_with_kv_cache`，因此它测的是 **CPU attention 核心算子路径**，不是完整的 vLLM request 执行链，也不是完整 decoder layer。证据：`benchmarks/kernels/cpu/benchmark_cpu_attn.py:46-158`
- 这个微基准不加载 `/models/Qwen3-30B-A3B` 的真实权重；它只用从模型配置推出来的 attention shape。证据：同文件 `main(...)` 参数与张量构造逻辑。
- 它内部自己构造输入：`block_tables` 用 `torch.randint(...)` 随机生成，`slot_mapping` 用连续 `torch.arange(...)` 生成，`query/key/value` 也都是合成张量。证据：`benchmarks/kernels/cpu/benchmark_cpu_attn.py:80-114`
- 因此，这份微基准适合回答“attention 核心算子在某种 shape 下怎么表现”，不适合直接回答“端到端 `Prefill/Decode` 差异主要来自哪一层”。

## 0.1 使用边界

- 如果当前问题是“`decode` 的高 `L3 miss` 到底主要来自 attention、MoE 还是 TP 通信”，优先做 `AMDuProfCLI hotspots + IBS L3-miss`，不要先看这份微基准。
- 如果当前问题是“在控制住 shape 以后，attention 核心算子本身对 `kv_split`、`q_len/kv_len`、局部性是否敏感”，这份文档才是合适入口。
- 解释结果时要记住：真实路径里 `cpu_attention_with_kv_cache` 之后还有 `o_proj`、可能的 TP `all_reduce`，以及更上游真实 `block_table/slot_mapping` 组织；这些都不在本微基准范围内。

## 1. 一次性前置

```bash
cd /home/zjj/vllm

EXP_ROOT=/home/zjj/vllm/test_results/attn_kernel/Qwen3-30B-A3B/
VLLM_ENV=${CONDA_PREFIX:-/home/zjj/.conda/envs/vllm-cpu}
mkdir -p "${EXP_ROOT}"
```

### 1.1 准备 AMDuProfPcm

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

## 2. 本次采用的 attention shape

- 总模型配置：`32` 个 query heads，`4` 个 KV heads，`head_dim=128`
- 当前对齐目标：`TP=2`
- 所以每个 TP rank 的本地 shape 取：
  - `--num-query-heads 16`
  - `--num-kv-heads 2`
  - `--head-size 128`

这不是拍脑袋设的，而是为了和当前 `Qwen3-30B-A3B + TP=2` 的 rank 内 attention 计算边界一致。

## 3. 

### 3.2 decode-like：启 `kv_split`

```bash
EXP_ROOT=/home/zjj/vllm/test_results/attn_kernel/Qwen3-30B-A3B/
python benchmarks/kernels/cpu/benchmark_cpu_attn.py \
  --batch-size 16 \
  --q-len-min 1 --q-len-max 1 \
  --kv-len-min 1024 --kv-len-max 1024 \
  --num-query-heads 16 \
  --num-kv-heads 2 \
  --head-size 128 \
  --dtype half \
  --block-size 128 \
  --enable-kv-split \
  --iters 2000 \
  | tee "${EXP_ROOT}/decode_q1_kv1024_split.log"
```

对应 PCM：

```bash
EXP_ROOT=/home/zjj/vllm/test_results/attn_kernel/Qwen3-30B-A3B/
SCENARIO=decode_like
METRIC=metric1_l3_dc_l2_memory
PCM_DIR=${EXP_ROOT}/pcm_res/${SCENARIO}/${METRIC}

AMDuProfPcm profile \
  -m ipc,l3,dc,l2,memory \
  -a -s --start-delay 10000 -d 120 \
  -O ${PCM_DIR} -- \
env LD_PRELOAD="${LD_PRELOAD:-}" \
    LD_LIBRARY_PATH="${VLLM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
python benchmarks/kernels/cpu/benchmark_cpu_attn.py \
  --batch-size 16 \
  --q-len-min 1 --q-len-max 1 \
  --kv-len-min 1024 --kv-len-max 1024 \
  --num-query-heads 16 \
  --num-kv-heads 2 \
  --head-size 128 \
  --dtype half \
  --block-size 128 \
  --enable-kv-split \
  --iters 1000000
```
### 3.4 prefill-like：启 `kv_split`

```bash
python benchmarks/kernels/cpu/benchmark_cpu_attn.py \
  --batch-size 16 \
  --q-len-min 1024 --q-len-max 1024 \
  --kv-len-min 1024 --kv-len-max 1024 \
  --num-query-heads 16 \
  --num-kv-heads 2 \
  --head-size 128 \
  --dtype half \
  --block-size 128 \
  --enable-kv-split \
  --iters 2000 \
  | tee "${EXP_ROOT}/prefill_q1024_kv1024_split.log"
```

对应 PCM：

```bash
EXP_ROOT=/home/zjj/vllm/test_results/attn_kernel/Qwen3-30B-A3B/
SCENARIO=prefill_like
METRIC=metric1_l3_dc_l2_memory
PCM_DIR=${EXP_ROOT}/pcm_res/${SCENARIO}/${METRIC}

AMDuProfPcm profile \
  -m ipc,l3,dc,l2,memory \
  -a -s --start-delay 15000 -d 120 \
  -O ${PCM_DIR} -- \
env LD_PRELOAD="${LD_PRELOAD:-}" \
    LD_LIBRARY_PATH="${VLLM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
python benchmarks/kernels/cpu/benchmark_cpu_attn.py \
  --batch-size 16 \
  --q-len-min 1024 --q-len-max 1024 \
  --kv-len-min 1024 --kv-len-max 1024 \
  --num-query-heads 16 \
  --num-kv-heads 2 \
  --head-size 128 \
  --dtype half \
  --block-size 128 \
  --enable-kv-split \
  --iters 500000 
```

- `事实`：`benchmark_cpu_attn.py` 内部固定先 warmup `5` 次，再正式跑 `--iters` 次。证据：`benchmarks/kernels/cpu/benchmark_cpu_attn.py:145-172`
- `假设`：attention 微基准单轮通常比端到端 `strict-batch` 短很多，所以上面的 PCM 命令把 `--iters` 放大到 `200/2000`；若稳定窗口仍太短，优先继续增大 `--iters`
- `建议`：如果要给 `3.1` 或 `3.4` 补 PCM，对应复用上面的 decode/prefill 命令，只改 `--enable-kv-split` 和 `BATCH`
- `建议`：性能结论优先绑定 `${EXP_ROOT}/pcm_res/<scenario>/<batch>/` 下的 timeseries/cumulative 文件，再和 `${EXP_ROOT}/*.log` 里的 `mean (ms)` 对齐，不要只看终端瞬时输出

## 4. 时间够再补 1 组 mixed

### 4.1 mixed：`q_len` 和 `kv_len` 同时拉开

```bash
python benchmarks/kernels/cpu/benchmark_cpu_attn.py \
  --batch-size 16 \
  --q-len-min 1 --q-len-max 1024 \
  --kv-len-min 1024 --kv-len-max 4096 \
  --num-query-heads 16 \
  --num-kv-heads 2 \
  --head-size 128 \
  --dtype half \
  --block-size 128 \
  --enable-kv-split \
  --iters 30 \
  | tee "${EXP_ROOT}/mixed_q1_1024_kv1024_4096_split.log"
```
