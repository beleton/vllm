# Qwen3-30B-A3B：attention-only TP 多进程实验步骤

> 更新时间：2026-04-07 16:38:21 +0800  
> 目的：用 `attention-only` 多进程脚本，在尽量复用真实 `CPUWorker.init_device()` / `init_cpu_threads_env()` 绑核口径的前提下，观察当前 CPU attention 路径在 `TP` 多进程环境下的 `L3/CCX` 局部性指标，并支持对照 `balanced` 与 `acc-local-l3`。  
> 入口脚本：`benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py`  
> 共享绑核逻辑：`vllm/v1/worker/cpu_binding.py`、`vllm/v1/worker/cpu_worker.py`、`csrc/cpu/utils.cpp`

## 0. 这份文档回答什么问题

- 这份脚本会先走 `cpu_attn_reshape_and_cache`，然后根据 `--attn-locality-mode` 选择 `cpu_attn_get_scheduler_metadata/cpu_attention_with_kv_cache` 或 `cpu_attn_get_scheduler_metadata_acc_locality/cpu_attention_with_kv_cache_acc_locality`，并在每个 rank 内调用 `torch.ops._C_utils.init_cpu_threads_env(...)` 绑定线程与内存。证据：`benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py`、`benchmarks/kernels/cpu/benchmark_cpu_attn.py`、`vllm/v1/attention/backends/cpu_attn.py`、`csrc/cpu/utils.cpp`。
- 它复用了真实 `CPUWorker.init_device()` 的 rank 选核规则：x86 下每个 physical core 只取 1 个 SMT 线程，按 `allowed_numa_nodes[local_rank]` 把 rank 映射到当前可见 NUMA node。证据：`vllm/v1/worker/cpu_binding.py`、`vllm/v1/worker/cpu_worker.py`。
- 它不加载真实模型权重，不复现真实 token 数据值，也不直接识别 `NPS`。它只能识别当前机器在运行时暴露出来的 `CPU/CORE/NODE` 拓扑。

## 1. 脚本怎么用

最小 dry-run：

```bash
cd /home/zjj/vllm

python benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py \
  --tp-size 2 \
  --partition-mode global-fixed \
  --workload decode-like \
  --batch-size 16 \
  --q-len 1 \
  --kv-len 1024 \
  --num-query-heads 32 \
  --num-kv-heads 4 \
  --head-size 128 \
  --dtype bfloat16 \
  --block-size 128 \
  --enable-kv-split \
  --warmup-iters 200 \
  --iters 20000 \
  --output-json /home/zjj/vllm/test_results/P2_AttnOnly/_examples/batch_16/q1_KV_1024/attn_only_tp2_decode_global_fixed.json
```

参数口径：

- `--tp-size`：进程数，也是 TP rank 数。
- `--batch-size`：每个 rank 共同处理的序列数，也就是 `seq_lens` 的 batch 维。
- `--q-len / --kv-len`：当前 attention kernel 的真实 shape 参数，也是推荐的实验口径。
  - `decode-like` 下通常写成 `q_len=1, kv_len=1024`。
  - `prefill-like` 下通常写成 `q_len=kv_len`，例如 `q_len=1024, kv_len=1024`。
- `--partition-mode global-fixed`：全局 head 数固定；`TP` 越大，每个 rank 的 head shard 越小。
- `--partition-mode per-rank-fixed`：每个 rank 的 head 数固定；`TP` 越大，总 head 工作量越大。
- `--workload decode-like`：默认 `q_len=1, kv_len=1024`。
- `--workload prefill-like`：默认 `q_len=kv_len=1024`。
- `--workload custom`：需显式给 `--q-len` 和 `--kv-len`。
- `--num-query-heads / --num-kv-heads`：
  - 在 `global-fixed` 下，它们表示全局总 head 数。
  - 在 `per-rank-fixed` 下，它们表示每个 rank 的本地 head 数。
- `--attn-locality-mode`：
  - `balanced`：旧路径，对应原有 `cpu_attention_with_kv_cache`。
  - `acc-local-l3`：新路径，对应 `cpu_attention_with_kv_cache_acc_locality`。
- `--attn-locality-group-span`：当前先固定传 `1`。脚本接口已保留，但当前实现还没有真正扩展到 `group_span > 1`。
- `--output-json`：写出当前 run 的 rank 绑核列表、head shard 口径和每 rank 计时结果。
- 结果目录建议显式带上解析后的 kernel shape，例如 `batch_16/q1_KV_1024`；脚本 summary 里也会返回同名的 `result_shape_dirname` 字段，便于外部 wrapper 复用。
- `--iters` 与 `--min-runtime-s`：二选一。
  - `--iters` 适合 dry-run 或短测，会保留逐轮 `times_ms`。
  - `--min-runtime-s` 适合 `AMDuProfPcm`，表示 warmup 之后至少继续跑这么多秒；该路径只保留聚合统计和实际运行轮数，不保留逐轮 `times_ms`，避免 decode 场景产生超大 JSON。

## 2. Qwen3-30B-A3B 的 head 口径

- `/models/Qwen3-30B-A3B/config.json` 里，`num_attention_heads=32`，`num_key_value_heads=4`，`head_dim=128`。
- 因此，对 `Qwen3-30B-A3B`：
  - `global-fixed` 口径应传 `--num-query-heads 32 --num-kv-heads 4`
  - 若要在 `per-rank-fixed` 里保持“TP=2 的本地 shape”不变，则传 `--num-query-heads 16 --num-kv-heads 2`

## 3. 下一步实验顺序

建议按下面顺序推进：

1. 先在当前 NPS 拓扑下做 `global-fixed` 的 `decode-like` dry-run，确认单轮耗时和绑核结果。
2. 在同一拓扑下补 `global-fixed` 的 `prefill-like`。
3. 正式 `AMDuProfPcm` 采样时，不再用固定 `--iters` 去“估算 110 秒”；统一用 `--min-runtime-s`，让 benchmark 在 warmup 之后持续运行到目标 steady-state 时长。
4. `AMDuProfPcm -d` 从 profiler 启动开始计时，包含 `--start-delay`；正式口径不再让 `-d` 去“包住 benchmark 全程”，而是让 benchmark 的 `--min-runtime-s` 明显长于 `-d`，保证 profiler 只截取稳定运行窗口。
5. 当前推荐关系：`benchmark_min_runtime_s > pcm_duration_s`，起步可取 `benchmark_min_runtime_s = pcm_duration_s + 10`。
6. `AMDuProfPcm` 默认采样间隔是 `1000ms`，正式实验应显式传 `-I <ms>`。第一轮建议从 `-I 100` 开始。
7. 完成 `NPS1_TP2 / NPS2_TP4 / NPS4_TP8` 的 `global-fixed` 后，再补 `per-rank-fixed`。
8. 写结论时，`global-fixed` 和 `per-rank-fixed` 必须分开解释，不能把收益直接写成 locality 改善。

## 4. 可直接复制执行的命令

### 4.1 一次性环境变量

```bash
cd /home/zjj/vllm

PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_ENV=${CONDA_PREFIX:-/home/zjj/.conda/envs/vllm-cpu}
PCM_BIN=/opt/AMDuProf_5.2-606/bin/AMDuProfPcm
RESULT_ROOT=/home/zjj/vllm/test_results/P3_AttnOnly/Qwen3-30B-A3B
SCRIPT=/home/zjj/vllm/benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py

mkdir -p "${RESULT_ROOT}"
```

如果还没准备 `AMDuProfPcm`：

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

```bash
# 运行时日志
conda run -n vllm-cpu env VLLM_CPU_ATTN_ACC_LOCALITY_DEBUG=1 python benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py \
  --tp-size 2 \
  --partition-mode global-fixed \
  --workload prefill-like \
  --batch-size 1 \
  --kv-len 128 \
  --num-query-heads 32 \
  --num-kv-heads 8 \
  --head-size 128 \
  --block-size 32 \
  --dtype float \
  --warmup-iters 0 \
  --iters 1 \
  --attn-locality-mode acc-local-l3 2>&1 | tee benchmark_cpu_attn_mp.log
  
```

### 4.2 定义 dry-run / PCM helper

```bash
dry_run_attn_only() {
  local exp_tag="$1"
  local tp_size="$2"
  local workload="$3"
  local mode="$4"
  local batch_size="$5"
  local q_len="$6"
  local kv_len="$7"
  local nq="$8"
  local nkv="$9"
  local locality_mode="${10}"
  local out_dir="${RESULT_ROOT}/${exp_tag}/${workload}/${mode}/batch_${batch_size}/q${q_len}_kv${kv_len}/${locality_mode}"

  mkdir -p "${out_dir}"
  lscpu -J -e=CPU,CORE,NODE > "${out_dir}/lscpu_topology.json"
  numactl -H > "${out_dir}/numactl_h.txt"

  env LD_PRELOAD="${LD_PRELOAD:-}" \
      LD_LIBRARY_PATH="${VLLM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
      --tp-size "${tp_size}" \
      --partition-mode "${mode}" \
      --workload "${workload}" \
      --batch-size "${batch_size}" \
      --q-len "${q_len}" \
      --kv-len "${kv_len}" \
      --num-query-heads "${nq}" \
      --num-kv-heads "${nkv}" \
      --head-size 128 \
      --dtype bfloat16 \
      --block-size 128 \
      --enable-kv-split \
      --omp-threads-bind auto \
      --warmup-iters 10 \
      --iters 300 \
      --attn-locality-mode "${locality_mode}" \
      --output-json "${out_dir}/dry_run_summary.json" \
    | tee "${out_dir}/dry_run.log"
}

run_attn_only_pcm() {
  local exp_tag="$1"
  local tp_size="$2"
  local workload="$3"
  local mode="$4"
  local batch_size="$5"
  local q_len="$6"
  local kv_len="$7"
  local nq="$8"
  local nkv="$9"
  local start_delay_ms="${10}"
  local pcm_duration_s="${11}"
  local benchmark_min_runtime_s="${12}"
  local interval_ms="${13}"
  local out_dir="${RESULT_ROOT}/${exp_tag}/${workload}/${mode}/batch_${batch_size}/q${q_len}_kv${kv_len}"

  mkdir -p "${out_dir}"
  lscpu -J -e=CPU,CORE,NODE > "${out_dir}/lscpu_topology.json"
  numactl -H > "${out_dir}/numactl_h.txt"

  if [ "${benchmark_min_runtime_s}" -le "${pcm_duration_s}" ]; then
    echo "benchmark_min_runtime_s must be > pcm_duration_s" >&2
    return 1
  fi

  "${PCM_BIN}" profile \
    -m ipc,l3,dc \
    -a -s --start-delay "${start_delay_ms}" -d "${pcm_duration_s}" -I "${interval_ms}" \
    -O "${out_dir}/pcm_l3_dc" -- \
  env LD_PRELOAD="${LD_PRELOAD:-}" \
      LD_LIBRARY_PATH="${VLLM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
      --tp-size "${tp_size}" \
      --partition-mode "${mode}" \
      --workload "${workload}" \
      --batch-size "${batch_size}" \
      --q-len "${q_len}" \
      --kv-len "${kv_len}" \
      --num-query-heads "${nq}" \
      --num-kv-heads "${nkv}" \
      --head-size 128 \
      --dtype bfloat16 \
      --block-size 128 \
      --enable-kv-split \
      --omp-threads-bind auto \
      --warmup-iters 200 \
      --min-runtime-s "${benchmark_min_runtime_s}"
}

run_attn_only_pcm_locality() {
  local exp_tag="$1"
  local tp_size="$2"
  local workload="$3"
  local mode="$4"
  local batch_size="$5"
  local q_len="$6"
  local kv_len="$7"
  local nq="$8"
  local nkv="$9"
  local locality_mode="${10}"
  local start_delay_ms="${11}"
  local pcm_duration_s="${12}"
  local benchmark_min_runtime_s="${13}"
  local interval_ms="${14}"
  local out_dir="${RESULT_ROOT}/${exp_tag}/${workload}/${mode}/batch_${batch_size}/q${q_len}_kv${kv_len}/${locality_mode}"

  mkdir -p "${out_dir}"
  lscpu -J -e=CPU,CORE,NODE > "${out_dir}/lscpu_topology.json"
  numactl -H > "${out_dir}/numactl_h.txt"

  if [ "${benchmark_min_runtime_s}" -le "${pcm_duration_s}" ]; then
    echo "benchmark_min_runtime_s must be > pcm_duration_s" >&2
    return 1
  fi

  "${PCM_BIN}" profile \
    -m ipc,l3,dc \
    -a -s --start-delay "${start_delay_ms}" -d "${pcm_duration_s}" -I "${interval_ms}" \
    -O "${out_dir}/pcm_l3_dc" -- \
  env LD_PRELOAD="${LD_PRELOAD:-}" \
      LD_LIBRARY_PATH="${VLLM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
      --tp-size "${tp_size}" \
      --partition-mode "${mode}" \
      --workload "${workload}" \
      --batch-size "${batch_size}" \
      --q-len "${q_len}" \
      --kv-len "${kv_len}" \
      --num-query-heads "${nq}" \
      --num-kv-heads "${nkv}" \
      --head-size 128 \
      --dtype bfloat16 \
      --block-size 128 \
      --enable-kv-split \
      --omp-threads-bind auto \
      --warmup-iters 200 \
      --min-runtime-s "${benchmark_min_runtime_s}" \
      --attn-locality-mode "${locality_mode}" \
      --attn-locality-group-span 1 \
      --output-json "${out_dir}/summary.json" \
    | tee "${out_dir}/run.log"
}
```

这里的 `pcm_duration_s` 只控制 profiler 采样时长；`benchmark_min_runtime_s` 要显式设得更长，避免 profiler 在程序收尾阶段采样。

### 4.3 测当前 `acc-local-l3` 实现

如果目标是验证新实现是否比旧路径更好，最简单的口径不是改脚本，也不是新建脚本，而是在同一个 case 下分别跑：

- `--attn-locality-mode balanced`
- `--attn-locality-mode acc-local-l3`

当前推荐先用已有证据最强的场景：

- `NPS1_TP2`
- `global-fixed`
- `prefill-like`
- `batch_size=16`
- `q_len=kv_len=128` 和 `q_len=kv_len=512`

直接命令：

```bash
for locality_mode in balanced acc-local-l3; do
  run_attn_only_pcm_locality NPS1_TP2 2 prefill-like global-fixed 16 128 128 32 4 "${locality_mode}" 20000 100 150 250
done

for locality_mode in balanced acc-local-l3; do
  run_attn_only_pcm_locality NPS1_TP2 2 prefill-like global-fixed 16 512 512 32 4 "${locality_mode}" 20000 100 150 250
done
```

如果只想先做不带 `AMDuProfPcm` 的短测，也应保持同一个 case 下成对对照：

```bash
for locality_mode in balanced acc-local-l3; do
  dry_run_attn_only NPS1_TP2 2 prefill-like global-fixed 16 128 128 32 4 "${locality_mode}"
done

for locality_mode in balanced acc-local-l3; do
  dry_run_attn_only NPS1_TP2 2 prefill-like global-fixed 16 512 512 32 4 "${locality_mode}"
done
```

### 4.4 第一轮：`global-fixed`

下面这组命令里：

- `32/4` 表示 `Qwen3-30B-A3B` 的全局总 head 数
- `16/1/1024` 或 `16/1024/1024` 分别表示 `batch_size/q_len/kv_len`

#### `NPS1_TP2`

```bash
for locality_mode in balanced acc-local-l3; do
  dry_run_attn_only NPS1_TP2 2 decode-like global-fixed 16 1 256 32 4 "${locality_mode}"
  dry_run_attn_only NPS1_TP2 2 prefill-like global-fixed 16 128 128 32 4 "${locality_mode}"
done

for locality_mode in balanced acc-local-l3; do
  run_attn_only_pcm_locality NPS1_TP2 2 decode-like global-fixed 16 1 128 32 4 "${locality_mode}" 20000 100 150 200
  run_attn_only_pcm_locality NPS1_TP2 2 prefill-like global-fixed 16 128 128 32 4 "${locality_mode}" 20000 100 150 200
done
```

#### `NPS2_TP4`

```bash
for locality_mode in balanced acc-local-l3; do
  dry_run_attn_only NPS2_TP4 4 decode-like global-fixed 16 1 1024 32 4 "${locality_mode}"
  dry_run_attn_only NPS2_TP4 4 prefill-like global-fixed 16 1024 1024 32 4 "${locality_mode}"
done

for locality_mode in balanced acc-local-l3; do
  run_attn_only_pcm_locality NPS2_TP4 4 decode-like global-fixed 16 1 1024 32 4 "${locality_mode}" 10000 120 130 100
  run_attn_only_pcm_locality NPS2_TP4 4 prefill-like global-fixed 16 1024 1024 32 4 "${locality_mode}" 10000 120 130 100
done
```

#### `NPS4_TP8`

```bash
for locality_mode in balanced acc-local-l3; do
  dry_run_attn_only NPS4_TP8 8 decode-like global-fixed 16 1 1024 32 4 "${locality_mode}"
  dry_run_attn_only NPS4_TP8 8 prefill-like global-fixed 16 1024 1024 32 4 "${locality_mode}"
done

for locality_mode in balanced acc-local-l3; do
  run_attn_only_pcm_locality NPS4_TP8 8 decode-like global-fixed 16 1 1024 32 4 "${locality_mode}" 10000 120 130 100
  run_attn_only_pcm_locality NPS4_TP8 8 prefill-like global-fixed 16 1024 1024 32 4 "${locality_mode}" 10000 120 130 100
done
```

### 4.5 第二轮：`per-rank-fixed`

如果要把“每个 rank 工作量固定”单独拿出来做对照，建议先固定成 `TP=2` 本地 shape，也就是每个 rank `16/2`：

#### `NPS1_TP2 / NPS2_TP4 / NPS4_TP8`

```bash
for locality_mode in balanced acc-local-l3; do
  dry_run_attn_only NPS1_TP2 2 decode-like per-rank-fixed 16 1 1024 16 2 "${locality_mode}"
  dry_run_attn_only NPS2_TP4 4 decode-like per-rank-fixed 16 1 1024 16 2 "${locality_mode}"
  dry_run_attn_only NPS4_TP8 8 decode-like per-rank-fixed 16 1 1024 16 2 "${locality_mode}"
done

for locality_mode in balanced acc-local-l3; do
  run_attn_only_pcm_locality NPS1_TP2 2 decode-like per-rank-fixed 16 1 1024 16 2 "${locality_mode}" 10000 120 130 100
  run_attn_only_pcm_locality NPS2_TP4 4 decode-like per-rank-fixed 16 1 1024 16 2 "${locality_mode}" 10000 120 130 100
  run_attn_only_pcm_locality NPS4_TP8 8 decode-like per-rank-fixed 16 1 1024 16 2 "${locality_mode}" 10000 120 130 100
done
```

prefill-like 对照：

```bash
for locality_mode in balanced acc-local-l3; do
  dry_run_attn_only NPS1_TP2 2 prefill-like per-rank-fixed 16 1024 1024 16 2 "${locality_mode}"
  dry_run_attn_only NPS2_TP4 4 prefill-like per-rank-fixed 16 1024 1024 16 2 "${locality_mode}"
  dry_run_attn_only NPS4_TP8 8 prefill-like per-rank-fixed 16 1024 1024 16 2 "${locality_mode}"
done

for locality_mode in balanced acc-local-l3; do
  run_attn_only_pcm_locality NPS1_TP2 2 prefill-like per-rank-fixed 16 1024 1024 16 2 "${locality_mode}" 10000 120 130 100
  run_attn_only_pcm_locality NPS2_TP4 4 prefill-like per-rank-fixed 16 1024 1024 16 2 "${locality_mode}" 10000 120 130 100
  run_attn_only_pcm_locality NPS4_TP8 8 prefill-like per-rank-fixed 16 1024 1024 16 2 "${locality_mode}" 10000 120 130 100
done
```

## 5. 结果解释边界

- 如果 `global-fixed` 改善，而 `per-rank-fixed` 不改善，不能直接写成 locality 变好；更可能是每个 rank 的工作量变小。
- 如果两种口径都同方向改善，而且 `L3 Miss %`、`Ave L3 Miss Latency`、`L3 Miss Latency From another CCX in same node (%)`、`Remote DRAM Reads %` 也同步变化，才可以写成“attention-only 场景暴露出可观测 locality 优化空间”。
- 即便如此，也不能直接把 attention-only 的结果写成端到端收益结论；端到端场景还包含 `o_proj`、`TP all_reduce`、`MoE/MLP` 等其他路径。
