# Qwen3-30B-A3B attention-only TP 实验步骤

## 目的
- 用 `attention-only` 多进程脚本，在尽量复用真实 `CPUWorker.init_device()` / `init_cpu_threads_env()` 绑核口径的前提下，观察当前 CPU attention 路径在 `TP` 多进程环境下的 `L3/CCX` 局部性指标，并支持对照 `balanced` 与 `acc-local-l3`。

## 这份文档回答什么问题
- 这份脚本会先走 `cpu_attn_reshape_and_cache`，然后根据 `--attn-locality-mode` 选择 `cpu_attn_get_scheduler_metadata/cpu_attention_with_kv_cache` 或 `cpu_attn_get_scheduler_metadata_acc_locality/cpu_attention_with_kv_cache_acc_locality`，并在每个 rank 内调用 `torch.ops._C_utils.init_cpu_threads_env(...)` 绑定线程与内存。证据：`benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py`、`benchmarks/kernels/cpu/benchmark_cpu_attn.py`、`vllm/v1/attention/backends/cpu_attn.py`、`csrc/cpu/utils.cpp`。
- 它复用了真实 `CPUWorker.init_device()` 的 rank 选核规则：x86 下每个 physical core 只取 `1` 个 SMT 线程，按 `allowed_numa_nodes[local_rank]` 把 rank 映射到当前可见 NUMA node。证据：`vllm/v1/worker/cpu_binding.py`、`vllm/v1/worker/cpu_worker.py`。
- 它不加载真实模型权重，不复现真实 token 数据值，也不直接识别 `NPS`。它只能识别当前机器在运行时暴露出来的 `CPU/CORE/NODE` 拓扑。

## 脚本怎么用

最小 dry-run：

```bash
cd /home/zjj/vllm

cleanup_vllm_all() {
  pkill -9 -f 'benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py' 2>/dev/null || true
  pkill -9 -f 'from multiprocessing.resource_tracker import main' 2>/dev/null || true
  pkill -9 -f 'from multiprocessing.spawn import spawn_main' 2>/dev/null || true
  pkill -9 -f 'vllm' 2>/dev/null || true
  sleep 1
  pgrep -af 'benchmark_cpu_attn_mp.py|multiprocessing.resource_tracker|multiprocessing.spawn|vllm' || true
}

cleanup_vllm_all

python benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py \
  --tp-size 2 \
  --partition-mode global-fixed \
  --workload prefill-like \
  --batch-size 1 \
  --q-len 4096 \
  --kv-len 4096 \
  --num-query-heads 32 \
  --num-kv-heads 16 \
  --head-size 128 \
  --dtype bfloat16 \
  --block-size 128 \
  --enable-kv-split \
  --warmup-iters 200 \
  --iters 2000

python benchmarks/kernels/cpu/benchmark_cpu_attn.py \
  --tp-size 2 \
  --partition-mode global-fixed \
  --workload decode-like \
  --batch-size 1 \
  --q-len 4096 \
  --kv-len 4096 \
  --num-query-heads 32 \
  --num-kv-heads 16 \
  --head-size 128 \
  --dtype bfloat16 \
  --block-size 128 \
  --enable-kv-split \
  --warmup-iters 200 \
  --iters 20000
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
- `--attn-locality-group-span`：现在会真实传进 `acc-local-l3` scheduler。
  - `1` 表示每个 `kv_head` 只覆盖一个 subgroup。
  - `>1` 表示一个 `kv_head` 覆盖多个连续 subgroup。
  - 截至 `2026-04-09`，`group_span=4` 在 `NPS1_TP2/prefill-like/global-fixed/batch=16/q=kv=128/512` 上虽然把 `attention_task_num` 恢复到接近 legacy，但会让多个 subgroup 重复执行同一批 legacy slot，因此当前只适合诊断，不是建议的优化配置。
- `--output-json`：写出当前 run 的 rank 绑核列表、head shard 口径和每 rank 计时结果。
- 结果目录建议显式带上解析后的 kernel shape，例如 `batch_16/q1_kv1024`；脚本 summary 里也会返回同名的 `result_shape_dirname` 字段，便于外部 wrapper 复用。
- `--iters` 与 `--min-runtime-s`：二选一。
  - `--iters` 适合 dry-run 或短测，会保留逐轮 `times_ms`。
  - `--min-runtime-s` 适合 `AMDuProfPcm`，表示 warmup 之后至少继续跑这么多秒；该路径只保留聚合统计和实际运行轮数，不保留逐轮 `times_ms`，避免 decode 场景产生超大 JSON。

## Qwen3-30B-A3B 的 head 口径
- `/models/Qwen3-30B-A3B/config.json` 里，`num_attention_heads=32`，`num_key_value_heads=4`，`head_dim=128`。
- 因此，对 `Qwen3-30B-A3B`：
  - `global-fixed` 口径应传 `--num-query-heads 32 --num-kv-heads 4`
  - 若要在 `per-rank-fixed` 里保持“TP=2 的本地 shape”不变，则传 `--num-query-heads 16 --num-kv-heads 2`

## 实验注意事项
1. 当前推荐关系：`benchmark_min_runtime_s > pcm_duration_s`，起步可取 `benchmark_min_runtime_s = pcm_duration_s + 10`。
2. `AMDuProfPcm` 默认采样间隔是 `1000ms`，正式实验应显式传 `-I <ms>`。第一轮建议从 `-I 100` 开始。

## 可直接复制执行的命令

### 一次性环境变量

```bash
cd /home/zjj/vllm

conda activate vllm-cpu
PYTHON_BIN=${PYTHON_BIN:-python}
VLLM_ENV=${CONDA_PREFIX:-/home/zjj/.conda/envs/vllm-cpu}
PCM_BIN=/opt/AMDuProf_5.2-606/bin/AMDuProfPcm
RESULT_ROOT=/home/zjj/vllm/test_results/P4_AttnOnly_L3Residency/Qwen3-30B-A3B
SCRIPT=/home/zjj/vllm/benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py

mkdir -p "${RESULT_ROOT}"

cleanup_vllm_all() {
  pkill -9 -f 'benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py' 2>/dev/null || true
  pkill -9 -f 'from multiprocessing.resource_tracker import main' 2>/dev/null || true
  pkill -9 -f 'from multiprocessing.spawn import spawn_main' 2>/dev/null || true
  pkill -9 -f 'vllm' 2>/dev/null || true
  sleep 1
  pgrep -af 'benchmark_cpu_attn_mp.py|multiprocessing.resource_tracker|multiprocessing.spawn|vllm' || true
}
```

如果还没准备 `AMDuProfPcm`：

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh
```

如果要打开当前 `acc-local-l3` runtime 日志：

```bash
DEBUG_OUT_DIR="${RESULT_ROOT}/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1/q32768_kv32768/acc-local-l3"
mkdir -p "${DEBUG_OUT_DIR}"

cleanup_vllm_all

conda run -n vllm-cpu env VLLM_CPU_ATTN_DEBUG=1 python benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py \
  --tp-size 2 \
  --partition-mode global-fixed \
  --workload prefill-like \
  --batch-size 1 \
  --q-len 32768 \
  --kv-len 32768 \
  --num-query-heads 32 \
  --num-kv-heads 16 \
  --head-size 128 \
  --block-size 128 \
  --dtype bfloat16 \
  --warmup-iters 0 \
  --iters 1 \
  --attn-locality-mode acc-local-l3 \
  --attn-locality-group-span 1 2>&1 | tee "${DEBUG_OUT_DIR}/debug.log"

DEBUG_OUT_DIR="${RESULT_ROOT}/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1/q32768_kv32768/balanced"
mkdir -p "${DEBUG_OUT_DIR}"

cleanup_vllm_all

conda run -n vllm-cpu env VLLM_CPU_ATTN_DEBUG=1 python benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py \
  --tp-size 2 \
  --partition-mode global-fixed \
  --workload prefill-like \
  --batch-size 1 \
  --q-len 32768 \
  --kv-len 32768 \
  --num-query-heads 32 \
  --num-kv-heads 16 \
  --head-size 128 \
  --block-size 128 \
  --dtype bfloat16 \
  --warmup-iters 0 \
  --iters 1 \
  --attn-locality-mode balanced \
  --attn-locality-group-span 1 2>&1 | tee "${DEBUG_OUT_DIR}/debug.log"

`VLLM_CPU_ATTN_DEBUG=1` 会把原来的 runtime summary 与 trace 合并到同一个 `debug.log`。默认只打印 rank 0；若要保留所有 rank，同时加 `VLLM_CPU_ATTN_DEBUG_ALL_RANKS=1`。

如果要打开当前 profiling 汇总：

`VLLM_CPU_ATTN_PROFILE=1` 会打印 scheduler metadata 构建时间与 runtime 汇总时间；`acc-local-l3` 额外打印 `legacy_scheduler_metadata_ns` 与 `locality_metadata_build_ns`。

`profile.json`、`summary.csv` 中各 profile 行的含义见 [CPU attention profiling指标介绍](../术语与背景/CPU_attention_profiling指标介绍.md)。

```bash
INPUT_LENS=(65536 32768 16384 8192 4096 2048 1024 512 256)
LOCALITY_MODES=(acc-local-l3 balanced)
BATCH_SIZE=1

for input_len in "${INPUT_LENS[@]}"; do
  for locality_mode in "${LOCALITY_MODES[@]}"; do
    PROFILE_OUT_DIR="${RESULT_ROOT}/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_${BATCH_SIZE}/q${input_len}_kv${input_len}/${locality_mode}"
    mkdir -p "${PROFILE_OUT_DIR}"

    cleanup_vllm_all

    conda run -n vllm-cpu env VLLM_CPU_ATTN_PROFILE=1 python benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py \
      --tp-size 2 \
      --partition-mode global-fixed \
      --workload prefill-like \
      --batch-size "${BATCH_SIZE}" \
      --q-len "${input_len}" \
      --kv-len "${input_len}" \
      --num-query-heads 32 \
      --num-kv-heads 16 \
      --head-size 128 \
      --block-size 128 \
      --dtype bfloat16 \
      --enable-kv-split \
      --warmup-iters 10 \
      --iters 200 \
      --attn-locality-mode "${locality_mode}" \
      --attn-locality-group-span 1 \
      --output-json "${PROFILE_OUT_DIR}/profile.json"
  done
done
```

### 定义 dry-run / PCM helper

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
  local locality_group_span="${11:-1}"
  local locality_tag="${locality_mode}"
  if [ "${locality_group_span}" -ne 1 ]; then
    locality_tag="${locality_mode}_span${locality_group_span}"
  fi
  local out_dir="${RESULT_ROOT}/qhead_${nq}_kvhead_${nkv}/${exp_tag}/${workload}/${mode}/batch_${batch_size}/${locality_tag}/${q_len}_kv${kv_len}}"

  mkdir -p "${out_dir}"
  cleanup_vllm_all

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
      --iters 150 \
      --attn-locality-mode "${locality_mode}" \
      --attn-locality-group-span "${locality_group_span}" \
      --output-json "${out_dir}/dry_run_summary.json" \
    | tee "${out_dir}/dry_run.log"
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
  local locality_group_span="${11:-1}"
  local start_delay_ms="${12}"
  local pcm_duration_s="${13}"
  local benchmark_min_runtime_s="${14}"
  local interval_ms="${15}"
  local locality_tag="${locality_mode}"
  if [ "${locality_group_span}" -ne 1 ]; then
    locality_tag="${locality_mode}_span${locality_group_span}"
  fi
  local out_dir="${RESULT_ROOT}/qhead_${nq}_kvhead_${nkv}/${exp_tag}/${workload}/${mode}/batch_${batch_size}/q${q_len}_kv${kv_len}/${locality_tag}"

  mkdir -p "${out_dir}"

  if [ "${benchmark_min_runtime_s}" -le "${pcm_duration_s}" ]; then
    echo "benchmark_min_runtime_s must be > pcm_duration_s" >&2
    return 1
  fi

  cleanup_vllm_all

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
      --warmup-iters 5 \
      --min-runtime-s "${benchmark_min_runtime_s}" \
      --attn-locality-mode "${locality_mode}" \
      --attn-locality-group-span "${locality_group_span}" \
      --output-json "${out_dir}/summary.json"
}
```

这里的 `pcm_duration_s` 只控制 profiler 采样时长；`benchmark_min_runtime_s` 要显式设得更长，避免 profiler 在程序收尾阶段采样。

### 补测系统 idle baseline

系统空闲基线只用于估计全局 `PCM` 计数的背景底噪，不用于从 workload 指标中直接相减得到“净 attention kernel 指标”。先用标准 `PCM` 指标，不使用自定义 XML：

```bash
IDLE_OUT_DIR="${RESULT_ROOT}/idle_baseline/NPS1_TP2/pcm_standard_metrics"
mkdir -p "${IDLE_OUT_DIR}"

"${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a -s -d 40 -I 200 \
  -O "${IDLE_OUT_DIR}/pcm_l3_dc_idle" -- \
  sleep 50
```

- 执行前保持系统尽量空闲，不启动 benchmark。
- 重点查看 `pcm_l3_dc_idle/report-cumulative.csv` 和 `pcm_l3_dc_idle/report-timeseries.csv`。
- 对比 workload 时优先看 raw `L3 Access`、raw `RetdInst`、`L3 Access / second` 的量级；`L3 Access (pti)` 在 idle 场景可能被很低的退休指令数放大。

### 补测 `L3 source latency` raw

- `wiki/术语与背景/访存延迟测量.md` 里的这份自定义 `PCM XML` 已足够回答当前问题：
  - raw `L3 Access/L3 Miss`、`L3 Access/L3 Miss (pti)`、`L3 Miss / second` 与 `L3 Miss %`
  - `Local Memory or I/O`、`another CCX in same node`、`Remote Memory or I/O`、`another CCX in remote node` 各自的 raw sampled latency 与 raw sampled requests
  - 各来源路径的平均 `L3 miss` 延迟
  - 各来源路径占总 sampled `L3 miss` requests 的比例；判断来源占比时看 `Request Share (%)`，不使用 latency share 代替请求占比
- 指标顺序把 `Local Memory or I/O` 与 `another CCX in same node` 相邻放置，便于直接判断 `L3 miss` 后有多少 sampled requests 落到本地内存、多少落到同节点其他 `CCX`。
- 这里直接把 `core ipc` 事件一并写进 XML，因此同一轮 `-i` 结果里也会带 `IPC/CPI`。
- 但它仍不包含标准 `dc` 指标；若还要 `Demand DC Fills` 这些列，仍需单独运行上面的 `run_attn_only_pcm_locality`。
- `AMDuProfPcm` 的 `-i` 和 `-m` 不能同时使用，因此自定义 `PCM` 只能作为补充批次。

```bash
run_attn_only_pcm_l3_source_latency() {
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
  local locality_group_span="${11:-1}"
  local start_delay_ms="${12}"
  local pcm_duration_s="${13}"
  local benchmark_min_runtime_s="${14}"
  local interval_ms="${15}"
  local locality_tag="${locality_mode}"
  if [ "${locality_group_span}" -ne 1 ]; then
    locality_tag="${locality_mode}_span${locality_group_span}"
  fi
  local out_dir="${RESULT_ROOT}/qhead_${nq}_kvhead_${nkv}/${exp_tag}/${workload}/${mode}/batch_${batch_size}/q${q_len}_kv${kv_len}/${locality_tag}"
  local pcm_xml=/home/zjj/vllm/test_results/amduprof_pcm_l3_source_latency.conf

  mkdir -p "${out_dir}"

  if [ "${benchmark_min_runtime_s}" -le "${pcm_duration_s}" ]; then
    echo "benchmark_min_runtime_s must be > pcm_duration_s" >&2
    return 1
  fi

  cleanup_vllm_all

  "${PCM_BIN}" profile \
    -i "${pcm_xml}" \
    -a -s --start-delay "${start_delay_ms}" -d "${pcm_duration_s}" -I "${interval_ms}" \
    -O "${out_dir}/pcm_l3_source_latency_raw" -- \
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
      --warmup-iters 5 \
      --min-runtime-s "${benchmark_min_runtime_s}" \
      --attn-locality-mode "${locality_mode}" \
      --attn-locality-group-span "${locality_group_span}"
}
```


```bash
for locality_mode in acc-local-l3 balanced; do
  for q_len in 65536 32768 16384 8192 4096 2048 1024 512 256; do
    run_attn_only_pcm_l3_source_latency \
      NPS1_TP2 2 prefill-like global-fixed 1 \
      "${q_len}" "${q_len}" 32 16 \
      "${locality_mode}" 1 \
      30000 110 130 200
  done
done

python tools/p3_attn_only/pcm_l3_source_latency_compare.py \
  --result-root "${RESULT_ROOT}/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1"
```

汇总脚本依赖同一 case 目录下已有 `dry_run_summary.json`；若目录中没有该文件，先执行对应的 `dry_run_attn_only`。

- 重点查看：
  - `pcm_l3_source_latency_raw/report-core.csv`
  - `pcm_l3_source_latency_raw/report-cumulative.csv`
  - `pcm_l3_source_latency_raw/report-timeseries.csv`
  - `summary_l3_source_latency.csv`
  - `detail_l3_source_latency.csv`

### 测当前 `acc-local-l3` 实现

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
  for q_len in 65536 32768 16384 8192 4096 2048 1024 512 256; do
    run_attn_only_pcm_locality \
      NPS1_TP2 2 prefill-like global-fixed 1 \
      "${q_len}" "${q_len}" 32 16 \
      "${locality_mode}" 1 \
      30000 110 150 200
  done
done

run_attn_only_pcm_locality NPS1_TP2 2 prefill-like global-fixed 1 512 512 32 4 balanced 4 20000 80 120 200
run_attn_only_pcm_locality NPS1_TP2 2 prefill-like global-fixed 1 512 512 32 4 acc-local-l3 4 20000 80 120 200

```
如果只想先做不带 `AMDuProfPcm` 的短测，也应保持同一个 case 下成对对照：

```bash
dry_run_attn_only NPS1_TP2 2 prefill-like global-fixed 1 65536 65536 32 16 acc-local-l3 1
dry_run_attn_only NPS1_TP2 2 prefill-like global-fixed 1 65536 65536 32 16 balanced 1

for locality_mode in balanced acc-local-l3; do
  for q_len in 65536 32768 16384 8192 4096 2048 1024; do
    dry_run_attn_only \
      NPS1_TP2 2 prefill-like global-fixed 1 \
      "${q_len}" "${q_len}" 32 16 \
      "${locality_mode}" 1
  done
done
```

如果要复现 `2026-04-09` 的 `group_span=4` 诊断批次，可把最后一个参数改成 `4`；结果目录会自动写成 `acc-local-l3_span4`，避免覆盖 `group_span=1`。


## 结果解释边界
- 如果 `global-fixed` 改善，而 `per-rank-fixed` 不改善，不能直接写成 locality 变好；更可能是每个 rank 的工作量变小。
- 如果两种口径都同方向改善，而且 `L3 Miss %`、`Ave L3 Miss Latency`、`Derived another CCX in same node L3 Miss Request Share (%)`、`Derived Local Memory L3 Miss Request Share (%)`、`Remote DRAM Reads %` 也同步变化，才可以写成“attention-only 场景暴露出可观测 locality 优化空间”。
- 即便如此，也不能直接把 attention-only 的结果写成端到端收益结论；端到端场景还包含 `o_proj`、`TP all_reduce`、`MoE/MLP` 等其他路径。

## 相关页面
- [2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md](../实验结果解读/2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md)
- [2026-04-05_Qwen3-30B-A3B_P3_AttnOnly_acc-local-l3对照观察.md](../实验结果解读/2026-04-05_Qwen3-30B-A3B_P3_AttnOnly_acc-local-l3对照观察.md)
