# Qwen3-30B-A3B attention-only L3 驻留验证实验步骤

## 目的
判断 chiplet-aware locality 是否值得继续投入。当前数据已经表明 `acc-local-l3` 会改变 `L3` 行为，但没有稳定转成 runtime 收益；跨 `CCD` demand fill 也一直很小，因此后续重点转为控制变量实验，判断单个 `CCD` 的 `L3` 容量是否真是性能杠杆。

## 当前判断
`acc-local-l3` 已经降低长序列的 `L3 Miss / attention task`，但 runtime 仍然更慢；跨 `CCD` demand fill 份额一直很小，主体 demand fill 仍是本地 `L2`。现阶段最需要补的是 `CAT`，不是继续扩大 `resctrl occupancy` 或长度 sweep。

## 需要继续的实验
`CAT` 是必做项。最小点位是 `q8192` 和 `q16384`，对比 `balanced` 与 `acc-local-l3`，看在限制本地 `L3` 容量后，runtime、`IPC` 和 `L3 Miss / task` 是否明显变化。`IBS L3-miss` 只在需要把 miss 来源和函数调用链说清时做，优先 `q8192` 单点。`resctrl llc_occupancy` 只做单点旁证，不做 sweep；如果只想先推进主结论，可以先不跑它。`span=8` 诊断已够用，不再扩展。

## CAT 容量限制
### 目标
把本地 `L3` 可用容量压小，观察 `acc-local-l3` 是否比 `balanced` 更敏感。当前机器 `cbm_mask=ffff`，可以先用极端点 `0001`；若需要更温和的容量点，再补 `00ff` 和 `0fff`。

### 操作
当前 `/sys/fs/resctrl` 是只读挂载，先临时 remount 为可写。每个 case 用同一个 resctrl group 顺序跑两次：先跑不带 `PCM` 的 dry-run，保留 `dry_run_summary.json`；若确实出现容量敏感性，再在同一路径下补 `pcm_l3_dc`。`benchmark_cpu_attn_mp.py` 用 `spawn` 拉起子进程，因此不能只把父进程写进 `tasks`；需要在启动后循环把整棵进程树都写进同一个 resctrl group。benchmark 直接用当前激活的 `vllm-cpu` Python 跑，只有 `resctrl` 的 `schemata/tasks` 写入和挂载切换需要 `sudo`。

```bash
conda activate vllm-cpu
export RESULT_ROOT=/home/zjj/vllm/test_results/P4_AttnOnly_L3Residency/Qwen3-30B-A3B/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1
export PYTHON_BIN=${PYTHON_BIN:-python}
export SCRIPT=/home/zjj/vllm/benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py
export PCM_BIN=/opt/AMDuProf_5.2-606/bin/AMDuProfPcm

sudo -v
sudo mount -o remount,rw /sys/fs/resctrl

cleanup_vllm_all() {
  pkill -9 -f 'benchmarks/kernels/cpu/benchmark_cpu_attn_mp.py' 2>/dev/null || true
  pkill -9 -f 'from multiprocessing.resource_tracker import main' 2>/dev/null || true
  pkill -9 -f 'from multiprocessing.spawn import spawn_main' 2>/dev/null || true
  pkill -9 -f 'vllm' 2>/dev/null || true
  sleep 1
}

cat_group_path() {
  local group="$1"
  printf '/sys/fs/resctrl/%s' "${group}"
}

make_cat_schemata() {
  local mask="$1"
  local smba=""
  local mb=""
  local l3=""
  for dom in $(seq 0 15); do
    smba="${smba}${dom}=4096;"
    mb="${mb}${dom}=4096;"
    l3="${l3}${dom}=${mask};"
  done
  printf 'SMBA:%s\nMB:%s\nL3:%s\n' "${smba%;}" "${mb%;}" "${l3%;}"
}

prepare_cat_group() {
  local group="$1"
  local mask="$2"
  local group_path
  group_path="$(cat_group_path "${group}")"
  sudo mkdir -p "${group_path}"
  make_cat_schemata "${mask}" | sudo tee "${group_path}/schemata" >/dev/null
}

collect_descendant_pids() {
  local root_pid="$1"
  local queue=("${root_pid}")
  local seen=""

  while ((${#queue[@]} > 0)); do
    local pid="${queue[0]}"
    queue=("${queue[@]:1}")
    [[ -d "/proc/${pid}" ]] || continue
    case " ${seen} " in
      *" ${pid} "*) continue ;;
    esac
    seen="${seen} ${pid}"
    echo "${pid}"
    while read -r child; do
      [[ -n "${child}" ]] && queue+=("${child}")
    done < <(pgrep -P "${pid}" || true)
  done
}

attach_cat_process_tree() {
  local group="$1"
  local root_pid="$2"
  local rounds="${3:-20}"
  local interval_s="${4:-0.2}"
  local group_path
  group_path="$(cat_group_path "${group}")"

  for _ in $(seq 1 "${rounds}"); do
    mapfile -t pid_list < <(collect_descendant_pids "${root_pid}" | sort -u)
    for pid in "${pid_list[@]}"; do
      [[ -n "${pid}" && -d "/proc/${pid}" ]] || continue
      echo "${pid}" | sudo tee "${group_path}/tasks" >/dev/null
    done
    [[ -d "/proc/${root_pid}" ]] || break
    sleep "${interval_s}"
  done
}

show_cat_group() {
  local group="$1"
  local out_file="$2"
  local group_path
  group_path="$(cat_group_path "${group}")"
  {
    echo "== ${group_path}/schemata =="
    sudo cat "${group_path}/schemata"
    echo "== ${group_path}/tasks =="
    sudo cat "${group_path}/tasks" || true
  } > "${out_file}"
}

run_cat_case_no_pcm() {
  local q_len="$1"
  local locality_mode="$2"
  local mask="${3:-0001}"
  local group="cat_${mask}"
  local out_dir="${RESULT_ROOT}/cat/q${q_len}_kv${q_len}/${locality_mode}/${mask}"

  cleanup_vllm_all
  mkdir -p "${out_dir}"
  prepare_cat_group "${group}" "${mask}"

  env VLLM_CPU_ATTN_PROFILE=1 \
      LD_PRELOAD="${LD_PRELOAD:-}" \
      LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --tp-size 2 \
      --partition-mode global-fixed \
      --workload prefill-like \
      --batch-size 1 \
      --q-len "${q_len}" \
      --kv-len "${q_len}" \
      --num-query-heads 32 \
      --num-kv-heads 16 \
      --head-size 128 \
      --dtype bfloat16 \
      --block-size 128 \
      --enable-kv-split \
      --omp-threads-bind auto \
      --warmup-iters 10 \
      --iters 150 \
      --attn-locality-mode "${locality_mode}" \
      --attn-locality-group-span 1 \
      --output-json "${out_dir}/dry_run_summary.json" &
  local bench_pid=$!
  attach_cat_process_tree "${group}" "${bench_pid}" 100 0.2
  show_cat_group "${group}" "${out_dir}/cat_group_state.txt"
  wait "${bench_pid}"
  cleanup_vllm_all
}

run_cat_case() {
  local q_len="$1"
  local locality_mode="$2"
  local mask="${3:-0001}"
  local group="cat_${mask}"
  local out_dir="${RESULT_ROOT}/cat/q${q_len}_kv${q_len}/${locality_mode}/${mask}"

  cleanup_vllm_all
  mkdir -p "${out_dir}"
  prepare_cat_group "${group}" "${mask}"

  env VLLM_CPU_ATTN_PROFILE=1 \
      LD_PRELOAD="${LD_PRELOAD:-}" \
      LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${PCM_BIN}" profile -m ipc,l2,l3,dc \
      -a -s --start-delay 30000 -d 110 -I 200 \
      -O "${out_dir}/pcm_l3_dc" -- \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --tp-size 2 \
      --partition-mode global-fixed \
      --workload prefill-like \
      --batch-size 1 \
      --q-len "${q_len}" \
      --kv-len "${q_len}" \
      --num-query-heads 32 \
      --num-kv-heads 16 \
      --head-size 128 \
      --dtype bfloat16 \
      --block-size 128 \
      --enable-kv-split \
      --omp-threads-bind auto \
      --warmup-iters 5 \
      --min-runtime-s 150 \
      --attn-locality-mode "${locality_mode}" \
      --attn-locality-group-span 1 &
  local bench_pid=$!
  attach_cat_process_tree "${group}" "${bench_pid}" 100 0.2
  show_cat_group "${group}" "${out_dir}/cat_group_state.txt"
  wait "${bench_pid}"
  cleanup_vllm_all
}

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
  local locality_tag="span${locality_group_span}"

  local out_dir="${RESULT_ROOT}/${locality_tag}/q${q_len}_kv${kv_len}/${locality_mode}"

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
      --output-json "${out_dir}/dry_run_summary.json" 
}
```

dry run:
```bash
for locality_mode in balanced acc-local-l3; do
  for q_len in 1024 65536 32768 16384 8192 4096 2048; do
    dry_run_attn_only \
      NPS1_TP2 2 prefill-like global-fixed 1 \
      "${q_len}" "${q_len}" 32 16 \
      "${locality_mode}" 1
  done
done
```

先跑不带 `PCM` 的主点：

```bash
sudo rmdir /sys/fs/resctrl/cat_*

for q_len in 1024 4096 8192 16384 32768 65536; do
  for locality_mode in balanced acc-local-l3; do
    run_cat_case_no_pcm "${q_len}" "${locality_mode}" ffff
  done
done
```

若 runtime 已经显示出容量敏感性，再补带 `PCM` 的版本：

```bash
for q_len in 8192 16384; do
  for locality_mode in balanced acc-local-l3; do
    run_cat_case "${q_len}" "${locality_mode}" 0001
  done
done
```

若需要更细容量点，再补：

```bash
for q_len in 8192 16384; do
  for locality_mode in balanced acc-local-l3; do
    run_cat_case "${q_len}" "${locality_mode}" 0fff
  done
done
```

## resctrl llc_occupancy 单点旁证
### 目标
观察 benchmark 进程组在各 `L3` monitoring domain 中的 `llc_occupancy`。该指标只表示采样时刻属于该 resctrl monitoring group 的 `LLC` 驻留字节数，不能直接等同于 `K/V` 工作集大小，也不能单独证明这些驻留数据决定 runtime。当前主判据仍是 `CAT` 容量限制；`llc_occupancy` 只用于补充展示驻留形态。

### 操作
只建议跑 `q8192` 或 `q16384` 单点，对比 `balanced` 与 `acc-local-l3`。`ffff` 用于观察不限制容量时的自然驻留量；`0001` 用于观察极端 CAT 容量限制下的驻留量。运行口径使用固定正式迭代数，不使用 `--min-runtime-s`，保证两种 locality mode 执行相同 attention 工作量。

输出目录按 mask 区分。`ffff` 结果写入 `${RESULT_ROOT}/span1/occupancy/q${q_len}_kv${q_len}/${locality_mode}/ffff/span1`；`0001` 结果写入 `${RESULT_ROOT}/span1/occupancy/q${q_len}_kv${q_len}/${locality_mode}/0001/span1`。

前置条件：

- `/sys/fs/resctrl` 已挂载，并且可以临时切到可写状态
- 机器支持 resctrl monitoring，创建 `mon_groups` 后能看到 `mon_data/mon_L3_*`
- 只把 benchmark 父进程和 `spawn` 出来的子进程写入同一个 control group 与 monitoring group

```bash
prepare_occ_group() {
  local group="$1"
  local mon_group="${2:-attn}"
  local mask="${3:-ffff}"
  local group_path
  group_path="$(cat_group_path "${group}")"

  sudo mkdir -p "${group_path}"
  make_cat_schemata "${mask}" | sudo tee "${group_path}/schemata" >/dev/null
  sudo mkdir -p "${group_path}/mon_groups/${mon_group}"
}

cleanup_occ_group() {
  local group="$1"
  local mon_group="${2:-attn}"
  local group_path
  group_path="$(cat_group_path "${group}")"

  sudo rmdir "${group_path}/mon_groups/${mon_group}" 2>/dev/null || true
  sudo rmdir "${group_path}" 2>/dev/null || true
}

attach_occ_process_tree() {
  local group="$1"
  local mon_group="$2"
  local root_pid="$3"
  local rounds="${4:-100}"
  local interval_s="${5:-0.2}"
  local group_path
  local mon_path
  group_path="$(cat_group_path "${group}")"
  mon_path="${group_path}/mon_groups/${mon_group}"

  for _ in $(seq 1 "${rounds}"); do
    mapfile -t pid_list < <(collect_descendant_pids "${root_pid}" | sort -u)
    for pid in "${pid_list[@]}"; do
      [[ -n "${pid}" && -d "/proc/${pid}" ]] || continue
      echo "${pid}" | sudo tee "${group_path}/tasks" >/dev/null
      echo "${pid}" | sudo tee "${mon_path}/tasks" >/dev/null
    done
    [[ -d "/proc/${root_pid}" ]] || break
    sleep "${interval_s}"
  done
}

sample_llc_occupancy_until_exit() {
  local group="$1"
  local mon_group="$2"
  local root_pid="$3"
  local out_csv="$4"
  local interval_s="${5:-0.2}"
  local mon_path
  mon_path="$(cat_group_path "${group}")/mon_groups/${mon_group}"

  printf 'ts_ms,domain,llc_occupancy_bytes\n' > "${out_csv}"
  while [[ -d "/proc/${root_pid}" ]]; do
    local ts_ms
    ts_ms="$(date +%s%3N)"
    for f in "${mon_path}"/mon_data/mon_L3_*/llc_occupancy; do
      [[ -e "${f}" ]] || continue
      printf '%s,%s,%s\n' \
        "${ts_ms}" \
        "$(basename "$(dirname "${f}")")" \
        "$(sudo cat "${f}")" >> "${out_csv}"
    done
    sleep "${interval_s}"
  done
}

show_occ_group() {
  local group="$1"
  local mon_group="$2"
  local out_file="$3"
  local group_path
  group_path="$(cat_group_path "${group}")"
  {
    echo "== ${group_path}/schemata =="
    sudo cat "${group_path}/schemata"
    echo "== ${group_path}/tasks =="
    sudo cat "${group_path}/tasks" || true
    echo "== ${group_path}/mon_groups/${mon_group}/tasks =="
    sudo cat "${group_path}/mon_groups/${mon_group}/tasks" || true
    echo "== ${group_path}/mon_groups/${mon_group}/mon_data =="
    sudo find "${group_path}/mon_groups/${mon_group}/mon_data" \
      -name llc_occupancy -print -exec cat {} \; 2>/dev/null || true
  } > "${out_file}"
}

run_llc_occupancy_case() {
  local q_len="$1"
  local locality_mode="$2"
  local mask="${3:-ffff}"
  local group="occ_probe_${mask}"
  local mon_group="attn"
  local out_dir

  out_dir="${RESULT_ROOT}/span1/occupancy/q${q_len}_kv${q_len}/${locality_mode}/${mask}/span1"

  cleanup_vllm_all
  cleanup_occ_group "${group}" "${mon_group}"
  mkdir -p "${out_dir}"
  prepare_occ_group "${group}" "${mon_group}" "${mask}"

  env VLLM_CPU_ATTN_PROFILE=1 \
      LD_PRELOAD="${LD_PRELOAD:-}" \
      LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --tp-size 2 \
      --partition-mode global-fixed \
      --workload prefill-like \
      --batch-size 1 \
      --q-len "${q_len}" \
      --kv-len "${q_len}" \
      --num-query-heads 32 \
      --num-kv-heads 16 \
      --head-size 128 \
      --dtype bfloat16 \
      --block-size 128 \
      --enable-kv-split \
      --omp-threads-bind auto \
      --warmup-iters 10 \
      --iters 400 \
      --attn-locality-mode "${locality_mode}" \
      --attn-locality-group-span 1 \
      --output-json "${out_dir}/occupancy_summary.json" &
  local bench_pid=$!

  attach_occ_process_tree "${group}" "${mon_group}" "${bench_pid}" 150 0.2 &
  local attach_pid=$!
  sample_llc_occupancy_until_exit \
    "${group}" "${mon_group}" "${bench_pid}" \
    "${out_dir}/llc_occupancy.csv" 0.2 &
  local sample_pid=$!

  wait "${bench_pid}"
  wait "${attach_pid}" 2>/dev/null || true
  wait "${sample_pid}" 2>/dev/null || true
  show_occ_group "${group}" "${mon_group}" "${out_dir}/resctrl_occ_state.txt"
  cleanup_vllm_all
  cleanup_occ_group "${group}" "${mon_group}"
}
```

单点采样命令：

```bash
for q_len in 1024 4096; do
  for locality_mode in balanced acc-local-l3; do
    run_llc_occupancy_case "${q_len}" "${locality_mode}" 0001
  done
done
```

补跑 `0001`，不覆盖已有 `ffff`：

```bash
for q_len in 1024 4096 8192 16384 32768 65536; do
  for locality_mode in balanced acc-local-l3; do
    run_llc_occupancy_case "${q_len}" "${locality_mode}" 0001
  done
done
```

若需要更长工作集，可把 `q_len` 改为 `16384`。结果文件为：

- `occupancy_summary.json`：benchmark runtime 与 rank 结果
- `llc_occupancy.csv`：各 `mon_L3_*` domain 的 `llc_occupancy_bytes` 时间序列
- `resctrl_occ_state.txt`：采样结束时的 resctrl group 状态

跑完后恢复只读：

```bash
sudo mount -o remount,ro /sys/fs/resctrl
```

可选汇总：

```bash
python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

root = Path("/home/zjj/vllm/test_results/P4_AttnOnly_L3Residency/Qwen3-30B-A3B/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1/span1/occupancy")
for csv_path in sorted(root.glob("q*_kv*/*/**/llc_occupancy.csv")):
    values = defaultdict(list)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            values[row["domain"]].append(int(row["llc_occupancy_bytes"]))
    print(f"== {csv_path.relative_to(root)} ==")
    for domain, xs in sorted(values.items()):
        mean_mib = sum(xs) / len(xs) / 1024 / 1024
        peak_mib = max(xs) / 1024 / 1024
        print(f"{domain}: mean={mean_mib:.2f} MiB peak={peak_mib:.2f} MiB samples={len(xs)}")
PY
```
