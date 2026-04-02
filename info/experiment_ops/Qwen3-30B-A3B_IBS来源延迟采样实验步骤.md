# Qwen3-30B-A3B：`AMDuProfCLI IBS` 来源延迟采样实验步骤

> 更新时间：2026-04-02 17:19:41 +0800  
> 适用机器：`2 x AMD EPYC 9745 128-Core Processor`  
> 目标：对 `L3 miss` 样本估算  
> `another CCX cache` 平均访问延迟，以及 `local memory` 平均访问延迟。  
> 主口径：`processor cycles`；若后续确实需要 `ns`，再结合同时间窗频率换算。

## 0. 这份文档回答什么问题

- `AMDuProfPcm` 能直接给：
  - `Ave L3 Miss Latency (ns)`
  - `L3 Miss Latency From Local Memory or I/O (%)`
  - `L3 Miss Latency From another CCX in same node (%)`
- 但 `AMDuProfPcm` 不能直接给：
  - “`another CCX cache` 平均是 `X ns`”
  - “`local memory` 平均是 `Y ns`”
- 要得到“每条来源自己的平均延迟”，当前最直接的工具是 `AMDuProfCLI` 的 `IBS OP`。

证据：

- `AMDuProfCLI collect` 支持 `event=ibs-op`、`ibsop-l3miss=1` 和 `ibsop-ldlat=<LATENCY>`。证据：`/opt/AMDuProf_5.2-606/bin/Help/text/AMDuProf_Collect_Help.txt:58-77`
- `AMDuProfCLI report --view ibs_op_ld` 提供：
  - `IBS_LD_PEER_CACHE_HIT_RATE_%`
  - `IBS_LD_LOCAL_DRAM_HIT_RATE_%`
  证据：`/opt/AMDuProf_5.2-606/bin/Data/Views/0x1a_0x1/ibs_op_ld.json:1-69`
- `AMDuProfCLI report --view ibs_op_ld_lat` 提供：
  - `%IBS_LD_PEER_CACHE_HIT_LAT`
  - `%IBS_LD_LOCAL_DRAM_HIT_LAT`
  - `IBS_LD_L1_DC_MISS_LAT_AVE`
  证据：`/opt/AMDuProf_5.2-606/bin/Data/Views/0x1a_0x1/ibs_op_ld_lat.json:1-69`
- 本机事件定义明确写了：
  - `IBS_LD_PEER_CACHE_HIT_LAT`：同一 NUMA node、不同 CCX 的 `L2/L3` 返回的 load，总延迟，单位是 `processor cycles`
  - `IBS_LD_LOCAL_DRAM_HIT_LAT`：同一 NUMA node、本地 DRAM 返回的 load，总延迟，单位是 `processor cycles`
  证据：`/opt/AMDuProf_5.2-606/bin/Data/Events/0x1a_0x1.xml:1087-1097`

## 1. 计算口径

### 1.1 推荐口径：先把采样限制到 `L3 miss`

推荐采样时显式加：

```bash
-e event=ibs-op,ibsop-l3miss=1,call-graph
```

这样保留下来的 `IBS OP` 样本只包含“`L3 miss` 的 load/store”。证据：`/opt/AMDuProf_5.2-606/bin/Help/text/AMDuProf_Collect_Help.txt:63-67`

在这个前提下，可直接用 `report` 视图里的指标反推来源平均延迟：

```text
another_ccx_avg_cycles
  = IBS_LD_L1_DC_MISS_LAT_AVE
    * (%IBS_LD_PEER_CACHE_HIT_LAT / IBS_LD_PEER_CACHE_HIT_RATE_%)

local_memory_avg_cycles
  = IBS_LD_L1_DC_MISS_LAT_AVE
    * (%IBS_LD_LOCAL_DRAM_HIT_LAT / IBS_LD_LOCAL_DRAM_HIT_RATE_%)
```

原因：

- `IBS_LD_L1_DC_MISS_LAT_AVE` = 所有被保留 miss 样本的平均 miss latency
- `%IBS_LD_PEER_CACHE_HIT_LAT` = `another CCX cache` 这一路占总 miss latency 的比例
- `IBS_LD_PEER_CACHE_HIT_RATE_%` = `another CCX cache` 这一路占样本数的比例

因此：

```text
来源平均 latency = 总平均 latency × (该来源 latency 占比 / 该来源样本占比)
```

### 1.2 若能拿到原始事件总量，可直接用“总延迟 / 总样本数”

若后续用 `report --view all` 或 `translate --ascii event-dump/raw-dump` 拿到原始事件总量，则最直接的公式是：

```text
another_ccx_avg_cycles = IBS_LD_PEER_CACHE_HIT_LAT / IBS_LD_PEER_CACHE_HIT
local_memory_avg_cycles = IBS_LD_LOCAL_DRAM_HIT_LAT / IBS_LD_LOCAL_DRAM_HIT
```

这里：

- `IBS_LD_PEER_CACHE_HIT_LAT` 与 `IBS_LD_LOCAL_DRAM_HIT_LAT` 都是“总 latency cycles”
- `IBS_LD_PEER_CACHE_HIT` 与 `IBS_LD_LOCAL_DRAM_HIT` 都是对应来源的 load 样本数

## 2. 一次性准备

```bash
sudo sysctl -w kernel.nmi_watchdog=0

export CLI_BIN=/opt/AMDuProf_5.2-606/bin/AMDuProfCLI
export VLLM_BIN=/home/zjj/.conda/envs/vllm-cpu/bin/vllm
export MODEL=/models/Qwen3-30B-A3B
export BASE=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/CLI_res
export LD_LIBRARY_PATH=/home/zjj/.conda/envs/vllm-cpu/lib:${LD_LIBRARY_PATH}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CPU_KVCACHE_SPACE=32
```

## 3. 可直接复制的采样命令

### 3.1 定义 helper

```bash
run_ibs_source_latency() {
  local case_name="$1"
  local input_len="$2"
  local output_len="$3"
  local start_delay_sec="$4"
  local duration_sec="$5"
  local num_rounds="$6"
  local out_base="${BASE}/${case_name}/ibs_source_latency"

  mkdir -p "${out_base}"

  "${CLI_BIN}" collect \
    -e event=ibs-op,ibsop-l3miss=1,call-graph \
    --call-graph-mode fp \
    --start-delay "${start_delay_sec}" \
    --duration "${duration_sec}" \
    --output-dir "${out_base}" \
    env \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
      VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
      VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
      "${VLLM_BIN}" bench strict-batch \
        --model "${MODEL}" \
        --dtype float16 \
        --block-size 128 \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 200000 \
        --no-enable-chunked-prefill \
        --no-enable-prefix-caching \
        --load-format dummy \
        --batch-size 16 \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --n 1 \
        --num-iters-warmup 1 \
        --num-rounds "${num_rounds}" \
        --disable-detokenize \
        --output-json "${out_base}/strict_batch_result.json"
}
```

### 3.2 `decode_B16_I1_O1024`

```bash
run_ibs_source_latency \
  decode_B16_I1_O1024 \
  1 \
  1024 \
  300 \
  240 \
  20
```

### 3.3 `prefill_B16_I1024_O1`

```bash
run_ibs_source_latency \
  prefill_B16_I1024_O1 \
  1024 \
  1 \
  300 \
  240 \
  60
```

说明：

- 这里沿用当前项目里已验证过的 `strict-batch + TP=2` 口径。
- `ibsop-l3miss=1` 是关键；没有它，后面的平均延迟反推公式就不成立。
- 如果只是想先做小样本 smoke test，可把 `start_delay_sec/duration_sec/num_rounds` 降低，但不要直接把 smoke 数值写成正式结论。

## 4. 怎么看采样结果

### 4.1 先定位 session 目录

`collect --output-dir <dir>` 会在 `<dir>` 下新建一个 session 子目录。先取最新那个：

```bash
OUT_BASE="${BASE}/decode_B16_I1_O1024/ibs_source_latency"
SESSION=$(find "${OUT_BASE}" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)
echo "${SESSION}"
```

### 4.2 看来源占比

```bash
${CLI_BIN} report \
  -i "${SESSION}" \
  --view ibs_op_ld \
  --show-event-count \
  --show-percentage \
  > "${SESSION}/ibs_op_ld.txt"

${CLI_BIN} report \
  -i "${SESSION}" \
  --view ibs_op_ld_lat \
  --show-event-count \
  --show-percentage \
  > "${SESSION}/ibs_op_ld_lat.txt"
```

重点看：

- 在 `ibs_op_ld.txt` 里：
  - `IBS_LD_PEER_CACHE_HIT_RATE_%`
  - `IBS_LD_LOCAL_DRAM_HIT_RATE_%`
- 在 `ibs_op_ld_lat.txt` 里：
  - `IBS_LD_L1_DC_MISS_LAT_AVE`
  - `%IBS_LD_PEER_CACHE_HIT_LAT`
  - `%IBS_LD_LOCAL_DRAM_HIT_LAT`

解释：

- `IBS_LD_PEER_CACHE_HIT_RATE_%`：同一 NUMA node、不同 CCX 的 cache 返回，在样本中的占比
- `%IBS_LD_PEER_CACHE_HIT_LAT`：这一路在总 miss latency cycles 里的占比
- `IBS_LD_LOCAL_DRAM_HIT_RATE_%` / `%IBS_LD_LOCAL_DRAM_HIT_LAT` 同理

### 4.3 反推来源平均延迟

把上一步抄出来的 5 个数代入下面脚本：

```bash
MISS_LAT_AVE=0
PEER_RATE=0
PEER_LAT_SHARE=0
LOCAL_DRAM_RATE=0
LOCAL_DRAM_LAT_SHARE=0

export MISS_LAT_AVE PEER_RATE PEER_LAT_SHARE LOCAL_DRAM_RATE LOCAL_DRAM_LAT_SHARE

python - <<'PY'
import os

miss_lat_ave = float(os.environ["MISS_LAT_AVE"])
peer_rate = float(os.environ["PEER_RATE"])
peer_lat_share = float(os.environ["PEER_LAT_SHARE"])
local_dram_rate = float(os.environ["LOCAL_DRAM_RATE"])
local_dram_lat_share = float(os.environ["LOCAL_DRAM_LAT_SHARE"])

def avg_cycles(total_avg, share_pct, rate_pct):
    if rate_pct == 0:
        return None
    return total_avg * (share_pct / rate_pct)

peer_avg = avg_cycles(miss_lat_ave, peer_lat_share, peer_rate)
local_dram_avg = avg_cycles(miss_lat_ave, local_dram_lat_share, local_dram_rate)

print("another_ccx_avg_cycles =", peer_avg)
print("local_memory_avg_cycles =", local_dram_avg)
PY
```

结果解释：

- 若 `another_ccx_avg_cycles > local_memory_avg_cycles`，说明在这一 workload 下，`another CCX cache` 这一路样本平均更慢。
- 若反过来，则说明本地 DRAM 这一路更慢。
- 这是“当前 workload + 当前时间窗 + 当前绑核/NUMA 口径”下的样本平均值，不要外推成硬件固定常数。

## 5. 如需看原始事件名

如果你需要进一步核对 raw 事件，可再跑：

```bash
${CLI_BIN} report \
  -i "${SESSION}" \
  --view all \
  --show-event-count \
  --show-percentage \
  > "${SESSION}/all.txt"
```

或者导出 ASCII dump：

```bash
${CLI_BIN} translate \
  -i "${SESSION}" \
  --ascii event-dump
```

证据：`/opt/AMDuProf_5.2-606/bin/Help/text/AMDuProf_Translate_Help.txt:132-140`

## 6. 当前口径下最稳的写法

- `AMDuProfPcm` 负责回答 phase 级：
  - `Ave L3 Miss Latency`
  - `L3 Miss Latency From another CCX in same node (%)`
  - `L3 Miss Latency From Local Memory or I/O (%)`
- `AMDuProfCLI IBS` 负责回答来源级：
  - `another CCX cache` 样本占比
  - `local memory` 样本占比
  - 以及它们各自的平均 `cycles`

不要把 `Pcm` 的“延迟占比”直接写成“该来源单次访问一定更慢”；若要比较来源绝对延迟，优先引用本实验文档里的 `IBS` 反推结果。
