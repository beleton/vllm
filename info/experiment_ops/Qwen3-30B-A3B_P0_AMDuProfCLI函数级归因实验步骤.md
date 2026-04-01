# Qwen3-30B-A3B：P0 `Prefill/Decode` 函数级归因实验步骤

> 更新时间：2026-04-01 21:33 +0800  
> 适用机器：`2 x AMD EPYC 9745 128-Core Processor`  
> 目标模型：`/models/Qwen3-30B-A3B`  
> 结论先行：下一步应先做 `P0`，再决定是否继续做 attention 局部性实验或拓扑优化。

## 0. 为什么先做 `P0`

- 当前已经有 phase 级证据，说明 `decode_B16_I1_O1024` 的 `L3 Miss %` 与 `Ave L3 Miss Latency` 明显高于 `prefill_B16_I1024_O1`。证据目录：
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/decode_B16_I1_O1024/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-48-58`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/prefill_B16_I1024_O1/metric2_l3_dc_l2_memory/AMDuProfPcm-Multi_Mar-23-2026_20-27-32`
- 但这些结果只回答“哪一阶段 miss 更高”，还没有回答“miss 主要由哪个函数/调用链制造”。
- 因此，下一步最应该先做 `AMDuProfCLI hotspots + IBS L3-miss`，确认高 miss 主要落在：
  - `cpu_attention_with_kv_cache`
  - 还是 `cpu_fused_moe`
  - 还是 `linear / TP all_reduce / 通信等待`

## 1. 本实验要回答的问题

1. `decode_B16_I1_O1024` 的 CPU 时间主要压在哪些函数。
2. `decode_B16_I1_O1024` 的 `L3 miss` 样本主要落在哪些函数。
3. `prefill_B16_I1024_O1` 的对应热点函数和 miss 函数是什么。
4. `decode` 与 `prefill` 的差异是否主要出现在 attention 路径。

## 2. 前置条件

### 2.1 已验证可用的工具

- `AMDuProfCLI`：`/opt/AMDuProf_5.2-606/bin/AMDuProfCLI`
- `AMDuProfPcm`：`/opt/AMDuProf_5.2-606/bin/AMDuProfPcm`
- `vllm` 可执行文件：`/home/zjj/.conda/envs/vllm-cpu/bin/vllm`

### 2.2 `strict-batch` 口径

- `strict-batch` 必须在 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 下运行。证据：`vllm/benchmarks/strict_batch.py:230-235`
- `strict-batch` 当前只支持 `--n 1`。证据：`vllm/benchmarks/strict_batch.py:225-227`
- 实验后要检查输出 JSON 里的：
  - `avg_first_step_scheduled_requests`
  - `avg_first_step_scheduled_tokens`
- 这两个字段由 `strict_batch.py` 直接写出。证据：`vllm/benchmarks/strict_batch.py:338-346`

## 3. 一次性准备

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo /opt/AMDuProf_5.2-606/bin/AMDPcmSetCapability.sh

export VLLM_BIN=/home/zjj/.conda/envs/vllm-cpu/bin/vllm
export CLI_BIN=/opt/AMDuProf_5.2-606/bin/AMDuProfCLI
export MODEL=/models/Qwen3-30B-A3B
export BASE=/home/zjj/vllm/test_results/PD_Test/Qwen3-30B-A3B/CLI_res
export LD_LIBRARY_PATH=/home/zjj/.conda/envs/vllm-cpu/lib:${LD_LIBRARY_PATH}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CPU_KVCACHE_SPACE=32

# 先用 AMDuProfPcm 或单独 strict-batch dry-run 校准 steady-state 时间窗。
# 注意：AMDuProfCLI 的 --start-delay / --duration 单位是秒；
# AMDuProfPcm 的 --start-delay 单位是毫秒，不能直接照抄。
# batchsize16,P1024时，初始化+warmup约120s
# batchsize16,D1024时，初始化+warmup约200s
export DECODE_START_DELAY_SEC=300
export DECODE_DURATION_SEC=240
export PREFILL_START_DELAY_SEC=300
export PREFILL_DURATION_SEC=240

# attach 模式下，start-delay 是“attach 成功后再等多久”，
# 不再包含模型加载与 worker 拉起阶段。
# 未重新校准前先设 0，优先保证 hotspots 能正常跑完。
export DECODE_ATTACH_START_DELAY_SEC=300
export PREFILL_ATTACH_START_DELAY_SEC=300
```
- `2026-04-01 11:16:07 +0800` 的 `hotspots` launcher 失败证据：
  - session 路径：`test_results/PD_Test/Qwen3-30B-A3B/CLI_res/decode_B16_I1_O1024/hotspots/AMDuProf-env-Hotspots_Apr-01-2026_11-16-07`
  - `AMDuProfCLI report` 显示 `Profile Duration=7.910 seconds`，而命令里设置了 `--start-delay 300 --duration 240`，说明目标进程在进入采样窗前就异常退出。
  - `report.csv` 的 `5 HOTTEST FUNCTIONS / PROCESSES / MODULES` 为空，说明这次 session 不可用于函数级归因。
  - 根因链：
    - CPU backend 强制 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。证据：`vllm/platforms/cpu.py:273-276`
    - `TP=2` 时，`MultiprocExecutor` 会用 multiprocessing context 拉 worker 子进程。证据：`vllm/v1/executor/multiproc_executor.py:136-165`、`vllm/v1/executor/multiproc_executor.py:613-624`
    - `AMDuProfCLI hotspots` 会给 Python 入口注入 `/opt/AMDuProf_5.2-606/bin/AMDTPyTracer.py`；该脚本对 `python -c ...` 直接 `exec(args)`。证据：`/opt/AMDuProf_5.2-606/bin/AMDTPyTracer.py:38-47`
    - spawned child 进入 `multiprocessing/spawn.py::spawn_main` 时要求 `is_forking(sys.argv)` 为真；被 tracer 改写后触发 `AssertionError: Not forking`。这与 `2026-04-01 11:16:07 +0800` 的实际报错一致。
- 因此：`hotspots` 在当前 `TP=2 + vllm Python launcher` 口径下不要再用 launcher 模式，改为“先启动 `strict-batch`，再 `-p PID` attach worker”；`IBS L3-miss` 当前仍可保留 launcher 模式。

## 4. 先跑 `decode_B16_I1_O1024`

### 4.1 `hotspots`

```bash
OUT=${BASE}/decode_B16_I1_O1024/hotspots
mkdir -p "${OUT}"

APP_LOG="${OUT}/strict_batch_stdout.log"

PYTHONUNBUFFERED=1 \
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
${VLLM_BIN} bench strict-batch \
  --model ${MODEL} \
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
  --input-len 1 \
  --output-len 1024 \
  --n 1 \
  --num-iters-warmup 1 \
  --num-rounds 20 \
  --disable-detokenize \
  --output-json ${OUT}/strict_batch_result.json \
>"${APP_LOG}" 2>&1 &
APP_PID=$!

while ! grep -q 'Worker_TP1 pid=' "${APP_LOG}"; do
  if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "strict-batch exited early, see ${APP_LOG}"
    wait "${APP_PID}"
    exit 1
  fi
  sleep 2
done

WORKER_PIDS=$(grep -o 'Worker_TP[0-9]\+ pid=[0-9]\+' "${APP_LOG}" \
  | sed 's/.*pid=//' \
  | sort -u \
  | paste -sd, -)

${CLI_BIN} collect \
  --config hotspots \
  -g \
  --call-graph-depth 64 \
  -p "${WORKER_PIDS}" \
  --start-delay ${DECODE_ATTACH_START_DELAY_SEC} \
  --duration ${DECODE_DURATION_SEC} \
  --output-dir "${OUT}"

wait "${APP_PID}"
```

- 这里 attach 的是实际执行 TP attention/MoE/linear 的 worker 子进程，不是父进程。

### 4.2 `IBS L3-miss`
collect 默认不会在采样结束时杀掉被启动的程序；只有显式加 -b/--terminate 才会在采样结束后终止 launched application

```bash
OUT=${BASE}/decode_B16_I1_O1024/ibs_l3miss
mkdir -p "${OUT}"

${CLI_BIN} collect \
  -e event=ibs-op,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp \
  --start-delay ${DECODE_START_DELAY_SEC} \
  --duration ${DECODE_DURATION_SEC} \
  --output-dir "${OUT}" \
  env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
    VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
    ${VLLM_BIN} bench strict-batch \
      --model ${MODEL} \
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
      --input-len 1 \
      --output-len 1024 \
      --n 1 \
      --num-iters-warmup 1 \
      --num-rounds 20 \
      --disable-detokenize \
      --output-json ${OUT}/strict_batch_result.json
```

## 5. 再跑 `prefill_B16_I1024_O1`

### 5.1 `hotspots`

```bash
OUT=${BASE}/prefill_B16_I1024_O1/hotspots
mkdir -p "${OUT}"

APP_LOG="${OUT}/strict_batch_stdout.log"

PYTHONUNBUFFERED=1 \
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
${VLLM_BIN} bench strict-batch \
  --model ${MODEL} \
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
  --input-len 1024 \
  --output-len 1 \
  --n 1 \
  --num-iters-warmup 1 \
  --num-rounds 60 \
  --disable-detokenize \
  --output-json ${OUT}/strict_batch_result.json \
>"${APP_LOG}" 2>&1 &
APP_PID=$!

while ! grep -q 'Worker_TP1 pid=' "${APP_LOG}"; do
  if ! kill -0 "${APP_PID}" 2>/dev/null; then
    echo "strict-batch exited early, see ${APP_LOG}"
    wait "${APP_PID}"
    exit 1
  fi
  sleep 2
done

WORKER_PIDS=$(grep -o 'Worker_TP[0-9]\+ pid=[0-9]\+' "${APP_LOG}" \
  | sed 's/.*pid=//' \
  | sort -u \
  | paste -sd, -)

${CLI_BIN} collect \
  --config hotspots \
  -g \
  --call-graph-depth 64 \
  -p "${WORKER_PIDS}" \
  --start-delay ${PREFILL_ATTACH_START_DELAY_SEC} \
  --duration ${PREFILL_DURATION_SEC} \
  --output-dir "${OUT}"

wait "${APP_PID}"
```

### 5.2 `IBS L3-miss`

```bash
OUT=${BASE}/prefill_B16_I1024_O1/ibs_l3miss
mkdir -p "${OUT}"

${CLI_BIN} collect \
  -e event=ibs-op,ibsop-l3miss=1,call-graph \
  --call-graph-mode fp \
  --start-delay ${PREFILL_START_DELAY_SEC} \
  --duration ${PREFILL_DURATION_SEC} \
  --output-dir "${OUT}" \
  env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} \
    VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE} \
    ${VLLM_BIN} bench strict-batch \
      --model ${MODEL} \
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
      --input-len 1024 \
      --output-len 1 \
      --n 1 \
      --num-iters-warmup 1 \
      --num-rounds 60 \
      --disable-detokenize \
      --output-json ${OUT}/strict_batch_result.json
```

## 6. 导出报告

### 6.1 `decode hotspots` 报告

```bash
SESSION=$(ls -td ${BASE}/decode_B16_I1_O1024/hotspots/*/ | head -n1)

${CLI_BIN} report \
  -i "${SESSION}" \
  --category cpu \
  --view timer \
  --sort-by metric=total_cpu_time \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_timer.txt"
```

### 6.2 `decode IBS L3-miss` 报告

```bash
SESSION=$(ls -td ${BASE}/decode_B16_I1_O1024/ibs_l3miss/*/ | head -n1)

AMDuProfCLI report \
  -i "${SESSION}" \
  --category cpu \
  --view ibs_op_ld \
  --sort-by event=ibs-op \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_ibs_l3miss_load.txt"
```

```bash
SESSION=$(ls -td ${BASE}/decode_B16_I1_O1024/ibs_l3miss/*/ | head -n1)

AMDuProfCLI report \
  -i "${SESSION}" \
  --category cpu \
  --view ibs_op_ls_overview \
  --sort-by event=ibs-op \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_ibs_l3miss_ls_overview.txt"
```

```bash
SESSION=$(ls -td ${BASE}/decode_B16_I1_O1024/ibs_l3miss/*/ | head -n1)

AMDuProfCLI report \
  -i "${SESSION}" \
  --category cpu \
  --view ibs_op_ld_lat \
  --sort-by event=ibs-op \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_ibs_l3miss_ld_lat.txt"
```

- `ibs_op_ld`：先看哪些函数拿到最多 `IBS_LD_L1_DC_MISS` 样本，以及 `LOCAL/PEER/RMT CACHE`、`LOCAL/RMT DRAM` 的 hit rate。
- `ibs_op_ls_overview`：补看 `IBS_STORE`、`IBS_ST_L1_DC_MISS`，避免只盯 load。
- `ibs_op_ld_lat`：补看 miss latency 更像耗在 `local cache`、`peer cache`、`remote cache`、`local dram` 还是 `remote dram`。

### 6.3 `prefill hotspots` 报告

```bash
SESSION=$(ls -td ${BASE}/prefill_B16_I1024_O1/hotspots/*/ | head -n1)

AMDuProfCLI report \
  -i "${SESSION}" \
  --category cpu \
  --view timer \
  --sort-by metric=total_cpu_time \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_timer.txt"
```

### 6.4 `prefill IBS L3-miss` 报告

```bash
SESSION=$(ls -td ${BASE}/prefill_B16_I1024_O1/ibs_l3miss/*/ | head -n1)

AMDuProfCLI report \
  -i "${SESSION}" \
  --category cpu \
  --view ibs_op_ld \
  --sort-by event=ibs-op \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_ibs_l3miss_load.txt"
```

```bash
SESSION=$(ls -td ${BASE}/prefill_B16_I1024_O1/ibs_l3miss/*/ | head -n1)

AMDuProfCLI report \
  -i "${SESSION}" \
  --category cpu \
  --view ibs_op_ls_overview \
  --sort-by event=ibs-op \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_ibs_l3miss_ls_overview.txt"
```

```bash
SESSION=$(ls -td ${BASE}/prefill_B16_I1024_O1/ibs_l3miss/*/ | head -n1)

AMDuProfCLI report \
  -i "${SESSION}" \
  --category cpu \
  --view ibs_op_ld_lat \
  --sort-by event=ibs-op \
  --detail \
  -g \
  --cutoff 20 \
  --stdout | tee "${SESSION}/report_ibs_l3miss_ld_lat.txt"
```

### 6.5 导出到 Windows GUI

- 第 `6.1` 到 `6.4` 节生成的 `report_*.txt` 适合文本分析；若要带到 Windows 上用 `AMDuProf` GUI 打开，建议额外导出 `session.tar.gz`。
- 下面命令都按“取该 workload 最新 session”执行，可直接复制到终端。

```bash
SESSION=$(ls -td ${BASE}/decode_B16_I1_O1024/hotspots/*/ | head -n1)

AMDuProfCLI translate \
  -i "${SESSION}" \
  --export-session
```

```bash
SESSION=$(ls -td ${BASE}/decode_B16_I1_O1024/ibs_l3miss/*/ | head -n1)

AMDuProfCLI translate \
  -i "${SESSION}" \
  --export-session
```

```bash
SESSION=$(ls -td ${BASE}/prefill_B16_I1024_O1/hotspots/*/ | head -n1)

AMDuProfCLI translate \
  -i "${SESSION}" \
  --export-session
```

```bash
SESSION=$(ls -td ${BASE}/prefill_B16_I1024_O1/ibs_l3miss/*/ | head -n1)

AMDuProfCLI translate \
  -i "${SESSION}" \
  --export-session
```

- 每条命令执行后，目标 session 目录下会生成 `session.tar.gz`；把这个 archive 拷到 Windows 再导入 GUI。

### 6.6 这份文档里会看到的指标含义

#### 6.6.1 `hotspots` 相关

- `CPU_TIME`：
  - `timer` 视图里的基础时间指标。
  - `AMDuProfCLI` 的定义是“CPU core 真正在执行代码的时间”。
  - 在函数表里更接近函数自身的 sampled CPU time。
- `TOTAL_CPU_TIME (s)`：
  - `timer` 视图里的派生指标。
  - `AMDuProfCLI` 的定义是“函数及其子调用一共花掉的 CPU time”。
  - 排函数主路径时，优先看它。
- `SELF SAMPLES (seconds)`：
  - callgraph 里函数自身命中的样本时间。
  - 若它高，说明时间主要压在该函数本体，而不是子调用。
- `DEEP SAMPLES (seconds)`：
  - callgraph 里“该函数 + 它下面整条子调用链”命中的总样本时间。
  - 若它高而 `SELF SAMPLES` 低，说明它更像是热点父路径入口。
- `% DEEP SAMPLES`：
  - `DEEP SAMPLES` 占全进程样本的比例。
  - 适合看结构占比，不适合替代 raw time。
- `PATH COUNT`：
  - callgraph function table 里命中该函数的调用路径数。
  - 可近似理解为“这个函数是从多少条不同调用链走进来的”。

#### 6.6.2 `IBS` 样本量相关

- `IBS_ALL_OPS`：
  - 收到的全部 IBS op 样本数。
  - 在 `ibs_op_ld` / `ibs_op_ls_overview` / `ibs_op_ld_lat` 里，它是最基础的样本量分母。
- `IBS_LOAD`：
  - 命中到 load 指令的 IBS 样本数。
- `IBS_STORE`：
  - 命中到 store 指令的 IBS 样本数。
- `IBS_LOAD_STORE`：
  - 命中到 load 或 store 的 IBS 样本数。
- `%IBS_LOAD`：
  - `IBS_LOAD / IBS_ALL_OPS`。
  - 用来看这批 IBS 样本里 load 占比高不高。

#### 6.6.3 `IBS` miss 与命中来源相关

- `IBS_LD_L1_DC_MISS`：
  - load 样本里，发生 L1 data cache miss 的样本数。
  - 这是这份文档里最直接的“load miss 样本量”指标。
- `IBS_ST_L1_DC_MISS`：
  - store 样本里，发生 L1 data cache miss 的样本数。
  - 若它也高，就不能只从 load 路径解释 miss。
- `IBS_LD_L1_DC_MISS_RATE_%`：
  - `IBS_LD_L1_DC_MISS / IBS_LOAD`。
  - 表示 load 样本里有多少比例发生了 L1 DC miss。
- `IBS_LD_L1_DC_HIT_RATE_%`：
  - load 样本里直接命中 L1 DC 的比例。
- `IBS_LD_L2_HIT_RATE_%`：
  - load 样本里 miss 掉 L1 后，在 L2 命中的比例。
- `IBS_LD_LOCAL_CACHE_HIT_RATE_%`：
  - `AMDuProfCLI` 定义为：load 样本被同一 `CCX` 内共享 `L3` 或其他 `L1/L2` 服务的比例。
  - 可以粗看成“同 CCX cache 命中”。
- `IBS_LD_PEER_CACHE_HIT_RATE_%`：
  - `AMDuProfCLI` 定义为：load 样本被同一 NUMA node、但不同 `CCX` 的 `L2/L3` 服务的比例。
  - 可以粗看成“同 node 跨 CCX cache 命中”。
- `IBS_LD_RMT_CACHE_HIT_RATE_%`：
  - `AMDuProfCLI` 定义为：load 样本被不同 NUMA node 的 `L2/L3` 服务的比例。
  - 可以粗看成“跨 node cache 命中”。
- `IBS_LD_LOCAL_DRAM_HIT_RATE_%`：
  - load 样本由本地 NUMA node DRAM 服务的比例。
- `IBS_LD_RMT_DRAM_HIT_RATE_%`：
  - load 样本由远端 NUMA node DRAM 服务的比例。

#### 6.6.4 `IBS` latency 相关

- `IBS_LD_L1_DC_MISS_LAT`：
  - 所有 load L1 DC miss 样本累计的 miss latency cycles。
  - 是总延迟，不是平均值。
- `IBS_LD_L1_DC_MISS_LAT_AVE`：
  - `IBS_LD_L1_DC_MISS_LAT / IBS_LD_L1_DC_MISS`。
  - 更适合直接比较“单次 miss 平均有多慢”。
- `%IBS_LD_L2_HIT_LAT`：
  - L1 DC miss latency 里，有多少比例耗在最终由 L2 命中的样本上。
- `%IBS_LD_LOCAL_CACHE_HIT_LAT`：
  - L1 DC miss latency 里，有多少比例耗在同 `CCX` cache 命中上。
- `%IBS_LD_PEER_CACHE_HIT_LAT`：
  - L1 DC miss latency 里，有多少比例耗在同 node 跨 `CCX` cache 命中上。
- `%IBS_LD_RMT_CACHE_HIT_LAT`：
  - L1 DC miss latency 里，有多少比例耗在跨 node cache 命中上。
- `%IBS_LD_LOCAL_DRAM_HIT_LAT`：
  - L1 DC miss latency 里，有多少比例耗在本地 DRAM 上。
- `%IBS_LD_RMT_DRAM_HIT_LAT`：
  - L1 DC miss latency 里，有多少比例耗在远端 DRAM 上。
- `IBS_TAG_TO_RETIRE_CYCLES`：
  - 所有 IBS op 样本从 tagged 到 retire 的总 cycles。
  - 可近似理解为“这些样本整体退休前一共拖了多久”。
- `%IBS_LD_L1_DC_MISS_LAT_CYCLES`：
  - `IBS_LD_L1_DC_MISS_LAT / IBS_TAG_TO_RETIRE_CYCLES`。
  - 表示 tag-to-retire 时间里，有多少比例可以归到 load miss latency。

## 7. 结果自检
### 7.2 先看哪些函数

优先检查这些符号是否排到前面：

- attention：
  - `cpu_attention_with_kv_cache`
  - `cpu_attn_get_scheduler_metadata`
  - `csrc/cpu/cpu_attn_impl.hpp` 对应函数
- MoE / MLP：
  - `cpu_fused_moe`
  - `linear`
- TP / 通信：
  - `tensor_model_parallel_all_reduce`
  - `gloo`
  - `cpu_communicator`

### 7.4 `IBS` 报告怎么看

- 判断“谁制造了 miss”时，优先看 `ibs_op_ld` / `ibs_op_ls_overview` 里的 raw sample count、函数 Top 排名和 callgraph deep samples。
- 百分比只适合辅助看结构占比，不适合单独拿来判定“主 miss 来源”。
- 如果 `ibs_op_ls_overview` 显示 `IBS_STORE` 或 `IBS_ST_L1_DC_MISS` 也很高，就不要只从 load 路径解释问题。
- 如果 `ibs_op_ld_lat` 显示 latency 主要落在 `PEER_CACHE` 或 `RMT_CACHE`，说明问题更像跨 `CCX` / 跨 node 缓存命中延迟；若主要落在 `LOCAL_DRAM` 或 `RMT_DRAM`，才更像 DRAM 路径主导。

## 8. 结果判读标准

- 如果 `decode hotspots` 和 `decode IBS L3-miss` 都主要指向 attention 路径，则下一步进入 attention 局部性实验是合理的。
- 如果 `decode` 的 CPU 时间在 attention，但 `L3 miss` 样本主要落在 `MoE/linear/通信`，则下一步不应直接把 attention 当主矛盾。
- 如果 `prefill` 与 `decode` 都显示 attention 不是主 miss 来源，则 `P2/P3` 的优先级应整体下调。

