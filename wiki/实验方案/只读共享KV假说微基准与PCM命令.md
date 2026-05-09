# 共享 KV 微基准与 PCM 命令

## 目的
- 用最小共享只读微基准，验证“同一块只读数据被多个线程反复复用时，线程落在同一 `CCD` 或不同 `CCD`，其 `L3/DC` 特征是否不同”。
- 用最小共享写微基准，比较“多个线程反复覆盖同一批 cache line”时，同 `CCD` 与跨 `CCD` 的访存与一致性特征。

## 只读版程序在读什么数据
- 程序源码：`tools/microbench/read_shared_cache.c`
- 它先分配一块 `64B` 对齐的共享数组 `shared_words`，总大小由 `--workset-bytes` 指定。
- 数组按 cache line 粒度初始化：第 `line` 条 cache line 的第一个 `uint64_t` 被写成 `line + 1`。
- 每个线程真正读取的是：

```c
shared_words[line * (64 / sizeof(uint64_t))]
```

- 也就是：
  - 每次内层循环只读取每条 `64B cache line` 的第一个 `8B word`
  - 不会把整条 cache line 的 `64B` 都逐字读一遍
  - 这样做的目的，是让每次循环都“触碰一次每条 cache line”，重点看 cache line 级别的共享只读行为，而不是把程序做成纯顺序带宽压测
- 所有线程共享同一块数组，没有线程写回这块数组；线程只把读取结果累加到各自私有的 `checksum`

## 写共享版程序在写什么数据
- 程序源码：`tools/microbench/write_shared_cache.c`
- 它同样分配一块 `64B` 对齐的共享数组 `shared_words`，总大小由 `--workset-bytes` 指定。
- 数组按 cache line 粒度初始化：第 `line` 条 cache line 的第一个 `_Atomic uint64_t` 初值为 `0`。
- 每个线程在热循环里反复写的是：

```c
atomic_store_explicit(
    &shared_words[line * (64 / sizeof(uint64_t))],
    thread_index + 1,
    memory_order_relaxed);
```

- 也就是：
  - 每次内层循环只覆盖每条 `64B cache line` 的第一个 `8B word`
  - 所有线程反复写同一批地址
  - 使用 `_Atomic uint64_t` 和 `memory_order_relaxed`，把“多线程同时写同一地址”的行为收敛为定义明确的原子写
- 程序结束后，主线程会把每条 cache line 第一个字的最终值求和，输出为 `final_checksum`
- `final_checksum` 只用于确认共享数组确实被写过；由于每条 cache line 的最终值取决于最后一次覆盖它的线程，因此这个值不适合作为性能比较指标

## 当前机器可直接使用的 CPU 组
- 以下 CPU 编号来自当前 `NPS1_TP2` dry-run 里已经落盘的 `locality_groups`
- `256-271` 属于同一个本地 `L3/CCD`
- `272-287` 属于另一个本地 `L3/CCD`
- 其他可用 `CCD` 首核可直接取：
  - `256`
  - `272`
  - `288`
  - `304`
  - `320`
  - `336`
  - `352`
  - `368`

## 编译

```bash
cd /home/zjj/vllm
conda activate vllm-cpu

gcc -O2 -pthread -std=c11 tools/microbench/read_shared_cache.c \
  -o tools/microbench/read_shared_cache

gcc -O2 -pthread -std=c11 tools/microbench/write_shared_cache.c \
  -o tools/microbench/write_shared_cache
```

## 只读版最小 dry-run

### 同 CCD，2 线程

```bash
cd /home/zjj/vllm

numactl --cpunodebind=0 --membind=0 \
  ./tools/microbench/read_shared_cache \
  --threads 2 \
  --cpus 256,257 \
  --workset-bytes 8123000 \
  --iters 600000
```

### 跨 CCD，2 线程

```bash
cd /home/zjj/vllm

numactl --cpunodebind=0 --membind=0 \
  ./tools/microbench/read_shared_cache \
  --threads 2 \
  --cpus 256,272 \
  --workset-bytes 8123000 \
  --iters 1000
```

### 同 CCD，8 线程

```bash
cd /home/zjj/vllm

numactl --cpunodebind=0 --membind=0 \
  ./tools/microbench/read_shared_cache \
  --threads 8 \
  --cpus 256,257,258,259,260,261,262,263 \
  --workset-bytes 16123000 \
  --iters 100000
```

### 跨 8 个 CCD，8 线程

```bash
cd /home/zjj/vllm

numactl --cpunodebind=0 --membind=0 \
  ./tools/microbench/read_shared_cache \
  --threads 8 \
  --cpus 256,272,288,304,320,336,352,368 \
  --workset-bytes 16123000 \
  --iters 500000
```

## 写共享版最小 dry-run

### 同 CCD，2 线程

```bash
cd /home/zjj/vllm

numactl --cpunodebind=0 --membind=0 \
  ./tools/microbench/write_shared_cache \
  --threads 2 \
  --cpus 256,257 \
  --workset-bytes 8388608 \
  --iters 20000
```

### 跨 CCD，2 线程

```bash
cd /home/zjj/vllm

numactl --cpunodebind=0 --membind=0 \
  ./tools/microbench/write_shared_cache \
  --threads 2 \
  --cpus 256,272 \
  --workset-bytes 8388608 \
  --iters 20000
```

### 同 CCD，8 线程

```bash
cd /home/zjj/vllm

numactl --cpunodebind=0 --membind=0 \
  ./tools/microbench/write_shared_cache \
  --threads 8 \
  --cpus 256,257,258,259,260,261,262,263 \
  --workset-bytes 8388608 \
  --iters 1000
```

### 跨 8 个 CCD，8 线程

```bash
cd /home/zjj/vllm

numactl --cpunodebind=0 --membind=0 \
  ./tools/microbench/write_shared_cache \
  --threads 8 \
  --cpus 256,272,288,304,320,336,352,368 \
  --workset-bytes 8388608 \
  --iters 1000
```

## PCM 前的迭代次数校准
- `AMDuProfPcm` 只在程序实际运行期间采样，因此正式采样前先用小 `iters` 做一次 dry-run，查看 JSON 里的 `elapsed_sec`
- 再按比例把 `iters` 放大到明显长于 `PCM` 的 `--start-delay + -d`
- 例如 dry-run 的 `iters=100` 若得到 `elapsed_sec=0.8`，目标总运行时间若想达到 `25s`，则可先把 `iters` 粗略放大到 `3125`

## PCM 命令

### 一次性环境变量

```bash
cd /home/zjj/vllm
conda activate vllm-cpu

PCM_BIN=/opt/AMDuProf_5.2-606/bin/AMDuProfPcm
OUT_ROOT=/home/zjj/vllm/test_results/microbench/read_shared_cache

mkdir -p "${OUT_ROOT}"
```

## 只读版 PCM 命令

### 同 CCD，2 线程

```bash
cd /home/zjj/vllm

"${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -I 100 \
  -O "${OUT_ROOT}/same_ccd_2t_8m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/read_shared_cache \
    --threads 2 \
    --cpus 256,257 \
    --workset-bytes 8554432 \
    --iters 300000
```

### 跨 CCD，2 线程

```bash
"${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -I 100 \
  -O "${OUT_ROOT}/cross_ccd_2t_8m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/read_shared_cache \
    --threads 2 \
    --cpus 256,272 \
    --workset-bytes 8554432 \
    --iters 300000
```

### 同 CCD，8 线程

```bash
cd /home/zjj/vllm

sudo "${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -d 20 \
  -I 100 \
  -O "${OUT_ROOT}/same_ccd_8t_32m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/read_shared_cache \
    --threads 8 \
    --cpus 256,257,258,259,260,261,262,263 \
    --workset-bytes 33554432 \
    --iters 5000
```

### 跨 8 个 CCD，8 线程

```bash
cd /home/zjj/vllm

sudo "${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -d 20 \
  -I 100 \
  -O "${OUT_ROOT}/cross_ccd_8t_32m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/read_shared_cache \
    --threads 8 \
    --cpus 256,272,288,304,320,336,352,368 \
    --workset-bytes 33554432 \
    --iters 5000
```

## 工作集 sweep
- 若要区分 `L2` 内、`L3` 内和超过本地 `L3` 的区间，可直接替换 `--workset-bytes`
- 当前建议点：
  - `1048576`：`1 MiB`
  - `8388608`：`8 MiB`
  - `33554432`：`32 MiB`
  - `67108864`：`64 MiB`

示例：

```bash
cd /home/zjj/vllm

sudo "${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -d 20 \
  -I 100 \
  -O "${OUT_ROOT}/cross_ccd_2t_64m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/read_shared_cache \
    --threads 2 \
    --cpus 256,272 \
    --workset-bytes 67108864 \
    --iters 5000
```

## 写共享版 PCM 命令

### 一次性环境变量

```bash
cd /home/zjj/vllm
conda activate vllm-cpu

PCM_BIN=/opt/AMDuProf_5.2-606/bin/AMDuProfPcm
OUT_ROOT_WRITE=/home/zjj/vllm/test_results/microbench/write_shared_cache

mkdir -p "${OUT_ROOT_WRITE}"
```

### 同 CCD，2 线程

```bash
cd /home/zjj/vllm

"${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -I 100 \
  -O "${OUT_ROOT_WRITE}/same_ccd_2t_8m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/write_shared_cache \
    --threads 2 \
    --cpus 256,257 \
    --workset-bytes 8388608 \
    --iters 220000
```

### 跨 CCD，2 线程

```bash
cd /home/zjj/vllm

"${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -I 100 \
  -O "${OUT_ROOT_WRITE}/cross_ccd_2t_8m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/write_shared_cache \
    --threads 2 \
    --cpus 256,272 \
    --workset-bytes 8388608 \
    --iters 220000
```

### 同 CCD，8 线程

```bash
cd /home/zjj/vllm

sudo "${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -d 20 \
  -I 100 \
  -O "${OUT_ROOT_WRITE}/same_ccd_8t_8m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/write_shared_cache \
    --threads 8 \
    --cpus 256,257,258,259,260,261,262,263 \
    --workset-bytes 8388608 \
    --iters 1000
```

### 跨 8 个 CCD，8 线程

```bash
cd /home/zjj/vllm

sudo "${PCM_BIN}" profile \
  -m ipc,l3,dc \
  -a \
  -s \
  --start-delay 1000 \
  -d 20 \
  -I 100 \
  -O "${OUT_ROOT_WRITE}/cross_ccd_8t_8m" -- \
  numactl --cpunodebind=0 --membind=0 \
    ./tools/microbench/write_shared_cache \
    --threads 8 \
    --cpus 256,272,288,304,320,336,352,368 \
    --workset-bytes 8388608 \
    --iters 1000
```

## 结果怎么看
- 重点先看：
  - `Demand DC Fills From another CCX in same node`
  - `Demand DC Fills From Local L2`
  - `Demand DC Fills From Local Memory or I/O`
  - `L3 Access`
  - `L3 Miss`
  - `IPC`
- 若“同一块只读共享数组在跨 CCD 情况下会持续更多地走 remote cache 补线”，则跨 CCD case 应更容易看到：
  - 更高的 `Demand DC Fills From another CCX in same node`
  - 更高的 `Ave L3 Miss Latency`
  - 在大工作集下更差的 `IPC`
- 若同 CCD 与跨 CCD 差异只体现在首次补线，而 steady-state 复用很快转为本地命中，则随着 `iters` 增大、工作集仍留在 cache 时，两边差距可能缩小。
- 对写共享版，`IPC/L3/DC` 指标反映的是反复覆盖共享 cache line 时的访存与一致性代价；它不再对应只读 `KV` 假说。

## 边界
- 这个程序只读每条 cache line 的第一个 `8B`，不是完整顺序带宽压测。
- 它验证的是“共享只读数据”的 cache 行为，不验证写共享或读写竞争。
- 写共享版只覆盖每条 cache line 的第一个 `8B`，不代表完整 cache line 带宽。
- 写共享版验证的是“多线程反复覆盖同一批共享地址”时的行为，不等价于 attention 的只读 `KV` 访问。
- 当前 `--iters 5000` 只是起始值，正式 `PCM` 前仍建议先用 dry-run 校准运行时间。
