# vLLM CPU 推理性能变化根因分析（NPS1/2/4 + TP2/4/8）

## 1. 分析范围与证据边界

### 1.1 目标

解释在双路 `AMD EPYC 9745` 平台上，为什么：

- `NPS2_TP4` 优于 `NPS1_TP2`
- `NPS4_TP8` 明显退化（与“`NPS4_TP8` 更优”的外部结论不一致）

### 1.2 使用的数据批次

本分析只使用仓库内可复查数据（`parallel=16, number=16` 为主观测点）：

- 吞吐与时延：
  - `test_results/DeepSeek-R1-Distill-Llama-8B/benchmark_latest_summary.csv`
- 系统计数器（System Aggregated）：
  - `test_results/DeepSeek-R1-Distill-Llama-8B/pcm_cumulative_system_by_nps.csv`
- L3 指标（conc=16）：
  - `test_results/DeepSeek-R1-Distill-Llama-8B/pcm_l3_metrics_conc16_system.csv`
- Roofline（参考）：
  - `test_results/DeepSeek-R1-Distill-Llama-8B/*/roofline_bs16/*/report-roofline.csv`

说明：

- `NPS2_TP4` 的 roofline 有两批次（`2026-02-05` 与 `2026-02-06`）且吞吐差异较大，本文将 roofline 作为辅助证据，不作为主结论依据。

---

## 2. 事实层：现象与关键指标

### 2.1 关键现象（concurrency=16）

| 配置 | Avg TTFT(s) | Avg TPOT(s) | Output tok/s |
| --- | ---: | ---: | ---: |
| `NPS1_TP2` | 5.5891 | 0.0755 | 197.5150 |
| `NPS2_TP4` | 4.8408 | 0.0717 | 209.4394 |
| `NPS4_TP8` | 10.2484 | 0.0945 | 153.2066 |

直接对比：

- `NPS2_TP4` vs `NPS1_TP2`：吞吐 `+6.0%`，TTFT `-13.4%`，TPOT `-5.0%`
- `NPS4_TP8` vs `NPS2_TP4`：吞吐 `-26.8%`，TTFT `+111.7%`，TPOT `+31.8%`

### 2.2 与退化强相关的硬件指标

| 指标 | NPS1 | NPS2 | NPS4 |
| --- | ---: | ---: | ---: |
| CPI (Sys+User) | 1.73 | 1.72 | 2.37 |
| IPC (Sys+User) | 0.58 | 0.58 | 0.42 |
| GIPS | 369.82 | 371.33 | 273.02 |
| Total Mem Bw (GB/s) | 271.03 | 302.40 | 231.04 |
| Remote Inbound Read (GB/s) | 1.04 | 4.32 | 8.95 |
| Remote DRAM Reads % | 0.03 | 0.12 | 0.43 |
| Ave L3 Miss Latency (ns) | 347.45 | 388.43 | 441.73 |
| Frontend_Bound (%) | 22.68 | 18.48 | 23.30 |
| Retiring (%) | 7.94 | 7.82 | 5.84 |
| System time (%) | 6.50 | 6.63 | 8.73 |

观察结论（事实）：

- `NPS2` 相比 `NPS1`：总带宽更高（`+11.6%`），吞吐更高。
- `NPS4` 相比 `NPS2`：带宽显著下降（`-23.6%`），同时出现 `CPI` 恶化、`IPC` 下滑、远端访存上升、L3 miss 延迟上升。

---

## 3. 源码路径：从 `vllm serve` 到 TP/NUMA/线程

下面只保留与本问题最相关的链路。

### 3.1 入口与进程结构

1. CLI 入口执行 `ServeSubcommand.cmd()` 并启动 `run_server(args)`  
   代码：`vllm/entrypoints/cli/serve.py:48-112`
2. API server 在 `run_server_worker()` 内构建 `engine_client`  
   代码：`vllm/entrypoints/openai/api_server.py:922-942`
3. `AsyncLLM` 把请求异步发送给 EngineCore（ZMQ）  
   代码：`vllm/v1/engine/async_llm.py:404-417`，`vllm/v1/engine/core_client.py:897-957`
4. EngineCore 通过 `MultiprocExecutor` 启动 TP worker 进程  
   代码：`vllm/v1/engine/core.py:104-119`，`vllm/v1/executor/multiproc_executor.py:99-167`

### 3.2 CPU 线程绑核与 NUMA 策略生效点

1. CPU 平台配置在创建配置时注入：
   - 强制 `spawn`
   - 非 `nobind` 时设置 `OMP_NUM_THREADS`
   代码：`vllm/platforms/cpu.py:179-285`
2. 每个 worker 在 `init_device()` 中决定绑核方案（默认 `auto`）  
   代码：`vllm/v1/worker/cpu_worker.py:54-89`
3. `auto` 逻辑：
   - 取 `lscpu -J -e=CPU,CORE,NODE`
   - 每个 rank 选一个 NUMA node
   - x86 每物理核只取一个 SMT 线程
   - 默认预留 `1` 个 core（当 `world_size>1`）  
   代码：`vllm/platforms/cpu.py:358-397`，`vllm/v1/worker/cpu_worker.py:126-187`
4. C++ 扩展实际执行：
   - `numa_set_membind` / `numa_set_interleave_mask`
   - `numa_migrate_pages`
   - `omp_set_num_threads` + `sched_setaffinity`  
   代码：`csrc/cpu/utils.cpp:25-165`

### 3.3 TP 通信触发点

1. 每轮调度由 EngineCore 广播到所有 worker 执行  
   代码：`vllm/v1/executor/multiproc_executor.py:302-374, 834-860`
2. 模型层中 `tp_size>1` 时触发 all-reduce/all-gather：
   - `RowParallelLinear.forward()` -> `tensor_model_parallel_all_reduce`  
     代码：`vllm/model_executor/layers/linear.py:1442-1463`
   - `ColumnParallelLinear.forward()` -> `tensor_model_parallel_all_gather`  
     代码：`vllm/model_executor/layers/linear.py:595-607`
3. TP 通信最终走 `GroupCoordinator` 的设备通信器（CPU 下为 `CpuCommunicator`）  
   代码：`vllm/distributed/parallel_state.py:356-367, 478-526`
4. `CpuCommunicator` 在满足条件时使用共享内存 collective（否则走 torch.distributed）  
   代码：`vllm/distributed/device_communicators/cpu_communicator.py:31-40, 158-213`

---

## 4. 根因闭环（事实 -> 机制 -> 推断）

### 4.1 为什么 `NPS2_TP4` 比 `NPS1_TP2` 更好

#### 事实

- `Output tok/s`：`197.5 -> 209.4`
- `Total Mem Bw`：`271 -> 302 GB/s`
- `Frontend_Bound`：`22.68% -> 18.48%`
- `CPI` 基本持平（`1.73 -> 1.72`）

#### 机制解释

- 从代码看，`NPS2_TP4` 意味着更多 TP worker 并行（4 个进程）且每个 worker 在独立 NUMA node 绑定计算线程与内存策略（`CPUWorker.init_device()` + `init_cpu_threads_env()`）。
- `TP=4` 虽然引入更多 collective，但在该点位上换来了更高有效内存带宽和更低前端阻塞，整体收益为正。

#### 推断

- 当前平台上，`NPS2 + TP4` 是“通信成本可控、内存子系统利用率更高”的甜点区间。

### 4.2 为什么 `NPS4_TP8` 退化

#### 事实

- 吞吐下降：`209.4 -> 153.2 tok/s`（`-26.8%`）
- 计算效率恶化：`IPC 0.58 -> 0.42`，`CPI 1.72 -> 2.37`，`GIPS 371 -> 273`
- 访存与局部性恶化：
  - `Total Mem Bw 302 -> 231 GB/s`
  - `Remote Inbound Read 4.32 -> 8.95 GB/s`
  - `Remote DRAM Reads % 0.12 -> 0.43`
  - `Ave L3 Miss Latency 388 -> 442 ns`
- 控制/调度开销迹象：`System time % 6.63 -> 8.73`

#### 机制解释

- `TP=8` 直接增加每轮 collective 参与 rank 数，放大跨进程同步成本（见 `linear.py` 的 all-reduce/all-gather 触发点 + `parallel_state.py` 的 group 通信）。
- `MultiprocExecutor.collective_rpc()` 每轮都要广播任务并等待目标 rank 返回，rank 数增大时“尾部慢 rank”更容易成为瓶颈。
- `auto` 绑核下每 rank 默认会预留 1 核：`NPS4_TP8` 的单 rank 可用 OMP 线程数进一步压缩（推断见 4.3），使 compute/comm overlap 能力下降。
- 远端访存与 L3 miss 延迟的系统性上升表明 NUMA 局部性被破坏或被通信/辅助线程访问模式抵消，最终拖慢 token 迭代。

#### 推断

- 主因是 `TP8` 下通信/同步开销放大叠加 NUMA 局部性恶化，导致“理论并行度增加但有效吞吐下降”。

### 4.3 线程规模推导（推断，待运行态日志确认）

基于 `CPUWorker._get_autobind_cpu_ids()` 规则（x86 每物理核选 1 个逻辑 CPU，`world_size>1` 默认预留 1 核）：

- `NPS1_TP2`：约 `127` OMP 线程 / rank
- `NPS2_TP4`：约 `63` OMP 线程 / rank
- `NPS4_TP8`：约 `31` OMP 线程 / rank

总 OMP 线程数近似不变，但单 rank 粒度变小，叠加 TP 通信频次/同步链路增加，更容易进入“通信主导”区间。

---

## 5. 为什么未复现“`NPS4_TP8` 更优”

当前最可能的差异源（按优先级）：

1. `dtype` 口径不同：当前主结果是 `bfloat16`，而外部结论提到 `float16`
2. 运行时绑核/NUMA策略差异：`VLLM_CPU_OMP_THREADS_BIND`、`VLLM_CPU_NUM_OF_RESERVED_CPU`、是否启用 NUMA
3. 不同批次数据混用风险：尤其是 roofline 的 `NPS2` 两批次差异明显
4. worker 非 OMP 线程调度噪声：现有亲和性采样（`test_results/vllm_worker_thread_affinity_20260204_211611.txt`）显示仅部分线程是单核 pin，TP 增大时此类噪声可能放大

---

## 6. 最小验证实验（命令级）

### 6.1 `bf16` vs `fp16` 对照（优先）

在 `NPS1_TP2 / NPS2_TP4 / NPS4_TP8` 三组下，保持其余参数不变，仅替换：

- `--dtype bfloat16`
- `--dtype float16`

并固定压测点 `parallel=16 --number 16`，同步采集 `benchmark_summary.json + report-cumulative.csv`。

### 6.2 显式绑核，隔离 auto 策略影响

示例（按实际拓扑替换）：

```bash
export VLLM_CPU_OMP_THREADS_BIND="0-30|32-62|128-158|160-190|256-286|288-318|384-414|416-446"
export VLLM_CPU_NUM_OF_RESERVED_CPU=0
```

对比 `auto` 与显式绑核在 `NPS4_TP8` 下的 `Remote DRAM Reads %`、`Ave L3 Miss Latency`、`Output tok/s`。

### 6.3 保持 NPS 不变，单独扫 TP

在同一 NPS 下做 TP sweep（例如 NPS4: TP2/4/8），观察通信扩展性拐点；优先看：

- `CPI`、`System time %`、`Remote Inbound Read`
- `Output tok/s` 与 `TPOT`

---

## 7. 最终结论

### 事实结论

- 当前实验数据已形成闭环证据：`NPS2_TP4` 最优，`NPS4_TP8` 明显退化。

### 机制结论

- 退化不是单一“算力不足”，而是 `TP8` 下跨进程 collective + 同步等待 + NUMA 远端访存升高的叠加结果。

### 待验证结论

- `float16` 是否能改变结论方向，当前仍需严格 A/B 复验；在完成 dtype 对照前，不建议把“`NPS4_TP8` 必然更优”作为既定结论。
