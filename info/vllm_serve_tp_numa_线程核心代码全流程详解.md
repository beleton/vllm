# vLLM `serve` 代码全流程详解（TP / NUMA / 线程 / 核心）

本文面向你当前这条命令，按源码逐层展开，从 CLI 到 EngineCore/Worker，并重点解释 `tensor parallel (TP)`、`NUMA`、线程和核心绑定逻辑。

```bash
vllm serve /models/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name DeepSeek-R1-Distill-Llama-8B \
  --port 8122 \
  --dtype bfloat16 \
  --block-size 128 \
  --max-num-seqs 64 \
  --max_model_len 8192 \
  --tensor-parallel-size 8
```

---

## 0. 先说结论（针对这条命令）

1. `--tensor-parallel-size 8` 会把 `ParallelConfig.tensor_parallel_size` 设为 `8`，最终 `world_size = pp * tp * pcp = 1 * 8 * 1 = 8`（`vllm/config/parallel.py:551-555`）。
2. 在 `mp` 执行器路径下，会启动 `8` 个 Worker 进程（`vllm/v1/executor/multiproc_executor.py:143-159`）。
3. TP 组只有 1 组：`[0,1,2,3,4,5,6,7]`（`vllm/distributed/parallel_state.py:1332-1354`）。
4. 只有 rank 0 会被标记为 driver worker：`rank % tp_size == 0`（`vllm/v1/executor/multiproc_executor.py:228-229`）。
5. 结果回传 `output_rank` 也是 0（`vllm/v1/executor/multiproc_executor.py:436-450`）。
6. NUMA/绑核逻辑是否执行，取决于平台分支：
   - CPU backend（你当前 NPS/TP 实验语境大概率是这个）：会执行 `CPUWorker.init_device()` 的自动绑核与 NUMA 内存策略（`vllm/v1/worker/cpu_worker.py:54-187`）。
   - CUDA backend：不会走 `CPUWorker` 的 NUMA 绑核路径，会走 `GPUWorker` 初始化分支（`vllm/v1/worker/gpu_worker.py:175-241`）。

---

## 1. 参数是怎么“落地”到配置对象的

### 1.1 CLI 参数定义与兼容写法

- 参数注册在 `EngineArgs.add_cli_args()`（`vllm/engine/arg_utils.py:619-1202`）。
- 你的 `--max_model_len`（下划线）会被 parser 自动归一化为 `--max-model-len`（中划线），逻辑在 `FlexibleArgumentParser.parse_args()`（`vllm/utils/argparse_utils.py:226-246`）。

### 1.2 从命令行到 `EngineArgs`

- `vllm` 入口：`main()`（`vllm/entrypoints/cli/main.py:16-74`）。
- `serve` 子命令：`ServeSubcommand.cmd()`（`vllm/entrypoints/cli/serve.py:48-112`）。
- 进入 API server 后会构造：`engine_args = AsyncEngineArgs.from_cli_args(args)`（`vllm/entrypoints/openai/api_server.py:139`）。

### 1.3 从 `EngineArgs` 到 `VllmConfig`

- `create_engine_config()`（`vllm/engine/arg_utils.py:1344-1775`）把参数写入：
  - `--tensor-parallel-size 8` -> `ParallelConfig.tensor_parallel_size`（`arg_utils.py:1591-1593`）。
  - `--max_model_len 8192` -> `ModelConfig.max_model_len`（`arg_utils.py:1246`）。
  - `--max-num-seqs 64` -> `SchedulerConfig.max_num_seqs`（`arg_utils.py:1639-1641`）。
  - `--block-size 128` -> `CacheConfig.block_size`（`arg_utils.py:1407-1410`）。
  - `--dtype bfloat16` -> `ModelConfig.dtype`（`arg_utils.py:1239`）。

### 1.4 `VllmConfig.__post_init__` 的关键作用

- `VllmConfig.__post_init__()` 最后会调用 `current_platform.check_and_update_config(self)`（`vllm/config/vllm.py:537-543, 825`）。
- 这一步决定了：
  - 默认 worker 类（CPU/GPU）。
  - 一些平台专属环境变量和调优（比如 CPU 的 OMP/NUMA 相关策略）。

### 1.5 这些参数最终影响的“运行时位置”

1. `--tensor-parallel-size 8`
   - 参与 `world_size` 计算（`vllm/config/parallel.py:551-555`）。
   - 影响 worker 数、TP 组划分、driver/output rank（`vllm/v1/executor/multiproc_executor.py:107-113, 228-229, 436-450`）。
2. `--max_model_len 8192`
   - 写入 `ModelConfig.max_model_len`（`vllm/engine/arg_utils.py:1246`）。
   - 参与 scheduler 参数校验（`vllm/config/scheduler.py:244-267`）。
   - 请求阶段也用于计算默认 `max_tokens`（`vllm/entrypoints/openai/chat_completion/serving.py:381-386`）。
3. `--max-num-seqs 64`
   - 写入 `SchedulerConfig.max_num_seqs`（`vllm/engine/arg_utils.py:1639-1641`）。
   - 与 `max_num_batched_tokens`、`max_model_len` 共同约束调度容量（`vllm/config/scheduler.py:260-265`）。
4. `--block-size 128`
   - 写入 `CacheConfig.block_size`（`vllm/engine/arg_utils.py:1407-1410`）。
   - 构造 scheduler 时会乘上 DCP/PCP，变成 `scheduler_block_size`（`vllm/v1/engine/core.py:132-136`）。
   - CPU backend 对 block size 的偏好是 32 的倍数（`vllm/platforms/cpu.py:190-194`）。
5. `--dtype bfloat16`
   - 写入 `ModelConfig.dtype`（`vllm/engine/arg_utils.py:1239`）。
   - GPU worker 初始化时会检查平台是否支持该 dtype（`vllm/v1/worker/gpu_worker.py:212`）。
   - CPU 支持 dtype 集合由 `CpuPlatform.supported_dtypes` 决定（`vllm/platforms/cpu.py:81-122`）。

---

## 2. 从 `vllm serve` 到模型 ready 的完整调用链

## 2.0 多进程启动方式（`spawn`）是怎么定的

- CLI 启动时会执行 `cli_env_setup()`（`vllm/entrypoints/cli/main.py:33`）。
- 若外部未设置 `VLLM_WORKER_MULTIPROC_METHOD`，默认设为 `spawn`（`vllm/entrypoints/utils.py:164-182`）。
- 获取 multiprocessing context 时还会二次检查 `_maybe_force_spawn()`，在 Ray/CUDA 已初始化/WSL 等条件下强制改成 spawn（`vllm/utils/system_utils.py:114-160`）。

## 2.1 CLI 到 API server

1. `vllm/entrypoints/cli/main.py:16-74`  
   解析子命令并调用 `serve.cmd`。
2. `vllm/entrypoints/cli/serve.py:48-112`  
   默认 `api_server_count=1`，执行 `uvloop.run(run_server(args))`。
3. `vllm/entrypoints/openai/api_server.py:912-920`  
   `run_server()` 先 `setup_server()` 绑定端口，再进入 `run_server_worker()`。

## 2.2 API server 构建异步引擎客户端

4. `build_async_engine_client()`（`api_server.py:121-153`）  
   先 `AsyncEngineArgs.from_cli_args`，再进入 `build_async_engine_client_from_engine_args()`。
5. `build_async_engine_client_from_engine_args()`（`api_server.py:157-207`）  
   调 `create_engine_config()`，然后 `AsyncLLM.from_vllm_config(...)`（`api_server.py:188-197`）。
6. `AsyncLLM.from_vllm_config()`（`vllm/v1/engine/async_llm.py:214-240`）  
   选择执行器类 `Executor.get_class(vllm_config)`（`async_llm.py:230`）。

## 2.3 进入多进程引擎核心（EngineCore）

7. `EngineCoreClient.make_async_mp_client()`（`vllm/v1/engine/core_client.py:99-123`）  
   DP=1 时返回 `AsyncMPClient`。
8. `MPClient.__init__()`（`core_client.py:444-552`）  
   在 `launch_core_engines(...)` 上下文中拉起 EngineCore 进程。
9. `launch_core_engines()`（`vllm/v1/engine/utils.py:785-943`）  
   创建 ZMQ 地址、启动本地 `EngineCoreProc`，并等待握手 `wait_for_engine_startup()`（`utils.py:945-1098`）。

## 2.4 EngineCore 进程初始化

10. `EngineCoreProc.run_engine_core()`（`vllm/v1/engine/core.py:880-954`）  
    设置进程标题、日志上下文，创建 `EngineCoreProc`。
11. `EngineCoreProc.__init__()`（`core.py:641-736`）  
    完成 handshake（HELLO/READY），并创建两个 IO 线程：
    - input_thread：`process_input_sockets`（`core.py:705-716`）
    - output_thread：`process_output_sockets`（`core.py:717-727`）
12. `EngineCore.__init__()`（`core.py:78-219`）  
    关键动作：
    - `self.model_executor = executor_class(vllm_config)`（`core.py:105`）
    - KV cache profiling + 初始化（`core.py:112-119`）
    - 初始化 scheduler（`core.py:123-145`）

## 2.5 Executor 选择与 Worker 拉起

13. `Executor.get_class()`（`vllm/v1/executor/abstract.py:46-85`）  
    `distributed_executor_backend == "mp"` 时选择 `MultiprocExecutor`。
14. `MultiprocExecutor._init_executor()`（`vllm/v1/executor/multiproc_executor.py:99-210`）  
    - 计算并校验 `world_size`（`107-113`）
    - `set_multiprocessing_worker_envs()`（`115-117`）
    - 创建 distributed init method（`119-121`）
    - spawn worker 进程（`146-159`）
    - 等待 worker READY（`165`）
    - 计算 `output_rank`（`210`）

## 2.6 Worker 进程初始化

15. `WorkerProc.worker_main()`（`multiproc_executor.py:694-790`）  
    子进程内创建 `WorkerProc(...)`。
16. `WorkerProc.__init__()`（`multiproc_executor.py:530-583`）  
    - `wrapper.init_worker(all_kwargs)`（`554`）
    - `self.worker.init_device()`（`569`）
    - 初始化消息队列（`577`）
    - `self.worker.load_model()`（`578`）

## 2.7 请求运行时链路（`/v1/chat/completions`）

1. 路由入口：`create_chat_completion()`（`vllm/entrypoints/openai/chat_completion/api_router.py:34-73`）。
2. 进入 `OpenAIServingChat.create_chat_completion()`（`vllm/entrypoints/openai/chat_completion/serving.py:328-483`）。
3. 通过 `self.engine_client.generate(...)` 把请求交给 AsyncLLM/EngineCore（`serving.py:436-446`）。
4. `AsyncLLM._add_request()` -> `await self.engine_core.add_request_async(request)`（`vllm/v1/engine/async_llm.py:404-417`）。
5. `AsyncMPClient.add_request_async()` 通过 ZMQ 发送 ADD 请求（`vllm/v1/engine/core_client.py:954-957`）。
6. EngineCore 主循环 `run_busy_loop()` 收到请求并调度（`vllm/v1/engine/core.py:958-999`）。
7. EngineCore 调 executor 的 `execute_model()`，走 `collective_rpc` 广播到全部 worker（`vllm/v1/executor/multiproc_executor.py:269-279, 302-340`）。
8. 每个 worker 在 `worker_busy_loop()` 里执行方法（通常是 `execute_model`），仅 `output_rank` 回传主输出（`multiproc_executor.py:834-860`）。

---

## 3. 本命令下 TP 拓扑与 rank 推导（具体数字）

你的参数未设置 `pp/dp/pcp/dcp`，默认都为 1。

1. `tp=8`
2. `pp=1`
3. `pcp=1`
4. `dp=1`

于是：

- `world_size = pp * tp * pcp = 8`（`vllm/config/parallel.py:551-555`）
- `local_world_size = world_size / nnodes_within_dp`，单机时通常也是 8（`parallel.py:469-471`）

### 3.1 Worker 数量

- `for local_rank in range(self.local_world_size)` 会 spawn 8 个 worker（`multiproc_executor.py:146-159`）。
- rank 通常是 `0..7`（`global_rank = global_start_rank + local_rank`，`143-148`）。

### 3.2 driver worker 与 output rank

- driver worker 判断：`rank % tp_size == 0`（`multiproc_executor.py:228-229`），所以只有 rank 0。
- `output_rank = world_size - tp_size * pcp_size = 8 - 8*1 = 0`（`multiproc_executor.py:436-450`）。

### 3.3 进程名中的 TP rank

- worker 初始化后会调用 `setup_proc_title_and_log_prefix()`，拼出 `Worker_TP{tp_rank}` 等标识（`multiproc_executor.py:862-889`）。

---

## 4. TP 组是如何构建出来的

核心在 `initialize_model_parallel()`（`vllm/distributed/parallel_state.py:1280-1435`）。

### 4.1 先 reshape 全部 rank

```python
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_parallel_size,
    prefill_context_model_parallel_size, tensor_model_parallel_size
)
```

对应你的配置是：

- `all_ranks = torch.arange(8).reshape(-1, 1, 1, 1, 8)`
- 形状是 `[1,1,1,1,8]`

### 4.2 TP 组

- `group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)`（`parallel_state.py:1343-1344`）。
- 结果只有一个 TP 组：`[[0,1,2,3,4,5,6,7]]`。
- `_TP = init_model_parallel_group(...)`（`1347-1353`）。

### 4.3 其他组

- DCP（`1362-1370`）：`dcp=1` 时每组 1 个 rank。
- PCP（`1372-1382`）：`pcp=1` 时每组 1 个 rank。
- PP（`1387-1393`）：`pp=1` 时每组 1 个 rank。
- DP（`1397-1401`）：`dp=1` 时每组 1 个 rank。

所以此命令下，真正形成“多 rank 协同”的是 TP。

---

## 5. TP 如何影响权重加载与执行

## 5.1 权重按 TP rank 切片加载

参数对象初始化时记录 `tp_rank`、`tp_size`（`vllm/model_executor/parameter.py:65-67`）。

- 列并行权重：`load_column_parallel_weight` 按输出维切片（`parameter.py:148-152`）。
- 行并行权重：`load_row_parallel_weight` 按输入维切片（`parameter.py:220-224`）。
- 通用 loader 也同样按 `tp_rank * shard_size` 切（`weight_utils.py:1014-1044`）。

这意味着 TP=8 时，很多大矩阵的参数每个 rank 只加载 1/8 分片（具体是否切分取决于层类型与参数元信息）。

## 5.2 推理时 TP 通信点

在典型线性层里：

- ColumnParallelLinear：需要时做 `all_gather`（`vllm/model_executor/layers/linear.py:605-607`）。
- RowParallelLinear：需要时做 `all_reduce`（`linear.py:1461-1463`）。

这两类 collective 通过 TP group 执行，TP 组规模就是 8。

## 5.3 Executor 级别的“谁回传结果”

`MultiprocExecutor` 每轮把调度输出广播给所有 worker（`multiproc_executor.py:336`），
但只让 `output_rank` 回主结果（`339-340`，`855-860`），你这条命令里 `output_rank=0`。

---

## 6. NUMA / 线程 / 核心：真正执行逻辑

本节分 CPU backend 与 GPU backend。你当前实验语境（NPS + 双路 EPYC）通常是 CPU backend，NUMA 逻辑重点在 CPU 分支。

## 6.1 平台分支如何决定

- `VllmConfig.__post_init__` 调 `current_platform.check_and_update_config(self)`（`vllm/config/vllm.py:825`）。
- `cpu_platform_plugin()` 触发条件包括 CPU 版本构建（`vllm/platforms/__init__.py:158-179`）。
- 默认 worker：
  - CPU：`vllm.v1.worker.cpu_worker.CPUWorker`（`vllm/platforms/cpu.py:230-231`）
  - CUDA：`vllm.v1.worker.gpu_worker.Worker`（`vllm/platforms/cuda.py:156-157`）

## 6.2 线程数默认值与 OMP 控制

### 6.2.1 MultiprocExecutor 的兜底降并发

- `set_multiprocessing_worker_envs()`（`multiproc_executor.py:892-918`）会在 **未设置** `OMP_NUM_THREADS` 时，把 torch 线程数降到 1，避免多进程 + 多线程严重抢核。

### 6.2.2 CPU 平台会提前设置 `OMP_NUM_THREADS`

- CPU 分支中，若 `VLLM_CPU_OMP_THREADS_BIND != "nobind"`，会设置 `OMP_NUM_THREADS = torch.get_num_threads()`（`vllm/platforms/cpu.py:282-285`）。
- 所以在 CPU backend 下，`MultiprocExecutor` 的“降到 1”通常不会触发（因为 `OMP_NUM_THREADS` 已存在）。

### 6.2.3 前端输入处理线程

- `InputProcessor.process_inputs()` 里会读取 `OMP_NUM_THREADS`，并通过 `set_default_torch_num_threads(num_threads)` 包裹预处理（`vllm/v1/engine/input_processor.py:537-546`，`vllm/utils/torch_utils.py:107-113`）。
- 这里影响的是前端输入处理（tokenize/preprocess）线程数，不是 worker 模型计算线程池本体。

## 6.3 CPU backend 的自动绑核与 NUMA 绑定（重点）

入口：`CPUWorker.init_device()`（`vllm/v1/worker/cpu_worker.py:54-107`）。

### 6.3.1 `VLLM_CPU_OMP_THREADS_BIND` 三种模式

1. `auto`（默认）：按架构 + NUMA 自动选核心（`cpu_worker.py:57-73`）。
2. `nobind`：不绑核（`74-75`）。
3. 显式列表：如 `"0-31|32-63|..."`，按 rank 取对应片段（`77-85`）。

### 6.3.2 `auto` 模式的核心选择

`_get_autobind_cpu_ids()`（`cpu_worker.py:126-187`）逻辑：

1. 读 `lscpu -J -e=CPU,CORE,NODE`，并叠加 `sched_getaffinity` 过滤（`vllm/platforms/cpu.py:358-383`）。
2. 得到允许 NUMA node 列表，若设置了 `CPU_VISIBLE_MEMORY_NODES` 再过滤（`cpu.py:390-395`）。
3. 断言允许 NUMA 节点数至少覆盖 `world_size`（`cpu_worker.py:143-148`）。
4. 对每个 worker，用 `selected_numa_node = allowed_numa_nodes[local_rank]`（`151`）。
5. 在该 node 内按 physical core 分组，x86 默认每个物理核只取一个 SMT 线程（`64-68`, `156-166`）。
6. 默认会预留核心：
   - 若 `VLLM_CPU_NUM_OF_RESERVED_CPU` 未设置，且 world_size>1，则默认预留 1 core（`169-176`）。
   - 去掉尾部保留核（`180-182`）。

### 6.3.3 C++ 层真正做了什么

调用点：`torch.ops._C_utils.init_cpu_threads_env(cpu_ids)`（`cpu_worker.py:87`）。  
实现：`csrc/cpu/utils.cpp:25-165`。

核心动作：

1. 解析 CPU 列表得到 `omp_cpu_ids`（`26-45`）。
2. 反推 NUMA node 集合（`47-55`）。
3. 尝试页迁移到目标 node（`72-78`）。
4. 设置 NUMA 内存策略：
   - 单 node：`numa_set_membind`（`97-109`）
   - 多 node：`numa_set_interleave_mask`（`82-95`）
   - `numa_set_strict(1)`（`112`）
5. 设置线程数量：
   - `omp_set_num_threads(N)`（`125`）
   - `torch::set_num_threads(N)`（`126`）
6. 对每个 OMP 线程调用 `sched_setaffinity` 绑到指定 CPU（`135-149`）。
7. 返回线程到 core 的映射日志（`155-164`）。

这就是“线程和核心绑定”真正发生的位置。

## 6.4 GPU backend 下和 NUMA 相关的现实情况

若实际平台是 CUDA：

- 会走 `GPUWorker.init_device()`（`vllm/v1/worker/gpu_worker.py:175-241`）。
- 主要是设置 CUDA device、初始化 distributed、建立 NCCL/Gloo 组等。
- 不会执行 `CPUWorker` 的 `init_cpu_threads_env` NUMA/membind/sched_setaffinity 逻辑。

---

## 7. 这条命令对应的线程/进程拓扑（运行时）

## 7.1 进程层

1. APIServer 进程（`run_server_worker`）。
2. EngineCore 进程（`EngineCoreProc.run_engine_core`）。
3. 8 个 Worker 进程（`VllmWorker-0` 到 `VllmWorker-7`）。

## 7.2 EngineCore 内线程

- 主线程：`run_busy_loop()`（`core.py:958-967`）。
- 输入线程：`process_input_sockets`（`core.py:705-716`）。
- 输出线程：`process_output_sockets`（`core.py:717-727`）。

## 7.3 Worker 内线程

- 主线程：`worker_busy_loop()`。
- `WorkerDeathMonitor`（`multiproc_executor.py:724-739`）。
- 若 async scheduling 开启，还会有 `WorkerAsyncOutputCopy`（`559-567`）；但 CPU 平台强制 `async_scheduling=False`（`vllm/platforms/cpu.py:196-199`），所以 CPU 下通常没有这条线程。
- OpenMP 线程池：数量由 OMP/绑定列表确定。

---

## 8. 本命令“会触发/不会触发”的关键分支表

| 逻辑 | 是否触发 | 条件与说明 |
| --- | --- | --- |
| `serve` 单 API server 路径 | 会 | `api_server_count` 默认 1（`serve.py:86-112`） |
| `AsyncMPClient` | 会 | DP=1（`core_client.py:116-123`） |
| `MultiprocExecutor` | 会 | `world_size=8 > 1` 默认 backend 为 `mp`（`parallel.py:602-644`） |
| 启动 8 个 worker | 会 | `local_world_size=8`（`multiproc_executor.py:146-159`） |
| TP group 创建 | 会 | `initialize_model_parallel()`（`parallel_state.py:1280-1354`） |
| DP coordinator | 不会 | 需要在线 DP 模式且 rank0 等条件（`engine/utils.py:830-853`） |
| Ray executor | 不会 | backend 不是 ray |
| CPU NUMA 自动绑核 | 取决于平台 | 仅 CPU backend 执行（`cpu_worker.py:54-187`） |
| CUDA device 初始化 | 取决于平台 | 仅 CUDA backend 执行（`gpu_worker.py:175-241`） |

---

## 9. 针对 TP=8 + NPS 实验最容易忽略的点

1. `local_rank -> NUMA node` 是直接按索引映射（`allowed_numa_nodes[local_rank]`）。  
   如果允许 NUMA node 不足 8，会直接断言失败（`cpu_worker.py:143-148`）。
2. 默认“预留 1 核”会减少每个 rank 的 OMP 线程数（`cpu_worker.py:169-176`），对高 TP 时每 rank 算力有直接影响。
3. `OMP_NUM_THREADS` 有两层影响：
   - Worker 计算线程（CPU backend 绑核时会重设）。
   - 前端输入处理线程（`input_processor.py:537-546`）。
4. `output_rank` 固定回传策略让 rank0 更“忙”，这是 TP>1 下的控制平面热点之一（`multiproc_executor.py:336-340` + `436-450`）。

---

## 10. 快速源码索引（按主题）

### 10.1 启动主链路

- `vllm/entrypoints/cli/main.py:16-74`
- `vllm/entrypoints/cli/serve.py:48-112`
- `vllm/entrypoints/openai/api_server.py:121-207`
- `vllm/v1/engine/async_llm.py:214-240`
- `vllm/v1/engine/core_client.py:99-123, 444-552, 807-958`
- `vllm/v1/engine/utils.py:785-943, 945-1098`
- `vllm/v1/engine/core.py:636-954`

### 10.2 配置与参数落地

- `vllm/engine/arg_utils.py:619-1202, 1344-1775`
- `vllm/config/vllm.py:537-543, 825`
- `vllm/config/parallel.py:540-555`
- `vllm/config/model.py:1064-1075`
- `vllm/config/scheduler.py:211-265`
- `vllm/config/cache.py:41-50, 224-243`

### 10.3 TP / 分布式组

- `vllm/v1/executor/multiproc_executor.py:99-229, 336-340, 436-450`
- `vllm/distributed/parallel_state.py:1162-1278, 1280-1437`
- `vllm/v1/worker/gpu_worker.py:936-963`
- `vllm/model_executor/parameter.py:65-67, 148-152, 220-224`
- `vllm/model_executor/layers/linear.py:605-607, 1461-1463`

### 10.4 NUMA / 线程 / 核心

- `vllm/platforms/cpu.py:179-189, 276-289, 358-397`
- `vllm/v1/worker/cpu_worker.py:54-107, 126-187`
- `csrc/cpu/utils.cpp:25-165`
- `vllm/v1/engine/input_processor.py:537-546`
- `vllm/v1/executor/multiproc_executor.py:892-918`
