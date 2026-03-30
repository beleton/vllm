# 01. vllm serve 启动到 server ready

> 更新时间：2026-03-17  
> 范围：默认讨论 `vllm serve`、单机、CPU backend、V1 engine。  
> 假设：未特别说明时，走 `ServeSubcommand.cmd -> run_server -> run_server_worker` 这条单 API server 路径。

## 结论

### 事实
- `vllm serve` 的主入口是 `vllm/entrypoints/cli/main.py::main`，真正进入 serve 子命令后落到 `vllm/entrypoints/cli/serve.py::ServeSubcommand.cmd`。
- 对单 API server 路径，核心调用链是：`cmd` -> `run_server` -> `run_server_worker` -> `build_async_engine_client` -> `AsyncLLM.from_vllm_config` -> `EngineCoreClient.make_async_mp_client` -> `launch_core_engines` -> `EngineCoreProc.run_engine_core` -> `EngineCore.__init__` -> `Executor/Worker` 初始化。
- 端口绑定发生在 `vllm/entrypoints/openai/api_server.py::setup_server`，早于 engine 初始化；因此“端口已占住”不等于“模型已 ready”。
- 从代码顺序看，`run_server_worker` 只有在 `build_async_engine_client(...)` 完成、`init_app_state(...)` 完成之后，才会调用 `serve_http(...)` 启动 FastAPI/uvicorn。也就是说，server ready 之前，EngineCore 与 worker 已经完成初始化。
- `api_server_count > 1` 时，路径会切到 `run_multi_api_server`；`--headless` 时走 `run_headless`。本文件只把它们当分叉点，不展开内部负载均衡细节。

### 推断
- 如果启动日志里 API server 已开始监听，但第一批请求仍然长时间阻塞，问题更可能位于 EngineCore/worker 初始化尾部或模型 load/warmup，而不是 CLI 层。

### 建议
- 看启动慢问题时，先区分三个时间点：端口绑定、EngineCore READY、`serve_http` 启动。
- 真正要做 CPU 启动性能归因时，应把 `run_server_worker`、`EngineCoreProc.__init__`、`WorkerProc.__init__`、`CPUWorker.init_device` 分段打点。

## 证据

### 主调用链

| 阶段 | 源码路径 | 关键动作 |
| --- | --- | --- |
| CLI 总入口 | `vllm/entrypoints/cli/main.py` | 解析子命令，把 `serve` 分发给 `ServeSubcommand.cmd` |
| serve 分叉 | `vllm/entrypoints/cli/serve.py` | 在 `headless / multi-api-server / single-api-server` 间选择 |
| 提前绑端口 | `vllm/entrypoints/openai/api_server.py::setup_server` | 先创建/绑定 socket，避免初始化期间端口竞争 |
| API worker 启动 | `vllm/entrypoints/openai/api_server.py::run_server_worker` | 建 `engine_client`、建 app、初始化 app state、最后 `serve_http` |
| Engine client | `vllm/entrypoints/openai/api_server.py::build_async_engine_client_from_engine_args` | 先 `create_engine_config`，再构造 `AsyncLLM` |
| AsyncLLM | `vllm/v1/engine/async_llm.py::AsyncLLM.__init__` | 初始化 InputProcessor / OutputProcessor / EngineCoreClient |
| EngineCore 进程 | `vllm/v1/engine/utils.py::launch_core_engines` | 启动 `CoreEngineProcManager`，等待 HELLO/READY 握手 |
| EngineCore 主体 | `vllm/v1/engine/core.py::EngineCoreProc.run_engine_core` | 创建 `EngineCoreProc`，进入 busy loop |
| Executor/worker | `vllm/v1/engine/core.py::EngineCore.__init__` | 选 executor，初始化 KV cache，建立 scheduler |
| 多 worker | `vllm/v1/executor/multiproc_executor.py::MultiprocExecutor._init_executor` | 拉起每个 worker 进程并等待 READY |
| CPU worker | `vllm/v1/worker/cpu_worker.py::CPUWorker.init_device` | 绑核、初始化 distributed、创建 `CPUModelRunner` |
| 模型 load | `vllm/v1/executor/multiproc_executor.py::WorkerProc.__init__` -> `worker.load_model()` | 完成每个 worker 的模型实例化 |

### 关键时序

```text
vllm/entrypoints/cli/main.py::main
  -> vllm/entrypoints/cli/serve.py::ServeSubcommand.cmd
  -> vllm/entrypoints/openai/api_server.py::run_server
  -> vllm/entrypoints/openai/api_server.py::setup_server      # 先绑端口
  -> vllm/entrypoints/openai/api_server.py::run_server_worker
  -> vllm/entrypoints/openai/api_server.py::build_async_engine_client
  -> vllm/entrypoints/openai/api_server.py::build_async_engine_client_from_engine_args
  -> vllm/v1/engine/async_llm.py::AsyncLLM.from_vllm_config
  -> vllm/v1/engine/async_llm.py::AsyncLLM.__init__
  -> vllm/v1/engine/core_client.py::EngineCoreClient.make_async_mp_client
  -> vllm/v1/engine/utils.py::launch_core_engines
  -> vllm/v1/engine/core.py::EngineCoreProc.run_engine_core
  -> vllm/v1/engine/core.py::EngineCoreProc.__init__
  -> vllm/v1/engine/core.py::EngineCore.__init__
  -> vllm/v1/executor/abstract.py::Executor.get_class
  -> vllm/v1/executor/multiproc_executor.py::MultiprocExecutor._init_executor
  -> vllm/v1/executor/multiproc_executor.py::WorkerProc.make_worker_process
  -> vllm/v1/executor/multiproc_executor.py::WorkerProc.__init__
  -> vllm/v1/worker/cpu_worker.py::CPUWorker.init_device
  -> vllm/v1/worker/cpu_model_runner.py::CPUModelRunner.load_model
  -> worker READY / engine READY
  -> vllm/entrypoints/openai/api_server.py::init_app_state
  -> vllm/entrypoints/launcher.py::serve_http
```

### READY 握手证据
- `vllm/v1/engine/core.py::EngineCoreProc._perform_handshake` 先发 `HELLO`，收到初始化元数据后继续初始化；退出上下文前才回发 `READY`。
- `vllm/v1/engine/utils.py::wait_for_engine_startup` 会等待全部 engine 从 `NEW -> CONNECTED -> READY`，否则不会把启动流程放过去。
- `vllm/entrypoints/openai/api_server.py::run_server_worker` 中，`serve_http(...)` 在 `async with build_async_engine_client(...)` 和 `await init_app_state(...)` 之后才执行。

## 机制解释

### 1. 为什么 `setup_server` 要先于 engine 初始化
- `setup_server` 明确先创建 socket，再做 engine 初始化，这是为避免初始化期端口竞争。
- 因此“端口可见”只是前端占坑成功，不代表 worker 已 load 模型。

### 2. `server ready` 前有哪些后台进程
- API server 进程：负责 HTTP、请求解析、`EngineClient`。
- EngineCore 进程：负责 scheduler、KV cache 配置、驱动 executor。
- Worker 进程：负责真正的 CPU 模型执行，数量通常是本地 `TP * PP`。

### 3. EngineCore 在 READY 前已经做了什么
- 完成与前端的 HELLO/初始化元数据握手。
- 创建 executor。
- 通过 executor 拉起 worker 进程。
- 每个 worker 完成 `init_device`、distributed 初始化和 `load_model()`。
- 回传 `num_gpu_blocks`（CPU backend 实际对应 cache block 容量结果）等 READY 元数据。

### 4. 为什么这里对 CPU 性能分析很重要
- 启动期就已经决定了：worker 数量、rank 编号、绑核字符串、NUMA 策略、TP group、共享内存通信对象名。
- 如果这一步的 rank/绑核映射错了，后面所有 attention / linear / moe 的性能分析都会被污染。
