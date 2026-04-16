# vllm serve 启动到 server ready

## 问题
- `vllm serve` 从 CLI 进入后，端口绑定、EngineCore 初始化、worker 初始化、`server ready` 各发生在什么阶段。

## 结论
- `vllm serve` 的主入口是 `vllm/entrypoints/cli/main.py::main`，进入 `serve` 后落到 `vllm/entrypoints/cli/serve.py::ServeSubcommand.cmd`。
- 单 API server 路径的主链路是：
  `cmd -> run_server -> run_server_worker -> build_async_engine_client -> AsyncLLM.from_vllm_config -> EngineCoreClient.make_async_mp_client -> launch_core_engines -> EngineCoreProc.run_engine_core -> EngineCore.__init__ -> Executor/Worker 初始化`。
- 端口绑定发生在 `vllm/entrypoints/openai/api_server.py::setup_server`，早于 engine 初始化；因此“端口已占住”不等于“模型已 ready”。
- `run_server_worker` 只有在 `build_async_engine_client(...)` 和 `init_app_state(...)` 完成后，才会调用 `serve_http(...)` 启动 FastAPI/uvicorn。
- `server ready` 之前，EngineCore 与 worker 已完成初始化。

## 关键时序

```text
vllm/entrypoints/cli/main.py::main
  -> vllm/entrypoints/cli/serve.py::ServeSubcommand.cmd
  -> vllm/entrypoints/openai/api_server.py::run_server
  -> vllm/entrypoints/openai/api_server.py::setup_server
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

## 关键分叉点
- `api_server_count > 1` 时会切到 `run_multi_api_server`
- `--headless` 时走 `run_headless`
- 本页只覆盖单 API server、单机、CPU backend、V1 engine

## READY 握手
- `EngineCoreProc._perform_handshake` 会先发 `HELLO`
- 前端把初始化元数据发回后，EngineCore 才继续初始化
- `wait_for_engine_startup` 会等待 engine 进入 `READY`
- 因此 `serve_http(...)` 启动前，worker 已完成 `init_device`、distributed 初始化和 `load_model()`

## 分析
- 如果启动日志里 API server 已开始监听，但首批请求仍明显阻塞，更可能是 EngineCore、worker 初始化或模型加载尾部的问题，而不是 CLI 层问题。
- 对 CPU 启动性能归因，应至少拆开：
  - `run_server_worker`
  - `EngineCoreProc.__init__`
  - `WorkerProc.__init__`
  - `CPUWorker.init_device`

## 证据
- 关键源码路径：
  - `vllm/entrypoints/cli/main.py`
  - `vllm/entrypoints/cli/serve.py`
  - `vllm/entrypoints/openai/api_server.py`
  - `vllm/v1/engine/async_llm.py`
  - `vllm/v1/engine/core_client.py`
  - `vllm/v1/engine/utils.py`
  - `vllm/v1/engine/core.py`
  - `vllm/v1/executor/multiproc_executor.py`
  - `vllm/v1/worker/cpu_worker.py`
  - `vllm/v1/worker/cpu_model_runner.py`

## 边界
- 本页不展开 `run_multi_api_server` 的内部负载均衡细节。
- 本页也不展开 `--headless` 路径的内部实现。

## 后续验证点
- 若后续做启动期性能归因，优先在 `run_server_worker`、`EngineCoreProc.__init__`、`WorkerProc.__init__`、`CPUWorker.init_device` 加分段时间戳。

