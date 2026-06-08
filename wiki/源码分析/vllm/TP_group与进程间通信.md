# TP group 与进程间通信

## 问题
- 当前这台 x86 机器上，`Qwen3-30B-A3B` 在 CPU backend、只开 `TP`、`DP=PP=PCP=1` 时，跨进程/跨线程通信实际走哪条路径。

## 结论
- 当前 `vllm-cpu` 环境下，`Qwen3-30B-A3B + TP=2` 的 TP 数据面通信实际走 `gloo`，不是 `csrc/cpu/shm.cpp` 的自定义 CPU SHM collective。
- 只开 TP 时，vLLM 会拉起 `TP` 个 worker 进程；相邻 global rank 直接组成一个 TP group。
- 控制面和数据面是两套通信：
  - 控制面：executor 通过 `MessageQueue` 把 RPC / `SchedulerOutput` 广播给 worker
  - 数据面：张量同步通过 TP group 的 collective，入口是 `tensor_model_parallel_all_reduce` / `tensor_model_parallel_all_gather`
- 在 `Qwen3-30B-A3B` 里，只开 TP、不启 EP 时，常见 TP 通信落点主要是：
  - attention 的 `o_proj`
  - dense MLP 的 `down_proj`
  - sparse MoE block 末尾
- `qkv_proj` / `gate_up_proj` 这类 column-parallel 层默认不会在 forward 后立刻做 TP 通信；只有 `gather_output=True` 时才会 `all_gather`。
- 虽然所有 TP rank 都参与计算和 collective，但通常只有 `TP rank 0` 把结果回给上层 executor。

## 当前环境实际路径
- 直接日志：`test_results/qwen3_tp2_init_2026-03-18_2338.log`
- 当前环境应认定为：
  `GroupCoordinator -> CpuCommunicator -> torch.distributed(gloo)`
- `csrc/cpu/shm.cpp` 是源码里的可选优化路径，但当前环境未生效

## TP 进程组织
- `MultiprocExecutor` 按 `local_rank=0..local_world_size-1` 创建 worker，并给每个 worker 分配 `global_rank`
- `initialize_model_parallel()` 里，TP group 用 `all_ranks.view(-1, tensor_model_parallel_size)` 构造
- 只开 TP 时，不同 TP 的进程如何通信，本质上就是同一 TP group 内 rank 互相做 collective

## 控制面与数据面

### 控制面
- executor 调 `collective_rpc()` 时，会把 `(method, args, kwargs, output_rank)` 放进 `rpc_broadcast_mq`
- 每个 worker 在 `worker_busy_loop()` 里从这个 MQ 取任务并执行

### 数据面
- 张量同步通过 TP group collective
- 入口是：
  - `tensor_model_parallel_all_reduce`
  - `tensor_model_parallel_all_gather`

## `Qwen3-30B-A3B` 的常见通信落点

### attention
- `qkv_proj` 是 `QKVParallelLinear`
- `QKVParallelLinear` 继承自 `ColumnParallelLinear`
- `ColumnParallelLinear.forward()` 只有在 `gather_output=True` 时才 `all_gather`
- `o_proj` 是 `RowParallelLinear`
- `RowParallelLinear.forward()` 在 `reduce_results=True && tp_size > 1` 时执行 `tensor_model_parallel_all_reduce`

### dense MLP
- `gate_up_proj` 是 `MergedColumnParallelLinear`
- `down_proj` 是 `RowParallelLinear`
- dense MLP 的跨 TP 通信重点在 `down_proj` 末尾的 `all_reduce`

### sparse MoE block
- 只开 TP、不启 `enable_expert_parallel` 时，Qwen3 MoE 不走 EP token dispatch/combine 的 All2All 路径
- 此时 `ep_size=1`、`use_ep=False`
- 主要通信仍然是 TP all-reduce

## 分析
- 当前环境的跨 rank 通信粒度是“进程/rank”，不是“每个 OMP 线程都显式做 peer-to-peer 通信”。
- 每个 TP worker 内部的 OMP 线程主要负责本地计算；跨 rank 的同步点主要集中在 `RowParallelLinear.forward` 和 MoE block 末尾的 TP collective。
- `csrc/cpu/shm.cpp` 是源码里的可选优化路径，但当前环境未生效，不能当成当前实跑路径的主分析对象。

## 证据
- 直接日志：
  - `test_results/qwen3_tp2_init_2026-03-18_2338.log`
- 关键源码路径：
  - `vllm/v1/executor/multiproc_executor.py`
  - `vllm/distributed/parallel_state.py`
  - `vllm/distributed/communication_op.py`
  - `vllm/distributed/device_communicators/cpu_communicator.py`
  - `vllm/model_executor/models/qwen3_moe.py`
  - `vllm/model_executor/layers/linear.py`
  - `vllm/model_executor/layers/fused_moe/layer.py`
  - `vllm/model_executor/layers/fused_moe/config.py`
  - `csrc/cpu/shm.cpp`

## 边界
- 本页只覆盖当前这台机器、CPU backend、只开 TP 的路径。
- `gloo` 内部具体 helper 线程和 socket/poll 机制不在当前 vLLM 源码主线里直接暴露，本页不展开。

## 后续验证点
- 若继续查 TP 通信瓶颈，优先对 `RowParallelLinear.forward` 和 MoE block 末尾的 `tensor_model_parallel_all_reduce` 抓火焰图或 uProf 调用栈。

