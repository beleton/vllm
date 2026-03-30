# 04. TP group 与进程间通信

> 更新时间：2026-03-18 23:49 +0800  
> 范围：只看当前这台 x86 机器上，`Qwen3-30B-A3B` 在 CPU backend、只开 `TP`、`DP=PP=PCP=1` 时的跨进程/跨线程通信。  
> 当前机器：`2 x AMD EPYC 9745 128-Core Processor`  
> 当前环境说明：本文优先写“当前环境实际生效路径”；源码里存在但当前环境未启用的可选路径，会单独标明。  
> 直接证据日志：`test_results/qwen3_tp2_init_2026-03-18_2338.log`

## 结论

### 事实
- 当前这台机器、当前 `vllm-cpu` 环境下，`Qwen3-30B-A3B + TP=2` 的 **TP 数据面通信实际走 `gloo`**，不是 `csrc/cpu/shm.cpp` 那条自定义 CPU SHM collective。证据是：
  - `test_results/qwen3_tp2_init_2026-03-18_2338.log` 里 worker distributed 初始化明确打印 `backend=gloo`，见 `test_results/qwen3_tp2_init_2026-03-18_2338.log:148`。
  - `vllm/distributed/device_communicators/cpu_communicator.py` 只有在 `torch.ops._C.init_shm_manager` 可用时才切到 `_CPUSHMDistributed`，否则保留 `torch.distributed` 路径，见 `vllm/distributed/device_communicators/cpu_communicator.py:29-40`、`vllm/distributed/device_communicators/cpu_communicator.py:55-57`、`vllm/distributed/device_communicators/cpu_communicator.py:92-108`。
- 只开 TP 时，vLLM 会拉起 `TP` 个 worker 进程；相邻 global rank 直接组成一个 TP group。证据见 `vllm/v1/executor/multiproc_executor.py:143-158` 与 `vllm/distributed/parallel_state.py:1332-1353`。
- 当前 `TP=2` 启动日志里，两个 TP worker 分别绑在两组不同核心集合上，而不是“整机所有核心都一起参与”。证据见：
  - `test_results/qwen3_tp2_init_2026-03-18_2338.log:16`
  - `test_results/qwen3_tp2_init_2026-03-18_2338.log:17`
  - `test_results/qwen3_tp2_init_2026-03-18_2338.log:18-20`
  - `test_results/qwen3_tp2_init_2026-03-18_2338.log:148-151`
- 控制面和数据面是两套通信：
  - 控制面：executor 把 RPC / `SchedulerOutput` 广播给 worker，用 `MessageQueue`，见 `vllm/v1/executor/multiproc_executor.py:302-337`、`vllm/v1/executor/multiproc_executor.py:834-860`。
  - 数据面：张量同步通过 TP group 的 collective，入口是 `tensor_model_parallel_all_reduce` / `tensor_model_parallel_all_gather`，见 `vllm/distributed/communication_op.py:12-21`。
- 在 `Qwen3-30B-A3B` 里，只开 TP、不启 EP 时，真正常见的 TP 通信落点主要是：
  - attention 的 `o_proj`：`RowParallelLinear -> all_reduce`，见 `vllm/model_executor/models/qwen3_moe.py:300-306`、`vllm/model_executor/layers/linear.py:1461-1463`。
  - dense MLP 的 `down_proj`：`RowParallelLinear -> all_reduce`，见 `vllm/model_executor/models/qwen3_moe.py:94-108`、`vllm/model_executor/models/qwen3_moe.py:116-124`、`vllm/model_executor/layers/linear.py:1461-1463`。
  - sparse MoE block 末尾：`maybe_all_reduce_tensor_model_parallel -> all_reduce`，见 `vllm/model_executor/models/qwen3_moe.py:237-245`、`vllm/model_executor/layers/fused_moe/layer.py:1538-1545`。
- `qkv_proj` / `gate_up_proj` 这种 column-parallel 层默认不会在 forward 后立刻做 TP 通信；只有 `gather_output=True` 时才会 `all_gather`。证据见 `vllm/model_executor/models/qwen3_moe.py:290-298`、`vllm/model_executor/layers/linear.py:605-609`。
- 只开 TP、不启 `enable_expert_parallel` 时，Qwen3 MoE 不走 EP token dispatch/combine 的 All2All 路径；此时 `ep_size=1`、`use_ep=False`。证据见 `vllm/model_executor/layers/fused_moe/config.py:992-1018`。
- 虽然所有 TP rank 都参与计算和 collective，但通常只有 `TP rank 0` 把结果回给上层 executor。证据见 `vllm/v1/executor/multiproc_executor.py:228-229`、`vllm/v1/executor/multiproc_executor.py:436-450`、`vllm/v1/executor/multiproc_executor.py:855-860`。

### 推断
- 从 vLLM 视角，当前环境的 **跨 rank 通信粒度是“进程/rank”**，不是“每个 OMP 线程都显式做 peer-to-peer 通信”。OMP 线程主要是各 worker 进程内部的本地计算线程。
- `gloo` 内部是否再启 helper 线程、具体用哪些 socket/poll 机制，本文不展开；这不在当前 vLLM 源码主线里直接暴露。

### 建议
- 分析 `Remote DRAM Reads %`、`Ave L3 Miss Latency` 时，先把 `MessageQueue` 控制面、TP collective 数据面、worker 内部 OpenMP 访存三类路径分开。
- 如果要继续查 TP 通信瓶颈，优先对 `RowParallelLinear.forward` 和 MoE block 末尾的 `tensor_model_parallel_all_reduce` 抓火焰图或 uProf 调用栈。
- 只有在你重新编译出 `torch.ops._C.init_shm_manager/shm_allreduce` 后，`csrc/cpu/shm.cpp` 才值得作为“当前实跑路径”的主分析对象。

## 证据拆解

### 1. 当前环境实际走哪条通信路径

#### 事实
- 本次用于证据的复现实验时间是 `2026-03-18 23:38 +0800`。
- 命令对应日志保存在 `test_results/qwen3_tp2_init_2026-03-18_2338.log`。
- 日志里 `EngineCore` 明确显示 `tensor_parallel_size=2`，见 `test_results/qwen3_tp2_init_2026-03-18_2338.log:11`。
- worker distributed 初始化明确显示 `backend=gloo`，见 `test_results/qwen3_tp2_init_2026-03-18_2338.log:148`。

#### 机制
- `GroupCoordinator` 初始化时，无论 backend 是什么，都会先建一个 `device_group` 和一个 `cpu_group(gloo)`，见 `vllm/distributed/parallel_state.py:324-343`。
- 真正 `all_reduce/all_gather` 走哪条实现，由 `device_communicator` 决定，见 `vllm/distributed/parallel_state.py:356-367`、`vllm/distributed/parallel_state.py:478-526`。
- 对 CPU backend：
  - 默认实现是 `torch.distributed.*`
  - 只有在 x86/ARM 且 `torch.ops._C.init_shm_manager` 存在，同时 group name 以 `tp`/`pp` 开头时，才切换成 `_CPUSHMDistributed`，见 `vllm/distributed/device_communicators/cpu_communicator.py:31-40`

#### 结论
- 当前环境的 TP 数据面通信主路径应认定为：
  `GroupCoordinator -> CpuCommunicator -> torch.distributed(gloo)`
- `csrc/cpu/shm.cpp` 是源码里的可选优化路径，但当前环境未生效。

### 2. TP 进程是怎么组织起来的

#### 事实
- `MultiprocExecutor` 按 `local_rank=0..local_world_size-1` 创建 worker，并给每个 worker 分配 `global_rank`，见 `vllm/v1/executor/multiproc_executor.py:143-158`。
- `initialize_model_parallel()` 里，TP group 用 `all_ranks.view(-1, tensor_model_parallel_size)` 构造，见 `vllm/distributed/parallel_state.py:1332-1353`。

#### 示例
- 若 `DP=PP=PCP=1, TP=2`：TP group 就是 `[0, 1]`
- 若 `DP=PP=PCP=1, TP=4`：TP group 就是 `[0, 1, 2, 3]`

#### 结论
- 只开 TP 时，“不同 TP 的进程如何通信”本质上就是：**同一 TP group 内的 rank 互相做 collective**。

### 3. 控制面通信：谁发命令给谁

#### 事实
- executor 调 `collective_rpc()` 时，会把 `(method, args, kwargs, output_rank)` 放进 `rpc_broadcast_mq`，见 `vllm/v1/executor/multiproc_executor.py:302-337`。
- 每个 worker 在 `worker_busy_loop()` 里从这个 MQ 取任务并执行，见 `vllm/v1/executor/multiproc_executor.py:834-860`。
- `MessageQueue.create_from_process_group()` 建立 handle 时，会对 handle 做一次 `broadcast_object_list(...)`，见 `vllm/distributed/device_communicators/shm_broadcast.py:705-777`。

#### 结论
- 控制面不是 TP all-reduce。
- 它更像“1 个写者 -> 多个读者”的广播队列：executor 发，所有 TP worker 收。

### 4. 数据面通信：Qwen3-30B-A3B 在哪几层真正通信

#### 4.1 attention

##### 事实
- `Qwen3MoeAttention.qkv_proj` 是 `QKVParallelLinear`，见 `vllm/model_executor/models/qwen3_moe.py:290-298`。
- `QKVParallelLinear` 继承自 `ColumnParallelLinear`，而 `ColumnParallelLinear.forward()` 只有在 `gather_output=True` 时才 `all_gather`，见 `vllm/model_executor/layers/linear.py:605-609`。
- `Qwen3MoeAttention.o_proj` 是 `RowParallelLinear`，见 `vllm/model_executor/models/qwen3_moe.py:300-306`。
- `RowParallelLinear.forward()` 在 `reduce_results=True && tp_size > 1` 时执行 `tensor_model_parallel_all_reduce`，见 `vllm/model_executor/layers/linear.py:1461-1463`。

##### 结论
- attention 里通常是：
  - `qkv_proj`：各 rank 本地算自己的 shard
  - `o_proj`：各 rank 输出再做一次 TP `all_reduce`

#### 4.2 dense MLP

##### 事实
- `Qwen3MoeMLP.gate_up_proj` 是 `MergedColumnParallelLinear`，见 `vllm/model_executor/models/qwen3_moe.py:94-100`。
- `Qwen3MoeMLP.down_proj` 是 `RowParallelLinear`，见 `vllm/model_executor/models/qwen3_moe.py:101-108`。
- forward 里真正同步点在 `down_proj` 返回时，见 `vllm/model_executor/models/qwen3_moe.py:116-124` 与 `vllm/model_executor/layers/linear.py:1461-1463`。

##### 结论
- dense MLP 的跨 TP 通信重点不在上投影，而在 `down_proj` 末尾的 `all_reduce`。

#### 4.3 sparse MoE block

##### 事实
- `Qwen3MoeDecoderLayer` 不是每层都一定是 sparse MoE；是否走 sparse 取决于 `decoder_sparse_step` 和 `mlp_only_layers`，见 `vllm/model_executor/models/qwen3_moe.py:382-400`。
- 对 sparse MoE block，若 `tp_size > 1` 且当前不是 sequence parallel，末尾会调用 `self.experts.maybe_all_reduce_tensor_model_parallel(...)`，见 `vllm/model_executor/models/qwen3_moe.py:237-245`。
- `maybe_all_reduce_tensor_model_parallel()` 默认会落到 `tensor_model_parallel_all_reduce(...)`，见 `vllm/model_executor/layers/fused_moe/layer.py:1538-1545`。
- 只开 TP、不启 EP 时，`ep_size=1`、`use_ep=False`，因此不会走 `get_ep_group().dispatch/combine()` 那条 All2All 路径，见 `vllm/model_executor/layers/fused_moe/config.py:992-1018` 与 `vllm/model_executor/layers/fused_moe/layer.py:1792-1794`。

##### 结论
- 只开 TP 时，Qwen3 MoE 的主要通信仍然是 **TP all-reduce**，不是 EP All2All。

### 5. 线程、核心、NUMA：到底是不是“所有核心都通信”

#### 事实
- 自动绑核按 `local_rank` 选 NUMA node，不是按 `tp_rank`，见 `vllm/v1/worker/cpu_worker.py:140-152`。
- 本次 `TP=2` 日志里：
  - 一个 worker 的 auto bind list 是 `256-382` 这一组物理核，见 `test_results/qwen3_tp2_init_2026-03-18_2338.log:16`
  - 另一个 worker 的 auto bind list 是 `384-510` 这一组物理核，见 `test_results/qwen3_tp2_init_2026-03-18_2338.log:17`
  - 之后 OMP 线程也确实落在各自那组核心上，见 `test_results/qwen3_tp2_init_2026-03-18_2338.log:19-20`、`test_results/qwen3_tp2_init_2026-03-18_2338.log:150-151`

#### 推断
- 从 vLLM 调用栈看，当前环境里“通信参与者”主要是 rank/worker 进程；每个 worker 内部的 OMP 线程是本地计算线程，而不是 vLLM 显式管理的跨 rank 通信端点。
- 所以更准确的说法是：**不是整机所有核心互相通信，而是每个 TP worker 使用自己绑到的那部分核心参与本地计算，同时各 worker 进程之间做 collective。**

#### 假设
- 本文不展开 gloo backend 自己在进程内部可能使用的辅助线程；那属于 PyTorch/gloo 内部实现细节，不是当前 vLLM 代码直接可见的 rank/core 映射层。

### 6. 为什么旧文档里会让人误以为一定走 `csrc/cpu/shm.cpp`

#### 事实
- 源码里确实实现了 `_CPUSHMDistributed` 和 `csrc/cpu/shm.cpp`，包括：
  - `shm_allreduce`
  - `shm_gather`
  - `shm_all_gather`
  见 `vllm/distributed/device_communicators/cpu_communicator.py:158-212` 与 `csrc/cpu/shm.cpp:549-568`、`csrc/cpu/shm.cpp:571-607`、`csrc/cpu/shm.cpp:803-834`。
- 若这条路径生效，`csrc/cpu/shm.cpp` 会让每个 rank 内的 OpenMP 线程进入 `shm_cc_loop()`，按线程分块，通过共享内存 buffer、自旋等待、stamp/fence 完成同步，见 `csrc/cpu/shm.cpp:396-427`、`csrc/cpu/shm.cpp:167-195`。

#### 结论
- 这条 SHM 路径是“源码支持的可选实现”，不是“当前环境已经启用的实现”。
- 以后如果重新编译环境并确认 `torch.ops._C.init_shm_manager == True`，再把它升级为主分析对象更合适。

## 最小心智模型

### 当前环境下，`Qwen3-30B-A3B + TP=2` 可以先记成下面这张图

```text
API/bench 进程
  -> EngineCore
    -> MessageQueue 广播控制消息
      -> Worker_TP0
      -> Worker_TP1

Worker_TP0 / Worker_TP1：
  1. 本地算各自 shard 的 qkv / gate_up / matmul
  2. 在 o_proj / down_proj / sparse MoE block 末尾做 TP all_reduce
  3. 通常只由 TP0 把结果回给 executor
```

## 后续建议

1. 如果你只关心这台机子的真实性能瓶颈，优先盯 `RowParallelLinear.forward` 和 MoE 末尾 `maybe_all_reduce_tensor_model_parallel`。
2. 如果你要继续解释 `Remote DRAM Reads %`，建议把 `TP=1`、`TP=2`、`TP=4` 三组启动日志里的 bind list、uProf 栈、`all_reduce` 热点放到一起对照。
3. 如果你想，我下一步可以继续把这篇文档压缩成一页“TP=2 时序图 + 通信点表格”。
