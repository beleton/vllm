# vLLM CPU NPS/TP 根因分析（基于 Qwen3-30B-A3B）

## 1. 结论

在这台双路 `AMD EPYC 9745` 机器上，`NPS` 会影响 LLM 推理性能，不是因为它单独改变了算力，而是因为它改变了：

- NUMA 节点划分方式
- 每个 TP worker 的绑核范围
- Linux 的内存分配与页迁移策略
- TP collective 需要跨越的 NUMA 边界数量

结合当前 `Qwen3-30B-A3B` 的实测数据和 vLLM 代码，当前最可靠的结论是：

- `NPS2_TP4` 是中高并发下的最佳点
- `NPS4_TP8` 在低并发有少量优势，但在中高并发明显退化
- `NPS4_TP8` 的问题不是“总带宽不够”，而是“远端访问、cache miss 延迟、rank 间同步等待”显著增加

一句话概括：

`NPS2_TP4` 拿到了更好的局部性，但还没有把 TP 通信成本推高到失控；`NPS4_TP8` 则进入了过度切分区间。

---

## 2. 实验范围与口径

本分析只基于以下数据与代码：

- 实验结果目录：
  - `/home/zjj/vllm/test_results/Qwen3-30B-A3B`
- 汇总分析报告：
  - `/home/zjj/vllm/test_results/Qwen3-30B-A3B_analysis/docs/Qwen3-30B-A3B_analysis_report.md`
- 实验命令文档：
  - `/home/zjj/vllm/info/Qwen3-30B-A3B_AMDuProfPcm分批实验命令.md`
- 关键源码：
  - `vllm/v1/worker/cpu_worker.py`
  - `vllm/platforms/cpu.py`
  - `csrc/cpu/utils.cpp`
  - `vllm/distributed/device_communicators/cpu_communicator.py`
  - `vllm/distributed/parallel_state.py`
  - `vllm/distributed/communication_op.py`
  - `vllm/model_executor/layers/linear.py`
  - `vllm/model_executor/models/qwen3_moe.py`
  - `vllm/model_executor/layers/fused_moe/layer.py`
  - `csrc/cpu/shm.cpp`

实验配置不是“纯 NPS A/B”，而是联动修改了 TP：

- `NPS1_TP2`
- `NPS2_TP4`
- `NPS4_TP8`

命令文档明确要求修改 `--tensor-parallel-size`，并固定：

- `evalscope perf --parallel 1 2 4 8 16 32 64 --number 1 2 4 8 16 32 64`
- PCM 采样主对齐点为 `parallel=16, number=16`

因此本文讨论的是：

**NPS 改变 NUMA 拓扑后，TP 大小如何通过 vLLM 的 CPU 执行路径把这些变化放大或削弱。**

---

## 3. 关键事实

### 3.1 Evalscope 结果

来自 `/home/zjj/vllm/test_results/Qwen3-30B-A3B_analysis/docs/Qwen3-30B-A3B_analysis_report.md`：

`parallel=16`：

| 配置 | Output tok/s | Avg latency(s) | Avg TTFT(s) |
| --- | ---: | ---: | ---: |
| `NPS1_TP2` | 254.48 | 64.32 | 9.98 |
| `NPS2_TP4` | 279.18 | 58.66 | 10.97 |
| `NPS4_TP8` | 222.74 | 73.55 | 17.80 |

`parallel=64`：

| 配置 | Output tok/s | Avg latency(s) | Avg TTFT(s) |
| --- | ---: | ---: | ---: |
| `NPS1_TP2` | 364.08 | 179.54 | 32.96 |
| `NPS2_TP4` | 387.36 | 168.98 | 33.14 |
| `NPS4_TP8` | 291.95 | 224.34 | 47.98 |

扩展倍数（`parallel=64 / parallel=1`）：

| 配置 | 扩展倍数 |
| --- | ---: |
| `NPS1_TP2` | 9.94x |
| `NPS2_TP4` | 8.69x |
| `NPS4_TP8` | 6.52x |

结论：

- `NPS4_TP8` 在 `parallel=1~2` 的总延迟有一点优势
- 从 `parallel>=4` 开始，`NPS2_TP4` 基本成为最优
- `NPS4_TP8` 的高并发扩展性最差

### 3.2 PCM 结果

固定负载 `parallel=16, number=16` 下，最关键的指标如下。

Batch1：

| 指标 | `NPS1_TP2` | `NPS2_TP4` | `NPS4_TP8` |
| --- | ---: | ---: | ---: |
| Total Mem Bw (GB/s) | 235.16 | 273.80 | 299.53 |
| Remote Inbound Read (GB/s) | 0.88 | 3.56 | 12.14 |

Batch2：

| 指标 | `NPS1_TP2` | `NPS2_TP4` | `NPS4_TP8` |
| --- | ---: | ---: | ---: |
| L3 Miss % | 43.78 | 45.32 | 50.88 |
| Ave L3 Miss Latency (ns) | 261.88 | 266.04 | 292.53 |
| Remote DRAM Reads % | 0.03 | 0.15 | 0.48 |

通用 CPU 指标：

| 指标 | `NPS1_TP2` | `NPS2_TP4` | `NPS4_TP8` |
| --- | ---: | ---: | ---: |
| IPC | 1.41 | 1.47 | 1.33 |
| CPI | 0.71 | 0.68 | 0.75 |

Topdown：

| 指标 | `NPS1_TP2` | `NPS2_TP4` | `NPS4_TP8` |
| --- | ---: | ---: | ---: |
| Frontend_Bound.Latency | 4.04 | 4.33 | 4.90 |
| Retiring | 21.48 | 22.66 | 21.37 |

结论：

- `NPS4_TP8` 的总带宽最高
- 但它的远端读、远端 DRAM、L3 miss 延迟也最高
- `IPC/CPI/Retiring` 同时恶化

这说明 `NPS4_TP8` 不是“内存太闲”，而是“内存很忙，但忙在远端访问和同步上”。

---

## 4. vLLM 代码中的机制

### 4.1 CPU worker 如何按 NUMA 绑核

`CPUWorker.init_device()` 在 `VLLM_CPU_OMP_THREADS_BIND=auto` 时，会调用 `_get_autobind_cpu_ids()` 为每个 rank 自动选 CPU。[`/home/zjj/vllm/vllm/v1/worker/cpu_worker.py`](../vllm/v1/worker/cpu_worker.py)

关键逻辑：

- 读取可用 NUMA node 列表
- 按 `allowed_numa_nodes[self.local_rank]` 选择一个 node
- x86 下每个物理核只取一个 SMT 线程
- `world_size > 1` 时默认预留 1 个 CPU

这意味着：

- `TP=2` 时通常只会起 2 个本地 worker，各绑定到前 2 个可见 NUMA node
- `TP=4` 时会落到前 4 个可见 node
- `TP=8` 时会落到前 8 个可见 node

### 4.2 vLLM 不只是绑核，还会改内存分配策略

`torch.ops._C_utils.init_cpu_threads_env()` 的 C++ 实现在 `csrc/cpu/utils.cpp`。[`/home/zjj/vllm/csrc/cpu/utils.cpp`](../csrc/cpu/utils.cpp)

它做了 4 件关键事：

1. 根据绑定的 CPU 推导 NUMA node
2. 调用 `numa_migrate_pages` 迁移已有页面
3. 单 node 时调用 `numa_set_membind`
4. 对 OMP 线程逐个 `sched_setaffinity`

所以这里影响的不是“线程在哪里跑”这么简单，还包括：

- KV cache 和临时 buffer 更倾向分配到哪个 NUMA node
- 远端访问是否容易发生
- 已分配页面是否会被迁到新的内存策略下

### 4.3 TP 在 CPU 上是多进程 + collective

CPU 平台会强制：

- `distributed_executor_backend = "mp"`
- `dist_backend = "gloo"`

见 [`/home/zjj/vllm/vllm/platforms/cpu.py`](../vllm/platforms/cpu.py)。

`MultiprocExecutor` 按 `local_rank=0..local_world_size-1` 启动 worker 进程。[`/home/zjj/vllm/vllm/v1/executor/multiproc_executor.py`](../vllm/v1/executor/multiproc_executor.py)

TP group 在 `initialize_model_parallel()` 中按连续 rank 建立。[`/home/zjj/vllm/vllm/distributed/parallel_state.py`](../vllm/distributed/parallel_state.py)

通信本身通过：

- `tensor_model_parallel_all_reduce()`
- `tensor_model_parallel_all_gather()`

进入 TP group。[`/home/zjj/vllm/vllm/distributed/communication_op.py`](../vllm/distributed/communication_op.py)

CPU 下 `CpuCommunicator` 可能切到共享内存 collective，但本质上仍然要在多个 rank 间同步与交换数据。[`/home/zjj/vllm/vllm/distributed/device_communicators/cpu_communicator.py`](../vllm/distributed/device_communicators/cpu_communicator.py)

`csrc/cpu/shm.cpp` 里能看到明显的：

- stamp
- spin wait
- memory fence

所以 rank 越多，等待慢 rank 的概率越高。[`/home/zjj/vllm/csrc/cpu/shm.cpp`](../csrc/cpu/shm.cpp)

### 4.4 Qwen3-30B-A3B 为什么特别敏感

`Qwen3-30B-A3B` 在仓库里走的是 `Qwen3Moe` 路径。[`/home/zjj/vllm/vllm/model_executor/models/qwen3_moe.py`](../vllm/model_executor/models/qwen3_moe.py)

每个 decoder layer 都包含：

- attention
- dense MLP 或 sparse MoE block

attention 里的：

- `qkv_proj` 是并行线性层
- `o_proj` 是 `RowParallelLinear`

MoE 里的：

- `gate_up_proj` 是 `MergedColumnParallelLinear`
- `down_proj` 是 `RowParallelLinear`
- 还可能发生 expert dispatch/combine

而 `RowParallelLinear.forward()` 在 `tp_size > 1` 时直接做 `tensor_model_parallel_all_reduce()`。[`/home/zjj/vllm/vllm/model_executor/layers/linear.py`](../vllm/model_executor/layers/linear.py)

MoE 路径里还会：

- `get_ep_group().dispatch(...)`
- `get_ep_group().combine(...)`
- `maybe_all_reduce_tensor_model_parallel(...)`

因此 `TP` 变大时，不只是 attention 在通信，MoE 也在把跨 rank 的代价往上推。[`/home/zjj/vllm/vllm/model_executor/layers/fused_moe/layer.py`](../vllm/model_executor/layers/fused_moe/layer.py)

---

## 5. 为什么 NPS 会影响推理性能

### 5.1 OS/硬件层

对 AMD EPYC 而言，`NPS` 改变的是每个 socket 如何被切成 NUMA 域。

`NPS` 增大后：

- 单个 NUMA 域更小
- 本地内存访问路径更短
- 理论上的本地性更强
- 但 NUMA 边界变多

如果软件能把计算、数据和线程都留在本地 node，`NPS` 增大可能带来收益。

如果软件的一个计算步骤需要跨多个 node 协同，`NPS` 增大也会带来代价：

- 远端内存访问更多
- cache coherence 成本更高
- 集体通信更容易被慢 rank 拖尾

### 5.2 vLLM 层

在 vLLM CPU 后端里，`NPS` 会通过以下链路生效：

1. NPS 改变可见 NUMA 拓扑
2. `auto` 绑核按 NUMA node 给 rank 分配 CPU
3. `init_cpu_threads_env()` 把内存也绑到这些 node
4. TP collective 在这些 rank 之间反复发生

因此，`NPS` 影响的是：

- rank 的本地工作集大小
- rank 是否容易触发远端访存
- TP collective 跨越多少个 NUMA 边界

---

## 6. 为什么会出现这样的性能变化

### 6.1 `NPS1_TP2 -> NPS2_TP4` 为什么提升

这是“局部性收益大于新增通信成本”的阶段。

从数据看：

- 吞吐从 `254.48` 提升到 `279.18 tok/s`
- 延迟下降
- IPC 提升、CPI 改善
- 总内存带宽提升，但远端访存还没有失控

合理解释是：

- `NPS2` 让每个 rank 的本地 NUMA 域更集中
- `TP4` 把模型切分得更细，单 rank 的权重和活跃工作集更小
- 这时 TP 通信虽然变多，但还在可控范围内

所以 `NPS2_TP4` 正好落在“局部性增强、通信尚可接受”的甜点区。

### 6.2 `NPS2_TP4 -> NPS4_TP8` 为什么退化

这是“过度切分后通信/远端访问收益反噬”的阶段。

从数据看：

- 总带宽继续升高
- 但 `Remote Inbound Read` 急剧升高
- `Remote DRAM Reads %` 急剧升高
- `Ave L3 Miss Latency` 变高
- `IPC` 下降，`CPI` 上升
- TTFT 恶化最明显

这说明：

- 更多 NUMA 域并没有转化成更高的有效吞吐
- 反而把一个 token 的前向过程切成了更多 rank 间同步点
- 更多时间花在等待数据、等待 collective、等待慢 rank 上

### 6.3 为什么 TTFT 恶化尤其明显

你当前压测是：

- `prefix-length 0`
- prompt 固定 `1024`
- output 固定 `1024`
- prefix caching 关闭

所以每个请求都要做完整 prefill。

TTFT 对以下因素最敏感：

- 第一次大规模 prompt prefill
- 第一次 KV cache 写入
- 第一次全层同步
- 第一次远端 miss/远端 DRAM 访问

这就是为什么 `NPS4_TP8` 在低并发总延迟未必最差，但 `parallel=16` 时 TTFT 会从 `10.97s` 明显拉到 `17.80s`。

---

## 7. `NPS4 + TP2` 会不会只用 2 个 NUMA 节点

默认 `auto` 绑核下，答案是：**大概率会。**

机制如下：

1. `TP=2` 时，CPU 后端通常只会起 2 个本地 worker
2. `local_rank=0` 和 `local_rank=1` 分别取 `allowed_numa_nodes[0]`、`allowed_numa_nodes[1]`
3. 如果没有额外设置 `CPU_VISIBLE_MEMORY_NODES`，而可见节点列表是升序 `[0,1,2,3,4,5,6,7]`
4. 那么两个 rank 通常就落在 `node0` 和 `node1`

结果是：

- 主要推理计算只会集中在这两个 node
- 其他 node 不承担主要 TP worker 计算
- 在你这台机器上，这甚至可能意味着第二颗 CPU 基本没有被主推理 worker 利用到

这不是因为 vLLM “只能用两个 node”，而是因为默认 `auto` 策略就是“一个 rank 对应一个 NUMA node”。

### 7.1 边界条件

以下情况会改变结论：

- 手工设置 `VLLM_CPU_OMP_THREADS_BIND`
- 手工设置 `CPU_VISIBLE_MEMORY_NODES`
- 改变 data parallel / pipeline parallel 规模

例如：

- 若设 `CPU_VISIBLE_MEMORY_NODES=0,4`
- 再用 `TP=2`

那么默认 `auto` 更可能把两个 worker 放到 `node0` 和 `node4`，即跨 socket 分布。

### 7.2 这意味着什么

`NPS4 + TP2` 不等于“自动均匀用满 8 个 node”。

它更像是：

- 把整机切成 8 个更小的 NUMA 域
- 但只从中挑 2 个域给 2 个 TP rank 使用

这会导致两类可能结果：

- 若选中的两个 node 局部性好、带宽足够，单 rank 体验可能不错
- 但整机资源利用率可能很差

所以 `NPS4 + TP2` 通常不是默认最合理的组合，除非你明确设计了跨 socket 绑核策略。

---

## 8. 当前最可信的根因闭环

### 事实

- `NPS2_TP4` 在中高并发下最好
- `NPS4_TP8` 的总带宽最高，但远端访存和 miss 延迟也最高
- `NPS4_TP8` 的 IPC/CPI/Retiring 变差

### 机制

- NPS 改变 NUMA 划分
- vLLM 按 NUMA 给 TP worker 绑核并绑内存
- TP 越大，collective 越多
- Qwen3 MoE 会进一步增加跨 rank 的通信与同步点

### 推断

在这台机器和这条 vLLM CPU 路径上：

1. `NPS2_TP4` 达到了 locality 和 TP 通信成本的平衡点
2. `NPS4_TP8` 因过度切分导致远端访问、cache miss 和慢 rank 等待一起恶化
3. 当前数据不支持“`NPS4_TP8` 必然优于 `NPS2_TP4`”这个结论

---

## 9. 建议的后续验证

若要把因果再拆得更干净，下一轮建议按下面顺序做：

1. 固定 `TP`，只改 `NPS`
2. 固定 `NPS`，只扫 `TP`
3. 在 `NPS4` 下比较：
   - 默认 `auto`
   - 显式绑到 `node0,node4`
   - 显式绑到 `node1,node5`
4. 继续盯以下指标：
   - `Remote Inbound Read`
   - `Remote DRAM Reads %`
   - `Ave L3 Miss Latency`
   - `IPC`
   - `CPI`
   - `Output token throughput`

如果这些指标在显式绑核后明显改善，就可以进一步确认：

**当前瓶颈主要不是“算不动”，而是“算得太分散、同步太多、远端访问太重”。**
