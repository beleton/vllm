# 02. CPU worker 初始化与绑核

> 更新时间：2026-03-17  
> 范围：只看 CPU worker、OpenMP 线程、CPU affinity、NUMA 内存策略。  
> 假设：Linux、NUMA 功能编译开启；否则 `csrc/cpu/utils.cpp` 中的绑核/绑内存逻辑不会真正生效。

## 结论

### 事实
- CPU backend 的全局线程环境首先在 `vllm/platforms/cpu.py::CpuPlatform.check_and_update_config` 中设置：包括 `VLLM_WORKER_MULTIPROC_METHOD=spawn`、默认 `OMP_NUM_THREADS`、`LOCAL_WORLD_SIZE` 等。
- 每个worker通过 MultiprocExecutor._init_executor() 里循环调用 `WorkerProc.make_worker_process(...)` 拉起，每个worker进程内部有OpenMP/Torch线程
- 具体到每个 worker，绑核逻辑在 `vllm/v1/worker/cpu_worker.py::CPUWorker.init_device` 中执行；它会先解析 `VLLM_CPU_OMP_THREADS_BIND`，再调用 `torch.ops._C_utils.init_cpu_threads_env(...)`。
- 自动绑核时，`CPUWorker._get_autobind_cpu_ids` 先通过 `CpuPlatform.get_allowed_cpu_core_node_list()` 取到“当前进程允许看到的 CPU/NUMA 拓扑”，再按 `local_rank -> allowed_numa_nodes[local_rank]` 选 NUMA 节点。
- 在 x86 上，自动绑核不是“一个物理核的两个 SMT 线程都拿”，而是每个物理核只取一个逻辑 CPU：`cpu_list[-1:]`。
- `VLLM_CPU_NUM_OF_RESERVED_CPU` 会在自动选出的逻辑 CPU 列表尾部保留若干 CPU 不给 OMP 线程使用；若用户未设置，且 `world_size > 1` 或本地 DP>1，则默认保留 1 个 CPU。
- `csrc/cpu/utils.cpp::init_cpu_threads_env` 同时做三件事：
  1. 解析 CPU 列表；
  2. 根据这些 CPU 所属 NUMA node 执行 `numa_migrate_pages + membind/interleave`；
  3. 设置 `omp_set_num_threads` / `torch::set_num_threads`，并在 OpenMP 并行区内逐线程 `sched_setaffinity`。
- `OMP_NUM_THREADS` 不只影响算子并行，也会影响 `vllm/v1/engine/input_processor.py` 中输入预处理的 Torch 线程数。

### 推断
- 对 `2 x AMD EPYC 9745` 这类多 NUMA 节点平台，`NPS` 改变的是 `allowed_numa_nodes` 的数量和每个 node 上的 core 集合，所以它会直接改变自动绑核的 rank->CPU 集合映射，而不是只改变“内存远近”。
- 如果 `TP` 大于 `allowed_numa_nodes` 数量，自动绑核会直接触发断言失败；此时只能手动指定 `VLLM_CPU_OMP_THREADS_BIND`。

### 建议
- 做性能对比时，必须同时记录：`TP`、`NPS`、`VLLM_CPU_OMP_THREADS_BIND`、`VLLM_CPU_NUM_OF_RESERVED_CPU`、`OMP_NUM_THREADS`。
- 若怀疑自动绑核不符合预期，优先看 worker 日志里的 `auto thread-binding list`，再看 `init_cpu_threads_env` 返回的 OMP 线程->core 映射。

## 证据

### 绑核主路径

```text
vllm/platforms/cpu.py::CpuPlatform.check_and_update_config
  -> 设置 OMP_NUM_THREADS / LOCAL_WORLD_SIZE / spawn 等默认环境
vllm/v1/worker/cpu_worker.py::CPUWorker.init_device
  -> 解析 VLLM_CPU_OMP_THREADS_BIND
  -> auto: CPUWorker._get_autobind_cpu_ids(...)
  -> manual: 从 VLLM_CPU_OMP_THREADS_BIND 按 rank 取子串
  -> torch.ops._C_utils.init_cpu_threads_env(cpu_ids)
csrc/cpu/utils.cpp::init_cpu_threads_env
  -> numa_migrate_pages / numa_set_membind / numa_set_interleave_mask
  -> omp_set_num_threads / torch::set_num_threads
  -> #pragma omp parallel for -> sched_setaffinity
```

### 自动绑核的 rank 映射规则
- 拓扑来源：`vllm/platforms/cpu.py::CpuPlatform.get_allowed_cpu_core_node_list`
  - 用 `lscpu -J -e=CPU,CORE,NODE` 读取逻辑 CPU、物理 core、NUMA node。
  - 再用 `os.sched_getaffinity(0)` 过滤成当前进程真正允许使用的 CPU。
  - 若设置了 `CPU_VISIBLE_MEMORY_NODES`，还会进一步过滤可见 NUMA node。
- rank 选择：`vllm/v1/worker/cpu_worker.py::CPUWorker._get_autobind_cpu_ids`
  - `selected_numa_node = allowed_numa_nodes[self.local_rank]`
  - 即自动绑核按 `local_rank` 取第 N 个可用 NUMA node，而不是按 `tp_rank` 单独取。
- x86 的 SMT 选择：`lambda cpus: cpus[-1:]`
  - 对同一物理核的多个逻辑 CPU，只保留排序后的最后一个。

### 手动绑核的 rank 映射规则
- 环境变量：`vllm/envs.py` 中 `VLLM_CPU_OMP_THREADS_BIND`
- 解析点：`vllm/v1/worker/cpu_worker.py::CPUWorker.init_device`
- 规则：
  - 字符串按 `|` 分割，每一段对应一个 rank 的 CPU 列表；
  - 若存在本地 DP，先按 `data_parallel_rank_local * world_size` 切出本 DP 的子列表；
  - 再用当前 `rank` 取对应字符串。

### NUMA 内存策略的实际生效点
- `csrc/cpu/utils.cpp::init_cpu_threads_env`
  - 先根据目标 CPU 集合推导其 NUMA node 集合；
  - 对已有页调用 `numa_migrate_pages`；
  - 单 node 时走 `numa_set_membind`；
  - 多 node 时走 `numa_set_interleave_mask`；
  - 最后 `numa_set_strict(1)`。

## 机制解释

### 1. 自动绑核为什么是 `local_rank -> NUMA node`
- 多 worker 的创建顺序在 `vllm/v1/executor/multiproc_executor.py::MultiprocExecutor._init_executor` 中是按 `local_rank` 递增创建。
- `CPUWorker` 不再额外计算“chiplet 级映射”，而是直接拿 `allowed_numa_nodes[self.local_rank]`。
- 所以自动绑核的基本粒度是“每个 worker 占一个 NUMA node”，而不是“每个 worker 占任意 CPU 子集”。

### 2. 为什么 `VLLM_CPU_NUM_OF_RESERVED_CPU` 会影响线程数
- `init_cpu_threads_env` 里最终 `omp_set_num_threads` 的值等于保留后 CPU 列表长度。
- 也就是说，这个变量不只是“留几个核给系统”，它会直接减少 worker 内部 OMP 线程数。

### 3. `OMP_NUM_THREADS` 与实际线程数的关系
- 平台初始化阶段：`CpuPlatform.check_and_update_config` 会在非 `nobind` 情况下把 `OMP_NUM_THREADS` 设为当前 Torch 线程数。
- worker 绑核阶段：`init_cpu_threads_env` 又会根据实际 CPU 列表重新设置 `omp_set_num_threads` 和 `torch::set_num_threads`。
- 所以对 CPU worker 来说，真正决定线程数的是“最终被绑定的 CPU 列表长度”。

### 4. 对 NUMA/Chiplet 分析的含义
- 静态代码上，worker 的 CPU 集与 NUMA node 一一对应；是否进一步映射到特定 CCD/Chiplet，要看 `lscpu` 导出的 NUMA 划分，也就是平台的 `NPS` 设置。
- 因此同一套 TP 配置，在不同 `NPS` 下，自动绑核得到的 CPU 集合可能完全不同。

## CPU 相关环境变量

| 变量 | 生效位置 | 影响 |
| --- | --- | --- |
| `VLLM_CPU_OMP_THREADS_BIND` | `vllm/v1/worker/cpu_worker.py::CPUWorker.init_device` | 选择 auto / nobind / 手工 CPU 列表 |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | `vllm/v1/worker/cpu_worker.py::CPUWorker._get_autobind_cpu_ids` | 从自动选出的 CPU 列表尾部保留若干 CPU |
| `OMP_NUM_THREADS` | `vllm/platforms/cpu.py`、`csrc/cpu/utils.cpp`、`vllm/v1/engine/input_processor.py` | 影响 OMP/Torch 线程数，也影响输入预处理线程数 |
| `CPU_VISIBLE_MEMORY_NODES` | `vllm/platforms/cpu.py::CpuPlatform.get_allowed_cpu_core_node_list` | 收窄自动绑核可见的 NUMA node |
| `LOCAL_WORLD_SIZE` | `vllm/platforms/cpu.py` | 给 IPEX / shared-memory all-reduce 提示 TP 局部规模 |

> 详细变量说明见 `info/vllm_analysis/06_CPU环境变量与生效路径.md`。