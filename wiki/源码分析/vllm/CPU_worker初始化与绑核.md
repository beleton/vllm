# CPU worker 初始化与绑核

## 问题
- CPU worker 在哪里决定 CPU affinity、OMP 线程数和 NUMA 内存策略。

## 结论
- 全局线程环境首先在 `vllm/platforms/cpu.py::CpuPlatform.check_and_update_config` 中设置，包括 `VLLM_WORKER_MULTIPROC_METHOD=spawn`、默认 `OMP_NUM_THREADS`、`LOCAL_WORLD_SIZE` 等。
- 每个 worker 通过 `MultiprocExecutor._init_executor()` 循环调用 `WorkerProc.make_worker_process(...)` 拉起；每个 worker 进程内部再有 OpenMP / Torch 线程。
- 每个 worker 的绑核逻辑在 `vllm/v1/worker/cpu_worker.py::CPUWorker.init_device` 中执行；它先解析 `VLLM_CPU_OMP_THREADS_BIND`，再调用 `torch.ops._C_utils.init_cpu_threads_env(...)`。
- 自动绑核时，`CPUWorker._get_autobind_cpu_ids` 通过 `CpuPlatform.get_allowed_cpu_core_node_list()` 读取当前进程可见的 `CPU/CORE/NODE` 拓扑，再按 `local_rank -> allowed_numa_nodes[local_rank]` 选 NUMA 节点。
- 在 x86 上，自动绑核每个物理核只取一个逻辑 CPU，不取两个 SMT 线程。
- `VLLM_CPU_NUM_OF_RESERVED_CPU` 会从自动选出的 CPU 列表尾部保留若干 CPU，不给 OMP 线程使用；这会直接影响 worker 内部线程数。
- `csrc/cpu/utils.cpp::init_cpu_threads_env` 会同时做三件事：解析 CPU 列表、设置 NUMA 内存策略、设置 OMP/Torch 线程数并在 OpenMP 并行区逐线程 `sched_setaffinity`。
- 真正决定 worker 内部 OMP 线程数的不是初始 `OMP_NUM_THREADS`，而是最终被绑定的 CPU 列表长度。

## 绑核主路径

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

## 自动绑核规则
- `CpuPlatform.get_allowed_cpu_core_node_list()`
  - 用 `lscpu -J -e=CPU,CORE,NODE` 读取逻辑 CPU、物理 core、NUMA node
  - 用 `os.sched_getaffinity(0)` 过滤成当前进程真正允许使用的 CPU
  - 若设置了 `CPU_VISIBLE_MEMORY_NODES`，还会进一步过滤可见 NUMA node
- `CPUWorker._get_autobind_cpu_ids`
  - `selected_numa_node = allowed_numa_nodes[self.local_rank]`
  - 自动绑核按 `local_rank` 取第 N 个可用 NUMA node，而不是按 `tp_rank` 单独取

## 手动绑核规则
- 环境变量：`VLLM_CPU_OMP_THREADS_BIND`
- 解析点：`CPUWorker.init_device`
- 规则：
  - 字符串按 `|` 分割，每一段对应一个 rank 的 CPU 列表
  - 若存在本地 DP，先按本地 DP 切片
  - 再用当前 rank 取对应字符串

## NUMA 内存策略
- `init_cpu_threads_env` 会：
  - 根据目标 CPU 集推导其 NUMA node 集合
  - 对已有页调用 `numa_migrate_pages`
  - 单 node 时走 `numa_set_membind`
  - 多 node 时走 `numa_set_interleave_mask`

## 分析
- 对 `AMD EPYC` 这类多 NUMA 节点平台，`NPS` 改变的不只是内存远近，也会改变 `allowed_numa_nodes` 的数量和每个 node 上的 core 集合，因此会直接改变自动绑核结果。
- 若 `TP` 大于 `allowed_numa_nodes` 数量，自动绑核会失败，此时需要手动指定 `VLLM_CPU_OMP_THREADS_BIND`。
- `VLLM_CPU_NUM_OF_RESERVED_CPU` 不只是“留几个核给系统”，它会真实减少 worker 内部线程数。

## 证据
- 关键源码路径：
  - `vllm/platforms/cpu.py`
  - `vllm/v1/worker/cpu_worker.py`
  - `vllm/v1/executor/multiproc_executor.py`
  - `csrc/cpu/utils.cpp`
  - `vllm/v1/engine/input_processor.py`

## 边界
- 本页只看 CPU worker、OpenMP 线程、CPU affinity、NUMA 内存策略。
- 具体 `CCD/CCX/L3` 粒度如何映射，要看平台在当前 `NPS` 下导出的 NUMA 划分，不是这页单独能回答的。

## 后续验证点
- 继续分析绑核问题时，优先看 worker 日志里的 `auto thread-binding list`，再对照 `init_cpu_threads_env` 返回的 OMP 线程到 core 映射。
