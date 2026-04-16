# CPU 环境变量与生效路径

## 问题
- CPU backend 直接相关的关键环境变量有哪些，它们分别在哪些路径生效。

## 结论
- 对 CPU 推理最关键的用户可配置变量包括：
  - `VLLM_CPU_KVCACHE_SPACE`
  - `VLLM_CPU_OMP_THREADS_BIND`
  - `VLLM_CPU_NUM_OF_RESERVED_CPU`
  - `CPU_VISIBLE_MEMORY_NODES`
  - `OMP_NUM_THREADS`
  - `VLLM_CPU_SGL_KERNEL`
  - `VLLM_FLOAT32_MATMUL_PRECISION`
- CPU backend 还会主动写入一组运行时变量，包括：
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn`
  - `LOCAL_WORLD_SIZE=<tp_size>`
  - `NUMEXPR_MAX_THREADS`
  - `TORCHINDUCTOR_COMPILE_THREADS=1`
  - `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`
- 若检测到 `libiomp5.so`，还会补写 `KMP_*`。
- `VLLM_CPU_KVCACHE_SPACE` 的“未设置”和“显式设为 0”语义不同：未设置时按运行时代码自动估算；显式设为 `0` 时就是 `0 GiB`。
- `VLLM_CPU_OMP_THREADS_BIND != nobind` 时，最终线程数由绑定后的 CPU 列表长度决定，不再由外部 `OMP_NUM_THREADS` 单独决定。
- `VLLM_CPU_NUM_OF_RESERVED_CPU` 只在 `auto` 绑核路径生效，并且会直接减少 OpenMP / Torch 线程数。
- `VLLM_CPU_SGL_KERNEL` 会影响 unquantized linear、x86 MoE 和部分 CPU quantized scaled-mm 路径。
- `VLLM_CPU_MOE_PREPACK` 在当前仓库只出现在导出名单里，未找到解析或读取点；当前不能把它当成有效调优开关。
- `VLLM_DIST_IDENT` 是 worker 初始化时自动写入的内部变量，用来给 CPU SHM collective 命名，一般不应手工设置。

## 关键变量与入口

| 变量 | 直接源码入口 | 主要影响 |
| --- | --- | --- |
| `VLLM_CPU_KVCACHE_SPACE` | `vllm/envs.py`、`vllm/platforms/cpu.py` | CPU KV cache 容量、`num_blocks`、最大可承载上下文 / 并发 |
| `VLLM_CPU_OMP_THREADS_BIND` | `vllm/envs.py`、`vllm/v1/worker/cpu_worker.py` | auto / nobind / 手工绑核，决定 rank 的 CPU 集 |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | `vllm/envs.py`、`vllm/v1/worker/cpu_worker.py` | auto 绑核时保留 CPU 数，直接减少线程数 |
| `CPU_VISIBLE_MEMORY_NODES` | `vllm/platforms/cpu.py` | 收窄 auto 绑核可见 NUMA node，并改变 rank->node 映射 |
| `OMP_NUM_THREADS` | `vllm/platforms/cpu.py`、`csrc/cpu/utils.cpp`、`vllm/v1/engine/input_processor.py` | OpenMP / Torch 线程数；在 bind 路径下会被重写 |
| `VLLM_CPU_SGL_KERNEL` | `vllm/model_executor/layers/utils.py`、`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | linear / MoE / quantized scaled-mm 是否走 SGL kernel |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `vllm/envs.py`、`vllm/v1/worker/gpu_worker.py` | PyTorch float32 matmul 精度模式 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `vllm/platforms/cpu.py`、`vllm/utils/system_utils.py` | worker 进程创建方式，CPU 路径强制 `spawn` |
| `LOCAL_WORLD_SIZE` | `vllm/platforms/cpu.py` | 传给下层 TP / SHM all-reduce 相关 runtime |
| `VLLM_DIST_IDENT` | `vllm/v1/worker/cpu_worker.py`、`vllm/distributed/device_communicators/cpu_communicator.py` | CPU SHM collective 的实例名 |

## `VLLM_CPU_KVCACHE_SPACE`
- 变量不存在时返回 `None`，不是 `0`
- `cpu.py` 会把 `None` 解释为“按 NUMA 节点均分整机内存，再取 50%”估算
- 显式设为 `0` 时就是 `0 GiB`
- `num_blocks` 最终会按所有 rank 中的最小值收缩

因此：
- 对 `TP > 1` 的 CPU 场景，`VLLM_CPU_KVCACHE_SPACE` 应按单 rank 所在 NUMA 节点容量来理解
- “未设置”和“设置为 0”不是同一个配置

## `VLLM_CPU_OMP_THREADS_BIND` / `VLLM_CPU_NUM_OF_RESERVED_CPU` / `CPU_VISIBLE_MEMORY_NODES`
- `VLLM_CPU_OMP_THREADS_BIND` 默认值是 `auto`
- `CPUWorker.init_device` 中分成三条路径：`auto`、`nobind`、手工字符串
- 手工字符串支持用 `|` 给不同 rank 分段
- `CPU_VISIBLE_MEMORY_NODES` 会先改变可见 NUMA node 集合，再影响 auto 绑核的 `local_rank -> node`
- `VLLM_CPU_NUM_OF_RESERVED_CPU` 只在 `auto` 路径生效，通过裁掉列表尾部 CPU 直接减少线程数

## `OMP_NUM_THREADS`
- 只有在 `VLLM_CPU_OMP_THREADS_BIND == nobind` 时，外部 `OMP_NUM_THREADS` 才保留最终控制权
- 一旦走 bind 路径，`init_cpu_threads_env` 会把 OMP / Torch 线程数重设为绑定 CPU 列表长度
- `input_processor.py` 还会在输入预处理阶段读取 `OMP_NUM_THREADS`

因此：
- 同一个 `OMP_NUM_THREADS=64`，在 `nobind` 与 `auto` 两种模式下，实际线程数可能完全不同

## `VLLM_CPU_SGL_KERNEL` / `VLLM_FLOAT32_MATMUL_PRECISION`
- `VLLM_CPU_SGL_KERNEL` 影响：
  - unquantized linear
  - x86 MoE
  - 一部分 CPU quantized scaled-mm
- `VLLM_FLOAT32_MATMUL_PRECISION` 在 worker 初始化时通过 `torch.set_float32_matmul_precision(...)` 生效，CPU backend 也会受影响

## 分析
- 若实验只记录 `OMP_NUM_THREADS`，而没同时记录 `VLLM_CPU_OMP_THREADS_BIND` / `VLLM_CPU_NUM_OF_RESERVED_CPU`，后续很难精确复盘实际线程数。
- `VLLM_CPU_SGL_KERNEL` 往往不是“小优化开关”，而是直接切 kernel 家族的开关。
- `LOCAL_WORLD_SIZE`、`KMP_*` 这类变量在仓库里多是“写出给下层 runtime”，其真实性能效果仍要结合 worker 日志或 profiler 再确认。

## 证据
- 关键源码路径：
  - `vllm/envs.py`
  - `vllm/platforms/cpu.py`
  - `vllm/v1/worker/cpu_worker.py`
  - `csrc/cpu/utils.cpp`
  - `vllm/v1/engine/input_processor.py`
  - `vllm/model_executor/layers/utils.py`
  - `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
  - `vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py`
  - `vllm/v1/worker/gpu_worker.py`
  - `vllm/utils/system_utils.py`

## 边界
- 本页主要来自静态源码阅读，不绑定新的运行日志。
- 内部变量如 `LOCAL_WORLD_SIZE`、`VLLM_DIST_IDENT` 的真实性能影响需要结合 worker 日志、`htop`、`perf` 或 uProf 再确认。

## 后续验证点
- 继续做性能实验时，至少同时保存：
  - `TP`
  - `NPS`
  - `VLLM_CPU_KVCACHE_SPACE`
  - `VLLM_CPU_OMP_THREADS_BIND`
  - `VLLM_CPU_NUM_OF_RESERVED_CPU`
  - `CPU_VISIBLE_MEMORY_NODES`
  - `OMP_NUM_THREADS`
  - `VLLM_CPU_SGL_KERNEL`
  - `VLLM_FLOAT32_MATMUL_PRECISION`

