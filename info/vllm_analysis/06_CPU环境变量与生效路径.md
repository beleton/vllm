# 06. CPU 环境变量与生效路径

> 更新时间：2026-03-18  
> 范围：只看 CPU backend 直接相关的环境变量，以及 CPU backend 会主动写入的运行时变量。  
> 假设：目标平台为 `2 x AMD EPYC 9745 128-Core Processor`；当前结论主要来自静态源码阅读，未绑定新的运行日志。

## 结论

### 事实
- 对 CPU 推理最关键的用户可配置变量共有 7 个：`VLLM_CPU_KVCACHE_SPACE`、`VLLM_CPU_OMP_THREADS_BIND`、`VLLM_CPU_NUM_OF_RESERVED_CPU`、`CPU_VISIBLE_MEMORY_NODES`、`OMP_NUM_THREADS`、`VLLM_CPU_SGL_KERNEL`、`VLLM_FLOAT32_MATMUL_PRECISION`。
- CPU backend 还会主动写入一组运行时变量：`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`LOCAL_WORLD_SIZE=<tp_size>`、`NUMEXPR_MAX_THREADS`、`TORCHINDUCTOR_COMPILE_THREADS=1`、`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`；若检测到 `libiomp5.so`，还会补写 `KMP_*`。
- `VLLM_CPU_KVCACHE_SPACE` 的“未设置”和“显式设为 0”语义不同：未设置时会按 `总内存 / NUMA 节点数 * 0.5` 估算；显式设为 `0` 时就是 `0 GiB`。
- `VLLM_CPU_OMP_THREADS_BIND != nobind` 时，最终线程数由绑定后的 CPU 列表长度决定，不再由外部 `OMP_NUM_THREADS` 单独决定。
- `VLLM_CPU_NUM_OF_RESERVED_CPU` 只在 `auto` 绑核路径生效，并且会直接减少 OpenMP / Torch 线程数。
- `VLLM_CPU_SGL_KERNEL` 目前会影响 unquantized linear、x86 MoE，以及一部分 CPU quantized scaled-mm 路径。
- `VLLM_CPU_MOE_PREPACK` 在当前仓库只出现在导出名单里，未找到解析或读取点；当前不能把它当成有效调优开关。
- `VLLM_DIST_IDENT` 是 worker 初始化时自动写入的内部变量，用来给 CPU SHM collective 命名，一般不应手工设置。

### 推断
- 在 `TP > 1` 的 CPU 场景里，`VLLM_CPU_KVCACHE_SPACE` 应按“单 rank 所在 NUMA 节点”的容量估算；因为最终 `num_blocks` 会按所有 rank 中的最小值收缩。
- 如果实验只记录 `OMP_NUM_THREADS`，而没有同时记录 `VLLM_CPU_OMP_THREADS_BIND` / `VLLM_CPU_NUM_OF_RESERVED_CPU`，后续几乎无法精确复盘实际线程数。
- `LOCAL_WORLD_SIZE`、`KMP_*` 这类变量在仓库里多是“写出给下层 runtime”，其真实性能效果需要用 worker 日志、`htop`、`perf` 或 uProf 交叉确认。

### 建议
- 记录实验配置时，至少同时保存：`TP`、`NPS`、`VLLM_CPU_KVCACHE_SPACE`、`VLLM_CPU_OMP_THREADS_BIND`、`VLLM_CPU_NUM_OF_RESERVED_CPU`、`CPU_VISIBLE_MEMORY_NODES`、`OMP_NUM_THREADS`、`VLLM_CPU_SGL_KERNEL`、`VLLM_FLOAT32_MATMUL_PRECISION`。
- 解释 KV cache 容量时，以 `vllm/platforms/cpu.py` 的实际实现为准，不要直接沿用“默认 0”或“默认 4GB”的旧说法。
- 分析算子热点前，先确认 `VLLM_CPU_SGL_KERNEL` 是否切到了 SGL；否则 linear / MoE 的线程行为和 cache 行为可能完全不同。

## 证据

### 1. 建议优先关注的变量

| 变量 | 直接源码入口 | 主要影响 |
| --- | --- | --- |
| `VLLM_CPU_KVCACHE_SPACE` | `vllm/envs.py:700`、`vllm/platforms/cpu.py:142` | CPU KV cache 容量、`num_blocks`、最大可承载上下文 / 并发 |
| `VLLM_CPU_OMP_THREADS_BIND` | `vllm/envs.py:705`、`vllm/v1/worker/cpu_worker.py:54` | auto / nobind / 手工绑核，决定 rank 的 CPU 集 |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | `vllm/envs.py:708`、`vllm/v1/worker/cpu_worker.py:168` | auto 绑核时保留 CPU 数，直接减少线程数 |
| `CPU_VISIBLE_MEMORY_NODES` | `vllm/platforms/cpu.py:78`、`vllm/platforms/cpu.py:390` | 收窄 auto 绑核可见 NUMA node，并改变 rank->node 映射 |
| `OMP_NUM_THREADS` | `vllm/platforms/cpu.py:282`、`csrc/cpu/utils.cpp:125`、`vllm/v1/engine/input_processor.py:537` | OpenMP / Torch 线程数；在 bind 路径下会被重写 |
| `VLLM_CPU_SGL_KERNEL` | `vllm/model_executor/layers/utils.py:227`、`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:235` | linear / MoE / quantized scaled-mm 是否走 SGL kernel |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `vllm/envs.py:495`、`vllm/v1/worker/gpu_worker.py:85` | PyTorch float32 matmul 精度模式 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `vllm/platforms/cpu.py:276`、`vllm/utils/system_utils.py:151` | worker 进程创建方式，CPU 路径强制 `spawn` |
| `LOCAL_WORLD_SIZE` | `vllm/platforms/cpu.py:342` | 传给下层 TP / SHM all-reduce 相关 runtime |
| `VLLM_DIST_IDENT` | `vllm/v1/worker/cpu_worker.py:92`、`vllm/distributed/device_communicators/cpu_communicator.py:160` | CPU SHM collective 的实例名 |

### 2. `VLLM_CPU_KVCACHE_SPACE`

#### 事实
- `vllm/envs.py:700` 只有在环境变量存在时才把它解析为整数；不存在时返回 `None`，不是 `0`。
- `vllm/platforms/cpu.py:148` 到 `vllm/platforms/cpu.py:163` 把 `None` 分支解释为“按 NUMA 节点均分整机内存，再取 50%”；非 `None` 分支则按 “GiB 整数 * 2^30” 转成字节。
- `vllm/platforms/cpu.py:214` 把结果写入 `cache_config.cpu_kvcache_space_bytes`；`vllm/v1/worker/cpu_worker.py:117` 直接把它作为 worker 的可用 KV memory 返回。
- `vllm/v1/core/kv_cache_utils.py:841` 用 `available_memory // page_size // num_layers` 计算 KV block 数；`vllm/v1/core/kv_cache_utils.py:1550` 到 `vllm/v1/core/kv_cache_utils.py:1556` 再把所有 rank 的 `num_blocks` 收缩到最小值。
- `vllm/v1/core/kv_cache_utils.py:615` 到 `vllm/v1/core/kv_cache_utils.py:621` 对 `available_memory <= 0` 直接报错；因此显式 `export VLLM_CPU_KVCACHE_SPACE=0` 对常规自回归模型通常不是“默认值”，而是“没有 KV cache 空间”。
- 仓库文档存在版本差异：`docs/getting_started/installation/cpu.md:141` 写默认值是 `0`，`docs/getting_started/installation/cpu.md:241` 与 `docs/configuration/conserving_memory.md:83` 又写默认值是 `4GB`；但当前运行时代码以 `vllm/platforms/cpu.py:148` 到 `vllm/platforms/cpu.py:163` 为准。

#### 推断
- 对 `TP > 1`，如果某个 rank 所在 NUMA node 可用内存偏小，它会把所有 rank 的 KV block 上限一起拉低。

#### 建议
- 评估 `VLLM_CPU_KVCACHE_SPACE` 时，按“单 rank 的权重 shard + KV cache”是否能落进单 NUMA node 内存来算，不要按整机总内存来算。
- 做实验时区分三种状态：未设置、显式设置较小值、显式设置较大值；这三者不是一回事。

### 3. `VLLM_CPU_OMP_THREADS_BIND` / `VLLM_CPU_NUM_OF_RESERVED_CPU` / `CPU_VISIBLE_MEMORY_NODES`

#### 事实
- `vllm/envs.py:705` 让 `VLLM_CPU_OMP_THREADS_BIND` 默认值为 `auto`。
- `vllm/v1/worker/cpu_worker.py:56` 到 `vllm/v1/worker/cpu_worker.py:84` 把它分成三条路径：`auto`、`nobind`、手工字符串；手工字符串支持用 `|` 给不同 rank 分段。
- x86 的 `auto` 路径在 `vllm/v1/worker/cpu_worker.py:64` 到 `vllm/v1/worker/cpu_worker.py:68` 只取每个物理核的一个 SMT 线程。
- `vllm/platforms/cpu.py:390` 到 `vllm/platforms/cpu.py:395` 会先用 `CPU_VISIBLE_MEMORY_NODES` 过滤可见 NUMA node；`vllm/v1/worker/cpu_worker.py:151` 再按 `allowed_numa_nodes[self.local_rank]` 选当前 rank 对应节点。
- `vllm/v1/worker/cpu_worker.py:169` 到 `vllm/v1/worker/cpu_worker.py:181` 说明 `VLLM_CPU_NUM_OF_RESERVED_CPU` 只在 `auto` 选出 CPU 列表之后才生效；它通过裁掉列表尾部 CPU 的方式减少线程。
- `csrc/cpu/utils.cpp:25` 到 `csrc/cpu/utils.cpp:164` 会把最终 CPU 列表同时用于：NUMA memory migrate / membind / interleave、`omp_set_num_threads`、`torch::set_num_threads`、`sched_setaffinity`。

#### 推断
- 在 `2 x AMD EPYC 9745` 上，`NPS` 实际上会通过 `lscpu` 暴露的 NUMA 切分，改变 `CPU_VISIBLE_MEMORY_NODES` 可见集合和 auto 绑核的 rank->CPU 集，而不是只影响“内存远近”。

#### 建议
- 如果要手工绑核，优先显式写出每个 rank 的 CPU 集；如果走 `auto`，务必同时保存 worker 日志中的 `auto thread-binding list`。
- `VLLM_CPU_NUM_OF_RESERVED_CPU` 不适合只当“给系统留 1 个核”的装饰项；它会真实改变线程级并行宽度。

### 4. `OMP_NUM_THREADS`

#### 事实
- `vllm/platforms/cpu.py:282` 到 `vllm/platforms/cpu.py:288` 表明：只有在 `VLLM_CPU_OMP_THREADS_BIND == nobind` 时，CPU backend 才把 `OMP_NUM_THREADS` 的最终控制权留给外部环境。
- 一旦走 bind 路径，`csrc/cpu/utils.cpp:125` 到 `csrc/cpu/utils.cpp:128` 会把 OMP / Torch 线程数重设为绑定 CPU 列表长度。
- `vllm/v1/engine/input_processor.py:537` 到 `vllm/v1/engine/input_processor.py:545` 还会在输入预处理阶段读取 `OMP_NUM_THREADS`，临时设置 Torch 线程数。

#### 推断
- 同一个 `OMP_NUM_THREADS=64`，在 `nobind` 与 `auto` 两种模式下，实际线程数可能完全不同。

#### 建议
- 分析线程数或 CPI 时，不要只看 shell 里的 `OMP_NUM_THREADS`；要同时看 `init_cpu_threads_env` 打印出来的 `OMP tid -> core` 映射。

### 5. `VLLM_CPU_SGL_KERNEL` / `VLLM_FLOAT32_MATMUL_PRECISION`

#### 事实
- `vllm/model_executor/layers/utils.py:206` 到 `vllm/model_executor/layers/utils.py:212` 说明 SGL kernel 的 CPU 形状 / 类型前提是：AMX tile 支持、`dtype in {bfloat16, int8}`、`k % 32 == 0`、`n % 16 == 0`。
- `vllm/model_executor/layers/utils.py:227` 到 `vllm/model_executor/layers/utils.py:238` 让 unquantized linear 在条件满足时切到 `weight_packed_linear`。
- `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:234` 到 `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:252` 让 x86 MoE 在条件满足时切到 `SGLFusedMOE`。
- `vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py:41` 到 `vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py:50` 说明 quantized scaled-mm CPU 路径也会检查 `VLLM_CPU_SGL_KERNEL`。
- `vllm/envs.py:495` 到 `vllm/envs.py:499` 把 `VLLM_FLOAT32_MATMUL_PRECISION` 限制为 `highest` / `high` / `medium`；`vllm/v1/worker/gpu_worker.py:85` 到 `vllm/v1/worker/gpu_worker.py:87` 在 worker 初始化时调用 `torch.set_float32_matmul_precision(...)`，CPUWorker 会继承这条路径。

#### 推断
- 对 CPU backend 来说，`VLLM_CPU_SGL_KERNEL` 往往是“切 kernel 家族”的开关，而不是“同一 kernel 的小优化项”。

#### 建议
- 只要打开 `VLLM_CPU_SGL_KERNEL`，实验记录里就应补充模型层 shape / dtype；否则很难判断它是否真的命中 SGL 路径。

### 6. 平台自动写入的运行时变量

#### 事实
- `vllm/platforms/cpu.py:276` 强制把 `VLLM_WORKER_MULTIPROC_METHOD` 设为 `spawn`；`vllm/utils/system_utils.py:151` 到 `vllm/utils/system_utils.py:160` 会据此选择 multiprocessing context。
- `vllm/platforms/cpu.py:280` 写 `NUMEXPR_MAX_THREADS`，用于避免 numexpr 线程数报错。
- `vllm/platforms/cpu.py:291` 写 `TORCHINDUCTOR_COMPILE_THREADS=1`，禁用异步 compile 线程。
- `vllm/platforms/cpu.py:294` 写 `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`，CPU 上禁用 shared experts multi-stream。
- `vllm/platforms/cpu.py:301` 到 `vllm/platforms/cpu.py:307` 只有在 `LD_PRELOAD` 已包含 `libiomp5.so` 时才补写 `KMP_BLOCKTIME`、`KMP_TPAUSE` 和三类 barrier pattern。
- `vllm/platforms/cpu.py:342` 写 `LOCAL_WORLD_SIZE=<tensor_parallel_size>`；源码注释写明这是“给 IPEX 提示使用 shared memory based AllReduce”。
- `vllm/v1/worker/cpu_worker.py:92` 设置 `VLLM_DIST_IDENT`；`vllm/distributed/device_communicators/cpu_communicator.py:160` 到 `vllm/distributed/device_communicators/cpu_communicator.py:167` 用它拼 CPU SHM collective 的 group name。

#### 推断
- 这些变量多数不是给用户直接调优业务逻辑用的，而是给 multiprocessing / OpenMP / IPEX / SHM runtime 传递上下文。

#### 建议
- 除非在 debug multiprocessing 或 SHM 通信，不建议手工覆写 `VLLM_DIST_IDENT`、`LOCAL_WORLD_SIZE` 这类内部变量。

### 7. 当前不建议依赖的变量

#### 事实
- `VLLM_CPU_MOE_PREPACK` 在当前仓库只出现在 `vllm/envs.py:1753` 的导出名单里；未找到解析器，也未找到任何读取点。
- `VLLM_CPU_CI_ENV` 只在 `vllm/platforms/cpu.py:248` 到 `vllm/platforms/cpu.py:254` 作为 CPU CI 内部变量使用，用来把 compile backend 从 `inductor` 切到 `eager`。

#### 建议
- 对生产或性能实验，不要把 `VLLM_CPU_MOE_PREPACK` 作为正式调优变量使用；除非后续源码引入了真实读取点。
- `VLLM_CPU_CI_ENV` 仅适合测试 / CI 场景，不应混入正式性能结果。

## 机制解释

### 1. 为什么“未设置”和“设置为 0”会得到不同的 KV cache 行为
- `envs.py` 在变量不存在时返回 `None`；`cpu.py` 把 `None` 解释为“自动估算”，把整数 `0` 解释为“0 GiB”。
- 所以 shell 里“什么都不设”和 `export VLLM_CPU_KVCACHE_SPACE=0` 不是同一个配置。

### 2. 为什么 `OMP_NUM_THREADS` 经常不是最后的线程数
- CPU backend 先根据 `VLLM_CPU_OMP_THREADS_BIND` 选择绑核策略。
- 只要进入 bind 路径，`init_cpu_threads_env` 就会把线程数重写成 CPU 列表长度。
- 因此真正要看的不是环境变量文本，而是最终 `sched_setaffinity` 后的线程 -> core 映射。

### 3. 为什么 `CPU_VISIBLE_MEMORY_NODES` 会改变 rank 映射
- 它不是“只限制内存分配”的被动掩码。
- 在 auto 绑核路径里，它先改变 `allowed_numa_nodes` 的候选集合和顺序，后续 `local_rank` 直接按这个列表取节点。
- 所以它会同时影响：
  1. rank 的 CPU 集
  2. rank 的 NUMA memory policy
  3. TP worker 之间的自然分布方式

## 下一步实验

1. 固定 `TP=4`，对比三组：未设置 `VLLM_CPU_KVCACHE_SPACE`、`VLLM_CPU_KVCACHE_SPACE=8`、`VLLM_CPU_KVCACHE_SPACE=0`，确认 worker 初始化日志与 KV cache block 变化。
2. 固定 `OMP_NUM_THREADS=64`，切换 `VLLM_CPU_OMP_THREADS_BIND=auto/nobind/手工字符串`，验证实际 `OMP tid -> core` 映射是否一致。
3. 固定模型与 `TP`，切 `VLLM_CPU_SGL_KERNEL=0/1`，结合 `perf` 或 uProf 调用栈确认 linear / MoE 是否真的切到了 SGL kernel。
