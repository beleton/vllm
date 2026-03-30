# 05. CPU 关键算子路径

> 更新时间：2026-03-17  
> 范围：只看 CPU backend 中 attention / linear / matmul / moe 的主路径与线程级工作划分。  
> 假设：当前讨论的是非量化或常见 CPU 路径；更细的量化分支（如特定 compressed kernels）未在本轮展开。

## 结论

### 事实
- attention 的 CPU 主路径是：`CPUAttentionBackend` -> `cpu_attn_get_scheduler_metadata` -> `cpu_attn_reshape_and_cache` -> `cpu_attention_with_kv_cache`，底层实现位于 `csrc/cpu/cpu_attn.cpp` 和 `csrc/cpu/cpu_attn_impl.hpp`。
- attention 在线程级不是简单“每线程一个 token”；调度器先在 `cpu_attn_get_scheduler_metadata` 里生成 workitem、KV split、归约信息，主循环再按线程消费这些 workitem。
- unquantized linear / matmul 的 CPU 分发顺序是：
  1. 若 `VLLM_CPU_SGL_KERNEL=1` 且 shape/dtype 符合，走 SGL `weight_packed_linear`；
  2. 否则若 oneDNN 可用，走 `create_onednn_mm` / `onednn_mm`；
  3. 否则退回 `torch.nn.functional.linear`。
- SGL GEMM 自己做 tile 级并行；oneDNN 路径的线程划分对 Python 层是黑盒，封装层只负责创建 primitive 和 execute。
- MoE 的 CPU 主路径在 `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`：
  - x86 + `VLLM_CPU_SGL_KERNEL` + shape 合适时，走 `SGLFusedMOE`；
  - 否则走 `CPUFusedMOE`；
  - `CPUFusedMOE` 内部再分成 grouped-gemm path 与 torch/oneDNN fallback path。
- `VLLM_CPU_MOE_PREPACK` 在当前仓库只出现在 `vllm/envs.py` 的导出列表里，未找到实际读取点；当前源码看不出它对 CPU 路径有真实影响。
- `VLLM_FLOAT32_MATMUL_PRECISION` 在 `vllm/v1/worker/gpu_worker.py::Worker.__init__` 中统一设置，`CPUWorker` 继承该逻辑，因此 CPU backend 也会受这个全局 PyTorch matmul 精度设置影响。

### 推断
- 对 CPU backend 来说，真正决定线程拆分方式的不是 Python 层模块名，而是底层具体选中的实现：`cpu_attention_with_kv_cache`、oneDNN primitive、SGL gemm、SGL moe。

### 建议
- 分析热点时先确认“最终走的是哪条内核路径”，再谈线程数和 NUMA。
- 若想稳定复现实验，应把 `VLLM_CPU_SGL_KERNEL` 和 oneDNN/ACL 支持状态一起记录；它们会改变最终 CPU kernel。

## 证据

## 1. Attention

### 路径
- Python 入口：`vllm/v1/attention/backends/cpu_attn.py::CPUAttentionBackendImpl.forward`
- C++ 入口：`csrc/cpu/cpu_attn.cpp`
  - `get_scheduler_metadata(...)`
  - `cpu_attn_reshape_and_cache(...)`
  - `cpu_attention_with_kv_cache(...)`
- 主循环：`csrc/cpu/cpu_attn_impl.hpp::AttentionMainLoop`

### 线程级工作划分
- `get_scheduler_metadata(...)` 会先算出：
  - `reduction_split_num`
  - `thread_num`
  - `effective_thread_num`
  - `split_kv_q_token_num_threshold`
  - `cu_workitem_num_per_thread`
- `AttentionMainLoop` 中使用 `#pragma omp parallel for schedule(static, 1)` 按 `thread_id` 启线程。
- 每个线程根据 `cu_workitem_num_per_thread` 取自己的 workitem 范围。
- 对长 KV 或需要分块归约的场景，会把一个 query 的 KV 范围拆成多个 split；各线程先产出 partial output，再通过 flag + atomic fence 等待和归约。

### 结论化解释
- attention 的线程划分是“调度器先离线切 workitem，线程再消费”，不是在主循环里临时平均分 token。
- 这类拆分天然会引入两种成本：
  1. 不同线程对 KV cache 的并发读；
  2. split reduction 的同步等待。

## 2. Linear / Matmul

### 路径
- 分发入口：`vllm/model_executor/layers/utils.py::dispatch_cpu_unquantized_gemm`
- Python 调用：`vllm/model_executor/layers/linear.py::UnquantizedLinearMethod.apply`
- oneDNN C++：`csrc/cpu/dnnl_kernels.cpp::create_onednn_mm_handler`、`onednn_mm`
- SGL GEMM：`csrc/cpu/sgl-kernels/gemm.cpp`

### 分发规则
- `dispatch_cpu_unquantized_gemm(...)`：
  - `VLLM_CPU_SGL_KERNEL=1` 且 `check_cpu_sgl_kernel(...)` 成立 -> `torch.ops._C.weight_packed_linear`
  - 否则 oneDNN 可用 -> `ops.create_onednn_mm(...)` + `ops.onednn_mm(...)`
  - 否则 fallback 到 `F.linear`

### 线程级工作划分
- oneDNN 路径：
  - Python 层只创建 handler 并调用 `ptr->execute(exec_args)`；
  - 线程级 tile/调度由 oneDNN primitive 内部处理；
  - 可直接受进程 `OMP_NUM_THREADS` 约束。
- SGL GEMM 路径：
  - `csrc/cpu/sgl-kernels/gemm.cpp` 按 `(MB, NB)` tile 划分；
  - 注释直接写明 `parallel on [MB, NB]`；
  - 再在 tile 内调用 `tinygemm_kernel`，必要时走 `brgemm`。

### 结论化解释
- linear/matmul 的线程级拆分方式并不唯一；必须先确认最终走的是 oneDNN 还是 SGL。
- 若同样的模型在两次实验里切换了这条分支，CPI / L3 miss 的可比性会很差。

## 3. MoE

### 路径
- Python 选择器：`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py::process_weights_after_loading`
- CPU 实现：`vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`
- SGL C++：`csrc/cpu/sgl-kernels/moe.cpp`

### 分发规则
- `SGLFusedMOE`
  - 条件：x86 + `VLLM_CPU_SGL_KERNEL=1` + 权重 shape 通过 `check_cpu_sgl_kernel(...)`
  - 最终调用 `torch.ops._C.fused_experts_cpu(...)`
- `CPUFusedMOE`
  - grouped-gemm path：权重先 `cpu_prepack_moe_weight(...)`，执行 `cpu_fused_moe(...)`
  - torch fallback path：为每个 expert 建 `gate_up_linear` / `down_linear`，内部可再选 oneDNN 或 `F.linear`

### 线程级工作划分
- gating / top-k：`cpu_fused_moe.py::select_experts` 先在 Python 中选 expert。
- token 排序与 padding：`csrc/cpu/sgl-kernels/moe.cpp`
  - `at::parallel_for(...)` 初始化 `sorted_ids` / `expert_ids` / `total_cnts`
  - `moe_align_block_size(...)` 生成按 expert 排序后的 token 块
- 计算阶段：
  - `at::parallel_for(0, MB * NB, ...)`
  - 每个线程通过 `at::get_thread_num()` 取得自己的私有 scratch buffer
  - 再按 expert block 做 gemm / silu_and_mul / down projection
  - 需要时调用 `brgemm`

### TP 同步点
- `vllm/model_executor/layers/fused_moe/layer.py::maybe_all_reduce_tensor_model_parallel`
  - 若当前 MoE kernel 没自己把 TP 结果 reduce 完，就追加一次 `tensor_model_parallel_all_reduce(...)`
- 所以 MoE 既有“进程内线程并行”，也可能有“TP 多进程 all-reduce”。

## 4. CPU 计算相关环境变量

| 变量 | 生效位置 | 影响 |
| --- | --- | --- |
| `VLLM_CPU_SGL_KERNEL` | `vllm/model_executor/layers/utils.py`、`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | 决定 linear / MoE 是否切到 SGL kernel |
| `OMP_NUM_THREADS` | `csrc/cpu/utils.cpp`、oneDNN/OpenMP runtime、`vllm/v1/engine/input_processor.py` | 影响 attention / gemm / MoE 的线程数 |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `vllm/v1/worker/gpu_worker.py::Worker.__init__` | 影响 PyTorch float32 matmul 精度策略 |
| `VLLM_CPU_KVCACHE_SPACE` | `vllm/platforms/cpu.py::get_device_total_memory` | 影响 CPU KV cache 容量 |
| `VLLM_CPU_MOE_PREPACK` | 仅见 `vllm/envs.py` 导出列表 | 当前源码未发现实际读取点 |

> 变量默认值、解析位置和 runtime 自动设置项见 `info/vllm_analysis/06_CPU环境变量与生效路径.md`。

## 机制解释

### 1. 哪些并行属于 TP 多进程级
- `ColumnParallelLinear` 的 all-gather
- `RowParallelLinear` 的 all-reduce
- MoE 最终 hidden states 的 TP all-reduce
- 这些都依赖 `get_tp_group()` / `tensor_model_parallel_*`

### 2. 哪些并行属于进程内线程级
- attention 的 OpenMP `thread_id -> workitem` 映射
- oneDNN primitive 内部线程划分
- SGL GEMM 的 `(MB, NB)` tile 并行
- SGL MoE 的 token 排序和 block 计算并行

### 3. 哪些地方存在同步和等待
- attention KV split reduction：`csrc/cpu/cpu_attn_impl.hpp`
- SHM collective：`csrc/cpu/shm.cpp`
- TP 层 all-reduce / all-gather：`linear.py`、`fused_moe/layer.py`

## 下一步实验

1. 先通过日志确认 linear 走 oneDNN 还是 SGL，再去解释 CPI 差异。
2. 对 attention 热点抓一次调用栈，确认时间是否落在 `cpu_attention_with_kv_cache` 而不是前后的 metadata 构建。
3. 对 MoE 模型额外记录 `top_k`、expert 数量和 `TP`，否则很难区分“线程内 block 计算”与“TP all-reduce”谁是主瓶颈。