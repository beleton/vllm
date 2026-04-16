# CPU 关键算子路径

## 问题
- CPU backend 中 attention / linear / matmul / MoE 各自的主路径是什么，线程级并行大致由谁控制。

## 结论
- attention 的 CPU 主路径是：`CPUAttentionBackend -> cpu_attn_get_scheduler_metadata -> cpu_attn_reshape_and_cache -> cpu_attention_with_kv_cache`，底层实现位于 `csrc/cpu/cpu_attn.cpp` 和 `csrc/cpu/cpu_attn_impl.hpp`。
- attention 在线程级不是“每线程一个 token”；调度器先在 `cpu_attn_get_scheduler_metadata` 里生成 workitem、KV split、归约信息，主循环再按线程消费这些 workitem。
- unquantized linear / matmul 的 CPU 分发顺序是：
  1. 若 `VLLM_CPU_SGL_KERNEL=1` 且 shape/dtype 符合，走 SGL `weight_packed_linear`
  2. 否则若 oneDNN 可用，走 `create_onednn_mm` / `onednn_mm`
  3. 否则退回 `torch.nn.functional.linear`
- SGL GEMM 自己做 tile 级并行；oneDNN 路径的线程划分对 Python 层是黑盒，封装层只负责创建 primitive 和 execute。
- MoE 的 CPU 主路径在 `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`：
  - x86 + `VLLM_CPU_SGL_KERNEL` + shape 合适时走 `SGLFusedMOE`
  - 否则走 `CPUFusedMOE`
  - `CPUFusedMOE` 内部再分成 grouped-gemm path 与 torch/oneDNN fallback path
- `VLLM_CPU_MOE_PREPACK` 在当前仓库只出现在 `vllm/envs.py` 的导出列表里，未找到实际读取点；当前源码看不出它对 CPU 路径有真实影响。
- `VLLM_FLOAT32_MATMUL_PRECISION` 在 `vllm/v1/worker/gpu_worker.py::Worker.__init__` 中统一设置，CPU backend 也会受这个全局 PyTorch matmul 精度设置影响。

## Attention
- Python 入口：`vllm/v1/attention/backends/cpu_attn.py::CPUAttentionBackendImpl.forward`
- C++ 入口：
  - `get_scheduler_metadata(...)`
  - `cpu_attn_reshape_and_cache(...)`
  - `cpu_attention_with_kv_cache(...)`
- 主循环：`csrc/cpu/cpu_attn_impl.hpp::AttentionMainLoop`
- 线程级工作划分：
  - 先计算 `reduction_split_num`、`thread_num`、`effective_thread_num`、`cu_workitem_num_per_thread`
  - OpenMP 按 `thread_id` 启线程
  - 每个线程根据 `cu_workitem_num_per_thread` 取自己的 workitem 范围
  - 对长 KV 或需要分块归约的场景，会先产出 partial output，再做 reduction

## Linear / Matmul
- 分发入口：`vllm/model_executor/layers/utils.py::dispatch_cpu_unquantized_gemm`
- Python 调用：`vllm/model_executor/layers/linear.py::UnquantizedLinearMethod.apply`
- oneDNN C++：
  - `csrc/cpu/dnnl_kernels.cpp::create_onednn_mm_handler`
  - `onednn_mm`
- SGL GEMM：
  - `csrc/cpu/sgl-kernels/gemm.cpp`

## MoE
- Python 选择器：`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py::process_weights_after_loading`
- CPU 实现：`vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`
- SGL C++：`csrc/cpu/sgl-kernels/moe.cpp`
- 计算阶段既有进程内线程并行，也可能在末尾追加 TP all-reduce

## 分析
- 对 CPU backend，真正决定线程拆分方式的不是 Python 层模块名，而是最终选中的底层实现：`cpu_attention_with_kv_cache`、oneDNN primitive、SGL gemm、SGL MoE。
- 如果两次实验切换了 oneDNN / SGL / fallback 路径，即使模型和 batch 看起来一样，CPI / L3 miss 也可能不可直接比较。
- attention 的线程成本主要来自两类：
  - 多线程并发读 KV cache
  - split reduction 的同步等待

## 证据
- 关键源码路径：
  - `vllm/v1/attention/backends/cpu_attn.py`
  - `csrc/cpu/cpu_attn.cpp`
  - `csrc/cpu/cpu_attn_impl.hpp`
  - `vllm/model_executor/layers/utils.py`
  - `vllm/model_executor/layers/linear.py`
  - `csrc/cpu/dnnl_kernels.cpp`
  - `csrc/cpu/sgl-kernels/gemm.cpp`
  - `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
  - `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`
  - `csrc/cpu/sgl-kernels/moe.cpp`
  - `vllm/v1/worker/gpu_worker.py`
  - `vllm/envs.py`

## 边界
- 本页只看 CPU backend 中 attention / linear / matmul / MoE 的主路径与线程级工作划分。
- 更细的量化分支和特殊 compressed kernels 不在本轮展开。

## 后续验证点
- 分析热点前，先确认 linear / MoE 是否真的切到了 SGL 或 oneDNN，再谈线程数和 NUMA。
