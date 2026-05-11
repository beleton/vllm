# LLM 推理链路 Chiplet 优化空间扫描

生成时间：2026-05-11

## 目的

基于三篇 chiplet 论文（OLAP / Sorting / CHARM）提炼的三条筛选条件，扫描 vllm CPU backend 中 LLM 推理各链路是否存在 chiplet L3 局部性优化的有效空间。扫描方式是结合源码做理论分析，非实验平台验证。

## 筛选条件

1. **跨 chiplet 数据访问在关键路径上占比有意义的份额**——PCM/PMU 可测。三篇论文 workload 中跨 chiplet 通信占 15%–34%，CPU attention 仅 0.14%–0.81%。
2. **该访问能通过改变数据/计算的放置来消除**——"数据被哪个 chiplet 的线程访问"是可控变量。
3. **工作集在 L3 容量边界附近且存在复用**——数据在 L3 内被多次访问。单 chiplet L3 约 32 MB，聚合 L3 约 256–512 MB。

## 排除的链路

### 1. Decode Attention

**源码路径**：`vllm/v1/attention/backends/cpu_attn.py:280-382` → `ops.cpu_attention_with_kv_cache()` → `csrc/cpu/cpu_attn_impl.hpp` 中 `AttentionMainLoop::operator()`

**计算特征**：单 token、逐层扫描全 KV cache。每层 KV cache 体积 = seq_len × num_kv_heads × head_dim × 2 (K+V) × 2 bytes。32K seq × 8 heads = 131 MB/层，64 层共 8.4 GB/step。

**排除理由**：
- 条件 3 不满足：KV cache 远大于 L3 总容量（512 MB），数据必然从 DRAM streaming 流过。L3 是通道，不是复用缓存。CAT 0001 把 L3 压到 1/16 仅退化 0.03%–2.11%（已实验验证）。
- 条件 1 也不满足：跨 CCD demand fill 仅 0.14%–0.81%。

### 2. Prefill Attention

**源码路径**：同 decode attention，但调用 `torch.nn.functional.scaled_dot_product_attention` 或相同的 `ops.cpu_attention_with_kv_cache`，取决于 batch 组成。

**计算特征**：QK^T 产生 seq_len² 中间结果。q=4096 时 QK^T = 32 heads × 4096 × 4096 × 4 bytes ≈ 2 GB，远超 L3。Q、K、V 各读一次，K/V 是只读共享但每元素只读一次，coherence 副本扩散不被复用放大。

**排除理由**：条件 3 不满足。中间结果 > L3 容量，K/V streaming access 无复用。

### 3. GEMM / 线性层

**源码路径**：`vllm/model_executor/layers/utils.py:207-261`（`dispatch_cpu_unquantized_gemm`）→ `ops.onednn_mm()` → `csrc/cpu/dnnl_kernels.cpp:520-570` → oneDNN 内部 OpenMP 并行。

**计算特征**：
- Decode（M=1）：GEMV。输入激活 22 KB → L2。权重 225 MB（down_proj [11008, 10240] × 2 bytes）→ 远超单 chiplet L3。输出 20 KB → L2。
- Prefill（M=32768 for batch=16, seq=2048）：GEMM。输入激活 720 MB → 远超总 L3。权重同上。

**排除理由**：条件 3 不满足。Decode 的工作集要么 fit in L2（输入/输出），要么远超 L3（权重从 DRAM streaming 读取）。权重是只读的，多个 chiplet 读同一份权重不产生 coherence 写失效，开销主要是 DRAM 带宽。Prefill 的所有数据都 > 总 L3 容量，全部走 DRAM。

### 4. 层间激活传递

**源码路径**：`vllm/model_executor/models/qwen2.py:286-306`（`Qwen2DecoderLayer.forward`）。一层输出（`hidden_states`）直接作为下一层输入，中间经过 PyTorch 张量传递。

**计算特征**：
- Prefill：层输出 [32768, 11008] × 2 = 720 MB → 远超 L3。
- Decode：层输出 [1, 11008] × 2 = 22 KB → fit in L2。

**潜在问题**：不同层的 oneDNN 内部线程调度不同，层 N 的线程 chiplet A 把结果写入了 chiplet A 的 L3，层 N+1 的线程 chiplet B 需要读该数据时会产生跨 chiplet coherence。但由于 prefill 激活 > L3 容量（无论如何走 DRAM），decode 激活 < L2 容量（不涉及 L3），实际影响很小。

**排除理由**：条件 3 不满足。激活要么 > L3，要么 < L2。L3 在两个场景下都不参与复用。

### 5. KV Cache Reshape and Cache（写入阶段）

**源码路径**：`ops.cpu_attn_reshape_and_cache()` → `csrc/cpu/cpu_attn_vec.hpp:197-244`，`#pragma omp parallel for collapse(2)` 在 (token_idx, head_idx) 上并行。

**计算特征**：将新产生的 K/V 写入 KV cache 中对应 slot。每个 (token, head) 写入不重叠的内存位置。Key 列主序写入时，相邻 token 写入同一 block 的相邻列，可能产生 cache line 伪共享（64B 对齐边界）。

**排除理由**：写入量极小（decode: 2 KB/step，prefill: batch × seq_len × 8 heads × 256 bytes），占整体执行时间比例微不足道。主瓶颈在后续的 attention 读（8.4 GB/step），不在写入阶段。

### 6. TP AllReduce 通信

**源码路径**：`ops.shm_allreduce()` → `csrc/cpu/shm.cpp:828` → `all_reduce_sum_impl()`，使用 POSIX 共享内存（`shm_open` + `mmap`）跨进程通信。每线程 2 MB 共享内存缓冲区，双缓冲设计。

**计算特征**：Decode 时单 token allreduce 数据量为 num_heads × head_dim × 2 bytes = 8 KB（32 heads, 128 dim），远小于每线程 2 MB 缓冲区。是 barrier 同步操作，延迟绑定。

**排除理由**：条件 1/3 均不满足。数据必须跨进程交换——无法通过 chiplet 放置"消除"。buffer 大小远小于 L3 容量，但共享内存区域在 DRAM 中（`mmap`），不涉及 L3 复用。操作是延迟敏感的，chiplet 放置改变不了同步开销。

## 排除的链路汇总

| 链路 | 排除的关键条件 | 根因 |
|------|---------------|------|
| Decode Attention | 条件 1、3 | KV cache streaming from DRAM，CAT 实验已证 L3 不敏感 |
| Prefill Attention | 条件 3 | QK^T > L3，K/V 无复用 |
| GEMM/线性层 | 条件 3 | 工作集要么 < L2 要么 > L3 |
| 层间激活传递 | 条件 3 | 激活 > L3（prefill）或 < L2（decode） |
| KV Cache 写入 | 条件 1 | 写入时间占比极小 |
| TP AllReduce | 条件 1、2 | 数据必须跨进程交换，延迟绑定非带宽绑定 |

## 根因总结

LLM 推理（dense model，Qwen3-30B-A3B）的工作集呈现两极分化：

| 规模 | 典型数据 | 大小 | 所在缓存层级 | 复用？ |
|------|---------|------|------------|-------|
| 极小 | 单 token 激活、attention tile | < 1 MB | L2（~1 MB/core） | 无 |
| 极大 | KV cache / QK^T / 层间激活 | 100 MB–8 GB | 必须走 DRAM | 跨 step 有复用但被跨层冲刷 |

缺少的是**中等规模（10–100 MB）且被反复访问**的工作集——这正是 chiplet L3 优化的适用区间。三篇论文的 workload 都在这个区间内（排序 15–150 MB 数组反复扫描、OLAP 32–256 MB 数据反复 join/scan、图遍历 working set 在 cache 边界）。

## 未排除的边际机会（理论存在，收益预期小）

### 机会 A：短上下文 Batched Decode 的 KV Cache L3 驻留

seq_len = 256–1024，B=32 时单层 KV cache ≈ 128 MB（4 个 chiplet 聚合 L3 可容纳）。若将 requests 按 chiplet 分组，每组 requests 的 KV cache 可稳定在本地 L3。跨 step 间 KV cache 仅增 1 token，复用率高。

**限制**：上下文增长后失效（seq_len > 2K 时单层 KV cache 已超 256 MB），且跨层冲刷问题仍在。

**验证方法**：CAT resctrl 在 B=16–32、seq_len=256–1024 的 decode-only 场景测 runtime 变化。

### 机会 B：NPS/TP × Chiplet 协同放置

NPS2/4 配置下每个 TP rank 独占一个 NUMA domain（对应一组 chiplet）。当前 `get_auto_local_omp_cpuid()`（`vllm/v1/worker/cpu_binding.py:42-95`）已在做"每 rank 一个 NUMA node"，但未做 chiplet 级细粒度绑定。

**限制**：这是 NUMA 级优化，非 chiplet 级，novelty 有限。

**验证方法**：对比 NPS1 vs NPS2 vs NPS4 下 TP=2/4/8 的端到端吞吐。

## 可能的 pivot 方向

### Pivot 1：MoE 模型 + Chiplet-Aware Expert Placement

MoE 中每个 expert 的权重只有 dense 的几分之一。Expert 权重可在 chiplet 间做 affinity-aware 分区放置。当前 vllm 已有 CPU MoE 实现（`csrc/cpu/cpu_fused_moe.cpp`），`#pragma omp parallel for schedule(static, 1)` 用 atomic counter 动态抢任务，未考虑 chiplet 拓扑。

与 OLAP/Sorting 论文的方法对齐点：expert 权重 ↔ 数据 fragment，expert 调用分布 ↔ 访问模式，chiplet ↔ worker 绑定粒度。

### Pivot 2：切换到非 LLM 推理的应用场景

三篇论文覆盖的 OLAP/排序/图计算/SGD 等领域都有已证明有效的 chiplet 优化方法线，可以直接做增量工作。选择空间更大，风险更低。

## 证据

- 三篇 chiplet 论文解读：`wiki/论文解读/OLAP_on_Modern_Chiplet_Based_Processors解读.md`、`Optimizing_Sorting_for_Chiplet_Based_CPUs解读.md`、`CHARM解读.md`
- 三篇论文对照：`wiki/论文解读/Chiplet三篇论文对照与Attention结论验证.md`
- 2026-05-06 ACC 结论：`wiki/实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md`
- vllm 源码关键路径：
  - Attention: `vllm/v1/attention/backends/cpu_attn.py`、`csrc/cpu/cpu_attn_impl.hpp`、`csrc/cpu/cpu_attn_vec.hpp`
  - GEMM: `vllm/model_executor/layers/utils.py`、`csrc/cpu/dnnl_kernels.cpp`、`csrc/cpu/dnnl_helper.h`
  - 线程绑定: `csrc/cpu/utils.cpp`、`vllm/v1/worker/cpu_binding.py`
  - TP 通信: `csrc/cpu/shm.cpp`
  - 模型结构: `vllm/model_executor/models/qwen2.py`
  - MoE: `csrc/cpu/cpu_fused_moe.cpp`
