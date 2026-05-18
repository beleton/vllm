# MoE Chiplet-Aware Expert Placement 优化分析

## 说明

本文是对"LLM 推理链路 Chiplet 优化空间扫描"中 Pivot 1（MoE 模型 + Chiplet-Aware Expert Placement）方向的详细展开。分析结论：**MoE 的 expert weight 与 dense attention 的 K/V 一样是 streaming access，每个 weight 元素在单次 forward pass 中仅被读一次，L3 不参与复用。chiplet 优化同样不适用。**

vllm CPU backend 已完整支持 MoE 推理，dispatch 路径如下（`vllm/model_executor/layers/fused_moe/`）：
- **C++ Grouped GEMM 路径**（优先）：`cpu_fused_moe.py:CPUFusedMOE.forward_grouped_gemm` → `csrc/cpu/cpu_fused_moe.cpp` 的 `cpu_fused_moe` kernel，以 atomic counter 调度 expert tile 任务
- **Per-expert oneDNN 路径**（fallback）：`CPUFusedMOE.forward_torch` → 每个 expert 独立调用 oneDNN GEMM
- **SGLang 路径**：`SGLFusedMOE` → `torch.ops._C.fused_experts_cpu`

即便如此，chiplet-aware expert placement 的收益预期极低——原因如下。

## 1. MoE 的计算流程与访存特征

### 1.1 MoE Layer 的四个阶段

以典型的 MoE transformer layer 为例（如 Qwen3-MoE、Mixtral 8×7B），MoE 层替代 dense MLP，处理流程为（`vllm/model_executor/layers/fused_moe/fused_moe.py`、`csrc/cpu/cpu_fused_moe.cpp`）：

**阶段 1：Router → 阶段 2：Token-to-Expert 重排 → 阶段 3：Per-Expert GEMM → 阶段 4：Weighted Sum**

阶段 3 是计算和访存的主体，`fused_moe_impl()` 的执行流程为：

1. 线程通过 atomic counter 抢任务，每个任务是一个 `(expert_id, output_tile_id)` 对（第 321-325 行）
2. 对于分配到的任务，将该 expert 的所有 token 的输入拷贝到 thread-local buffer（第 382-406 行）
3. 在 N 维度上分 subtile 遍历（第 419 行：`for (int32_t i = 0; i < actual_n_tile_size; i += min_w13_n_tile_size)`）：
   - `gemm.gemm(input_buffer, weight_subtile, ...)` —— 读当前 weight subtile，做 GEMM
   - `w13_weight_ptr += w13_n_tile_stride`（第 448 行）—— pointer 单调递增，指向下一个 subtile
4. `apply_gated_act(...)` —— 对 GEMM 输出做激活
5. 线程回到第 1 步，抢下一个任务

### 1.2 Expert Weight 的访问模式：Streaming，无复用

**关键代码证据在第 419-448 行**：

```cpp
// 遍历 N 维度的所有 subtile
for (int32_t i = 0; i < actual_n_tile_size; i += min_w13_n_tile_size) {
    gemm.gemm(curr_w13_input_buffer_iter, w13_weight_ptr_0_iter, ...);  // 读 weight_subtile[i]
    gemm.gemm(curr_w13_input_buffer_iter, w13_weight_ptr_1_iter, ...);  // 读 weight_subtile[i+1]
    w13_weight_ptr_0_iter += w13_n_tile_stride;  // 永不回头
    w13_weight_ptr_1_iter += w13_n_tile_stride;
}
```

每个 weight subtile 在单次任务中**被读恰好一次**。weight pointer 只向前移动，从不回头。一个任务完成后，线程抢下一个任务（可能是另一个 expert 或同一 expert 的另一个 output tile），但不会重读之前的 weight subtile。

**唯一的复用发生在 micro-GEMM 的 L2 内部**：同一个 weight subtile 加载到 L2 寄存器/tile buffer 中，用于处理该 expert 的多批 token（第 377 行的 `for (int32_t token_idx = 0; ...)` 内循环）。这个复用完全在 L2 内完成，L3 不参与。当 weight subtile 从 L2 被 evict 到 L3 时，不会再被访问。

### 1.3 为什么 L3 不参与复用

对比三篇论文中 chiplet 优化有效的 workload：

| | OLAP | Sorting | Graph | MoE (expert weight) |
|---|---|---|---|---|
| 数据被访问次数 | 多次（scan→repartition 重读） | 多次（多轮 pass 反复扫描） | 多次（frontier 邻居反复访问） | **1 次**（每个 weight subtile 被读一次后不再访问） |
| L3 的角色 | 复用缓存（两次访问之间缓存数据） | 复用缓存（pass 间缓存数据） | 复用缓存（遍历间缓存数据） | **Victim cache**（数据从 L2 evict 到 L3 后永不再读） |
| Chiplet 放置的收益 | 确保复用访问走本地 L3（24 ns 而非 106 ns） | 确保多轮 pass 走本地 L3 | 确保邻居访问走本地 L3 | **N/A——不存在复用访问** |

在三篇论文的 workload 中，同一份数据被多次访问，L3 在两次访问之间起到了缓存作用。Chiplet 放置确保"第二次访问"发生在同一个 chiplet 内，从而命中本地 L3。

在 MoE 中，没有"第二次访问"。每个 weight 元素在一次 GEMM 调用中被读一次后，就再也不需要了。Weight 从 DRAM 经过 L3（作为 victim cache）到达 L2，在 L2 中被反复使用处理多批 token，然后被 evict。**L3 永远是从 DRAM 到 L2 的单向通道，不是复用的缓存。** 这与 Attention 中 K/V 的访存模式在结构上一致。

### 1.4 多 chiplet 访问 weight 的实际 coherence 开销

未优化情况下，多个 chiplet 的线程可能处理同一 expert 的不同 output tile：
- chiplet 0 的线程处理 expert E 的 tile 0 → weight_tile_0_E 进入 chiplet 0 的 L2/L3
- chiplet 3 的线程处理 expert E 的 tile 1 → weight_tile_1_E 进入 chiplet 3 的 L2/L3

两者读的是同一 expert 的**不同 tile**（不同地址范围），不存在 cache line 重叠，不需要 coherence 消息。Weight 是只读的，即使不同 chiplet 读同一 tile（小概率，因为 atomic counter 动态抢任务），coherence 也只产生 probe（确认无 modified 副本），不存在 modified→shared 的 data transfer。

**跨 chiplet coherence 开销本身就很低，消除它换不回多少性能。** 即便做 chiplet-aware expert 分区让同一 expert 的所有 tile 只被同一 chiplet 处理，也改变不了"每个 weight 元素被读一次"的根本事实——数据仍然从 DRAM 来，L3 仍然只是 victim cache。

### 1.5 跨层冲刷

与 KV cache 相同的瓶颈：Layer 0 MoE 的 expert weight 进入 L3 后，被 Layer 1–63 的 MoE expert weight 和 attention K/V 冲刷。即使某一层的某个 expert weight 意外地留在 L3 中，等不到该层的下一次 forward pass（下一个 decode step 或下一个 prefill batch），就被清空了。

总数据量估计（以 DeepSeek-R1 细粒度 expert 为例）：
- 64 层 × 每层 2 个活跃 expert × 44 MB ≈ 5.6 GB weight 会经过 L3
- 加上每层 attention 的 K/V 访问（~8.4 GB/step for 32K context）
- **总数据量约 14 GB/step，对比总 L3 容量 512 MB（8 chiplet × 32 MB）或 256 MB（单 socket）**
- 数据量是 L3 容量的 27–55 倍，完全没有跨层驻留的可能

## 2. 排除结论

MoE expert weight 的 chiplet-aware placement 与 Attention 的 acc-local-l3 有相同的根因缺陷：**数据是 streaming access，L3 不参与复用。** 唯一的不同是：
- Attention 的 K/V 是**读写共享**（写产生 modified 状态，读需要 modified→shared 传输）→ 单次跨 chiplet 访问代价更大 → 但工作集 > L3，数据必然从 DRAM 来
- MoE 的 expert weight 是**只读共享**（只产生 probe）→ 单次跨 chiplet 访问代价更小 → 消除它换回的性能也更小 → 且同样受限于工作集 > L3

两者殊途同归：chiplet L3 级优化的前提条件不满足。

## 3. MoE 中可能有 chiplet 优化空间的地方（如果存在）

以下不是 expert weight placement，而是可能的其他角度，仅做记录：

- **All-to-All 通信**（expert parallelism 跨机器场景）：MoE 的 token dispatch 和 combine 涉及不同 expert 间的 all-to-all 通信。在单机多 chiplet 场景下，如果 expert 分布在多个 NUMA node 上，token 的跨 chiplet 搬运可能产生 coherence 开销。但当前 vllm CPU backend 不支持 expert parallelism。
- **Shared Expert**（部分 MoE 模型有共享 expert）：少量 expert 处理所有 token。共享 expert 的 weight 在每层被所有 token 访问，访问密度最高。如果单共享 expert weight < 32 MB（取决于模型），仍有讨论空间——但同样受限于 streaming access 无复用。

这两点的收益预期均低于 expert weight placement 本身（而 expert weight placement 已经被排除），不推荐作为主攻方向。

## 证据

- MoE C++ 实现：`csrc/cpu/cpu_fused_moe.cpp`（`fused_moe_impl`，第 142-663 行；关键循环第 419-451 行）
- MoE Python 层（GPU）：`vllm/model_executor/layers/fused_moe/fused_moe.py`
- 三篇芯片论文解读：`wiki/论文解读/OLAP_on_Modern_Chiplet_Based_Processors解读.md`、`Optimizing_Sorting_for_Chiplet_Based_CPUs解读.md`、`CHARM解读.md`
- LLM 链路扫描：`wiki/论文解读/LLM推理链路Chiplet优化空间扫描.md`
