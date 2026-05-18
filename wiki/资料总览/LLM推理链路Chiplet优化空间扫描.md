# LLM 推理链路 Chiplet 优化空间扫描

生成时间：2026-05-11

## 目的

基于三篇 chiplet 论文（OLAP / Sorting / CHARM）提炼的三条筛选条件，扫描 vllm CPU backend 中 LLM 推理各链路是否存在 chiplet L3 局部性优化的有效空间。扫描方式是结合源码做理论分析，非实验平台验证。

## 筛选条件

1. **跨 chiplet 数据访问在关键路径上占比有意义的份额**——PCM/PMU 可测。三篇论文 workload 中跨 chiplet 通信占 15%–34%，CPU attention 仅 0.14%–0.81%。
2. **该访问能通过改变数据/计算的放置来消除**——"数据被哪个 chiplet 的线程访问"是可控变量。
3. **工作集在 L3 容量边界附近且存在复用**——数据在 L3 内被多次访问。单 chiplet L3 约 32 MB，聚合 L3 约 256–512 MB。

## 附：工作集大小的判定方法

工作集不是一个绝对值，而是取决于"算子的访存模式 + 缓存层级 + 复用距离"三者共同决定的结果。对于 chiplet L3 优化来说，关键问题是：**在 L3 的存活时间窗口内，哪些数据会被多次访问？这些数据总量是否在 L3（单 chiplet 或聚合）的容量边界附近？**

下面分别分析三个场景中工作集的判定逻辑，说明为什么同样的概念框架可以统一解释 attention 的"不敏感"和 OLAP/Sorting 的"敏感"。

### 定义框架

对于给定缓存层级，工作集 ≈ 在**复用间隔**内必须驻留的数据总量。这里的"复用间隔"指：数据第一次被访问到下一次被访问之间，计算会触碰多少其他数据。如果复用间隔内触碰的数据总量超过了该级缓存的容量，数据在复用前就会被 evict（容量 miss）；即使容量够，如果复用间隔内的访问模式导致冲突（associativity miss），也可能 evict。

因此判断一个 workload 对 L3 是否敏感，需要回答三个子问题：

1. **有没有复用**——同一份数据在一个有意义的短时间窗口内被多次访问？
2. **复用距离落在哪个缓存层级**——复用间隔内触碰的其他数据量是几 KB（L1 级）、几百 KB–几 MB（L2 级），还是几十 MB（L3 级）？
3. **复用数据的总量是否在目标缓存容量的边界附近**——如果复用数据总量远小于缓存，无优化空间（本来就全命中）；如果远大于缓存，也无优化空间（无论如何都命中不了）。

### Attention：分块计算将工作集压到了 L2 以内

**算子流程**（`csrc/cpu/cpu_attn_impl.hpp` 中 `AttentionMainLoop`）：

对每个 attention 操作，将 Q 沿 sequence 维度切成 tile（Q_tile），将 K/V 沿 sequence 维度切成 chunk（K_chunk/V_chunk），每个线程一次处理一个 (Q_tile, K_chunk) 对：

```
for each Q_tile (size: tile_q × head_dim):
    for each K_chunk (size: tile_kv × head_dim):
        S = Q_tile @ K_chunk^T          // [tile_q, tile_kv]
        S = softmax(S * scale)
        O_tile += S @ V_chunk            // [tile_q, head_dim]
```

**Tile 大小由 L2 容量决定**（`cpu_attn_impl.hpp` 中 `calcu_default_tile_size`）：

```cpp
const int64_t available_cache_size = cpu_utils::get_available_l2_size();
const int32_t default_tile_size = AttentionScheduler::calcu_default_tile_size(
    available_cache_size, head_dim, ...);
```

`get_available_l2_size()` 返回单核 L2 大小（典型 1 MB）。`calcu_default_tile_size` 的计算逻辑是：确保 Q_tile + K_chunk + S(score) 三者同时放得进 L2。以 head_dim=128、bf16 为例：Q_tile [4, 128] + K_chunk [2048, 128] + S [4, 2048] ≈ 1 KB + 512 KB + 16 KB ≈ 530 KB，在典型 L2（1 MB/core）内。

**工作集判定**：

在任一时刻，每个线程的工作集 = **当前 tiles 之和**，约 500 KB–1 MB。这份数据在 tile 计算期间被反复访问（Q_tile 的每行要与整个 K_chunk 的所有行逐次相乘累加）。复用发生在 L1/L2 内——Q_tile 元素在 K_chunk 遍历期间被反复读取数百次，K_chunk 行被至少 tile_q 次读取。

这个工作集完全 fit in L2。L3 里虽然可能残留之前 tile 的 K_chunk 副本（来自不同请求或同一请求的其他 tile），但后续不会再被复用——attention 的 KV 访问是 streaming 的，每个 K_chunk 只被当前 tile 消费一次，下一个 tile 需要的是下一个 K_chunk。

**L3 为什么测出不敏感**：因为复用发生在 L2 级的 tile 内，L3 只是 K_chunk 从 DRAM 到 L2 的通道。压 L3 容量不会增加当前 tile 的 eviction——tile 本来就在 L2 里。CAT 0001 把 L3 从 32 MB 压到 2 MB，只是减少了"通道"的缓冲深度，不影响 L2 内的 tile 复用。这就是为什么 q8192+ 仅退化 0.03%–2.11%。

**结论**：attention 的工作集 = 当前 tile ≈ 500 KB–1 MB，处于 L2 域。L3 不是复用缓存，因此 chiplet 级 L3 放置策略无效。

### Sorting：分块之间通过分区缓冲传递状态，复用发生在 L3 级

**算子流程**（LSB Radix Sort，每轮 pass）：

```
1. 扫描全数组 → 直方图统计（256 桶）
2. 前缀和 → 计算每桶输出偏移量
3. 再次扫描全数组 → 按桶号写入输出缓冲区
```

Radix Sort 确实也分块——每个线程扫描输入数组的一个 chunk，统计局部直方图。但是：

- 每个线程的局部直方图（256 × 8 bytes × num_threads ≈ 64 threads × 2 KB = 128 KB）和各线程的输出缓冲之间**不是独立的关系**——线程 A 写入桶 0 的位置和线程 B 写入桶 0 的位置是由全局前缀和决定的，它们写入的是**同一个输出数组的相邻区域**。
- 第 N 轮 pass 的输出 = 第 N+1 轮 pass 的输入。pass 之间，数据布局被全面重组（按桶号重新排列）。

**工作集判定**：

关键区别在于 pass 之间。pass N 的输出缓冲区是 pass N+1 的输入。对于 150 MB 的输入数组，pass N 产生 150 MB 输出，pass N+1 要从这 150 MB 中再次扫描。

- 如果 pass N 的线程全部在同一个 chiplet（Chiplet_Local）：150 MB 的输出全部写入 chiplet 0 的 L3 → L3 只有 32 MB → 118 MB 溢出到 DRAM。pass N+1 读输入时，只有最近写入的 32 MB 在 L3，其余 118 MB 从 DRAM 读 → 慢。
- 如果 pass N 的线程分散在 8 个 chiplet（Chiplet_Mixed）：每个 chiplet 写 ~19 MB 的输出到本 chiplet L3 → 19 MB < 32 MB，全部留在 L3。pass N+1 每个 chiplet 的线程读本 chiplet L3 内的 19 MB → 全部 L3 命中。

**复用发生在**：pass N 写入的数据，被 pass N+1 读取。复用距离 = 一轮 pass 的计算时间（毫秒到秒级），远大于 L2 的存活时间，但在 L3 的存活时间内。复用的数据总量 = 每个 chiplet 负责的分区大小。当这个量在 32 MB 边界附近时，Chiplet_Local vs Chiplet_Mixed 的选择就决定了 L3 命中率。

**attention 与 sorting 的核心不同**：attention 的 tile 之间**没有状态传递**——tile 0 的 K_chunk 处理完后，tile 1 不需要 tile 0 的任何结果。数据流是单向的：DRAM → L3 → L2（tile 复用）→ 丢弃。Sorting 的 pass 之间**有状态传递**——pass N 整个输出被 pass N+1 读回。数据流是循环的：DRAM → L3（pass N 写入）→ pass N+1 从 L3 读回。这个循环中的"同时驻留数据量"才是 L3 工作集。

**结论**：sorting 的工作集 = 各 chiplet 在 pass 间需同时驻留的分区数据总量。当此量在 32 MB（单 chiplet L3）边界附近时，chiplet 放置策略可以决定它走本地 L3 命中还是 DRAM。

### OLAP：两个独立机制——Coherence 消除 + L3 局部性

**算子流程**（Hash Join）：

```
Build 阶段：扫描 Build 表 → 构建 hash table
Probe 阶段：扫描 Probe 表 → 对每行查 hash table → 匹配 join
```

**WICP 的收益来自两个独立机制，不可混为一谈**：

**机制 A——消除跨 chiplet cache coherence 流量（与数据大小无关）**：

WICP 下每个 chiplet 跑一个独立 worker 进程，各进程有独立的虚拟地址空间。chiplet A 的 worker 线程写的数据只进 chiplet A 的 L3；chiplet B 的 worker 看不到这段地址，硬件 coherence 协议不会自动去 snoop/probe/invalidate。跨 worker 的数据交换走网络栈（同机 TCP loopback），是 MB 级的批量传输，不是数百万次 64 字节 cache line 的细碎 coherence 消息。

WIM 则相反——全部线程跑在同一进程内共享地址空间。chiplet A 上的线程写数据，chiplet B 上的线程随后读同一地址，硬件 coherence 自动把 cache line 跨 IF 拖过来。64 个线程 × 16 个 chiplet 的交叉访问每秒产生数百万次 coherence 消息，Infinity Fabric 被这些细碎往返占满。论文报告 SparkSQL WIM 下互联拥塞占执行时间 34%。

**机制 B——L3 局部性（仅在数据量接近 L3 边界时有效）**：

即 WICP_Local vs WICP_Mixed 的选择。论文在 STREAM benchmark 上用 6 MB 到 38 GB 的数组标定了 L3 容量边界（Fig. 10）：数组 < 32 MB 时 WICP_Local 带宽更高（全部命中本地 L3，24 ns）；32–256 MB 时 WICP_Mixed 更优（聚合 L3 容纳）；> 256 MB 时两者趋同到 DRAM 带宽，无差异。

**大数据（几 GB hash table）下 WICP 为什么还有效**：

机制 B 失效（数据远超聚合 L3），但机制 A 仍然有效。Fig. 7 中 Presto/SparkSQL 在 TPC-H SF100（100 GB 数据库）上 2.61x–6.97x 的 speedup，主要驱动力是消除 coherence 风暴，而非 L3 缓存命中。Q13 的 Orders 表扫描从 5.36s 降至 0.4s、Repartition 从 5.06s 降至 0.25s——这些数据远超 L3，但消除了 WIM 下线程间交叉访问产生的 coherence 拥塞后，IF 带宽从"碎片化"恢复到"可被有效利用"。

机制 B 只解释了 WICP_Local 与 WICP_Mixed 之间的差异（SingleStore 的 3.32x vs 1.85x），以及为什么这种差异在 Presto/SparkSQL 上缩小了（数据超出聚合 L3，两种放置的 L3 效果趋同）。

**复用和 coherence 的双重效应**：

当 hash table 同时满足"在 L3 边界附近"和"被反复随机访问"时，两种机制叠加：机制 A 消除了跨 chiplet 的 coherence 碎片，机制 B 让剩余的访问尽可能走本地 L3。当 hash table 远超 L3 时，只有机制 A 有效，但单独机制 A 仍然可以带来显著收益（因为 WIM 下的 coherence 风暴本身就是一个巨大的性能税）。

**与 attention 的核心不同**：

Attention 既不走机制 A 也不走机制 B：
- 机制 A 不可用：attention 的并行模式是线程各自处理不同的 head/tile，线程间几乎不访问对方写的数据。PCM 数据显示跨 CCD demand fill 仅 0.14%–0.81%，说明 coherence 风暴在 attention 中不存在——WIM 的"税"在这里本来就没交。
- 机制 B 不可用：tile 在 L2 内复用，L3 只是通道。

**结论**：OLAP 的有效性取决于 coherence 消除 + L3 局部性的叠加效应。大数据场景下只有前者有效，但单独前者已经值得做。论文实验数据规模：TPC-H SF100 = 100 GB 数据库，STREAM 微基准 6 MB–38 GB。

### 三者对比

|               | Attention (vLLM)        | Sorting                               | OLAP                                       |
| ------------- | ----------------------- | ------------------------------------- | ------------------------------------------ |
| 分块？           | 是，tile 按 L2 容量切         | 每个线程扫描输入 chunk，但各 chunk 的输出通过分区逻辑相互关联 | 是，TableScan 按 page，但 hash table 是全局结构      |
| 复用对象          | 当前 tile 内 Q 行与 K 列的反复乘加 | pass N 的输出 → pass N+1 的输入             | hash table（Build → Probe）、partition buffer |
| 复用距离          | L2 级（tile 内，微秒）         | L3 级（pass 间，毫秒到秒）                     | L3 级（Scan → Probe，可更长）                     |
| 复用数据总量        | ~500 KB–1 MB（单 tile）    | 几十 MB–150 MB（分区数据）                    | 几十 MB–几 GB（hash table + buffer）            |
| 所在缓存层级        | L2                      | L3（边界附近）                              | L3（边界附近）                                   |
| L3 敏感？        | 否（tile 在 L2 复用，L3 只是通道） | 是（pass 间复用数据量在 L3 边界）                 | 部分（hash table 在 L3 边界时叠加 L3 收益，超出后仅剩 coherence 消除） |
| 跨 chiplet coherence 风暴？ | 否（线程各自处理不同 head/tile，几乎不交叉访问对方数据，跨 CCD fill 仅 0.14%–0.81%） | 是（shuffling 和 pass 间数据重组产生大量跨 chiplet 访问） | 是（WIM 下多 chiplet 线程交叉访问共享内存，互联拥塞占 34% 执行时间） |
| Chiplet 放置有效？ | 否（coherence 风暴本就不存在，L3 局部性也无杠杆） | 是 | 是（coherence 消除在所有数据规模下有效，L3 局部性在边界附近叠加收益） |

### 通用判定方法

对一个算子判断工作集大小和 chiplet 优化空间时，不应只看"总共要处理多少数据"，而应按以下步骤：

1. **识别复用**——哪些数据在计算过程中被访问超过一次？画出数据流图，标出读出-再读的路径。
2. **度量复用距离**——两次访问之间，计算逻辑触碰了多少**其他**数据（以字节计）？这决定了复用能在哪级缓存存活。
3. **估算复用数据总量**——在同一级复用距离内，一共有多少数据需要同时驻留？这个量是否接近目标缓存（如单 chiplet L3 32 MB、聚合 L3 256 MB）的容量边界？
4. **确定瓶颈层级**——如果复用距离在 L2 以内且复用数据量 < L2，则瓶颈不在 L3；如果复用距离在 L3 以内且复用数据量在 L3 容量边界附近，则 L3 是瓶颈层级，chiplet 放置可介入。
5. **用 CAT 或类似实验验证**——人为限制目标缓存容量，观察性能退化。退化显著 → 确认该缓存层级是瓶颈；退化可忽略 → 瓶颈不在该层级。

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
| TP AllReduce | 条件 1、2 | 数据必须跨进程交换，non-temporal store 绕过 cache |
| MoE Expert Weight | 条件 3 | Weight 元素每 forward pass 只读一次，streaming access 无复用 |
| SHM / Reduction / Counter / 层间流（通信） | 条件 1、3 | 数据量要么极小（amortized），要么绕过 L3（non-temporal store） |

### 7. MoE Expert Weight

**源码路径**：CPU 有完整 MoE 支持，dispatch 路径为 `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py:CPUFusedMOE`：
- **C++ Grouped GEMM 路径**（优先）：`forward_grouped_gemm` → `csrc/cpu/cpu_fused_moe.cpp` 的 `fused_moe_impl`，`#pragma omp parallel for schedule(static, 1)` + atomic counter 调度 expert tile 任务
- **Per-expert oneDNN 路径**（fallback）：`forward_torch` → 每个 expert 独立 oneDNN GEMM

**计算特征**：每个 expert weight subtile 在单次 forward pass 中被读恰好一次——代码第 419 行 `for (int32_t i = 0; i < actual_n_tile_size; i += min_w13_n_tile_size)` 中 weight pointer 单调递增（第 448 行），从不回头。唯一的复用是同一 weight subtile 在 micro-GEMM 的 L2 内被复用处理多批 token（第 377 行内循环）。细粒度 expert 模型中单 expert weight 约 44 MB（DeepSeek-R1），但 32 MB L3 装不下，且 streaming access 无多 pass 复用。

**排除理由**：条件 3 不满足。每个 weight 元素被读一次后永不再访问。L3 只是 victim cache（DRAM → L2 通道），不是复用缓存。与 attention 相同的 streaming 缺陷。详细分析见 [MoE_Chiplet_Expert_Placement优化分析](MoE_Chiplet_Expert_Placement优化分析.md)。

### 8. 层间/算子间通信（完整分析）

LLM 推理中存在四种通信模式。逐一分析，均不满足 chiplet L3 优化的前提。

**模式 8a：TP SHM AllReduce**（`csrc/cpu/shm.cpp`）

SHM 通信已在代码层面最优处理：写入用 non-temporal store（`_mm512_stream_si512`，bypass cache 直接到 DRAM）、读取用 streaming load（第 446 行 `INT8Vec64 data(true, ...)` 的 `true` 参数即 non-temporal load，不污染 L3）。数据全程走 DRAM 通道，不经过 L3 也不触发 cache coherence。在同一 NUMA node 内（NPS1），所有 CCD 通过同一个 I/O die 访问 DRAM，CCD 间延迟无差异。通信量：decode 时 KB 级（8 KB–22 KB per token），prefill 时 linear in batch×seq_len（batch=16, seq=4096 时约 1.4 GB per allreduce，但 non-temporal store 决定了它不走 L3）。

**结论**：芯片拓扑对此路径无影响。CCD 放置改变不了 SHM 通信的代价。

**模式 8b：Attention KV Split Reduction**（`csrc/cpu/cpu_attn_impl.hpp`）

当单 request 的 KV 序列超出单线程处理能力时，scheduler 将其切为多个 split，各线程处理一部分 KV 范围，最后由 reduction 线程做 online softmax 重缩放。Reduction 线程读其他线程的 partial results 和 spin-flag，如果相关线程跨 CCD，会产生 coherence 读取（modified→shared 的 cache line 传输）。

**结论**：数据量极小（每 split 约 hidden_dim × 4 bytes = 44 KB for Qwen3-30B-A3B），且 split 只在长序列时触发。即使把 split 限制在同 CCD 内，收益测不出来。

**模式 8c：Atomic Counter Contention**（`csrc/cpu/cpu_attn_impl.hpp`、`csrc/cpu/cpu_fused_moe.cpp`）

Attention 和 MoE kernel 均用 `alignas(64) cpu_utils::Counter` 做动态任务抢取。所有 CCD 的线程都在同一 counter 上做 fetch-and-add，counter 所在的 cache line 在多个 CCD 的 L2/L3 之间 bounce。每次 bounce 涉及 cache line 在 IF 上的传输。

**结论**：每个 task 涉及数十万次乘加运算（attention tile: ~tile_q × tile_kv × head_dim × 2 ≈ 4 × 2048 × 128 × 2 ≈ 2M FLOPS for QK^T alone），一次 counter 访问的 cache line bounce 成本被摊销在大量计算上。在 attention 中，任务数 = KV heads × tile groups（通常数百到数千），总 bounces 控制在千次量级。在 MoE 中，任务数 = experts × tiles per expert（256 experts × ~10 tiles = 2560），同样被 GEMM 计算摊销。开销可忽略。

**模式 8d：层间激活传递（Operator-to-Operator Data Flow）**

层 N 的 GEMM 输出被层 N+1 的 GEMM 读取。如果两个层的 oneDNN 内部线程调度将线程放在了不同 CCD 上，层 N 输出中由 CCD A 的线程写入的 cache lines（modified 状态）被层 N+1 中 CCD B 的线程读取时，需要 coherence 做 modified→shared 传输。但 decode 激活仅 22 KB（fit in L2），prefill 激活 720 MB（远超 L3，全部从 DRAM 走）。无论哪种情况，L3 都不参与复用。

**结论**：与层间激活的排除结论一致。L3 不是层间数据流的关键缓存层级。

**通信模式汇总**：

| 通信模式 | 数据量 | 主要通道 | CCD 放置是否影响？ | Chiplet 优化空间 |
|----------|--------|---------|-------------------|-----------------|
| TP SHM AllReduce | Decode: KB / Prefill: MB–GB | DRAM（non-temporal store + streaming load） | 否 | 无 |
| KV Split Reduction | ~44 KB per split | L2/L3 coherence | 极小影响 | 测不出收益 |
| Atomic Counter | 1 cache line per acquire | L2/L3 coherence (bounce) | 极小影响 | Amortized |
| 层间激活传递 | Decode: 22 KB / Prefill: 720 MB | Decode: L2 / Prefill: DRAM | 极小影响 | 无 |

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

### Pivot 1：切换到非 LLM 推理的应用场景

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
  - MoE: `csrc/cpu/cpu_fused_moe.cpp`、`vllm/model_executor/layers/fused_moe/cpu_fused_moe.py`、[MoE_Chiplet_Expert_Placement优化分析](MoE_Chiplet_Expert_Placement优化分析.md)
