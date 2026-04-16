# GPU Attention NUMA Effects 解读
## 问题
- 在采用 chiplet / multi-die 组织方式并显式暴露 NUMA 特性的 GPU 上，FlashAttention2 这类 attention kernel 若忽略物理拓扑，会不会因为工作块映射不当而显著损失缓存复用和性能。

## 核心内容
- 论文把共享同一组 `K/V` 的 workgroup 集合定义为 `ACC (Attention Compute Cluster)`。
- 论文的关键优化不是改 attention 公式，而是让同一 `ACC` 尽量落在同一 `XCD`，提高本地 `L2` 复用并减少重复 HBM 读取。
- 其核心映射方法是 `Swizzled Head-first Mapping`：同时按 head-first 顺序执行，并通过 workgroup ID 重映射，把同一 head / 同一 ACC 的 blocks 尽量收敛到同一个 `XCD`。

## 背景与问题意识
- 论文强调现代 AI GPU 正在从“统一缓存、统一访问代价”的架构，走向“分布式缓存、多 memory controller、多 chiplet”的架构。
- 在这种架构下：
  - 本地访问更快，本地带宽更高
  - 跨 die、跨 chiplet 的远程访问更慢
  - 某个 die 上的 `L2` 不能天然服务另一个 die 上的计算
  - 同一份数据如果被多个 die 分别加载，会造成缓存碎片化和 HBM 重复访存
- 论文要回答的问题不是重新发明 attention 算法，而是：对于已经很高效的 FlashAttention2，如果把 workgroup 映射方式改成 NUMA-aware，是否还能显著提速。

## 方法

### 1. 共享工作集的定义：ACC
- 论文先分析 FlashAttention2 的 tiled 执行结构。
- 对一个 attention head 而言，`Q` 会被切成多个 row blocks，每个 workgroup 负责其中一块；但这些 workgroup 都要访问同一整组 `K/V`。
- 因此真正值得放进同一 NUMA 域、争取本地 `L2` 复用的，不是“相邻 block”本身，而是共享同一输入张量的那一组 workgroups。
- 作者把这类集合定义为 `ACC`：
  - 在 `MHA` 中，一个 head 基本对应一个 `ACC`
  - 在 `GQA` 中，多个 query heads 共享一组 `K/V`，因此一个 group 对应一个更大的 `ACC`

### 2. 四种映射方式

#### Naive Block-first
- 先处理所有 head 的 block `0`，再处理所有 head 的 block `1`，依此类推。
- 问题：
  - 同一 `ACC` 会被拆散到多个 `XCD`
  - 同一份 `K/V` 可能在多个 `XCD` 的 `L2` 里重复加载

#### Swizzled Block-first
- 保留 block-first 的遍历顺序，但通过 swizzle 让一些 logically related 的 workgroup 尽量落到同一个 `XCD`。
- 对 `GQA` 场景可能有效，尤其是 group 数量刚好匹配 `XCD` 数量时。
- 但它对 `MHA` 不够稳定，也无法覆盖所有 ACC / XCD 拓扑不匹配的情况。

#### Naive Head-first
- 先做完一个 head 的所有 row blocks，再做下一个 head。
- 它比 block-first 更接近作者想要的局部性，因为至少把同一 head 的工作在时间顺序上放在一起。
- 但如果 workgroup 仍 round-robin 地分发到所有 `XCD`，一个 head 的所有 blocks 仍会被条带化到多个 `XCD`。

#### Swizzled Head-first
- 这是论文的核心方法。
- 它把：
  - `Head-first`
  - `Swizzled`
 结合起来，让同一 head / 同一 `ACC` 的 blocks 尽量落在同一 `XCD`。
- 论文实现表明，这个优化不需要重写整个内核，只需要对 workgroup ID 到 `(batch, head, block)` 的映射逻辑做较小改动。

## 实验设置
- 平台：`AMD MI300X`
  - `8` 个 `XCD`
  - 每个 `XCD` `38` 个 `CU`
  - 每个 `XCD` `4 MB L2`
  - 总 `32 MB L2`
  - `192 GB HBM3`
- 实现方式：`Triton`
- 统计工具：`ROCProfiler v3`
- 对比对象：
  - `Naive Block-first`
  - `Swizzled Block-first`
  - `Naive Head-first`
  - `Swizzled Head-first`
- 实验分成四部分：
  1. `MHA` 敏感性分析
  2. `GQA` 敏感性分析
  3. `DeepSeek-V3 prefill` 案例
  4. `FlashAttention2 backward`

## 结果与解释

### MHA：长序列、高 head 数时收益最明显
- 当 head 数较少、序列较短时，四种方法差别不大。
- 当 `H_Q >= 64` 且序列长度增大时，差距迅速拉开。
- 在 `H_Q = 128, N_CTX = 128K` 的极端配置下，`Swizzled Head-first` 相比 block-first 方法可达到显著提升。

### 真正解释性能差异的是 L2 命中率
- 轻载配置下，各方法的 `L2 hit rate` 都能接近 `90%`。
- 在重载配置，特别是 `H_Q = 128, N_CTX = 128K` 时：
  - `Swizzled Head-first` 仍可维持在 `90%-96%`
  - `Naive Head-first` 会明显下降
  - 两种 block-first 方法在极端配置下会大幅掉到更低水平
- 因此收益并不是“调度碰巧更快”，而是缓存复用确实更强。

### GQA：分组与 XCD 数量匹配时，Swizzled Block-first 也能很强
- 在 `GQA` 实验中，作者固定 `8` 个 KV heads，并测试 `H_Q = 32, 64, 128`。
- 这里 `Swizzled Block-first` 也表现很好，因为 `8` 个 KV heads 与 MI300X 的 `8` 个 `XCD` 在拓扑上恰好匹配。
- 这说明最优映射不仅由算法决定，还取决于“模型分组结构”和“硬件 NUMA 域数量”是否匹配。

### DeepSeek-V3 Prefill：典型 chiplet-aware 案例
- DeepSeek-V3 prefill 的配置是：
  - `H_Q = 128`
  - `H_K = 128`
  - `D_HEAD = 56`
- 这是高 head 数真实工作负载，因此很容易暴露缓存碎片化问题。
- 论文显示 `Swizzled Head-first` 在几乎所有配置下都是最优。

### Backward 也受益，但小于 Forward
- 在 backward pass 上，`Swizzled Head-first` 仍是最优，但提升幅度明显小于 forward。
- 作者的解释是：backward 中标量操作更多，瓶颈更复杂，因此单靠 cache locality 优化不能像 forward 那样充分释放收益。

## 分析
- 论文真正先回答的是“FlashAttention2 里哪些 workgroup 之间真的共享数据”。作者的判断是：同一 attention head，或 GQA 下共享同一组 `K/V` 的 grouped heads，天然构成一个共享工作集单元；这正是 `ACC` 的定义来源。
- 它比较四种映射的核心差异不在 attention 数学，而在 workgroup 被怎样映射到不同 `XCD`。如果同一 `ACC` 被条带化到多个 `XCD`，同一份 `K/V` 就会在多个 `L2` 中重复加载。
- 这篇论文最贴近当前课题的点，不是 GPU 平台本身，而是“共享 `K/V` 工作集应该作为拓扑放置单元”这一抽象。
- 它和当前 CPU 路径的可迁移映射是：
  - GPU 里的 `ACC`
  - 对应当前 CPU `prefill` 里共享同一 `kv_head` 或共享重叠 `KV` 前缀的一组任务
- 当前 `acc-local-l3` 的价值，也正是验证“把共享 `K/V` 工作集收进更小局部域”是否真能改善 locality。

## 证据
- 原始资料：
  - [../原始资料/papers/GPU_Attention_NUMA_Optimization.pdf](../原始资料/papers/GPU_Attention_NUMA_Optimization.pdf)
- 当前研究里的对应页面：
  - [../实验结果解读/2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md](../实验结果解读/2026-04-07_Qwen3-30B-A3B_注意力KV工作集与32MiBL3容量估算.md)
  - [../实验方案/Qwen3-30B-A3B_attention-only_TP实验步骤.md](../实验方案/Qwen3-30B-A3B_attention-only_TP实验步骤.md)

## 边界
- 论文平台是 `AMD MI300X GPU`，不是 `AMD EPYC CPU`。
- 论文直接证明的是 `XCD/L2` 局部性与 workgroup 映射的关系，不是当前 `CCX/L3` 场景的直接结论。
- 因此它只能作为方法启发，不能直接当作“当前 CPU 方案已被论文证明有效”的证据。

## 可迁移点
- 共享 `K/V` 工作集可以作为 locality 单元。
- 最值得改的是任务映射方式，而不是先改 attention 数学形式。
- 映射策略需要同时考虑模型分组结构与硬件局部域数量。

## 不可直接迁移点
- GPU 的 `XCD/L2` 拓扑、workgroup 调度和当前 CPU 线程子池机制并不相同。
- 论文的收益数字不能直接外推到当前 `vLLM CPU`。
