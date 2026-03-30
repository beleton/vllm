# Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects

## 论文基本信息

- 标题：*Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects*
- 作者：Mansi Choudhary, Karthik Sangaiah, Sonali Singh, Muhammad Osama, Lisa Wu Wills, Ganesh Dasika
- 版本：arXiv:2511.02132v1，2025 年 11 月 3 日
- 核心对象：AMD MI300X 这类采用 chiplet / multi-die 组织方式、显式暴露 NUMA 特性的 GPU

## 一句话概括

这篇论文的核心贡献，是把大家通常从“算法复杂度”角度理解的 Attention 优化，进一步推进到“GPU 物理拓扑感知”的层面：作者指出，在 chiplet GPU 上，Attention 的性能瓶颈不只是算子本身，而是**工作块被如何映射到不同 die / XCD 上**。如果映射方式忽略 NUMA，就会严重破坏缓存复用；如果映射方式顺着 NUMA 域去安排，就能用非常小的代码改动换来很大的性能提升。

## 1. 研究背景与问题意识

作者首先强调一个重要趋势：现代 AI GPU 正在从“统一缓存、统一访问代价”的架构，走向“分布式缓存、多 memory controller、多 chiplet”的架构。这样做的好处是扩展性更强、制造良率更高，但代价是引入了显著的 NUMA 效应。

在这种架构下：

- 本地访问更快，本地带宽更高；
- 跨 die、跨 chiplet 的远程访问更慢；
- 某个 die 上的 L2 cache 不能天然服务另一个 die 上的计算；
- 同一份数据如果被多个 die 分别加载，就会造成缓存碎片化和 HBM 重复访存。

论文的关键判断是：现有很多 Attention 优化工作，包括 FlashAttention 系列，主要假设 GPU 的内存访问近似均匀，因此没有把“空间拓扑”当成一等公民来处理。但到了 AMD MI300X 这样的架构上，这个假设已经不成立。

换句话说，这篇论文不是在重新发明 Attention 算法，而是在问一个更偏体系结构的问题：

**对于已经很高效的 FlashAttention2，如果把 workgroup 映射方式改成 NUMA-aware，是否还能显著提速？**

作者的答案是：可以，而且收益很可观。

## 2. Attention 与 FlashAttention2 中真正可复用的数据在哪里

为了建立后续优化的逻辑，作者先分析了 FlashAttention2 的计算结构。

标准 attention 计算可写为：

- \(S = QK^T\)
- \(P = softmax(S / \sqrt{d})\)
- \(O = PV\)

训练时还要做 backward，计算 \(dQ\)、\(dK\)、\(dV\) 等梯度。

论文关注的是 FlashAttention2 的 tiled 执行方式。对一个 attention head 而言，Q 会被切成多个 row blocks，每个 workgroup 负责其中一块；但这些 workgroup 在计算时都要访问**同一整个 K 和 V**。这意味着：

- 同一个 head 内，不同 row block 的 workgroup 之间存在天然的数据共享；
- backward 中这种共享也存在，因为多个 workgroup 仍会共同依赖同一组 Q/K/V/dO；
- 真正值得放进同一 NUMA 域、争取本地 L2 复用的，不是“相邻 block”本身，而是“共享同一 K/V 的那一组 workgroups”。

作者把这类共享同一输入张量的 workgroup 集合定义为 **Attention Compute Cluster, ACC**。

这个定义很关键：

- 在 MHA 中，一个 head 基本对应一个 ACC；
- 在 GQA 中，多个 query heads 共享同一组 K/V，因此一个 group 对应一个更大的 ACC。

论文最重要的优化洞见就是：

**只要把同一个 ACC 尽量限制在同一个 XCD 内执行，就能最大化该 XCD 的 L2 cache 复用，并减少同一份 K/V 被多个 XCD 重复从 HBM 拉取。**

## 3. 现有映射方式为什么不够好

论文系统比较了 FlashAttention2 在 MI300X 上的四种 grid-to-XCD 映射思路，其中前三种可视为现有方法或基线。

### 3.1 Naive Block-first

这种方法按 block 优先遍历：先处理所有 head 的 block 0，再处理所有 head 的 block 1，依此类推。与此同时，workgroup 按 round-robin 分到不同 XCD。

问题在于：

- 同一 ACC 会被拆散到多个 XCD；
- 每个 XCD 只看到这个 ACC 的一部分工作；
- 同一份 K/V 可能在多个 XCD 的 L2 里重复加载；
- 结果就是缓存命中率下降、HBM 流量上升。

### 3.2 Swizzled Block-first

它保留 block-first 的遍历顺序，但用 swizzle 技术把某些 logically related 的 workgroup 尽量映射到同一个 XCD。

这在 GQA 场景下可能有效，特别是当：

- GQA 的 group 数量恰好与 XCD 数量匹配；
- 每个 group 能自然落到一个 XCD 上。

但它的问题也很明显：

- 它对 GQA 比较友好；
- 对 MHA 不够理想；
- 当 ACC 数量与 XCD 拓扑关系不匹配时，仍然会出现 cache split。

论文指出，AMD AITER 中已有类似思路，但并不能稳定覆盖 MHA 这类更难的配置。

### 3.3 Naive Head-first

这种方法按 head 优先遍历：先做完一个 head 的所有 row blocks，再做下一个 head。

它比 block-first 更接近作者想要的局部性，因为至少“时间顺序”上把同一 head 的工作放在一起了。但如果 workgroup 仍是 round-robin 地分发到所有 XCD，那么：

- 一个 head 的所有 blocks 仍会被条带化到多个 XCD；
- 各个 XCD 都会各自缓存这份 head 的 K/V；
- 局部性虽然比 block-first 强，但仍然存在跨 XCD 冗余。

所以它只是在“时间局部性”上进步了，但还没有解决“空间局部性”的根本问题。

## 4. 论文提出的方法：Swizzled Head-first Mapping

这篇论文最核心的贡献，就是 **Swizzled Head-first Mapping**。

它把两件事结合了起来：

- `Head-first`：先把一个 head 的所有 blocks 做完，再切下一个 head；
- `Swizzled`：通过 workgroup ID 重映射，让同一 head / 同一 ACC 的 blocks 尽量落在同一个 XCD。

于是，论文实现了这样一种效果：

- 一个 XCD 在一段时间内主要服务一个 ACC；
- 这个 ACC 所需的 K/V 可以集中进入该 XCD 的 L2；
- 后续同一 ACC 的其他 blocks 直接复用本地 L2；
- 避免多个 XCD 同时为同一 ACC 重复拉取相同数据。

作者给出的 Triton 伪代码显示，这个优化并不需要重写整个内核，只需要对 workgroup ID 到 `(batch, head, block)` 的映射逻辑做较小改动。其关键变量包括：

- `heads_per_xcd`
- `blocks_per_head`
- `chunk_size`
- 基于线性 `wid` 计算 `head_offset`、`block_offset`、`batch_offset`

从体系结构视角看，这个方法的本质不是“让算子更少算”，而是：

**让本来就必须执行的那些 workgroups，以更符合硬件拓扑的方式执行。**

这也是我认为这篇论文最有价值的地方：它把“软件中的调度顺序”与“硬件中的 chiplet / cache 拓扑”精确对应起来了。

## 5. 实验平台与评测设计

作者在 AMD MI300X 上评测，平台参数包括：

- 8 个 XCD
- 每个 XCD 38 个 Compute Units，总计 304 个 CU
- 每个 CU 64 个 stream processors
- 每个 CU 16 KB L1
- 每个 XCD 4 MB L2，总计 32 MB
- 192 GB HBM3
- 5.3 TB/s HBM3 带宽

实现和测量方式也比较标准：

- 内核用 Triton 实现；
- 用 ROCProfiler v3 统计 L2 hit rate；
- 比较四种映射：Naive Block-first、Swizzled Block-first、Naive Head-first、Swizzled Head-first。

实验分为四部分：

1. MHA 敏感性分析  
   扫描 head 数、序列长度、batch size，观察不同映射方式的变化趋势。

2. GQA 敏感性分析  
   重点看 Llama 3 家族这类共享 KV 的场景。

3. DeepSeek-V3 prefill 案例  
   这是一个高 head 数的真实工作负载，非常适合验证该方法。

4. FlashAttention2 backward  
   检查这一优化是否不仅适用于 forward，也能迁移到 backward。

## 6. 最重要的实验结论

## 6.1 MHA：长序列、高 head 数时收益最明显

在 MHA 配置中，作者测试：

- `H_Q = H_K = 8, 16, 32, 64, 128`
- `N_CTX = 8K, 32K, 128K`
- batch size = `1, 2, 4, 8`
- `D_HEAD = 128`

结论非常清楚：

- 当 head 数较少、序列较短时，四种方法差别不大；
- 当 `H_Q >= 64` 且序列长度增大时，差距迅速拉开；
- 在 `H_Q = 128, N_CTX = 128K` 的极端配置下，Swizzled Head-first 相比 block-first 方法最高可提升 **50%**。

这个结论说明：NUMA-aware 映射不是“锦上添花”，而是在大模型、长上下文场景中越来越接近“必要条件”。

## 6.2 真正解释性能差异的是 L2 命中率

作者进一步给出 L2 cache hit rate 的证据，这是论文非常扎实的一点。

在轻载配置下，各方法的 L2 命中率都能接近 90%。但当配置变重，特别是在：

- `H_Q = 128`
- `N_CTX = 128K`

时，情况发生了根本分化：

- Swizzled Head-first：L2 命中率仍可维持在 **90%-96%**
- Naive Head-first：下降到 **40%-60%**
- 两种 block-first 方法：在极端配置下几乎崩到 **约 1%**

这组结果非常有说服力，因为它直接把“性能更高”追溯到了“缓存复用更强、HBM 重取更少”这一硬件层机制，而不是停留在经验结论上。

## 6.3 GQA：当分组与 XCD 数量匹配时，Swizzled Block-first 也能很强

在 GQA 实验中，作者固定 `8` 个 KV heads，并测试：

- `H_Q = 32, 64, 128`
- 对应 Llama 3 8B、70B、405B

这里的结果更细腻一些：

- Swizzled Head-first 依然稳定优秀；
- Swizzled Block-first 也表现很好；
- 这是因为 8 个 KV heads 与 MI300X 的 8 个 XCD 在拓扑上刚好匹配；
- Naive Block-first 仍然在大规模配置下明显掉速；
- Naive Head-first 在高 batch、长序列下会有一定不稳定，通常只有 Swizzled Head-first 的约 90%-95%。

这说明一个更深层的事实：

**最优映射并不是只由算法决定的，还取决于“模型分组结构”和“硬件 NUMA 域数量”是否共振匹配。**

## 6.4 DeepSeek-V3 Prefill：非常典型的 chiplet-aware 案例

作者专门分析了 DeepSeek-V3 prefill：

- Attention 类型：MHA
- `H_Q = 128, H_K = 128`
- `D_HEAD = 56`

这是一个很好的案例，因为：

- head 数很多；
- 明显超过 XCD 数量；
- 很容易暴露缓存碎片化问题。

结果显示：

- Swizzled Head-first 在几乎所有配置下都是最优；
- 当序列长度达到 `128K` 时，Naive Block-first 的相对性能会跌到 **0.65x 以下**；
- Swizzled Block-first 在 `batch = 8, 128K` 时也会降到约 **0.76x**；
- Naive Head-first 虽然比 block-first 好，但在极端长度下仍然波动明显。

这部分结果很重要，因为它证明该方法并非只在合成实验里有效，而是对真实大模型 prefill 场景有直接价值。

## 6.5 Backward 也受益，但收益小于 Forward

在 backward pass 上，作者使用 AMD AITER 中的 FlashAttention2 backward kernel 做测试，配置为：

- `H_Q = 128`
- `N_CTX = 8K, 32K, 128K`
- batch size = `1, 2`

结论是：

- Swizzled Head-first 仍然最好；
- 在长序列下，speedup 大约可以达到 **1.10x**；
- 但相比 forward 中最高 50% 的提升，backward 收益明显更小。

作者的解释是合理的：backward 中标量操作更多，内核瓶颈更复杂，因此单靠 cache locality 优化，不能像 forward 那样充分释放收益。

这里还有一个值得注意的小细节：图 16 的标题与正文中的归一化描述有一点口径上的不完全一致，但不影响核心判断，即 **Swizzled Head-first 在 backward 中依然最优，只是收益幅度缩小了。**

## 7. 这篇论文最值得重视的学术价值

如果站在一名体系结构研究者而不是单纯算子优化工程师的角度，这篇论文至少有五点价值。

### 7.1 它证明了“Attention 优化”已经进入拓扑感知阶段

过去讨论 Attention，重点往往是：

- 降低 IO；
- 融合 kernel；
- 提高并行度；
- 减少中间张量落地。

这篇论文进一步说明，在 chiplet GPU 上，上述优化还不够，**work distribution 与物理拓扑的一致性**也已经成为核心性能因素。

### 7.2 它把 NUMA 从“系统软件问题”推进成“算子设计问题”

通常 NUMA 更常出现在 CPU、操作系统或运行时调度的语境里。本文的价值在于，它表明在 MI300X 这类架构上，NUMA 不只是 runtime 的问题，而是 kernel designer 必须主动编码利用的结构性特征。

### 7.3 它说明“最小代码改动”也可能带来“最大体系结构收益”

论文没有重新设计 FlashAttention2 的数学形式，也没有增加复杂的近似策略，主要修改的是 workgroup 映射逻辑。可结果却带来了非常大的缓存命中率提升和性能提升。这说明：

**在 chiplet 架构上，调度映射有时比进一步微调算术流水更关键。**

### 7.4 它强调了厂商实现差异的重要性

论文提到一个很有意思的对比：

- NVIDIA Blackwell 倾向于通过更强的一致性机制在硬件层吸收 NUMA 影响；
- AMD MI300X 则把 NUMA 特征更显式地暴露给软件。

这意味着同一种优化策略未必在所有 GPU 上同样有效。未来分析 GPU attention kernel 时，不能再抽象成“某种通用 GPU”，而应当具体到缓存组织、一致性模型和 die 间通信机制。

### 7.5 它为 chiplet 时代的算子库设计提供了方向

从 GEMM 到 Attention，论文都在传递同一个趋势：未来高性能库不会只依赖“线程块怎么切”，还要关心“切好的线程块被放到哪一块物理硅片上执行”。

这是一个非常典型的 chiplet-aware software stack 研究方向。

## 8. 论文的局限与我认为仍可继续追问的问题

这篇论文很有启发，但从严格研究角度看，仍有一些边界。

### 8.1 主要验证对象是 MI300X

论文结论对 MI300X 非常强，但对其他 multi-die GPU 的可迁移性，还需要更多证据。特别是：

- 不同厂商是否显式暴露 NUMA；
- 不同架构是否具备相同的 L2 私有化程度；
- runtime 调度策略是否允许同样的 swizzle 映射稳定生效。

### 8.2 更偏 kernel 级别，而非端到端训练 / 推理级别

论文主要评测单个 attention kernel 的性能和缓存命中率。它没有进一步展示：

- 端到端训练吞吐提升多少；
- 对整模型 latency / throughput 的贡献占比；
- 在多算子串联时是否会与其他优化产生冲突。

### 8.3 Backward 的收益说明瓶颈并不只有 cache

backward 只有约 1.10x 的改进，说明随着内核复杂度增加，性能可能受寄存器压力、指令调度、标量路径等多种因素共同限制。也就是说，NUMA-aware mapping 很重要，但并不是万能钥匙。

## 9. 对 Chiplet 方向阅读这篇论文的启发

如果你是从 chiplet / 异构体系结构角度读这篇文章，我建议抓住下面三个启发。

### 9.1 Chiplet 的问题不只是“互连带宽够不够”

很多 chiplet 论文会强调互连、封装、带宽、延迟，但这篇文章提醒我们：**软件是否把相关工作放在同一个局部域内执行**，同样会决定最终性能。

### 9.2 Cache locality 在 chiplet 时代必须重新定义

在传统单 die GPU 上，说“有 locality”往往已经足够；但在 chiplet GPU 上，还必须问：

- locality 发生在哪个 die 上？
- 复用的数据是否停留在同一个 XCD 的 L2？
- 是否因为调度方式而把本可复用的数据拆散？

也就是说，locality 已经从单纯的时空局部性，升级为**拓扑约束下的时空局部性**。

### 9.3 后续很多 AI kernel 都值得按这个思路重看

这篇论文虽然讨论的是 Attention，但它的方法论完全可以推广到：

- MoE 路由相关算子；
- KV cache 访问相关 kernel；
- GEMM / fused MLP；
- collectives 与片上缓存协同的问题。

凡是存在“多个 workgroups 共享一部分输入张量”的场景，都值得重新检查其 workgroup-to-chiplet 映射方式。

## 10. 总结

这篇论文最重要的贡献，不在于提出了一个更复杂的 Attention 公式，而在于提出了一个很清楚、也很有普适性的体系结构观点：

**在 chiplet GPU 上，Attention 的高性能实现必须同时满足算法高效与拓扑感知。**

作者提出的 Swizzled Head-first Mapping，本质上是把“共享同一 K/V 的工作块”尽量收拢到同一个 XCD 中执行，从而减少缓存碎片化、提高 L2 命中率、降低 HBM 重复访存。实验表明，这种方法在 MHA 的长序列高 head 数场景下尤其有效，在 MI300X 上可带来最高 50% 的性能提升，并把 L2 命中率稳定在 80%-97% 的高水平。

如果用一句更学术的话来概括本文，那就是：

**它证明了在下一代 disaggregated / chiplet GPU 上，NUMA-aware kernel mapping 不再是可选优化，而正在成为 Attention 这类核心 AI 算子的基础设计原则。**
