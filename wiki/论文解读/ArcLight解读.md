# ArcLight 解读
## 问题
- 论文要解决的是 many-core CPU 上跨 `NUMA node` 扩展时的远端内存访问开销。作者将这一瓶颈概括为 `cross-NUMA memory access wall`，并指出主流 CPU LLM 框架没有把它作为首要优化对象处理（摘要，Fig. 1，Sec. 1）。

## 核心内容
- `ArcLight` 从内存管理、线程组织和计算图执行三层同时做 `NUMA-aware` 设计，而不是只增加线程绑核选项（Fig. 3-9）。
- 内存侧，它为每个 `NUMA node` 分配本地 buffer，并对 activation buffer 做双缓冲，按 layer parity 交替使用（Fig. 3，Fig. 4）。
- 线程侧，它在统一线程池中引入可拆分/合并的 `thread group`，并区分 `local barrier` 与 `global barrier`（Fig. 5，Fig. 6）。
- 计算侧，它引入跨 `NUMA` 的 tensor parallel、`Scatter/Gather` 和异步子图执行，使各 `NUMA node` 尽量在本地 shard 与本地 buffer 上工作（Fig. 8，Fig. 9）。
- 论文摘要报告 `ArcLight` 相对主流框架最高可达 `46%` 的吞吐提升；正文实验对象是 `llama.cpp`，并分别给出单 `NUMA`、多 `NUMA`、长 prompt decode 和 prefill 的结果（摘要，Fig. 10-13）。

## 论文先证明了什么
- 在作者的 `4-NUMA` 测试机上，本地内存访问带宽约 `101-103 GB/s`，跨节点远端访问约 `22-26 GB/s`，本地约为远端 `4x`，这是论文后续所有 `NUMA-aware` 设计的前提（Tab. 1）。
- 论文进一步说明，仅把线程均匀分散到多个 `NUMA node` 并不能保证 locality。`llama.cpp` 在 `--numa distribute` 下会把线程分散到各 node，但 tensor 不绑定到特定 node，导致计算位置与数据位置频繁失配（Sec. 3.1，Fig. 7）。
- `Fig. 7` 给出的具体失配链条是：若初始 activation 只在 `node 0`，则 `GEMM1` 时位于 `node 1-3` 的权重分区需要远端读取 activation；`GEMM1` 之后 activation 分散到各 node；到 `GEMM2` 时，每个权重分区只有约四分之一 activation 访问是本地，其余仍是远端访问（Fig. 7）。

## 方法与系统设计
### 内存管理
- `ArcLight` 启动时预分配内存池；启用 `NUMA` 后，为每个 `NUMA node` 单独分配本地 buffer，而不是把 `NUMA` 对上层透明化。这一设计的直接作用是简化显式的 tensor-to-`NUMA node` 绑定（Fig. 3，Sec. 2.3）。
- activation buffer 使用双缓冲机制，两块 buffer 按 layer parity 交替使用，以降低逐层推理时的运行期内存占用（Fig. 4，Sec. 2.3）。

### 线程组织
- 线程管理器在推理开始前创建 worker 线程。与只支持单一线程池视图的常见设计不同，`ArcLight` 在池内引入逻辑 `thread group`，允许在初始化和图执行时动态拆分与合并（Fig. 5，Sec. 2.4）。
- 当线程池被拆成 `n` 个组时，`n` 个组可以并行执行 `n` 个独立 tensor operation。为配合这种多视图组织，论文引入跨整个线程池的 `global barrier`，与组内同步使用的 `local barrier` 区分开来（Fig. 5，Fig. 6，Sec. 2.4）。

### 跨 NUMA Tensor Parallel
- 论文把 tensor parallel 作为 many-core CPU 上缓解跨节点访存的核心手段。作者说明：在连续 `GEMM` 序列中，合适的权重分区可以消除跨节点数据访问（Fig. 8a，Fig. 8b，Sec. 3.2）。
- 在 transformer 中，`Wq/Wk/Wv/Wgate/Wup` 做 row partition，`Wo/Wdown` 做 column partition；其中 `Wq/Wk/Wv` 按 attention head 切分（Sec. 3.2）。
- 论文以 MLP 为例写出 TP 后的形式：原始计算是 `Y=σ(AX), Z=BY`；TP 后变成 `[Y1,Y2]=[σ(A1X),σ(A2X)]`、`[Z1,Z2]=[B1Y1,B2Y2]`、`Z=Z1+Z2`。参与 TP 的 tensor 被分别放入各 `NUMA node` 的 buffer 中，从而隔离跨节点访问（Fig. 8a，Fig. 8b，Sec. 3.2）。

### 计算图执行
- 引入 TP 后，CPU 上不再是“所有线程协作执行同一个 op”，而是多个 operator 并发执行，因此计算图会从单图转为并行子图集合（Fig. 8，Sec. 3.3）。
- `Scatter` 用于把线程池重组为多个 group，并为各子图创建输入 view tensor；`Gather` 用于收集并求和各子图输出，同时把线程池恢复为单 group，回到非 TP 模式（Fig. 8，Sec. 3.3）。
- 在线程同步上，论文比较了两种方式：`Sync A` 在每个 operator 后做全局同步；`Sync B` 只在开始和结束做全局同步，中间只做组内同步。论文说明异步子图执行可显著减少线程空转时间（Fig. 9，Sec. 3.4）。

## 实验设置
- 测试机为 `192-core`、`4 NUMA nodes`。每个 node 有 `48 x HUAWEI Kunpeng-920 cores (ARMv8.2)` 和 `6 x DDR4 memory channels`（Sec. 4）。
- OS 为 `Ubuntu 22.04`，tensor operation 使用 `NEON` 指令；模型是 `Qwen3-4B`，量化格式为 `Q4_0`（Sec. 4）。
- 主文 decode 测试使用 `prompt length = 15`、`generation length = 256`；附录 `A.2` 额外报告 `prompt length = 300` 时的 decode 和 prefill 吞吐（Sec. 4，Fig. 12，Fig. 13）。
- 对比对象是 `llama.cpp`。正文先比较单 `NUMA node`，再比较多 `NUMA node` 下的 `2-node` 与 `4-node`；附录继续沿用多 `NUMA` 配置（Fig. 10，Fig. 11，Fig. 12，Fig. 13）。

## 结果与解释
- 单 `NUMA node` 下，随着线程数从 `6` 增加到 `48`，两套框架的 decode 吞吐都上升。论文将这一现象解释为：单节点内存带宽较高时，吞吐会随核心数扩展；`ArcLight` 略高于 `llama.cpp`，原因是前者做了 node-local allocation，而后者 buffer 页面由 OS 跨 node 分布（Fig. 10，Sec. 4）。
- 多 `NUMA node` 下，`ArcLight + TP` 优于只分散线程绑定、但把内存放置留给 OS 的 `llama.cpp`。论文把这部分收益归因于 cross-`NUMA` TP 打破了 `single-node memory access wall`（Fig. 11，Sec. 4）。
- 在多 `NUMA` 结果上，论文还写明 Sec. 3.4 的异步子图执行可再带来约 `5 token/s` 的额外增益（Fig. 11，Sec. 4）。
- 对长 prompt，附录写明 `prompt length = 300` 时 decode 吞吐略低于短 prompt；同时 `ArcLight` 在 prefill 吞吐上仍优于 `llama.cpp`，但优势小于 decode。论文给出的解释是：`ArcLight` 的 TP 主要缓解 memory access wall，而 prefill 更偏 `compute-bound`（Fig. 12，Fig. 13，App. A.2）。

## 分析
- 从论文可直接提炼的事实是：作者先用 `Tab. 1` 和 `Fig. 7` 把 many-core CPU 上的 locality 失配写成明确瓶颈，再分别在内存放置、线程组织和图执行三个层面给出对应控制机制（Tab. 1，Fig. 3-9）。
- 论文中的 locality 粒度是 `NUMA node` 与本地/远端 DRAM 访问，不是更细的 `CCX/L3 slice` 或 kernel 内部任务分片粒度（Tab. 1，Fig. 7）。
- 论文报告的性能提升是条件化结论，依赖其 `ARM Kunpeng-920 + 4 NUMA` 平台、`Qwen3-4B Q4_0` 模型和指定 prompt/generation 设置，不是无条件结论（Sec. 4，App. A.2）。

## 边界
- 论文实验平台只有 `ARM Kunpeng-920`；`Limitations` 明确写到当前仅在 ARM 平台评估，其他架构如 `x86` 留待后续工作。
- baseline 是 `llama.cpp`，不是 `vLLM`。
- 论文关注的是整框架的推理吞吐，没有直接给出 `L3/LLC` 计数器、`CCX` 粒度缓存行为或 kernel 级 locality 数据。
- `Scatter/Gather` 实现本身在 `Limitations` 中也被标注为仍较初步，后续还需继续优化内存开销和并行效率。

## 可迁移点
- 论文可直接借鉴的点是：若 locality 是目标，tensor 放置不能只依赖 OS `first-touch`，而要由框架显式控制（Fig. 3，Fig. 7）。
- 若并行执行会改变数据访问域，线程组织和计算图切分要一起设计，而不是只改绑核或只改内存分配（Fig. 5-9）。
- decode 与 prefill 的收益预期需要分开看；论文已明确给出“prefill 优势更小”的条件化结果（Fig. 12，Fig. 13）。

## 不可直接迁移点
- 论文里的整模型吞吐提升不能直接映射到当前 `AMD EPYC chiplet`、`vLLM` 或 `attention-only` 场景，因为硬件、软件栈和研究粒度都不同。
- 论文的 TP 设计是整层权重切分与并行子图调度，不等于 kernel 内部的 locality 调度或 `CCX/L3 slice` 级数据放置。
- 论文没有覆盖 `x86/AMD EPYC` 实测，因此不能把其数值结果直接外推到当前平台。

## 证据
- 原始资料：`wiki/原始资料/papers/ArcLight.pdf`
- 关键锚点：摘要；`Tab. 1`；`Fig. 3-13`；`Sec. 2.3-2.7`；`Sec. 3.1-3.4`；`Sec. 4`；`App. A.2`；`Limitations`
