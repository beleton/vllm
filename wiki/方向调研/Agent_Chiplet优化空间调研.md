# Agent 场景在 Chiplet CPU 上的优化空间调研

## 目的

从 Agent 场景对 CPU 侧的实际需求出发，调研 chiplet 架构是否存在有意义的优化空间。两次迭代：
- **第一版**：从 LLM 推理视角（multi-turn decode、tool-call prefill）分析 → 结论为排除，因为 attention 对 L3 不敏感的约束未变
- **第二版**：从 CPU 侧的 tool 执行、内存管理、并发隔离需求出发 → 结论反转，存在 chiplet 架构特有的优化杠杆

## Agent 场景对 CPU 侧的真实需求

### 需求全景

Agent 场景对 CPU 的消耗远超 LLM 推理。关键数据来自两篇 2025–2026 年的测量论文：

| 来源                                                                                      | 发现                                                                                            |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **"A CPU-Centric Perspective on Agentic AI"** (Raj et al., arXiv:2511.00739, 2025–2026) | Tool 处理在 CPU 上占用高达 **90.6%** 的端到端延迟；CPU 动态能耗占系统总能耗的 **44%**（大 batch 下）                        |
| **"AgentCgroup"** (Zheng et al., arXiv:2602.09345, 2026)                                | OS 级执行（容器初始化 + tool 调用）占端到端延迟的 **56–74%**；**内存（非 CPU）是并发瓶颈**；内存峰均比高达 **15.4×**；85–97% 任务含重试循环 |

### 按组件分解的 CPU 需求

```
Agent 端到端延迟 = LLM推理(26–44%) + 容器/Agent初始化(31–48%) + 工具执行(25–26%)
```

具体计算模式：

#### 1. Tool 执行沙箱（CPU 最大消耗者）

| 特征 | 数据 |
|------|------|
| 沙箱类型 | 微虚拟机（Firecracker/KVM/Hyperlight）或 V8 isolates |
| 冷启动延迟 | 60ms（CubeSandbox）到 <1s（GKE Agent Sandbox） |
| 单沙箱内存基线 | ~185 MB（Agent 框架 + Node.js 运行时） |
| 内存突发峰值 | 500 MB–2+ GB（持续 1–2 秒） |
| 内存峰值变化速率 | 最高 3 GB/s |
| 生命周期 | 创建 → 执行 → 销毁，单次 1–2 秒 |
| 并发密度 | ~2000 sandbox / 96-core 服务器（CubeSandbox） |

Tool 执行的访存模式：
- **内存分配密集**：沙箱启动时的 JIT 编译、库加载、数据加载
- **I/O 混合**：文件读写、网络 I/O、管道通信
- **计算特征**：异构——linter 是 CPU 密集，grep 是 I/O 密集，Python 脚本是混合型
- **burst 特征**：98.5% 的内存突发发生在 tool 调用期间（而非 LLM 推理阶段）

#### 2. Agent 记忆系统

| 记忆类型 | 存储后端 | 访问模式 | 典型数据量 |
|----------|---------|---------|-----------|
| 工作记忆（Working） | 内存中，对话上下文 | 高频读写，临时 | KB–MB |
| 情景记忆（Episodic） | 向量数据库 | 相似度检索 | GB 级（历史交互 embedding） |
| 语义记忆（Semantic） | 知识图谱 | 图遍历 / 子图查询 | MB–GB 级 |
| 程序记忆（Procedural） | Key-value 存储 | 元数据查找 | 小（技能定义） |

其中情景记忆（向量检索）和语义记忆（图遍历）的访存模式与 chiplet 优化框架有交互——详见 [RAG 调研](RAG_Chiplet优化空间调研.md) 中的 HNSW 和知识图谱分析。

#### 3. 多 Agent 并发编排

| 特征 | 数据 |
|------|------|
| CPU 平均利用率 | 仅 7.6%–13.2%（24 核机器，单 agent） |
| 内存利用率 | 先于 CPU 成为瓶颈（峰值 2–4 GB/任务） |
| 重试循环 | 85–97% 任务含连续 3+ 次相同 tool 调用；最坏 56 次连续重试 |
| 重试内存累积 | 最坏情况累计 502 MB 未释放内存 |
| 跨任务差异 | 资源需求差异 20× across tasks，同一任务差异 1.8× across runs |

## 从 Chiplet 三篇论文的筛选条件出发

将筛选条件应用于 CPU 侧的三个主要需求：

### Tool 执行沙箱：**条件 1 满足，条件 2 部分满足，条件 3 不适用**

**条件 1（跨 chiplet 访问占比）**：
- **sandbox 私有堆、栈、COW 页由本进程独占**。若进程没有迁移且页面由本地 NUMA first-touch 分配，访问路径是本 CCD → IOD → DRAM，不经过其他 CCD 的 cache。不同 sandbox 访问各自私有数据不会触发彼此之间的 cache coherence 数据搬移
- **真正可能产生跨 CCD 开销的是共享 OS 控制路径**：进程迁移后的远端 NUMA 页访问、共享内核/文件系统元数据（VFS/overlayfs/dentry/inode）、page cache、cgroup 计数器（memcg 高频更新 cache line bounce）、pipe/socket buffer、日志队列等
- 并发 DRAM 访问会争用内存控制器和带宽，但这是共享内存系统的带宽竞争，不是"跨 chiplet 数据流"
- **该方向需用 `perf c2c` / IBS / PCM 定位共享 cache line、内核锁、cgroup 计数器、VFS 元数据或 IPC 队列热点，而非测量私有 sandbox 数据的 DRAM 访问**

**条件 2（放置可控性）**：
- 沙箱进程可绑定到特定 chiplet（cgroup cpuset + mempolicy）
- Sandbox 间无共享内存（独立进程），放置互相独立
- **可控制，且放置决策空间大**

**条件 3（L3 边界复用）**：
- 沙箱内存 > 单 chiplet L3（32 MB）→ L3 不是复用缓存
- **但 chiplet 优化在此处的杠杆不在 L3，而在 DRAM 带宽隔离和内存分配的 NUMA locality**
- 这超出了三篇论文的 L3 中心框架

### Agent 记忆系统：**同 RAG 分析**

- 工作记忆：fit in L2，不涉及 chiplet
- 情景记忆（向量检索）：HNSW 图遍历有 chiplet 优化空间（详见 RAG 调研）
- 语义记忆（知识图谱）：图遍历类似 CHARM，但有 scale 差异

### 多 Agent 并发：**新场景——chiplet 作为资源隔离边界**

- 不是 "优化单个 workload 的 L3 局部性"
- 而是 "chiplet 拓扑作为多租户资源隔离的硬件边界"
- 这与三篇论文的问题设定根本不同

## Chiplet 架构适配 Agent CPU 需求的三个方向

### 方向 A：Agent Tool OS-Shared-Path Sharding（微架构视角）⭐⭐⭐⭐

**对应需求**：多 sandbox 并发时的共享 OS 路径 coherence 热点

**核心思路**：多 sandbox 并发时，私有数据不会产生跨 CCD 数据流。真正的跨 CCD 开销来自 VFS、overlayfs、page cache、cgroup 计数器、pipe/socket、日志队列等共享 OS 路径的 coherence/NUMA 干扰。将共享 OS 对象按 CCD/NUMA domain 分片，减少跨 CCD cache line bounce。

**与 OLAP WICP 的差异**：
- WICP 用独立地址空间切断同构 worker 的 coherence
- 此处 sandbox 已经是独立进程（自然无 coherence），热点来自**内核共享对象**而非用户态数据
- 问题从"消除 coherence"变为"分片共享 OS 路径以消除跨 CCD 争用"

**与 CHRAM 的差异**：
- CHARM 优化单个 workload 的图遍历放置
- 此处优化多 sandbox 共享的内核元数据和 IPC 路径
- 度量工具不同：perf c2c / lock stat / tracepoint，而非 PCM remote cache fill

**具体实现路径**：
1. 每个 CCD 或 NUMA domain 维护独立 sandbox worker pool
2. 每个 pool 使用独立 temp dir、日志队列、pipe/socket acceptor、cgroup subtree
3. Agent orchestrator 将同一 repo / 同一 task 的连续 tool calls 固定到同一 pool
4. 避免所有 sandbox 共享同一 cgroup、日志队列和 overlayfs 热路径
5. 用 `perf c2c`、IBS、lock stat 和 VFS/overlayfs/cgroup tracepoint 验证热点来源

**预期收益**：
- 减少共享 OS 对象（cgroup 计数器、dentry/inode、overlayfs 元数据、page cache）的 cross-CCD cache line bounce
- 减少共享锁争用（VFS、overlayfs、pipe/socket buffer lock）
- 提高并发 agent 密度

**风险**：
- 如果瓶颈主要是磁盘 I/O、网络或解释器初始化，该方向收益低
- 分片不能破坏 sandbox 隔离和文件系统语义
- 需 `perf c2c` / IBS 先定位热点，确认热点来自共享 OS 路径而非私有用户态数据

### 方向 B：NUMA-aware Warm Sandbox and Page-cache Affinity ⭐⭐⭐

**对应需求**：连续 tool calls 的 session locality 和 page cache 复用

**核心思路**：连续 tool calls 重复读取相同 repo 文件、解释器库、site-packages、node_modules、测试缓存。将同一 agent/repo 的 tool calls 固定到同一 NUMA domain，提升 page cache / dentry / L3 热度。这个方向优化的是 NUMA/page-cache locality，不是"CCD-local DRAM"。

注意：CCD 不包含内存控制器——内存控制器在 IOD，OS 能直接控制的是 socket / NUMA / NPS domain，不是"每 CCD 一个 DRAM 池"。

**与三篇论文的本质不同**：
- 三篇论文优化的是 L3 内的数据复用
- 此处优化的是 **连续 tool calls 间 session 固定带来的 page cache 和 dentry 热度**
- 评估指标：page fault 下降、远端 NUMA 页访问下降、page-cache miss 下降

**具体实现路径**：
1. 每个 NUMA/NPS domain 一个 warm sandbox pool
2. 同一 agent session 固定 home domain
3. Repo checkout、venv、node_modules、pytest cache 放在 home domain first-touch
4. 重试循环不迁移，除非该 domain memory pressure 超阈值
5. 用 PCM/NUMA stat 对比默认调度 vs session-affine 调度的 page fault 和远端访问

**预期收益**：
- 减少连续 tool calls 的 page fault 和远端 NUMA 页访问
- 提升 page cache 和 dentry cache 热度
- 对含大量文件扫描（pytest/grep/mypy）和库加载的 tool calls 效果最明显

**风险**：
- 若连续 tool calls 访问不同 repo 或不同文件集，session affinity 可能降低并发度
- 固定 home domain 可能导致负载不均，排队抵消 locality 收益
- 该方向优化的是 session locality，不是单次 sandbox 冷启动

### 方向 C：vLLM CPU 推理与 Tool Sandbox 的资源隔离 ⭐⭐⭐

**对应需求**：CPU LLM 推理与 tool execution 并发时的 L3/DRAM 带宽干扰

**核心思路**：CPU LLM 推理（vllm CPU backend）与 tool execution 并发时，tool 可能抢占 DRAM bandwidth、LLC occupancy、core time，导致 TTFT 或 decode p99 变差。利用 chiplet/NUMA 作为隔离边界，用 `resctrl` CAT/MBA 做 L3 和 memory bandwidth 隔离。

该方向不要求 tool 私有数据跨 CCD。它利用 chiplet/NUMA 作为隔离边界，目标是 tail latency isolation。

**具体实现路径**：
- vLLM worker 固定一组 CCD/NUMA domain
- Tool sandbox pool 固定另一组
- 用 `resctrl` CAT/MBA 做 L3 和 memory bandwidth 隔离
- Agent scheduler 根据 LLM phase 调整 sandbox 并发

**前提条件**：
- 推理和 tool 执行必须同时发生（而非串行），这需要多 agent 并发或流水线
- 需先确认 tool 并发显著扰动 vLLM 的 TTFT、decode p99 或 DRAM bandwidth

**风险**：单 agent 下 CPU 利用率很低（7–13%），隔离可能只是浪费核心。该方向的贡献是 tail isolation，不是单独加速 tool 或 attention。

## 关键论文

### 直接相关的 Agent CPU 测量论文

1. **"A CPU-Centric Perspective on Agentic AI"** (Raj et al., arXiv:2511.00739, 2025–2026)
   - 首次系统性地从 CPU 视角 profiling agentic workload
   - Tool 处理占 90.6% 延迟；CPU 占 44% 能耗
   - 提出 COMB（CPU-Aware Overlapped Micro-Batching）和 MAS（Mixed Agentic Scheduling）
   - Profiled: Haystack RAG, Toolformer, ChemCrow, LangChain, SWE-Agent
   - **与 chiplet 的关系**：论文识别了 CPU 瓶颈（cache coherence、同步、core over-subscription），但未落到 chiplet 级别

2. **"AgentCgroup"** (Zheng et al., arXiv:2602.09345, 2026)
   - Profiled 144 SWE-rebench 任务，发现 OS 级执行占 56–74% 延迟
   - 内存是并发瓶颈（非 CPU）；峰均比 15.4×；98.5% 突发在 tool 调用
   - 提出 eBPF 驱动的自适应资源控制（sched_ext + memcg_bpf_ops）
   - **与 chiplet 的关系**：AgentCgroup 的 cgroup 层级结构可以天然扩展为 chiplet-aware——每个 chiplet 一个顶层 cgroup，tool-call 子 cgroup 继承 chiplet 亲和性

### Agent Serving 系统论文

3. **"Autellix"** (Luo et al., arXiv:2502.13965, 2025) — UC Berkeley. Program-level scheduling, KV cache affinity. 4–15× throughput vs vLLM.
4. **"ThunderAgent"** (Kang et al., arXiv:2602.13692, 2026) — Unified KV cache + tool resource management. 1.5–3.6× serving throughput.
5. **"Cortex"** (Pagonas et al., arXiv:2510.14126, 2025) — Workflow-aware resource pooling, stage isolation. KV cache memory ↓40%.
6. **"Tokencake"** (Bian et al., arXiv:2510.18586, 2025) — KV-cache-centric multi-agent serving. Latency ↓47%.
7. **"Continuum"** (2025) — KV cache TTL for multi-turn agent scheduling.
8. **"Agent.xpu"** (2025) — Heterogeneous SoC scheduling for agents. **与 chiplet 最接近的工作**，但针对的是异构加速器（CPU+GPU+NPU）而非同构 chiplet。

### Agent 基础设施论文

9. **CubeSandbox** (Tencent, 2026) — 开源微虚拟机沙箱。60ms 冷启动，<5MB overhead，~2000 并发/96-core。
10. **Hyperlight + CodeAct** (Microsoft, 2026) — 微虚拟机 + 代码折叠。52% latency 减少，64% token 减少。

## 与 Chiplet 三篇论文的方法线关系

| 维度 | OLAP/Sorting/CHARM | Agent CPU 需求 |
|------|-------------------|---------------|
| 优化层级 | L3 cache（数据复用/coherence） | **共享 OS 路径 coherence + NUMA page-cache affinity** |
| 优化对象 | 单个 workload 的数据放置 | **多 sandbox 共享内核对象的 CCD 级分片** |
| 数据特征 | 读写混合（coherence 风暴） | **私有数据只读/独占；热点在共享内核对象** |
| 生命周期 | 持久（worker 进程、排序 pass） | **短生命周期（sandbox 1–2 秒），但 session/task 持久** |
| 瓶颈 | L3 miss → DRAM 访问 | **共享 OS 元数据 lock/cache-line bounce、远端 NUMA 页、page-cache miss** |
| 放置决策 | Chiplet_Local vs Chiplet_Mixed（按工作集大小） | **Session-to-NUMA-domain 固定 + OS 路径分片** |

**Agent CPU 需求打开了 chiplet 优化的一个新维度：共享 OS 路径 coherence 和 session locality**，而非三篇论文关注的单 workload L3 局部性。这意味着：
- 方法线不同：不是 "Chiplet_Local vs Chiplet_Mixed" 的缓存容量决策，而是 "如何分片共享 OS 对象以消除跨 CCD 干扰"
- 评估指标不同：不是 L3 hit rate / L3 Miss per task，而是 perf c2c 热点、远端 NUMA 页比例、page-cache hit rate、dentry/inode lock contention
- 测量工具不同：`perf c2c`、lock stat、IBS、NUMA stat 而非 PCM `Another CCX same node Demand Fill`

## 下一步验证

### 关键前提测量

1. **定位跨 CCD 开销的真实来源**：
   - 构造 N 个 sandbox 并发运行相同 repo 上的 `pytest/grep/mypy/npm test`
   - 用 `perf c2c`、IBS、PCM 检查热点来自私有数据、VFS/page cache、cgroup、pipe/socket 还是 remote NUMA
   - 对比默认调度、cpuset-only、cpuset+mems、pool-local cgroup/log/temp-dir
   - 若热点不在共享 OS 路径，停止 Agent chiplet 方向

2. **共享 OS 路径分片的 counterfactual 验证**：
   - 做人工 counterfactual：分片 cgroup、日志、temp dir、IPC acceptor
   - 或固定 session home domain
   - 若分片不能降低 p99，停止该方向

3. **Session locality 的 page-cache 复用测量**：
   - 多次连续 tool calls 间，测量 page fault、远端 NUMA 页访问和 page-cache hit
   - 固定 session home domain 后对比指标变化
   - 若连续 tool calls 访问不同文件集，locality 收益可能很低

### 如果测量正面

可以进行原型实现：
1. 基于 AgentCgroup 的 eBPF/cgroup 框架，添加 chiplet/NUMA 拓扑感知
2. 每个 CCD/NUMA domain 维护独立 sandbox worker pool + 独立 cgroup subtree + 独立 temp/log 目录
3. 与 CubeSandbox 或 Hyperlight 集成，做 chiplet-aware sandbox pool
4. Agent orchestrator 将同一 agent session 固定到同一 pool
5. 在 EPYC 上测量端到端 agent task 的 p99 改善

## 结论

**Agent 场景在 chiplet CPU 上存在有意义的优化空间，但优化对象不是 sandbox 私有数据，而是共享 OS 控制路径的 coherence/NUMA 干扰和 session locality。**

核心逻辑链：
1. Agent 的计算瓶颈不在 LLM 推理（26–44%），而在 CPU 侧的 tool 执行 + 容器初始化（56–74%）
2. Tool 执行的 CPU 瓶颈不在 compute（利用率仅 7–13%），而在**内存**（并发瓶颈、峰均比 15.4×、重试累积）
3. **Sandbox 私有数据不会因运行在不同 CCD 上产生跨 CCD 数据流**。可能的跨 CCD 开销来自共享内核/文件系统元数据（VFS/overlayfs/dentry/inode）、page cache、cgroup 计数器、pipe/socket buffer、日志队列等共享控制路径，以及进程迁移后的远端 NUMA 页
4. Chiplet/NUMA 拓扑为分片这些共享 OS 路径和固定 session locality 提供了自然的隔离边界
5. **这是三篇 chiplet 论文未覆盖的新维度——不是 L3 数据复用优化，也不是"CCD-local DRAM"，而是共享 OS 路径的 chiplet 级分片和 session/page-cache affinity**

## 参考文献

- [A CPU-Centric Perspective on Agentic AI](https://arxiv.org/abs/2511.00739) (Raj et al., 2025–2026)
- [AgentCgroup: Understanding and Controlling OS Resources of AI Agents](https://arxiv.org/abs/2602.09345) (Zheng et al., 2026)
- [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965) (Luo et al., 2025)
- [ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System](https://arxiv.org/abs/2602.13692) (Kang et al., 2026)
- [Cortex: Workflow-Aware Resource Pooling and Scheduling for Agentic Serving](https://arxiv.org/abs/2510.14126) (Pagonas et al., 2025)
- [Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](https://arxiv.org/abs/2510.18586) (Bian et al., 2025)
- [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](https://arxiv.org/abs/2511.02230) (2025)
- [Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC](https://arxiv.org/abs/2506.24045) (2025)
- [CubeSandbox (Tencent, Apache 2.0)](https://github.com/tencent/cubesandbox)
- [Hyperlight + CodeAct](https://devblogs.microsoft.com/agent-framework/codeact-with-hyperlight/) (Microsoft, 2026)
- [How Agentic AI Strains Modern Memory Hierarchies](https://www.theregister.com/2026/01/28/how_agentic_ai_strains_modern_memory_heirarchies/) (The Register, Jan 2026)
- [G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems](https://bytez.com/docs/neurips/116187/paper) (NeurIPS 2025)

## 相关内部文档

- 三篇 chiplet 论文解读：`wiki/论文解读/`
- 2026-05-06 ACC 结论：`wiki/实验结果解读/2026-05-06_Qwen3-30B-A3B_CPU_Attention_ACC局部性优化结论.md`
- LLM 推理链路扫描：`wiki/资料总览/LLM推理链路Chiplet优化空间扫描.md`
- RAG 调研：`wiki/资料总览/RAG_Chiplet优化空间调研.md`
