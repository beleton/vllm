# A CPU-Centric Perspective on Agentic AI 解读

**一句话概述**：Agentic AI 在 LLM 推理之外引入了大量 CPU 工具处理（检索、摘要、代码执行、Web/API 访问等），论文发现这些 CPU 阶段可占端到端延迟的 90.6%，大 batch 下 CPU 能耗占比可达 44%，吞吐瓶颈也不再只来自 GPU。为此，作者在 Intel Emerald Rapids + NVIDIA B200 平台上系统测量了 5 类代表性 Agentic workload 的 latency、throughput 和 energy，并据此提出两种调度优化——面向同构 CPU-heavy 负载的 CGAM（micro-batching）和面向异构混合负载的 MAWS（自适应并行策略），分别将 P50 延迟最高降低 2.1x 和 1.41x。

---

## 背景：Agentic AI 基础知识

### 从单次 LLM 推理到 Agentic AI
传统 LLM 服务通常是单次推理流程：用户输入 prompt，模型生成 answer，服务结束。系统性能分析也常围绕 GPU 上的 prefill、decode、KV cache、batching 和 kernel efficiency 展开。

Agentic AI 在 LLM 外增加了编排器、工具、记忆和循环控制，使模型或宿主程序能够根据中间结果继续行动。典型流程是：

```
用户问题
  -> LLM/Host 判断下一步
  -> 调用工具
  -> 读取工具结果
  -> 再次调用 LLM
  -> 继续调用工具或输出最终答案
```

论文把这种系统描述为"在 monolithic LLM 之上增加 decision-making orchestrator 和 external tools"，工具包括 web search、Python interpreter、contextual database 等。Agentic AI 因此从"被动文本 oracle"变为可计划、调用工具、记忆历史步骤并动态调整执行路径的系统（Abstract）。

### Agent、orchestrator 与 tool call

**Agent** 指围绕目标自动推进任务的执行单元。它可以调用 LLM，也可以调用外部工具，并根据中间反馈决定后续动作。

**Orchestrator（编排器）** 是决定下一步执行什么的控制组件。它负责选择是否继续推理、调用哪个工具、如何把工具结果放回上下文、何时结束任务。论文按 orchestrator 所在位置将 Agentic AI 分为两类（Sec. 3.1, Fig. 1）：

- **LLM-orchestrated**：LLM 自己决定控制流，例如 ReAct、AutoGPT、BabyAGI、CAMEL、MetaGPT。
- **Host-orchestrated**：Python/host 程序决定控制流，例如 LangChain、Semantic Kernel、Haystack、LlamaIndex、DSPy。LLM 更像无状态推理引擎，host 程序负责任务调度、工具调用和结果聚合。

**Tool call（工具调用）** 指 Agent 请求外部工具完成非语言模型擅长的操作。工具可以是 WolframAlpha 计算器、Google search、FAISS 检索、Arxiv/Pubmed 文献搜索、Bash/Python 执行、文件 I/O、LexRank 摘要器等（Table 1, Appendix A）。这些工具大多运行在 CPU、操作系统、网络或外部 API 上，不属于 GPU 上的模型推理。

### Static path、dynamic path 与 single-step、multi-step

论文从执行路径和重复性两个维度补充分类（Sec. 3.2, Sec. 3.3, Fig. 1）。

**Static path（静态路径）** 指工具调用顺序预先固定。例如 RAG pipeline 通常是检索 -> 拼接上下文 -> LLM 生成。系统仍可并行处理多个请求，但单个请求内部路径较确定。

**Dynamic path（动态路径）** 指执行图在运行时由中间结果决定。例如 SWE-Agent 可能先读文件，再执行测试，再修改代码，再根据错误继续搜索；ChemCrow 可能根据化学问题选择不同文献或分子工具。

**Single-step（单步）** 指一次工具/推理链路即可完成任务，例如单轮 QA、zero-shot tool use、RAG。

**Multi-step（多步）** 指需要反复观察、计划、行动和修正，例如 WebArena、AgentBench、SWE-Agent、ChemCrow。多步流程会放大 CPU 工具执行、上下文切换、同步和外部 I/O 的影响。

### RAG、ENNS 与工具侧 CPU 开销

RAG（Retrieval-Augmented Generation，检索增强生成）把外部文档检索结果放入 prompt，再由 LLM 生成答案。检索阶段可以用精确最近邻搜索或近似最近邻搜索。

**ENNS（Exact Nearest Neighbor Search，精确最近邻搜索）** 会精确比较查询向量与文档向量。它比 ANNS（Approximate Nearest Neighbor Search，近似最近邻搜索）开销更高。论文引用的 QA-RAG 研究显示，在 K=1 和 K=16 下，ENNS 在多种模型上比 ANNS 有更高生成准确率，并在 throughput-accuracy Pareto frontier 上占优。本文 Haystack 使用 C4 英文语料 305 GB、FAISS FLAT、top-5 ENNS；文档规模超过 GPU 显存，因此检索放在 CPU 侧（Sec. 3.4.4, Appendix A.3）。

### CPU 并行：multi-threading 与 multi-processing

Agentic AI 的 CPU 工具阶段常需要并行处理多个请求。论文区分两种 CPU 并行方式（Sec. 2.2, Sec. 4.3.1）：

- **Multi-threading（多线程）**：一个进程内启动多个线程，共享内存空间。线程创建和切换开销较低，适合共享大数据结构，但 Python GIL 和同步开销可能限制 CPU-bound 任务。
- **Multi-processing（多进程）**：启动多个独立进程，每个进程有独立地址空间。能绕开 Python GIL，适合 CPU-bound 工具任务，但会带来更高进程开销和内存复制压力。

论文在 LangChain workload 上比较二者：batch size 128 时，multi-processing 相比单核基线达到 26.8x 加速，相比 multi-threading 达到 1.6x 加速（Fig. 3）。Haystack 不采用 multi-processing，因为 305 GB C4 检索语料使独立进程复制内存不可行，论文改用 multi-threading 共享内存（Sec. 4.3.1）。
## 研究问题

论文反对只从 GPU kernel 或 KV-cache scheduling 的视角分析 Agentic AI。Agentic AI 的执行链路中，LLM 推理只是其中一段；工具处理、host orchestration、检索、摘要、代码执行和 API 访问会引入大量 CPU、内存、同步和 I/O 开销（Sec. 1, Sec. 2.1）。

作者提出三个问题：

1. Agentic AI workload 应该如何按系统行为分类，而不是只按算法能力分类。
2. CPU 工具处理、CPU 并行和 CPU 能耗在端到端 Agentic AI 中占多大比例。
3. 是否能通过 CPU/GPU 联合调度改善 homogeneous workload（一批请求包含相同或极度相似的资源需求特征）和 heterogeneous workload（一批请求中存在不同资源特征和偏好）的 latency、throughput 与 energy。

## 核心内容

1. **系统分类**：按 orchestrator、path、flow/repetitiveness 三个正交维度刻画 Agentic AI（Sec. 3, Fig. 1）。
2. **系统 profiling**：选取 5 个 workload，测量延迟分解、吞吐饱和和动态能耗（Sec. 4, Fig. 2-5）。
3. **调度优化**：提出 CGAM 与 MAWS。CGAM 面向 homogeneous CPU-heavy agentic workload，限制 batch cap 并做 micro-batching；MAWS 面向 CPU-heavy 与 LLM-heavy 混合 workload，自适应选择 multi-processing 或 multi-threading（Sec. 5, Fig. 6-9）。

## 代表性 Workload

论文选择 5 个 workload 覆盖不同 orchestrator、path、flow 和工具类型（Table 1, Appendix A）。

| Workload   | Orchestrator |    Path |        Flow | 主要工具                                                | 测试任务                                     |
| ---------- | -----------: | ------: | ----------: | --------------------------------------------------- | ---------------------------------------- |
| Toolformer |          LLM | Dynamic | Single-step | WolframAlpha calculator                             | ASDiv、SVAMP、MAWPS 数学题                    |
| SWE-Agent  |          LLM | Dynamic |  Multi-step | File I/O、Bash/Python execution                      | APPS、BigCodeBench、DS-1000 代码任务           |
| Haystack   |         Host |  Static | Single-step | FAISS FLAT ENNS retrieval                           | NQ、HotpotQA、TriviaQA                     |
| ChemCrow   |          LLM | Dynamic |  Multi-step | Arxiv/Pubmed literature search、化学工具                 | nicotine、warfarin、caffeine、aspirin 化学 QA |
| LangChain  |         Host |  Static | Single-step | Google search、LexRank summarizer、Python interpreter | FreshQA、MusiQue、QASC                     |

模型和软件环境也不同：Toolformer 使用 GPT-J 6B；SWE-Agent 使用 Qwen2.5-Coder-32B；ChemCrow 使用 GPT-4-0613 OpenAI API；LangChain 使用 GPT-OSS-20B；本地推理使用 PyTorch 2.8.0 和 vLLM 0.11.0（Appendix A）。

## 实验观察

论文在 48-core Intel Emerald Rapids CPU + NVIDIA B200 GPU 上进行了三组 profiling 实验（能耗实验因设备限制使用 AMD Threadripper + H200）。以下为三个核心发现。

### 观察一：CPU 工具处理主导端到端延迟

5 个 workload 的端到端时间分解（Sec. 4.1, Fig. 2）：

- **Haystack RAG**：ENNS retrieval 是主瓶颈。NQ、HotpotQA、TriviaQA 上检索分别消耗 6.0 s、8.0 s、7.7 s，占总时间 84.5%-90.6%；LLM inference 不超过 0.5 s（Fig. 2a, Sec. 4.2）。
- **Toolformer**：第一次 GPT-J 6B inference 约 1 s；WolframAlpha API 添加 1.4-1.7 s；最终 inference 也较稳定，因为工具输出只增加少量 token（Fig. 2b, Sec. 4.2）。
- **ChemCrow**：文献搜索工具消耗 4.0-10.1 s，包含 Arxiv paper search、download/read、Pubmed search 和 abstract read；GPT-4-0613 final inference 消耗 5.6-7.6 s，总时间 12.6-18.8 s（Fig. 2c, Sec. 4.2）。
- **LangChain**：web search 最高 4.2 s，LexRank summarization 最高 3.5 s，工具阶段可超过端到端延迟的一半。作者指出，约束 web search 和 summarization 的网站数量是主要优化手段，而不只是替换 base model（Fig. 2d, Sec. 4.2）。
- **SWE-Agent**：Bash/Python execution 分别占 APPS、BigCodeBench、DS-1000 总延迟的 64.7%、78.7%、43.8%（Fig. 2e, Sec. 4.2）。

**核心结论**：CPU 工具处理对 Agentic AI 延迟影响显著，最高达到 90.6%，优化对象不能只放在 GPU inference 上。

### 观察二：吞吐由 CPU 或 GPU 共同限制

**CPU 并行选择**：LangChain 的 `Runnable.batch` 使用线程池实现 multi-threading；multi-processing 则启动多个独立 Python 进程并绕开 GIL。低 batch size 下二者接近；batch size 128 时，multi-processing 比单核基线快 26.8x，比 multi-threading 快 1.6x（Fig. 3, Sec. 4.3.1）。
Haystack 检索面向1个RAG库多请求的场景，使用 305 GB C4 corpus，即多个请求加载的是同一份 FAISS 索引和 mmap 文档，多个进程独立持有内存会导致内存压力过高，因此论文对 Haystack 选择 multi-threading，其余 workload 主要使用 multi-processing（Sec. 4.3.1, Appendix A.3）。

**GPU throughput saturation**：vLLM + GPT-OSS-20B 在 batch size 1-128 下测试不同输入/输出 token 长度。throughput 在 batch size 64 前接近线性增长，之后增益趋缓。作者将原因归为 KV cache 随 batch 增长导致 GPU HBM 容量和带宽压力，必要时还会触发 GPU-host offloading，受 PCIe 带宽限制（Fig. 4a, Sec. 4.3.2）。

**CPU throughput saturation**：CPU 侧吞吐不一定随 core 数线性增长，论文讨论了三类 CPU 限制（Sec. 4.3.3）：
- **cache coherence**：共享 cache line、false sharing 和 MESI 协议带来的 coherence traffic 会限制扩展性。
- **synchronization**：barrier、lock、atomic 操作会让慢线程决定整体进度，形成 straggler。
- **over-subscription**：进程数超过可用 core 后，OS scheduler contention、context switching 和 cache/TLB state 丢失会主导开销。

Fig. 4b 展示不同 workload 的 batch scaling：Toolformer 增益从 batch size 2 的 2.8x 降到 batch size 128 的 1.4x；Haystack 在 batch size 32 后受 LLC pressure 和 disk I/O contention 限制；LangChain 和 SWE-Agent 在 batch size 128 受 core over-subscription 限制。Fig. 4c 中，LangChain summarization 平均延迟从 batch size 64 的 2.9 s 升到 batch size 128 的 6.3 s，LLM inference 从 2.6 s 升到 3.9 s（Fig. 4b-c, Sec. 4.3.4）。

**核心结论**：Agentic AI 的 throughput 可能被 CPU 侧的 core over-subscription、cache coherence、synchronization 限制，也可能被 GPU 的 device memory capacity 和 bandwidth 限制。

#### 深入分析：为什么 LangChain multi-processing 只快 1.6x 而非 128x？
直觉上，Python 多线程受 GIL 限制应该是串行的，128 进程理应比单进程多线程快 128 倍。但实际只快 1.6x，原因在于这个 workload 的特点和实现方式。
从 `figure_3.sh` 的执行命令可以直接看出（代码仓库 `langchain/`）：
- **Multi-threading**：单个 Python 进程，`--batch-size 128`，内部 `RunnableConfig(max_concurrency=128)`，LangGraph 用线程池并发执行 128 个 query。
- **Multi-processing**：128 个独立 `python orchestrator.py --skip-web-search &` 后台进程，每个进程默认 `batch_size=1`，独自处理 1 个 query。
区别：1 个进程带 `--batch-size 128` 靠线程池并发，还是 128 个进程各带默认 `batch-size 1` 靠 OS 调度并发。

**Multi-threading GIL 在三个关键阶段都被绕开了。** 看 `orchestrator.py` 的 pipeline：
1. **URL 抓取（网络 I/O）**：`requests.get(url)` 等待 HTTP 响应时释放 GIL，其他线程可以继续执行。
2. **LexRank 摘要（子进程）**：代码在 `summarize()` 内部使用了 `ProcessPoolExecutor(max_workers=1)`，把 `_lexrank_one()` 放到独立子进程中执行——这部分已经绕过 GIL，即使在上层"多线程"模式下。128 个 query 会产生 128 个 LexRank 子进程，真正在多核上并行。
3. **LLM 推理（HTTP 调用 vLLM）**：`llm.invoke()` 是到 `localhost:5000` 的 HTTP 请求，同样释放 GIL。
所以这个 workload 中，真正被 GIL 串行化的只有 LangGraph 线程池的编排开销（调度、数据传递、结果拼接），CPU 密集计算（LexRank）和 I/O 密集操作（URL fetch、vLLM 调用）都不受 GIL 制约。

### 观察三：大 batch 下 CPU 能耗不可忽略

能耗实验因设备限制在另一台机器上进行：AMD Ryzen Threadripper PRO 7985WX 64-core + NVIDIA H200 GPU。CPU energy 用 pyRAPL 读取 RAPL counter；GPU energy 用 `nvidia-smi` 每 100 ms 采样 board power 并做梯形积分；CPU idle 113 W、GPU idle 115 W，从总功耗中扣除得到 dynamic power（Sec. 4.4）。

LangChain FreshQA 上，batch size 从 1 增到 128 时，总 dynamic energy 从 108 J 增到 4114 J，增长 38.1x。GPU dynamic energy 从 86 J 增到 2307 J，增长 26.8x；CPU dynamic energy 从 22 J 增到 1807 J，增长 86.7x。CPU dynamic energy 占比从小 batch 的 20% 增到 batch size 128 的 44%（Fig. 5, Sec. 4.4）。

**核心结论**：大 batch 下 CPU dynamic energy 占比可达到 44%，CPU multiprocessing 的能效低于 GPU parallelism。作者也明确说明，能耗趋势是在不同硬件上测量的，后续多硬件配置研究属于 future work（Sec. 4.4, Sec. 8）。

---

## 方法一：CGAM

CGAM（CPU and GPU-Aware Micro-batching，CPU/GPU 感知微批处理）面向 homogeneous agentic workload，目标是在吞吐已经饱和时降低 P50/P90 延迟、减少 KV cache 占用和 CPU 能耗（Sec. 5.1, Fig. 6）。

### Bcap 选择

论文定义 batch size `B` 下吞吐为 `T(B)`，吞吐增益比为：
```
r(B) = T(B) / T(B/2)
```

选择 `Bcap` 的准则是：当继续翻倍 batch 只能带来低于阈值 `λ` 的吞吐增益时停止。论文设置 `λ = 1.1`，即 batch 翻倍带来的吞吐增益低于 10% 时不再继续增大 batch（Eq. 1, Sec. 5.1.1）。

Table 2 给出三个 workload 的结果：

| Workload | r(64) | r(128) | Bcap |
|---|---:|---:|---:|
| LangChain | 1.52 | 1.09 | 64 |
| Haystack | 1.15 | 1.08 | 64 |
| SWE-Agent | 1.32 | 1.10 | 64 |

因此论文在后续 CGAM 中使用 `Bcap = 64`（Table 2）。

### CGAM 执行方式

基线 multi-processing 会一次性并行执行全部 128 个请求。CGAM 将 128 个请求拆成两个 64 请求的 micro-batch，按 micro-batch 顺序执行。每个时刻只让最多 `Bcap` 个请求占用 CPU/GPU 资源（Fig. 6）。

论文给出三项预期收益（Sec. 5.1.2）：

- **P50 latency 约 2x 改善**：第一个 micro-batch 在总时间约一半处完成。
- **KV cache usage 约减半**：同一时刻只运行一半 batch，GPU KV cache 压力下降。
- **CPU energy 约下降**：限制并行 core 数。论文以 `128 / 64 = 2x` 估算最大 core 使用下降带来的 CPU energy 降低。

### CGAM overlap

CGAM overlap 在第一个 micro-batch 的 CPU 工具阶段结束后，立即启动第二个 micro-batch 的 CPU 工具阶段，同时第一个 micro-batch 在 GPU 上做 LLM inference。它利用空闲 CPU 换取更低 tail latency，但并发阶段会增加 CPU contention，因此 P50 通常不如普通 CGAM，P90 更好（Fig. 6, Sec. 5.1.3）。

## 方法二：MAWS

MAWS（Mixed Agentic Workload Scheduling，混合 Agentic workload 调度）面向 heterogeneous workload。论文区分两类请求（Sec. 5.2）：

- **CPU-heavy**：工具执行主导，例如 web search + LexRank summarization + LLM 的 LangChain pipeline。
- **LLM-heavy**：LLM inference 主导，工具很轻，例如 guardrail -> LLM inference，其中 guardrail 只是简单 if-else 判断恶意 prompt。

若所有请求都用 multi-processing，LLM-heavy 请求也会占用大量 CPU 进程，导致 CPU-heavy 请求更容易 over-subscription。MAWS 对 CPU-heavy 任务使用 multi-processing，对 LLM-heavy 任务使用较轻的 multi-threading 来并行 vLLM API I/O，释放 CPU 资源给真正 CPU-heavy 的工具任务（Sec. 5.2）。
这里的多进程/多线程是对python Agentic AI场景而言的，python的多线程受限于GIL全局解释器锁，只能串行执行，而多进程才能正在进行并行

## 实验设置

Profiling 和调度评估使用 Intel Emerald Rapids CPU + NVIDIA B200 GPU，CPU 为 48-core、每 core 2 threads，内存为 DDR5，GPU 为 HBM3e（Sec. 4.1, Sec. 6.1）。

CGAM 和 MAWS 评估采用 closed-loop arrival system：所有 `B` 个请求在 `t=0` 同时到达。CGAM 和 MAWS 单独评估时 `B=128`；MAWS+CGAM 组合评估时 `B=256`（Sec. 6.1）。

CGAM 评估 workload 为 Haystack RAG、LangChain、SWE-Agent，因为它们有显著本地 CPU 工具处理：ENNS retrieval、LexRank summarization、Bash/Python execution。Toolformer 和 ChemCrow 不用于 CGAM 评估，因为它们包含大量外部处理：WolframAlpha API 和 OpenAI API（Sec. 6.1）。

基线设置与 Sec. 4.3.1 保持一致：LangChain 和 SWE-Agent 因 CPU-bound tool processing 使用 multi-processing；Haystack 因 C4 corpus 过大使用 multi-threading。所有评估报告单次运行时间，多次运行观察到约 5% statistical variance，对 P50 speedup 影响较低（Sec. 6.1）。

## 结果与解释

### CGAM 结果

在 `Bcap=64` 下，CGAM 相比基线在三个 workload 上降低 P50 latency（Fig. 7, Sec. 6.2）：

- LangChain FreshQA：2.11x，11.21 s -> 5.32 s。
- Haystack NQ：1.94x，42.87 s -> 22.12 s。
- SWE-Agent APPS：1.72x，65.08 s -> 37.82 s。

CGAM 使用总 CPU 的 3/4，即 96 个 CPU hardware threads，同时保持近似相同 tail latency。论文在等功耗假设下估算 CGAM 可节省约 1.5x CPU dynamic energy（Fig. 7, Sec. 6.2）。

CGAM overlap 的 P50 改善分别为 1.69x、1.82x、1.37x，P90 改善分别为 1.33x、1.15x、1.16x，对应 LangChain、Haystack、SWE-Agent。论文解释为：overlap 让第二个 micro-batch 更早开始，因此 P90 更好；但 CPU/GPU overlap 阶段带来更多 CPU contention，因此 P50 弱于普通 CGAM。LangChain 的 P90 改善更明显，因为它的 CPU 和 GPU 执行时长更接近，overlap 更有效；Haystack 和 SWE-Agent 更偏 CPU-dominant（Fig. 6, Fig. 7, Sec. 6.2）。

### MAWS 结果

MAWS 使用两条 LangChain pipeline 构造异构 workload：一半是 CPU-heavy 的 profiling pipeline，另一半是 LLM-heavy 的 guardrail -> LLM inference pipeline。128 个 mixed LangChain tasks 上，MAWS 相比 multi-processing 基线的 P99 latency 改善 1.17x，同时保持接近的 P50 latency（Fig. 8, Sec. 6.3）。

### MAWS+CGAM 结果

MAWS+CGAM 在 256 个 mixed LangChain tasks 上评估，其中 128 个 CPU-heavy tasks 达到吞吐饱和。相比 multi-processing 基线，MAWS+CGAM 的 P50 latency 改善为：CPU-heavy tasks 2.1x、LLM-heavy tasks 1.2x、all tasks 1.4x；overall P99 latency 改善 1.15x（Fig. 9, Sec. 6.4）。

## 分析

- 论文的核心贡献不是提出新的 LLM 算法，而是把 Agentic AI 的系统瓶颈从 GPU-only 视角扩展到 CPU/GPU 联合视角。
- Agentic AI 的 CPU 开销来自工具处理与编排路径，不等同于传统 LLM serving 中的轻量 tokenizer 或 HTTP wrapper。Haystack 的 305 GB ENNS、LangChain 的 LexRank、SWE-Agent 的 Bash/Python execution 都是本地 CPU workload（Fig. 2, Appendix A）。
- CPU 并行策略必须按工具特征选择。multi-processing 适合 CPU-bound、内存占用可控的工具；multi-threading 适合需要共享大内存数据结构的 Haystack 检索（Sec. 4.3.1）。
- 大 batch 的优化目标不能只看 throughput。Fig. 4 显示吞吐会在 CPU 或 GPU 端饱和，继续增大 batch 会提高 P50/P90/P99 latency 和 CPU energy。CGAM 通过 `Bcap=64` 把"吞吐增益低于 10%"作为停止扩 batch 的判据（Table 2, Eq. 1）。
- MAWS 的本质是按请求类型限制 CPU 消耗。LLM-heavy 请求不应以多进程方式抢占 CPU core；它们主要等待 GPU inference，用 multi-threading 做 API I/O 即可（Sec. 5.2）。

## 边界

- 论文不是 chiplet CPU 论文，也没有研究 CCD/L3 分片、NUMA placement 或跨 chiplet cache locality。主 profiling 平台是 Intel Emerald Rapids CPU + NVIDIA B200 GPU；energy 平台是 AMD Threadripper PRO 7985WX + NVIDIA H200 GPU（Sec. 4.1, Sec. 4.4）。
- 论文未给出 CPU cache miss、LLC occupancy、NUMA remote access、IPC、context switch count 等底层计数器，只在 throughput 解释中讨论 coherence、synchronization 和 over-subscription（Sec. 4.3.3）。
- CGAM/MAWS 是调度策略，不是工具实现优化。它们不改变检索算法、摘要算法、Bash/Python 执行逻辑或 LLM kernel。
- 评估采用 closed-loop arrival system，所有请求在 `t=0` 同时到达；这与真实在线服务中的随机到达、优先级队列、超时和重试机制不同（Sec. 6.1）。
- ChemCrow 使用 OpenAI API，Toolformer 使用 WolframAlpha API。这些外部服务延迟不可完全由本地 CPU/GPU 调度控制，因此论文未把它们纳入 CGAM 评估（Sec. 6.1）。
- 论文明确把 diverse hardware configurations 作为 future work，不能把 B200/Emerald Rapids 上的阈值直接外推到其他 CPU/GPU 平台（Sec. 8）。

## 可迁移点
- 当前研究若讨论 Agent/RAG/Tool sandbox，必须把 CPU 工具阶段作为一等 workload，而不是仅把它看作 LLM inference 的外围逻辑。
- 对 Agentic AI 做性能分析时，应拆分 LLM inference、retrieval、summarization、Python/Bash execution、web/API I/O、orchestration，而不是只测端到端时间。
- CPU 工具阶段的并行方式需要按数据共享和内存占用选择：大共享语料检索优先 multi-threading；CPU-bound 且内存可复制的工具可用 multi-processing。
- batch size 选择可采用 CGAM 的思想：用 `T(B) / T(B/2)` 判断吞吐增益是否进入低收益区，而不是盲目增加 batch。
- heterogeneous agentic serving 需要区分 CPU-heavy 与 LLM-heavy 请求。前者保留 CPU 并行资源，后者降低 CPU footprint，避免无效进程并发造成 over-subscription。

## 不可直接迁移点
- 论文没有验证 chiplet/CCD 局部性，因此不能直接支持"Agentic AI 适合 chiplet L3 locality 优化"的结论。
- Haystack 的 ENNS 使用 305 GB C4 corpus 和 FAISS FLAT，代表大规模精确检索；它不能代表所有 RAG 系统，尤其不能代表小索引、GPU ANN、HNSW 或 IVF 检索。
- LangChain 的 LexRank summarization 是作者为避免 LLM summarizer 幻觉和节省 GPU 成本而选择的 CPU 摘要器；真实产品若使用 LLM summarizer，CPU/GPU 分解会不同（Sec. 3.4.6, Appendix A.5）。
- SWE-Agent 的 Bash/Python execution 延迟与 benchmark、代码执行安全策略、文件系统和 sandbox 实现强相关，不能直接外推到所有 coding agent。

## 与当前 Chiplet CPU 研究的关系

论文能支持的结论是：Agentic AI 的 tool/retrieval/execution 阶段确实可能成为 CPU 主导路径，值得作为 attention 之外的 CPU workload 候选。它不能直接证明 chiplet 优化空间存在。

若要把该论文迁移到 chiplet CPU 研究，需要新增证据：
- 对 Haystack/FAISS ENNS、LexRank、Python/Bash execution 分别测 LLC miss、remote NUMA/CCD access、IPC、context switch、page cache 和 disk I/O。
- 判断 CPU 工具阶段的瓶颈是内存带宽、LLC 容量、锁/同步、进程调度、文件系统、网络/API 还是外部服务。
- 只有当瓶颈包含可通过分片、复制、亲和、隔离或迁移改变的 CPU 数据/执行路径时，chiplet-aware 调度才有研究价值。

因此，本论文适合作为"Agentic AI 有真实 CPU 工具负载"的背景证据，不适合作为"Agentic AI 天然适合 chiplet L3 优化"的直接证据。

## 证据

- 原始资料：`wiki/原始资料/papers/A CPU-Centric Perspective on Agentic AI.pdf`
- 论文元数据：arXiv `2511.00739v2`，DOI `10.48550/arXiv.2511.00739`
- 关键锚点：Abstract；Fig. 1-9；Table 1-2；Sec. 2.1-2.2；Sec. 3.1-3.4；Sec. 4.1-4.4；Sec. 5.1-5.2；Sec. 6.1-6.4；Sec. 8；Appendix A
