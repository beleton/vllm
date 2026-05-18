# The Fake-Busy and True-Idle Problems of Running Graph Applications on Chiplet-Based Multi-Cores 解读

## 说明
- 依据版本：ISPASS 2025 会议论文本地 PDF 版，原始资料见 `../原始资料/papers/The_Fake-Busy_and_True-Idle_Problems_of_Running_Graph_Applications_on_Chiplet-Based_Multi-Cores.pdf`。
- 论文研究对象是 chiplet-based OoO multi-core 上的图应用，不是 LLM 推理或 attention。

## 问题
Chiplet-based 多核把 core-die、M/I/O-die 和 DRAM 分开后，local LLC miss 需要经过 IS 接口、data switch 和 memory controller 才能到 DRAM（Fig. 1）。图应用的访问模式不规则，分支多，容易在等待 branch-resolving load 时把 ROB 和 LQ 填满。论文要回答的是：这类 workload 的 pipeline 资源到底是在做有效工作，还是被投机执行和 squash 浪费。

## 核心概念
- **fake-busy**：core 等待分支解析 load 时，投机跳过当前 branch，先执行下一轮循环并继续发出新的 speculative branch-load。off-chiplet load latency 很长时，ROB 和 LQ 很快被填满，整条 pipeline 停住。
- **true-idle**：core 先在 branch 内部投机执行，但 branch 内的指令仍在等缺失的 branch-load，能被 fetch、decode、rename，却不能真正执行，ROB 里堆的是 stalled instructions。
- 两者都来自同一类控制流依赖，只是投机路径不同。论文把它们视为 graph traversal 中常见的两种失效形态。

## 方法与实验
论文用 gem5 做系统仿真，采用 x86_64 OoO cores 和 Ruby memory model，并用 HeteroGarnet 模拟 IS 和 DS，McPAT 负责功耗相关结果（Sec. 2）。Benchmark 包括 SPEC CPU2017 的 `omnetpp`、`leela`、`deepsjeng`、`mcf`，以及 GAPBS 的 `bc`、`bfs`、`cc`、`pr`、`sssp`、`tc`（Sec. 2）。

Branch misprediction ratio 的定义是 committed but mispredicted conditional branches / all committed conditional branches（Fig. 2）。

## 观察
- 平均 branch misprediction ratio 为 9.06%，说明 graph workloads 里的误判不是边缘事件，而是常态（Fig. 2）。
- `omnetpp`、`bc` 的 misprediction 相对低，论文把它归因于周期性或半规律访问模式对 TAGE 更友好。
- `leela`、`deepsjeng`、`mcf`、`tc` 更偏 true-idle；`bfs`、`cc`、`pr`、`sssp` 更偏 fake-busy。两者会在不同图区域交替出现，论文称它们是 mutualistic（Fig. 2）。

## 结果与解释
ROB 基线是 352 entries，LQ 基线是 128 entries（Fig. 3, Fig. 4）。

| 结构 | 基线 | 75% 容量 | 50% 容量 | 论文结果 |
| --- | --- | --- | --- | --- |
| ROB | 352 entries | 264 entries | 176 entries | 平均执行时间只增加 2.8% / 12.3%，排除 `sssp` 和 `tc` 后，50% ROB 的增幅为 4.3% |
| LQ | 128 entries | 96 entries | 64 entries | 执行时间变化很小 |

论文给出的直接解释是：大量 ROB 和 LQ 入口被分支误判后的 squash 指令，或等待 branch-resolving load 的指令占用，所以减少容量对总性能的影响有限（Fig. 3, Fig. 4）。

## 分析
这篇论文的核心不是“ROB/LQ 越小越好”，而是“对 graph workloads，pipeline 队列里很多占位并不对应有效工作”。它说明图应用在 chiplet OoO 多核上的主要矛盾，不是单纯的队列容量，而是长延迟 off-chiplet 访问叠加分支投机带来的无效占用。

论文没有提出新的调度、编译或硬件结构，只用敏感性实验证明：当前 ROB/LQ 配置对这类 workload 留有明显 slack。这个结论是 workload-specific 的，不能直接外推成通用设计规则。

## 边界
- 研究对象是 graph applications，不是 LLM serving、attention 或通用 streaming kernel。
- 结论依赖 chiplet-based OoO multi-core、长 off-chiplet latency 和不规则图遍历；对规则流式 workload 不一定成立。
- 实验基于 gem5 模拟，不是实机测量；数值反映的是该模型和该 benchmark set 下的行为。

## 可迁移点
- 评估 chiplet CPU 上的性能瓶颈时，不能只看 ROB 和 LQ 是否“满”，要区分有效执行、投机占用和 squash 占用。
- 对分支密集、load 依赖强的 workload，branch prediction 和 speculative control flow 可能比简单队列容量更关键。
- 这类分析适合作为 graph runtime、编译器和微架构设计的诊断框架参考。

## 不可直接迁移点
- 不能据此推出通用 CPU 应该缩小 ROB 或 LQ。
- 不能直接外推到 attention、prefill、decode 这类以 streaming 为主的 workload。
- 不能把“减少 ROB/LQ 影响不大”理解为所有 graph workload 都同样如此；这篇论文只覆盖 SPEC 和 GAPBS 这一组样本。

## 证据
- 原始资料：`../原始资料/papers/The_Fake-Busy_and_True-Idle_Problems_of_Running_Graph_Applications_on_Chiplet-Based_Multi-Cores.pdf`
- 关键锚点：摘要；Fig. 1-4；Sec. 1-3
