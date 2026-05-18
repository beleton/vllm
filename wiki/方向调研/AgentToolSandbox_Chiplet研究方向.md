# Agent Tool Sandbox 的 Chiplet 研究方向

## 结论

Agent tool sandbox 是可做但不宜作为第一主线的系统方向。它的优势是应用新、CPU/OS 开销真实存在、已有论文尚未把问题落到 AMD EPYC 多 `CCD` 拓扑；劣势是系统噪声大、可控性弱、实验归因比 HNSW 困难。

该方向的 chiplet 优化对象不是 sandbox 私有堆、栈和匿名页。若进程不迁移、页面按本地 first-touch 分配，私有数据主要走本核到本地内存路径，不会天然造成跨 `CCD` 共享热点。真正可能结合 chiplet 优化的是：

- 共享 OS 路径：`VFS/overlayfs/page cache/dentry/inode/slab/cgroup/IPC/log queue`
- session/page-cache 亲和
- tool 与 vLLM 混部时的 tail latency 隔离
- sandbox 生命周期频率降低

推荐定位：

```text
per-domain sandbox pool
  + shared OS path sharding
  + session/page-cache affinity
  + vLLM/tool tail isolation
```

该方向适合作为 HNSW 主线之外的系统备选。若要推进，第一阶段必须用 `perf c2c`、lock contention、PSI、page fault、`numa_maps` 和 `resctrl` 证明热点确实来自共享 OS 路径或混部资源争用。

## Agent 基础

### Agent 的含义

`Agent` 指由 LLM 驱动、能够多轮决策并调用外部工具完成任务的系统。普通 LLM serving 的典型输入是 prompt，输出是 token。Agent 的输入是任务目标，输出可能需要经过多轮推理、工具调用、观察结果和重试。

普通 LLM serving：

```text
user prompt
  -> LLM inference
  -> generated text
```

Agent execution：

```text
user task
  -> LLM plans next step
  -> tool call
  -> tool runs in OS/sandbox
  -> observation returned to LLM
  -> LLM decides next step
  -> repeat until done
```

例子：

- 代码修复 agent：读取 repo，运行 `grep`，修改文件，执行 `pytest`，根据错误继续修复。
- 数据分析 agent：读取 CSV，执行 Python 脚本，生成图表，检查异常。
- 浏览器 agent：打开网页，点击元素，读取页面状态，继续操作。
- RAG agent：检索资料，调用数据库或搜索工具，把结果交给 LLM 继续推理。

### Tool Call 的含义

`tool call` 指 LLM 不直接回答，而是请求系统执行一个外部动作。常见 tool 包括：

| Tool 类型 | 例子 | 主要资源 |
| --- | --- | --- |
| 文件系统工具 | `ls`、`grep`、读写文件 | page cache、dentry/inode、磁盘 I/O |
| 代码执行工具 | Python、Node.js、shell script | CPU、内存、解释器启动、依赖加载 |
| 测试工具 | `pytest`、`npm test`、`mypy` | CPU、文件系统、进程创建 |
| 网络工具 | HTTP request、API call | socket、TLS、网络 I/O |
| 数据库/检索工具 | SQL、vector search | 内存、索引、RPC |
| 浏览器工具 | Playwright、Chrome DevTools | 进程树、共享内存、GPU/CPU、文件缓存 |

Tool call 的关键特点是：它把 Agent 从“只跑 LLM 推理”变成“LLM + OS + 文件系统 + 进程管理 + 容器/沙箱 + I/O”的混合 workload。

### Sandbox 的含义

`sandbox` 是隔离 tool 执行环境的机制。Agent 会执行不完全可信的代码、命令或文件操作，因此需要限制权限、资源和可见文件系统。常见实现包括：

- 普通进程 + seccomp/namespace/cgroup
- 容器，如 Docker/containerd/gVisor
- 微虚拟机，如 Firecracker、Hyperlight、CubeSandbox
- 预热环境池，如 warm container pool、snapshot restore

Sandbox 通常提供：

- 文件系统隔离：每个任务有独立 working directory。
- 进程隔离：tool 不能任意访问宿主进程。
- 网络隔离：限制出站连接。
- 资源限制：CPU、内存、I/O、进程数上限。
- 生命周期管理：创建、初始化、执行、收集日志、销毁。

### Agent 端到端流程

一个代码 agent 的端到端路径可拆成：

```text
1. LLM 生成下一步动作
2. Agent runtime 解析 tool call
3. Scheduler 选择 sandbox
4. Sandbox 创建或复用
5. 准备文件系统、环境变量、依赖路径
6. 启动 tool 进程
7. Tool 读取 repo、依赖、缓存、测试文件
8. Tool 输出 stdout/stderr/log
9. Runtime 收集结果并返回给 LLM
10. LLM 根据 observation 继续下一步
```

与 chiplet 相关的是第 3-8 步。它们会触发进程调度、内存分配、page cache、文件系统元数据、IPC、日志队列、cgroup 计数和资源隔离。

## OS 术语

### VFS、dentry、inode

`VFS`（Virtual File System，虚拟文件系统）是 Linux 内核中文件系统的统一抽象。应用执行 `open/read/stat` 时，不直接操作 ext4、xfs 或 overlayfs，而是先经过 VFS。

`dentry` 是目录项缓存，表示路径名到 inode 的映射，例如 `src/main.py -> inode X`。`inode` 是文件元数据对象，保存权限、大小、时间戳、block 映射等。大量 `grep/pytest/npm test` 会反复访问路径、目录和文件元数据，因此 dentry/inode cache 可能成为共享热点。

### Page Cache

`page cache` 是 Linux 用内存缓存文件内容的机制。第一次读取文件可能从磁盘或后端文件系统读取，后续读取可直接命中内存页。

Agent tool 常反复读取：

- repo 源码文件
- Python/Node.js 依赖
- 测试文件
- 编译或解释器缓存
- 配置文件

若同一 session 的连续 tool call 被调度到同一拓扑域，相关文件页和元数据更可能保持在本地 cache 和本地内存路径上。若 tool 在多个 `CCD` 间迁移，可能引入冷启动、远端页访问或重复 cache 填充。

### Overlayfs

`overlayfs` 是容器常用的联合文件系统。它把只读 lower layer 和可写 upper layer 叠在一起，让每个容器看到独立文件系统视图。

典型结构：

```text
lowerdir: 只读基础镜像
upperdir: 当前 sandbox 的写入层
workdir : overlayfs 内部工作目录
merged  : tool 看到的文件系统
```

多个 sandbox 共享同一 lower layer 时，lower 的 page cache、dentry、inode 可能被大量复用；upper/work 目录若集中在同一路径，也可能形成元数据热点。

### cgroup

`cgroup`（control group，控制组）是 Linux 的资源控制机制。它可以限制和统计一组进程的 CPU、内存、I/O、进程数等资源。Agent sandbox 通常用 cgroup 做：

- `cpu.max`：限制 CPU 时间。
- `cpuset.cpus`：限制可运行 CPU。
- `cpuset.mems`：限制可使用 NUMA memory nodes。
- `memory.max/memory.high`：限制内存或提前触发回收。
- `pids.max`：限制进程数量。
- `io.max`：限制块设备 I/O。

多个 sandbox 如果挂在同一个 cgroup subtree 下，内核会频繁更新共享统计和状态。高并发下这些计数器、锁或回收路径可能引入共享 cache line 和锁竞争。

### IPC、pipe、socket、vsock

`IPC`（Inter-Process Communication，进程间通信）是进程之间传递数据的机制。Agent runtime 与 sandbox/tool 之间常用：

- pipe：收集 stdout/stderr。
- Unix domain socket：传控制消息。
- TCP socket：连接本地服务或远端服务。
- vsock：宿主与微虚拟机通信。

若所有 sandbox 共用同一个 acceptor、日志队列或 IPC broker，该共享队列可能成为跨 `CCD` 争用点。

### PSI、resctrl、perf c2c

`PSI`（Pressure Stall Information，压力停顿信息）是 Linux 暴露 CPU、memory、I/O 压力的接口。它能反映任务因资源不足而停顿的时间比例。

`resctrl` 是 Linux x86 资源控制文件系统，可用于：

- `CAT`（Cache Allocation Technology）：限制 LLC 可用 ways。
- `MBA`（Memory Bandwidth Allocation）：限制内存带宽。
- `MBM`（Memory Bandwidth Monitoring）：监测内存带宽。
- `llc_occupancy`：监测 LLC 占用。

`perf c2c` 用于分析 cache-to-cache 共享和 HITM（Hit Modified，命中其他核心修改态 cache line）。如果 Agent 方向要证明 chiplet 相关性，需要看到共享 OS 对象或队列导致跨核/跨 `CCD` cache line 争用，而不是只看到普通 DRAM miss。

## 为什么 Agent 与 Chiplet 可能相关

### 与普通 LLM serving 的差异

普通 LLM serving 的主要开销是模型推理：prefill、decode、KV cache、GEMM/attention。Agent serving 的端到端执行还包含 tool execution、sandbox/container 初始化、文件系统访问、日志、IPC、网络和重试循环。

已有 agent serving 与 agent OS 资源论文表明，CPU/OS 侧开销在 agent 场景中很高。它们主要关注资源控制、workflow scheduling 或 tool environment management，尚未直接研究 AMD EPYC 多 `CCD` 拓扑下的放置和隔离。

### 私有数据不是主要 chiplet 优化对象

一个 sandbox 进程的私有堆、栈和匿名页通常只被该进程访问。若该进程固定在一个 `CCD` 或 NUMA domain，页面由本地 first-touch 分配，访问路径不需要其他 `CCD` 的 cache 参与。

因此，不能把“sandbox 很多、内存很多”直接等同于“跨 chiplet 共享开销很大”。真正需要验证的是：

- 进程是否在 `CCD` 间迁移，导致远端页访问。
- 多个 sandbox 是否共享同一 VFS/page-cache/cgroup/IPC/log 对象。
- tool 与 vLLM 是否争用同一 LLC/DRAM bandwidth/I/O。
- sandbox 生命周期是否频繁触发冷路径。

### 可优化对象

Agent tool sandbox 的 chiplet 优化对象可分为三类：

| 类别 | 对象 | 可控变量 |
| --- | --- | --- |
| 共享 OS 路径 | VFS、overlayfs、dentry/inode、page cache、cgroup、IPC、日志队列 | 分片、独立目录、独立 cgroup subtree、独立 acceptor |
| 会话局部性 | 同一 repo/session/tool-class 的重复文件访问 | session stickiness、warm sandbox pool、page-cache 亲和 |
| 混部隔离 | tool 与 vLLM 争用 core/LLC/DRAM/I/O | cpuset、resctrl CAT/MBA、admission control |

## 研究点一：共享 OS 路径分片

### 含义

共享 OS 路径分片指每个拓扑域维护独立 sandbox pool、cgroup subtree、tmp/log/workdir、IPC acceptor，减少多个 `CCD` 同时访问同一共享内核对象。

```text
agent session / repo / tool class
  -> home topology domain
  -> local sandbox pool
  -> local cgroup subtree
  -> local tmp/log/workdir
  -> local IPC endpoint
```

这里的拓扑域可以是：

- 单个 `CCD`
- NPS domain
- socket

AMD EPYC 上单 `CCD` L3 很小但低延迟，NPS/socket 更容易通过 OS 控制。第一版实验可先做 NPS/socket，再视工具支持细化到 `CCD` 级绑核。

### 为什么可能有效

多 sandbox 并发时，如果所有 sandbox 共用同一个 manager、cgroup subtree、overlayfs 根目录、日志队列和 IPC acceptor，跨 `CCD` 的 worker 会频繁访问同一批内核对象或用户态队列。分片后，每个拓扑域访问本域对象，减少 cache line bounce、锁竞争和远端页访问。

### 可能热点

- `overlayfs` upper/work/lower 元数据
- `dentry/inode` cache
- page cache 和 file workingset
- `memcg` 计数器
- pipe/socket/vsock buffer
- 日志队列
- orchestrator 主线程或 acceptor

### 设计变量

| 变量 | 说明 |
| --- | --- |
| 分片粒度 | `CCD`、NPS domain、socket |
| sandbox pool | 每域固定 warm sandbox 数量 |
| cgroup 结构 | 全局单树 vs 每域 subtree |
| 文件路径 | 全局 tmp/log/workdir vs 每域目录 |
| IPC | 全局 acceptor vs 每域 acceptor |
| 路由键 | session、repo、tool-class、tenant |

### 验证指标

- tool call `p50/p95/p99`
- sandbox cold/warm start latency
- `perf c2c` HITM 热点
- `perf lock contention`
- `memory.stat` 中 `slab`、`workingset_refault_file`、`pgfault`
- `cpu.pressure/memory.pressure/io.pressure`
- `resctrl llc_occupancy/mbm`

### 风险

- 热点可能在磁盘、网络、解释器初始化或外部服务，而非共享 OS 对象。
- 分片后 page cache 和 slab 复制增加内存压力。
- 过细分片会导致负载不均和排队尾部。
- 真实 sandbox 系统复杂，工程噪声可能掩盖 chiplet 增量。

## 研究点二：Session/Page-Cache 亲和

### 含义

`session` 指同一个用户任务或同一个 repo 上的一串连续 agent steps。`page-cache affinity` 指把同一 session 的连续 tool calls 固定到同一拓扑域，使 repo 文件、依赖目录和解释器库保持热度。

```text
session A:
  grep -> edit -> pytest -> mypy -> pytest

without affinity:
  CCD0 -> CCD5 -> CCD2 -> CCD7 -> CCD1

with affinity:
  CCD3 -> CCD3 -> CCD3 -> CCD3 -> CCD3
```

### 为什么可能有效

代码 agent 经常在同一 repo 上多次读取相同文件和依赖。第一次 `pytest` 会加载大量 Python 文件、测试文件和依赖包；后续 `pytest` 或 `mypy` 可能复用 page cache、dentry/inode cache、解释器路径和测试缓存。若这些 tool calls 在同一拓扑域运行，更可能保留本地 cache 和内存局部性。

### 设计变量

- session stickiness TTL
- warm sandbox reuse 次数
- snapshot clone 与 persistent sandbox 切换阈值
- `cpuset.cpus + cpuset.mems`
- repo checkout、venv、cache 目录 first-touch
- 按 tool-class 建池：`pytest/grep/mypy/npm test`

### 验证指标

- minor/major page fault
- `workingset_refault_file`
- `/proc/<pid>/numa_maps`
- `numastat`
- 首次执行与重复执行的 p99 差异
- 同一 session 内连续 tool call 的 cache warmup 曲线

### 风险

- 若连续调用访问文件集差异大，stickiness 只会增加排队。
- warm pool 过大可能挤压内存并触发 reclaim。
- sandbox 复用增加状态清理和隔离风险。
- page cache 亲和可能主要是 NUMA/socket 级收益，不一定能精确落到 `CCD`。

## 研究点三：Tool 与 vLLM 混部隔离

### 含义

混部指 CPU LLM 推理和 agent tool sandbox 同时运行在同一台 chiplet CPU 上。tool 会争用核心、LLC、DRAM bandwidth、I/O 和 page cache，可能导致 `TTFT` 或 decode p99 抖动。

`TTFT`（Time To First Token，首 token 延迟）是从请求到生成第一个 token 的时间。decode p99 是生成阶段单步或端到端延迟的尾部指标。Agent 服务通常既关心 tool 完成时间，也关心 LLM 交互延迟。

### 机制

- vLLM worker 固定一组 `CCD/NPS/socket`
- tool sandbox pool 固定另一组
- `resctrl CAT/MBA` 做 LLC/MBA 隔离
- `cpu.max/memory.high/cpuset` 做资源控制
- 用 `PSI` 和 `memory.events` 做 admission control

### 设计空间

| 策略 | 含义 | 适用条件 |
| --- | --- | --- |
| 空间隔离 | vLLM 和 tool 使用不同 CCD/NPS/socket | tail latency 比平均吞吐更重要 |
| 时间隔离 | decode 高峰期限制 tool 并发 | LLM SLO 严格，tool 可等待 |
| 带宽隔离 | MBA 限制 tool 内存带宽 | tool 造成 DRAM bandwidth 抖动 |
| LLC 隔离 | CAT 给 vLLM 保留 LLC ways | vLLM 或检索端对 LLC 敏感 |
| admission control | PSI 高时暂停新 tool | 系统出现 memory/I/O pressure |

### 指标

- `TTFT`
- decode `p95/p99`
- tool `p95/p99`
- `llc_occupancy`
- `mbm_total_bytes`
- `memory.pressure`
- context switches、CPU migrations

### 风险

- 低并发时隔离浪费硬件资源。
- CAT/MBA 可能降低平均吞吐但不改善尾部。
- 错误配额会引入 CFS throttling。
- 已有 attention 实验显示 vLLM attention 对 L3 容量不敏感，因此该方向更可能是 tail isolation，而不是单独加速 vLLM。

## 研究点四：减少 Sandbox 生命周期频率

### 含义

Agent 常出现多个细粒度 tool calls。每次都创建 sandbox 会反复触发进程创建、文件系统挂载、依赖加载、IPC 建连和日志初始化。减少 sandbox 生命周期频率不是 chiplet 专属优化，但它是评估 chiplet 分片前必须做的强基线。

### 机制

- tool chain collapse：把多次细粒度 tool call 合并为单次 episode。
- per-domain warm pool：每个拓扑域保留预热 sandbox。
- snapshot clone：从已初始化快照恢复 sandbox。
- async environment preparation：提前准备依赖和文件系统。
- 长命令中复用单一 IPC channel。

### 为什么是强基线

如果 warm pool、snapshot 或 tool 合并已经消除了大部分 sandbox 冷启动，chiplet 分片的增量可能很小。论文实验必须证明 chiplet-aware 策略在这些强基线之后仍有收益。

### 风险

- 单 sandbox 生命周期变长后，失败影响范围变大。
- 状态清理复杂，隔离风险上升。
- tool 合并可能改变 agent 行为和调试粒度。

## 与已有论文和系统的差异

| 已有方向 | 已有重点 | 本方向差异 |
| --- | --- | --- |
| AgentCgroup | OS 资源动态控制、sched_ext、memcg | 拓扑域分片共享 OS 路径和 page-cache 亲和 |
| A CPU-Centric Perspective on Agentic AI | agent execution 的 CPU/OS 开销画像 | 将 CPU/OS 开销映射到 `CCD/NPS/socket` 放置和隔离 |
| Autellix/Cortex/ThunderAgent | workflow/stage/tool environment scheduling | 将 stage/resource pool 映射到 `CCD/NPS` |
| Kubernetes Agent Sandbox / Firecracker / Hyperlight / CubeSandbox | sandbox 生命周期和隔离机制 | 用作实验底座，不把系统资料写成 chiplet 结论 |
| OLAP WICP | worker per chiplet、独立地址空间 | sandbox 已是独立进程，热点在共享内核对象而非用户态共享堆 |

## 最小验证方案

### 阶段一：建立可重复 workload

选择一个代码仓库，构造连续 tool call：

```text
grep -> sed/edit -> pytest -> mypy -> pytest -> npm test
```

同时准备两类负载：

- 单 session 重复调用：验证 page-cache/session 亲和。
- 多 session 并发调用：验证共享 OS 路径分片。

### 阶段二：默认调度归因

记录：

- tool call p50/p95/p99
- sandbox cold/warm start latency
- minor/major page fault
- `numastat`、`/proc/<pid>/numa_maps`
- `perf c2c` HITM
- `perf lock contention`
- PSI：`cpu.pressure/memory.pressure/io.pressure`
- `resctrl llc_occupancy/mbm`

判据：若热点不在共享 OS 路径、page cache、cgroup、IPC、资源争用或 NUMA 迁移，停止该方向。

### 阶段三：人工 counterfactual

依次加入：

1. per-domain sandbox pool
2. per-domain tmp/log/workdir
3. per-domain cgroup subtree
4. per-domain IPC acceptor
5. session stickiness
6. `cpuset.cpus + cpuset.mems`

比较每一步对 p99、page fault、HITM、lock contention、memory pressure 的影响。

### 阶段四：混部隔离

在同一机器上同时运行 vLLM CPU serving 和 tool workload：

1. 默认混部。
2. cpuset 空间隔离。
3. cpuset + resctrl CAT/MBA。
4. PSI-driven admission control。

观察 vLLM `TTFT/decode p99` 和 tool p99 是否同时可控。

## 预期优化空间

若 workload 含大量重复文件访问、容器/沙箱初始化和重试循环，session/page-cache 亲和可能显著改善 warm path。chiplet 专属增量预计小于通用 sandbox warmup，合理目标是 `5%-15%` p99 改善或更稳定的 SLO。若瓶颈在外部磁盘、网络、解释器计算或 LLM 推理排队，该方向收益低。

## 判停条件

- `perf c2c/lock` 无共享对象热点。
- page-cache 亲和改善 fault 指标但 p99 不变。
- 分片导致内存压力、reclaim 或 OOM 风险上升。
- 热点转移到单一 orchestrator 或 acceptor。
- 强基线 warm pool/snapshot 已消除大部分可优化空间。
- 与 vLLM 混部时隔离降低平均吞吐但不改善 tail latency。

## 参考资料

- A CPU-Centric Perspective on Agentic AI：https://arxiv.org/abs/2511.00739
- AgentCgroup：https://arxiv.org/abs/2602.09345
- Autellix：https://arxiv.org/abs/2502.13965
- Cortex：https://arxiv.org/abs/2510.14126
- Tokencake：https://arxiv.org/abs/2510.18586
- Continuum：https://arxiv.org/abs/2511.02230
- ThunderAgent：https://arxiv.org/abs/2602.13692
- Agent.xpu：https://arxiv.org/abs/2506.24045
- CubeSandbox：https://github.com/TencentCloud/CubeSandbox
- Kubernetes Agent Sandbox：https://kubernetes.io/blog/2026/03/20/running-agents-on-kubernetes-with-agent-sandbox/
- Hyperlight：https://github.com/hyperlight-dev/hyperlight
- Firecracker snapshot：https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md?plain=1
- Linux cgroup v2：https://docs.kernel.org/admin-guide/cgroup-v2.html
- PSI：https://docs.kernel.org/6.10/accounting/psi.html
- resctrl：https://docs.kernel.org/filesystems/resctrl.html
