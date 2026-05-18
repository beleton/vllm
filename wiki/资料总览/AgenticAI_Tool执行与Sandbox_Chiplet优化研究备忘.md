# Agentic AI Tool执行与Sandbox的 Chiplet 优化研究备忘

## 结论

多 chiplet CPU 上，`Agentic AI` 的可做方向不在 attention 局部性，而在 tool execution 对共享 OS 路径和资源控制路径的压力：`sandbox/container` 生命周期、`page cache`、`dentry/inode/slab`、`cgroup` 计数与回收、`IPC` 队列、以及与 `vLLM` 并发时的 tail latency 隔离。

按本次检索，**2025-2026 年已存在一批真实论文研究 agent workload 的 CPU、OS、workflow 和 tool environment 开销，但尚未发现直接把这些开销映射到 AMD EPYC 类多 chiplet CPU 的 CCD/NPS 拓扑并做系统优化的论文**。该空白可作为本课题的研究入口。

一个前提需要固定：**sandbox 私有堆/栈/匿名页本身不是跨 chiplet 共享热点**。若进程不迁移、页按本地 first-touch 分配，其主要路径是本核到本 NUMA 内存。真正可能产生跨 chiplet 干扰的是共享内核对象、共享文件缓存、共享控制面、以及与推理任务竞争的 LLC/内存带宽。

## 2025-2026 论文真实性核查

| 项目 | 结论 | 关系 |
| --- | --- | --- |
| `A CPU-Centric Perspective on Agentic AI` / `arXiv:2511.00739` | 真实存在。当前 arXiv 页面标题更新为 `Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective` | 直接给出 agent workload 的 CPU 瓶颈：tool processing 可占总时延 `90.6%`，CPU 动态能耗可占 `44%` |
| `AgentCgroup` / `arXiv:2602.09345` | 真实存在 | 最直接。研究 sandboxed AI coding agents 的 OS 级资源动态；指出 `56-74%` 端到端时延在 tool call、container、agent init；memory 是并发瓶颈；提出 `sched_ext + memcg_bpf_ops` |
| `Autellix` / `arXiv:2502.13965` | 真实存在 | 从 program-level scheduling 降低 agent 端到端等待时间，适合作为“减少 tool/LLM 交替排队”的上层基线 |
| `Cortex` / `arXiv:2510.14126` | 真实存在 | 提出 stage isolation，把 agent workflow 拆成独立资源池，适合映射到 chiplet 级 pool sharding |
| `Tokencake` / `arXiv:2510.18586` | 真实存在 | 重点在 tool call stall 期间的 KV cache 时间利用，不直接研究 OS 路径 |
| `Continuum` / `arXiv:2511.02230` | 真实存在 | 利用 tool duration 的 TTL 保留 KV cache，重点在多轮 agent scheduling，不直接研究 OS 路径 |
| `ThunderAgent` / `arXiv:2602.13692` | 真实存在 | 明确把 tool execution environment 作为受管资源，适合映射到 chiplet-aware environment preparation |
| `Agent.xpu` / `arXiv:2506.24045` | 真实存在 | 研究异构 SoC，不是多 chiplet CPU；只可借其“分阶段、分资源池、带宽隔离”方法线 |

## 官方系统与内核资料

以下资料真实存在，但不是学术论文，不应混写为论文结论：

| 项目 | 结论 | 关系 |
| --- | --- | --- |
| `CubeSandbox` 官方仓库 | 真实存在 | 给出 agent sandbox 的 warm pool、snapshot clone、`<60ms` 冷启动、`<5MB` 基线开销与高并发密度，适合作为实验实现底座 |
| Kubernetes `Agent Sandbox` | 真实存在 | 给出 `SandboxWarmPool`、stable identity、scale-to-zero/resume、gVisor/Kata runtime 抽象 |
| `Hyperlight` 官方仓库 | 真实存在 | 无 guest OS 的 micro-VM，低启动开销，适合把 sandbox manager 嵌入 agent runtime |
| `Firecracker` snapshot 文档 | 真实存在 | 明确 snapshot restore、on-demand memory load、`cgroups v2` 对恢复时延更友好 |
| Kubernetes `Pod-Level Resource Managers` | 真实存在 | 官方给出 pod 级 NUMA 对齐与 shared pool/exclusive slice 混合模型，适合多容器 agent runtime 与 sidecar 隔离 |
| Linux `cgroup v2` / `PSI` / `resctrl` / `perf c2c` 文档 | 真实存在 | 给出 page cache/slab/workingset/pressure/LLC occupancy/HITM 的一手口径 |

## 与 chiplet 结合的主方向

### 方向一：共享 OS 路径分片

目标是减少 `VFS`、`overlayfs`、`page cache`、`dentry/inode`、`memcg`、`IPC`、日志队列等共享对象在多个 chiplet/NUMA domain 间的 cache line bounce 和锁争用。

适合映射的变量：

- 分片粒度：`socket` / `NPS domain` / `CCD-cluster`
- 每域独立 `sandbox pool`
- 每域独立 `cgroup subtree`
- 每域独立 `tmp/work/log` 目录
- 每域独立 `IPC acceptor`、`vsock/pipe/socket` 端点
- `repo/session/tool-class -> home domain` 的固定策略
- 根据工作集或排队深度，在“严格 stickiness”和“跨域偷空”之间切换

优先验证指标：

- tool call `p50/p95/p99`
- sandbox cold/warm start latency
- `perf c2c` 的 HITM/cacheline 热点位置
- `perf lock contention` 的热点锁
- `memory.stat` 中 `slab`、`slab_reclaimable`、`workingset_refault_file`、`pgfault`、`pgmajfault`
- `cpu.pressure`、`memory.pressure`、`io.pressure`
- `resctrl` 的 `llc_occupancy`、`mbm_total_bytes`、`mbm_local_bytes`

主要风险：

- 共享热点可能不在 OS 元数据，而在磁盘、网络或解释器初始化
- `overlayfs` lower 层和共享 inode 仍可能跨域共享，单独分 `upperdir` 不一定足够
- 过细分片会放大负载不均和排队尾延迟
- page cache 与 slab 被复制后，内存压力和 reclaim 可能更差

### 方向二：session/page-cache 亲和

目标是利用 agent 重试、同 repo 连续 tool calls、相同解释器与依赖树重复装载带来的复用，把 page cache、dentry、共享库文本页、`site-packages/node_modules` 读热点留在同一 NUMA/chiplet 域附近。

适合映射的变量：

- session stickiness 开关与 TTL
- warm sandbox reuse 次数
- snapshot clone 与 persistent sandbox 的切换阈值
- `cpuset.cpus + cpuset.mems`
- `numactl --membind/--preferred/--interleave`
- repo checkout、venv、缓存目录的 first-touch 域
- 工具类型感知：`pytest/grep/mypy/npm test` 分开建池

优先验证指标：

- `minor/major page fault`
- `workingset_refault_file`
- `inactive_file/active_file`
- tool call 首次与重复执行的 `p95/p99`
- `/proc/<pid>/numa_maps` 与 `numastat`
- `page cache` 命中变化对应的 wall-time 改善

主要风险：

- 连续调用若访问文件集差异大，stickiness 只会增加排队
- 预热过度会把文件缓存挤成 reclaim 噪声
- sandbox 复用与 snapshot 复用会引入状态清理与隔离风险

### 方向三：tool 与 vLLM 并发时的 tail latency 隔离

目标不是加速单次 tool，而是控制 `TTFT`、decode `p99`、tool `p99` 在混部时的抖动。

适合映射的变量：

- `vLLM worker` 与 sandbox pool 的 chiplet/NUMA 划分
- `resctrl CAT/MBA` 配额
- `cpu.max`、`cpuset`、`memory.high`
- 并发 admission control：按 `PSI` 或 `memory.events` 动态降载
- pod/shared-pool 与 exclusive-slice 的组合

优先验证指标：

- `TTFT`、decode `p95/p99`
- tool `p95/p99`
- `memory.pressure full/some`
- `io.pressure`
- `llc_occupancy`
- `mbm_total_bytes`
- `context-switches`、`cpu-migrations`

主要风险：

- 单 agent 或低并发下会浪费核和 LLC ways
- `CAT/MBA` 可能只改变均值，不改善尾部
- 错误的配额会让 CFS throttling 成为新瓶颈

### 方向四：减少 sandbox 生命周期频率

该方向不直接改变芯片拓扑，但会显著降低共享 OS 路径的触发频率，是 chiplet 优化前的强基线。

可做变量：

- tool chain collapse：把多次细粒度 tool call 合并为单次 sandbox episode
- per-domain warm pool size
- snapshot clone vs fresh start
- async environment preparation
- 长命令合并后的 `IPC` 模式：`pipe` / `uds` / `vsock`

优先指标：

- 每任务 sandbox create 数
- 每任务 container init 数
- 单任务 `p99`
- `disk writes`、`file_writeback`
- IPC round-trip latency

主要风险：

- 单 sandbox 生命周期变长后，失败影响范围变大
- 状态残留与权限边界更难清理
- 该方向可能掩盖而非解决 chiplet 共享路径热点

## chiplet 侧可控变量总表

| 类别 | 变量 |
| --- | --- |
| 拓扑绑定 | `socket` / `NPS` / `CCD-cluster` 粒度；`cpuset.cpus`；helper thread 是否同域 |
| 内存绑定 | `cpuset.mems`；`membind` / `preferred` / `interleave`；snapshot/预热页 first-touch 域 |
| 资源池 | 每域 warm pool 大小；每域 sandbox manager；每域 tool-class pool |
| 共享路径 | 每域 `cgroup subtree`；`tmp/log/workdir`；`IPC acceptor`；是否分离 `overlayfs upper/work` |
| QoS | `CAT` way 掩码；`MBA`；`cpu.max`；`memory.high`；按 PSI 触发的 admission 控制 |
| 调度策略 | session stickiness；重试保域；跨域偷空阈值；工作集大时转 pooled mode |

## 验证口径

建议把验证拆成四层：

1. `端到端`
   - 单 task、单 agent、多 agent 的 `p50/p95/p99`
   - cold start、warm start、resume latency

2. `OS 共享路径`
   - `perf c2c`
   - `perf lock contention`
   - `tracefs` 上的 `vfs`、`writeback`、`sched`、`cgroup` 相关 tracepoint

3. `内存与文件缓存`
   - `memory.stat`
   - `cpu.pressure/memory.pressure/io.pressure`
   - `numastat`
   - `/proc/<pid>/numa_maps`

4. `芯片资源`
   - `resctrl llc_occupancy`
   - `mbm_total_bytes`
   - `mbm_local_bytes`
   - `perf stat` 的 `page-faults`、`context-switches`、`migrations`
   - AMD 平台可补 `IBS` 或 `AMDuProf` 做 kernel call-path 与 load latency 归因

## 研究判据

以下情况成立时，值得继续投入：

- `perf c2c` 或 `perf lock` 明确出现共享内核对象热点
- page cache / slab / workingset 指标改善能转化为 `p99` 改善
- `tool + vLLM` 混部时存在稳定可重复的 tail latency 干扰
- 分片收益大于排队与复制代价

以下情况成立时，应停止该方向：

- 热点主要在私有用户态计算或外部磁盘/网络
- stickiness 改善局部指标但 `p99` 不降
- 内存复制导致 `PSI`、reclaim 或 OOM 风险上升
- 共享对象分片后，热点只转移到 orchestrator 主线程或单一 acceptor

## 来源 URL

- https://arxiv.org/abs/2511.00739
- https://arxiv.org/abs/2602.09345
- https://arxiv.org/abs/2502.13965
- https://arxiv.org/abs/2510.14126
- https://arxiv.org/abs/2510.18586
- https://arxiv.org/abs/2511.02230
- https://arxiv.org/abs/2602.13692
- https://arxiv.org/abs/2506.24045
- https://github.com/TencentCloud/CubeSandbox
- https://kubernetes.io/blog/2026/03/20/running-agents-on-kubernetes-with-agent-sandbox/
- https://github.com/hyperlight-dev/hyperlight
- https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md?plain=1
- https://kubernetes.io/blog/2026/05/01/kubernetes-v1-36-feature-pod-level-resource-managers-alpha/
- https://docs.kernel.org/admin-guide/cgroup-v2.html
- https://docs.kernel.org/6.10/accounting/psi.html
- https://docs.kernel.org/filesystems/resctrl.html
- https://man7.org/linux/man-pages/man1/perf-c2c.1.html
- https://man7.org/linux/man-pages/man1/perf-lock.1.html
- https://www.vldb.org/pvldb/vol17/p3428-fogli.pdf
- https://www.doc.ic.ac.uk/~af6618/publication/Optimizing_Sorting_for_Chiplet-Based_CPUs/
- https://saksham.web.illinois.edu/assets/pdf/tina.pdf
