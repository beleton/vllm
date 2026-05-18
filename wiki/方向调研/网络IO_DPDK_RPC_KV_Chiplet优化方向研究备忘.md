# 网络 I/O、DPDK、RPC、KV Serving 在 Chiplet CPU 上的优化方向研究备忘

## 结论

可优先做的软件原型有六类：

1. **queue/worker/mempool 分区基线**：按 chiplet 或近似拓扑域固定 `RX/TX queue -> polling core -> worker group`，避免 queue 共享与跨域抢占。
2. **active-buffer-aware overflow tiering**：仅保留为 Intel SPR / TiNA 类硬件条件下的低优先级备选。AMD Zen 上远端 `CCD` L3/cache 路径接近本地 DRAM，TiNA 的 Remote-tier 容量套利不成立。
3. **RPC/KV size-aware queueing**：把小请求和大请求分到不同 queue set 与 core set，先压 head-of-line blocking，再叠加 chiplet 亲和。
4. **tail-latency 驱动的弹性 spill/steal**：保留每个 chiplet 的 latency core，只有在本地 p99、queue depth 或 in-flight bytes 越界时才向远端 spill。
5. **KV 热元数据复制与冷值分片**：把小而热的索引、slab class metadata、session metadata 做多副本本地化，把大 value 或冷对象分片。
6. **chiplet-aware BDP/credit 控制**：按本地路径和远端路径分别维护 credit/window，避免统一窗口把慢路径尾部放大。

不建议把 TiNA 原论文方案直接当成纯软件原型目标。TiNA 的关键 placement 决策发生在 NIC receive path，论文明确指出软件在 packet 完成 DMA 后才看见精确信息，已经错过 placement 时机。未修改 RSS 只能做静态 per-flow 分配，不能按 active mbuf size 动态切换。

## 事实前提

- **TiNA 的核心矛盾**：SNC 提供更低的本地 core-to-LLC/DRAM 延迟，但会缩小单处理核可直接利用的 DCA/DDIO 容量；当 active mbuf size 超过本地 DCA 容量后，DMA leak 和 DMA bloat 会推高 tail latency。TiNA 对 KVS 的 p99 改善最大，原因是 KVS 还有较大的非 I/O working set。该结论依赖 Intel SPR 上远端 LLC 仍明显快于 DRAM。AMD Zen 上远端 `CCD` L3/cache 路径接近本地 DRAM，因此 Remote-tier 策略缺少稳定净收益。
- **DPDK 官方约束**：receive queue 应由单个逻辑核轮询；queue 不应被多核共享。`rte_eth_rx_queue_setup()` 可为队列指定 `socket_id` 和 `mb_pool`；`rte_mempool` 提供 per-lcore local cache。`rte_flow` 支持把流直接导向指定 queue，`RSS` action 支持把流分发到给定 queue set。
- **Server Chiplet Networking 的含义**：chiplet server 的网络路径会出现更长数据路径、异构带宽域和不一致 BDP（bandwidth-delay product，带宽时延积）。这意味着 transport、queueing 和 congestion control 不能再假设 socket 内路径近似均匀。
- **RPC/KV tail latency 的已知机制**：Minos 证明小请求和大请求混跑会产生 head-of-line blocking；Arachne、Shenango、Caladan 证明 core-aware 调度和细粒度 core 重新分配可以显著改善 tail latency 或 SLO 吞吐。

## 可做软件原型

| 研究点 | 软件原型 | 硬件依赖 | 主要风险 |
| --- | --- | --- | --- |
| `P1` 拓扑感知 queue/worker 基线 | 每个 chiplet 或近似拓扑域固定一组 `RX/TX queue + polling core + worker`；RSS/`rte_flow` 只把流导入本组；每组独立 mempool、独立统计 | commodity NIC 需要足够多 hardware queues；DPDK；CPU 绑核 | 当前平台未必暴露精确 per-chiplet 内存分配，只能先做到 per-NUMA/per-group；流量偏斜会造成局部过载 |
| `P2` active-buffer-aware overflow tiering | Zen 上不作为主方向。仅在 Intel SPR / TiNA 类远端 LLC 明显快于 DRAM的平台上，为每组配置 `local queue set` 与 `overflow queue set` | 需要 RSS/`rte_flow`/RETA 可改写到 queue set；若想复现 TiNA 语义，还需 DDIO/DCA、SNC 和 NIC 侧 queue 选择 | Zen 上远端 cache 不比 DRAM 快；软件看到的信息滞后于 NIC DMA；切换过快会抖动 |
| `P3` RPC/KV size-aware queueing | 参照 Minos，把 `small RPC/get`、`large RPC/scan/set`、后台 compaction/rehash 分到不同 queue set 和 core set；小请求优先保本地，长请求允许跨域 spill | multi-queue NIC；请求大小或成本可在 ingress 早期判断；应用层可分类 | 需要可预测的 cost class；分类错误会伤吞吐；多类队列会增大调度与统计复杂度 |
| `P4` tail-latency 驱动的弹性 spill/steal | 每个 chiplet 预留 latency core；正常只处理本地队列，超过阈值后把新流导向远端组，或允许远端偷取长请求；阈值可用 `p99`、`queue depth`、`service time EWMA`、`in-flight bytes` | 只需软件；最好有 PMU/队列统计；适合 DPDK、eRPC、用户态 KV server | 过早 spill 会损失局部性；过晚 spill 会放大 tail；需要 hysteresis，避免来回迁移 |
| `P5` KV 热元数据复制与冷值分片 | 每个 chiplet 本地保留哈希目录、slab class metadata、连接状态、热点 key 索引副本；大 value、冷 value 或日志仍做分片；远端访问尽量只落在 value path | 主要是内存容量；无需特殊 NIC；适合 KV server 原型 | 内存放大；副本一致性和回收策略复杂；写多场景收益可能被同步成本抵消 |
| `P6` chiplet-aware BDP/credit 控制 | 对跨 chiplet 的 software pipeline、RPC hop 或 service chain 维护独立 credit/window；把本地路径和远端路径分开定速，避免统一窗口把慢路径尾部放大 | 适合有明确跨域 hop 的用户态网络栈或多阶段 RPC；需要较细队列遥测 | 若平台上跨域带宽差异不大，收益可能有限；控制面参数多，验证周期长 |

## 优先级

建议按 `P1 -> P3 -> P4 -> P5 -> P6` 推进。`P2` 在 Zen 上降为排除项或平台特定附录。

- `P1` 是所有后续实验的基线。
- `P3` 直接针对 RPC/KV tail latency，且完全软件可做。
- `P4` 可以在 `P1/P3` 之上增量实现。
- `P2` 最接近 TiNA，但 Zen 上 Remote-tier 收益模型不成立，不作为 AMD EPYC 主线。
- `P5/P6` 更偏系统化改造，工程量较大。

## 硬件依赖归纳

### 强依赖硬件语义

- **TiNA 式精确 tiering**：需要 `SNC + DDIO/DCA + 多 hardware queue NIC + NIC 侧 queue 选择`。TiNA 原型还用了 FPGA bump-in-the-wire。
- **按 packet 动态选 tier**：需要 placement 发生在 NIC receive path；纯软件在 DMA 完成后再决策已经过晚。

### 只依赖 commodity NIC/软件栈

- **queue/core/request-class 分区**
- **RSS/RETA/rte_flow 导流**
- **per-queue 或 per-group mempool**
- **tail-latency controller**
- **KV 热元数据复制**

### 当前 AMD chiplet 平台的边界

- 现有 TiNA 证据来自 Intel Sapphire Rapids。即使 AMD 平台存在与 `DDIO/DCA` 等价的 `NIC -> LLC` 放置语义，也不能默认远端 `CCD` cache 比本地 DRAM 更快。
- 若平台只能暴露 socket 或 NPS 级内存域，而不能暴露更细粒度 chiplet-local 内存域，`P1/P2` 先按 **per-NUMA/per-topology-group** 实现，再看是否需要更细粒度页放置或额外硬件支持。

## 不建议优先投入

1. **直接复现 TiNA 的 per-packet placement**：没有可编程 NIC 或等价硬件时，软件原型无法等价复现。
2. **把未修改 RSS 当成 TiNA 替代品**：RSS 只能做静态流分配，不能根据 active mbuf size 动态切换 local/remote tier。
3. **先做全局 work stealing**：如果没有 request class 和 spill 阈值，跨 chiplet 抢占很容易把小请求 tail latency 拉高。

## 来源 URL

1. TiNA 论文 PDF：<https://saksham.web.illinois.edu/assets/pdf/tina.pdf>
2. TiNA 主页：<https://saksham.web.illinois.edu/publications/tina>
3. Server Chiplet Networking PDF：<https://pages.cs.wisc.edu/~mgliu/papers/ChipletNet-hotnets25.pdf>
4. Server Chiplet Networking 项目页：<https://www.bingyangwei.com/publication/server-chiplet-networking/>
5. DPDK Poll Mode Driver 文档：<https://doc.dpdk.org/guides-19.02/prog_guide/poll_mode_drv.html>
6. DPDK `rte_ethdev` API：<https://doc.dpdk.org/api/rte__ethdev_8h.html>
7. DPDK `rte_mempool` API：<https://doc.dpdk.org/api/rte__mempool_8h.html>
8. DPDK flow offload / `QUEUE` / `RSS` action：<https://doc.dpdk.org/guides/prog_guide/ethdev/flow_offload.html>
9. Intel DDIO 技术说明：<https://www.intel.com/content/www/us/en/developer/articles/technical/ddio-analysis-performance-monitoring.html>
10. Minos 论文：<https://www.usenix.org/conference/nsdi19/presentation/didona>
11. Arachne 论文：<https://www.usenix.org/conference/osdi18/presentation/qin>
12. Shenango 论文：<https://www.usenix.org/conference/nsdi19/presentation/ousterhout>
13. Caladan 论文：<https://www.usenix.org/conference/osdi20/presentation/fried>
14. eRPC 论文：<https://www.usenix.org/conference/nsdi19/presentation/kalia>
