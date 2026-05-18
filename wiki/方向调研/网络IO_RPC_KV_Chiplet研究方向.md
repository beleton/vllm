# 网络 I/O、RPC 与 KV Serving 的 Chiplet 研究方向

## 结论

网络 I/O 方向可做，但不建议在 AMD Zen 平台上复现 `TiNA` 的 remote-tier / per-packet tiering。原因有两层：第一，TiNA 的 placement 决策发生在 NIC receive path，纯软件在 packet DMA 完成后才看到精确信息，已经错过等价 placement 时机；第二，Zen 上远端 `CCD` L3/cache 路径与本地 DRAM 延迟接近，TiNA 在 Intel SPR 上依赖的“远端 LLC hit 明显快于 DRAM”收益模型不成立。现阶段更现实的软件方向是：

- `queue/worker/mempool` 拓扑分区
- RPC/KV size-aware queueing
- tail-latency 驱动的 bounded spill/steal
- KV 热元数据复制与冷值分片

这些方向只依赖 commodity NIC 多队列、RSS/`rte_flow`、DPDK 绑核、per-queue/per-group 统计和用户态调度，适合作为系统型备选。

## 背景

网络应用在 Zen 上的 chiplet 敏感点与普通计算不同：

- 不应假设远端 `CCD` L3 是比 DRAM 更优的容量池。
- DPDK 使用多 RX/TX queue、polling core、mempool 和 per-lcore cache。
- RPC/KV serving 的 tail latency 常受 head-of-line blocking、小请求/大请求混跑、queue depth 和 core allocation 影响。
- 多 chiplet server 中，本地路径和远端路径的延迟、带宽、队列和内存页归属可能不同。

## 与 TiNA 的关系

TiNA 证明了 Intel SPR 上本地低延迟和跨 chiplet 聚合 DCA 容量之间存在条件化取舍。active mbuf size 小时，本地 tier 低延迟；active mbuf size 超过本地 DCA ways 后，使用远端 tier 可避免 DMA leak/bloat。

该取舍不能作为 Zen 研究动机。Zen 上跨 `CCD` L3/cache 路径接近本地 DRAM，远端 cache hit 替代 DRAM access 的延迟差不足，Remote-tier 容量套利缺少明确收益。

不可直接迁移点：

- TiNA 依赖 `SNC + DDIO/DCA + 多 hardware queue NIC + NIC receive path 上的 queue 选择`。
- TiNA 原型使用 FPGA bump-in-the-wire。
- 未修改 RSS 只能做静态 per-flow 分配，不能按 active mbuf size 动态切换 local/remote tier。
- AMD Zen 平台即使存在可控 NIC-to-cache 路径，也不能默认远端 `CCD` cache 比本地 DRAM 更快。

## 研究点一：Topology-Aware Queue/Worker/Mempool

### 思路

将 `RX/TX queue -> polling core -> worker group -> mempool` 固定到同一拓扑域，避免 queue 共享、core migration、mempool 远端页和跨域抢占。

```text
flow hash / rte_flow
  -> queue set i
  -> polling core group i
  -> worker pool i
  -> mempool i
```

### 设计变量

- queue set 数量
- polling core 所在 `CCD/NPS/socket`
- mempool 的 `socket_id`
- per-lcore cache 大小
- RSS RETA 或 `rte_flow` 规则
- flow 到 queue 的映射方式

### 实验方案

- baseline：默认 DPDK queue 分配。
- 方案：按 topology group 分配 queue、polling core、mempool。
- 指标：
  - packet/RPC p99
  - queue depth
  - drops
  - cache miss
  - `resctrl llc_occupancy`
  - `mbm_total/local`

## 研究点二：RPC/KV Size-Aware Queueing

### 思路

将小请求和大请求分到不同 queue set 和 core set。小请求保本地低延迟；大请求、scan、large value、background task 允许跨域 spill。

### 与 chiplet 的结合

size-aware queueing 本身不是 chiplet 创新，但与 chiplet 拓扑结合后，可让 latency-sensitive 小请求固定在本地 queue/core/mempool，避免被跨域大请求拖慢。该方向不依赖远端 L3 容量收益。

### 设计变量

- 请求分类：small get、large get、scan、set、background compaction/rehash
- 每类 core 配额
- 每类 queue set
- spill 阈值
- long request 是否优先远端执行

### 风险

- 请求成本早期不可预测。
- 分类错误会降低吞吐。
- 多队列会增加统计和调度复杂度。

## 研究点三：Tail-Latency Spill/Steal

### 思路

正常情况下每个 topology group 只处理本地队列；当本地 p99、queue depth 或 in-flight bytes 超阈值时，才把新请求导向远端组，或允许远端偷取长请求。spill 是 SLO 保护和负载均衡手段，不是把数据放入远端 L3 的局部性优化。

### 控制信号

- queue depth
- in-flight bytes
- service time EWMA
- p99 violation
- mempool in-use count
- `PSI`
- `resctrl` bandwidth/occupancy

### 关键设计

必须有 hysteresis，避免频繁来回迁移。spill 只应处理新请求或长请求，不应迁移正在执行的小请求。

## 研究点四：KV 热元数据复制与冷值分片

### 思路

KV serving 中哈希目录、slab metadata、connection/session metadata、小热点 key 索引较小且访问频繁，可复制到各 `CCD`。大 value、冷对象、日志和持久化路径继续分片。

### 适用条件

- read-heavy
- 热 key 稳定
- metadata 远小于 value
- 写入比例低或可批量同步

### 风险

- 写多时副本同步成本高。
- metadata 复制增大内存占用。
- 一致性和回收复杂。

## 优先级

1. 拓扑感知 queue/worker/mempool 分区。
2. RPC/KV size-aware queueing。
3. tail-latency bounded spill/steal。
4. KV 热元数据复制。
5. active-buffer-aware overflow tiering（仅作为 Intel-SPR/TiNA 类硬件条件下的低优先级方向）。

TiNA 式 per-packet tiering 在 Zen 上默认排除，除非实测证明目标 AMD 平台存在“远端 cache 明显快于本地 DRAM”的 NIC-to-cache 路径。

## 预期优化空间

该方向更可能改善 p99，而不是显著提高平均吞吐。纯软件 queue/core/mempool 分区若能减少 queue 共享、cache pollution 和 head-of-line blocking，合理目标是 `5%-20%` p99 改善。若平台瓶颈在 NIC、PCIe、外部网络或应用逻辑，chiplet 增量会很小。

## 判停条件

- 默认 DPDK 已满足 queue 单核轮询且无跨域抢占。
- p99 主要由外部网络或应用计算决定。
- 拓扑分区造成负载不均，tail latency 上升。
- `resctrl/PCM/perf` 看不到缓存、带宽或 queue-level 改善。

## 参考资料

- Zen 平台 TiNA 适用性复核：[Zen平台_TiNA适用性复核与新增场景.md](./Zen平台_TiNA适用性复核与新增场景.md)
- TiNA：https://saksham.web.illinois.edu/assets/pdf/tina.pdf
- TiNA 项目页：https://saksham.web.illinois.edu/publications/tina
- Server Chiplet Networking：https://pages.cs.wisc.edu/~mgliu/papers/ChipletNet-hotnets25.pdf
- DPDK Poll Mode Driver：https://doc.dpdk.org/guides-19.02/prog_guide/poll_mode_drv.html
- DPDK flow offload：https://doc.dpdk.org/guides/prog_guide/ethdev/flow_offload.html
- Minos：https://www.usenix.org/conference/nsdi19/presentation/didona
- Arachne：https://www.usenix.org/conference/osdi18/presentation/qin
- Shenango：https://www.usenix.org/conference/nsdi19/presentation/ousterhout
- Caladan：https://www.usenix.org/conference/osdi20/presentation/fried
- eRPC：https://www.usenix.org/conference/nsdi19/presentation/kalia
