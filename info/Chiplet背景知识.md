
# 介绍
## Chiplet架构的CPU
传统的 CPU 采用 Monolithic（单片）设计，即所有核心、缓存、内存控制器都在一块巨大的硅片上。而 Chiplet 设计则是将不同功能的硅片（Die）像搭积木一样封装在一起。

单体/Monolithic架构的CPU，L3 Cache在物理上也是分块的，但核心访问隔壁核心的L3 Slice只比访问自己的慢一点点（通常几ns的延迟，被称为UCA），但Chiplet架构的访问远端L3的时间要远大于本地L3的时间，本质上是NUMA系统。

同规模下的纯物理性能，Chiplet不如单片架构，但单片架构无法制作更大的芯片（可扩展性不足），且良率低（一个地方出错则整个芯片报废）

**AMD**
- **CCD (Core Complex Die):** 计算模块。它里面包含了 CPU 核心（Cores）和 L3 缓存。它**不包含**内存控制器或 PCIe 控制器。
- **CCX (Core Complex):** 这是 CCD 内部的逻辑单元。
    - **Zen 2:** 1 个 CCD 包含 2 个 CCX。每个 CCX 有自己的 L3。
    - **Zen 3/4:** 1 个 CCD 包含 1 个 CCX。意味着 CCD 内的所有核心共享同一块统一的 L3 缓存。
- **IOD (I/O Die):** 中央枢纽，包含了内存控制器（IMC）、PCIe 控制器和与所有 CCD 通信的路由。
- **Infinity Fabric (IF):** 连接 CCD 和 IOD 的高速互联总线
跨chiplet的L3访问需要经过Infinity Fabric以及IO Die

>**AMD EPYC** 处理器演进史：
	- **第 1 代 (Naples)**：Zen 1，32核。
	- **第 2 代 (Rome)**：Zen 2，64核。
	- **第 3 代 (Milan)**：Zen 3，64核 
	- **第 4 代 (Genoa/Bergamo)**：Zen 4/4c，96/128核。
	- **第 5 代 (Turin)**：Zen 5/5c，128/192核 

| 资源层级            | 物理位置    | 独占/共享属性                   | OS/调度视角                                                                                       |
| :-------------- | :------ | :------------------------ | :-------------------------------------------------------------------------------------------- |
| **L1/L2 Cache** | Core 内部 | **Core 独占**               | 线程切换会导致冷缓存 (Cache Trashing)。                                                                  |
| **L3 Cache**    | CCD 内部  | **CCD 内共享** (8核或16核共享)    | **关键点**：Core 0 访问同 CCD 内 Core 7 的数据极快（走内部 L3）；但访问另一个 CCD 的数据极慢（必须出 CCD -> 进 IOD -> 进另一个 CCD）。 |
| **内存 (DRAM)**   | 连接在 IOD | **全局共享** (UMA 物理，NUMA 逻辑) | 虽然内存控制器在中间的 IOD，但物理距离和拥塞导致 OS 仍需将其视为 NUMA。                                                    |
| **I/O (PCIe)**  | IOD     | **全局共享**                  | 所有核心访问网卡/磁盘的物理距离基本一致。                                                                         |

Zen 5架构的L3为受害者L3, L3只存储从L2中被（淘汰）的数据,通常情况下，同一份数据不会同时驻留在 L2 和 L3 中。
与之相对应的是包含式缓存：规定高级别缓存（如 L3）必须包含低级别缓存（如 L2 和 L1）中的所有数据。如果数据存在于 L1 中，它也必然备份在 L2 和 L3 中。

L3PMCx04属于L3缓存性能监控计数器，按照CCX的层级定义。命中其他CCX的L3也算作L3 Miss。

- CCX内部通信：当两个线程运行在同一个 CCX 的不同核心上时，它们的数据交换主要通过共享的 L3 缓存完成。延迟通常为几十个时钟周期。
- 跨 CCX 通信：必须经过 IOD 上的 DF 链路。

# 本机配置
## 系统信息
 2个AMD EPYC 9745 128-Core Processor处理器，16个物理核心共享1个32MB的本地L3，每个CPU8个CCD
`AMDuProfCLI info --system` 查看系统信息




# 相关研究
## Sandwich
### 问题
CPU进行LLM推理效率低下，同时体现在Prefill阶段和Decdoe阶段：
- 现有主流框架的CPU后端在推理过程中采用固定的进程绑核策略，通常按照NUMA节点均匀划分，但没有考虑到NUMA架构更下层的资源竞争，如对L3 Cache，内存控制器的竞争
- 像 TVM 这样的自动调优编译器，搜索空间巨大，调优一个模型可能需要数天，且难以处理 LLM 中动态变化的输入形状（Dynamic Shapes）

###  解决方案
- 对于Decode阶段，论文发现Decode阶段是内存带宽密集型，减少活跃核心数量、依据物理拓扑选择特定核心，反而能减少对共享资源（如内存带宽、LLC Tags）的争抢，从而提高 Token 生成速度。
- 文章提出了一种树状结构（TopoTree）来抽象 CPU 的硬件拓扑（从 Socket 到 NUMA 再到具体的共享 Cache 结构），并在树上进行变换搜索，以找到最佳的核心绑定配置。


## ARCAS: Adaptive Runtime System for Chiplet-Aware  Scheduling
### 问题
Chiplet架构的CPU由多个小硅片通过互联技术拼接起来，L3 Cache被物理分割，核心访问本Chiplet的L3最快(25ns)，访问同一 Socket 但不同 Chiplet 的 L3 较慢(80-90ns)，跨Socket的访问最慢(150ns)。

传统的操作系统调度器和现有的NUMA感知运行时通常只看到 Socket 级别，导致缓存利用率低下。

CPU的物理核心数量增多，但内存通道却只有8-12个，单核可用的平均内存带宽急剧下降

Socket内部线程的放置策略：
1. **LocalCache（集中策略）**：把 8 个线程都挤在**同一个 Chiplet** 上。
    - 优势：线程间通信极快（25ns），且大家共享本地 L3。
    - 劣势：只能用这一个 Chiplet 的 32MB L3。
2. **DistributedCache（分散策略）**：把 8 个线程分散到 **8 个不同的 Chiplet** 上。
    - 优势：可以使用 8 个 Chiplet 的总和 L3（32MB * 8 = 256MB）。
    - 劣势：线程间通信慢（80ns+）。
数据量小于单个L3 Cache的大小时，线程共享同一个L3 Cache，访问速度更快。数据量大于单个L3 Cache的大小时，分散到不同Chiplet上的时候速度会更快。
（数据独立无依赖的情况，是否分散永远更好？）

### 设计
1. 系统利用硬件计数器（Hardware Counters）监控 `RMT_CHIP_ACCESS_RATE`（远程 Chiplet 访问率），如果远程访问率过高，系统会增加 spread_rate（分散率），让任务利用更多的 Chiplet，意图是利用更大的聚合 L3 Cache 来减少对主存的访问。当远程访问率低时，ARCAS 会收缩（Compact）任务到更少的 Chiplet 上，以享受低延迟
2. 实现了轻量级协程的作为任务调度的单位
