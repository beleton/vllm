# Chiplet 与硬件背景

## 核心概念

Chiplet CPU 的关键不是封装形式，而是局部性：本地 L3 和本地内存访问更快，跨 CCD/CCX 或跨 socket 访问更慢，本质上是更细粒度的 NUMA 问题。

相比单片架构，Chiplet 牺牲了一部分统一访问特性，换来更好的可扩展性和良率。性能通常在两件事之间权衡：一是把线程集中到更少 chiplet 以获得更低通信延迟，二是把线程分散到更多 chiplet 以获得更大的聚合 L3 容量。

## AMD 术语

- **CCD**：计算模块，包含核心和 L3，不包含内存控制器或 PCIe 控制器。
- **CCX**：CCD 内部的逻辑单元。Zen 2 通常 1 CCD = 2 CCX，Zen 3/4/5 通常 1 CCD = 1 CCX。
- **IOD**：I/O Die，包含内存控制器、PCIe 和路由逻辑。
- **Infinity Fabric**：连接 CCD 和 IOD 的高速互联，跨 chiplet 访问要经过它。

## 缓存层级

- L1/L2 是核心私有资源；线程迁移会带来冷缓存。
- L3 主要按 CCD/CCX 共享；访问本地 L3 更快，跨 chiplet L3 更慢。
- Zen 5 的 L3 是 non-inclusive victim cache，不要求稳定包含 L1/L2 中的同份数据。
- 从其他 CCX 的 L3 或内存取回的数据，先回填到本地 L1/L2，不会直接驻留在本地 L3。
- L3PMCx04 等按 CCX 层级统计时，命中其他 CCX 的 L3 也可能被记作 L3 miss。

## 本机事实

- 平台：`2 x AMD EPYC 9745 128-Core Processor`，family `0x1a`
- 每 16 个物理核心共享约 32 MiB 本地 L3，每个 CPU 8 个 CCD
- 系统信息可用 `AMDuProfCLI info --system` 查看

## 本地延迟量级

| 路径 | 大致量级 |
| --- | --- |
| L1 | ~1.7 ns |
| L2 | ~8.7 ns |
| 本地 L3 | ~26.2 ns |
| 本地内存 | ~120.2 ns |
| 跨 CCD 远端 cache 路径 | ~110–150 ns |

跨 CCD 访问代价已接近本地内存（120 ns），因此同一 socket 不是低延迟共享域，低延迟共享范围更接近本地 L3/CCD。

## AMD / Intel Chiplet 与 AMX/L3 的关系

- chiplet/MCM、tile、SNC domain、socket 不是同一层概念。
- Intel AMX 作用在矩阵乘法执行层；L3 感知放置作用在缓存/拓扑/内存层。两者是不同层优化，在 attention/KV cache 场景里拓扑与缓存局部性问题相对更独立。
- 当前 AMD EPYC 9745 不支持 AMX，但不影响 L3 感知放置作为独立研究方向。

## 对 LLM 推理的启发

- Decode 阶段通常更容易受内存带宽、L3 局部性和线程放置影响。
- 只按 NUMA 均匀切分不一定最优，下层还有 CCD/CCX/L3 竞争。
- 小工作集更适合集中放置；大工作集或高 miss 场景更可能受益于分散放置。
