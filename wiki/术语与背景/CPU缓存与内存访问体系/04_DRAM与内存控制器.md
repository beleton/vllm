# 04 — DRAM 与内存控制器

## 4.1 DRAM 单元与阵列结构

### 1T1C 单元

DRAM 的基本存储单元是 **1T1C**（1 晶体管 + 1 电容）。电容充电代表 1，放电代表 0。电容会随时间泄漏电荷，因此需要**定期刷新**（refresh）。

### Bank 结构

DRAM 芯片内部组织为分层的阵列结构：

```
Channel → DIMM → Rank → Bank Group → Bank → Row (page) → Column
```

| 层级 | 数量示例（DDR5-4800） | 说明 |
|------|----------------------|------|
| Channel | 8-12 / socket | 独立 64-bit 数据总线，可完全并行操作 |
| DIMM | 1-2 / channel | 物理内存条 |
| Rank | 1-2 / DIMM | 共享数据总线但独立 chip select 的信号组 |
| Bank Group | 4-8 (DDR5) | Bank 组，同组内不能同时做不同操作 |
| Bank | 4 / bank group (DDR5 16Gb) | 独立行列解码，独立的 sense amplifier |
| Row | 2^16 = 65536 | 一行，也称 page（例如 8 KiB） |
| Column | 64-128 / row | 一行中的列（一次 burst 传输 64 B） |

### 层次之间的并行性

不同 **bank** 可以同时执行不同操作（activate to row X in bank A, read from row Y in bank B），是实现**内存级并行**（MLP）的硬件基础。但同一 bank 同时只能访问一行（即一次只能有一个 open page）。

Bank group 引入额外约束：同一组内的 bank 共享部分 datapath 逻辑，因此不能同时执行列命令（read/write），但可以同时进行不同 bank 的行激活（activate）。DDR5 强化了 bank group 以提高带宽。

### Sense Amplifier（行缓冲区）

每个 bank 有一组 sense amplifier（即行缓冲），大小等于一行。读取/写入操作的作用于此缓冲区：

1. **ACTIVATE**（激活）命令：将目标行从存储电容阵列加载到 sense amplifier（破坏性读出——原存储单元的电荷被耗尽，需写回）
2. **READ/WRITE**（列命令）：从 sense amplifier 读出指定列/将数据写入 sense amplifier
3. **PRECHARGE**（预充电）：将 sense amplifier 的内容写回存储电容，关闭当前行，准备 bank 接受新的 ACTIVATE

**Open Page Policy**：行激活后保持打开，后续访问同一行的不同列只需列命令（低延迟）。代价是激活了新行时必须先 precharge。

**Closed Page Policy**：每次访问后立即 precharge 关闭行。适合随机访问模式，避免 precharge 延迟发生在关键路径。

内存控制器根据访问模式动态选择策略。

## 4.2 DRAM 时序参数

所有参数单位是 **tCK**（DRAM 时钟周期）或 ns。

| 参数 | 含义 | DDR5-4800 典型值 |
|------|------|------------------|
| tRCD | RAS-to-CAS Delay：从 ACTIVATE 到 READ/WRITE 的最小间隔 | ~14 ns (28 tCK) |
| tCL | CAS Latency：从 READ 到数据出现在总线的最小间隔 | ~16 ns (32 tCK) |
| tRAS | Row Active Time：ACTIVATE 到 PRECHARGE 之间的最小时间 | ~32 ns (64 tCK) |
| tRP | Row Precharge Time：PRECHARGE 完成的最小时间 | ~14 ns (28 tCK) |
| tRC | Row Cycle Time：同一 bank 连续两个 ACTIVATE 的最小间隔 | tRAS + tRP ≈ 46 ns |
| tRTP | Read-to-Precharge：READ 到 PRECHARGE 的最小间隔 | ~7.5 ns |
| tWR | Write Recovery：WRITE 到 PRECHARGE 的最小间隔 | ~15 ns |
| tFAW | Four-Activate Window：同一 rank 内四次 ACTIVATE 的最小间隔 | ~21-30 ns |
| tREFI | Refresh Interval：平均每 bank 刷新间隔 | 7.8 μs (DDR5 16Gb) |
| tRFC | Refresh Cycle Time：一次刷新所需的时间 | ~295 ns (DDR5 16Gb) |

### 延迟计算

一次简单的随机行访问（关闭行→读取）：

```
延迟 = tRP + tRCD + tCL
     ≈ 14ns + 14ns + 16ns ≈ 44 ns  (DRAM 时钟域)
     + 总线传输 + 内存控制器排队 ≈ 60-80 ns（端到端）
```

一次**行命中**（open page hit）则只需：

```
延迟 = tCL ≈ 16 ns (DRAM) + 少量传输 ≈ 40 ns 端到端
```

这就是为什么**行局部性（row locality）**对 DRAM 性能至关重要：行命中比行冲突快 2-3 倍。

### tREFI / tRFC 与刷新开销

刷新是 DRAM 无法回避的带宽税：

```
刷新带宽损失 ≈ (tRFC / tREFI) / num_banks_per_refresh_group

例如：tRFC = 295 ns, tREFI = 7800 ns / per bank
→ 约 3.8% 的 bank-时间用于刷新
```

DDR5 的 bank group 架构允许同一 bank group 内其他 bank 在刷新期间仍然被访问（per-bank refresh），约节省一半刷新开销相比 DDR4 的 all-bank refresh。

## 4.3 地址映射（Address Mapping）

内存控制器将物理地址映射到具体的 channel、rank、bank、row、column。映射策略直接影响实际延迟和带宽：

### 典型映射方案（Intel/AMD 未公开精确方案，此处为代表性描述）

```
物理地址 bits:
| high ... | ... | row | rank | bank | bank_group | column | low |
  页颜色       通道     rank    bank   bg          列偏移
```

关键权衡：

- **低位 bank 位**：让连续地址（如 64 B stride）跨越 bank，增加 bank 级并行度（MLP）
- **高位 row 位**：使连续地址在同一 row 内尽可能聚合，增加 open page hit
- **通道分散**：连续的物理页跨通道，利用多通道带宽
- **页着色（Page Coloring）**：bank/row bit 的选取如果恰好与 stride 对齐，会导致所有访问陷在同一 bank 而无法利用 bank 并行的 "热点 bank"

### XOR/哈希映射

现代内存控制器在行列映射中加入 XOR/哈希（如地址高位 XOR 后选择 channel），防止规律性地址步长（如页大小步长：4096）陷入同一 bank。AMD Zen 架构在地址到 CCX/IOD 的映射中大量使用 XOR 打破规律性。

### 异构内存与 Sub-NUMA Clustering（SNC）

AMD EPYC 支持 SNC（Sub-NUMA Clustering），将同一 socket 的物理核心和内存控制器分为多个 NUMA 节点（如 SNC4 = 每 socket 4 个 NUMA 域）：

- 优化的局部性：核心被限制到一个 NUMA 域
- 内存交错禁用，每个 NUMA 域直接挂接该域的内存控制器
- 简化了 LLC 和内存控制器的扩展性

**intel 等效特性**：SNC（Sub-NUMA Clustering，Sapphire Rapids 同取名）或 COD（Cluster-on-Die）。

## 4.4 内存控制器（IMC）调度

### 请求队列与重排序

内存控制器维护一个请求队列，包含所有未完成的 DRAM 读写需求。调度器的目标：最大化带宽利用率 + 最小化平均延迟。

**FR-FCFS（First-Ready, First-Come-First-Served）** 是经典调度策略：

1. 优先调度 "ready" 的请求（即 open page hit —— row 已激活，只需列命令）
2. 多个请求 equally "ready" 时按到达时间排序（FCFS）
3. 也有其他因子干预：写请求可以批量提交减少总线翻转；延迟敏感的读请求可以得到优先级跃升

### 读写调度

- **读优先级**：读是同步操作（CPU 在等待），直接出现在关键路径上；写可以推迟，因此读通常有调度优先级
- **写收集（Write Batching）**：延迟写，积累一批后 burst 写入。减少总线翻转（读→写→读的 turnaround 开销）
- **写 drained**：当写队列超过高线阈值，加速排空避免溢出

### 刷新调度

内存控制器将 tREFI 内的刷新需求平均分散为 **每 bank 单独刷新**（DDR5 per-bank refresh），使刷新的延迟干扰降到最低。但温度升高时刷新频率会加速（Tcase refresh management）。

## 4.5 DRAM 的可靠性机制

### ECC（Error-Correcting Code）

- 标准 DDR5 ECC：每 64-bit 数据 + 8-bit ECC，可纠正单位错、检测双位错（SEC-DED）
- DDR5 芯片内置 on-die ECC，用于内部阵列的错误纠正（与系统级 ECC 不同）
- AMD EPYC/Intel Xeon 全线支持 ECC 内存，桌面级通常不支持

### Row Hammer

Row Hammer 是由于激活同一 bank 中的物理相邻行（aggressor row）激发的干扰效应——相邻行（victim row）的电容在反复激活 aggressor 行时发生异常电荷泄漏，导致位错误。

**缓解措施**：

- 增加 tREFI 频率（对温度/工作载敏感时刷新更频繁）
- **PRAC**（Per-Row Activation Counting，DDR5）：内存控制器跟踪每行的激活次数，达到阈值时暂停激活相邻行并对其进行预防性刷新
- **TRR**（Target Row Refresh）：对潜在受害行额外刷新
- 软件层面通过加密/ASLR 减小恶意利用面

## 4.6 内存访问模式与性能调优

### 延迟 vs 带宽——Roofline 视角

DRAM 系统的关键性能剖面是两条线：

1. **延迟墙**：单次串行访问（无 MLP）时随机访问的延迟下限 ~100 ns = 10M IOPS 上限
2. **带宽墙**：所有 channel 的峰值吞吐上限（例如 12ch DDR5-4800 ≈ 460 GB/s per socket）

实现高性能的关键是**并发**：(a) 足够多的 in-flight 请求保持 bank 并行，(b) access pattern 充分利用行局部性避免重复 precharge/activate 代价。

### Streaming vs Random

- **Streaming** 顺序访问（stride=1 cache line）：硬件预取器自动覆盖 → L3/L2 均被预取提前填充，DRAM 端表现为整行连续激活，bank 并行利用良好
- **Random** 随机访问（指针追踪）：L1/L2 逐个 miss，L3 可能命中（工作集小）或 miss，DRAM 端每访问一次要 precharge+activate+read → 延迟全路径，带宽利用率极低（IOPS bound, not bandwidth bound）

### NUMA 感知

跨 Socket 访问的代价（~200-300 ns）是本地访问的 ~2-2.5 倍。NUMA 感知调度需要：

- **首触（First-Touch）**：数据页在第一次写入时被分配到写入线程的 NUMA 节点
- **交错分配（Interleaving）**：页按轮替分布在所有 NUMA 节点（mbind / numactl --interleave），可用于工作集均匀分布、线程化访问的场景
- **迁移**：`move_pages` / `migrate_pages` 将页迁移到访问最频繁的 NUMA 节点

## 4.7 关键小节

- DRAM 的分层并行结构（bank/bank group/rank/channel）决定了可实现的内存级并行（MLP）上限
- tRP + tRCD + tCL 构成一次随机行的最终延迟，加上内存控制器排队和总线传输
- 行局部性（open page hit）是获得 DRAM 最佳性能的关键——行命中比随机访问快 2-3 倍
- 内存控制器调度算法（FR-FCFS）在最大化带宽利用率和最小化延迟之间做权衡
- 地址映射的 XOR/哈希粒度决定了流式访问是否能充分利用多 bank 和多通道
- NUMA 架构中，远端访问代价接近本地访问的 2-3 倍，软件调度至关重要
