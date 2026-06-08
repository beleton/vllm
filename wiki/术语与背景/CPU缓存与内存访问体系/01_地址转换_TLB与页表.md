# 01 — 地址转换：TLB 与页表遍历

## 1.1 为什么需要地址转换

操作系统为每个进程维护独立的虚拟地址空间。虚拟地址提供：

- **隔离**：进程 A 无法访问进程 B 的内存
- **连续假象**：虚拟地址连续，物理页可以散布在任意物理帧中
- **按需分页**：不常用的页可以换出到磁盘，释放物理内存
- **权限控制**：每页独立设置 R/W/X 权限位

虚拟→物理的翻译粒度是 **4 KiB 页**（标准 x86-64 小页），此外支持 2 MiB 大页和 1 GiB 巨页。翻译由 MMU（Memory Management Unit）完成，核心数据结构是**多级页表**。

## 1.2 x86-64 页表结构（4 级 & 5 级）

### 4 级页表（48-bit 虚拟地址，4 KiB 页）

48-bit 虚拟地址被分割为：

```
| 47..39 | 38..30 | 29..21 | 20..12 | 11..0 |
|  PML4   |  PDP   |   PD   |   PT   | offset |
  9 bits   9 bits   9 bits   9 bits   12 bits
```

- **PML4E**（Page Map Level 4 Entry）：指向 PDPT 的基址
- **PDPTE**：指向 Page Directory 基址
- **PDE**：指向 Page Table 基址（若 PS=1，则直接映射 2 MiB 大页）
- **PTE**：指向 4 KiB 物理页帧基址
- **offset**：页内偏移（12 bits 对应 4 KiB）

每一级表项 8 字节，一张表恰好占一页（512 条目 × 8 字节 = 4 KiB），因此每级索引恰好 9 bits。

### 物理地址宽度

物理地址受实现限制，不是 64 bits。AMD Zen 5 支持最高 52-bit 物理地址（受 `CPUID` 查询 `MAXPHYADDR` 限制）。每个 PTE 中的物理页帧号（PFN）指向物理地址的高位。

### 5 级页表（57-bit 虚拟地址）

Intel Ice Lake 服务器和 AMD Zen 5 均支持 5 级页表。增加额外一级 PML5，形如：

```
| 56..48 | 47..39 | ... | 11..0 |
|  PML5   |  PML4  | ... | offset |
```

5 级页表在 `CR4.LA57 = 1` 时启用。对大多数应用无影响（页表深度 +1 仅当虚拟地址实际使用 57-bit 空间时才触发额外遍历）。

### 页表项（PTE）关键位

| 位 | 名称 | 含义 |
|----|------|------|
| 0 | P (Present) | 1 = 页在内存中，0 = 触发缺页异常 |
| 1 | R/W | 0 = 只读，1 = 读写 |
| 2 | U/S | 0 = 内核，1 = 用户可访问 |
| 5 | A (Accessed) | 硬件在访问时自动置 1 |
| 6 | D (Dirty) | 硬件在写入时自动置 1（仅 PTE 级别） |
| 7 | PS (Page Size) | PDE: 1 = 2 MiB 大页；PDPTE: 1 = 1 GiB 巨页 |
| 8 | G (Global) | 全局页，上下文切换时不刷出 TLB |
| 11..9 | Ignored | 软件可用（操作系统用 bit 9-11 标记换出/COW 等） |
| 63 | NX (No Execute) | 禁止执行 |

### 页表遍历成本

一次 4 级页表遍历需要 **4 次顺序内存访问**（PML4 → PDPT → PD → PT），每次都可能触发 cache miss。若不使用大页，一次 TLB miss 的遍历代价可高达 100+ ns。这是 TLB 存在的核心动机。

## 1.3 TLB 层次结构

TLB（Translation Lookaside Buffer）是页表项的专用缓存，实质上是虚拟地址 → 物理地址的**全相联或组相联查找表**。每条 TLB 条目缓存：

- 虚拟页号（VPN）→ 物理页号（PPN）
- 权限位（R/W、U/S、NX）
- 页大小（4K / 2M / 1G）
- ASID/PCID 标签（用于区分不同进程的地址空间）

### AMD Zen 5 典型 TLB 参数

| TLB 级别 | 条目数 | 相联度 | 覆盖页大小 | 备注 |
|---------|--------|--------|-----------|------|
| L1 iTLB | 64 | 全相联 | 4K/2M/1G | 每周期 1 次查找 |
| L1 dTLB | 72 | 全相联 | 4K | 每周期可处理 2 次 load + 1 次 store |
| L1 dTLB (large) | 32 | 全相联 | 2M/1G | 独立结构，与普通 dTLB 并行查找 |
| L2 TLB (STLB) | 2048 | 16 路 | 4K/2M/1G | 共享 I/D，命中的代价约 7-8 cycle |

### Intel Golden Cove 典型 TLB 参数

| TLB 级别 | 条目数 | 备注 |
|---------|--------|------|
| L1 iTLB | 96 | 含 4K + 2M/4M |
| L1 dTLB | 96 | 含 4K + 2M/4M |
| STLB | 4096 | 共享 I/D，仅 4K（大页走独立路径） |

### TLB 寻址机制

TLB 设计与其下游的 L1 缓存之间的交互至关重要。**关键命题：如何让 TLB 查找与 L1D 缓存访问并行进行？** 答案在于 VIPT（Virtually Indexed, Physically Tagged）——第 2 章详细展开。

### TLB 替换策略

现代 TLB 普遍使用**伪 LRU** 或 **基于重用的替换策略**（如 RRIP）。L1 TLB 由于条目少、要求访问极快，常使用类似 NRU（Not Recently Used）的简化策略。

## 1.4 页表遍历器（Page Table Walker）

当 L1 dTLB 和 L2 STLB 均 miss，MMU 中的硬件页表遍历器接管：

1. 从 **CR3** 寄存器加载 PML4 基址
2. 用虚拟地址的 PML4 索引（bits 47:39）从 PML4 表中读取 PML4E
3. PML4E 中的物理地址指向 PDPT 基址
4. 用 bits 38:30 索引 PDPTE
5. 递归至 PTE
6. PTE 中的 PFN + offset → 最终物理地址
7. 将翻译结果填充到 STLB 和 dTLB

### 页表遍历加速

- **Page Walk Cache（PWC）**：缓存中间级页表项（PML4E、PDPTE、PDE），避免每次遍历都要重新加载所有级别。AMD Zen 上的 PWC 可缓存数十至上百条中间翻译。
- **嵌套遍历时的复用**：操作系统往往将页表自身映射到虚拟地址空间（递归映射），使页表页本身也享受 cache 加速。

### 缺页异常（Page Fault）

如果任何一级表项的 P 位为 0，硬件触发缺页异常（#PF），CR2 寄存器被写入引发错误的虚拟地址。操作系统处理流程：

1. 检查访问合法性（是否超出 VMA 范围 → SIGSEGV）
2. 分配物理页帧
3. 从磁盘/swap 读入内容（major fault），或零页/COW（minor fault）
4. 填充 PTE，设置 P=1
5. IRET 返回，重新执行故障指令

## 1.5 大页（Huge Page）与 TLB Reach

### TLB Reach 问题

TLB 能覆盖的地址空间总量称为 **TLB reach**：

```
TLB Reach = TLB 条目数 × 页大小
```

对于 64 条 L1 dTLB × 4 KiB = 256 KiB reach。对于工作集超过 256 KiB 的应用（几乎所有现实应用），L1 dTLB 频繁 miss 不可避免。L2 STLB 2048 × 4 KiB = 8 MiB reach 也远不足以覆盖 GB 级工作集。

### 大页解决方案

| 页大小 | 单条 TLB 覆盖 | 64 条目 L1 dTLB reach | 2048 条目 STLB reach |
|--------|-------------|----------------------|----------------------|
| 4 KiB | 4 KiB | 256 KiB | 8 MiB |
| 2 MiB | 2 MiB | 128 MiB | 4 GiB |
| 1 GiB | 1 GiB | 64 GiB | 2 TiB |

**因此大页的首要性能收益不是减少页表遍历次数，而是扩大 TLB 覆盖范围减少 miss 率**。对于 in-memory 数据库、LLM 推理等大工作负载，透明大页（THP）或显式 hugetlbfs 是标配优化。

### 副作用

- **内部碎片**：2 MiB 页即使只用 4 KiB 也占据整个 2 MiB 物理帧
- **换页成本**：大页换出/换入耗时远大于 4 KiB 页（连续 I/O 的优势被总量抵消）
- **内存膨胀**：THP 的 khugepaged 线程在后台尝试合并 4 KiB 页→2 MiB 页，会产生额外 CPU 和内存带宽开销

## 1.6 VIPT 与 TLB–L1D 并行访问

### 问题定义

Load 指令需要**同时**知道：(a) 物理地址（用于 tag 比较和最终的重填/驱逐决策），(b) 缓存中应查找的 set。

如果 L1D 缓存使用物理索引（PIPT），则必须先完成 TLB 翻译，再访问缓存——这是**串行化**路径，代价约为 TLB 延迟 + Cache 延迟。

### VIPT 原理

VIPT 将虚拟地址的低位（落入 page offset 的部分）直接用作 cache index。对于 **4 KiB 页 + 64 B 行**：offset 为 12 bits（bits 11:0），其中低 6 bits 为行内偏移（2^6 = 64B），剩余 bits 11:6 为 6 bits 可索引 64 个 set。

对于 **32 KiB、8 路、64 B 行**的 L1D：

```
32 KiB / (64 B × 8 路) = 64 sets → 需要 6 bits 索引
```

正好 bits 11:6 全部落入 page offset 内（bits 11:0），与物理地址的对应位完全相同（虚拟→物理翻译不改变低 12 位）。这意味着：

> **TLB 翻译和 L1D tag 读取可以完全并行**：TLB 用虚拟地址高位查物理地址的同时，L1D 用虚拟地址低位选 set 并读取 tag array。

### 约束条件

VIPT 的约束条件：

```
Cache_Size ≤ (Page_Size × Associativity)
```

即 `32 KiB = 4 KiB × 8`，恰好满足。若想扩容 L1D 到 64 KiB 同时保持 4 KiB 页和 8 路，则 `64 KiB > 4 KiB × 8 = 32 KiB`，出现**同义词/混叠（aliasing）**问题——两个不同的虚拟地址可能映射到同一物理地址的不同 cache set。

处理方式包括：限制 L1D 大小、加大相联度、使用 PIPT（如部分 ARM 实现），或由硬件做同义词检测。

## 1.7 PCID/ASID 与上下文切换

### 问题

传统方案在上下文切换时刷新全部 TLB 条目（写 CR3 即触发）。这导致新进程运行初期大量 TLB miss。

### PCID（Process-Context Identifier）

x86 引入 PCID（12-bit），将 TLB 条目与特定 PCID 关联。写 CR3 时可以不刷新 PCID≠0 的条目（通过 CR4.PCIDE 和适当的 CR3 写入方式）。INVPCID 指令支持精细的 TLB 失效：

- Type 0：使能单个 PCID 的特定虚拟地址
- Type 1：使能单个 PCID 的所有条目
- Type 2：使能所有 PCID（含 PCID 0）

### AMD ASID

AMD 等价机制是 ASID（Address Space Identifier），功能和 PCID 类似但实现细节不同。AMD 处理器支持最多 32768 个 ASID（可同时跟踪此量的地址空间）。

### 实践意义

高频上下文切换场景下，PCID/ASID 显著降低 TLB miss 率。这对虚拟化环境同等重要——VMM 可以给每个 VM 分配独立 ASID/PCID，使 VM 之间切换不需要刷 TLB。

## 1.8 KPTI 与 Meltdown 对 TLB 的影响

KPTI（Kernel Page-Table Isolation）自 Meltdown 披露后成为 Linux 标准配置。其核心是维护两套页表：

- **用户态页表**：映射用户空间 + 最小内核映射（中断入口/出口蹦床）
- **内核态页表**：映射用户空间 + 完整内核映射

每次用户态↔内核态切换伴随 CR3 写入（= 页表切换），PCID 在此必不可少——否则每次系统调用都会因 TLB 刷新导致 20-40% 的性能损失。

即使有 PCID，KPTI 仍然引入可测量的开销（~3-5% 典型），来源是内核入口/出口的额外 CR3 操作和 trampoline 栈切换。

## 1.9 关键小节

- TLB 是页表遍历的专用缓存，使其 miss 率控制在可接受范围。对于大工作集，大页 > 增加 TLB 条目数
- VIPT 是 L1D 低延迟的基础保证：使得 TLB 翻译和 cache 访问可完全并行
- 页表遍历成本取决于页表自身的缓存局部性——中间级页表项由 PWC 缓存，减少内存访问
- PCID/ASID 缓解上下文切换 TLB miss，KPTI 下尤为重要
