# CPU 缓存体系与内存访问机制

本系列文档围绕**一次 Load/Store 指令在现代乱序执行 CPU 中的完整生命周期**展开，自顶向下追溯从地址生成到 DRAM 数据返回的全路径。目标读者具备体系结构基础，期望获得课程深度 + 工程剖面级别的理解。

## 文档结构

| 章节 | 文件 | 核心内容 |
|------|------|----------|
| 0 | 本文 | 总体架构概览与阅读指南 |
| 1 | [01_地址转换_TLB与页表](01_地址转换_TLB与页表.md) | 虚拟→物理地址翻译、TLB 层次、页表遍历、VIPT 原理 |
| 2 | [02_L1数据缓存与存储转发](02_L1数据缓存与存储转发.md) | L1D 微架构、tag/data 并行访问、Store Buffer、Store-to-Load Forwarding |
| 3 | [03_L2与L3缓存子系统](03_L2与L3缓存子系统.md) | L2 组织、L3 切片与哈希、Ring/Mesh 互连、Victim Cache、NUMA |
| 4 | [04_DRAM与内存控制器](04_DRAM与内存控制器.md) | DRAM 单元到 channel/rank/bank 层级、时序参数、地址映射、内存控制器调度 |
| 5 | [05_乱序执行引擎中的访存](05_乱序执行引擎中的访存.md) | ROB、RS、LDQ/STQ、内存消歧（Memory Disambiguation）、机器清除 |
| 6 | [06_硬件预取机制](06_硬件预取机制.md) | 各层级预取器（stride/stream/spatial/IP-based）、训练与节流、软件预取 |
| 7 | [07_缓存一致性协议](07_缓存一致性协议.md) | MESI/MOESI/MESIF、Snoop Filter、目录协议、跨 Socket 一致性与原子操作 |

## 一次 Load 的完整路径（总览）

以下以 AMD Zen 5 一条普通的整数 Load 指令为例，给出关键步骤与涉及硬件，后续章节对每一步做详细展开：

```
1. 指令译码 → 进入 μOP 队列 → ROB 分配条目 → RS 等待操作数
2. AGU 计算有效地址（base + index*scale + displacement）
3. 虚拟地址拆分：低 12 位送 L1D tag 阵列做 index 选择，高位送 dTLB
4. TLB 查找与 L1D tag 读取并行执行（VIPT 保证低 12 位不变，index 落入 page offset）
5. TLB 命中 → 获得物理地址；TLB miss → L2 TLB → 页表遍历器
6. L1D tag 比较：若命中 → 根据 way 选择读出 data array，数据返回寄存器
7. L1D miss → 查询 Store Buffer → miss → 分配 LFB/MSHR → 发请求到 L2
8. L2 查找：tag 比较 → 命中 → 数据回填 L1D，Load 完成
9. L2 miss → 查询本地 L3 slice（由物理地址哈希决定 slice）
10. L3 miss → 查询 snoop filter → 可能需要跨 CCX/CCD snoop
11. 所有片上缓存 miss → 内存控制器发出 DRAM 读写命令
12. DRAM 激活行 → 列读取 → 数据经 Infinity Fabric / UPI 返回
13. 数据沿缓存层次逐级回填 → Load 完成 → ROB 退休
```

## 关键层次延迟量级（AMD Zen 5 EPYC 参考值）

| 访问路径 | 典型延迟 | 备注 |
|----------|---------|------|
| L1D | ~4-5 cycle (~1.7 ns) | 4 cycle 流水线访问 |
| L2 | ~12-14 cycle (~8.7 ns) | 统一 I+D，8 路组相联 |
| 本地 L3 | ~40-50 cycle (~26 ns) | 同一 CCX 内 |
| 跨 CCX L3 | ~110-150 ns | 经 Infinity Fabric |
| 本地 DRAM | ~120 ns | 同一 socket 内存控制器 |
| 跨 Socket DRAM | ~200-300 ns | 经 Infinity Fabric 跨 socket |

## 分层架构图

```
Core 0                  Core 1                  Core N
 ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
 │  ROB / Scheduler │    │  ROB / Scheduler │    │  ROB / Scheduler │
 │  LDQ │  STQ      │    │  LDQ │  STQ      │    │  LDQ │  STQ      │
 │  L1I │ L1D │ dTLB│    │  L1I │ L1D │ dTLB│    │  L1I │ L1D │ dTLB│
 │       L2 (私有)   │    │       L2 (私有)   │    │       L2 (私有)   │
 └────────┬────────-┘    └────────┬────────-┘    └────────┬────────-┘
          │                       │                       │
     ┌────┴───────────────────────┴───────────────────────┴────┐
     │              L3 (CCX 内共享, 多 slice)                    │
     │              Snoop Filter / Probe Filter                 │
     └───────────────────────────┬──────────────────────────────┘
                                 │ Infinity Fabric / Ring / Mesh
     ┌───────────────────────────┴──────────────────────────────┐
     │                    内存控制器 (IMC)                        │
     │                  Channel 0 ... Channel N                  │
     └───────────────────────────┬──────────────────────────────┘
                                 │
                          ┌──────┴──────┐
                          │  DDR5 DIMM  │
                          │ R0  R1  ... │
                          └─────────────┘
```

## 与具体工具的对应关系

本文档讨论的微架构概念对应于以下观测工具的实际含义：

- **AMDuProfPcm**：L3PMCx04（L3 hit/miss per CCX）、L2PMCx00（L2 access/miss）、L1PMCx00（L1D access）
- **perf mem**：data_src 字段直接映射到缓存命中层级（L1/L2/L3 hit、local/remote DRAM）
- **IBS**（Instruction-Based Sampling）：OpDataSrc 可反推数据来源延迟（L1/L2/L3/local DRAM/remote cache/remote DRAM）
- **perf stat** 通用事件：l1d_cache_access/l1d_cache_miss、l2_cache_access/l2_cache_miss、llc_cache_access/llc_cache_miss

## 参考文献与进一步阅读

- Intel 64 and IA-32 Architectures Software Developer's Manual, Vol. 3A: System Programming Guide
- AMD64 Architecture Programmer's Manual, Vol. 2: System Programming
- AMD Processor Programming Reference (PPR) for specific family/model
- _What Every Programmer Should Know About Memory_, Ulrich Drepper (2007)
- _A Primer on Memory Consistency and Cache Coherence_, Sorin/Hill/Wood (2011)
- Intel _Optimization Reference Manual_, Chapters on cache and memory subsystem
