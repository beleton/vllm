# 论文解读
- [论文解读写作规范.md](./论文解读写作规范.md)：后续撰写论文解读时应遵循的证据、图表引用、措辞与边界规则
- [论文方法线对比总览.md](./论文方法线对比总览.md)：按方法线归类比较现有论文，提炼各篇最核心思想
- [GPU_Attention_NUMA_Optimization解读.md](./GPU_Attention_NUMA_Optimization解读.md)：GPU attention 中按共享 `K/V` 工作集做拓扑感知放置
- [ArcLight解读.md](./ArcLight解读.md)：many-core CPU 上把 NUMA-local 内存、线程组和 TP 联合设计的系统思路
- [CHARM解读.md](./CHARM解读.md)：chiplet CPU 上在 `locality` 与聚合 `L3` 容量之间动态切换的 runtime 设计
- [AMD_EPYC_Rome与Intel_Cascade_Lake_SP内存性能解读.md](./AMD_EPYC_Rome与Intel_Cascade_Lake_SP内存性能解读.md)：Rome/CLX 的本地与远端缓存、NUMA、双路主存延迟和带宽对比
- [Optimizing_Sorting_for_Chiplet_Based_CPUs解读.md](./Optimizing_Sorting_for_Chiplet_Based_CPUs解读.md)：chiplet CPU 上按本地/聚合 `L3` 容量切换放置策略并避免 `data shuffling` 的排序优化
- [Chiplet三篇论文对照与Attention结论验证.md](./Chiplet三篇论文对照与Attention结论验证.md)：三篇 chiplet 论文（OLAP/Sorting/CHARM）的横向对比，以及它们与 Attention L3 不敏感结论的对照验证
