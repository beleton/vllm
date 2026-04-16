# 访存延迟测量基础资料检索摘要

- 时间：`2026-04-14 22:05:32 +0800`
- 目标：补充 `wiki/术语与背景/访存延迟测量.md` 所需的基础知识，包括访问链路、排队/合并点，以及常见软件工具测到的延迟口径。

## 关键结论

- 常见软件工具测到的不是“单段物理链路延迟”，而是从请求发出到数据或 cache line 所有权可继续被消费的端到端服务时间。
- 若 benchmark 用 dependent pointer chase 或低并发 ping-pong，可把 benchmark 自己引入的并发压到较低，但仍会保留 cache lookup、一致性协议、fabric、内存控制器与 DRAM 时序。
- 一旦工具显式引入背景流量或系统本身已有竞争，fill buffer、L2 资源、page walk、cache miss merge、memory controller page policy 与 DRAM 调度都会进入观测延迟。

## 资料

### Intel Memory Latency Checker

- 官方页：<https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html>
- 要点：
  - `MLC` 测 `idle latency`、`loaded latency`、`cache-to-cache transfer latencies`。
  - `loaded latency` 的方法是：一个 latency thread 做 dependent reads，其余线程持续生成内存流量，并通过注入 delay 改变负载。
  - 因而 `loaded latency` 明确包含负载条件下的等待与争用，不是固定物理链路延迟。

### lmbench `lat_mem_rd`

- 手册：<https://lmbench.sourceforge.net/man/lat_mem_rd.8.html>
- 要点：
  - `lat_mem_rd` 明确测 `memory read latency`，结果以 `nanoseconds per load` 给出。
  - 其覆盖范围包括 `onboard cache latency and size, external cache latency and size, main memory latency, and TLB miss latency`。
  - 实现方式是 backward pointer ring 上的串行 dependent load。

### Intel 公开 PMU 事件文档

- 文档：<https://perfmon-events.intel.com/platforms/graniterapids/core-events/core/>
- 要点：
  - `DTLB_LOAD_MISSES.WALK_PENDING` 直接描述 demand load 的 page walk 在 `PMH` 中挂起。
  - `L1D_PEND_MISS.FB_FULL`、`L1D_PEND_MISS.L2_STALLS`、`L1D_PEND_MISS.PENDING` 直接对应 fill buffer 不可用、`L2` 资源不足、未完成 miss 在途数量。
  - `L2_RQSTS.*MISS` 明确说明 `true-miss excludes misses that were merged with ongoing L2 misses`，可用于支撑“同一 cache line 并发访问会被合并”的表述。

### AMD 5th Gen EPYC 架构白皮书

- 官方白皮书：<https://docs.amd.com/v/u/en-US/5th-gen-amd-epyc-processor-architecture-white-paper>
- 要点：
  - `EPYC 7002` 起把 memory controller 放到 `I/O die`，以降低 NUMA latency 差异。
  - `4th/5th Gen EPYC` 继续通过 `Infinity Fabric` 优化减小延迟差异。
  - 可用于支撑 chiplet 平台上“CPU die -> fabric -> I/O die memory controller”的大致链路描述。

### Intel DDR5 page policy 文档

- 官方文档：<https://cdrdv2-public.intel.com/826015/826015_Perf_Diff_Open_Pg_Rev0-9.pdf>
- 要点：
  - 文档明确写到不同 page policy 会改变 `MLC latency test result`。
  - open-page/closed-page policy 分别针对不同 DRAM page 命中模式优化调度。
  - 可用于支撑“进入内存控制器后，调度与 page hit/miss 状态会改变观测延迟”的表述。

### `core-to-core-latency`

- 项目页：<https://github.com/nviennot/core-to-core-latency>
- 要点：
  - README 明确写明：该工具测量的是 CPU 通过 `cache coherence protocol` 向另一 CPU 发送消息所需时间。
  - 方法是两个绑核线程反复做 `compare-exchange`，结果显示为 `round-trip-time/2`。
  - 可用于支撑“core-to-core latency 测到的是 coherence transfer latency，而不是单段互连裸延迟”的表述。
