# AMD Intel Chiplet CPU 与 AMX L3 关系

## 问题
- AMD / Intel 的 chiplet、tile、`SNC`、`LLC` 局部域应如何区分。
- `L3` 感知的数据放置 / 任务放置，与 `AMX` 这类矩阵扩展到底是不是同一类优化。

## 结论
- `chiplet / MCM`、`tile`、`SNC domain`、`socket` 不是同一层概念。
- Intel `AMX` 作用在矩阵乘法执行层；`L3` 感知放置作用在缓存 / 拓扑 / 内存层。两者不是同一层优化。
- 更稳的说法是：
  - 在支持 `AMX` 的 Intel CPU 上，它们是“部分正交、整体互补”
  - 在 attention / KV cache 场景里，拓扑与缓存局部性问题相对更独立
- 当前这台 AMD EPYC 9745 不支持 `AMX`，但这不影响 `L3` 感知放置成为独立且合理的研究方向。

## 分析
- Intel 侧需要拆开看：
  - 封装层的 `MCM / chiplet`
  - 封装内的 `tile`
  - 软件可见的 `SNC` 局部域
  - 跨 socket 的 `UPI`
- AMD 侧当前更直接的是：
  - `CCD / CCX / 32 MiB L3`
  - `NPS`
  - `LLC as NUMA`
- 对当前课题，attention / KV cache 更像“共享工作集局部化”问题，而不是先找 `AMX` 的 AMD 对等物。
- 因为本地源码和 `PCM` 已经给出两个直接支撑：
  - prefill 里同一 `kv_head` 存在多线程重复读重叠 `KV` 前缀
  - decode 高 `L3 Miss` 同时 `Remote DRAM Reads %` 很低，说明不能先把主因写成远端 DRAM

## 证据
- 本地锚点：
  - `csrc/cpu/cpu_attn_impl.hpp:436-589`
  - `csrc/cpu/cpu_attn_impl.hpp:1424-1459`
  - `csrc/cpu/cpu_attn_impl.hpp:1527-1561`
  - `csrc/cpu/cpu_attn_impl.hpp:1631-1727`
  - `test_results/PD_Test/Qwen3-30B-A3B/PCm_res/2026-03-26_qwen3-30b-a3b_pd_pcm_summary.md`

## 边界
- 本页保留的是当前课题直接相关的结论，不展开完整文献综述。
- 里面关于 Intel `disaggregated / tile-based` 与当前课题的映射，属于基于官方材料的推断，不是本地实验结论。
