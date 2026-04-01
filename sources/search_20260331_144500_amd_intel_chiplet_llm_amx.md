# 2026-03-31 AMD/Intel chiplet CPU, LLM推理与AMX 检索笔记

检索时间：`2026-03-31 14:45:00 +0800`

说明：
- `PARALLEL_API_KEY`、`OPENROUTER_API_KEY` 未配置，因此未使用 `parallel-web` / `research-lookup` API。
- 本次改为直接抓取官方 PDF / HTML，并把本机事实输出到 `sources/`。
- 后续正式分析文档：`info/ai-research/2026-03-31_AMD_Intel_Chiplet_LLM_AMX_L3研究.md`

## 1. AMD 官方资料

- 标题：`AMD EPYC 9005 Processor Architecture Overview`
  - URL：`https://docs.amd.com/api/khub/documents/iZ5eCtQ5v6PyPN7wEd8HQg/content`
  - 本地副本：`sources/ref_20260331_amd_epyc_9005_arch_overview.pdf`
  - 关键点：
    - p.3：EPYC 9005 保留 `MCM chiplet` 架构；`CCD` 围绕中心 `I/O Die`；一个 `CCX` 共享一份 `L3`，`Zen5` 为最多 `8` 核 / `32 MB L3`，`Zen5c` 为最多 `16` 核 / `32 MB L3`。
    - p.4：每个 `CCD` 只有一个 `CCX`；`CCD` 通过 `Infinity Fabric / GMI` 连到 `IOD`；`IOD` 提供 `12` 个 DDR5 UMC、PCIe Gen5、CXL 2.0。
    - p.6-p.8：`NPS` 会改变 NUMA 划分；`NPS=4` 时每象限最多 `4 CCD + 3 UMC`；`LLC as NUMA` 可把每个共享 `L3/CCX` 视为独立 NUMA 节点；双路通过 `xGMI` 互连。

- 标题：`5th Gen AMD EPYC Processor Architecture`
  - URL：`https://docs.amd.com/api/khub/documents/UIqhAbjRhgnzgzzdVU4pUw/content`
  - 本地副本：`sources/ref_20260331_amd_5th_gen_epyc_arch_whitepaper.pdf`
  - 关键点：
    - p.8：`Zen 5` 从核心开始把数据通路扩到 `512-bit`；如果 BIOS 更偏功耗，也可让 `AVX-512` 以两个 `256-bit` 顺序执行。
    - p.9-p.10：`IOD` 负责内存与 I/O；EPYC 9005 有 `12` 个 DDR5 控制器；内部 `Infinity Fabric` / `GMI` 连接 `IOD` 与各 `CPU die`；2P 下最多 `512 GB/s` socket 间带宽。

## 2. Intel 官方资料

- 标题：`4th Gen Intel Xeon Processor Scalable Family, sapphire rapids`
  - URL：`https://www.intel.com/content/www/us/en/developer/articles/technical/fourth-generation-xeon-scalable-family-overview.html`
  - 本地副本：网页已抓取分析，未单独落 HTML。
  - 关键点：
    - 表 1：每核 `2 MB private L2`、每核 `1.875 MB LLC`，并支持 `4` 个 `Sub-NUMA clusters`。
    - `UMA` 模式下，地址在所有内存控制器间交错，核心可访问整颗 die 上的 `LLC slice` 和内存。
    - `SNC` 模式下，本地域核心优先使用本地域 `memory controller + LLC slices`，可降低 `LLC / memory latency`；但 socket 全部 LLC 容量仍对每核可见。
    - `AMX` 是新 64-bit 编程范式：由二维 `tile registers` 与 `TMUL` 组成；AMX 指令与 Intel 架构指令流同步，和 `AVX-512` 可交错执行；`FP32` 仍走 `AVX-512` 路径。

- 标题：`Tuning Guide for AI on the 4th Generation Intel Xeon Scalable Processors`
  - URL：`https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-guide-for-ai-on-the-4th-generation.html`
  - 关键点：
    - 对 AI 推理，4th Gen Xeon 同时支持 `AVX512_VNNI`、`AVX512_BF16`、`AMX_INT8`、`AMX_BF16`。
    - `AMX_INT8 / AMX_BF16` 属于 `Matrix` 路径，较 `AVX512_VNNI / AVX512_BF16` 更“newer and more advanced”，框架会优先选 AMX。
    - `BF16` / `INT8` 不只提速，也会减少内存占用或带宽需求。

- 标题：`Intel Architecture Instruction Set Extensions Programming Reference`
  - URL：`https://cdrdv2.intel.com/v1/dl/getContent/671368`
  - 本地副本：`sources/ref_20260331_intel_ise_programming_reference.pdf`
  - 关键点：
    - p.15：`Intel AMX` 在 `Sapphire Rapids` 引入，包含 `AMX-BF16 / AMX-TILE / AMX-INT8`。
    - p.98-p.99：AMX 由主机负责算法、blocking、循环与指针；tile load/store 与 accelerator 命令发往多周期执行单元；AMX 访问对 host memory coherent，且与 `AVX-512` 可并行/交错。
    - p.101：示例 `C += A * B` 明确显示 `TILELOADD + TDPBUSD + TILESTORED` 的矩阵乘法执行模式。

- 标题：`Intel Launches 4th Gen Xeon Scalable Processors...`
  - URL：`https://download.intel.com/newsroom/archive/2025/en-us-2023-01-10-intel-launches-4th-gen-xeon-scalable-processors-max-series-cpus.pdf`
  - 关键点：
    - p.3：4th Gen Xeon 依靠 `Intel AMX` 做 AI 推理/训练加速。
    - p.5：4th Gen Xeon 由最多 `4` 个 tiles 组成，使用 `EMIB` 封装，并提升 DDR5 / PCIe5 / CXL 带宽。

- 标题：`Are the Intel Xeon Scalable Processor Families Single-Chip or MCM?`
  - URL：`https://www.intel.com/content/www/us/en/support/articles/000099181/processors/intel-xeon-processors.html`
  - 关键点：
    - 4th Gen Xeon 是 `MCM`，含多个 compute tiles。
    - Xeon 6 转向更彻底的 `modular, tile-based architecture`，可含 compute / I/O tiles。

## 3. 学术 / 高质量论文

- 标题：`SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills`
  - URL：`https://www.microsoft.com/en-us/research/publication/sarathi-efficient-llm-inference-by-piggybacking-decodes-with-chunked-prefills/`
  - 关键点：
    - 明确区分 `prefill` 与 `decode` 两阶段。
    - `prefill` 在小 batch 下可较充分占用算力；`decode` 因逐 token 生成而出现低计算利用率。

- 标题：`vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention`
  - URL：`https://arxiv.org/abs/2405.04437`
  - 关键点：
    - `PagedAttention` 会把 KV cache 的虚拟地址布局从连续改成不连续，从而带来额外编程和性能开销。
    - `vAttention` 通过保留 KV cache 的虚拟地址连续性来降低这类开销。

## 4. 本机事实文件

- `sources/local_20260331_lscpu.txt`
  - `Model name: AMD EPYC 9745 128-Core Processor`
  - `L3 cache: 512 MiB (16 instances)`
  - `NUMA node(s): 2`

- `sources/local_20260331_cpu_flags.txt`
  - 可见 `avx_vnni`、`avx512_bf16`、`avx512_vnni`
  - 未检出任何 `amx_*` flags
