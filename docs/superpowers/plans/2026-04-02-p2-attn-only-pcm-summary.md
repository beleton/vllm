# P2 Attention-Only PCM 汇总与出图脚本 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `test_results/P2_AttnOnly/...` 下的 attention-only PCM 结果提供可复用的解析与出图脚本，统一导出 `summary.csv`、`summary.md` 和 `plots/*.png`。

**Architecture:** 采用两阶段方案。第一阶段扫描结果目录中的 `dry_run_summary.json` 与 `report.json`，抽取 `system` 级 L3 指标与可选 `ccx` 级明细，输出结构化表格与摘要文档；第二阶段从结构化表格读取数据，按 `decode-like/prefill-like` 生成折线图与 latency 组成图，避免将解析逻辑和绘图逻辑耦合在一起。

**Tech Stack:** Python 3 标准库、`matplotlib`、`numpy`

---

### Task 1: 任务登记与文件布局

**Files:**
- Modify: `tasks.json`
- Create: `docs/superpowers/plans/2026-04-02-p2-attn-only-pcm-summary.md`

- [ ] **Step 1: 在 `tasks.json` 新增任务**

将任务描述写成“通用 PCM 结果整理与出图脚本”，状态先设为 `pending`。

- [ ] **Step 2: 更新 `updated_at`**

写入本轮开始时间戳。

- [ ] **Step 3: 保存计划文档**

把本计划写到当前文件，作为后续实现依据。

### Task 2: 先写失败测试，固定解析口径

**Files:**
- Create: `tests/tools/test_p2_attn_only_pcm.py`
- Create: `tools/p2_attn_only/pcm_summary.py`

- [ ] **Step 1: 写目录扫描测试**

覆盖从结果路径中解析 `workload/partition_mode/batch_size/q_len/kv_len/session_dir`。

- [ ] **Step 2: 写指标提取测试**

覆盖从 `report.json` 中只提取 `group=[{'l3': [{'id': -1, 'name': 'system'}]}]` 的目标指标，并验证 `L3 Miss Latency From ... (%)` 指标集合完整。

- [ ] **Step 3: 写 dry-run 摘要拼接测试**

覆盖 `dry_run_summary.json` 新旧字段兼容，至少验证 `slowest_rank_mean_ms`、`head_plan.partition_mode`、`resolved_lengths/result_shape_dirname`。

- [ ] **Step 4: 运行单测，确认失败**

Run: `python -m unittest tests.tools.test_p2_attn_only_pcm -v`
Expected: 因模块/函数不存在而失败。

### Task 3: 实现解析脚本

**Files:**
- Create: `tools/p2_attn_only/pcm_summary.py`
- Modify: `tests/tools/test_p2_attn_only_pcm.py`

- [ ] **Step 1: 实现结果扫描与路径元数据解析**

为每个 case 建立一行 summary，字段至少包含：

```python
{
    "case_dir": "...",
    "session_dir": "...",
    "workload": "decode-like",
    "partition_mode": "global-fixed",
    "batch_size": 16,
    "q_len": 1,
    "kv_len": 1024,
    "slowest_rank_mean_ms": 0.0509,
}
```

- [ ] **Step 2: 实现 `report.json` system 级指标提取**

抽取以下指标的 `aggregated.average`：

```python
[
    "L3 Access (pti)",
    "L3 Miss (pti)",
    "L3 Miss %",
    "Ave L3 Miss Latency (ns)",
    "L3 Miss Latency From Local Memory or I/O (%)",
    "L3 Miss Latency From Remote Memory or I/O (%)",
    "L3 Miss Latency From another CCX in same node (%)",
    "L3 Miss Latency From another CCX in remote node (%)",
    "L3 Miss Latency From Local Extension Memory (CXL) (%)",
    "L3 Miss Latency From Remote Extension Memory (CXL) (%)",
]
```

- [ ] **Step 3: 实现可选 `ccx` 明细导出**

额外输出 `ccx_summary.csv`，保留 `ccx_id` 与相同指标，便于后续深挖，但默认图只读 system 级 summary。

- [ ] **Step 4: 输出 `summary.csv` 和 `summary.md`**

`summary.md` 只做简短表格与数据路径索引，不写无证据结论。

- [ ] **Step 5: 运行单测，确认通过**

Run: `python -m unittest tests.tools.test_p2_attn_only_pcm -v`
Expected: PASS

### Task 4: 实现出图脚本

**Files:**
- Create: `tools/p2_attn_only/plot_pcm_summary.py`
- Modify: `tests/tools/test_p2_attn_only_pcm.py`

- [ ] **Step 1: 写最小绘图测试**

覆盖读取 `summary.csv` 后按 workload 分组、按长度升序排序，以及 latency 组成列求和。

- [ ] **Step 2: 先运行测试，确认失败**

Run: `python -m unittest tests.tools.test_p2_attn_only_pcm -v`
Expected: 因绘图辅助函数不存在而失败。

- [ ] **Step 3: 实现折线图**

输出：
- `plots/decode_l3_miss_vs_kv_len.png`
- `plots/decode_l3_latency_vs_kv_len.png`
- `plots/prefill_l3_miss_vs_q_len.png`
- `plots/prefill_l3_latency_vs_q_len.png`

- [ ] **Step 4: 实现 latency 组成图**

输出：
- `plots/decode_l3_latency_breakdown_vs_kv_len.png`
- `plots/prefill_l3_latency_breakdown_vs_q_len.png`

- [ ] **Step 5: 再跑测试**

Run: `python -m unittest tests.tools.test_p2_attn_only_pcm -v`
Expected: PASS

### Task 5: 实跑生成产物并收尾

**Files:**
- Modify: `tasks.json`
- Modify: `process.txt` (only if handoff is needed)
- Create: `test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.csv`
- Create: `test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/ccx_summary.csv`
- Create: `test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.md`
- Create: `test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/plots/*.png`

- [ ] **Step 1: 运行解析脚本**

Run:

```bash
python tools/p2_attn_only/pcm_summary.py \
  --result-root test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2
```

Expected: 生成 `summary.csv`、`ccx_summary.csv`、`summary.md`

- [ ] **Step 2: 运行出图脚本**

Run:

```bash
python tools/p2_attn_only/plot_pcm_summary.py \
  --summary-csv test_results/P2_AttnOnly/Qwen3-30B-A3B/NPS1_TP2/summary.csv
```

Expected: `plots/` 下生成 6 张图

- [ ] **Step 3: 抽查产物**

确认 `summary.csv` 行数与当前 case 数一致，图文件存在且非空。

- [ ] **Step 4: 更新任务状态**

把 `tasks.json` 中该任务标为 `completed`，`passes=true`。

- [ ] **Step 5: 只在确有必要时补 `process.txt`**

若脚本后续复用方式、输入口径或输出路径需要交接，再写一条。
