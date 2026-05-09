from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tools.p3_attn_only.attn_mapping_compare_to_html import (
    ModeSummary,
    build_mode_summary,
)
from tools.p3_attn_only.attn_trace_to_html import load_trace_records


@dataclass(frozen=True)
class CoreKVRow:
    ccd: int
    core: int
    kv_head_idx: int
    task_rows: int
    total_q_tiles: int
    total_q_tokens: int
    q_start_min: int
    q_end_max: int
    workitem_group_idxs: list[int]


@dataclass(frozen=True)
class CoreTaskRow:
    ccd: int
    core: int
    kv_head_idx: int
    q_tile_num: int
    q_token_num: int
    q_start: int
    q_end: int
    workitem_group_idx: int
    line_no: int


@dataclass(frozen=True)
class RankSection:
    rank: int
    balanced_mapping: ModeSummary
    acc_mapping: ModeSummary
    balanced_task_rows: list[CoreTaskRow]
    acc_task_rows: list[CoreTaskRow]


@dataclass(frozen=True)
class RuntimeModeSummary:
    core_count: int
    kv_count: int
    task_nums: int
    total_q_tiles: int


@dataclass(frozen=True)
class AlignedTaskRow:
    core: int
    show_core: bool
    balanced_task_nums: int | None
    balanced_task: CoreTaskRow | None
    acc_task_nums: int | None
    acc_task: CoreTaskRow | None


@dataclass(frozen=True)
class CCDRuntimeComparison:
    ccd: int
    balanced_summary: RuntimeModeSummary
    acc_summary: RuntimeModeSummary
    rows: list[AlignedTaskRow]


def _extract_summary_json(log_text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(log_text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(log_text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "rank_results" in candidate:
            return candidate
    return None


def extract_summary_ranks(log_path: str | Path) -> list[int]:
    log_path = Path(log_path).resolve()
    summary = _extract_summary_json(log_path.read_text(encoding="utf-8"))
    if not summary:
        return []
    ranks: list[int] = []
    for item in summary.get("rank_results", []):
        rank = item.get("rank")
        if rank is None:
            continue
        ranks.append(int(rank))
    return sorted(set(ranks))


def collect_trace_ranks(log_path: str | Path) -> list[int]:
    records = load_trace_records(log_path)
    return sorted({int(record.get("rank", -1)) for record in records})


def build_core_kv_rows(log_path: str | Path, rank: int = 0) -> list[CoreKVRow]:
    records = [
        record
        for record in load_trace_records(log_path)
        if int(record.get("rank", -1)) == rank
    ]
    if not records:
        raise ValueError(f"no trace rows for rank={rank}: {Path(log_path).resolve()}")

    grouped: dict[tuple[int, int, int], dict[str, object]] = {}
    for record in records:
        ccd = int(record.get("ccd", -1))
        core = int(record.get("core", -1))
        kv_head_idx = int(record["kv_head_idx"])
        key = (ccd, core, kv_head_idx)
        row = grouped.setdefault(
            key,
            {
                "ccd": ccd,
                "core": core,
                "kv_head_idx": kv_head_idx,
                "task_rows": 0,
                "total_q_tiles": 0,
                "total_q_tokens": 0,
                "q_start_min": None,
                "q_end_max": None,
                "workitem_group_idxs": set(),
            },
        )
        q_tile_num = int(record.get("q_tile_num", len(record.get("q_tiles", []))))
        q_token_num = int(record["q_token_num"])
        q_start = int(record["q_token_id_start"])
        q_end = q_start + q_token_num

        row["task_rows"] = int(row["task_rows"]) + 1
        row["total_q_tiles"] = int(row["total_q_tiles"]) + q_tile_num
        row["total_q_tokens"] = int(row["total_q_tokens"]) + q_token_num
        row["q_start_min"] = (
            q_start
            if row["q_start_min"] is None
            else min(int(row["q_start_min"]), q_start)
        )
        row["q_end_max"] = (
            q_end
            if row["q_end_max"] is None
            else max(int(row["q_end_max"]), q_end)
        )
        if "workitem_group_idx" in record:
            workitem_group_idxs = row["workitem_group_idxs"]
            assert isinstance(workitem_group_idxs, set)
            workitem_group_idxs.add(int(record["workitem_group_idx"]))

    rows: list[CoreKVRow] = []
    for row in grouped.values():
        workitem_group_idxs = row["workitem_group_idxs"]
        assert isinstance(workitem_group_idxs, set)
        rows.append(
            CoreKVRow(
                ccd=int(row["ccd"]),
                core=int(row["core"]),
                kv_head_idx=int(row["kv_head_idx"]),
                task_rows=int(row["task_rows"]),
                total_q_tiles=int(row["total_q_tiles"]),
                total_q_tokens=int(row["total_q_tokens"]),
                q_start_min=int(row["q_start_min"]),
                q_end_max=int(row["q_end_max"]),
                workitem_group_idxs=sorted(workitem_group_idxs),
            )
        )
    rows.sort(key=lambda row: (row.ccd, row.core, row.kv_head_idx))
    return rows


def build_core_task_rows(log_path: str | Path, rank: int = 0) -> list[CoreTaskRow]:
    records = [
        record
        for record in load_trace_records(log_path)
        if int(record.get("rank", -1)) == rank
    ]
    if not records:
        raise ValueError(f"no trace rows for rank={rank}: {Path(log_path).resolve()}")

    rows = [
        CoreTaskRow(
            ccd=int(record.get("ccd", -1)),
            core=int(record.get("core", -1)),
            kv_head_idx=int(record["kv_head_idx"]),
            q_tile_num=int(record.get("q_tile_num", len(record.get("q_tiles", [])))),
            q_token_num=int(record["q_token_num"]),
            q_start=int(record["q_token_id_start"]),
            q_end=int(record["q_token_id_start"]) + int(record["q_token_num"]),
            workitem_group_idx=int(record["workitem_group_idx"]),
            line_no=int(record["line_no"]),
        )
        for record in records
    ]
    rows.sort(
        key=lambda row: (
            row.ccd,
            row.core,
            row.q_start,
            row.workitem_group_idx,
            row.kv_head_idx,
            row.line_no,
        )
    )
    return rows


def build_rank_section(
    balanced_log: str | Path,
    acc_log: str | Path,
    rank: int,
) -> RankSection:
    return RankSection(
        rank=rank,
        balanced_mapping=build_mode_summary(balanced_log, rank=rank),
        acc_mapping=build_mode_summary(acc_log, rank=rank),
        balanced_task_rows=build_core_task_rows(balanced_log, rank=rank),
        acc_task_rows=build_core_task_rows(acc_log, rank=rank),
    )


def _heat_color(value: int, max_value: int) -> str:
    if max_value <= 0 or value <= 0:
        return "#f3f1ea"
    ratio = value / max_value
    lightness = 94.0 - ratio * 46.0
    return f"hsl(16 78% {lightness:.1f}%)"


def _render_matrix(summary: ModeSummary) -> str:
    max_value = max(max(row) for row in summary.matrix) if summary.matrix else 0
    parts = [
        '<div class="panel">',
        f'<div class="panel-title">{html.escape(summary.mode)}</div>',
        '<table class="matrix-table">',
        "<thead><tr><th>CCD</th>",
    ]
    for kv_head in summary.kv_heads:
        parts.append(f"<th>kv{kv_head}</th>")
    parts.append("</tr></thead><tbody>")
    for group, row in zip(summary.l3_groups, summary.matrix, strict=True):
        label = f"l3_{group.l3_cache_id}"
        parts.append(f"<tr><th>{html.escape(label)}</th>")
        for value in row:
            parts.append(
                f'<td style="background:{_heat_color(value, max_value)}">{value}</td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _render_summary_facts(balanced: ModeSummary, acc: ModeSummary) -> str:
    balanced_max_kv_span = max(balanced.kv_span_counts.values())
    balanced_max_l3_load = max(balanced.l3_kv_counts.values())
    acc_max_kv_span = max(acc.kv_span_counts.values())
    acc_max_l3_load = max(acc.l3_kv_counts.values())
    return (
        '<div class="facts">'
        '<div class="fact-card">'
        '<div class="fact-label">balanced</div>'
        f'<div class="fact-value">kv_head 最多横跨 {balanced_max_kv_span} 个 CCD</div>'
        f'<div class="fact-sub">单个 CCD 最多同时计算 {balanced_max_l3_load} 个 kv_head</div>'
        "</div>"
        '<div class="fact-card">'
        '<div class="fact-label">acc-local-l3</div>'
        f'<div class="fact-value">每个 kv_head 只落在 {acc_max_kv_span} 个 CCD</div>'
        f'<div class="fact-sub">单个 CCD 最多同时计算 {acc_max_l3_load} 个 kv_head</div>'
        "</div>"
        "</div>"
    )


def _group_task_rows_by_ccd(rows: list[CoreTaskRow]) -> dict[int, list[CoreTaskRow]]:
    grouped: dict[int, list[CoreTaskRow]] = defaultdict(list)
    for row in rows:
        grouped[row.ccd].append(row)
    return dict(sorted(grouped.items()))


def _group_task_rows_by_ccd_and_core(
    rows: list[CoreTaskRow],
) -> dict[int, dict[int, list[CoreTaskRow]]]:
    grouped: dict[int, dict[int, list[CoreTaskRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row.ccd][row.core].append(row)
    return {
        ccd: {core: core_rows for core, core_rows in sorted(core_map.items())}
        for ccd, core_map in sorted(grouped.items())
    }


def _summarize_runtime_rows(rows_by_core: dict[int, list[CoreTaskRow]]) -> RuntimeModeSummary:
    rows = [row for core_rows in rows_by_core.values() for row in core_rows]
    return RuntimeModeSummary(
        core_count=len(rows_by_core),
        kv_count=len({row.kv_head_idx for row in rows}),
        task_nums=len(rows),
        total_q_tiles=sum(row.q_tile_num for row in rows),
    )


def build_ccd_runtime_comparisons(
    balanced_rows: list[CoreTaskRow],
    acc_rows: list[CoreTaskRow],
) -> list[CCDRuntimeComparison]:
    balanced_grouped = _group_task_rows_by_ccd_and_core(balanced_rows)
    acc_grouped = _group_task_rows_by_ccd_and_core(acc_rows)

    comparisons: list[CCDRuntimeComparison] = []
    for ccd in sorted(set(balanced_grouped) | set(acc_grouped)):
        balanced_by_core = balanced_grouped.get(ccd, {})
        acc_by_core = acc_grouped.get(ccd, {})
        aligned_rows: list[AlignedTaskRow] = []

        for core in sorted(set(balanced_by_core) | set(acc_by_core)):
            balanced_core_rows = balanced_by_core.get(core, [])
            acc_core_rows = acc_by_core.get(core, [])
            total_rows = max(len(balanced_core_rows), len(acc_core_rows))

            for idx in range(total_rows):
                aligned_rows.append(
                    AlignedTaskRow(
                        core=core,
                        show_core=idx == 0,
                        balanced_task_nums=(
                            len(balanced_core_rows)
                            if idx == 0 and balanced_core_rows
                            else None
                        ),
                        balanced_task=(
                            balanced_core_rows[idx]
                            if idx < len(balanced_core_rows)
                            else None
                        ),
                        acc_task_nums=(
                            len(acc_core_rows) if idx == 0 and acc_core_rows else None
                        ),
                        acc_task=acc_core_rows[idx] if idx < len(acc_core_rows) else None,
                    )
                )

        comparisons.append(
            CCDRuntimeComparison(
                ccd=ccd,
                balanced_summary=_summarize_runtime_rows(balanced_by_core),
                acc_summary=_summarize_runtime_rows(acc_by_core),
                rows=aligned_rows,
            )
        )
    return comparisons


def _render_runtime_summary(mode: str, summary: RuntimeModeSummary, css_class: str) -> str:
    return (
        f'<div class="mode-summary {css_class}">'
        f'<div class="mode-summary-title">{html.escape(mode)}</div>'
        "<div>"
        f"cores={summary.core_count} | kv_heads={summary.kv_count} | "
        f"task_nums={summary.task_nums} | total q_tile={summary.total_q_tiles}"
        "</div></div>"
    )


def _task_cells(task: CoreTaskRow | None) -> list[str]:
    if task is None:
        return ["", "", "", "", ""]
    return [
        f"kv{task.kv_head_idx}",
        str(task.q_tile_num),
        str(task.q_token_num),
        str(task.workitem_group_idx),
        f"[{task.q_start},{task.q_end})",
    ]


def _render_runtime_comparison_panel(section: RankSection) -> str:
    comparisons = build_ccd_runtime_comparisons(
        section.balanced_task_rows, section.acc_task_rows
    )
    parts = [
        '<div class="panel span-2 runtime-panel">',
        '<div class="panel-title">balanced vs acc-local-l3</div>',
        '<div class="panel-subtitle">'
        '共享列只有 <code>Core</code>；左右两侧分别保留各自的 <code>Task Nums</code>、'
        '<code>KV Head</code>、<code>Q Tiles</code>、<code>Q Tokens</code>、'
        '<code>Workitem</code>、<code>Q Range</code>。'
        "</div>",
    ]
    if not comparisons:
        parts.append('<p class="muted">No runtime rows.</p></div>')
        return "".join(parts)

    for comparison in comparisons:
        parts.append('<section class="ccd-block">')
        parts.append(f'<div class="ccd-title">CCD {comparison.ccd}</div>')
        parts.append(
            '<div class="mode-summary-grid">'
            f"{_render_runtime_summary('balanced', comparison.balanced_summary, 'mode-balanced')}"
            f"{_render_runtime_summary('acc-local-l3', comparison.acc_summary, 'mode-acc')}"
            "</div>"
        )
        parts.append(
            '<div class="table-wrap"><table class="runtime-table runtime-compare-table">'
            "<thead><tr>"
            '<th class="shared-head core-head">Core</th>'
            '<th class="balanced-head">balanced Task Nums</th>'
            '<th class="balanced-head">balanced KV Head</th>'
            '<th class="balanced-head">balanced Q Tiles</th>'
            '<th class="balanced-head">balanced Q Tokens</th>'
            '<th class="balanced-head">balanced Workitem</th>'
            '<th class="balanced-head">balanced Q Range</th>'
            '<th class="acc-head">acc-local-l3 Task Nums</th>'
            '<th class="acc-head">acc-local-l3 KV Head</th>'
            '<th class="acc-head">acc-local-l3 Q Tiles</th>'
            '<th class="acc-head">acc-local-l3 Q Tokens</th>'
            '<th class="acc-head">acc-local-l3 Workitem</th>'
            '<th class="acc-head">acc-local-l3 Q Range</th>'
            "</tr></thead><tbody>"
        )
        for row in comparison.rows:
            balanced_cells = _task_cells(row.balanced_task)
            acc_cells = _task_cells(row.acc_task)
            parts.append(
                "<tr>"
                f'<td class="core-cell">{html.escape(str(row.core) if row.show_core else "")}</td>'
                f"<td>{html.escape(str(row.balanced_task_nums) if row.balanced_task_nums is not None else '')}</td>"
                f"<td>{html.escape(balanced_cells[0])}</td>"
                f"<td>{html.escape(balanced_cells[1])}</td>"
                f"<td>{html.escape(balanced_cells[2])}</td>"
                f"<td>{html.escape(balanced_cells[3])}</td>"
                f"<td>{html.escape(balanced_cells[4])}</td>"
                f"<td>{html.escape(str(row.acc_task_nums) if row.acc_task_nums is not None else '')}</td>"
                f"<td>{html.escape(acc_cells[0])}</td>"
                f"<td>{html.escape(acc_cells[1])}</td>"
                f"<td>{html.escape(acc_cells[2])}</td>"
                f"<td>{html.escape(acc_cells[3])}</td>"
                f"<td>{html.escape(acc_cells[4])}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></div></section>")
    parts.append("</div>")
    return "".join(parts)


def _render_rank_section(section: RankSection) -> str:
    return (
        '<section class="rank-section">'
        f'<div class="rank-title">Rank {section.rank}</div>'
        f"{_render_summary_facts(section.balanced_mapping, section.acc_mapping)}"
        '<div class="grid">'
        '<div class="panel span-2">'
        '<div class="panel-title">CCD × KV Head 任务计数矩阵</div>'
        '<div class="panel-subtitle">'
        '单元格数字表示该 CCD 上命中的该 kv_head trace 行数。'
        '这个区块用于直接看 kv_head 是否扩散到多个 CCD。'
        "</div></div>"
        f"{_render_matrix(section.balanced_mapping)}"
        f"{_render_matrix(section.acc_mapping)}"
        '<div class="panel span-2">'
        '<div class="panel-title">CCD 内核心执行明细</div>'
        '<div class="panel-subtitle">'
        '这个区块回答：一个 CCD 上的几个核心，分别算了哪些 kv_head，'
        '以及这些 kv_head 一共经历了多少个 q_tile。'
        "</div></div>"
        f"{_render_runtime_comparison_panel(section)}"
        "</div></section>"
    )


def render_html(
    rank_sections: list[RankSection],
    balanced_log: str | Path,
    acc_log: str | Path,
    trace_note: str,
) -> str:
    balanced_log = Path(balanced_log).resolve()
    acc_log = Path(acc_log).resolve()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attention Locality Compare Report</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --panel: #fffdf8;
      --ink: #1f1a14;
      --muted: #6c6258;
      --line: #d7cdbf;
      --balanced: #cf5c36;
      --acc: #2f6c62;
      --shadow: 0 18px 40px rgba(56, 40, 22, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #efe7d8 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }}
    .hero, .rank-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 10px 0 12px;
      font-size: 34px;
      line-height: 1.15;
    }}
    .subtitle, .panel-subtitle, .hero-note, .logs {{
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
    }}
    .hero-note {{
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 14px;
      background: #f5eee1;
      border: 1px solid var(--line);
    }}
    .logs {{
      margin-top: 18px;
      font-size: 13px;
    }}
    .rank-title {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 18px;
    }}
    .facts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .fact-card {{
      background: #f9f4ea;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
    }}
    .fact-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .fact-value {{
      margin-top: 8px;
      font-size: 24px;
      font-weight: 700;
    }}
    .fact-sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px 18px 20px;
    }}
    .span-2 {{
      grid-column: span 2;
    }}
    .panel-title {{
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .matrix-table, .runtime-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 14px;
    }}
    .matrix-table th, .matrix-table td,
    .runtime-table th, .runtime-table td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: center;
    }}
    .matrix-table th:first-child {{
      width: 140px;
      text-align: left;
      background: #faf5ec;
    }}
    .runtime-table th:first-child {{
      width: 96px;
    }}
    .matrix-table td {{
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    .runtime-panel {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .mode-summary-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .mode-summary {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 14px;
      color: var(--muted);
      background: #fffaf1;
    }}
    .mode-summary-title {{
      margin-bottom: 4px;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .mode-balanced {{
      background: #fbefe8;
    }}
    .mode-acc {{
      background: #edf6f3;
    }}
    .ccd-block {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #faf6ee;
      padding: 14px 14px 16px;
    }}
    .ccd-title {{
      font-size: 18px;
      font-weight: 700;
    }}
    .ccd-meta {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 14px;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin-top: 12px;
    }}
    .runtime-compare-table {{
      table-layout: auto;
      min-width: 1400px;
    }}
    .runtime-compare-table th,
    .runtime-compare-table td {{
      white-space: nowrap;
    }}
    .runtime-compare-table .core-head,
    .runtime-compare-table .core-cell {{
      position: sticky;
      left: 0;
      z-index: 2;
      background: #f8f1e5;
      box-shadow: 1px 0 0 0 var(--line), 10px 0 18px rgba(31, 26, 20, 0.06);
    }}
    .runtime-compare-table .core-head {{
      z-index: 3;
    }}
    .shared-head {{
      background: #f8f1e5;
    }}
    .balanced-head {{
      background: #fbefe8;
    }}
    .acc-head {{
      background: #edf6f3;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      background: #f5eee1;
      padding: 1px 5px;
      border-radius: 6px;
    }}
    @media (max-width: 980px) {{
      .facts, .grid {{
        grid-template-columns: 1fr;
      }}
      .mode-summary-grid {{
        grid-template-columns: 1fr;
      }}
      .span-2 {{
        grid-column: span 1;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">CPU Attention Locality</div>
      <h1>balanced vs acc-local-l3 对比报告</h1>
      <div class="subtitle">
        同时解析两份 <code>debug.log</code>，把 <code>CCD ↔ kv_head</code> 任务关系和
        <code>CCD/core ↔ kv_head ↔ q_tile</code> 的实际执行分布放到同一页。
      </div>
      <div class="hero-note">{html.escape(trace_note)}</div>
      <div class="logs">
        generated_at=<code>{html.escape(generated_at)}</code><br>
        balanced_log=<code>{html.escape(str(balanced_log))}</code><br>
        acc_log=<code>{html.escape(str(acc_log))}</code>
      </div>
    </section>
    {''.join(_render_rank_section(section) for section in rank_sections)}
  </div>
</body>
</html>
"""


def _parse_ranks(value: str | None) -> list[int] | None:
    if value is None:
        return None
    ranks: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        ranks.append(int(token))
    return ranks or None


def _resolve_case_mode_paths(case_dir: Path) -> tuple[Path, Path, Path]:
    case_dir = case_dir.resolve()
    balanced_log = case_dir / "balanced" / "debug.log"
    acc_candidates = [
        case_dir / "acc-local-l3" / "debug.log",
        case_dir / "acc" / "debug.log",
    ]
    acc_log = next((path for path in acc_candidates if path.exists()), None)

    missing: list[str] = []
    if not balanced_log.exists():
        missing.append(str(balanced_log))
    if acc_log is None:
        missing.extend(str(path) for path in acc_candidates)
    if missing:
        raise FileNotFoundError(
            "failed to resolve case logs; missing: " + ", ".join(missing)
        )
    return balanced_log, acc_log, case_dir / "attn_locality_compare.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a single HTML report comparing balanced vs acc-local-l3 "
            "for CCD/KV-head placement and CCD/core/q_tile execution."
        )
    )
    parser.add_argument(
        "case_dir",
        nargs="?",
        type=Path,
        help=(
            "Case directory containing balanced/debug.log and "
            "acc-local-l3/debug.log."
        ),
    )
    parser.add_argument("--balanced-log", type=Path, default=None)
    parser.add_argument("--acc-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--ranks",
        default=None,
        help="Comma-separated ranks to render. Defaults to trace ranks common to both logs.",
    )
    args = parser.parse_args(argv)

    using_case_dir = args.case_dir is not None
    using_manual_logs = args.balanced_log is not None or args.acc_log is not None
    if using_case_dir and using_manual_logs:
        parser.error("case_dir cannot be used together with --balanced-log/--acc-log")
    if using_manual_logs and (
        args.balanced_log is None or args.acc_log is None
    ):
        parser.error("--balanced-log and --acc-log must be provided together")
    if not using_case_dir and not using_manual_logs:
        parser.error(
            "pass either case_dir, or both --balanced-log and --acc-log"
        )
    return args


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.case_dir is not None:
        balanced_log, acc_log, default_output = _resolve_case_mode_paths(args.case_dir)
        output_path = args.output.resolve() if args.output else default_output
    else:
        assert args.balanced_log is not None
        assert args.acc_log is not None
        balanced_log = args.balanced_log.resolve()
        acc_log = args.acc_log.resolve()
        output_path = (
            args.output.resolve()
            if args.output
            else balanced_log.parent / "attn_locality_compare.html"
        )

    balanced_trace_ranks = collect_trace_ranks(balanced_log)
    acc_trace_ranks = collect_trace_ranks(acc_log)
    selected_ranks = _parse_ranks(args.ranks)
    if selected_ranks is None:
        selected_ranks = sorted(set(balanced_trace_ranks) & set(acc_trace_ranks))
        if not selected_ranks:
            raise ValueError(
                "no common trace ranks found between the two logs; pass --ranks explicitly if needed"
            )

    rank_sections = [
        build_rank_section(balanced_log, acc_log, rank=rank)
        for rank in selected_ranks
    ]
    summary_ranks = sorted(
        set(extract_summary_ranks(balanced_log)) | set(extract_summary_ranks(acc_log))
    )
    trace_note = (
        "仅展示同时存在 trace 的 rank: "
        + ", ".join(str(rank) for rank in selected_ranks)
    )
    hidden_ranks = [rank for rank in summary_ranks if rank not in selected_ranks]
    if hidden_ranks:
        trace_note += (
            "；日志 summary 还包含 rank: "
            + ", ".join(str(rank) for rank in hidden_ranks)
            + "，但这些 rank 没有同时出现在两份 trace 里。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(rank_sections, balanced_log, acc_log, trace_note),
        encoding="utf-8",
    )
    print(f"wrote html: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
