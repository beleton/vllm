from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tools.p3_attn_only.attn_trace_to_html import load_trace_records


@dataclass(frozen=True)
class L3Group:
    l3_cache_id: int
    cpu_ids: list[int]

    @property
    def label(self) -> str:
        if not self.cpu_ids:
            return f"l3_{self.l3_cache_id}"
        return f"l3_{self.l3_cache_id} ({self.cpu_ids[0]}-{self.cpu_ids[-1]})"


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    rank: int
    source_log: Path
    trace_rows: int
    kv_heads: list[int]
    l3_groups: list[L3Group]
    matrix: list[list[int]]
    kv_span_counts: dict[int, int]
    l3_kv_counts: dict[int, int]


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


def _extract_rank_l3_groups(log_text: str) -> dict[int, list[L3Group]]:
    summary = _extract_summary_json(log_text)
    if not summary:
        return {}

    result: dict[int, list[L3Group]] = {}
    for item in summary.get("rank_results", []):
        rank = item.get("rank")
        locality_groups = item.get("locality_groups")
        if rank is None or not isinstance(locality_groups, list):
            continue
        groups: list[L3Group] = []
        for group in locality_groups:
            if not isinstance(group, dict):
                continue
            l3_cache_id = group.get("l3_cache_id")
            cpu_ids = group.get("cpu_ids")
            if l3_cache_id is None or not isinstance(cpu_ids, list):
                continue
            groups.append(
                L3Group(
                    l3_cache_id=int(l3_cache_id),
                    cpu_ids=[int(cpu_id) for cpu_id in cpu_ids],
                )
            )
        groups.sort(key=lambda group: group.l3_cache_id)
        result[int(rank)] = groups
    return result


def build_mode_summary(log_path: str | Path, rank: int = 0) -> ModeSummary:
    source_log = Path(log_path).resolve()
    log_text = source_log.read_text(encoding="utf-8")
    records = [
        record for record in load_trace_records(source_log)
        if int(record.get("rank", -1)) == rank
    ]
    if not records:
        raise ValueError(f"no trace rows for rank={rank}: {source_log}")

    mode = str(records[0].get("mode", "unknown"))
    kv_heads = sorted({int(record["kv_head_idx"]) for record in records})
    l3_groups = _extract_rank_l3_groups(log_text).get(rank, [])
    if not l3_groups:
        raise ValueError(f"no locality_groups for rank={rank}: {source_log}")

    l3_ids = [group.l3_cache_id for group in l3_groups]
    matrix: list[list[int]] = []
    kv_span_counts: dict[int, int] = {}
    l3_kv_counts: dict[int, int] = {}

    for l3_id in l3_ids:
        row: list[int] = []
        for kv_head in kv_heads:
            count = sum(
                1
                for record in records
                if int(record.get("ccd", -1)) == l3_id
                and int(record["kv_head_idx"]) == kv_head
            )
            row.append(count)
        matrix.append(row)

    for col_idx, kv_head in enumerate(kv_heads):
        kv_span_counts[kv_head] = sum(1 for row in matrix if row[col_idx] > 0)

    for row_idx, l3_id in enumerate(l3_ids):
        l3_kv_counts[l3_id] = sum(1 for count in matrix[row_idx] if count > 0)

    return ModeSummary(
        mode=mode,
        rank=rank,
        source_log=source_log,
        trace_rows=len(records),
        kv_heads=kv_heads,
        l3_groups=l3_groups,
        matrix=matrix,
        kv_span_counts=kv_span_counts,
        l3_kv_counts=l3_kv_counts,
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
        parts.append(f"<tr><th>{html.escape(group.label)}</th>")
        for value in row:
            parts.append(
                f'<td style="background:{_heat_color(value, max_value)}">{value}</td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _render_bar_table(
    title: str,
    labels: list[str],
    balanced_values: list[int],
    acc_values: list[int],
) -> str:
    max_value = max(balanced_values + acc_values) if (balanced_values or acc_values) else 0
    parts = [
        '<div class="panel span-2">',
        f'<div class="panel-title">{html.escape(title)}</div>',
        '<table class="bar-table"><thead><tr><th>Item</th><th>balanced</th><th>acc-local-l3</th></tr></thead><tbody>',
    ]
    for label, balanced_value, acc_value in zip(labels, balanced_values, acc_values, strict=True):
        balanced_width = 0 if max_value == 0 else balanced_value / max_value * 100
        acc_width = 0 if max_value == 0 else acc_value / max_value * 100
        parts.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f'<td><div class="bar"><div class="bar-fill balanced" style="width:{balanced_width:.1f}%"></div><span>{balanced_value}</span></div></td>'
            f'<td><div class="bar"><div class="bar-fill acc" style="width:{acc_width:.1f}%"></div><span>{acc_value}</span></div></td>'
            "</tr>"
        )
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
        f'<div class="fact-sub">每个 CCD 只计算 {acc_max_l3_load} 个 kv_head</div>'
        "</div>"
        "</div>"
    )


def render_html(balanced: ModeSummary, acc: ModeSummary) -> str:
    kv_labels = [f"kv{kv_head}" for kv_head in balanced.kv_heads]
    l3_labels = [f"l3_{group.l3_cache_id}" for group in balanced.l3_groups]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>balanced vs acc-local-l3 CCD Mapping</title>
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
      max-width: 1480px;
      margin: 0 auto;
      padding: 32px 28px 48px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
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
    .subtitle {{
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
    }}
    .facts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 22px;
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
      margin-top: 20px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px 18px 20px;
      box-shadow: var(--shadow);
    }}
    .span-2 {{
      grid-column: span 2;
    }}
    .panel-title {{
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 14px;
    }}
    .matrix-table, .bar-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 14px;
    }}
    .matrix-table th, .matrix-table td, .bar-table th, .bar-table td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: center;
    }}
    .matrix-table th:first-child, .bar-table th:first-child {{
      width: 180px;
      text-align: left;
      background: #faf5ec;
    }}
    .matrix-table td {{
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    .bar {{
      position: relative;
      height: 28px;
      background: #efe8dd;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      position: absolute;
      top: 0;
      left: 0;
      bottom: 0;
    }}
    .bar-fill.balanced {{ background: var(--balanced); }}
    .bar-fill.acc {{ background: var(--acc); }}
    .bar span {{
      position: relative;
      z-index: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      height: 100%;
      font-weight: 700;
      color: #17110c;
    }}
    .notes {{
      margin-top: 20px;
      background: #1f1a14;
      color: #f7f1e8;
      border-radius: 22px;
      padding: 22px 24px;
    }}
    .notes h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .notes ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.7;
    }}
    .logs {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    @media (max-width: 980px) {{
      .facts, .grid {{ grid-template-columns: 1fr; }}
      .span-2 {{ grid-column: span 1; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">CPU Attention Mapping</div>
      <h1>CCD × KV Head 对照图</h1>
      <div class="subtitle">
        口径：rank {balanced.rank}，由日志中的 trace + omp_cpuids + locality_groups 还原。<br>
        这个页面用于说明：balanced 会让一个 kv_head 分散到多个 CCD，且一个 CCD 同时计算多个 kv_head；acc-local-l3 则收敛到一对一映射。
      </div>
      {_render_summary_facts(balanced, acc)}
      <div class="logs">
        generated_at={html.escape(generated_at)}<br>
        balanced_log=<code>{html.escape(str(balanced.source_log))}</code><br>
        acc_log=<code>{html.escape(str(acc.source_log))}</code>
      </div>
    </section>

    <section class="grid">
      <div class="panel span-2">
        <div class="panel-title">CCD × KV Head 任务计数矩阵</div>
        <div class="subtitle">单元格数字表示该 CCD 上命中的该 kv_head trace 行数。颜色越深，表示该 CCD 在该 kv_head 上承担的任务越多。</div>
      </div>
      {_render_matrix(balanced)}
      {_render_matrix(acc)}
      {_render_bar_table("KV Head 跨 CCD 数", kv_labels, [balanced.kv_span_counts[kv] for kv in balanced.kv_heads], [acc.kv_span_counts[kv] for kv in acc.kv_heads])}
      {_render_bar_table("CCD 同时覆盖的 KV Head 数", l3_labels, [balanced.l3_kv_counts[group.l3_cache_id] for group in balanced.l3_groups], [acc.l3_kv_counts[group.l3_cache_id] for group in acc.l3_groups])}
    </section>

    <section class="notes">
      <h2>口述要点</h2>
      <ul>
        <li>balanced 下，单个 kv_head 最多横跨 {max(balanced.kv_span_counts.values())} 个 CCD；acc-local-l3 下，每个 kv_head 都只落在 1 个 CCD。</li>
        <li>balanced 下，单个 CCD 最多同时计算 {max(balanced.l3_kv_counts.values())} 个 kv_head；acc-local-l3 下，每个 CCD 只计算 1 个 kv_head。</li>
        <li>这张图展示的是任务放置关系，不是性能结论本身；它只回答“谁在算哪个 kv_head，以及是否跨 CCD 扩散”。</li>
      </ul>
    </section>
  </div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side HTML comparing balanced vs acc-local-l3 CCD mapping."
    )
    parser.add_argument("--balanced-log", required=True, help="Path to balanced log.")
    parser.add_argument("--acc-log", required=True, help="Path to acc-local-l3 log.")
    parser.add_argument("--rank", type=int, default=0, help="Rank to visualize.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    args = parser.parse_args(argv)

    balanced = build_mode_summary(args.balanced_log, rank=args.rank)
    acc = build_mode_summary(args.acc_log, rank=args.rank)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(balanced, acc), encoding="utf-8")
    print(f"wrote html: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
