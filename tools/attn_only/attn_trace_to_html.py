from __future__ import annotations

import argparse
import json
import html
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

TRACE_LINE_RE = re.compile(
    r"^(?:rank=(?P<rank>-?\d+)\s+)?CPU attention trace\s+(?P<fields>.+)$"
)
TILE_PLAN_RE = re.compile(
    r"^\s+tile_plan:\s+q_tile_num=(?P<q_tile_num>\d+),\s+"
    r"default_q_tile_token_num=(?P<default_q_tile_token_num>\d+)\s*$"
)
Q_TILE_RE = re.compile(
    r"^\s+q_tile\s+(?P<q_tile_idx>\d+):\s+"
    r"q_range=\[(?P<q_start>\d+),(?P<q_end>\d+)\),\s+"
    r"kv_range=\[(?P<kv_start>\d+),(?P<kv_end>\d+)\),\s+"
    r"kv_tile_size=(?P<kv_tile_size>\d+),\s+"
    r"kv_tile_num=(?P<kv_tile_num>\d+)\s*$"
)

TABLE_COLUMNS = [
    ("line_no", "Line"),
    ("rank", "Rank"),
    ("mode", "Mode"),
    ("thread_id", "Thread"),
    ("core", "Core"),
    ("ccd", "CCD"),
    ("kv_head_idx", "KV Head"),
    ("req_id", "Req"),
    ("workitem_group_idx", "Workitem"),
    ("q_token_id_start", "Q Start"),
    ("q_token_num", "Q Num"),
    ("kv_split_pos_start", "KV Start"),
    ("kv_split_pos_end", "KV End"),
    ("split_id", "Split"),
    ("local_split_id", "Local Split"),
    ("task_idx", "Task"),
    ("thread_offset", "Thread Offset"),
    ("subgroup_id", "Subgroup"),
    ("legacy_thread_offset", "Legacy Offset"),
]

FILTER_KEYS = [
    ("rank", "Rank"),
    ("mode", "Mode"),
    ("thread_id", "Thread"),
    ("kv_head_idx", "KV Head"),
    ("req_id", "Req"),
]

TAG_COLORS = [
    "tag-a",
    "tag-b",
    "tag-c",
    "tag-d",
    "tag-e",
    "tag-f",
]


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _coerce_value(value: str) -> int | str:
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _expand_cpu_ids(cpu_ids: str) -> list[int]:
    expanded: list[int] = []
    for token in cpu_ids.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            expanded.extend(range(start, end + 1))
        else:
            expanded.append(int(token))
    return expanded


def parse_trace_line(line: str) -> dict[str, Any] | None:
    match = TRACE_LINE_RE.match(line.strip())
    if not match:
        return None

    record: dict[str, Any] = {}
    rank = match.group("rank")
    if rank is not None:
        record["rank"] = int(rank)

    fields = match.group("fields").split()
    for field in fields:
        if "=" not in field:
            continue
        key, value = field.split("=", maxsplit=1)
        record[key] = _coerce_value(value)
    return record


def _extract_summary_json(log_text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", log_text):
        try:
            candidate, _ = decoder.raw_decode(log_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "rank_results" in candidate:
            return candidate
    return None


def extract_rank_thread_to_core(log_text: str) -> dict[int, dict[int, int]]:
    summary = _extract_summary_json(log_text)
    if not summary:
        return {}

    mapping: dict[int, dict[int, int]] = {}
    for item in summary.get("rank_results", []):
        rank = item.get("rank")
        omp_cpuids = item.get("omp_cpuids")
        if rank is None or not omp_cpuids:
            continue
        cpu_ids = _expand_cpu_ids(str(omp_cpuids))
        mapping[int(rank)] = {
            thread_id: cpu_id for thread_id, cpu_id in enumerate(cpu_ids)
        }
    return mapping


def extract_rank_core_to_ccd(log_text: str) -> dict[int, dict[int, int]]:
    summary = _extract_summary_json(log_text)
    if not summary:
        return {}

    mapping: dict[int, dict[int, int]] = {}
    for item in summary.get("rank_results", []):
        rank = item.get("rank")
        locality_groups = item.get("locality_groups")
        if rank is None or not isinstance(locality_groups, list):
            continue
        rank_mapping: dict[int, int] = {}
        for group in locality_groups:
            if not isinstance(group, dict):
                continue
            l3_cache_id = group.get("l3_cache_id")
            cpu_ids = group.get("cpu_ids")
            if l3_cache_id is None or not isinstance(cpu_ids, list):
                continue
            for cpu_id in cpu_ids:
                rank_mapping[int(cpu_id)] = int(l3_cache_id)
        mapping[int(rank)] = rank_mapping
    return mapping


def build_request_kv_coverage(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    for record in records:
        if "rank" not in record or "mode" not in record:
            continue
        key = (
            int(record["rank"]),
            str(record["mode"]),
            int(record["req_id"]),
            int(record["kv_head_idx"]),
        )
        row = grouped.setdefault(
            key,
            {
                "rank": key[0],
                "mode": key[1],
                "req_id": key[2],
                "kv_head_idx": key[3],
                "threads": set(),
                "cores": set(),
                "ccds": set(),
                "task_rows": 0,
                "q_start_min": None,
                "q_end_max": None,
            },
        )
        row["threads"].add(int(record["thread_id"]))
        if "core" in record:
            row["cores"].add(int(record["core"]))
        if "ccd" in record:
            row["ccds"].add(int(record["ccd"]))
        row["task_rows"] += 1
        q_start = int(record["q_token_id_start"])
        q_end = q_start + int(record["q_token_num"])
        row["q_start_min"] = q_start if row["q_start_min"] is None else min(row["q_start_min"], q_start)
        row["q_end_max"] = q_end if row["q_end_max"] is None else max(row["q_end_max"], q_end)

    rows = []
    for row in grouped.values():
        rows.append(
            {
                **row,
                "threads": sorted(row["threads"]),
                "cores": sorted(row["cores"]),
                "ccds": sorted(row["ccds"]),
            }
        )
    rows.sort(
        key=lambda row: (
            row["rank"],
            row["req_id"],
            row["kv_head_idx"],
            row["mode"],
        )
    )
    return rows


def load_trace_records(log_path: str | Path) -> list[dict[str, Any]]:
    log_path = _as_path(log_path)
    log_text = log_path.read_text(encoding="utf-8")
    rank_thread_to_core = extract_rank_thread_to_core(log_text)
    rank_core_to_ccd = extract_rank_core_to_ccd(log_text)
    records: list[dict[str, Any]] = []
    trace_line_no = 0
    active_record: dict[str, Any] | None = None
    for raw_line in log_text.splitlines():
        record = parse_trace_line(raw_line)
        if record is None:
            if active_record is not None:
                tile_plan_match = TILE_PLAN_RE.match(raw_line)
                if tile_plan_match:
                    active_record["q_tile_num"] = int(
                        tile_plan_match.group("q_tile_num")
                    )
                    active_record["default_q_tile_token_num"] = int(
                        tile_plan_match.group("default_q_tile_token_num")
                    )
                    active_record.setdefault("q_tiles", [])
                    continue

                q_tile_match = Q_TILE_RE.match(raw_line)
                if q_tile_match:
                    active_record.setdefault("q_tiles", []).append(
                        {
                            "q_tile_idx": int(q_tile_match.group("q_tile_idx")),
                            "q_start": int(q_tile_match.group("q_start")),
                            "q_end": int(q_tile_match.group("q_end")),
                            "kv_start": int(q_tile_match.group("kv_start")),
                            "kv_end": int(q_tile_match.group("kv_end")),
                            "kv_tile_size": int(q_tile_match.group("kv_tile_size")),
                            "kv_tile_num": int(q_tile_match.group("kv_tile_num")),
                        }
                    )
                    continue
            continue
        record["line_no"] = trace_line_no
        record["raw_line"] = raw_line.rstrip("\n")
        rank = int(record.get("rank", -1))
        thread_id = int(record["thread_id"])
        core = rank_thread_to_core.get(rank, {}).get(thread_id)
        if core is not None:
            record["core"] = core
            ccd = rank_core_to_ccd.get(rank, {}).get(core)
            if ccd is not None:
                record["ccd"] = ccd
        records.append(record)
        active_record = record
        trace_line_no += 1
    return records


def _tag_class(value: int | str) -> str:
    return TAG_COLORS[hash(str(value)) % len(TAG_COLORS)]


def _render_tag(value: int | str, prefix: str) -> str:
    escaped = html.escape(str(value))
    return (
        f'<span class="tag {_tag_class(value)}" '
        f'title="{html.escape(prefix)}={escaped}">{html.escape(prefix)}={escaped}</span>'
    )


def _format_set(values: set[int | str], prefix: str) -> str:
    if not values:
        return '<span class="muted">none</span>'
    return " ".join(_render_tag(value, prefix) for value in sorted(values, key=str))


def _build_filter_options(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    options: dict[str, set[str]] = {key: set() for key, _ in FILTER_KEYS}
    for record in records:
        for key in options:
            if key in record:
                options[key].add(str(record[key]))
    return {key: sorted(values, key=lambda item: (len(item), item)) for key, values in options.items()}


def _build_thread_groups(
    records: list[dict[str, Any]]
) -> list[tuple[tuple[int, int], list[dict[str, Any]]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        rank = int(record.get("rank", -1))
        thread_id = int(record["thread_id"])
        grouped[(rank, thread_id)].append(record)

    return sorted(grouped.items(), key=lambda item: item[0])


def _render_filter_bar(records: list[dict[str, Any]]) -> str:
    options = _build_filter_options(records)
    parts = ['<section class="panel filters">']
    parts.append('<div class="panel-title">Filters</div>')
    parts.append('<div class="filter-grid">')
    for key, label in FILTER_KEYS:
        parts.append(f'<label class="filter-item" for="filter-{key}">{html.escape(label)}</label>')
        parts.append(f'<select id="filter-{key}" class="filter-select" data-filter-key="{key}">')
        parts.append('<option value="">All</option>')
        for value in options[key]:
            escaped = html.escape(value)
            parts.append(f'<option value="{escaped}">{escaped}</option>')
        parts.append('</select>')
    parts.append("</div>")
    parts.append(
        '<div class="filter-actions"><button type="button" id="reset-filters">Reset</button>'
        '<button type="button" id="expand-all">Expand All</button>'
        '<button type="button" id="collapse-all">Collapse All</button></div>'
    )
    parts.append("</section>")
    return "".join(parts)


def _render_summary(records: list[dict[str, Any]], source_log: Path) -> str:
    ranks = sorted({int(record.get("rank", -1)) for record in records})
    modes = sorted({str(record.get("mode", "")) for record in records if "mode" in record})
    threads = {int(record["thread_id"]) for record in records}
    kv_heads = {int(record["kv_head_idx"]) for record in records if "kv_head_idx" in record}
    req_ids = {int(record["req_id"]) for record in records if "req_id" in record}
    cores = {int(record["core"]) for record in records if "core" in record}
    ccds = {int(record["ccd"]) for record in records if "ccd" in record}

    cards = [
        ("Total Trace Rows", str(len(records))),
        ("Ranks", ", ".join(str(rank) for rank in ranks) if ranks else "none"),
        ("Threads", str(len(threads))),
        ("Cores", str(len(cores)) if cores else "n/a"),
        ("CCDs", str(len(ccds)) if ccds else "n/a"),
        ("Modes", ", ".join(modes) if modes else "none"),
        ("KV Heads", str(len(kv_heads))),
        ("Req IDs", str(len(req_ids))),
    ]
    parts = ['<section class="panel summary">']
    parts.append('<div class="panel-title">Summary</div>')
    parts.append(
        f'<div class="source-path">Source Log: '
        f'<code>{html.escape(str(source_log))}</code></div>'
    )
    parts.append('<div class="summary-grid">')
    for label, value in cards:
        parts.append(
            '<div class="summary-card">'
            f'<div class="summary-label">{html.escape(label)}</div>'
            f'<div class="summary-value">{html.escape(value)}</div>'
            "</div>"
        )
    parts.append("</div></section>")
    return "".join(parts)


def _row_data_attr(record: dict[str, Any], key: str) -> str:
    if key not in record:
        return ""
    return html.escape(str(record[key]))


def _render_thread_groups(records: list[dict[str, Any]]) -> str:
    groups = _build_thread_groups(records)
    if not groups:
        return (
            '<section class="panel empty-state"><div class="panel-title">Threads</div>'
            '<p>No attention trace rows found in the selected log.</p></section>'
        )

    parts = ['<section class="thread-groups">']
    for group_index, ((rank, thread_id), group_records) in enumerate(groups):
        modes = {str(record.get("mode", "")) for record in group_records if "mode" in record}
        kv_heads = {int(record["kv_head_idx"]) for record in group_records if "kv_head_idx" in record}
        req_ids = {int(record["req_id"]) for record in group_records if "req_id" in record}
        cores = {int(record["core"]) for record in group_records if "core" in record}
        ccds = {int(record["ccd"]) for record in group_records if "ccd" in record}
        open_attr = " open" if group_index < 4 else ""
        core_text = (
            ", ".join(str(core) for core in sorted(cores))
            if cores else "n/a"
        )
        ccd_text = (
            ", ".join(str(ccd) for ccd in sorted(ccds))
            if ccds else "n/a"
        )
        parts.append(
            f'<section class="thread-card" data-rank="{rank}" data-thread-id="{thread_id}">'
            f'<details{open_attr}><summary>'
            f'<span class="thread-title">Thread {thread_id}</span>'
            f'<span class="thread-meta">rank {rank} | core {html.escape(core_text)} | CCD {html.escape(ccd_text)} | '
            f'<span class="visible-count">{len(group_records)}</span> / {len(group_records)} rows</span>'
            "</summary>"
            '<div class="thread-summary">'
            f'<div><span class="label">Modes</span>{_format_set(modes, "mode")}</div>'
            f'<div><span class="label">Cores</span>{_format_set(cores, "core") if cores else "<span class=\"muted\">unavailable</span>"}</div>'
            f'<div><span class="label">CCDs</span>{_format_set(ccds, "CCD") if ccds else "<span class=\"muted\">unavailable</span>"}</div>'
            f'<div><span class="label">KV Heads</span>{_format_set(kv_heads, "kv_head")}</div>'
            f'<div><span class="label">Req IDs</span>{_format_set(req_ids, "req_id")}</div>'
            "</div>"
            '<div class="table-wrap"><table><thead><tr>'
        )
        for _, label in TABLE_COLUMNS:
            parts.append(f"<th>{html.escape(label)}</th>")
        parts.append("</tr></thead><tbody>")
        for record in group_records:
            summary_text = " ".join(
                f"{key}={record[key]}" for key, _ in TABLE_COLUMNS if key in record
            )
            parts.append(
                "<tr "
                f'data-rank="{_row_data_attr(record, "rank")}" '
                f'data-mode="{_row_data_attr(record, "mode")}" '
                f'data-thread-id="{_row_data_attr(record, "thread_id")}" '
                f'data-kv-head-idx="{_row_data_attr(record, "kv_head_idx")}" '
                f'data-req-id="{_row_data_attr(record, "req_id")}" '
                f'data-summary="{html.escape(summary_text)}">'
            )
            for key, _ in TABLE_COLUMNS:
                value = record.get(key, "")
                if key in {"mode"} and value != "":
                    cell = _render_tag(str(value), key)
                elif key in {"kv_head_idx", "req_id"} and value != "":
                    cell = _render_tag(value, key)
                else:
                    cell = html.escape(str(value))
                parts.append(f"<td>{cell}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div></details></section>")
    parts.append("</section>")
    return "".join(parts)


def _render_coverage_table(records: list[dict[str, Any]]) -> str:
    coverage_rows = build_request_kv_coverage(records)
    if not coverage_rows:
        return (
            '<section class="panel empty-state"><div class="panel-title">'
            'Request × KV Head Coverage</div><p>No coverage rows available.</p></section>'
        )

    parts = [
        '<section class="panel coverage"><div class="panel-title">'
        'Request × KV Head Coverage</div>',
        '<div class="table-wrap"><table><thead><tr>',
        '<th>Rank</th><th>Mode</th><th>Req</th><th>KV Head</th><th>Threads</th>'
        '<th>Cores</th><th>CCDs</th><th>Task Rows</th><th>Q Range</th>',
        '</tr></thead><tbody>',
    ]
    for row in coverage_rows:
        threads = " ".join(_render_tag(thread_id, "thread") for thread_id in row["threads"])
        cores = (
            " ".join(_render_tag(core, "core") for core in row["cores"])
            if row["cores"]
            else '<span class="muted">unavailable</span>'
        )
        ccds = (
            " ".join(_render_tag(ccd, "CCD") for ccd in row["ccds"])
            if row["ccds"]
            else '<span class="muted">unavailable</span>'
        )
        q_range = f'{row["q_start_min"]} -> {row["q_end_max"]}'
        parts.append("<tr>")
        parts.append(f'<td>{row["rank"]}</td>')
        parts.append(f'<td>{_render_tag(str(row["mode"]), "mode")}</td>')
        parts.append(f'<td>{_render_tag(row["req_id"], "req_id")}</td>')
        parts.append(f'<td>{_render_tag(row["kv_head_idx"], "kv_head_idx")}</td>')
        parts.append(f"<td>{threads}</td>")
        parts.append(f"<td>{cores}</td>")
        parts.append(f"<td>{ccds}</td>")
        parts.append(f'<td>{row["task_rows"]}</td>')
        parts.append(f"<td>{html.escape(q_range)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div></section>")
    return "".join(parts)


def render_html(records: list[dict[str, Any]], source_log: str | Path) -> str:
    source_log = _as_path(source_log)
    generated_at = datetime.now().isoformat(timespec="seconds")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attention Trace Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f1e8;
      --panel: #fffaf0;
      --panel-border: #d9ccb8;
      --text: #1f2933;
      --muted: #6b7280;
      --accent: #a44a3f;
      --accent-soft: #f2ddd8;
      --table-stripe: #fcf7ee;
      --table-border: #eadfce;
      --shadow: rgba(41, 33, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background: linear-gradient(180deg, #efe3d0 0%, var(--bg) 28%, #f8f4ed 100%);
      color: var(--text);
    }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0.02em; }}
    .subtitle {{ color: var(--muted); margin-bottom: 24px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      box-shadow: 0 10px 24px var(--shadow);
      padding: 18px 20px;
      margin-bottom: 18px;
    }}
    .panel-title {{ font-size: 18px; font-weight: 700; margin-bottom: 12px; }}
    .source-path {{ color: var(--muted); margin-bottom: 12px; }}
    code {{
      background: #f7ead8;
      padding: 2px 6px;
      border-radius: 6px;
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .summary-card {{
      border: 1px solid var(--table-border);
      border-radius: 12px;
      background: #fff;
      padding: 12px 14px;
    }}
    .summary-label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .summary-value {{ font-size: 24px; margin-top: 6px; font-weight: 700; }}
    .filter-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px 12px;
      align-items: center;
    }}
    .filter-item {{ color: var(--muted); font-size: 13px; }}
    .filter-select, button {{
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 8px 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }}
    button {{
      cursor: pointer;
      transition: background 120ms ease, transform 120ms ease;
    }}
    button:hover {{ background: var(--accent-soft); transform: translateY(-1px); }}
    .filter-actions {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
    .thread-groups {{ display: grid; gap: 14px; }}
    .thread-card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      box-shadow: 0 8px 20px var(--shadow);
      overflow: hidden;
    }}
    details > summary {{
      list-style: none;
      cursor: pointer;
      padding: 16px 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: space-between;
      align-items: baseline;
      background: linear-gradient(90deg, #fff7ee 0%, #f5e8d8 100%);
      border-bottom: 1px solid var(--table-border);
    }}
    details > summary::-webkit-details-marker {{ display: none; }}
    .thread-title {{ font-size: 20px; font-weight: 700; }}
    .thread-meta {{ color: var(--muted); font-size: 14px; }}
    .thread-summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      padding: 14px 18px 8px;
    }}
    .label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
      letter-spacing: 0.06em;
    }}
    .tag {{
      display: inline-block;
      margin: 0 6px 6px 0;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
    }}
    .tag-a {{ background: #f6d8cb; color: #723429; }}
    .tag-b {{ background: #dcead6; color: #36552f; }}
    .tag-c {{ background: #d7e7f6; color: #244a70; }}
    .tag-d {{ background: #f2e0b8; color: #6b4d18; }}
    .tag-e {{ background: #ead9ef; color: #69417a; }}
    .tag-f {{ background: #d6eceb; color: #235758; }}
    .muted {{ color: var(--muted); }}
    .table-wrap {{ padding: 8px 18px 18px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1300px; }}
    th, td {{
      border-bottom: 1px solid var(--table-border);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f7ead8;
      z-index: 1;
    }}
    tbody tr:nth-child(odd) {{ background: var(--table-stripe); }}
    .empty-state p {{ margin: 0; color: var(--muted); }}
    @media (max-width: 800px) {{
      main {{ padding: 16px; }}
      h1 {{ font-size: 28px; }}
      details > summary {{ padding: 14px; }}
      .table-wrap {{ padding: 6px 14px 14px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Attention Trace Viewer</h1>
    <div class="subtitle">Generated at {html.escape(generated_at)}. Grouped by real runtime thread.</div>
    {_render_summary(records, source_log)}
    {_render_coverage_table(records)}
    {_render_filter_bar(records)}
    {_render_thread_groups(records)}
  </main>
  <script>
    const filterSelects = Array.from(document.querySelectorAll('.filter-select'));
    const rows = Array.from(document.querySelectorAll('tbody tr'));
    const cards = Array.from(document.querySelectorAll('.thread-card'));

    function matchesFilter(row, filters) {{
      if (filters.rank && row.dataset.rank !== filters.rank) return false;
      if (filters.mode && row.dataset.mode !== filters.mode) return false;
      if (filters.thread_id && row.dataset.threadId !== filters.thread_id) return false;
      if (filters.kv_head_idx && row.dataset.kvHeadIdx !== filters.kv_head_idx) return false;
      if (filters.req_id && row.dataset.reqId !== filters.req_id) return false;
      return true;
    }}

    function applyFilters() {{
      const filters = {{}};
      for (const select of filterSelects) {{
        filters[select.dataset.filterKey] = select.value;
      }}
      for (const card of cards) {{
        const cardRows = Array.from(card.querySelectorAll('tbody tr'));
        let visibleRows = 0;
        for (const row of cardRows) {{
          const visible = matchesFilter(row, filters);
          row.style.display = visible ? '' : 'none';
          if (visible) visibleRows += 1;
        }}
        card.style.display = visibleRows > 0 ? '' : 'none';
        const counter = card.querySelector('.visible-count');
        if (counter) {{
          counter.textContent = String(visibleRows);
        }}
      }}
    }}

    for (const select of filterSelects) {{
      select.addEventListener('change', applyFilters);
    }}
    document.getElementById('reset-filters').addEventListener('click', () => {{
      for (const select of filterSelects) {{
        select.value = '';
      }}
      applyFilters();
    }});
    document.getElementById('expand-all').addEventListener('click', () => {{
      for (const details of document.querySelectorAll('.thread-card details')) {{
        details.open = true;
      }}
    }});
    document.getElementById('collapse-all').addEventListener('click', () => {{
      for (const details of document.querySelectorAll('.thread-card details')) {{
        details.open = false;
      }}
    }});
    applyFilters();
  </script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CPU attention trace log into a grouped HTML viewer."
    )
    parser.add_argument("--log", required=True, type=Path, help="Path to trace log file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <log>.html.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    log_path = args.log.resolve()
    output_path = args.output.resolve() if args.output else log_path.with_suffix(".html")
    records = load_trace_records(log_path)
    output_path.write_text(render_html(records, log_path), encoding="utf-8")
    print(f"wrote html: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
