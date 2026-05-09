from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_RESULT_ROOT = Path(
    "test_results/P4_AttnOnly_L3Residency/Qwen3-30B-A3B/"
    "qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1/span1/occupancy"
)
DEFAULT_OUTPUT_NAME = "occupancy_compare.html"
DEFAULT_BASELINE_MASK = "ffff"
DEFAULT_COMPARE_MASK = "0001"
MB = 1_000_000.0
DOMAIN_COUNT = 16
DOMAIN_LABELS = [f"L3_{idx:02d}" for idx in range(DOMAIN_COUNT)] + ["Total"]
MODE_LABELS = {
    "balanced": "balanced",
    "acc-local-l3": "acc-local-l3",
}
MASK_LABELS = {
    DEFAULT_BASELINE_MASK: "ffff",
    DEFAULT_COMPARE_MASK: "0001",
}


@dataclass(frozen=True)
class DomainOccupancy:
    label: str
    mean_mb: float | None
    peak_mb: float | None


@dataclass(frozen=True)
class CaseOccupancy:
    workload: str | None
    batch_size: int | None
    q_len: int | None
    kv_len: int | None
    partition_mode: str | None
    mode: str | None
    mask: str
    group_span: int | None
    slowest_rank_mean_ms: float | None
    case_dir: Path
    domains: list[DomainOccupancy]
    total_mean_mb: float | None
    total_peak_mb: float | None


@dataclass(frozen=True)
class ChartSeries:
    label: str
    values: list[float | None]
    color: str


@dataclass(frozen=True)
class CaseKey:
    q_len: int | None
    mode: str | None
    mask: str


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", "--", "N/A", "nan"}:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _safe_div(numerator: float | None,
              denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _latest_path(paths: list[Path]) -> Path | None:
    latest: Path | None = None
    for path in paths:
        if latest is None:
            latest = path
            continue
        candidate_key = (path.stat().st_mtime_ns, path.parent.name)
        latest_key = (latest.stat().st_mtime_ns, latest.parent.name)
        if candidate_key > latest_key:
            latest = path
    return latest


def _load_occupancy_summary(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved_lengths = data.get("resolved_lengths") or {}
    requested_lengths = data.get("requested_lengths") or {}
    head_plan = data.get("head_plan") or {}
    return {
        "workload": data.get("workload"),
        "batch_size": data.get("batch_size"),
        "q_len": resolved_lengths.get("q_len") or requested_lengths.get("q_len"),
        "kv_len": resolved_lengths.get("kv_len") or requested_lengths.get("kv_len"),
        "partition_mode": head_plan.get("partition_mode"),
        "mode": data.get("attn_locality_mode"),
        "group_span": data.get("attn_locality_group_span"),
        "slowest_rank_mean_ms": _to_float(data.get("slowest_rank_mean_ms")),
    }


def _load_occupancy_csv(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {
            "domains": [DomainOccupancy(label=f"L3_{idx:02d}",
                                        mean_mb=None,
                                        peak_mb=None)
                        for idx in range(DOMAIN_COUNT)],
            "total_mean_mb": None,
            "total_peak_mb": None,
        }

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                rows.append(row)

    if not rows:
        return {
            "domains": [DomainOccupancy(label=f"L3_{idx:02d}",
                                        mean_mb=None,
                                        peak_mb=None)
                        for idx in range(DOMAIN_COUNT)],
            "total_mean_mb": None,
            "total_peak_mb": None,
        }

    total_by_ts: dict[str, float] = {}
    domain_sum_by_name: dict[str, float] = {}
    domain_count_by_name: dict[str, int] = {}
    domain_peak_by_name: dict[str, float] = {}

    for row in rows:
        ts_ms = str(row.get("ts_ms", "")).strip()
        domain = str(row.get("domain", "")).strip()
        occupancy_bytes = _to_float(row.get("llc_occupancy_bytes")) or 0.0
        total_by_ts[ts_ms] = total_by_ts.get(ts_ms, 0.0) + occupancy_bytes
        domain_sum_by_name[domain] = domain_sum_by_name.get(domain, 0.0) + occupancy_bytes
        domain_count_by_name[domain] = domain_count_by_name.get(domain, 0) + 1
        domain_peak_by_name[domain] = max(domain_peak_by_name.get(domain, 0.0),
                                          occupancy_bytes)

    total_series = [total_by_ts[key] for key in sorted(total_by_ts)]
    total_mean_mb = _safe_div(mean(total_series) if total_series else None, MB)
    total_peak_mb = _safe_div(max(total_series) if total_series else None, MB)

    domains: list[DomainOccupancy] = []
    for idx in range(DOMAIN_COUNT):
        domain_key = f"mon_L3_{idx:02d}"
        total_bytes = domain_sum_by_name.get(domain_key)
        count = domain_count_by_name.get(domain_key)
        peak_bytes = domain_peak_by_name.get(domain_key)
        mean_bytes = _safe_div(total_bytes, count) if total_bytes is not None and count else None
        domains.append(
            DomainOccupancy(
                label=f"L3_{idx:02d}",
                mean_mb=_safe_div(mean_bytes, MB),
                peak_mb=_safe_div(peak_bytes, MB) if peak_bytes is not None else None,
            )
        )

    return {
        "domains": domains,
        "total_mean_mb": total_mean_mb,
        "total_peak_mb": total_peak_mb,
    }


def _load_case(case_dir: Path, mask: str) -> CaseOccupancy:
    summary = _load_occupancy_summary(case_dir / "occupancy_summary.json")
    occupancy = _load_occupancy_csv(case_dir / "llc_occupancy.csv")
    return CaseOccupancy(
        workload=summary["workload"],
        batch_size=summary["batch_size"],
        q_len=summary["q_len"],
        kv_len=summary["kv_len"],
        partition_mode=summary["partition_mode"],
        mode=summary["mode"],
        mask=mask,
        group_span=summary["group_span"],
        slowest_rank_mean_ms=summary["slowest_rank_mean_ms"],
        case_dir=case_dir.resolve(),
        domains=occupancy["domains"],
        total_mean_mb=occupancy["total_mean_mb"],
        total_peak_mb=occupancy["total_peak_mb"],
    )


def collect_case_index(result_root: str | Path,
                       *,
                       masks: tuple[str, str] = (DEFAULT_BASELINE_MASK,
                                                 DEFAULT_COMPARE_MASK)) -> dict[CaseKey, CaseOccupancy]:
    result_root = Path(result_root)
    grouped: dict[CaseKey, list[Path]] = {}

    for summary_path in result_root.glob("q*_kv*/**/occupancy_summary.json"):
        case_dir = summary_path.parent
        mask = case_dir.parent.name
        if mask not in masks:
            continue
        summary = _load_occupancy_summary(summary_path)
        key = CaseKey(
            q_len=summary["q_len"],
            mode=summary["mode"],
            mask=mask,
        )
        grouped.setdefault(key, []).append(case_dir)

    case_index: dict[CaseKey, CaseOccupancy] = {}
    for key, case_dirs in grouped.items():
        latest_dir = _latest_path(case_dirs)
        if latest_dir is None:
            continue
        case_index[key] = _load_case(latest_dir, key.mask)
    return case_index


def _case_for(case_index: dict[CaseKey, CaseOccupancy],
              q_len: int | None,
              mode: str,
              mask: str) -> CaseOccupancy | None:
    return case_index.get(CaseKey(q_len=q_len, mode=mode, mask=mask))


def _q_labels(lengths: list[int | None]) -> list[str]:
    labels: list[str] = []
    for length in lengths:
        labels.append(str(length) if length is not None else "na")
    return labels


def _build_total_series(case_index: dict[CaseKey, CaseOccupancy],
                       lengths: list[int | None],
                       mask: str) -> list[ChartSeries]:
    series: list[ChartSeries] = []
    for mode in ("balanced", "acc-local-l3"):
        cases = [_case_for(case_index, q_len, mode, mask) for q_len in lengths]
        series.extend([
            ChartSeries(
                label=f"{MODE_LABELS[mode]} mean",
                values=[case.total_mean_mb if case else None for case in cases],
                color="#1f77b4" if mode == "balanced" else "#d62728",
            ),
            ChartSeries(
                label=f"{MODE_LABELS[mode]} peak",
                values=[case.total_peak_mb if case else None for case in cases],
                color="#79aeea" if mode == "balanced" else "#f28e8e",
            ),
        ])
    return series


def _build_runtime_series(case_index: dict[CaseKey, CaseOccupancy],
                          lengths: list[int | None]) -> list[ChartSeries]:
    palette = {
        "balanced-ffff": "#1f77b4",
        "balanced-0001": "#79aeea",
        "acc-local-l3-ffff": "#d62728",
        "acc-local-l3-0001": "#f28e8e",
    }
    series: list[ChartSeries] = []
    for mode in ("balanced", "acc-local-l3"):
        for mask in (DEFAULT_BASELINE_MASK, DEFAULT_COMPARE_MASK):
            values: list[float | None] = []
            for q_len in lengths:
                case = _case_for(case_index, q_len, mode, mask)
                values.append(case.slowest_rank_mean_ms if case else None)
            series.append(
                ChartSeries(
                    label=f"{MODE_LABELS[mode]}-{mask}",
                    values=values,
                    color=palette[f"{mode}-{mask}"],
                )
            )
    return series


def _split_runtime_lengths(lengths: list[int]) -> list[tuple[str, list[int]]]:
    short_lengths = [length for length in lengths if length <= 8192]
    long_lengths = [length for length in lengths if length >= 16384]
    groups: list[tuple[str, list[int]]] = []
    if short_lengths:
        groups.append(("slowest_rank_mean_ms q1024-q8192", short_lengths))
    if long_lengths:
        groups.append(("slowest_rank_mean_ms q16384-q65536", long_lengths))
    return groups


def _build_occupancy_series_pair(balanced_case: CaseOccupancy | None,
                                 acc_case: CaseOccupancy | None) -> list[ChartSeries]:
    balanced_mean = ([domain.mean_mb for domain in balanced_case.domains] +
                     [balanced_case.total_mean_mb]
                     if balanced_case else [None] * len(DOMAIN_LABELS))
    balanced_peak = ([domain.peak_mb for domain in balanced_case.domains] +
                     [balanced_case.total_peak_mb]
                     if balanced_case else [None] * len(DOMAIN_LABELS))
    acc_mean = ([domain.mean_mb for domain in acc_case.domains] +
                [acc_case.total_mean_mb]
                if acc_case else [None] * len(DOMAIN_LABELS))
    acc_peak = ([domain.peak_mb for domain in acc_case.domains] +
                [acc_case.total_peak_mb]
                if acc_case else [None] * len(DOMAIN_LABELS))
    return [
        ChartSeries(
            label="balanced mean",
            values=balanced_mean,
            color="#1f77b4",
        ),
        ChartSeries(
            label="balanced peak",
            values=balanced_peak,
            color="#79aeea",
        ),
        ChartSeries(
            label="acc-local-l3 mean",
            values=acc_mean,
            color="#d62728",
        ),
        ChartSeries(
            label="acc-local-l3 peak",
            values=acc_peak,
            color="#f28e8e",
        ),
    ]


def _render_grouped_bar_svg(title: str,
                            categories: list[str],
                            series: list[ChartSeries],
                            *,
                            y_label: str,
                            width: int | None = None,
                            height: int = 420) -> str:
    chart_width = width or max(1000, 120 + len(categories) * 90)
    left = 78
    right = 24
    top = 50
    bottom = 86
    plot_width = chart_width - left - right
    plot_height = height - top - bottom
    series_count = max(1, len(series))
    slot_width = plot_width / max(1, len(categories))
    group_width = min(slot_width * 0.82, 64 + series_count * 10)
    bar_gap = 3
    bar_width = max(6.0, (group_width - bar_gap * (series_count - 1)) / series_count)
    group_left_offset = (slot_width - (bar_width * series_count + bar_gap * (series_count - 1))) / 2

    all_values = [
        value
        for series_item in series
        for value in series_item.values
        if value is not None
    ]
    y_max = max(all_values) if all_values else 1.0
    y_limit = y_max * 1.15 if y_max > 0 else 1.0

    def y_pos(value: float) -> float:
        return top + plot_height - (value / y_limit) * plot_height

    def esc(text: str) -> str:
        return html.escape(text, quote=True)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {chart_width} {height}" width="100%" height="{height}" role="img" aria-label="{esc(title)}">',
        '<style>',
        '.axis { stroke: #444; stroke-width: 1; }',
        '.grid { stroke: #e5e5e5; stroke-width: 1; }',
        '.tick { fill: #555; font-size: 11px; }',
        '.xlabel { fill: #222; font-size: 11px; }',
        '.title { fill: #111; font-size: 18px; font-weight: 700; }',
        '.ylabel { fill: #444; font-size: 12px; }',
        '.legend { fill: #222; font-size: 12px; }',
        '</style>',
        f'<text class="title" x="{chart_width / 2}" y="24" text-anchor="middle">{esc(title)}</text>',
        f'<text class="ylabel" x="20" y="{top + 6}" transform="rotate(-90 20 {top + 6})">{esc(y_label)}</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" />',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" />',
    ]

    for tick_index in range(5):
        ratio = tick_index / 4
        y = top + plot_height - ratio * plot_height
        value = y_limit * ratio
        label = f"{value:.0f}" if y_limit >= 100 else f"{value:.1f}"
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" />')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')

    legend_x = chart_width - 20
    legend_y = 16
    for idx, series_item in enumerate(series):
        x = legend_x - (idx % 2) * 170
        y = legend_y + (idx // 2) * 18
        parts.append(
            f'<rect x="{x - 14}" y="{y - 11}" width="12" height="12" fill="{series_item.color}" />'
        )
        parts.append(f'<text class="legend" x="{x + 4}" y="{y}" text-anchor="end">{esc(series_item.label)}</text>')

    for cat_index, category in enumerate(categories):
        group_x = left + cat_index * slot_width + group_left_offset
        center_x = left + cat_index * slot_width + slot_width / 2
        parts.append(
            f'<text class="xlabel" x="{center_x}" y="{top + plot_height + 26}" text-anchor="middle" '
            f'transform="rotate(45 {center_x} {top + plot_height + 26})">{esc(category)}</text>'
        )
        for series_index, series_item in enumerate(series):
            value = series_item.values[cat_index] if cat_index < len(series_item.values) else None
            if value is None:
                continue
            bar_x = group_x + series_index * (bar_width + bar_gap)
            bar_y = y_pos(value)
            bar_height = top + plot_height - bar_y
            parts.append(
                f'<g><title>{esc(series_item.label)} {esc(category)}: {value:.3f} {esc(y_label)}</title>'
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{series_item.color}" /></g>'
            )

    parts.append('</svg>')
    return ''.join(parts)


def _render_mask_card(title: str,
                      svg: str) -> str:
    return (
        '<div class="chart-card">'
        f'<div class="chart-card-title">{html.escape(title)}</div>'
        f'<div class="chart-scroll">{svg}</div>'
        '</div>'
    )


def _render_runtime_section(case_index: dict[CaseKey, CaseOccupancy],
                            lengths: list[int | None]) -> str:
    cards = []
    for title, sub_lengths in _split_runtime_lengths([int(length) for length in lengths if length is not None]):
        svg = _render_grouped_bar_svg(
            title,
            _q_labels(sub_lengths),
            _build_runtime_series(case_index, sub_lengths),
            y_label="ms",
            height=420,
        )
        cards.append(_render_mask_card(title, svg))
    return (
        '<section class="section">'
        '<h2>Total runtime compare</h2>'
        '<p class="section-note">balanced/acc-local-l3 together, ffff and 0001 together.</p>'
        f'<div class="chart-grid">{"".join(cards)}</div>'
        '</section>'
    )


def _render_total_section(case_index: dict[CaseKey, CaseOccupancy],
                          lengths: list[int | None]) -> str:
    cards = []
    for mask in (DEFAULT_BASELINE_MASK, DEFAULT_COMPARE_MASK):
        svg = _render_grouped_bar_svg(
            f"Total occupancy across lengths ({mask})",
            _q_labels(lengths),
            _build_total_series(case_index, lengths, mask),
            y_label="MB",
            height=420,
        )
        cards.append(_render_mask_card(mask, svg))
    return (
        '<section class="section">'
        '<h2>Total occupancy across lengths</h2>'
        '<p class="section-note">Total mean and peak in MB, grouped by balanced and acc-local-l3.</p>'
        f'<div class="chart-grid">{"".join(cards)}</div>'
        '</section>'
    )


def _render_length_section(case_index: dict[CaseKey, CaseOccupancy],
                           q_len: int) -> str:
    title = f"q{q_len}_kv{q_len}"
    cards = []
    for mask in (DEFAULT_BASELINE_MASK, DEFAULT_COMPARE_MASK):
        balanced_case = _case_for(case_index, q_len, "balanced", mask)
        acc_case = _case_for(case_index, q_len, "acc-local-l3", mask)
        if balanced_case is None and acc_case is None:
            continue
        svg = _render_grouped_bar_svg(
            f"{title} {mask}",
            DOMAIN_LABELS,
            _build_occupancy_series_pair(balanced_case, acc_case),
            y_label="MB",
            height=430,
        )
        cards.append(_render_mask_card(mask, svg))
    return (
        '<details class="section" open>'
        f'<summary>{html.escape(title)}</summary>'
        '<p class="section-note">Each chart shows balanced and acc-local-l3 mean/peak together. Domain axis includes L3_00 to L3_15 and Total.</p>'
        f'<div class="chart-grid">{"".join(cards)}</div>'
        '</details>'
    )


def build_html(case_index: dict[CaseKey, CaseOccupancy]) -> str:
    lengths = sorted({key.q_len for key in case_index if key.q_len is not None})
    lengths = [int(length) for length in lengths]
    body_parts = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN">',
        '<head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>LLC occupancy compare</title>',
        '<style>',
        ':root { --bg: #f6f7fb; --panel: #ffffff; --ink: #141824; --muted: #667085; --line: #d7dbe7; }',
        '* { box-sizing: border-box; }',
        'body { margin: 0; font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--ink); }',
        '.page { max-width: 1720px; margin: 0 auto; padding: 24px 24px 40px; }',
        'h1 { margin: 0 0 8px; font-size: 28px; }',
        '.meta { color: var(--muted); margin-bottom: 18px; }',
        '.section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 16px 18px; margin: 16px 0; }',
        '.section > h2, .section > summary { margin: 0 0 12px; font-size: 20px; font-weight: 700; cursor: pointer; }',
        '.section-note { margin: 0 0 14px; color: var(--muted); font-size: 13px; }',
        '.chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }',
        '.chart-card { border: 1px solid var(--line); border-radius: 8px; padding: 10px 10px 8px; background: #fff; }',
        '.chart-card-title { font-size: 14px; font-weight: 700; margin-bottom: 8px; color: var(--ink); }',
        '.chart-scroll { overflow-x: auto; }',
        'svg { display: block; }',
        '@media (max-width: 1200px) { .chart-grid { grid-template-columns: 1fr; } }',
        '</style>',
        '</head>',
        '<body>',
        '<div class="page">',
        '<h1>LLC occupancy compare</h1>',
        '<div class="meta">MB unit. Each length section shows balanced and acc-local-l3 mean/peak for ffff and 0001.</div>',
        _render_runtime_section(case_index, lengths),
        _render_total_section(case_index, lengths),
    ]
    for q_len in lengths:
        body_parts.append(_render_length_section(case_index, q_len))
    body_parts.extend(['</div>', '</body>', '</html>'])
    return ''.join(body_parts)


def write_html(result_root: str | Path,
               output_path: str | Path | None = None,
               *,
               masks: tuple[str, str] = (DEFAULT_BASELINE_MASK, DEFAULT_COMPARE_MASK)) -> Path:
    result_root = Path(result_root)
    case_index = collect_case_index(result_root, masks=masks)
    html_text = build_html(case_index)
    output_path = (Path(output_path) if output_path is not None else
                   result_root / DEFAULT_OUTPUT_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML report for LLC occupancy comparisons.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    output_path = write_html(args.result_root, args.output)
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
