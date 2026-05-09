from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

METRIC_SPECS = [
    ("IPC (Sys + User)", 4),
    ("IPC (Sys)", 4),
    ("IPC (User)", 4),
    ("CPI (Sys + User)", 4),
    ("CPI (Sys)", 4),
    ("CPI (User)", 4),
    ("L3 Access", 2),
    ("L3 Miss", 2),
    ("L3 Access (pti)", 2),
    ("L3 Miss (pti)", 2),
    ("L3 Miss / second", 2),
    ("L3 Miss %", 2),
    ("L3 Hit %", 2),
    ("Raw L3SampledLatencyAll", 2),
    ("Raw L3SampledLatencyRequestsAll", 2),
    ("Derived Avg L3 Miss Latency (ns)", 2),
    ("Raw L3SampledLatencyFromLocalMemory", 2),
    ("Raw L3SampledLatencyRequestsFromLocalMemory", 2),
    ("Derived Local Memory Avg L3 Miss Latency (ns)", 2),
    ("Derived Local Memory L3 Miss Request Share (%)", 2),
    ("Raw L3SampledLatencyFromExternalCacheLocal", 2),
    ("Raw L3SampledLatencyRequestsFromExternalCacheLocal", 2),
    ("Derived another CCX in same node Avg L3 Miss Latency (ns)", 2),
    ("Derived another CCX in same node L3 Miss Request Share (%)", 2),
    ("Raw L3SampledLatencyFromRemoteMemory", 2),
    ("Raw L3SampledLatencyRequestsFromRemoteMemory", 2),
    ("Derived Remote Memory Avg L3 Miss Latency (ns)", 2),
    ("Derived Remote Memory L3 Miss Request Share (%)", 2),
    ("Raw L3SampledLatencyFromExternalCacheRemote", 2),
    ("Raw L3SampledLatencyRequestsFromExternalCacheRemote", 2),
    ("Derived another CCX in remote node Avg L3 Miss Latency (ns)", 2),
    ("Derived another CCX in remote node L3 Miss Request Share (%)", 2),
]

CASE_METRIC_NAMES = [metric_name for metric_name, _ in METRIC_SPECS]

CASE_FIELDNAMES = [
    "workload",
    "partition_mode",
    "batch_size",
    "q_len",
    "kv_len",
    "mode_dir",
    "group_span",
    "slowest_rank_mean_ms",
    "note",
    "case_dir",
    *CASE_METRIC_NAMES,
]

DEFAULT_RESULT_ROOT = Path(
    "test_results/P3_AttnOnly/Qwen3-30B-A3B/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1")


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _parse_shape_dirname(shape_dirname: str) -> dict[str, int] | None:
    match = re.fullmatch(r"q(?P<q_len>\d+)_kv_?(?P<kv_len>\d+)",
                         shape_dirname,
                         re.IGNORECASE)
    if not match:
        return None
    return {
        "q_len": int(match.group("q_len")),
        "kv_len": int(match.group("kv_len")),
    }


def _latest_report_cumulative_csv(case_dir: Path) -> Path | None:
    latest: Path | None = None
    for report_csv in case_dir.glob("pcm_l3_source_latency_raw/*/report-cumulative.csv"):
        if latest is None:
            latest = report_csv
            continue
        candidate_key = (report_csv.stat().st_mtime_ns, report_csv.parent.name)
        latest_key = (latest.stat().st_mtime_ns, latest.parent.name)
        if candidate_key > latest_key:
            latest = report_csv
    return latest


def _load_dry_run(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved_lengths = data.get("resolved_lengths") or {}
    requested_lengths = data.get("requested_lengths") or {}
    head_plan = data.get("head_plan") or {}
    return {
        "workload": data.get("workload"),
        "batch_size": data.get("batch_size"),
        "q_len": data.get("q_len") or resolved_lengths.get("q_len")
        or requested_lengths.get("q_len"),
        "kv_len": data.get("kv_len") or resolved_lengths.get("kv_len")
        or requested_lengths.get("kv_len"),
        "partition_mode": head_plan.get("partition_mode"),
        "slowest_rank_mean_ms": data.get("slowest_rank_mean_ms"),
        "mode_name": data.get("attn_locality_mode"),
        "group_span": data.get("attn_locality_group_span"),
    }


def _infer_mode_name(mode_dir: str) -> str:
    return re.sub(r"_span\d+$", "", mode_dir)


def _header_index(row: list[str], target: str) -> int | None:
    for idx, value in enumerate(row):
        if value.strip() == target:
            return idx
    return None


def extract_system_metrics(report_csv_path: str | Path) -> dict[str, float]:
    report_csv_path = _as_path(report_csv_path)
    metrics: dict[str, float] = {}
    system_column_idx: int | None = None

    with report_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            metric_name = row[0].strip()
            if metric_name == "Metric":
                system_column_idx = _header_index(row, "System (Aggregated)")
                continue
            if system_column_idx is None or metric_name not in CASE_METRIC_NAMES:
                continue
            if system_column_idx >= len(row):
                continue
            raw_value = row[system_column_idx].strip()
            if not raw_value:
                continue
            metrics[metric_name] = float(raw_value)

    return metrics


def discover_cases(result_root: str | Path) -> list[dict[str, Any]]:
    result_root = _as_path(result_root).resolve()
    cases = []
    for summary_path in sorted(result_root.glob("**/dry_run_summary.json")):
        case_dir = summary_path.parent
        rel_parts = case_dir.relative_to(result_root).parts
        workload = rel_parts[0] if len(rel_parts) > 0 else None
        partition_mode = rel_parts[1] if len(rel_parts) > 1 else None
        batch_size = None
        q_len = None
        kv_len = None
        for part in rel_parts:
            if part.startswith("batch_"):
                batch_size = int(part.split("_", 1)[1])
            parsed = _parse_shape_dirname(part)
            if parsed:
                q_len = parsed["q_len"]
                kv_len = parsed["kv_len"]

        report_csv = _latest_report_cumulative_csv(case_dir)
        cases.append({
            "workload": workload,
            "partition_mode": partition_mode,
            "batch_size": batch_size,
            "q_len": q_len,
            "kv_len": kv_len,
            "mode_dir": case_dir.name,
            "case_dir": str(case_dir),
            "dry_run_summary_json": str(summary_path),
            "report_cumulative_csv": str(report_csv) if report_csv is not None else
            None,
            "session_dir": str(report_csv.parent) if report_csv is not None else
            None,
        })

    cases.sort(key=lambda case: (
        case.get("workload") or "",
        case.get("partition_mode") or "",
        case.get("batch_size") or -1,
        case.get("q_len") or -1,
        case.get("kv_len") or -1,
        0 if case.get("mode_dir", "").startswith("balanced") else
        1 if case.get("mode_dir", "").startswith("acc-local-l3") else 2,
        case.get("mode_dir") or "",
    ))
    return cases


def build_case_rows(result_root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for case in discover_cases(result_root):
        dry_run = _load_dry_run(Path(case["dry_run_summary_json"]))
        metrics = (extract_system_metrics(case["report_cumulative_csv"])
                   if case["report_cumulative_csv"] else {})
        row = {
            **case,
            **metrics,
            "workload": dry_run.get("workload") or case["workload"],
            "partition_mode": dry_run.get("partition_mode")
            or case["partition_mode"],
            "batch_size": dry_run.get("batch_size") or case["batch_size"],
            "q_len": dry_run.get("q_len") or case["q_len"],
            "kv_len": dry_run.get("kv_len") or case["kv_len"],
            "mode_dir": case["mode_dir"],
            "mode_name": dry_run.get("mode_name")
            or _infer_mode_name(case["mode_dir"]),
            "group_span": dry_run.get("group_span"),
            "slowest_rank_mean_ms": dry_run.get("slowest_rank_mean_ms"),
            "note": "latest PCM session" if case["report_cumulative_csv"] else
            "missing report-cumulative.csv",
        }
        rows.append(row)

    rows.sort(key=lambda row: (
        row.get("workload") or "",
        row.get("partition_mode") or "",
        row.get("batch_size") or -1,
        row.get("q_len") or -1,
        row.get("kv_len") or -1,
        0 if str(row.get("mode_dir") or "").startswith("balanced") else
        1 if str(row.get("mode_dir") or "").startswith("acc-local-l3") else 2,
        row.get("mode_dir") or "",
    ))
    return rows


def _fmt(value: float | None, precision: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def build_transposed_rows(
        case_rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    case_headers = []
    for row in case_rows:
        shape = f"q{row['q_len']}_kv{row['kv_len']}"
        case_headers.append(f"{shape}_{row['mode_dir']}")

    fieldnames = ["metric", *case_headers]
    metadata_specs = [
        ("workload", lambda row: str(row.get("workload") or "-")),
        ("partition_mode", lambda row: str(row.get("partition_mode") or "-")),
        ("batch_size", lambda row: str(row.get("batch_size") or "-")),
        ("q_len", lambda row: str(row.get("q_len") or "-")),
        ("kv_len", lambda row: str(row.get("kv_len") or "-")),
        ("mode_dir", lambda row: str(row.get("mode_dir") or "-")),
        ("group_span", lambda row: str(row.get("group_span") or "-")),
        ("slowest_rank_mean_ms",
         lambda row: _fmt(row.get("slowest_rank_mean_ms"), 6)),
        ("note", lambda row: str(row.get("note") or "-")),
    ]

    rows = []
    for metric_name, value_fn in metadata_specs:
        row = {"metric": metric_name}
        for case_row, case_header in zip(case_rows, case_headers):
            row[case_header] = value_fn(case_row)
        rows.append(row)

    for metric_name, precision in METRIC_SPECS:
        row = {"metric": metric_name}
        for case_row, case_header in zip(case_rows, case_headers):
            row[case_header] = _fmt(case_row.get(metric_name), precision)
        rows.append(row)
    return fieldnames, rows


def _write_csv(rows: list[dict[str, Any]], csv_path: Path,
               fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(
    result_root: str | Path,
    output_csv: str | Path | None = None,
    output_detail_csv: str | Path | None = None,
) -> tuple[Path, Path]:
    result_root = _as_path(result_root).resolve()
    output_csv = (_as_path(output_csv)
                  if output_csv else result_root / "summary_l3_source_latency.csv")
    output_detail_csv = (
        _as_path(output_detail_csv)
        if output_detail_csv else result_root / "detail_l3_source_latency.csv")

    case_rows = build_case_rows(result_root)
    summary_fieldnames, summary_rows = build_transposed_rows(case_rows)
    _write_csv(summary_rows, output_csv, summary_fieldnames)
    _write_csv(case_rows, output_detail_csv, CASE_FIELDNAMES)
    return output_csv, output_detail_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize PCM L3 source latency cumulative CSV files.")
    parser.add_argument("--result-root",
                        type=Path,
                        default=DEFAULT_RESULT_ROOT,
                        help="Root directory containing P3 attention-only results.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to transposed summary CSV. Default: <result-root>/summary_l3_source_latency.csv",
    )
    parser.add_argument(
        "--output-detail-csv",
        type=Path,
        default=None,
        help="Path to detail CSV. Default: <result-root>/detail_l3_source_latency.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv, detail_csv = build_outputs(
        result_root=args.result_root,
        output_csv=args.output_csv,
        output_detail_csv=args.output_detail_csv,
    )
    print(f"wrote summary csv: {summary_csv}")
    print(f"wrote detail csv: {detail_csv}")


if __name__ == "__main__":
    main()
