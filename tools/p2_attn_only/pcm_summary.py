from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SUMMARY_METRIC_NAMES = [
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

LATENCY_BREAKDOWN_METRIC_NAMES = [
    "L3 Miss Latency From Local Memory or I/O (%)",
    "L3 Miss Latency From Remote Memory or I/O (%)",
    "L3 Miss Latency From another CCX in same node (%)",
    "L3 Miss Latency From another CCX in remote node (%)",
    "L3 Miss Latency From Local Extension Memory (CXL) (%)",
    "L3 Miss Latency From Remote Extension Memory (CXL) (%)",
]

SUMMARY_FIELDNAMES = [
    "workload",
    "partition_mode",
    "batch_size",
    "q_len",
    "kv_len",
    "slowest_rank_mean_ms",
    "tp_size",
    "global_num_query_heads",
    "global_num_kv_heads",
    "local_num_query_heads",
    "local_num_kv_heads",
    "result_shape_dirname",
    "case_dir",
    "session_dir",
    "dry_run_summary_json",
    "report_json",
    *SUMMARY_METRIC_NAMES,
]

CCX_FIELDNAMES = [
    "workload",
    "partition_mode",
    "batch_size",
    "q_len",
    "kv_len",
    "ccx_id",
    "case_dir",
    "session_dir",
    "report_json",
    *SUMMARY_METRIC_NAMES,
]

SHAPE_RE = re.compile(r"q(?P<q_len>\d+)_kv_?(?P<kv_len>\d+)", re.IGNORECASE)


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _parse_shape_dirname(shape_dirname: str) -> dict[str, int] | None:
    match = SHAPE_RE.fullmatch(shape_dirname)
    if not match:
        return None
    return {
        "q_len": int(match.group("q_len")),
        "kv_len": int(match.group("kv_len")),
    }


def _extract_group_target(metric: dict[str, Any]) -> tuple[str | None, int | None]:
    for group_entry in metric.get("group", []):
        l3_entries = group_entry.get("l3")
        if not l3_entries:
            continue
        l3_entry = l3_entries[0]
        return l3_entry.get("name"), l3_entry.get("id")
    return None, None


def _extract_average(metric: dict[str, Any]) -> float | None:
    aggregated = metric.get("aggregated", {})
    value = aggregated.get("average")
    if value is None:
        return None
    return float(value)


def parse_case_dir(case_dir: Path, result_root: Path) -> dict[str, Any]:
    rel_parts = case_dir.relative_to(result_root).parts
    workload = rel_parts[0] if len(rel_parts) > 0 else None
    partition_mode = rel_parts[1] if len(rel_parts) > 1 else None

    batch_size = None
    q_len = None
    kv_len = None
    for part in rel_parts:
        if part.startswith("batch_"):
            batch_size = int(part.split("_", 1)[1])
        parsed_shape = _parse_shape_dirname(part)
        if parsed_shape:
            q_len = parsed_shape["q_len"]
            kv_len = parsed_shape["kv_len"]

    return {
        "workload": workload,
        "partition_mode": partition_mode,
        "batch_size": batch_size,
        "q_len": q_len,
        "kv_len": kv_len,
        "case_dir": str(case_dir),
    }


def discover_cases(result_root: str | Path) -> list[dict[str, Any]]:
    result_root = _as_path(result_root).resolve()
    latest_report_by_case: dict[Path, Path] = {}
    for report_json in result_root.glob("**/report.json"):
        if report_json.parent.parent.name != "pcm_l3_dc":
            continue
        case_dir = report_json.parent.parent.parent
        current = latest_report_by_case.get(case_dir)
        report_key = (report_json.stat().st_mtime_ns, report_json.parent.name)
        current_key = ((current.stat().st_mtime_ns, current.parent.name)
                       if current is not None else None)
        if current is None or report_key > current_key:
            latest_report_by_case[case_dir] = report_json

    cases = []
    for case_dir in sorted(latest_report_by_case):
        report_json = latest_report_by_case[case_dir]
        case = parse_case_dir(case_dir, result_root)
        case.update({
            "session_dir": str(report_json.parent),
            "report_json": str(report_json),
            "dry_run_summary_json": str(case_dir / "dry_run_summary.json"),
        })
        cases.append(case)
    return cases


def load_dry_run_summary(summary_path: str | Path) -> dict[str, Any]:
    summary_path = _as_path(summary_path)
    with summary_path.open(encoding="utf-8") as f:
        data = json.load(f)

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
        "slowest_rank_mean_ms": data.get("slowest_rank_mean_ms"),
        "result_shape_dirname": data.get("result_shape_dirname"),
        "partition_mode": head_plan.get("partition_mode"),
        "tp_size": head_plan.get("tp_size"),
        "global_num_query_heads": head_plan.get("global_num_query_heads"),
        "global_num_kv_heads": head_plan.get("global_num_kv_heads"),
        "local_num_query_heads": head_plan.get("local_num_query_heads"),
        "local_num_kv_heads": head_plan.get("local_num_kv_heads"),
    }


def extract_system_metrics(report_json_path: str | Path) -> dict[str, float]:
    report_json_path = _as_path(report_json_path)
    with report_json_path.open(encoding="utf-8") as f:
        report = json.load(f)

    metrics = {}
    for metric in report.get("metrics", []):
        name = metric.get("name")
        if name not in SUMMARY_METRIC_NAMES:
            continue
        target_name, target_id = _extract_group_target(metric)
        if (target_name, target_id) != ("system", -1):
            continue
        average = _extract_average(metric)
        if average is not None:
            metrics[name] = average
    return metrics


def extract_ccx_metrics(report_json_path: str | Path) -> list[dict[str, Any]]:
    report_json_path = _as_path(report_json_path)
    with report_json_path.open(encoding="utf-8") as f:
        report = json.load(f)

    rows: dict[int, dict[str, Any]] = defaultdict(dict)
    for metric in report.get("metrics", []):
        name = metric.get("name")
        if name not in SUMMARY_METRIC_NAMES:
            continue
        target_name, target_id = _extract_group_target(metric)
        if target_name != "ccx" or target_id is None or target_id < 0:
            continue
        average = _extract_average(metric)
        if average is None:
            continue
        row = rows[target_id]
        row["ccx_id"] = target_id
        row[name] = average

    return [rows[ccx_id] for ccx_id in sorted(rows)]


def build_summary_rows(result_root: str | Path) -> list[dict[str, Any]]:
    result_root = _as_path(result_root).resolve()
    rows = []
    for case in discover_cases(result_root):
        dry_run_summary = load_dry_run_summary(case["dry_run_summary_json"])
        metrics = extract_system_metrics(case["report_json"])

        row = {
            **case,
            **metrics,
            "workload": dry_run_summary.get("workload") or case["workload"],
            "partition_mode":
            dry_run_summary.get("partition_mode") or case["partition_mode"],
            "batch_size": dry_run_summary.get("batch_size") or case["batch_size"],
            "q_len": dry_run_summary.get("q_len") or case["q_len"],
            "kv_len": dry_run_summary.get("kv_len") or case["kv_len"],
            "slowest_rank_mean_ms": dry_run_summary.get("slowest_rank_mean_ms"),
            "tp_size": dry_run_summary.get("tp_size"),
            "global_num_query_heads":
            dry_run_summary.get("global_num_query_heads"),
            "global_num_kv_heads": dry_run_summary.get("global_num_kv_heads"),
            "local_num_query_heads": dry_run_summary.get("local_num_query_heads"),
            "local_num_kv_heads": dry_run_summary.get("local_num_kv_heads"),
            "result_shape_dirname": dry_run_summary.get("result_shape_dirname")
            or (
                f"batch_{case['batch_size']}/q{case['q_len']}_KV_{case['kv_len']}"
                if case["batch_size"] is not None and case["q_len"] is not None
                and case["kv_len"] is not None else None
            ),
        }
        rows.append(row)

    rows.sort(key=lambda row: (
        row.get("workload") or "",
        row.get("partition_mode") or "",
        row.get("batch_size") or -1,
        row.get("q_len") or -1,
        row.get("kv_len") or -1,
    ))
    return rows


def build_ccx_rows(result_root: str | Path) -> list[dict[str, Any]]:
    result_root = _as_path(result_root).resolve()
    rows = []
    for case in discover_cases(result_root):
        for ccx_row in extract_ccx_metrics(case["report_json"]):
            rows.append({
                **{
                    key: case[key]
                    for key in (
                        "workload",
                        "partition_mode",
                        "batch_size",
                        "q_len",
                        "kv_len",
                        "case_dir",
                        "session_dir",
                        "report_json",
                    )
                },
                **ccx_row,
            })
    rows.sort(key=lambda row: (
        row.get("workload") or "",
        row.get("q_len") or -1,
        row.get("kv_len") or -1,
        row.get("ccx_id") or -1,
    ))
    return rows


def _write_csv(rows: list[dict[str, Any]], csv_path: Path,
               fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(rows: list[dict[str, Any]], markdown_path: Path,
                           result_root: Path) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    lines = [
        "# P2 Attention-Only PCM Summary",
        "",
        f"> generated_at: {generated_at}",
        f"> result_root: {result_root}",
        "",
        f"- case_count: {len(rows)}",
        f"- metrics: {', '.join(SUMMARY_METRIC_NAMES)}",
        "",
    ]

    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[row["workload"]].append(row)

    for workload in sorted(by_workload):
        workload_rows = by_workload[workload]
        workload_rows.sort(key=lambda row: (row["q_len"], row["kv_len"]))
        lines.extend([
            f"## {workload}",
            "",
            "| q_len | kv_len | slowest rank mean (ms) | L3 Access (pti) | L3 Miss (pti) | L3 Miss % | Ave L3 Miss Latency (ns) | same-node another CCX % | local memory / I/O % | remote memory / I/O % | report.json |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in workload_rows:
            report_rel = Path(row["report_json"]).resolve().relative_to(
                result_root.resolve())
            lines.append(
                "| {q_len} | {kv_len} | {slowest_rank_mean_ms:.4f} | {l3_access:.2f} | {l3_miss_pti:.2f} | {l3_miss_pct:.2f} | {l3_latency:.2f} | {same_node:.2f} | {local_mem:.2f} | {remote_mem:.2f} | `{report_rel}` |"
                .format(
                    q_len=row["q_len"],
                    kv_len=row["kv_len"],
                    slowest_rank_mean_ms=row["slowest_rank_mean_ms"],
                    l3_access=row["L3 Access (pti)"],
                    l3_miss_pti=row["L3 Miss (pti)"],
                    l3_miss_pct=row["L3 Miss %"],
                    l3_latency=row["Ave L3 Miss Latency (ns)"],
                    same_node=row[
                        "L3 Miss Latency From another CCX in same node (%)"],
                    local_mem=row[
                        "L3 Miss Latency From Local Memory or I/O (%)"],
                    remote_mem=row[
                        "L3 Miss Latency From Remote Memory or I/O (%)"],
                    report_rel=report_rel,
                ))
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(result_root: str | Path,
                  output_csv: str | Path | None = None,
                  output_ccx_csv: str | Path | None = None,
                  output_md: str | Path | None = None) -> tuple[Path, Path, Path]:
    result_root = _as_path(result_root).resolve()
    output_csv = _as_path(output_csv) if output_csv else result_root / "summary.csv"
    output_ccx_csv = (_as_path(output_ccx_csv)
                      if output_ccx_csv else result_root / "ccx_summary.csv")
    output_md = _as_path(output_md) if output_md else result_root / "summary.md"

    rows = build_summary_rows(result_root)
    ccx_rows = build_ccx_rows(result_root)
    _write_csv(rows, output_csv, SUMMARY_FIELDNAMES)
    _write_csv(ccx_rows, output_ccx_csv, CCX_FIELDNAMES)
    write_summary_markdown(rows, output_md, result_root)
    return output_csv, output_ccx_csv, output_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize attention-only PCM results into csv/md tables.")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-ccx-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv, output_ccx_csv, output_md = build_outputs(
        result_root=args.result_root,
        output_csv=args.output_csv,
        output_ccx_csv=args.output_ccx_csv,
        output_md=args.output_md,
    )
    print(f"summary_csv={output_csv}")
    print(f"ccx_summary_csv={output_ccx_csv}")
    print(f"summary_md={output_md}")


if __name__ == "__main__":
    main()
