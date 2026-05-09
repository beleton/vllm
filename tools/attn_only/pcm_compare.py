from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

METRIC_SPECS = [
    ("ipc_sys_user", "IPC (Sys + User)", 4),
    ("ipc_sys", "IPC (Sys)", 4),
    ("ipc_user", "IPC (User)", 4),
    ("cpi_sys_user", "CPI (Sys + User)", 4),
    ("cpi_sys", "CPI (Sys)", 4),
    ("cpi_user", "CPI (User)", 4),
    ("all_dc_fills_pti", "All DC Fills (pti)", 2),
    ("dc_fills_local_l2_pti", "DC Fills From Local L2 (pti)", 2),
    ("dc_fills_local_same_ccx_pti",
     "DC Fills From Local L3 or different L2 in same CCX (pti)", 2),
    ("dc_fills_another_ccx_same_node_pti",
     "DC Fills From another CCX in same node (pti)", 2),
    ("dc_fills_local_memory_pti", "DC Fills From Local Memory or I/O (pti)", 2),
    ("dc_fills_another_ccx_remote_node_pti",
     "DC Fills From another CCX in remote node (pti)", 2),
    ("dc_fills_remote_memory_pti",
     "DC Fills From Remote Memory or I/O (pti)", 2),
    ("all_demand_dc_fills_pti", "All Demand DC Fills (pti)", 2),
    ("demand_local_l2_pti", "Demand DC Fills From Local L2 (pti)", 2),
    ("demand_local_same_ccx_pti",
     "Demand DC Fills From Local L3 or different L2 in same CCX (pti)", 2),
    ("demand_another_ccx_same_node_pti",
     "Demand DC Fills From another CCX in same node (pti)", 2),
    ("demand_local_memory_pti",
     "Demand DC Fills From Local Memory or I/O (pti)", 2),
    ("demand_another_ccx_remote_node_pti",
     "Demand DC Fills From another CCX in remote node (pti)", 2),
    ("demand_remote_memory_pti",
     "Demand DC Fills From Remote memory or I/O (pti)", 2),
    ("l3_access_pti", "L3 Access (pti)", 2),
    ("l3_miss_pti", "L3 Miss (pti)", 2),
    ("l3_miss_pct", "L3 Miss %", 2),
    ("ave_l3_miss_latency_ns", "Ave L3 Miss Latency (ns)", 2),
    ("latency_local_memory_pct",
     "L3 Miss Latency From Local Memory or I/O (%)", 2),
    ("latency_another_ccx_same_node_pct",
     "L3 Miss Latency From another CCX in same node (%)", 2),
    ("latency_another_ccx_remote_node_pct",
     "L3 Miss Latency From another CCX in remote node (%)", 2),
    ("latency_remote_memory_pct",
     "L3 Miss Latency From Remote Memory or I/O (%)", 2),
]

PROFILE_METRIC_SPECS = [
    ("profile_scheduler_metadata_avg_ns", "Scheduler Metadata Avg (ns)", 2),
    ("profile_legacy_scheduler_metadata_avg_ns",
     "Legacy Scheduler Metadata Avg (ns)", 2),
    ("profile_locality_metadata_build_avg_ns",
     "Locality Metadata Build Avg (ns)", 2),
    ("profile_attention_task_body_avg_ns", "Attention Task Body Avg (ns)", 2),
    ("profile_execute_attention_avg_ns", "Execute Attention Avg (ns)", 2),
    ("profile_attention_task_non_execute_pct",
     "Attention Task Non-execute (%)", 4),
    ("profile_slowest_rank_attention_parallelism",
     "Slowest Rank Attention Effective Threads", 2),
    ("profile_slowest_rank_execute_parallelism",
     "Slowest Rank Execute Effective Threads", 2),
]

PROFILE_DELTA_SPECS = [
    ("profile_scheduler_metadata_avg_ns_delta_pct",
     "Scheduler Metadata Avg Delta (%)", 2),
    ("profile_attention_task_body_avg_ns_delta_pct",
     "Attention Task Body Avg Delta (%)", 2),
    ("profile_execute_attention_avg_ns_delta_pct",
     "Execute Attention Avg Delta (%)", 2),
    ("profile_slowest_rank_attention_parallelism_delta_pct",
     "Slowest Rank Attention Effective Threads Delta (%)", 2),
    ("profile_slowest_rank_execute_parallelism_delta_pct",
     "Slowest Rank Execute Effective Threads Delta (%)", 2),
]

DISPLAY_METRIC_NAMES = {
    "L3 Miss %": "L3 Miss (%)",
}

CASE_METRIC_NAMES = [
    metric_name for _, metric_name, _ in METRIC_SPECS
]

COMPARE_METRICS = [
    (slug, metric_name) for slug, metric_name, _ in METRIC_SPECS
]

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
    *[slug for slug, _, _ in PROFILE_METRIC_SPECS],
    *CASE_METRIC_NAMES,
]

COMPARE_FIELDNAMES = [
    "workload",
    "partition_mode",
    "batch_size",
    "q_len",
    "kv_len",
    "balanced_mode_dir",
    "acc_local_l3_mode_dir",
    "balanced_slowest_rank_mean_ms",
    "acc_local_l3_slowest_rank_mean_ms",
    "slowest_rank_mean_ms_delta_pct",
    *[slug for slug, _, _ in PROFILE_DELTA_SPECS],
]

DEFAULT_RESULT_ROOT = Path(
    "test_results/P3_AttnOnly/Qwen3-30B-A3B/qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1")
for prefix in ("balanced", "acc_local_l3"):
    COMPARE_FIELDNAMES.extend([
        f"{prefix}_note",
    ])
    for slug, _ in COMPARE_METRICS:
        COMPARE_FIELDNAMES.append(f"{prefix}_{slug}")
    for slug, _, _ in PROFILE_METRIC_SPECS:
        COMPARE_FIELDNAMES.append(f"{prefix}_{slug}")


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


def _latest_report_json(case_dir: Path) -> Path | None:
    latest: Path | None = None
    for report_json in case_dir.glob("pcm_l3_dc/*/report.json"):
        if latest is None:
            latest = report_json
            continue
        candidate_key = (report_json.stat().st_mtime_ns, report_json.parent.name)
        latest_key = (latest.stat().st_mtime_ns, latest.parent.name)
        if candidate_key > latest_key:
            latest = report_json
    return latest


def _profile_json(case_dir: Path) -> Path | None:
    profile_json = case_dir / "profile.json"
    if profile_json.exists():
        return profile_json
    return None


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


def _safe_avg(total: float | int | None, count: float | int | None) -> float | None:
    if total is None or count in (None, 0):
        return None
    return float(total) / float(count)


def _coalesce(*values: float | int | None) -> float | int | None:
    for value in values:
        if value is not None:
            return value
    return None


def _extract_slowest_rank_runtime_metrics(
    profile_data: dict[str, Any],
) -> dict[str, float | None]:
    rank_results = profile_data.get("rank_results") or []
    if not rank_results:
        return {
            "profile_slowest_rank_attention_parallelism": None,
            "profile_slowest_rank_execute_parallelism": None,
        }

    def _rank_key(rank_result: dict[str, Any]) -> float:
        return float((rank_result.get("result") or {}).get("time_mean_ms") or 0.0)

    slowest_rank = max(rank_results, key=_rank_key)
    elapsed_ms = slowest_rank.get("elapsed_ms")
    runtime = ((slowest_rank.get("profile") or {}).get("runtime") or {})
    if elapsed_ms in (None, 0):
        return {
            "profile_slowest_rank_attention_parallelism": None,
            "profile_slowest_rank_execute_parallelism": None,
        }

    elapsed_ns = float(elapsed_ms) * 1e6
    attention_task_body_ns = runtime.get("attention_task_body_ns")
    execute_attention_ns = runtime.get("execute_attention_ns")
    return {
        "profile_slowest_rank_attention_parallelism": (
            float(attention_task_body_ns) / elapsed_ns
            if attention_task_body_ns is not None
            else None
        ),
        "profile_slowest_rank_execute_parallelism": (
            float(execute_attention_ns) / elapsed_ns
            if execute_attention_ns is not None
            else None
        ),
    }


def extract_profile_metrics(profile_json_path: str | Path) -> dict[str, float | None]:
    profile_json_path = _as_path(profile_json_path)
    profile_data = json.loads(profile_json_path.read_text(encoding="utf-8"))
    profile_summary = profile_data.get("profile_summary") or {}
    scheduler = profile_summary.get("scheduler") or {}
    runtime = profile_summary.get("runtime") or {}

    scheduler_call_count = scheduler.get("call_count")
    attention_task_body_ns = runtime.get("attention_task_body_ns")
    attention_task_non_execute_ns = runtime.get("attention_task_non_execute_ns")

    metrics: dict[str, float | None] = {
        "profile_scheduler_metadata_avg_ns":
        _coalesce(
            scheduler.get("scheduler_metadata_avg_ns"),
            _safe_avg(scheduler.get("scheduler_metadata_ns"), scheduler_call_count),
        ),
        "profile_legacy_scheduler_metadata_avg_ns":
        _safe_avg(scheduler.get("legacy_scheduler_metadata_ns"),
                  scheduler_call_count),
        "profile_locality_metadata_build_avg_ns":
        _safe_avg(scheduler.get("locality_metadata_build_ns"),
                  scheduler_call_count),
        "profile_attention_task_body_avg_ns":
        _coalesce(
            runtime.get("attention_task_body_avg_ns"),
            _safe_avg(attention_task_body_ns, runtime.get("attention_task_count")),
        ),
        "profile_execute_attention_avg_ns":
        _coalesce(
            runtime.get("execute_attention_avg_ns"),
            _safe_avg(runtime.get("execute_attention_ns"),
                      runtime.get("execute_attention_count")),
        ),
        "profile_attention_task_non_execute_pct": (
            float(attention_task_non_execute_ns) / float(attention_task_body_ns) *
            100.0
            if attention_task_non_execute_ns is not None and attention_task_body_ns
            not in (None, 0)
            else None
        ),
    }
    metrics.update(_extract_slowest_rank_runtime_metrics(profile_data))
    return metrics


def _extract_group_target(metric: dict[str, Any]) -> tuple[str | None, int | None]:
    for group_entry in metric.get("group", []):
        for entries in group_entry.values():
            if not entries:
                continue
            entry = entries[0]
            return entry.get("name"), entry.get("id")
    return None, None


def _extract_average(metric: dict[str, Any]) -> float | None:
    aggregated = metric.get("aggregated", {})
    value = aggregated.get("average")
    if value is None:
        return None
    return float(value)


def extract_system_metrics(report_json_path: str | Path) -> dict[str, float]:
    report_json_path = _as_path(report_json_path)
    report = json.loads(report_json_path.read_text(encoding="utf-8"))

    metrics = {}
    for metric in report.get("metrics", []):
        name = metric.get("name")
        if name not in CASE_METRIC_NAMES:
            continue
        target_name, target_id = _extract_group_target(metric)
        if (target_name, target_id) != ("system", -1):
            continue
        average = _extract_average(metric)
        if average is not None:
            metrics[name] = average
    return metrics


def _infer_mode_name(mode_dir: str) -> str:
    return re.sub(r"_span\d+$", "", mode_dir)


def _mode_prefix(mode_name: str) -> str:
    return mode_name.replace("-", "_")


def _delta_pct(base: float | None, new: float | None) -> float | None:
    if base in (None, 0) or new is None:
        return None
    return (new - base) / base * 100.0


def _build_case_header(shape: str, mode_dir: str | None) -> str:
    if not mode_dir:
        return shape
    return f"{shape}_{mode_dir}"


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

        report_json = _latest_report_json(case_dir)
        profile_json = _profile_json(case_dir)
        cases.append({
            "workload": workload,
            "partition_mode": partition_mode,
            "batch_size": batch_size,
            "q_len": q_len,
            "kv_len": kv_len,
            "mode_dir": case_dir.name,
            "case_dir": str(case_dir),
            "dry_run_summary_json": str(summary_path),
            "report_json": str(report_json) if report_json is not None else None,
            "profile_json": str(profile_json) if profile_json is not None else None,
            "session_dir": str(report_json.parent) if report_json is not None else
            None,
        })

    cases.sort(key=lambda case: (
        case.get("workload") or "",
        case.get("partition_mode") or "",
        case.get("batch_size") or -1,
        case.get("q_len") or -1,
        case.get("kv_len") or -1,
        case.get("mode_dir") or "",
    ))
    return cases


def build_case_rows(result_root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for case in discover_cases(result_root):
        dry_run = _load_dry_run(Path(case["dry_run_summary_json"]))
        mode_name = dry_run.get("mode_name") or _infer_mode_name(case["mode_dir"])
        metrics = (extract_system_metrics(case["report_json"])
                   if case["report_json"] else {})
        profile_metrics = (
            extract_profile_metrics(case["profile_json"])
            if case["profile_json"] else {}
        )
        row = {
            **case,
            **metrics,
            **profile_metrics,
            "workload": dry_run.get("workload") or case["workload"],
            "partition_mode": dry_run.get("partition_mode")
            or case["partition_mode"],
            "batch_size": dry_run.get("batch_size") or case["batch_size"],
            "q_len": dry_run.get("q_len") or case["q_len"],
            "kv_len": dry_run.get("kv_len") or case["kv_len"],
            "mode_name": mode_name,
            "mode_prefix": _mode_prefix(mode_name),
            "group_span": dry_run.get("group_span"),
            "slowest_rank_mean_ms": dry_run.get("slowest_rank_mean_ms"),
            "note": "latest PCM session" if case["report_json"] else
            "missing report.json",
        }
        rows.append(row)
    rows.sort(key=lambda row: (
        row.get("workload") or "",
        row.get("partition_mode") or "",
        row.get("batch_size") or -1,
        row.get("q_len") or -1,
        row.get("kv_len") or -1,
        0 if row.get("mode_prefix") == "balanced" else
        1 if row.get("mode_prefix") == "acc_local_l3" else 2,
        row.get("mode_dir") or "",
    ))
    return rows


def build_compare_rows(result_root: str | Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in build_case_rows(result_root):
        key = (
            row.get("workload"),
            row.get("partition_mode"),
            row.get("batch_size"),
            row.get("q_len"),
            row.get("kv_len"),
        )
        grouped[key][row["mode_prefix"]] = row

    rows = []
    for key in sorted(grouped):
        modes = grouped[key]
        balanced = modes.get("balanced", {})
        acc = modes.get("acc_local_l3", {})
        row = {
            "workload": key[0],
            "partition_mode": key[1],
            "batch_size": key[2],
            "q_len": key[3],
            "kv_len": key[4],
            "balanced_mode_dir": balanced.get("mode_dir"),
            "acc_local_l3_mode_dir": acc.get("mode_dir"),
            "balanced_slowest_rank_mean_ms":
            balanced.get("slowest_rank_mean_ms"),
            "acc_local_l3_slowest_rank_mean_ms":
            acc.get("slowest_rank_mean_ms"),
            "slowest_rank_mean_ms_delta_pct":
            _delta_pct(balanced.get("slowest_rank_mean_ms"),
                       acc.get("slowest_rank_mean_ms")),
        }
        for slug, _, _ in PROFILE_DELTA_SPECS:
            base_slug = slug.removesuffix("_delta_pct")
            row[slug] = _delta_pct(balanced.get(base_slug), acc.get(base_slug))
        for prefix, source in (("balanced", balanced), ("acc_local_l3", acc)):
            row[f"{prefix}_note"] = source.get("note")
            row[f"{prefix}_report_json"] = source.get("report_json")
            for slug, metric_name in COMPARE_METRICS:
                row[f"{prefix}_{slug}"] = source.get(metric_name)
            for slug, _, _ in PROFILE_METRIC_SPECS:
                row[f"{prefix}_{slug}"] = source.get(slug)
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], csv_path: Path,
               fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_transposed_rows(compare_rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    paired_headers = []
    for row in compare_rows:
        shape = f"q{row['q_len']}_kv{row['kv_len']}"
        paired_headers.extend([
            _build_case_header(shape, row.get("balanced_mode_dir")),
            _build_case_header(shape, row.get("acc_local_l3_mode_dir")),
        ])
    fieldnames = ["metric", *paired_headers]
    metric_specs = [
        ("slowest rank mean (ms)",
         lambda row: _fmt(row["balanced_slowest_rank_mean_ms"], 6),
         lambda row: _fmt(row["acc_local_l3_slowest_rank_mean_ms"], 6)),
        ("slowest rank mean delta (%)",
         lambda row: "-",
         lambda row: _fmt(row["slowest_rank_mean_ms_delta_pct"], 2)),
    ]
    for slug, metric_name, precision in PROFILE_METRIC_SPECS:
        metric_specs.append((
            metric_name,
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(f"balanced_{slug}"), precision),
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(f"acc_local_l3_{slug}"), precision),
        ))
    for slug, metric_name, precision in PROFILE_DELTA_SPECS:
        metric_specs.append((
            metric_name,
            lambda row: "-",
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(slug), precision),
        ))
    for slug, metric_name, precision in METRIC_SPECS:
        metric_specs.append((
            DISPLAY_METRIC_NAMES.get(metric_name, metric_name),
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(f"balanced_{slug}"), precision),
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(f"acc_local_l3_{slug}"), precision),
        ))
    metric_specs.append((
        "note",
        lambda row: row.get("balanced_note") or "-",
        lambda row: row.get("acc_local_l3_note") or "-",
    ))
    rows = []
    for metric_name, balanced_value_fn, acc_value_fn in metric_specs:
        row = {"metric": metric_name}
        for compare_row in compare_rows:
            shape = f"q{compare_row['q_len']}_kv{compare_row['kv_len']}"
            row[_build_case_header(shape, compare_row.get("balanced_mode_dir"))] = (
                balanced_value_fn(compare_row)
            )
            row[_build_case_header(shape, compare_row.get("acc_local_l3_mode_dir"))] = (
                acc_value_fn(compare_row)
            )
        rows.append(row)
    return fieldnames, rows


def build_detail_transposed_rows(case_rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    case_headers = []
    for row in case_rows:
        shape = f"q{row['q_len']}_kv{row['kv_len']}"
        case_headers.append(f"{shape}_{row['mode_dir']}")
    fieldnames = ["metric", *case_headers]

    detail_metrics = [
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
        *[slug for slug, _, _ in PROFILE_METRIC_SPECS],
        *CASE_METRIC_NAMES,
    ]
    rows = []
    for metric_name in detail_metrics:
        row = {"metric": metric_name}
        for case_row, case_header in zip(case_rows, case_headers):
            value = case_row.get(metric_name)
            if value is None:
                row[case_header] = "-"
            elif isinstance(value, float):
                row[case_header] = str(value)
            else:
                row[case_header] = str(value)
        rows.append(row)
    return fieldnames, rows


def build_compare_transposed_rows(compare_rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    compare_headers = []
    for row in compare_rows:
        shape = f"q{row['q_len']}_kv{row['kv_len']}"
        compare_headers.extend([
            _build_case_header(shape, row.get("balanced_mode_dir")),
            _build_case_header(shape, row.get("acc_local_l3_mode_dir")),
        ])
    fieldnames = ["metric", *compare_headers]
    metric_specs = [
        ("workload",
         lambda row: str(row.get("workload") or "-"),
         lambda row: str(row.get("workload") or "-")),
        ("partition_mode",
         lambda row: str(row.get("partition_mode") or "-"),
         lambda row: str(row.get("partition_mode") or "-")),
        ("batch_size",
         lambda row: str(row.get("batch_size") or "-"),
         lambda row: str(row.get("batch_size") or "-")),
        ("q_len",
         lambda row: str(row.get("q_len") or "-"),
         lambda row: str(row.get("q_len") or "-")),
        ("kv_len",
         lambda row: str(row.get("kv_len") or "-"),
         lambda row: str(row.get("kv_len") or "-")),
        ("mode_dir",
         lambda row: str(row.get("balanced_mode_dir") or "-"),
         lambda row: str(row.get("acc_local_l3_mode_dir") or "-")),
        ("slowest_rank_mean_ms",
         lambda row: _fmt(row.get("balanced_slowest_rank_mean_ms"), 6),
         lambda row: _fmt(row.get("acc_local_l3_slowest_rank_mean_ms"), 6)),
        ("slowest_rank_mean_delta_pct",
         lambda row: "-",
         lambda row: _fmt(row.get("slowest_rank_mean_ms_delta_pct"), 2)),
    ]
    for slug, metric_name, precision in PROFILE_METRIC_SPECS:
        metric_specs.append((
            metric_name,
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(f"balanced_{slug}"), precision),
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(f"acc_local_l3_{slug}"), precision),
        ))
    for slug, metric_name, precision in PROFILE_DELTA_SPECS:
        metric_specs.append((
            metric_name,
            lambda row: "-",
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(slug), precision),
        ))
    for slug, metric_name, precision in METRIC_SPECS:
        metric_specs.append((
            DISPLAY_METRIC_NAMES.get(metric_name, metric_name),
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(f"balanced_{slug}"), precision),
            lambda row, slug=slug, precision=precision: _fmt(
                row.get(f"acc_local_l3_{slug}"), precision),
        ))
    metric_specs.append((
        "note",
        lambda row: str(row.get("balanced_note") or "-"),
        lambda row: str(row.get("acc_local_l3_note") or "-"),
    ))
    rows = []
    for metric_name, balanced_value_fn, acc_value_fn in metric_specs:
        row = {"metric": metric_name}
        for compare_row in compare_rows:
            shape = f"q{compare_row['q_len']}_kv{compare_row['kv_len']}"
            row[_build_case_header(shape, compare_row.get("balanced_mode_dir"))] = (
                balanced_value_fn(compare_row)
            )
            row[_build_case_header(shape, compare_row.get("acc_local_l3_mode_dir"))] = (
                acc_value_fn(compare_row)
            )
        rows.append(row)
    return fieldnames, rows


def write_summary_markdown(compare_rows: list[dict[str, Any]], markdown_path: Path,
                           result_root: Path) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    mode_dirs = []
    for row in compare_rows:
        for mode_dir in (row.get("balanced_mode_dir"), row.get("acc_local_l3_mode_dir")):
            if mode_dir and mode_dir not in mode_dirs:
                mode_dirs.append(mode_dir)
    lines = [
        "# P3 Attention-Only PCM Compare",
        "",
        f"> generated_at: {generated_at}",
        f"> result_root: {result_root}",
        "",
        f"- pair_count: {len(compare_rows)}",
        "- modes: " + ", ".join(f"`{mode_dir}`" for mode_dir in mode_dirs),
        "- key DC metric: `Demand another CCX same node` means `Demand DC Fills From another CCX in same node (pti)`",
        "- key profile metric: `Slowest Rank Attention Effective Threads` = slowest rank `attention_task_body_ns / elapsed_ns`",
        "",
        "## Dry-run + Profile + IPC/CPI + DC/L3",
        "",
    ]
    shape_headers, transposed_rows = build_transposed_rows(compare_rows)
    lines.extend([
        "| metric | " + " | ".join(shape_headers[1:]) + " |",
        "| --- | " + " | ".join(["---"] * (len(shape_headers) - 1)) + " |",
    ])
    for row in transposed_rows:
        values = [row[header] for header in shape_headers[1:]]
        lines.append(f"| {row['metric']} | " + " | ".join(values) + " |")
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: float | None, precision: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def build_outputs(
    result_root: str | Path,
    output_csv: str | Path | None = None,
    output_compare_csv: str | Path | None = None,
    output_md: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    result_root = _as_path(result_root).resolve()
    output_csv = _as_path(output_csv) if output_csv else result_root / "summary.csv"
    output_compare_csv = (_as_path(output_compare_csv)
                          if output_compare_csv else
                          result_root / "compare_summary.csv")
    output_md = _as_path(output_md) if output_md else result_root / "summary.md"
    output_detail_csv = result_root / "detail_summary.csv"

    case_rows = build_case_rows(result_root)
    compare_rows = build_compare_rows(result_root)
    transposed_fieldnames, transposed_rows = build_transposed_rows(compare_rows)
    detail_fieldnames, detail_transposed_rows = build_detail_transposed_rows(
        case_rows)
    compare_fieldnames, compare_transposed_rows = build_compare_transposed_rows(
        compare_rows)
    _write_csv(transposed_rows, output_csv, transposed_fieldnames)
    _write_csv(compare_transposed_rows, output_compare_csv, compare_fieldnames)
    _write_csv(detail_transposed_rows, output_detail_csv, detail_fieldnames)
    write_summary_markdown(compare_rows, output_md, result_root)
    return output_csv, output_compare_csv, output_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize P3 attention-only balanced vs acc-local-l3 PCM results."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=f"default: {DEFAULT_RESULT_ROOT}",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-compare-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv, output_compare_csv, output_md = build_outputs(
        result_root=args.result_root,
        output_csv=args.output_csv,
        output_compare_csv=args.output_compare_csv,
        output_md=args.output_md,
    )
    print(f"summary_csv={output_csv}")
    print(f"compare_summary_csv={output_compare_csv}")
    print(f"summary_md={output_md}")


if __name__ == "__main__":
    main()
