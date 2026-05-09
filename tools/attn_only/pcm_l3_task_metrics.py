from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_RESULT_ROOT = Path(
    "test_results/P4_AttnOnly_L3Residency/Qwen3-30B-A3B/"
    "qhead_32_kvhead_16/NPS1_TP2/prefill-like/global-fixed/batch_1/span1"
)

FIELDNAMES = [
    "case",
    "mode",
    "q_len",
    "kv_len",
    "case_dir",
    "pcm_source",
    "pcm_report",
    "benchmark_runtime_s",
    "collected_s",
    "total_attention_task_count",
    "estimated_sampled_task_count",
    "l3_miss",
    "l3_access",
    "l3_miss_per_second",
    "l3_access_per_second",
    "l3_miss_per_task",
    "l3_access_per_task",
    "ipc_sys_user",
    "l2_access_pti",
    "l2_miss_pti",
    "l3_access_pti",
    "l3_miss_pti",
    "all_demand_dc_fills_pti",
    "demand_dc_fills_local_l2_pti",
    "demand_dc_fills_local_l3_same_ccx_pti",
    "demand_dc_fills_another_ccx_same_node_pti",
    "demand_dc_fills_local_memory_io_pti",
    "attention_task_body_avg_ns",
    "note",
]

CSV_PRECISION = {
    "benchmark_runtime_s": 6,
    "collected_s": 6,
    "total_attention_task_count": 2,
    "estimated_sampled_task_count": 2,
    "l3_miss": 0,
    "l3_access": 0,
    "l3_miss_per_second": 2,
    "l3_access_per_second": 2,
    "l3_miss_per_task": 2,
    "l3_access_per_task": 2,
    "ipc_sys_user": 4,
    "l2_access_pti": 4,
    "l2_miss_pti": 4,
    "l3_access_pti": 4,
    "l3_miss_pti": 4,
    "all_demand_dc_fills_pti": 4,
    "demand_dc_fills_local_l2_pti": 4,
    "demand_dc_fills_local_l3_same_ccx_pti": 4,
    "demand_dc_fills_another_ccx_same_node_pti": 4,
    "demand_dc_fills_local_memory_io_pti": 4,
    "attention_task_body_avg_ns": 2,
}

STANDARD_PCM_AVG_METRICS = {
    "ipc_sys_user": ("IPC (Sys + User)", "ipc"),
    "l2_access_pti": ("L2 Access (pti)", "l2"),
    "l2_miss_pti": ("L2 Miss (pti)", "l2"),
    "l3_access_pti": ("L3 Access (pti)", "l3"),
    "l3_miss_pti": ("L3 Miss (pti)", "l3"),
    "all_demand_dc_fills_pti": ("All Demand DC Fills (pti)", "dc"),
    "demand_dc_fills_local_l2_pti": ("Demand DC Fills From Local L2 (pti)",
                                     "dc"),
    "demand_dc_fills_local_l3_same_ccx_pti": (
        "Demand DC Fills From Local L3 or different L2 in same CCX (pti)",
        "dc",
    ),
    "demand_dc_fills_another_ccx_same_node_pti": (
        "Demand DC Fills From another CCX in same node (pti)",
        "dc",
    ),
    "demand_dc_fills_local_memory_io_pti": (
        "Demand DC Fills From Local Memory or I/O (pti)",
        "dc",
    ),
}


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _parse_shape_dirname(shape_dirname: str) -> dict[str, int] | None:
    match = re.fullmatch(
        r"q(?P<q_len>\d+)_kv_?(?P<kv_len>\d+)",
        shape_dirname,
        re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "q_len": int(match.group("q_len")),
        "kv_len": int(match.group("kv_len")),
    }


def _safe_div(numerator: float | None,
              denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


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


def _latest_report_cumulative_csv(case_dir: Path) -> Path | None:
    return _latest_path(
        list(case_dir.glob("pcm_l3_source_latency_raw/*/report-cumulative.csv"))
    )


def _latest_custom_report_json(case_dir: Path) -> Path | None:
    return _latest_path(list(case_dir.glob("pcm_l3_source_latency_raw/*/report.json")))


def _latest_standard_report_json(case_dir: Path) -> Path | None:
    return _latest_path(list(case_dir.glob("pcm_l3_dc/*/report.json")))


def _header_index(row: list[str], target: str) -> int | None:
    for idx, value in enumerate(row):
        if value.strip() == target:
            return idx
    return None


def extract_custom_system_metrics(report_csv_path: str | Path) -> dict[str, float]:
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
            if system_column_idx is None or system_column_idx >= len(row):
                continue
            if metric_name not in {"L3 Miss", "L3 Access", "L3 Miss / second"}:
                continue
            value = _to_float(row[system_column_idx])
            if value is not None:
                metrics[metric_name] = value

    return metrics


def _load_profile(profile_path: Path) -> dict[str, float | None]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    runtime = (profile.get("profile_summary") or {}).get("runtime") or {}
    slowest_rank_mean_ms = _to_float(profile.get("slowest_rank_mean_ms"))
    if slowest_rank_mean_ms is None:
        rank_times = [
            _to_float((item.get("result") or {}).get("time_mean_ms"))
            for item in profile.get("rank_results", [])
        ]
        rank_times = [value for value in rank_times if value is not None]
        slowest_rank_mean_ms = max(rank_times) if rank_times else None

    call_count = _to_float(runtime.get("call_count"))
    task_count = _to_float(runtime.get("attention_task_count"))
    task_body_avg_ns = _to_float(runtime.get("attention_task_body_avg_ns"))
    if task_body_avg_ns is None:
        task_body_avg_ns = _safe_div(
            _to_float(runtime.get("attention_task_body_ns")),
            task_count,
        )

    benchmark_runtime_s = None
    if slowest_rank_mean_ms is not None and call_count is not None:
        benchmark_runtime_s = slowest_rank_mean_ms * call_count / 1000.0

    return {
        "benchmark_runtime_s": benchmark_runtime_s,
        "total_attention_task_count": task_count,
        "attention_task_body_avg_ns": task_body_avg_ns,
    }


def _metadata_value(report: dict[str, Any], name: str) -> str | None:
    for item in report.get("metadata", []):
        if item.get("name") == name:
            value = item.get("value")
            return str(value) if value is not None else None
    return None


def _metric_targets_system(metric: dict[str, Any],
                           group_key: str | None) -> bool:
    for group_entry in metric.get("group", []):
        entries_by_key = ([group_entry.get(group_key)]
                          if group_key is not None else group_entry.values())
        for entries in entries_by_key:
            if not entries:
                continue
            entry = entries[0]
            if entry.get("name") == "system" and entry.get("id", -1) == -1:
                return True
    return False


def _system_metric(report: dict[str, Any], metric_name: str,
                   group_key: str | None) -> dict[str, Any] | None:
    for metric in report.get("metrics", []):
        if metric.get("name") != metric_name:
            continue
        if not _metric_targets_system(metric, group_key):
            continue
        return metric
    return None


def _series_from_metric(metric: dict[str, Any]) -> list[float]:
    series = []
    for value in metric.get("series", []):
        number = _to_float(value)
        if number is not None:
            series.append(number)
    return series


def _system_series(report: dict[str, Any], metric_name: str,
                   group_key: str | None) -> list[float]:
    metric = _system_metric(report, metric_name, group_key)
    if metric is not None:
        return _series_from_metric(metric)
    raise KeyError(f"missing metric series: {metric_name}")


def _metric_sum(metric: dict[str, Any]) -> float | None:
    aggregated_sum = _to_float((metric.get("aggregated") or {}).get("sum"))
    if aggregated_sum is not None:
        return aggregated_sum
    series = _series_from_metric(metric)
    if series:
        return sum(series)
    return None


def _metric_average(metric: dict[str, Any]) -> float | None:
    average = _to_float((metric.get("aggregated") or {}).get("average"))
    if average is not None:
        return average
    series = _series_from_metric(metric)
    if series:
        return sum(series) / len(series)
    return None


def _sample_interval_s(report: dict[str, Any]) -> float | None:
    interval_ms = _to_float(_metadata_value(report, "SAMPLE_INTERVAL (ms)"))
    if interval_ms in (None, 0):
        return None
    return interval_ms / 1000.0


def extract_custom_report_json_counts(
        report_json_path: str | Path) -> dict[str, float] | None:
    report_json_path = _as_path(report_json_path)
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    miss_metric = _system_metric(report, "L3 Miss", "l3")
    access_metric = _system_metric(report, "L3 Access", "l3")
    if miss_metric is None or access_metric is None:
        return None

    l3_miss = _metric_sum(miss_metric)
    l3_access = _metric_sum(access_metric)
    if l3_miss is None or l3_access is None:
        return None

    miss_per_second_metric = _system_metric(report, "L3 Miss / second", "l3")
    l3_miss_per_second = (
        _metric_average(miss_per_second_metric)
        if miss_per_second_metric is not None else None
    )
    interval_s = _sample_interval_s(report)
    sample_count = len(_series_from_metric(miss_metric))
    collected_s = (
        sample_count * interval_s
        if sample_count > 0 and interval_s is not None else
        _safe_div(l3_miss, l3_miss_per_second)
    )
    if collected_s in (None, 0):
        return None

    return {
        "collected_s": collected_s,
        "l3_miss": l3_miss,
        "l3_access": l3_access,
        "l3_miss_per_second": (
            l3_miss_per_second
            if l3_miss_per_second is not None else l3_miss / collected_s
        ),
        "l3_access_per_second": l3_access / collected_s,
    }


def extract_standard_l3_counts(report_json_path: str | Path) -> dict[str, float]:
    report_json_path = _as_path(report_json_path)
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    interval_s = _sample_interval_s(report)
    if interval_s in (None, 0):
        raise ValueError(f"missing SAMPLE_INTERVAL (ms): {report_json_path}")

    gips = _system_series(report, "Giga Instructions Per Sec", "ipc")
    miss_pti = _system_series(report, "L3 Miss (pti)", "l3")
    access_pti = _system_series(report, "L3 Access (pti)", "l3")
    sample_count = min(len(gips), len(miss_pti), len(access_pti))
    if sample_count == 0:
        raise ValueError(f"empty PCM series: {report_json_path}")

    l3_miss = sum(
        miss_pti[idx] * gips[idx] * interval_s * 1_000_000
        for idx in range(sample_count)
    )
    l3_access = sum(
        access_pti[idx] * gips[idx] * interval_s * 1_000_000
        for idx in range(sample_count)
    )
    collected_s = sample_count * interval_s
    return {
        "collected_s": collected_s,
        "l3_miss": l3_miss,
        "l3_access": l3_access,
        "l3_miss_per_second": l3_miss / collected_s,
        "l3_access_per_second": l3_access / collected_s,
    }


def _build_custom_pcm_values(case_dir: Path,
                             profile_metrics: dict[str, float | None],
                             report_csv: Path) -> dict[str, float | str | None]:
    metrics = extract_custom_system_metrics(report_csv)
    l3_miss = metrics.get("L3 Miss")
    l3_access = metrics.get("L3 Access")
    l3_miss_per_second = metrics.get("L3 Miss / second")
    collected_s = _safe_div(l3_miss, l3_miss_per_second)
    if collected_s is None:
        collected_s = profile_metrics["benchmark_runtime_s"]
    l3_access_per_second = _safe_div(l3_access, collected_s)
    return {
        "pcm_source": "custom_cumulative_csv",
        "pcm_report": str(report_csv),
        "collected_s": collected_s,
        "l3_miss": l3_miss,
        "l3_access": l3_access,
        "l3_miss_per_second": l3_miss_per_second,
        "l3_access_per_second": l3_access_per_second,
        "note": "latest pcm_l3_source_latency_raw session",
    }


def _build_standard_pcm_values(
        report_json: Path) -> dict[str, float | str | None]:
    counts = extract_standard_l3_counts(report_json)
    averages = extract_standard_pcm_averages(report_json)
    return {
        "pcm_source": "standard_report_json",
        "pcm_report": str(report_json),
        "note": "latest pcm_l3_dc session; count inferred from pti and GIPS",
        **counts,
        **averages,
    }


def extract_standard_pcm_averages(
        report_json_path: str | Path) -> dict[str, float | None]:
    report_json_path = _as_path(report_json_path)
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    averages: dict[str, float | None] = {}
    for fieldname, (metric_name, group_key) in STANDARD_PCM_AVG_METRICS.items():
        metric = _system_metric(report, metric_name, group_key)
        averages[fieldname] = (
            _metric_average(metric) if metric is not None else None
        )
    return averages


def _build_custom_report_json_values(
        report_json: Path) -> dict[str, float | str | None] | None:
    counts = extract_custom_report_json_counts(report_json)
    if counts is None:
        return None
    return {
        "pcm_source": "custom_report_json",
        "pcm_report": str(report_json),
        "note": "latest pcm_l3_source_latency_raw report.json session",
        **counts,
    }


def _build_missing_pcm_values() -> dict[str, float | str | None]:
    return {
        "pcm_source": "missing",
        "pcm_report": None,
        "collected_s": None,
        "l3_miss": None,
        "l3_access": None,
        "l3_miss_per_second": None,
        "l3_access_per_second": None,
        "note": "missing pcm_l3_source_latency_raw and pcm_l3_dc reports",
    }


def _empty_standard_pcm_averages() -> dict[str, float | None]:
    return {
        fieldname: None
        for fieldname in STANDARD_PCM_AVG_METRICS
    }


def _empty_profile_metrics() -> dict[str, float | None]:
    return {
        "benchmark_runtime_s": None,
        "total_attention_task_count": None,
        "attention_task_body_avg_ns": None,
    }


def _append_note(base_note: str | None, extra_note: str | None) -> str | None:
    if base_note and extra_note:
        return f"{base_note}; {extra_note}"
    return base_note or extra_note


def _discover_case_dirs(result_root: Path) -> list[Path]:
    case_dirs: list[Path] = []
    seen: set[Path] = set()
    patterns = [
        "q*_kv*/*",
        "span*/q*_kv*/*",
    ]
    for pattern in patterns:
        for candidate in sorted(result_root.glob(pattern)):
            if candidate in seen:
                continue
            seen.add(candidate)
            if not candidate.is_dir():
                continue
            shape = _parse_shape_dirname(candidate.parent.name)
            if shape is None and candidate.parent.parent != result_root:
                shape = _parse_shape_dirname(candidate.parent.parent.name)
            if shape is None:
                continue
            profile_json = candidate / "profile.json"
            has_profile = profile_json.exists()
            has_custom_csv = _latest_report_cumulative_csv(candidate) is not None
            has_custom_json = _latest_custom_report_json(candidate) is not None
            has_standard_json = _latest_standard_report_json(candidate) is not None
            if has_profile or has_custom_csv or has_custom_json or has_standard_json:
                case_dirs.append(candidate)
    return case_dirs


def _case_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    mode = str(row.get("mode") or "")
    return (
        row.get("q_len") or -1,
        row.get("kv_len") or -1,
        0 if mode.startswith("balanced") else
        1 if mode.startswith("acc-local-l3") else 2,
        mode,
    )


def build_rows(result_root: str | Path) -> list[dict[str, Any]]:
    result_root = _as_path(result_root).resolve()
    rows: list[dict[str, Any]] = []
    for case_dir in _discover_case_dirs(result_root):
        shape = _parse_shape_dirname(case_dir.parent.name)
        if shape is None and case_dir.parent.parent != result_root:
            shape = _parse_shape_dirname(case_dir.parent.parent.name)
        if shape is None:
            continue

        profile_path = case_dir / "profile.json"
        profile_metrics = (
            _load_profile(profile_path)
            if profile_path.exists() else _empty_profile_metrics()
        )
        custom_report = _latest_report_cumulative_csv(case_dir)
        custom_report_json = _latest_custom_report_json(case_dir)
        standard_report = _latest_standard_report_json(case_dir)
        standard_averages = (
            extract_standard_pcm_averages(standard_report)
            if standard_report is not None else _empty_standard_pcm_averages()
        )
        if custom_report is not None:
            pcm_values = _build_custom_pcm_values(case_dir, profile_metrics,
                                                  custom_report)
        elif custom_report_json is not None:
            custom_json_values = _build_custom_report_json_values(
                custom_report_json)
            pcm_values = (
                custom_json_values
                if custom_json_values is not None else
                _build_standard_pcm_values(standard_report)
                if standard_report is not None else _build_missing_pcm_values()
            )
        elif standard_report is not None:
            pcm_values = _build_standard_pcm_values(standard_report)
        else:
            pcm_values = _build_missing_pcm_values()

        collected_s = _to_float(pcm_values.get("collected_s"))
        benchmark_runtime_s = profile_metrics["benchmark_runtime_s"]
        task_count = profile_metrics["total_attention_task_count"]
        sampled_task_count = None
        if (task_count is not None and collected_s is not None and
                benchmark_runtime_s not in (None, 0)):
            sampled_task_count = _safe_div(task_count * collected_s,
                                           benchmark_runtime_s)

        l3_miss = _to_float(pcm_values.get("l3_miss"))
        l3_access = _to_float(pcm_values.get("l3_access"))
        note = pcm_values.get("note")
        if not profile_path.exists():
            note = _append_note(
                str(note) if note is not None else None,
                "missing profile.json, per-task metrics unavailable",
            )
        rows.append({
            "case": case_dir.parent.name,
            "mode": case_dir.name,
            "q_len": shape["q_len"],
            "kv_len": shape["kv_len"],
            "case_dir": str(case_dir),
            **profile_metrics,
            **pcm_values,
            **standard_averages,
            "note": note,
            "estimated_sampled_task_count": sampled_task_count,
            "l3_miss_per_task": _safe_div(l3_miss, sampled_task_count),
            "l3_access_per_task": _safe_div(l3_access, sampled_task_count),
        })

    rows.sort(key=_case_sort_key)
    return rows


def _format_csv_value(fieldname: str, value: Any) -> str:
    if value is None:
        return ""
    if fieldname in CSV_PRECISION:
        number = _to_float(value)
        if number is None:
            return ""
        return f"{number:.{CSV_PRECISION[fieldname]}f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    output_csv = _as_path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                fieldname: _format_csv_value(fieldname, row.get(fieldname))
                for fieldname in FIELDNAMES
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize L3 Miss/Access per attention task from PCM reports.")
    parser.add_argument("--result-root",
                        type=Path,
                        default=DEFAULT_RESULT_ROOT,
                        help="Root containing q*_kv*/<mode> PCM result directories.")
    parser.add_argument("--output-csv",
                        type=Path,
                        default=None,
                        help="Output CSV path. Defaults to RESULT_ROOT/l3_task_metrics.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or args.result_root / "l3_task_metrics.csv"
    rows = build_rows(args.result_root)
    write_csv(rows, output_csv)
    print(f"wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
