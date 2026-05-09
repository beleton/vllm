from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

TARGET_MODES = ("balanced", "acc-local-l3")
DEFAULT_OUTPUT_NAME = "cat_compare_summary.csv"


def _load_dry_run(summary_path: Path) -> dict[str, Any]:
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
        "slowest_rank_mean_ms": data.get("slowest_rank_mean_ms"),
    }


def _pct_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return (float(value) / float(baseline) - 1.0) * 100.0


def discover_cat_masks(result_root: str | Path) -> list[str]:
    result_root = Path(result_root)
    masks = {
        summary_path.parent.name
        for summary_path in result_root.glob(
            "cat/q*_kv*/**/dry_run_summary.json")
        if summary_path.parent.parent.parent.parent.name == "cat"
    }
    return sorted(masks)


def collect_rows(result_root: str | Path) -> list[dict[str, Any]]:
    result_root = Path(result_root)
    masks = discover_cat_masks(result_root)
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(result_root.glob("q*_kv*/**/dry_run_summary.json")):
        case_dir = summary_path.parent
        shape_dir = case_dir.parent
        if shape_dir.name == "cat":
            continue

        baseline = _load_dry_run(summary_path)
        mode = baseline["mode"]
        if mode not in TARGET_MODES:
            continue

        row: dict[str, Any] = {
            "workload": baseline["workload"],
            "partition_mode": baseline["partition_mode"],
            "batch_size": baseline["batch_size"],
            "q_len": baseline["q_len"],
            "kv_len": baseline["kv_len"],
            "mode": mode,
            "group_span": baseline["group_span"],
            "baseline_slowest_rank_mean_ms": baseline["slowest_rank_mean_ms"],
            "baseline_case_dir": str(case_dir),
        }
        for mask in masks:
            cat_summary = (result_root / "cat" / shape_dir.name / mode / mask
                           / "dry_run_summary.json")
            cat_value = None
            cat_case_dir = ""
            if cat_summary.exists():
                cat_case = _load_dry_run(cat_summary)
                cat_value = cat_case["slowest_rank_mean_ms"]
                cat_case_dir = str(cat_summary.parent)
            row[f"cat_{mask}_slowest_rank_mean_ms"] = cat_value
            row[f"cat_{mask}_delta_pct"] = _pct_delta(
                cat_value,
                baseline["slowest_rank_mean_ms"],
            )
            row[f"cat_{mask}_case_dir"] = cat_case_dir
        rows.append(row)

    rows.sort(key=lambda item: (item["q_len"], item["kv_len"],
                                TARGET_MODES.index(item["mode"])))
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary(result_root: str | Path,
                  output_path: str | Path | None = None) -> Path:
    result_root = Path(result_root)
    masks = discover_cat_masks(result_root)
    rows = collect_rows(result_root)
    output_path = (Path(output_path) if output_path is not None else
                   result_root / DEFAULT_OUTPUT_NAME)

    fieldnames = [
        "workload",
        "partition_mode",
        "batch_size",
        "group_span",
        "q_len",
        "kv_len",
        "mode",
        "baseline_slowest_rank_mean_ms",
    ]
    for mask in masks:
        fieldnames.extend([
            f"cat_{mask}_slowest_rank_mean_ms",
            f"cat_{mask}_delta_pct",
        ])
    fieldnames.extend([
        "baseline_case_dir",
    ])
    for mask in masks:
        fieldnames.append(f"cat_{mask}_case_dir")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _format_value(row.get(key))
                for key in fieldnames
            })
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare span baseline dry-run results against CAT masks.")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    output_path = write_summary(args.result_root, args.output)
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
