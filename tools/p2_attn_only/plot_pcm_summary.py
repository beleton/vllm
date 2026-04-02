from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.p2_attn_only.pcm_summary import LATENCY_BREAKDOWN_METRIC_NAMES

INT_FIELDS = {"batch_size", "q_len", "kv_len", "tp_size"}
FLOAT_FIELDS = {
    "slowest_rank_mean_ms",
    "global_num_query_heads",
    "global_num_kv_heads",
    "local_num_query_heads",
    "local_num_kv_heads",
    "L3 Access (pti)",
    "L3 Miss (pti)",
    "L3 Miss %",
    "Ave L3 Miss Latency (ns)",
    *LATENCY_BREAKDOWN_METRIC_NAMES,
}

PLOT_SPECS = [
    ("decode-like", "L3 Miss %", "decode_l3_miss_vs_kv_len.png"),
    ("decode-like", "Ave L3 Miss Latency (ns)",
     "decode_l3_latency_vs_kv_len.png"),
    ("prefill-like", "L3 Miss %", "prefill_l3_miss_vs_q_len.png"),
    ("prefill-like", "Ave L3 Miss Latency (ns)",
     "prefill_l3_latency_vs_q_len.png"),
]


def _parse_value(key: str, value: str) -> Any:
    if value == "":
        return None
    if key in INT_FIELDS:
        return int(float(value))
    if key in FLOAT_FIELDS:
        return float(value)
    return value


def load_summary_rows(summary_csv: str | Path) -> list[dict[str, Any]]:
    summary_csv = Path(summary_csv)
    with summary_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{key: _parse_value(key, value)
                 for key, value in row.items()} for row in reader]


def sort_rows_for_plot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    workload = rows[0]["workload"]
    if workload == "decode-like":
        return sorted(rows, key=lambda row: (row["kv_len"], row["q_len"]))
    return sorted(rows, key=lambda row: (row["q_len"], row["kv_len"]))


def get_x_values(rows: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    workload = rows[0]["workload"]
    if workload == "decode-like":
        return [row["kv_len"] for row in rows]
    return [row["q_len"] for row in rows]


def get_x_label(workload: str) -> str:
    if workload == "decode-like":
        return "KV Length"
    return "Q Length"


def get_latency_breakdown_values(row: dict[str, Any]) -> list[float]:
    return [float(row[metric]) for metric in LATENCY_BREAKDOWN_METRIC_NAMES]


def plot_metric(rows: list[dict[str, Any]], workload: str, metric_name: str,
                output_path: Path) -> None:
    rows = sort_rows_for_plot(rows)
    x_values = get_x_values(rows)
    y_values = [row[metric_name] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel(get_x_label(workload))
    plt.ylabel(metric_name)
    plt.title(f"{workload}: {metric_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_latency_breakdown(rows: list[dict[str, Any]], workload: str,
                           output_path: Path) -> None:
    rows = sort_rows_for_plot(rows)
    x_values = get_x_values(rows)
    positions = list(range(len(rows)))
    bottoms = [0.0] * len(rows)

    plt.figure(figsize=(9, 5.5))
    for metric in LATENCY_BREAKDOWN_METRIC_NAMES:
        values = [row[metric] for row in rows]
        plt.bar(
            positions,
            values,
            bottom=bottoms,
            label=metric,
            width=0.7,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    plt.xlabel(get_x_label(workload))
    plt.ylabel("Latency Breakdown (%)")
    plt.title(f"{workload}: L3 Miss Latency Breakdown")
    plt.ylim(0, 100)
    plt.xticks(positions, x_values)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_plots(summary_csv: str | Path,
                   output_dir: str | Path | None = None) -> list[Path]:
    summary_csv = Path(summary_csv).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else summary_csv.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary_rows(summary_csv)
    rows_by_workload: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_workload.setdefault(row["workload"], []).append(row)

    generated = []
    for workload, metric_name, filename in PLOT_SPECS:
        if workload not in rows_by_workload:
            continue
        output_path = output_dir / filename
        plot_metric(rows_by_workload[workload], workload, metric_name,
                    output_path)
        generated.append(output_path)

    for workload, filename in (
        ("decode-like", "decode_l3_latency_breakdown_vs_kv_len.png"),
        ("prefill-like", "prefill_l3_latency_breakdown_vs_q_len.png"),
    ):
        if workload not in rows_by_workload:
            continue
        output_path = output_dir / filename
        plot_latency_breakdown(rows_by_workload[workload], workload,
                               output_path)
        generated.append(output_path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot summarized attention-only PCM metrics.")
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = generate_plots(args.summary_csv, args.output_dir)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
