from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TARGET_METRIC = "slowest rank mean (ms)"
TARGET_MODES = ("balanced", "acc-local-l3")
DEFAULT_OUTPUT_NAME = "slowest_rank_mean_balanced_vs_acc-local-l3.png"

CASE_COLUMN_RE = re.compile(r"^q(?P<q_len>\d+)_kv(?P<kv_len>\d+)_(?P<mode>.+)$")


def parse_case_column(column_name: str) -> tuple[int, int, str]:
    match = CASE_COLUMN_RE.fullmatch(column_name)
    if match is None:
        raise ValueError(f"Unsupported summary column: {column_name}")
    return (
        int(match.group("q_len")),
        int(match.group("kv_len")),
        match.group("mode"),
    )


def normalize_mode_name(mode_name: str) -> str:
    for target_mode in TARGET_MODES:
        if mode_name == target_mode or mode_name.startswith(f"{target_mode}_"):
            return target_mode
    return mode_name


def load_slowest_rank_mean_series(
        summary_csv: str | Path) -> dict[str, list[tuple[int, int, float]]]:
    summary_csv = Path(summary_csv)
    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"Empty summary csv: {summary_csv}")

    header = rows[0]
    metric_row = None
    for row in rows[1:]:
        if row and row[0] == TARGET_METRIC:
            metric_row = row
            break
    if metric_row is None:
        raise ValueError(f"Missing metric row: {TARGET_METRIC}")

    series = {mode: [] for mode in TARGET_MODES}
    for column_name, value in zip(header[1:], metric_row[1:]):
        if value in {"", "-"}:
            continue
        q_len, kv_len, mode = parse_case_column(column_name)
        mode = normalize_mode_name(mode)
        if mode not in series:
            continue
        series[mode].append((q_len, kv_len, float(value)))

    for mode in series:
        series[mode].sort(key=lambda item: (item[0], item[1]))
    return series


def _shape_label(q_len: int, kv_len: int) -> str:
    if q_len == kv_len:
        return str(q_len)
    return f"q{q_len}/kv{kv_len}"


def make_plot_title(summary_csv: str | Path) -> str:
    summary_csv = Path(summary_csv)
    batch_dir = summary_csv.parent.name
    workload = next(
        (part for part in summary_csv.parts if part in ("prefill-like", "decode-like")),
        None,
    )
    if workload is None:
        return TARGET_METRIC
    return f"{workload.capitalize()} {batch_dir}"


def generate_plot(summary_csv: str | Path,
                  output_path: str | Path | None = None) -> Path:
    summary_csv = Path(summary_csv).resolve()
    output_path = (Path(output_path).resolve()
                   if output_path else summary_csv.with_name(DEFAULT_OUTPUT_NAME))

    series = load_slowest_rank_mean_series(summary_csv)
    balanced_points = series["balanced"]
    acc_points = series["acc-local-l3"]
    if not balanced_points or not acc_points:
        raise ValueError("Both balanced and acc-local-l3 series are required")

    shape_keys = [(q_len, kv_len) for q_len, kv_len, _ in balanced_points]
    acc_shape_keys = [(q_len, kv_len) for q_len, kv_len, _ in acc_points]
    if shape_keys != acc_shape_keys:
        raise ValueError("balanced and acc-local-l3 shapes do not align")

    x_positions = list(range(len(shape_keys)))
    x_labels = [_shape_label(q_len, kv_len) for q_len, kv_len in shape_keys]
    balanced_values = [value for _, _, value in balanced_points]
    acc_values = [value for _, _, value in acc_points]

    bar_width = 0.36
    balanced_positions = [x - bar_width / 2 for x in x_positions]
    acc_positions = [x + bar_width / 2 for x in x_positions]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.bar(
        balanced_positions,
        balanced_values,
        width=bar_width,
        label="balanced",
        color="#1f77b4",
    )
    ax.bar(
        acc_positions,
        acc_values,
        width=bar_width,
        label="acc-local-l3",
        color="#d62728",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Sequence length (q=kv)")
    ax.set_ylabel("mean (ms)")
    ax.set_title(make_plot_title(summary_csv))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=True)

    for x_pos, value in zip(balanced_positions, balanced_values):
        ax.annotate(
            f"{value:.3f}",
            (x_pos, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    for x_pos, value in zip(acc_positions, acc_values):
        ax.annotate(
            f"{value:.3f}",
            (x_pos, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Plot slowest rank mean comparison between balanced and "
                     "acc-local-l3 from a P3 summary.csv."))
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    output_path = generate_plot(args.summary_csv, args.output)
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
