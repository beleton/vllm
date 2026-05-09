from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TARGET_METRICS = (
    "All Demand DC Fills (pti)",
    "Demand DC Fills From Local L3 or different L2 in same CCX (pti)",
    "Demand DC Fills From Local Memory or I/O (pti)",
)
TOTAL_METRIC = TARGET_METRICS[0]
LOCAL_L3_METRIC = TARGET_METRICS[1]
LOCAL_MEMORY_METRIC = TARGET_METRICS[2]
TARGET_MODES = ("balanced", "acc-local-l3")
DEFAULT_OUTPUT_NAME = "selected_demand_dc_fills_balanced_vs_acc-local-l3.png"
IMAGE_SIZE = (1360, 500)

CASE_COLUMN_RE = re.compile(r"^q(?P<q_len>\d+)_kv(?P<kv_len>\d+)_(?P<mode>.+)$")

STACK_SEGMENTS = (
    ("Other demand fills", "#9aa0a6"),
    ("Local L3 / same CCX", "#4e79a7"),
    ("Local Memory / I/O", "#f28e2b"),
)

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)

GRID = "#d9d9d9"
TEXT = "#111111"
AXIS = "#444444"
BORDER = "#6b7280"


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


def _load_rows(summary_csv: Path) -> tuple[list[str], list[list[str]]]:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty summary csv: {summary_csv}")
    return rows[0], rows[1:]


def load_metric_series(
        summary_csv: str | Path
) -> dict[str, dict[str, list[tuple[int, int, float]]]]:
    summary_csv = Path(summary_csv)
    header, rows = _load_rows(summary_csv)

    row_by_metric = {}
    for row in rows:
        if row:
            row_by_metric[row[0]] = row

    series = {
        metric: {
            mode: []
            for mode in TARGET_MODES
        }
        for metric in TARGET_METRICS
    }
    for metric in TARGET_METRICS:
        metric_row = row_by_metric.get(metric)
        if metric_row is None:
            raise ValueError(f"Missing metric row: {metric}")
        for column_name, value in zip(header[1:], metric_row[1:]):
            if value in {"", "-"}:
                continue
            q_len, kv_len, mode = parse_case_column(column_name)
            mode = normalize_mode_name(mode)
            if mode not in TARGET_MODES:
                continue
            series[metric][mode].append((q_len, kv_len, float(value)))

    for metric in TARGET_METRICS:
        for mode in TARGET_MODES:
            series[metric][mode].sort(key=lambda item: (item[0], item[1]))
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
        return "Selected demand DC fills"
    return f"{workload.capitalize()} {batch_dir}: selected demand DC fills"


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str,
               font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_centered_text(draw: ImageDraw.ImageDraw, center_x: float, top_y: float,
                        text: str,
                        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
                        fill: str = TEXT) -> None:
    width, height = _text_size(draw, text, font)
    draw.text((center_x - width / 2, top_y), text, font=font, fill=fill)


def _draw_right_text(draw: ImageDraw.ImageDraw, right_x: float, top_y: float,
                     text: str,
                     font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
                     fill: str = TEXT) -> None:
    width, _ = _text_size(draw, text, font)
    draw.text((right_x - width, top_y), text, font=font, fill=fill)


def _draw_legend(draw: ImageDraw.ImageDraw,
                 body_font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> None:
    start_x = 720
    top_y = 16
    box_size = 14
    for index, (label, color) in enumerate(STACK_SEGMENTS):
        x = start_x + index * 185
        draw.rectangle((x, top_y, x + box_size, top_y + box_size),
                       fill=color, outline=color)
        draw.text((x + box_size + 8, top_y - 2), label, font=body_font, fill=TEXT)
    draw.text((28, top_y - 2), "each group: left balanced, right acc-local-l3",
              font=body_font, fill=AXIS)


def compute_bar_layout(plot_width: float, group_count: int) -> dict[str, float]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    cell_width = plot_width / group_count
    group_width = min(210.0, cell_width * 0.70)
    gap = max(8.0, round(group_width * 0.07, 2))
    bar_width = round((group_width - gap) / 2, 2)
    return {
        "cell_width": round(cell_width, 2),
        "group_width": round(group_width, 2),
        "gap": gap,
        "bar_width": bar_width,
    }


def _draw_stacked_chart(
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        cases: list[dict[str, object]],
        title_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        body_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        tick_font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y1),
                           radius=12,
                           outline="#cfcfcf",
                           width=1,
                           fill="#ffffff")
    _draw_centered_text(draw, (x0 + x1) / 2, y0 + 10,
                        "All Demand DC Fills stacked by selected sources",
                        title_font)

    plot_left = x0 + 58
    plot_top = y0 + 48
    plot_right = x1 - 12
    plot_bottom = y1 - 78
    plot_height = plot_bottom - plot_top
    plot_width = plot_right - plot_left

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS, width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS, width=1)
    y_max = max(
        bar["total"] for case in cases for bar in case["bars"]) if cases else 0.0
    y_limit = y_max * 1.16 if y_max > 0 else 1.0

    tick_count = 4
    for tick_index in range(tick_count + 1):
        ratio = tick_index / tick_count
        y = plot_bottom - ratio * plot_height
        value = y_limit * ratio
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        label = f"{value:.2f}" if y_limit < 10 else f"{value:.1f}"
        _draw_right_text(draw, plot_left - 8, y - 8, label, tick_font, AXIS)
    draw.text((plot_left - 34, plot_top - 28), "pti", font=tick_font, fill=AXIS)

    group_count = len(cases)
    if group_count == 0:
        raise ValueError("No valid cases to draw")
    layout = compute_bar_layout(plot_width, group_count)
    cell_width = layout["cell_width"]
    gap = layout["gap"]
    bar_width = layout["bar_width"]

    for index, case in enumerate(cases):
        center_x = plot_left + cell_width * (index + 0.5)
        bar_positions = (
            (center_x - gap / 2 - bar_width, center_x - gap / 2, "bal"),
            (center_x + gap / 2, center_x + gap / 2 + bar_width, "acc"),
        )
        for bar, (bar_x0, bar_x1, mode_label) in zip(case["bars"], bar_positions):
            current_bottom = plot_bottom
            for segment_value, (_, color) in zip(bar["segments"], STACK_SEGMENTS):
                segment_height = (segment_value / y_limit) * plot_height
                segment_top = current_bottom - segment_height
                if segment_height > 0:
                    draw.rectangle((bar_x0, segment_top, bar_x1, current_bottom),
                                   fill=color, outline=color)
                current_bottom = segment_top
            draw.rectangle((bar_x0, current_bottom, bar_x1, plot_bottom),
                           outline=BORDER, width=1)
            label_y = max(plot_top + 2, current_bottom - 16)
            _draw_centered_text(draw, (bar_x0 + bar_x1) / 2, label_y,
                                f"{bar['total']:.2f}", tick_font)
            _draw_centered_text(draw, (bar_x0 + bar_x1) / 2, plot_bottom + 10,
                                mode_label, tick_font, AXIS)
        _draw_centered_text(draw, center_x, plot_bottom + 30, case["shape_label"],
                            body_font)


def _align_shared_points(
    balanced_points: list[tuple[int, int, float]],
    acc_points: list[tuple[int, int, float]],
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    balanced_by_shape = {(q_len, kv_len): value
                         for q_len, kv_len, value in balanced_points}
    acc_by_shape = {(q_len, kv_len): value for q_len, kv_len, value in acc_points}
    shared_shapes = sorted(set(balanced_by_shape) & set(acc_by_shape))
    if not shared_shapes:
        raise ValueError("No shared shapes between balanced and acc-local-l3")
    return (
        [(q_len, kv_len, balanced_by_shape[(q_len, kv_len)])
         for q_len, kv_len in shared_shapes],
        [(q_len, kv_len, acc_by_shape[(q_len, kv_len)])
         for q_len, kv_len in shared_shapes],
    )


def _value_by_shape(
    points: list[tuple[int, int, float]]
) -> dict[tuple[int, int], float]:
    return {(q_len, kv_len): value for q_len, kv_len, value in points}


def build_stacked_case_series(
    series: dict[str, dict[str, list[tuple[int, int, float]]]]
) -> list[dict[str, object]]:
    shared_shapes: set[tuple[int, int]] | None = None
    for metric in TARGET_METRICS:
        for mode in TARGET_MODES:
            current_shapes = {
                (q_len, kv_len)
                for q_len, kv_len, _ in series[metric][mode]
            }
            shared_shapes = (current_shapes if shared_shapes is None else
                             shared_shapes & current_shapes)
    if not shared_shapes:
        raise ValueError("No shared shapes between balanced and acc-local-l3")

    total_by_mode = {
        mode: _value_by_shape(series[TOTAL_METRIC][mode])
        for mode in TARGET_MODES
    }
    local_l3_by_mode = {
        mode: _value_by_shape(series[LOCAL_L3_METRIC][mode])
        for mode in TARGET_MODES
    }
    local_memory_by_mode = {
        mode: _value_by_shape(series[LOCAL_MEMORY_METRIC][mode])
        for mode in TARGET_MODES
    }

    cases = []
    for q_len, kv_len in sorted(shared_shapes):
        bars = []
        for mode in TARGET_MODES:
            total = total_by_mode[mode][(q_len, kv_len)]
            local_l3 = local_l3_by_mode[mode][(q_len, kv_len)]
            local_memory = local_memory_by_mode[mode][(q_len, kv_len)]
            other = total - local_l3 - local_memory
            if other < -1e-6:
                raise ValueError(
                    f"Negative residual for {mode} q={q_len} kv={kv_len}: {other}"
                )
            other = max(other, 0.0)
            bars.append({
                "mode": mode,
                "segments": [
                    round(other, 2),
                    round(local_l3, 2),
                    round(local_memory, 2),
                ],
                "total": round(total, 2),
            })
        cases.append({
            "shape": (q_len, kv_len),
            "shape_label": _shape_label(q_len, kv_len),
            "bars": bars,
        })
    return cases


def generate_plot(summary_csv: str | Path,
                  output_path: str | Path | None = None) -> Path:
    summary_csv = Path(summary_csv).resolve()
    output_path = (Path(output_path).resolve()
                   if output_path else summary_csv.with_name(DEFAULT_OUTPUT_NAME))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series = load_metric_series(summary_csv)

    title_font = _load_font(24)
    panel_title_font = _load_font(16)
    body_font = _load_font(14)
    tick_font = _load_font(12)

    image = Image.new("RGB", IMAGE_SIZE, "#f7f7f7")
    draw = ImageDraw.Draw(image)

    _draw_centered_text(draw, IMAGE_SIZE[0] / 2, 16, make_plot_title(summary_csv),
                        title_font)
    _draw_legend(draw, body_font)
    cases = build_stacked_case_series(series)
    _draw_stacked_chart(draw, (18, 68, IMAGE_SIZE[0] - 16, IMAGE_SIZE[1] - 18),
                        cases, panel_title_font, body_font, tick_font)

    image.save(output_path)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Plot selected demand DC fills metrics from a P3 "
                     "summary.csv as grouped bar charts."))
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
