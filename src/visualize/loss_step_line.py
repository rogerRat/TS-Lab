#!/usr/bin/env python3
"""Render smoothed training-loss lines for model run directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont


COLORS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SMOOTHING_WINDOW = 9


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size=size)


def _rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def _run_directories(input_dirs: Sequence[Path]) -> list[tuple[str, Path]]:
    # Search in priority order (archived train_meta first, then live train).
    # A missing/empty fallback directory is expected when pruning is disabled;
    # only the absence of every loss file is an error.
    runs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for input_dir in input_dirs:
        root = input_dir.resolve()
        if not root.is_dir():
            continue
        candidates = [(root.name, root)] if (root / "losses.csv").is_file() else [
            (child.name, child) for child in sorted(root.iterdir())
            if child.is_dir() and (child / "losses.csv").is_file()
        ]
        for name, run_dir in candidates:
            if name in seen:
                continue
            seen.add(name)
            runs.append((name, run_dir))
    if not runs:
        searched = ", ".join(str(path.resolve()) for path in input_dirs)
        raise FileNotFoundError("no losses.csv found under supplied directories: %s" % searched)
    return runs


def collect_rows(input_dirs: Sequence[Path]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for method, run_dir in _run_directories(input_dirs):
        with (run_dir / "losses.csv").open(newline="", encoding="ascii") as handle:
            for row in csv.DictReader(handle):
                try:
                    step, loss = int(row["optimizer_step"]), float(row["loss"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("invalid loss row in %s" % (run_dir / "losses.csv")) from error
                rows.append({"method": method, "step": step, "loss": loss})
    if not rows:
        raise ValueError("loss CSV files contain no rows")
    return rows


def _smoothed_points(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    points = sorted(rows, key=lambda row: int(row["step"]))
    if len(points) < 3:
        return points
    window = min(SMOOTHING_WINDOW, len(points))
    if window % 2 == 0:
        window -= 1
    radius = window // 2
    smoothed: list[dict[str, float | int | str]] = []
    for index, point in enumerate(points):
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        average = sum(float(item["loss"]) for item in points[start:end]) / (end - start)
        smoothed.append({"method": point["method"], "step": point["step"], "loss": average})
    return smoothed


def _method_points(rows: list[dict[str, float | int | str]]) -> dict[str, list[dict[str, float | int | str]]]:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    return {method: _smoothed_points([row for row in rows if row["method"] == method]) for method in methods}


def write_png(rows: list[dict[str, float | int | str]], output_path: Path, title: str) -> None:
    """Render smoothed lines with Pillow without point markers."""
    points_by_method = _method_points(rows)
    methods = list(points_by_method)
    steps = sorted({int(row["step"]) for row in rows})
    step_positions = {step: position for position, step in enumerate(steps)}
    values = [float(row["loss"]) for row in rows]
    lower, upper = min(values), max(values)
    padding = (upper - lower) * 0.08 if upper > lower else max(abs(upper) * 0.08, 0.1)
    lower -= padding
    upper += padding
    width, height = 2400, 1400
    left, top, right, bottom = 180, 170, width - 100, height - 180
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 45), title, font=_font(50, bold=True), fill="#111111", anchor="ma")
    draw.rectangle((left, top, right, bottom), outline="#777777", width=3)
    for fraction in range(1, 5):
        y = top + (bottom - top) * fraction // 5
        value = upper - (upper - lower) * fraction / 5
        draw.line((left, y, right, y), fill="#e6e6e6", width=2)
        draw.text((left - 20, y), "%.4g" % value, font=_font(24), fill="#555555", anchor="rm")
    step_span = len(steps) - 1 if len(steps) > 1 else 0
    tick_stride = max(1, len(steps) // 8)
    for position, step in [(step_positions[step], step) for step in steps[::tick_stride]]:
        x = (left + right) // 2 if step_span == 0 else left + position * (right - left) / step_span
        draw.line((int(x), bottom, int(x), bottom + 8), fill="#777777", width=2)
        draw.text((int(x), bottom + 25), str(step), font=_font(24), fill="#555555", anchor="ma")
    draw.text((width // 2, height - 45), "Optimizer step", font=_font(28), fill="#222222", anchor="ms")
    draw.text((45, (top + bottom) // 2), "Loss", font=_font(28), fill="#222222", anchor="mm")
    for index, method in enumerate(methods):
        color = _rgb(COLORS[index % len(COLORS)])
        coordinates = []
        for point in points_by_method[method]:
            x = (left + right) // 2 if step_span == 0 else left + step_positions[int(point["step"])] * (right - left) / step_span
            y = bottom - (float(point["loss"]) - lower) * (bottom - top) / (upper - lower)
            coordinates.append((int(x), int(y)))
        if len(coordinates) > 1:
            draw.line(coordinates, fill=color, width=5, joint="curve")
        legend_x = 220 + index * 300
        draw.line((legend_x, 125, legend_x + 42, 125), fill=color, width=5)
        draw.text((legend_x + 55, 125), method, font=_font(25), fill="#222222", anchor="lm")
    canvas.save(output_path, format="PNG", optimize=True)


def main(argv: Sequence[str] | None = None) -> int:
    from configs.paths import output_path

    parser = argparse.ArgumentParser(description="Plot smoothed training loss by optimizer step")
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True, help="run directories or one common training root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Training loss by optimizer step")
    args = parser.parse_args(argv)
    rows = collect_rows(args.input_dir)
    steps = sorted({int(row["step"]) for row in rows})
    step_positions = {step: position for position, step in enumerate(steps)}
    points_by_method = _method_points(rows)
    destination = output_path("vis", args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "losses_by_step.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "step", "loss"))
        writer.writeheader()
        writer.writerows(rows)
    figure = go.Figure()
    for index, (method, points) in enumerate(points_by_method.items()):
        figure.add_trace(go.Scatter(
            x=[step_positions[int(row["step"])] for row in points],
            y=[row["loss"] for row in points],
            customdata=[int(row["step"]) for row in points],
            name=method,
            mode="lines",
            line={"color": COLORS[index % len(COLORS)], "width": 2.5},
            hovertemplate="method=%{fullData.name}<br>step=%{customdata}<br>loss=%{y:.6g}<extra></extra>",
        ))
    figure.update_layout(
        template="plotly_white",
        title={"text": args.title, "x": 0.5},
        xaxis={"title": "Checkpoint order (sorted by optimizer step)", "tickmode": "array", "tickvals": list(range(len(steps))), "ticktext": [str(step) for step in steps]},
        yaxis_title="Loss",
        hovermode="x unified",
    )
    figure.write_html(destination / "losses_by_step.html", include_plotlyjs=True)
    write_png(rows, destination / "losses_by_step.png", args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
