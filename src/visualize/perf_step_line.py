#!/usr/bin/env python3
"""Render FEV performance-versus-training-step line charts from result CSVs.

This is a rendering-only component: it reads the immutable ``summary.csv``
files written by evaluation and does not run an analysis stage.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRICS = ("SQL", "WQL", "MASE", "WAPE", "MAE", "MSE")
COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#A52A2A", "#6A3D9A", "#1B9E77", "#E31A1C",
    "#8C6D31", "#006D77",
)
MARKERS = ("circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star", "hexagon", "triangle-left", "triangle-right", "pentagon")
STEP_DIRECTORY = re.compile(r"([1-9][0-9]*)K")
PNG_SERIF_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
PNG_SERIF_BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def _rgb(value: str) -> tuple[int, int, int]:
    """Convert a fixed Plotly hex colour into a Pillow RGB colour."""
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load the server's serif fallback for Plotly's Times New Roman theme."""
    return ImageFont.truetype(PNG_SERIF_BOLD_FONT if bold else PNG_SERIF_FONT, size=size)


def _summary_value(path: Path, metric: str) -> float:
    """Read one numeric benchmark summary value from a completed evaluation."""
    with path.open(newline="", encoding="ascii") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None or row.get("task") != "mean" or row.get(metric) is None:
        raise ValueError("invalid FEV summary for %s: %s" % (metric, path))
    return float(row[metric])


def _evaluation_directories(input_dirs: Sequence[Path]) -> list[tuple[str, Path]]:
    """Discover named method evaluation directories from supplied result folders."""
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for input_dir in input_dirs:
        root = input_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError("results directory does not exist: %s" % root)
        if root.name == "evaluation":
            candidates = [(root.parent.name, root)]
        elif (root / "evaluation").is_dir():
            candidates = [(root.name, root / "evaluation")]
        else:
            direct_method_dirs = [
                child for child in sorted(root.iterdir())
                if child.is_dir() and any((step / "summary.csv").is_file() for step in child.iterdir())
            ]
            candidates = [(child.name, child) for child in direct_method_dirs]
            if not candidates:
                candidates = [(child.name, child / "evaluation") for child in sorted(root.iterdir()) if (child / "evaluation").is_dir()]
        if not candidates:
            raise FileNotFoundError("no method evaluation directory under: %s" % root)
        for method, evaluation_dir in candidates:
            if method in names:
                raise ValueError("duplicate method directory: %s" % method)
            names.add(method)
            result.append((method, evaluation_dir))
    return result


def collect_rows(input_dirs: Sequence[Path]) -> list[dict[str, float | int | str]]:
    """Load every completed FEV summary discovered below the supplied folders."""
    rows: list[dict[str, float | int | str]] = []
    for method, evaluation_dir in _evaluation_directories(input_dirs):
        method_rows: list[dict[str, float | int | str]] = []
        for checkpoint_dir in evaluation_dir.iterdir():
            match = STEP_DIRECTORY.fullmatch(checkpoint_dir.name)
            summary = checkpoint_dir / "summary.csv"
            if match is None or not summary.is_file():
                continue
            step = int(match.group(1)) * 1000
            row: dict[str, float | int | str] = {"method": method, "step": step}
            for metric in METRICS:
                row[metric] = _summary_value(summary, metric)
            method_rows.append(row)
        if not method_rows:
            raise FileNotFoundError("no completed checkpoint summary under: %s" % evaluation_dir)
        rows.extend(sorted(method_rows, key=lambda row: int(row["step"])))
    return rows


def write_summary_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    """Publish the exact values rendered in the chart as a compact CSV."""
    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "step", *METRICS))
        writer.writeheader()
        writer.writerows(rows)


def relative_to_rand_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Express every non-random metric as its signed percentage change from rand."""
    rand_by_step = {int(row["step"]): row for row in rows if row["method"] == "rand"}
    if not rand_by_step:
        raise ValueError("relative-to-rand plot requires a method directory named rand")
    relative_rows: list[dict[str, float | int | str]] = []
    for row in rows:
        if row["method"] == "rand":
            continue
        step = int(row["step"])
        baseline = rand_by_step.get(step)
        if baseline is None:
            raise ValueError("missing rand summary at step %d" % step)
        relative_row: dict[str, float | int | str] = {"method": row["method"], "step": step}
        for metric in METRICS:
            rand_value = float(baseline[metric])
            if rand_value == 0:
                raise ValueError("cannot calculate relative change from zero rand %s at step %d" % (metric, step))
            relative_row[metric] = (float(row[metric]) - rand_value) / rand_value * 100
        relative_rows.append(relative_row)
    if not relative_rows:
        raise ValueError("relative-to-rand plot requires at least one non-rand method")
    return relative_rows


def build_figure(rows: list[dict[str, float | int | str]], title: str, relative_to_rand: bool = False) -> go.Figure:
    """Build six metric curves, optionally as signed changes relative to rand."""
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    method_colors = {method: COLORS[index % len(COLORS)] for index, method in enumerate(methods)}
    method_markers = {method: MARKERS[index % len(MARKERS)] for index, method in enumerate(methods)}
    steps = sorted({int(row["step"]) for row in rows})
    step_positions = {step: position for position, step in enumerate(steps)}
    figure = make_subplots(rows=2, cols=3, subplot_titles=METRICS, vertical_spacing=0.15)
    for position, metric in enumerate(METRICS):
        row_index, column_index = divmod(position, 3)
        for method in methods:
            points = [row for row in rows if row["method"] == method]
            figure.add_trace(
                go.Scatter(
                    x=[step_positions[int(row["step"])] for row in points],
                    y=[float(row[metric]) for row in points],
                    customdata=[int(row["step"]) // 1000 for row in points],
                    mode="lines+markers",
                    name=method,
                    legendgroup=method,
                    showlegend=position == 0,
                    line={"color": method_colors[method], "width": 2},
                    marker={"size": 9, "symbol": method_markers[method], "line": {"width": 1, "color": "white"}},
                    hovertemplate="method=%{fullData.name}<br>step=%{customdata}K<br>" + metric + (" change=%{y:.3f}%" if relative_to_rand else "=%{y:.6g}") + "<extra></extra>",
                ),
                row=row_index + 1,
                col=column_index + 1,
            )
            figure.update_xaxes(
                title_text="Training steps (K)",
                tickmode="array",
                tickvals=list(range(len(steps))),
                ticktext=[step // 1000 for step in steps],
                row=row_index + 1,
                col=column_index + 1,
            )
            if relative_to_rand:
                figure.update_yaxes(title_text=metric + " (method - rand) / rand (%)", zeroline=True, zerolinecolor="#666666", row=row_index + 1, col=column_index + 1)
            else:
                figure.update_yaxes(title_text=metric + " (lower is better)", row=row_index + 1, col=column_index + 1)
    figure.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.5},
        font={"family": "Times New Roman, serif", "size": 18, "color": "#222222"},
        title_font={"family": "Times New Roman, serif", "size": 30},
        width=1440,
        height=940,
        legend={"title": {"text": "Method"}, "orientation": "v", "x": 1.02, "xanchor": "left", "y": 1.0, "yanchor": "top"},
        margin={"t": 100, "b": 60, "l": 65, "r": 285},
        hovermode="x unified",
    )
    return figure


def _draw_marker(draw: ImageDraw.ImageDraw, marker: str, x: int, y: int, color: tuple[int, int, int], size: int = 10) -> None:
    if marker == "circle":
        draw.ellipse((x - size, y - size, x + size, y + size), fill=color, outline="white", width=3)
    elif marker == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), fill=color, outline="white", width=3)
    elif marker == "diamond":
        draw.polygon(((x, y - size - 2), (x + size + 2, y), (x, y + size + 2), (x - size - 2, y)), fill=color, outline="white")
    elif marker in {"triangle-up", "triangle-down", "triangle-left", "triangle-right"}:
        vertices = {
            "triangle-up": ((x, y - size - 2), (x + size + 1, y + size - 1), (x - size - 1, y + size - 1)),
            "triangle-down": ((x, y + size + 2), (x + size + 1, y - size + 1), (x - size - 1, y - size + 1)),
            "triangle-left": ((x - size - 2, y), (x + size - 1, y - size - 1), (x + size - 1, y + size + 1)),
            "triangle-right": ((x + size + 2, y), (x - size + 1, y - size - 1), (x - size + 1, y + size + 1)),
        }[marker]
        draw.polygon(vertices, fill=color, outline="white")
    elif marker == "cross":
        draw.line((x - size, y, x + size, y), fill=color, width=6)
        draw.line((x, y - size, x, y + size), fill=color, width=6)
    elif marker == "x":
        draw.line((x - size + 1, y - size + 1, x + size - 1, y + size - 1), fill=color, width=6)
        draw.line((x - size + 1, y + size - 1, x + size - 1, y - size + 1), fill=color, width=6)
    else:
        draw.ellipse((x - size, y - size, x + size, y + size), fill=color, outline="white", width=3)


def write_png(rows: list[dict[str, float | int | str]], output_path: Path, title: str, relative_to_rand: bool = False) -> None:
    """Render the Plotly figure's data as a PNG through the headless Pillow backend."""
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    method_colors = {method: COLORS[index % len(COLORS)] for index, method in enumerate(methods)}
    method_markers = {method: MARKERS[index % len(MARKERS)] for index, method in enumerate(methods)}
    steps = sorted({int(row["step"]) for row in rows})
    width, height = 2880, 1800
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(52, bold=True)
    legend_font = _font(32)
    panel_font = _font(32, bold=True)
    axis_font = _font(24)
    draw.text((width // 2, 38), title, font=title_font, fill="#111111", anchor="ma")
    legend_columns = 4
    legend_cell_width = width // legend_columns
    legend_row_height = 58
    for legend_index, method in enumerate(methods):
        legend_row, legend_column = divmod(legend_index, legend_columns)
        x = legend_column * legend_cell_width + 25
        y = 125 + legend_row * legend_row_height
        color = _rgb(method_colors[method])
        draw.line((x, y, x + 48, y), fill=color, width=7)
        _draw_marker(draw, method_markers[method], x + 24, y, color, size=9)
        draw.text((x + 62, y), method, font=legend_font, fill="#222222", anchor="lm")

    legend_rows = (len(methods) + legend_columns - 1) // legend_columns
    outer_left, outer_top = 90, 150 + legend_rows * legend_row_height
    panel_width, panel_height = 870, 650
    panel_gap_x, panel_gap_y = 50, 70
    first_step = steps[0]
    last_step = steps[-1]
    step_positions = {step: position for position, step in enumerate(steps)}
    step_span = len(steps) - 1
    for position, metric in enumerate(METRICS):
        row, column = divmod(position, 3)
        left = outer_left + column * (panel_width + panel_gap_x)
        top = outer_top + row * (panel_height + panel_gap_y)
        right, bottom = left + panel_width, top + panel_height
        plot_left, plot_top = left + 120, top + 70
        plot_right, plot_bottom = right - 35, bottom - 85
        values = [float(item[metric]) for item in rows]
        lower, upper = min(values), max(values)
        padding = (upper - lower) * 0.08 if upper > lower else max(abs(upper) * 0.08, 0.1)
        lower -= padding
        upper += padding
        panel_title = metric + (" (method - rand) / rand (%)" if relative_to_rand else " (lower is better)")
        draw.text((left, top), panel_title, font=panel_font, fill="#111111")
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#9a9a9a", width=3)
        for fraction in range(1, 5):
            y = plot_top + (plot_bottom - plot_top) * fraction // 5
            value = upper - (upper - lower) * fraction / 5
            draw.line((plot_left, y, plot_right, y), fill="#e6e6e6", width=1)
            draw.text((plot_left - 12, y), "%.3g" % value, font=axis_font, fill="#555555", anchor="rm")
        if relative_to_rand and lower <= 0 <= upper:
            zero_y = int(plot_bottom - (0 - lower) * (plot_bottom - plot_top) / (upper - lower))
            draw.line((plot_left, zero_y, plot_right, zero_y), fill="#666666", width=3)
        for step in steps:
            x = (plot_left + plot_right) // 2 if step_span == 0 else plot_left + step_positions[step] * (plot_right - plot_left) / step_span
            draw.line((int(x), plot_bottom, int(x), plot_bottom + 6), fill="#777777", width=1)
            draw.text((int(x), plot_bottom + 22), "%dK" % (step // 1000), font=axis_font, fill="#555555", anchor="ma")
        for method in methods:
            points = [item for item in rows if item["method"] == method]
            coordinates = []
            for item in points:
                x = (plot_left + plot_right) // 2 if step_span == 0 else plot_left + step_positions[int(item["step"])] * (plot_right - plot_left) / step_span
                y = plot_bottom - (float(item[metric]) - lower) * (plot_bottom - plot_top) / (upper - lower)
                coordinates.append((int(x), int(y)))
            color = _rgb(method_colors[method])
            draw.line(coordinates, fill=color, width=7)
            for x, y in coordinates:
                _draw_marker(draw, method_markers[method], x, y, color)
    canvas.save(output_path, format="PNG", optimize=True)


def main(argv: Sequence[str] | None = None) -> int:
    from configs.paths import output_path
    parser = argparse.ArgumentParser(description="Plot completed FEV checkpoint curves")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-dir", type=Path, nargs="+", help="one or more method directories, evaluation directories, or a common results root")
    source_group.add_argument("--results-root", type=Path, help="one common results root; retained for compatibility")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="FEV performance by training step")
    parser.add_argument("--relative-to-rand", action="store_true", help="plot signed metric changes against rand")
    args = parser.parse_args(argv)
    input_dirs = args.input_dir if args.input_dir is not None else [args.results_root]
    rows = collect_rows(input_dirs)
    if args.relative_to_rand:
        rows = relative_to_rand_rows(rows)
    args.output_dir = output_path("vis", args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_name = "relative_to_rand_by_step.csv" if args.relative_to_rand else "performance_by_step.csv"
    figure_name = "relative_to_rand_steps" if args.relative_to_rand else "performance_steps"
    write_summary_csv(rows, args.output_dir / summary_name)
    figure = build_figure(rows, args.title, relative_to_rand=args.relative_to_rand)
    figure.write_html(args.output_dir / (figure_name + ".html"), include_plotlyjs=True)
    write_png(rows, args.output_dir / (figure_name + ".png"), args.title, relative_to_rand=args.relative_to_rand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
