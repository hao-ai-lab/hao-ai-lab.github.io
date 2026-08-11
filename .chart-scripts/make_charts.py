"""Generate the FastMetal-QAD blog figures as standalone SVGs.

Usage:
    python3 .chart-scripts/make_charts.py [output_dir]

Default output directory is content/blogs/fastmetal-qad-apple-silicon/img.
Figures are deterministic: rerunning with the same run_data.py reproduces them
byte for byte.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import run_data as data
import svg_kit as svg

HW = "Mac Studio · Apple M4 Max · 36 GB unified memory"


def fmt_seconds(value: float) -> str:
    return f"{value:.0f} s" if value >= 100 else f"{value:.1f} s"


def mode_speed_figure() -> str:
    """Small multiples: denoise seconds per mode, one panel per model."""
    body = [
        svg.title(40, 36, "Denoise time by generation mode",
                  f"{HW} · three-step DMD, INT8 DiT · each panel is scaled to its own model"),
    ]
    cols = {model: 190 + i * 322 for i, model in enumerate(data.models())}
    bar_w, row_h, top = 150, 60, 126

    for model, col_x in cols.items():
        peak = max(r.denoise_s for r in data.RUNS if r.model == model)
        body.append(svg.text(col_x, 96, f"FastMetal-{model}-QAD", cls="lbl"))
        body.append(svg.text(col_x, 114, data.MODEL_SETUP[model], cls="sub"))
        body.append(svg.line(col_x - 20, 84, col_x - 20, top + 5 * row_h - 12, svg.GRID))
        base = data.lookup(model, "baseline")
        for i, mode in enumerate(data.MODES):
            y = top + i * row_h + 10
            run = data.lookup(model, mode)
            if run is None:
                body.append(svg.text(col_x, y + 17, "not run", cls="na"))
                continue
            width = bar_w * run.denoise_s / peak
            body.append(svg.bar(col_x, y, width, 22, svg.MODEL_COLORS[model]))
            body.append(svg.text(col_x + bar_w + 14, y + 17, fmt_seconds(run.denoise_s), cls="val"))
            if base is not None and run is not base:
                ratio = base.denoise_s / run.denoise_s
                body.append(svg.text(col_x + 290, y + 17, f"{ratio:.2f}×", cls="vs", anchor="end"))

    for i, mode in enumerate(data.MODES):
        body.append(svg.text(40, top + i * row_h + 27, mode, cls="lbl"))
    body.append(svg.text(40, 460, "Speed-up factors are relative to that model's own baseline run.",
                         cls="cap"))
    return svg.render(1160, 480, "Denoise time by generation mode", body)


def _memory_rows() -> list[tuple[data.Run, float]]:
    rows, y = [], 118.0
    for model in data.models():
        for run in [r for r in data.RUNS if r.model == model]:
            rows.append((run, y))
            y += 30
        y += 16
    return rows


def peak_memory_figure() -> str:
    """Shared-scale peak memory with Mac unified-memory reference lines."""
    x0, x1, span = 210, 1080, 36.0
    scale = lambda gib: x0 + (x1 - x0) * gib / span  # noqa: E731
    rows = _memory_rows()
    axis_y = rows[-1][1] + 44
    body = [
        svg.title(40, 36, "Peak MLX memory by model and mode",
                  f"{HW} · peak during denoising, INT8 DiT, TAEHV decode"),
    ]
    for gib in range(0, 37, 4):
        body.append(svg.line(scale(gib), 100, scale(gib), axis_y, svg.GRID))
        body.append(svg.text(scale(gib), axis_y + 20, str(gib), cls="tick", anchor="middle"))
    # Apple markets RAM in decimal GB; the axis is GiB, so 16 GB is 14.90 GiB
    # and 24 GB is 22.35 GiB. macOS and running apps also claim several GiB, so
    # a usable budget sits well below the nameplate line.
    for gb, label in ((16, "16 GB Mac"), (24, "24 GB Mac")):
        gib = gb * 1000**3 / 1024**3
        body.append(svg.line(scale(gib), 92, scale(gib), axis_y, svg.RULE, 1.5, "6 5"))
        body.append(svg.text(scale(gib), 86, f"{label} ({gib:.1f} GiB)", cls="sub", anchor="middle"))

    for run, y in rows:
        body.append(svg.text(x0 - 14, y + 15, f"{run.model} · {run.mode}", cls="lbl", anchor="end"))
        body.append(svg.bar(x0, y, scale(run.peak_gib) - x0, 21, svg.MODEL_COLORS[run.model]))
        body.append(svg.text(scale(run.peak_gib) + 10, y + 15, f"{run.peak_gib:.2f} GiB", cls="val"))
    body.append(svg.line(x0, axis_y, x1, axis_y, svg.RULE))
    body.append(svg.text(x1 + 12, axis_y + 20, "GiB", cls="tick", anchor="start"))
    body.append(svg.text(40, axis_y + 46,
                         "Nameplate RAM is decimal, so a 24 GB Mac holds 22.35 GiB. macOS and running apps "
                         "claim several GiB on top, so a peak near a line will not fit that machine in practice.",
                         cls="cap"))
    return svg.render(1160, int(axis_y + 64), "Peak MLX memory by model and mode", body)


def _stage_segments(run: data.Run, model: str) -> list[tuple[str, float, str]]:
    return [
        ("prompt encode", run.encode_s, "#7b8bb5"),
        ("denoise", run.denoise_s, svg.MODEL_COLORS[model]),
        ("decode + export", run.decode_s, "#f0b429"),
        ("frame interpolation", run.rife_s, "#8fd4b0"),
        ("other", run.other_s, "#cbd3e0"),
    ]


def time_breakdown_figure() -> str:
    """Percent-stacked wall clock, showing what a cold prompt encode costs."""
    x0, x1 = 250, 990
    body = [
        svg.title(40, 36, "Where the wall clock goes",
                  f"{HW} · share of end-to-end time per stage; fast-mode rows reuse a cached prompt"),
    ]
    for i, (label, _, color) in enumerate(_stage_segments(data.RUNS[0], "1.3B")):
        body.append(svg.swatch(250 + i * 165, 78, color, label))

    y = 128.0
    for model in data.models():
        for mode in ("baseline", "fast"):
            run = data.lookup(model, mode)
            if run is None:
                continue
            body.append(svg.text(x0 - 14, y + 19, f"{model} · {mode}", cls="lbl", anchor="end"))
            cursor = float(x0)
            for label, seconds, color in _stage_segments(run, model):
                width = (x1 - x0) * seconds / run.total_s
                if width < 0.6:
                    continue
                body.append(svg.bar(cursor, y, width, 27, color, radius=3))
                if width > 62:
                    body.append(svg.text(cursor + width / 2, y + 19,
                                         f"{100 * seconds / run.total_s:.0f}%",
                                         cls="vsw", anchor="middle"))
                cursor += width
            body.append(svg.text(x1 + 14, y + 19, f"{run.total_s:.1f} s total", cls="val"))
            y += 42
        y += 14

    body.append(svg.text(40, y + 12,
                         "A cold run pays UMT5 prompt encoding once (18 s at 1.3B/14B, 47 s at 5B); "
                         "repeat prompts hit the cache and start at denoise.", cls="cap"))
    return svg.render(1160, int(y + 40), "Wall clock breakdown by stage", body)


def reconstruction_figure() -> str:
    """Log-scale weight reconstruction error per quantization format."""
    x0, x1 = 250, 880
    lo, hi = 0.004, 0.2
    scale = lambda v: x0 + (x1 - x0) * (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))  # noqa: E731
    best = data.RECONSTRUCTION[0][2]
    body = [
        svg.title(40, 36, "Weight reconstruction error by quantization format",
                  "Relative L2 against FP16 weights · lower is better · log scale"),
    ]
    for tick in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        body.append(svg.line(scale(tick), 78, scale(tick), 356, svg.GRID))
        body.append(svg.text(scale(tick), 378, f"{tick:g}", cls="tick", anchor="middle"))

    for i, (name, bits, value) in enumerate(data.RECONSTRUCTION):
        y = 92 + i * 52
        color = "#1f8a4c" if i == 0 else ("#d1521a" if i == len(data.RECONSTRUCTION) - 1 else "#8fa3bd")
        body.append(svg.text(x0 - 16, y + 20, f"{name} ({bits:g} bits)", cls="lbl", anchor="end"))
        body.append(svg.bar(x0, y, scale(value) - x0, 30, color))
        body.append(svg.text(scale(value) + 12, y + 20, f"{value:.4g}", cls="val"))
        if i:
            body.append(svg.text(scale(value) + 78, y + 20, f"{value / best:.1f}× INT8", cls="vs"))
    body.append(svg.line(x0, 356, x1 + 60, 356, svg.RULE))
    body.append(svg.text(40, 412,
                         "Per-group scaling already supplies dynamic range, so affine INT8 can spend "
                         "all eight bits on 256 uniform levels across the group's actual range.",
                         cls="cap"))
    return svg.render(1000, 436, "Weight reconstruction error by quantization format", body)


FIGURES = {
    "fig_mode_speed.svg": mode_speed_figure,
    "fig_peak_memory.svg": peak_memory_figure,
    "fig_time_breakdown.svg": time_breakdown_figure,
    "fig_int8_reconstruction.svg": reconstruction_figure,
}


def main() -> None:
    default = (Path(__file__).resolve().parents[1] / "content" / "blogs" /
               "fastmetal-qad-apple-silicon" / "img")
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, build in FIGURES.items():
        path = out_dir / name
        path.write_text(build())
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
