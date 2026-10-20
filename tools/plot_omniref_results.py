#!/usr/bin/env python3
"""Render the blog's figures from its published JSON (Matplotlib 3.11.2)."""
import argparse
import json
from pathlib import Path


def render_charts(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "fasth3-omniref-20261020"})
    output.mkdir(parents=True, exist_ok=True)
    for seconds in [5, 10, 15]:
        rows = [row for row in payload["profiles"] if row["target_seconds"] == seconds]
        fig, ax = plt.subplots(figsize=(11, 6.9), layout="constrained")
        y = np.arange(len(rows))
        values_base = [row["performance"]["base"]["mean_s"] for row in rows]
        values_fast = [row["performance"]["fast"]["mean_s"] for row in rows]
        ax.barh(y - .18, values_base, height=.32, color="#697a91", label="Base · 50 evaluations")
        ax.barh(y + .18, values_fast, height=.32, color="#168a70", label="FastH3 Omni · 8 evaluations")
        xmax = max(values_base)
        for i, row in enumerate(rows):
            perf = row["performance"]
            ax.text(values_base[i] + xmax*.015, i-.18, f"{values_base[i]:.2f}s", va="center", fontsize=9)
            ax.text(values_fast[i] + xmax*.015, i+.18, f"{values_fast[i]:.2f}s  ·  {perf['speedup']:.2f}×",
                    va="center", color="#126f5c", fontsize=9, weight="bold")
        ax.set_yticks(y, [row["label"] for row in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax * 1.19)
        ax.set_xlabel("End-to-end latency, seconds (lower is better)")
        ax.set_title(f"{seconds}s requests · 1344×768 with audio · 8×GB200", loc="left", pad=43, weight="bold", fontsize=15)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=1, frameon=False, fontsize=10)
        ax.xaxis.grid(True, alpha=.17)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        target = output / f"latency-{seconds}s.svg"
        fig.savefig(target, metadata={"Date": None,
            "Title": f"Base versus FastH3 Omni latency for {seconds}s requests",
            "Description": "Mean across seeds 0,1,2 after warmup 42. C1. Base/FastH3 multipliers use unrounded means. Original VAE for both lanes."})
        plt.close(fig)
        target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render_charts(json.loads(args.data.read_text()), args.output)
