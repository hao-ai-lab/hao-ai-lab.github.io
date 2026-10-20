#!/usr/bin/env python3
"""Render a separate speedup chart for each output length."""
import argparse
import json
from pathlib import Path


def render_charts(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "fasth3-omniref"})
    output.mkdir(parents=True, exist_ok=True)
    maximum = max(row["performance"]["speedup"] for row in payload["profiles"])
    for seconds in [5, 10, 15]:
        rows = [row for row in payload["profiles"] if row["target_seconds"] == seconds]
        ratios = [row["performance"]["speedup"] for row in rows]
        fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
        positions = np.arange(len(rows))
        ax.barh(positions, ratios, height=.62, color="#168a70")
        for position, ratio in zip(positions, ratios):
            ax.text(ratio + .13, position, f"{ratio:.2f}×",
                    va="center", weight="bold", color="#126f5c")
        ax.set_yticks(positions, [row["label"] for row in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, maximum * 1.14)
        ax.axvline(1, color="#697a91", linestyle="--", linewidth=1)
        ax.set_xlabel("Speedup over Base (higher is better)")
        ax.set_title(f"{seconds}-second video + audio · 8×GB200 · 1344×768",
                     loc="left", pad=15, fontsize=15, weight="bold")
        ax.xaxis.grid(True, alpha=.17)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        target = output / f"speedup-{seconds}s.svg"
        fig.savefig(target, metadata={
            "Date": None, "Creator": None,
            "Title": f"FastH3 Omni speedup over Base for {seconds}-second outputs",
            "Description": "Base mean latency divided by FastH3 Omni mean latency. Original video VAE in both lanes.",
        })
        plt.close(fig)
        target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render_charts(json.loads(args.data.read_text()), args.output)
