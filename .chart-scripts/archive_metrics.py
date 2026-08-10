"""Archive the run metrics behind the blog figures into version control.

The figures in the post are generated from `run_data.py`, which was transcribed
by hand from `*_metrics.json` files written by the generation runs. Those files
originally lived under a scratch directory that does not survive a reboot, so
this script copies the relevant ones next to the chart code.

Paths in the source files embed a local home directory and scratch locations.
Those values are redacted here, since this repository is public.

Usage:
    python3 .chart-scripts/archive_metrics.py [source_dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SOURCE_DEFAULT = Path("/tmp/fastvideo_pr1638_quality_runs")
DEST = Path(__file__).resolve().parent / "data"

# Run directory -> metrics files that feed a figure in the post.
WANTED = {
    "blog_modes_forest_1p3b": "forest_1p3b_832x480_81_{mode}_metrics.json",
    "blog_modes_neon_5b": "neon_woman_5b_1280x704_81_{mode}_metrics.json",
    "blog_modes_stretch_14b": "stretch_14b_832x480_81_{mode}_seed1024_metrics.json",
}
MODES = ("baseline", "fast", "refine", "fast_refine", "prompt_enhance")

# The 5B refine and fast+refine rows in the post come from a re-run.
EXTRA = {
    "blog_modes_neon_5b_retry_seed2027_nocompile": (
        "neon_woman_5b_1280x704_81_refine_seed2027_metrics.json",
        "neon_woman_5b_1280x704_81_fast_refine_seed2027_metrics.json",
    ),
}

PATH_KEYS = {
    "mlx_checkpoint", "model_root", "output_path", "prompt_embeds_cache",
    "enhance_model", "taehv_checkpoint_path", "vae_root", "text_encoder_root",
}
HOME = re.compile(r"/(?:private/)?(?:Users|home)/[^/\s\"]+")
SCRATCH = re.compile(r"/(?:private/)?tmp/[^\s\"]*")


def redact(value: object) -> object:
    """Strip local filesystem identity from a metrics value."""
    if not isinstance(value, str):
        return value
    cleaned = HOME.sub("~", value)
    return SCRATCH.sub("<run-dir>", cleaned)


def sanitize(payload: dict) -> dict:
    """Return a copy of the metrics with path-bearing fields redacted."""
    return {
        key: redact(value) if key in PATH_KEYS else value
        for key, value in payload.items()
    }


def wanted_files(source: Path) -> list[Path]:
    paths = []
    for run_dir, template in WANTED.items():
        for mode in MODES:
            candidate = source / run_dir / template.format(mode=mode)
            if candidate.exists():
                paths.append(candidate)
    for run_dir, names in EXTRA.items():
        paths.extend(source / run_dir / name for name in names
                     if (source / run_dir / name).exists())
    return paths


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE_DEFAULT
    if not source.exists():
        raise SystemExit(f"source directory not found: {source}")

    files = wanted_files(source)
    if not files:
        raise SystemExit(f"no metrics files matched under {source}")

    for path in files:
        payload = sanitize(json.loads(path.read_text()))
        target = DEST / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"archived {target.relative_to(DEST.parent)}")


if __name__ == "__main__":
    main()
