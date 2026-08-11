"""Measured FastMetal-QAD runs used by the blog figures.

Every row is transcribed from the `*_metrics.json` file emitted by the run that
produced the matching video in the post. Hardware for all rows: Mac Studio,
Apple M4 Max, 36 GB unified memory, macOS, three-step DMD, INT8 DiT, TAEHV decode.

The source files are archived under `.chart-scripts/data/` by
`archive_metrics.py`, with local paths redacted.

Provenance (PR hao-ai-lab/FastVideo#1638 quality runs):
  1.3B -> blog_modes_forest_1p3b/forest_1p3b_832x480_81_*
          two-pass rows re-measured in blog_modes_forest_1p3b_pr51ede39/
  5B   -> blog_modes_neon_5b/neon_woman_5b_1280x704_81_{baseline,fast,prompt_enhance}
          blog_modes_neon_5b_pr51ede39/*_{refine,fast_refine}_seed2027
  14B  -> blog_modes_stretch_14b/stretch_14b_832x480_81_*_seed1024
          two-pass rows re-measured in blog_modes_stretch_14b_pr51ede39/

The refine and fast+refine rows were re-measured after FastVideo commit
51ede39e9, which stopped mx.compile from keeping a second copy of the DiT
weights alive across the two denoising resolutions.

Peak memory for 1.3B and 14B is `mlx_denoise_peak_bytes / 2**30`; the 5B runner
reports `peak_gib` directly.
"""

from __future__ import annotations

from typing import NamedTuple

GIB = 1024**3


class Run(NamedTuple):
    model: str
    mode: str
    encode_s: float
    denoise_s: float
    decode_s: float
    rife_s: float
    total_s: float
    peak_gib: float

    @property
    def other_s(self) -> float:
        """Wall time not attributed to a named stage (DiT load, export, glue)."""
        named = self.encode_s + self.denoise_s + self.decode_s + self.rife_s
        return max(self.total_s - named, 0.0)


MODES = ["baseline", "fast", "refine", "fast + refine", "prompt enhance"]

MODEL_SETUP = {
    "1.3B": "832×480 · 81 frames",
    "5B": "1280×704 · 81 frames",
    "14B": "832×480 · 81 frames",
}

RUNS: tuple[Run, ...] = (
    Run("1.3B", "baseline", 18.416, 89.768, 1.883, 0.0, 110.136, 4159476950 / GIB),
    Run("1.3B", "fast", 0.001, 31.306, 2.429, 11.428, 45.188, 3441337422 / GIB),
    Run("1.3B", "refine", 0.003, 96.240, 4.310, 0.0, 100.590, 4186736826 / GIB),
    Run("1.3B", "fast + refine", 0.003, 34.650, 1.980, 2.770, 39.430, 3453476922 / GIB),
    Run("1.3B", "prompt enhance", 19.304, 88.862, 4.196, 0.0, 112.432, 4159476950 / GIB),
    Run("5B", "baseline", 47.022, 98.504, 4.045, 0.0, 151.423, 9.337),
    Run("5B", "fast", 0.020, 41.155, 0.936, 4.342, 47.238, 7.972),
    Run("5B", "refine", 0.015, 102.880, 2.930, 0.0, 107.640, 10.801),
    Run("5B", "fast + refine", 0.018, 44.920, 1.510, 4.580, 51.580, 8.130),
    Run("5B", "prompt enhance", 47.377, 98.769, 3.338, 0.0, 151.268, 9.337),
    Run("14B", "baseline", 18.431, 554.224, 29.108, 0.0, 601.816, 23283634390 / GIB),
    Run("14B", "fast", 0.003, 204.729, 3.858, 2.511, 211.137, 19431739606 / GIB),
    Run("14B", "refine", 0.003, 581.050, 36.970, 0.0, 618.060, 23310897338 / GIB),
    Run("14B", "fast + refine", 0.003, 231.190, 4.020, 2.100, 237.350, 19446091962 / GIB),
    Run("14B", "prompt enhance", 17.274, 550.776, 10.852, 0.0, 578.965, 23283634390 / GIB),
)

# Weight reconstruction relative L2, lower is better. Format label -> (bits, error).
RECONSTRUCTION = (
    ("affine INT8", 8.5, 0.0055),
    ("MXFP8", 8.25, 0.068),
    ("INT4", 4.5, 0.092),
    ("NVFP4", 4.5, 0.103),
    ("MXFP4", 4.25, 0.121),
)


def lookup(model: str, mode: str) -> Run | None:
    for run in RUNS:
        if run.model == model and run.mode == mode:
            return run
    return None


def models() -> list[str]:
    seen: list[str] = []
    for run in RUNS:
        if run.model not in seen:
            seen.append(run.model)
    return seen
