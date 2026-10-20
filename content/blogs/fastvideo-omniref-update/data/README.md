# Frozen OmniRef comparison data

Snapshot: 2026-10-20 06:53:34 UTC. The selected cohort contains all 27 complete
eight-GB200 workload profiles for each of three output variants.

- `benchmark.json`: protocol, model/runtime revisions, exact prompts, ordered
  reference files, output hashes, Base and FastH3 Omni measured latencies,
  arithmetic means, sample standard deviations and unrounded Base/FastH3 ratios.
- `latency.csv`: the 27 timing rows. Both lanes use the original video VAE.
- `media-manifest.json`: hashes and sizes of the original reference media,
  unchanged seed-0 output MP4s, and derived JPEG preview frames.

The gallery's third output variant has no timing fields. Timing statistics
use seeds 0, 1 and 2; the gallery consistently displays seed 0. Shared seeds
and inputs do not establish numerical or perceptual parity.

Rebuild the three SVG figures using Python 3.12, Matplotlib 3.11.2 and NumPy
2.5.4, from the repository root:

```bash
python tools/plot_omniref_results.py \
  --data content/blogs/fastvideo-omniref-update/data/benchmark.json \
  --output content/blogs/fastvideo-omniref-update/img
```

`tools/export_omniref_blog.py` creates this sanitized data and copies media from
a frozen matrix snapshot plus its verified local artifacts. It checks the
original successful requests, measured seeds, summary validation and matching
inputs before exporting. Raw deployment logs, service addresses, job IDs and
36L timing values are not included in this page bundle.
