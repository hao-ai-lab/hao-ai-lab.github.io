# FastH3 Omni comparison data

- `benchmark.json`: measurement settings, full prompts, ordered reference files, output video paths, measured latencies and speedups.
- `latency.csv`: all 27 timing rows, with the original video VAE in both timing lanes.

Each timing uses measured seeds 0, 1 and 2 after one excluded warmup. The galleries display seed 0. Shared inputs and seeds do not establish output parity. The 36-layer VAE appears in the galleries without timing fields.

Base uses SGLang's lossless numerical policy and Diffusers reference sizing, recorded in `settings.base`. It keeps unquantized weights and dense attention but enables fusions that can change intermediate rounding.

Rebuild the three speedup charts with Python, Matplotlib and NumPy:

```bash
python tools/plot_omniref_results.py \
  --data content/blogs/fastvideo-omniref-update/data/benchmark.json \
  --output content/blogs/fastvideo-omniref-update/img
```
