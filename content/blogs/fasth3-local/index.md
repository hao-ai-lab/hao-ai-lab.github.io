+++
title = "FastH3 on Apple Silicon and DGX Spark"
date = 2026-08-31T00:00:00-07:00
url = "/blogs/fasth3-local/"
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
draft = true
contentClass = "fasth3-local-article"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
[cover]
    image = "img/cover.png"
    alt = "FastH3 on Apple Silicon and DGX Spark"
    caption = "FastH3 on Apple Silicon and DGX Spark"
    hidden = true
+++

<!--
Publication checklist:
- Keep draft = true until weights, videos, benchmarks, and commands are final.
- Confirm checkpoint IDs, step count, attention backend, licenses, and authors.
- Confirm spatial fast mode, Spark staged loading, and Spark FP8 before
  treating them as shipped.
- Fill the video grids, phase chart, and Spark table from final runs.
- Test cookbook commands and every public link.

Wide-M A/B, now in the release: M4 Max 36 GB, MLX 0.31.2, affine INT6,
832x480x124, 4 steps, seed 2026, dense attention, no temporal fast mode.
DiT 386.47s to 348.75s. Bit-exact video and audio latents.
MLX peak 19.31 to 19.46 GiB. Not an end-to-end VSA number.
-->

{{< image src="img/cover.png" alt="FastH3 on Apple Silicon and DGX Spark" width="100%" >}}

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fastvideo-fasth3" >}}

FastH3 now runs on a Mac, and on NVIDIA DGX Spark.

H3 generates video and audio together. That used to mean a data-center GPU.
Our [FastH3 Preview](/blogs/fasth3-preview/) distilled it into a few steps on
Blackwell. This release puts that model on Apple Silicon through MLX, and on a
desktop GB10 through CUDA 13. The Mac path targets 36 GB of unified memory or
more. Spark has 128 GB.

Start in the [FastVideo cookbook](/FastVideo/cookbook/). H3 is the first family.

## Generated locally

The same prompts on an Apple M4 Max and a DGX Spark. Base H3 on B200 is the
quality reference. Turn the audio on.

<div class="fasth3-local-grid fasth3-local-grid--platforms">
  <div class="fasth3-local-grid__header">Apple M4 Max<br><small>FastH3 MLX, TODO precision</small></div>
  <div class="fasth3-local-grid__header">DGX Spark GB10<br><small>FastH3 CUDA, TODO precision</small></div>
  <div class="fasth3-local-grid__header">NVIDIA B200<br><small>Base H3 reference</small></div>
  <div class="fasth3-local-media-placeholder" data-label="Apple M4 Max, prompt A">TODO: prompt A video</div>
  <div class="fasth3-local-media-placeholder" data-label="DGX Spark, prompt A">TODO: prompt A video</div>
  <div class="fasth3-local-media-placeholder" data-label="B200 Base H3, prompt A">TODO: prompt A video</div>
  <div class="fasth3-local-media-placeholder" data-label="Apple M4 Max, prompt B">TODO: prompt B video</div>
  <div class="fasth3-local-media-placeholder" data-label="DGX Spark, prompt B">TODO: prompt B video</div>
  <div class="fasth3-local-media-placeholder" data-label="B200 Base H3, prompt B">TODO: prompt B video</div>
</div>

<!-- TODO: Replace placeholders with final videos. Add collapsible full prompts
and per-cell checkpoint, precision, seed, resolution, frames, duration, steps,
attention backend, and revision. Match prompts, output dimensions, and duration.
Base H3 and FastH3 differ in model and schedule; do not imply a hardware speed test. -->

## Three formats for Mac

Every Mac number in this post comes from an M4 Max with 36 GB of unified
memory. H3 is large enough that memory capacity matters as much as GPU speed.

INT8 keeps the most weight precision. INT6 is the format behind the runtime
number below. INT4 leaves the most room for activations. Lower precision can
change detail and motion. The grid holds the generation settings fixed.

<div class="fasth3-local-precision-grid">
  <div class="fasth3-local-grid__header">Format</div>
  <div class="fasth3-local-grid__header">Prompt A</div>
  <div class="fasth3-local-grid__header">Prompt B</div>
  <div class="fasth3-local-grid__header">Prompt C</div>
  <div class="fasth3-local-precision-label"><strong>INT8</strong><small>Time: TODO<br>Peak memory: TODO</small></div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt A">TODO: INT8 A</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt B">TODO: INT8 B</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt C">TODO: INT8 C</div>
  <div class="fasth3-local-precision-label"><strong>INT6</strong><small>Time: TODO<br>Peak memory: TODO</small></div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt A">TODO: INT6 A</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt B">TODO: INT6 B</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt C">TODO: INT6 C</div>
  <div class="fasth3-local-precision-label"><strong>INT4</strong><small>Time: TODO<br>Peak memory: TODO</small></div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt A">TODO: INT4 A</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt B">TODO: INT4 B</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt C">TODO: INT4 C</div>
</div>

<!-- TODO: Fill each row with end-to-end time and the largest measured phase
peak in GiB. Preserve identical prompts, seeds, resolution, frame count, steps,
attention/decode settings, prompt-cache state, and runtime revision. Link weights
from the format labels. Put complete run records in a collapsible caption. -->

## How a data-center model fits

The weights are only part of the problem. H3 also needs a large text encoder,
working memory for denoising, and decoders for video and audio. Keep all of
that loaded on a 36 GB Mac and the generation does not start.

The pipeline runs in phases. Each phase loads what it needs and frees the rest.

<div class="fasth3-local-pipeline" aria-label="FastH3 MLX generation stages">
  <div><strong>Encode</strong><span>Stream the text encoder</span></div>
  <b aria-hidden="true">&rarr;</b>
  <div><strong>Denoise</strong><span>Video and audio together</span></div>
  <b aria-hidden="true">&rarr;</b>
  <div><strong>Decode</strong><span>Frames and soundtrack</span></div>
  <b aria-hidden="true">&rarr;</b>
  <div><strong>Export</strong><span>One synchronized MP4</span></div>
</div>

H3 reads an intermediate layer of Qwen3-VL, so the encoder skips the last 14
layers. It streams the rest one layer at a time and keeps only the embedding
rows the prompt uses. Cache those embeddings and a repeated prompt skips this
stage. After denoising, tiled video decode and a native audio decoder finish
the clip without reconstructing the whole frame buffer at once.

<div class="fasth3-local-chart-placeholder">
  <strong>TODO: Time and memory through one generation</strong>
  <span>Prompt encoding · Denoising · Video and audio decode · Export</span>
</div>

<!-- TODO: Two aligned panels: time per phase and measured phase peak. Include
RIFE when enabled. Identify cached versus uncached prompts and memory accounting;
do not equate MLX allocation peaks with whole-system memory use. -->

Smaller weights are not always faster. H3 multiplies large video and audio
matrices. At those shapes, unpacking a quantized weight into BF16 and using
MLX's dense matrix multiply beat the quantized kernel. The unpacked copy is
discarded after the multiply.

On a four-step INT6 run, denoising dropped from 386.47 seconds to 348.75
seconds. The video and audio latents were bit-exact. Peak MLX memory rose from
19.31 GiB to 19.46 GiB. That run used 832×480, 124 frames, and dense attention,
so the number is the matrix path, not sparse attention.

FastH3 also uses Video Sparse Attention, so only selected video tiles attend.
The MLX runtime implements that policy on Apple GPUs.

## FastH3 on DGX Spark

[DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
is a desktop Blackwell machine: a GB10 GPU, 128 GB of unified LPDDR5X, CUDA 13,
ARM64. The FastH3 CUDA path from the Preview release now runs on that box.

The model fits. The memory math does not look like a discrete GPU.

There is no separate VRAM. CPU and GPU share one pool, at roughly 270 GB/s,
about a tenth of datacenter HBM. On a B200, offloading weights to the host
frees GPU memory. On Spark, that copy lands in the same pool. It duplicates
the weights and frees nothing. We stopped that copy.

H3 never uses the last layers of its text encoder. The Spark path skips them,
same as the Mac path. Video Sparse Attention stays on CUDA. The FastVideo
kernel builds from source for `sm_121`.

Install from the
[CUDA 13 Spark guide](/FastVideo/getting_started/installation/spark/), then
open the [H3 recipes](/FastVideo/cookbook/minimax-h3/) and pick the CUDA
runtime.

{{< table title="DGX Spark · FastH3 release measurements" >}}

| Output | Precision | Total generation time | Peak allocated memory |
| :----- | :-------- | --------------------: | --------------------: |
| TODO: resolution and duration | TODO | TODO | TODO |

{{</ table >}}

<!-- TODO: Final public checkpoint, exact command, hardware/runtime revision,
cache state, and memory flags. Use a full-run peak including activations.
Do not reuse the 902s rank-reduced candidate as a final release benchmark. -->

## Faster drafts

`--fast` denoises fewer video frames, then interpolates back to the requested
length. Audio keeps its full duration. Drafts finish sooner, sometimes with
softer motion. Compare it with the baseline before a final render.

<div class="fasth3-local-grid fasth3-local-grid--modes">
  <div class="fasth3-local-grid__header">Baseline</div>
  <div class="fasth3-local-grid__header">Temporal fast</div>
  <div class="fasth3-local-grid__header">Spatial fast</div>
  <div class="fasth3-local-media-placeholder" data-label="Baseline">TODO: fixed-prompt baseline</div>
  <div class="fasth3-local-media-placeholder" data-label="Temporal fast">TODO: same prompt and seed</div>
  <div class="fasth3-local-media-placeholder" data-label="Spatial fast">TODO: same prompt and seed</div>
</div>

<!-- TODO: Verify spatial fast mode ships and replace with the final algorithm
and flag. Add matched examples with time, memory, output dimensions, and audio. -->

## Start in the cookbook

The [FastVideo cookbook](/FastVideo/cookbook/) is where this release starts.
It is a catalog of model families. Each family has recipes tied to checked-in
FastVideo sources, so the command you copy matches a combination we actually
run.

H3 is first: CUDA, the four-step preview, LoRA, and the native MLX path. Open
the [H3 recipes](/FastVideo/cookbook/minimax-h3/), pick a runtime, generate.
Spark users should install from the
[CUDA 13 Spark guide](/FastVideo/getting_started/installation/spark/) first.

<div class="fasth3-local-cookbook">
  <strong><a href="/FastVideo/cookbook/">FastVideo cookbook</a></strong>
  <span>Choose a model family, then a recipe. Inference is live. Training and
  evaluation land in the same catalog as they ship.</span>
</div>

> TODO: Add the three weight downloads and one verified generation command
> for each platform.

RTX 5090 and RTX 4090 are next. We have not measured FastH3 on Apple's new
[M5 Max and M5 Ultra Mac Studio](https://www.apple.com/mac-studio/) or
[M5 Pro Mac mini](https://www.apple.com/mac-mini/) yet.

If you work on MLX, Metal, or CUDA, send a kernel, a memory fix, or a
reproducible run. Start from the
[contribution guide](/FastVideo/contributing/overview/) or the
[repository](https://github.com/hao-ai-lab/FastVideo). Include hardware,
settings, timings, and the output.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
We thank the MiniMax team for releasing its weights and code, and our
collaborators at Nuva Lab and NVIDIA FastGen for their work on FastH3 Preview.

<!-- TODO: Final acknowledgements, licenses, and partner wording. -->

## FastVideo team

**Contributor:** Aryan Kumar
<a href="https://github.com/aryan5v" aria-label="Aryan Kumar GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/aryan-kumar01" aria-label="Aryan Kumar LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/aryan_xv" aria-label="Aryan Kumar X"><i class="fab fa-x-twitter"></i></a>  
**Tech lead:** Will Lin
<a href="https://github.com/SolitaryThinker" aria-label="Will Lin GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/will-lin-294920100" aria-label="Will Lin LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/wlsaidhi" aria-label="Will Lin X"><i class="fab fa-x-twitter"></i></a>  
**Advisor:** Hao Zhang
<a href="https://github.com/zhisbug" aria-label="Hao Zhang GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/haozhangml" aria-label="Hao Zhang LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/haozhangml" aria-label="Hao Zhang X"><i class="fab fa-x-twitter"></i></a>

<style>
.fasth3-local-grid,
.fasth3-local-precision-grid {
  display: grid;
  gap: 0.75rem;
  margin: 1.4rem 0 2rem;
}

.fasth3-local-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.fasth3-local-precision-grid {
  grid-template-columns: minmax(6rem, 0.55fr) repeat(3, minmax(0, 1fr));
}

.fasth3-local-grid__header,
.fasth3-local-precision-label,
.fasth3-local-media-placeholder {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: 10px;
}

.fasth3-local-grid__header {
  padding: 0.75rem;
  background: var(--code-bg);
  text-align: center;
  font-weight: 650;
}

.fasth3-local-grid__header small {
  font-weight: 400;
  opacity: 0.7;
}

.fasth3-local-media-placeholder {
  display: grid;
  min-height: 10rem;
  padding: 1rem;
  place-content: center;
  background: linear-gradient(145deg, var(--entry), var(--code-bg));
  color: var(--secondary);
  text-align: center;
  font-size: 0.8rem;
}

.fasth3-local-precision-label {
  display: flex;
  padding: 0.85rem;
  flex-direction: column;
  justify-content: center;
  background: var(--entry);
}

.fasth3-local-precision-label strong {
  font-size: 1.15rem;
}

.fasth3-local-precision-label small {
  margin-top: 0.25rem;
  opacity: 0.7;
}

.fasth3-local-pipeline {
  display: grid;
  grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
  gap: 0.65rem;
  margin: 1.5rem 0 2rem;
  align-items: stretch;
}

.fasth3-local-pipeline > div {
  display: flex;
  min-width: 0;
  padding: 0.9rem;
  flex-direction: column;
  gap: 0.35rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--entry);
}

.fasth3-local-pipeline span {
  font-size: 0.78rem;
  line-height: 1.4;
  opacity: 0.75;
}

.fasth3-local-pipeline b {
  align-self: center;
  opacity: 0.55;
}

.fasth3-local-chart-placeholder {
  display: flex;
  min-height: 13rem;
  margin: 1.4rem 0 2rem;
  padding: 1.2rem;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 0.4rem;
  border: 1px dashed var(--border);
  border-radius: 12px;
  background:
    linear-gradient(to top, transparent 24%, var(--border) 25%, transparent 26%) 0 0 / 100% 25%,
    var(--entry);
  text-align: center;
}

.fasth3-local-chart-placeholder span {
  font-size: 0.8rem;
  opacity: 0.7;
}

.fasth3-local-cookbook {
  display: grid;
  gap: 0.4rem;
  margin: 1.5rem 0 2rem;
  padding: 1.1rem 1.2rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--entry);
}

.fasth3-local-cookbook span {
  font-size: 0.95rem;
  line-height: 1.5;
  opacity: 0.8;
}

@media (max-width: 760px) {
  .nav {
    overflow-x: auto;
    scrollbar-width: none;
  }

  .nav::-webkit-scrollbar {
    display: none;
  }

  .post-content figure,
  .post-content figure div {
    width: 100% !important;
    max-width: 100%;
    min-width: 0;
    overflow-x: auto;
  }

  .post-content figure table {
    width: 100% !important;
    max-width: 100%;
    table-layout: auto;
  }

  .fasth3-local-grid,
  .fasth3-local-precision-grid,
  .fasth3-local-pipeline {
    grid-template-columns: 1fr;
  }

  .fasth3-local-grid__header,
  .fasth3-local-precision-grid > .fasth3-local-grid__header:first-child {
    display: none;
  }

  .fasth3-local-pipeline b {
    transform: rotate(90deg);
  }

  .fasth3-local-precision-label {
    margin-top: 0.65rem;
  }

  .fasth3-local-media-placeholder::before {
    display: block;
    margin-bottom: 0.4rem;
    color: var(--primary);
    content: attr(data-label);
    font-weight: 650;
  }
}
</style>
