+++
title = "FastH3 on Apple Silicon and DGX Spark"
date = 2026-08-31T00:00:00-07:00
url = "/blogs/fasth3-local/"
authors = ["Aryan Kumar", "Satyam Srivastava", "Will Lin", "Hao Zhang"]
author = "Aryan Kumar, Satyam Srivastava, Will Lin, Hao Zhang"
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
- Keep draft = true until weights, videos, and public commands are final.
- Confirm checkpoint IDs, licenses, and partner wording.
- Fill the video grids and phase chart from final runs.
- Test cookbook commands and every public link.

Mac evidence, M4 Max 36 GB:
- Wide-M affine INT6, 832x480x124, 4 steps, dense attention: DiT 386.47s to
  348.75s, bit-exact latents, peak 19.31 to 19.46 GiB.
- Uncached conditioner: about 80s to about 15s, exact hidden features.
- TAEH3 decode vs tiled H3 VAE: 1.44s vs 107.90s, 3.62 vs 11.03 GiB. Preview
  quality, not lossless.

Spark GB10, FastH3 4-step VSA-DataFree, 768x1344x124, seed 2026:
- Sequential start, GPU-direct DiT, full VAE: DiT load 445s to 39s, e2e 772s
  to 336s.
- TAEH3 preview decode: 2.4s vs 68s VAE, 224s e2e.
- Lazy load: Qwen, then DiT, then VAE. Geometry from checkpoint JSON.
-->

{{< image src="img/cover.png" alt="FastH3 on Apple Silicon and DGX Spark" width="100%" >}}

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fastvideo-fasth3" >}}

FastH3 now runs on a Mac, and on NVIDIA DGX Spark.

H3 generates video and audio together. That used to mean a data-center GPU.
Our [FastH3 Preview](/blogs/fasth3-preview/) distilled it into a few steps on
Blackwell. This release puts that model on Apple Silicon through MLX, and on a
desktop GB10 through CUDA 13. The Mac path targets 36 GB of unified memory or
more. Spark has 128 GB.

With this post we are also introducing the [FastVideo Cookbook](/FastVideo/cookbook/).
This is the first time we are putting it out in public. MiniMax H3 is on it,
with CUDA and native MLX paths.

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

## INT8, INT6, and INT4

Every Mac number in this post comes from an M4 Max with 36 GB of unified
memory.

INT8 keeps more of the original weights. INT6 is the format we used for the
Mac timings later in this post. INT4 leaves the most room for activations.
Lower precision can change detail and motion. Same prompt, same seed, three
formats.

<div class="fasth3-local-grid">
  <div class="fasth3-local-grid__header">INT8<br><small>Time: TODO · Peak: TODO</small></div>
  <div class="fasth3-local-grid__header">INT6<br><small>Time: TODO · Peak: TODO</small></div>
  <div class="fasth3-local-grid__header">INT4<br><small>Time: TODO · Peak: TODO</small></div>
  <div class="fasth3-local-media-placeholder" data-label="INT8">TODO: INT8</div>
  <div class="fasth3-local-media-placeholder" data-label="INT6">TODO: INT6</div>
  <div class="fasth3-local-media-placeholder" data-label="INT4">TODO: INT4</div>
</div>

<!-- TODO: One matched prompt across INT8, INT6, and INT4. Fill time and peak
memory. Link weights from the headers. -->

## How H3 runs on a Mac

The weights are only part of the problem. H3 also needs a large text encoder,
working memory for denoising, and decoders for video and audio. Load all of
that at once on a 36 GB Mac and there is nothing left to generate with.

So the runtime never tries. It runs in phases. Encode the prompt, denoise,
decode, export. Each phase loads what it needs and frees the rest.

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
rows the prompt uses. An uncached prompt used to spend about 80 seconds here.
A single bounded read per tensor brings that to about 15 seconds, with the
same hidden features. Cache those embeddings and a repeated prompt skips this
stage.

After denoising, tiled video decode and a native audio decoder finish the clip
without reconstructing the whole frame buffer at once.

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
is a desktop Blackwell machine. GB10 GPU, 128 GB of unified LPDDR5X, CUDA 13,
ARM64. The FastH3 CUDA path from the Preview release now runs on that box.

The model fits. Loading it the usual way does not.

There is no separate VRAM. CPU and GPU share one pool, at roughly 270 GB/s,
about a tenth of datacenter HBM. FastH3's encoder, transformer, and decoders
add up to more than the 121 GB a workload actually gets. Keep them all
resident and the process dies before the first frame.

So the pipeline never holds them together. It encodes the prompt, drops the
text encoder, loads the transformer, denoises, drops the transformer, then
loads the VAE. Patch size and compression ratios come from the checkpoint
config, so decode does not keep a 65 GB DiT loaded just to read a patch size.

On a discrete GPU, copying weights to the host frees device memory. On Spark
that copy lands in the same pool. We stopped it. DiT weights load straight
onto the GPU. On one GB10 that cut the transformer load from 445 seconds to
39 seconds, and a 768 by 1344, 124-frame run from 772 seconds to 336 seconds.

H3 never uses the last layers of its text encoder. The Spark path skips them,
same as the Mac path. Video Sparse Attention stays on CUDA. The FastVideo
kernel builds from source for `sm_121`.

Install from the
[CUDA 13 Spark guide](/FastVideo/getting_started/installation/spark/), then
pick a CUDA recipe in the [Cookbook](/FastVideo/cookbook/minimax-h3/).

{{< table title="DGX Spark · FastH3 on one GB10" >}}

| Output | Decode | Generation time |
| :----- | :----- | --------------: |
| 768×1344, 124 frames, 4-step FastH3 | Full H3 VAE | 336 s |
| 768×1344, 124 frames, 4-step FastH3 | TAEH3 preview | 224 s |

{{</ table >}}

<!-- TODO: Confirm these rows against the final public checkpoint and command.
GB10, VSA-DataFree, seed 2026, sequential/lazy load, GPU-direct DiT, n=1.
TAEH3 is preview quality. Add peak allocated memory if the public example
prints it. -->

## Full VAE versus TAEH3

The full H3 VAE is the quality path. [TAEH3](https://github.com/madebyollin/taehv)
is the preview path. It reconstructs the same latents much faster, and fine
detail goes softer. Hair, fabric, and distant backgrounds lose bite. That is
expected.

On an M4 Max, tiled H3 VAE decode took 108 seconds. TAEH3 took 1.44 seconds.
On a GB10, VAE decode was 68 seconds and TAEH3 was 2.4 seconds. End-to-end
dropped from 336 seconds to 224 seconds. Use TAEH3 to check a prompt. Render
with the full VAE when you like what you see.

Same prompt and seed on each row. Left is the full VAE. Right is TAEH3.

<div class="fasth3-local-grid fasth3-local-grid--taeh3">
  <div class="fasth3-local-grid__header">Apple M4 Max · full VAE</div>
  <div class="fasth3-local-grid__header">Apple M4 Max · TAEH3</div>
  <div class="fasth3-local-media-placeholder" data-label="Mac, full VAE">TODO: Mac full VAE</div>
  <div class="fasth3-local-media-placeholder" data-label="Mac, TAEH3">TODO: Mac TAEH3</div>
  <div class="fasth3-local-grid__header">DGX Spark · full VAE</div>
  <div class="fasth3-local-grid__header">DGX Spark · TAEH3</div>
  <div class="fasth3-local-media-placeholder" data-label="Spark, full VAE">TODO: Spark full VAE</div>
  <div class="fasth3-local-media-placeholder" data-label="Spark, TAEH3">TODO: Spark TAEH3</div>
</div>

<!-- TODO: Matched prompt, seed, resolution, and duration. Mac and Spark may
differ in precision. Caption the quality drop. Do not autoplay muted. -->

## Faster drafts

A native-resolution clip takes a while. That is fine for a final render. It
is a lot to pay to find out the prompt is wrong.

`--fast` denoises fewer video frames, then interpolates back to the requested
length. Audio keeps its full duration.

`--fast-spatial` is the MLX preview knob. It denoises a smaller canvas, then
resamples the frames up. Composition and fine detail get softer. That is the
trade. Glance at a prompt this way, then run without it when you are ready
to keep the clip.

<div class="fasth3-local-grid fasth3-local-grid--modes">
  <div class="fasth3-local-grid__header">Baseline</div>
  <div class="fasth3-local-grid__header">Temporal fast</div>
  <div class="fasth3-local-grid__header">Spatial fast</div>
  <div class="fasth3-local-media-placeholder" data-label="Baseline">TODO: fixed-prompt baseline</div>
  <div class="fasth3-local-media-placeholder" data-label="Temporal fast">TODO: same prompt and seed</div>
  <div class="fasth3-local-media-placeholder" data-label="Spatial fast">TODO: same prompt and seed</div>
</div>

<!-- TODO: Matched examples with time, memory, output dimensions, and audio.
Spatial fast is MLX. Call out the quality drop in the caption. -->

## FastVideo Cookbook

Open the [Cookbook](/FastVideo/cookbook/), pick a model, pick a recipe, and copy
a command we actually run.

H3 is there with CUDA, the four-step preview, LoRA, and native MLX. Open the
[H3 recipes](/FastVideo/cookbook/minimax-h3/) and generate. Spark users should
install from the
[CUDA 13 Spark guide](/FastVideo/getting_started/installation/spark/) first.

<div class="fasth3-local-cookbook">
  <strong><a href="/FastVideo/cookbook/">FastVideo Cookbook</a></strong>
  <span>Maintained inference recipes, starting with MiniMax H3. Distillation,
  training, and evaluation land in the same catalog as they ship.</span>
</div>

> TODO: Add the three weight downloads and one verified generation command
> for each platform.

We are still cutting latency, adding more distilled models, and looking at
schedules with fewer than four steps. The RTX family, including the 5090 and
4090, is the next CUDA focus.

We have not run FastH3 on M5 Max, M5 Ultra Mac Studio, M5 Mac Pro, M5 Mac mini,
or M6 yet. Every Mac number here is from an M4 Max. Those chips should be
faster. We want to keep making the Apple Silicon path better.

If you measure a new machine, improve a kernel, or hit a bug, start from the
[contribution guide](/FastVideo/contributing/overview/) or the
[repository](https://github.com/hao-ai-lab/FastVideo). Include hardware,
settings, timings, and the output.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
We thank the MiniMax team for releasing its weights and code.

[Nuva Lab](https://nuvalab.ai/) and
[NVIDIA FastGen](https://github.com/NVlabs/FastGen) collaborated on
[FastH3 Preview](/blogs/fasth3-preview/).

We thank Ollin Boer Bohan for [TAEH3](https://github.com/madebyollin/taehv),
the optional preview decoder this release uses. The Mac path is built on
[MLX](https://github.com/ml-explore/mlx) and the community around it.

<!-- TODO: Final licenses and partner wording. -->

## FastVideo team

**Contributors:** Aryan Kumar
<a href="https://github.com/aryan5v" aria-label="Aryan Kumar GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/aryan-kumar01" aria-label="Aryan Kumar LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/aryan_xv" aria-label="Aryan Kumar X"><i class="fab fa-x-twitter"></i></a>,
Satyam Srivastava
<a href="https://github.com/Satyam-53" aria-label="Satyam Srivastava GitHub"><i class="fab fa-github"></i></a>  
**Tech lead:** Will Lin
<a href="https://github.com/SolitaryThinker" aria-label="Will Lin GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/will-lin-294920100" aria-label="Will Lin LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/wlsaidhi" aria-label="Will Lin X"><i class="fab fa-x-twitter"></i></a>  
**Advisor:** Hao Zhang
<a href="https://github.com/zhisbug" aria-label="Hao Zhang GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/haozhangml" aria-label="Hao Zhang LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/haozhangml" aria-label="Hao Zhang X"><i class="fab fa-x-twitter"></i></a>

<style>
.fasth3-local-grid {
  display: grid;
  gap: 0.75rem;
  margin: 1.4rem 0 2rem;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.fasth3-local-grid--taeh3 {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.fasth3-local-grid__header,
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
  .fasth3-local-grid--taeh3,
  .fasth3-local-pipeline {
    grid-template-columns: 1fr;
  }

  .fasth3-local-grid__header {
    display: none;
  }

  .fasth3-local-pipeline b {
    transform: rotate(90deg);
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
