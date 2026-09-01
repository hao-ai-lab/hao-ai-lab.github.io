+++
title = "FastH3 on Apple Silicon and DGX Spark"
date = 2026-09-01T00:00:00-07:00
url = "/blogs/fasth3-local/"
authors = ["Aryan Kumar", "Will Lin", "Hao Zhang"]
author = "Aryan Kumar, Will Lin, Hao Zhang"
ShowReadingTime = true
draft = false
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

{{< image src="img/cover.png" alt="FastH3 on Apple Silicon and DGX Spark" width="100%" >}}

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fastvideo-fasth3" >}}

FastH3 now runs on a Mac and on NVIDIA DGX Spark. Two Sparks can generate
one clip together.

H3 generates video and audio together. That used to mean a data-center GPU.
Our [FastH3 Preview](/blogs/fasth3-preview/) distilled it into a few steps on
Blackwell. This release puts that model on Apple Silicon through MLX, and on a
desktop GB10 through CUDA 13. The Mac path needs 36 GB of unified memory or
more. Spark has 128 GB.

This post also publishes the [FastVideo Cookbook](/FastVideo/cookbook/) for
the first time. MiniMax H3 is on it, with CUDA, native MLX, and a local
OpenAI-compatible server.

## Generated locally

The same prompts on an Apple M4 Max, a DGX Spark, and four GB200s.
Turn the audio on.

<div class="fasth3-local-grid fasth3-local-grid--platforms">
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="M4 Max, space gate">
      <source src="img/videos/platforms/mac-space-gate.mp4" type="video/mp4">
    </video>
    <figcaption><b>M4 Max</b><span>504 s · cold</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="DGX Spark, space gate">
      <source src="img/videos/platforms/spark-astronaut-gateway.mp4" type="video/mp4">
    </video>
    <figcaption><b>DGX Spark</b><span>264 s · cold</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="Four GB200s, space gate">
      <source src="img/videos/platforms/gb200-space-gate.mp4" type="video/mp4">
    </video>
    <figcaption><b>4 GB200s</b><span>10.2 s · first request</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="M4 Max, desert fashion">
      <source src="img/videos/platforms/mac-desert-fashion.mp4" type="video/mp4">
    </video>
    <figcaption><b>M4 Max</b><span>465 s · cold</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="DGX Spark, desert fashion">
      <source src="img/videos/platforms/spark-fashion-highway.mp4" type="video/mp4">
    </video>
    <figcaption><b>DGX Spark</b><span>243 s · cold</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="Four GB200s, desert fashion">
      <source src="img/videos/platforms/gb200-desert-fashion.mp4" type="video/mp4">
    </video>
    <figcaption><b>4 GB200s</b><span>5.1 s · warm</span></figcaption>
  </figure>
</div>

Those six clips are the same recipe: 832×480, 124 frames, four-step FastH3,
full VAE. The chart splits a first generation from a repeat. Spark repeat is
a second generate in the same process, not a loaded server. Qwen and DiT
still reload. Most of those 20 to 31 seconds are VAE compile already paid.
GB200 is a loaded server. 350 s of model start sits outside both bars.

On the Mac, denoising is most of the wait. On Spark with the full VAE,
decode is.

{{< image src="img/fig_platform_stages.svg" alt="Share of end-to-end time per stage, first generation versus repeat, on M4 Max, Spark, and four GB200s" width="100%" title="Figure 1. Same 832×480, 124-frame, full-VAE recipe. Mac repeat is a cached prompt. Spark repeat is the same process. GB200 is a loaded server." >}}

## INT8, INT6, and INT4

Every Mac number in this post comes from an M4 Max with 36 GB of unified
memory.

INT8 keeps more of the original weights. INT4 leaves the most room for
activations. INT6 is the default we timed. Wall clock barely moves across
the three. Peak memory does. Same prompt, same seed.

<div class="fasth3-local-grid">
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="INT8 meadow dialogue">
      <source src="img/videos/quant/int8.mp4" type="video/mp4">
    </video>
    <figcaption><b>INT8</b><span>481 s · 24.2 GiB peak</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="INT6 meadow dialogue">
      <source src="img/videos/quant/int6.mp4" type="video/mp4">
    </video>
    <figcaption><b>INT6</b><span>456 s · 19.5 GiB peak</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="INT4 meadow dialogue">
      <source src="img/videos/quant/int4.mp4" type="video/mp4">
    </video>
    <figcaption><b>INT4</b><span>467 s · 14.8 GiB peak</span></figcaption>
  </figure>
</div>

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
A single bounded read per tensor brings that to about 17 seconds, with the
same hidden features. Cache those embeddings and the next prompt skips this
stage.

After denoising, tiled video decode and a native audio decoder finish the clip
without rebuilding the whole frame buffer at once.

{{< image src="img/fig_mac_phases.svg" alt="Time and memory through one FastH3 generation on an M4 Max" width="100%" title="Figure 2. Cache the prompt and the 17 s encode disappears. TAEH3 drops decode from 102 s to 1 s, and peak decode from 11.0 GiB to 3.6 GiB." >}}

Smaller weights are not always faster. H3 multiplies large video and audio
matrices. At those shapes, unpacking a quantized weight into BF16 and using
MLX's dense matrix multiply beat the quantized kernel. The unpacked copy is
discarded after the multiply.

On a four-step INT6 run, denoising dropped from 386.47 seconds to 348.75
seconds. The video and audio latents were bit-exact. Peak MLX memory rose from
19.31 GiB to 19.46 GiB. That run used 832×480, 124 frames, and dense attention,
so this is the matmul path, not Video Sparse Attention.

The Mac path still implements Video Sparse Attention. Selected video tiles
attend. The rest do not.

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
39 seconds, and a 768×1344, 124-frame run from 772 seconds to 336 seconds.

H3 never uses the last layers of its text encoder. The Spark path skips them,
same as the Mac path. Video Sparse Attention stays on CUDA. The FastVideo
kernel builds from source for `sm_121`.

Two Sparks can run that clip together. Sequence parallel splits denoising and
decode across both GB10s over the QSFP link. Each box still loads components
in phases. The transformer is copied onto both, so neither can skip the
phased load.

On the same 768×1344, 124-frame recipe, two Sparks finished in 292 seconds.
One Spark took 374. TAEH3 on that pair was 195 seconds. A 345-frame clip,
about 14 seconds of video, finished in 581 seconds on the pair with the
full H3 VAE.

{{< image src="img/fig_spark_setups.svg" alt="DGX Spark end-to-end generation time for one Spark and two Sparks, full VAE and TAEH3, at 832 by 480 and 768 by 1344" width="100%" title="Figure 3. Cold process starts on GB10. Two Sparks help. TAEH3 helps more. At 768×1344, TAEH3 decode is 12.5 s, not the 1 s you see at 480p." >}}

Install from the
[CUDA 13 Spark guide](/FastVideo/getting_started/installation/spark/). For two
boxes, follow the
[pair guide](/FastVideo/getting_started/installation/spark_pair/), then pick a
CUDA recipe in the [Cookbook](/FastVideo/cookbook/minimax-h3/).

## Full VAE versus TAEH3

The full H3 VAE is the quality path. [TAEH3](https://github.com/madebyollin/taehv)
is the preview path. It reconstructs the same latents much faster, and fine
detail goes softer. Hair, fabric, and distant backgrounds lose bite.

On an M4 Max, tiled H3 VAE decode took 104 seconds. TAEH3 took one second.
On one Spark, VAE decode was 114 seconds and TAEH3 was 1.3 seconds.
End-to-end with TAEH3 was 134 seconds on that box, 119 seconds on two.
At 768×1344 on two Sparks, TAEH3 decode was 12.5 seconds and the clip
finished in 195 seconds. Use TAEH3 to check a prompt. Render with the full
VAE when you like what you see.

Same prompt and seed. First row is the full VAE. Second row is TAEH3.
Columns are an M4 Max, one Spark, and two Sparks.

<div class="fasth3-local-grid">
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="M4 Max, full VAE">
      <source src="img/videos/taeh3/mac-full-vae.mp4" type="video/mp4">
    </video>
    <figcaption><b>M4 Max</b><span>Full VAE · 451 s</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="One Spark, full VAE">
      <source src="img/videos/taeh3/1spark-full-vae.mp4" type="video/mp4">
    </video>
    <figcaption><b>1 Spark</b><span>Full VAE · 243 s</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="Two Sparks, full VAE">
      <source src="img/videos/taeh3/2spark-full-vae.mp4" type="video/mp4">
    </video>
    <figcaption><b>2 Sparks</b><span>Full VAE · 209 s</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="M4 Max, TAEH3">
      <source src="img/videos/taeh3/mac-taeh3.mp4" type="video/mp4">
    </video>
    <figcaption><b>M4 Max</b><span>TAEH3 · 350 s</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="One Spark, TAEH3">
      <source src="img/videos/taeh3/1spark-taeh3.mp4" type="video/mp4">
    </video>
    <figcaption><b>1 Spark</b><span>TAEH3 · 134 s</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="Two Sparks, TAEH3">
      <source src="img/videos/taeh3/2spark-taeh3.mp4" type="video/mp4">
    </video>
    <figcaption><b>2 Sparks</b><span>TAEH3 · 119 s</span></figcaption>
  </figure>
</div>

## Faster drafts

A native-resolution clip takes a while. That is fine for a final render. It
is a lot to pay to find out the prompt is wrong.

`--fast` denoises fewer video frames, then interpolates back to 124.
This run generated 73 frames. RIFE filled the rest. Audio keeps its full
duration.

`--fast-spatial` denoises a smaller canvas, then resamples up. The clip
below is the conservative setting, 672×384 up to 832×480. Composition and
fine detail get softer. Treat it as a preview knob, not a final render.

Same prompt and seed on the M4 Max. INT6, cached prompt, full VAE.

<div class="fasth3-local-grid fasth3-local-grid--modes">
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="Baseline forest clip">
      <source src="img/videos/drafts/baseline.mp4" type="video/mp4">
    </video>
    <figcaption><b>Baseline</b><span>457 s · 19.5 GiB</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="Temporal fast forest clip">
      <source src="img/videos/drafts/temporal-fast.mp4" type="video/mp4">
    </video>
    <figcaption><b>Temporal fast</b><span>248 s · 1.84×</span></figcaption>
  </figure>
  <figure class="fasth3-local-clip">
    <video controls playsinline preload="metadata" aria-label="Spatial fast forest clip">
      <source src="img/videos/drafts/spatial-fast.mp4" type="video/mp4">
    </video>
    <figcaption><b>Spatial fast</b><span>272 s · 1.68×</span></figcaption>
  </figure>
</div>

## FastVideo Cookbook

Open the [Cookbook](/FastVideo/cookbook/), pick a model, pick a recipe, and copy
a command we actually run.

H3 is there with CUDA, the four-step preview, LoRA, native MLX, and a
two-Spark recipe.

You can also serve FastH3. Start the server once on CUDA or on MLX. Then
change prompts from the playground, from cURL, or from an OpenAI-compatible
SDK in your app. Later prompts reuse that process. You do not reload the
model for every try. Open the [H3 recipes](/FastVideo/cookbook/minimax-h3/)
or the [server guide](/FastVideo/cookbook/openai-api/). Spark users should
install from the
[CUDA 13 Spark guide](/FastVideo/getting_started/installation/spark/) first.
Two boxes should follow the
[pair guide](/FastVideo/getting_started/installation/spark_pair/).

<div class="fasth3-local-cookbook">
  <strong><a href="/FastVideo/cookbook/">FastVideo Cookbook</a></strong>
  <span>Maintained inference recipes, starting with MiniMax H3. Distillation,
  training, and evaluation land in the same catalog as they ship.</span>
</div>

We are still cutting latency, adding distilled models, and looking at
schedules with fewer than four steps. The RTX family, including the 5090 and
4090, is the next CUDA focus.

Apple just announced
[M6 in the Mac mini and M5 Ultra in the Mac Studio](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/).
We have not run FastH3 on those, or on M5 Max or M5 Pro Mac mini. Every Mac
number here is from an M4 Max. The MLX path is the same on those chips. They
should be faster, especially M5 Ultra with its unified memory and GPU. We
want to measure them.

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

## FastVideo team

**Contributors:** Aryan Kumar
<a href="https://github.com/aryan5v" aria-label="Aryan Kumar GitHub"><i class="fab fa-github"></i></a>
<a href="https://x.com/aryan_xv" aria-label="Aryan Kumar X"><i class="fab fa-x-twitter"></i></a>,
Satyam Srivastava
<a href="https://github.com/Satyam-53" aria-label="Satyam Srivastava GitHub"><i class="fab fa-github"></i></a>
<a href="https://x.com/Sat_53" aria-label="Satyam Srivastava X"><i class="fab fa-x-twitter"></i></a>,
Kyle Hu
<a href="https://github.com/KyleNeverGivesUp" aria-label="Kyle Hu GitHub"><i class="fab fa-github"></i></a>,
Ishan Vaish
<a href="https://github.com/Ishxn20" aria-label="Ishan Vaish GitHub"><i class="fab fa-github"></i></a>  
**Tech lead:** Will Lin
<a href="https://github.com/SolitaryThinker" aria-label="Will Lin GitHub"><i class="fab fa-github"></i></a>
<a href="https://x.com/wlsaidhi" aria-label="Will Lin X"><i class="fab fa-x-twitter"></i></a>  
**Advisor:** Hao Zhang
<a href="https://github.com/zhisbug" aria-label="Hao Zhang GitHub"><i class="fab fa-github"></i></a>
<a href="https://x.com/haozhangml" aria-label="Hao Zhang X"><i class="fab fa-x-twitter"></i></a>

<style>
.fasth3-local-article .fasth3-local-grid {
  display: grid;
  width: 100%;
  margin: 1.6rem 0 2.2rem;
  gap: 1.35rem 0.7rem;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.fasth3-local-article .fasth3-local-clip {
  min-width: 0;
  margin: 0;
  overflow: visible;
  background: transparent;
}

.fasth3-local-article .fasth3-local-clip video {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 8px;
  background: #000;
}

.fasth3-local-article .fasth3-local-clip > figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.5rem;
  align-items: baseline;
  margin: 0.45rem 0 0;
  color: var(--primary);
  font-size: 0.8rem;
  font-weight: 400;
  line-height: 1.3;
  letter-spacing: 0.01em;
}

.fasth3-local-article .fasth3-local-clip > figcaption b {
  font-weight: 600;
}

.fasth3-local-article .fasth3-local-clip > figcaption span {
  color: var(--secondary);
  font-weight: 400;
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

  .fasth3-local-article .fasth3-local-grid,
  .fasth3-local-pipeline {
    width: 100%;
    margin-inline: 0;
    grid-template-columns: 1fr;
  }

  .fasth3-local-article .fasth3-local-clip {
    width: 100% !important;
    max-width: 100%;
    overflow: visible;
  }

  .fasth3-local-pipeline b {
    transform: rotate(90deg);
  }
}
</style>
