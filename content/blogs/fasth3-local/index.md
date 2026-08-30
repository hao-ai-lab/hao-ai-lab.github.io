+++
title = "FastH3 goes local: Video and audio on Apple Silicon and DGX Spark"
date = 2026-08-31T00:00:00-07:00
url = "/blogs/fasth3-local/"
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
draft = true
contentClass = "post-content-justified"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
+++

<!--
Publication checklist:
- Replace the cover placeholder and add a [cover] block to the front matter.
- Confirm the release name, step count, model IDs, license text, and final author list.
- Replace every media placeholder with an audible MP4 from the final checkpoint.
- Fill the fixed-protocol INT8, INT6, and INT4 table from one quiet M4 Max run.
- Confirm that the final FastVideo release revision contains PR #1788.
- Fill the Spark benchmark row from the final public checkpoint and merged FastVideo revision.
- Replace the architecture and phase-breakdown TODOs with final figures.
- Confirm whether spatial fast mode ships in this release. Upstream MLX does not wire it yet.
- Confirm that Spark PRs #1761, #1780, and #1785 are merged or remove their dependent claims.
- Run all final commands from a clean checkout and check every public link.
- Review the MiniMax H3 Community License and add the final acknowledgement list.
-->

<div class="fasth3-local-cover-placeholder" role="img" aria-label="Placeholder for the FastH3 local AI cover image">
  <span>TODO: cover image</span>
  <strong>FastH3 on Apple Silicon and DGX Spark</strong>
  <small>Local text-to-video-and-audio generation</small>
</div>

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fastvideo-fasth3" >}}

**TL;DR:** FastH3 now has local inference paths for Apple Silicon through MLX
and for NVIDIA DGX Spark through CUDA. The Apple release provides INT8, INT6,
and INT4 weights for Macs with at least 36 GB of unified memory. It generates
video and synchronized audio in one local pipeline. Temporal fast mode denoises
fewer video frames, then reconstructs the requested frame count with MLX RIFE.
The DGX Spark path brings the same FastH3 family to a 128 GB desktop
Grace-Blackwell system. We are releasing the weights, inference recipes, and
the runtime work so that the community can measure, change, and improve the
whole path. The MLX release also includes the exact-quality wide-matrix dispatch
from FastVideo PR #1788.

<!-- TODO: Replace "now" and the release statements above with the exact public
artifact status on publication day. -->

## A large video model belongs on your desk

Our first [FastH3 Preview](/blogs/fasth3-preview/) showed what post-training can
do on data-center Blackwell GPUs. MiniMax H3 is one of the strongest open-weight
models we have tested for joint video and audio. FastH3 distills it into a few
denoising steps and uses Video Sparse Attention to cut attention work. The
result preserves H3's joint video-and-audio generation while removing most of
the diffusion steps.

This release asks a different question. How much of that system can run beside
you, on a machine you own?

Local generation changes the development loop. A prompt, an input image, and a
draft video do not need to leave the machine. There is no per-generation API
charge. More useful for systems work, every slow operation is visible. You can
profile the text encoder, replace an attention kernel, test another quantization
format, or change how components move through memory. The local machine becomes
both the product and the lab bench.

Video makes this a hard systems problem. H3 is much larger than the language
models that made local AI popular, and it generates two synchronized outputs.
The runtime must condition the prompt, denoise packed video and audio tokens,
decode both latent streams, and mux them into one playable file. Porting the
transformer alone would not make the model local. We had to make the entire
pipeline fit.

## What we are releasing

The two local paths share the FastH3 model family, but each runtime follows its
hardware.

{{< table title="Table 1. FastH3 local release surface. Final artifact names and measurements remain publication TODOs." >}}

| Platform | Runtime | Release formats | Initial scope | Memory target |
| :------- | :------ | :-------------- | :------------ | :------------ |
| Apple Silicon Mac | Native MLX and Metal | Affine INT8, INT6, and INT4 weights | T2VA, temporal fast mode, optional VSA | 36 GB unified memory or more |
| NVIDIA DGX Spark | PyTorch and CUDA 13 on GB10 | TODO: final FastH3 checkpoint and precision | T2VA on one desktop GB10 | 128 GB unified memory |
| NVIDIA B200 | FastVideo CUDA reference | Base H3 for the cross-platform comparison | T2VA reference | Data-center reference only |

{{</ table >}}

The Mac release target starts at 36 GB because that is the smallest machine on
which we validated the complete H3 MLX pipeline. Our measurements use an Apple
M4 Max with 36 GB of unified memory. Treat that configuration as the release
floor, not as a prediction for every Mac with a similar chip name.

Apple's new desktop lineup gives local AI developers more room. The new
[Mac Studio](https://www.apple.com/mac-studio/) offers M5 Max and M5 Ultra with
36 GB to 512 GB of unified memory. The new
[Mac mini](https://www.apple.com/mac-mini/) offers M5 Pro configurations with
up to 64 GB. The M6 Mac mini tops out at 32 GB, so it falls below our current
FastH3 memory target even though it advances Apple's on-device AI hardware.
We have not benchmarked FastH3 on these new chips yet. Our published numbers
will remain M4 Max numbers until we do.

DGX Spark attacks the same problem from the CUDA side. Its GB10 combines an Arm
CPU, a Blackwell GPU, and
[128 GB of unified memory](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
in a desktop system. That memory capacity is the reason H3 is interesting on
Spark. Its unified design is also the reason ordinary GPU offload assumptions
break, as we will explain below.

## The same prompt on three classes of hardware

This grid will compare the local releases against Base H3 on B200. Mac and
Spark use FastH3. The B200 column uses Base H3 as a quality reference. Every
row will use the exact prompt printed below the videos, the same requested
duration, and the same aspect ratio.

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

<!-- TODO: Add the two full prompts in collapsible blocks. Include speech tags,
seed, duration, frames, resolution, steps, checkpoint revision, and runtime
revision. Do not compare unmatched seeds or schedules. -->

The comparison is deliberately not a speed chart. Base H3 and FastH3 use
different denoising schedules. The three devices also have different memory
systems and kernels. The grid answers a visual question. The performance tables
later in the post answer a timing question under a fixed model and protocol.

## What it took to run H3 through MLX

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) gives arrays on
Apple Silicon a shared CPU and GPU memory model, lazy evaluation, compiled graph
transforms, and access to custom Metal kernels. Those properties are useful,
but they do not make a 30-billion-parameter-class video pipeline fit by
themselves.

We built a native path for each phase instead of wrapping the CUDA pipeline.
The runtime produces the same kind of output as H3, an H.264 video with stereo
AAC audio, but it schedules the work around unified memory.

<div class="fasth3-local-pipeline" aria-label="FastH3 MLX phase diagram">
  <div><strong>1. Condition</strong><span>Stream Qwen3-VL weights and cache prompt embeddings</span></div>
  <b aria-hidden="true">&rarr;</b>
  <div><strong>2. Denoise</strong><span>Run the quantized H3 DiT on packed video and audio tokens</span></div>
  <b aria-hidden="true">&rarr;</b>
  <div><strong>3. Decode</strong><span>Decode video tiles and stereo audio with native MLX VAEs</span></div>
  <b aria-hidden="true">&rarr;</b>
  <div><strong>4. Mux</strong><span>Write one synchronized MP4</span></div>
</div>

### Stream the text encoder instead of loading it

H3 reads an intermediate hidden state from Qwen3-VL. It does not need the top
14 decoder layers or the final norm. Even the part it needs is too large to
keep beside the DiT on a 36 GB Mac.

The MLX conditioner reads only the token embedding rows in the prompt. It then
loads one decoder layer at a time, evaluates the layer, and releases its weights
before opening the next one. The full embedding table never reaches the GPU,
and the 50-layer graph never accumulates in memory. At each point, the runtime
holds the current hidden state and roughly one BF16 decoder layer. Prompt
embeddings can then be cached on disk, so a repeated prompt skips the encoder.

This is more than an optimization. Without streaming, the local pipeline does
not fit.

### Keep one heavyweight phase resident at a time

The same rule governs the rest of the pipeline. After conditioning, the runtime
clears the conditioner before it loads the DiT. It releases the DiT before the
native video VAE decodes tiled frames. The audio VAE and muxer follow. MLX cache
cleanup and peak-memory accounting run between phases.

This means the peak is set by the largest active phase, not by the sum of the
text encoder, DiT, video VAE, and audio VAE. The final benchmark chart will show
that phase boundary directly.

<div class="fasth3-local-chart-placeholder">
  <strong>TODO: MLX phase time and peak-memory chart</strong>
  <span>Conditioning, denoising, video decode, RIFE, audio decode, and mux</span>
</div>

### Quantize the weights that dominate memory

The Mac checkpoints store the DiT's large matrix weights in affine INT8, INT6,
or INT4 groups. Norms, modulation values, and attention activations stay at
higher precision. This is weight-only quantization. It lowers the checkpoint
and resident weight cost without forcing every operation into low-bit math.

Shipping three formats matters because a local runtime has more than one useful
operating point. INT8 keeps more weight precision. INT6 is our balanced format
for the 36 GB M4 Max development machine. INT4 leaves the most memory for
activations and higher resolutions. The final comparison will let readers judge
the image and motion tradeoff instead of reducing it to one aggregate score.

The runtime work also exposed a counterintuitive result. At H3's very wide
packed-token matrices, MLX's affine quantized matrix multiplication can be
slower than dequantizing a weight for the operation and calling a dense BF16
GEMM. The release includes the H3-specific dispatch from
[FastVideo PR #1788](https://github.com/hao-ai-lab/FastVideo/pull/1788). Stored
INT4, INT6, and INT8 weights stay quantized. When the packed row count reaches
the measured crossover, the runtime dequantizes the matrix for that operation
and calls the dense GEMM without caching a second weight copy. The default
crossover is 768 rows, and `FASTVIDEO_MLX_DQ_GEMM=0` restores the original
quantized matrix multiplication path.

On an M4 Max with 36 GiB of unified memory, this dispatch reduced four-step INT6
DiT time from 386.47 seconds to 348.75 seconds for the 832 by 480 by 124
workload. That is a 9.8 percent reduction with bit-exact video and audio latents.
Peak memory changed from 19.31 GiB to 19.46 GiB. INT8 and INT4 also crossed over
at the production matrix shape, while small INT4 matrices still favored the
quantized path. That last result is why the release uses a shape gate instead of
changing every low-bit multiplication.

### Make sparse attention native to Metal

FastH3's trained Video Sparse Attention policy keeps about 10 percent of
eligible video-to-video attention tiles. Our first MLX backend gathered those
tiles in chunks and sent them through MLX scaled dot-product attention. We then
built a SIMD-group Metal backend for H3's tile size 64 and head dimension 128.

On the development workload, the SIMD backend reduced the DiT forward from
499.3 seconds to 436.4 seconds and reduced the sparse attention microbenchmark
from 586 milliseconds to 444 milliseconds. It is opt-in. Dynamic sparse routing
produced a different valid sample in our comparison, so the reference VSA path
remains the automatic choice. The merged implementation and its validation are
in [FastVideo PR #1776](https://github.com/hao-ai-lab/FastVideo/pull/1776).

This is exactly why we want the work in public. There is large performance
headroom in Metal attention, quantized matrix multiplication, compilation, and
decode. Each candidate still needs an end-to-end quality gate.

## Two ways to spend less denoising compute

Temporal fast mode reduces the number of video latent rows that pass through
the DiT. After denoising, the native MLX RIFE backend interpolates the video to
the requested frame count. Audio remains at full duration, so the shortcut does
not shorten the soundtrack. This mode is available through `--fast`.

Spatial fast mode applies the same idea across pixels. It denoises a smaller
latent grid, then reconstructs the requested output size before decode. The
mode can compose with temporal fast mode, reducing both video rows and spatial
tokens.

<!-- TODO: Spatial fast mode is part of the launch plan but is not wired into
the upstream MiniMax H3 MLX pipeline as of FastVideo a4d9a75e2. Add the exact
algorithm, flag, quality gate, and measurements only after the implementation
lands. -->

<div class="fasth3-local-grid fasth3-local-grid--modes">
  <div class="fasth3-local-grid__header">Baseline</div>
  <div class="fasth3-local-grid__header">Temporal fast</div>
  <div class="fasth3-local-grid__header">Spatial fast</div>
  <div class="fasth3-local-media-placeholder" data-label="Baseline">TODO: fixed-prompt baseline</div>
  <div class="fasth3-local-media-placeholder" data-label="Temporal fast">TODO: same prompt and seed</div>
  <div class="fasth3-local-media-placeholder" data-label="Spatial fast">TODO after mode lands</div>
</div>

The first validated INT6 M4 Max run produced 124 frames at 832 by 480 in
565.37 seconds end to end. Denoising took 383.73 seconds and peaked at 19.63
GiB of MLX memory. That was a bring-up measurement from the merged MLX runtime,
not the final release benchmark. We will rerun the full matrix after the weights
and performance stack are frozen.

## INT8, INT6, and INT4 under one protocol

The precision grid will use the same prompts, seeds, dimensions, frame count,
step schedule, decode settings, and FastVideo revision for every cell. Each row
will report end-to-end time, denoise time, and the largest phase peak. That
protocol prevents a lower resolution or a cached prompt from masquerading as a
quantization speedup.

<div class="fasth3-local-precision-grid">
  <div class="fasth3-local-grid__header">Format</div>
  <div class="fasth3-local-grid__header">Prompt A</div>
  <div class="fasth3-local-grid__header">Prompt B</div>
  <div class="fasth3-local-grid__header">Prompt C</div>
  <div class="fasth3-local-precision-label"><strong>INT8</strong><small>TODO time and peak</small></div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt A">TODO: INT8 A</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt B">TODO: INT8 B</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt C">TODO: INT8 C</div>
  <div class="fasth3-local-precision-label"><strong>INT6</strong><small>TODO time and peak</small></div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt A">TODO: INT6 A</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt B">TODO: INT6 B</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt C">TODO: INT6 C</div>
  <div class="fasth3-local-precision-label"><strong>INT4</strong><small>TODO time and peak</small></div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt A">TODO: INT4 A</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt B">TODO: INT4 B</div>
  <div class="fasth3-local-media-placeholder" data-label="Prompt C">TODO: INT4 C</div>
</div>

{{< table title="Table 2. Fixed-protocol Apple M4 Max benchmark. Fill from one bounded final run." >}}

| Format | Checkpoint size | End-to-end | Denoise | Largest phase peak | Quality notes |
| :----- | --------------: | ---------: | ------: | -----------------: | :------------ |
| INT8 | TODO | TODO | TODO | TODO | TODO: fixed-prompt review |
| INT6 | TODO | TODO | TODO | TODO | TODO: fixed-prompt review |
| INT4 | TODO | TODO | TODO | TODO | TODO: fixed-prompt review |

{{</ table >}}

## Why Spark needed different memory work

CUDA software often treats CPU memory and GPU memory as separate pools.
Offloading a component to the host can free device memory on a discrete GPU.
GB10 is integrated. Its CPU and GPU share the same physical memory, so copying
a component from a CUDA allocation into a host allocation can briefly hold both
copies in the same 128 GB pool.

We saw this while loading H3's text encoder. The old offload path added about
34 GB and took 5 minutes 49 seconds after the checkpoint read. Detecting unified
memory and skipping that copy reduced the post-read load step to milliseconds.
The fix is merged in
[FastVideo PR #1710](https://github.com/hao-ai-lab/FastVideo/pull/1710).

We also stopped building the top 14 Qwen3-VL layers that H3 never reads. The
change removed 13.7 GB of BF16 parameters while preserving the selected hidden
state exactly. That work is part of the merged H3 text-encoder optimization in
[FastVideo PR #1732](https://github.com/hao-ai-lab/FastVideo/pull/1732).

FastH3 still needs staged component loading on one Spark. Its eager components
sum to about 124 GiB, while the system exposes about 121 GiB to the workload.
The release candidate loads a component on first use and releases it after its
last pipeline stage. A 124-frame, 1344 by 768 candidate run then reached a 69.2
GiB allocated peak and completed in 902 seconds with a rank-reduced FastH3 v1
checkpoint. The implementation is still under review in
[FastVideo PR #1761](https://github.com/hao-ai-lab/FastVideo/pull/1761), so this
is engineering evidence, not the final launch number.

{{< table title="Table 3. DGX Spark release benchmark. Replace the candidate row with the final public artifact." >}}

| Checkpoint | Output | Precision | End-to-end | Peak allocated | FastVideo revision |
| :--------- | :----- | :-------- | ---------: | -------------: | :----------------- |
| TODO: final FastH3 Spark release | TODO | TODO | TODO | TODO | TODO |

{{</ table >}}

The final Spark grid will include audible examples and the exact peak reported
by `GenerationResult.peak_memory_mb`. We are adding that output to the public
H3 examples in
[FastVideo PR #1785](https://github.com/hao-ai-lab/FastVideo/pull/1785) so the
number is reproducible without a local patch.

## Run it through the FastVideo cookbook

The new [FastVideo model-family cookbook](/FastVideo/cookbook/) is the shortest
way to find a maintained inference path. The
[MiniMax H3 page](/FastVideo/cookbook/minimax-h3/) lets you choose a full H3,
FastH3, or MLX recipe and keeps device claims tied to recorded runs.

<!-- TODO: Replace these commands with the final pre-quantized model IDs. The
current upstream guide converts the MLX DiT locally. -->

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
uv pip install -e ".[mlx]"

python examples/inference/basic/mlx_fasth3.py \
  --model-root ./FastH3-RELEASE \
  --mlx-checkpoint ./FastH3-MLX/int6 \
  --prompt "(S1) A presenter says <d>[English] FastH3 runs on my Mac.</d>" \
  --height 480 --width 832 --num-frames 124 --seed 2026 \
  --output-path ./outputs/fasth3_int6.mp4
```

Add `--fast` to enable temporal interpolation. The final release instructions
will also list the spatial fast flag if that path clears its quality gate.

DGX Spark uses the CUDA 13 installation path in the
[FastVideo Spark guide](/FastVideo/getting_started/installation/spark/). The
final command will pin the public Spark checkpoint and every memory flag needed
to reproduce Table 3.

## Where local FastH3 goes next

This release is a starting point for local video systems work. We are working
on stronger quality at each low-bit format, faster Metal kernels, lower decode
cost, and better memory scheduling. On NVIDIA hardware, RTX 5090 and RTX 4090
support is next. Those cards have smaller memory pools than Spark, so they will
need their own checkpoint and activation-memory plan. We will publish support
after an end-to-end run passes, not based on component-level fit.

We also want FastVideo's cookbook to grow beyond inference. The first release
provides maintained commands for supported model families. Future entries will
cover post-training, evaluation, optimization, and deployment as those recipes
become reproducible.

There is plenty of useful work for contributors now. Metal attention still has
headroom. H3's wide low-bit matrix multiplications behave differently from
small language-model shapes. Spatial fast mode needs an H3 implementation and
quality study. Spark needs better low-memory checkpoints and fewer reloads
between generations. New M5 Max, M5 Ultra, and higher-memory M5 Pro systems
also need clean, comparable measurements.

If you want to help, start with the
[FastVideo repository](https://github.com/hao-ai-lab/FastVideo), reproduce one
recorded workload, and report the exact commit, checkpoint, hardware, runtime,
timings, memory peak, and output. A small result with a complete receipt is more
useful than a large speedup measured against a different workload.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
We thank the MiniMax team for releasing the model weights and code that made
this work possible. FastH3 Preview was developed by FastVideo with Nuva Lab and
the NVIDIA FastGen team.

<!-- TODO: Confirm the complete acknowledgement list, contributor names,
funding, licenses, and partner wording before publication. -->

<style>
.fasth3-local-cover-placeholder {
  display: grid;
  min-height: 22rem;
  margin: 0 0 1.5rem;
  padding: 2rem;
  place-content: center;
  gap: 0.55rem;
  border: 1px dashed color-mix(in srgb, var(--primary) 45%, transparent);
  border-radius: 16px;
  background:
    radial-gradient(circle at 20% 20%, rgba(49, 130, 246, 0.2), transparent 32%),
    radial-gradient(circle at 80% 75%, rgba(132, 78, 245, 0.2), transparent 35%),
    var(--entry);
  text-align: center;
}

.fasth3-local-cover-placeholder span {
  width: fit-content;
  margin: 0 auto;
  padding: 0.25rem 0.65rem;
  border-radius: 999px;
  background: var(--code-bg);
  font-size: 0.75rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.fasth3-local-cover-placeholder strong {
  font-size: clamp(1.6rem, 4vw, 3rem);
  line-height: 1.08;
}

.fasth3-local-cover-placeholder small {
  font-size: 1rem;
  opacity: 0.72;
}

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
