+++
title = "FastH3 on Consumer Hardware: RTX GPUs, DGX Spark and Macs, Plus FastH3 Trim"
date = 2026-10-06T00:00:00-07:00
url = "/blogs/fasth3-rtx/"
authors = ["Aryan Kumar", "Will Lin", "Hao Zhang"]
author = "Aryan Kumar, Will Lin, Hao Zhang"
ShowReadingTime = true
draft = false
contentClass = "fasth3-rtx-article"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
[cover]
    image = "img/cover.jpg"
    alt = "Blueprint drawings of a DGX Spark, a Mac Studio, an RTX 4090 and an RTX 5090"
    caption = "FastH3 Trim"
    hidden = true
+++

{{< image src="img/cover.jpg" alt="Blueprint drawings of a DGX Spark, a Mac Studio, an RTX 4090 and an RTX 5090" width="100%" >}}

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fastvideo-fasth3" >}}

FastH3 V2 generates video with synchronized audio in eight steps, and at those eight steps it already beats base H3 on quality. Until now it needed data-center GPUs: its weights take 138 GiB. Today FastH3 V2 runs on a single consumer machine: an RTX 5090, RTX 4090 or RTX PRO 6000, a DGX Spark, or an Apple Silicon Mac. We are also releasing **FastH3 Trim**, an experimental smaller version that removes 8 of the 50 transformer blocks to run faster still.

This post continues [FastH3 Goes Local](/blogs/fasth3-local/), which brought FastH3 to the DGX Spark and the Mac and named the RTX family as the next target.

## TL;DR

- **FastH3 V2 now runs on one consumer GPU, a DGX Spark or a Mac.** We ship it in NVFP4 for Blackwell GPUs and DGX Spark, FP8 for the RTX 4090 and GPUs with less memory, and INT6 for Apple Silicon.
- **Quantization keeps the quality.** Every FP4 layer uses activation scales calibrated on 1,000 prompts, so large activations are not clipped.
- **FastH3 Trim is an experiment in making the model smaller.** It is 4.2× smaller than base H3, faster than V2 on every device and runs in as little as 8 GB of GPU memory, at minimal cost in quality.
- **There is room to improve, and we will keep working on it.** Both models are eight-step distillations; we expect better quality and smaller models in future releases.

## Same prompt, every machine

Each row is one machine. The left clip is FastH3 V2 and the right clip is FastH3 Trim, from the same prompt and seed at 832×480 for 5 s with audio. Turn the audio on. Three more prompts are under [More samples](#more-samples).

<div class="fasth3-rtx-gallery">
  <div></div>
  <div class="fasth3-rtx-colhead">FastH3 V2</div>
  <div class="fasth3-rtx-colhead">FastH3 Trim</div>
  <div class="fasth3-rtx-rowhead"><b>RTX 5090</b><span>32 GB · NVFP4</span></div>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx5090/v2-nvfp4-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 V2 on RTX 5090">
        <source src="img/videos/rtx5090/v2-nvfp4-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>V2 · NVFP4</b><span>14.8 s</span></figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx5090/trim-nvfp4-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 Trim on RTX 5090">
        <source src="img/videos/rtx5090/trim-nvfp4-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>Trim · NVFP4</b><span>13.4 s</span></figcaption>
  </figure>
  <div class="fasth3-rtx-rowhead"><b>RTX PRO 6000</b><span>96 GB · NVFP4</span></div>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx-pro-6000/v2-nvfp4-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 V2 on RTX PRO 6000">
        <source src="img/videos/rtx-pro-6000/v2-nvfp4-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>V2 · NVFP4</b><span>13.5 s</span></figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx-pro-6000/trim-nvfp4-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 Trim on RTX PRO 6000">
        <source src="img/videos/rtx-pro-6000/trim-nvfp4-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>Trim · NVFP4</b><span>12.0 s</span></figcaption>
  </figure>
  <div class="fasth3-rtx-rowhead"><b>RTX 4090</b><span>24 GB · FP8</span></div>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx4090-24gb/v2-fp8-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 V2 on RTX 4090">
        <source src="img/videos/rtx4090-24gb/v2-fp8-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>V2 · FP8</b><span>54.6 s</span></figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx4090-24gb/trim-fp8-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 Trim on RTX 4090">
        <source src="img/videos/rtx4090-24gb/trim-fp8-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>Trim · FP8</b><span>43.9 s</span></figcaption>
  </figure>
  <div class="fasth3-rtx-rowhead"><b>DGX Spark</b><span>128 GB unified · NVFP4</span></div>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="spark-1x/v2-nvfp4-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 V2 on DGX Spark">
        <source src="img/videos/spark-1x/v2-nvfp4-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>V2 · NVFP4</b><span>141.4 s</span></figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="spark-1x/trim-nvfp4-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 Trim on DGX Spark">
        <source src="img/videos/spark-1x/trim-nvfp4-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>Trim · NVFP4</b><span>125.8 s</span></figcaption>
  </figure>
  <div class="fasth3-rtx-rowhead"><b>Mac (M4 Max)</b><span>36 GB unified · INT6</span></div>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="m4max/v2-int6-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 V2 on Mac (M4 Max)">
        <source src="img/videos/m4max/v2-int6-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>V2 · INT6</b><span></span></figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="m4max/trim-int6-corgi-weather.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 Trim on Mac (M4 Max)">
        <source src="img/videos/m4max/trim-int6-corgi-weather.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption><b>Trim · INT6</b><span>925.2 s</span></figcaption>
  </figure>
</div>

## How fast it runs

We report two numbers per machine: a 5 s clip at 832×480 and a 5 s clip at 1344×768, both with audio. Times are end to end on a warm server, from prompt to finished MP4: text encoding, eight denoising steps, video and audio decoding, and export. Each is the median of two runs on each of two prompts.

{{< image src="img/fig_e2e.svg" alt="Paired thin bars per machine, FastH3 V2 and FastH3 Trim, end-to-end seconds for a 5 s, 832×480 clip on a log scale." width="100%" title="Figure 1. End-to-end time for a 5 s, 832×480 clip with audio. Every row is one machine." >}}

<!-- results-table:start -->
| Machine | Memory | V2, 480p | Trim, 480p | V2, 768p | Trim, 768p |
|---|---|---:|---:|---:|---:|
| RTX PRO 6000 | 96 GB | 13.5 s | 12.0 s | 36.5 s | 32.5 s |
| RTX 5090 | 32 GB | 14.8 s | 13.4 s | 38.6 s | 35.4 s |
| RTX 4090 | 24 GB | 54.6 s | 43.9 s | 154.6 s | 132.8 s |
| RTX 4090, 16 GB limit | 16 GB | 79.9 s | 72.1 s | 153.7 s | 139.9 s |
| RTX 4090, 12 GB limit | 12 GB | 91.2 s | 74.1 s | 170.7 s | 147.1 s |
| RTX 4090, 8 GB limit | 8 GB | — | 82.0 s | — | — |
| DGX Spark | 128 GB unified | 141.4 s | 125.8 s | — | 340.1 s |
| 2× DGX Spark | 128 GB each | 87.2 s | 78.3 s | — | — |
| Mac, M4 Max | 36 GB unified | — | 925.2 s | — | — |
<!-- results-table:end -->

## Fitting H3 on one machine

H3 is three networks. A Qwen3-VL text encoder reads the prompt, a 50-block diffusion transformer (DiT) denoises the video and audio latents together, and two VAEs decode the latents into frames and sound. In BF16 these weights total 137.7 GiB, and a 32 GB RTX 5090 cannot hold even the DiT. We reduced each network separately.

{{< image src="img/fig_memory_stack.svg" alt="Stacked bars. BF16 H3: text encoder 62.1 GiB, DiT 65.3 GiB, VAEs 10.3 GiB, 137.7 GiB total. FastH3 Trim with NVFP4: text encoder 15.3, DiT 11.1, VAEs 6.5, 33.0 GiB total." width="100%" title="Figure 2. Checkpoint size by component. The FastH3 Trim NVFP4 release is 4.2× smaller than BF16 H3." >}}

- **Text encoder: 62.1 → 15.3 GiB.** H3 reads hidden state 50 of a 64-layer Qwen3-VL and never generates text, so we remove the last 14 layers and the language-model head without changing the features H3 uses. The remaining linear layers are stored in NVFP4.
- **Transformer: 65.3 GiB in BF16.** For FastH3 V2 we store the attention, MLP and sparse-attention gate weights in NVFP4 (FP8 on GPUs without FP4 support). FastH3 Trim also removes 8 of the 50 blocks and replaces each block's timestep projection with a rank-16 factorization, which brings it to 11.1 GiB in NVFP4.
- **VAEs: 10.3 → 6.5 GiB.** We decode video with the [LynnReal lightweight video VAE](https://huggingface.co/stdstu123/LynnReal-Onmi-light-vae), a distilled 26-block decoder with the same latent interface as the H3 VAE, loaded with [Kijai's INT8 weights](https://huggingface.co/Kijai/MiniMax-H3-experimental). It uses 2.3 GiB of GPU memory, and every device in this post uses the same one.

Size matters for speed. On a 32 GB GPU, the question is whether the DiT can stay in GPU memory between requests. When it can, each request only moves the text encoder in and out.

### RTX 5090

Our first FP4 export quantized only the MLPs, the setting we use on data-center GPUs. On a 5090 that left a 20 GB DiT, because the BF16 attention projections alone take 9.7 GB, and the text encoder no longer fit beside it. Every request moved the DiT to host memory and back: a 480p clip took 26.4 s, and a 768p clip did not fit at all.

With attention and the gate also in NVFP4, the DiT is 11.1 GiB and stays on the GPU, and the text encoder streams in one layer at a time. The same 480p clip now takes 13.4 s, and 768p fits.

PyTorch's pinned-memory allocator rounds each block up to a power of two, so 2.87 GiB of FP4 weights took 5.06 GiB of host RAM, enough to get the process killed in a 60 GB cloud container. We pin one exact-size buffer per module with `cudaHostRegister` instead, which uses 2.90 GiB.

### RTX 4090 and smaller memory

The RTX 4090 has no FP4 tensor cores, so it uses FP8: 8-bit weights with one scale per output channel and 8-bit activations with one scale per token. PyTorch's FP8 matrix multiply with these scales runs at about 70 TFLOPS on a 4090, slower than BF16 at about 160 TFLOPS. The per-tensor FP8 kernel runs at 220–305 TFLOPS, so we call it with unit scales and apply both scale vectors to the output in one fused pass. The result matches per-token, per-channel scaling and costs 5–10% more than per-tensor scaling.

Three more changes bring the 4090 to 43.9 s for a 5 s clip:

- **Sparse attention:** queries and keys are quantized to INT8 for the score computation, while values stay in BF16. The fine attention kernel runs 1.6× faster with about 0.6% relative error.
- **Text encoder:** it streams to the GPU one layer at a time through exact-size pinned buffers, and one fused kernel expands its NVFP4 weights.
- **VAE:** the same INT8 lightweight VAE as every other device, with a fused dequantization step and one shared quantized input for the Q, K and V projections. Decoded frames are bit-identical to the unoptimized path.

For the 16 GB, 12 GB and 8 GB rows, we cap GPU memory on the same 4090. Both models fit in 12 GB, and FastH3 Trim fits in 8 GB; a real card with less memory will be slower.

### DGX Spark and Apple Silicon

A DGX Spark keeps the text encoder, the transformer and both VAEs in its 128 GB of unified memory, so nothing moves between requests. It uses the same NVFP4 checkpoints and lightweight VAE as the 5090, and two Sparks split each request with sequence parallelism. In [FastH3 Goes Local](/blogs/fasth3-local/), a 5 s clip took 243 s on one Spark with the four-step preview model; the eight-step models now do twice as many denoising steps and still finish faster.

On Apple Silicon, both models run in MLX with INT6 weights, the NVFP4 text encoder and the lightweight VAE.

## Four-bit weights without clipping

NVFP4 stores each group of 16 values as 4-bit floats with a shared 8-bit scale, plus one scale per tensor that sets the overall range. For weights, we compute that per-tensor scale from the weights themselves. For activations it must be fixed before the data arrives. The simplest choice, a unit scale, covers magnitudes up to 6 × 448 = 2,688, and H3 activations are much larger.

{{< image src="img/fig_fc_out_amax.svg" alt="Bar chart of the largest input to each block's MLP output projection across 42 blocks, on a log scale. 40 of 42 bars exceed the 2,688 line; block 37 reaches 368,640." width="100%" title="Figure 3. Largest input to each block's MLP output projection, over 1,000 calibration prompts and all eight steps. With a unit scale, everything above the dashed line is clipped." >}}

In 40 of the 42 blocks, the input to the MLP output projection exceeds 2,688. In block 37 it reaches 368,640, 137 times the limit, and a unit scale clips these values on every forward pass. So we calibrate: we ran 1,000 prompts through the full eight-step sampler, recorded the largest input to each linear layer, and stored one static scale per layer in the checkpoint. FastH3 Trim has 294 calibrated layers, covering the attention projections, the MLPs and the sparse-attention gate. This extends the FastH3 V2 NVFP4 recipe from the MLPs to every quantized layer.

## FastH3 Trim: an experiment in pruning

Pruning is our path to smaller models. We remove the blocks we measured as least important, which makes the model smaller and faster but takes some of what it learned with them. FastH3 Trim is our first step.

{{< image src="img/fig_squares.svg" alt="Squares drawn to scale, area equal to transformer checkpoint size. H3 BF16, 65.3 GiB, is tiled with blocks 0 to 49; blocks 6, 7, 9, 13, 15, 16, 22 and 23 are red (removed), and blocks 0, 1, 5, 47, 48 and 49 are outlined as most sensitive. Arrows lead to smaller squares tiled with the same 42 kept blocks: Trim BF16 34.8 GiB (1.9× smaller), FP8 19.9 (3.3×), INT6 14.3 (4.6×), NVFP4 11.1 (5.9×)." width="100%" title="Figure 4. The FastH3 Trim transformer in each format we ship, drawn to scale: area is checkpoint size." >}}

### Choosing the blocks

Our first pruned model chose blocks by their activations, which clearly beat removing blocks at even intervals. We recovered that model with teacher guidance and then used DMD to reduce its sampling steps. Motion coherence, fine detail and prompt adherence stayed weak. Quantization-aware distillation (QAD) did not beat post-training quantization in our side-by-side comparisons.

For FastH3 Trim we measured each block directly. Starting from base H3, we skipped one block at a time and recorded how much the video and audio predictions changed. We tested four examples (motion, speech, music and sound events) at three noise levels, for 600 measurements in total. Each block was ranked by the largest change it caused under any condition, so a block that matters to either video or audio is kept. The first and last blocks changed the output the most. The eight blocks we removed are all in the first half of the network.

### Compressing the timestep conditioning

H3 conditions each block on the diffusion timestep through an AdaLN projection, which maps a 2,688-dimensional time embedding to six modulation vectors. These projections take 24 GiB in BF16 across the 50 blocks. Their input, however, is a smooth function of a single number, the timestep, so it uses very few of its 2,688 dimensions. FastH3 Trim replaces the projections with one shared 2,688→16 basis and a small projection per block. We store the factorized weights in FP16, because BF16 gives about 1.7× larger reconstruction error.

### Training

We trained the new 42-block model directly with eight-step DMD2, using the FastH3 V2 objective. Base H3 initializes both the frozen teacher and the trainable critic, and attention is 80% sparse. The model samples at timesteps 999, 874, 749, 624, 500, 375, 250 and 125.

## Where the time goes

{{< image src="img/fig_stages.svg" alt="100% stacked bars. RTX PRO 6000 Trim NVFP4, 480p, 5 s, 12.0 s: denoise 78%, decode 17%, rest 5%. RTX 5090 Trim NVFP4, 480p, 5 s, 13.4 s: denoise 68%, decode 16%, rest 16%. RTX 4090 Trim FP8, 480p, 5 s, 43.9 s: denoise 75%, decode 17%, rest 7%. DGX Spark Trim NVFP4, 480p, 5 s, 125.8 s: denoise 65%, decode 34%, rest 1%. Mac, M4 Max Trim MLX INT6, 480p, 5 s, 925.2 s: denoise 91%, decode 8%, rest 2%." width="100%" title="Figure 5. Share of end-to-end time per stage." >}}

Figure 5 splits one 5 s, 480p FastH3 Trim clip by stage on each machine. On the RTX PRO 6000 every model stays in GPU memory, and denoising is 78% of the 12.0 s. The RTX 5090 denoises just as fast (9.1 s) and finishes in 13.4 s, because only the text encoder and decoders move between host and GPU. The RTX 4090 has no FP4 tensor cores, so it runs FP8 and streams part of the transformer from host memory each step; denoising is 33.1 s of its 43.9 s. On a DGX Spark, video decoding takes a third of the time (42.6 s), and it is the next stage we will optimize. On a Mac, denoising is 91% of the time because attention still runs on a reference path rather than a tuned Metal kernel.

## Limitations and what comes next

- **FastH3 Trim is experimental.** Busy scenes can show less detail than V2; use V2 when quality matters most.
- **Memory tiers are emulated.** The 16 GB, 12 GB and 8 GB numbers cap memory on a 4090; real cards with that memory will be slower.

## Get the models

| Hardware | FastH3 V2 | FastH3 Trim |
|---|---|---|
| RTX 5090, RTX PRO 6000, DGX Spark (NVFP4) | [`FastVideo-FastH3-8-Step-V2-NVFP4-Consumer`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer) | [`FastVideo-FastH3-Trim-8-Step-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4) |
| RTX 4090 and GPUs down to 8 GB (FP8) | [`FastVideo-FastH3-8-Step-V2-FP8`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-FP8) | [`FastVideo-FastH3-Trim-8-Step-FP8`](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-FP8) |
| Apple Silicon (MLX INT6) | [`FastVideo-FastH3-8-Step-V2-MLX-INT6`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-MLX-INT6) | [`FastVideo-FastH3-Trim-8-Step-MLX-INT6`](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-MLX-INT6) |
| Source weights (BF16) | [`FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) | [`FastVideo-FastH3-Trim-8-Step`](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step) |

All repositories are under the [FastVideo](https://huggingface.co/FastVideo) organization. Multi-GPU data-center serving keeps using [`FastVideo-FastH3-8-Step-V2-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4). Each consumer repository includes the NVFP4 text encoder, the lightweight VAE and a `fastvideo_inference.json` file with the sampling schedule, which FastVideo reads automatically. On one RTX 5090:

```python
import os

from huggingface_hub import hf_hub_download

repo = "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer"
os.environ["FASTVIDEO_H3_PARK_MODULES"] = "vae,audio_vae"  # keep the transformer on the GPU during text encoding
os.environ["FASTVIDEO_H3_ENCODER_LAYERWISE"] = "1"  # stream the text encoder layer by layer
os.environ["FASTVIDEO_H3_ADALN_TABLE"] = hf_hub_download(repo, "transformer/adaln_tables.pt")  # skip 26 GB of AdaLN weights

from fastvideo import VideoGenerator

generator = VideoGenerator.from_config({
    "model_path": repo,
    "engine": {
        "num_gpus": 1,
        "quantization": {"transformer_quant": "NVFP4", "layer_profile": "h3_dit_vsa"},
        "offload": {"text_encoder": True, "pin_cpu_memory": True},
    },
    "pipeline": {"experimental": {"attention_backend": "VIDEO_SPARSE_ATTN_H3", "h3_sequential_load": True}},
})
generator.generate_video(
    prompt="A red fox leaps into deep snow at sunrise and pops back up with snow on its face.",
    height=480, width=832, num_frames=124, guidance_scale=1.0, output_path="out.mp4",
)
```

Swap in `FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4` for the faster experimental model, without the AdaLN table line.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), and we thank the MiniMax team for releasing its weights and code.

The lightweight decoder is the [LynnReal Lightweight Video VAE](https://huggingface.co/stdstu123/LynnReal-Onmi-light-vae) ([paper](https://arxiv.org/abs/2609.15863), [code](https://github.com/LynnReal-AI/LynnReal-Omni)), loaded with the INT8 weights from [Kijai](https://huggingface.co/Kijai)'s [MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental). We thank both.

We thank the NVIDIA Enterprise Products team (Pengcheng Li and Cliff Woolley) for the Video Sparse Attention kernel, and the FlashInfer and NVIDIA Model Optimizer teams for the FP4 kernels and calibration tools. The FP4 sparse attention on RTX GPUs builds on [SageAttention](https://github.com/thu-ml/SageAttention).

The FastVideo team worked closely with [Nuva Lab](https://nuvalab.ai/), [NVIDIA FastGen](https://github.com/NVlabs/FastGen) (Julius Berner, Chao Liu, Arash Vahdat) and the NVIDIA Enterprise Products team on [FastH3](/blogs/fasth3-preview/). We also thank the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/) and [MBZUAI](https://mbzuai.ac.ae/) for their continued sponsorship and support of FastVideo.

## FastVideo team

**Contributor:** [Aryan Kumar](https://github.com/aryan5v)
<a href="https://github.com/aryan5v" aria-label="Aryan Kumar GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/aryan-kumar01" aria-label="Aryan Kumar LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/aryan_xv" aria-label="Aryan Kumar X"><i class="fab fa-x-twitter"></i></a>  
**Tech lead:** [Will Lin](https://github.com/SolitaryThinker)
<a href="https://github.com/SolitaryThinker" aria-label="Will Lin GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/will-lin-294920100" aria-label="Will Lin LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/wlsaidhi" aria-label="Will Lin X"><i class="fab fa-x-twitter"></i></a>  
**Advisor:** [Hao Zhang](https://github.com/zhisbug)
<a href="https://github.com/zhisbug" aria-label="Hao Zhang GitHub"><i class="fab fa-github"></i></a>
<a href="https://www.linkedin.com/in/haozhangml" aria-label="Hao Zhang LinkedIn"><i class="fab fa-linkedin"></i></a>
<a href="https://x.com/haozhangml" aria-label="Hao Zhang X"><i class="fab fa-x-twitter"></i></a>

## More samples

FastH3 V2 (top) and FastH3 Trim (bottom) on one RTX 5090, 832×480, 5 s with audio.

<div class="fasth3-rtx-more">
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx5090/v2-nvfp4-street-food-jingle.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 V2 on RTX 5090: Street-food jingle">
        <source src="img/videos/rtx5090/v2-nvfp4-street-food-jingle.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption>V2 · Street-food jingle</figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx5090/v2-nvfp4-parrot-pirate.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 V2 on RTX 5090: Pirate and parrot">
        <source src="img/videos/rtx5090/v2-nvfp4-parrot-pirate.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption>V2 · Pirate and parrot</figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx5090/v2-nvfp4-grandma-dj.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 V2 on RTX 5090: Grandma DJ">
        <source src="img/videos/rtx5090/v2-nvfp4-grandma-dj.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption>V2 · Grandma DJ</figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx5090/trim-nvfp4-street-food-jingle.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 Trim on RTX 5090: Street-food jingle">
        <source src="img/videos/rtx5090/trim-nvfp4-street-food-jingle.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption>Trim · Street-food jingle</figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx5090/trim-nvfp4-parrot-pirate.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 Trim on RTX 5090: Pirate and parrot">
        <source src="img/videos/rtx5090/trim-nvfp4-parrot-pirate.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption>Trim · Pirate and parrot</figcaption>
  </figure>
  <figure class="fasth3-rtx-clip">
    <div class="fasth3-rtx-frame" data-file="rtx5090/trim-nvfp4-grandma-dj.mp4">
      <video controls playsinline preload="metadata" aria-label="FastH3 Trim on RTX 5090: Grandma DJ">
        <source src="img/videos/rtx5090/trim-nvfp4-grandma-dj.mp4#t=0.1" type="video/mp4">
      </video>
    </div>
    <figcaption>Trim · Grandma DJ</figcaption>
  </figure>
</div>

<style>
.fasth3-rtx-article .fasth3-rtx-todo {
  margin: 1.4rem 0;
  padding: 0.85rem 1rem;
  border: 1.5px dashed #eb6834;
  border-radius: 10px;
  background: rgba(235, 104, 52, 0.07);
  font-size: 0.9rem;
  line-height: 1.5;
}

.fasth3-rtx-article .fasth3-rtx-more {
  display: grid;
  max-width: 640px;
  margin: 0.8rem 0 1.4rem;
  gap: 0.6rem 0.5rem;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.fasth3-rtx-article .fasth3-rtx-more figcaption {
  margin: 0.25rem 0 0;
  color: var(--secondary);
  font-size: 0.72rem;
  line-height: 1.3;
}

.fasth3-rtx-article .fasth3-rtx-grid {
  display: grid;
  margin: 1rem 0 1.6rem;
  gap: 0.9rem 0.6rem;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.fasth3-rtx-article .fasth3-rtx-clip {
  min-width: 0;
  margin: 0;
}

.fasth3-rtx-article .fasth3-rtx-hero {
  margin: 1.4rem 0 0.4rem;
}

.fasth3-rtx-article .fasth3-rtx-frame {
  position: relative;
  aspect-ratio: 832 / 480;
  border: 1.5px dashed var(--border);
  border-radius: 8px;
}

.fasth3-rtx-article .fasth3-rtx-frame--wide {
  aspect-ratio: 1344 / 768;
}

.fasth3-rtx-article .fasth3-rtx-frame::before {
  content: "";
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 0.4rem;
  color: var(--secondary);
  font-size: 0.72rem;
  text-align: center;
}

.fasth3-rtx-article .fasth3-rtx-frame video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 8px;
  background: transparent;
}

.fasth3-rtx-article .fasth3-rtx-frame.is-loaded {
  border-style: solid;
}

.fasth3-rtx-article .fasth3-rtx-frame.is-loaded::before {
  display: none;
}

.fasth3-rtx-article .fasth3-rtx-frame.is-loaded video {
  background: #000;
}

.fasth3-rtx-article .fasth3-rtx-clip > figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 0.2rem 0.45rem;
  margin: 0.4rem 0 0;
  font-size: 0.8rem;
  line-height: 1.35;
}

.fasth3-rtx-article .fasth3-rtx-clip > figcaption span {
  color: var(--secondary);
}

.fasth3-rtx-article table {
  font-size: 0.92rem;
}

.fasth3-rtx-article td {
  font-variant-numeric: tabular-nums;
}

@media (max-width: 760px) {
  .fasth3-rtx-article .fasth3-rtx-grid {
    grid-template-columns: 1fr;
  }
}

.fasth3-rtx-article .fasth3-rtx-gallery {
  display: grid;
  margin: 1rem 0 1.6rem;
  gap: 0.8rem 0.6rem;
  align-items: start;
  grid-template-columns: 7.5rem repeat(2, minmax(0, 1fr));
}

.fasth3-rtx-article .fasth3-rtx-colhead {
  font-size: 0.85rem;
  font-weight: 700;
}

.fasth3-rtx-article .fasth3-rtx-rowhead {
  display: flex;
  flex-direction: column;
  align-self: center;
  gap: 0.15rem;
  font-size: 0.8rem;
  line-height: 1.3;
}

.fasth3-rtx-article .fasth3-rtx-rowhead span {
  color: var(--secondary);
}

@media (max-width: 760px) {
  .fasth3-rtx-article .fasth3-rtx-gallery {
    grid-template-columns: 5.2rem repeat(2, minmax(0, 1fr));
  }
}
</style>

<script>
document.querySelectorAll(".fasth3-rtx-frame video").forEach(function (v) {
  var mark = function () { v.closest(".fasth3-rtx-frame").classList.add("is-loaded"); };
  if (v.readyState >= 1) mark(); else v.addEventListener("loadedmetadata", mark);
});
</script>
