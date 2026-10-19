+++
title = "FastH3-QAD-v2 and NVFP4 VAE Decoders: Better Quality, Faster Decoding, and Community LoRAs"
date = 2026-10-19T00:00:00-07:00
url = "/blogs/fasth3-v21/"
authors = ["Aryan Kumar", "Will Lin", "Hao Zhang"]
author = "Aryan Kumar, Will Lin, Hao Zhang"
ShowReadingTime = true
draft = false
contentClass = "fasth3-v21-article"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
[cover]
    image = "img/cover.png"
    alt = "FastH3-QAD-v2"
    hidden = false
+++

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fastvideo-fasth3" >}}

FastH3 V2 runs an eight-step video and audio model in NVFP4 on consumer GPUs. FastH3-QAD-v2 raises quality without changing the architecture, memory footprint or file layout by training the 4-bit model instead of only calibrating it.

We are releasing three pieces:

- **FastH3-QAD-v2**, the quantization-aware distillation (QAD) checkpoint of FastH3 V2. It keeps V2's memory footprint and file layout, and it matches or beats V2 end to end with FP4 attention.
- **Two NVFP4 VAE decoders**, trained on FastH3's own outputs. The 36-layer decoder matches the original H3 decoder. The 8-layer decoder runs 15× faster. Both beat LightVAE on its own benchmark.
- **Community LoRAs in FastVideo.** LoRAs trained for MiniMax H3 in ComfyUI and other common formats now load directly, including on NVFP4 models.

Together they form the foundation for tomorrow's release of **FastH3 OmniRef**, which uses all three. More [at the end](#tomorrow-fasth3-omniref).

<figure class="v21-hero-video">
  <video controls playsinline preload="metadata" poster="img/cover.png" width="1344" height="768" aria-label="FastH3-QAD-v2 world tour generated on two DGX Sparks">
    <source src="img/videos/world-tour.mp4#t=0.1" type="video/mp4">
  </video>
  <figcaption>Six 10 s clips generated on two DGX Sparks with FastH3-QAD-v2 and both NVFP4 decoders. Turn the audio on.</figcaption>
</figure>

## FastH3-QAD-v2: QAD closes the PTQ quality gap

FastH3-QAD-v2 is the quantization-aware distillation (QAD) checkpoint of FastH3 V2. V2 shipped with post-training quantization (PTQ): calibrate activation scales on 1,000 prompts, then round each weight to 4 bits. QAD goes one step further. The teacher is V2 in full precision. The student runs the deployed 4-bit weights and activations on all 350 linear layers and trains through the rounding, so the weights adapt to the quantized forward pass instead of carrying its error into the output.

- **Matching every step.** At each of the eight sampling steps, the student sees the teacher's input and learns the teacher's output. A quarter of the samples run on the student's own trajectory, so it also learns to recover from its own small errors.
- **Text-only training.** The recipe uses 3,000 text prompts, 300 steps on two GB200 nodes, and about five hours.
- **A shared recipe.** The same QAD method is the training path for the next FastH3 models, including OmniRef. <!-- TODO: link the QAD recipe docs page once FastVideo #1989 merges. -->

The result, measured as error against full-precision V2 on 30 held-out prompts (lower is better):

<div class="v21-table">
<table>
<thead><tr><th>Error vs full-precision V2</th><th>V2 NVFP4</th><th>FastH3-QAD-v2 NVFP4</th><th>Change</th></tr></thead>
<tbody>
<tr><td>Each sampling step, video + audio</td><td>0.431</td><td class="win">0.351</td><td class="win">−19%</td></tr>
<tr><td>Video</td><td>0.471</td><td class="win">0.382</td><td class="win">−19%</td></tr>
<tr><td>Audio</td><td>0.391</td><td class="win">0.319</td><td class="win">−18%</td></tr>
<tr><td>Final video after all 8 steps</td><td>0.817</td><td class="win">0.713</td><td class="win">−13%</td></tr>
<tr><td>Memory, file layout</td><td colspan="3">unchanged</td></tr>
</tbody>
</table>
</div>

### Why the VSA gates stay in BF16

Video Sparse Attention uses small gate projections to decide which video tiles each token attends to. Quantizing the linears changes the arithmetic. Changing the gates changes the selection itself, so the attention pattern no longer matches the one the model was trained with. FastH3-QAD-v2 keeps the gates and normalization layers in BF16, and trains only the q/k/v/out and FFN projections, the same set the NVFP4 export quantizes.

The result preserves tile selection while letting every quantized projection adapt to 4-bit arithmetic. We evaluate each QAD checkpoint on rendered clips through the deployed pipeline, in addition to teacher-forced metrics, before promoting it.

## NVFP4 VAE decoders

Every FastH3 clip ends in the VAE decoder, which turns latents into pixels. After the transformer and the text encoder moved to NVFP4, the decoder was the last part of FastH3 not in 4-bit, and on a DGX Spark it had become a third of the total time: the INT8 lightweight VAE took **43 s** to decode a 5 s, 480p clip, next to 68 s of denoising.

{{< image src="img/fig_spark_decode.svg" alt="Video decode time for a 5 s 832x480 clip on one DGX Spark: INT8 light VAE 42.8 s, dense light VAE 9.6 s, our 8-layer NVFP4 decoder 1.6 s." width="100%" title="Figure 1. Video decode time for a 5 s, 832×480 clip on one DGX Spark." >}}

We trained two NVFP4 decoders, both distilled from the original 36-layer H3 decoder:

- **Quality, 36 layers.** The full decoder with every block in NVFP4. 1.29 GiB instead of 4.5 GiB, and within 0.1 dB of the original.
- **Fast, 8 layers.** 8 of the 36 blocks, distilled and quantized. 0.31 GiB and 15× faster than the original.

### Trained on FastH3's own outputs

A decoder only has to be good on the latents it is given. We train on **latents generated by FastH3** itself, mixed with encoded real video, so the decoder sees the same distribution at training and inference:

- **Teacher:** the original H3 decoder in full precision. **Loss:** pixel L1 plus LPIPS against the teacher's output, on random latent tiles.
- **Students:** NVFP4 weights and activations in every transformer block; input/output projections and norms stay in higher precision.
- **What the error is:** with 4-bit weights alone the decoder is nearly lossless. What remains comes from 4-bit activations, spread evenly across the blocks.

### Fused FP4 kernels

A 4-bit GEMM is only fast if everything around it is too. The decoder's blocks run on fused Triton and CUTLASS FP4 kernels, which quantize activations, apply scales and run the GEMM without round trips through memory. CUDA graphs remove launch overhead, and weights stay packed in 4 bits on the GPU. The fused path is bit-exact with the unfused NVFP4 reference and cut the 36-layer decoder from 1.88 s to 0.53 s at 480p.

### Head to head with LightVAE

[LightVAE](https://huggingface.co/lightx2v/LightVAE) is a widely used public lightweight decoder for H3. LightVAE-Lite keeps 12 of the 36 decoder layers and reports about 3× the original's speed. We compared against it **on LightVAE's own protocol**: 1344×768, 124 frames, bf16, eager mode, one warm-up and the median of three runs, on one GB200. LightVAE runs with its own code and weights, and every decoder receives the same latents from the original H3 encoder. The 20 test clips are 10 H3 generations and 10 live-action clips that no decoder has seen. Our harness reproduces LightVAE's PSNR exactly, and their speed claims hold (2.85× vs their claimed 2.96×).

<div class="v21-table">
<table>
<thead><tr><th>Decoder</th><th>Decode FPS, eager</th><th>Decode FPS, best config</th><th>PSNR (dB)</th><th>LPIPS</th><th>Size</th></tr></thead>
<tbody>
<tr><td>Original H3</td><td>17.9</td><td>62.1</td><td class="best">33.73</td><td class="best">0.071</td><td>4.5 GiB</td></tr>
<tr><td>LightVAE-Lite (decoder)</td><td>51.0 (2.86×)</td><td>183.7</td><td>31.74 (−2.0)</td><td>0.109</td><td>1.5 GiB</td></tr>
<tr><td>LightVAE-Lite (encoder + decoder)</td><td>49.3</td><td>183.1</td><td>30.57 (−3.2)</td><td>0.140</td><td>1.5 GiB</td></tr>
<tr class="ours"><td>Ours, 36-layer NVFP4</td><td class="win">70.0 (3.9×)</td><td>104.3</td><td class="win">33.63 (−0.10)</td><td class="win">0.073</td><td class="win">1.29 GiB</td></tr>
<tr class="ours"><td>Ours, 8-layer NVFP4</td><td class="best">274.6 (15.4×)</td><td class="best">418</td><td>31.73 (= Lite)</td><td class="win">0.096</td><td class="best">0.31 GiB</td></tr>
</tbody>
</table>
</div>

<p class="v21-legend"><span class="best">best in column</span> <span class="win">ours, better than LightVAE-Lite</span> "Best config" puts all spatial tiles in one call and adds <code>torch.compile</code>, applied equally to every model.</p>

{{< image src="img/fig_vae_pareto.svg" alt="Decode latency against PSNR for the original H3 decoder, LightVAE-Lite, TAEH3 and our two NVFP4 decoders." width="100%" title="Figure 2. Quality against decode latency for H3 VAE decoders on LightVAE's protocol (1344×768, one GB200). Lower latency and higher PSNR are better." >}}

- **36-layer vs LightVAE-Lite:** 1.9 dB better and essentially lossless (−0.10 dB against the original, where Lite loses 2.0), faster on their protocol (70 vs 51 FPS) and smaller. The one place it trails: in the best config, Lite's decoder is faster (184 vs 104 FPS), at 1.9 dB lower quality.
- **8-layer vs LightVAE-Lite:** the same PSNR (31.73 vs 31.74 dB) with better LPIPS (0.096 vs 0.109), 5.4× faster on their protocol, 2.3× faster in the best config, and 5× smaller.
- **On real footage and on generated video, at 480p, 720p and 768p,** the ranking is the same.
- **On our 20-clip protocol, Lite loses 2.0 dB; its README reports 1.25 dB.** Its paper's real-dataset results, about −3 dB on UCF-101 and DAVIS, are consistent with our measurement.

<!-- TODO(fig): zoomed crop grid on hard content (face, small text, foliage): original | ours 36L | ours 8L | LightVAE-Lite | TAEH3. Optional: a wipe slider original vs 8L. -->

## The whole stack in NVFP4

With the decoder done, every large component of FastH3 text-to-video now runs in NVFP4: the text encoder, the transformer and the VAE decoder. Here is what that buys against the same model kept in BF16.

{{< image src="img/fig_stack_size.svg" alt="Stacked bars of the weights FastH3 loads for text-to-video: 118.3 GiB for the same model in BF16 and 30.0 GiB for the FastH3-QAD-v2 NVFP4 stack." width="100%" title="Figure 3. Weights loaded for text-to-video, by component." >}}

<div class="v21-table">
<table>
<thead><tr><th>Component</th><th>Same model in BF16</th><th>FastH3-QAD-v2 stack</th><th>Smaller by</th></tr></thead>
<tbody>
<tr><td>Transformer linear layers (21.2B parameters)</td><td>39.5 GiB</td><td class="win">11.1 GiB, NVFP4</td><td class="win">3.6×</td></tr>
<tr><td>Timestep (AdaLN) projections (13.0B)</td><td>24.2 GiB</td><td class="win">0.14 GiB, precomputed table</td><td class="win">170×</td></tr>
<tr><td>Other transformer weights (0.8B)</td><td>1.6 GiB</td><td>1.6 GiB, BF16</td><td>—</td></tr>
<tr><td>Text encoder, Qwen3-VL, 50 layers (25.8B)</td><td>48.0 GiB</td><td class="win">15.3 GiB, NVFP4</td><td class="win">3.1×</td></tr>
<tr><td>VAE decoder (2.4B)</td><td>4.5 GiB</td><td class="win">1.29 GiB (36 layers) / 0.31 GiB (8 layers)</td><td class="win">3.5× / 15×</td></tr>
<tr><td>Audio VAE</td><td>0.6 GiB</td><td>0.6 GiB</td><td>—</td></tr>
<tr class="ours"><td>Total</td><td>118.3 GiB</td><td class="best">30.0 GiB / 29.0 GiB</td><td class="best">3.9× / 4.1×</td></tr>
</tbody>
</table>
</div>

- **NVFP4 stores a weight in about 4.5 bits** (4-bit values plus a shared FP8 scale for every 16), so 4-bit layers come out about 3.6× smaller than BF16.
- **The timestep projections are not quantized at all.** FastH3 samples on a fixed eight-step schedule, so their outputs are precomputed once into a 155 MB table and the 24 GiB of weights are never loaded.
- **What stays in BF16** is small and sensitive: norms, embeddings, the input and output projections, the text encoder's vision tower and the audio VAE.
- **The BF16 column is the same layers FastH3 runs.** H3 reads layer 50 of its 64-layer Qwen3-VL text encoder, so we keep only those 50 layers. The BF16 H3 checkpoint as released is 137.7 GiB, because it also carries the last 14 encoder layers, the language-model head, and the FP32 VAE encoder and decoder.
- **Not all of it is on the GPU at once.** On a 32 GB RTX 5090, the transformer (12.7 GiB) and the decoder stay on the GPU, while the text encoder streams in one layer at a time from host memory and the timestep projections are never loaded. A DGX Spark keeps everything in its 128 GB of unified memory.

### End-to-end latency

Same protocol as our last post: end to end on a warm server from prompt to finished MP4, the median of four runs, with model load excluded. The figure uses the FP4 VSA attention lane.

{{< image src="img/fig_e2e.svg" alt="Two-panel grouped bar chart. Top: 5 s 832×480 clips for V2, FastH3-QAD-v2 with the 36-layer decoder, and FastH3-QAD-v2 with the 8-layer decoder. Bottom: 10 s 1344×768 clips for the two FastH3-QAD-v2 decoders. Log scale." width="100%" title="Figure 4. End-to-end time with audio. Top: a 5 s, 832×480 clip. Bottom: a 10 s, 1344×768 clip. Every row is one machine; all times use FP4 VSA." >}}

## Community LoRAs

The H3 community has published many LoRAs, from camera moves to film looks. Most use ComfyUI or trainer layouts. FastVideo loads them directly, including on NVFP4 models:

```python
from fastvideo import VideoGenerator
from fastvideo.api import ComponentConfig, GeneratorConfig, PipelineSelection

generator = VideoGenerator.from_config(
    GeneratorConfig(
        model_path="FastVideo/FastH3-QAD-v2",
        pipeline=PipelineSelection(
            components=ComponentConfig(lora_path="pablodawson/MiniMax-H3-360-Orbit-LoRA", lora_strength=0.8),
        ),
    ))
```

- **Formats:** ComfyUI, fal, ai-toolkit and musubi-tuner layouts, with `lora_A/lora_B` or `lora_down/lora_up` factors and per-layer alpha.
- **On 4-bit models:** NVFP4 and FP8 weights stay packed and untouched; each LoRA runs as a small bf16 side branch next to its layer. Strength 0 reproduces the base model exactly, and strength can change at runtime.
- **Clear load errors:** a LoRA that matches no layers, or too few, fails with a list of what did not match instead of silently returning the base model.

Most community LoRAs target base H3's 50-step schedule. FastH3 samples in eight steps, so style and motion LoRAs usually carry over, but the best strength can differ. Turbo LoRAs are unnecessary.

## Tomorrow: FastH3 OmniRef

Tomorrow we release **FastH3 OmniRef**: the general model and its training code in BF16, plus a consumer-hardware release. The consumer release uses the QAD recipe above, extended to 4-bit attention, and the NVFP4 decoders.

<!-- TODO: one teaser clip. -->

## Known limitations

- **The 8-layer decoder trades quality for speed.** Use the 36-layer decoder when detail matters.
- **The NVFP4 decoders need a Blackwell GPU:** RTX 50 series, RTX PRO 6000, DGX Spark, or B200/GB200.
- **LoRAs on FastH3 Trim are not supported yet.** Trim removes transformer blocks, so LoRAs trained on full H3 do not line up with its layers.

## Get the models

| | |
|---|---|
| FastH3-QAD-v2, NVFP4 (RTX 5090, RTX PRO 6000, DGX Spark) | [`FastH3-QAD-v2`](https://huggingface.co/FastVideo/FastH3-QAD-v2) |
| NVFP4 VAE decoders: `quality-36L/` and `fast-8L/` | [`FastH3-NVFP4-VAE-Decoder`](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder) |

<!-- TODO: usage snippet selecting the decoder (env/config) once the public loader path is final. -->

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3); we thank the MiniMax team for releasing its weights and code. We thank the [LightX2V](https://github.com/ModelTC/LightX2V) team for releasing LightVAE with its benchmark code, which made a fair comparison possible, and the H3 community for its LoRAs.

We thank the NVIDIA Enterprise Products team (Pengcheng Li and Cliff Woolley) for the Video Sparse Attention kernel, and the FlashInfer and NVIDIA Model Optimizer teams for the FP4 kernels and calibration tools. We also thank [Nuva Lab](https://nuvalab.ai/), [NVIDIA FastGen](https://github.com/NVlabs/FastGen), the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/) and [MBZUAI](https://mbzuai.ac.ae/) for their collaboration and support.

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

<style>
.fasth3-v21-article .v21-table { overflow-x: auto; margin: 1rem 0 0.6rem; }
.fasth3-v21-article .v21-table table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
.fasth3-v21-article .v21-table th, .fasth3-v21-article .v21-table td { padding: 0.45rem 0.6rem; text-align: right; border-bottom: 1px solid var(--border); }
.fasth3-v21-article .v21-table th:first-child, .fasth3-v21-article .v21-table td:first-child { text-align: left; }
.fasth3-v21-article .v21-table tr.ours td:first-child { font-weight: 700; }
.fasth3-v21-article .v21-table tr.ours { background: rgba(42, 111, 219, 0.06); }
.fasth3-v21-article td.win { font-weight: 700; color: #1f5fc4; }
.fasth3-v21-article td.best { font-weight: 700; color: #1f5fc4; box-shadow: inset 0 0 0 2px #2a6fdb; border-radius: 4px; }
.fasth3-v21-article .v21-legend { font-size: 0.8rem; color: var(--secondary); margin: 0.2rem 0 1.4rem; }
.fasth3-v21-article .v21-legend span { display: inline-block; margin-right: 0.8rem; padding: 0 0.35rem; font-weight: 700; color: #1f5fc4; }

.fasth3-v21-article .v21-legend span.best { box-shadow: inset 0 0 0 2px #2a6fdb; border-radius: 4px; }
.fasth3-v21-article .v21-hero-video { margin: 1.2rem 0 1.8rem; }
.fasth3-v21-article .v21-hero-video video { display: block; width: 100%; height: auto; border-radius: 10px; background: #000; }
.fasth3-v21-article .v21-hero-video figcaption { margin-top: 0.5rem; font-size: 0.9rem; color: var(--secondary); }
</style>
