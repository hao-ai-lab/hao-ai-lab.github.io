+++
title = "FastH3 V2.1 and NVFP4 VAE Decoders: Better Quality, Faster Decoding, and Community LoRAs"
date = 2026-10-19T00:00:00-07:00
url = "/blogs/fasth3-v21/"
authors = ["Aryan Kumar", "Will Lin", "Hao Zhang"]
author = "Aryan Kumar, Will Lin, Hao Zhang"
ShowReadingTime = true
draft = true
contentClass = "fasth3-v21-article"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
[cover]
    image = "img/cover.jpg"
    alt = "TODO cover"
    caption = "FastH3 V2.1"
    hidden = true
+++

<div class="v21-placeholder">Video montage placeholder: V2.1 clips with audio (close-up faces, fast motion), a decoder side-by-side, and one community LoRA. Replaces the cover image.</div>

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fastvideo-fasth3" >}}

Two weeks ago we brought [FastH3 to consumer hardware](/blogs/fasth3-rtx/): an RTX 5090, RTX 4090, DGX Spark or Mac. Today we are releasing three things that make it better:

- **FastH3 V2.1**, a new NVFP4 checkpoint of FastH3 V2 with higher quality at the same speed and memory.
- **Two NVFP4 VAE decoders**, trained on FastH3's own outputs. The 36-layer decoder matches the original H3 decoder; the 8-layer decoder is 15× faster than it. Both beat LightVAE, believed to be the best public lightweight decoder for H3.
- **Community LoRAs in FastVideo.** LoRAs trained for MiniMax H3 in ComfyUI and other common formats now load directly, including on NVFP4 models.

Together they lay the foundation for tomorrow's release of **FastH3 OmniRef**, which uses all three. More [at the end](#tomorrow-fasth3-omniref).

## TL;DR

- **V2.1 improves V2 NVFP4** with quantization-aware distillation: sharper faces and fine texture, with no new artifacts, and no change in speed, memory or file layout. It is a drop-in replacement.
- **The VAE decoder was the last part of FastH3 that was not in NVFP4.** We trained two NVFP4 decoders on FastH3 latents and wrote fused FP4 kernels for them. On one DGX Spark, decoding a 5 s, 480p clip drops from 43 s to 1.6 s, and the whole clip from 141 s to 62 s.
- **Both decoders beat LightVAE on its own benchmark.** The 36-layer decoder is faster than LightVAE-Lite and 1.9 dB better. The 8-layer decoder matches Lite's quality at 5.4× its speed and a fifth of its size.
- **The whole stack is now 4-bit.** Text encoder, transformer and VAE decoder all run in NVFP4: 30 GiB of weights instead of 118 GiB for the same model in BF16, 3.9× smaller.
- **Bring your own LoRA.** Point FastVideo at a MiniMax H3 LoRA. On NVFP4 and FP8 models it runs beside the 4-bit weights without modifying them.

## FastH3 V2.1

V2.1 is a new NVFP4 checkpoint of FastH3 V2. Same model, same files, same speed; better videos.

<!-- TODO(fig): side-by-side clips, V2 | V2.1, same seed, DGX Spark: elderly-man close-up, laughing woman, mountain biker, night-market violinist. Source: ~/Desktop/V2-QAD-Spark/fg300/ -->

Our first NVFP4 release used post-training quantization: calibrate activation scales on 1,000 prompts, then round every weight to 4 bits. Rounding nudges every prediction slightly, and over eight sampling steps the nudges add up to softer faces and less fine detail. For V2.1 we trained the 4-bit model to compensate, with **quantization-aware distillation (QAD)**:

- **Teacher and student.** The teacher is V2 in full precision. The student is V2 with 4-bit weights and activations on all 350 linear layers, and it trains *through* the 4-bit rounding, so it learns weights that work well after rounding.
- **Matching every step.** At each of the eight sampling steps, the student sees the teacher's input and learns to make the teacher's prediction, for video and audio. A quarter of the samples run on the student's own trajectory, so it also learns to recover from its own small errors.
- **No data needed.** Training uses only text prompts: 3,000 prompts, 300 steps on two GB200 nodes, about five hours.

The result, measured as error against full-precision V2 on 30 held-out prompts (lower is better):

<div class="v21-table">
<table>
<thead><tr><th>Error vs full-precision V2</th><th>V2 NVFP4</th><th>V2.1 NVFP4</th><th>Change</th></tr></thead>
<tbody>
<tr><td>Each sampling step, video + audio</td><td>0.431</td><td class="win">0.351</td><td class="win">−19%</td></tr>
<tr><td>Video</td><td>0.471</td><td class="win">0.382</td><td class="win">−19%</td></tr>
<tr><td>Audio</td><td>0.391</td><td class="win">0.319</td><td class="win">−18%</td></tr>
<tr><td>Final video after all 8 steps</td><td>0.817</td><td class="win">0.713</td><td class="win">−13%</td></tr>
<tr><td>Speed, memory, file layout</td><td colspan="3">unchanged</td></tr>
</tbody>
</table>
</div>

### What we learned: freeze what you don't quantize

Our first QAD run trained every linear layer, including the small gates that Video Sparse Attention uses to decide which parts of the video each token attends to. The training metric improved. The videos got worse: on fast motion, a dancer grew a ghostly double.

The gates are not where quantization error lives, but changing them changes *what the model looks at*, and no per-step error metric catches that. Restoring the original gates removed the artifacts completely. V2.1 freezes the gates and normalization layers, and it came out both cleaner and better on the metric (0.351 vs 0.362 at the same step).

<!-- TODO(fig): 3-panel still, V2 | QAD with trained gates (ghost dancer) | V2.1. Source: ~/Desktop/V2-QAD-Spark/gate-test/frame_dancers_60.png -->

We now check every QAD run on real videos, rendered through the deployed pipeline, before trusting its numbers.

## NVFP4 VAE decoders

Every FastH3 clip ends in the VAE decoder, which turns latents into pixels. After the transformer and the text encoder moved to NVFP4, the decoder was the last part of FastH3 not in 4-bit, and on a DGX Spark it had become a third of the total time: the INT8 lightweight VAE took **43 s** to decode a 5 s, 480p clip, next to 68 s of denoising.

{{< image src="img/fig_spark_decode.svg" alt="Video decode time for a 5 s 832x480 clip on one DGX Spark: INT8 light VAE 42.8 s, dense light VAE 9.6 s, our 8-layer NVFP4 decoder 1.6 s." width="100%" title="Figure 1. Video decode time for a 5 s, 832×480 clip on one DGX Spark." >}}

We trained two NVFP4 decoders, both distilled from the original 36-layer H3 decoder:

- **Quality, 36 layers.** The full decoder with every block in NVFP4. 1.29 GiB instead of 4.5 GiB, and within 0.1 dB of the original.
- **Fast, 8 layers.** 8 of the 36 blocks, distilled and quantized. 0.31 GiB and 15× faster than the original.

### Trained on FastH3's own outputs

A decoder only has to be good on the latents it is given. Our first attempt trained on encoded real video and looked fine on real clips, but lost detail on FastH3 generations. The final decoders train on **latents generated by FastH3** itself (V2, V1, Trim and OmniRef), mixed with encoded real video:

- **Teacher:** the original H3 decoder in full precision. **Loss:** pixel L1 plus LPIPS against the teacher's output, on random latent tiles.
- **Students:** NVFP4 weights and activations in every transformer block; input/output projections and norms stay in higher precision.
- **What the error is:** with 4-bit weights alone the decoder is nearly lossless. What remains comes from 4-bit activations, spread evenly across the blocks.

They are trained on FastH3's latent distribution, which is H3's: we expect them to work with other H3-based models as well.

### Fused FP4 kernels

A 4-bit GEMM is only fast if everything around it is too. The decoder's blocks run on fused Triton and CUTLASS FP4 kernels, which quantize activations, apply scales and run the GEMM without round trips through memory. CUDA graphs remove launch overhead, and weights stay packed in 4 bits on the GPU. The fused path is bit-exact with the unfused NVFP4 reference and cut the 36-layer decoder from 1.88 s to 0.53 s at 480p.

### Head to head with LightVAE

[LightVAE](https://huggingface.co/lightx2v/LightVAE) is believed to be the best public lightweight decoder for H3: LightVAE-Lite keeps 12 of the 36 decoder layers and reports about 3× the original's speed. We compared against it **on LightVAE's own protocol**: 1344×768, 124 frames, bf16, eager mode, one warm-up and the median of three runs, on one GB200. LightVAE runs with its own code and weights, and every decoder receives the same latents from the original H3 encoder. The 20 test clips are 10 H3 generations and 10 live-action clips that no decoder has seen. Our harness reproduces LightVAE's PSNR exactly, and their speed claims hold (2.85× vs their claimed 2.96×).

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

{{< image src="img/fig_vae_pareto.svg" alt="Decoder FPS against PSNR for the original H3 decoder, LightVAE-Lite, TAEH3 and our two NVFP4 decoders." width="100%" title="Figure 2. Speed and quality of H3 VAE decoders on LightVAE's protocol (1344×768, one GB200). Up and to the right is better." >}}

- **36-layer vs LightVAE-Lite:** 1.9 dB better and essentially lossless (−0.10 dB against the original, where Lite loses 2.0), faster on their protocol (70 vs 51 FPS) and smaller. The one place it trails: in the best config, Lite's decoder is faster (184 vs 104 FPS), at 1.9 dB lower quality.
- **8-layer vs LightVAE-Lite:** the same PSNR (31.73 vs 31.74 dB) with better LPIPS (0.096 vs 0.109), 5.4× faster on their protocol, 2.3× faster in the best config, and 5× smaller.
- **On real footage and on generated video, at 480p, 720p and 768p,** the ranking is the same. Full tables are in the [appendix](#appendix-decoder-benchmark).
- **Their quality claim did not hold on our clips:** Lite lost 2.0 dB here, against the 1.25 dB in their README. The numbers in their paper on real datasets (about −3 dB on UCF-101 and DAVIS) agree with ours.

<!-- TODO(fig): zoomed crop grid on hard content (face, small text, foliage): original | ours 36L | ours 8L | LightVAE-Lite | TAEH3. Optional: a wipe slider original vs 8L. -->

## The whole stack in NVFP4

With the decoder done, every large component of FastH3 text-to-video now runs in NVFP4: the text encoder, the transformer and the VAE decoder. Here is what that buys against the same model kept in BF16.

{{< image src="img/fig_stack_size.svg" alt="Stacked bars of the weights FastH3 loads for text-to-video: 118.3 GiB for the same model in BF16 and 30.0 GiB for the FastH3 V2.1 NVFP4 stack." width="100%" title="Figure 3. Weights loaded for text-to-video, by component." >}}

<div class="v21-table">
<table>
<thead><tr><th>Component</th><th>Same model in BF16</th><th>FastH3 V2.1 stack</th><th>Smaller by</th></tr></thead>
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

### Faster end to end

Same protocol as our last post: a 5 s clip with audio at 832×480, end to end on a warm server from prompt to finished MP4, the median of two runs on each of two prompts.

| Machine | Last post (V2) | **V2.1 + 36-layer decoder** | **V2.1 + 8-layer decoder** |
|---|---:|---:|---:|
| DGX Spark | 141.4 s | **66.4 s** | **61.7 s** |
| 2× DGX Spark | 87.2 s | **50.1 s** | **44.7 s** |
| RTX 5090 | 14.8 s | TODO | TODO |
| RTX PRO 6000 | 13.5 s | TODO | TODO |

<!-- TODO: 1344x768 column (runs in progress), 2x Spark, RTX rows. -->

A 5 s clip on one DGX Spark now takes about a minute instead of almost two and a half: **2.3× faster**. The transformer gets faster from cuDNN FP4 GEMMs and from removing a padding copy before every FP4 activation, and the decoder goes from 43 s to 1.6 s. The NVFP4 decoders also lower peak GPU memory on the Spark by 5–6 GiB. Repeated runs with the same seed produce identical videos.

## Community LoRAs

The H3 community has already published many LoRAs, from camera moves to film looks. Most are published in ComfyUI or trainer layouts, and until now FastVideo could not read them. It now loads them with one line:

```python
from fastvideo import VideoGenerator
from fastvideo.api import ComponentConfig, GeneratorConfig, PipelineSelection

generator = VideoGenerator.from_config(
    GeneratorConfig(
        model_path="FastVideo/FastVideo-FastH3-8-Step-V2.1-NVFP4-Consumer",
        pipeline=PipelineSelection(
            components=ComponentConfig(lora_path="pablodawson/MiniMax-H3-360-Orbit-LoRA", lora_strength=0.8),
        ),
    ))
```
<!-- TODO: verify each example LoRA below loads and works on V2.1 NVFP4 before publishing; add the engine settings from the last post if needed. -->

- **Formats:** ComfyUI, fal, ai-toolkit and musubi-tuner layouts, with `lora_A/lora_B` or `lora_down/lora_up` factors and per-layer alpha.
- **On 4-bit models:** NVFP4 and FP8 weights stay packed and untouched; each LoRA runs as a small bf16 side branch next to its layer. Strength 0 reproduces the base model exactly, and strength can change at runtime.
- **No silent failures:** a LoRA that matches no layers, or too few, fails to load with a list of what did not match, instead of quietly giving you the base model.

Some popular MiniMax H3 LoRAs to try:

<!-- TODO(fig): base | LoRA gallery on V2.1 NVFP4 for 3-4 of these. Check each license. -->

| LoRA | What it does |
|---|---|
| [`fal/MiniMax-H3-Realism-People-LoRA`](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA) | Photorealistic people |
| [`pablodawson/MiniMax-H3-360-Orbit-LoRA`](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA) | 360° orbiting camera |
| [`Jojocodex/minimax-h3-Camera-Motion-lora`](https://huggingface.co/Jojocodex/minimax-h3-Camera-Motion-lora) | Camera motion control |
| [`KennethFal/vh5tape-vhs-lora-minimax-h3`](https://huggingface.co/KennethFal/vh5tape-vhs-lora-minimax-h3) | VHS tape look |
| [`lovis93/studio-1939-old-animation-lora-minimax-h3`](https://huggingface.co/lovis93/studio-1939-old-animation-lora-minimax-h3) | 1930s animation style |

Most community LoRAs are trained on base H3, which samples in 50 steps; FastH3 samples in 8. Style and motion LoRAs usually carry over, but the best strength can differ. Step-reduction ("turbo") LoRAs are not needed: FastH3 is already an 8-step model. See the community LoRA guide for details. <!-- TODO: link the docs page once FastVideo #1973 merges -->

## Tomorrow: FastH3 OmniRef

Tomorrow we release **FastH3 OmniRef** for consumer GPUs: video with audio generated from reference images, video and audio. It is built from today's pieces: the same QAD recipe, extended to 4-bit attention, and the NVFP4 decoders. Today's release is the foundation it runs on.

<!-- TODO: one teaser clip. -->

## Limitations

- **The 8-layer decoder trades quality for speed.** Use the 36-layer decoder when detail matters.
- **The NVFP4 decoders need a Blackwell GPU:** RTX 50 series, RTX PRO 6000, DGX Spark, or B200/GB200.
- **LoRAs on FastH3 Trim are not supported yet.** Trim removes transformer blocks, so LoRAs trained on full H3 do not line up with its layers.

## Get the models

| | |
|---|---|
| FastH3 V2.1, NVFP4 (RTX 5090, RTX PRO 6000, DGX Spark) | [`FastVideo-FastH3-8-Step-V2.1-NVFP4-Consumer`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2.1-NVFP4-Consumer) |
| NVFP4 VAE decoders: `quality-36L/` and `fast-8L/` | [`FastH3-NVFP4-VAE-Decoder`](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder) |
| Community LoRAs | Guide <!-- TODO: docs link after #1973 merges --> |

<!-- TODO: usage snippet selecting the decoder (env/config) once the public loader path is final. -->

## Appendix: decoder benchmark

<!-- TODO: by-content table (generated vs real), 480p and 720p tables, method notes. Source: ~/Desktop/VAE-benchmark.md -->

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

.fasth3-v21-article .v21-placeholder { margin: 1rem 0 1.6rem; padding: 3.2rem 1rem; text-align: center; border: 1.5px dashed #2a6fdb; border-radius: 10px; background: rgba(42, 111, 219, 0.06); color: var(--secondary); font-size: 0.9rem; aspect-ratio: 16 / 9; display: flex; align-items: center; justify-content: center; }
.fasth3-v21-article .v21-legend span.best { box-shadow: inset 0 0 0 2px #2a6fdb; border-radius: 4px; }
</style>
