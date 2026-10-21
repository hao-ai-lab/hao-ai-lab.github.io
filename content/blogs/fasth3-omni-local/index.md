+++
title = "FastH3 Omni Goes Local: DGX Spark, RTX and Apple Silicon"
description = "The same FastH3 Omni request, with matched prompts, references and seeds, run end to end on DGX Spark, RTX GPUs and an Apple M5 Max."
date = 2026-10-21T00:00:00Z
authors = ["Aryan Kumar", "Will Lin", "Hao Zhang"]
author = "Aryan Kumar, Will Lin, Hao Zhang"
ShowReadingTime = true
ShowToc = false
draft = false
contentClass = "post-content-justified omniref-article"
omnirefInteractive = true
url = "/blogs/fasth3-omni-local/"
[cover]
    image = "img/cover.jpg"
    alt = "A stack of NVIDIA DGX Spark systems beside the FastH3 Omni wordmark"
    relative = true
    hiddenInSingle = true
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
+++

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fasth3-omni" >}}

Yesterday we released [**FastH3 Omni**](/blogs/fastvideo-omniref-update/): give it a prompt plus images, video clips and a voice, and it returns a finished video with synchronized sound. On eight GB200s, a 5-second 768p clip from one reference photo takes **5.7 seconds**.

Today it comes to your desk. We ran the **exact same requests**, with identical prompts, references and seeds, end to end on one and two **NVIDIA DGX Sparks**, an **RTX 5090**, an **RTX 3090** and an **Apple M5 Max** (64 GB).

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata" poster="media/posters/spark1x-36l-image-768p.jpg"
    aria-label="FastH3 Omni 768p output generated on one DGX Spark"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/outputs/spark1x-36l-image-768p.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
  <p style="margin: 0.5rem 0 0; text-align: center; font-size: 0.9em; opacity: 0.75;">One reference photo and a prompt in, 1344×768 video with dialogue and sound out, on a single DGX Spark. Turn the audio on.</p>
</div>

## TL;DR

- **DGX Spark runs every reference mix.** A 5-second 480p clip from a photo takes **about 3 minutes on one Spark and about 2 minutes on two**, and the 128 GB of unified memory handles everything we tested, from one photo to three images or a long reference video, at 480p and 768p.
- **Across local hardware.** The same 1-photo request takes **62 s on an RTX 5090**, about **17 minutes on an RTX 3090** (plus model loading), and **19.5 minutes on an Apple M5 Max** (single run) with FastVideo's new MLX runtime.
- **4.6× faster in seven weeks.** Text-to-video on the Spark went from 243 s at our first local release to 52.6 s today. FastH3 Omni runs on the same stack, so those gains carry over.
- **References are the real cost.** A reference photo makes a request 3.3–4.4× slower than text-to-video on the same machine. We show why, and how to trade some of it back.

## Same request, every machine

Choose a reference mix and press **Play together**. Every panel uses the same prompt, references and seed; outputs differ across hardware, so compare quality rather than pixels.

{{< omni-local-gallery >}}

## End-to-end latency

Every bar is the full request, from prompt and references to a finished MP4 with audio.

{{< omni-local-latency >}}

**DGX Spark, every reference mix** (three-seed means, 36-layer decoder, 5-second clips):

<div class="omni-table-wrap">

| References | 2× Spark 480p | 1× Spark 480p | 2× Spark 768p | 1× Spark 768p |
|---|---:|---:|---:|---:|
| 1 image | **130.1 s** | 192.0 s | **217.8 s** | 361.3 s |
| Image + voice | **128.6 s** | 198.3 s | **220.9 s** | 378.5 s |
| 3 images | **390.5 s** | 640.3 s | **594.8 s** | 957.9 s |
| 1 short clip | **116.2 s** | 194.8 s | **200.6 s** | 332.6 s |
| 1 long clip | **261.4 s** | 504.6 s | **366.5 s** | 656.9 s |

</div>

The two-Spark runs used the earlier FastH3 Omni QAD checkpoint, so read the 1× vs 2× ratio as indicative.

**Across local hardware** (480p, 36-layer decoder):

<div class="omni-table-wrap">

| References | 1× DGX Spark | RTX 5090 | RTX 3090 | Apple M5 Max |
|---|---:|---:|---:|---:|
| 1 image | 192.0 s | 61.9 s | 16.9 min | 19.5 min |
| Image + voice | 198.3 s | 63.6 s | 18.0 min | 20.1 min |
| 1 short clip | 194.8 s | 57.2 s | 14.5 min | 14.4 min |

</div>

The M5 Max numbers are single runs; the RTX 3090 numbers exclude its ~7-minute model load per request.

- **Spark runs every mix in minutes**, at both resolutions and with either decoder. The 3-image and long-clip requests hit the memory limits of today's 24–32 GB GPU profiles; the M5 Max runs them, but in 44–59 minutes.
- **Two Sparks are 1.5–1.9× faster than one**, and the long-clip request scales best (1.9× at 480p).
- **The decoder no longer matters much.** The 8-layer decoder saves only a few percent on Spark, because the [NVFP4 decoders](/blogs/fasth3-v21/#nvfp4-vae-decoders) take seconds and the transformer dominates.
- **Every class of local machine works.** The RTX 5090 is the quickest single GPU at about a minute, and the RTX 3090 and the Apple M5 Max land within 15–20 minutes.

## From text-to-video to references

Text-to-video on local hardware has become fast. On the Spark, where FastH3 first went local, the same 5-second 480p clip is **4.6× faster on one Spark and 6.8× faster on two** than seven weeks ago (8-layer decoder; the first release was measured cold), on top of the [8.4× we measured over the public vLLM-Omni recipe](/blogs/fasth3-local/) at launch:

{{< spark-history >}}

The gains came from eight-step distillation in [NVFP4](/blogs/fasth3-rtx/), NVIDIA's 4-bit floating-point format, then [quantization-aware distillation with 4-bit attention and NVFP4 decoders](/blogs/fasth3-v21/), which cut decoding from 43 s to under 2 s.

Adding a reference photo to the same request takes **3.3–4.4× more time** on the same machine:

{{< t2v-vs-omni >}}

The reason is tokens. A 5-second 480p clip is about **14,400 video tokens**. FastH3 Omni reads each reference image on the **2048-pixel canvas** it was trained with, which adds about **7,100 VAE tokens** and another **7,100 vision tokens** in the prompt. One photo doubles the sequence to about 29,000 tokens, and three photos roughly quadruple it to nearly 60,000. This is why the 3-image request is the slowest everywhere.

The reference canvas is therefore the biggest lever you control. `--reference-max-pixels match` shrinks each reference to the clip's own size: on the M5 Max, the 1-photo request drops from **19.5 to 4.7 minutes (4.2×)**. It is opt-in because it departs from training and changes framing and action:

<div class="omni-output-grid" style="grid-template-columns: repeat(2, minmax(0,1fr));">
  <figure class="omni-output"><figcaption><strong>Trained reference canvas</strong><span>M5 Max · 19.5 min</span></figcaption>
    <video controls playsinline preload="none" poster="media/posters/m5max-36l-image-480p.jpg" aria-label="M5 Max output with the trained reference canvas"><source src="media/outputs/m5max-36l-image-480p.mp4" type="video/mp4"></video></figure>
  <figure class="omni-output"><figcaption><strong><code>--reference-max-pixels match</code></strong><span>M5 Max · 4.7 min</span></figcaption>
    <video controls playsinline preload="none" poster="media/posters/m5max-36l-image-480p-match.jpg" aria-label="M5 Max output with the reference canvas matched to the clip"><source src="media/outputs/m5max-36l-image-480p-match.mp4" type="video/mp4"></video></figure>
</div>

## How it runs on each machine

FastH3 Omni pairs a ~35B-parameter transformer with a ~32B-parameter Qwen3-VL text encoder, well over 100 GB in BF16. Each machine fits it differently.

### DGX Spark

Everything stays resident in the Spark's 128 GB of unified memory: the 4-bit NVFP4 transformer, text encoder and decoder, with no offloading between stages. Attention uses the 4-bit sparse attention on 128-token tiles the model was trained with, while the text, reference and audio rows run densely at 8-bit so the conditioning stays out of FP4 rounding.

Two directly connected Sparks split the token sequence: each holds the full model and processes half the tokens. Conditioning does not split as cleanly as denoising, one reason the speedup stays below 2×.

Get started with the [Spark guide](https://haoailab.com/FastVideo/getting_started/installation/spark/) or the [Spark pair guide](https://haoailab.com/FastVideo/getting_started/installation/spark_pair/).

### RTX GPUs

The **RTX 5090** keeps the NVFP4 transformer on its 32 GB. A precomputed 155 MB table replaces 26 GB of timestep-projection weights, shrinking the transformer from 44.5 GB to 18.5 GB, and the text encoder loads only while the prompt is encoded.

The **RTX 3090** has no FP4 or FP8 hardware, so it runs FP8 weights with BF16 math through the [24 GB profile](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_rtx4090.yaml), streaming the transformer a few blocks at a time.

### Apple Silicon

FastH3 Omni runs natively on Macs through FastVideo's [MLX runtime](https://github.com/hao-ai-lab/FastVideo/pull/2000):

- **INT6 transformer (17 GB).** Timestep projections are precomputed, as on the RTX 5090, and INT6 fits the rest in memory; Apple GPUs have no FP4 units.
- **No base VAE needed.** The [LightVAE](https://huggingface.co/lightx2v/LightVAE) Lite encoder (406 MB) reads the references, and the same NVFP4 decoders as on Blackwell write the frames.
- **The NVFP4 text encoder runs unchanged**, including its vision tower.
- **Trained sparsity on Metal**, through a new Metal kernel for the model's 128-token sparse-attention tiles.

Peak memory on the M5 Max stays between 27 and 31 GB across all five workloads.

## Quantization-aware distillation

Rounding a distilled eight-step model to 4 bits after training costs quality. **Quantization-aware distillation (QAD)** instead trains the NVFP4 student against the full-precision teacher with FP4 numerics active from the first step, so the weights learn to absorb the rounding. The sparse-attention gates and norms stay frozen, because retraining them changes which tiles are selected.

FastH3 Omni's local weights, [`FastVideo/FastH3-Omni-NVFP4-QAD`](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD), extend the [FastH3-QAD-v2 recipe](/blogs/fasth3-v21/) to 4-bit attention. The shared training recipe is available in [FastVideo #1989](https://github.com/hao-ai-lab/FastVideo/pull/1989).

## Try it

| Machine | Weights and recipe |
| --- | --- |
| DGX Spark | [NVFP4 QAD weights](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD) · [`spark` recipe](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_spark.yaml) |
| 2× DGX Spark | [NVFP4 QAD weights](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD) · [`spark-pair` recipe](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_spark_pair.yaml) |
| RTX 5090 | [NVFP4 QAD weights](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD) · [`rtx-5090` recipe](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_rtx5090.yaml) |
| 24 GB RTX GPUs | [FP8 weights](https://huggingface.co/FastVideo/FastH3-Omni-FP8-Comfy) · [`rtx-4090` recipe](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_rtx4090.yaml) |
| Apple Silicon | [BF16 release](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1), converted locally · [Apple Silicon recipe](https://haoailab.com/FastVideo/cookbook/minimax-h3/models/fasth3-omniref-mlx/) |

Start with the [FastVideo cookbook](https://haoailab.com/FastVideo/cookbook/) and the [FastH3 Omni recipe builder](https://haoailab.com/FastVideo/cookbook/minimax-h3/fasth3-omniref/).

Prefer a visual workflow? The local community has been making [great things](#from-the-community) with our [ComfyUI release](https://huggingface.co/FastVideo/FastH3-Omni-Comfy), available in [NVFP4 QAD](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD-Comfy) and [FP8](https://huggingface.co/FastVideo/FastH3-Omni-FP8-Comfy).

## From the community

Within a day of release, people were running FastH3 Omni on their own GPUs:

{{< community-embeds >}}

<p class="cwall-note">Community-reported, each on its own setup and settings. Scroll for more; click a post to open it.</p>

## What's next

We will keep pushing what FastH3 Omni can do on Apple Silicon, keep improving quality and speed across every machine, and release smaller models that fit a wider range of consumer hardware.

## Limitations

- NVFP4, FP8 and INT6 are different numerical variants, so outputs differ from the BF16 release and from each other for the same seed.
- The M5 Max numbers are single runs, the RTX 3090 numbers exclude model loading, and the two-Spark runs used the earlier QAD checkpoint.
- `--reference-max-pixels match` trades reference detail for speed.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). We thank the MiniMax team for releasing its weights and code.

We thank the NVIDIA Enterprise Products team (Pengcheng Li and Cliff Woolley) for the Video Sparse Attention kernel used by FastH3. We thank the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team (Julius Berner, Chao Liu and Arash Vahdat) for the DMD2 framework and H3 reference work, and for introducing [Parallel Decoding Distillation](https://research.nvidia.com/labs/genair/pdd/).

We thank [Nuva Lab](https://nuvalab.ai/) for bringing production-grounded creative video workflows to FastH3, and the upstream serving, kernel and open-source communities whose implementations make these experiments possible. We also acknowledge [Reactor](https://www.reactor.inc/) for supporting FastH3 Omni, and the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/) and [MBZUAI](https://mbzuai.ac.ae/) for their support of FastVideo.

Finally, thanks to community member [@posi_posi8](https://x.com/posi_posi8), who volunteered an Apple M5 Max to benchmark FastH3 Omni for this post.

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
