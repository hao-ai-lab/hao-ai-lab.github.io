+++
title = "FastVideo FastH3 Omni: Open Source Real-Time Minimax FastH3 Ref2VA on NVIDIA Blackwell, RTX Spark, and RTX GPUs."
description = "MiniMax H3 video and audio with up to approximately 12× lower end-to-end latency on eight GB200 GPUs, plus optimized NVFP4 QAD checkpoints for local AI."
date = 2026-10-19T00:00:00Z
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = false
ShowToc = false
draft = false
contentClass = "post-content-justified omniref-article"
omnirefInteractive = true
[cover]
    image = "img/cover.svg"
    alt = "FastH3 Omni: 5.6–11.9 times lower end-to-end latency across 27 matched workloads on eight GB200 GPUs, compared with Base’s lossless-policy serving recipe."
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


FastVideo is open-sourcing **FastH3 Omni**, a distilled MiniMax H3 model for reference-guided video and audio. Give it a prompt plus images, video clips and a voice reference to generate a finished video with synchronized sound.

[Model weights](https://huggingface.co/collections/FastVideo/fasth3-omni) · [UniServe deployment](#deploy-fasth3-omni-with-uniserve) · [Local AI with FastVideo](#local-ai-with-fastvideo)

ComfyUI support is available. Our local AI blog post and release are planned for tomorrow, **October 21**.

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata"
    aria-label="FastH3 Omni trilingual sample with background music"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/hero/trilingual-bgm-1080p.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
  <p style="margin: 0.5rem 0 0; text-align: center; font-size: 0.9em; opacity: 0.75;">This entire video was generated with FastH3 Omni on FastVideo.</p>
</div>

## TL;DR

- **Eight transformer evaluations**, down from Base’s 50, with image, video and audio references.
- **5.6–11.9× lower end-to-end latency** across 27 matched workloads on eight GB200 GPUs. This compares full serving recipes, using Base’s `lossless` numerical policy; [methodology](#benchmark-methodology).
- **Real-time Generation on FastVideo UniServe**: On particular FastH3 Omni tasks, FastVideo UniServe achieves real time generation speeds for 768p videos for 5s, 10s, and 15s clips.
- **Local AI.** Run on [DGX Spark or RTX 5090 with FastVideo](#local-ai-with-fastvideo), using the NVFP4 QAD weights and device recipes.
- **See and hear the outputs below.** Change the references or clip length, then play the matched examples together. Reference inputs are expanded; open the full prompt when you need it.
- **Partners.** Thanks to [Nuva Lab](https://nuvalab.ai/), [Reactor](https://www.reactor.inc/), NVIDIA FastGen Team, and NVIDIA Enterprise products for their collaboration and support.

## Matched reference inputs and outputs

Choose a reference mix and clip length to compare **Base, FastH3 Omni and FastH3 Omni with the 36-layer VAE**. All examples use seed 0 and matched inputs; these are qualitative comparisons, not evidence of output parity.

{{< omniref-gallery >}}

The [36-layer VAE](/blogs/fasth3-v21/#nvfp4-vae-decoders) is shown for quality comparison; its latency is not included in the timing data.

## Results at a glance

Across 27 matched workloads, FastH3 Omni delivers **5.6–11.9× lower end-to-end latency** than Base on eight GB200 GPUs. Timings include reference encoding, generation, decoding and the finished MP4, at a 1344×768 target canvas.

Both lanes use the original video VAE. Base uses SGLang’s `quality="lossless"` policy, not its stricter `exact` policy. This compares complete model and serving recipes; see the [benchmark methodology](#benchmark-methodology).

{{< omniref-results >}}

## PDD + DMD

FastH3 Omni combines **Parallel Decoding Distillation (PDD)** with **Distribution Matching Distillation (DMD)**. PDD predicts several denoising updates per transformer evaluation; DMD helps train the completed video and audio output. Reference conditioning is retained throughout generation. For the method and its illustrations, see the [PDD paper](https://arxiv.org/abs/2607.26004).

- **PDD: many updates per forward.** The student's output heads are widened from one velocity prediction to a 32-interval fine time grid. At inference those heads run as eight fused blocks of four, so one transformer evaluation advances four intervals and the full trajectory takes eight evaluations instead of Base's 50. A teacher-supervised head loss keeps every head's update consistent with the frozen Base model.
- **DMD: match the finished output.** On top of PDD, a trainable critic models the distribution of the student's samples while the frozen Base teacher supplies the target; the student is updated to close the gap between them on the completed video and audio, not just on each individual step. Critic and student updates alternate, with several critic updates per student update.
- **References stay clean conditioning.** Reference images, videos and audio are never noised: they condition the student, teacher and critic in every forward, and only the target video and audio enter the noising and the losses.
- **Sparse throughout.** The student trains and samples with Video Sparse Attention on 128-row tiles, so the sparsity used for serving is the sparsity it was trained with.

Huge thanks to the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team (Julius Berner, Chao Liu and Arash Vahdat), who introduced PDD and whose DMD2 framework and H3 reference work this model builds on. FastVideo's PDD and DMPDD trainers adapt the PDD objectives from the [FastGen repository](https://github.com/NVlabs/FastGen) to FastVideo's MiniMax H3 model, sequence-parallel and checkpoint interfaces.

| Training | Amount |
| --- | --- |
| GPUs | 64 NVIDIA B300 |
| Training steps | 3,500 |
| Estimated compute | Approximately 7,800 GPU-hours |

The training code is in FastVideo: see the [FastH3 Omni DMPDD training guide](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/training/omniref_dmpdd.md).

## Local AI with FastVideo

FastH3 Omni also runs locally on **NVIDIA DGX Spark and RTX 5090 through FastVideo**. Start with the [FastVideo cookbook](https://haoailab.com/FastVideo/cookbook/) and [FastH3 Omni recipe builder](https://haoailab.com/FastVideo/cookbook/minimax-h3/fasth3-omniref/), then follow the [local inference guide](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/inference/fasth3_omniref.md) for model setup and reference inputs.

| Device | Weights and inference recipe |
| --- | --- |
| DGX Spark (GB10, 128 GB unified memory) | [NVFP4 QAD weights](https://huggingface.co/FastVideo/FastH3-OmniRef-NVFP4-QAD) · [`spark` recipe](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_spark.yaml) |
| RTX 5090 (32 GB) | [NVFP4 QAD weights](https://huggingface.co/FastVideo/FastH3-OmniRef-NVFP4-QAD) · [`rtx-5090` recipe](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_rtx5090.yaml) |

**QAD means quantization-aware distillation:** these weights are trained for NVFP4 execution on Blackwell. Install FastVideo with the [DGX Spark setup guide](https://haoailab.com/FastVideo/getting_started/installation/spark/) or [NVIDIA GPU setup guide](https://haoailab.com/FastVideo/getting_started/installation/gpu/). Both device profiles default to 480p; Spark keeps the model components resident, while the RTX 5090 profile offloads the text encoder and requires a precomputed AdaLN table. The local inference guide covers preparing that table, composing the model directory and selecting the decoder. Keep the recipe's attention and precision settings.

For **FP8**, use the [FastH3 Omni FP8 weights](https://huggingface.co/FastVideo/FastH3-OmniRef-FP8) and the guide's [FP8 device profile](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/inference/fasth3_omniref.md#device-profiles); the [RTX 4090 recipe](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_rtx4090.yaml) shows layerwise offload for a device without FP4 support. Its text encoder must also match the hardware: the bundled NVFP4 encoder needs Blackwell, so other GPUs need compatible BF16 encoder components. NVFP4 QAD and FP8 are separate numerical variants; their outputs can differ from the BF16 release.

Optional [NVFP4 VAE decoders](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder) provide a **36-layer quality** tier and an **8-layer fast** tier. Follow the guide's `--decoder` setup; these local configurations have their own resolution, memory and precision choices, so the eight-GB200 timings above do not describe their speed.

## Deploy FastH3 Omni with UniServe

Install [UniServe](https://github.com/hao-ai-lab/uniserve) on every node:

```bash
git clone https://github.com/hao-ai-lab/uniserve.git && cd uniserve
apt-get update && apt-get install -y build-essential python3.12-dev pkg-config cmake \
  git curl ca-certificates nasm zlib1g-dev xz-utils
scripts/install-ffmpeg.sh
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export CUDA_HOME=/usr/local/cuda MAX_JOBS=2
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$CUDA_HOME/bin:$PATH"
uv sync --locked --python 3.12 --extra gpu
source .venv/bin/activate
```

Then pick your GPUs and run the steps from the UniServe checkout:

{{< uniserve_deploy model="omni" >}}

The [UniServe serving guide](https://github.com/hao-ai-lab/uniserve/blob/main/docs/minimax_h3/minimax_h3.md) covers every option and request field.

## Benchmark methodology

Nine reference compositions × three clip lengths (5, 10 and 15 seconds), measured on eight GB200 GPUs. Each profile uses one warmup and three measured seeds; the gallery consistently shows seed 0.

{{< omniref-details title="Read measurement settings and numerical policy" >}}

- **Hardware and scope.** Eight NVIDIA GB200 GPUs across two four-GPU trays. GB200 is our proxy for B200; B200 was not separately measured. These results describe the eight-GPU cohort.
- **Matched inputs.** Nine reference compositions × three requested lengths, all at 1344×768 and 24 FPS. The aligned outputs contain 124, 243 or 362 frames. Images use the released enlarged-reference mode with a 2048-pixel short edge. Reference clips retain their real soundtracks and standalone audio remains a real input.
- **Measurement.** One request at a time, one warmup with seed 42 for each profile, then measured seeds 0, 1 and 2. The tables report arithmetic mean client end-to-end time until the complete MP4 is available; the downloadable data also includes sample standard deviations. Model loading and server startup are excluded. No latency is inferred from missing cells or memory failures, and no timing is normalized to another workload.
- **Model and runtime.** Base is MiniMax H3 Ref2VA with 50 denoising evaluations; FastH3 Omni uses the 32-interval PDD student with eight fused evaluations and VSA. The Base lane uses Ulysses 4 × Ring 2 with the native FlashAttention transformer backend; the student uses eight-way Ulysses. Conditioning caches are disabled. Model weights, attention patterns and serving implementations differ, so this compares complete model/runtime recipes, not an isolated distillation or kernel ablation.
- **Base numerical policy.** These Base measurements use SGLang's `quality="lossless"` policy with original unquantized weights, dense attention and the model's mixed-precision rules. Its DiT normalization/modulation and VAE fusions can change intermediate rounding. They are not measurements of SGLang's stricter `quality="exact"` policy, and reference-exact or perceptual equivalence has not been established. The original model is not uniformly BF16: it retains FP32-sensitive DiT operations and the released VAE's mixed-precision execution. The displayed latencies have not been converted to an exact-policy estimate.
- **Reading the examples.** The timing statistics use three seeds; the gallery consistently uses seed 0. A shared seed does not imply numerical or perceptual output parity across different models or runtimes. The examples support direct inspection and are not a quantitative quality evaluation.

{{< /omniref-details >}}

## Get started

Download the model weights from the [FastH3 Omni collection](https://huggingface.co/collections/FastVideo/fasth3-omniref), then follow the [UniServe setup](#deploy-fasth3-omni-with-uniserve) to generate video and audio from your own references.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). We thank the MiniMax team for releasing its weights and code.

We thank the NVIDIA Enterprise Products team (Pengcheng Li and Cliff Woolley) for the Video Sparse Attention kernel used by FastH3. We thank the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team (Julius Berner, Chao Liu and Arash Vahdat) for the DMD2 framework and H3 reference work, and for introducing [Parallel Decoding Distillation](https://research.nvidia.com/labs/genair/pdd/).

We thank [Nuva Lab](https://nuvalab.ai/) for bringing production-grounded creative video workflows to FastH3, and the upstream serving, kernel and open-source communities whose implementations make these experiments possible. We also acknowledge [Reactor](https://www.reactor.inc/) for supporting FastH3 Omni, and the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/) and [MBZUAI](https://mbzuai.ac.ae/) for their support of FastVideo.
