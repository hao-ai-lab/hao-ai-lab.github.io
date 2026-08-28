+++
title = "FastVideo FastH3 V1: Open-Weight 4-Step Sparse Distilled Minimax H3 for 14x Speedup on NVIDIA Blackwell GPU"
date = 2026-08-27T00:00:00-07:00
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
draft = false
contentClass = "post-content-justified"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
+++

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fastvideo-fasth3" >}}

<!--
Publication checklist (remove before publishing):
- Replace every `TBD` in the performance and human-evaluation tables with a
  receipt-backed result from the exact release artifact.
- Run Base H3, every selected FastH3 variant, and H3 Max on the same 60 prompts,
  duration, resolution, and audio-enabled protocol. Do not mix fal-reported
  latency with our measured rows.
- Add matched-prompt video comparisons with working audio controls.
- Rerun the merged FastH3 LoRA path with regional compile on B200 and record the
  exact FastVideo revision with the final performance results.
- Confirm that revision includes the H3 DMD model adapter and training
  implementation, not only the release configs.
- Confirm all four model repositories are public.
- Confirm the combined FastH3 LoRA repository is public.
- Add the public training-config and synthetic-dataset links; verify their
  licenses and manifests.
- Confirm final author list, acknowledgements, citation URL, and MiniMax H3
  Community License review.
- Confirm the final publication timestamp.
-->

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata"
    aria-label="Garfield announces the open-source FastH3 four-step release"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="img/hero/000_garfield_fasth3_open_source.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
</div>

## **TL;DR:** 
FastVideo, in collaboration with [Nuva Lab](https://nuvalab.ai/)
and the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team, is open sourcing
FastH3 Preview v1 for text-to-video-and-audio (T2VA), post-trained on Minimax H3.

We took production readiness, quality, user experience and openness seriously.
We hope this joint effort will lead to a solid foundation for people who would
love to use it in real commerical workload beyond an academic experimentation.

### Speed:
- FastH3 can generate 15s 768p video in less than 13s with sub-realtime generation on 8xB200 GPUs.
- Up to 14x speedup on a single Nvidia Blackwell GPU

### Quality:
- We used 1k+ B200 training hours, paired with real world multi-shot, visual audio synced input distribution and output formats for best possible quality preservation.
- FastH3 natively supports variable resolution, aspect ratio, and duration. In a single checkpoint.

### Openness

- Start with the 4-step VSA / Data-Free checkpoint, our recommended FastH3
  Preview v1 release. We provide full weights and a pre-extracted LoRA, plus
  dense and synthetic-data ablations.
- Fully open source with training (coming soon!) and inference code recipe for your customization.


### What’s Next

- Follow us along for image ref (FL2VA) and full omni ref (Ref2VA) coming in the next a few weeks   
- Motion and more generation quality improvements
- Nvfp4 and GPU memory reduction. 
- Optimizations targeting local AI devices including RTX, DGX Sparks, and Apple MLX.
- New training runs using FastGen team's new [Parallel Decoding Distillation (PDD)](https://research.nvidia.com/labs/genair/pdd/) method!


<!-- TODO: Add an audible, matched-prompt hero comparison across the four
FastH3 releases, Base H3, and H3 Max. Do not autoplay because autoplay mutes
audio. -->

## Why open H3 matters

The strongest video systems were mostly closed until MiniMax released the
[H3-Base weights](https://huggingface.co/MiniMaxAI/MiniMax-H3). With downloadable
weights, the community can inspect H3, post-train it, replace kernels, and run it
on its own hardware.

This continues our work on
[FastWan sparse distillation](/blogs/fastvideo_post_training/) and
[FastWan-QAD](/blogs/fastwan-qad/). Both releases paired checkpoints with their
FastVideo training and inference stacks. FastH3 does the same for H3.


## FastH3 Preview v1

FastH3 distills H3's base transformer for text-to-video-and-audio (T2VA) and
reuses the H3-Base text encoder, video VAE, audio VAE, tokenizers, and schedulers.

FastH3 is not limited to the 1344×768 benchmark resolution. The checkpoints
were trained and validated at multiple aspect ratios, including square,
portrait, landscape, and ultrawide 768p video. FastVideo accepts custom heights
and widths in multiples of 32.

Our recommended Preview v1 checkpoint is
[4-step VSA / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree).
It trains from prompts without target videos and is the checkpoint to try
first.

We provide both full weights and a pre-extracted LoRA for the recommended
checkpoint.

{{< table title="Recommended FastH3 Preview v1 checkpoint. Uses four DiT calls." >}}
| Checkpoint | Pre-extracted LoRA | Training source | Attention | Training step |
|---|---|---|---|---:|
| [VSA / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) | [LoRA folder](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-datafree) | Prompts only, mixed shapes | VSA, 90% sparse, tile 64 | 1300 |
{{</ table >}}

The validation gallery and VSA performance rows below use this checkpoint.

The recommended checkpoint and its LoRA require FastVideo's Video Sparse
Attention (VSA-H3) backend and kernel to achieve the reported speed and quality.
Dense attention is not a drop-in substitute.

All Preview v1 checkpoints currently support T2VA only. H3 uses the base
transformer for both T2VA and first/last-frame-to-video-and-audio (FL2VA), but
these students were not trained with first/last-frame conditioning.
Reference-to-video-and-audio (Ref2VA) uses a separate reference transformer and
needs its own distilled checkpoint. FL2VA and Ref2VA checkpoints are in
development.

## Validation samples

Fifteen samples from the recommended VSA / Data-Free checkpoint, with generated
stereo audio and full prompts.

{{< fasth3-validation-gallery manifest="validation_v1_random15.json" >}}

## Performance

Each local test uses FastVideo's optimized inference at 1344×768 and 24 FPS
with audio. The 5s, 10s, and 15s shapes contain 124, 243, and 345 frames.
We report the median of three timed requests after one full
warmup. Model loading and compilation are excluded. End-to-end time includes
encoding, denoising, decoding, audio, muxing, and file output.

{{< table title="Warm end-to-end latency and same-hardware speedup for FastVideo on B200." >}}
| Model / runtime | Duration | 1× B200 E2E (s) | 4× B200 E2E (s) | 8× B200 E2E (s) | **Speedup over Base H3 (1× / 4×)** |
|---|---:|---:|---:|---:|---:|
| **Base H3 · Dense FA4** | 5s | 132.5 | 40.6 |  | 1.0× / 1.0× |
|  | 10s | 377.4 | 108.7 |  | 1.0× / 1.0× |
|  | 15s | 678.7 | 193.1 |  | 1.0× / 1.0× |
| **Preview v1 VSA / Data-Free · 90% sparse** | 5s | 16.2 | 6.1 | 8.02 | **8.16× / 6.65×** |
|  | 10s | 31.1 | 12.0 | 10.54 | **12.13× / 9.03×** |
|  | 15s | 47.2 | 15.5 | 13.90 | **14.38× / 12.48×** |
| **Preview v1 Dense / Data-Free · Dense FA4** | 5s | 18.3 | 6.8 |  | 7.24× / 5.97× |
|  | 10s | 50.2 | 15.0 |  | 7.52× / 7.25× |
|  | 15s | 91.3 | 25.6 |  | 7.43× / 7.54× |
{{</ table >}}

Speedup uses the unrounded timings for the same duration and GPU count; no 8×
speedup is claimed without a matched Base H3 run.

VSA / Data-Free is our recommended release and default performance path.
On B200, this path uses FastVideo's optimized tile-64 CUDA VSA kernel, regional
DiT compilation, H3 fusions, and compiled parallel video VAE. The 8× B200
measurements use H3 fusions and the compiled parallel VAE, but predate regional
compilation of the sparse DiT.

## How FastH3 works

Base H3 calls its 33B audio-video diffusion transformer 49 times. FastH3 lowers
that cost in two ways: four calls instead of 49, and less attention work inside
each call.

**Distribution Matching Distillation (DMD2) cuts the number of calls.**
[DMD2](https://arxiv.org/abs/2405.14867) trains the student with a frozen Base H3
teacher and a learned critic. The difference between their score estimates
supplies the training signal. For prompt-only runs, backward simulation exposes
the student to the few-step states it will see at inference. The synthetic-video
runs instead start from forward-noised Base-H3 video-and-audio latents.

**VSA makes each call cheaper.** The
[VSA paper](https://arxiv.org/abs/2505.13389) introduces trainable sparse
attention for video diffusion. In FastH3, the student keeps about 10% of eligible
video-to-video tiles (90% sparsity) with 64-token blocks. Text and audio remain
dense. The teacher and critic also use dense attention, giving the sparse
student a full-attention target. This extends the
[FastVideo sparse-distillation recipe](/blogs/fastvideo_post_training/) to H3.

### Other runs and ablations

We publish three comparison runs covering training source, training duration,
and dense attention. All use four DiT calls. Their pre-extracted LoRAs are
grouped in one
[FastH3 Preview LoRA repository](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA).

{{< table title="Research ablations." >}}
| Checkpoint | Purpose | Pre-extracted LoRA | Training source | Attention | Training step |
|---|---|---|---|---|---:|
| [VSA / Synthetic / Step 1300](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1300) | Training source at matched step | [LoRA folder](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-synthetic-step1300) | Synthetic Base-H3 videos | VSA, 90% sparse, tile 64 | 1300 |
| [VSA / Synthetic / Step 1900](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1900) | Longer synthetic training | [LoRA folder](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-synthetic-step1900) | Synthetic Base-H3 videos | VSA, 90% sparse, tile 64 | 1900 |
| [Dense / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree) | Dense-attention reference | [LoRA folder](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/dense-datafree) | Prompts only, mixed shapes | Dense FA4 | 1000 |
{{</ table >}}

Here, “data-free” means prompt-only training; the synthetic runs use videos
generated by Base H3. Dense / Data-Free is the full-attention reference.

The two synthetic checkpoints use the same VSA architecture and runtime as the
recommended checkpoint, so we do not repeat their latency in the performance
table. Both require the VSA-H3 backend and tile-64 kernel; Dense / Data-Free
uses FA4.

## Try FastH3

The setup below targets four NVIDIA B200 GPUs with CUDA 13. On first use, the
launcher downloads Base H3 and the VSA / Data-Free adapter. Each new process
loads them and compiles the fast inference path; warmup and measured generations
in one process reuse that work.

For guided setup, use FastVideo's
[agent-guided installation](https://github.com/hao-ai-lab/FastVideo#install-with-an-ai-coding-agent).
For a manual install, first install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate

UV_TORCH_BACKEND=cu130 uv pip install \
  --no-sources-package fastvideo-kernel \
  -e ".[fasth3]"

hf auth login
```

The `--no-sources-package` option installs the published
`fastvideo-kernel 0.3.4` wheel, which includes the B200 `sm100a` VSA kernel,
instead of compiling the kernel locally. Accept the MiniMax H3 Community
License before downloading Base H3. Never put a Hub token in a script or output
directory.

Use the pre-extracted
[VSA / Data-Free LoRA](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-datafree)
with the
[FastVideo launcher](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh).
The launcher downloads the adapter, loads it on top of `MiniMaxAI/MiniMax-H3`,
and enables the required VSA-H3 backend and tile-64 kernel.

Try VSA / Data-Free:

```bash
PROMPT='integrated_multimodal_description: A red fox runs through fresh snow at dawn. overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds.'

bash examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh \
  --prompt "$PROMPT" \
  --no-warmup \
  --repeats 1
```

Do not remove `--vsa` or run this LoRA through the dense path. The other three
checkpoints remain available in the ablation table above.

The launcher uses the shared
[`basic_fasth3_lora_preview.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_lora_preview.py)
runner. Its defaults enable FastVideo's H3 fusions, regional full-graph DiT
compile, FA4 for dense attention, compiled sequence-parallel VAE decoding,
replicated DiT weights, and pinned CPU offload. VSA variants also select 90%
sparsity, tile size 64, and the `sm100a` block-sparse kernel. Five scheduler
points mean exactly four DiT calls. Guidance stays at 1.0, matching training.

By default, the runner performs one compile warmup and then saves three measured
generations. Use `--no-warmup --repeats 1` for one clip, as above. Use
`--num-frames 243` for about 10 seconds or `--num-frames 345` for about 15
seconds.

Adapter strength is also available:

```bash
bash examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh \
  --prompt "$PROMPT" \
  --lora-strength 0.5 \
  --no-warmup \
  --repeats 1
```

Strength `1.0` applies the published rank-64 adapter at its trained scale.
Strength `0` removes its weight deltas but keeps the selected attention backend,
so a VSA run remains sparse. FastH3 adapters also contain exact parameter deltas
and, for VSA, compression-gate weights. FastVideo therefore applies the adapter
while building the pipeline and rejects unsafe runtime switching. Create a new
generator when changing variants or strength.

For a local adapter file or custom flags, call the shared Python runner directly.
For a JSONL prompt set, use
[`minimax_h3_lora_inference.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/lora/minimax_h3_lora_inference.py).

## What's coming next

Preview v1 is an early checkpoint family. FastVideo's next priorities are:

### 1. Improve motion and offer an eight-step option

We will finish the four-step low-noise A/B and 8-step model post-training.
We will then test stronger final-step training, more low-noise critic samples,
motion-sensitive losses, and learned timestep placement. The four-step model
targets minimum latency; eight steps may be a better quality setting. We will
release an 8-step checkpoint only after a matched comparison.

### 2. Add FL2VA and Ref2VA

T2VA is only one H3 workflow. FL2VA uses the base transformer but needs new
conditioning training. Ref2VA uses the separate reference transformer, so
FastVideo must distill it separately. We will evaluate mixed reference types,
long clips, and reference fidelity before releasing either workflow.

### 3. Apply Parallel Decoding Distillation methods to H3

We will continue to collaborate with NVIDIA's FastGen team to try their new PDD
algorithm and obtain the highest quality step-distilled H3 possible!

### 4. Make H3 easier to run for local AI and to extend

At four steps, encoding, VAE decode, audio, and file output become a larger
share of latency. FastVideo will keep improving VAE compilation and parallelism,
sparse compilation, portable kernels, cold start, and multi-GPU serving. We are 
also exploring FP8 and NVFP4 variants.

## Help us test more hardware

Our published latency numbers use NVIDIA B200 GPUs because that is our
controlled benchmark platform, not because the checkpoints require B200. The
weights are hardware-independent and can run anywhere with enough memory and a
compatible H3 runtime. The VSA checkpoints additionally need a compatible
FastVideo VSA kernel.

We are preparing optimized FastVideo recipes for NVIDIA RTX GPUs, NVIDIA DGX
Spark, and Apple Silicon through MLX. Stay tuned, and help us test hardware we
do not have locally.

Share results, unsupported hardware, regressions, and new ideas on
[GitHub](https://github.com/hao-ai-lab/FastVideo) or in the
[FastVideo Slack](https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ).
Start with the recommended
[VSA / Data-Free checkpoint](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
and its [optimized FastVideo launcher](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh).
Use the [full collection](https://huggingface.co/collections/FastVideo/fastvideo-fasth3)
for the ablations, and share your videos with us!

## Acknowledgements


We thank [Nuva Lab](https://nuvalab.ai/) for bringing production grounding to
FastH3 through its experience with real-world creative video-agent workloads.
Its production-aligned post-training insights help bridge open-source research
to practical data-assisted distillation for commercial video workflows, with
Omni Ref as the next focus.

We thank the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team for
the DMD2 framework and H3 reference experiment that helped us align the score
clock, modality shifts, and backward simulation. 

We also thank MiniMax for releasing H3-Base, and the [vLLM
project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/), and
[MBZUAI](https://mbzuai.ac.ae/) for their continued sponsorship and support of
FastVideo.

<!-- TODO: Add named contributors and affiliations after author approval. -->

## Citations

Method background: the [DMD2 paper](https://arxiv.org/abs/2405.14867), the
[VSA paper](https://arxiv.org/abs/2505.13389), our earlier
[FastWan sparse-distillation blog](/blogs/fastvideo_post_training/), and the
[FastWan-QAD blog](/blogs/fastwan-qad/). If you build on FastH3, please cite this
release, DMD2, VSA, and FastVideo.

```bibtex
@misc{fastvideo_fasth3_2026,
  title        = {FastH3 Preview v1: Four Open-Weight H3 Models in Four Steps},
  author       = {FastVideo Team},
  year         = {2026},
  howpublished = {\url{https://haoailab.com/blogs/fasth3-preview/}},
}

@misc{yin2024improved,
  title         = {Improved Distribution Matching Distillation for Fast Image Synthesis},
  author        = {Tianwei Yin and Michaël Gharbi and Taesung Park and Richard Zhang and Eli Shechtman and Fredo Durand and William T. Freeman},
  year          = {2024},
  eprint        = {2405.14867},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
}

@article{zhang2025vsa,
  title   = {VSA: Faster Video Diffusion with Trainable Sparse Attention},
  author  = {Peiyuan Zhang and Yongqi Chen and Haofeng Huang and Will Lin and Zhengzhong Liu and Ion Stoica and Eric Xing and Hao Zhang},
  journal = {arXiv preprint arXiv:2505.13389},
  year    = {2025},
}

@software{fastvideo2024,
  title  = {FastVideo: A Unified Framework for Accelerated Video Generation},
  author = {The FastVideo Team},
  url    = {https://github.com/hao-ai-lab/FastVideo},
  year   = {2024},
}
```
