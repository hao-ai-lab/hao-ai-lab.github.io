+++
title = "FastH3 Preview v1: Four Open-Weight H3 Models in Four Steps"
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
- Land the explicit H3 DMD inference-ladder fix on public FastVideo, rerun the
  default B200 route, and replace `FASTVIDEO_RELEASE_COMMIT` below with that
  exact revision.
- Confirm that revision includes the H3 DMD model adapter and training
  implementation, not only the release configs.
- Confirm all four model repositories are public.
- Add the public training-config and synthetic-dataset links; verify their
  licenses and manifests.
- Confirm final author list, acknowledgements, citation URL, and MiniMax H3
  Community License review.
- Confirm the final publication timestamp.
-->


**TL;DR.** FastVideo, in collaboration with [Nuva Lab](https://nuvalab.ai/)
and the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team, is releasing
four FastH3 Preview v1 checkpoints for text-to-video-and-audio (T2VA). They
distill MiniMax H3 from 49 DiT calls to four and generate 768p video with audio.
The family compares prompt-only with synthetic-video training and sparse with
dense attention. In our qualitative checks, fast motion remain
weaker than Base H3. FL2VA was not a training target and can produce low-motion
results. Ref2VA is not supported yet.

This is an open release: weights, FastVideo inference code are released in this blog. Training code, configs, synthetic Base-H3 data will be released at a later date. We want the community to reproduce the recipe and improve it with us.

<!-- TODO: Add an audible, matched-prompt hero comparison across the four
FastH3 releases, Base H3, and H3 Max. Do not autoplay because autoplay mutes
audio. -->

## Why open H3 matters

The strongest video systems were mostly closed until MiniMax released the
[H3-Base weights](https://huggingface.co/MiniMaxAI/MiniMax-H3). With downloadable
weights, the community can inspect H3, post-train it, replace kernels, and run it
on its own hardware.

That ecosystem is already growing. On August 26, fal announced
[H3 Max](https://fal.ai/learn/devs/introducing-h3-max-by-fal), a hosted
post-trained H3 model. fal reports stronger quality and a latency below three
seconds for a five-second 768p request. fal has not announced downloadable H3 Max weights or training code.
FastH3 takes a complementary path by making the acceleration recipe a shared
FastVideo asset, not only an endpoint.

This continues our work on
[FastWan sparse distillation](/blogs/fastvideo_post_training/) and
[FastWan-QAD](/blogs/fastwan-qad/). Both releases paired checkpoints with their
FastVideo training and inference stacks. FastH3 brings the same open development
model to H3.

FastVideo code uses Apache 2.0. H3 and FastH3 weights use the custom
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
not a standard open-source model license. Please read it before using or
redistributing the weights.

## Four preview checkpoints

FastH3 distills H3's base transformer for text-to-video-and-audio (T2VA) and
reuses the H3-Base text encoder, video VAE, audio VAE, tokenizers, and schedulers.

Alongside the weights, FastVideo will later be releasing:

- Full training code and recipe for DMD2 and Video Sparse Attention (VSA) kernels.
- Prompts and synthetic Base-H3 videos.

{{< table title="The four checkpoints in FastH3 Preview v1. Each uses four DiT calls." >}}
| Variant | Training source | Attention | Training step |
|---|---|---|---:|
| [VSA / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) | Prompts only, mixed shapes | VSA, 90% sparse, tile 64 | 1300 |
| [VSA / Synthetic / Step 1300](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1300) | Synthetic Base-H3 videos | VSA, 90% sparse, tile 64 | 1300 |
| [VSA / Synthetic / Step 1900](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1900) | Synthetic Base-H3 videos | VSA, 90% sparse, tile 64 | 1900 |
| [Dense / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree) | Prompts only, mixed shapes | Dense FA4 | 1000 |
{{</ table >}}

“Data-free” (also known as backward simulation in DMD2) means training uses prompts but no target videos. The synthetic
variants use videos generated by Base H3. The dense variant provides a full-attention comparison.
[Nuva Lab](https://nuvalab.ai/) contributed the prompt suite and most of the
synthetic Base-H3 T2VA training corpus.

All four checkpoints support T2VA only. H3 uses the base transformer for both
T2VA and first/last-frame-to-video-and-audio (FL2VA), but these students were not
trained with first/last-frame conditioning. Reference-to-video-and-audio
(Ref2VA) uses a separate reference transformer and needs its own distilled
checkpoint.

## Performance
Each local test uses FastVideo's optimized inference at a resolution of 1344x768x124 @ 24 FPS (5s) with audio. 
We report the median of three timed requests after one full
warmup. Model loading and compilation are excluded. End-to-end time includes
encoding, denoising, decoding, audio, muxing, and file output; denoise time
measures only the DiT loop.

{{< table title="Performance table for final publication. TBD cells are intentionally not estimates." >}}
| Model / variant | DiT forwards | Attention | 1× B200 E2E | 4× B200 E2E | 4× B200 denoise | Hosted E2E | Peak GiB / GPU |
|---|---:|---|---:|---:|---:|---:|---:|
| Base H3, FastVideo | 49 | Dense FA4 | TBD | TBD | TBD | N/A | TBD |
| Preview v1 VSA family | 4 | VSA, 90% sparse, tile 64 | TBD | TBD | TBD | N/A | TBD |
| Preview v1 Dense / Data-Free | 4 | Dense FA4 | TBD | TBD | TBD | N/A | TBD |
| H3 Max, fal API | Not disclosed | Not disclosed | N/A | N/A | N/A | Under 3 s, fal-reported; observed TBD | Not disclosed |
{{</ table >}}

The three VSA checkpoints share one latency row because their runtime paths are
identical.
On B200, the default path uses FastVideo's tile-64 CUDA VSA kernel, regional DiT
compilation, H3 fusions, and compiled parallel video VAE. We will evaluate the
three checkpoints' quality separately. H3 Max is a hosted service on undisclosed
hardware, so fal's number is not an apples-to-apples local result.

## How FastH3 works

Base H3 calls its 33B audio-video diffusion transformer 49 times. FastH3 lowers
that cost in two ways: four calls instead of 49, and less attention work inside
each call.

** Distribution Matching Distillation (DMD2) cuts the number of calls.**
[DMD2](https://arxiv.org/abs/2405.14867) trains a student against a frozen Base
H3 teacher and a learned critic. The student learns from the gap between their
scores. For prompt-only runs, backward simulation exposes the student to the
few-step states it will see at inference. The synthetic-video runs instead start
from forward-noised Base-H3 video-and-audio latents.

**VSA makes each call cheaper.** The
[VSA paper](https://arxiv.org/abs/2505.13389) introduces trainable sparse
attention for video diffusion. In FastH3, the student keeps about 10% of eligible
video-to-video tiles (90% sparsity) with 64-token blocks. Text and audio remain
dense. The teacher and critic also uses dense attention, giving the sparse student a
full-attention target. This extends the
[FastVideo sparse-distillation recipe](/blogs/fastvideo_post_training/) to H3.

## Where four steps still fall short

### Fast motion


### FL2VA



## Try FastH3

The optimized FastVideo path targets NVIDIA B200 GPUs with CUDA 13. Startup
loads the model, and the first generation compiles it. Use a warm request for
steady-state timing.

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
git checkout FASTVIDEO_RELEASE_COMMIT
uv venv --python 3.12 --seed
source .venv/bin/activate
UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"
```

```bash
PROMPT='integrated_multimodal_description: A red fox runs through fresh snow at dawn. overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds.'
MODEL_PATH='FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1900'
export FASTVIDEO_DMD_DENOISING_STEPS=999,749,500,250

python examples/inference/basic/basic_fasth3.py \
  --model-path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --output outputs/fasth3_fox \
  --num-gpus 4 \
  --repeats 1
```

The example uses five scheduler points, which produce exactly four DiT calls.
Keep the trained schedule, 90% VSA sparsity, tile size 64, and guidance 1.0;
changing them can hurt quality. This command uses VSA / Synthetic / Step 1900.
The other VSA checkpoints use the same command with a different model path. See
the Dense / Data-Free model card for its command.

## What comes next

Preview v1 is an early checkpoint family, not a claim that four steps have
solved H3. FastVideo's next priorities are:

### 1. Publish a evaluation between our checkpoints and H3 Max

We will compare every checkpoint with Base H3 and H3 Max on 60 held-out prompts.

### 2. Improve motion and offer an eight-step option

We will finish the four-step low-noise A/B and the 8-step control. We will then
test stronger final-step training, more low-noise critic samples,
motion-sensitive losses, and learned timestep placement. The four-step model
targets minimum latency; eight steps may be a better quality setting. We will
release an 8-step checkpoint only after a matched comparison.

### 3. Add FL2VA and Ref2VA

T2VA is only one H3 workflow. FL2VA uses the base transformer but needs new
conditioning training. Ref2VA uses the separate reference transformer, so
FastVideo must distill it separately. We will evaluate mixed reference types,
long clips, and reference fidelity before releasing either workflow.


### 4. Make H3 easier to run and extend

At four calls, encoding, VAE decode, audio, and file output become a larger
share of latency. FastVideo will keep improving VAE compilation and parallelism,
sparse compilation, portable kernels, cold start, and multi-GPU serving. We will
also study FP8 and NVFP4. Stable releases will include
the checkpoint, config, sampling contract, code revision, data provenance, and
negative ablations.

## Help us test more hardware

Our published numbers use NVIDIA B200 GPUs. We welcome results from other
single- and multi-GPU systems that can load FastH3. Run the default example and
report the GPU model and count, driver and CUDA versions, FastVideo commit,
checkpoint, output shape, warmup, end-to-end and denoising time, peak memory,
and a sample clip.

Share results, unsupported hardware, regressions, and new ideas on
[GitHub](https://github.com/hao-ai-lab/FastVideo) or in the
[FastVideo Slack](https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ).
Explore the
[four-checkpoint collection](https://huggingface.co/collections/FastVideo/fastvideo-fasth3),
run the [FastVideo example](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3.py),
and show us where FastH3 works—and where it does not.

## Acknowledgements

We thank [Nuva Lab](https://nuvalab.ai/) for the prompt suite and synthetic
Base-H3 training corpus, and the [NVIDIA FastGen](https://github.com/NVlabs/FastGen)
team for the DMD2 framework and H3 reference experiment that helped us align
the score clock, modality shifts, and backward simulation. We also thank
MiniMax for releasing H3-Base, fal for adding another H3 direction to the
ecosystem, the DMD2 and VSA authors, the FlashAttention and CUTLASS teams, and
every FastVideo contributor who built and tested the model, kernels, training
path, evaluation tools, and serving stack.

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
