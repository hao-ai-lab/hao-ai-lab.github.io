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


**TL;DR.** FastVideo, in collaboration with [Nuva Lab](https://nuvalab.ai/)
and the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team, is releasing
FastH3 Preview v1 for text-to-video-and-audio (T2VA). Our highlighted and
recommended checkpoint is
[VSA / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree).
It distills MiniMax H3 from 49 DiT calls to four, generates up to 768p video with
audio, and uses 90% sparse Video Sparse Attention. Three companion checkpoints
provide training-data and dense-attention ablations. In our qualitative checks,
fast motion remains weaker than Base H3. FL2VA was not a training target and can
produce low-motion results. Ref2VA is not supported yet.

This is an open release: full weights, pre-extracted LoRAs, and FastVideo
inference code are released in this blog. Training code, configs, and synthetic
Base-H3 data will be released at a later date. We want the community to
reproduce the recipe and improve it with us.

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
FastVideo takes the open source path by making the acceleration checkpoint and recipe a shared
open source asset, not just an API.

This continues our work on
[FastWan sparse distillation](/blogs/fastvideo_post_training/) and
[FastWan-QAD](/blogs/fastwan-qad/). Both releases paired checkpoints with their
FastVideo training and inference stacks. FastH3 brings the same open development
model to H3.

FastVideo code uses Apache 2.0. H3 and FastH3 weights use the custom
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
not a standard open-source model license. Please read it before using or
redistributing the weights.

## FastH3 Preview v1

FastH3 distills H3's base transformer for text-to-video-and-audio (T2VA) and
reuses the H3-Base text encoder, video VAE, audio VAE, tokenizers, and schedulers.

The highlighted release is
[VSA / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree).
It trains from prompts without target videos and is the version we recommend
trying first. The other three checkpoints are ablations for studying synthetic
training data, training duration, and dense attention.

Alongside the weights, FastVideo will later be releasing:

- Full training code and recipe for DMD2 and Video Sparse Attention (VSA) kernels.
- Prompts and synthetic Base-H3 videos.

We have already extracted the LoRA for every checkpoint. They are grouped in
one [FastH3 Preview LoRA repository](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA),
so inference does not require extracting an adapter from the full checkpoint.

{{< table title="The highlighted checkpoint and three ablations. Each uses four DiT calls." >}}
| Variant | Role | Pre-extracted LoRA | Training source | Attention | Training step |
|---|---|---|---|---|---:|
| [VSA / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) | **Highlighted** | [LoRA folder](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-datafree) | Prompts only, mixed shapes | VSA, 90% sparse, tile 64 | 1300 |
| [VSA / Synthetic / Step 1300](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1300) | Ablation | [LoRA folder](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-synthetic-step1300) | Synthetic Base-H3 videos | VSA, 90% sparse, tile 64 | 1300 |
| [VSA / Synthetic / Step 1900](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1900) | Ablation | [LoRA folder](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-synthetic-step1900) | Synthetic Base-H3 videos | VSA, 90% sparse, tile 64 | 1900 |
| [Dense / Data-Free](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree) | Ablation | [LoRA folder](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/dense-datafree) | Prompts only, mixed shapes | Dense FA4 | 1000 |
{{</ table >}}

“Data-free” (also known as backward simulation in DMD2) means training uses prompts but no target videos. The synthetic
variants use videos generated by Base H3. The dense variant provides a full-attention comparison.
[Nuva Lab](https://nuvalab.ai/) contributed the prompt suite and most of the
synthetic Base-H3 T2VA training corpus.

The three VSA checkpoints and their LoRAs require FastVideo's Video Sparse
Attention (VSA-H3) backend and kernel. This is part of the model, not an
optional speed setting. Dense attention is not a drop-in substitute; use the
Dense / Data-Free variant for the dense path.

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
| Preview v1 VSA / Data-Free (highlighted) | 4 | VSA, 90% sparse, tile 64 | TBD | TBD | TBD | N/A | TBD |
| Preview v1 Dense / Data-Free (ablation) | 4 | Dense FA4 | TBD | TBD | TBD | N/A | TBD |
{{</ table >}}

VSA / Data-Free is the highlighted performance path. The two synthetic VSA
ablations use the same runtime path and therefore have the same latency.
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


## Try FastH3

The highlighted example targets four NVIDIA B200 GPUs with CUDA 13. The first
run downloads and loads Base H3 plus the VSA / Data-Free adapter, then compiles
the fast inference path. Warmup and measured generations in the same runner
process reuse that work; a new process loads and compiles again.

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

The recommended path is the highlighted
[VSA / Data-Free LoRA](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-datafree),
which is already extracted and ready to load. Its
[FastVideo launcher](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh)
downloads the adapter, loads it on top of `MiniMaxAI/MiniMax-H3`, and enables
the required VSA-H3 backend and tile-64 kernel.

Try VSA / Data-Free:

```bash
PROMPT='integrated_multimodal_description: A red fox runs through fresh snow at dawn. overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds.'

bash examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh \
  --prompt "$PROMPT" \
  --no-warmup \
  --repeats 1
```

Do not remove `--vsa` or run this LoRA through the dense path. The other three
checkpoints remain available as ablations in the release table above.

The launcher uses the shared
[`basic_fasth3_lora_preview.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_lora_preview.py)
runner. Its defaults enable FastVideo's H3 fusions, regional full-graph DiT
compile, FA4, compiled sequence-parallel VAE decoding, replicated DiT weights,
and pinned CPU offload. VSA variants also select 90% sparsity, tile size 64,
and the `sm100a` block-sparse kernel. Five scheduler points mean exactly four
DiT calls. Guidance stays at 1.0, matching training.

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
Start with the highlighted
[VSA / Data-Free checkpoint](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
and its [optimized FastVideo launcher](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh).
Use the [full collection](https://huggingface.co/collections/FastVideo/fastvideo-fasth3)
for the ablations, and show us where FastH3 works—and where it does not.

## Acknowledgements

We thank [Nuva Lab](https://nuvalab.ai/) for the prompt suite and synthetic
data corpus, and the [NVIDIA FastGen](https://github.com/NVlabs/FastGen)
team for the DMD2 framework and H3 reference experiment that helped us align
the score clock, modality shifts, and backward simulation. We also thank
MiniMax for releasing H3-Base, and the [vLLM project](https://vllm.ai/), NVIDIA,
and [MBZUAI](https://mbzuai.ac.ae/) for their continued sponsorship and support
for FastVideo.

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
