+++
title = "FastVideo Open Sources FastH3 T2AV: V1 release of 4/8-step Sparse Distilled Minimax H3"
date = 2026-08-25T17:52:53-07:00
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
Publication checklist (remove before publishing):
- Re-run the 5-second table on the exact public Preview v0.2 artifact. The
  current checkpoint-comparable numbers below were measured with the
  architecture-identical Preview v0.1 artifact; this is disclosed in the text.
- Refresh `h3_perf.md` with the newer public-head acceptance numbers and publish
  stable benchmark receipts; its current branch copy still shows the older grid.
- Align the v0.2 model card's direct-API ladder example with current FastVideo.
  This draft intentionally routes users through `basic_fasth3.py` meanwhile.
- Refresh the live Artificial Analysis ranks and Elo values on publication day.
- Add matched-prompt base H3 versus Preview v0.2 video/audio embeds.
- Confirm final author list, acknowledgements, and citation metadata.
- Complete the MiniMax H3 Community License and territory review for this
  derivative checkpoint and the release announcement.
-->


**TL;DR.** FastVideo team is releasing first version of [FastVideo FastH3 T2AV](https://huggingface.co/FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2),
a set of T2AV checkpoints that reduces MiniMax H3's denoising loop from 49
transformer forwards to 4/8 while still producing 768p video with synchronized
stereo audio. On four NVIDIA GB200 GPUs, 4-step FastH3 can generate 5 second videos in 5.90 seconds.  Fast motion and fine audio detail still trail
Base H3, and this release distills only T2VA. Stay tuned for improved checkpoints and also Reference to Video checkpoints!

We are releasing the [checkpoint](https://huggingface.co/FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2),
the [FastVideo inference path](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3.py),
and the exact four-forward sampling contract. This is the first verion of a larger
effort: make frontier open-weight video practical to study, post-train, and
serve, for everyone.

<!-- TODO: Insert hero comparison: Base H3 49-forward vs FastH3 Preview v0.2
four-forward, identical prompt/seed, both with audible synchronized audio. -->

## Why H3 matters for open video

Until August 2026, the strongest video generators were overwhelmingly
closed-weight. Systems such as Seedance 2.0 became quality reference points,
but developers could not download the official weights, inspect the model,
change its inference path, distill it, or self-host it. Open-weight video
models existed, but their visual quality was of lower tier.

[MiniMax launched H3 on July 31 and released the H3-Base weights on August 3](https://www.minimax.io/news/minimax-h3-open-source).
That release sharply narrowed the quality gap. In the Artificial Analysis
"With Audio" arenas on August 25, 2026, H3 ranked first among open-weight models in
all three evaluated workflows and near the top of the overall leaderboards:

| Artificial Analysis arena | H3 rank | H3 Elo | Selected comparison systems |
|---|---:|---:|---|
| [Text-to-video](https://artificialanalysis.ai/video/leaderboard/text-to-video) | #3 overall, #1 open-weight | 1,226 ± 8 | Wan 3.0 1,241; Gemini Omni Flash 1,237; Seedance 2.0 1,220 |
| [Image-to-video](https://artificialanalysis.ai/video/leaderboard/image-to-video/) | #2 overall, #1 open-weight | 1,185 ± 9 | Seedance 2.0 1,191; Gemini Omni Flash 1,180 |
| [Video editing](https://artificialanalysis.ai/video/leaderboard/video-editing) | #2 overall, #1 open-weight | 1,128 ± 6 | Wan 3.0 1,190; Gemini Omni Flash 1,123; Seedance 2.0 1,037 |



This is not the complete hosted H3 product. The public Base weights generate
locally at 768p, while MiniMax's Context-IR prompt-processing stage and 2K
regeneration stage remain hosted. MiniMax's initial release also did not include
its production sparse-attention implementation; that is precisely the kind of
systems gap downloadable weights allow the broader community to address. The
weights also use the custom
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
not a standard open-source license. FastVideo's code is open source under
Apache 2.0; H3 and FastH3 are open-weight models whose use and redistribution
remain subject to MiniMax's terms, including territory restrictions. Please
review the license before using or redistributing the checkpoint.



## What we are releasing

FastH3 Preview is an early post-trained version of the H3 T2VA transformer.
The encoder, visual VAE, audio VAE, tokenizers, processors, and schedulers are
the unmodified H3-Base components; only the transformer weights are distilled.

| Property | FastH3 Preview v0.2 |
|---|---|
| Base model | [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| Workflow distilled | Text-to-audio-video only |
| Native training shape | 768x1344, 124 frames at 24 fps, synchronized stereo audio |
| Denoising contract | `[999, 749, 500, 250]`: exactly four DiT forwards |
| Attention contract | VSA prunes 90% of eligible video-to-video tiles using 64-token blocks |
| Training | Data-free DMD2, about 258,000 text prompts, 32 NVIDIA GB200 GPUs |
| Checkpoint | Step 2900 of a 4000-step run; bf16 transformer |
| Code | [FastVideo FastH3 example](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3.py) |
| Weights license | [MiniMax H3 Community License](https://huggingface.co/FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2/blob/main/LICENSE) |

There is one easy-to-miss sampling detail. "Four-step" means four transformer
forwards, but the public FastVideo CLI accepts sigma-grid points. The correct
command is therefore `--steps 5`: the scheduler traverses four non-zero points
and then lands at zero. Passing `--steps 4` produces only three transformer
forwards on a different grid. The student was not trained for that route.

The same rule applies to sparsity. VSA prunes 90% of eligible video-to-video
tiles using 64-token blocks; text, audio, and exempt-prefix interactions remain
dense. Setting 90% sparsity while leaving the tile size at 256 is a different,
untrained attention pattern.

## Performance

We benchmarked 768x1344, 124-frame T2VA with guidance 1.0 and synchronized
audio on NVIDIA GB200 GPUs. This is H3's nominal five-second shape: 124 frames
at 24 fps encode about 5.17 seconds. Every row uses one excluded warmup and
three timed requests with seed 1000; model load and compilation warmup are excluded.
"Denoise" measures only the DiT loop, while "end to end" also includes text
encoding, video/audio decoding, and saving the MP4.

| System and route | GPUs | DiT forwards | End to end | Denoise |
|---|---:|---:|---:|---:|
| vLLM-Omni Base H3, regional compile | 1 | 49 | 136.00 s | 127.74 s |
| FastVideo Base H3, optimized (report-only) | 1 | 49 | 134.38 s | 126.50 s |
| FastH3, same-student implementation-parity route | 1 | 4 | **17.95 s** | **10.70 s** |
| FastH3, H3 fusions + compiled/parallel VAE (eager sparse DiT; report-only) | 1 | 4 | **16.23 s** | **8.68 s** |
| vLLM-Omni Base H3, sequence parallel | 4 | 49 | 40.80 s | 37.22 s |
| FastVideo Base H3, sequence parallel (report-only) | 4 | 49 | 39.95 s | 36.90 s |
| FastH3, same-student implementation-parity route | 4 | 4 | **6.56 s** | **3.49 s** |
| FastH3, H3 fusions + compiled/parallel VAE (eager sparse DiT; report-only) | 4 | 4 | **6.11 s** | **2.97 s** |
| FastH3, current `all` profile (regional sparse DiT; report-only) | 4 | 4 | **5.90 s** | **2.25 s** |

This table establishes two separate results. First, FastVideo's optimized
49-forward Base H3 route matches vLLM-Omni at both one and four GPUs, so the
FastH3 comparison is not benefiting from an artificially slow base runtime.
Second, the four-forward student reduces the five-second end-to-end latency by
about 7.5x on one GPU and 6.1x on four GPUs on the implementation-parity route.

The optimized Base H3 rows are also performance routes, not parity routes:
packed-varlen FA4 and regional compilation change reduction order and can
produce materially different samples from fixed-FA4 eager execution.

The table's exact FastH3 artifact is public Preview v0.1 at step 1400. Preview
v0.2 changes transformer values but not tensor shapes or the serving graph. We
are nevertheless keeping the distinction explicit: the final publication
table should be refreshed on the exact v0.2 artifact rather than assuming
identical latency. Benchmark provenance is Base FastVideo `e0bb6a5`, eager
FastH3 `bbc8d35`, regional FastH3 `fe9ad53`, and vLLM-Omni `73b623f`. The run
guide below pins their later FastVideo descendant `6388db8`.

The fastest row adds regional full-graph compilation of the sparse DiT. On the
same Preview v0.1 architecture, it prepared all 50 trained VSA gates and 52
full-graph compiled regions with no graph break or Triton fallback.

That number is a speed ceiling, not a parity claim. H3 fusions and regional
compilation change floating-point operation order; the resulting videos are
repeat-deterministic but can diverge materially from the eager route. We call
this the `all` profile and report it separately. "Implementation parity" means
agreement with the eager FastH3 serving reference; it does not mean quality
parity between FastH3 and Base H3. For the reported eager sparse-DiT path,
disable the fusions and regional DiT compile with
`--profile strict --no-inference-torch-compile`.

<!-- TODO: Add a waterfall chart with the measured contribution from four-step
distillation, VSA, compiled VAE, temporal-parallel VAE, H3 fusions, and regional
sparse-DiT compilation. -->

## Video makes the systems problem larger

Low-batch autoregressive language-model decoding often streams weights and KV
cache data to produce one token at a time. A video diffusion transformer has a
different shape: every denoising evaluation processes tens or hundreds of
thousands of latent tokens in parallel. It is closer to repeatedly running a
very long, non-causal prefill than to token-by-token decoding.

H3 makes that workload concrete. Its local T2VA pipeline contains a Qwen3-VL
text encoder, a 33B single-stream audio-video diffusion transformer, a visual
VAE, and an audio VAE:

```text
text prompt
    |
Qwen3-VL text encoder
    |
H3 Omni DiT x N  ---- jointly predicts video and audio latents
    |                                      |
visual VAE                              audio VAE
    |                                      |
    +-------- 768p video + stereo audio ---+
```

FL2VA and Ref2VA add separately encoded frame or reference-conditioning tokens
to this core path.

The DiT dominates. Our optimized one-GPU Base H3 run spends 126.50 of 134.38
seconds—94.1% of end-to-end latency—inside denoising for a five-second,
768x1344 output. That request packs 38,224 tokens and invokes the transformer
49 times.

The pressure grows with duration and conditioning. H3 supports
text-to-audio-video (T2VA), first/last-frame-to-audio-video (FL2VA), and
reference-to-audio-video (Ref2VA). Ref2VA is one of the most flexible and
computationally demanding workflows because reference tokens are processed
alongside the output tokens. Our nominal 15-second, 768p Ref2VA workload packs
220,628 tokens. At that shape, attention accounts for 91.5% of the dense DiT
wall time. Because attention scales quadratically with sequence length,
reference-conditioned long video is where step reduction and structured
sparsity should compound most strongly.

## From 49 transformer forwards to 4

FastH3 combines step distillation with sparse attention. DMD2 learns a
four-forward trajectory, backward simulation keeps training on the states that
trajectory actually visits, and VSA reduces the cost of each student forward.

**Distribution matching distillation.** We train a student, retain a frozen
dense H3 teacher, and train a dense fake-score critic. The critic learns the
score of the student's current distribution. The student then follows the
difference between the teacher score and critic score, using the
[DMD2 objective](https://arxiv.org/abs/2405.14867) to learn a four-forward
sampling trajectory without a paired video dataset.

Here, "data-free" means that training does not require paired target videos;
it still uses text prompts.

Training alternates four critic updates with one student update, using a global
batch of 64, FP32 master weights, and BF16 block compute.

**Backward simulation.** Training must expose the student to the states it
will actually encounter at inference. Instead of supervising only isolated
noise levels, carried student trajectories walk the same four-rung ODE ladder
used at serving time. Each training stream advances one rung at a time. This
keeps the training distribution aligned with the student's own accumulated
errors.

**Sparse distillation.** The student is trained with
[Video Sparse Attention](https://arxiv.org/abs/2505.13389), which prunes 90% of
eligible video-to-video tiles using 64-token blocks; text, audio, and exempt
prefix interactions remain dense. The teacher and critic are fully dense.
Sparsity is part of the learned student rather than a switch applied after
training. On Blackwell, FastVideo executes that pattern with its tile-64
`sm100a` CUDA kernel and uses FlashAttention-4 for eligible dense attention
paths.

The result is 49 to four DiT calls—a 12.25x reduction in evaluations—plus a
sparser student forward. End-to-end speedup is smaller because the text
encoder, VAEs, audio decode, and file writing do not disappear. That is why
FastVideo also compiles and parallelizes the video VAE and regionally compiles
the sparse DiT in the fastest serving profile.

## Quality: what works and what does not yet

FastH3 Preview is not Base H3 compressed into four lossless steps. The Base H3
leaderboard position belongs to the Base model and hosted H3 system; it does
not transfer automatically to this student. We are releasing the Preview so
the community can evaluate the tradeoff directly and help us improve it.

| Works today | Preview limitations |
|---|---|
| 768p, five-second T2VA at the native 124-frame shape | Quality remains below 49-forward Base H3 |
| Synchronized H.264 video and 32 kHz stereo AAC audio | Fast and intricate motion can lose detail or temporal stability |
| Four-forward trained ladder with guidance 1.0 | Fine audio detail can be weaker than Base H3 |
| Learned VSA pruning 90% of eligible video tiles with 64-token blocks | Other step counts and timestep grids are out of distribution |
| One- and four-GPU execution on high-memory GPUs | The measured v0.1 one-GPU route peaked near 77 GiB; 80 GB accelerators remain unvalidated |
| Nominal 10- and 15-second runtime paths | The student was trained at five seconds; longer-duration quality is not yet a release guarantee |

The v0.2 checkpoint continues the same run for 1500 additional updates. In our
qualitative review, it shows sharper still-image detail and steadier audio-video
synchronization relative to v0.1, while high-motion detail remains the clearest
weakness. We have regenerated all 64 held-out five-second prompts with the
correct four-forward ladder and completed a 64-prompt structural test at
nominal 15 seconds. Those tests establish that the schedule, sparse route,
frame count, video stream, and stereo audio stream are correct. They are not a
substitute for a preference benchmark against Base H3 or closed systems.

This release does **not** distill FL2VA or Ref2VA. H3-Base supports both, but
FastH3 Preview does not package a distilled `transformer_ref`. Substituting the
T2VA student into the reference pipeline is useful for latency research, but
it is an untrained cross-workflow transplant and should not be presented as a
quality result.

<!-- TODO: Insert matched-seed evaluation grid from the corrected v0.2
four-forward, 64-prompt revalidation. Include audible clips and state that any
montage audio comes from the labeled side. -->

## Run FastH3

The measured route currently targets Blackwell and CUDA 13. The checkpoint is
about 138 GiB. The measured v0.1 one-GPU route peaked near 77 GiB of device
memory; budget more than 80 GiB and treat 80 GB accelerators as unvalidated.
The four-GPU default replicates the DiT, so use high-memory GPUs and leave ample
host memory and storage for model loading.

The commands below pin the FastVideo revision audited for this draft. The
reported profile requires Blackwell and CUDA 13; other CUDA and PyTorch
combinations may use fallback kernels but are not represented by these latency
numbers.

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
git checkout 6388db815b2f3d6b5fb05ff143c674a23a489878
uv venv --python 3.12 --seed
source .venv/bin/activate
UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"
```

Then generate one video. The first request can take several minutes because it
includes model loading and regional compilation; the reported latency is
steady-state after one full excluded warmup generation. This quickstart
disables that warmup and changes the default three timed requests to one:

```bash
PROMPT='integrated_multimodal_description: A cinematic close-up of a red fox walking through fresh snow at dawn, its breath visible in the cold air. overall_soundscape: Soft pawsteps compress the snow beneath a quiet winter wind and distant birds.'

python examples/inference/basic/basic_fasth3.py \
  --prompt "$PROMPT" \
  --output outputs/fasth3_fox \
  --profile all \
  --num-gpus 4 \
  --no-warmup \
  --repeats 1
```

This uses the fastest `all` profile. It is the latency route, not a
cross-route parity guarantee: regional compilation and H3 fusions change
floating-point operation order.

The example defaults to the public v0.2 checkpoint, 768x1344, 124 frames,
VSA sparsity 0.9, tile size 64, the Blackwell `sm100a` sparse kernel, compiled
and temporal-parallel video VAE, regional sparse-DiT compilation, and exactly
four transformer forwards. It runs an excluded warmup and keeps each timed
output so the printed median can be audited when the quickstart overrides are
not supplied.

To reproduce the benchmark protocol, use the exact same prompt, remove
`--no-warmup`, and set `--repeats 3`. The script will save one excluded warmup
plus three timed MP4s. For the reported eager sparse-DiT comparison route, add
`--profile strict --no-inference-torch-compile`. This disables H3 fusions and
regional DiT compilation, but deliberately leaves the compiled/parallel VAE
and other serving optimizations enabled.

On systems without the Blackwell sparse kernel, replace `--profile all` with
`--profile strict` and add `--no-inference-torch-compile`,
`--vsa-kernel triton`, and `--no-fa4`, then explicitly select the available GPU
count with `--num-gpus`. This is a slower, unbenchmarked fallback, not the
reported GB200 profile.

Do not change `--steps 5`, `--vsa-sparsity 0.9`, or
`--vsa-tile-size 64` when trying to reproduce the trained operating point with
`basic_fasth3.py`. The five-point CLI convention is specific to this example's
scheduler path. Use the included example for this release; do not copy
`--steps 5` into another API's `num_inference_steps` field without verifying
that it executes four denoiser calls on the trained ladder.

## What comes next

FastH3 Preview is a starting point, not the end state. Our next work falls into
four tracks.

**Improve the student.** We will train beyond the current preview, compare
data-free backward simulation with video-anchored training recipes,
and publish matched human and automated evaluations rather than inferring
student quality from Base H3's leaderboard position.

We will also benchmark alternative few-step methods—progressive or
consistency-style distillation, adversarial auxiliary objectives, and hybrids
with quantization-aware training—under the same quality protocol. Each changes
a different failure mode; none should be treated as a free speedup.

**Distill the workflows people build with.** T2VA is the cleanest starting
point, but FL2VA and Ref2VA are where downloadable weights enable the most
control. Ref2VA also creates the largest systems opportunity. We are exploring
reference-aware sparse policies that preserve the information users provide
instead of treating reference and output tokens identically.

**Reduce fixed pipeline cost.** At four forwards, the DiT is no longer the
whole story. Text encoding, reference encoding, video decoding, audio decoding,
and file output become first-order. FastVideo has already matched vLLM-Omni on
Base H3 and added compiled, temporal-parallel H3 VAE decoding. We will keep
hardening sparse full-graph compilation, parallel encode/decode, portable
kernels, and cold-start behavior.

**Make it fit more hardware.** The current one-GPU route still requires a
high-memory accelerator. We will evaluate FP8, NVFP4, quantization-aware
distillation, and CPU/GPU placement—but every smaller or faster configuration
will ship with an explicit quality gate. We do not want a better latency number
whose visual tradeoff is hidden.

## Open weights turn a model into an ecosystem

An API exposes the outputs a provider chose to serve. Downloadable weights
expose a research surface. With H3, the community can now ask questions that
were impossible to answer from a hosted endpoint: Which attention blocks can
be sparse? Which reference tokens matter? Can a four-step student retain
native audio? Can the 33B transformer be quantized without breaking motion?
Can a compiler capture the whole sparse forward? Which parts should scale
across GPUs, and which should stay local?

FastH3 Preview is our first set of answers. It reduces H3's denoising loop from
49 transformer forwards to four, trains those forwards with structured sparse
attention, and serves the complete audio-video pipeline through FastVideo. It
is fast, useful, and visibly unfinished. That is exactly why we are releasing
it now.

Try the [checkpoint](https://huggingface.co/FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2),
run the [example](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3.py),
and tell us where the four-step model holds up—and where it does not.

## Acknowledgements

We thank the MiniMax team for releasing H3-Base, the FlashAttention and CUTLASS
teams for the Blackwell attention stack, and the FastVideo contributors who
built and validated H3 model support, VSA kernels, compilation, sequence
parallelism, and VAE parallelism.

<!-- TODO: Add named contributors and affiliations after author approval. -->

## Citation

```bibtex
@misc{fastvideo_fasth3_2026,
  title        = {From 49 H3 Forwards to Four: Introducing FastH3 Preview},
  author       = {FastVideo Team},
  year         = {2026},
  howpublished = {\url{https://haoailab.com/blogs/}},
}
```
