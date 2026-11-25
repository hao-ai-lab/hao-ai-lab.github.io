+++
title = "FastPan2: Distilling a 29B World Model to 8 Steps with VSA2"
description = "How we turned PAN 2's 100-forward sampler into 8 sparse forwards: the DMPDD recipe, VSA2 attention, the data, and what it buys in speed and human preference."
date = 2026-11-25T00:00:00-08:00
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
ShowToc = true
TocOpen = false
draft = false
contentClass = "post-content-justified"
# Unlisted: the page is built and reachable by its link, but kept out of /blogs/ and RSS. Remove [_build] to list it.
[_build]
    list = "never"
[cover]
    image = "img/cover.jpg"
    alt = "A frame from a FastPan2 video: a swimmer diving into a pool, seen half above and half below the water line."
    relative = true
    hidden = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
+++

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" >}}

<!-- TODO before publishing: link the FastPan2 weights and the training code, and confirm the author list. -->

[PAN 2](https://github.com/ifm-ai/PAN2) is a 29B-parameter world model: given a starting image (or none) and a description of what happens next, it simulates up to 20 seconds of video. Its release sampler runs the transformer **100 times per video**: 50 denoising steps, each with a conditional and an unconditional pass for classifier-free guidance. A 10 s clip takes six minutes on one GPU.

**FastPan2** runs the same transformer **8 times**, with guidance distilled in and 90% of its video attention skipped. On eight B300 GPUs it renders a 5 s, 704×1248 clip in **2.2 s**, a 10 s clip in **3.8 s** and a 20 s clip in **7.7 s**. In a blind human-preference arena it is preferred at least as often as the 50-step PAN 2 it was distilled from.

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline muted loop preload="metadata" poster="media/hero-swimmer.jpg"
    aria-label="FastPan2 video of a swimmer diving into a pool, the view moving between above and below the water"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/hero-swimmer.mp4" type="video/mp4">
  </video>
  <p style="margin: 0.5rem 0 0; text-align: center; font-size: 0.9em; opacity: 0.75;">FastPan2, 8 transformer calls, 10 s at 704×1248. "A swimmer in his early 30s dives into the water, the view shifting between above and below as he powers through the race."</p>
</div>

## TL;DR

- **8 forwards instead of 100.** One distillation stage, **DMPDD**, combines distribution matching (DMD2) with Parallel Decoding Distillation (PDD). The student has 32 output heads on a fine time grid and fuses 4 of them per call. Guidance is distilled into it.
- **Sparse attention while distilling.** The student attends with **VSA2** at 90% sparsity: a gradient-free router over 2×2×2 windows picks which 4×4×8 video tiles each tile reads, and a gated coarse branch keeps global context. Attention does 13–17% of the dense FLOPs, and a 20 s forward is 1.67× cheaper.
- **Data and scale.** 1M clips at 704×1248, 5–20 s, 70% text-to-video and 30% image-to-video, trained on 1,024 B300 GPUs at batch 512 for 2,000 steps (about one pass). We release step 1,800.
- **Speed.** Against PAN 2 running in FastVideo on the same 8 GPUs, FastPan2 is **8.1×, 10.8× and 14.7×** faster at 5, 10 and 20 s. Against PAN 2's release launcher it is **68–127×** faster ([details](#part-4-speed)).
- **Quality.** In a blind arena of 275 votes, FastPan2 wins **60%** of its games, PAN 2 58%, FastH3 43% and Cosmos3-Super 34% ([details](#part-5-the-arena)).

## How this post is organized

- **[Part 1](#part-1-fewer-passes-dmpdd)** is how the number of transformer calls drops from 100 to 8: PDD's multi-head student, the DMD half, and how the two losses share one forward.
- **[Part 2](#part-2-cheaper-passes-vsa2)** is how each call gets cheaper: VSA2's router, its fine and coarse branches, and the fused training kernel on Blackwell.
- **[Part 3](#part-3-data-and-training-at-scale)** is the data, the mixture, and the training run on 128 nodes.
- **[Part 4](#part-4-speed)** is the latency of the 4-, 8- and 16-call students, and where the remaining time goes.
- **[Part 5](#part-5-the-arena)** is the human-preference arena.
- **[Appendix A](#appendix-a-attention-cost)** works through the attention FLOPs.

## Part 0: what one PAN 2 call costs

PAN 2's world simulator is a 29B multimodal diffusion transformer: 44 joint-attention blocks of 32 heads × 128 channels over a single sequence of text and video tokens, plus a 4-layer text refiner. The video goes through the Wan2.2 VAE, whose latent has 48 channels and is 16× smaller than the frame in each spatial direction and 4× in time; the transformer then patches it 2×2. At 704×1248 that is a grid of **22 × 39 = 858 tokens per latent frame**, and a clip of *T* seconds at 24 fps has 6*T* + 1 latent frames.

| Clip | Frames | Latent frames | Video tokens | Text tokens |
|---|---|---|---|---|
| 5 s | 121 | 31 | 26,598 | up to 1,536 |
| 10 s | 241 | 61 | 52,338 | up to 1,536 |
| 20 s | 481 | 121 | 103,818 | up to 1,536 |

Captions are PAN 2's structured JSON: video style, lighting, viewpoint, a subject list, a description of the first frame, time-stamped events and quality notes, encoded by a 9B Qwen3.5 text encoder. Image-to-video conditioning does not add tokens: the clean first latent frame and a one-channel mask are stacked on the channels of the noisy latent (48 + 48 + 1 = 97 input channels).

The release sampler integrates the rectified flow with 50 Euler steps at flow shift 9. Each step evaluates the transformer on the prompt and on PAN 2's structured negative caption (a description of artifacts to avoid, rendered to the clip's length), and mixes the two with guidance 2.5:

$$
x_{k+1} = x_k + (\sigma_{k+1} - \sigma_k)\,\big[v_\theta(x_k, \sigma_k, c_-) + 2.5\,\big(v_\theta(x_k, \sigma_k, c_+) - v_\theta(x_k, \sigma_k, c_-)\big)\big], \quad k = 0, \dots, 49.
$$

That is 100 transformer evaluations over 27k to 104k tokens, and attention alone is about 2 PFLOP per evaluation at 10 s ([Appendix A](#appendix-a-attention-cost)). We attack both factors: fewer evaluations (Part 1) and cheaper evaluations (Part 2).

## Part 1: fewer passes (DMPDD)

### 1.1 A student with 32 output heads

Parallel Decoding Distillation ([Shaul et al. 2026](https://arxiv.org/abs/2607.26004)) changes only the last layer of the transformer. PAN 2's output projection is copied **32 times**, so one forward produces 32 velocity predictions $q_0, \dots, q_{31}$, one for each interval of a fixed fine time grid:

$$
u_j = 0.999\,(1 - j/32), \qquad \sigma(u) = \frac{s\,u\,M}{u\,(s-1) + M}, \quad s = 9,\ M = 0.999.
$$

The grid is uniform on a base clock $u$, and each interval's noise level comes from PAN 2's own flow shift. Head $j$ is the velocity $\epsilon - x_0$ held constant on interval $j$, so a call that starts at node $n$ and owns heads $n, \dots, n+b-1$ advances the latent by

$$
x_{u_{n+b}} = x_{u_n} + \sum_{j=n}^{n+b-1} \big[\sigma(u_{j+1}) - \sigma(u_j)\big]\, q_j(x_{u_n}, u_n).
$$

All $b$ heads come from **one** forward at the call's starting point: the student has learned to predict, in parallel, where the trajectory goes over the next $b$ intervals. At inference the heads of a block are mixed inside the linear layer, so a call costs no more than one PAN 2 forward. With $b = 4$, eight calls cover the grid.

{{< image src="img/fig_schedule.svg" alt="Noise level against sampling progress. PAN 2 takes 50 small steps; FastPan2 takes 8 calls, each integrating 4 sub-intervals of a 32-interval grid." width="100%" title="Figure 1. PAN 2's 50 steps against FastPan2's 8 calls. Each call predicts 4 heads and integrates 4 intervals of the 32-interval grid." >}}

The same widened layer supports other budgets: the grid stays at 32 intervals and only the block size changes. We trained 16-, 8- and 4-call students with otherwise identical recipes and data order ([Part 4](#part-4-speed) has their speed).

{{< image src="img/fig_heads.svg" alt="The 32 heads grouped as 16 blocks of 2, 8 blocks of 4, or 4 blocks of 8." width="100%" title="Figure 2. One head grid, three step budgets." >}}

### 1.2 Two losses from one forward

DMPDD trains the student with two objectives at once. Each student step draws a training clip, picks a call boundary $u_n$ on the 8-call partition, noises the clip's latent to that level and runs the sparse student **once**. That one prediction is used twice.

**The DMD half (distribution matching).** Integrating all of the student's heads from $u_n$ to the end of the grid gives a clean estimate $\hat x_0$, the multi-head analog of DMD2's one-step prediction. $\hat x_0$ is re-noised to a random score time $\tau$, and two networks score it: the frozen **teacher** (PAN 2, dense attention) with guidance 2.5 against the structured negative, and a trainable **critic** (a dense copy of PAN 2) that tracks the distribution of the student's own outputs. The student's gradient is the difference of the two scores, pulled back through $\hat x_0$:

$$
\nabla_\theta \mathcal{L}_{\text{DMD}} \;\propto\; \mathbb{E}_{\tau,\,\epsilon}\Big[\big(s_{\text{critic}}(\tilde x_\tau, \tau) - s^{\,\text{cfg}=2.5}_{\text{teacher}}(\tilde x_\tau, \tau)\big)\,\frac{\partial \hat x_0}{\partial \theta}\Big], \qquad \tilde x_\tau = (1-\sigma_\tau)\,\hat x_0 + \sigma_\tau\,\epsilon.
$$

This is what lets 8 calls produce sharp samples: the student is judged on the distribution of what it produces, not on reproducing one teacher trajectory.

**The PDD half (trajectory matching).** One head $k$ inside the current call's block is sampled. The student's own heads are integrated from $u_n$ up to $u_k$ (without gradient), and the teacher, again with guidance 2.5, gives its velocity at that point. Head $k$ regresses onto it:

$$
\mathcal{L}_{\text{PDD}} = \big\| q_k(x_{u_n}, u_n) - v^{\,\text{cfg}=2.5}_{\text{teacher}}(x_{u_k}, u_k) \big\|^2 .
$$

This anchors every head to the teacher's flow, so the student stays on PAN 2's trajectories while DMD sharpens it. The student loss is $\mathcal{L}_{\text{DMD}} + 0.3\,\mathcal{L}_{\text{PDD}}$.

**The critic** is trained on the other steps, by flow matching on the student's clean estimates (its loss computed in $x_0$ space). It takes four updates for every student update. Its condition dropout is off, so it keeps the teacher's unconditional branch instead of relearning it from student samples.

{{< image src="img/fig_dmpdd.svg" alt="DMPDD loop: a training clip is noised and passed once through the sparse student; its heads give a clean estimate scored by the teacher and the critic (DMD loss) and one head is matched to the teacher's velocity (PDD loss)." width="100%" title="Figure 3. One student forward feeds both losses. The teacher and critic stay dense; only the student is sparse." >}}

### 1.3 Distilling guidance and conditioning

- **Guidance is distilled.** The teacher's score, the critic's score and the PDD target all use guidance 2.5 against PAN 2's structured negative caption, PAN 2's own release setting. Using the same scale in all three places matters: mismatched scales pull the student two ways. The student samples with no unconditional pass at all.
- **Image-to-video.** The first latent frame of an image-to-video clip is conditioning: never noised, never part of the integrated state, and excluded from guidance in every role, as in PAN 2's sampler. Under the causal Wan2.2 VAE, the first latent frame of a pre-encoded clip encodes its first pixel frame alone, so training needs no separate image encoding.
- **Score times.** Score times are drawn continuously over $[0.001, 0.999]$ on the shifted clock (score-time shift 1.0).

## Part 2: cheaper passes (VSA2)

After Part 1 every call still attends over up to 104k video tokens. Attention grows with the square of the sequence: at 20 s it is the largest part of a forward. FastPan2 replaces the student's dense attention with **VSA2** ([Zhang et al., ICML 2026](https://arxiv.org/abs/2609.32882)), the successor of our [Video Sparse Attention](https://arxiv.org/abs/2505.13389), while it distills. The teacher and the critic stay dense.

### 2.1 Blocks and tiles

The sequence is cut into blocks of 128 rows. Text comes first, as dense prefix blocks. The video follows as **4 × 4 × 8 tiles** (time × height × width) over the latent token grid. For image-to-video, the conditioning first latent frame gets a time tile of its own, so the clean frame never shares a tile with noisy ones. At 10 s the 61 × 22 × 39 grid becomes 16 × 6 × 5 = **480 tiles**.

### 2.2 Router, fine branch, coarse branch, gate

{{< image src="img/fig_vsa2.svg" alt="VSA2 layer: rows in 128-row blocks; a gradient-free router over 2x2x2 windows selects tile pairs per head; a fine branch attends within the kept tiles, a coarse branch attends between block means, and a zero-initialized gate merges them." width="100%" title="Figure 4. One VSA2 attention layer, per head." >}}

- **Router (no gradient).** Queries and keys are mean-pooled over 2×2×2 windows, 16 per tile. Each query window takes a softmax over every key window, and the probabilities are pooled back to tile pairs. Per head, VSA2 keeps the best $N_v \times K$ of the $N_v^2$ video tile pairs, plus the diagonal, with $K = \lceil 0.1\,N_v \rceil$ (sparsity 0.9). The budget is spread over query tiles by score: a busy tile can read more than its share and a static one less. Text keys are always kept.
- **Fine branch.** Exact attention of each query row over the rows of the tiles it kept. At 10 s a video query reads about 6,650 of 52,850 keys.
- **Coarse branch.** Q, K and V are mean-pooled per 128-row block, and attention between all block means is dense. This is cheap (about 480 blocks at 10 s) and carries global context that the fine branch skips.
- **Gated merge.** A learned projection gives a gate per row, head and channel:

$$
\text{out} = \text{Attn}\big(q, K_{\text{kept}}, V_{\text{kept}}\big) + g \odot \text{Attn}\big(\bar q, \bar K, \bar V\big), \qquad g = W_g\, x .
$$

$W_g$ is initialized to zero, so at the start of distillation every student block computes pure sparse attention over PAN 2's weights, and the gate learns how much global context to add back. The router decouples its pooling granularity (2×2×2 windows) from the attention granularity (4×4×8 tiles), which is what makes gradient-free routing precise enough. Distilling with VSA2 from a dense checkpoint is close to the paper's *sparse rebasing*: sparse attention takes over from a model that was pretrained dense.

### 2.3 One fused kernel on Blackwell

Training needs VSA2's forward and backward at PAN 2's shapes, 4 heads per GPU under 8-way sequence parallelism. Our sm_100a/sm_103a kernel runs the whole layer as **one forward and one backward**: the router, the top-K, the coarse branch, the fine branch and the gated merge. It is built on `tcgen05` tensor-core instructions with tensor memory, TMA and cluster launch control. On one GB200 GPU, for the forward and backward of one layer:

| Clip | Tokens | VSA2 (this kernel) | VSA at 128-token tiles | Dense flash SDPA |
|---|---|---|---|---|
| 8 s | 42,554 | **1.94 ms** | 4.56 ms | 8.75 ms |
| 12 s | 63,146 | **3.71 ms** | 6.17 ms | 18.85 ms |
| 16 s | 83,738 | **6.30 ms** | 8.94 ms | 40.06 ms |
| 20 s | 104,330 | **9.85 ms** | 12.21 ms | 61.90 ms |
| geomean | | **4.05 ms** | 6.80 ms (1.68×) | 20.6 ms (5.1×) |

- **Where the time goes.** The two fine-branch kernels take 83–90% of a call. The backward one runs at about 68% of its tensor-core ideal per kept tile pair; its limit is draining the fp32 dQ partials through shared memory.
- **Exact routing.** The top-K breaks ties at the threshold by index, not by arrival order, so repeated runs select the same tiles.
- **Two bugs worth knowing.** The kernels were first guarded for sm_100 only. On B300 (sm_103) they compiled to empty bodies that launched, wrote nothing and returned no error. Every kernel now traps on an architecture it was not built for. Second, an earlier 128-tile VSA kernel dropped work items on B300 through a hand-decoded cluster-launch-control response; VSA2 uses the documented decode and passed 3.46M forward and backward calls with an all-gather always in flight, with no NaNs and no run-to-run differences.

In the full attention layer on B300, including placing the rows into tiles, VSA2's forward and backward are 1.24–1.48× faster than the previous 128-tile VSA layer across 5–20 s clips.

### 2.4 What sparsity buys per forward

{{< image src="img/fig_forward.svg" alt="Transformer time per sample-forward on one B300: dense 1.10, 2.81, 7.97 s against VSA2 1.02, 2.17, 4.77 s at 5, 10 and 20 s." width="100%" title="Figure 5. Transformer time per sample per forward on one B300, dense PAN 2 against the VSA2 student." >}}

Attention does 13–17% of the dense FLOPs, but a forward also runs 29B parameters of linear layers that sparsity does not touch. The saving therefore grows with the clip: 1.08× at 5 s, 1.29× at 10 s and **1.67×** at 20 s. (Dense PAN 2 batches its two guidance passes, which makes its per-sample cost slightly lower, so these ratios are conservative.)

## Part 3: data and training at scale

### 3.1 The data

FastPan2 trains on **1M video clips** at 704×1248 and 24 fps, 5 to 20 s long, from PAN 2's supervised fine-tuning distribution. <!-- TODO: confirm how to name the data source publicly. -->

- **Pre-encoded once.** Every clip was encoded once with the Wan2.2 VAE and stored as a latent of shape [48, T, 44, 78]. Training reads latents from shared storage and never decodes video.
- **Full length.** Clips train at their full length in exact-length buckets: at each step every replica trains a clip of the same length, so sequence-parallel groups stay in lockstep without padding.
- **Captions online.** Captions are PAN 2's structured JSON and are encoded during training by the 9B text encoder, streamed one layer at a time from pinned host memory, so it takes no GPU memory between encodes.
- **The task mix follows PAN 2.** Each clip is image-to-video with probability 0.3 and text-to-video otherwise, PAN 2's own pretraining and fine-tuning ratio. The draw is a fixed function of the seed and the clip id, so every rank of a sequence-parallel group and every resumed run agree on it.

{{< image src="img/fig_data.svg" alt="Clip length mix: 19.6% 5-8 s, 26.4% 8-12 s, 24.2% 12-16 s, 29.4% 16-20 s; task mix 70% text-to-video, 30% image-to-video." width="100%" title="Figure 6. Length and task mixture of the 1M training clips. The mean clip is 12.7 s." >}}

By caption, the clips are mostly third-person footage (about 85%), with about 10% first-person and a little aerial. Lifestyle, cinematic, documentary, nature, wildlife, tutorial and cooking footage are the largest styles. A 64-clip validation set picked across 14 world-model and physics topics (liquids, collisions, deformables, fracture, fire, granular media, tools, vehicles, first-person and aerial camera motion, animals, sports, weather and machines) never trains.

### 3.2 The run

| | |
|---|---|
| Hardware | 128 nodes × 8 B300 = **1,024 GPUs** |
| Parallelism | One model replica per node: 8-way sequence parallelism + FSDP over its 8 GPUs; 128-way data parallelism across nodes |
| Batch | 128 replicas × 1 clip × 4 accumulation steps = **512 clips** |
| Steps | **2,000** (about one pass over the data); released checkpoint: step **1,800** |
| Optimizer | AdamW, β = (0.9, 0.999), weight decay 0.01, constant learning rate **5e-6** for the student and the critic |
| Update ratio | 4 critic updates per student update |
| Losses | DMD + 0.3 × PDD; guidance 2.5 in the teacher score, the critic score and the PDD target |
| Precision | fp32 master weights and bf16 compute for the student and the critic; the frozen teacher is stored in bf16 |
| Stability | gradient clipping at 1.0 |
| Throughput | critic step 15.0 s (29% MFU), student step 25.6 s (35% MFU) |
| Wall clock | 11.1 h for 2,000 steps, including a 128-video validation and a checkpoint export every 100 steps |

- **Memory.** Three 29B roles live on each node: two trainable ones with fp32 masters, gradients and two Adam moments, and the bf16 teacher, about 123 GB per GPU of model state before activations. Every node is one replica, so the sequence all-to-alls and the weight gathers stay on NVLink, and only the gradient all-reduce crosses the network.
- **Communication.** The gradient all-reduce across the 128 replicas runs once per step instead of once per micro-batch. FSDP sums the earlier micro-batches' shards locally.
- **Fused kernels.** Each block's Q/K RMSNorm + RoPE and LayerNorm + modulation run as fused Triton kernels, forward and backward. The student keeps the same fused kernels at inference.
- **Fault tolerance.** A resumable training state is saved every 50 steps. A job that loses a node resubmits itself without it and resumes from the newest complete state.

## Part 4: speed

All timings below are text-to-video at 704×1248 on B300 GPUs, warm, from the prompt to 8-bit frames in host memory. Image-to-video costs the same to within 0.1–0.3 s. "PAN 2 in FastVideo" is the 50-step, guidance-2.5 sampler running in FastVideo with the same decode and fused kernels as the students. Its 8-GPU numbers are medians of **339 full 50-step renders** from our arena runs. PAN 2's release launcher renders one video per GPU, so its latency does not fall with more GPUs, and its 50 steps are extrapolated from 3 timed steps.

{{< image src="img/fig_latency.svg" alt="Latency on 8 B300 GPUs. 5 s: release 149.7 s, PAN 2 in FastVideo 17.9 s, FastPan2 16/8/4 calls 3.5/2.2/1.5 s. 10 s: 361.5, 41.2, 6.4, 3.8, 2.5 s. 20 s: 975.6, 113, 12.9, 7.7, 4.9 s." width="100%" title="Figure 7. Latency per video on eight B300 GPUs (log scale)." >}}

{{< table title="Table 1. Latency per video on 8 B300 GPUs, and FastPan2-8's speedup." >}}
| Clip | PAN 2 release | PAN 2 in FastVideo | FastPan2, 16 calls | **FastPan2, 8 calls** | FastPan2, 4 calls | 8-call speedup vs. FastVideo PAN 2 | vs. release |
|---|---|---|---|---|---|---|---|
| 5 s | 149.7 s | 17.9 s | 3.5 s | **2.2 s** | 1.5 s | **8.1×** | 68× |
| 10 s | 361.5 s | 41.2 s | 6.4 s | **3.8 s** | 2.5 s | **10.8×** | 94× |
| 20 s | 975.6 s | 113.0 s | 12.9 s | **7.7 s** | 4.9 s | **14.7×** | 127× |
{{</ table >}}

On a single B300 the 8-call student takes **9.4 s, 19.5 s and 42.4 s** for 5, 10 and 20 s clips, and the 4-call student 5.3, 10.9 and 23.3 s. FastPan2-8 renders a 10 s clip in 3.8 s and a 20 s clip in 7.7 s on 8 GPUs, both faster than real time.

The speedup over PAN 2 is less than the 12.5× cut in forwards at 5 s and more than it at 20 s. At 5 s, fixed costs that do not shrink with the number of calls (text encoding, decoding, the copy to host) are about 40% of the 2.2 s. At 20 s, VSA2 also makes each forward 1.67× cheaper (Figure 5).

{{< image src="img/fig_breakdown.svg" alt="FastPan2 8-call latency breakdown on 8 B300s: 5 s 2.21 s (denoise 1.35), 10 s 3.84 s (denoise 2.62), 20 s 7.66 s (denoise 5.43)." width="100%" title="Figure 8. Where FastPan2's time goes on eight B300 GPUs." >}}

With 8 calls, the parts outside the transformer start to matter. Three changes to FastVideo's PAN 2 pipeline keep them small:

- **A Wan2.2 VAE decode kernel.** It replaces the fp32 decode of the reference pipeline. A 20 s clip decodes in 2.7 s instead of 20.1 s on one GPU, with SSIM 0.998–0.999 against the fp32 decode.
- **Split decode.** The decode is split across the sequence-parallel GPUs, so a 20 s clip decodes in 0.47 s on 8 of them.
- **Fused kernels at inference.** The student samples with the same fused Q/K-norm + RoPE and LayerNorm + modulation kernels it trained with, and its VSA2 tiles are written straight into the kernel's layout.

## Part 5: the arena

Benchmarks capture parts of quality. To see whether people actually prefer the videos, we ran a blind pairwise arena. Raters see two videos for the same prompt, from two randomly chosen models, side by side, and pick the better one or call a tie.

- **Prompts.** 141 prompts, 5–20 s, text-to-video and image-to-video: human motion, sports, hand manipulation, physics, camera motion and aesthetics.
- **PAN 2 models.** FastPan2 and PAN 2 (50 steps, guidance 2.5) render the same prompts with the same seed, rewritten into PAN 2's structured caption.
- **External baselines.**
  - **[FastH3](/blogs/fasth3-v21/)**, 8 steps, text-to-video only, up to 15 s at 1344×768.
  - **Cosmos3-Super**, 35 steps, guidance 6, 8.2 s at 1280×720, with prompts upsampled by its own templates.
- **Encoding.** Every video is re-encoded through the same encoder.
- **Votes.** 275 votes from 8 raters. Two other FastPan2 checkpoints were also in the pool and count as opponents.

{{< image src="img/fig_arena.svg" alt="Arena win rates with 95% intervals: FastPan2 0.60, PAN 2 0.58, FastH3 0.43, Cosmos3-Super 0.34." width="100%" title="Figure 9. Win rate against the rest of the pool, ties counting half, with 95% bootstrap intervals." >}}

{{< table title="Table 2. FastPan2-8 head to head (wins–losses, ties)." >}}
| FastPan2-8 against | FastPan2 wins | Opponent wins | Ties |
|---|---|---|---|
| PAN 2, 50 steps + CFG | 10 | 7 | 8 |
| FastH3, 8 steps | 13 | 8 | 2 |
| Cosmos3-Super | 17 | 6 | 5 |
{{</ table >}}

FastPan2 and its teacher are statistically tied: their intervals overlap almost entirely, and the head-to-head is 10–7 with 8 ties. **12.5× fewer forwards and 90% sparse attention cost no measurable preference.** Both are preferred over FastH3 and over Cosmos3-Super. Split by task, FastPan2's win rate is 0.60 on text-to-video and 0.60 on image-to-video.

### Side by side

<div style="display: grid; gap: 1rem; margin: 1.2rem 0;">
  <figure style="margin: 0;">
    <video controls playsinline muted loop preload="metadata" poster="media/compare-judo.jpg" style="display: block; width: 100%; height: auto; border-radius: 8px; background: #000;" aria-label="Judo throw, PAN 2 on the left and FastPan2 on the right">
      <source src="media/compare-judo.mp4" type="video/mp4">
    </video>
    <figcaption style="font-size: 0.85em; text-align: center; opacity: 0.8;">Left: PAN 2, 100 forwards. Right: FastPan2, 8 forwards. "Two martial artists grip-fight on a padded mat; the woman in blue turns in, loads the man in white across her hip, throws him onto his back, and retains sleeve control."</figcaption>
  </figure>
  <figure style="margin: 0;">
    <video controls playsinline muted loop preload="metadata" poster="media/compare-swimmer.jpg" style="display: block; width: 100%; height: auto; border-radius: 8px; background: #000;" aria-label="Swimmer, PAN 2 on the left and FastPan2 on the right">
      <source src="media/compare-swimmer.mp4" type="video/mp4">
    </video>
    <figcaption style="font-size: 0.85em; text-align: center; opacity: 0.8;">Left: PAN 2. Right: FastPan2. Same prompt and seed as the video at the top.</figcaption>
  </figure>
</div>

### More FastPan2 samples

All are 8-call FastPan2 renders from the arena set. The captions give the short prompt, which is rewritten into PAN 2's structured caption before sampling.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 1rem; margin: 1.2rem 0;">
  <figure style="margin: 0;"><video controls playsinline muted loop preload="metadata" poster="media/gallery-cat.jpg" style="display:block;width:100%;height:auto;border-radius:8px;background:#000;"><source src="media/gallery-cat.mp4" type="video/mp4"></video><figcaption style="font-size:0.8em;opacity:0.8;">10 s. "A cat jumps onto a kitchen counter, sniffs a glass of water, taps the rim with one paw, and watches the glass wobble in place."</figcaption></figure>
  <figure style="margin: 0;"><video controls playsinline muted loop preload="metadata" poster="media/gallery-air-hockey.jpg" style="display:block;width:100%;height:auto;border-radius:8px;background:#000;"><source src="media/gallery-air-hockey.mp4" type="video/mp4"></video><figcaption style="font-size:0.8em;opacity:0.8;">12.6 s. "A striped-sweater boy and a dark-haired woman sustain a fast air-hockey rally across a blue arcade table."</figcaption></figure>
  <figure style="margin: 0;"><video controls playsinline muted loop preload="metadata" poster="media/gallery-cable-car.jpg" style="display:block;width:100%;height:auto;border-radius:8px;background:#000;"><source src="media/gallery-cable-car.mp4" type="video/mp4"></video><figcaption style="font-size:0.8em;opacity:0.8;">10 s. "At a high-altitude overlook, rugged terrain tumbles into mist-filled valleys. A cable car glides silently across a chasm."</figcaption></figure>
  <figure style="margin: 0;"><video controls playsinline muted loop preload="metadata" poster="media/gallery-badminton.jpg" style="display:block;width:100%;height:auto;border-radius:8px;background:#000;"><source src="media/gallery-badminton.mp4" type="video/mp4"></video><figcaption style="font-size:0.8em;opacity:0.8;">20 s. "Four doubles players sustain a long badminton rally on a green indoor court."</figcaption></figure>
  <figure style="margin: 0;"><video controls playsinline muted loop preload="metadata" poster="media/gallery-downhill.jpg" style="display:block;width:100%;height:auto;border-radius:8px;background:#000;"><source src="media/gallery-downhill.mp4" type="video/mp4"></video><figcaption style="font-size:0.8em;opacity:0.8;">20 s. "A first-person downhill cyclist leaves a crowded elevated start, descends constructed ramps, crosses a lower plaza, and continues down a tight brick-and-stone stair alley."</figcaption></figure>
  <figure style="margin: 0;"><video controls playsinline muted loop preload="metadata" poster="media/gallery-judo.jpg" style="display:block;width:100%;height:auto;border-radius:8px;background:#000;"><source src="media/gallery-judo.mp4" type="video/mp4"></video><figcaption style="font-size:0.8em;opacity:0.8;">10 s. "Two martial artists grip-fight on a padded mat; the woman in blue throws the man in white onto his back."</figcaption></figure>
</div>

## Putting it together

- **Fewer passes.** PDD's 32-head student and the DMD critic turn 50 guided steps into 8 unguided calls.
- **Cheaper passes.** VSA2, trained in the same stage, removes 83–87% of the attention FLOPs.
- **Scale.** One pass over 1M clips on 1,024 B300s takes about 11 hours.
- **Inference.** A fast VAE decode and fused kernels keep the rest of the pipeline from eating the gain.

The result renders a 10 s, 704×1248 world-model rollout in 3.8 s on one node, and people prefer it as often as the model it came from.

## Appendix A: attention cost

Per layer and head, with head dimension $d = 128$, $N_v$ video tokens, $L$ text tokens ($L = 512$ below) and $N = N_v + L$:

$$
F_{\text{dense}} = 4\,d\,N^2,
$$

$$
F_{\text{VSA2}} = \underbrace{4\,d\,\big[N_v\,(128K + L) + L\,N\big]}_{\text{fine}} + \underbrace{4\,d\,\lceil N/128 \rceil^2}_{\text{coarse}} + \underbrace{2\,d\,(N_v/8)^2}_{\text{router}}, \qquad K = \lceil 0.1\, n_t \rceil,
$$

where $n_t$ is the number of video tiles. Text queries attend densely, and every video query keeps all text keys.

| Clip | $N_v$ | Tiles $n_t$ | $K$ | Keys read per video query | Dense, per forward (44 layers × 32 heads) | VSA2 / dense |
|---|---|---|---|---|---|---|
| 5 s | 26,598 | 270 | 27 | 3,968 of 27,110 | 0.52 PFLOP | 17.0% |
| 10 s | 52,338 | 480 | 48 | 6,656 of 52,850 | 2.01 PFLOP | 14.2% |
| 20 s | 103,818 | 930 | 93 | 12,416 of 104,330 | 7.84 PFLOP | 13.1% |

- **The router is cheap.** It costs 0.8% of dense attention: it scores pooled windows, 8 tokens each.
- **The arithmetic saving is not the time saving.** Tiles are padded at the edges of the grid, the softmax and the gate add work, and the kernel runs at about half its floor. The measured transformer saving (Figure 5) also includes the linear layers, which sparsity does not touch.

## Acknowledgements

FastPan2 builds on [PAN 2](https://github.com/ifm-ai/PAN2) and its release. PDD comes from NVIDIA's FastGen team, and VSA2 from [Zhang et al.](https://arxiv.org/abs/2609.32882). Thanks to the raters who voted in the arena.

## Citation

```bibtex
@misc{fastpan2_2026,
  title  = {FastPan2: Distilling a 29B World Model to 8 Steps with VSA2},
  author = {FastVideo Team},
  year   = {2026},
  month  = {November},
  url    = {https://haoailab.com/blogs/fastpan2/}
}
```
