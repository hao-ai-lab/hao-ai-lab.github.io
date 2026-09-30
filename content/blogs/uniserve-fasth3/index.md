+++
title = "UniServe: Serving FastH3 at Its Fastest"
date = 2026-09-29T00:00:00-07:00
url = "/blogs/uniserve-fasth3/"
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
ShowToc = true
TocOpen = false
draft = false
math = true
contentClass = "post-content-justified"
summary = "UniServe serves FastH3 8-Step text-to-video-with-audio with lower median latency and higher throughput than FastVideo, vLLM-Omni and SGLang on GB200 and RTX PRO 6000, and Reactor runs it in production."
[cover]
    image = "img/architecture.png"
    relative = true
    alt = "UniServe architecture on four GB200 GPUs"
    hidden = true
[socialIcons]
   [[socialIcons.icon]]
     name = "twitter"
     url = "https://twitter.com/haoailab"
+++

{{< socialBadges github="hao-ai-lab/UniServe" >}}

**TL;DR**

- [UniServe](https://github.com/hao-ai-lab/UniServe) is a serving engine for FastH3 8-Step text-to-video-with-audio generation (more models to come!), built for production serving with our partners [Reactor](https://www.reactor.inc/) and [Nuva Lab](https://nuvalab.ai/).
- **Faster than real time.** On eight GB200 GPUs, a 5-second 1344×768 clip with audio arrives as a finished MP4 in 3.1 s and a 10-second clip in 6.3 s (5.4 s with NVFP4).
- **Throughput.** About 11,000 videos a day on eight GB200 GPUs and 3,400 on eight RTX PRO 6000 GPUs for a mix of 5- to 15-second clips: 6.4 and 21.2 GPU-seconds per second of generated video.
- **Against open-source serving.** 1.2–1.7× lower median latency and 15–45% higher throughput than FastVideo, vLLM-Omni and SGLang, each in its fastest exact configuration.
- **Hardware.** Tested on NVIDIA GB200 (four GPUs, and eight across two nodes) and RTX PRO 6000 Blackwell, with BF16 and NVFP4 checkpoints. UniServe also runs as an experimental backend for NVIDIA Dynamo.

{{< image src="img/results-at-a-glance.svg" alt="UniServe latency and throughput against FastVideo, vLLM-Omni and SGLang" width="100%" title="Figure 1. Median end-to-end latency with one request at a time (top) and the best throughput per GPU under load (bottom), on each hardware configuration. vLLM-Omni has no eight-GB200 latency result because one of its replicas cannot span two nodes." >}}
FastH3 turns a text prompt into a video with a synchronized soundtrack in eight denoising steps. Serving it efficiently means keeping a long-sequence multimodal diffusion transformer (DiT) busy, decoding hundreds of overlapping video tiles, and encoding the video and audio and delivering them to the client. A faster attention kernel addresses only one part of that path.

UniServe optimizes the whole request path. It shards the attention projections by head for Ulysses sequence parallelism, fuses sparse-attention preprocessing so that it writes directly into the layouts the attention kernels read, batches spatial tiles into each VAE decoder call, and streams decoded video to CPU encoders while later frames are still decoding. Precomputed schedule-dependent values, preallocated request buffers and CUDA graphs support this execution path.

This post first shows the results, then the mechanisms behind them. Unless noted otherwise, all systems run the FastH3 8-Step V2 checkpoint with BF16 text encoding and denoising and FP16 video-decoder projection weights. Latency is measured end to end: from request submission until the client receives the last byte of an MP4 containing both video and audio.

## Results at a glance

**Latency.** With one request at a time, UniServe's median latency is 1.25–1.32× lower than the baselines' on four GB200 GPUs, 1.30–1.73× lower on eight GB200 GPUs across two nodes and 1.22–1.44× lower on eight RTX PRO 6000 GPUs. On eight GB200 GPUs the median request finishes faster than real time at every clip length and prompt size: a 5-second clip in 3.1 s, a 10-second clip in 6.3 s and a 15-second clip with a 10K-token prompt in 14.1 s.

{{< image src="img/realtime-gb200-8.svg" alt="End-to-end latency against clip length on eight GB200 GPUs" width="100%" title="Figure 2. Median end-to-end latency per clip length on eight GB200 GPUs across two nodes, one request at a time." >}}
**Throughput.** Under load, UniServe's best throughput is 24–33% above the baselines' on four GB200 GPUs, 15–22% above on eight GB200 GPUs and 21–45% above on eight RTX PRO 6000 GPUs. For our mix of 5-, 10- and 15-second clips, that is about 6,100 videos a day on four GB200 GPUs, 11,000 on eight and 3,400 on eight RTX PRO 6000 GPUs in BF16, or 5.8, 6.4 and 21.2 GPU-seconds per second of generated video.

{{< image src="img/pareto.svg" alt="Throughput per GPU against latency under load" width="100%" title="Figure 3. Each point is one concurrency level: 1, 2, 4 and 8 concurrent requests, and 16 on eight GPUs. With two replicas, one request at a time leaves one replica idle, hence the jump from the first point to the second." >}}
**NVFP4.** FastVideo's NVFP4 checkpoint lowers UniServe's median latency by a further 10–15% and raises its throughput by 15–19%; [NVFP4](#nvfp4) explains what it quantizes and what it changes. The [methodology and full results](#benchmark-methodology-and-full-results) describe the workload and configurations and list every measured point.

## The FastH3 workload

Each request runs a Qwen3-VL-based text encoder, a two-layer text refiner, eight forward passes of a 50-layer DiT, and separate video and audio VAE decoders. The DiT attends over a joint sequence of text, video and audio tokens. Its hidden size is 5,376; attention has 56 heads of dimension 128, so the Q, K, V and gate projections each produce 7,168 channels. With four-way Ulysses sequence parallelism (SP4), each GPU computes attention for 14 heads.

The checkpoint is distilled for a fixed schedule of eight timesteps, `[999, 874, 749, 624, 500, 375, 250, 125]`, with timestep shifts of 10 for video and 3 for audio. The last solver update reaches the clean sample without another DiT forward. Because the schedule is fixed, every timestep-dependent value can be computed before any request arrives, which UniServe exploits below.

Video sparse attention (VSA) operates on tiles of 64 tokens. Text and audio queries attend to every key tile. Video queries attend densely to the text and audio keys and, among the video keys, to the highest-scoring 20% of tiles, rounded to a whole number of tiles. A second, coarse branch attends over mean-pooled tiles, and a learned gate adds its output to the fine sparse-attention output. UniServe computes exactly this, including the dense multimodal prefix and the coarse branch.

At 1344 × 768 and 24 fps, a 10-second request aligns to 243 frames. With a 1K-token prompt it has 72,576 video tokens, 810 audio tokens and 1,000 text tokens: 74,386 tokens in total. Tile alignment and the capacity padding of UniServe's preallocated buffers extend this to 78,080 rows. A 15-second request with a 10K-token prompt has 119,062 tokens and uses 125,696 buffer rows in the measured deployment; other systems pad differently. At these sequence lengths, every additional full-sequence intermediate tensor costs substantial memory traffic.

## Architecture

UniServe separates the control plane, the numerical code and the execution resources. A Rust server and engine handle the HTTP API, admission control, component placement and batch scheduling. Python GPU workers run the model and own every execution resource: request buffers, CUDA streams, transfers and CUDA graphs. Model code is ordinary PyTorch modules operating on tensors; the worker supplies their buffers and execution context.

{{< image src="img/architecture.svg" alt="UniServe architecture on four GB200 GPUs" width="100%" title="Figure 4. The measured four-GPU deployment runs text encoding, denoising and media decoding on the same four GPUs. Four CPU worker processes (host ranks) encode video units to H.264, and a separate host process encodes the audio and muxes the MP4." >}}
This deployment runs the text encoder with four-way tensor parallelism (TP4) and the DiT with Ulysses SP4. The video and audio decoders distribute temporal units (the groups of latent frames the VAE decodes together) across the same four GPUs. Placing each component separately matters because the text encoder, the long-sequence DiT and the tiled VAE decoder parallelize along different axes: raising the DiT's sequence-parallel degree alone does nothing for the decode path.

Each request's conditioning occupies a preallocated request slot, and its latents are double-buffered in two banks of pages: a denoising step reads the current latent from one bank and writes the next latent into the other, and the runtime commits the result when the batch completes. CUDA graphs are captured at startup for each shape bucket and address request slots and latent pages through device-side index tensors, so every request in the same bucket replays the same graphs on its own data.

The diffusion runner allocates its workspace once, for the largest supported shape, and smaller shapes run on views of that allocation. Constants and step graphs stay resident in worker-owned memory pools, so serving a request neither allocates a new workspace nor captures a new graph. The runner orders the calls that share scratch memory; the worker owns collectives and resource lifetimes.

## Precompute schedule-dependent values

### Precompute AdaLN modulation

Each DiT block projects the timestep embedding into per-modality shift, scale and gate vectors (AdaLN modulation). These depend only on the weights and the timestep, not on the prompt or the latent. Since the checkpoint fixes its eight timesteps, UniServe computes every modulation vector at load time and keeps only the results.

Each of the 50 layers produces 96,768 values (three modalities × six modulation terms × hidden size 5,376), and the final output modulation adds 10,752. With eight steps, each with separate video and audio timesteps, the BF16 table occupies:

$$
8 \times 2 \times (50 \times 96{,}768 + 10{,}752) \times 2\ \text{bytes}
= 147.984\ \text{MiB}.
$$

The projection weights and biases this table replaces occupy **24.288 GiB** in the checkpoint. UniServe streams them through the precomputation at load time and frees them afterwards. Both figures are tensor sizes only, excluding other weights, activations, workspaces and CUDA graphs.

### Run the text refiner once

The two-layer text refiner depends only on the encoded prompt, not on the timestep. UniServe runs it once per request and reuses its output across all eight DiT forwards.

## Head-sharded projections for Ulysses

Standard Ulysses sequence parallelism starts with each GPU holding a shard of the sequence. In the basic FastVideo path, each GPU applies the full, replicated Q, K, V and gate projections to its local tokens; an all-to-all then switches from sequence sharding to head sharding, so attention sees the full sequence for the local heads. A second all-to-all switches back to sequence sharding before the output projection.

UniServe changes the first half of this data flow. It shards the Q/K/V/gate weights by attention head, all-gathers the hidden states, which are narrower than the projected Q/K/V/gate tensors, and runs one merged projection for its 14 heads over the full sequence. Local rows are projected immediately. Remote rows arrive through chunked asynchronous all-gathers, and each chunk enters the projection and attention preprocessing as soon as it lands while later chunks are still in flight.

{{< image src="img/projection-dataflow.svg" alt="Projection and communication data flow" width="100%" title="Figure 5. Both paths compute the full token × head product across four GPUs. They differ in which tensor crosses the input collective and which projection weights each GPU stores." >}}
For a padded sequence length $S$, hidden size $H=5376$ and attention width $A=7168$, each GPU receives the following number of BF16 elements in the input collective:

$$
\begin{aligned}
\text{Projected Q/K/V/gate exchange} &: \frac{3}{4}\frac{S}{4}(4A),\\
\text{Hidden-state gather} &: \frac{3S}{4}H.
\end{aligned}
$$

The hidden-state all-gather moves **75% of the data** of the projected all-to-all, the ratio H/A. The output all-to-all is unchanged. This is a payload calculation for equal padding that excludes transport overhead; it is not a measured bandwidth or latency reduction, and because the systems pad differently it is not an exact ratio of traced bytes either.

Head sharding also reduces weight memory. The 200 Q/K/V/gate matrices (four per layer across 50 layers) each have shape `[7168, 5376]`. They occupy **14.355 GiB per GPU** when replicated and **3.589 GiB per GPU** when sharded four ways by head. The total projection FLOPs are unchanged: the work is repartitioned from local tokens × all heads to all tokens × local heads.

## Kernel fusion in the DiT

### Fused attention preprocessing

An unfused path runs Q/K RMSNorm, RoPE, gathering the valid rows, filling the padding, tile pooling and layout transposes as separate operations, and each materialized full-sequence tensor adds another round trip through memory.

UniServe's preprocessing kernel performs the learned Q/K RMSNorm, partial RoPE, tile pooling over valid tokens, tile packing and gate copying in a single pass per projected chunk. It writes the fine-attention inputs and the FP32 pooled statistics directly into the buffers the attention kernels read. Padding is handled inside this kernel, so it never becomes an input to later computation. Top-k tile selection and the fine and coarse attention kernels remain separate.

The fusion granularity is a chunk, not a whole transformer block: the 10 s/1K request makes 6,400 preprocessing calls on GPU 0, totaling **0.373 s**. Working per chunk lets preprocessing start as rows arrive from the all-gather.

For comparison, FastVideo's packing and pooling read separately materialized, normalized projections, and SGLang's fused packing and pooling also run after a separate RMSNorm/RoPE step. vLLM-Omni builds padded attention inputs through indexed writes and contiguous layout conversions. UniServe's wider fusion avoids these intermediate full-sequence tensors.

### Fused output combination

The fine sparse-attention output and the gated coarse output combine as:

$$
O_i = O_i^{\mathrm{fine}} + G_i \odot O_{\mathrm{tile}(i)}^{\mathrm{compressed}}.
$$

UniServe's combination kernel evaluates this expression and writes each row directly into its position in the send buffer of the output all-to-all. This fuses the combination with the layout conversion, avoiding a separately materialized combined tensor followed by a permute-and-copy. The all-to-all then restores the sequence sharding that the output projection and residual path consume.

The traces show what unfused layout conversions cost. In vLLM-Omni, the PyTorch profiler records 1,200 indexed Q/K/V writes into `[1, 77952, 14, 128]` buffers and 1,200 `contiguous()` calls on `[1, 14, 77952, 128]` tensors: padded token-major inputs are materialized and then converted to head-major layout. Nested operator records are counted once.

### Fused modulation, normalization and residuals

Each block applies AdaLN modulation around RMSNorm and a gated residual update. Schematically, the attention input is `RMSNorm(x) * (1 + scale_attn) + shift_attn`. After attention, the residual becomes `r = x + gate_attn * attention_output`, and the MLP input is the modulated RMSNorm of `r`.

UniServe fuses these steps. The pre-attention kernel looks up each token's modulation row, selected by its modality and the current timestep, while it normalizes, and saves the residual input in the same pass as chunks arrive. The post-attention kernel produces both the gated residual and the modulated, normalized MLP input. A fused gate/up projection with SwiGLU feeds the down projection, followed by the MLP's gated residual update. In the BF16 path, finished chunks flow into the next stage without first concatenating the whole shard.

### End-to-end DiT time matters more than one kernel

In this sample, the three optimized baselines' native sparse-attention kernels are faster than UniServe's:

| GPU 0 kernel time, 10 s/1K       | UniServe | FastVideo | SGLang  | vLLM-Omni |
| -------------------------------- | -------- | --------- | ------- | --------- |
| Fine sparse attention, 400 calls | 3.779 s  | 3.373 s   | 3.476 s | 3.251 s   |
| NCCL SendRecv, including waits   | 1.068 s  | 1.558 s   | 1.687 s | 1.323 s   |

*Summed kernel durations on one GPU; they are not additive request phases.*

UniServe runs fine sparse attention with its own CuTe kernel. Even so, its combined refiner and denoising time is shorter: **10.052 s**, versus **11.723 s** for FastVideo, **11.111 s** for SGLang and **12.580 s** for vLLM-Omni. The advantage comes from the computation and data movement around attention; these traces do not attribute it to individual mechanisms.

## A batched, streaming decode path

### Batch spatial tiles, distribute temporal units

The video VAE decoder reconstructs each temporal unit from overlapping spatial tiles. At 1344 × 768 the tile grid is 4 × 7: 28 tiles with a 256-pixel minimum tile size and a 64-pixel overlap. A straightforward implementation loops over the tiles and calls the decoder once per tile. Parallelizing the DiT leaves this loop untouched: in the basic SP4 trace, all video decoder kernels still run on GPU 0.

UniServe concatenates a unit's 28 tiles along the batch dimension and decodes them in one call, then restores the grid and blends the overlaps. Tiles remain independent samples in the batch, so the tiled computation itself is unchanged; no attention crosses tile boundaries. The worker distributes temporal units across the four GPUs.

{{< image src="img/media-placement.svg" alt="Spatial batching and temporal placement" width="100%" title="Figure 6. For four temporal units, UniServe issues four batch-28 decoder calls. FastVideo distributes temporal units across GPUs but loops over their spatial tiles. SGLang and vLLM-Omni distribute each unit's spatial tiles across GPUs. The diagram shows ownership and call counts, not timing or FLOP reductions." >}}
For the 10-second sample's 14 temporal units, the traces show **14 decoder calls in UniServe**, versus **392 per-tile calls** in FastVideo and SGLang. Each batched call amortizes the host-side launch work over a whole unit.

### Fuse the VAE decoder and replay it as a CUDA graph

Batching combines with kernel fusion inside the VAE decoder. UniServe's video decoder transformer fuses Q/K normalization with RoPE, fuses the scaled attention residual with the following RMSNorm, and defers each MLP residual addition into the next layer's normalization; the last layer folds its pending addition into the output LayerNorm. The normalization reads the FP32 residual sum, so FP16 projection weights do not round the residual stream to FP16. SwiGLU and the final unpatchify step, which turns tokens back into pixels, have their own fused kernels.

The worker captures the batched decode as a CUDA graph and replays it for each temporal unit. Across UniServe's video decoding and postprocessing, the 10-second trace records **14 graph launches**. Inside FastVideo's VAE stage, rank 0 issues 73,024 `cuLaunchKernel` and 24,420 `cuLaunchKernelEx` calls alongside 116 partial-graph launches, which take 1.024 s of CPU API time under profiling. Capturing the whole batched decode removes most of these returns to the host between kernels.

The video decode spans **1.107 s on the GPUs in UniServe and 4.818 s in FastVideo**. On rank 0, video kernels are active for 1.043 s of UniServe's 1.107-second span, versus 1.343 s of FastVideo's 4.817-second span, so UniServe keeps the GPU far busier during decoding. Batching, fusion, graph replay and scheduling act together here; this comparison does not isolate the benefit of any one of them.

### Encode while the GPU is still decoding

A video request is not finished when its pixels are reconstructed. UniServe copies each decoded RGB unit to host memory and schedules its encoding independently of the remaining GPU decoding. On the same host, the encoder reads the published host buffer directly instead of making another copy. Four CPU encoder processes encode units with libx264 while the GPUs continue decoding. The audio decoder produces PCM, which the host encodes to AAC. The muxer appends the already encoded video packets with adjusted timestamps and adds the audio track, without decoding or re-encoding any video.

The trace shows the overlap: the first video encode starts **10.469 s** after submission, while the last video decoder kernel finishes at **11.315 s**. The complete MP4 arrives at **11.414 s**, only **98.8 ms** after the last video kernel.

vLLM-Omni offers a similar option, `preencode_mp4`, which encodes each decoded chunk on the worker while later chunks decode; the vLLM-Omni benchmark results below use it. FastVideo and SGLang encode the MP4 after decoding finishes.

## Where the time goes

The following traces use the same 10 s/1K prompt and seed on four GB200 GPUs, one request at a time. Model warmup and IPC initialization complete before collection starts; both traced shapes are then warmed with the profiler active before the recorded requests. Each capture runs alone on its four-GPU node. PyTorch profiler traces capture eager operator shapes and calls; Nsight Systems captures the CUDA and NVTX timeline, including kernels inside CUDA graphs.

{{< image src="img/request-timeline-10s-1k.svg" alt="Five-system complete-request timeline" width="100%" title="Figure 7. All panels share one time axis, starting at request submission. The dashed line marks receipt of the complete MP4. GPU bars span the first to the last correlated kernel across the four GPUs, including idle gaps. Host output bars can overlap GPU work and include any format conversion inside those functions." >}}
| 10 s/1K request, seconds | Text encoder (GPU) | Refiner + denoising (GPU) | Decode (GPU) | End-to-end, profiled | End-to-end, unprofiled |
| ------------------------ | ------------------ | ------------------------- | ------------ | -------------------- | ---------------------- |
| UniServe                 | 0.020              | 10.052                    | 1.107        | 11.414               | 11.341†                |
| FastVideo                | 0.113              | 11.723                    | 4.953        | 17.735               | 16.496                 |
| SGLang                   | 0.161              | 11.111                    | 5.318        | 17.215               | 16.089                 |
| vLLM-Omni                | 0.119              | 12.580                    | 3.940        | 17.018               | 16.116                 |
| Basic SP4                | 0.112              | 15.752                    | 25.346       | 42.622               | 34.315                 |

*Decode covers video decoding, audio decoding and the final reconstruction steps; it excludes the GPU frame conversion performed by the output encoder. The intervals may overlap and should not be added. The unprofiled column is a warmed request served by the Nsight-traced process before collection starts; † UniServe's comes from its separate PyTorch-profiler process, also before profiling starts.*

*These are single-request diagnostics, not p50 estimates. Instrumentation perturbs execution, especially for the basic reference's many small kernel launches: its profiled request takes 42.622 s against 34.315 s unprofiled. We therefore do not derive speedup claims from these ratios. FastVideo, SGLang and vLLM-Omni are traced in the configurations their benchmark results use, with added NVTX instrumentation.*

## NVFP4

FastVideo also publishes an NVFP4 checkpoint, `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4`, and UniServe serves it natively. NVFP4 is the 4-bit floating-point format with native tensor-core support on Blackwell GPUs: it stores E2M1 values with one FP8 (E4M3) scale per 16-element block and one FP32 scale per tensor.

The checkpoint quantizes 150 DiT MLP projections (gate, up and down in each of the 50 layers) and 252 video-VAE decoder projections (Q/K/V, attention output and the three MLP projections in each of 36 layers). The text encoder and the DiT's attention projections stay in BF16. UniServe loads the packed weights as they are stored, two E2M1 values per byte plus the block scales, and quantizes activations on the fly against each module's calibrated global scale from the checkpoint, so every chunk of rows uses the same scale no matter how the sequence is split. The packed values and block scales go directly into a block-scaled FP4 GEMM that produces BF16 outputs.

The traces confirm that the FP4 path runs: 14,400 FlashInfer `DeviceGemmFp4` kernels in the DiT and 3,528 in the video decoder across the four GB200 GPUs of a 10 s/1K request. On GPU 0, DiT time falls from 10.243 s with BF16 to 8.512 s, and video decoder time from 1.195 s to 0.766 s. End to end, NVFP4 lowers UniServe's median latency by 9.6–14.7% and raises its throughput by 15.4–18.6% across the three hardware configurations; see [NVFP4 against BF16](#nvfp4-against-bf16).

**What is exact and what is lossy.** Every optimization in the sections above is exact: it changes where and when the same computation runs, so its output differs from a reference implementation only by floating-point summation order, and every collective, including the Ulysses all-gathers and all-to-alls, stays in BF16. Precision is a separate, explicit choice. The BF16 results in this post use UniServe's `quality` preset (BF16 DiT and text encoder, FP16 video-decoder projections) and the NVFP4 results FastVideo's calibrated NVFP4 checkpoint; for dense checkpoints UniServe also offers runtime presets that trade precision for speed: NVFP4 video-decoder projections in `balanced`, the default, plus an FP8 DiT MLP in `performance`, and an NVFP4 DiT MLP with an FP8 text encoder in `maximum`, with attention always in BF16. NVFP4 is lossy, and a same-seed comparison shows it: the clips below come from the same prompt and seed, yet details such as a performer's hair or apparent age differ. FastH3's eight-step sampler turns any numerical difference into a different sample of the same scene; in our tests, changing only the FP32 summation order of the pooled attention scores already produced visibly different videos. Same-seed divergence therefore does not by itself mean lower quality, but we have not yet evaluated the quality of the NVFP4 checkpoint or the runtime presets systematically, so we report NVFP4 as a separate speed tier rather than a free default.

The clips compare UniServe's BF16 (left) and NVFP4 (right) outputs for the same benchmark requests, 10 seconds with 1K-token prompts, from the eight-RTX PRO 6000 latency runs.

<figure>
<video style="width: 100%; height: auto;" controls muted loop playsinline preload="metadata" poster="img/videos/market-poster.jpg" src="img/videos/market-side-by-side.mp4"></video>
<figcaption style="font-size: 16px; font-weight: normal; color: #808080; text-align: center;">Market: BF16 (left) and NVFP4 (right).</figcaption>
</figure>

<figure>
<video style="width: 100%; height: auto;" controls muted loop playsinline preload="metadata" poster="img/videos/percussion-poster.jpg" src="img/videos/percussion-side-by-side.mp4"></video>
<figcaption style="font-size: 16px; font-weight: normal; color: #808080; text-align: center;">Percussion performance: BF16 (left) and NVFP4 (right).</figcaption>
</figure>

<figure>
<video style="width: 100%; height: auto;" controls muted loop playsinline preload="metadata" poster="img/videos/river-poster.jpg" src="img/videos/river-side-by-side.mp4"></video>
<figcaption style="font-size: 16px; font-weight: normal; color: #808080; text-align: center;">River: BF16 (left) and NVFP4 (right).</figcaption>
</figure>

<figure>
<video style="width: 100%; height: auto;" controls muted loop playsinline preload="metadata" poster="img/videos/ceramics-poster.jpg" src="img/videos/ceramics-side-by-side.mp4"></video>
<figcaption style="font-size: 16px; font-weight: normal; color: #808080; text-align: center;">Ceramics studio: BF16 (left) and NVFP4 (right).</figcaption>
</figure>

The side-by-side clips are silent because the two samples have different soundtracks. NVFP4 changes numerical precision in both the DiT and the video decoder, and in this tier the decoder's remaining dense projections run in BF16 rather than FP16, so we report NVFP4 as a separate UniServe tier instead of comparing it with the BF16 baselines. The other systems do not load this serialized checkpoint natively, and converting it would change the evaluated artifact.

## Serving with NVIDIA Dynamo

UniServe also runs as an experimental backend for [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo). Dynamo's frontend serves the OpenAI-style `/v1/videos` endpoint and handles service discovery and request routing.

`uniserve-dynamo-worker` embeds UniServe's Rust scheduler and engine in a Dynamo worker process and registers it as a video endpoint. Each request follows the same path as UniServe's own `/v1/videos` route, from validation through the finished MP4, and returns as one completed response with the MP4 embedded. The worker rejects request options that FastH3 does not implement instead of silently ignoring them. The integration currently supports text-to-video requests with complete, non-streaming responses. The [Dynamo quickstart](https://github.com/hao-ai-lab/UniServe/blob/main/docs/fast_h3/dynamo.md) shows how to serve FastH3 through Dynamo 1.5.0 on four GB200 GPUs.

## Benchmark methodology and full results

This section describes how we measured and lists every result behind the charts. Tables are folded; open them for the numbers.

### Baselines

We compare UniServe against two kinds of baselines.

**Basic SP4 reference.** To show what UniServe's optimizations contribute, we build a basic reference from the pinned FastVideo implementation. It already has the standard parallel setup: Ulysses SP4 denoising, TP4 text encoding, resident weights, the native VSA kernel, one text-refiner pass per request and FP16 VAE decoder projections. FastVideo's optional fusions, `torch.compile`, AdaLN precomputation and parallel VAE decoding are disabled.

**Optimized baselines.** FastVideo, vLLM-Omni and SGLang each run their fastest exact configuration. On GB200 all three use the native SM100a VSA kernel; vLLM-Omni and SGLang would otherwise use their Triton VSA kernels there, so we force the native kernel. Where a system lacked a deployment optimization that UniServe uses, we added it locally, so that the comparison measures execution pipelines rather than missing features: fixed-schedule AdaLN precomputation (added to FastVideo and vLLM-Omni; SGLang's existing implementation enabled for this checkpoint), FP16 storage of the video decoder's projection weights (added to FastVideo and vLLM-Omni; SGLang already stores them in FP16), and text-to-video loading that skips the unused image-conditioning components and VAE encoders. vLLM-Omni's worker-side MP4 encoding (`preencode_mp4`), which is off by default, is turned on, and every system encodes H.264 with the same libx264 settings as UniServe. Every system keeps its weights resident on the GPU. [Setup](#setup) lists the per-system settings and patches.

### Setup

**Hardware.** Four GB200 GPUs in one node; eight GB200 GPUs across two nodes connected by multi-node NVLink; and one node with eight RTX PRO 6000 Blackwell Server Edition GPUs with full peer-to-peer access. We treat each as a separate environment and do not read the change from four to eight GPUs as strong scaling, because the topology changes too. On the GB200 nodes every system runs with NCCL's NVLS multicast disabled (`NCCL_NVLS_ENABLE=0`).

**Systems and checkpoints.** Every system serves the BF16 checkpoint `FastVideo/FastVideo-FastH3-8-Step-V2` with its trained schedule, VSA sparsity and full decoders; only UniServe also serves the NVFP4 checkpoint `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4`. Approximations that change the computation are excluded: caching features across steps, skipping softmax work, approximate decoders, progressive resolution and fewer denoising steps. Optimizations that preserve the computation are allowed: compilation, exact fused kernels, sequence and tensor parallelism, component placement, precomputing request-independent values and asynchronous media processing. Every system selects the checkpoint's top-k video tiles exactly. vLLM-Omni loads the video and audio VAEs from the base MiniMax-H3 repository that its release pins; every tensor it loads is identical to the V2 release's.

| System                       | Checkpoint  | 4 x GB200                    | 8 x GB200, two nodes                       | 8 x RTX PRO 6000      |
| ---------------------------- | ----------- | ---------------------------- | ------------------------------------------ | --------------------- |
| UniServe                     | BF16, NVFP4 | Supported                    | Supported                                  | Supported             |
| FastVideo                    | BF16        | Supported                    | Supported                                  | Supported             |
| vLLM-Omni                    | BF16        | Supported, native SM100a VSA | Throughput layout only, native SM100a VSA† | Supported, Triton VSA |
| SGLang                       | BF16        | Supported, native SM100a VSA | Supported, native SM100a VSA               | Supported, Triton VSA |
| FastVideo, vLLM-Omni, SGLang | NVFP4       | N/A                          | N/A                                        | N/A                   |

*Supported configurations. On GB200, vLLM-Omni and SGLang are switched from their default Triton VSA kernels to the native SM100a kernel. † vLLM-Omni runs the sequence-parallel workers of a diffusion stage as local processes on one host, so a single replica cannot span the two GB200 nodes; it serves the two-replica throughput layout and has no eight-GB200 latency result. N/A: these systems do not load the serialized NVFP4 checkpoint natively, and converting it would change the evaluated artifact.*

**Layouts.** On each hardware configuration, every system serves latency in one shared layout and throughput in another. A single request is fastest when all GPUs work on it, so every latency layout is one replica spanning the whole machine. The throughput layouts come from qualification runs over each hardware configuration's replica × sequence-parallel options. A shared layout is not every system's own best: in qualification on four GB200 GPUs, FastVideo's four single-GPU replicas delivered 2.5% more throughput than one four-GPU replica, and SGLang's two two-GPU replicas 0.8% more.

| Hardware             | Latency layout           | Throughput layout                              |
| -------------------- | ------------------------ | ---------------------------------------------- |
| 4 x GB200            | `r1s4`, text encoder TP4 | `r1s4`, text encoder TP4                       |
| 8 x GB200, two nodes | `r1s8`, text encoder TP8 | `r2s4`, one replica per node, text encoder TP4 |
| 8 x RTX PRO 6000     | `r1s8`, text encoder TP8 | `r2s4`, text encoder TP4                       |

*Layouts.* `r` *is the number of replicas and* `s` *the DiT's sequence-parallel degree; the text encoder is tensor-parallel across each replica's GPUs.*

**Baseline configuration.** Within a layout, each system runs its fastest exact configuration. No system offloads weights to host memory or shards DiT weights with tensor parallelism; with these settings every system keeps all weights resident on every hardware configuration.

| Setting                                 | UniServe                                                                                     | FastVideo                         | vLLM-Omni                                      | SGLang                                          |
| --------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| DiT weights                             | Resident                                                                                     | Resident                          | Resident                                       | Resident                                        |
| Text encoder                            | TP over the replica                                                                          | TP over the replica (`tp_size`)   | TP over the replica (`--text-encoder-tp-size`) | TP over the replica (`--encoder-parallel fold`) |
| AdaLN modulation for the fixed schedule | Precomputed at load                                                                          | Precomputed at load               | Precomputed at load                            | Precomputed on first use                        |
| Video VAE decoder projections           | FP16                                                                                         | FP16                              | FP16                                           | FP16                                            |
| Components unused by text-to-video      | Not loaded                                                                                   | Not loaded                        | Not loaded                                     | Not loaded                                      |
| Compilation                             | CUDA graphs for four-GPU replicas on GB200; eager for eight-GPU replicas and on RTX PRO 6000 | DiT: regional `torch.compile` on GB200 (it requires the SM100a kernel); VAE decoder compiled on all hardware | Regional `torch.compile`                       | None: eager, because SGLang's VSA-H3 validation refuses `torch.compile` and breakable CUDA graphs |
| MP4 encoding                            | Host workers, overlapped with decoding                                                       | After decoding                    | On the worker, overlapped with decoding (`preencode_mp4`) | After decoding, followed by an `ffprobe` check  |
| H.264 encoder                           | libx264 `ultrafast`, CRF 23                                                                  | libx264 `ultrafast`, CRF 23       | libx264 `ultrafast`, CRF 23; CRF 18 on RTX PRO 6000 | libx264 `ultrafast`, CRF 23                     |
| Allocator                               | Expandable segments                                                                          | Expandable segments               | Expandable segments                            | Expandable segments                             |

*Per-system settings within each layout.*

The baselines reach these settings through small local changes, published as patches against their served revisions:

- **AdaLN precomputation.** SGLang already implements it (`--minimax-h3-adaln-online`) but rejects the Diffusers-layout checkpoint because it looks up weights by their native tensor names; the patch resolves the names through SGLang's own checkpoint mapping. vLLM-Omni and FastVideo gain the same precomputation: the projections are evaluated once per schedule step with the model's own time embedder and the same kernels and row counts as the per-step path, and the transformer blocks read the results.
- **FP16 video-VAE decoder weights.** FastVideo and vLLM-Omni run the video decoder under FP16 autocast, which casts the FP32 projection weights to FP16 on every call. The patch stores these weights in FP16 at load time, which removes the per-call casts and halves their memory; the stored weights are the values the casts would produce. SGLang already stores its decoder weights in FP16 and needs no change.
- **Text-to-video components only.** The video and audio VAE encoders and the Qwen3-VL vision tower are not loaded, and any request that would need them is rejected.
- **H.264 encoder settings.** SGLang gains the `x264_preset` request field from [sglang#41797](https://github.com/sgl-project/sglang/pull/41797), applied to its served revision.

These changes sit on top of small support fixes the baselines need to serve the V2 checkpoint correctly on GB200: SGLang loads the eight-step checkpoint through a local model overlay and selects the SM100a VSA kernel explicitly, vLLM-Omni derives its selected video tiles from the checkpoint's sparsity, and FastVideo sizes its `torch.compile` cache for all six served shapes.

Every system encodes the H.264 stream with libx264's `ultrafast` preset at CRF 23, the settings UniServe and FastVideo use by default. vLLM-Omni receives them per request through `video_codec_options`, replacing its default CRF 18. SGLang receives `x264_preset` set to `ultrafast` and `output_compression` set to 53, which its encoder maps to CRF 23, replacing its default `fast` preset at CRF 25; it encodes with the host's current FFmpeg build, selected through `IMAGEIO_FFMPEG_EXE`, instead of the FFmpeg 4.2.2 bundled with imageio-ffmpeg, and it still probes each written file with `ffprobe`, as it does when serving. On the RTX PRO 6000, vLLM-Omni was measured at its default CRF 18 with the `ultrafast` preset; on four GB200 GPUs, moving vLLM-Omni from CRF 18 to CRF 23 changed its per-shape median latencies by at most 0.07 s. At the same settings UniServe's files are about 9% larger (median video bitrate 10.7 Mb/s against 9.7–9.9 Mb/s on the four-GB200 latency outputs), consistent with its encoding each temporal unit, in parallel, as its own segment that starts with a keyframe (seven keyframes in a 5-second clip, against one for the other systems).

**Workload.** Six independently written scene families (a harbor, a ceramics studio, a percussion performance, a river, an observatory and a market) supply structured prompts describing visual action, camera, material detail and environmental sound. Each family has a 1,000-token and a 10,000-token version under the checkpoint's tokenizer; the longer version adds meaningful description of the same continuous event, never repeated filler. Every request has unique prompt bytes, so no prefix or result cache can turn repetition into a speedup. Prompt text, token IDs, SHA-256 digests, seeds and execution order are frozen in the published manifest.

**Metrics.** The latency benchmark sends 72 requests at concurrency 1, after all six shapes are warmed: six scene families × two prompt lengths × three durations × two seeds, in one fixed random order shared by every system. We report p50, p90 and p95 over all 72 requests and the median of the 12 requests of each shape (duration × prompt length). The throughput benchmark sends 32 requests per concurrency level, five or six per shape, in one fixed shuffled order; the client keeps at most `C` requests in flight with no think time, for `C` = 1, 2, 4 and 8 on four GPUs and additionally 16 on eight, and each level is preceded by `2C` priming requests that are not counted. Throughput is the number of valid complete MP4s divided by the time from the first submission to the last response; invalid, failed or timed-out requests stay in the denominator. Each concurrency level is one run, so the numbers are completion rates over a finite request set, including ramp-up and drain, not sustained-capacity claims, and small differences between levels should not be read as rankings. A response is valid if it contains exactly one H.264 stream at 1344 × 768 and exactly 24 fps with a frame count within one frame of the aligned count, a stereo 32 kHz AAC track whose duration is within one frame period of the aligned media duration, and nonzero video variance and audio RMS. Within one environment, runs execute one at a time with one server deployment and one client, and each benchmark starts a fresh deployment that performs its own shape warmup and priming.

### Latency

{{< image src="img/latency-per-shape.svg" alt="Median latency per shape on every hardware configuration" width="100%" title="Figure 8. Median end-to-end latency of each shape (12 requests per point), one request at a time. The two panels of a row share their y-axis. vLLM-Omni has no eight-GB200 latency result because one of its replicas cannot span two nodes. The folded tables below list every value and the quantiles over all 72 requests." >}}
**Four GB200 GPUs.** **Across the 72-request workload, UniServe BF16 has a median end-to-end latency of 14.24 s, compared with 17.80 s for SGLang, 18.17 s for FastVideo and 18.83 s for vLLM-Omni.** Across the six shapes, SGLang's median is 1.18 to 1.30 times UniServe's, FastVideo's 1.23 to 1.33 times and vLLM-Omni's 1.28 to 1.37 times. SGLang is the fastest baseline here and on eight GB200 GPUs although its DiT runs eager: its VSA-H3 validation refuses `torch.compile` and breakable CUDA graphs, while FastVideo and vLLM-Omni compile theirs.

<details><summary>End-to-end latency on four GB200 GPUs</summary>

| System    | Precision | p50 (s)   | p90 (s)   | p95 (s)   | Valid / attempted |
| --------- | --------- | --------- | --------- | --------- | ----------------- |
| UniServe  | BF16      | **14.24** | **27.30** | **27.32** | 72 / 72           |
| FastVideo | BF16      | 18.17     | 33.62     | 33.77     | 72 / 72           |
| vLLM-Omni | BF16      | 18.83     | 34.89     | 34.94     | 72 / 72           |
| SGLang    | BF16      | 17.80     | 32.59     | 32.76     | 72 / 72           |
| UniServe  | NVFP4     | **12.15** | **24.16** | **24.27** | 72 / 72           |

| System    | Precision | 5 s / 1K | 5 s / 10K | 10 s / 1K | 10 s / 10K | 15 s / 1K | 15 s / 10K |
| --------- | --------- | -------- | --------- | --------- | ---------- | --------- | ---------- |
| UniServe  | BF16      | **5.62** | **8.80**  | **11.73** | **16.75**  | **20.27** | **27.31**  |
| FastVideo | BF16      | 7.47     | 10.87     | 15.65     | 20.65      | 26.35     | 33.70      |
| vLLM-Omni | BF16      | 7.69     | 11.34     | 15.93     | 21.47      | 26.84     | 34.90      |
| SGLang    | BF16      | 7.07     | 10.35     | 15.23     | 20.43      | 25.44     | 32.59      |
| UniServe  | NVFP4     | **4.49** | **7.48**  | **9.67**  | **14.46**  | **17.32** | **24.18**  |

*End-to-end latency on four GB200 GPUs (*`r1s4`*): quantiles over all 72 requests (top) and median per shape in seconds, 12 requests each (bottom).*

</details>

**Eight GB200 GPUs across two nodes.** With one eight-GPU replica spanning both nodes, **UniServe BF16 has a median latency of 7.51 s, 47% below its four-GPU median, compared with 9.79 s for SGLang and 13.02 s for FastVideo**. SGLang's per-shape medians are 1.24 to 1.35 times UniServe's and FastVideo's 1.52 to 1.88 times.

<details><summary>End-to-end latency on eight GB200 GPUs across two nodes</summary>

| System    | Precision | p50 (s)  | p90 (s)   | p95 (s)   | Valid / attempted |
| --------- | --------- | -------- | --------- | --------- | ----------------- |
| UniServe  | BF16      | **7.51** | **14.09** | **14.12** | 72 / 72           |
| FastVideo | BF16      | 13.02    | 23.58     | 24.39     | 72 / 72           |
| vLLM-Omni | BF16      | N/A†     |           |           |                   |
| SGLang    | BF16      | 9.79     | 17.46     | 17.59     | 72 / 72           |
| UniServe  | NVFP4     | **6.56** | **12.53** | **12.60** | 72 / 72           |

| System    | Precision | 5 s / 1K | 5 s / 10K | 10 s / 1K | 10 s / 10K | 15 s / 1K | 15 s / 10K |
| --------- | --------- | -------- | --------- | --------- | ---------- | --------- | ---------- |
| UniServe  | BF16      | **3.13** | **4.74**  | **6.25**  | **8.76**   | **10.55** | **14.11**  |
| FastVideo | BF16      | 5.37     | 7.19      | 11.38     | 13.54      | 19.86     | 23.97      |
| SGLang    | BF16      | 4.22     | 6.02      | 8.38      | 10.99      | 13.88     | 17.51      |
| UniServe  | NVFP4     | **3.05** | **4.15**  | **5.35**  | **7.62**   | **9.13**  | **12.54**  |

*End-to-end latency on eight GB200 GPUs across two nodes (*`r1s8`*). † A single vLLM-Omni replica cannot span two nodes.*

</details>

**Eight RTX PRO 6000 GPUs.** **UniServe BF16 has a median latency of 35.57 s, compared with 43.38 s for SGLang, 49.76 s for FastVideo and 51.35 s for vLLM-Omni.** The margin grows with duration and prompt length. SGLang's median is 0.96 times UniServe's for 5-second videos with 1K-token prompts, where SGLang is 0.85 s faster, and 1.30 times for 15-second videos with 10K-token prompts; FastVideo's ratio ranges from 1.15 to 1.45 and vLLM-Omni's from 1.12 to 1.55. No system uses an SM100a kernel on this hardware: UniServe runs FlashInfer's SM120 block-sparse attention kernel, and the others run their Triton VSA kernels.

<details><summary>End-to-end latency on eight RTX PRO 6000 GPUs</summary>

| System    | Precision | p50 (s)   | p90 (s)   | p95 (s)   | Valid / attempted |
| --------- | --------- | --------- | --------- | --------- | ----------------- |
| UniServe  | BF16      | **35.57** | **58.72** | **58.75** | 72 / 72           |
| FastVideo | BF16      | 49.76     | 85.03     | 85.18     | 72 / 72           |
| vLLM-Omni | BF16      | 51.35     | 91.13     | 91.24     | 72 / 72           |
| SGLang    | BF16      | 43.38     | 76.49     | 76.53     | 72 / 72           |
| UniServe  | NVFP4     | **32.15** | **53.54** | **53.54** | 72 / 72           |

| System    | Precision | 5 s / 1K  | 5 s / 10K | 10 s / 1K | 10 s / 10K | 15 s / 1K | 15 s / 10K |
| --------- | --------- | --------- | --------- | --------- | ---------- | --------- | ---------- |
| UniServe  | BF16      | 19.12     | **24.51** | **31.39** | **39.80**  | **47.52** | **58.72**  |
| FastVideo | BF16      | 21.97     | 31.31     | 42.53     | 55.87      | 67.95     | 85.12      |
| vLLM-Omni | BF16      | 21.44     | 33.28     | 43.45     | 59.16      | 70.21     | 91.15      |
| SGLang    | BF16      | **18.27** | 27.10     | 37.09     | 49.71      | 60.36     | 76.50      |
| UniServe  | NVFP4     | **17.39** | **22.54** | **28.15** | **36.15**  | **42.61** | **53.54**  |

*End-to-end latency on eight RTX PRO 6000 GPUs (*`r1s8`*). Quantiles use linear interpolation over all 72 request latencies.*

</details>

### Throughput

{{< image src="img/throughput-concurrency.svg" alt="Throughput at each concurrency on every hardware configuration" width="100%" title="Figure 9. Valid videos per second at each concurrency, one 32-request run per point. On four GB200 GPUs, FastVideo's and vLLM-Omni's lines overlap. The folded tables below list every value with its p50 and p95 latency." >}}
On four GB200 GPUs with one replica, every system's throughput is essentially flat across concurrency: higher concurrency adds queueing latency without adding throughput. **At each system's best concurrency, UniServe BF16's throughput is 24.0% above SGLang's, 32.6% above vLLM-Omni's and 32.7% above FastVideo's; all 640 responses pass media validation.** With two replicas on eight GPUs, C=1 leaves one replica idle and both are busy from C=2. **On two GB200 nodes, UniServe BF16 reaches 0.1283 videos/s, 14.9% above SGLang's best (0.1117 videos/s at C=8), 20.3% above FastVideo's (0.1067 videos/s at C=4) and 22.0% above vLLM-Omni's (0.1052 videos/s at C=8).** On the RTX PRO 6000, **UniServe BF16 reaches 0.0389 videos/s, 20.8% above SGLang's best (0.0322 videos/s at C=4), 39.9% above FastVideo's (0.0278 videos/s at C=4) and 45.4% above vLLM-Omni's (0.0267 videos/s at C=4)**. Each cell below gives throughput in valid videos per second, followed by p50 / p95 end-to-end latency in seconds, for one 32-request run.

<details><summary>Throughput and latency under load on four GB200 GPUs</summary>

| System    | Precision | C=1                      | C=2                      | C=4                      | C=8                        |
| --------- | --------- | ------------------------ | ------------------------ | ------------------------ | -------------------------- |
| UniServe  | BF16      | **0.0702 (11.6 / 26.6)** | **0.0708 (30.4 / 45.6)** | **0.0705 (54.0 / 74.8)** | **0.0707 (104.8 / 140.5)** |
| FastVideo | BF16      | 0.0531 (16.1 / 34.0)     | 0.0532 (37.4 / 55.1)     | 0.0534 (74.2 / 90.9)     | 0.0531 (141.0 / 173.7)     |
| vLLM-Omni | BF16      | 0.0533 (15.8 / 34.2)     | 0.0534 (39.6 / 55.3)     | 0.0534 (74.2 / 97.9)     | 0.0532 (143.5 / 170.9)     |
| SGLang    | BF16      | 0.0567 (15.0 / 31.8)     | 0.0571 (36.8 / 52.0)     | 0.0571 (69.5 / 89.6)     | 0.0571 (133.7 / 160.3)     |
| UniServe  | NVFP4     | **0.0819 (9.5 / 23.7)**  | **0.0824 (24.4 / 40.0)** | **0.0824 (46.9 / 64.0)** | **0.0823 (90.3 / 120.9)**  |

*Throughput and latency under load on four GB200 GPUs (*`r1s4`*).*

</details>

<details><summary>Throughput and latency under load on eight GB200 GPUs across two nodes</summary>

| System    | Precision | C=1                      | C=2                      | C=4                      | C=8                      | C=16                       |
| --------- | --------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | -------------------------- |
| UniServe  | BF16      | **0.0692 (11.7 / 27.3)** | **0.1223 (12.4 / 34.0)** | **0.1251 (29.4 / 49.0)** | **0.1283 (53.1 / 78.6)** | **0.1186 (118.4 / 151.7)** |
| FastVideo | BF16      | 0.0506 (20.6 / 33.8)     | 0.1039 (15.7 / 33.7)     | 0.1067 (36.2 / 54.3)     | 0.1042 (68.7 / 101.7)    | 0.0958 (114.3 / 182.2)     |
| vLLM-Omni | BF16      | 0.0529 (16.0 / 34.6)     | 0.1021 (15.9 / 34.3)     | 0.1043 (37.0 / 55.3)     | 0.1052 (70.9 / 111.7)    | 0.1023 (136.8 / 167.0)     |
| SGLang    | BF16      | 0.0564 (15.3 / 32.6)     | 0.1079 (15.2 / 32.7)     | 0.1111 (34.7 / 51.4)     | 0.1117 (63.0 / 93.7)     | 0.1002 (106.5 / 173.5)     |
| UniServe  | NVFP4     | **0.0804 (9.7 / 24.3)**  | **0.1420 (9.7 / 34.6)**  | **0.1437 (25.2 / 42.2)** | **0.1467 (46.3 / 68.5)** | **0.1480 (92.3 / 119.7)**  |

*Throughput and latency under load on eight GB200 GPUs across two nodes (*`r2s4`*). UniServe's rows on this configuration were measured with a later revision that routes each decoded video unit only to the host that encodes it; the earlier revision also published every unit to the other node.*

</details>

<details><summary>Throughput and latency under load on eight RTX PRO 6000 GPUs</summary>

| System    | Precision | C=1                      | C=2                      | C=4                       | C=8                        | C=16                       |
| --------- | --------- | ------------------------ | ------------------------ | ------------------------- | -------------------------- | -------------------------- |
| UniServe  | BF16      | **0.0207 (39.7 / 86.2)** | **0.0389 (39.7 / 88.2)** | **0.0389 (97.8 / 131.3)** | **0.0362 (186.0 / 274.9)** | **0.0359 (379.2 / 499.2)** |
| FastVideo | BF16      | 0.0138 (61.9 / 129.2)    | 0.0271 (58.7 / 129.2)    | 0.0278 (140.3 / 210.8)    | 0.0269 (266.7 / 415.7)     | 0.0270 (413.2 / 706.0)     |
| vLLM-Omni | BF16      | 0.0135 (58.9 / 139.5)    | 0.0260 (59.0 / 139.5)    | 0.0267 (144.9 / 225.4)    | 0.0263 (273.8 / 419.9)     | 0.0260 (475.8 / 794.6)     |
| SGLang    | BF16      | 0.0163 (50.0 / 114.3)    | 0.0313 (49.9 / 114.6)    | 0.0322 (120.7 / 185.1)    | 0.0317 (227.6 / 345.4)     | 0.0289 (368.1 / 616.3)     |
| UniServe  | NVFP4     | **0.0244 (32.6 / 74.8)** | **0.0442 (33.7 / 75.0)** | **0.0461 (80.5 / 110.8)** | **0.0412 (170.2 / 264.7)** | **0.0434 (320.5 / 399.2)** |

*Throughput and latency under load on eight RTX PRO 6000 GPUs (*`r2s4`*).*

</details>

### NVFP4 against BF16

| Hardware         | BF16 p50 (s) | NVFP4 p50 (s) | Latency change | BF16 best (videos/s) | NVFP4 best (videos/s) | Throughput change |
| ---------------- | ------------ | ------------- | -------------- | -------------------- | --------------------- | ----------------- |
| 4 x GB200        | 14.24        | 12.15         | −14.7%         | 0.0708               | 0.0824                | +16.4%            |
| 8 x GB200        | 7.51         | 6.56          | −12.6%         | 0.1283               | 0.1480                | +15.4%            |
| 8 x RTX PRO 6000 | 35.57        | 32.15         | −9.6%          | 0.0389               | 0.0461                | +18.6%            |

*UniServe BF16 and NVFP4 in the same layouts. NVFP4 is a separate precision tier; see [NVFP4](#nvfp4).*

## Acknowledgments

FastVideo FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). We thank the MiniMax team for releasing its weights and code.

We thank Guan Luo, Qi Wang, Ryan McCormick, and the rest of the [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) team for their support in integrating Dynamo with UniServe, bringing UniServe closer to a production-grade serving engine.

We thank [Nuva Lab](https://nuvalab.ai/) and [Reactor](https://www.reactor.inc/) for their collaboration, feedback and insights during UniServe's development, and Nuva Lab for access to the production hardware used in our benchmarks.

We also thank MiniMax for releasing H3-Base, and the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/) and [MBZUAI](https://mbzuai.ac.ae/) for their continued sponsorship and support of FastVideo.
