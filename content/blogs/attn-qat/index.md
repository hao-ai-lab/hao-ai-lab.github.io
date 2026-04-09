+++
title = "Attn-QAT: Making 4-Bit Attention Actually Work"
date = 2026-04-06T12:00:00-07:00
authors = ["Peiyuan Zhang", "Matthew Noto", "Wenxuan Tan", "Chengquan Jiang", "Will Lin", "Wei Zhou", "Hao Zhang"]
author = "Peiyuan Zhang*, Matthew Noto*, Wenxuan Tan*, Chengquan Jiang, Will Lin, Wei Zhou, Hao Zhang"
ShowReadingTime = true
draft = false
math = true
contentClass = "post-content-justified"
description = ""
summary = "Attn-QAT is the first systematic study of 4-bit quantization-aware training for attention, recovering FP4 attention quality without inference-time outlier mitigation while also enabling faster kernels."
+++

{{< socialBadges arxiv-index="2603.00040" github="hao-ai-lab/FastVideo" huggingface="https://huggingface.co/FastVideo/14B_qat_400/tree/main">}}

**TL;DR:** FP4 hardware is finally here, and FP4 linear layers are already being used in production. However, FP4 attention still causes significant quality degradation, preventing true end-to-end FP4 serving and limiting full hardware utilization. In this work, we present **Attn-QAT**, the first systematic study of 4-bit quantization-aware training for attention. We identify two key principles for stable FP4 attention QAT: (1) matching low-precision recomputation of attention probabilities in the backward pass and (2) resolving implicit precision assumptions in FlashAttention's gradient calculation. Across video diffusion models and language models, Attn-QAT **recovers the quality drop** of 4-bit attention without the extra outlier-mitigation heuristics used by prior FP4 attention methods, while also delivering a **1.1x--1.5x** speedup over SageAttention3 on an RTX 5090 and up to a **1.39x** speedup over FlashAttention-4 on a B200. 

## 4-bit attention remains unsolved

Native FP4 tensor core support on new NVIDIA GPUs makes 4-bit computation increasingly attractive. In principle, FP4 offers lower memory traffic and up to 4x higher arithmetic intensity than higher-precision BF16 execution. But in practice, attention is much harder to quantize than linear layers. The reason is twofold. First, FP4 has an extremely small dynamic range and only a tiny set of representable values. Second, attention activations are [heavy-tailed](https://haoailab.com/blogs/sta/) and contain [more outliers](https://arxiv.org/pdf/2411.10958) than standard matrix multiplies. These two properties make attention much more sensitive to quantization errors.

Naively applying FP4 attention layers results in severely degraded video quality. In an attempt to mitigate this, prior training-free FP4 attention methods such as [SageAttention3](https://arxiv.org/pdf/2505.11594) use specialized outlier-mitigation techniques such as [Q/K smoothing and two-level quantization of attention probabilities](https://arxiv.org/pdf/2505.11594#page=4). However, even with these heuristics, we find that there is still video quality degradation on Wan 2.1-14B compared to using standard BF16 attention.


<div class="video-embed">
{{<youtube eHW78v2H4bU>}}
</div>


Instead of devising more sophisticated tricks to reduce quantization error in a training-free manner, we take a different approach: **employing quantization-aware training (QAT) for attention**, which enables models to adapt to quantization errors during training and thus preserve model quality. The goal of this work is simple: make FP4 attention work without any outlier-mitigation techniques. In the context of video generation, this means making FP4 attention produce videos that are indistinguishable in quality from those of BF16 attention at inference time.

{{< figure src="img/inference_algo.png" alt="attn-qat inference" width="50%" align="center" >}}

## The two fixes that make Attn-QAT work

For linear layers, the [QAT recipe](https://pytorch.org/blog/quantization-aware-training/) is well known: simulate low-precision execution in the forward pass using fake quantization, keep the backward pass in higher precision (typically BF16), and train through the quantization noise. That works well for GEMMs. It does **not** work for attention.

The reason is that FlashAttention-style kernels are **heavily fused operators** whose backward pass relies on recomputation of attention scores and an algebraic identity that quietly assumes that the forward and backward passes share the same numerical behavior.

Attn-QAT stabilizes 4-bit attention by enforcing two forms of precision consistency.

### 1. Store a high-precision attention output for gradient computation


We first clarify notation. Let $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ denote the query, key, and value matrices, and define

\[
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d} \in \mathbb{R}^{N\times N}, \quad
\mathbf{P} = \mathrm{softmax}(\mathbf{S}) \in \mathbb{R}^{N\times N}, \quad
\mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N\times d}.
\]

We use $(\cdot)^F$ to denote fake quantized (FP4-simulated) tensors, e.g., $\mathbf{P}^F = \phi^{-1}(\phi(\mathbf{P}))$.

In FlashAttention, the backward pass relies on a memory-efficient formulation of the softmax gradient. For a single row $i$,

\[
\mathbf{P}_i = \mathrm{softmax}(\mathbf{S}_i) \in \mathbb{R}^d,
\]

the gradient can be written as

\[
\begin{aligned}
\mathbf{dS}_i
&= \left(\mathrm{diag}(\mathbf{P}_i) - \mathbf{P}_i \mathbf{P}_i^\top\right)\mathbf{dP}_i \\
&= \mathbf{P}_i \odot \mathbf{dP}_i - (\mathbf{P}_i^\top \mathbf{dP}_i)\mathbf{P}_i.
\end{aligned}
\]

The key difficulty is the scalar term $\mathbf{P}_i^\top \mathbf{dP}_i$, which naively requires access to the full attention row and thus $ O(n^2) $ memory in the sequence length for the full matrix $\mathbf{P}$.

FlashAttention keeps memory $O(n)$ in the sequence length by rewriting the scalar as

\[
\begin{aligned}
\mathbf{P}_i^\top \mathbf{dP}_i
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dP}_{ij} \\
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dO}_i^\top \mathbf{V}_j \\
&= \mathbf{dO}_i^\top \sum_j \mathbf{P}_{ij} \mathbf{V}_j \\
&= \mathbf{dO}_i^\top \mathbf{O}_i,
\end{aligned}
\]

which reduces the computation to a dot product with the attention output $\mathbf{O}_i$.

This identity implicitly assumes that the forward pass computes

\[
\mathbf{O}_i = \sum_j \mathbf{P}_{ij} \mathbf{V}_j.
\]

However, under Attn-QAT, the forward pass ([Algorithm 2](#training-algo)) instead uses **fake quantized probabilities and values**:

\[
\mathbf{O}_i = \sum_j \mathbf{P}_{ij}^{F} \mathbf{V}_j^{F}.
\]

This introduces a precision mismatch: a naive backward pass depends on the high-precision $\mathbf{P}$ matrix, while the Attn-QAT forward pass uses $\mathbf{P}^F$. As a result,

\[
\mathbf{dO}_i^\top \mathbf{O}_i \neq \mathbf{P}_i^\top \mathbf{dP}_i,
\]

which leads to incorrect gradients and unstable training.

<a id="grad-norm"></a>
{{< figure src="img/grad_norm.png" alt="grad norm" width="50%" align="center" >}}

To resolve this, we compute an additional auxiliary output during the forward pass:

\[
\mathbf{O}_i' = \sum_j \mathbf{P}_{ij} \mathbf{V}_j^{F}.
\]

Here, $\mathbf{P}$ remains in high precision (FP32 row-wise softmax over $\mathbf{S}$), while $\mathbf{V}^F$ is still fake quantized (and stored in BF16 precision). This adds only a small amount (25% increase) of extra storage, while still preventing the need to materialize the full attention matrix.

In the backward pass, we then replace the scalar term with

\[
\mathbf{P}_i^\top \mathbf{dP}_i = \mathbf{dO}_i^\top \mathbf{O}_i',
\]

which restores the exact identity:

\[
\begin{aligned}
\mathbf{P}_i^\top \mathbf{dP}_i
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dP}_{ij} \\
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dO}_i^\top \mathbf{V}_j^{F} \\
&= \mathbf{dO}_i^\top \sum_j \mathbf{P}_{ij} \mathbf{V}_j^{F} \\
&= \mathbf{dO}_i^\top \mathbf{O}_i'.
\end{aligned}
\]

Intuitively, $\mathbf{O}$ is the low-precision output used by the model, while $\mathbf{O}'$ is a minimal high-precision correction that ensures the backward pass remains mathematically consistent. Without $\mathbf{O}'$, the gradient computation silently assumes a different forward computation than what actually occurred, which is the root cause of instability in naive FP4 QAT for attention.

This small modification preserves the fully low-precision forward path while restoring correctness in the backward pass, **eliminating the need for heuristic outlier-mitigation techniques**.


### 2. Recompute attention probabilities in the same low precision used in the forward pass

In FlashAttention, the full matrix $\mathbf{P}$ is not stored. It is recomputed during the backward pass from the saved [logsumexp statistics](https://arxiv.org/pdf/2307.08691#page=6). Under QAT, this recomputation **must match the low-precision forward pass**. Attn-QAT therefore fake-quantizes the recomputed attention probabilities in the backward pass, so gradients are computed with respect to the same quantized activations seen in the forward pass. Empirically, we found that this had the effect of [stabilizing training dynamics](#grad-norm). 


<a id="training-algo"></a>
{{< figure src="img/training_algo.png" alt="attn-qat training" width="100%" align="center" >}}

## Results: Quality is Recovered

We evaluated the efficacy of Attn-QAT for both video diffusion models and language models. 

For the **non-cherry picked** example videos below (generated by Wan-2.1-14B), we see that with Attn-QAT, FP4 attention **produces videos comparable to BF16 attention**, whereas SageAttention3 produces videos with artifacts.


<div class="video-embed">
{{<youtube 5_19ypV3E3o>}}
</div>


We also evaluate Attn-QAT on LLMs in two settings: continued pretraining and supervised fine-tuning.

<a id="llm-benchmarks"></a>
{{< figure src="img/LLM_benchmarks.png" alt="attn-qat training" width="100%" align="center" >}}

For continued pretraining, Attn-QAT recovers most of the quality loss caused by FP4 attention on Qwen3-14B and partially recovers it on Llama 3.1-70B. We hypothesize that the remaining gap on the 70B model is likely due to limited training budget rather than a failure of the method itself.

For supervised fine-tuning, Attn-QAT can be used as a **drop-in replacement** for BF16 attention. On Qwen3-14B, it achieves nearly identical downstream benchmark performance to BF16 attention. On Llama 3.1-70B, it remains close with a small gap. This is an important practical result: Attn-QAT is not only a specialized recovery stage for quantization, but can also be **integrated directly into standard fine-tuning pipelines**.


## Faster Inference on an RTX 5090

Because Attn-QAT eliminates the need for extra smoothing and two-level quantization overhead used by SageAttention3, this results in faster inference. On an RTX 5090, we're able to achieve approximately **1.1x-1.5x** higher throughput than SageAttention3. The key reason is straightforward: by removing extra preprocessing for $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{P}$, the kernel becomes lighter while preserving quality through training rather than inference heuristics.

{{< figure src="img/5090_speedup.png" alt="5090 speedup" width="60%" align="center" >}}

## B200/B300 FP4 attention kernel

To make Attn-QAT practical on Blackwell GPUs, we also developed [FlashAttention-4 FP4](https://github.com/hao-ai-lab/flash-attention-fp4), an NVFP4-quantized FA4 kernel implemented in CuTeDSL. It reaches up to 1801 TFLOPS and 1.39x speedup over FA4. This section explains the hardware constraints that shaped the kernel.

### Block-scaled MMAs and TMEM

A block-scaled MMA (matrix-multiply accumulate) is the following operation:

\[
\mathbf{D} = (\mathbf{A} \cdot \mathbf{s}_A) @ (\mathbf{B} \cdot \mathbf{s}_B) + \mathbf{C}.
\]

{{< figure src="img/BS.png" alt="B200 kernel" width="70%" align="center" caption="<span style=\"display:block; text-align:center;\">Source: [NVIDIA PTX Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-block-scaling)</span>" >}}

Block scaling compensates for the limited dynamic range of FP4/FP8 formats. The scale factors map higher-precision weights or activations into a more uniform range before quantization. The main design choice is granularity: one scale per element is expensive, while one scale for the whole matrix is too coarse. Blackwell Tensor Cores support an intermediate scheme in which each row or column is split into 16- or 32-element chunks along the reduction dimension, with one scale per chunk.

This matters for attention because Blackwell can consume both the quantized values and their scales directly in hardware through `tcgen05.mma.cta_group.kind.block_scale`. In practice, `tcgen05.mma` supports MXFP8/MXFP4 and NVFP4 block-scaled GEMMs, with group sizes of 32 for the MX formats and 16 for NVFP4. 

Unlike the `wgmma` instruction on Hopper GPUs, where A/B live in SMEM/registers and the outputs stay in registers, Blackwell introduces **Tensor Memory (TMEM)** to hold MMA outputs. This reduces register pressure for larger tiles, but it also adds extra movement in attention kernels: outputs must be copied from TMEM to registers for softmax and then written back to TMEM. TMEM consists of 128 lanes across four warps times 512 columns gives **64K 32-bit cells**, so it quickly becomes a real resource constraint.

{{< figure src="img/TMEM.png" alt="SM" width="100%" align="center" caption="<span style=\"display:block; text-align:center;\">Source: [Colfax Research Tutorial On Writing Blackwell GEMM Kernels](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)</span>" >}}

### Softmax becomes the bottleneck
| Spec | A100 (SXM4) | H100 (SXM5) | B200 (HGX) | B300 / GB300 | R200 |
|---|---|---|---|---|---|
| **Architecture** | Ampere | Hopper | Blackwell | Blackwell Ultra | Rubin |
| **Year** | 2020 | 2022 | 2024 | 2025 | 2026 |
| **Die Config** | 1 die, 826 mm² | 1 die, 814 mm² | 2× ~800 mm² ≈ 1,600 mm² | 2× ~800 mm² ≈ 1,600 mm² | 2× near-reticle + 2 I/O dies |
| **Transistors** | 54.2B | 80B | 208B | 208B | 336B |
| **TDP** | 400W | 700W | 1,000W | 1,100W (HGX) / 1,400W (GB300) | ~1,800W |
| **SMs (enabled)** | 108 | 132 | 148 | 160 | 224 |
| **CUDA Cores (FP32)** | 6,912 | 16,896 | 18,944 | 20,480 | TBD |
| **BF16 Tensor TFLOPS (dense)** | 312 | 989 | 2,250 | ~2,500 | TBD |
| **FP8 Tensor TFLOPS (dense)** | — | 1,979 | 4,500 | 5,000 | TBD |
| **FP4 Tensor PFLOPS (dense)** | — | — | 9.0 | 14–15 | TBD |
| **Registers/SM** | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit | 64K × 32-bit | 64K x 32-bit (est.) |
| **TMEM/SM** | — | — | 256 KB | 256 KB | 256 KB+ |
| **Shared Mem/SM (max)** | 164 KB | 228 KB | 228 KB | 228 KB | TBD |
| **MUFU.EX2 ops/clk/SM** | **16** | **16** | **16** | **32** | **32 (fp32)/64 (fp16)** | 

Recent NVIDIA gains come mostly from **larger tensor cores**, which primarily accelerate GEMMs. As the table shows, BF16/FP8 Tensor Core throughput jumps by ~2.27x from H100 to B200, while CUDA core throughput increases by only 1.1x and MUFU.EX2 throughput does not improve. Since MUFU.EX2 is used to compute `exp2` in softmax, softmax becomes an increasingly important bottleneck in attention kernels, alongside the GEMMs themselves, as discussed in [FlashAttention-4](https://arxiv.org/pdf/2603.05451).

FA4 addresses this with **warp specialization**, overlapping MMA and softmax work across warp groups (WGs).

{{< figure src="img/FA4.png" alt="B200 kernel" width="100%" align="center" caption="<span style=\"display:block; text-align:center;\">Source: [Colfax Research FlashAttention-4 Blog Post](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)</span>" >}}

The overlap is still imperfect because of pipeline warmup and bookkeeping overheads such as address computation, MMA issue, row-max updates, and cross-WG copies. FA4 also uses a [polynomial exp2 approximation](https://arxiv.org/pdf/2603.05451#page=8) to reduce softmax cost, but higher-degree polynomials increase register use and CUDA core pressure, so this optimization only applies to 10%-25% of the softmax scores.

This imbalance shows up most clearly in how Blackwell manages TMEM. Once softmax and MMA are both competing for a tightly constrained pipeline, the exact placement of intermediate tensors and scale factors starts to matter.

### TMEM overlap schedule
The scale factors still need to travel GMEM -> SMEM (via TMA) -> TMEM and then be duplicated across four warps in a WG via `tcgen05.cp` multicast. At tile size `m128n128`, FA4 already uses all available TMEM: S1 and S2 hold the $\mathbf{Q}\mathbf{K}$ outputs, and O1/O2 use the remaining columns.

To fit the scale factors without stalling the pipeline, we reuse S2 for `sfqk 1` and S1 for `sfqk 2`, adding barriers only where sequential execution is not guaranteed. We considered moving P to SMEM to free TMEM, but rejected that approach because the available `ldmatrix` shapes make the copy path too restrictive.

{{< figure src="img/B200_plot.png" alt="B200 kernel" width="100%" align="center" >}}

A TMEM-overlapped schedule is required because B200 kernels are often TMEM-bound, which limits effective MMA pipeline depth. We also found that quantizing $ \mathbf{P} $ does not help in the same way as quantizing $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ because it adds too much pressure to the softmax warps. Interleaving quantization with register-to-TMEM copies does not fully relieve that pressure.

### Results

Even with careful scheduling, quantized PV did not speed things up because softmax remains the dominant bottleneck. Computing group-wise scale factors and `cvt.rn.satfinite` quantization instructions adds enough overhead to erase the gain, so on B200 we use NVFP4 QK and BF16 PV. This reaches 1801 TFLOPS and 1.39x speedup over FA4.

That number is still a lower bound: we have not yet tested NVFP4 QK + FP8 PV, which removes the group-quant overhead in the softmax WG. On B300 and Rubin, faster exp units should make quantized PV more attractive.

 | Config                         | FP4 (ms) | FP4 TFLOPS | BF16 (ms) | BF16 TFLOPS | Speedup |
|--------------------------------|----------|------------|-----------|-------------|---------|
| b=1 s=256 h=16 d=128           | 0.015    | 37         | 0.015     | 35          | 1.01x   |
| b=1 s=1024 h=16 d=128          | 0.023    | 379        | 0.025     | 338         | 1.12x   |
| b=4 s=4096 h=16 d=128          | 0.336    | 1637       | 0.389     | 1413        | 1.16x   |
| b=4 s=8192 h=16 d=128          | 1.259    | 1747       | 1.511     | 1455        | 1.20x   |
| b=2 s=16384 h=16 d=128         | 2.467    | 1782       | 3.003     | 1464        | 1.22x   |
| b=1 s=32768 h=16 d=128         | 4.884    | 1801       | 6.771     | 1299        | 1.39x   |
| b=4 s=4096 h=32 d=128          | 0.655    | 1678       | 0.775     | 1418        | 1.18x   |
| b=4 s=8192 h=32 d=128          | 2.501    | 1759       | 3.027     | 1453        | 1.21x   |
| b=1 s=4096 h=12 d=128          | 0.104    | 986        | 0.117     | 882         | 1.12x   |
| b=1 s=32768 h=12 d=128         | 3.856    | 1711       | 5.056     | 1305        | 1.31x   |
| b=1 s=4096 h=24 d=128          | 0.152    | 1352       | 0.172     | 1198        | 1.13x   |
| b=1 s=32768 h=24 d=128         | 7.551    | 1747       | 10.061    | 1311        | 1.33x   |
| b=1 s=32768 h=24 d=64          | 7.170    | 920        | 7.284     | 906         | 1.02x   |

### Precision Results
At the time of writing, the FA4 repo had received an [FP8 non-block-scaled PR](https://github.com/Dao-AILab/flash-attention/pull/2109), so we compare kernel-level precision below. Even with NVFP4, our kernel achieves roughly 2-2.5x lower max absolute error with group size 16 than FA4's per-head grouping (e.g., 128).

| batch | seqlen | nheads | hdim | FP4 max | FP4 mean | FP8 max | FP8 mean |
|-------|--------|--------|------|---------|----------|---------|----------|
| 1 | 256 | 16 | 128 | 0.158 | 0.01058 | 0.352 | 0.00591 |
| 1 | 512 | 16 | 128 | 0.148 | 0.00771 | 0.379 | 0.00555 |
| 1 | 1024 | 16 | 128 | 0.070 | 0.00553 | 0.260 | 0.00463 |
| 1 | 2048 | 16 | 128 | 0.078 | 0.00395 | 0.184 | 0.00372 |
| 2 | 4096 | 16 | 128 | 0.068 | 0.00280 | 0.136 | 0.00286 |
| 1 | 4096 | 24 | 128 | 0.043 | 0.00280 | 0.132 | 0.00284 |
| 1 | 8192 | 24 | 128 | 0.033 | 0.00199 | 0.075 | 0.00213 |
| 1 | 16384 | 16 | 128 | 0.025 | 0.00141 | 0.044 | 0.00157 |
| 1 | 32768 | 16 | 128 | **0.011** | **0.00100** | 0.022 | 0.00114 |
| 1 | 32768 | 24 | 128 | **0.016** | **0.00100** | 0.031 | 0.00114 |

### Agent-assisted Kernel Development
During debugging, we found LLMs like Claude useful even for CuTeDSL and low-level PTX. It surfaced an obscure uninitialized register bug in FA4, which had already been fixed in a [large commit](https://github.com/Dao-AILab/flash-attention/commit/c79976218fb71f282f76cb959a5aad48a2d23e86) before we found it; in practice, we estimate that LLMs saved us about 1-2 weeks and were especially helpful for SASS inspection, instruction dependency analysis, and structured performance debugging.


## What this paper really changes

Prior to this paper, attention quantization was mostly treated as an inference problem: improve smoothing, calibration, or other post-hoc fixes. Attn-QAT argues that this view is incomplete. Since modern attention kernels are fused and precision-sensitive, **training methods and low-bit kernels must be co-designed**.

Despite NVIDIA’s headline FP4/FP8 (MMA) TFLOPS coming from scaling pure GEMMs, attention usually dominates wall-clock time in long-context agentic serving and video generation. Across Hopper -> Blackwell -> Rubin, algorithms and hardware are becoming increasingly coupled as hardware headroom shrinks.

Moving forward, we are excited to try **hardware-specific mixed-precision QAT recipes** and combine distillation and sparse attention with FP4.

For more details, see [our paper](https://arxiv.org/abs/2603.00040). All code is available in [FastVideo](https://github.com/hao-ai-lab/FastVideo), our unified framework for video diffusion post-training and [real-time inference](https://haoailab.com/blogs/fastvideo_realtime_1080p/).  

## Citation

```bibtex
@misc{zhang2026attnqat4bitattentionquantizationaware,
  title={Attn-QAT: 4-Bit Attention With Quantization-Aware Training},
  author={Peiyuan Zhang and Matthew Noto and Wenxuan Tan and Chengquan Jiang and Will Lin and Wei Zhou and Hao Zhang},
  year={2026},
  eprint={2603.00040},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.00040}
}
```
