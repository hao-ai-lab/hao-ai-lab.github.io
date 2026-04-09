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
\mathbf{P}\_i = \mathrm{softmax}(\mathbf{S}\_i) \in \mathbb{R}^d,
\]

the gradient can be written as

\[
\begin{aligned}
\mathbf{dS}\_i
&= \left(\mathrm{diag}(\mathbf{P}\_i) - \mathbf{P}\_i \mathbf{P}\_i^\top\right)\mathbf{dP}\_i \\
&= \mathbf{P}\_i \odot \mathbf{dP}\_i - (\mathbf{P}\_i^\top \mathbf{dP}\_i)\mathbf{P}\_i.
\end{aligned}
\]

The key difficulty is the scalar term $\mathbf{P}_i^\top \mathbf{dP}_i$, which naively requires access to the full attention row and thus $ O(n^2) $ memory in the sequence length for the full matrix $\mathbf{P}$.

FlashAttention keeps memory $O(n)$ in the sequence length by rewriting the scalar as

\[
\begin{aligned}
\mathbf{P}_i^\top \mathbf{dP}\_i
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dP}_{ij} \\
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dO}_i^\top \mathbf{V}\_j \\
&= \mathbf{dO}\_i^\top \sum_j \mathbf{P}_{ij} \mathbf{V}\_j \\
&= \mathbf{dO}\_i^\top \mathbf{O}\_i,
\end{aligned}
\]

which reduces the computation to a dot product with the attention output $\mathbf{O}_i$.

This identity implicitly assumes that the forward pass computes

\[
\mathbf{O}_i = \sum_j \mathbf{P}_{ij} \mathbf{V}\_j.
\]

However, under Attn-QAT, the forward pass ([Algorithm 2](#training-algo)) instead uses **fake quantized probabilities and values**:

\[
\mathbf{O}_i = \sum_j \mathbf{P}_{ij}^{F} \mathbf{V}\_j^{F}.
\]

This introduces a precision mismatch: a naive backward pass depends on the high-precision $\mathbf{P}$ matrix, while the Attn-QAT forward pass uses $\mathbf{P}^F$. As a result,

\[
\mathbf{dO}\_i^\top \mathbf{O}\_i \neq \mathbf{P}\_i^\top \mathbf{dP}\_i,
\]

which leads to incorrect gradients and unstable training.

<a id="grad-norm"></a>
{{< figure src="img/grad_norm.png" alt="grad norm" width="50%" align="center" >}}

To resolve this, we compute an additional auxiliary output during the forward pass:

\[
\mathbf{O}_i' = \sum_j \mathbf{P}_{ij} \mathbf{V}\_j^{F}.
\]

Here, $\mathbf{P}$ remains in high precision (FP32 row-wise softmax over $\mathbf{S}$), while $\mathbf{V}^F$ is still fake quantized (and stored in BF16 precision). This adds only a small amount (25% increase) of extra storage, while still preventing the need to materialize the full attention matrix.

In the backward pass, we then replace the scalar term with

\[
\mathbf{P}\_i^\top \mathbf{dP}\_i = \mathbf{dO}\_i^\top \mathbf{O}\_i',
\]

which restores the exact identity:

\[
\begin{aligned}
\mathbf{P}_i^\top \mathbf{dP}\_i
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dP}_{ij} \\
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dO}_i^\top \mathbf{V}\_j^{F} \\
&= \mathbf{dO}\_i^\top \sum_j \mathbf{P}_{ij} \mathbf{V}\_j^{F} \\
&= \mathbf{dO}\_i^\top \mathbf{O}\_i'.
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

For the randomly-selected example videos below (generated by Wan-2.1-14B), we see that with Attn-QAT, FP4 attention **produces videos comparable to BF16 attention**, whereas SageAttention3 produces videos with artifacts.

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

## For the GPU Enjoyers: B200/B300 FP4 Attention Kernel

To make Attn-QAT **usable on data-center grade Blackwell GPUs (e.g. B200s/B300s)**, we also developed [FlashAttention-4 FP4](https://github.com/hao-ai-lab/flash-attention-fp4), an NVFP4-quantized FA4 kernel implemented in CuTeDSL, achieving up to a 1.39x speedup over FA4 and 1801 TFLOPS. The rest of this section explains hardware constraints, the Blackwell programming model and how they influenced kernel design and development.

### Block-scaled MMAs and TMEM

Let $\mathbf{A}$ and $\mathbf{B}$ be quantized (e.g. NVFP4) matrices, $\mathbf{s}_{A}$ and $\mathbf{s}_{B}$ be the dequantizing scale factors for the two matrices, and $\mathbf{D}$ be the BF16 output matrix. A block-scaled MMA (matrix-multiply accumulate) is the following operation:

\[
\mathbf{D} = (\mathbf{A} \cdot \mathbf{s}\_A) @ (\mathbf{B} \cdot \mathbf{s}\_B) + \mathbf{C}.
\]

{{< figure src="img/BS.png" alt="B200 kernel" width="70%" align="center" caption="<span style=\"display:block; text-align:center;\">Source: [NVIDIA PTX Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-block-scaling)</span>" >}}

FP4 and FP8 have limited dynamic range, so directly quantizing a tensor can introduce large errors when its values vary too much. Block scaling reduces this problem by assigning a scale to each group of values before quantization, so that each group can be mapped into a range that the low-precision format represents more accurately.

Equivalently, during the MMA, the hardware multiplies the quantized values by the corresponding dequantization scale; with the dequantization scale being the element-wise inverse of the scale factor used to quantize the matrix to low-precision.

The main design choice for block-scaled quantization is how many values share one scale: one scale per element is too expensive, while one scale for the whole matrix is often too coarse. Blackwell Tensor Cores support a middle ground, where each row or column is split into 16- or 32-element chunks along the reduction dimension, and each chunk gets its own scale.

More specifically, on Blackwell GPUs, the `tcgen05.mma` instruction bakes in MXFP8/MXFP4 and NVFP4 GEMMs into the hardware with these finer group sizes (32 and 16, respectively), providing better precision and freeing registers. It also enables FP8/FP6/FP4 GEMM without block scales via `tcgen05.mma.cta_group.kind`.

{{< figure src="img/SM.png" alt="SM" width="70%" align="center" >}}

Let A and B be the inputs for an MMA. Unlike the `wgmma` instruction on Hopper GPUs, where A/B live in SMEM/registers and the outputs stay in registers, Blackwell introduces **Tensor Memory (TMEM)** to hold MMA outputs. This reduces register pressure for larger tiles, but it also adds extra movement in attention kernels: outputs must be copied from TMEM to registers for softmax and then written back to TMEM. Note that TMEM consists of 128 lanes (across four warps) x 512 columns, which gives **64K 32-bit cells**. We will see how this quickly becomes a limiting resource alongside registers.

{{< figure src="img/TMEM.png" alt="SM" width="70%" align="center" caption="<span style=\"display:block; text-align:center;\">Source: [Colfax Research Tutorial On Writing Blackwell GEMM Kernels](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)</span>" >}}

### Overbloated Tensor Cores $\rightarrow$ Softmax Becomes A Bottleneck

| Spec                           | A100 (SXM4)    | H100 (SXM5)    | B200 (HGX)              | B300 / GB300                  | R200                         |
| ------------------------------ | -------------- | -------------- | ----------------------- | ----------------------------- | ---------------------------- |
| **Architecture**               | Ampere         | Hopper         | Blackwell               | Blackwell Ultra               | Rubin                        |
| **Year**                       | 2020           | 2022           | 2024                    | 2025                          | 2026                         |
| **Die Config**                 | 1 die, 826 mm² | 1 die, 814 mm² | 2× ~800 mm² ≈ 1,600 mm² | 2× ~800 mm² ≈ 1,600 mm²       | 2× near-reticle + 2 I/O dies |
| **Transistors**                | 54.2B          | 80B            | 208B                    | 208B                          | 336B                         |
| **TDP**                        | 400W           | 700W           | 1,000W                  | 1,100W (HGX) / 1,400W (GB300) | ~1,800W                      |
| **SMs (enabled)**              | 108            | 132            | 148                     | 160                           | 224                          |
| **CUDA Cores (FP32)**          | 6,912          | 16,896         | 18,944                  | 20,480                        | TBD                          |
| **BF16 Tensor TFLOPS (dense)** | 312            | 989            | 2,250                   | ~2,500                        | TBD                          |
| **FP8 Tensor TFLOPS (dense)**  | —              | 1,979          | 4,500                   | 5,000                         | TBD                          |
| **FP4 Tensor PFLOPS (dense)**  | —              | —              | 9.0                     | 14–15                         | TBD                          |
| **Registers/SM**               | 64K × 32-bit   | 64K × 32-bit   | 64K × 32-bit            | 64K × 32-bit                  | 64K x 32-bit (est.)          |
| **TMEM/SM**                    | —              | —              | 256 KB                  | 256 KB                        | 256 KB+                      |
| **Shared Mem/SM (max)**        | 164 KB         | 228 KB         | 228 KB                  | 228 KB                        | TBD                          |
| **MUFU.EX2 ops/clk/SM**        | **16**         | **16**         | **16**                  | **32**                        | **32 (fp32)/64 (fp16)**      |

The performance gains of newer NVIDIA GPUs come mostly from **larger tensor cores**, which primarily accelerate GEMMs. In the above table, we see that BF16/FP8 Tensor Core throughput increases by ~2.27x from H100 to B200, while CUDA core throughput increases by only 1.1x and MUFU.EX2 throughput does not improve. Since the MUFU.EX2 instruction is used to compute `exp2` in softmax, in [FlashAttention-4](https://arxiv.org/pdf/2603.05451), attention is jointly bound by softmax and GEMM (both taking 1024 cycles for 128x128 tiles).

FA4 addresses this with a **warp specialized** pipeline that overlaps MMA and softmax work across warp groups (WGs).

{{< figure src="img/FA4.png" alt="B200 kernel" width="100%" align="center" caption="<span style=\"display:block; text-align:center;\">Source: [Colfax Research FlashAttention-4 Blog Post](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)</span>" >}}

However, the overlap is never perfect because of pipeline warmup (launching two $\mathbf{Q}\mathbf{K}$ MMAs in a row) and bookkeeping overheads such as address computation, issuing MMA instructions, updating the softmax row-max, and copying results across WGs, so we can still yield meaningful speedups by reducing either bottleneck.

FA4 tries to mitigate the softmax bottleneck by using a software emulated [polynomial approximation of exp2](https://arxiv.org/pdf/2603.05451#page=8) which increase the effective exp throughput by utilizing FMA (fused multiply-add) units in addition to the SFUs (special function units). The trade-off with this optimization is that higher-degree polynomials are more accurate but incur additional register usage and CUDA core instructions. Hence it’s only applied to 10%-25% of the softmax scores. Despite this, the softmax operation still remains register-heavy and a persistent bottleneck.

### TMEM overlap schedule

To leverage low-precision block-scaled MMAs to speedup the GEMMs in attention, understanding the data-flow of the scale factors is critical. On Blackwell GPUs, the scale factors must be loaded from GMEM $\rightarrow$ SMEM (via a TMA load) $\rightarrow$ TMEM and then must be [duplicated across four warps](https://github.com/NVIDIA/cutlass/issues/2961#issuecomment-3771068790) in a WG via a `tcgen05.cp` multicast in order to be usable by `tcgen05.mma`.

However, with 128x128 tiles, FA4's pipeline **already uses all available TMEM**:

- S1 and S2 ($\mathbf{Q}\mathbf{K}$ outputs): 128 columns each
- O1/O2 use the remaining columns: remaining 256 columns

{{< figure src="img/pipeline.png" alt="B200 kernel" width="100%" align="center" >}}

To avoid conflicts, we use a **TMEM overlap schedule** to squeeze in the scale factors while **minimizing pipeline stalls**: we reuse S2 for `sfqk 1` and S1 for `sfqk 2`. Because `tcgen05.mma` ops issued by a thread using the same shape are [guaranteed to execute sequentially](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05#tcgen05-memory-consistency-model-pipelined-instructions), we only insert barriers between S1 T2R and sfqk 2 load. Note that `sfvp 1` is guaranteed not to stomp on S1 due to being after QK2, and `sfvp 2` again uses a barrier to wait for S2 copy out (which rarely fires due to having 2 MMAs between QK2 and PV2).
We also considered storing $\mathbf{P}$ in SMEM to free up TMEM, but rejected it due to insufficient `ldmatrix` instruction shapes for an R2S copy (max of 8x8 vs 32x16 for `tcgen05.st`).

Despite careful scheduling, we still did not observe a speedup with NVFP4 block-scaled $\mathbf{P}\mathbf{V}$ GEMMs due to the aforementioned softmax bottleneck. Quantizing $\mathbf{P}$ and $\mathbf{V}$ requires computing group-wise scale factors and `cvt.rn.satfinite` quantization instructions, which adds to the existing softmax bottleneck.

Therefore, we choose to run block-scaled NVFP4 $\mathbf{Q}\mathbf{K}$ and BF16 $\mathbf{P}\mathbf{V}$ on a B200 which achieves a up to 1801 TFLOPS and a 1.39x speedup over FA4. Note that this is **only a lower bound of the speedup**: we have yet to experiment **NVFP4/MXFP8 + FP8** $\mathbf{P}\mathbf{V}$ (cutting MMA by ½ or to ¼ hardly matters for a softmax-bound kernel), which eliminates the group quantization overhead in a softmax WG.

Another detail worth mentioning is that a B300 GPU has 2x the exp throughput, and Rubin has 4x the exp throughput (w/ fp16 exp), which should make NVFP4 $\mathbf{P}\mathbf{V}$ GEMMs faster. **We are excited to test more QAT recipes for different hardware!**

| Config                 | FP4 (ms) | FP4 TFLOPS | BF16 (ms) | BF16 TFLOPS | Speedup |
| ---------------------- | -------- | ---------- | --------- | ----------- | ------- |
| b=1 s=256 h=16 d=128   | 0.015    | 37         | 0.015     | 35          | 1.01x   |
| b=1 s=1024 h=16 d=128  | 0.023    | 379        | 0.025     | 338         | 1.12x   |
| b=4 s=4096 h=16 d=128  | 0.336    | 1637       | 0.389     | 1413        | 1.16x   |
| b=4 s=8192 h=16 d=128  | 1.259    | 1747       | 1.511     | 1455        | 1.20x   |
| b=2 s=16384 h=16 d=128 | 2.467    | 1782       | 3.003     | 1464        | 1.22x   |
| b=1 s=32768 h=16 d=128 | 4.884    | 1801       | 6.771     | 1299        | 1.39x   |
| b=4 s=4096 h=32 d=128  | 0.655    | 1678       | 0.775     | 1418        | 1.18x   |
| b=4 s=8192 h=32 d=128  | 2.501    | 1759       | 3.027     | 1453        | 1.21x   |
| b=1 s=4096 h=12 d=128  | 0.104    | 986        | 0.117     | 882         | 1.12x   |
| b=1 s=32768 h=12 d=128 | 3.856    | 1711       | 5.056     | 1305        | 1.31x   |
| b=1 s=4096 h=24 d=128  | 0.152    | 1352       | 0.172     | 1198        | 1.13x   |
| b=1 s=32768 h=24 d=128 | 7.551    | 1747       | 10.061    | 1311        | 1.33x   |
| b=1 s=32768 h=24 d=64  | 7.170    | 920        | 7.284     | 906         | 1.02x   |

### Precision Results

At the time of writing this blog, we received updates from an [FP8 non-block-scaled PR](https://github.com/Dao-AILab/flash-attention/pull/2109), in the FA4 repo, so we show the kernel-level precision comparison below. Despite using NVFP4, we can see that our kernel achieves a 2-2.5x lower max absolute error with a group size of 16, compared to their per-head group (e.g., 128).

| batch | seqlen | nheads | hdim | FP4 max   | FP4 mean    | FP8 max | FP8 mean |
| ----- | ------ | ------ | ---- | --------- | ----------- | ------- | -------- |
| 1     | 256    | 16     | 128  | 0.158     | 0.01058     | 0.352   | 0.00591  |
| 1     | 512    | 16     | 128  | 0.148     | 0.00771     | 0.379   | 0.00555  |
| 1     | 1024   | 16     | 128  | 0.070     | 0.00553     | 0.260   | 0.00463  |
| 1     | 2048   | 16     | 128  | 0.078     | 0.00395     | 0.184   | 0.00372  |
| 2     | 4096   | 16     | 128  | 0.068     | 0.00280     | 0.136   | 0.00286  |
| 1     | 4096   | 24     | 128  | 0.043     | 0.00280     | 0.132   | 0.00284  |
| 1     | 8192   | 24     | 128  | 0.033     | 0.00199     | 0.075   | 0.00213  |
| 1     | 16384  | 16     | 128  | 0.025     | 0.00141     | 0.044   | 0.00157  |
| 1     | 32768  | 16     | 128  | **0.011** | **0.00100** | 0.022   | 0.00114  |
| 1     | 32768  | 24     | 128  | **0.016** | **0.00100** | 0.031   | 0.00114  |

### Agent-assisted Kernel Development

During debugging, we found LLM-based tools (e.g., Claude) surprisingly effective—even for low-level PTX and CuTeDSL code. It found an obscure uninitialized register bug in FA4, and we confirmed that it was fixed a week before we found it (buried in a [large commit](https://github.com/Dao-AILab/flash-attention/commit/c79976218fb71f282f76cb959a5aad48a2d23e86)). We estimate that Claude cut down at least 1-2 weeks of debugging time. In particular, it was very useful for SASS inspection (e.g. CuTeDSL $\rightarrow$ PTX $\rightarrow$ SASS mapping), instruction dependency analysis, and guided performance debugging via structured task lists in a `.md` file.

## What this work really changes

Prior to this paper, attention quantization was mostly treated as an inference problem: improve smoothing, calibration, or other post-hoc fixes. Attn-QAT argues that this view is incomplete. Since modern attention kernels are fused and precision-sensitive, **training methods and low-bit kernels must be co-designed**.

Despite NVIDIA’s headline FP4/FP8 (MMA) TFLOPS come from stacking units for pure GEMMs, attention often takes up the bulk of the wall-clock time in long-context agentic serving & video gen. Across the Hopper $\rightarrow$ Blackwell $\rightarrow$ Rubin evolution, we see a trend **toward algorithms and hardware becoming increasingly coupled as hardware headroom diminishes**.

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
