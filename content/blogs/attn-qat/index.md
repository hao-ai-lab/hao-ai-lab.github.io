+++
title = "Attn-QAT: Making 4-Bit Attention Actually Work"
date = 2026-04-05T12:00:00-07:00
authors = ["Peiyuan Zhang", "Matthew Noto", "Wenxuan Tan", "Chengquan Jiang", "Will Lin", "Wei Zhou", "Hao Zhang"]
author = "Peiyuan Zhang*, Matthew Noto*, Wenxuan Tan*, Chengquan Jiang, Will Lin, Wei Zhou, Hao Zhang"
ShowReadingTime = true
draft = false
math = true
contentClass = "post-content-justified"
description = "Attn-QAT makes FP4 attention trainable by enforcing precision consistency across FlashAttention's forward and backward passes during quantization-aware training."
summary = "Attn-QAT is the first systematic study of 4-bit quantization-aware training for attention, recovering FP4 attention quality without inference-time outlier mitigation while also enabling faster kernels."
+++

{{< socialBadges arxiv-index="2603.00040" github="hao-ai-lab/FastVideo" >}}

**TL;DR:** FP4 hardware is finally here, and FP4 linear layers are already being used in production. However, FP4 attention still causes significant quality degradation, preventing true end-to-end FP4 serving and limiting full hardware utilization. In this work, we present **Attn-QAT**, the first systematic study of 4-bit quantization-aware training for attention. We identify two key principles for stable FP4 attention QAT: (1) matching low-precision recomputation of attention scores in the backward pass and (2) resolving implicit precision assumptions in FlashAttention's gradient calculation. Across video diffusion models and language models, Attn-QAT recovers the quality drop of 4-bit attention without the extra outlier-mitigation heuristics used by prior FP4 attention methods, while also delivering up to 1.5x higher throughput than SageAttention3 on an RTX 5090.

## 4-bit attention remains unsolved

Native FP4 tensor core support on new NVIDIA GPUs makes 4-bit computation increasingly attractive. In principle, FP4 offers lower memory traffic and up to 4x higher arithmetic intensity than higher-precision BF16 execution. But in practice, attention is much harder to quantize than linear layers. The reason is twofold. First, FP4 has an extremely small dynamic range and only a tiny set of representable values. Second, attention activations are heavy-tailed and contain more outliers than standard matrix multiplies. These two properties make attention much more sensitive to quantization error.

This is exactly why prior training-free FP4 attention methods such as SageAttention3 still suffer in quality. Even with specialized outlier-mitigation techniques such as Q/K smoothing and two-level quantization of attention probabilities, we find that FP4 attention still degrades video quality on Wan 2.1-14B.


{{<youtube t7axGk-ev3E>}}

Instead of devising more sophisticated tricks to reduce quantization error in a training-free manner, we take a different approach: we employ quantization-aware training (QAT) for attention, which enables models to adapt to quantization error during training and thus preserve model quality. The goal of this work is simple: make FP4 attention work without any outlier-mitigation techniques. In the context of video generation, this means making FP4 attention produce videos that are indistinguishable in quality from BF16 videos at inference time.

<div style="text-align: center;">
  {{< image src="img/inference_algo.png" alt="attn-qat inference" width="50%">}}
</div>

## The two fixes that make Attn-QAT work

For linear layers, the QAT recipe is well known: simulate low-precision execution in the forward pass using fake quantization, keep the backward pass in higher precision, and train through the quantization noise. That works well for GEMMs. It does not work for attention.

The reason is that FlashAttention-style kernels are heavily fused operators whose backward pass relies on recomputation and algebraic identities that quietly assume that forward and backward passes share the same numerical behavior.

Attn-QAT stabilizes 4-bit attention by enforcing two forms of precision consistency.

### 1. Store a high-precision attention output for gradient computation


We first clarify notation. Let $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ denote the query, key, and value matrices, and define

\[
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d}, \quad
\mathbf{P} = \mathrm{softmax}(\mathbf{S}), \quad
\mathbf{O} = \mathbf{P}\mathbf{V}.
\]

We use $(\cdot)^F$ to denote fake-quantized (FP4-simulated) tensors, e.g., $\mathbf{P}^F = \phi^{-1}(\phi(\mathbf{P}))$.

In FlashAttention, the backward pass relies on a memory-efficient formulation of the softmax gradient. For a single row $i$,

\[
\mathbf{P}_i = \mathrm{softmax}(\mathbf{S}_i),
\]

and the gradient can be written as

\[
\mathbf{dS}_i
= \mathbf{P}_i \odot \mathbf{dP}_i
= (\mathbf{P}_i^\top \mathbf{dP}_i)\mathbf{P}_i.
\]

The key difficulty is the scalar term $\mathbf{P}_i^\top \mathbf{dP}_i$, which naively requires access to the full attention row and thus quadratic memory.

FlashAttention avoids this by rewriting the scalar as

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

However, under Attn-QAT, the forward pass instead uses fake-quantized probabilities:

\[
\mathbf{O}_i = \sum_j \mathbf{P}_{ij}^{F} \mathbf{V}_j^{F}.
\]

This introduces a forward-backward mismatch: the backward derivation depends on high-precision $\mathbf{P}$, while the forward pass uses $\mathbf{P}^F$. As a result,

\[
\mathbf{dO}_i^\top \mathbf{O}_i \neq \mathbf{P}_i^\top \mathbf{dP}_i,
\]

which leads to incorrect gradients and unstable training.

To resolve this, we compute an additional auxiliary output during the forward pass:

\[
\mathbf{O}_i' = \sum_j \mathbf{P}_{ij} \mathbf{V}_j^{F}.
\]

Here, $\mathbf{P}$ remains in high precision (FP32 softmax), while $\mathbf{V}^F$ is still fake-quantized. This adds only a small amount of extra storage, without materializing the full attention matrix.

In the backward pass, we then replace the scalar term with

\[
\mathbf{P}_i^\top \mathbf{dP}_i = \mathbf{dO}_i^\top \mathbf{O}_i'.
\]

This restores the exact identity:

\[
\begin{aligned}
\mathbf{P}_i^\top \mathbf{dP}_i
&= \sum_j \mathbf{P}_{ij} \, \mathbf{dO}_i^\top \mathbf{V}_j^{F} \\
&= \mathbf{dO}_i^\top \sum_j \mathbf{P}_{ij} \mathbf{V}_j^{F} \\
&= \mathbf{dO}_i^\top \mathbf{O}_i'.
\end{aligned}
\]

Intuitively, $\mathbf{O}$ is the low-precision output used by the model, while $\mathbf{O}'$ is a minimal high-precision correction that ensures the backward pass remains mathematically consistent. Without $\mathbf{O}'$, the gradient computation silently assumes a different forward computation than what actually occurred, which is the root cause of instability in naive FP4 QAT for attention.

This small modification preserves the fully low-precision forward path while restoring correctness in the backward pass, eliminating the need for heuristic outlier-mitigation techniques.

<div style="text-align: center;">
  {{< image src="img/grad_norm.png" alt="grad norm" width="50%">}}
</div>

### 2. Recompute attention probabilities in the same low precision used in the forward pass

In FlashAttention, the full attention probability matrix is not stored. It is recomputed during the backward pass from the saved log-sum-exp statistics. Under QAT, this recomputation must match the low-precision forward pass. Attn-QAT therefore fake-quantizes the recomputed attention probabilities in the backward pass, so gradients are computed with respect to the same quantized activations seen in the forward pass.


<div style="text-align: center;">
  {{< image src="img/training_algo.png" alt="attn-qat training" width="100%">}}
</div>

## Experimental results: quality is recovered

The strongest evidence comes from video diffusion, where attention errors are immediately visible as degraded motion or temporal inconsistency. In the example videos below, we see that with Attn-QAT, FP4 attention produces videos comparable to BF16 attention, whereas SageAttention3 produces videos with artifacts, and naive NVFP4 attention without QAT and outlier mitigation produces blurry videos.


{{<youtube gZipc43qvNE>}}


We also evaluate Attn-QAT on LLMs in two settings: continued pretraining and supervised fine-tuning.

For continued pretraining, Attn-QAT recovers most of the quality loss caused by FP4 attention on Qwen3-14B and partially recovers it on Llama 3.1-70B. The authors note that the remaining gap on the 70B model is likely due to limited training budget rather than a failure of the method itself.

For supervised fine-tuning, Attn-QAT can be used as a drop-in replacement for BF16 attention. On Qwen3-14B, it achieves nearly identical downstream benchmark performance to BF16. On Llama 3.1-70B, it remains close with a small gap. This is an important practical result: Attn-QAT is not only a specialized recovery stage for quantization, but can also be integrated directly into standard fine-tuning pipelines.


## Faster kernels

Because Attn-QAT no longer needs the extra smoothing and two-level quantization overhead used by SageAttention3, it also enables a faster inference path.

The paper implements Triton kernels for training and improved CUDA kernels for inference. On an RTX 5090, the Attn-QAT inference kernel achieves approximately 1.1x-1.5x higher throughput than SageAttention3, depending on the setup. The key reason is straightforward: by removing extra preprocessing for Q, K, and P, the kernel becomes lighter while preserving quality through training rather than inference heuristics.

<div style="text-align: center;">
  {{< image src="img/5090_speedup.png" alt="5090 speedup" width="50%">}}
</div>


## What this paper really changes

For a long time, quantization for attention has been treated mostly as an inference problem: find better clipping, smoothing, calibration, or other post-hoc fixes. Attn-QAT argues that this view is incomplete. Since modern attention kernels are fused and precision-sensitive, training methods and low-bit kernels must be co-designed.

This work also enables the development of training recipes that support more extreme inference optimizations, such as combining few-step distillation and full end-to-end FP4 execution during inference. We are actively working on this.

Please see [the paper](https://arxiv.org/abs/2603.00040) for more details. Code is available in [FastVideo](https://github.com/hao-ai-lab/FastVideo), our unified framework for video diffusion post-training and inference.

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

## Appendix: B200/B300 FP4 attention kernel

**TL;DR:** We release FlashAttention-4 FP4, an NVFP4-quantized FA4 kernel implemented in CuTeDSL, achieving up to 1.39x speedup and 1801 TFLOPS, along with a deeper look at implementation challenges and NVIDIA hardware evolution.

### FP4/FP8 support on Blackwell

Blackwell is the first GPU generation to provide native FP4/FP8 GEMM support via the `tcgen05.mma.cta_group.kind.block_scale` instruction family. Previous quantization methods on Hopper (H100) and Ampere (A100), such as W4A8 (QServe) and W4A16 (AWQ), used software dequantization: the kernel allocates extra registers and CUDA cores to dequantize tensors group-wise after loading them into the kernel. On Blackwell, this is no longer practical because tensor core throughput roughly doubles relative to Hopper while CUDA core count stays roughly unchanged.

The broader issue is that softmax remains the bottleneck while tensor-core throughput keeps rising. NVIDIA increases advertised TFLOPS primarily by scaling chip size and power, with most of the growth allocated to tensor cores [[1]](https://newsletter.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis). Software tricks such as exponential approximation can only mitigate the problem. FA4 only applies them to 25% of the softmax scores before register spilling becomes a concern. The practical lesson is to design attention operators around the actual balance of available hardware units and their capacity.

Other implementation takeaways include:

1. A TMEM-overlapped schedule is required because B200 is often TMEM-bound, which limits effective MMA pipeline depth.
2. Quantizing $ P $ does not help in the same way as quantizing Q, K, and V because it adds too much pressure to the softmax warps.
3. Interleaving quantization with register-to-TMEM copies does not resolve that pressure, though quantized PV may become more attractive on B300 and Rubin with improved FP16 exponential support.


<div style="text-align: center;">
  {{< image src="img/B200_plot.png" alt="B200 kernel" width="100%">}}
</div>

### Kernel performance

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


### Conclusion and future work

NVIDIA's headline FP4/FP8 GEMM TFLOPS increasingly come from allocating more silicon and power to tensor cores while other units scale more slowly. But long-context agentic serving and video generation are not purely GEMM-bound. Looking from Hopper to Blackwell and onward to Rubin, attention speed will likely improve more slowly as long as softmax and $ O(n^2) $ computation remain central to model capacity.
