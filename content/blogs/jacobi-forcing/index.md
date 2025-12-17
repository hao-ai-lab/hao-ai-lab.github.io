+++
title = "Fast and Accurate Causal Parallel Decoding using Jacobi Forcing"
date = 2025-12-16T12:00:00-08:00
authors = ["Lanxiang Hu*", "Siqi Kou*", "Yichao Fu", "Samyam Rajbhandari", "Tajana Rosing", "Yuxiong He", "Zhijie Deng", "Hao Zhang"]
author = "Lanxiang Hu*, Siqi Kou*, Yichao Fu, Samyam Rajbhandari, Tajana Rosing, Yuxiong He, Zhijie Deng, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/JacobiForcing"
[cover]
      image = "/img/jacobi-forcing/decoding_comparison.gif"
      alt = "jacobi forcing decoding"
      caption = "Side-by-side comparison between Jacobi forcing decoding and text diffusion decoding, where Jacobi forcing decoding comes with more efficient KV cache reuse and is trained to generate higher quality drafts over a long horizon."
+++


{{< socialBadges arxiv-index="2512.14681" github="hao-ai-lab/JacobiForcing" huggingface="https://huggingface.co/JacobiForcing">}}


{{< justify >}}
**TL;DR**: Today’s Best LLMs mostly decode autoregressively from left-to-right, which gives great quality but is terribly slow. Diffusion LLM can decode many tokens in parallel thanks to their non-casual, any-order generation, but they must be trained from scratch, or heavily adapted from autoregressive (AR) checkpoints with a non-casual diffusion objective; we find this mismatch often hurts quality and breaks many effective KV-cache related serving optimizations. This blog introduces Jacobi Forcing, a new training technique that converts LLMs into native casual parallel decoders. Jacobi forcing keeps the casual AR backbone and fixes the AR-to-diffusion mismatch by training the model to handle noisy future blocks along its own Jacobi decoding trajectories. This yields an AR model which behaves like a diffusion-style decoder—decoding multiple tokens per pass, but still from left to right—with up to $4.5\times$ higher tokens-per-forward and $4\times$ wall-clock speedup on coding and math tasks, while retaining near-AR generation quality. 
{{< /justify >}}


{{< two_images
    src1="img/ar_example_demo.gif"
    src2="img/jacobi_forcing_example_demo.gif"
    alt1="ar_demo"
    alt2="jacobi_forcing_demo"
    width1="50%"
    width2="50%"
    title="Figure 1: Demo of on average more than $4\times$ speedup (181.8 TPS vs. 39.81 TPS) by Jacobi Forcing Model in comparison with the AR baseline (Qwen2.5-Coder-7B-Instruct) on coding sessions."
>}}

## Background

{{< justify >}}
Modern LLM inference has a simple but painful bottleneck: **decoding is mostly serial**. With autoregressive decoding, each new token depends on all previous ones, so we pay (roughly) one forward pass per token.

Most existing work on faster decoding falls into two broad families: 

1. **Diffusion-style LLMs (dLLMs):** use non-causal, often bidirectional attention and denoising objectives to update *many* tokens in parallel.

2. **Speculative decoding (SD)**: keeps a causal AR backbone but relies on a draft model or extra heads that propose multiple future tokens per verification step.

Table 1 (row 2 and row 3) summarizes their pros and cons. At a high-level, dLLMs offer strong parallelism but demand expensive non-causal post-training and custom infrastructure; SD preserves AR quality but adds FLOPs and system complexity for modest net gains. Let’s dive deeper to discuss their trade-offs.
{{< /justify >}}



### Diffusion LLMs

{{< justify >}}
Diffusion-style dLLMs iteratively denoise entire token blocks with non-causal (often bidirectional) attention. At each step, the model sees a globally noised sequence and tries to predict a cleaner one, updating many positions in parallel. This offers a natural form of parallel decoding, but comes with several trade-offs.

From a modeling perspective, the cleanest way to get a high-quality dLLM would be to pretrain it from scratch with a diffusion-style objective. But at today’s scales, fully training a non-causal dLLM to match a strong AR baseline (where we have invested multiple billions) is prohibitively expensive, so almost nobody does this in practice.
Instead, most recent work starts from a strong AR-pretrained checkpoint and then converts it into a diffusion-style model by oftentimes heavy post-training with a denoising objective. This AR-to-dLLM conversion introduces two kinds of mismatch.
- The first is a training objective mismatch. AR pre-training sees clean, causal prefixes, while diffusion-style post-training sees globally noised sequences and learns to denoise them. The model is now being asked to serve two different goals, and the resulting distribution shift makes it hard to fully recover AR-level quality.
- The second is an attention and infrastructure mismatch. To denoise whole token blocks in parallel, these methods typically switch from causal masking to non-causal (often bidirectional) attention. That breaks exact KV-cache reuse and many low-level optimizations baked into today’s AR-optimized kernels and serving stacks, and it complicates batching and scheduling in production systems.


In practice, recent dLLMs of this form often require billions to hundreds of billions of additional post-training tokens on top of AR pre-training, and still either lag behind strong AR baselines in accuracy or struggle to turn their theoretical parallelism into proportional wall-clock speedups.
{{< /justify >}}



{{< image src="img/baselines_comparison.png" alt="comparison_plot" width="60%" title="Figure 2: Comparison across various diffusion LLMs techniques with speed and accuracy trade-off (as well as associated training cost).">}}



### Speculative Decoding

{{< justify >}}
**Speculative decoding (SD)** keeps the causal AR backbone and its lossless quality, but introduces an additional draft stage. A draft model (or draft head) proposes multiple future tokens. The target model (the main AR backbone) then verifies these proposals and accepts or rejects them in parallel. If drafting were free and most tokens were accepted, SD would give a clean speedup: multiple tokens per verification step without any loss in quality. In reality, SD introduces several overheads:
- The **draft model still consumes FLOPs, memory, and latency**. Strong SD methods like EAGLE-3 and HASS achieve impressive speedups, but  also involve training the draft models or draft heads and integrating them into the serving stack (see these GitHub issues as examples: [SGL-6949](https://github.com/sgl-project/sglang/issues/6949), [vLLM-9565](https://github.com/vllm-project/vllm/issues/9565), [vLLM-15025](https://github.com/vllm-project/vllm/issues/15025)).
- Integrating SD into production serving systems adds **engineering complexity**: two-model orchestration, heuristics for drafting length, and extra complexity in batching and scheduling.


As a result, end-to-end speedups therefore often plateau at around $2-3\times$ even when the "acceptance length per step" looks impressive.
{{< /justify >}}



### Where Does Jacobi Forcing Fit?

{{< justify >}}
Table 1 summarizes the trade-offs of all three families discussed above: 
- **Standard AR decoding**: simple, high quality, but strictly serial.
- **SD**: keeps AR quality but adds draft overhead and system complexity.
- **dLLMs**: strongly parallel but require expensive non-causal post-training and custom infrastructure, and often lower quality.
{{< /justify >}}

{{< justify >}}
This leads to the central question behind Jacobi Forcing:

Can we build a **native causal parallel decoder** that (i) runs fast like diffusion-style methods, (ii) preserves AR-level quality, and (iii) fits naturally into existing KV-cache-based serving systems without extra models or heavy architectural changes?

**Jacobi Forcing answers "yes" to this question.**
{{< /justify >}}

### Can We Get both Quality and Parallelism using Jacobi Forcing?

{{< justify >}}
Jacobi forcing builds on top of jacobing decoding, which is a causal parallel decoding procedure that repeatedly updates all tokens in a block in parallel until they match the greedy AR output, tracing a parallel refinement trajectory while preserving the causal attention mechanism. See these papers ([Parallelizing feedforward with Jacobi iterations](https://arxiv.org/pdf/2002.03629), [Parallel Decoding](https://arxiv.org/pdf/2305.10427)) and blogpost ([Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)) describing Jacobi decoding in details. 

Our prior work on CLLMs showed that fine-tuning on Jacobi trajectories can shorten this trajectory and enable faster decoding, but it did not fully exploit hardware constraints or longer-horizon noise.

Jacobi Forcing pushes this idea further: we keep the original causal attention and minimize pre-/post-train mismatch, and train the model so that Jacobi-style decoding **produces high-quality drafts that stay close to the AR distribution even under noisy long-horizon context**. This is realized via a noise-conditioned training, along with an inference algorithm that exploits high-quality n-grams appearing in the draft. as summarized in Figure 2, Jacobi Forcing turns standard AR models into highly efficient parallel decoders while retaining competitive AR-like quality.

{{< /justify >}}




| Method        | Attention      | Parallelism                      | Training Cost | Single-model Decoding (no draft–verifier)   | Efficient KV Reuse        | Real Speedup | Generation Quality         |
|:----------------------:|:------------------------:|:-----------------------------------------------:|:------------------------------------:|:-------------------------------------------:|:----------------------:|:-------------------------------:|:---------------------------:|
| **AR**        | Causal | 💔                          | 🆓                                            | 😃   |  😃               | 🐢            | 🏅️            |
| **SD** | Causal                 | 😃             | 🆓/💰        | 💔 | 😃     | ⚡️/⚡️⚡️ |      🏅️       | 
| **dLLMs**            | Non-causal    | 😃   | 💰/💰💰💰      | 😃 | 💔 | ⚡️ | 🥉/🥈 |
| **Jacobi Forcing**   | Causal                 | 😃   | 💰 | 😃 |  😃   |  ⚡️⚡️⚡️    | 🥈/🏅️   |

{{< image src="" alt="" width="100%" title="Table 1: Qualitative comparison of parallel decoding methods.">}}



## Jacobi Forcing

### Noise schedule and Training Sequence Preparation

{{< justify >}}
Training with Jacobi Forcing starts from collecting Jacobi trajectories of the base AR model. For each prompt: (1) for all $N$ blocks of size $n$ in its generation, the base model runs Jacobi decoding on each block $i \in \{N\}$ to obtain intermediate states and the final fixed point that matches greedy AR decoding. (2) Treat each intermediate state as a “noisy” view of the fixed point, with an associated noise ratio $s_i^k (\text{number of unconverged tokens}/n)$ for the $k$-th Jacobi iteration.

To make learning feasible for large blocks, Jacobi Forcing packs the training sequences the following way and uses a **progressive noise schedule**:

- We split the response into $N$ blocks of size $n$ and assign each block a target noise ratio $t_i \in [0, 1]$ taken from a small set $W$ denoted to the noise schedule, with a linear progressive noise schedule we have $W  = \{0, 1/w, 2/w, \dots, (w-1)/w\}$ (for some window size $w$). Noise ratios could repeat cyclically along the sequence: $t_i = W[i \bmod w]$.

- For each block’s Jacobi trajectory, we then find the intermediate state whose noise ratio $s_i^{(k)}$ is closest to $t_i$, and use that state as the **noisy block** for block $i$, with its fixed point as the **clean block**.

- Arrange time steps ${t_i}$ in short cyclic windows (from nearly clean to heavily noised), so a single packed training sequence always contains a structured mixture of easy (low-noise) and hard (high-noise) denoising subproblems across blocks, rather than long stretches of uniformly high noise.


Progressive noise schedule shortens long runs of corrupted tokens and keeps each denoising problem local and learnable, especially when scaling the block size, while still covering a rich range of noise levels within every packed sequence as illustrated in Video 1.
{{< /justify >}}

{{<youtube KQUiKdxHugs>}}
{{< image src="" alt="" width="100%" title="Video 1: Illustration of the training sequence packing process with an example (linear progressive) noise schedule mapping.">}}


### Noisy-Context Conditioned Training

{{< justify >}} 
Naively training on each Jacobi state would require many passes. Instead, Jacobi Forcing:


- Packs **noisy blocks** $\tilde{\mathbf y}_i$ and their **fixed-point (clean) versions** $\mathbf y_i^{*}$ into a single long sequence: 

$$
\tilde{\mathbf y}_{1:N} = (\tilde{\mathbf y}_1, \dots, \tilde{\mathbf y}_N) 
\text{ and } \mathbf y^{*}_{1:N} = (\mathbf y^{*}_1, \dots, \mathbf y^{*}_N).
$$

- Uses a **noise-conditioned causal attention mask** as shown in Figure 3 so each token:
    - Sees the prompt and earlier blocks at their assigned noise levels.
    - Knows which positions in its block are noisy or clean. 
    - Exposes the fixed-point tokens needed to compute a teacher distribution.


This lets a single forward–backward pass compute losses for multiple noise levels $t_i$ and blocks $i = 1,\dots,N$. Concretely, the training objective combines:

- A **progressive consistency loss** that pushes the model to map noisy blocks $\tilde{\mathbf y}_i$ to their fixed points $\mathbf y_i^{*}$ in one Jacobi update:

  $$
  \mathcal L_{\text{pc}}(\theta)
  = \mathbb E_{(\mathbf x, \tilde{\mathbf y}_{1:N}, \mathbf y^{*}_{1:N})}
  \Biggl[
      \frac{1}{N}
      \sum_{i=1}^{N}
      D_{\mathrm{KL}}\Bigl(
        p_{\theta^-}(\cdot \mid \mathbf x, \mathbf y^{*}_{1:i})
        \,\Big\|\, 
        p_{\theta}(\cdot \mid \mathbf x, \tilde{\mathbf y}_{1:i})
      \Bigr)
  \Biggr],
  $$ 
  where $\tilde{\mathbf y}_{1:i} = (\tilde{\mathbf y}_1, \dots, \tilde{\mathbf y}_i)$ and $\mathbf y^{*}_{1:i} = (\mathbf y^{*}_1, \dots, \mathbf y^{*}_i)$.

- A standard **AR loss** that keeps overall generation quality anchored to the base model’s greedy output $\mathbf l = (l_1,\dots,l_L)$:

  $$
  \mathcal{L}_{\text{AR}}(\theta)
  = \mathbb{E}_{(\mathbf{x}, \mathbf{l})}
  \big[
    -\sum_{t=1}^{L}
      \log p_{\theta}\big(l_t \mid \mathbf{x}, \mathbf{l}_{< t}\big)
  \big]
  $$

The final objective is therefore: 
  
  $$\mathcal{L}(\theta) = \mathcal{L}_{\text{pc}}(\theta) + \lambda \mathcal{L}_{\text{AR}}(\theta)
  $$

where $\lambda > 0$ balances progressive consistency and AR fidelity.
{{< /justify >}} 

{{< two_images
    src1="img/clean_context_attention_mask.png" 
    src2="img/noisy_context_attention_mask.png" 
    alt1="clean_context_attention_mask" 
    alt2="noisy_context_attention_mask"
    width1="50%" 
    width2="50%" 
    title="Figure 3: Jacobi Forcing uses the attention implementation on the right, whereas CLLMs' training doesn't involve noise conditioning and is therefore more similar to the one on the left. Both implementations allow logits from clean blocks and noisy blocks to be generated with single forward pass to calculate the progressive consistency loss and AR loss.">}}


## Jacobi Forcing Model Inference

### Observation: Jacobi Forcing Model with Higher-quality Drafts

{{< justify >}}
After training, Jacobi Forcing model is still a standard AR checkpoint, but its Jacobi trajectories change qualitatively:


- Intermediate Jacobi states now contain **long n-grams in the draft that already match the final greedy AR output**. 
- Once an n-gram becomes correct, it tends to stay correct across later iterations, even if neighboring tokens are still wrong and the positions are wrong. 
- As a result, we can cache these stable n-grams and reuse them at the right positions in subsequent verification steps for further speedup.


This “stability under noisy futures” is precisely what the noise-conditioned training objective encourages and is what makes Jacobi Forcing model a strong self-speculative decoder without any extra model.
{{< /justify >}}

{{< image src="img/trajectory.png" alt="high_quality_draft_illustration" width="100%" title="Figure 4: Visualization of Jacobi Forcing model’s trajectory under vanilla Jacobi decoding. The figure shows a partial segment of the trajectory. Blue tokens denote accepted tokens that match the fixed point at their positions. Black tokens denote unconverged noisy tokens, and we highlight them in red if more than three consecutive tokens match the fixed point regardless of position.">}}


### Multiblock decoding

{{< justify >}}
To better utilize the GPU, Jacobi Forcing model employs multiblock Jacobi decoding:

- Maintain up to $K$ blocks in flight.
- Mark one block as **real-active**, whose tokens are verified and committed into the KV cache.
- Treat other blocks as **pseudo-active**: (1) They are updated under Jacobi iterations using the current prefix. (2) Their tokens are not committed to the KV cache yet.
- When the real-active block converges, it promotes a pseudo-active block and re-verify all of its tokens under the updated prefix with all tokens converged.
{{< /justify >}}


### Rejection recycling

{{< justify >}}
Jacobi Forcing model also leverages **rejection recycling** to reuse high-quality n-grams from earlier iterations as illustrated in Figure 4 to expedite convergence:

- Cache promising n-grams, where its first token matches the last token in the committed KV, from previous Jacobi iterations in an n-gram pool.
- Verify those candidate n-grams in parallel along the batch dimension during the next Jacobi step. 
- Choose the path with the highest acceptance rate (TPF) count.


Because Jacobi Forcing model’s intermediate states are much higher quality than those of the base AR model, this recycling step becomes highly effective, turning previously “wasted” speculative work into real progress.
{{< /justify >}}

{{<youtube 8t3oda5gnHs>}}
{{< image src="" alt="" width="100%" title="Video 2: Illustration of multiblock Jacobi decoding with rejection recycling. High-quality n-grams from earlier iterations are reused as drafts. Up to $K$ blocks (here $K = 2$) are maintained: earlier blocks are real-active and commit tokens to the KV cache, while later pseudo-active blocks run Jacobi updates under noisy context and are only verified and committed after all preceding blocks have been finalized in the KV cache.">}}


### Hardware-aware Configuration Search


{{< justify >}}

We do not pick Jacobi Forcing model’s inference hyperparameters by trial-and-error alone. Instead, we tune the decoding configuration so that it sits near the compute–memory “knee” of the hardware roofline while still producing high-quality drafts.

In our inference algorithm, the main knobs are:

- **Block size $n$** (how many tokens are updated in parallel).
- **Number of blocks $K$** (depth of multiblock decoding).
- **Verification budget `pool_size`** (how many recycled candidates are checked per step).
- **Activation ratio $r$** (how far a block should converge before we activate additional pseudo-active blocks).


In practice, we fix $r = 0.85$ and $K = 2$ as the heuristic optimal. The choices are constrained by training: later pseudo-active blocks must still see enough clean context to draft meaningful tokens that actually boost the acceptance rate. If we lower $r$ or increase $K$ too aggressively, later blocks are conditioned on overly noisy prefixes and tend to generate “trash” tokens that rarely get accepted, hurting both quality and speed.


With $r$ and $K$ fixed, we then sweep over **block size** and **verification size** (Figure 5a) and find that the best tradeoff is achieved at block size $n = 64$ and verification size $= 4$. This configuration also aligns with the roofline profiling on H200 and B200 GPUs (Figure 5b), where these settings sit closest to the compute–memory roofline while keeping latency overhead modest.
{{< /justify >}}

{{< two_images
    src1="img/B200_profiling.png"
    src2="img/best_fit_contour_K2_r0.85.png"
    alt1="blocksize-verif-sweep"
    alt2="roofline"
    width1="50%"
    width2="50%"
    title="Figure 5a (left): Roofline profiling on B200 GPUs, The 256 batched token setting sits closest to the roof “knee,” where compute and memory are both well utilized. This matches the profiling trend in Figure 5b and justifies our choice of this configuration as the default forJacobi Forcing model. Figure 5b (right): Sweep over block size and verification size (pool_size), showing the best tradeoff at block size n = 64 and verification size = 4, while fixing $r=0.85, K=2$. Contour is obtained by interpolation and Gaussian-like smoothing."
>}}

{{< justify >}}
While Figure 5 focuses on throughput in tokens-per-second (TPS), the same sweeps reveal that TPF for Jacobi Forcing scales almost monotonically as we spend more FLOPs on larger blocks and higher verification budgets. On B200, our default configuration ($n = 64$, `pool_size` = 4) already achieves a strong TPF at $4.2\times$, but pushing to a more aggressive setting with **block size $n = 256$ and verification size $= 16$** increases TPF further to **$4.57\times$**, at the cost of substantially higher per-step compute. We do not adopt this configuration as default today because, on current Blackwell-class hardware, it starts to move beyond the roofline “knee” and yields diminishing TPS gains for a given latency budget.
{{< /justify >}}

### Why Jacobi Forcing Works?

{{< justify >}}

In summary, Jacobi Forcing works at two levels:

1. **Intra-trajectory (within a block):**  For each block, we keep the same idea as CLLM: the model is trained so that any intermediate Jacobi state is mapped to the fixed point. And we found training models this way can effectively allow fast forwarding across commonly-used phrases in natural language.

2. **Inter-trajectory (across blocks):**  Across blocks, we introduce a noise schedule where earlier blocks in a window see lighter corruption, later blocks see heavier corruption. This creates a curriculum from “denoise a few tokens” to “denoise many tokens,” making the objective much easier than asking the model to fix a fully corrupted long block in one shot. Empirically, this schedule encourages the model to produce higher-quality drafts even when conditioned on noisy futures. 

Our ablation study training models on a 10k subset of data shows that linear progressive noise schedule outperforms both random and reverse progressive schedules, where **reverse progressive (putting the heaviest noise first) is clearly harmful**, leading to the slowest convergence.
{{< /justify >}}

<div style="margin: 0 auto; width: fit-content;">

| Strategy            | Acc.  | iter/token |
|---------------------|-------|-----------:|
| Random              | 83.5  | 0.53       |
| Linear Progressive  | 84.7  | **0.48**   |
| Reverse Progressive | 82.3  | 0.62       |

</div>

{{< image src="" alt="" width="100%" title="Table 2: Ablation study on different noise schedules with 10k training examples.">}}



## Experiments

{{< justify >}}
Jacobi Forcing is evaluated on:

- **Coding benchmarks:** HumanEval and MBPP with Qwen2.5-Coder-7B-Instruct.
- **Math benchmarks:** GSM8K and MATH with Qwen2.5-Math-7B-Instruct.


Compared to **dLLM baselines at 7B scale**, Jacobi Forcing model offers a much better accuracy–speed trade-off:


- On HumanEval, the strongest diffusion model baseline (D2F) reaches $1.8\times$ speedup with 54.3% accuracy, while Jacobi Forcing model (MR) reaches $4.0\times$ speedup with 82.3% accuracy.
- On GSM8K, D2F yields $2.2\times$ speedup with 77.6% solve rate; Jacobi Forcing model (MR) pushes this to $3.7\times$ speedup at 91.4%.
- Similar trends hold on MBPP and MATH: Jacobi Forcing model matches or exceeds dLLMs’ speed while maintaining substantially higher task accuracy.


Compared to **CLLM-style parallel decoders at the ssame 7B scales**, Jacobi Forcing model consistently provides ~1.7× higher throughput at similar or slightly lower accuracy, while keeping the pure AR backbone and KV reuse: 

- On HumanEval, CLLM achieves $2.5\times$ speedup with 88.0% accuracy, whereas Jacobi Forcing model (MR) achieves $4.0\times$ speedup with 82.3%.
- On GSM8K and MATH, CLLM reaches about $2.1\times$ speedup; Jacobi Forcing model (MR) pushes this to $3.7\times$ with negligible accuracy change.
{{< /justify >}}



### Detailed Results (on A100, at 7B scale)

| Task      | Method           | Family      | Speedup $\uparrow$ | TPF $\uparrow$ | TPS $\uparrow$ | Acc / Solve $\uparrow$ |
|----------|--------------|------------|-----------|-------|-------|---------------|
| HumanEval| AR           | AR           | $1.00\times$    | 1.0  | 41.3  | 87.8%       |
|          | D2F          | dLLM         | $1.8\times$    |  2.5   | 73.2    | 54.3%   |
|          | Fast-dLLM    | dLLM         | $1.5\times$    |  1.8   | 60.0    | 53.0%   |
|          | dParallel    | dLLM-distilled  | $2.1\times$  |  2.9   | 88.5    | 54.3%   |
|          | EAGLE-3      | SD           |  $2.9\times$ | 6.4 | 120.7 | 68.9%$^*$ |
|          | HASS         | SD           |  $3.4\times$ | 5.5 | 138.7 | 61.6%$^*$ |
|          | CLLM$^*$        | causal parallel | $2.5\times$  | 2.7  | 103.3 | 88.0%       |
|          | **Jacobi Forcing model**    | causal parallel | $3.9\times$    | 4.0  | 159.5 | 83.5%  |
|          | **Jacobi Forcing model (MR)** | causal parallel | **$4.0\times$** | 4.1  | 163.9 | 83.5% |
| GSM8K | AR               | AR           | $1.0\times$     | 1.0  | 41.8   | 92.4%      |
|       | D2F                | dLLM       | $2.2\times$   |  2.3  |  91.2   | 77.6%    |
|       | Fast-dLLM      | dLLM           | $1.2\times$       |  2.1 | 49.8   | 75.0%      |
|       | dParallel      | dLLM-distilled | $3.1\times$  |  3.8   |  128.0   | 82.9%   |
|       | EAGLE-3        | SD             | $3.3\times$  | 7.2   | 138.6  |  63.9%$^*$           |
|       | HASS           | SD             | $3.1\times$  | 5.0   | 128.1  |  74.0%$^*$           |
|       | CLLM*          | causal parallel    | $2.1\times$     | 2.3  | 86.8   | 92.2%      |
|       | **Jacobi Forcing model**        | causal parallel | $3.5\times$    | 3.7  | 146.1  | 91.4% |
|       | **Jacobi Forcing model (MR)**   |causal parallel | **$3.7\times$** | 4.0  | 154.9 | 91.4% |


{{< image src="" alt="" width="100%" title="Table 3: Generation quality and efficiency comparison among Jacobi Forcing model, baseline SD and baseline dLLM methods.">}}


{{< justify >}}
<small><em>Footnote<sup>*</sup>:</em> Here we report the strongest checkpoints released by the authors; in principle EAGLE-3 and HASS are lossless in comparison with greedy AR checkpoints if they were trained with the Qwen2.5-7B backbone. Note that SD has a worse acceptance length (TPF) to TPS conversion ratio due to other overheads in the algorithm like token drafting using draft head, tree-like verification overhead, feature merging from different layers etc. </small>
{{< /justify >}}



{{< justify >}}
Overall, Jacobi Forcing model consistently delivers **up to $3-4\times$ wall-clock speedup** on coding and math tasks with only minor accuracy changes versus greedy AR, while significantly outperforming both dLLMs and prior consistency-based parallel decoders in the accuracy–throughput tradeoff.
{{< /justify >}}

{{< justify >}}
On a single B200 GPU with much higher FLOPs, the same Jacobi Forcing model with multiblock + rejection recycling can achieve an even more significant speedup at around 330 tokens/s (vs. around 80 tokens/s using AR), showing that the design continues to scale on newer accelerators.
{{< /justify >}}




## Get started

{{< justify >}}
For more details on the noise-conditioned objective, multiblock decoding, and rejection recycling, please see the Jacobi Forcing paper and appendix. We also invite you to try the Jacobi Forcing codebase and Jacobi Forcing model checkpoints at:
{{< /justify >}}

- GitHub: https://github.com/hao-ai-lab/JacobiForcing
- Huggingface: http://huggingface.co/JacobiForcing

{{< justify >}}
Because Jacobi Forcing model preserves the original AR architecture and KV layout, you can often deploy it as a near drop-in replacement for an existing AR model: just switch the checkpoint and use a Jacobi-style decoding loop with the recommended multiblock and recycling configuration.
{{< /justify >}}


