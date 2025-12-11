+++
title = "AUP: when Accuracy Meets Parallelism in Diffusion Language Models"
date = 2025-12-10T12:00:00-08:00
authors = ["Yu-Yang Qian", "Junda Su", "Lanxiang Hu", "Peiyuan Zhang", "Zhijie Deng", "Peng Zhao", "Hao Zhang"]
author = "Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Zhijie Deng, Peng Zhao, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/d3LLM"
[cover]
      image = "/img/dllm_leaderboard.png"
      alt = "d3LLM: Ultra-Fast Diffusion LLM 🚀"
      caption = "AUP: when Accuracy Meets Parallelism in Diffusion Language Models"
      hidden = true
+++

{{< socialBadges github="hao-ai-lab/d3LLM" demo="https://d3llm-team.github.io/" huggingface="https://huggingface.co/d3LLM">}}

{{< justify >}}

**TL;DR:** Diffusion large language models (dLLMs) promise things that autoregressive LLMs cannot:  parallel decoding, error correction, and random-order generation. Over the past year, a wave of papers has pushed this vision, and closed-source systems like [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/) and [Mercury](https://arxiv.org/abs/2506.17298) report impressive throughput numbers. In this blog, we take a step back and ask a simple question: if we look at both speed and accuracy together, are diffusion LLMs actually better decoders than strong autoregressive (AR) models?


<div style="margin-top: -12px;"></div>

In our study of open-source systems, we find a consistent accuracy–parallelism trade-off: pushing more tokens per forward pass almost always costs accuracy. We introduce Accuracy Under Parallelism (**AUP**), a hardware-robust metric that scores this trade-off in one number, and we present [d3LLM](https://github.com/hao-ai-lab/d3LLM), a distillation + decoding framework that improves AUP and narrows the gap to strong AR + speculative decoding baselines. Our d3LLM achieves up to 5× speedup over the AR baseline (Qwen-2.5-7B-it) on H100 GPU and 3.5× speedup on A100 GPU. Feel free to try 🕹️ [our demo](https://d3llm-team.github.io/).



{{< /justify >}}


{{< justify >}}


<style>
.responsive-img-grid {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 20px;
  flex-wrap: wrap;
}
.responsive-img-grid img {
  height: auto;
  flex-shrink: 0;
}
@media (max-width: 768px) {
  .responsive-img-grid img {
    width: 100% !important;
  }
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.responsive-img-grid img[data-width]').forEach(function(img) {
    var width = img.getAttribute('data-width');
    if (width) {
      img.style.width = width + '%';
    }
  });
});
</script>


## Background

Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs.  Conceptually, they promise things AR models fundamentally struggle with:
- **Parallel decoding:** update many tokens per forward pass instead of generating one token at a time.
- **Error correction:** revise earlier positions during refinement.
- **Random-order generation:** tokens need not be produced strictly left-to-right.

In the best-case story, dLLMs could be "the future of LLM inference": faster decoding without giving up quality, plus extra capabilities that AR decoding doesn’t naturally offer.
Recently, several diffusion large language models have been announced, including [Mercury](https://arxiv.org/abs/2506.17298), [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/), and [Seed Diffusion](https://arxiv.org/abs/2508.02193), which demonstrate impressive efficiency and performance and achieve extremely high throughput - sometimes reported at 1000+ tokens per second in certain settings.

But the open-source reality today is much more mixed. Many open diffusion models are still slow in common inference stacks, and they often trail similarly sized AR models in accuracy. For example, [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487) reach only about 20 tokens per second, sometimes even slower than AR baselines if accounting for the number of refinement steps and cache behavior.

This raises a simple question that we think has been under-emphasized: If we evaluate both speed and accuracy together, are today’s diffusion LLMs actually better decoders than strong AR models (especially AR + speculative decoding)? In this blog post, we attempt to study that question with evidence, and then use what we learned to: (1) propose a better metric, and (2) build a better diffusion system guided by that metric.

### Key Finding: a Fundamental Accuracy-Parallelism Trade-off in dLLMs

To answer this question, we conduct a comprehensive evaluation of SOTA dLLM methods (including [Fast-dLLM](https://arxiv.org/abs/2505.22618), [D2F](https://arxiv.org/abs/2508.09192), [dParallel](https://arxiv.org/abs/2509.26488), and [Fast-dLLM-v2](https://arxiv.org/abs/2509.26328)) on several widely used benchmarks in dLLM literature:
- Math / reasoning: [GSM8K-CoT (zero-shot CoT)](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml), [MATH](https://openreview.net/forum?id=IFXTZERXdM7)
- Coding: [HumanEval](https://arxiv.org/abs/2107.03374), [MBPP](https://arxiv.org/pdf/2108.07732)
- Long-context reasoning: 5-shot [GSM8K](https://arxiv.org/abs/2110.14168) (prompt length ~1000)

We measure two quantities for each model and decoding configuration:
- Accuracy (solve rate / pass@1 depending on the benchmark)
- Parallelism, measured by tokens per forward pass (**TPF**)

Why TPF? Because it captures the algorithmic “how many tokens do I advance per model evaluation” effect that diffusion-style decoding and speculative methods aim to improve (We'll come back to this in [Section AUP](#aup-considering-both-accuracy-and-parallelism)). 
The results are summarized in the table below.

{{< /justify >}}


{{< dllm_leaderboard_previous >}}
<div style="margin-top: -30px;"></div>
<figcaption style="text-align: center; color: #808080; margin-top: 0px;">Table 1. Benchmark results of previous dLLM models compared with AR models.</figcaption>

<div style="margin-top: 20px;"></div>


{{< justify >}}

Upon careful examination of previous dLLM methods (e.g., [dKV](https://arxiv.org/abs/2505.15781), [MMaDA](https://arxiv.org/abs/2505.15809), [SDAR](https://arxiv.org/abs/2510.06303), [Fast-dLLM](https://arxiv.org/abs/2505.22618), and [D2F](https://arxiv.org/abs/2508.09192)), the answer to this question is clear: the speedup offered by dLLMs is ***not a free lunch***. It almost always comes with accuracy degradation – different dLLMs simply land at different points on the same curve:
- Methods like D2F push hard on parallelism (higher TPF), but take a visible hit in accuracy compared to similarly sized AR models.
- Methods like Fast-dLLM-v2 preserve accuracy better, but at the cost of lower parallelism (lower TPF).

In other words, most diffusion decoding improvements implicitly slide along a trade-off frontier: *more parallelism usually means lower accuracy, and vice versa.*


It is worth noting that, in parallel, a separate line of work seeks to improve the efficiency of AR models through speculative decoding and multi-token prediction (e.g., [Medusa](https://arxiv.org/abs/2401.10774), [Hydra](https://arxiv.org/abs/2402.05109), [CLLM](https://arxiv.org/abs/2403.00835), [OSD](https://arxiv.org/abs/2310.07177)). By combining AR models with speculative decoding, i.e., the state-of-the-art [EAGLE-3](https://arxiv.org/abs/2503.01840) method with the LLaMA-Instruct 3.1 8B model, parallelism can be improved **without sacrificing accuracy**. This approach achieves superior results and significantly outperforms current dLLM methods.


Now here’s the part that surprised us the most when we looked at the data “with both axes turned on”:
***When judged jointly on speed and accuracy, strong AR models combined with speculative decoding in fact deliver the best overall trade-offs in our study. (see row 1 of Table 1).*** For example, state-of-the-art speculative decoding (e.g., EAGLE-3 on LLaMA-3.1 8B) increases effective parallelism while remaining (in principle) lossless relative to the target AR model. Under this joint view, diffusion systems do not currently dominate. We clarify that this does not mean diffusion is “bad”:
- Diffusion decoding is genuinely parallel and can be very fast.
- But open diffusion systems today pay for speed with accuracy, and the cost is often non-trivial.
- AR + speculative decoding remains a very strong baseline when you measure the full trade-off (although the drafting overhead is non-negligible and may increase system complexity).


### Why Do We Need a New Metric?

At this point, we ran into a practical problem: the literature (and many blog discussions) tends to report diffusion progress using single, isolated metrics:
- Efficiency-only metrics: tokens per second (TPS) or tokens per forward (TPF)
- Performance-only metrics: accuracy (solve rate / pass@1)

Unlike AR, dLLM ***by nature lives on an accuracy–parallelism curve***, hence single metrics become misleading, as it overlooks the fundamental trade-off between efficiency and performance to answer the real question: How well does a method maintain accuracy as we push parallelism higher?

These insights motivate us to ***design a new unified metric*** – **AUP**, which we describe next.


{{< /justify >}}


## AUP: Considering Both Accuracy and Parallelism

{{< justify >}}


Most dLLM methods already expose certain knobs that trade off speed and quality. e.g., Fast-dLLM employs a logit “threshold”,where tokens with logits above this threshold can be decoded in parallel. By sweeping this threshold, we can adjust the quality–speed trade-off and obtain multiple parallelism–accuracy pairs, which can then be used to plot a curve of accuracy versus parallelism. We refer to this curve as the ***accuracy–parallelism curve*** (see the white curve in Figure 1 for an illustration), which characterizes the trade-off frontier dLLMs navigate.

A natural first attempt is to summarize the curve by the area under the curve (AUC). But plain AUC has a serious failure mode: it can reward models that become extremely fast by letting accuracy collapse. The right side of the curve can contribute lots of area even if the model is not useful in practice. We want a metric that strongly prefers staying in a high-accuracy regime, and only then rewards higher parallelism.

We propose AUP (Accuracy Under Parallelism) as a weighted area under the accuracy–parallelism curve, where the weight penalizes accuracy drops relative to the best achievable accuracy on that task. 
Formally, let $\mathcal{S} = \{(\rho_i, y_i)\}_{i=1}^m$ be a set of parallelism-accuracy pairs, where $\rho_1 < \rho_2 < \dots < \rho_m$, $\rho_i \in \mathbb{R}^{+}$ denotes the parallelism (measured in _tokens per forward_, TPF), and $y_i \in [0, 100]$ represents accuracy in percentage. We define a minimum accuracy threshold $y_{\min} = y_1 - 5$ to avoid measuring in regimes of significant accuracy degradation. Only points satisfying $y_i \ge y_{\min}$ are included. AUP is then defined as:

$$\operatorname{AUP} \triangleq \rho_1 y_1 + \frac{1}{2} \sum_{i=2}^{m} (\rho_{i} - \rho_{i-1}) \left( y_i \cdot W(y_i) + y_{i-1} \cdot W(y_{i-1}) \right),$$

where the weighting function is defined as $W(y) = \min(e^{-\alpha \left(1 - {y}/{y_\max}\right)}, 1)$, with a penalty factor $\alpha = 3$ and $y_\max$ denotes the highest accuracy achieved on that task.


{{< /justify >}}

<!-- {{< image src="img/aup_illustration.png" alt="AUP Illustration" width="50%" title="Figure 1: Illustration of the AUP metric. The metric captures both parallelism (TPF) and accuracy, with a weighting function that penalizes accuracy degradation.">}} -->



<figure>
<div class="responsive-img-grid">
  <img src="img/aup_illustration.png" alt="AUP Illustration" data-width="50">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 1: Illustration of the AUP metric. The metric captures both parallelism (TPF) and accuracy, with a weighting function that penalizes accuracy degradation.</figcaption>
</figure>

The intuition behind AUP is simple:
- If you increase parallelism without losing accuracy, your AUP increases a lot.
- If you increase parallelism by sacrificing accuracy, your AUP increases only a little (or not at all), because the penalty suppresses low-accuracy regimes.



{{< justify >}}

**Choice of $\alpha$.** The hyperparameter $\alpha$ controls the penalty for accuracy degradation. A larger $\alpha$ increases sensitivity to performance drops, causing the contribution of throughput to decay exponentially with the error rate. In the ideal case, where a method improves parallelism without compromising accuracy, the AUP reduces to the standard area under the parallelism-accuracy curve (AUC). In our setting, we set $\alpha = 3$ as it balances the importance of parallelism and accuracy.

**AUP is hardware-independent** because AUP is built on TPF (token per forward), not TPS (token per second). TPS is heavily affected by hardware generation (H100 vs A100), kernel fusion, cache implementation, and the inference framework. The same algorithm can look dramatically different depending on system details.
For instance, in our experiments, our [d3LLM-LLaDA model](https://huggingface.co/d3LLM/d3LLM_LLaDA) (which will be introduced in the next section) demonstrates around 5× higher TPS than an AR baseline (Qwen-2.5-7B-it) on an NVIDIA H100 GPU (289 vs. 57 tokens/s). However, this advantage shrinks significantly on an NVIDIA A100 GPU (175 vs. 50 tokens/s). In contrast, the TPF captures the algorithmic parallelism: how many tokens you progress per forward pass. This is much more stable across hardware. Therefore, AUP gives a fairer view of algorithmic progress, without requiring everyone to run on the exact same GPU or inference engine, helping the community focus on algorithmic design without requiring access to particular GPUs.

{{< /justify >}}


{{< justify >}}

### What AUP Reveals about Today’s Landscape

Once we scored existing methods using AUP, the landscape became clearer (see Table 1): Recent diffusion acceleration methods do improve AUP over vanilla diffusion baselines (e.g., vanilla Dream / LLaDA). This is real progress. However, state-of-the-art AR + speculative decoding methods still achieve the top overall AUP in our evaluation. We need methods that move the entire accuracy–parallelism curve up and to the right, not just push parallelism at the expense of accuracy. This is where d3LLM comes in: we treat AUP as the optimization target, and design a diffusion framework specifically to increase AUP.


{{< /justify >}}

## d3LLM: Jointly Achieving Accuracy and Parallelism 🚀


{{< justify >}}
{{< image src="img/example.gif" alt="d3LLM: Ultra-fast diffusion language model" width="100%" title="Figure 2. Demo of our d3LLM, which achieves up to 5× speedup over the AR (Qwen-2.5-7B-it) on H100 GPU and 3.5× speedup on A100 GPU. You can try 🕹️ [our demo](https://d3llm-team.github.io/).">}}

{{< /justify >}}

{{< justify >}}

Following the guidance of the AUP score, we introduce ***[d3LLM](https://github.com/hao-ai-lab/d3LLM)*** (*pseuDo-Distillated-Diffusion Large Language Model*), a novel framework for constructing dLLMs with both high accuracy and high parallelism. d3LLM combines two main ideas:
1.	*Pseudo-trajectory distillation (training)*: Instead of distilling only from a teacher’s final answers, we distill from the teacher diffusion model’s decoding order (the order in which it unmasks tokens). This provides intermediate supervision about which tokens can be safely decoded earlier, which directly improves parallelism. we design a ***curriculum learning strategy*** that gradually increases the masking ratio from easier scenarios (few masks) to more difficult ones (many masks) during training, resulting in a more robust distillation process.
2.	*Multi-block decoding with KV-cache refresh (inference)*: At inference time, we decode multiple blocks in parallel based on confidence (entropy), and we introduce a ***KV-cache refresh mechanism*** to prevent quality degradation that can occur with aggressive multi-block parallelism.


Together, these techniques enable d3LLM to strike a balance between accuracy and parallelism and to obtain the highest AUP score among all dLLMs.

{{< /justify >}}


{{< justify >}}

### (i) Pseudo-Trajectory-based Distillation Recipe

{{< image src="img/fig_distillation.png" alt="Distillation Illustration" width="100%" title="Figure 3. Illustration of our pseudo-trajectory-based distillation recipe.">}}

{{< /justify >}}


{{< justify >}}

- **Utilizing the Teacher dLLM's Pseudo-Trajectory (15%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

    A key challenge in distillation is that dLLM's intermediate supervision is unavailable: we usually only have prompt–response pairs, without teacher's intermediate states. Ideally, when the teacher’s output matches the ground truth, its decoding trajectory provides an ideal real-trajectory for teaching the student the correct generation order, but such cases are rare. To overcome this, we instead use the teacher dLLM’s own decoding trajectory as a pseudo-trajectory, even when its final answer differs from the ground truth.


    <div style="margin-top: 2mm;"></div>

    Specifically, given a prompt \$\mathbf{x}\$ and a predefined maximum output length \$n\$, we first let the teacher dLLM to generate and record its own decoding trajectory \$\\{\mathcal{T}_1,\ldots,\mathcal{T}_n\\}\$, where \$\mathcal{T}_i \in \mathbb{R}^n, \forall i \in \\{1,\ldots,n\\}\$. Rather than relying on the content of the teacher's response, we extract only the order in which tokens are decoded. This order forms what we refer to as the **_pseudo-trajectory_** of the teacher. Combine the pseudo-trajectory with the ground-truth prompt-response pair \$(\mathbf{x}, \mathbf{y})\$ and construct a _noisy sequence_ \$\widetilde{\mathbf{y}} \in \mathbb{R}^n\$ that simulates teacher's intermediate state during the decoding process. Formally, let \$t \in [0, 1]\$ denote mask ratio, and \$w = \\{s, \ldots, s + k\\}\$ be a decoding window of length \$k\$, the noisy sequence \$\widetilde{\mathbf{y}}\$ is

    $$[\widetilde{\mathbf{y}}]_i= \begin{cases}[\mathbf{y}]_i & \text { if } i \leqslant s \text {  or  }\left[\mathcal{T}_{s+\lceil k t \rceil}\right]_i \neq \texttt{mask}, \\ {\texttt{mask} } & \text { if } i>s+k \text { or }\left[\mathcal{T}_{s+\lceil k t \rceil}\right]_i=\texttt{mask},\end{cases}$$

    where \$\texttt{mask}\$ is the special mask token ID, and \$[\cdot]_i\$ denotes the \$i\$-th token in the trajectory sequence. This leads to a **15% improvement in TPF** compared to strategies that use random masking.

{{< /justify >}}


{{< justify >}}

- **Progressive Noise Level (Further get 18%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

  To preserve accuracy during distillation, we introduce a _progressive noise schedule_ by gradually increasing the mask ratio \$t\$ from 0.0 to 0.8 during the training process. This curriculum learning approach encourages the model to learn from easier to harder decoding scenarios, thereby enhancing its robustness and decoding efficiency while maintaining generation quality. Empirically, this strategy further improves the model's tokens-per-forward (TPF) by approximately **18%** compared to using a fixed mask ratio. Without this curriculum strategy, we observe that the distillation process becomes unstable and the model is more likely to suffer accuracy degradation.

{{< /justify >}}


{{< justify >}}
- **Progressive Window Size (Further 8%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

  We also employ a _progressive window sizing_ as another curriculum learning technique: instead of fixing the decoding window length \$k\$, we gradually increase it from 16 to 32 during the training process. This allows the model to adapt to increasingly larger context spans, facilitating a smoother distillation process and stable token generation while maintaining accuracy. This approach leads to an additional **8%** improvement in TPF compared to a constant window size.

{{< /justify >}}


---

{{< justify >}}


### (ii) Multi-Block Decoding Strategy

{{< image src="img/fig_decoding.png" alt="Decoding Illustration" width="100%" title="Figure 4. Illustration of our multi-block decoding strategy with KV-cache and refresh.">}}


In addition to the distillation recipe, we also introduce an efficient decoding mechanism tailored for d3LLM.

{{< /justify >}}

{{< justify >}}

- **Entropy-Based Multi-Block Parallel Decoding (20%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

  Inspired by the approach in [D2F](https://arxiv.org/abs/2508.09192), we propose an _entropy-based multi-block decoding_ method. Unlike conventional diffusion decoding, which operates strictly within a single block, our method enables decoding of both the current and future blocks in parallel. We select tokens to decode based on the entropy threshold, in which lower-entropy (more confident) predictions are first to be unmasked. 
  
  <div style="margin-top: 2mm;"></div>

  Each block can be in one of five states: `Inactive`, `Activated`, `Fully-Activated`, `Completed but Stabilizing`, and `Completed`. We create a new `Activated` block when its preceding block reaches 10% completion and employ a conservative decoding strategy for this block, generating tokens only when they meet the entropy threshold. When the preceding block reaches 95% completion, the `Activated` block transitions to a `Fully-Activated` state, where a more aggressive strategy is used by decoding at least one token per forward pass, regardless of the threshold. Once all tokens in a block are unmasked, the block enters the `Completed but Stabilizing` state, during which we perform forward passes without using the KV cache and refresh previous caches. After 1-2 rounds, the block becomes `Completed`, and we store its KV cache. In addition, we apply a periodic-refresh strategy that updates the KV cache every few rounds. This multi-block decoding strategy increases TPF by **20%**, and the KV-refresh mechanism helps maintain the accuracy.

{{< /justify >}}

{{< justify >}}

- **Multi-Block Decoding with KV-Cache and Refresh (20%↑ TPS under Long Contexts)**

    <div style="margin-top: 2mm;"></div>

  To further improve decoding throughput while maintaining generation quality, particularly in long-context settings, we incorporate a _KV-cache_ mechanism alongside a periodic _KV-refresh_. Specifically, after completing each block, we introduce a short delay before caching its key–value states to ensure that the cache remains reliable and does not lead to performance degradation. Simultaneously, we perform full forward passes to refresh previous caches. This hybrid strategy maintains decoding accuracy while significantly improving TPS by approximately **20%** in long-context scenarios.

{{< /justify >}}

{{< justify >}}

- **Early Stopping on EOS Token (5%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

  We implement an *early stopping mechanism* that halts decoding once the end-of-sequence (EOS) token is generated. This simple yet effective optimization reduces unnecessary computation and yields a **5%** improvement in TPF on average.

{{< /justify >}}


{{< justify >}}

## Benchmark Results

We present comprehensive benchmark results across five representative tasks: GSM8K-CoT (chain-of-thought reasoning), MATH (mathematical problem solving), HumanEval (code generation), MBPP (Python programming), and a long-context math reasoning task (5-shot GSM8K reasoning, with a prompt length ≈ 1000). These datasets span diverse domains and problem types and are widely used in the research community. In addition, their relatively long output lengths allow us to effectively evaluate the models' parallel decoding capabilities together with their accuracy.

Our experiments are conducted on three foundational diffusion models: [LLaDA](https://arxiv.org/abs/2502.09992), [Dream](https://arxiv.org/abs/2508.15487), and [Dream-Coder](https://arxiv.org/abs/2509.01142). From these, we derive three distilled models, [d3LLM-LLaDA](https://huggingface.co/d3LLM/d3LLM_LLaDA), [d3LLM-Dream](https://huggingface.co/d3LLM/d3LLM_Dream), and [d3LLM-Coder](https://huggingface.co/d3LLM/d3LLM_Dream_Coder), each trained using the same trajectory-based distillation recipe and multi-block decoding strategy outlined previously. We use a single GPU and fix the batch size to 1 for all models.

**Implementation Details.** Our d3LLM begins with a block diffusion model (either LLaDA or Dream) with a block size of 32 as the teacher model. For fair comparison, we adopt the same distillation dataset as dParallel, which includes approximately 122k samples for Dream and 92k samples for LLaDA, sourced from the PRM12K, AceCode, GSM8K (training split), and Numina-Math datasets. The learning rate is set to 2e-5. We train 6 epochs for LLaDA and 3 for Dream. More implementation details can be found in our [GitHub code](https://github.com/hao-ai-lab/d3LLM).

{{< /justify >}}

{{< justify >}}

For the **LLaDA-based models**, we compare our *d3LLM-LLaDA* with _vanilla LLaDA, Fast-dLLM-LLaDA, D2F_, and _dParallel-LLaDA_. The parallelism-accuracy curves are as below.

{{< /justify >}}

<figure>
<div class="responsive-img-grid">
  <img src="img/data_llada_aup_curve_gsm8k_cot.png" alt="LLaDA GSM8K-CoT" data-width="30.2">
  <img src="img/data_llada_aup_curve_humaneval.png" alt="LLaDA HumanEval" data-width="33">
  <img src="img/data_llada_aup_curve_mbpp.png" alt="LLaDA MBPP" data-width="31.2">
</div>
<div class="responsive-img-grid" style="margin-top: 20px;">
  <img src="img/data_llada_aup_curve_math.png" alt="LLaDA MATH" data-width="30">
  <img src="img/data_llada_aup_curve_long-gsm8k.png" alt="LLaDA Long-GSM8K" data-width="30.2">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 5: AUP curves for LLaDA-based models across five benchmark tasks (GSM8K-CoT, HumanEval, MBPP, MATH, and Long-GSM8K).</figcaption>
</figure>

<figure>
<div class="responsive-img-grid">
  <img src="img/data_llada_aup_histogram.png" alt="LLaDA AUP Histogram" data-width="51">
  <img src="img/data_llada_aup_radar.png" alt="LLaDA AUP Radar" data-width="40.5">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 6: AUP scores and radar chart comparing different LLaDA-based methods.</figcaption>
</figure>

{{< justify >}}

For the **Dream-based models**, we compare our *d3LLM-Dream* with _vanilla Dream, Fast-dLLM-Dream, Fast-dLLM-v2-7B_, and _dParallel-Dream_. The parallelism-accuracy curves are as below.

{{< /justify >}}

<figure>
<div class="responsive-img-grid">
  <img src="img/data_dream_aup_curve_gsm8k_cot.png" alt="Dream GSM8K-CoT" data-width="29.8">
  <img src="img/data_dream_aup_curve_humaneval_instruct.png" alt="Dream HumanEval_Instruct" data-width="30.5">
  <img src="img/data_dream_aup_curve_mbpp_instruct.png" alt="Dream MBPP_Instruct" data-width="30.2">
</div>
<div class="responsive-img-grid" style="margin-top: 20px;">
  <img src="img/data_dream_aup_curve_math.png" alt="Dream MATH" data-width="31.2">
  <img src="img/data_dream_aup_curve_long-gsm8k.png" alt="Dream Long-GSM8K" data-width="30.2">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 7: AUP curves for Dream-based models across five benchmark tasks (GSM8K-CoT, HumanEval_Instruct, MBPP_Instruct, MATH, and Long-GSM8K).</figcaption>
</figure>

<figure>
<div class="responsive-img-grid">
  <img src="img/data_dream_aup_histogram.png" alt="Dream AUP Histogram" data-width="56.5">
  <img src="img/data_dream_aup_radar.png" alt="Dream AUP Radar" data-width="40">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 8: AUP scores and radar chart comparing different Dream-based methods.</figcaption>
</figure>

{{< justify >}}


**Results on Different Models and Datasets.** As shown by the results above, the proposed distillation recipe and multi-block decoding strategy are robust and improve efficiency across various domains. Specifically, our d3LLM achieves the highest AUP score on 9 out of 10 tasks, and accelerates the vanilla LLaDA by approximately 5–10× on TPF across different tasks. 
Remarkably, we note that for Fast-dLLM-v2, the accuracy scores on Math and HumanEval are notably higher than those of other diffusion models derived from Dreams. We suspect that this stems from the fact that Fast-dLLM-v2 is finetuned directly from Qwen-2.5-7B with an additional 1B tokens (i.e., the LLaMA–Nemotron post-training dataset). In contrast, our [d3LLM-Dream](https://huggingface.co/d3LLM/d3LLM_Dream) is distilled based on the vanilla Dream and uses only 60M additional tokens.



The experimental results also validate the reliability of our AUP metric. For example, on the MBPP dataset with the LLaDA-based model, although many methods achieve parallelism (TPF) greater than 1, their accuracy degradation compared with the best-performing model (Qwen-2.5-7B-it) is substantial, leading to low overall utility. This demonstrates that the AUP metric more faithfully reflects the practical efficiency–performance trade-off.

{{< /justify >}}


{{< justify >}}

**Efficient Diffusion Coder.** Beyond LLaDA and Dream, we further apply our distillation approach and multi-block decoding method to a more realistic and challenging application: efficient coding models. Specifically, we use _Dream-Coder-7B-Instruct_ as the teacher dLLM and collect 120k samples from the Ling-Coder-SFT and AceCode datasets, along with a small amount of math-reasoning data, to distill our [d3LLM-Coder](https://huggingface.co/d3LLM/d3LLM_Dream_Coder). The results are demonstrated as below.


{{< /justify >}}

<figure>
<div class="responsive-img-grid">
  <img src="img/data_dream_coder_aup_curve_humaneval.png" alt="Dream-Coder HumanEval" data-width="22.4">
  <img src="img/data_dream_coder_aup_curve_humaneval+.png" alt="Dream-Coder HumanEval+" data-width="22.4">
  <img src="img/data_dream_coder_aup_curve_mbpp.png" alt="Dream-Coder MBPP" data-width="25.4">
  <img src="img/data_dream_coder_aup_curve_mbpp+.png" alt="Dream-Coder MBPP+" data-width="22.7">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 9: Evaluation for Coders across four coding benchmarks (HumanEval, HumanEval+, MBPP, MBPP+).</figcaption>
</figure>

<figure>
<div class="responsive-img-grid">
  <img src="img/data_dream_coder_aup_histogram.png" alt="Dream-Coder AUP Histogram" data-width="53">
  <img src="img/data_dream_coder_aup_radar.png" alt="Dream-Coder AUP Radar" data-width="40.6">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 10: AUP scores and radar chart comparing different Coder-based methods.</figcaption>
</figure>




{{< justify >}}

**Wall-Clock Speed Comparison.** In addition to AUP scores, we further evaluate different methods on multiple hardware platforms, including H100 and A100 GPUs, to measure their wall-clock throughput (measured by tokens per second, TPS) and speedup. 
For the *LLaDA-8B-Instruct*, we report speed (TPS) and accuracy on GSM8K-CoT dataset.
The results are presented below.

{{< /justify >}}



{{< table title="Table 2. TPS and performance comparison on LLaDA-based models on GSM8K-CoT dataset." >}}

|                 | H100's TPS | A100's TPS | Acc   |
|-----------------|:----------:|:----------:|:-----:|
| Qwen-2.5-7B-it  | 57.32      | 50.36      | 74.10 |
| LLaDA           | 27.89      | 19.15      | 72.55 |
| Fast-dLLM-LLaDA | 114.29     | 79.14      | 74.68 |
| D2F             | 102.13     | 76.24      | 74.39 |
| dParallel-LLaDA | 172.23     | 105.85     | 72.63 |
| **d3LLM-LLaDA** | **288.73** | **174.57** | **73.10** |

{{< /table >}}


For the *Dream-7B-Instruct*, we again report speed and accuracy on GSM8K-CoT dataset.


{{< table title="Table 3. TPS and performance comparison on Dream-based models on GSM8K-CoT dataset." >}}

|                 | H100's TPS | A100's TPS | Acc   |
|---------------|:----------:|:----------:|:-----:|
| Qwen-2.5-7B-it  | 57.32      | 50.36      | 74.10 |
| Dream           | 13.41      | 8.32       | 83.94 |
| Fast-dLLM-Dream | 77.25      | 51.55      | 79.00 |
| Fast-dLLM-v2-7B | 150.01     | 109.68     | 81.48 |
| dParallel-Dream | 168.36     | 80.23      | 82.12 |
| **d3LLM-Dream** | **235.34** | **128.19** | **81.86** |

{{< /table >}}


{{< justify >}}

To summarize, our d3LLM framework achieves the highest AUP score with negligible performance degradation, successfully balancing both parallelism and accuracy and striking a balance between accuracy and parallelism. It delivers up to a **5× speedup** over autoregressive decoding (Qwen-2.5-7B-it) on H100 GPUs (288.73 TPS vs. 57.32 TPS), and approximately **3.5× speedup** on A100 GPUs (174.57 TPS vs. 50.36 TPS) with comparable performance. This makes dLLMs more practical for real-world deployment.

Note that all experiments are using the HuggingFace inference backend. System-level optimizations, including GPU kernel fusion and integration with vLLM, are left for future work to further improve TPS.


{{< /justify >}}

## 🏆 Diffusion LLM Leaderboard using AUP Score

We present the leaderboard of diffusion LLMs, using the AUP score as the evaluation metric.

{{< dllm_leaderboard >}}

{{< justify >}}

Our d3LLM-Coder achieves higher TPF and maintains the highest AUP score among all diffusion LLMs.
Notably, the state-of-the-art speculative decoding method, EAGLE-3 (with LLaMA-Instruct 3.1 8B), attains the top overall AUP score. This is expected, as speculative decoding includes an additional verification step and therefore does not suffer from accuracy degradation as in dLLMs under high parallelism. Moreover, our evaluation does not constrain total FLOPs, and speculative decoding methods may take more FLOPs than diffusion-based approaches. Nevertheless, our d3LLM framework substantially narrows the gap between diffusion-based models and SOTA speculative decoding methods, offering valuable insights for future research directions.


All our distillation code, data, model weights, and benchmark evaluation code are available at [https://github.com/hao-ai-lab/d3LLM](https://github.com/hao-ai-lab/d3LLM). The full paper about AUP and our d3LLM framework will be released soon. Stay tuned!

{{< /justify >}}

{{< justify >}}

<!-- 
## Reference

{{< /justify >}}


{{< justify >}}
[1] [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)
{{< /justify >}}

{{< justify >}}
[2] [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/abs/2505.15781)
{{< /justify >}}

{{< justify >}}
[3] [Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](https://arxiv.org/abs/2508.09192)
{{< /justify >}}

{{< justify >}}
[4] [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488)
{{< /justify >}}

{{< justify >}}
[5] [Fast-dLLM v2: Efficient Block-wise Diffusion LLM](https://arxiv.org/abs/2509.26328)
{{< /justify >}}

{{< justify >}}
[6] [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666)
{{< /justify >}}

{{< justify >}}
[7] [LLaDA 2.0](https://huggingface.co/collections/inclusionAI/llada-20)
{{< /justify >}}

{{< justify >}}
[8] [SDAR: A Synergistic Diffusion-AutoRegression Paradigm for
Scalable Sequence Generation](https://arxiv.org/abs/2510.06303)
{{< /justify >}}

{{< justify >}}
[9] [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809)
{{< /justify >}}

{{< justify >}}
[10] [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
{{< /justify >}}

{{< justify >}}
[11] [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)
{{< /justify >}}

{{< justify >}}
[12] [Dream-Coder 7B: An Open Diffusion Language Model for Code](https://arxiv.org/abs/2509.01142)
{{< /justify >}} -->
