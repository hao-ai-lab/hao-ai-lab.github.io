+++
title = "AUP: when Accuracy Meets Parallelism in Diffusion Language Models"
date = 2025-12-09T12:00:00-08:00
authors = ["Yu-Yang Qian", "Junda Su", "Lanxiang Hu", "Peiyuan Zhang", "Peng Zhao", "Hao Zhang"]
author = "Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Peng Zhao, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/text-diffusion"
[cover]
      image = "/img/dllm_leaderboard.png"
      alt = "AUP: when Accuracy Meets Parallelism in Diffusion Language Models"
      caption = "AUP: when Accuracy Meets Parallelism in Diffusion Language Models"
      hidden = true
+++

{{< socialBadges github="hao-ai-lab/text-diffusion" demo="https://d3llm-team.github.io/" huggingface="https://huggingface.co/d3LLM">}}

{{< justify >}}

**TL;DR:** Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs, offering advantages such as parallel decoding, error correction, and random-order generation—capabilities that AR models lack. In this blog, we conduct a comprehensive study of open-source dLLMs and uncover a fundamental trade-off between accuracy and parallelism. However, prior work typically focuses on only one side of this trade-off, targeting either efficiency or performance. When jointly considering speed and accuracy, AR models combined with speculative decoding still deliver the best results. We argue that this is due to previous dLLMs rely on single, isolated metrics, which fail to capture the full picture. To address this, we introduce **Accuracy Under Parallelism (AUP)**, a unified metric that evaluates both speed and accuracy. Furthermore, guided by AUP, we propose **d3LLM** (*pseuDo-Distilled Diffusion LLM*), a framework that incorporates a novel distillation method and decoding strategy, outperforming previous SOTA approaches and achieving a balance between efficiency and performance. 


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

Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs. A key advantage of dLLMs is the use of *bidirectional attention*, which enables parallel decoding, error correction, and random-order generation capabilities, especially promising the high parallelism and speedup, that are not feasible for AR models. Recently, several diffusion large language models have been released, including [Mercury](https://arxiv.org/abs/2506.17298), [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/), and [Seed Diffusion](https://arxiv.org/abs/2508.02193), which demonstrate impressive efficiency and performance and achieve high throughput compared to AR models, and can output 1000+ tokens per second. 


Despite this progress, open-source dLLMs have attracted growing interest but currently exhibit *significantly lower throughput*, in some cases running even slower than AR LLMs. For example, [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487) reach only about 20 tokens per second, whereas closed-source dLLMs exceed 1000 tokens per second. Moreover, the *inferior accuracy* of open-source dLLMs relative to similarly sized AR models further limits their practical utility. These raise natural questions:


<div style="text-align: center;">

***Are dLLMs really faster than AR? If so, at what cost? Or is it a free lunch?***

</div>

To answer these questions, we conduct a comprehensive evaluation of state-of-the-art dLLM methods [[1](https://arxiv.org/abs/2505.22618), [3](https://arxiv.org/abs/2508.09192), [4](https://arxiv.org/abs/2509.26488), [5](https://arxiv.org/abs/2509.26328)] on several widely used benchmarks (GSM8K-CoT-Zero, MATH, HumanEval, MBPP, and 5-shot GSM8K reasoning) and compare them with an AR models. We evaluate both parallelism (measured by tokens per forward pass, TPF) and accuracy. The results are summarized in the table below.


{{< /justify >}}


{{< dllm_leaderboard_previous >}}
<div style="margin-top: -30px;"></div>
<figcaption style="text-align: center; color: #808080; margin-top: 0px;">Table 1. Benchmark results of previous dLLM models compared with AR models.</figcaption>

<div style="margin-top: 20px;"></div>


{{< justify >}}

Upon careful examination of previous dLLM methods, the answer to these questions is clear: the speedup offered by dLLMs is ***not a free lunch***. It comes at the cost of accuracy degradation, and existing dLLM approaches implicitly navigate a trade-off between accuracy and parallelism. For example, as shown in the table below, D2F prioritizes efficiency by achieving, but its accuracy declines compared to the similar-sized AR model. In contrast, Fast-dLLM-v2 attains higher accuracy than other dLLM methods but at the cost of lower TPF. This fundamental trade-off between accuracy and parallelism represents a core challenge for dLLMs.

In parallel, a separate line of work seeks to improve the efficiency of AR models through *speculative decoding*. By combining AR models with speculative decoding, i.e., the state-of-the-art [EAGLE-3](https://arxiv.org/abs/2503.01840) method with the LLaMA-Instruct 3.1 8B model, parallelism can be improved **without sacrificing accuracy**. This approach achieves superior results and significantly outperforms current dLLM methods.

To summarize the current landscape: although numerous methods have been proposed to improve dLLMs [[1](https://arxiv.org/abs/2505.22618), [2](https://arxiv.org/abs/2505.15781), [3](https://arxiv.org/abs/2508.09192), [4](https://arxiv.org/abs/2509.26488), [5](https://arxiv.org/abs/2509.26328), [6](https://arxiv.org/abs/2510.08666), [7](https://huggingface.co/collections/inclusionAI/llada-20), [8](https://arxiv.org/abs/2510.06303), [9](https://arxiv.org/abs/2505.15809)], they ***typically focus on only one side of the coin, targeting either efficiency or performance***. We argue that this issue stems from the limitations of relying on single, isolated metrics, such as reporting only tokens per second (TPS) or tokens per forward (TPF) for efficiency, or only accuracy for model performance. This practice overlooks the fundamental *trade-off* between efficiency and performance: improving efficiency often reduces accuracy, and vice versa. These insights motivate us to ***design a new unified metric*** that jointly captures both efficiency and performance, guiding models to ***strike a balance*** between both.

{{< /justify >}}


## AUP: Considering Both Performance and Parallelism

{{< justify >}}

Building on the above comprehensive studies, we observe that it is essential to *look at two sides of the coin*, due to the inherent accuracy-speed tradeoff in dLLMs. This motivates the design of a new metric that jointly captures efficiency and performance.

Next, we elaborate on how we define our new metric.
A common practice is that, most dLLM methods employ a "threshold" in decoding, where tokens with logits above this threshold can be decoded in parallel. By varying this threshold, we can adjust the quality–speed trade-off of the decoding process and obtain multiple parallelism–accuracy pairs, which can then be used to plot a curve of accuracy versus parallelism. We refer to this curve as the ***accuracy–parallelism curve*** (see the white curve in Figure 1 for an illustration), which characterizes the trade-off between efficiency and performance.

A naïve metric based on this curve is the area under the curve (AUC). However, this is not appropriate, because it is heavily influenced by parallelism even when accuracy degrades significantly, allowing low-quality but fast models to obtain high scores. To address this limitation, we propose ***AUP*** (Accuracy Under Parallelism). AUP quantifies how well a model maintains accuracy as the degree of parallelism increases, providing a unified measure of both the *efficiency* and *performance* of a dLLM.


Formally, let $\mathcal{S} = \{(\rho_i, y_i)\}_{i=1}^m$ be a set of parallelism-accuracy pairs, where $\rho_1 < \rho_2 < \dots < \rho_m$, $\rho_i \in \mathbb{R}^{+}$ denotes the parallelism (measured in _tokens per forward_, TPF), and $y_i \in [0, 100]$ represents accuracy in percentage. We define a minimum accuracy threshold $y_{\min} = y_1 - 5$ to avoid measuring in regimes of significant accuracy degradation. Only points satisfying $y_i \ge y_{\min}$ are included. AUP is then defined as the weighted area under the accuracy-parallelism curve:

$$\operatorname{AUP} \triangleq \rho_1 y_1 + \frac{1}{2} \sum_{i=2}^{m} (\rho_{i} - \rho_{i-1}) \left( y_i \cdot W(y_i) + y_{i-1} \cdot W(y_{i-1}) \right),$$

where the weighting function is defined as $W(y) = \min(e^{-\alpha \left(1 - {y}/{y_\max}\right)}, 1)$, with a penalty factor $\alpha = 3$ and $y_\max$ denotes the highest accuracy achieved on that task. This weight penalizes lower-accuracy regions to emphasize both high parallelism and high performance.

{{< /justify >}}

<!-- {{< image src="img/aup_illustration.png" alt="AUP Illustration" width="50%" title="Figure 1: Illustration of the AUP metric. The metric captures both parallelism (TPF) and accuracy, with a weighting function that penalizes accuracy degradation.">}} -->



<figure>
<div class="responsive-img-grid">
  <img src="img/aup_illustration.png" alt="AUP Illustration" data-width="50">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 1: Illustration of the AUP metric. The metric captures both parallelism (TPF) and accuracy, with a weighting function that penalizes accuracy degradation.</figcaption>
</figure>


{{< justify >}}

**Choice of $\alpha$.** The hyperparameter $\alpha$ controls the penalty for accuracy degradation. A larger $\alpha$ increases sensitivity to performance drops, causing the contribution of throughput to decay exponentially with the error rate. In the ideal case, where a method improves parallelism without compromising accuracy, the AUP reduces to the standard area under the parallelism-accuracy curve (AUC). In our setting, we set $\alpha = 3$ as it balances the importance of parallelism and accuracy.

**Hardware-Independence.** Unlike traditional throughput metrics such as TPS (tokens per second), which are highly dependent on hardware capabilities, AUP offers a more robust and hardware-independent measure. For instance, in our experiments, our d3LLM-LLaDA model (which will be introduced in the next section) demonstrates around 5× higher TPS than an AR baseline (Qwen-2.5-7B-it) on an NVIDIA H100 GPU (289 vs. 57 tokens/s). However, this advantage shrinks significantly on an NVIDIA A100 GPU (175 vs. 50 tokens/s). In contrast, the TPF (tokens per forward pass) and our AUP score remain consistent across hardware platforms. Therefore, AUP provides a robust and fair evaluation metric that reflects both efficiency and accuracy while remaining independent of specific hardware configurations, helping the community focus on algorithmic design without requiring access to particular GPUs.

{{< /justify >}}

{{< justify >}}

**Application of AUP.** We further evaluate previous methods using our AUP metric. As shown in the results in the [Leaderboard](#-diffusion-llm-leaderboard-using-aup-score), although existing dLLM methods achieve higher AUP scores than vanilla Dream and LLaDA, they still lag behind speculative decoding methods (e.g., EAGLE-3). We argue that this does not negate the potential of dLLMs. In the following, we introduce our proposed d3LLM framework, which attains the highest AUP score among all dLLMs.

{{< /justify >}}


## d3LLM: Jointly Achieving Accuracy and Parallelism 🚀


{{< justify >}}
{{< image src="img/example.gif" alt="d3LLM: Ultra-fast diffusion language model" width="100%" title="Figure 2. Demo of our d3LLM, which achieves up to 5× speedup over the AR (Qwen-2.5-7B-it) on H100 GPU and 3.5× speedup on A100 GPU. You can try 🕹️ [our demo](https://d3llm-team.github.io/).">}}

{{< /justify >}}

{{< justify >}}

Following the guidance of the AUP score, we introduce ***d3LLM*** (*pseuDo-Distillated-Diffusion Large Language Model*), a novel framework for constructing dLLMs with both high accuracy and high parallelism. Our d3LLM approach not only improves parallelism but also preserves model performance, with only minimal accuracy degradation compared to standard LLaDA/Dream models.

First, to ***improve parallelism***, we carefully study the behavior of dLLMs and find that the key to high parallelism is enabling the model to unmask multiple tokens at each forward pass. Previous work often overlooks the trajectory in distillation process and typically adopts a single-block decoding strategy. This motivates us to adopt ***trajectory-based distillation*** and ***multi-block decoding*** as two key techniques for improving parallelism. Distillation is used to guide the model to unmask as many tokens as possible, while multi-block decoding is designed to fully exploit the parallel decoding capability of the dLLM.

Second, to ***maintain accuracy***, we find that robustness in both distillation and decoding is crucial. Therefore, we design a ***curriculum learning strategy*** that gradually increases the masking ratio from easier scenarios (few masks) to more difficult ones (many masks) during training, resulting in a more robust distillation process. Moreover, we observe that the multi-block decoding process may cause a performance drop, as the *bidirectional attention* in dLLMs may harm accuracy. This motivates us to design a ***KV-cache refresh*** mechanism to update the KV cache and maintain accuracy.

Together, these techniques enable d3LLM to strike a balance between accuracy and parallelism and to obtain the highest AUP score among all dLLMs. We will introduce details of the d3LLM framework in the following.


{{< /justify >}}


{{< justify >}}

### (i) Pseudo-Trajectory-based Distillation Recipe

{{< image src="img/fig_distillation.png" alt="Distillation Illustration" width="100%" title="Figure 3. Illustration of our pseudo-trajectory-based distillation recipe.">}}

We first introduce our **pseudo-trajectory-based distillation** recipe with curriculum learning, which is crucial for enhancing parallelism while maintaining model accuracy. This advanced distillation recipe aims at improving decoding efficiency and alignment with the teacher model's generation pattern. Specifically, it consists of the following key techniques:

{{< /justify >}}


{{< justify >}}

- **Utilizing the Teacher dLLM's Pseudo-Trajectory (15%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

    A fundamental challenge in distillation is the limited availability of intermediate supervision for the diffusion process, where only prompt-response pairs are accessible, and the teacher model's final response may be suboptimal or incorrect. When a strong teacher dLLM produces a response that exactly matches the ground-truth output, its decoding trajectory can help the student model learn the correct generation order; we refer to this as a _real-trajectory_. However, such real-trajectories are hard to obtainable in practice because the teacher’s response often differs from the ground truth. To address this limitation, we propose using the trajectory produced by the teacher dLLM (denoted as _pseudo-trajectory_, due to it is not the exact diffusion trajectory for the ground-truth response) to guide the student model.

    <div style="margin-top: 2mm;"></div>

    Specifically, given a prompt \$\mathbf{x}\$ and a predefined maximum output length \$n\$, we first let the teacher dLLM to generate and record its own decoding trajectory \$\\{\mathcal{T}_1,\ldots,\mathcal{T}_n\\}\$, where \$\mathcal{T}_i \in \mathbb{R}^n, \forall i \in \\{1,\ldots,n\\}\$. Rather than relying on the content of the teacher's response, we extract only the order in which tokens are decoded. This order forms what we refer to as the **_pseudo-trajectory_** of the teacher.

    <div style="margin-top: 2mm;"></div>
    
    To train the student model, we combine the pseudo-trajectory \$\\{\mathcal{T}_1,\ldots,\mathcal{T}_n\\}\$ with the ground-truth prompt-response pair \$(\mathbf{x}, \mathbf{y})\$ and construct a _noisy sequence_ \$\widetilde{\mathbf{y}} \in \mathbb{R}^n\$ that simulates teacher's intermediate state during the decoding process. Formally, let \$t \in [0, 1]\$ denote the mask ratio, and let \$w = \\{s, s+1, \ldots, s + k\\}\$ be a decoding window of length \$k\$ starting at position \$s\$, the noisy sequence \$\widetilde{\mathbf{y}}\$ is defined as

    $$[\widetilde{\mathbf{y}}]_i= \begin{cases}[\mathbf{y}]_i & \text { if } i \leqslant s \text {  or  }\left[\mathcal{T}_{s+\lceil k t \rceil}\right]_i \neq \texttt{mask}, \\ {\texttt{mask} } & \text { if } i>s+k \text { or }\left[\mathcal{T}_{s+\lceil k t \rceil}\right]_i=\texttt{mask},\end{cases}$$

    where \$\texttt{mask}\$ is the special mask token ID, and \$[\cdot]_i\$ denotes the \$i\$-th token in the trajectory sequence. By training the student dLLM on this noisy input by requiring it to predict the labels of the masked tokens, the model learns to unmask tokens sequentially in a manner aligned with the teacher's decoding order. This leads to smoother and more efficient token generation, yielding a **15% improvement in TPF** compared to strategies that use random masking.

{{< /justify >}}


{{< justify >}}

- **Progressive Noise Level (Further get 18%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

  To preserve accuracy during distillation, we introduce a _progressive noise schedule_ by gradually increasing the mask ratio \$t\$ from 0.0 to 0.8 during the training process. This curriculum learning approach encourages the model to learn from easier to harder decoding scenarios, thereby enhancing its robustness and decoding efficiency while maintaining generation quality. Empirically, this strategy further improves the model's tokens-per-forward (TPF) by approximately **18%** compared to using a fixed mask ratio. Without this curriculum strategy, we observe that the distillation process becomes unstable and the model is more likely to suffer accuracy degradation.

{{< /justify >}}


{{< justify >}}
- **Progressive Window Size (Further 8%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

  We also employ a _progressive window sizing_ as another curriculum learning technique: instead of fixing the decoding window length \$k\$, we gradually increase it from 16 to 32 during the training process. This allows the model to adapt to increasingly larger context spans, facilitating smoother distillation process and stable token generation while maintaining accuracy. This approach leads to an additional **8% improvement in TPF** compared to a constant window size.

{{< /justify >}}


---

{{< justify >}}


### (ii) Multi-Block Decoding Strategy

{{< image src="img/fig_decoding.png" alt="Decoding Illustration" width="100%" title="Figure 4. Illustration of our multi-block decoding strategy with KV-cache and refresh.">}}


In addition to the novel distillation recipe, we also introduce an efficient decoding mechanism tailored for dLLM, designed to maximize parallelism while maintaining generation quality. Our decoding strategy includes the following components:

{{< /justify >}}

{{< justify >}}

- **Entropy-Based Multi-Block Parallel Decoding (20%↑ TPF Improvement)**

    <div style="margin-top: 2mm;"></div>

  Inspired by the approach in [D2F](https://arxiv.org/abs/2508.09192), we propose an _entropy-based multi-block decoding_ method. Unlike conventional diffusion decoding, which operates strictly within a single block, our method enables decoding of both the current and future blocks in parallel. We select tokens to decode based on the entropy threshold, in which lower-entropy (more confident) predictions across blocks are first to be unmasked. 
  
  <div style="margin-top: 2mm;"></div>

  Each block can be in one of five states: **Inactive**, **Activated**, **Fully-Activated**, **Completed but Stabilizing**, and **Completed**. We create a new *Activated* block when its preceding block reaches 10% completion and employ a conservative decoding strategy for this block, generating tokens only when they meet the entropy threshold. When the preceding block reaches 95% completion, the *Activated* block transitions to a *Fully-Activated* state, where a more aggressive strategy is used by decoding at least one token per forward pass regardless of the threshold. Once all tokens in a block are unmasked, the block enters the *Completed but Stabilizing* state, during which we perform forward passes without using the KV cache and refresh previous caches. After 1-2 rounds, the block becomes *Completed*, and we store its KV cache. In addition, we apply a periodic-refresh strategy that updates the KV cache every few rounds. This multi-block decoding strategy increases TPF by **20%**, and the KV-refresh mechanism helps maintain the accuracy.

{{< /justify >}}

{{< justify >}}

- **Multi-Block Decoding with KV-Cache and Refresh (20%↑ TPS under Long Contexts)**

    <div style="margin-top: 2mm;"></div>

  To further improve decoding throughput while maintaining generation quality, particularly in long-context settings, we incorporate a _KV-cache_ mechanism alongside a periodic _KV-refresh_. Specifically, after completing each block, we introduce a short delay before caching its key–value states to ensure that the cache remains reliable and does not lead to performance degradation. Simultaneously, we perform full forward passes to refresh previous caches. This hybrid strategy maintains decoding accuracy while significantly improving TPS by approximately **20%** in long-context scenarios.

{{< /justify >}}

{{< justify >}}

- **Early Stopping on EOS Token (5%↑ TPF Improvement)**

  We implement an **early stopping mechanism** that halts decoding once the end-of-sequence (EOS) token is generated. This simple yet effective optimization reduces unnecessary computation and yields a **5% improvement in TPF** on average.

{{< /justify >}}


{{< justify >}}

## Benchmark Results

We present comprehensive benchmark results across five representative tasks: GSM8K-CoT (chain-of-thought reasoning), MATH (mathematical problem solving), HumanEval (code generation), MBPP (Python programming), and a long-context math reasoning task (5-shot GSM8K reasoning, with a prompt length ≈ 1000). These datasets span diverse domains and problem types and are widely used in the research community. In addition, their relatively long output lengths allow us to effectively evaluate the models' parallel decoding capabilities together with their accuracy.

Our experiments are conducted on three foundational diffusion models: LLaDA, Dream, and Dream-Coder. From these, we derive three distilled models, d3LLM-LLaDA, d3LLM-Dream, and d3LLM-Coder, each trained using the same trajectory-based distillation recipe and multi-block decoding strategy outlined previously. We use a single GPU and fix the batch size to 1 for all models.

**Implementation Details.** Our d3LLM begins with a block diffusion model (either LLaDA or Dream) with a block size of 32 as the teacher model. For fair comparison, we adopt the same distillation dataset as dParallel, which includes approximately 122k samples for Dream and 92k samples for LLaDA, sourced from the PRM12K, AceCode, GSM8K (training split), and Numina-Math datasets. The learning rate is set to 2e-5. We train 6 epochs for LLaDA and 3 for Dream. More implementation details can be found in our [GitHub code](https://github.com/hao-ai-lab/text-diffusion).

{{< /justify >}}

{{< justify >}}

**Results on LLaDA-8B-Instruct Model:** For the foundation model of _LLaDA-8B-Instruct_, we compare our *d3LLM-LLaDA* with _vanilla LLaDA, Fast-dLLM-LLaDA, D2F_, and _dParallel-LLaDA_.

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

**Results on Dream-7B-Instruct Model:** For the foundation model of _Dream-7B-Instruct_, we compare our *d3LLM-Dream* with _vanilla Dream, Fast-dLLM-Dream, Fast-dLLM-v2-7B_, and _dParallel-Dream_.

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
Remarkably, we note that for Fast-dLLM-v2, the accuracy scores on Math and HumanEval are notably higher than those of other diffusion models derived from Dreams. We suspect that this stems from the fact that Fast-dLLM-v2 is finetuned directly from Qwen-2.5-7B with an additional 1B tokens (i.e., the LLaMA–Nemotron post-training dataset). In contrast, our d3LLM-Dream is distilled based on the vanilla Dream and uses only 60M additional tokens.



The experimental results also validate the reliability of our AUP metric. For example, on the MBPP dataset with the LLaDA-based model, although many methods achieve parallelism (TPF) greater than 1, their accuracy degradation compared with the best-performing model (Qwen-2.5-7B-it) is substantial, leading to low overall utility. This demonstrates that the AUP metric more faithfully reflects the practical efficiency–performance trade-off.

{{< /justify >}}


{{< justify >}}

**Efficient Diffusion Coder.** Beyond LLaDA and Dream, we further apply our distillation approach and multi-block decoding method to a more realistic and challenging application: efficient coding models. Specifically, we use _Dream-Coder-7B-Instruct_ as the teacher dLLM and collect 120k samples from the Ling-Coder-SFT and AceCode datasets, along with a small amount of math-reasoning data, to distill our d3LLM-Coder. The results are demonstrated as below.


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


All our distillation code, data, model weights, and benchmark evaluation code are available at [https://github.com/hao-ai-lab/text-diffusion](https://github.com/hao-ai-lab/text-diffusion). The full paper about AUP and our d3LLM framework will be released soon. Stay tuned!

{{< /justify >}}

{{< justify >}}

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
{{< /justify >}}
