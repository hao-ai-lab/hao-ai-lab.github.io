+++
title = "Who is the Most Parallel Parallel-Decoder?"
date = 2025-11-21T12:00:00-08:00
authors = ["Yu-Yang Qian", "Junda Su", "Peiyuan Zhang", "Lanxiang Hu", "Peng Zhao", "Hao Zhang"]
author = "Yu-Yang Qian, Junda Su, Peiyuan Zhang, Lanxiang Hu, Peng Zhao, Hao Zhang"
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
      alt = "Who is the Most Parallel Parallel-Decoder?"
      caption = "Who is the Most Parallel Parallel-Decoder?"
      hidden = true
+++

{{< socialBadges github="hao-ai-lab/text-diffusion">}}

## 🏆 Parallel-Decoder Leaderboard

{{< justify >}}

We present a leaderboard that compares different parallel-decoders across five representative benchmark tasks, using the AUP score (which will be described in detail later) as the primary evaluation metric.

{{< /justify >}}

{{< dllm_leaderboard >}}

{{< justify >}}

## Background

Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs. A key advantage of dLLMs is the use of _bidirectional attention_, enabling capabilities such as parallel decoding, error correction, and random-order generation, which are features that are not feasible with AR models. Recently, several closed-source diffusion models, including [Mercury](https://arxiv.org/abs/2506.17298), [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/), and [Seed Diffusion](https://arxiv.org/abs/2508.02193), have demonstrated impressive efficiency and performance, achieving high decoding speeds and competitive results compared to AR models. In contrast, open-source diffusion language models have exhibited significantly lower throughput, often performing even slower than AR models. For example, [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487) achieve only around 20 tokens per second, whereas closed-source dLLMs exceed 1000 tokens per second.
 
With growing interest from the research community, an increasing number of methods have been proposed to accelerate diffusion large language models (dLLMs) [[1](https://arxiv.org/abs/2505.22618), [2](https://arxiv.org/abs/2508.09192), [3](https://arxiv.org/abs/2509.26488), [4](https://arxiv.org/abs/2509.26328), [5](https://arxiv.org/abs/2510.08666)]. Upon carefully examining the behavior of dLLMs, our key observation is that: **dLLMs are inherently parallel-decoders with bidirectional attention.** As a result, many recent efforts to equip AR models with parallel-decoding capabilities [[6](https://arxiv.org/abs/2401.10774), [7](https://arxiv.org/abs/2403.00835)] are closely related to the design principles of dLLMs. This leads to a natural question:

<center><i>

**Who is the most parallel parallel-decoder?**</i>

</center>


Although many acceleration techniques have been proposed, there is still no standardized benchmark or metric for evaluating and comparing their degrees of parallelism. In this work, we focus on models with *native parallelism capabilities*, with diffusion language models as a representative example. The speculative decoding framework falls out of our scope because it depends on different model architectures and requires an additional verification step to ensure token correctness, which introduces more complexity when serving models of different sizes. Consequently, our parallel-decoder leaderboard includes only models with native parallelism capabilities, and speculative decoding is orthogonal to our study and can be further incorporated to achieve additional speedup.


{{< /justify >}}

## AUP: A New Metric for Evaluating Parallel-Decoders

{{< justify >}}

The challenge in defining a measure for parallel decoders primarily arises from their *dependence on hardware*. Because GPU computational capacity varies across platforms, traditional throughput (often measured in tokens per second, TPS) can differ significantly across different devices. To this end, we propose a new metric, **_AUP_** (_Accuracy Under Parallelism_), to quantify how well the accuracy is maintained as the degree of parallelism increases, which jointly measures the *efficiency and performance* of a parallel-decoder in a *device-independent* manner.

Let \$\mathcal{S} = \\{(\rho_i, y_i)\\}\_{i=1}^m\$ be a set of parallelism-accuracy pairs, where \$\rho_1 < \rho_2 < \dots < \rho_m\$, \$\rho_i \in \mathbb{R}^{+}\$ denotes the parallelism (measured in _tokens per forward_, TPF), and \$y_i \in [0, 100]\$ represents accuracy in percentage. We define a minimum accuracy threshold \$y\_{\min} = y_1 - 5\$ to avoid measuring in regimes of significant accuracy degradation. Only points satisfying \$y_i \ge y\_{\min}\$ are included. 

The most naïve approach is to calculate a score as the area under the accuracy–parallelism curve (AUC), but this is not an effective metric. This quantity is strongly influenced by parallelism even when accuracy degrades substantially, allowing low-quality but fast models to obtain high scores. To this end, we establish **AUP**, which takes the accuracy degradation into account, which is defined as the weighted area under the accuracy-parallelism curve:

$$\operatorname{AUP} \triangleq \rho_1 y_1 + \frac{1}{2} \sum_{i=2}^{m} (\rho_{i} - \rho_{i-1}) \left( y_i \cdot W(y_i) + y_{i-1} \cdot W(y_{i-1}) \right),$$

where the weighting function is defined as \$W(y) = \min(e^{-\alpha \left(1 - {y}/{y\_\max}\right)}, 1)\$, with a penalty factor \$\alpha = 3\$ and \$y\_\max\$ denotes the highest accuracy achieved on that task. This weight penalizes lower-accuracy regions to emphasize both high parallelism and stable performance. AUP thus provides a unified measure of decoding quality under increasing parallelism.

{{< /justify >}}

{{< image src="img/aup_illustration.png" alt="AUP Illustration" width="50%" title="Figure 1: Illustration of the AUP metric. The metric captures both parallelism (TPF) and accuracy, with a weighting function that penalizes accuracy degradation.">}}

{{< justify >}}

**Choice of \$\alpha\$.** The hyperparameter \$\alpha\$ controls the penalty for accuracy degradation. A larger \$\alpha\$ increases sensitivity to performance drops, causing the contribution of throughput to decay exponentially with the error rate. In the ideal case, where a method improves parallelism without compromising accuracy, the AUP reduces to the standard area under the parallelism-accuracy curve (AUC). In our setting, we set \$\alpha = 3\$ as it balances the importance of parallelism and accuracy.

**Hardware-Independence.** Unlike traditional throughput metrics such as TPS (tokens per second), which are highly dependent on hardware capabilities, AUP offers a more robust and hardware-independent measure. For instance, in our experiments, our d3LLM-LLaDA model (which will be introduced in the next section) demonstrated around 5× higher TPS than an AR baseline (Qwen-2.5-7B-it) on an NVIDIA H100 GPU (280 vs. 57 tokens/s). However, this advantage shrank significantly on an NVIDIA A100 GPU (180 vs. 50 tokens/s). In contrast, the TPF (tokens per forward pass) remained consistent across hardware platforms. Therefore, AUP provides a robust and fair evaluation metric that reflects both efficiency and accuracy while remaining independent of specific hardware configurations, helping the community focus on algorithmic design without requiring access to particular GPUs.

{{< /justify >}}

## [d3LLM:](../d3llm) the Most Parallel Parallel-Decoder so far 🚀


{{< justify >}}
{{< image src="img/example.gif" alt="d3LLM: Ultra-fast diffusion language model" width="100%" title="Demo of our d3LLM-Dream, which can be 5× faster than the AR (Qwen) on H100 GPU and 3.5× faster on A100 GPU.">}}

{{< /justify >}}

{{< justify >}}

In addition to proposing a new evaluation metric for parallel decoders, we introduce **_d3LLM_** (_dequeued-distillate-diffusion Large Language Model_), a novel recipe for building highly efficient diffusion language models. The d3LLM framework combines two key innovations: a trajectory-based distillation approach and an entropy-based multi-block decoding strategy.

{{< /justify >}}

### (i) Distillation Recipe: Pseudo-Trajectory Distillation


Distillation for dLLMs faces a fundamental challenge: we typically only have access to final prompt-response pairs, without visibility into how the teacher model arrived at its answer through intermediate states. This is particularly problematic because the teacher's decoding trajectory, the order in which it unmasks tokens, contains valuable information about efficient generation patterns. Our key insight is to leverage this trajectory as a form of intermediate supervision.


To this end, we propose leveraging the teacher’s **pseudo-trajectory** to guide the student model. Given a prompt, we first let the teacher diffusion model generate a full output. Instead of using the content of the response, we extract the *dequeuing order*, i.e., the sequence in which the teacher chooses to unmask tokens at each step. This sequence forms a **pseudo-trajectory** that reflects the teacher’s decoding strategy. We then reconstruct noisy sequences that approximate the intermediate states of the teacher. This trajectory-based method alone yields a **15% increase in tokens per forward pass** compared with naive random masking.


We further enhance the distillation recipe with two curriculum learning techniques. First, we use a **progressive noise schedule**, gradually increasing the masking ratio from easy scenarios (few masks) to harder ones (many masks) during training. This curriculum approach helps the model build robust unmasking strategies incrementally, contributing an additional **18% TPF improvement**. Second, we employ a **progressive window sizing**, starting with small decoding windows of 16 tokens and gradually expanding to 32 tokens. This improves another **8% to TPF performance**.


### (ii) Decoding Strategy: Entropy-based Multi-Block Decoding


While our distillation recipe produces an efficient student model, we also need a decoding strategy that fully exploits its parallel generation capabilities. Standard diffusion decoding operates within fixed-size blocks, processing one block at a time. We push this further with **entropy-based multi-block parallel decoding**. This multi-block parallel decoding delivers approximately **20% TPF improvement**. 


For long-context scenarios, we further combine this with a **KV-cache mechanism with periodic refresh**. After completing each block, we delay for several rounds before caching its key-value states, and simultaneously perform full forward passes to refresh previous caches. This hybrid approach maintains generation quality while boosting throughput by roughly **20% in long-context scenarios**. Finally, we implement **early stopping** when the model generates an EOS token, which contributes an additional **5% TPF gain**.



Together, these distillation and decoding innovations enable d3LLM to achieve substantial efficiency gains while maintaining generation quality, making diffusion language models practical for real-world deployment. More details about our d3LLM framework can be found in our [code repository](https://github.com/hao-ai-lab/text-diffusion) and in [this blog post](../d3llm).

## Benchmark Results

{{< justify >}}

We present comprehensive benchmark results across five representative tasks: GSM8K-CoT (chain-of-thought reasoning), MATH (mathematical problem solving), HumanEval (code generation), MBPP (Python programming), and a long-context math reasoning task (5-shot GSM8K reasoning, with a prompt length ≈ 1000). These datasets span diverse domains and problem types and are widely used in the research community. In addition, their relatively long output lengths allow us to effectively evaluate the models' native parallel decoding capabilities.

Our experiments are conducted on three foundational diffusion models: LLaDA, Dream, and Dream-Coder. From these, we derive three distilled models, d3LLM-LLaDA, d3LLM-Dream, and d3LLM-Coder, each trained using the same trajectory-based distillation recipe and multi-block decoding strategy outlined previously. We use a single GPU and fix the batch size to 1 for all models to ensure that hardware factors are controlled and do not influence the comparison.

{{< /justify >}}

{{< justify >}}

**Results on LLaDA-8B-Instruct Model:** For the foundation model of _LLaDA-8B-Instruct_, we compare our *d3LLM-LLaDA* with _vanilla LLaDA, Fast-dLLM-LLaDA, D2F_, and _dParallel-LLaDA_.

{{< /justify >}}

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
.responsive-img-grid img[data-width="39"] {
  width: 39%;
}
.responsive-img-grid img[data-width="38"] {
  width: 38%;
}
.responsive-img-grid img[data-width="30"] {
  width: 30%;
}
.responsive-img-grid img[data-width="31"] {
  width: 31%;
}
.responsive-img-grid img[data-width="30.5"] {
  width: 30.5%;
}
.responsive-img-grid img[data-width="29"] {
  width: 29%;
}
.responsive-img-grid img[data-width="22.5"] {
  width: 22.5%;
}
.responsive-img-grid img[data-width="45"] {
  width: 45%;
}
.responsive-img-grid img[data-width="50"] {
  width: 50%;
}
.responsive-img-grid img[data-width="40"] {
  width: 40%;
}
.responsive-img-grid img[data-width="44"] {
  width: 44%;
}
.responsive-img-grid img[data-width="24"] {
  width: 24%;
}
@media (max-width: 768px) {
  .responsive-img-grid img {
    width: 100% !important;
  }
}
</style>

<figure>
<div class="responsive-img-grid">
  <img src="img/data_llada_aup_curve_gsm8k_cot.png" alt="LLaDA GSM8K-CoT" data-width="30">
  <img src="img/data_llada_aup_curve_humaneval.png" alt="LLaDA HumanEval" data-width="30">
  <img src="img/data_llada_aup_curve_mbpp.png" alt="LLaDA MBPP" data-width="29">
</div>
<div class="responsive-img-grid" style="margin-top: 20px;">
  <img src="img/data_llada_aup_curve_math.png" alt="LLaDA MATH" data-width="30">
  <img src="img/data_llada_aup_curve_long-gsm8k.png" alt="LLaDA Long-GSM8K" data-width="30">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 2: AUP curves for LLaDA-based models across five benchmark tasks (GSM8K-CoT, HumanEval, MBPP, MATH, and Long-GSM8K).</figcaption>
</figure>

<figure>
<div class="responsive-img-grid">
  <img src="img/data_llada_aup_histogram.png" alt="LLaDA AUP Histogram" data-width="45">
  <img src="img/data_llada_aup_radar.png" alt="LLaDA AUP Radar" data-width="40">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 3: AUP scores and radar chart comparing different LLaDA-based methods.</figcaption>
</figure>

{{< justify >}}

**Results on Dream-7B-Instruct Model:** For the foundation model of _Dream-7B-Instruct_, we compare our *d3LLM-Dream* with _vanilla Dream, Fast-dLLM-Dream, Fast-dLLM-v2-7B_, and _dParallel-Dream_.

{{< /justify >}}

<figure>
<div class="responsive-img-grid">
  <img src="img/data_dream_aup_curve_gsm8k_cot.png" alt="Dream GSM8K-CoT" data-width="30">
  <img src="img/data_dream_aup_curve_humaneval_instruct.png" alt="Dream HumanEval_Instruct" data-width="29">
  <img src="img/data_dream_aup_curve_mbpp_instruct.png" alt="Dream MBPP_Instruct" data-width="29">
</div>
<div class="responsive-img-grid" style="margin-top: 20px;">
  <img src="img/data_dream_aup_curve_math.png" alt="Dream MATH" data-width="30">
  <img src="img/data_dream_aup_curve_long-gsm8k.png" alt="Dream Long-GSM8K" data-width="31">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 4: AUP curves for Dream-based models across five benchmark tasks (GSM8K-CoT, HumanEval_Instruct, MBPP_Instruct, MATH, and Long-GSM8K).</figcaption>
</figure>

<figure>
<div class="responsive-img-grid">
  <img src="img/data_dream_aup_histogram.png" alt="Dream AUP Histogram" data-width="50">
  <img src="img/data_dream_aup_radar.png" alt="Dream AUP Radar" data-width="44">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 5: AUP scores and radar chart comparing different Dream-based methods.</figcaption>
</figure>


**Results on Different Models and Datasets.** As shown by the results above, the proposed distillation recipe and multi-block decoding strategy are robust and improve efficiency across various domains. Specifically, our d3LLM achieves the highest AUP score on 9 out of 10 tasks, and accelerates the vanilla LLaDA by approximately 5–10× on TPF across different tasks. 
Remarkably, we note that for Fast-dLLM-v2, the accuracy scores on Math and HumanEval are notably higher than those derived from Dreams. We suspect that this stems from the fact that Fast-dLLM-v2 is finetuned directly from Qwen-2.5-7B with an additional 1B tokens. In contrast, our d3LLM-Dream is distilled based on the vanilla Dream and uses only 60M additional tokens.

The experimental results also validate the reliability of our AUP metric. For example, on the MBPP dataset with the LLaDA model, although many methods achieve parallelism (TPF) greater than 1, their accuracy degradation compared with the best-performing model (Qwen-2.5-7B-it) is substantial, leading to low overall utility, highlighting that the AUP metric better reflects practical efficiency-performance trade-off.




<!-- **Wall-Clock Speed Comparison.** We further evaluate different methods on multiple hardware platforms, including H100 and A100 GPUs, to measure their wall-clock throughput (measured by tokens per second, TPS) and speedup. The results are presented below.


For the *LLaDA-8B-Instruct*, we report speed (TPS) and accuracy on GSM8K-CoT dataset.

{{< table title="Table 1: Performance comparison of d3LLM-LLaDA with baseline methods on GSM8K-CoT." >}}

|                 | H100's TPS | A100's TPS | Acc   |
|-----------------|:----------:|:----------:|:-----:|
| Qwen-2.5-7B-it  | 57.32      | 50.36      | 74.10 |
| LLaDA           | 27.89      | 19.15      | 72.55 |
| Fast-dLLM-LLaDA | 114.29     | 79.14      | 74.68 |
| D2F             | 102.13     | 76.24      | 73.24 |
| dParallel-LLaDA | 172.23     | 105.85     | 72.63 |
| **d3LLM-LLaDA** | **280.97** | **180.23** | **73.10** |

{{< /table >}}


For the *Dream-7B-Instruct*, we again report speed and accuracy on GSM8K-CoT dataset.


{{< table title="Table 2: Performance comparison of d3LLM-Dream with baseline methods on GSM8K-CoT." >}}

|                 | H100's TPS | A100's TPS | Acc   |
|:---------------:|:----------:|:----------:|:-----:|
| Qwen-2.5-7B-it  | 57.32      | 50.36      | 74.10 |
| Dream           | 27.62      | 8.32       | 83.94 |
| Fast-dLLM-Dream | 77.25      | 51.55      | 79.00 |
| Fast-dLLM-v2-7B | 150.01     | 109.68     | 77.48 |
| dParallel-Dream | 168.36     | 80.23      | 82.12 |
| **d3LLM-Dream** | **235.34** | **128.19** | **81.86** |

{{< /table >}}


Across both models, our d3LLM achieves the highest TPS with minimal accuracy degradation. It delivers up to a **4.5× speedup** over autoregressive decoding (Qwen-2.5-7B-it) on H100 GPUs, and approximately **3× speedup** on A100 GPUs. All experiments are conducted using the HuggingFace inference backend. We leave system-level optimizations including GPU kernel fusion and integration with vLLM, to future work for further TPS improvements. -->

**Efficient Diffusion Coder.** Beyond LLaDA and Dream, we further apply our distillation approach and multi-block decoding method to a more realistic and challenging application: an efficient LLM-based coding model. Specifically, we use _Dream-Coder-7B-Instruct_ as the teacher dLLM and collect 120k samples from the Ling-Coder-SFT and AceCode datasets, along with a small amount of math-reasoning data, to distill our d3LLM-Coder. The results are demonstrated as below.


<style>
.responsive-img-grid img[data-width="23"] {
  width: 23%;
}
</style>

<figure>
<div class="responsive-img-grid">
  <img src="img/data_dream_coder_aup_curve_humaneval.png" alt="Dream-Coder HumanEval" data-width="23">
  <img src="img/data_dream_coder_aup_curve_humaneval+.png" alt="Dream-Coder HumanEval+" data-width="23">
  <img src="img/data_dream_coder_aup_curve_mbpp.png" alt="Dream-Coder MBPP" data-width="22.5">
  <img src="img/data_dream_coder_aup_curve_mbpp+.png" alt="Dream-Coder MBPP+" data-width="23">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 6: Evaluation for Coders across four coding benchmarks (HumanEval, HumanEval+, MBPP, MBPP+).</figcaption>
</figure>

<figure>
<div class="responsive-img-grid">
  <img src="img/data_dream_coder_aup_histogram.png" alt="Dream-Coder AUP Histogram" data-width="50">
  <img src="img/data_dream_coder_aup_radar.png" alt="Dream-Coder AUP Radar" data-width="39">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 7: AUP scores and radar chart comparing different Coder-based methods.</figcaption>
</figure>

{{< justify >}}

Our d3LLM-Coder achieves higher TPF and maintains the highest AUP score across all four tasks with negligible performance degradation, accelerating the vanilla Dream-Coder on various coding tasks. All our distillation code, data, model weights, and benchmark evaluation code are available at [https://github.com/hao-ai-lab/text-diffusion](https://github.com/hao-ai-lab/text-diffusion). The full paper about AUP and our d3LLM framework will be released soon. Stay tuned!

{{< /justify >}}

{{< justify >}}
## Reference

{{< /justify >}}


{{< justify >}}
[1] [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)
{{< /justify >}}

{{< justify >}}
[2] [Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](https://arxiv.org/abs/2508.09192)
{{< /justify >}}

{{< justify >}}
[3] [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488)
{{< /justify >}}

{{< justify >}}
[4] [Fast-dLLM v2: Efficient Block-wise Diffusion LLM](https://arxiv.org/abs/2509.26328)
{{< /justify >}}

{{< justify >}}
[5] [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666)
{{< /justify >}}

{{< justify >}}
[6] [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
{{< /justify >}}

{{< justify >}}
[7] [CLLMs: Consistency Large Language Models](https://arxiv.org/abs/2403.00835)
{{< /justify >}}

{{< justify >}}
[8] [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
{{< /justify >}}

{{< justify >}}
[9] [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)
{{< /justify >}}

{{< justify >}}
[10] [Dream-Coder 7B: An Open Diffusion Language Model for Code](https://arxiv.org/abs/2509.01142)
{{< /justify >}}
