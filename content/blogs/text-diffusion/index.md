+++
title = "Who is the Most Parallel Parallel-Decoder?"
date = 2025-11-21T12:00:00-08:00
authors = ["Yu-Yang Qian", "Junda Su", "Peng Zhao", "Hao Zhang"]
author = "Yu-Yang Qian, Junda Su, Peng Zhao, Hao Zhang"
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
      image = "/img/text-diffusion-demo.gif"
      alt = "d3LLM: Ultra-fast diffusion language model"
      caption = "d3LLM: the Most Parallel Parallel-Decoder"
      hidden = true
+++

{{< socialBadges github="hao-ai-lab/text-diffusion">}}

## 🏆 Parallel-Decoder Leaderboard

{{< justify >}}

We present a leaderboard comparing different parallel-decoding methods across five benchmark tasks. The **AUP score** (which will be detailed later) is the metric.

{{< /justify >}}

{{< dllm_leaderboard >}}

{{< justify >}}

## Introduction

Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs. A key advantage of dLLMs is the use of _bidirectional attention_, enabling capabilities such as parallel decoding, error correction, and random-order generation, which are features that are not feasible with AR models. Recently, several closed-source diffusion models, including [Mercury](https://arxiv.org/abs/2506.17298), [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/), and [Seed Diffusion](https://arxiv.org/abs/2508.02193), have demonstrated impressive efficiency and performance, achieving high decoding speeds and competitive results compared to AR models. In contrast, open-source diffusion language models have exhibited significantly lower throughput, often performing even slower than AR models. For example, [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487) achieve only around 20 tokens per second, whereas closed-source dLLMs exceed 1000 tokens per second.
 
With growing interest from the research community, an increasing number of methods have been proposed to accelerate diffusion large language models (dLLMs) [[1](https://arxiv.org/abs/2505.22618), [2](https://arxiv.org/abs/2508.09192), [3](https://arxiv.org/abs/2509.26488), [4](https://arxiv.org/abs/2509.26328), [5](https://arxiv.org/abs/2510.08666)]. Upon carefully examining the behavior of dLLMs, our key observation is that: **dLLMs are inherently parallel-decoders with bidirectional attention.** As a result, many recent efforts to equip AR models with parallel-decoding capabilities [[6](https://arxiv.org/abs/2401.10774), [7](https://arxiv.org/abs/2403.00835)] are closely related to the design principles of dLLMs. This leads to a natural question:

_Who is the most parallel parallel-decoder?_

Although numerous acceleration techniques have been introduced, there is currently no standardized benchmark or metric to evaluate and compare their degree of parallelism.

{{< /justify >}}

## AUP: A New Metric for Evaluating Parallel-Decoders

{{< justify >}}

The challenge in defining a measure for parallel decoders primarily arises from their *dependence on hardware*. Because GPU computational capacity varies across platforms, traditional throughput (often measured in tokens per second, TPS) can differ significantly across different devices. To this end, we propose a new metric, **_AUP_** (_Accuracy Under Parallelism_), to quantify how well the accuracy is maintained as the degree of parallelism increases, which jointly measures the *efficiency and performance* of a parallel-decoder in a *device-independent* manner.

Let \$\mathcal{S} = \\{(\rho_i, y_i)\\}\_{i=1}^m\$ be a set of parallelism-accuracy pairs, where \$\rho_1 < \rho_2 < \dots < \rho_m\$, \$\rho_i \in \mathbb{R}^{+}\$ denotes the parallelism (measured in _tokens per forward_, TPF), and \$y_i \in [0, 100]\$ represents accuracy in percentage. We define a minimum accuracy threshold \$y\_{\min} = y_1 - 5\$ to avoid measuring in regimes of significant accuracy degradation. Only points satisfying \$y_i \ge y\_{\min}\$ are included. The AUP is then defined as the weighted area under the accuracy-parallelism curve:

$$\operatorname{AUP} \triangleq \rho_1 y_1 + \frac{1}{2} \sum_{i=2}^{m} (\rho_{i} - \rho_{i-1}) \left( y_i \cdot W(y_i) + y_{i-1} \cdot W(y_{i-1}) \right),$$

where the weighting function is defined as \$W(y) = \min(e^{-\alpha \left(1 - {y}/{y\_\max}\right)}, 1)\$, with a penalty factor \$\alpha = 3\$ and \$y\_\max\$ denotes the highest accuracy achieved on that task. This weight penalizes lower-accuracy regions to emphasize both high parallelism and stable performance. AUP thus provides a unified measure of decoding quality under increasing parallelism.

{{< /justify >}}

{{< image src="img/aup_illustration.png" alt="AUP Illustration" width="50%" title="Figure 1: Illustration of the AUP metric. The metric captures both parallelism (TPF) and accuracy, with a weighting function that penalizes accuracy degradation.">}}

{{< justify >}}

**Remark 1 (Choice of \$\alpha\$).** The hyperparameter \$\alpha\$ controls the penalty for accuracy degradation. A larger \$\alpha\$ increases sensitivity to performance drops, causing the contribution of throughput to decay exponentially with the error rate. In the ideal case, where a method improves parallelism without compromising accuracy, the AUP reduces to the standard area under the parallelism-accuracy curve (AUC). In our setting, we set \$\alpha = 3\$.

**Remark 2 (Hardware-Independence).** Unlike traditional throughput metrics such as TPS (tokens per second), which are highly dependent on hardware capabilities, AUP offers a more robust and hardware-independent measure. For instance, in our experiments, our d3LLM-LLaDA model (which will be introduced in the next section) demonstrated around 5× higher TPS than an AR baseline (Qwen 2.5 7B it) on an NVIDIA H100 GPU (280 vs. 57 tokens/s). However, this advantage shrank significantly on an NVIDIA A100 GPU (180 vs. 50 tokens/s). In contrast, the TPF (tokens per forward pass) remained consistent across hardware platforms. Therefore, AUP provides a robust and fair evaluation metric that reflects both efficiency and accuracy while remaining independent of specific hardware configurations, helping the community focus on algorithmic design without requiring access to particular GPUs.

{{< /justify >}}

## d3LLM: the Most Parallel Parallel-Decoder so far 🚀


{{< image src="img/example.gif" alt="d3LLM: Ultra-fast diffusion language model" width="100%" title="d3LLM: the Most Parallel Parallel-Decoder">}}

{{< justify >}}

In addition to proposing a new evaluation metric for parallel-decoders, we introduce a novel recipe for building a highly efficient and high-performing diffusion language model: **_d3LLM_** (_dequeued-distillate-diffusion Large Language Model_). The d3LLM framework comprises two key components: _distillation_ and _decoding_. Each plays a critical role in enhancing the model's parallelism and efficiency.

{{< /justify >}}

### (i) Trajectory-based Distillation Recipe

{{< justify >}}

We propose a novel **trajectory-based distillation** recipe, which introduces an advanced distillation recipe aimed at improving both decoding efficiency and alignment with the teacher model's generation pattern. Specifically, it consists of the following key techniques:

{{< /justify >}}

{{< justify >}}

**Utilizing the Teacher dLLM's Pseudo-Trajectory (15%↑ TPF Improvement)**

A fundamental challenge in distillation is the limited availability of intermediate supervision, where only prompt-response pairs are accessible, and the teacher model's final response may be suboptimal or incorrect. To address this, we propose leveraging the trajectory generated by the teacher dLLM to guide the student model.

Specifically, given a prompt \$\mathbf{x}\$ and a predefined maximum output length \$n\$, we first let the teacher dLLM to generate and record its own decoding trajectory \$\\{\mathcal{T}_1,\ldots,\mathcal{T}_n\\}\$, where \$\mathcal{T}_i \in \mathbb{R}^n, \forall i \in \\{1,\ldots,n\\}\$. Rather than relying on the content of the teacher's response, we extract only the order in which tokens are **_dequeued_**, that is, the sequence in which masked tokens are predicted and revealed. This order forms what we refer to as the **_pseudo-trajectory_** of the teacher.

To train the student model, we combine the pseudo-trajectory \$\\{\mathcal{T}_1,\ldots,\mathcal{T}_n\\}\$ with the ground-truth prompt-response pair \$(\mathbf{x}, \mathbf{y})\$ and construct a _noisy sequence_ \$\widetilde{\mathbf{y}} \in \mathbb{R}^n\$ that simulates teacher's intermediate state during the decoding process. Formally, let \$t \in [0, 1]\$ denote the mask ratio, and let \$w = \\{s, s+1, \ldots, s + k\\}\$ be a decoding window of length \$k\$ starting at position \$s\$, the noisy sequence \$\widetilde{\mathbf{y}}\$ is defined as

$$[\widetilde{\mathbf{y}}]_i= \begin{cases}[\mathbf{y}]_i & \text { if } i \leqslant s \text {  or  }\left[\mathcal{T}_{s+\lceil k t \rceil}\right]_i \neq \texttt{mask}, \\ {\texttt{mask} } & \text { if } i>s+k \text { or }\left[\mathcal{T}_{s+\lceil k t \rceil}\right]_i=\texttt{mask},\end{cases}$$

where \$\texttt{mask}\$ is the special mask token ID, and \$[\cdot]_i\$ denotes the \$i\$-th token in the trajectory sequence. By training the student dLLM on this noisy input by requiring it to predict the labels of the masked tokens, the model learns to unmask tokens sequentially in a manner aligned with the teacher's decoding order. This leads to smoother and more efficient token generation, yielding a **15% improvement in TPF** compared to strategies that use random masking.

{{< /justify >}}

{{< justify >}}

**Progressive Noise Level (Further get 18%↑ TPF Improvement)**

Rather than applying a fixed masking ratio throughout training, we introduce a _progressive noise schedule_ by gradually increasing the mask ratio \$t\$ from 0.0 to 0.8 during the training process. This dynamic adjustment encourages the model to learn from easier to harder decoding scenarios in a curriculum-like manner, thereby enhancing its robustness and decoding efficiency. Empirically, this strategy further improves the model's tokens-per-forward (TPF) by approximately **18%** compared to using a fixed mask ratio.

{{< /justify >}}

{{< justify >}}

**Progressive Window Size (Further 8%↑ TPF Improvement)**

Inspired by [CLLMs](https://arxiv.org/abs/2403.00835), we also employ a _progressive window size_ during training: instead of fixing the decoding window length \$k\$, we gradually increase it from 16 to 32 during the training process. This allows the model to adapt to increasingly larger context spans, facilitating smoother distillation process and stable token generation. This approach leads to an additional **8% improvement in TPF** compared to a constant window size.

{{< /justify >}}

### (ii) Multi-Block Decoding Strategy

{{< justify >}}

In addition to the novel distillation recipe, we also introduce an efficient decoding mechanism tailored for dLLM, designed to maximize parallelism across multiple-block decoding. Our decoding strategy includes the following components:

{{< /justify >}}

{{< justify >}}

**Entropy-Based Multi-Block Parallel Decoding (20%↑ TPF Improvement)**

Inspired by the approach in [D2F](https://arxiv.org/abs/2508.09192), we propose an _entropy-based multi-block decoding_ method. Unlike conventional diffusion decoding, which operates strictly within a single block, our method enables decoding of both the current and future blocks in parallel. We select tokens to decode based on entropy threshold, in which lower-entropy (more confident) predictions across blocks are first to be unmasked. This strategy significantly enhances decoding efficiency and increases TPF by approximately **20%**.

{{< /justify >}}

{{< justify >}}

**Multi-Block Decoding with KV-Cache and Refresh (20%↑ TPS under Long Contexts)**

To further improve decoding throughput, particularly in long-context settings, we incorporate a _KV-cache_ mechanism alongside a periodic _KV-refresh_. Specifically, after completing each block, we delay for 1–2 iterations to store the block's key-value cache. Simultaneously, we perform a full forward pass without caching to refresh all prior KV caches. This hybrid strategy maintains decoding accuracy while significantly improving TPS by approximately **20%** in long-context scenarios.

{{< /justify >}}

{{< justify >}}

**Early Stopping on EOS Token (5%↑ TPF Improvement)**

We implement an **early stopping mechanism** that halts decoding once the end-of-sequence (EOS) token is generated. This simple yet effective optimization reduces unnecessary computation and yields a **5% improvement in TPF** on average.

{{< /justify >}}

## Benchmark Results

{{< justify >}}

By combining our distillation recipe with the proposed decoding strategy, our d3LLM framework surpasses previous state-of-the-art methods on both TPF and TPS, without sacrificing accuracy. The results are presented below.

**Implementation Details.** Our approach begins with a semi-autoregressive diffusion model (either LLaDA or Dream) with a block size of 32 as the teacher model. This setup enforces causal dependencies between blocks while enabling parallel decoding of multiple tokens within each block. For fair comparison, we adopt the same distillation dataset as dParallel, which includes approximately 122k samples for Dream and 92k samples for LLaDA, sourced from the PRM12K, AceCode, GSM8K (training split), and Numina-Math datasets. The learning rate is set to 2e-5. We train six epochs for LLaDA and three for Dream. More implementation details can be found in our [code](https://github.com/hao-ai-lab/text-diffusion).

**Benchmark Datasets.** We present comprehensive benchmark results across five representative tasks: GSM8K-CoT (chain-of-thought reasoning), MATH (mathematical problem solving), HumanEval (code generation), MBPP (Python programming), and a long-context math reasoning task (5-shot GSM8K reasoning, with a prompt length ≈ 1000). These evaluations assess the performance of our proposed model, d3LLM, against state-of-the-art diffusion-based language models using three key metrics: tokens per forward (TPF), accuracy, and our proposed Accuracy Under Parallelism (AUP).

Our experiments are conducted on three foundational diffusion models: LLaDA, Dream, and Dream-Coder. From these, we derive three distilled models, d3LLM-LLaDA, d3LLM-Dream, and d3LLM-Coder, each trained using the same trajectory-based distillation recipe and multi-block decoding strategy outlined previously. All experiments were performed on NVIDIA H100 GPUs.

{{< /justify >}}

{{< justify >}}

**Results on LLaDA-8B-Instruct Model**

For **_LLaDA-8B-Instruct_** model, we compare our *d3LLM-LLaDA* with _vanilla LLaDA, Fast-dLLM-LLaDA, D2F_, and _dParallel-LLaDA_.

{{< /justify >}}

<figure>
<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
  <img src="img/data_llada_aup_curve_gsm8k_cot.png" alt="LLaDA GSM8K-CoT" style="width: 30%; height: auto;">
  <img src="img/data_llada_aup_curve_humaneval.png" alt="LLaDA HumanEval" style="width: 30%; height: auto;">
  <img src="img/data_llada_aup_curve_long-gsm8k.png" alt="LLaDA Long-GSM8K" style="width: 30%; height: auto;">
</div>
<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: wrap; margin-top: 20px;">
  <img src="img/data_llada_aup_curve_math.png" alt="LLaDA MATH" style="width: 30%; height: auto;">
  <img src="img/data_llada_aup_curve_mbpp.png" alt="LLaDA MBPP" style="width: 30%; height: auto;">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 2: AUP curves for d3LLM-LLaDA across five benchmark tasks (GSM8K-CoT, HumanEval, Long-GSM8K, MATH, MBPP).</figcaption>
</figure>

{{< two_images src1="img/data_llada_aup_histogram.png" src2="img/data_llada_aup_radar.png" alt1="LLaDA AUP Histogram" alt2="LLaDA AUP Radar" width1="45%" width2="45%" title="Figure 3: AUP histogram and radar chart comparing d3LLM-LLaDA with baseline methods.">}}

{{< justify >}}

**Results on Dream-7B-Instruct Model**

For **_Dream-7B-Instruct_** model, we compare our *d3LLM-Dream* with _vanilla Dream, Fast-dLLM-Dream, Fast-dLLM-v2-7B_, and _dParallel-Dream_.

{{< /justify >}}

<figure>
<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
  <img src="img/data_dream_aup_curve_gsm8k_cot.png" alt="Dream GSM8K-CoT" style="width: 30%; height: auto;">
  <img src="img/data_dream_aup_curve_humaneval_instruct.png" alt="Dream HumanEval_Instruct" style="width: 30%; height: auto;">
  <img src="img/data_dream_aup_curve_long-gsm8k.png" alt="Dream Long-GSM8K" style="width: 30%; height: auto;">
</div>
<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: wrap; margin-top: 20px;">
  <img src="img/data_dream_aup_curve_math.png" alt="Dream MATH" style="width: 30%; height: auto;">
  <img src="img/data_dream_aup_curve_mbpp_instruct.png" alt="Dream MBPP_Instruct" style="width: 30%; height: auto;">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 4: AUP curves for d3LLM-Dream across five benchmark tasks (GSM8K-CoT, HumanEval_Instruct, Long-GSM8K, MATH, MBPP_Instruct).</figcaption>
</figure>

{{< two_images src1="img/data_dream_aup_histogram.png" src2="img/data_dream_aup_radar.png" alt1="Dream AUP Histogram" alt2="Dream AUP Radar" width1="50%" width2="45%" title="Figure 5: AUP histogram and radar chart comparing d3LLM-Dream with baseline methods.">}}

{{< justify >}}

**Results on Different Models and Datasets.** As shown by the results above, the proposed distillation recipe and multi-block decoding strategy are robust and improve efficiency across various domains. Specifically, our d3LLM achieves the highest AUP score on 9 out of 10 tasks, and accelerates the vanilla LLaDA by approximately 5–10× on TPF across different tasks. The experimental results also validate the reliability of our AUP metric. For example, on the MBPP dataset with the LLaDA model, although many methods achieve parallelism (TPF) greater than 1, their accuracy degradation compared with the best-performing model (Qwen-2.5-7B-it) is substantial, leading to low overall utility. Remarkably, we note that for Fast-dLLM-v2, the accuracy scores on Math and HumanEval are notably higher than those of other diffusion models derived from Dreams. We suspect that this stems from the fact that Fast-dLLM-v2 is finetuned directly from Qwen-2.5-7B with an additional 1B tokens (i.e., the LLaMA–Nemotron post-training dataset). In contrast, our d3LLM-Dream is distilled based on the vanilla Dream and uses only 60M additional tokens.

{{< /justify >}}

{{< justify >}}

**Wall-Clock Speed Comparison.** We further evaluate different methods on multiple hardware platforms, including H100 and A100 GPUs, to measure their wall-clock throughput (measured by tokens per second, TPS) and speedup. The results are presented below.

{{< /justify >}}

{{< justify >}}

For the *LLaDA-8B-Instruct*, we report speed (TPS) and accuracy on GSM8K-CoT dataset.

{{< /justify >}}

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

{{< justify >}}

For the *Dream-7B-Instruct*, we again report speed and accuracy on GSM8K-CoT dataset.

{{< /justify >}}

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

{{< justify >}}

Across both models, our d3LLM achieves the highest TPS with minimal accuracy degradation. It delivers up to a **4.5× speedup** over autoregressive decoding (Qwen-2.5-7B-it) on H100 GPUs, and approximately **3× speedup** on A100 GPUs. All experiments are conducted using the HuggingFace inference backend. We leave system-level optimizations including GPU kernel fusion and integration with vLLM, to future work for further TPS improvements.

{{< /justify >}}

{{< justify >}}

**Efficient Diffusion Coder.** Beyond LLaDA and Dream, we further apply our distillation approach and multi-block decoding method to a more realistic and challenging application: an efficient LLM-based coding model. Specifically, we use _Dream-Coder-7B-Instruct_ as the teacher dLLM and collect 120k samples from the Ling-Coder-SFT and AceCode datasets, along with a small amount of math-reasoning data, to distill our d3LLM-Coder. The results are demonstrated as below.

{{< /justify >}}

<figure>
<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
  <img src="img/data_dream_coder_aup_curve_humaneval.png" alt="Dream-Coder HumanEval" style="width: 23%; height: auto;">
  <img src="img/data_dream_coder_aup_curve_humaneval+.png" alt="Dream-Coder HumanEval+" style="width: 23%; height: auto;">
  <img src="img/data_dream_coder_aup_curve_mbpp.png" alt="Dream-Coder MBPP" style="width: 23%; height: auto;">
  <img src="img/data_dream_coder_aup_curve_mbpp+.png" alt="Dream-Coder MBPP+" style="width: 23%; height: auto;">
</div>
<figcaption style="text-align: center; color: #808080; margin-top: 10px;">Figure 6: Evaluation for d3LLM-Coder across four coding benchmarks (HumanEval, HumanEval+, MBPP, MBPP+).</figcaption>
</figure>

{{< two_images src1="img/data_dream_coder_aup_histogram.png" src2="img/data_dream_coder_aup_radar.png" alt1="Dream-Coder AUP Histogram" alt2="Dream-Coder AUP Radar" width1="50%" width2="40%" title="Figure 7: AUP histogram and radar chart comparing d3LLM-Coder with baseline methods.">}}

{{< justify >}}

Our d3LLM-Coder achieves higher TPF and maintains the highest AUP score across all four tasks with negligible performance degradation, accelerating the vanilla Dream-Coder on various coding tasks. All our distillation code, data, model weights, and benchmark evaluation code are available at [https://github.com/hao-ai-lab/text-diffusion](https://github.com/hao-ai-lab/text-diffusion). The full paper about AUP and our d3LLM framework will be released soon. Stay tuned!

{{< /justify >}}

## Reference

{{< justify >}}

[1] [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)

[2] [Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](https://arxiv.org/abs/2508.09192)

[3] [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488)

[4] [Fast-dLLM v2: Efficient Block-wise Diffusion LLM](https://arxiv.org/abs/2509.26328)

[5] [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666)

[6] [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)

[7] [CLLMs: Consistency Large Language Models](https://arxiv.org/abs/2403.00835)

[8] [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)

[9] [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)

[10] [Dream-Coder 7B: An Open Diffusion Language Model for Code](https://arxiv.org/abs/2509.01142)

{{< /justify >}}