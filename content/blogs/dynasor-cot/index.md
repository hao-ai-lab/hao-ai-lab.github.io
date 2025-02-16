+++
title = "Reasoning Without Hesitating: More Efficient Chain-of-Thought Through Certainty Probing"
date = 2000-02-16T12:00:00-08:00
authors = ["Yichao Fu", "Junda Chen", "Yonghao Zhuang", "Zheyu Fu", "Ion Stoica", "Hao Zhang"]
author = "Yichao Fu, Junda Chen, Yonghao Zhuang, Zheyu Fu, Ion Stoica, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/Dynasor"
[cover]
      image = "img/dynasor-cot-illustration.jpg"
      alt = "demo-dynasor"
      caption = "Illustration of Dynasor-CoT"
+++

{{< socialBadges arxiv-index="2412.20993" github="hao-ai-lab/Dynasor" >}}

{{< justify >}}

**TL;DR:** Reasoning models often exhibit poor token efficiency, with self-doubt appearing as one significant factor --- models frequently spend excessive tokens verifying already-correct answers. Using our Probe-In-The-Middle technique to analyze model states during reasoning, we propose **Dynasor-CoT**, a certainty-based approach for dynamic reasoning termination. The method is completely training-free and plug-and-play, requiring no model modifications or fine-tuning. We efficiently achieves up to 29\% token reduction while maintaining accuracy across mathematical reasoning tasks like AMC23, AIME24, and MATH500. Try our [demo](https://e4d417385887b7e801.gradio.live) now!

{{< /justify >}}

{{< image src="img/dynasor-cot-demo.gif" alt="dynasor-cot-acc-demo" width="120%" title="Figure 1: [Demo link](https://e4d417385887b7e801.gradio.live) of DeepSeek-R1-Distill-Qwen-7B achieving a $\sim 4.3 \times$ speedup compared to the baseline when using Dynasor-CoT on MATH500">}}


## Self-Doubt in Reasoning LLMs

{{< justify >}}
Recent advances in large language models (LLMs) with long Chain-of-Thought (CoT) reasoning capabilities, such as DeepSeek-R1 and OpenAI o1/o3, have demonstrated remarkable performance on complex tasks (e.g., math and code). However, compared to previous LLMs, these models exhibit markedly lower token efficiency—requiring more tokens to achieve comparable accuracy—as shown in Figure 2.
{{< /justify >}}

{{< image src="img/accuracy_vs_tokens_01.jpg" alt="efficiency_curve" width="40%" title="Figure 2: The token efficiency curve for the traditional model is much steeper than reasoning model.">}}

{{< justify >}}
One major source of this inefficiency stems from our observation that LLMs hesitate, a phenomenon we call self-doubt: models often reach the correct answer early but engage in extended verification behaviors such as double-checking, reassessment, re-verification, and so on. Such self-doubt patterns can lead to significantly increased token consumption. For instance, Figure 3 compares the traditional Qwen-7B model with a reasoning Deepseek-distilled Qwen-7B model on a simple question. While the traditional model reaches its answer in 180 tokens, the reasoning model expends 1K tokens on iterative verification steps but already got the correct answer at token 340.
{{< /justify >}}

{{< image src="img/example-hesitation.png" alt="hesitation" width="70%" title="Figure 3: An example answer from reasoning model (Deepseek-distilled Qwen-2.5 7B) vs traditional model (Qwen-2.5 7B) on one of the problem in MATH500 dataset.">}}

{{< justify >}}
To systematically investigate this phenomenon, we developed a "Probe-In-The-Middle" technique (or "Probe" for short) that extracts the model's intermediate thinking by appending specific prompts such as "Oh, I suddenly got the answer to the whole problem, Final Answer: boxed\{". Figure 4 shows the analysis of the accuracy comparing directly asking vs probing the model. Taking AMC23 as an example, reasoning models frequently arrive at correct answers early (median: 830 tokens), but continue generating unnecessary tokens due to self-doubt (median: 2.7K tokens). This self-doubt phenomenon significantly impacts token efficiency, as models continue reasoning despite having internal confidence in their answers. Our key insight is that LLMs exhibit detectable levels of certainty during their reasoning process, which can be leveraged to determine effective stopping points.
{{< /justify >}}


{{< image src="img/r1_amc_standard_1.png" alt="token-deprivation" width="70%" title="Figure 4: DeepSeek R1 performance on AMC23 and AIME24 (lowest to highest scores over 10 attempts) at varying token budgets. (Left) Standard reasoning with late answer outputs. (Right) Early answer extraction using Probe-In-The-Middle technique, demonstrating equivalent accuracy with 50% token reduction.">}}



{{< justify >}}
To address self-doubt, we propose Dynasor-CoT, a training-free, least intrusive, but simple approach for long CoT reasoning. Our method combines certainty-based heuristics with the probe-in-the-middle technique to dynamically determine termination points. This approach efficiently truncates reasoning chains while maintaining accuracy, demonstrating significant improvements over fixed-token-budget baselines. Notably, it achieves up to 29\% token reduction without compromising accuracy or requiring additional training and introduces no extra latency to the critical reasoning path.
{{< /justify >}}


## Dynasor-CoT: Efficiently Scaling Long Chain-of-Thought Reasoning


{{< justify >}}
We present an efficient reasoning framework Dynasor-CoT for early termination that enhances token-to-accuracy efficiency in long CoT LLM reasoning through three key mechanisms: answer extraction by probe, certainty assessment, and post-generation validation. Figure 5 shows an example of our methods.
{{< /justify >}}


{{< image src="img/dynasor-cot-illustration.jpg" alt="illustration" width="100%" title="Figure 5: Illustration of Dynasor-CoT: (1) Probe-In-The-Middle for answer extraction, (2) early exit based on certainty (case 1), (2) post-generation validation for hesitation words (e.g., wait) (case 3), and (4) continue if not certain enough (case 2)">}}


### Probe-In-The-Middle

{{< justify >}}
Instead of waiting for complete reasoning chains, we introduce strategic interventions called Probe-In-The-Middle (or probe in short) during the generation process. Our approach appends carefully designed guidance at intermediate stages of reasoning to explicitly elicit the model's current answer (e.g., "Oh, I suddenly got the answer to the whole problem, Final Answer: boxed\{"). This method capitalizes on our observation that reasoning LLMs often reach the correct solution before completing their full reasoning chain. When the LLM has already reached its conclusion internally, this early extraction technique significantly reduces computational costs. 
{{< /justify >}}

### Certainty Assessment through Answer Consistency

{{< justify >}}
We implement a dynamic certainty assessment mechanism that monitors the model's outputs at regular intervals (e.g., every 32, 64, or 128 tokens). At each interval, we probe the model to extract and store the current answer, then allow the LLM to continue its generation. Importantly, the subsequent generation remains unaffected by the probing tokens, enabling parallel execution of answer extraction and original generation. When the model produces consistent answers across multiple intervals, we interpret this pattern as an indicator of certainty, following the certaindex approach [Dynasor](https://arxiv.org/abs/2412.20993). This methodology provides a quantitative measure of the model's certainty.
{{< /justify >}}

### Post-generation Validation

{{< justify >}}
We empirically observed DeepSeek-R1 and DeepSeek-Distill models' generations and identified that they generate specific words like "wait" or "hmm" when lacking certainty in their previous generations. Based on this finding, we specifically monitor for these uncertainty indicators following probed answers. Responses containing these indicators are automatically discarded. This validation mechanism works in conjunction with the certainty assessment to create a comprehensive certainty metric. Figure 5 shows an example.
{{< /justify >}}


### Summary

{{< justify >}}
These three components operate synergistically to optimize the token-to-accuracy trade-off. At regular intervals, the framework injects probe words after the current generation to extract the model's answer at that reasoning stage. It then discards answers that exhibit low certainty indicators. Finally, it terminates the process early if answers remain consistent across several consecutive intervals. This approach leverages the model's ability to reach conclusions during intermediate stages while maintaining robust safeguards against premature or uncertain responses. Our method requires no additional training or model changing, making it readily applicable to existing LLM deployments.
{{< /justify >}}

## Certaindex: Generalize To More Reasoning Algorithms

{{< justify >}}
Building upon our exploration of Chain-of-Thought reasoning, where we discovered that measuring model certainty could effectively reduce computational costs, a broader question emerged: Could this approach to quantifying certainty extend beyond a single reasoning algorithm to benefit LLM reasoning more generally?

Our investigation across commonly adopted reasoning algorithms - from Self-Consistency (SC) to Monte Carlo Tree Search (MCTS) - revealed a promising pattern that led to the development of Certaindex. This generalized metric quantifies model certainty across various LLM reasoning methods by leveraging two key indicators: semantic entropy and reward model scores. By serving as a simple yet effective proxy for model confidence, Certaindex enables dynamic resource allocation and informed early termination decisions while maintaining accuracy across different reasoning algorithms.
{{< /justify >}}


## Dynasor: System For LLM Reasoning Built Upon Certaindex

{{< justify >}}
Dynasor is a system optimized for LLM reasoning algorithms, built upon the Certaindex proxy variable. It introduces a reasoning program abstraction to formalize the structure of reasoning tasks. The application runtime handles intra-program scheduling and dynamically allocates resources based on Certaindex statistics, while the system runtime manages request scheduling and prefix cache optimization across multiple programs. The architecture, as shown in the figure 6, enables efficient resource allocation through the interplay between local application components and server-side system management. See our [paper](https://arxiv.org/abs/2412.20993) for more details about Certaindex and Dynasor!
{{< /justify >}}

{{< image src="img/arch-dynasor.png" alt="result-r1" width="100%" title="Figure 6: Architecture of Dynasor">}}


## Evaluation

### Higher Token Efficiency with Dynasor-CoT

{{< justify >}}
We evaluate our certainty-based early termination method Dynasor-CoT against baseline uniform token allocation across multiple scales of distilled DeepSeek models (7B, 14B, and 32B) on mathematical reasoning benchmarks AIME24 and AMC23, and MATH500. Unlike the baseline approach that uniformly increases token budgets, our method enables early termination by monitoring model certainty at various intervals. As illustrated in Figure 6, we evaluate variable probing intervals (32, 64, and so on) represented by distinct colored lines, with a maximum token budget of 16K. For each interval, we vary the early termination parameter N (the required number of consecutive consistent answers), generating different points along each line. All configurations achieve significant token savings, with our approach reducing token usage by up to 29\% while maintaining comparable accuracy to the baseline. For fair comparison, appropriate accuracy thresholds were calibrated to model scale - with 32B models evaluated against stricter thresholds above QwQ levels and reduced thresholds for smaller models - while setting higher targets for simpler tasks where greater accuracy is achievable. For the 10\% of problems where our method achieves the highest token reduction, we observe savings of 34\% on AIME and 53\% on MATH500. This extends further for the top 1\% of problems, where we achieve even more substantial reductions of 53\% on AIME and 81\% on MATH500. These results, particularly the substantial token savings on certain problems (up to 81\% reduction), demonstrate our method's ability to adapt token allocation to different problem types. This variable performance shows the advantage of our dynamic approach over fixed token budgets, as problems vary in their token requirements for reaching solutions.
{{< /justify >}}

{{< image src="img/token_deprivation_comparison_01.jpg" alt="result-main" width="100%" title="Figure 7: Comparing Dynasor-CoT Performance Across Model Scales and Datasets">}}

{{< justify >}}
To validate scalability, we extended our experiments to the larger DeepSeek-R1 model on AIME and AMC datasets (Figure 8). The results align with our findings from smaller distill models, demonstrating consistent efficiency gains: DeepSeek-R1 achieves 12\% token savings on AIME problems and 24\% on AMC problems while maintaining baseline accuracy levels.
{{< /justify >}}

{{< image src="img/token_deprivation_comparison_r1_01.jpg" alt="result-r1" width="100%" title="Figure 8: Applying Dynasor-CoT on DeepSeek-R1">}}

{{< justify >}}
We conduct ablation studies across MATH500, AIME24, and AMC23 using DeepSeek Distill 32B to evaluate our framework's components. Our analysis compares four configurations: (1) baseline (uniform token budget), (2) baseline + probing, (3) our certainty-based early exit without post-generation validation, and (4) our full Dynasor-CoT framework. Results (Figure 9) demonstrate that both the basic probe implementation and the version without validation achieve lower token efficiency compared to our complete framework across all settings.
{{< /justify >}}


{{< image src="img/ablation_study_dynasor_cot_01.jpg" alt="result-r1" width="100%" title="Figure 9: Effectiveness of Components using DeepSeek Distill 32B on mathematic datasets">}}


## Get started
{{< justify >}}
Please see [our paper](https://arxiv.org/abs/2412.20993) for more details. We also invite you to try out [our codebase](https://github.com/hao-ai-lab/Dynasor) and [our demo](https://e4d417385887b7e801.gradio.live)!
{{< /justify >}}


## Acknowledgement

We would like to thank Jishen Zhao, Lanxiang Hu, Vikranth Srivatsa‬, Runlong Su, Peiyuan Zhang, Siqi Zhu, Zhongdongming Dai for providing insightful feedback.

## Citation

```
@article{fu2024efficiently,
  title={Efficiently Serving LLM Reasoning Programs with Certaindex},
  author={Fu, Yichao and Chen, Junda and Zhu, Siqi and Fu, Zheyu and Dai, Zhongdongming and Qiao, Aurick and Zhang, Hao},
  journal={arXiv preprint arXiv:2412.20993},
  year={2024}
}
```
