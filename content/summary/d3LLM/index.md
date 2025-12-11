+++
title = "d3LLM"
description = "Ultra-Fast Diffusion LLM 🚀"
date = 2025-12-10T12:00:00-08:00
authors = ["Yu-Yang Qian", "Junda Su", "Lanxiang Hu", "Peiyuan Zhang", "Zhijie Deng", "Peng Zhao", "Hao Zhang"]
author = "Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Zhijie Deng, Peng Zhao, Hao Zhang"
ShowReadingTime = true
draft = false 
urlblog = "d3LLM"
type= "summary"
# arxiv = "2403.00835"
github = "hao-ai-lab/d3LLM"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/d3LLM"
[cover]
      image = "/img/dllm_demo.gif"
      alt = "d3LLM: Ultra-Fast Diffusion LLM 🚀"
      caption = "d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation 🚀"

+++

{{< socialBadges github="hao-ai-lab/d3LLM" demo="https://d3llm-team.github.io/" huggingface="https://huggingface.co/d3LLM">}}

{{< justify >}}


**TL;DR:** We introduce an ultra-fast diffusion-based language model framework, named d3LLM (*pseuDo-Distillated Diffusion Large Language Model*), which balances accuracy and parallel decoding through two key innovations: First, we propose a pseudo-trajectory based distillation method that leverages the teacher model’s decoding order, combined with curriculum strategies that progressively increase the noise level and window size. This stabilizes training and improves token-per-forward efficiency. Second, we employ an entropy-based multi-block decoding algorithm with KV-cache and refresh, enabling multiple future blocks to be decoded in parallel while preserving output quality, especially in long-context scenarios. Across LLaDA/Dream backbones on five benchmark datasets, d3LLM consistently achieves the highest AUP scores on 9 of 10 benchmarks and delivers substantial real-world speedups. It attains up to a 5× speedup over AR models (Qwen-2.5-7B-it) on an H100 GPU with minimal accuracy degradation.

{{< /justify >}}