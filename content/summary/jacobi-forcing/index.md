+++
title = "JacobiForcing"
date = 2025-12-16T12:00:00-08:00
authors = ["Lanxiang Hu*", "Siqi Kou*", "Yichao Fu", "Samyam Rajbhandari", "Tajana Rosing", "Yuxiong He", "Zhijie Deng", "Hao Zhang"]
author = "Lanxiang Hu*, Siqi Kou*, Yichao Fu, Samyam Rajbhandari, Tajana Rosing, Yuxiong He, Zhijie Deng, Hao Zhang"
ShowReadingTime = true
draft = false 
description = "Fast and Accurate Causal Parallel Decoding using Jacobi Forcing"
github = "hao-ai-lab/JacobiForcing"
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
**TL;DR**: Today’s Best LLMs mostly decode autoregressively from left-to-right, which gives great quality but is terribly slow. Diffusion LLM can decode many tokens in parallel thanks to their non-casual, any-order generation, but they must be trained from scratch, or heavily adapted from autoregressive (AR) checkpoints with a non-casual diffusion objective; we find this mismatch often hurts quality and breaks many effective KV-cache related serving optimizations. This blog introduces Jacobi Forcing, a new training technique that converts LLMs into native casual parallel decoders. Jacobi forcing keeps the casual AR backbone and fixes the AR-to-diffusion mismatch by training the model to handle noisy future blocks along its own Jacobi decoding trajectories. This yields an AR model which behaves like a diffusion-style decoder—decoding multiple tokens per pass, but still from left to right—with up to $4.5\times$ higher tokens-per-forward and $4\times$ wall-clock speedup on coding and math tasks, while retraining near-AR generation quality. 
{{< /justify >}}
