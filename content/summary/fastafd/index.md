+++
title = "FastAFD"
description = "When Does Attention-FFN Disaggregation Pay on Rack-Scale NVLink?"
date = 2026-11-07T12:00:00-08:00
authors = ["Yichao Fu*", "Yuxuan Zhang*", "Minshen Zhang*", "Ruitian Wang", "Junda Chen", "Hao Zhang"]
author = "Yichao Fu*, Yuxuan Zhang*, Minshen Zhang*, Ruitian Wang, Junda Chen, Hao Zhang"
ShowReadingTime = true
draft = false
urlblog = "fastafd"
type = "summary"
arxiv = "2611.07149"
github = "hao-ai-lab/FastAFD"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastAFD"
[cover]
    image = "/blogs/fastafd/img/afd-token-flow.gif"
    alt = "FastAFD: Attention-FFN Disaggregation"
    caption = "FastAFD disaggregates attention servers and MLP/MoE servers with micro-batch ping-pong overlap on rack-scale NVLink."
+++

{{< socialBadges github="hao-ai-lab/FastAFD" arxiv-index="2611.07149" >}}

{{< justify >}}

**TL;DR:** [**FastAFD**](https://github.com/hao-ai-lab/FastAFD) is an open-source Attention–FFN Disaggregation (AFD) serving engine for large Mixture-of-Experts (MoE) models on rack-scale NVLink systems such as NVIDIA Blackwell GB200 NVL72. By separating stateful KV-cache attention servers from stateless expert servers with micro-batch ping-pong overlap, fused N2M/M2N MoE megakernels, and EAGLE-3 speculative decoding support, FastAFD outperforms tuned vLLM wide expert parallelism by up to 1.20× on hybrid-attention models and up to 1.08× with speculative decoding. We also derive a three-term analytical model—memory gain, overlap gain, and disaggregation cost—that predicts AFD throughput within ~3% from colocated profiles. See the [arXiv paper](https://arxiv.org/abs/2611.07149) and [blog post](/blogs/fastafd/) for details.

{{< /justify >}}
