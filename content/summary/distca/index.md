+++
title = "DistCA"
description = "Core Attention Disaggregation for Efficient Long-context Language Model Training"
date = 2025-12-21T12:00:00-08:00
authors = ["Yonghao Zhuang", "Junda Chen", "Bo Pang", "Yi Gu", "Yibo Zhu", "Yimin Jiang", "Ion Stoica", "Eric Xing", "Hao Zhang"]
author = "Yonghao Zhuang, Junda Chen, Bo Pang, Yi Gu, Yibo Zhu, Yimin Jiang, Ion Stoica, Eric Xing, Hao Zhang"
ShowReadingTime = true
draft = false
urlblog = "distca"
type= "summary"
arxiv = "2510.18121"
github = "hao-ai-lab/distca"
[cover]
    image = "/img/distca.gif"
    alt = "DistCA"
    caption = "DistCA eliminates core attention imbalance in long-context LLM training by disaggregates core attention from other components and treats core attention as an individual unit of work."

+++

{{< socialBadges arxiv-index="2401.09670" >}}

{{< justify >}}

**TL;DR:** Workload imbalance is one of the major problems in training long-context LLM models. Imbalance among data parallel (DP) and pipeline parallel (PP) workers introduces stragglers or bubbles that causes severe slowdown, and the problem becomes more severe as we scale to longer context lengths or more GPUs.

In this blog post, we show how core attention disaggregation can fundamentally eliminate the imbalance and achieve near-linear scaling for long-context LLM training. We also build a system prototype [**DistCA**](https://github.com/hao-ai-lab/DistCA), which achieves up to 1.35× speedup over state-of-the-art training systems.

{{< /justify >}}
