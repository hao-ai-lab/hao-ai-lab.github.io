+++
title = "vLLM-LTR"
description = "Efficient LLM Scheduling by Learning to Rank"
date = 2025-01-13T12:00:00-08:00
authors = ["Yichao Fu", "Siqi Zhu", "Runlong Su", "Aurick Qiao", "Ion Stoica", "Hao Zhang"]
author = "Yichao Fu, Siqi Zhu, Runlong Su, Aurick Qiao, Ion Stoica, Hao Zhang"
ShowReadingTime = true
urlblog = "vllm-ltr"
draft = false
type= "summary"
arxiv = "2408.15792"
github = "hao-ai-lab/vllm-ltr"

[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/vllm-ltr"
[cover]
      image = "/img/llm-ltr-cover.jpg"
      alt = "llm-ltr-cover"
      
+++

{{< socialBadges arxiv-index="2408.15792" github="hao-ai-lab/vllm-ltr" >}}

**TL;DR:** Traditional Large Language Model (LLM) serving systems rely on first-come-first-serve (FCFS) scheduling. When longer requests block shorter ones in the queue, this creates a cascade of delays that severely impacts overall system latency. 
LLM inference jobs are particularly challenging to schedule due to their highly unpredictable workload and variable output lengths. We developed a novel *learning to rank* approach that predicts the relative ranking of output lengths, enabling a more efficient Shortest Job First-like scheduling policy. This scheduling approach reduced chatbot latency by 6.9x in high-load scenarios compared to commonly adopted FCFS scheduling.
