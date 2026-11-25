+++
title = "vstack"
description = "Formal Verification Toolkit and Skill Stack for Coding Agents in AI Infrastructure"
date = 2026-11-11T12:00:00-08:00
authors = ["Yuxuan Zhang", "Yichao Fu", "Minshen Zhang", "Junda Su", "Yi Sun", "Junda Chen", "Hao Zhang"]
author = "Yuxuan Zhang, Yichao Fu, Minshen Zhang, Junda Su, Yi Sun, Junda Chen, Hao Zhang"
ShowReadingTime = true
draft = false
urlblog = "verifiable-llm-infra"
type = "summary"
github = "hao-ai-lab/vstack"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://x.com/haoailab/status/2120595980659917024"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/vstack"
[cover]
    image = "/blogs/verifiable-llm-infra/img/cover.png"
    alt = "vstack: Towards Verifiable LLM Infra"
    caption = "vstack enforces mechanical anti-reward-hacking gates across TLA+, Apalache, TLAPS, and Z3 for coding agents."
+++

{{< socialBadges github="hao-ai-lab/vstack" x="https://x.com/haoailab/status/2120595980659917024" >}}

{{< justify >}}

**TL;DR:** As coding agents grow capable of building complex AI infrastructure from scratch, the bottleneck shifts from capability to verification. [**vstack**](https://github.com/hao-ai-lab/vstack) is an open-source formal verification toolkit and skill stack for coding agents that enforces mechanical anti-reward-hacking gates—role isolation, hash-pinned specifications, line-anchored models, and mutation/trace validation—across TLA+, Apalache, TLAPS, and Z3. Together with our prototype verifiable serving engine [**VeriServe**](https://github.com/hao-ai-lab/veriserve), `vstack` has uncovered 40+ real concurrency and correctness bugs across SGLang and vLLM. See our [blog post](/blogs/verifiable-llm-infra/) and [verified bug tracker](https://docs.google.com/spreadsheets/d/1FL5QqgzEpja9k8Zw6GmIQDtHveddjj1coHe1MJ3Ts-Q/edit) for details.

{{< /justify >}}
