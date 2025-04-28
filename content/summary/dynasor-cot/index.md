+++
title = "Dynasor"
date = 2025-02-16T12:00:00-08:00
authors = ["Yichao Fu", "Junda Chen", "Yonghao Zhuang", "Zheyu Fu", "Ion Stoica", "Hao Zhang"]
author = "Yichao Fu, Junda Chen, Yonghao Zhuang, Zheyu Fu, Ion Stoica, Hao Zhang"
ShowReadingTime = true
draft = false
urlblog = "dynasor-cot"
type= "summary"
arxiv = "2412.20993"
github = "hao-ai-lab/Dynasor"
demo = "https://hao-ai-lab.github.io/demo/dynasor-cot"
description = "Making Reasoning Models More Token-Efficient"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/Dynasor"
[cover]
      image = "/img/dynasor-cot-illustration.jpg"
      alt = "demo-dynasor"
      caption = "Illustration of Dynasor-CoT"
+++

{{< socialBadges arxiv-index="2412.20993" github="hao-ai-lab/Dynasor" demo="https://hao-ai-lab.github.io/demo/dynasor-cot">}}

{{< justify >}}

**TL;DR:** We observe reasoning models often exhibit poor token efficiency: they waste many tokens second-guessing themselves. We develop **Dynasor-CoT**, a certainty-based approach for dynamically allocating inference compute for reasoning models. The intuition is that by probing reasoning models at intermediate steps, we can identify and early terminate problems where they maintain consistently high certainty in their answers. The method is **plug-and-play, requiring no model modifications or training**, but matches baseline accuracy on benchmarks like AMC23, AIME24, and MATH500 while reducing token consumption by 29% dataset-wide and up to 81% for single problems.

🚀👉Try our [demo](https://e4d417385887b7e801.gradio.live) now!

{{< /justify >}}