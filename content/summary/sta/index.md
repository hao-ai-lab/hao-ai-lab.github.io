+++
title = "Fast Video"
date = 2025-02-18T11:00:00-08:00
authors = ["Peiyuan Zhang", "Yongqi Chen", "Runlong Su", "Hangliang Ding", "Ion Stoica", "Zhengzhong Liu", "Hao Zhang"]
author = "Peiyuan Zhang, Yongqi Chen*, Runlong Su*, Hangliang Ding, Ion Stoica, Zhengzhong Liu, Hao Zhang"
ShowReadingTime = true
draft = false
description = "Make Video Generation Faster"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
[cover]
      image = "/img/attn_sliding_gif.gif"
      alt = "STA Sliding visialization"
      caption = "Visualization of 2D Sliding Tile Attention"
      hidden = true
+++

{{< socialBadges arxiv-index="2502.04507" github="hao-ai-lab/FastVideo" >}}

{{< justify >}}
**TL;DR:** Video generation with DiTs is **painfully slow** -- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) takes 16 minutes to generate just a 5-second video on an H100 with FlashAttention3. Our sliding tile attention (STA) slashes this to **5 minutes** with **zero quality loss, no extra training required**.
Specifically, STA accelerates attention alone by **2.8–17x** over FlashAttention-2 and **1.6–10x** over FlashAttention-3. 
With STA and other optimizations, our solution boosts end-to-end generation speed by **2.98×** compared to the FA3 full attention baseline, without quality loss or the need for training. Enabling finetuning unlocks even greater speedups!

👉Try out kernel in our [FastVideo project](https://github.com/hao-ai-lab/FastVideo) project and we'd love to hear what you think!
{{< /justify >}}
