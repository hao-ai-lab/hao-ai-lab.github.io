+++
title = "MuxServe"
date = 2024-05-20T12:00:00-08:00
authors = ["Jiangfei Duan", "Runyu Lu", "Haojie Duanmu", "Xiuhong Li", "Xingcheng Zhang", "Dahua Lin", "Ion Stoica", "Hao Zhang"]
author = "Jiangfei Duan, Runyu Lu, Haojie Duanmu, Xiuhong Li, Xingcheng Zhang, Dahua Lin, Ion Stoica, Hao Zhang"
ShowReadingTime = true
draft = false
type= "summary"
description = "Serving Multiple LLMs with Flexible Spatial-Temporal Multiplexing"
urlblog = "muxserve"
[cover]
      image = "/img/muxserve_cover.gif"
      alt = "MuxServe"
      caption = "The workflow of serving 2 LLMs with flexible spatal-temporal multiplexing."
      
+++

{{< socialBadges arxiv-index="2404.02015" github="hao-ai-lab/MuxServe" >}}

{{< justify >}}

**TL;DR:** Efficiently serving *multiple* LLMs have emerged as a crucial and time-sensitive demand within the community, especially for LLM endpoint providers. In this blog, we show that the dynamic popularity of LLMs and the unbalanced resource utilization of LLM inference can be leveraged to achieve high GPU utilization and reduce serving cost. We introduce MuxServe, a novel serving system that efficiently serves multiple LLMs with flexible spatial-temporal multiplexing. MuxServe outperforms the spatial partitioning and temporal multiplexing baselines by up to $1.8\times$ in throughput and up to $2.9\times$ in SLO attainment on synthetic workloads.

{{< /justify >}}
