+++
title = "FastLTX-2.3: The Fastest and Cheapest 1080p TI2AV Pipeline Ever"
date = 2026-03-11T12:00:00-08:00
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
draft = false
[cover]
    image = "/img/fastvideo_realtime_1080p/t2v-price.png"
    video = "Main_4.mp4"
    alt = "FastLTX-2.3 text-to-video pricing comparison"
    caption = "FastLTX-2.3 reduces 1080p text-to-video cost dramatically."
    hidden = true
+++

**TL;DR:** If you work in media generation, you know the frustration: an idea pops into your mind, you type a prompt, maybe provide a reference image, and want to see the result immediately – *while the idea is still alive*. But today’s video generation APIs break the loop. You wait minutes for clips, and each costs enough to make your wallet wince. This is exactly the problem we fix.

We introduce FastLTX-2.3, a production-grade 1080p TI2AV pipeline built on top of FastVideo: our unified post-training and inference framework for video diffusion models. We turn [LTX-2.3](https://ltx.io/model/ltx-2-3), a strong text-image-to-audio-video model into a single GPU speed machine for interactive media generation: it generates 5 second 1080p videos with audio with an end-to-end latency of **\~4.5 seconds** on a *single* B200 GPU. For the first time, we optimize 1080p video-audio generation to **\$0.075/min** (estimated), **32x lower** than the next cheapest option.[^cost] We believe this is a major turning point to make high-quality videogen fast and economic enough to feel interactive.[^code]

If you have the *need for speed (and quality)*, try our demo (**TODO: LINK DEMO HERE**) for free.  

### **Comparison Video** 

The biggest bottleneck in video generation is no longer just model quality. It is the *broken feedback loop* during creative iteration.

As a creator, designer, or builder working with AI video models, you don't want just a single generated video, you want to have the ability to explore with multiple generations. You want to try multiple ideas, change the framing, swap the style, adjust the motion, add a reference image, rerun, and keep going until perfection. This is how real creative work happens. But when each attempt takes minutes and hits your budget, the creation loop just collapses\! A tool is not effective if it becomes slower than your imagination\!

For example, generating an 8-second video with Google's Veo-3 Fast takes about 55 seconds and costs around \$1.20. While Veo is an impressive model, it is still too slow and expensive for the rapid iteration that **modern media generation** workflows demand. If **generation is slower than the pace of ideation, then frequent iteration becomes impractical.** 

Recent research efforts have greatly reduced video generation latency, but most of these systems are still limited to 480p, or at best 720p, and often do not produce audio. **1080p is where things get serious.** It is where outputs become much more usable for storytelling, content creation, and real products—but it is also where the systems challenges become much harder. The spatial workload grows dramatically, attention becomes more expensive, memory pressure rises, and every inefficiency in the stack gets amplified. Achieving *interactive latency* at full-HD resolution requires deep optimization of model execution, scheduling, and kernel efficiency.

In this post, we release the fastest and cheapest 1080p TI2AV pipeline ever. We show that 1080p video generation can be made ***interactive*** on a single NVIDIA B200 GPU with [FastVideo](https://github.com/hao-ai-lab/FastVideo). By combining full-stack optimization techniques, we reduced end-to-end latency by **3.9x**, and inference cost by **32x for T2V** and **31.2x for I2V** (see the two figures below).

For the first time, we achieved **\~4.5-second** latency for 5-second video generation at 1088 x 1920 resolution at 24 FPS, costing only \$0.075/min\! We believe this is an important milestone that will truly unblock the potential of video gen in creative and interactive video generation: faster feedback, lower cost, and a much more seamless creative loop. 

## **![][image1]**

## **![][image2]**

### Creating The Speed Demon with the FastVideo Stack

**![][image3]**

**TODO: Insert I2V graph**

Achieving **sub-5-second end-to-end latency** for 1080p generation (see figures above) requires optimization across every layer of the stack: model implementation, kernels, low-bit precision, compilation, scheduling, and even the infrastructure around frame and audio processing. FastLTX-2.3 is specifically optimized for data-center grade **Blackwell GPUs** (**B200/B300**).

**Fast attention for videogen.** Modern video generation models are dominated by attention computation. DiTs rely on 3D spatiotemporal attention so that tokens can exchange information across both space and time. That means attention is one of the biggest consumers of FLOPs in the entire system. We use optimized attention kernels for NVIDIA’s **SM100/SM103** architectures, which gives FastLTX-2.3 the attention engine it needs to keep 1080p generation interactive.

**Aggressive low-precision execution with NVFP4.** Another key feature of Blackwell chips is native Tensor Core support for the NVFP4 data-type. Compared to BF16, NVFP4 offers dramatically higher theoretical throughput, making low-precision execution essential if we want to fully utilize the capabilities of a B200/B200. FastVideo takes advantage of this by using **NVFP4-quantized linear layers** in the DiT while preserving output quality. 

**End-to-End Graph and Kernel Optimization.** We also applied aggressive graph-level and kernel-level optimization throughout the inference pipeline. Using **torch.compile** with the **TorchInductor** backend, along with custom fused kernels where appropriate, we optimized every major stage of generation: prompt encoding, latent preparation, denoising, and decoding. This matters because real latency is not merely the model’s core diffusion denoising loop, but everything around it too. FastVideo treats the entire path as a first-class systems problem.

**System-Level Efficiency Beyond the Model.** To push end-to-end latency below 5 seconds, optimizing the model alone is not enough. We also reduced IPC overhead and streamlined the surrounding I/O pipeline for processing and saving frames and audio. This includes using an optimized ffmpeg build for the target CPU so that media handling does not become a hidden bottleneck.

### Serving Optimizations

The demo is served on a GB200 NVL72 (72x B200), with each GPU acting as a serving replica with load balancing. We also deploy Rust-based middleware for better serving efficiency. Due to capacity limits, we allow each user to interact for a limited amount of time. If you are interested in **dedicated serving capacity** or an **endpoint**, join our [waitlist](https://docs.google.com/forms/d/e/1FAIpQLSePSPqH5ZoLTn88LvLU7C-UM6EVxU3EODCIjhDsNisabFa7JA/viewform?usp=publish-editor).

### Outlook and Conclusion

Running high-quality 1080p video generation on a **single GPU** dramatically simplifies deployment. It eliminates the need for heavyweight techniques such as context parallelism or other multi-GPU serving strategies that are often required for fast video inference. That means lower operational complexity, simpler scaling, and a much cleaner path from research to production.

For creators, it means something even more important: **faster iteration**. You can test more ideas in less time, keep momentum in the creative loop, and actually use video generation as an interactive tool rather than a slow batch job.

For products, it opens the door to entirely new classes of applications in **media generation**, including:

* Rapid creative ideation  
* Interactive storytelling tools  
* Personalized content generation  
* Real-time visual experiences  
* Future on-device and local-generation workflows

And with continued advances in hardware and models, we believe **sub-5-second local generation on consumer devices** is coming sooner than most people expect.

## About FastVideo

**FastVideo** is our unified framework for video diffusion post-training and inference. We build the system layers that turn cutting-edge generative media models into fast, deployable, real-world products. It powers the diffusion backend in the open ecosystems, including SGLang and Dynamo, and [techniques](https://arxiv.org/abs/2505.13389) created in FastVideo have been adopted by top frontier labs to train media generation models.

If you are excited about the future of fast, interactive generative video, **please check out the [FastVideo repo](https://github.com/hao-ai-lab/FastVideo).** We will also be at [NVIDIA GTC 2026](https://www.nvidia.com/gtc/), so keep an eye out for us and come say hi\! 

### Acknowledgement

We thank [NVIDIA](https://www.nvidia.com/en-us/) and [Coreweave](https://www.coreweave.com/) for supporting the development and release of FastLTX-2.3, and [Lightricks](https://www.lightricks.com/) for creating a strong base model and open-sourcing it to the community. We are also excited to announce that NVIDIA's [Dynamo](https://docs.nvidia.com/dynamo/dev/user-guides/diffusion/fastvideo) supports FastVideo as a backend. Stay tuned for more developments as this collaboration continues!  

### The Team

**Core contributors:** Matthew Noto\*, Yechen Xu,\* Junda Su\*, Will Lin\* (\* equal contribution)  
**Contributors:** Shao Duan, Kevin Lin, Minshen Zhang, Wei Zhou  
**Tech leads:** Will Lin, Peiyuan Zhang, Hao Zhang  
**Advisors:** Hao Zhang (corresponding), Danyang Zhuo, Eric Xing, Zhengzhong Liu

[image1]: /img/fastvideo_realtime_1080p/t2v-price.png
[image2]: /img/fastvideo_realtime_1080p/i2v-price.png
[image3]: /img/fastvideo_realtime_1080p/t2v-latency.png

[^code]: Code will be released in the near future.

[^cost]: We estimate video generation cost from GPU rental rates. At the time of this blog post, a B200 instance on Runpod costs \\$4.99/hour. With a T2V latency of 4.5 seconds per 5-second clip, generating 60 seconds of video requires 4.5 × 12 = 54 seconds of GPU time. The cost is therefore \\$4.99 / 3600 × 54 ≈ \\$0.075, or about \\$0.075 per 1-minute video.
