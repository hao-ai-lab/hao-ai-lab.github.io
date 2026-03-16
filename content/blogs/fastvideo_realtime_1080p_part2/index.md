+++
title = "Into The Dreamverse: Interactive Vibe Directing in FastVideo"
date = 2026-03-15T12:00:00-08:00
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
draft = false
contentClass = "post-content-justified"
[cover]
    image = ""
    alt = "Vibed Into the VibeVerse: Interactive Vibe Directing in FastVideo"
    caption = "Vibed Into the VibeVerse: Interactive Vibe Directing in FastVideo"
    hidden = true
+++

{{< socialBadges github="hao-ai-lab/FastVideo" demo="https://1080p.fastvideo.org/" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3sbq6wr37-TozNr4xtGHOoa4byxFXmPg" discord="https://discord.gg/Dm8F2peD3e" >}}

**[TODO: update demo url]**

AI video generation is already good enough to make a clip. But real creative work is not about getting a clip in one shot. It's about *iteration*. An idea appears, you test it: keep the subject, change the camera angle, continue the scene, and try again. The problem is that ideas move faster than generations. **If every attempt takes minutes, the creative loop breaks;** your imagination moves on before the video does.

We think there is a better interface for AI video generation: **Vibe directing**.

Vibe directing is to video what vibe coding is to software. Instead of rewriting giant prompts from scratch, you talk to the system in natural language and steer the video through *fast* revision. Keep the subject, change the background, slow the camera, etc. Rather than jamming everything into a single prompt, iterate with multiple simple prompts.

**[TODO: Insert demo video]**

That only works if the video generation is fast enough. In our FastVideo prototype, **we generated a 5s, 1080p video clip (with audio) in ~4.55 seconds.** In other words, FastVideo generates a clip faster than you can watch it. This capability completely changes the feel of video generation inference; it stops feeling like a passive experience and **starts feeling like directing your own scenes**. A longer 30-second scene unfolds as a chain of these 5-second segments, while the chat window stays open so you can keep directing in real time.

This matters because serious video creation is almost never perfect on the first try. A shot may look wrong. Motion may break halfway through. Characters may drift between frames. In addition, creators may have multiple versions of a scene and want to play them out to determine which is better. In practice, creators are constantly making small adjustments and trying again. When revisions are slow, it's much more difficult to explore many ideas. When the next result comes back almost immediately, it becomes possible to quickly try many ideas rather than just one. Better creative work comes from a faster loop, not just a better model.

We think this is where video generation is going – a way to direct the video as it unfolds. The best systems will not just generate impressive clips. They will let people explore ideas at the speed of their imagination.

That is what real-time vibe directing is all about…

**[TODO: Insert side-by-side gif comparison]**

### For the GPU Enjoyers: FastVideo makes vibe directing real-time.

Vibe-directing is enabled by FastVideo's real-time inference stack: fast attention backends, 4-bit quantization, fused kernels, optimized multi-user serving, and more. It is currently served on an **NVIDIA GB200 NVL72 system**, where each user request is assigned **a single B200 GPU**. With our optimizations, **one B200 is enough to generate a 30-second video with a latency of only ~4 seconds**, enabling an interactive creative loop.

As a unified framework for both post-training and inference, FastVideo provides a simple interface for integrating new models, performing step distillation, and applying heavy inference optimizations. We are also actively working on RTX 5090 support. Stay tuned for this!


### **The Team**

**Core contributors:** [Will Lin*](https://solitarythinker.github.io/), [Matthew Noto*](https://github.com/RandNMR73), [Junda Su*](https://davids048.github.io/), [Yechen Xu*](https://github.com/XOR-op), [Peiyuan Zhang*](https://github.com/jzhang38) (* equal contribution)

**Contributors:** [Shao Duan](https://github.com/shaoxiongduan), [Minshen Zhang](alexzms.github.io), [Loay Rashid](https://github.com/loaydatrain), [Kevin Lin](https://github.com/kevin314)

**UI:** [Tina Mai](https://tinabmai.com/)

**Tech leads:** [Will Lin](https://solitarythinker.github.io/), [Hao Zhang](https://haozhang.ai/)

**Advisors:** [Hao Zhang](https://haozhang.ai/) (corresponding), [Danyang Zhuo](https://danyangzhuo.com/), [Eric Xing](https://www.cs.cmu.edu/~epxing/), [Zhengzhong Liu](https://hunterhector.github.io/)


### **Learn More**

- [FastVideo Documentation](https://haoailab.com/FastVideo/)
- [FastVideo Roadmap for 26Q1](https://github.com/hao-ai-lab/FastVideo/issues/899)
- [Previous Update: Create a 5s 1080p Video in 4.5s with FastVideo on a Single GPU](https://haoailab.com/blogs/fastvideo_realtime_1080p/)

*Note: vibe-directing is not yet pushed to the public branch of FastVideo as we are still cleaning up the code.*
