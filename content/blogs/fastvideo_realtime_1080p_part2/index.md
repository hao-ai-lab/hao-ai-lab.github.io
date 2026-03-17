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
    alt = "Into the Dreamverse: Vibe Directing in FastVideo"
    caption = "Into the Dreamverse: Vibe Directing in FastVideoo"
    hidden = true
+++

{{< socialBadges github="hao-ai-lab/FastVideo" demo="https://1080p.fastvideo.org/" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3sbq6wr37-TozNr4xtGHOoa4byxFXmPg" discord="https://discord.gg/Dm8F2peD3e" >}}

**TL;DR:** Real-time video generation enables Dreamverse, a prototype for a new interface where users can vibe-direct their own “multiverse” of videos. 

**[TODO: Insert demo video]**

AI video generation is already good enough to make a convincing clip. But real creative work is not about getting a clip in one shot. It’s about iteration. An idea appears, you test it: keep the subject, change the camera angle, continue the scene, and try again. The problem is that ideas move faster than generations. **If every attempt takes minutes, the creative loop breaks**; your imagination moves on before the video does.

We think there is a better interface for AI video generation, which is why we created Dreamverse, an interface that enables a new workflow called vibe-directing. 

Vibe directing is to video what vibe coding is to software. Instead of rewriting giant prompts from scratch, you talk to the system in natural language and steer the video through *fast* revision. Keep the subject, change the background, slow the camera, etc. Rather than jamming everything into a single prompt, iterate with multiple simple prompts. 

This kind of workflow is only possible when video generation is fast enough. Current video generation models like Sora take 1-2 minutes to generate a 5 s 1080p clip. We can do it in [**~4.55 seconds**](https://haoailab.com/blogs/fastvideo_realtime_1080p/). In other words, our inference stack in FastVideo can generate a clip faster than you can watch it. This capability completely changes the feel of video generation inference; it stops feeling like a passive experience and **starts feeling like directing your own scenes**. This allows us to create a longer 30-second scene that unfolds as a chain of these 5-second clips, while the chat window stays open so you can keep directing in real time.


This matters because serious video creation is almost never perfect on the first try. A shot may look off. Motion may break halfway through. Characters may drift between frames. In addition, creators may have multiple versions of a scene and want to play them out to determine which is better. In practice, creators are constantly making small adjustments and trying again. When revisions are slow, it’s much more difficult to explore many ideas. However, when the next result comes back almost immediately, it becomes possible to quickly try many ideas rather than just one. Better creative work comes from a faster loop, not just a better model.

We think this is where video generation is going: a way to direct the video as it unfolds. The best systems will not just generate impressive clips. They will let people explore ideas at the speed of their imagination.
That is what vibe directing is all about. Step in the Dreamverse today with our free demo…

**[TODO: Insert side-by-side gif comparison]**

### **The Team**

**Core contributors:** [Will Lin*](https://solitarythinker.github.io/), [Matthew Noto*](https://github.com/RandNMR73), [Junda Su*](https://davids048.github.io/), [Yechen Xu*](https://github.com/XOR-op), [Peiyuan Zhang*](https://github.com/jzhang38) (* equal contribution)

**Contributors:** [Shao Duan](https://github.com/shaoxiongduan), [Minshen Zhang](alexzms.github.io), [Loay Rashid](https://github.com/loaydatrain), [Kevin Lin](https://github.com/kevin314)

**UI:** [Tina Mai](https://tinabmai.com/)

**Tech leads:** [Will Lin](https://solitarythinker.github.io/), [Hao Zhang](https://haozhang.ai/)

**Advisors:** [Hao Zhang](https://haozhang.ai/) (corresponding), [Danyang Zhuo](https://danyangzhuo.com/), [Eric Xing](https://www.cs.cmu.edu/~epxing/), [Zhengzhong Liu](https://hunterhector.github.io/)


### **Learn More**

- [FastVideo Documentation](https://haoailab.com/FastVideo/)
- [FastVideo Roadmap for 26Q1](https://github.com/hao-ai-lab/FastVideo/issues/899)

*Note: The Dreamverse is not yet pushed to the public branch of FastVideo as we are still cleaning up the code.*
