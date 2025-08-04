+++
title = "Introducing FastWan: 15 Times Faster Video Generation With Sparse Distillation"
date = 2025-08-01T11:00:00-08:00
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
[cover]
    image = "/img/speed.gif"
    alt = "Denoising speedup of FastWan"
    caption = "A gif of a graph showing FastWan achieving 72.8x speedup for denoising"
    hidden = true
+++


{{< socialBadges github="hao-ai-lab/FastVideo" arxiv-index="2505.13389" demo="https://fastwan.fastvideo.org/" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ" discord="https://discord.gg/Dm8F2peD3e" huggingface="https://huggingface.co/FastVideo" >}}

**TL;DR:** We introduce **FastWan**, a family of video generation models, trained via a new recipe we term as “sparse distillation”, to achieve near-realtime video generation. FastWan matches Wan in video quality but is blazingly faster: 50x speedup on diffusion time and **15x end-to-end speedup**. FastWan2.1-1.3B can generate a 5-second 480P video in **12 seconds** on a single RTX 4090 and **near real time** on a single H200. FastWan2.2-5B can generate a 5-second 720P video in 16 seconds on a single H200. All resources — model weights, training recipe, and dataset — are released under the Apache-2.0 license.


{{<youtube gQD6woeeMXA>}}

## The FastWan Series

With this blog, we are releasing the following models and their recipes:
|                                            Model                                            	|                                               Sparse Distillation                                               	|                                                  Dataset                                                 	|
|:-------------------------------------------------------------------------------------------:	|:---------------------------------------------------------------------------------------------------------------:	|:--------------------------------------------------------------------------------------------------------:	|
| [FastWan2.1-T2V-1.3B](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers)       	|    [Recipe](https://github.com/hao-ai-lab/FastVideo/tree/main/examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P)    	| [FastVideo Synthetic Wan2.1 480P](https://huggingface.co/datasets/FastVideo/Wan-Syn_77x448x832_600k)     	|
| [FastWan2.1-T2V-14B-Preview](https://huggingface.co/FastVideo/FastWan2.1-T2V-14B-Diffusers) 	|                                                   Coming soon!                                                  	|   [FastVideo Synthetic Wan2.1 720P](https://huggingface.co/datasets/FastVideo/Wan-Syn_77x768x1280_250k)  	|
| [FastWan2.2-TI2V-5B](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-Diffusers)         	| [Recipe](https://github.com/hao-ai-lab/FastVideo/tree/main/examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free) 	| [FastVideo Synthetic Wan2.2 720P](https://huggingface.co/datasets/FastVideo/Wan2.2-Syn-121x704x1280_32k) 	|


We are actively working on applying sparse distillation to 14B models for both Wan2.1 and Wan2.2 and will be releasing those checkpoints over the following weeks. Follow our progress at our [Github](https://github.com/hao-ai-lab/FastVideo), [Slack](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ) and [Discord](https://discord.gg/Dm8F2peD3e)!

### How good is FastWan?
{{< image src="img/speed.gif" alt="denoising speedup" width="100%" >}}

### Online Demo using FastVideo
Try the FastWan demo [here](https://fastwan.fastvideo.org/)!
Our demo is served on 16 H200s generously provided by [GMI Cloud](https://www.gmicloud.ai/).

{{< image src="img/demo.png" alt="screenshot of demo" width="100%" >}}


### Try FastWan Locally!
FastWan is runnable on a wide range of hardware with [FastVideo](https://github.com/hao-ai-lab/FastVideo). 

<!-- We list below the VRAM needed for the 1.3B and 5B models under variable resolution. 

#### FastWan2.1-T2V-1.3B
[VRAM v.s. Model size v.s. Resolution Table]
#### FastWan2.2-TI2V-5B -->


## Sparse Distillation: Making Video Generation Go Blurrrr
Video diffusion models are incredibly powerful, but they've long been held back by two major bottlenecks: 
1. The huge number of denoising steps needed to generate a video. 
2. The quadratic cost of attention when handling long sequences — which are unavoidable for high-resolution videos. Taking Wan2.2-14B as an example, the models run for 50 diffusion steps, and generating just a 5-second 720P video involves processing over 100K tokens. Even worse, attention operations can eat up more than 85% of total inference time.

Sparse distillation is our core innovation in FastWan2.2 — the first method to **jointly train sparse attention and denoising step distillation in a unified framework**. At its heart, sparse distillation answers a fundamental question: *Can we retain the speedups from sparse attention while applying extreme diffusion compression (e.g., 3 steps instead of 50)?* Prior work says no — and in the following sections we show why that answer changes with Video Sparse Attention (VSA). 

### Why Existing Sparse Attention Fails Under Distillation
Most prior sparse attention methods (e.g., [STA](https://arxiv.org/pdf/2502.04507), [SVG](https://svg-project.github.io/)) rely on redundancy in multi-step denoising to prune attention maps. They often sparsify only late-stage denoising steps and retain full attention in early steps. However, when distillation compresses 50 steps into 1–4 steps, there’s no “later stage” to sparsify — and the redundancy they depend on vanishes. As a result, these sparse patterns no longer hold up. Our preliminary experiments confirm that existing sparse attention schemes degrade sharply under sub-10 step setups. This is a critical limitation. While sparse attention alone can yield up to 3× speedup, distillation offers more than 20× gains. We argue that to make sparse attention truly effective and production-ready, it must be compatible with training and distillation.

### Why Video Sparse Attention is Compatible With Distillation
[Video Sparse Attention](https://arxiv.org/pdf/2505.13389) is a sparse attention kernel we developed that learns to dynamically identify important tokens in the sequence. Rather than relying on training-free techniques such as profiling or heuristics, VSA can directly replace [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main) during training to learn data-dependent sparsity while minimizing quality degradation. During step-distillation, as the student model learns to denoise in fewer steps, VSA does not need to rely on redundancy in multi-step denoising to prune attention maps and can instead directly learn and adjust to new sparse patterns, allowing VSA to be fully compatible with distillation techniques. **To our knowledge, VSA is the first sparse attention mechanism to be fully compatible with distillation** (we even train VSA together with distillation)! We will be releasing a technical blog on VSA next week, so stay tuned!

### How Sparse Distillation Works
Building upon Video Sparse Attention (VSA), we propose **sparse distillation**, a post-training technique that combines sparse attention training and step-distillation. Figure 1 shows an overview of sparse distillation.

{{< image src="img/overview.png" alt="sparse distillation overview" width="100%" >}}

**Figure 1**: Sparse Distillation. The student model (FastWan2.2) uses **video sparse attention (VSA)** during generation, while both the real and fake score networks use **full attention**. This allows the student to benefit from efficient sparse computation, while leveraging full-attention supervision during training.

----

The core idea of sparse distillation is to teach a few-step and sparse student model to match the *distribution* from a full-step and dense teacher. When applying sparse distillation with VSA and [Distribution Matching Distillation](https://tianweiy.github.io/dmd2/), this is done through three components:
1. a few-step and sparse student network with VSA (trainable).
2. a real score network (frozen, full attention).
3. a fake score network (trainable, full attention).

All three components are initialized with Wan2.2. During training, the sparse-distilled student takes a noisy video input and performs one denoising step with VSA, producing the current output. This output is then noised again and passed to both the real and fake score functions, each of which performs one denoising step under full attention. The outputs from these two branches define the **real and fake score**, whose difference forms the **distribution matching gradient** that is backpropagated to improve the student. In parallel, the fake score model is updated via a diffusion loss on the student outputs.
Importantly, while the student model adopts **video sparse attention (VSA)** for efficiency, both the real and fake score functions remain full-attention to ensure high-fidelity supervision during training. This separation allows us to decouple runtime acceleration (in the student) from distillation quality (in the score estimators), making sparse attention compatible with aggressive step reduction. More broadly, since sparse attention is only applied to the student, it remains fully compatible with any distillation method, such as consistency distillation, progressive distillation, or GAN-based distillation loss.




## Distillation Details

High-quality data is critical for any training recipe, especially in the context of diffusion models. However, it is extremely challenging for academic teams to access video data that is both comparable in quality to Wan’s proprietary post-training data and license-friendly. Collecting data independently often results in lower quality, which can actually degrade generation performance—even when training with full attention. To address this, we choose to generate our own **synthetic dataset** using high-quality Wan models. Specifically, we use **Wan2.1-T2V-14B**  to produce 600k 480P videos and 250k 720P videos; and **Wan2.2-TI2V-5B** to produce 32k videos. All data is released under an open license to facilitate reproducibility and community research (see Artifact Release below).

Sparse distillation with DMD requires fitting three large 14B models into GPU memory: the student model, the trainable fake score model, and the frozen real score model. Two of these models (student and fake score) are actively trained and thus require both optimizer states and gradient storage, making memory efficiency a major challenge—especially given the long sequence lengths involved. To address this, we wrap all three models with FSDP2, enabling parameter sharding across GPUs to significantly reduce memory overhead. We also apply activation checkpointing to further reduce high activation memory caused by long sequence length. Careful attention is paid to which models require gradients and which can remain in inference mode during different stages of distillation (e.g., updating the student vs. updating the fake score model). In addition, we incorporate gradient accumulation to increase the effective batch size within constrained GPU memory budgets.

We perform sparse distillation for Wan2.1-T2V-1.3B on 64 H200 GPUs for 4k steps, totaling 768 GPU hours. Our data, training recipe and even slurm script are released. With a cloud pricing of \\$3.39/H200/hour, you can reproduce FastWan2.1-T2V-1.3B with a \\$2603 budget with FastVideo.

 



## Acknowledgement
We thank [Anyscale](https://www.anyscale.com/), [MBZUAI](https://mbzuai.ac.ae/), and [GMI Cloud](https://www.gmicloud.ai/) for supporting the development and release of FastWan. We are especially grateful to the developers of the [Wan series](https://github.com/Wan-Video), whose work laid the foundation for our advancements. Our implementation of DMD distillation and Video Sparse Attention would not be possible without the effort from [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), [Triton](https://github.com/triton-lang/triton), [DMD2](https://github.com/tianweiy/DMD2) and [CausalVid](https://github.com/tianweiy/CausVid).  



## The Team
## Citation
If you use FastWan for your research, please cite our work:
```bibtex
@software{fastvideo2024,
 title        = {FastVideo: A Unified Framework for Accelerated Video Generation},
 author       = {The FastVideo Team},
 url          = {https://github.com/hao-ai-lab/FastVideo},
 year         = {2024},
}

@misc{zhang2025vsafastervideodiffusion,
     title={VSA: Faster Video Diffusion with Trainable Sparse Attention},
     author={Peiyuan Zhang and Haofeng Huang and Yongqi Chen and Will Lin and Zhengzhong Liu and Ion Stoica and Eric Xing and Hao Zhang},
     year={2025},
     eprint={2505.13389},
     archivePrefix={arXiv},
     primaryClass={cs.CV},
     url={https://arxiv.org/abs/2505.13389},
}
```


Ready to experience lightning-fast video generation? Check out our [documentation](https://hao-ai-lab.github.io/FastVideo/index.html) and [FastVideo](https://github.com/hao-ai-lab/FastVideo) to get started today.
Available now with native support for ComfyUI, Apple Silicon, Windows WSL, and Gradio web interface!


*The FastVideo team continues to push the boundaries of real-time video generation. Stay tuned for more exciting developments!*

