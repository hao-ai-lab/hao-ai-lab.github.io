+++
title = "FastVideo Update: OmniRef Distillation and Open Training"
date = 2026-10-19T00:00:00Z
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
ShowToc = true
math = true
draft = true
contentClass = "post-content-justified"
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
+++

<!--
Working release draft. The introduction and TL;DR will expand as other sections land.
Before publication:
- Confirm the release name, date, authors and checkpoint links.
- Add the other release sections at the marked insertion point.
- Add matched teacher/student videos with audio and the supplied references.
- Add quality and end-to-end latency results for the exact release checkpoint.
- Replace implementation PR links with merged documentation when available.
Keep private dataset versions, corpus composition, internal paths and run IDs out of this post.
-->

Our [first FastH3 release](/blogs/fasth3-preview/) brought few-step distillation to text-to-video generation with audio. Since then, we have worked on [production serving](/blogs/uniserve-fasth3/) and [running FastH3 on consumer hardware](/blogs/fasth3-rtx/). We are now extending FastVideo's distillation stack to **OmniRef**: generating video and audio from text together with visual and audio references.

This update develops that training path around **Parallel Decoding Distillation (PDD) and Distribution Matching Distillation (DMD)**. PDD teaches the student to advance through several denoising intervals in one transformer evaluation. DMD adds a learning signal on the student's completed output. Together, they let us train a few-step student while retaining the multimodal conditioning that makes reference-guided generation useful.

## TL;DR

- **OmniRef distillation conditions on text and references.** The training path supports image and video references, optionally accompanied by audio references, for joint video/audio generation.
- **We combine PDD and DMD in one student objective.** Parallel interval predictions learn from a frozen teacher, while distribution matching trains the resulting clean video/audio estimate.
- **The training implementation and preprocessing are available for review.** The [OmniRef training PR](https://github.com/hao-ai-lab/FastVideo/pull/1984) includes model adapters, the combined objective, a configurable recipe and reference-aware preprocessing.

<!-- Add the other release highlights to this list when their sections are ready. -->

## OmniRef: PDD + DMD for reference-guided video and audio

A text prompt describes what should happen. References can supply additional information about who or what appears, the scene, or the audio. An OmniRef request combines those inputs: for example, a character image, a short audio reference and a prompt asking for a new scene.

Distillation must preserve that conditioning throughout training. In FastVideo's reference-conditioned path, the ordered references stay available to the student, teacher and learned score model. Denoising and training losses act on the **target video and audio**, with reference information retained as conditioning. This separation also lets a target use a different resolution or duration from its references.

<!-- Demo placement: show the supplied reference image/video/audio, full prompt, then matched teacher and student outputs. Include audio controls and multiple seeds. -->

### Predicting several intervals in one evaluation

[Parallel Decoding Distillation](https://research.nvidia.com/labs/genair/pdd/), introduced by the NVIDIA FastGen team, keeps the pretrained transformer backbone and expands its output projections into interval-specific heads. A single evaluation predicts updates for several intervals; the sampler combines them to advance a whole block before evaluating the transformer again.

Our example configuration uses a **32-interval grid with blocks of four**, giving eight transformer evaluations over the full trajectory. The video and audio heads share the backbone but integrate using their respective noise schedules. At inference, the linear heads for a block can be fused into one projection.

{{< image src="img/pdd-blocks.svg" alt="A 32-interval PDD grid grouped into eight blocks, each advanced by one transformer evaluation" width="100%" >}}

During training, a selected interval prediction is supervised against a teacher update at a corresponding intermediate state. This gives the student a local trajectory target even though inference advances several intervals at once.

### Matching the completed output

We also want a useful learning signal on the result produced by those interval predictions. For DMD, we integrate the student's predicted remaining trajectory into a clean estimate of the target video and audio. A frozen teacher supplies the desired conditional score, and a trainable fake-score model estimates the student's distribution. Their difference supplies the distribution-matching update to the student.

The combined student objective is:

$$
\mathcal{L}_{\mathrm{student}}
= \mathcal{L}_{\mathrm{DMD}}
+ \lambda\,\mathcal{L}_{\mathrm{PDD}}.
$$

The example recipe sets $\lambda = 0.3$ and alternates four fake-score updates with one student update. Video and audio losses are normalized separately so the larger video latent does not determine their relative contribution just by having more elements. Both objectives use the same reference-conditioned request.

The implementation supports two ways to construct training states: noise real target latents at a sampled block boundary, or carry states along the student's own simulated trajectory. The public example uses target latents. This gives users an explicit starting point for adapting the method to their own reference-conditioned data.

### Making the training path practical

An OmniRef sequence contains the prompt, references and target video/audio tokens. Its length changes with the number of references, their geometry and the requested output. The trainer preserves those dimensions and groups compatible target shapes into microbatches. Optional capacity padding uses explicit masks so padded rows do not enter attention or loss normalization.

The sparse student uses Video Sparse Attention, with dense attention for the teacher and fake-score model. The training code also includes native Blackwell sparse-attention backward support, BF16 communication for H3 attention inputs and sharded FP32-sensitive modules. Optional compilation and activation offload provide further controls for the compute/memory tradeoff. Their benefit depends on the shape mix and hardware, so they are configuration choices rather than a fixed speedup claim.

Reference-aware preprocessing is included with the training code. It encodes text, clean reference conditions and target video/audio latents while preserving reference order and geometry. Users can prepare their own data and change the training recipe through the modular trainer's YAML configuration.

### Results and examples

**Results and reference-conditioned examples will be added before publication.** The comparison will use the same prompts, supplied references, target dimensions and audio-enabled output settings for the teacher and student. We will report the checkpoint, transformer evaluation count, hardware and end-to-end latency alongside the samples.

<!-- Add the finalized quality results, multiple-seed gallery and timing table here. Do not carry over T2AV speedups or training MFU measurements as OmniRef results. -->

<!-- OTHER RELEASE SECTIONS: insert the additional sections here before finalizing the title, opening and TL;DR. -->

## Code and next steps

The [OmniRef PDD+DMD training PR](https://github.com/hao-ai-lab/FastVideo/pull/1984) contains the implementation, preprocessing and example recipe. The [T2AV DMD training PR](https://github.com/hao-ai-lab/FastVideo/pull/1985) separately publishes the training path used for FastH3 8-Step V2, updated with recent runtime optimizations. Both are draft PRs while validation continues.

We build on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and the NVIDIA FastGen team's [PDD work](https://research.nvidia.com/labs/genair/pdd/). The next additions to this post will cover the release checkpoints, reference-conditioned examples and measured results.
