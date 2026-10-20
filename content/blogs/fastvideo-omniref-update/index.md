+++
title = "FastVideo Update: FastH3 OmniRef, Matched Results and Open Training"
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
- Complete the release quality assessment; the matched examples are qualitative.
- Replace implementation PR links with merged documentation when available.
Keep private dataset versions, corpus composition, internal paths and run IDs out of this post.
-->

Our [first FastH3 release](/blogs/fasth3-preview/) brought few-step distillation to text-to-video generation with audio. Since then, we have worked on [production serving](/blogs/uniserve-fasth3/) and [running FastH3 on consumer hardware](/blogs/fasth3-rtx/). We are now extending FastVideo's distillation stack to **OmniRef**: generating video and audio from text together with visual and audio references.

This update develops that training path around **Parallel Decoding Distillation (PDD) and Distribution Matching Distillation (DMD)**. PDD teaches the student to advance through several denoising intervals in one transformer evaluation. DMD adds a learning signal on the student's completed output. Together, they let us train a few-step student while retaining the multimodal conditioning that makes reference-guided generation useful.

## TL;DR

FastH3 OmniRef extends FastH3 to reference-guided video and audio. This update brings the eight-forward student, matched results against Base, and the PDD+DMD training path together.

### Speed

- **5.6–11.9× lower end-to-end latency on eight GB200 GPUs**, across 27 matched reference workloads. Base runs 50 denoising evaluations and FastH3 Omni runs eight. Both timing lanes use the original video VAE; the comparison includes the complete model and serving recipes.

### Quality

- **Real references, full prompts and audio.** The [matched gallery](#matched-reference-inputs-and-outputs) shows Base, FastH3 Omni and FastH3 Omni with the trained 36-layer NVFP4 VAE for every workload, using a shared seed and request geometry.
- **Inspect the outputs directly.** The examples are qualitative comparisons; matching inputs and seeds do not establish output parity.

### Openness

- **PDD and DMD in one student objective.** Parallel interval predictions learn from a frozen teacher, while distribution matching trains the resulting clean video/audio estimate.
- **Training and reference-aware preprocessing are available for review.** The [OmniRef training PR](https://github.com/hao-ai-lab/FastVideo/pull/1984) includes model adapters, the combined objective, a configurable recipe and preprocessing for users' own reference-conditioned data.

<!-- Add the other release highlights to this list when their sections are ready. -->

## Results at a glance

We compare nine reference compositions at three requested clip lengths: 5, 10 and 15 seconds. Every point uses the same prompt, reference files, 1344×768 target canvas and eight GB200 GPUs. The bars include reference encoding, denoising, video/audio decoding and delivery of the finished MP4. Each speedup is **Base mean latency divided by FastH3 Omni mean latency**, using the original video VAE in both lanes.

{{< omniref-results >}}

The reference mix matters. Three short clips and nine images put different demands on the encoders and the joint attention sequence, even when the target dimensions match. We show all 27 completed profiles, with the sampling protocol and runtime differences [below](#benchmark-methodology), rather than carrying over text-only FastH3 results.

## OmniRef: PDD + DMD for reference-guided video and audio

A text prompt describes what should happen. References can supply additional information about who or what appears, the scene, or the audio. An OmniRef request combines those inputs: for example, a character image, a short audio reference and a prompt asking for a new scene.

Distillation must preserve that conditioning throughout training. In FastVideo's reference-conditioned path, the ordered references stay available to the student, teacher and learned score model. Denoising and training losses act on the **target video and audio**, with reference information retained as conditioning. This separation also lets a target use a different resolution or duration from its references.

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

## Matched reference inputs and outputs

Choose a reference composition and clip length to inspect the same request across **Base**, **FastH3 Omni**, and **FastH3 Omni · 36L VAE**. All three outputs use seed 0 on eight GB200 GPUs, with the same prompt, ordered reference files and target geometry. The gallery contains every one of the 27 profiles in the timing comparison, with no selection by output appearance.

The third column uses the trained quality decoder from our [NVFP4 VAE release](/blogs/fasth3-v21/#nvfp4-vae-decoders). It keeps FastH3 Omni's student and reference encoders and changes the output video decoder. It is included for visual and audio inspection; the performance comparison above covers Base and original-VAE FastH3 Omni.

The gallery's native decoder adapter preserves calibration boundaries with separate normalization, RoPE, residual and SwiGLU operations. It differs from the fused FastVideo decoder implementation described in that release.

Open **Reference inputs** to see the supplied images, clips and audio, and **Full prompt** for the complete instruction. **Play together** starts all three videos; **Listen to** selects the soundtrack. The MP4 downloads preserve the generated files unchanged.

{{< omniref-gallery >}}

## Benchmark methodology

- **Hardware and scope.** Eight NVIDIA GB200 GPUs across two four-GPU trays. GB200 is our proxy for B200; B200 was not separately measured. These results describe the eight-GPU cohort.
- **Matched inputs.** Nine reference compositions × three requested lengths, all at 1344×768 and 24 FPS. The aligned outputs contain 124, 243 or 362 frames. Images use the released enlarged-reference mode with a 2048-pixel short edge. Reference clips retain their real soundtracks and standalone audio remains a real input.
- **Measurement.** One request at a time, one warmup with seed 42 for each profile, then measured seeds 0, 1 and 2. The table reports the arithmetic mean and sample standard deviation of client end-to-end time until the complete MP4 is available. Model loading and server startup are excluded. No latency is inferred from historical results, missing cells or memory failures, and no timing is normalized to another workload.
- **Model and runtime.** Base is MiniMax H3 Ref2VA with 50 denoising evaluations; FastH3 Omni uses the 32-interval PDD student with eight fused evaluations and VSA. The Base lane uses Ulysses 4 × Ring 2 with the native FlashAttention transformer backend; the student uses eight-way Ulysses. Conditioning caches are disabled. Model weights, attention patterns and serving implementations differ, so this compares complete model/runtime recipes, not an isolated distillation or kernel ablation.
- **Provenance.** The frozen data records the [Base runtime source](https://github.com/sgl-project/sglang/tree/ab580b2e060059c0f8387b598a9f648dcec13f93) at `ab580b2e`, the [FastH3 Omni runtime source](https://github.com/hao-ai-lab/UniServe/tree/04a49444f3ba77db51d9eee231dac0a40bf1652a) at `04a49444`, Base checkpoint `9bfb6693` and OmniRef student `88d512c5`, with complete revisions and sample hashes in the downloadable JSON. The gallery's 36L decoder is pinned separately. Its timings are not included in the public performance data.
- **Reading the examples.** The timing statistics use three seeds; the gallery consistently uses seed 0. A shared seed does not imply numerical or perceptual output parity across different models or runtimes. The examples support direct inspection and are not a quantitative quality evaluation.

<!-- OTHER RELEASE SECTIONS: insert the additional sections here before finalizing the title, opening and TL;DR. -->

## Code and next steps

The [OmniRef PDD+DMD training PR](https://github.com/hao-ai-lab/FastVideo/pull/1984) contains the implementation, preprocessing and example recipe. The [T2AV DMD training PR](https://github.com/hao-ai-lab/FastVideo/pull/1985) separately publishes the training path used for FastH3 8-Step V2, updated with recent runtime optimizations. Both are draft PRs while validation continues.

Release checkpoint links, the broader quality assessment and the other update sections will be finalized before publication.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). We thank the MiniMax team for releasing its weights and code.

We thank the NVIDIA Enterprise Products team (Pengcheng Li and Cliff Woolley) for the Video Sparse Attention kernel used by FastH3. We thank the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team (Julius Berner, Chao Liu and Arash Vahdat) for the DMD2 framework and H3 reference work, and for introducing [Parallel Decoding Distillation](https://research.nvidia.com/labs/genair/pdd/).

We thank [Nuva Lab](https://nuvalab.ai/) for bringing production-grounded creative video workflows to FastH3, and the upstream serving, kernel and open-source communities whose implementations make these experiments possible. We also acknowledge the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/) and [MBZUAI](https://mbzuai.ac.ae/) for their support of FastVideo.
