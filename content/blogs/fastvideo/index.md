+++
title = "FastVideo V1: A Unified Framework for Accelerated Video Generation"
date = 2025-04-24T11:00:00-08:00
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
    image = "/img/cover_fastvideo_v1.png"
    alt = "FastVideo Logo"
    caption = "A logo of the FastVideo project"
    hidden = true
+++


{{< socialBadges github="hao-ai-lab/FastVideo" >}}


**TL;DR:** We are announcing [FastVideo V1](https://github.com/hao-ai-lab/FastVideo), a unified framework that accelerates video generation. This new version of FastVideo features a clean, consistent API that works across popular video models, making it easier for developers to author new models, and incorprate system- or kernel-level optimizations. For example, FastVideo V1 is able to provide 3x speedup for inference while maintaining quality by seamlessly integrating [SageAttention](https://arxiv.org/abs/2410.02367) and [Teacache](https://arxiv.org/pdf/2411.19108).

{{< image src="img/perf.png" alt="fastvideo logo" width="100%" >}}

## What's New

Modern open-source video generation models such as [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main) have reached impressive levels of quality, creating videos comparable to closed-source models. However, it is well known that using these models for creative work still remains highly impractical. Creating a few seconds of high-quality video can take **15+ minutes** even on high-end H100 GPUs using existing video generation tools. As a result, there are a significant number of research teams developing cutting edge techniques to accelerate these models, such as [Sliding Tile Attention](https://arxiv.org/pdf/2502.04507), [SageAttention](https://arxiv.org/abs/2410.02367), [TeaCache](https://arxiv.org/pdf/2411.19108), and many more.

In FastVideo V1, we aim to provide a platform to unify the work across the video generation ecosystem to provide highly accessible and performant video generation. FastVideo V1 offers:
1. A simple, consistent API that's easy to use and integrate
2. A collection of model performance optimizations and techniques that can be composed with each other
3. A clean and articulate way for model creators to define and distribute video generation models to end users

With all of these combined, FastVideo is able to perform high quality video generation up to 3x faster than existing systems.


## Quick Start

Requirements:
- NVIDIA GPU with CUDA 12.4
- Python 3.10-3.12

We recommend using Conda or virtualenv.

```python
pip install fastvideo
```
Create a Python file `generate_video.py` and copy the following:
```python
from fastvideo import VideoGenerator

def main():
    generator = VideoGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", num_gpus=1)

    video = generator.generate_video("a person walking in a forest")

if __name__ == '__main__':
    main()
```
Then simply:
```bash 
python generate_video.py
```
We next explore key features along with this release.

## Simple, Unified Python API with Multi-GPU Support

FastVideo V1 features a streamlined Python API with built-in multi-GPU support that eliminates the need for complex command-line tools or bash scripts. 





{{< case_study title="Generating Videos: FastVideo vs Diffusers vs xDiT" tabs="FastVideo,Diffusers,xDiT" >}}
In this example, we showcase how `PipelineConfig` is used to configure the pipeline initialization parameters and how `SamplingParam` is used to configure the generation time parameters:

```python
from fastvideo import VideoGenerator, SamplingParam, PipelineConfig

def main():
    # Initialization config
    model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    config = PipelineConfig.from_pretrained(model_name)

    # Can adjust any parameters
    # Other arguments will be set to best defaults
    config.num_gpus = 2 # how many GPUS to parallelize generation
    config.vae_config.vae_precision = "fp32"

    # Override default configurations while keeping optimal defaults for other settings
    generator = VideoGenerator.from_pretrained(model_name, pipeline_config=config)

    # Generation config
    param = SamplingParam.from_pretrained(model_name)

    # Adjust specific sampling parameters
    # Other arguments will be set to best defaults
    param.num_inference_steps=30 # higher quality
    param.guidance_scale=7.5 # stronger guidance
    param.width=1024  # Higher resolution
    param.height=576

    # Users can also directly override any field of sampling_param using kwargs
    video = generator.generate_video(
        "fox in the forest close-up quickly turned its head to the left",
        sampling_param=param,
        num_inference_steps=35  # will override param.num_inference_steps
    )

if __name__ == '__main__':
    main()
```
<!--tab-->
HuggingFace's [Diffusers](https://github.com/huggingface/diffusers) library has become a standard for diffusion models, but its architecture is limited to **only data parallelism** and requires launching processes from the CLI using `accelerate` or `torchrun`. 

How to use data parallelism in Diffusers:
(Taken from Diffusers example [here](https://huggingface.co/docs/diffusers/en/training/distributed_inference#-accelerate))

```python
# Diffusers with Accelerate: Requires CLI command and only helps for batch processing
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

# This only distributes DIFFERENT prompts to DIFFERENT GPUs
# It cannot accelerate a single video generation
with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```
Then finally, launch from command line using `accelerate`:
```bash
$ accelerate launch run_distributed.py --num_processes=2
```

<!--tab-->

[xDiT](https://github.com/xdit-project/xDiT) (also known as xFusers) improves upon Diffusers by supporting tensor and sequence parallelism for the diffusion model. However, its API still requires `torchrun` and complicated bash scripts to configure and launch. Below is the primary [example script](https://github.com/xdit-project/xDiT/blob/main/examples/run.sh) from xDiT:

```bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
export MODEL_TYPE="Flux"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_example.py /cfs/dit/PixArt-XL-2-1024-MS 20"
    ["Pixart-sigma"]="pixartsigma_example.py /cfs/dit/PixArt-Sigma-XL-2-2K-MS 20"
    ["Sd3"]="sd3_example.py /cfs/dit/stable-diffusion-3-medium-diffusers 20"
    ["Flux"]="flux_example.py /cfs/dit/FLUX.1-dev/ 28"
    ["HunyuanDiT"]="hunyuandit_example.py /cfs/dit/HunyuanDiT-v1.2-Diffusers 50"
    ["SDXL"]="sdxl_example.py /cfs/dit/stable-diffusion-xl-base-1.0 30"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"

# cache args
# CACHE_ARGS="--use_teacache"
# CACHE_ARGS="--use_fbcache"

# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)
N_GPUS=8
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 2"

# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Another compile option is `--use_onediff` which will use onediff's compiler.
# COMPILE_FLAG="--use_torch_compile"


# Use this flag to quantize the T5 text encoder, which could reduce the memory usage and have no effect on the result quality.
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

# export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "brown dog laying on the ground with a metal bowl in front of him." \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
$CACHE_ARGS \
```
After configuring the above script, run with:
```bash
bash examples/run.sh
```

{{< /case_study >}}

When `num_gpus > 1`, the best parallelism strategy is automatically applied without requiring `torchrun` or `accelerate` commands through bash scripts or CLI.
This API also allows users to easily integrate FastVideo into their applications. For an example, see our [Gradio example](https://github.com/hao-ai-lab/FastVideo/tree/main/examples/inference/gradio).
FastVideo automatically applies optimal configurations based on the model. With just a HuggingFace model string, it configures all pipeline components for high-quality output without manual tuning.

For advanced users who need fine-grained control, FastVideo provides access to all pipeline components through a comprehensive API. This config can be examined, modified, and passed to `VideoGenerator.from_pretrained()` to customize any aspect of the pipeline.

Both initialization parameters (model loading, component configuration) and sampling parameters (inference steps, guidance scale, dimensions) can be customized while keeping optimal defaults for everything else. Model authors and developers can contribute configurations for new or fine-tuned models to our repository, making their models immediately accessible with optimal settings for all FastVideo users.
Below is how it works in practice and how APIs from other popular video generation frameworks look like. 


### Modular Architecture with Clean Separation

FastVideo provides clear separation between model architecture and implementation, similar to modern LLM inference frameworks. This allows model authors to leverage FastVideo's distributed processing, optimized components, and parallelism strategies without rewriting their core model logic. With FastVideo's clean architecture, researchers can implement a new optimization once and have it benefit all compatible models in the ecosystem.

The following snippet demonstrates how a new model might be implemented with FastVideo's components:

{{< case_study title="Defining New Models" tabs="FastVideo" >}}
```python
from fastvideo.v1.layers.linear import QKVParallelLinear
from fastvideo.v1.attention import LocalAttention
from fastvideo.v1.layers.rotary_embedding import get_rope

# Define model with FastVideo optimized components
class CustomEncoderModel(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, head_dim=64):
        super().__init__()
        # Use FastVideo's optimized layers
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads
        )
        self.rotary_emb = get_rope(
            head_dim, 
            rotary_dim=head_dim,
            max_position=8192
        )
        self.attn = LocalAttention(
            num_heads=num_heads // get_tensor_model_parallel_world_size(),
            head_dim=head_dim,
            num_kv_heads=num_heads // get_tensor_model_parallel_world_size(),
            causal=True
        )
    
    def forward(self, positions, hidden_states):
        # Simple model logic, actual implementation called depends on user configuration and available optimizations
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return self.attn(q, k, v)
```
{{< /case_study >}}

### ComposablePipeline and PipelineStage Abstraction

FastVideo splits the diffusion pipeline into *functional and reusable* stages, avoiding code duplication and enabling pipeline-level optimizations. This modular approach lets developers easily customize specific parts of the generation process while reusing standard components.

{{< case_study title="Diffusion Pipelines: FastVideo vs Diffusers" tabs="FastVideo,Diffusers" >}}
Each of these `Stage` in FastVideo can be reused by other pipelines or composed for other purposes such as training or distillation (coming soon!)

Below we show FastVideo's entire file for Wan2.1 Pipeline. It is only 63 lines!
```python
# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.stages import (ConditioningStage, DecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage)

logger = init_logger(__name__)


class WanPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanPipeline
```

<!--tab-->
In Diffusers and many other repositories, the entire pipeline is redefined for each model. In the following example from [Diffusers Wan2.1](), we've heavily pruned the pipeline code to only the `forward()` method. The original file is almost 600 lines long, and this is repeated for each supported type (I2V, V2V, etc.)

```python
class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    # other steps ...
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
```
{{< /case_study >}}

## Performance and Results

{{< image src="img/perf.png" alt="fastvideo performance comparison" width="100%" >}}
Our benchmarks demonstrate that FastVideo significantly outperforms the official Wan2.1 implementation. Testing across multiple GPU configurations shows up to 3x faster video generation while maintaining the same output quality. Model loading time is reduced by up to 7x, dramatically improving startup experience. These performance gains come from transparently switching to SageAttention kernels for attention and enabling Teacache optimization.

## Getting Started

To try FastVideo, visit our [GitHub repository](https://github.com/hao-ai-lab/FastVideo). Our documentation provides comprehensive guidance for installation, configuration, and integration.

For those interested in technical details:
- [Sliding Tile Attention blog post](https://hao-ai-lab.github.io/blogs/sta/)
- [Developer documentation](https://hao-ai-lab.github.io/FastVideo/contributing/overview.html)
<!-- - [Model compatibility guide](https://github.com/hao-ai-lab/FastVideo/#supported-models) -->

We welcome your feedback on FastVideo V1. Share your results and experiences on Twitter or GitHub to help guide our continued development.


## Acknowledgements
FastVideo builds on contributions from many researchers and engineers. We're particularly grateful to the following teams and projects we learned and reused code from: [PCM](https://github.com/G-U-N/Phased-Consistency-Model), [diffusers](https://github.com/huggingface/diffusers), [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [xDiT](https://github.com/xdit-project/xDiT), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main). The developement of FastVideo V1 was partially supported by [Anyscale](https://www.anyscale.com/) and [MBZUAI](https://ifm.mbzuai.ac.ae/).

We also thank our early testers and community members who provided invaluable feedback throughout the development process, in particular, Jiao Dong provided valuable feedback as the first public adopter of V1.


### FastVideo Team
Here we want to acknowledge everyone on the FastVideo Team who contributed to FastVideo V1:
- Will Lin and Peiyuan Zhang: Project lead and architect
- Wei Zhou: Added Pipeline and Model Configs; Wan2.1 Pipeline; Torch compile
- Kevin Lin: Worked on CI/CD and Tests
- Yongqi Chen: Added STA support to Wan2.1
- Zihang He: Working on adding StepVideo
- You Zhou: Worked on CD
- Wenting Zhang: Worked on ComfyUI integration
- Cody Yu: Helped design V1 architecture
- Richard Liaw: Helped design API and advised the project
- Hao Zhang: Advised entire project
