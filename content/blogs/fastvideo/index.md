+++
title = "FastVideo: A Unified Framework for Accelerated Video Generation"
date = 2025-04-24T11:00:00-08:00
authors = ["Will Lin", "Peiyuan Zhang", "FastVideo Team", "Cody Yu", "Richard Liaw", "Hao Zhang"]
author = "Will Lin, Peiyuan Zhang, FastVideo Team, Cody Yu, Richard Liaw, Hao Zhang"
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
    image = "img/logo.jpg"
    alt = "FastVideo Logo"
    caption = "A logo of the FastVideo project"
    hidden = true

+++

{{< image src="img/logo.jpg" alt="fastvideo logo" width="100%" >}}

{{< socialBadges github="hao-ai-lab/FastVideo" >}}

{{< justify >}}

FastVideo v1 offers new APIs for accelerating video generation. In this release, FastVideo is able to generate videos up to 3x faster than alternative solutions, and provides a clean and consistent API across a wide variety of popular video models.

Modern open-source video generation models such as [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main) have reached impressive levels of quality, creating videos comparable to closed-source models.

However, it is well known that using these models for creative work still remains highly impractical. Creating a few seconds of high-quality video can take **15+ minutes** even on high-end H100 GPUs using existing video generation tools. As a result, there are a significant number of research teams developing cutting edge techniques to accelerate these models, such as Sliding Tile Attention, SageAttention, and TeaCache.

In FastVideo v1, we aim to provide a framework to unify the work across the video generation ecosystem to provide highly accessible and performant video generation.

FastVideo v1 offers:
- A a simple, consistent API that's easy to use and integrate
- A collection of model performance optimizations and techniques that can be composed with each other
- A clean and articulate way for model creators to define and distribute video generation models to end users

With all of these combined, FastVideo is able to perform high quality video generation up to 5x faster than existing systems.

{{< /justify >}}

## Quick Start

Requirements:
- NVIDIA GPU with CUDA 12.4
- Python 3.10-3.12

```python
pip install fastvideo
```

```python
def main():
    generator = VideoGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", num_gpus=1)

    video = generator.generate_video("a person walking in a forest")

if __name__ == '__main__':
    main()
```


## Features

 Below, we explore FastVideo's key features.

### Simple, Unified Python API with Multi-GPU Support

{{< justify >}}
A streamlined Python API with built-in multi-GPU support eliminates the need for complex command-line tools or bash scripts. When `num_gpus > 1`, the best parallelism strategy is automatically applied without requiring `torchrun` or `accelerate` commands through bash scripts or CLI.

This API also allows users to easily integrate FastVideo into their applications. For an example, see our [Gradio example](https://github.com/hao-ai-lab/FastVideo/tree/main/fastvideo/v1/examples/inference/gradio).

FastVideo automatically applies optimal configurations based on the model. With just a HuggingFace model string, it configures all pipeline components for high-quality output without manual tuning.

For advanced users who need fine-grained control, FastVideo provides access to all pipeline components through a comprehensive API. This config can be examined, modified, and passed to `VideoGenerator.from_pretrained()` to customize any aspect of the pipeline.

Both initialization parameters (model loading, component configuration) and sampling parameters (inference steps, guidance scale, dimensions) can be customized while keeping optimal defaults for everything else. Model authors and developers can contribute configurations for new or fine-tuned models to our repository, making their models immediately accessible with optimal settings for all FastVideo users.

Here's how it works in practice:
{{< /justify >}}

{{< case_study >}}
```python
from fastvideo import VideoGenerator, SamplingParam, PipelineConfig

def main():

    config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers")

    # Can adjust any parameters
    config.num_gpus = 4
    config.vae_config.vae_precision = "fp32"

    # Override default configurations while keeping optimal defaults for other settings
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        pipeline_config=config
    )

    # Fine-tune specific parameters while maintaining optimal defaults
    param = SamplingParam(
        "FastVideo/FastHunyuan-diffusers",  # Other arguments will be set to best defaults
    )
    param = SamplingParam.from_pretrained("FastVideo/FastHunyuan-diffusers")

    param.num_inference_steps=30 # higher quality
    param.guidance_scale=7.5 # stronger guidance
    param.width=1024  # Higher resolution
    param.height=576

    # Performance optimizations are still applied automatically. 
    # Users can also directly override any field of sampling_param using kwargs
    video = generator.generate_video(
        "detailed landscape with mountains and a lake",
        sampling_param=param,
        num_inference_steps=35  # will override param.num_inference_steps
    )

if __name__ == '__main__':
    main()
```
{{< /case_study >}}


### Modular Architecture with Clean Separation

{{< justify >}}
FastVideo provides clear separation between model architecture and implementation, similar to modern LLM inference frameworks.

This allows model authors to leverage FastVideo's distributed processing, optimized components, and parallelism strategies without rewriting their core model logic. With FastVideo's clean architecture, researchers can implement a new optimization once and have it benefit all compatible models in the ecosystem.

The following snippet demonstrates how a new model might be implemented with FastVideo's components:
{{< /justify >}}

{{< case_study >}}
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

{{< justify >}}
FastVideo splits the diffusion pipeline into functional and reusable stages, avoiding code duplication and enabling pipeline-level optimizations. This modular approach lets developers easily customize specific parts of the generation process while reusing standard components.
{{< /justify >}}

{{< case_study >}}
```python
class MyCustomPipeline(ComposedPipelineBase):
    # ...
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer")
            )
        )
        
        # Add standard stages for diffusion process
        self.add_stage("timestep_preparation_stage", 
                       TimestepPreparationStage(self.get_module("scheduler")))
        self.add_stage("latent_preparation_stage", 
                       LatentPreparationStage(self.get_module("scheduler"),
                                             self.get_module("vae")))
        self.add_stage("denoising_stage", 
                       DenoisingStage(self.get_module("transformer"),
                                     self.get_module("scheduler")))
        self.add_stage("decoding_stage", 
                       DecodingStage(self.get_module("vae")))
        # ...
```
{{< /case_study >}}

## Performance and Results

{{< justify >}}
<!-- In our benchmarks, FastVideo V1 consistently outperforms existing frameworks: -->
<!-- 
| Model | Resolution | Frames | FastVideo V1 | Diffusers | Speedup |
|-------|------------|--------|--------------|-----------|---------|
| FastHunyuan | 512×512 | 16 | 24s | 68s | 2.8× |
| CogVideo | 256×256 | 16 | 19s | 62s | 3.3× |
| AnimateDiff | 512×512 | 16 | 32s | 85s | 2.7× |

*Benchmarks performed on 4× NVIDIA A100 GPUs* -->

<!-- To make FastVideo accessible to different workflows, we provide:
- A Python SDK for direct integration
- A Gradio interface for code-free experimentation 
- A ComfyUI integration for node-based workflows -->
{{< /justify >}}

## Getting Started

{{< justify >}}
To try FastVideo, visit our [GitHub repository](https://github.com/hao-ai-lab/FastVideo). Our documentation provides comprehensive guidance for installation, configuration, and integration.

For those interested in technical details:
- [Sliding Tile Attention blog post](https://hao-ai-lab.github.io/blogs/sta/)
- [Developer documentation](https://hao-ai-lab.github.io/FastVideo/contributing/overview.html)
<!-- - [Model compatibility guide](https://github.com/hao-ai-lab/FastVideo/#supported-models) -->

We welcome your feedback on FastVideo V1. Share your results and experiences on Twitter or GitHub to help guide our continued development.
{{< /justify >}}


## Acknowledgements
{{< justify >}}
FastVideo builds on contributions from many researchers and engineers. We're particularly grateful to the following teams we learned and reused code from: [PCM](https://github.com/G-U-N/Phased-Consistency-Model), [diffusers](https://github.com/huggingface/diffusers), [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [xDiT](https://github.com/xdit-project/xDiT), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main).

We also thank our early testers and community members who provided invaluable feedback throughout the development process, in particular, Jiao Dong provided valuable feedback as the first public adopter of v1.

### FastVideo Team
Here we want to acknowledge everyone on the FastVideo Team who contributed to FastVideo V1:
- Yongqi Chen: Added STA to Wan
- Kevin Lin: Worked on CI/CD and Tests
- Zihang He: Working on adding StepVideo
- Wei Zhou: Adding Pipeline and Model Configs, Wan2.1
- You Zhou: Worked on CD
- Wenting Zhang: Worked on ComfyUI integration

{{< /justify >}}