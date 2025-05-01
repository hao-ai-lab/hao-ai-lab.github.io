+++
title = "FastVideo: A Unified Framework for Accelerated Video Generation"
date = 2025-04-24T11:00:00-08:00
authors = ["Will Lin", "Peiyuan Zhang", "Wei Zhou", "Kevin Lin", "Yongqi Chen", "Runlong Su", "Hangliang Ding", "Wenting Zhang", "You Zhou", "Cody Yu", "Richard Liaw", "Hao Zhang"]
author = "Will Lin, Peiyuan Zhang, Wei Zhou, Kevin Lin, Yongqi Chen, Runlong Su, Hangliang Ding, Wenting Zhang, You Zhou, Cody Yu, Richard Liaw, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
+++

{{< socialBadges github="hao-ai-lab/FastVideo" >}}

{{< justify >}}
**TL;DR:** FastVideo V1 is a **unified framework** for accelerating video generation, offering **3× faster** speeds without quality compromise while providing a clean, consistent API across models for seamless multi-GPU video generation!

All you need to do multi-gpu inference in FastVideo V1 is the following!
See full example [here]().
```python
# Automatically shards model and applies tensor parallelism and sequence parallelism!
generator = VideoGenerator.from_pretrained("FastVideo/FastHunyuan", num_gpus=4)

# Automatically defaults to the optimal parameters for the picked model!
video = generator.generate_video("a person walking in a forest")
```

👉 Try FastVideo in our [GitHub repository](https://github.com/hao-ai-lab/FastVideo)!
{{< /justify >}}

## The Video Generation Ecosystem is Broken

{{< justify >}}
Video generation models has reached impressive levels of quality, but at enormous computational cost. Creating just a few seconds of high-quality video can take **15+ minutes** even on high-end H100 GPUs, making iterative creative work virtually impossible on bigger models.

Behind this performance problem lies a deeper structural issue in how video generation frameworks are built, hindering both users and researchers. 

The video generation ecosystem currently inadequately serves three key groups:

- **Users** (artists, creators, developers) want access to diverse video models with fast performance through a simple, consistent API that's easy to use and integrate with
- **Researchers** develop performance and efficiency improvements that currently must be manually integrated into each model separately, creating duplicated effort and code across the ecosystem
- **Model Authors** focus on quality through larger models and novel architectures, often at the expense of performance and integration simplicity. Without significant effort, it is difficult for new models to benefit from existing optimization techniques.

**No current framework adequately serves any of these groups.** This fragmentation forces painful tradeoffs - users sacrifice performance for model variety, researchers waste effort reimplementing the same techniques across models, and model authors struggle to easily integrate performance improvements without breaking their architectures.

{{< /justify >}}

## FastVideo V1: A Unified Solution

FastVideo V1 addresses these challenges with a clean, modular architecture designed from the ground up for both performance and extensibility. Below, we explore FastVideo's key features and how they benefit different stakeholders in the video generation ecosystem.

### Simple, Unified Python API with Multi-GPU Support

A streamlined Python API with built-in multi-GPU support eliminates the need for complex command-line tools or bash scripts. In particular when `num_gpus > 1`, best parallelism strategy is automatically applied without requiring the use of `torchrun` or `accelerate` commands through bash scripts or CLI. 

This API also allows users to easily iFor an example, see our [Gradio example]().ntegrate FastVideo into their applications. 

{{< case_study >}}
```python
# FastVideo - Simple Python API. All configuration and optimizations are handled automatically
from fastvideo import VideoGenerator

def main():
    # if num_gpus > 1, parallelism is automatically handled
    generator = VideoGenerator.from_pretrained("FastVideo/FastHunyuan-diffusers", num_gpus=4)

    # output is automatically saved to outputs/
    video = generator.generate_video("a beautiful sunset over mountains")

if __name__ == '__main__':
    main()

```
{{< /case_study >}}

## Intelligent Default Configurations

FastVideo automatically applies optimal configurations based on the model. With just a HuggingFace model string, it configures all pipeline components for high-quality output without manual tuning.

For advanced users who need fine-grained control, FastVideo provides access to all pipeline components through a comprehensive API. 
This config can be examined, modified, and passed back to `VideoGenerator.from_pretrained()` to customize any aspect of the pipeline:


Both initialization parameters (model loading, component configuration) and sampling parameters (inference steps, guidance scale, dimensions) can be customized while keeping optimal defaults for everything else. 
Model authors and developers can contribute configurations for new or fine-tuned models to our repository, making their models immediately accessible with optimal settings for all FastVideo users.

Now we put it all together in the following example:

{{< case_study >}}
```python
from fastvideo import VideoGenerator, SamplingParam

config = PipelineConfig.from_pretrained("FastVideo/FastHunyuan-diffusers")

config.num_gpus = 4

# Override default configurations while keeping optimal defaults for other settings
generator = VideoGenerator.from_pretrained(
    "FastVideo/FastHunyuan-diffusers",
    pipeline_config=config
)

# Fine-tune specific parameters while maintaining optimal defaults
param = SamplingParam(
    "FastVideo/FastHunyuan-diffusers", # Other arguments will be set to best defaults
    num_inference_steps=30,  # Higher quality
    guidance_scale=7.5,      # Stronger guidance
    width=1024,              # Higher resolution
    height=576
)

# Performance optimizations still applied automatically
video = generator.generate_video(
    "detailed landscape with mountains and a lake",
    sampling_param=param
)
```
{{< /case_study >}}


### Modular Architecture with Clean Separation

FastVideo provides clear separation between model architecture and implementation, similar to modern LLM inference frameworks.

This allows model authors to leverage FastVideo's distributed processing, optimized components, and parallelism strategies without rewriting their core model logic. With FastVideo's clean architecture, researchers can implement a new optimization once and have it benefit all compatible models in the ecosystem.


The following snippet demonstrates how a new model might be implemented with FastVideo's components:
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

FastVideo splits the diffusion pipeline into functional and reusable stages, avoiding code duplication and enabling pipeline-level optimizations. This modular approach lets developers easily customize specific parts of the generation process while reusing standard components.

{{< case_study >}}
```python
class MyCustomPipeline(ComposedPipelineBase):
    ...
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=CLIPTextEncodingStage(
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
        ...
```
{{< /case_study >}}

## Performance and Results

{{< justify >}}
In our benchmarks, FastVideo V1 consistently outperforms existing frameworks:
<!-- 
| Model | Resolution | Frames | FastVideo V1 | Diffusers | Speedup |
|-------|------------|--------|--------------|-----------|---------|
| FastHunyuan | 512×512 | 16 | 24s | 68s | 2.8× |
| CogVideo | 256×256 | 16 | 19s | 62s | 3.3× |
| AnimateDiff | 512×512 | 16 | 32s | 85s | 2.7× |

*Benchmarks performed on 4× NVIDIA A100 GPUs* -->

To make FastVideo accessible to different workflows, we provide:
- A Python SDK for direct integration
- A Gradio interface for code-free experimentation 
- A ComfyUI integration for node-based workflows
{{< /justify >}}

## Getting Started

{{< justify >}}
To try FastVideo, visit our [GitHub repository](https://github.com/hao-ai-lab/FastVideo). Our documentation provides comprehensive guidance for installation, configuration, and integration.

For those interested in technical details:
- [Sliding Tile Attention blog post](https://hao-ai-lab.github.io/blogs/sta/)
- [Developer documentation](https://github.com/hao-ai-lab/FastVideo/tree/main/docs)
- [Model compatibility guide](https://github.com/hao-ai-lab/FastVideo/#supported-models)

We welcome your feedback on FastVideo V1. Share your results and experiences on Twitter or GitHub to help guide our continued development.
{{< /justify >}}

## Acknowledgements

{{< justify >}}
FastVideo builds on contributions from many researchers and engineers. We're particularly grateful to the teams behind FlashAttention, NATTEN, and ThunderKittens for their pioneering work in attention optimization.

We also thank our early testers and community members who provided invaluable feedback throughout the development process.
{{< /justify >}}