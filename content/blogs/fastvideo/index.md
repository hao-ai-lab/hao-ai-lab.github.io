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

The entire video generation ecosystem currently underserves all following groups of people:

- **Users** (artists, creators, developers) want access to diverse video models with fast performance through a simple, consistent API that's easy to use and integrate with
- **Researchers** develop performance and efficiency improvements that currently must be manually integrated into each model separately, creating duplicated effort and code across the ecosystem
- **Model Authors** focus on quality through larger models and novel architectures, often at the expense of performance and integration simplicity. Without significant effort, it is difficult for new models to benefit from existing optimization techniques.

**No current framework adequately serves any of these groups.** This fragmentation forces painful tradeoffs - users sacrifice performance for model variety, researchers waste effort reimplementing the same techniques across models, and model authors struggle to easily integrate performance improvements without breaking their architectures.

{{< /justify >}}

## Why Current Frameworks Fail

To demonstrate these structural problems concretely, let's examine how existing frameworks fail to serve the needs of all three groups and finally how FastVideo solves these challenges. 

{{< technical_note >}}
When accelerating video generation across multiple GPUs, there are three key model parallelism strategies to understand:

**Data Parallelism (DP)**: Processes multiple different inputs in parallel, with each GPU handling a complete copy of the model but working on different data. This is useful for batch processing multiple videos simultaneously but doesn't speed up generating a single video.

**Tensor Parallelism (TP)**: Splits neural network layers across GPUs, allowing larger models to fit in memory and accelerating computation within layers. This directly speeds up individual video generation and enables larger models that wouldn't fit on a single GPU.

**Sequence Parallelism (SP)**: Divides long sequences (like video frames) across GPUs, enabling efficient processing of longer videos without memory limitations. This is critical for high-resolution or longer video generation.

Most frameworks only support Data Parallelism, which helps with throughput but not with latency for single videos. FastVideo applies TP and SP to different pipeline components in order to maximize both performance and capability.
{{< /technical_note >}}

### HuggingFace Diffusers: Fundamental Limitations for All Users

HuggingFace's [Diffusers](https://github.com/huggingface/diffusers) library has become a standard for diffusion models, but its architecture creates significant problems for all three stakeholder groups. Let's examine how:

{{< case_study >}}
```python
# Diffusers with Accelerate: Requires CLI command and only helps for batch processing
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

# First, create a script file
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

# This only distributes DIFFERENT prompts to DIFFERENT GPUs
# It cannot accelerate a single video generation
with distributed_state.split_between_processes(
    ["a dog running", "a cat jumping", "a bird flying", "a fish swimming"]
) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```
Then finally launch from command line:
```bash
$ accelerate launch run_distributed.py --num_processes=4
```
{{< /case_study >}}

**For Users**: Limited to data parallelism (cannot accelerate single video generation) and requires CLI commands outside Python.

**For Researchers**: No modular architecture for optimization integration, requiring duplicated effort and complex code changes across each diffusion pipeline.

**For Model Authors**: New architectures cannot inherit existing optimizations, requiring specialized performance knowledge beyond model design.

For large models, Diffusers requires cumbersome manual memory management and model sharding.

### xDiT: Advanced But Still Challenging for All Users

xDiT (also known as xFusers) improves upon Diffusers by supporting tensor and sequence parallelism for the diffusion model. However, it still presents significant challenges for all three user groups:

{{< case_study >}}
Running a video generation model with xDiT typically requires configuring and executing a complex bash script like the following:

The following code has been simplified, see [`xDiT/examples/run.sh`](https://github.com/xdit-project/xDiT/blob/main/examples/run.sh) for a full example.

```bash
# Select from many predefined model types - each with different settings
MODEL_TYPE="Flux"
declare -A MODEL_CONFIGS=(
    ["Sd3"]="sd3_example.py /cfs/dit/stable-diffusion-3-medium-diffusers 20"
    ["Flux"]="flux_example.py /cfs/dit/FLUX.1-dev/ 28"
    # Multiple other model types and their configurations...
)

# Configure parallelism strategies - must be manually tuned
N_GPUS=8
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 2"
# Numerous optional flags for various optimizations - requires expertise
# CFG_ARGS="--use_cfg_parallel"
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
# COMPILE_FLAG="--use_torch_compile"
# QUANTIZE_FLAG="--use_fp8_t5_encoder"
# CACHE_ARGS="--use_teacache"

# Finally launch with torchrun - separate command outside Python
torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
--height 1024 --width 1024 \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "a beautiful sunset over mountains" \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
$CACHE_ARGS
```

Or alternatively, directly launching the example Python scripts with `torchrun`:

```bash
torchrun --nproc_per_node=8 \
    examples/hunyuandit_example.py \
    --model /models/HunyuanDiT-v1.2 \
    --pipefusion_parallel_degree 2 \
    --ulysses_degree 2 \
    --num_inference_steps 50 \
    --prompt "A beautiful sunset over mountains" \
    --use_cfg_parallel
```

These scripts require deep understanding of multiple optimization parameters, parallelism strategies, and each model's specific architecture.
{{< /case_study >}}
{{< justify >}}

**For Users**: Requires complex CLI execution with `torchrun`, manual configuration of multiple parallelism strategies, and lacks a standardized Python API. Each model requires different settings.

**For Researchers**: No universal abstraction for optimizations, which remain tightly coupled to specific model implementations.

**For Model Authors**: Requires adapting to xDiT-specific paradigms with no clean separation between model architecture and execution strategy.

Despite its advanced capabilities, xDiT's complexity limits adoption across all user groups.

## FastVideo V1: A Unified Solution

FastVideo V1 addresses these challenges with a clean, modular architecture designed to serve all stakeholders in the video generation ecosystem. Below, we'll explore how FastVideo V1 solves these problems for users, researchers, and model authors.

### For Users
1. **Simple Python API**
A simple Python API with built-in multi-GPU support, eliminating the need for command-line tools or bash scripts

{{< /justify >}}

{{< case_study >}}

```python
# FastVideo - Simple Python API. All configuration and optimizations will be configured through this interface
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained("FastVideo/FastHunyuan")
# No torchrun or configuration of parallelism settings needed
video = generator.generate_video("a beautiful sunset over mountains")
```

{{< /case_study >}}
{{< justify >}}

2. **Automatic parallelism and optimization** across multiple GPUs without requiring technical knowledge or configuration.
FastVideo will automatically shard and use tensor parallelism for text and image encoders and use Ulysses sequence parallelism for DiT attention. 

Users can easily switch to different attention backends configured through an environment variable. 

Currently FastVideo supports the following Attention backends:
- Torch SDPA
- [Flash Attention 2 and 3 (for Hopper)]()
- [Sliding Tile Attention]()
- [Sage Attention]()

More coming soon!
- [Chipmunk]()

{{< /justify >}}
{{< case_study >}}
```python
from fastvideo import VideoGenerator

# Can easily configure backends for attention computation
# Available options:
# - "TORCH_SDPA": use torch.nn.MultiheadAttention
# - "FLASH_ATTN": use FlashAttention
# - "SLIDING_TILE_ATTN" : use Sliding Tile Attention
# - "SAGE_ATTN": use Sage Attention
os.environ['FASTVIDEO_ATTENTION_BACKEND'] = "SLIDING_TILE_ATTN"
# Tensor and sequence parallelism are automatically applied with num_gpus
generator = VideoGenerator.from_pretrained("FastVideo/FastHunyuan", num_gpus=4)

# No torchrun or configuration of parallelism settings needed
video = generator.generate_video("a person walking in a forest")
```
{{< /case_study >}}
{{< justify >}}

3. **Consistent interfaces** across different video models, reducing the learning curve by defaulting to optimal configuration and parameters while still exposing all configuration knobs to user.

For any HuggingFace model string, FastVideo will attempt to detect and use the optimal configuration for initialization and sampling. 

{{< /justify >}}

{{< case_study >}}

```python
from fastvideo import VideoGenerator, SamplingParam

# Override default configurations. 
# Other arguments will be set to best defaults
generator = VideoGenerator.from_pretrained(
    "FastVideo/FastHunyuan-diffusers",
    num_gpus=2,
    flow_shift=16 # override scheduler flow_shift
)

# Users can fine-tune and override parameters. 
params = SamplingParam(
    "FastVideo/FastHunyuan-diffusers", # Other arguments will be set to best defaults
    num_inference_steps=30,  # Higher quality
    guidance_scale=7.5,      # Stronger guidance
    width=1024,              # Higher resolution
    height=576
)

# Performance optimizations still applied automatically
video = generator.generate_video(
    "detailed landscape with mountains and a lake",
    sampling_param=params
)
```
{{< /case_study >}}
{{< justify >}}

### For Researchers 
1. A **modular architecture** that separates model logic from execution strategy

```python
# add pipeline stuff here
```

2. **Unified attention backends** that can be reused across multiple models

Researchers face a significant challenge: implementing the same optimization technique (like a faster attention algorithm) across multiple video models requires duplicating work for each model architecture. FastVideo solves this with a pluggable attention system where:

- New attention implementations can be developed once and immediately used across all models
- Researchers can focus on optimizing just the attention algorithm, not model-specific code
- Performance improvements automatically propagate to all integrated models
- Benchmarking becomes standardized across different backend implementations

The code below demonstrates how a model author can leverage multiple attention backends without changing their model architecture:

```python
# FastVideo - Modular architecture with pluggable attention backends
from fastvideo.v1.attention import DistributedAttention
from fastvideo.v1.attention.backends import Backend
from fastvideo.v1.models.dit import BaseDiT

class NewDiT(BaseDiT):
    def __init__(self, hidden_size=1024, num_heads=16, head_dim=64, **kwargs):
        super().__init__(**kwargs)
        # Create attention module with multiple supported backends
        # The system will automatically select the best available backend
        self.attn = DistributedAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            supported_attention_backends=(
                Backend.TORCH_SDPA,      # Basic PyTorch attention
                Backend.FLASH_ATTN,      # FlashAttention for speed
                Backend.SLIDING_TILE_ATTN,  # Sliding Tile for video-specific optimization
            )
        )
        # Define the rest of the transformer blocks
        # ...
    
    def forward(self, x, attention_mask=None):
        # DistributedAttention automatically uses the configured backend with
        # the right parallelism strategy
        x = self.attn(x, attention_mask=attention_mask)
        # Rest of the forward pass
        # ...

```

### For Model Authors
1. **Clean separation** between model architecture and execution strategy

```python
# FastVideo - Clean separation of model and execution
from fastvideo.v1.models import register_model
from fastvideo.v1.pipeline import ComposedPipelineBase

# Define model without worrying about parallelism
class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ...
        self.decoder = ...
    
    def forward(self, x):
        # Model-specific logic
        return self.decoder(self.encoder(x))

# Register model without execution details
register_model("MyNewModel", MyCustomModel)

# Execution strategy is applied separately
generator = VideoGenerator.from_pretrained("MyNewModel", num_gpus=4)
```

2. **Automatic inheritance** of optimizations without additional implementation work

```python
# FastVideo - Automatic optimization inheritance
from fastvideo import VideoGenerator
from my_custom_model import MyModelClass

# Custom model automatically gets all FastVideo optimizations
generator = VideoGenerator(
    model_class=MyModelClass,
    model_weights="path/to/weights.safetensors",
    num_gpus=4
)

# Generated with optimized attention and parallelism
video = generator.generate_video("futuristic cityscape")
```

3. **Simplified integration** of new models into the ecosystem

```python
# FastVideo - Easy model integration
from fastvideo.v1.pipeline import ComposedPipelineBase
from fastvideo.v1.stages import StandardStages

class MyPipeline(ComposedPipelineBase):
    """Custom pipeline implementation."""
    
    def create_pipeline_stages(self, args):
        # Reuse standard stages
        StandardStages.add_text_encoding_stage(self)
        StandardStages.add_transformer_stage(self)
        StandardStages.add_vae_stage(self)
        
        # Add only custom stages as needed
        self.add_stage("my_custom_stage", MyCustomStage())
```

4. **Pipeline and Model Config Abstraction** allows authors to upload the best default configurations of their weights to give users best results out-of-the-box without requiring bash scripts or READMEs.


FastVideo V1 eliminates the need for users to understand complex parallelization strategies while still achieving 3× faster generation speeds. By automatically handling distribution across GPUs, it makes high-performance video generation accessible to everyone.
{{< /justify >}}

{{< justify >}}

FastVideo V1 addresses these structural issues with a unified, modular framework designed from the ground up for both performance and extensibility.

{{< /justify >}}

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

## Appendix: The Challenge of Distributed Video Generation

{{< justify >}}

### Manual Model Sharding in Diffusers: Tedious and Error-Prone

For large models that don't fit on a single GPU, Diffusers requires manual sharding of model components, which is tedious and error-prone:

```python
# Diffusers model sharding: Manual, complex, and fragile
import torch
import gc
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL

prompt = "a beautiful woman in a red dress walking down a street"

# Step 1: Load only text encoders with balanced sharding
pipeline = FluxPipeline.from_pretrained(
    "large-video-model",
    transformer=None,  # Don't load transformer yet
    vae=None,          # Don't load VAE yet
    device_map="balanced",
    max_memory={0: "16GB", 1: "16GB"},
    torch_dtype=torch.bfloat16
)

# Step 2: Compute text embeddings
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, max_sequence_length=512
    )

# Step 3: Manually free memory
del pipeline.text_encoder
del pipeline.text_encoder_2
del pipeline.tokenizer
del pipeline.tokenizer_2
del pipeline
gc.collect()
torch.cuda.empty_cache()

# Step 4: Load transformer model with auto sharding
transformer = FluxTransformer2DModel.from_pretrained(
    "large-video-model", 
    subfolder="transformer",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Step 5: Create new pipeline with only transformer
pipeline = FluxPipeline.from_pretrained(
    "large-video-model",
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    vae=None,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

# Step 6: Run denoising
latents = pipeline(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=50,
    height=768,
    width=1360,
    output_type="latent",
).images

# Step 7: Manually free more memory
del pipeline.transformer
del pipeline
gc.collect()
torch.cuda.empty_cache()

# Step 8: Load VAE and decode latents
vae = AutoencoderKL.from_pretrained(
    "large-video-model", 
    subfolder="vae", 
    torch_dtype=torch.bfloat16
).to("cuda")

# Step 9: Finally decode the image
with torch.no_grad():
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents, return_dict=False)[0]
    # Additional processing to save the video...
```

Compare this to FastVideo's approach, which handles all this complexity automatically:

```python
from fastvideo import VideoGenerator

# Automatic model sharding with multi-GPU support
generator = VideoGenerator.from_pretrained(
    "FastVideo/LargeVideoModel",
    num_gpus=4  # Uses tensor and sequence parallelism to accelerate a SINGLE generation
)

# Generate with a single call - no complex setup or manual memory management
video = generator.generate_video(
    "a beautiful woman in a red dress walking down a street",
    height=768,
    width=1360,
    num_frames=24
)
```
{{< /justify >}}