+++
title = "FastH3 OmniRef Release"
date = 2026-10-19T00:00:00Z
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
ShowToc = true
math = true
draft = false
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
- Confirm public access to the pinned OmniRef export before publishing the deployment guide.
- Review the disclosed SGLang lossless-policy baseline; do not relabel it as exact-policy inference.
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

- **5.6–11.9× lower end-to-end latency on eight GB200 GPUs**, across 27 matched reference workloads. Base runs 50 denoising evaluations and FastH3 Omni runs eight. Both timing lanes use the original video VAE; Base uses SGLang's `lossless` numerical policy ([methodology](#benchmark-methodology)). The comparison includes the complete model and serving recipes.

### Quality

- **Real references, full prompts and audio.** The [matched gallery](#matched-reference-inputs-and-outputs) shows Base, FastH3 Omni and FastH3 Omni with the trained 36-layer NVFP4 VAE for every workload, using a shared seed and request geometry.
- **Inspect the outputs directly.** The examples are qualitative comparisons; matching inputs and seeds do not establish output parity.

### Openness

- **PDD and DMD in one student objective.** Parallel interval predictions learn from a frozen teacher, while distribution matching trains the resulting clean video/audio estimate.
- **Training and reference-aware preprocessing are available for review.** The [OmniRef training PR](https://github.com/hao-ai-lab/FastVideo/pull/1984) includes model adapters, the combined objective, a configurable recipe and preprocessing for users' own reference-conditioned data.
- **Deploy with UniServe.** The [installation walkthrough](#deploy-fasth3-omniref-with-uniserve) includes pinned checkpoints, a four-GPU quickstart, the two-host configuration and a helper for preparing the gallery's real requests.

<!-- Add the other release highlights to this list when their sections are ready. -->

## Results at a glance

We compare nine reference compositions at three requested clip lengths: 5, 10 and 15 seconds. Every point uses the same prompt, reference files, 1344×768 target canvas and eight GB200 GPUs. The bars include reference encoding, denoising, video/audio decoding and delivery of the finished MP4. Each speedup is **Base mean latency divided by FastH3 Omni mean latency**, using the original video VAE in both lanes.

Base uses the original unquantized model with SGLang's **`quality="lossless"` numerical policy**. These timings do not measure its stricter `quality="exact"` policy; see the [numerical-policy disclosure](#benchmark-methodology).

**Explore the measurements:** the chart shows the selected requested clip length; the table includes all three lengths. Click an underlined timing or **View example** to see that request’s prompt, reference media and generated outputs directly below.

{{< omniref-results >}}

The reference mix matters. Three short clips and nine images put different demands on the encoders and the joint attention sequence, even when the target dimensions match. We show all 27 completed profiles, with the sampling protocol and runtime differences [below](#benchmark-methodology), rather than carrying over text-only FastH3 results.

## Matched reference inputs and outputs

Choose a reference composition and clip length to inspect the same request across **Base**, **FastH3 Omni**, and **FastH3 Omni · 36L VAE**. All three outputs use seed 0 on eight GB200 GPUs, with the same prompt, ordered reference files and target geometry. The gallery contains every one of the 27 profiles in the timing comparison, with no selection by output appearance.

The third column uses the trained quality decoder from our [NVFP4 VAE release](/blogs/fasth3-v21/#nvfp4-vae-decoders). It keeps FastH3 Omni's student and reference encoders and changes the output video decoder. It is included for visual and audio inspection; the performance comparison above covers Base and original-VAE FastH3 Omni.

The gallery's native decoder adapter preserves calibration boundaries with separate normalization, RoPE, residual and SwiGLU operations. It differs from the fused FastVideo decoder implementation described in that release.

**Click any underlined latency or “View example” in the table above** to open its matching request here. The reference images, clips, audio and full prompt are expanded alongside the three outputs. You can also browse with the reference and clip-length selectors. **Play together** starts all three videos; **Listen to** selects the soundtrack. The MP4 downloads preserve the generated files unchanged.

{{< omniref-gallery >}}

## Deploy FastH3 OmniRef with UniServe

The walkthrough below serves the same [eight-forward OmniRef student](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) used in the gallery, with its original video VAE. It starts with one image and one audio reference on **four GB200 GPUs**, then shows the **eight-GPU, two-host** configuration for the larger reference mixes. The model's task is **`ref2va`**: reference-conditioned video and audio.

These commands pin [UniServe `0519921`](https://github.com/hao-ai-lab/uniserve/tree/0519921767d3fdf6527dc78114e6698e5d821087), the public source revision checked for this guide, plus the [large-index correction](deploy/uniserve-wide-index.patch) applied below. The timing tables retain their separately recorded runtime revisions; installing a newer runtime does not change those measurements. For other hardware and deployment layouts, see the [UniServe FastH3 guide](https://github.com/hao-ai-lab/uniserve/blob/0519921767d3fdf6527dc78114e6698e5d821087/docs/minimax_h3/minimax_h3.md).

### 1. Install UniServe and its GPU dependencies

Use Linux with the **CUDA 13 development toolkit**, a compatible NVIDIA driver, **Python 3.12**, and a C++ compiler. The commands below assume Ubuntu 24.04. Install FFmpeg and `ffprobe` as well: reference media is processed on the server. Check `nvidia-smi` and `nvcc --version` before building; a CUDA runtime image alone does not supply the compiler.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config cmake git curl ca-certificates \
  python3.12 python3.12-dev python3.12-venv ffmpeg

curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- \
  -y --profile minimal --default-toolchain stable

export CUDA_HOME=/usr/local/cuda
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2

git clone https://github.com/hao-ai-lab/uniserve.git
cd uniserve
git checkout 0519921767d3fdf6527dc78114e6698e5d821087

export OMNIREF_BLOG="https://haoailab.com/blogs/fastvideo-omniref-update/"
curl --fail --location \
  "${OMNIREF_BLOG}deploy/uniserve-wide-index.patch" \
  --output omniref-wide-index.patch
git apply --check omniref-wide-index.patch
git apply omniref-wide-index.patch

uv sync --locked --python /usr/bin/python3.12 --extra gpu
source .venv/bin/activate
uniserve serve --help
ffprobe -version
```

`uv sync` builds the Rust server, host launcher, worker IPC extension and native kernels, and installs the locked GPU providers. Keep the `gpu` extra when syncing this environment again. Building on a machine without visible GPUs also requires `TORCH_CUDA_ARCH_LIST` for the deployment hardware; for GB200/B200, use `10.0`. The first model startup may compile additional kernels and prepare request layouts before reporting ready.

The patch widens activation indices **before** multiplying a block offset. Without it, large reference mixes can overflow signed 32-bit indexing and trigger an illegal GPU memory access. It changes indexing, not model weights, precision, reference geometry or the denoising schedule. Apply it on both hosts of a two-host deployment before installation.

### 2. Download the student and its pinned base components

Keep the shell in the UniServe checkout. Authenticate with a Hugging Face account that has access to the export when required, then download the pinned student:

```bash
hf auth login

export OMNIREF_MODEL="$PWD/models/FastH3-Omni-8-Step-V1"
hf download FastVideo/FastVideo-FastH3-Omni-8-Step-V1 \
  --revision 88d512c5e7cc38dda499fbb65688ce58611f9c80 \
  --local-dir "$OMNIREF_MODEL"
```

The export supplies `transformer_ref`, the schedulers and `fastvideo_inference.json`. Its text encoder, tokenizer, processor, original video VAE and audio VAE come from the base revision named in that inference contract. Populate the Hugging Face cache with those components before starting the server:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["OMNIREF_MODEL"])
contract = json.loads((root / "fastvideo_inference.json").read_text())
repository, revision = contract["base_model_revision"].removeprefix("hf://").rsplit("@", 1)
snapshot_download(
    repo_id=repository,
    revision=revision,
    allow_patterns=[
        "model_index.json", "text_encoder/*", "tokenizer/*",
        "processor/*", "vae/*", "audio_vae/*",
    ],
)
print(f"Base components cached: {repository}@{revision}")
print(f"Task: {contract['model_type']}; transformer forwards: {contract['transformer_forwards']}")
PY
```

For this export the base revision is `9bfb6693f2cf6de171db46d1aa586f67d773a1da`. The cache and local student directory together supply the complete model; the base model's large denoisers are not needed. Use the native export for UniServe, including its inference contract, rather than a ComfyUI-format transformer file.

### 3. Prepare a real gallery request

The [downloadable request helper](deploy/prepare_request.py) retrieves a profile from this article's frozen data, downloads and verifies its reference files, and writes a ready-to-send JSON request. It preserves the full prompt, ordered conditions, output geometry and seed. Run it from the same checkout before starting the server:

```bash
export OMNIREF_MEDIA="$PWD/omniref-media"
export OMNIREF_BLOG="https://haoailab.com/blogs/fastvideo-omniref-update/"

curl --fail --location \
  "${OMNIREF_BLOG}deploy/prepare_request.py" \
  --output prepare_omniref_request.py

python prepare_omniref_request.py \
  --blog-url "$OMNIREF_BLOG" \
  --composition image-voice --seconds 5 --seed 0 \
  --media-directory "$OMNIREF_MEDIA" \
  --output omniref-request.json
```

This selects the gallery's **Image + voice, 5-second** example. The generated conditions use absolute `file://` URIs under `OMNIREF_MEDIA`. Those paths must exist on the server; a file on a separate client machine is not automatically uploaded. The helper uses the public article URL by default; when previewing the draft locally, pass the preview's article URL with `--blog-url`.

### 4. Start the reference-conditioned server

In the same terminal, select four local GPUs and launch the **reference** deployment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_NVLS_ENABLE=0 \
uniserve serve "$OMNIREF_MODEL" \
  --workers configs/minimax_h3/ulysses4-reference.json \
  --served-model-name FastH3-OmniRef \
  --host 0.0.0.0 --port 8000 \
  --max-running-requests 2 \
  --max-video-seconds 5 \
  --video-resolutions 768p --video-aspect-ratios 16:9 \
  --max-model-len 8192 --video-text-capacities 8192 \
  --max-condition-rows 49152 \
  --mm-processor-kwargs '{"enlarge_references":true}' \
  --dtype bfloat16 --quantization-config '{"mode":"quality"}' \
  --mem-fraction-static 0.70 \
  --media-directory "$OMNIREF_MEDIA"
```

`ulysses4-reference.json` places `transformer_ref`, the text and reference encoders, and the video/audio decoders across the four GPUs. The CPU worker reads media and produces the finished MP4. `enlarge_references` matches the released reference sizing used for this article. The `quality` precision preset retains the original decoder; it is separate from SGLang's request-quality policy discussed in the methodology.

This quickstart admits 768p landscape outputs up to five seconds, with bounded prompt and reference storage. Its capacities cover the image-plus-voice example, not every gallery composition. The checkpoint supplies the **32-interval PDD schedule, eight transformer forwards, 128-row VSA tiles and trained reference policy** automatically. Do not add diffusion-step, CFG or SGLang-specific quality fields to a UniServe request.

### 5. Generate and download video with audio

Leave the server running. In a second terminal, return to the same checkout, activate the environment, and wait for readiness:

```bash
source .venv/bin/activate

curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/capabilities \
  | python -m json.tool
```

Continue when `/health` succeeds. The capabilities response should advertise **`ref2va`**, the served canvases and duration limit, condition capacity and the checkpoint's schedule. Startup preparation is excluded from the latency figures above; there is no fixed startup-time estimate for this walkthrough.

```bash
curl --fail-with-body \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  --data-binary @omniref-request.json \
  --output omniref-output.mp4

ffprobe -v error \
  -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json omniref-output.mp4
```

The response is the completed MP4, with 1344×768 H.264 video at 24 FPS and stereo 32-kHz AAC audio. A requested five seconds produces 124 frames under the model's frame-alignment rule. Open the file locally to inspect both the video and soundtrack.

For your own input, keep the same request structure and change `prompt`, `conditions` and `seed`. Each condition has `type`, `role: "reference"` and `uri`; the prompt names the ordered references as `<Picture 1>`, `<Video 1>` and `<Audio 1>`. The `model` field must match `FastH3-OmniRef`. Requests use `target.short_edge`, `target.aspect_ratio` and `target.duration_seconds`; `/v1/capabilities` gives the limits of the running deployment.

For a job that should be polled, send the same JSON to `POST /v1/videos`, then inspect `GET /v1/videos/{id}` and download `GET /v1/videos/{id}/content` after its status becomes `completed`. Keep the server running until the job's output has been downloaded.

### 6. Serve the larger mixes on eight GB200 GPUs

For four GPUs on each of two hosts, install the same pinned source and environment on both, with identical absolute paths for the checkout, Python environment, student and Hugging Face cache. Share or replicate those model/cache directories and the media directory at the same paths. Use a network that connects the workers' NCCL traffic and host-launcher connections.

Stop the four-GPU server before replacing it. On the head host, run from the same UniServe checkout with the paths from the quickstart:

```bash
source .venv/bin/activate
export OMNIREF_MODEL="$PWD/models/FastH3-Omni-8-Step-V1"
export OMNIREF_MEDIA="$PWD/omniref-media"

CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_NVLS_ENABLE=0 \
uniserve serve "$OMNIREF_MODEL" \
  --workers configs/minimax_h3/ulysses8-reference-two-node.json \
  --host-identity rank-0 \
  --served-model-name FastH3-OmniRef \
  --host 0.0.0.0 --port 8000 \
  --max-running-requests 2 \
  --max-video-seconds 15 \
  --video-resolutions 768p --video-aspect-ratios 16:9 \
  --max-model-len 48128 \
  --video-text-capacities 4096,7168,8192,10240,14336,15360,22528,37888 \
  --max-condition-rows 79104 \
  --mm-processor-kwargs '{"enlarge_references":true}' \
  --dtype bfloat16 --quantization-config '{"mode":"quality"}' \
  --mem-fraction-static 0.70 \
  --media-directory "$OMNIREF_MEDIA"
```

The head logs `awaiting a launcher` with an address such as `0.0.0.0:PORT`. **Use the port in that log, not the HTTP port 8000.** On the second host, activate the identical environment and connect using the head's reachable IP or hostname:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_NVLS_ENABLE=0 \
uniserve-host --head HEAD_HOST:PORT --host-identity rank-1
```

Replace `HEAD_HOST:PORT` with that head address and launcher port. Once the head becomes healthy, the same request endpoint works. The larger capacities accommodate this article's nine reference compositions at requested lengths of 5, 10 and 15 seconds, including nine square images plus three two-second clips. They are capacities for these specific inputs, not a promise that every nine-image/three-video request fits.

For example, in another terminal in the head's checkout, use the same helper to prepare the largest gallery profile:

```bash
source .venv/bin/activate
export OMNIREF_MEDIA="$PWD/omniref-media"
export OMNIREF_BLOG="https://haoailab.com/blogs/fastvideo-omniref-update/"

python prepare_omniref_request.py \
  --blog-url "$OMNIREF_BLOG" \
  --composition full --seconds 15 --seed 0 \
  --media-directory "$OMNIREF_MEDIA" \
  --output omniref-full-15s.json

curl --fail-with-body \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  --data-binary @omniref-full-15s.json \
  --output omniref-full-15s.mp4
```

The helper also accepts `image`, `images3`, `images9`, `clip`, `video`, `clips3` and `mixed`. Use [FFmpeg **8.1.2**](https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz) for matching reference-video preprocessing to the frozen benchmark; Ubuntu's default package may be older. If it is installed outside `PATH`, set `--ffmpeg /absolute/path/to/ffmpeg --ffprobe /absolute/path/to/ffprobe` on the server. Keep the original VAE for the timing lane in this article. The gallery's trained 36-layer decoder requires a separately composed checkpoint; selecting a generic quantization preset does not install that decoder.

### Deployment checks

| Symptom | What to check |
| --- | --- |
| A checkpoint download returns 401 or 404 | Confirm the exact repository name, the release's availability and the account used by `hf auth login`. |
| A base component is missing at startup | Populate the cache at the revision in `fastvideo_inference.json`; a different base revision in a local directory does not satisfy that pin. |
| A reference file is refused | Its absolute path must exist under `--media-directory` on the server. Keep FFmpeg and `ffprobe` available to that process. |
| Startup runs out of memory or a request exceeds a bound | Use the appropriate deployment and explicit duration, prompt and condition capacities. Larger or longer references need more storage; increasing a bound does not create GPU memory. |
| A two-host head waits for a launcher | Match `rank-0`/`rank-1`, connect to the logged launcher port, and check that both hosts use the same source, environment and absolute paths. |

For a container deployment, follow the [UniServe Docker build instructions](https://github.com/hao-ai-lab/uniserve/blob/0519921767d3fdf6527dc78114e6698e5d821087/docs/minimax_h3/minimax_h3.md#build-and-run-with-docker), include FFmpeg in the image, and mount the student, Hugging Face cache and reference directory at the paths the server uses. Worker IPC also needs sufficient shared memory and pinned-memory access. The native installation above is the complete walkthrough for this article.

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

## Benchmark methodology

- **Hardware and scope.** Eight NVIDIA GB200 GPUs across two four-GPU trays. GB200 is our proxy for B200; B200 was not separately measured. These results describe the eight-GPU cohort.
- **Matched inputs.** Nine reference compositions × three requested lengths, all at 1344×768 and 24 FPS. The aligned outputs contain 124, 243 or 362 frames. Images use the released enlarged-reference mode with a 2048-pixel short edge. Reference clips retain their real soundtracks and standalone audio remains a real input.
- **Measurement.** One request at a time, one warmup with seed 42 for each profile, then measured seeds 0, 1 and 2. The table reports the arithmetic mean and sample standard deviation of client end-to-end time until the complete MP4 is available. Model loading and server startup are excluded. No latency is inferred from historical results, missing cells or memory failures, and no timing is normalized to another workload.
- **Model and runtime.** Base is MiniMax H3 Ref2VA with 50 denoising evaluations; FastH3 Omni uses the 32-interval PDD student with eight fused evaluations and VSA. The Base lane uses Ulysses 4 × Ring 2 with the native FlashAttention transformer backend; the student uses eight-way Ulysses. Conditioning caches are disabled. Model weights, attention patterns and serving implementations differ, so this compares complete model/runtime recipes, not an isolated distillation or kernel ablation.
- **Base numerical policy.** These Base measurements use SGLang's `quality="lossless"` policy with original unquantized weights, dense attention and the model's mixed-precision rules. Its DiT normalization/modulation and VAE fusions can change intermediate rounding. They are not measurements of SGLang's stricter `quality="exact"` policy, and reference-exact or perceptual equivalence has not been established. The original model is not uniformly BF16: it retains FP32-sensitive DiT operations and the released VAE's mixed-precision execution. The displayed latencies have not been converted to an exact-policy estimate.
- **Provenance.** The frozen data records the [Base runtime source](https://github.com/sgl-project/sglang/tree/ab580b2e060059c0f8387b598a9f648dcec13f93) at `ab580b2e`, the [FastH3 Omni runtime source](https://github.com/hao-ai-lab/UniServe/tree/04a49444f3ba77db51d9eee231dac0a40bf1652a) at `04a49444`, Base checkpoint `9bfb6693` and OmniRef student `88d512c5`, with complete revisions and sample hashes in the downloadable JSON. The gallery's 36L decoder is pinned separately. Its timings are not included in the public performance data.
- **Reading the examples.** The timing statistics use three seeds; the gallery consistently uses seed 0. A shared seed does not imply numerical or perceptual output parity across different models or runtimes. The examples support direct inspection and are not a quantitative quality evaluation.

<!-- OTHER RELEASE SECTIONS: insert the additional sections here before finalizing the title, opening and TL;DR. -->

## Code and next steps

The [OmniRef PDD+DMD training PR](https://github.com/hao-ai-lab/FastVideo/pull/1984) contains the implementation, preprocessing and example recipe. The [T2AV DMD training PR](https://github.com/hao-ai-lab/FastVideo/pull/1985) separately publishes the training path used for FastH3 8-Step V2, updated with recent runtime optimizations. Both are draft PRs while validation continues.

Public checkpoint availability, the broader quality assessment and the other update sections will be finalized before publication.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). We thank the MiniMax team for releasing its weights and code.

We thank the NVIDIA Enterprise Products team (Pengcheng Li and Cliff Woolley) for the Video Sparse Attention kernel used by FastH3. We thank the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team (Julius Berner, Chao Liu and Arash Vahdat) for the DMD2 framework and H3 reference work, and for introducing [Parallel Decoding Distillation](https://research.nvidia.com/labs/genair/pdd/).

We thank [Nuva Lab](https://nuvalab.ai/) for bringing production-grounded creative video workflows to FastH3, and the upstream serving, kernel and open-source communities whose implementations make these experiments possible. We also acknowledge the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/) and [MBZUAI](https://mbzuai.ac.ae/) for their support of FastVideo.
