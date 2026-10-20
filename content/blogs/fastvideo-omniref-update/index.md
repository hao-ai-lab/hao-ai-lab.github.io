+++
title = "FastH3 OmniRef Release"
date = 2026-10-19T00:00:00Z
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = false
ShowToc = false
draft = false
contentClass = "post-content-justified omniref-article"
omnirefInteractive = true
[cover]
    image = "img/cover.svg"
    alt = "FastH3 OmniRef: 5.6–11.9 times lower end-to-end latency across 27 matched workloads on eight GB200 GPUs, compared with Base’s lossless-policy serving recipe."
    relative = true
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
+++

{{< socialBadges github="hao-ai-lab/FastVideo" slack="https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ" huggingface="https://huggingface.co/collections/FastVideo/fasth3-omniref" >}}

FastVideo is open-sourcing **FastH3 OmniRef**, a distilled MiniMax H3 model for reference-guided video and audio. Give it a prompt plus images, video clips and a voice reference to generate a finished video with synchronized sound.

[Model weights](https://huggingface.co/collections/FastVideo/fasth3-omniref) · [Deployment guide](#deploy-fasth3-omniref-with-uniserve)

ComfyUI support is available. Our local AI blog post and release are planned for tomorrow, **October 21**.

## TL;DR

- **Eight transformer evaluations**, down from Base’s 50, with image, video and audio references.
- **5.6–11.9× lower end-to-end latency** across 27 matched workloads on eight GB200 GPUs. This compares full serving recipes, using Base’s `lossless` numerical policy; [methodology](#benchmark-methodology).
- **See and hear the outputs below.** Change the references or clip length, then play the matched examples together. Reference inputs are expanded; open the full prompt when you need it.
- **Partners.** Thanks to [Nuva Lab](https://nuvalab.ai/) and [Reactor Inc.](https://www.reactor.inc/) for their collaboration and support.

## Matched reference inputs and outputs

Choose a reference mix and clip length to compare **Base, FastH3 Omni and FastH3 Omni with the 36-layer VAE**. All examples use seed 0 and matched inputs; these are qualitative comparisons, not evidence of output parity.

{{< omniref-gallery >}}

The [36-layer VAE](/blogs/fasth3-v21/#nvfp4-vae-decoders) is shown for quality comparison; its latency is not included in the timing data.

## Results at a glance

Across 27 matched workloads, FastH3 Omni delivers **5.6–11.9× lower end-to-end latency** than Base on eight GB200 GPUs. Timings include reference encoding, generation, decoding and the finished MP4, at a 1344×768 target canvas.

Both lanes use the original video VAE. Base uses SGLang’s `quality="lossless"` policy, not its stricter `exact` policy. This compares complete model and serving recipes; see the [benchmark methodology](#benchmark-methodology).

{{< omniref-results >}}

## PDD + DMD

FastH3 OmniRef combines **Parallel Decoding Distillation (PDD)** with **Distribution Matching Distillation (DMD)**. PDD predicts several denoising updates per transformer evaluation; DMD helps train the completed video and audio output. Reference conditioning is retained throughout generation. For the method and its illustrations, see the [PDD paper](https://arxiv.org/abs/2607.26004).

| Training | Amount |
| --- | --- |
| GPUs | 64 NVIDIA B300 |
| Training steps | 3,500 |
| Estimated compute | Approximately 7,800 GPU-hours |

## Deploy FastH3 OmniRef with UniServe

Find the checkpoints in the [FastH3 OmniRef weights collection](https://huggingface.co/collections/FastVideo/fasth3-omniref).

The walkthrough below serves the same [eight-forward OmniRef student](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) used in the gallery, with its original video VAE. It starts with one image and one audio reference on **four GB200 GPUs**, then shows the **eight-GPU, two-host** configuration for the larger reference mixes. The model's task is **`ref2va`**: reference-conditioned video and audio.

{{< omniref-details title="Open the complete deployment walkthrough" >}}

Install [UniServe](https://github.com/hao-ai-lab/uniserve) and apply the [large-index correction](deploy/uniserve-wide-index.patch) below for large reference mixes. For other hardware and deployment layouts, see the [UniServe FastH3 guide](https://github.com/hao-ai-lab/uniserve/blob/main/docs/minimax_h3/minimax_h3.md).

{{< omniref-details title="1. Install UniServe and its GPU dependencies" id="1-install-uniserve-and-its-gpu-dependencies" >}}

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

{{< /omniref-details >}}

{{< omniref-details title="2. Download the model weights" id="2-download-the-model-weights" >}}

Keep the shell in the UniServe checkout. Authenticate with a Hugging Face account that has access to the model when required, then download it:

```bash
hf auth login

export OMNIREF_MODEL="$PWD/models/FastH3-Omni-8-Step-V1"
hf download FastVideo/FastVideo-FastH3-Omni-8-Step-V1 \
  --local-dir "$OMNIREF_MODEL"
```

The download supplies `transformer_ref`, the schedulers and `fastvideo_inference.json`. UniServe automatically downloads the required text encoder, tokenizer, processor and video/audio VAEs into the Hugging Face cache when they are missing. Keep that cache writable and allow network access during the first startup. Use the native UniServe-compatible model directory, including `fastvideo_inference.json`.

{{< /omniref-details >}}

{{< omniref-details title="3. Prepare a real gallery request" id="3-prepare-a-real-gallery-request" >}}

The [downloadable request helper](deploy/prepare_request.py) downloads a gallery profile's reference files and writes a ready-to-send JSON request. It preserves the full prompt, ordered conditions, output geometry and seed. Run it from the same checkout before starting the server:

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

This selects the gallery's **Image + voice, 5-second** example. The generated conditions use absolute `file://` URIs under `OMNIREF_MEDIA`. Those paths must exist on the server; a file on a separate client machine is not automatically uploaded. The helper uses the public article URL by default; when previewing locally, pass the preview's article URL with `--blog-url`.

{{< /omniref-details >}}

{{< omniref-details title="4. Start the reference-conditioned server" id="4-start-the-reference-conditioned-server" >}}

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

{{< /omniref-details >}}

{{< omniref-details title="5. Generate and download video with audio" id="5-generate-and-download-video-with-audio" >}}

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

{{< /omniref-details >}}

{{< omniref-details title="6. Serve the larger mixes on eight GB200 GPUs" id="6-serve-the-larger-mixes-on-eight-gb200-gpus" >}}

For four GPUs on each of two hosts, install the same UniServe source and environment on both, with identical absolute paths for the checkout, Python environment, model and Hugging Face cache. Share or replicate those model/cache directories and the media directory at the same paths. Use a network that connects the workers' NCCL traffic and host-launcher connections.

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

The helper also accepts `image`, `images3`, `images9`, `clip`, `video`, `clips3` and `mixed`. Keep FFmpeg and `ffprobe` on `PATH`, or set `--ffmpeg /absolute/path/to/ffmpeg --ffprobe /absolute/path/to/ffprobe` on the server. Keep the original VAE for the timing lane in this article. The gallery's trained 36-layer decoder requires a separately composed checkpoint; selecting a generic quantization preset does not install that decoder.

{{< /omniref-details >}}

{{< omniref-details title="Deployment checks" id="deployment-checks" >}}

| Symptom | What to check |
| --- | --- |
| A checkpoint download returns 401 or 404 | Confirm the exact repository name, the release's availability and the account used by `hf auth login`. |
| A component is missing at startup | Check network access, Hugging Face authentication and write access to the cache so UniServe can download the required files. |
| A reference file is refused | Its absolute path must exist under `--media-directory` on the server. Keep FFmpeg and `ffprobe` available to that process. |
| Startup runs out of memory or a request exceeds a bound | Use the appropriate deployment and explicit duration, prompt and condition capacities. Larger or longer references need more storage; increasing a bound does not create GPU memory. |
| A two-host head waits for a launcher | Match `rank-0`/`rank-1`, connect to the logged launcher port, and check that both hosts use the same source, environment and absolute paths. |

For a container deployment, follow the [UniServe Docker build instructions](https://github.com/hao-ai-lab/uniserve/blob/main/docs/minimax_h3/minimax_h3.md#build-and-run-with-docker), include FFmpeg in the image, and mount the model, Hugging Face cache and reference directory at the paths the server uses. Worker IPC also needs sufficient shared memory and pinned-memory access. The native installation above is the complete walkthrough for this article.

{{< /omniref-details >}}

{{< /omniref-details >}}

## Benchmark methodology

Nine reference compositions × three clip lengths (5, 10 and 15 seconds), measured on eight GB200 GPUs. Each profile uses one warmup and three measured seeds; the gallery consistently shows seed 0.

{{< omniref-details title="Read measurement settings and numerical policy" >}}

- **Hardware and scope.** Eight NVIDIA GB200 GPUs across two four-GPU trays. GB200 is our proxy for B200; B200 was not separately measured. These results describe the eight-GPU cohort.
- **Matched inputs.** Nine reference compositions × three requested lengths, all at 1344×768 and 24 FPS. The aligned outputs contain 124, 243 or 362 frames. Images use the released enlarged-reference mode with a 2048-pixel short edge. Reference clips retain their real soundtracks and standalone audio remains a real input.
- **Measurement.** One request at a time, one warmup with seed 42 for each profile, then measured seeds 0, 1 and 2. The tables report arithmetic mean client end-to-end time until the complete MP4 is available; the downloadable data also includes sample standard deviations. Model loading and server startup are excluded. No latency is inferred from missing cells or memory failures, and no timing is normalized to another workload.
- **Model and runtime.** Base is MiniMax H3 Ref2VA with 50 denoising evaluations; FastH3 Omni uses the 32-interval PDD student with eight fused evaluations and VSA. The Base lane uses Ulysses 4 × Ring 2 with the native FlashAttention transformer backend; the student uses eight-way Ulysses. Conditioning caches are disabled. Model weights, attention patterns and serving implementations differ, so this compares complete model/runtime recipes, not an isolated distillation or kernel ablation.
- **Base numerical policy.** These Base measurements use SGLang's `quality="lossless"` policy with original unquantized weights, dense attention and the model's mixed-precision rules. Its DiT normalization/modulation and VAE fusions can change intermediate rounding. They are not measurements of SGLang's stricter `quality="exact"` policy, and reference-exact or perceptual equivalence has not been established. The original model is not uniformly BF16: it retains FP32-sensitive DiT operations and the released VAE's mixed-precision execution. The displayed latencies have not been converted to an exact-policy estimate.
- **Reading the examples.** The timing statistics use three seeds; the gallery consistently uses seed 0. A shared seed does not imply numerical or perceptual output parity across different models or runtimes. The examples support direct inspection and are not a quantitative quality evaluation.

{{< /omniref-details >}}

## Get started

Download the model weights from the [FastH3 OmniRef collection](https://huggingface.co/collections/FastVideo/fasth3-omniref), then follow the [UniServe setup](#deploy-fasth3-omniref-with-uniserve) to generate video and audio from your own references.

## Acknowledgements

FastH3 builds on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). We thank the MiniMax team for releasing its weights and code.

We thank the NVIDIA Enterprise Products team (Pengcheng Li and Cliff Woolley) for the Video Sparse Attention kernel used by FastH3. We thank the [NVIDIA FastGen](https://github.com/NVlabs/FastGen) team (Julius Berner, Chao Liu and Arash Vahdat) for the DMD2 framework and H3 reference work, and for introducing [Parallel Decoding Distillation](https://research.nvidia.com/labs/genair/pdd/).

We thank [Nuva Lab](https://nuvalab.ai/) for bringing production-grounded creative video workflows to FastH3, and the upstream serving, kernel and open-source communities whose implementations make these experiments possible. We also acknowledge [Reactor Inc.](https://www.reactor.inc/) for supporting FastH3 OmniRef, and the [vLLM project](https://vllm.ai/), [NVIDIA](https://www.nvidia.com/en-us/) and [MBZUAI](https://mbzuai.ac.ae/) for their support of FastVideo.
