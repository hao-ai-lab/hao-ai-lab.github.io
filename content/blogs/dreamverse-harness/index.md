+++
title = "FastVideo Dreamverse Harness: Local-First AI Filmmaking on Your Own GPUs"
date = 2026-11-19T00:00:00Z
authors = ["FastVideo Team"]
author = "FastVideo Team"
ShowReadingTime = true
ShowToc = false
draft = true
contentClass = "post-content-justified"
url = "/blogs/dreamverse-harness/"
[cover]
    image = "img/cover.jpg"
    alt = "Dreamverse Harness"
    relative = true
    hiddenInSingle = true
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
+++

Today we are open-sourcing **Dreamverse Harness**, an agent harness built upon [FastVideo](https://github.com/hao-ai-lab/FastVideo) that takes local video generation from single clips to complete films.

**Try it:** [GitHub](https://github.com/fastvideo-project/dreamverse-harness) · [Slack](https://join.slack.com/t/fastvideo/shared_invite/zt-4heayzk81-lOkgpvJRoceuZjNP7pXxJA) · [Discord](https://discord.gg/dqc2J2yNQ)

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls autoplay muted playsinline preload="metadata" poster="media/intro-poster.jpg"
    aria-label="Dreamverse Harness introduction video"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/intro.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
</div>

## **From Single-Clip Generation to Film Production**

With [FastVideo](https://github.com/hao-ai-lab/FastVideo) and [FastH3](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD), generating a single video clip on your own GPU is now faster than ever. However, a film is more than one clip. It takes a script, consistent characters across shots, and a final cut—things a video model cannot do on its own.

Today, cloud AIGC studios like Higgsfield, Runway, and TapNow can produce a full film, but they host the entire stack, so you pay credit markups on every generation instead of running on your own hardware. Customization stops at the UI as well: you can adjust presets and pick from their hosted models, but you cannot edit the agent's logic or plug in a model of your own.

Open-source projects have started on this problem, but each one covers only half of it. Some run only on your own GPU, with a fixed set of local models.¹˒² Others direct a whole film, but only through paid cloud APIs.³˒⁴ We built FastVideo Dreamverse Harness to close that gap: a fully open-source harness built upon FastVideo, and open to any model you connect.

## **What FastVideo Dreamverse Harness Does**

Dreamverse Harness is a workspace where you and an agent build a film together. Instead of hiding generation behind a single chat box, it connects an agent to a visual canvas for your assets and a timeline for your final cut—so you can see and edit every piece of the film as it comes together.

**Bring any models to your workflow.** Dreamverse Harness ships with a FastVideo server running [FastH3](https://huggingface.co/collections/FastVideo/fastvideo-fasth3), either on a remote cluster or locally on [DGX Spark](https://haoailab.com/blogs/fasth3-omni-local/). To add another local model or API, you write one provider plugin, or let a coding agent write it for you with the built-in dv-add-model-provider skill.

**An agent equipped to direct.** Dreamverse Harness pairs the agent with structured directing skills, domain-specific prompts, and a dedicated toolset that gives it control over the film—from breaking down a script and locking character references to scheduling shots across models.

{{< image src="img/plan-storyboard.png" alt="The Storyboard note open on the canvas, next to the agent's first shot plan in the chat" width="100%" >}}

**Interactive directing.** Typing one sentence and hoping a finished movie comes out never works for real projects, yet slow and expensive video models used to make back-and-forth iteration impractical. Dreamverse Harness fixes both sides of that loop. First, FastH3 renders in seconds at low cost, so you can watch a shot immediately and steer the next one without losing momentum. Second, every character, scene asset, and take lives on a shared canvas and timeline—so you and the agent can adjust a camera angle, re-roll a single shot, or recut the sequence together instead of regenerating from scratch.

{{< image src="img/chat-request.png" alt="The request in the chat, with the two Nailong pictures as the only references" width="100%" >}}

{{< image src="img/canvas-nodes.png" alt="The canvas, left to right: the notes, the two pictures, the Nailong character and the six rendered shots" width="100%" >}}

{{< image src="img/timeline.png" alt="The timeline paused on shot 3 of the six shots" width="100%" >}}

**Everything is a plugin.** Built on [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)⁵, Dreamverse Harness treats every part of the stack as a modular plugin. You can strip out features you do not need, directly tune the skills and context fed to the agent, or build and distribute your own custom canvas nodes and UI visualizations.

**Runs on DGX Spark, locally or with the cloud.** Dreamverse Harness runs on NVIDIA DGX Spark, the GB10 Grace Blackwell desktop with 128 GB of unified memory. On a single DGX Spark the whole stack runs locally: Qwen 3.8 27B in NVFP4 serves as the agent’s LLM, FastH3 renders the video, and scripts, references and takes never leave the machine. Local is the default, not a requirement. A cloud LLM or a remote FastVideo server is one more entry in the same catalog, so you can direct with a larger cloud model while video stays on your Spark, add remote render servers next to local ones, or grow from one DGX Spark to two, three or four. Up to four connect directly over QSFP cables without a switch, and the harness treats every combination the same way.

<!-- DGX Spark player: one video, four tabs that switch the clip. -->
<figure class="spark-tabs video-embed" style="margin: 1.5rem 0;">
  <div class="spark-tablist" role="tablist" aria-label="Number of DGX Sparks">
    <button type="button" role="tab" id="spark-tab-1" aria-selected="true" aria-controls="spark-panel" tabindex="0"
      data-src="media/spark-1-spark.mp4" data-poster="media/spark-1-spark-poster.jpg"
      data-caption="1 DGX Spark, fully local: Qwen 3.8 27B and FastH3 share one box">1 Spark</button>
    <button type="button" role="tab" id="spark-tab-2" aria-selected="false" aria-controls="spark-panel" tabindex="-1"
      data-src="media/spark-2-sparks.mp4" data-poster="media/spark-2-sparks-poster.jpg"
      data-caption="2 DGX Sparks: three FastH3 models across two boxes, then a 2K upscale">2 Sparks</button>
    <button type="button" role="tab" id="spark-tab-3" aria-selected="false" aria-controls="spark-panel" tabindex="-1"
      data-src="media/spark-3-sparks.mp4" data-poster="media/spark-3-sparks-poster.jpg"
      data-caption="3 DGX Sparks: local Qwen on one box, and a pair joins up for 2K">3 Sparks</button>
    <button type="button" role="tab" id="spark-tab-4" aria-selected="false" aria-controls="spark-panel" tabindex="-1"
      data-src="media/spark-4-sparks.mp4" data-poster="media/spark-4-sparks-poster.jpg"
      data-caption="4 DGX Sparks: four FastH3 models at once, then three 2K upscales at once">4 Sparks</button>
  </div>
  <div id="spark-panel" role="tabpanel" aria-labelledby="spark-tab-1">
    <video controls playsinline preload="metadata" src="media/spark-1-spark.mp4" poster="media/spark-1-spark-poster.jpg"
      aria-label="Dreamverse Harness on 1 DGX Spark"
      style="display: block; width: 100%; height: auto; aspect-ratio: 16 / 9; border-radius: 10px; background: #000;">
      Your browser does not support MP4 video.
    </video>
  </div>
  <figcaption class="spark-caption" aria-live="polite">1 DGX Spark, fully local: Qwen 3.8 27B and FastH3 share one box</figcaption>
</figure>

<style>
  .spark-tablist { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; }
  .spark-tablist [role="tab"] {
    padding: 0.4rem 0.9rem; border: 1px solid var(--tertiary); border-radius: 999px;
    background: var(--entry); color: var(--primary); font: inherit; font-size: 0.95rem; cursor: pointer;
  }
  .spark-tablist [role="tab"][aria-selected="true"] { background: var(--primary); color: var(--theme); border-color: var(--primary); }
  .spark-tablist [role="tab"]:focus-visible { outline: 2px solid #3b6cf6; outline-offset: 2px; }
  .post-content figure > figcaption.spark-caption { margin: 0.5rem 0 0; font-size: 16px; font-weight: 400; color: #808080; text-align: center; }
  @media (max-width: 600px) {
    .spark-tablist { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.35rem; }
    .spark-tablist [role="tab"] { padding: 0.4rem 0.2rem; font-size: 0.9rem; white-space: nowrap; }
  }
</style>

<script>
  // Switch the DGX Spark clip and play it when a tab is selected; arrow keys, Home and End move between tabs.
  (function () {
    var tabs = Array.prototype.slice.call(document.querySelectorAll('.spark-tablist [role="tab"]'));
    var panel = document.getElementById('spark-panel');
    var video = panel.querySelector('video');
    var caption = document.querySelector('.spark-tabs .spark-caption');
    function select(tab) {
      if (tab.getAttribute('aria-selected') === 'true') { play(); return; }
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute('aria-selected', on ? 'true' : 'false');
        t.tabIndex = on ? 0 : -1;
      });
      panel.setAttribute('aria-labelledby', tab.id);
      video.poster = tab.dataset.poster;
      video.src = tab.dataset.src;
      video.setAttribute('aria-label', 'Dreamverse Harness on ' + tab.dataset.caption.split(':')[0].replace(',', ''));
      caption.textContent = tab.dataset.caption;
      video.load();
      play();
    }
    // Play with sound when the browser allows it, otherwise muted.
    function play() {
      var p = video.play();
      if (p) p.catch(function (e) { if (e.name === 'NotAllowedError') { video.muted = true; video.play(); } });
    }
    tabs.forEach(function (tab, i) {
      tab.addEventListener('click', function () { select(tab); });
      tab.addEventListener('keydown', function (e) {
        var next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
        if (next === undefined) return;
        e.preventDefault();
        var target = tabs[(next + tabs.length) % tabs.length];
        target.focus();
        select(target);
      });
    });
  })();
</script>

**FastVideo built in, tuned for DGX Spark.** Dreamverse Harness ships FastVideo as its native video backend, and FastVideo is tuned for the GB10 down to the kernels. Every FastH3 model (V1, V2, Trim and Omni), along with Pika H3 Flex⁶, ships as an NVFP4 build for Blackwell and runs on GB10 kernels: fused NVFP4 matrix multiplies, sparse video attention and an NVFP4 video decoder. The gains add up. When FastH3 first ran on a DGX Spark, a five-second clip took 243 seconds; today five seconds of 480p video with audio renders in about 33 seconds on one Spark and 19 seconds on two. As a result, one DGX Spark can hold the agent’s LLM and a FastH3 model together, or two FastH3 models side by side, and the agent switches between them as the film needs them.

**Hardware-aware control through your agent.** Add each Spark on the new Hardware settings page, and the agent sees every device: whether it is reachable, whether it is busy, and which models it holds. It places each clip on an idle Spark with that model installed, prefers one where the model is already loaded, and renders independent clips in parallel. It loads and unloads models on each device on its own, and you keep the same control from the Generation models page.

<figure class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata" poster="media/spark-4-sparks-switch-poster.jpg"
    aria-label="Dreamverse Harness on 4 DGX Sparks with a 200G switch"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/spark-4-sparks-switch.mp4" type="video/mp4">
    Your browser does not support MP4 video.
  </video>
  <figcaption class="spark-caption">4 DGX Sparks on a 200G switch: the agent renders four shots in parallel, then all four Sparks split each 2K upscale</figcaption>
</figure>

**Upscale videos to 2K.** We adapted NVlabs’ Open-H3-Regenerate-2K⁷, which regenerates a draft at 2560×1440, to DGX Spark. It runs with sparse attention, MXFP8 matrix multiplies and a lighter 24-layer decoder on the GB10, and it splits each clip across two cabled Sparks. Together these take a 2K regeneration on two DGX Sparks from 454 to 158 seconds. The agent sends the takes you keep to the upscaler as a V2V task; with three or more Sparks, drafting with FastH3 continues on the others while a pair regenerates the keeper at 2K.

## **How the Harness Works**

Dreamverse Harness is built on [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) and runs on its plugin system and agent loop. Everything Dreamverse Harness adds is a plugin, and the plugins depend on each other in one direction only. Figure 1 shows the resulting layers from the bottom up. The project layer records the whole life of a project. Above it, the model generation layer holds the model catalog and the providers that serve the models. The feature layer then holds the parts of a film, such as characters, locations, shot renders and the timeline. At the top, workflows such as the story workflow combine these features into one production flow. In addition, each plugin ships the skills that tell the agent how to use it.

The agent is not one more layer: it works across all of them, from workflows down to features and models. It follows the video-directing skill to plan a film, and it acts through the tools that the plugins provide. For example, when a shot needs a render, the agent picks a model from the catalog and follows the prompting skill of that model. Before it changes anything, it reads the project history to know what has already been done.

{{< image src="img/fig1-plugin-layers-v2.png" alt="Figure 1 · The Dreamverse plugin layers; dashed boxes on the right are example skills that the plugins ship" width="100%" title="Figure 1 · The Dreamverse plugin layers; dashed boxes on the right are example skills that the plugins ship" >}}

### **1. The project layer: version control for a film**

Making a film with an agent is iterative. You try a take, compare it with the last one, change a reference photo, and sometimes go back two steps. A project that stores only its current state loses exactly the information this work depends on. We therefore built the project layer to work like version control for a film (Figure 2).

{{< image src="img/fig2-project-layer.png" alt="Figure 2 · The project layer: a history of records and versioned references" width="100%" title="Figure 2 · The project layer: a history of records and versioned references" >}}

Every change, whether you make it on the canvas or the agent makes it from chat, is one record in the project's append-only log, records.jsonl. A record keeps who made the change, from which view, with which inputs, and what it produced. The project state is computed from these records, so all three views always show the same project. Undo and redo move the project's current position along its history, and they delete nothing. As a result, an undone take stays in the log, and you can return to it or compare it with the new one.

References have versions as well. A character such as c1 starts at version c1@1 and becomes c1@2 when you swap its photo, and a shot plan gets a new version each time it is revised. Each render records exactly which versions it read. Consequently, every take can answer which photo and which plan made it. When a newer version replaces one that a take read, the take is flagged, so that you or the agent can decide whether to render it again.

### **2. Composable by design**

A harness for filmmaking has to grow. New models arrive every few weeks, and new kinds of films bring new things to track. In the first version of Dreamverse Harness, growth was expensive, because shared packages named the plugins that used them. For example, adding one reference noun edited the project kernel and four UI packages, and adding image generation touched 35 files.

We therefore rebuilt Dreamverse Harness around registries. The core provides the registries and names no plugin; a plugin adds itself by registering into them. On the host, the project kernel takes components and their operations, and the model registry takes model entries. In the browser, the canvas takes node types, while the history panel and the chat composer take their own rows and mention kinds. Every registration is tied to the lifetime of the plugin that made it, so unloading a plugin removes everything it added.

The canvas is the clearest example in the browser (Figure 3). ui-canvas provides a registry of node types and draws whatever is registered in it. A UI plugin such as ui-character or ui-shot-plan registers its own node type and editor. A new kind of node is therefore one new plugin, and removing that plugin removes its nodes from the canvas.

{{< image src="img/fig3-canvas-registry.png" alt="Figure 3 · The canvas registry: UI plugins register node types, and the canvas draws them" width="100%" title="Figure 3 · The canvas registry: UI plugins register node types, and the canvas draws them" >}}

The model catalog works the same way on the host (Figure 4). Each provider plugin registers one entry per model and task, and the shot render plugin finds the entry by the key that the agent names. As a result, neither the catalog nor the render code names any provider, and adding a model takes one new provider plugin.

{{< image src="img/fig4-model-catalog-v2.png" alt="Figure 4 · The model catalog: providers register model entries, and shot render finds one by key" width="100%" title="Figure 4 · The model catalog: providers register model entries, and shot render finds one by key" >}}

### **3. Workflows compose components**

Each noun of a film, such as a character, a location, a style, a shot plan or a timeline, is one component with its own state slice and operations, and no component imports another. The order of production lives elsewhere, in a workflow. The story workflow that ships with Dreamverse Harness owns two operations (Figure 5). story.preview is read-only: it reports which shots would render, the GPU estimate, and every reason a render would be refused. story.approve always asks for your confirmation and then writes the whole chain: one render for each new or changed shot, then one timeline layout. Every record in that chain carries started_by, which points back to the approval, so the canvas can group the takes of one approval, and a later retake stays in the same chain.

{{< image src="img/fig5-story-workflow.png" alt="Figure 5 · The story workflow: preview, approve, then one chain of records" width="100%" title="Figure 5 · The story workflow: preview, approve, then one chain of records" >}}

We keep the flow out of the components because flows differ while nouns stay the same. For example, an ad cut that renders one plan in two aspect ratios needs a different approval, but the same shot plan and the same render operation. A second workflow is then one new package, and no component changes.

## **Getting Started**

Dreamverse Harness needs two servers: a FastVideo server that renders the video, and an LLM endpoint that runs the agent.

**Dependencies**

* Node.js ^22.19.0 || >=24.0.0 and pnpm 11.7.0
* ffmpeg and ffprobe
* One or more NVIDIA GPUs for the FastVideo server (the stock FastH3 8-step config uses 4)
* An OpenAI-compatible LLM endpoint for the agent (the default model id is deepseek-v4.1)

**Steps**

1. Start a FastVideo server that serves FastH3:

   ```bash
   uv venv --python 3.12 --seed && source .venv/bin/activate
   git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
   UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"
   fastvideo serve --config examples/serving/openai_fasth3_8step.yaml \
     --server.host 127.0.0.1   # port 8000
   ```

2. Get Dreamverse Harness and create its profile:

   ```bash
   git clone https://github.com/fastvideo-project/dreamverse-harness.git dreamverse && cd dreamverse
   pnpm install && pnpm run build
   export DSH_HOME="$HOME/.local/state/dsh-video-harness"
   scripts/video-harness/setup-profile.sh
   ```

3. Point Dreamverse Harness at your servers. Add these rows to $DSH_HOME/profiles/video-harness/cordis.patch.yml:

   ```yaml
   - id: dv-fastvideo
     disabled: false
     config:
       models:
         fasth3:
           t2v:
             baseUrl: http://127.0.0.1:8000
   - id: dv-uniserve
     disabled: true
   ```

   Then export these variables:

   ```bash
   export DV_DEEPSEEK_BASE_URL=[your OpenAI-compatible endpoint, ending in /v1]
   export DV_DEEPSEEK_API_KEY=[your key, or leave unset for a keyless server]
   export DV_FFMPEG="$(command -v ffmpeg)"
   ```

4. Launch the web UI with scripts/video-harness/launch.sh web. It listens on port 8092; set DV_PORT to change it.

5. Open the URL that the launcher prints, for example http://127.0.0.1:8092, and describe your first video in the chat.

### Set up on DGX Spark

The quickest path is to let your coding agent do it. Install our setup skill in Codex or Claude Code and ask it to set up Dreamverse on your DGX Sparks:

```bash
npx skills add fastvideo-project/dreamverse-harness --skill dreamverse-dgx-spark-setup
```

The agent inspects each box, chooses a model layout for the hardware you have, installs FastVideo, the LLM and Dreamverse Harness, and checks every box with a test render. Prefer to do it yourself? The DGX Spark guide in the repository has every step, with a check after each one. The recipes below are the setups we run; mix and extend them freely.

One DGX Spark, fully local. Qwen 3.8 27B (about 39 GB under vLLM) and FastH3 Trim (about 45 GB while rendering) share one box:

1. Prepare the box: DGX OS with CUDA 13, at least 120 GB of free disk, a Hugging Face login, and a memory watchdog, because an out-of-memory condition can freeze a GB10.

2. Download the weights:

   ```bash
   hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir ~/models/Qwen3.8-27B-NVFP4
   hf download FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4 --local-dir ~/models/FastH3-Trim-8-Step-NVFP4
   ```

3. Serve Qwen with vLLM's nightly CUDA 13 wheels on port 8000, capped with `--gpu-memory-utilization 0.32` so FastH3 fits beside it. Start it first, because its first launch compiles kernels.

4. Install FastVideo by its [DGX Spark guide](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/getting_started/installation/spark_performance.md), and serve FastH3 Trim from `examples/serving/openai_fasth3_trim_spark.yaml` with the local weights on port 18000.

5. Build Dreamverse Harness, create its profile, and write a profile patch that points the agent at Qwen and `dv-fastvideo` at FastH3 Trim. Then launch the web UI with `scripts/video-harness/launch.sh web` and open the printed URL in the DGX Spark's browser.

Several DGX Sparks. Every DGX Spark runs its own FastVideo server, bound to its cable or LAN address, with a model catalog (`server.model_catalog`) so the agent can load and unload models on it. In the profile patch, give each model its server's `baseUrl` and set `managed: true`. One FastH3 NVFP4 model takes 40 to 55 GB, so two fit in a DGX Spark, or one beside Qwen.

2K across two DGX Sparks. Two cabled DGX Sparks run one Open-H3-Regenerate-2K server with sequence parallel over Ray (`sp_size: 2`), so each box refines half of each clip.

The [DGX Spark guide](https://github.com/fastvideo-project/dreamverse-harness/blob/main/scripts/video-harness/dgx-spark.md) in the Dreamverse Harness repository has the full commands, with a check after each step.

## **More Example Videos Built with Dreamverse Harness**

A product ad for NVIDIA DGX Spark.

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata" poster="media/example-spark-ad-poster-v2.jpg"
    aria-label="A product ad for NVIDIA DGX Spark"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/example-spark-ad-v2.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
</div>

A Chinese ink-wash animation of the Jingwei myth.

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata" poster="media/example-jingwei-poster.jpg"
    aria-label="A Chinese ink-wash animation of the Jingwei myth"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/example-jingwei.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
</div>

A nature documentary about fungi.

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata" poster="media/example-fungi-poster.jpg"
    aria-label="A nature documentary about fungi"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/example-fungi.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
</div>

A sci-fi scene aboard a starship.

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata" poster="media/example-scifi-poster.jpg"
    aria-label="A sci-fi scene aboard a starship"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/example-scifi.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
</div>

Rough Blender scenes rendered into action shots.

<div class="video-embed" style="width: 100%; margin: 1.5rem 0;">
  <video controls playsinline preload="metadata" poster="media/example-david-poster.jpg"
    aria-label="Rough Blender scenes rendered into action shots"
    style="display: block; width: 100%; height: auto; border-radius: 10px; background: #000;">
    <source src="media/example-david.mp4" type="video/mp4">
    Your browser does not support MP4 video with audio.
  </video>
</div>

Join us on [Slack](https://join.slack.com/t/fastvideo/shared_invite/zt-4heayzk81-lOkgpvJRoceuZjNP7pXxJA) or [Discord](https://discord.gg/dqc2J2yNQ), or open an issue on [GitHub](https://github.com/fastvideo-project/dreamverse-harness).

## **References**

1. WanGP. [github.com/deepbeepmeep/WanGP](https://github.com/deepbeepmeep/WanGP)
2. LTX Desktop. [github.com/Lightricks/LTX-Desktop](https://github.com/Lightricks/LTX-Desktop)
3. OpenMontage. [github.com/calesthio/OpenMontage](https://github.com/calesthio/OpenMontage)
4. ViMax. [github.com/HKUDS/ViMax](https://github.com/HKUDS/ViMax)
5. DeepSeek Harness. [github.com/deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)
6. Pika H3 Flex. [huggingface.co/Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex)
7. Open-H3-Regenerate-2K. [huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
8. vLLM-Omni. [github.com/vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)

## **Acknowledgement**

We thank NVIDIA for their compute support and collaboration.

## **FastVideo Team**

**Contributors:** [Junda Su](https://davids048.github.io/), [Aryan Kumar](https://github.com/aryan5v), [Kun Lin](https://github.com/klhhhhh), [Kyle Hu](https://github.com/KyleNeverGivesUp), [Kevin Lin](https://github.com/kevin314), [Loay Rashid](https://github.com/loaydatrain), [Junda Chen](https://github.com/GindaChen), [Ishan Vaish](https://github.com/ivaish)  
**Tech lead:** [Will Lin](https://solitarythinker.github.io/)  
**Advisor:** [Hao Zhang](https://haozhang.ai/) (corresponding)
