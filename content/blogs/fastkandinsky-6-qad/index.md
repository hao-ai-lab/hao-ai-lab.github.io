+++
title = "FastKandinsky 6 QAD"
date = 2026-11-24T00:00:00-08:00
url = "/blogs/fastkandinsky-6-qad/"
description = "30× faster. 39% lighter."
authors = ["Kirill Kozlov", "Lev Novitskiy", "Aryan Kumar", "Will Lin", "Hao Zhang"]
author = "Kirill Kozlov, Lev Novitskiy, Aryan Kumar, Will Lin, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com/haoailab"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
[cover]
    image = "assets/posters/promo-cheburashka.jpg"
    alt = "FastKandinsky 6 QAD"
    hiddenInSingle = true
+++

<div class="fk6-article" id="fk6" data-media-base="">
<style>
#fk6 {
  --fk6-ours: #1f5fc4;
  --fk6-ours-bg: rgba(42, 111, 219, .06);
  --fk6-chart-bg: #fcfcfb;
  --fk6-grid: #e4e3de;
  --fk6-axis: #9b9a95;
  --fk6-t1: #0b0b0b;
  --fk6-t2: #52514e;
  --fk6-s0: #b5afa5;
  --fk6-s1: #33557d;
  --fk6-s2: #b86a3c;
  --fk6-on-fill: #fff;
}
.dark #fk6 {
  --fk6-ours: #8fb3ee;
  --fk6-ours-bg: rgba(115, 147, 187, .12);
  --fk6-chart-bg: #1a1a19;
  --fk6-grid: #2e2e2c;
  --fk6-axis: #6b6a66;
  --fk6-t1: #fff;
  --fk6-t2: #c3c2b7;
  --fk6-s0: #69655e;
  --fk6-s1: #7393bb;
  --fk6-s2: #c27448;
  --fk6-on-fill: #111;
}
#fk6 .fk6-badges { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 var(--content-gap, 20px); }
#fk6 .fk6-badges a { display: inline-flex; }
#fk6 .fk6-badges img { display: block; height: 28px; margin: 0; border-radius: 0; }
#fk6 .fk6-video { display: block; width: 100%; height: auto; aspect-ratio: 3 / 2; margin: 0; border-radius: 6px; background: #000; object-fit: contain; }
#fk6 .fk6-missing { display: grid; place-items: center; aspect-ratio: 3 / 2; border: 1px solid var(--border); border-radius: 6px; color: var(--secondary); font-size: .8rem; }
#fk6 figure { margin: 0 0 var(--content-gap, 20px); }
#fk6 figure > figcaption { margin: .5rem 0 0; color: var(--secondary); font-size: .9rem; font-weight: 400; line-height: 1.5; }
#fk6 .fk6-hero { margin: 1.2rem 0 1.8rem; }
#fk6 .fk6-hero .fk6-video { border-radius: 10px; }
#fk6 .fk6-prompt summary { cursor: pointer; }
#fk6 .fk6-prompt-label { margin-right: .4rem; color: var(--primary); font-weight: 600; }
#fk6 .fk6-prompt[open] .fk6-prompt-preview { display: none; }
#fk6 .fk6-prompt p { margin: .5rem 0 0; font-size: .85rem; white-space: pre-wrap; overflow-wrap: anywhere; }
#fk6 .fk6-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.4rem 1rem; align-items: start; margin: 0 0 var(--content-gap, 20px); }
#fk6 .fk6-grid figure { margin: 0; }
#fk6 .fk6-compare { margin: 32px 0 var(--content-gap, 20px); }
#fk6 .fk6-compare > summary { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; list-style: none; cursor: pointer; }
#fk6 .fk6-compare > summary::-webkit-details-marker { display: none; }
#fk6 .fk6-compare > summary::before { content: '▸'; color: var(--secondary); transition: transform .15s; }
#fk6 .fk6-compare[open] > summary::before { transform: rotate(90deg); }
#fk6 .fk6-compare[open] > summary { margin-bottom: 24px; }
#fk6 .fk6-compare > summary h2 { margin: 0; }
#fk6 .fk6-count { color: var(--secondary); font-size: .9rem; }
#fk6 .fk6-compare-item { margin: 0 0 1.8rem; }
#fk6 .fk6-compare-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .75rem; }
#fk6 .fk6-compare-grid h3 { margin: 0 0 .35rem; color: var(--primary); font-size: .9rem; font-weight: 700; line-height: 1.3; }
#fk6 .fk6-compare-grid .ours h3 { color: var(--fk6-ours); }
#fk6 .fk6-compare-grid .ours .fk6-video { box-shadow: 0 0 0 2px var(--fk6-s1); }
#fk6 .fk6-setup { margin: -12px 0 1rem; color: var(--secondary); font-size: .9rem; }
#fk6 .fk6-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 0 0 1rem; }
#fk6 .fk6-stat-value { display: block; color: var(--fk6-ours); font-size: 1.9rem; font-weight: 700; line-height: 1.2; }
#fk6 .fk6-stat-label { display: block; color: var(--secondary); font-size: .8rem; line-height: 1.4; }
#fk6 .fk6-table { margin: 1rem 0 .6rem; overflow-x: auto; }
#fk6 .fk6-table table { display: table; width: 100%; min-width: 560px; margin: 0; border-collapse: collapse; font-size: .9rem; font-variant-numeric: tabular-nums; }
#fk6 .fk6-table th, #fk6 .fk6-table td { min-width: 0; padding: .45rem .6rem; border-bottom: 1px solid var(--border); line-height: 1.5; text-align: right; white-space: nowrap; }
#fk6 .fk6-table th:first-child, #fk6 .fk6-table td:first-child { text-align: left; white-space: normal; }
#fk6 .fk6-table thead th { color: var(--primary); font-weight: 700; }
#fk6 .fk6-table tbody th { color: var(--content); font-weight: 400; }
#fk6 .fk6-table tr.ours { background: var(--fk6-ours-bg); }
#fk6 .fk6-table tr.ours th { color: var(--primary); font-weight: 700; }
#fk6 .fk6-table td.win { color: var(--fk6-ours); font-weight: 700; }
#fk6 .fk6-model-details { display: block; color: var(--secondary); font-size: .8rem; font-weight: 400; }
#fk6 .fk6-note { margin: .2rem 0 1.4rem; color: var(--secondary); font-size: .8rem; }
#fk6 .fk6-figures { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: stretch; gap: 1rem; margin: 1rem 0 var(--content-gap, 20px); }
#fk6 .fk6-figures > figure { display: flex; flex-direction: column; min-width: 0; margin: 0; }
#fk6 .fk6-panel { position: relative; display: flex; flex: 1; flex-direction: column; padding: 18px 20px 14px; border: 1px solid var(--border); border-radius: var(--radius, 8px); background: var(--fk6-chart-bg); color: var(--fk6-t1); }
#fk6 .fk6-panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-bottom: 14px; }
#fk6 .fk6-panel-title { color: var(--fk6-t1); font-size: .85rem; font-weight: 700; line-height: 1.35; }
#fk6 .fk6-panel-sub { display: block; color: var(--fk6-t2); font-size: .75rem; font-weight: 400; }
#fk6 .fk6-panel-total { color: var(--fk6-t1); font-size: 1rem; font-weight: 700; white-space: nowrap; font-variant-numeric: tabular-nums; }
#fk6 .fk6-stages { display: grid; flex: 1; grid-template-columns: clamp(56px, 12vw, 80px) minmax(0, 1fr); align-items: center; gap: 12px clamp(12px, 3vw, 24px); }
#fk6 .fk6-stage-bar { display: grid; height: 232px; gap: 2px; }
#fk6 .fk6-segment { display: grid; place-items: center; min-height: 0; overflow: hidden; border-radius: 3px; background: var(--stage); color: var(--fk6-on-fill); font-size: .72rem; font-weight: 700; white-space: nowrap; }
#fk6 .stage-transformer { --stage: var(--fk6-s1); }
#fk6 .stage-vae { --stage: var(--fk6-s2); }
#fk6 .stage-other { --stage: var(--fk6-s0); }
#fk6 .fk6-timings { display: table; width: 100%; min-width: 0; margin: 0; border-collapse: collapse; font-size: .85rem; font-variant-numeric: tabular-nums; }
#fk6 .fk6-timings th, #fk6 .fk6-timings td { min-width: 0; padding: 6px 0 6px 10px; border-bottom: 1px solid var(--fk6-grid); color: var(--fk6-t1); line-height: 1.4; text-align: right; }
#fk6 .fk6-timings th:first-child { padding-left: 0; text-align: left; white-space: normal; }
#fk6 .fk6-timings thead th { color: var(--fk6-t2); font-size: .75rem; font-weight: 400; }
#fk6 .fk6-timings tr > :last-child { width: 1%; white-space: nowrap; }
#fk6 .fk6-timings tbody tr:last-child > * { border-bottom: 0; }
#fk6 .fk6-component { display: flex; align-items: center; gap: 8px; font-weight: 600; }
#fk6 .fk6-component-detail { display: block; color: var(--fk6-t2); font-size: .75rem; font-weight: 400; }
#fk6 .fk6-leader { flex: 1; min-width: 16px; border-top: 1px solid var(--stage); opacity: .55; }
#fk6 .fk6-swatch { flex: 0 0 10px; height: 10px; border-radius: 3px; background: var(--stage); }
#fk6 .fk6-plot { position: relative; display: grid; flex: 1 0 200px; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 26px; border-bottom: 1px solid var(--fk6-axis); }
#fk6 .fk6-col { position: relative; border-radius: 6px; }
#fk6 .fk6-bar { position: absolute; bottom: 0; left: 50%; width: 24px; min-height: 2px; transform: translateX(-50%); border-radius: 4px 4px 0 0; background: var(--fk6-s0); transition: filter .15s; }
#fk6 .fk6-col.ours .fk6-bar { background: var(--fk6-s1); }
#fk6 .fk6-col:hover .fk6-bar, #fk6 .fk6-col:focus-visible .fk6-bar { filter: brightness(1.12); }
#fk6 .fk6-value { position: absolute; left: -12px; right: -12px; color: var(--fk6-t1); font-size: .75rem; font-weight: 600; line-height: 16px; text-align: center; white-space: nowrap; font-variant-numeric: tabular-nums; }
#fk6 .fk6-col.ours .fk6-value { font-weight: 800; }
#fk6 .fk6-xlabels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 7px; }
#fk6 .fk6-xlabels span { padding: 0 2px; color: var(--fk6-t2); font-size: .72rem; line-height: 1.3; text-align: center; }
#fk6 .fk6-xlabels .ours { color: var(--fk6-t1); font-weight: 700; }
#fk6 .fk6-ref { position: absolute; height: 0; border-top: 1px solid var(--fk6-axis); opacity: .6; pointer-events: none; }
#fk6 .fk6-arrow { position: absolute; display: flex; flex-direction: column; align-items: center; gap: 6px; width: 44px; transform: translateX(-50%); color: var(--fk6-s1); pointer-events: none; }
#fk6 .fk6-arrow-line { position: relative; flex: 1; width: 2px; border-radius: 1px; background: currentColor; }
#fk6 .fk6-arrow-line::before { content: ''; position: absolute; left: 50%; width: 8px; height: 8px; border: 0 solid currentColor; }
#fk6 .fk6-arrow-line:first-child::before { top: 0; border-width: 2px 0 0 2px; transform: translate(-50%, 1px) rotate(45deg); }
#fk6 .fk6-arrow-line:last-child::before { bottom: 0; border-width: 0 2px 2px 0; transform: translate(-50%, -1px) rotate(45deg); }
#fk6 .fk6-arrow-label { font-size: 1.25rem; font-weight: 800; line-height: 1; white-space: nowrap; }
#fk6 .fk6-tip { position: absolute; z-index: 2; padding: 8px 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--entry); box-shadow: 0 6px 18px rgba(0, 0, 0, .14); color: var(--secondary); font-size: .75rem; line-height: 1.45; white-space: nowrap; transform: translate(-50%, calc(-100% - 8px)); pointer-events: none; }
#fk6 .fk6-tip strong { display: block; color: var(--primary); font-size: .9rem; font-variant-numeric: tabular-nums; }
#fk6 .fk6-tip-model { color: var(--primary); font-weight: 600; }
#fk6 .fk6-icon-sprite { position: absolute; width: 0; height: 0; overflow: hidden; }
#fk6 .fk6-team { line-height: 1.9; }
#fk6 .fk6-team strong { color: var(--primary); }
#fk6 .fk6-social { display: inline-block; margin-left: .15em; color: var(--primary); line-height: 1; vertical-align: -.1em; }
#fk6 .fk6-social:hover { opacity: .7; text-decoration: none; }
#fk6 .fk6-icon { display: block; width: 1em; height: 1em; fill: currentColor; }
@media (max-width: 760px) {
  #fk6 .fk6-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  #fk6 .fk6-figures { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 480px) {
  #fk6 .fk6-grid, #fk6 .fk6-compare-grid { grid-template-columns: minmax(0, 1fr); }
  #fk6 .fk6-stats { grid-template-columns: minmax(0, 1fr); gap: .6rem; }
  #fk6 .fk6-panel { padding: 14px 12px 10px; }
  #fk6 .fk6-stage-bar { height: 208px; }
}
</style>
<div class="fk6-badges">
<a href="https://github.com/hao-ai-lab/FastVideo" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/github/stars/hao-ai-lab/FastVideo?style=social" alt="FastVideo on GitHub"></a>
<a href="https://huggingface.co/kandinskylab/FastKandinsky-6.0-Pro-NVPF4-QAD" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20huggingface-model-green.svg?style=social" alt="Model on Hugging Face"></a>
</div>
<p data-fk6-slot="lead"></p>
<figure class="fk6-hero">
<div data-fk6-slot="intro"></div>
<figcaption>
<details class="fk6-prompt">
<summary><span class="fk6-prompt-label">Prompt</span><span class="fk6-prompt-preview">Cheburashka peeks over a crate of oranges and delivers the FastVideo line.</span></summary>
<p>A five-second cinematic teaser in a horizontal frame, filmed like handcrafted stop-motion in a miniature animation studio. Cheburashka, a small brown puppet with enormous round ears, a warm tan face, and shiny curious eyes, peeks over the edge of a wooden crate full of oranges. An amber desk lamp catches the texture of his fur and the dimpled orange peels. A turquoise rim light outlines his ears. The camera moves slowly closer at eye level in one continuous shot, with a dark workshop softly out of focus behind him.</p>
<p>0:00–0:01: Cheburashka lifts his head above the crate, and one orange rolls gently across the tabletop. 0:01–0:04.6: He looks straight into the camera and says with playful confidence: &lt;S&gt;Kandinsky by FastVideo is now hyper-fast. Cameras ready? Let's roll!&lt;E&gt; His mouth remains visible and moves in sync with the words. He holds a small blank turquoise clapperboard just below his chin. 0:04.6–0:05: On the word roll, he snaps the clapperboard shut with a grin. Hold the final close-up on his face.</p>
<p>&lt;AUDCAP&gt;Start with quiet workshop room tone and the soft rumble of one orange rolling on wood. Cheburashka speaks in lively, clearly articulated English with a warm, slightly high voice. The line is close and centered, with a quick playful pause after hyper-fast and after cameras ready. Keep the voice clear above the room tone, with no music or background voices under the dialogue. End with a crisp wooden clapperboard snap tightly synchronized to the final gesture.&lt;ENDAUDCAP&gt;</p>
</details>
</figcaption>
</figure>
<h2 id="fk6-samples">NVFP4 video and audio samples</h2>
<div class="fk6-grid" data-fk6-slot="gallery"></div>
<details class="fk6-compare">
<summary><h2 id="fk6-comparison">Base vs. Distill vs. NVFP4 QAD</h2><span class="fk6-count" data-fk6-slot="compare-count"></span></summary>
<div data-fk6-slot="compare"></div>
</details>
<h2 id="fk6-speed">Inference speed</h2>
<p class="fk6-setup" data-fk6-slot="setup"></p>
<div class="fk6-stats">
<div><strong class="fk6-stat-value" data-fk6-slot="stat-sft"></strong><span class="fk6-stat-label" data-fk6-slot="stat-sft-label">NVFP4 E2E vs. Pro Base BF16</span></div>
<div><strong class="fk6-stat-value" data-fk6-slot="stat-distill"></strong><span class="fk6-stat-label" data-fk6-slot="stat-distill-label">NVFP4 E2E vs. Distill BF16</span></div>
<div><strong class="fk6-stat-value" data-fk6-slot="stat-dit"></strong><span class="fk6-stat-label">NVFP4 DiT forward vs. Distill BF16</span></div>
</div>
<div class="fk6-table" tabindex="0" role="region" aria-label="Inference speed">
<table>
<thead>
<tr>
<th scope="col">Model</th>
<th scope="col" data-fk6-slot="total-heading" title="End-to-end generation time">E2E (s)</th>
<th scope="col">DiT / forward (s)</th>
<th scope="col" title="Peak PyTorch allocation during generation">Peak allocated VRAM (GiB)</th>
</tr>
</thead>
<tbody data-fk6-slot="metrics"></tbody>
</table>
</div>
<p class="fk6-note" data-fk6-slot="setup-note"></p>
<h2 id="fk6-timings">FastVideo component timings</h2>
<div class="fk6-figures">
<figure>
<div class="fk6-panel">
<div class="fk6-panel-head" data-fk6-slot="stages-head"></div>
<div class="fk6-stages">
<div class="fk6-stage-bar" data-fk6-slot="stages-bar" aria-hidden="true"></div>
<table class="fk6-timings">
<thead><tr><th scope="col">Component</th><th scope="col">Time (s)</th></tr></thead>
<tbody data-fk6-slot="stages-table"></tbody>
</table>
</div>
</div>
<figcaption data-fk6-slot="stages-caption"></figcaption>
</figure>
<figure role="group" aria-label="End-to-end generation time" data-fk6-slot="speed-figure">
<div class="fk6-panel" data-fk6-slot="speed"></div>
<figcaption data-fk6-slot="speed-caption"></figcaption>
</figure>
</div>
<h2 id="fk6-team">Team</h2>
<svg class="fk6-icon-sprite" aria-hidden="true" focusable="false"><symbol id="fk6-icon-github" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></symbol><symbol id="fk6-icon-linkedin" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></symbol><symbol id="fk6-icon-x" viewBox="0 0 24 24"><path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z"/></symbol></svg>
<p class="fk6-team">
<strong>Contributor:</strong> <a href="https://github.com/KozlovKY">Kirill Kozlov</a> <a class="fk6-social" href="https://github.com/KozlovKY" aria-label="Kirill Kozlov GitHub"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-github"></use></svg></a> <a class="fk6-social" href="https://www.linkedin.com/in/kirill-kozlov-288464273/" aria-label="Kirill Kozlov LinkedIn"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-linkedin"></use></svg></a><br>
<strong>Contributor:</strong> <a href="https://github.com/leffff">Lev Novitskiy</a> <a class="fk6-social" href="https://github.com/leffff" aria-label="Lev Novitskiy GitHub"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-github"></use></svg></a> <a class="fk6-social" href="https://www.linkedin.com/in/lev-novitskiy/" aria-label="Lev Novitskiy LinkedIn"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-linkedin"></use></svg></a> <a class="fk6-social" href="https://x.com/leffffffffffff1" aria-label="Lev Novitskiy X"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-x"></use></svg></a><br>
<strong>Contributor:</strong> <a href="https://github.com/aryan5v">Aryan Kumar</a> <a class="fk6-social" href="https://github.com/aryan5v" aria-label="Aryan Kumar GitHub"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-github"></use></svg></a> <a class="fk6-social" href="https://www.linkedin.com/in/aryan-kumar01" aria-label="Aryan Kumar LinkedIn"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-linkedin"></use></svg></a> <a class="fk6-social" href="https://x.com/aryan_xv" aria-label="Aryan Kumar X"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-x"></use></svg></a><br>
<strong>Tech lead:</strong> <a href="https://github.com/SolitaryThinker">Will Lin</a> <a class="fk6-social" href="https://github.com/SolitaryThinker" aria-label="Will Lin GitHub"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-github"></use></svg></a> <a class="fk6-social" href="https://www.linkedin.com/in/will-lin-294920100" aria-label="Will Lin LinkedIn"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-linkedin"></use></svg></a> <a class="fk6-social" href="https://x.com/wlsaidhi" aria-label="Will Lin X"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-x"></use></svg></a><br>
<strong>Advisor:</strong> <a href="https://github.com/zhisbug">Hao Zhang</a> <a class="fk6-social" href="https://github.com/zhisbug" aria-label="Hao Zhang GitHub"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-github"></use></svg></a> <a class="fk6-social" href="https://www.linkedin.com/in/haozhangml" aria-label="Hao Zhang LinkedIn"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-linkedin"></use></svg></a> <a class="fk6-social" href="https://x.com/haozhangml" aria-label="Hao Zhang X"><svg class="fk6-icon" aria-hidden="true"><use href="#fk6-icon-x"></use></svg></a>
</p>
<!-- Set the opening clip and its optional poster relative to this HTML file. -->
<script type="application/json" data-fk6="intro">{"video": "assets/videos/promo-cheburashka.mp4", "poster": "assets/posters/promo-cheburashka.jpg"}</script>
<script type="application/json" data-fk6="gallery">[{"id": "3", "prompt": "Two-dimensional anime with crisp outlines, contrasting color planes, and luminous geometric magic, 16:9 frame. On a transparent circular platform above violet storm clouds, a young woman in a white-and-blue coat holding a silver staff faces a young man in a black-and-crimson mantle wearing an amber magical gauntlet. One continuous medium-wide shot; the camera moves quickly through a short arc around both characters. 0:00–0:01: the woman warns: \u003cS>Watch out!\u003cE> 0:01–0:03: she swings her staff and launches a single blue energy crescent; the man meets it with a punch from his glowing gauntlet. 0:03–0:04: the crescent shatters into blue and amber fragments of light. 0:04–0:05: both hold their combat stances as the sparks fade between them. Hair and coat hems flutter in the wind. \u003cAUDCAP>A short urgent female warning, strong wind, a staff whoosh, one sharp magical impact, and fading spark crackles. Fabric snaps and tense breathing. No music.\u003cENDAUDCAP>", "video": "assets/videos/samples/3.mp4", "poster": "assets/posters/opening-fp4/3.jpg"}, {"id": "4", "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic. The camera holds a static medium close-up on a beautiful woman with long flowing brown hair and expressive green eyes, sitting alone in the corner of a dimly lit room. She wears a simple, elegant black dress that hugs her curves; her posture is slumped, her head bowed slightly, her shoulders drooping, and her eyes glisten under the low light with her mouth turned downward. Behind her, the room is sparsely furnished with a few pieces of vintage furniture, including a worn wooden dresser and an old upholstered armchair, while a single table lamp casts soft shadows across the walls. She draws a slow, trembling breath, then lowers her gaze toward the floor.\n\noverall_soundscape: The room is nearly silent, filled only with the woman's slow, shallow breathing and a soft sigh escaping her lips. Her black dress rustles faintly as her shoulders shift, and a vintage clock ticks quietly somewhere in the room.\n\nnon_diegetic_music: A sparse solo piano plays slow, widely spaced notes in a low register at a soft volume, with long pauses between phrases and no crescendo.", "video": "assets/videos/samples/4.mp4", "poster": "assets/posters/opening-fp4/4.jpg"}, {"id": "5", "prompt": "Cinematic live-action realism, wide horizontal 16:9 frame. A large wooden sailing ship moves slowly across a dark ocean beneath a densely star-filled sky. Warm lanterns hanging from the rigging illuminate weathered planks, ropes, and the lower edges of tall sails; moonlit reflections tremble on the water. One continuous wide shot: over five seconds, the camera moves smoothly through a small arc alongside the hull, keeping the entire ship in the composition. The hull gently rises and falls over one low swell, the sails fill with wind, and the lantern light sways slightly. The distant horizon remains calm and level. \u003cAUDCAP>Steady wind through the rigging, soft sail flaps, long wooden and rope creaks, and water lapping against the hull. A quiet sustained chord of strings and low brass with one gentle piano note underneath. No speech.\u003cENDAUDCAP>", "video": "assets/videos/samples/5.mp4", "poster": "assets/posters/opening-fp4/5.jpg"}, {"id": "6", "prompt": "Cinematic live-action realism, horizontal 16:9 frame. An evening exterior view through the closed window of a cozy study; the dark wooden window frame borders the composition. Inside, a warm desk lamp illuminates a writing desk with an open laptop, a notebook, pens, and loose sheets of paper. Behind it are book-filled shelves, framed photographs, and green plants. One continuous five-second shot with a very slow push toward the glass. A light breeze gently moves the reflection of a nearby branch on the window; the interior remains almost still. Warm highlights shift along the desk edge as the camera moves. No people are visible. \u003cAUDCAP>Quiet wind outside, a subdued laptop hum, and one barely audible wooden furniture creak. Soft, sparse piano notes over a low sustained string accompaniment. No speech.\u003cENDAUDCAP>", "video": "assets/videos/samples/6.mp4", "poster": "assets/posters/opening-fp4/6.jpg"}, {"id": "11", "prompt": "Cinematic fantasy realism, horizontal 16:9 composition. A majestic emerald dragon flies above a misty forest at sunset. Detailed scales, an amber eye, broad leathery wings, and a long winding tail are clearly visible. The sky is pink and orange near the horizon, shading into violet overhead; bands of sunlight illuminate the treetops. One continuous wide side view: for five seconds, the camera flies alongside the dragon, keeping its body and wings in frame. The dragon makes one powerful wingbeat, banks slightly toward the camera, then settles into a smooth glide. Ribbons of mist spread beneath its wings, and nearby treetops sway in the passing air current. \u003cAUDCAP>Continuous high-altitude wind, one deep leathery wingbeat, a fading rush of air, and soft rustling from disturbed treetops. Distant, subdued forest ambience. No speech or music.\u003cENDAUDCAP>", "video": "assets/videos/samples/11.mp4", "poster": "assets/posters/opening-fp4/11.jpg"}, {"id": "tokyo-neon", "prompt": "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.", "video": "assets/videos/samples/tokyo-neon.mp4", "poster": "assets/posters/opening-fp4/tokyo-neon.jpg"}, {"id": "woolly-mammoths", "prompt": "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.", "video": "assets/videos/samples/woolly-mammoths.mp4", "poster": "assets/posters/opening-fp4/woolly-mammoths.jpg"}, {"id": "91", "prompt": "A cinematic night landscape shot with a wide angle from the foot of a rocky cliff. The camera slowly glides upward from the dark ground and sparse grass at the base of the rock, following its sharp, jagged faces to the summit. There, against the deep night sky, the silhouette of a solitary pine comes into view, its branches swaying almost imperceptibly from side to side. The space above the cliff opens up, filling the upper half of the frame with a majestic dome of stars and the broad silver band of the Milky Way. The thin crescent of a waxing moon in the upper-left corner casts a cold, soft light that traces the rock's contours and creates long, sharp shadows from the pine branches. Below, right on the horizon, an unmoving, dense veil of mist conceals the bases of distant mountains. At the third second, a bright greenish flash streaks across the sky from the upper-right corner. A meteor travels down diagonally toward the center of the frame. Its short, bright trail briefly illuminates the edges of the banks of fog and the summit of the cliff, then dissolves into absolute darkness. The palette uses deep indigo and violet, cold silvery highlights on the rocks, and a single contrasting burst of acid green from the meteor. The camera's smooth ascent gives a feeling of slowly floating and of awe at the scale of nature. \u003cAUDCAP>The muted rustle of pine needles in the wind, an occasional quiet cricket chirp, and a barely audible low hum of night wind flowing around the rocks.\u003cENDAUDCAP>", "video": "assets/videos/comparison/Quant_Distill/91.mp4", "poster": "assets/posters/comparison/quant/91.jpg"}, {"id": "96", "prompt": "A cinematic shot captures the relentless movement of volcanic lava. The frame tightens around a thick, red-orange flow as it slowly descends the slope and winds around sharp black formations of solidified rock. The camera glides from right to left, following the current. The pressure of the flow continually forms a dark, cooling crust that cracks open, exposing pulses of incandescent magma in the fissures. Thick gray smoke and heat haze rise above the lava. A predominantly orange glow shines through, revealing textures in the cooled rock. A shiny strand of lava runs through the foreground, with a vast, uneven field receding behind it. During the five-second shot, the foreground crust splits with a dull crack and a brighter mass bursts out, reshaping the surface. Light from the molten rock pulses with its movement, casting deep, shifting shadows and setting the glowing lava against the cold, dark volcanic rock. \u003cAUDCAP>A low, muffled rumble, hissing hot gases, and the crackle of the cooling lava crust.\u003cENDAUDCAP>", "video": "assets/videos/comparison/Quant_Distill/96.mp4", "poster": "assets/posters/comparison/quant/96.jpg"}]</script>
<!-- Each comparison joins Base (videos in the RL folders), Distill and NVFP4 QAD (videos in the Quant_Distill folders) by the Russian prompt CSV ID. -->
<script type="application/json" data-fk6="comparisons">[{"id": "15", "prompt": "The shot opens in extreme close-up, filling the frame with a viscous, glossy, pearlescent texture. Deep, cool blues grade into shimmering silver, like liquid metal. The dense substance flows slowly and evenly downward over an invisible vertical plane. Soft directional light glides across it, letting lustrous highlights spill smoothly from one contour and droplet to the next. Tiny motions reveal the slow stretching and breaking of translucent threads. Focus holds in the foreground as the edges blur into a velvety bokeh, immersing the viewer in this alien substance. At the end of the fifth second, the flow maintains its speed, leaving a perfectly symmetrical, slowly descending drop in the center of the frame. The camera remains completely still throughout. Sharp, crystalline light creates a dramatic contrast between shining reflections and deep shadows. The grade is cinematic blue and silver, with a glow on the brightest surfaces. The material stretches elastically and flows under gravity, while silver particles drift through the blue mass and give it depth. \u003cAUDCAP>A muted, wet squelch as the viscous material slowly moves, accompanied by a low-frequency hum.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/15.mp4", "poster": "assets/posters/comparison/rl/15.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/15.mp4", "poster": "assets/posters/comparison/distill/15.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/15.mp4", "poster": "assets/posters/comparison/quant/15.jpg"}]}, {"id": "18", "prompt": "A cinematic urban landscape in the rays of the setting sun. The movement begins with an extreme close-up of the texture of a classic-cut dark gray wool coat, with each thread in the weave visible. A man of Slavic appearance, around 45, with a trace of weariness in his features, slowly lowers himself onto a wooden slatted bench supported by cast-iron legs. The space gradually opens up. The camera smoothly zooms out to reveal a gray cylindrical concrete litter bin, followed by the austere silhouette of a black wrought-iron lamppost at the edge of the path. The view glides farther to take in a concrete staircase with cold, gleaming metal railings. By the end of the fifth second, the movement resolves into a wide establishing shot. The whole square, with all its avenues and paths, fits in the frame. The man sits motionless, shoulders slightly dropped, breathing steadily. Soft, warm sunset light floods the scene, lengthening tree shadows and leaving a golden glow on the concrete. A light evening haze hangs in the air and diffuses the rays. 0:00–0:02 (Setup): The view holds on the texture of the coat's wool and the moment the man touches the wooden slat of the bench. The movement begins with a slow, fluid pullback. 0:02–0:04 (Development): The space expands, bringing the bin and lamppost into frame in succession. The man settles into a relaxed pose and his coat falls slightly open. 0:04–0:05 (Resolution): The zoom stops on the wide view of the square. The man holds still in profile, looking toward the setting sun as its light turns a richer golden orange. Emphasize the realism of the materials. The heavy wool drapes smoothly around the body without creasing, the old cast iron of the bench has a porous texture, and the concrete bin is covered in fine cracks. Micro-movements are minimal, with only a barely perceptible ripple in the coat's nap from the evening breeze and an occasional natural blink. The light shifts from bright gold to a deeper copper tone as the sun sinks, coloring the treetops in fiery shades. The lighting combines backlight with warm fill. The setting sun is behind the man, tracing a thin luminous halo around him and the bench. The palette combines deep graphite in the coat, warm ocher light, and cool gray concrete. The atmosphere recalls cinema shot on film, with soft gradients and no digital clipping. \u003cAUDCAP>A distant, muted, low city hum, the quiet rustle of leaves in the trees, and occasional sparrow chirps.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/18.mp4", "poster": "assets/posters/comparison/rl/18.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/18.mp4", "poster": "assets/posters/comparison/distill/18.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/18.mp4", "poster": "assets/posters/comparison/quant/18.jpg"}]}, {"id": "24", "prompt": "The shot opens on a head-on view and slowly, smoothly pushes inward, compressing the space between the lens and the facade. Against a clean, uniform gray sky, the monumental rectangular form of a modern office building dominates the frame. Its dark, glossy glass panels reflect the cold tones of the sky, creating an almost mirrorlike effect that still seems to absorb light. An external glass elevator car glides silently and smoothly upward along a vertical line, cutting across the plane of glass. The elevator's movement is perfectly synchronized with the smooth push inward, as the focus gradually brings out the texture of the glass and its narrow metal frames against the sky dissolving into haze. The composition is strictly symmetrical, emphasizing the architecture's geometry and scale. Even, shadowless lighting typical of an overcast day bathes the scene in cold, diffuse light with no sharp contrasts. The palette is monochromatic, from lead gray at the horizon to deep anthracite in the building's glass. At the end of the movement, the focus settles on the reflection of a drifting cloud in the dark, glossy panel beside the elevator. The frame holds only the architecture and the mechanical motion of the elevator car. The soundscape is a dense low-frequency metropolitan hum, a mix of engines, tires brushing over asphalt, and the faint, blurred echo of many voices drifting up from lower levels and from outside the frame. \u003cAUDCAP>A steady, low-frequency hum of city traffic, the swish of tires on asphalt, and the distant, indistinct murmur of many voices.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/24.mp4", "poster": "assets/posters/comparison/rl/24.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/24.mp4", "poster": "assets/posters/comparison/distill/24.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/24.mp4", "poster": "assets/posters/comparison/quant/24.jpg"}]}, {"id": "32", "prompt": "Soft daylight floods a minimalist composition on a light-colored table. A white ceramic plate sits at the center of the frame with a neat arrangement of snacks: bright cubes of yellow cheese, round golden crackers, and large, glossy green olives. A woman's hand with neatly groomed fingers glides into frame from the right. She reaches slowly and delicately for a golden cracker, her fingertips barely grazing its dry, porous surface. Her hand pauses for a moment before lifting it. The depth of field keeps the cheese's texture and the olives' sheen visible, while the background softly dissolves into warm bokeh. The composition is static, with the hand providing the only movement and creating a sense of comfort and calm. In the slightly out-of-focus background, the silhouettes of tall crystal glasses catch the light. \u003cAUDCAP>The quiet rustle of fingers on the dry surface of the cracker, the faintest touch against the plate, and the distant, melodic clink of crystal glasses.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/32.mp4", "poster": "assets/posters/comparison/rl/32.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/32.mp4", "poster": "assets/posters/comparison/distill/32.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/32.mp4", "poster": "assets/posters/comparison/quant/32.jpg"}]}, {"id": "33", "prompt": "The look of a Marvel superhero blockbuster, with dramatic high-contrast lighting and saturated red and gold accents. The shot begins with a wide view of a kitchen alive with movement. In one quick, practiced gesture, a chef tosses a massive T-bone steak in a scorching cast-iron skillet, sending droplets of hot oil flying through the sidelight. The chef confidently flips the meat with a heavy metal spatula, producing a sharp, high scrape of metal against cast iron. The camera keeps gliding sideways to a second pan, where bright green asparagus, glossy scarlet cherry tomato halves, and juicy slices of yellow bell pepper meet with a fierce hiss. The framing gradually tightens on the vegetables. The camera glides toward the pan through clouds of steam, picking out moist, oily highlights on each piece. The pace abruptly shifts into cinematic slow motion. A side-on close-up shows the perfectly cooked steak, with its crosshatched grill marks, landing heavily on a substantial dark plate with a dull, wet sound. The bright, steaming vegetable medley is plated beside it with surgical precision. In the final moment, all motion comes to a stop, revealing the velvety background bokeh as the kitchen's golden lights dissolve into soft focus. In close-up, the chef's skin and the texture of the meat show complex subsurface light scattering, while steam slowly rises from the food through backlighting and changes density from one moment to the next. \u003cAUDCAP>Loud, aggressive sizzling of hot oil in two pans and the sharp metallic scrape of a spatula against cast iron, giving way to the dull, wet sound of food landing on the plate.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/33.mp4", "poster": "assets/posters/comparison/rl/33.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/33.mp4", "poster": "assets/posters/comparison/distill/33.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/33.mp4", "poster": "assets/posters/comparison/quant/33.jpg"}]}, {"id": "37", "prompt": "A table-level close-up glides across a light linen tablecloth. In the right foreground is a white ceramic plate of pasta carbonara generously sprinkled with fresh herbs. Closer to the center, a silver-colored knife and fork catch a warm glint in their matte metal. To the left stands a tall, slender wineglass into which a waiter smoothly pours white wine from a clear glass carafe. The golden stream curves as it fills the glass, while the waiter's hand moves with precision and control, a dark fabric apron visible behind it. The silhouettes of other tables and the warm restaurant interior dissolve into thick, velvety bokeh. Directional light from a low pendant lamp produces soft, dimensional lighting, casting deep but gentle side shadows across the linen and picking out the fine texture of the Parmesan. The palette combines creamy linen, the white of the plate, deep brown sauce, the cool silver gleam of the cutlery, and pale golden wine with touches of green basil. The focus moves forward slowly, almost imperceptibly. It first holds on the suspended stream of wine, then shifts smoothly to bubbles rising from the bottom of the glass, and finally fixes on the texture of the pasta. It holds there for a moment as the carafe comes softly to rest on the tablecloth, a millimeter from the edge. \u003cAUDCAP>The quiet clink of porcelain touching silverware, the soft splash of wine being poured, and the subdued murmur of conversation and tableware in the restaurant.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/37.mp4", "poster": "assets/posters/comparison/rl/37.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/37.mp4", "poster": "assets/posters/comparison/distill/37.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/37.mp4", "poster": "assets/posters/comparison/quant/37.jpg"}]}, {"id": "38", "prompt": "A cinematic crime-thriller aesthetic with a near-obsessive attention to textures. The camera glides smoothly and hypnotically from a medium shot to an extreme close-up, synchronized with the movement of a hand. 0:00–0:02 (Setup): In a dark, stylish kitchen with subdued light and deep shadows, a medium shot shows a chef's dark-sleeved arm plunging a heavy metal spoon into a glass jar filled to the brim with bright yellow turmeric. The movement is swift and assured. The fingers wrap firmly around the cold metal, and the spoon disappears into the powder. 0:02–0:04 (Development): The hand emerges from the jar with a scoop of spice and carries it over a scorching pan of sizzling pieces of chicken breast. As the spoon hovers over the rim, a sudden, percussive cut tightens the framing to an extreme close-up. 0:04–0:05 (Resolution): A thin, continuous stream of golden turmeric falls into the hot oil. The spice instantly dissolves, erupting into thousands of amber and graphite-colored streaks across the meat. The focus picks out the detailed texture of the chicken, the glossy sheen of the oil, and the tiny droplets scattered by the impact. Hard, high-contrast sidelight picks out only this action, leaving the surroundings in velvety shadow. The palette is deep graphite black, dark brown, and metallic steel, with the warm, almost luminous amber yellow of the spice as its sole bright accent. Pores and a slight sheen are visible on the skin of the hand, and the metal spoon reflects a distorted image of the chef's face. Visual style: crime-thriller aesthetic, high detail, macro, high-contrast lighting, and color isolation. The hand moves with mechanical precision and no fuss. In close-up, tiny particles of turmeric hang in the air for a fraction of a second before touching the oil. The oil bubbles and churns, scorching the falling golden dust. Hard, directional light creates long, sharp shadows. The color grade has deep contrast and subdued shadows, emulating film with pronounced grain in the dark areas. The turmeric's warm yellow remains the only visually warm element in the cold, somber surroundings. \u003cAUDCAP>The intense, aggressive hiss of hot oil, with a sharp, brief whistling whoosh at the cut.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/38.mp4", "poster": "assets/posters/comparison/rl/38.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/38.mp4", "poster": "assets/posters/comparison/distill/38.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/38.mp4", "poster": "assets/posters/comparison/quant/38.jpg"}]}, {"id": "45", "prompt": "The shot opens on a deep, velvety black star field dotted with the faint points of distant stars. The nucleus of a large comet emerges from the lower-left corner, cutting through the darkness of space. It travels in a straight line from the lower left to the upper right, leaving a blurred trail behind it. The camera smoothly tracks its movement, creating a floating, weightless view. Comet Hale–Bopp appears in all its splendor, its bright, white-hot center contrasting with the surrounding void. A luminous blue coma pulses around the nucleus, a cloud of gas illuminated by the offscreen Sun. Its rays cast hard, directional light that emphasizes the volume of the icy body. A long, forked tail stretches behind it in two distinct streams, one dense with dust and the other made of thinner gas. The tail points directly away from its trajectory, toward the light source, and gradually dissolves into darkness. At 0:02, the comet crosses the center of the frame, its tail filling the upper-right portion of the screen. By 0:04, it begins disappearing beyond the upper-right edge, leaving only a slowly fading trace of light against the motionless constellations. At 0:05, the screen is once again immersed in the primordial silence of space, with the barely perceptible flicker of the stars as the only remaining movement. The comet's nucleus has a complex, pitted surface and rotates almost imperceptibly. Dust particles drift slowly in the tail, stretching out for millions of kilometers and creating a sense of depth and scale. The lighting remains cold and constant, with a bright highlight on the icy nucleus and the soft, diffuse glow of the coma. The palette is limited to deep indigo and black in space, the nucleus's brilliant white light, and the neon-blue glow of the surrounding envelope, evoking absolute cold and vacuum. \u003cAUDCAP>A quiet, low, mysterious cinematic hum that underscores the silence of space.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/45.mp4", "poster": "assets/posters/comparison/rl/45.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/45.mp4", "poster": "assets/posters/comparison/distill/45.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/45.mp4", "poster": "assets/posters/comparison/quant/45.jpg"}]}, {"id": "87", "prompt": "A cinematic shot from extremely close range shows the edge of a rocky ledge thickly covered in fresh snow. Under the effects of gravity and tiny cracks, the mass of snow loses its grip and breaks free, falling uncontrollably off the ledge. When it strikes the slope below, a cloud of white powder bursts in all directions and fills the lower part of the frame. The camera glides downward from the point of separation, tracking the fall. A sharp cut opens instantly onto a majestic view of the same mountain range from a respectful distance. An enormous mass of snow races down the steep slope, gathering speed and turning into a giant avalanche. A colossal cloud of sunlit, sparkling snow powder rises behind it, following the flow, expanding, and swallowing the lower part of the mountain. Bright sun behind the summit creates powerful backlight, piercing the suspended snow and making millions of crystals shine while casting long, sharp shadows across the contours of the slope. The camera floats fluidly downward with the avalanche, keeping it at the center of the composition until the powder cloud begins to fill the lower third of the frame. In the first two seconds, the view fixes on the microtexture of the snow crust and the lone cap of snow on the ledge. At the breakaway, the focus follows the falling mass and the framing tightens around the cloud of powder. By the fourth second, the action peaks. The avalanche fills two-thirds of the frame and the sun throws dazzling highlights on airborne particles. The emphasis is on the physics of frozen water. Heavy, dense snow moves like a viscous liquid, while the lighter powder responds to the slightest eddies in the air. The lighting relies on sharp contrast, with deep, cold shadows on shaded parts of the slope and blindingly bright, warm outlines on the backlit flying snow. The palette combines the piercing azure of the sky, brilliant white snow with blue reflections, and a golden halo around the airborne powder. A low, vibrating rumble builds with the movement, filling the space. \u003cAUDCAP>A low, deafening roar of moving snow that steadily builds, with the dull rumble of shifting masses of ice.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/87.mp4", "poster": "assets/posters/comparison/rl/87.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/87.mp4", "poster": "assets/posters/comparison/distill/87.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/87.mp4", "poster": "assets/posters/comparison/quant/87.jpg"}]}, {"id": "91", "prompt": "A cinematic night landscape shot with a wide angle from the foot of a rocky cliff. The camera slowly glides upward from the dark ground and sparse grass at the base of the rock, following its sharp, jagged faces to the summit. There, against the deep night sky, the silhouette of a solitary pine comes into view, its branches swaying almost imperceptibly from side to side. The space above the cliff opens up, filling the upper half of the frame with a majestic dome of stars and the broad silver band of the Milky Way. The thin crescent of a waxing moon in the upper-left corner casts a cold, soft light that traces the rock's contours and creates long, sharp shadows from the pine branches. Below, right on the horizon, an unmoving, dense veil of mist conceals the bases of distant mountains. At the third second, a bright greenish flash streaks across the sky from the upper-right corner. A meteor travels down diagonally toward the center of the frame. Its short, bright trail briefly illuminates the edges of the banks of fog and the summit of the cliff, then dissolves into absolute darkness. The palette uses deep indigo and violet, cold silvery highlights on the rocks, and a single contrasting burst of acid green from the meteor. The camera's smooth ascent gives a feeling of slowly floating and of awe at the scale of nature. \u003cAUDCAP>The muted rustle of pine needles in the wind, an occasional quiet cricket chirp, and a barely audible low hum of night wind flowing around the rocks.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/91.mp4", "poster": "assets/posters/comparison/rl/91.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/91.mp4", "poster": "assets/posters/comparison/distill/91.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/91.mp4", "poster": "assets/posters/comparison/quant/91.jpg"}]}, {"id": "96", "prompt": "A cinematic shot captures the relentless movement of volcanic lava. The frame tightens around a thick, red-orange flow as it slowly descends the slope and winds around sharp black formations of solidified rock. The camera glides from right to left, following the current. The pressure of the flow continually forms a dark, cooling crust that cracks open, exposing pulses of incandescent magma in the fissures. Thick gray smoke and heat haze rise above the lava. A predominantly orange glow shines through, revealing textures in the cooled rock. A shiny strand of lava runs through the foreground, with a vast, uneven field receding behind it. During the five-second shot, the foreground crust splits with a dull crack and a brighter mass bursts out, reshaping the surface. Light from the molten rock pulses with its movement, casting deep, shifting shadows and setting the glowing lava against the cold, dark volcanic rock. \u003cAUDCAP>A low, muffled rumble, hissing hot gases, and the crackle of the cooling lava crust.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/96.mp4", "poster": "assets/posters/comparison/rl/96.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/96.mp4", "poster": "assets/posters/comparison/distill/96.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/96.mp4", "poster": "assets/posters/comparison/quant/96.jpg"}]}, {"id": "98", "prompt": "A kinetic, soaring bird's-eye view glides above a boundless field. Gusts of wind make the golden-copper ears of wheat tremble continuously, creating a living, shimmering texture. Below, a bright-red combine harvester with a massive metal body and deep-tread tires rolls along a dusty country road, leaving a trail of dry earth. The framing gradually tightens and the focus shifts to the machine as it reaches the edge of the field, lines up with the perfectly straight crop rows, and begins harvesting. The camera follows closely, sweeping around the machine in a smooth arc. The heavy header's metal fingers catch the stalks, which give slightly before being cut evenly at the base and disappearing into the intake. 0:00–0:02: The shot opens wide on an endless sea of wheat and a slowly moving red speck. 0:02–0:04: The main movement is a dynamic sweep around the combine, which settles at the center of the composition as the header starts its destructive but rhythmic passage through the rows. 0:04–0:05: A close-up shows the header's sharp blades at work and the severed ears disappearing inside. The camera then lifts slightly to reveal the even trail of cut stubble behind the machine. The combine moves heavily, as if straining. Its body vibrates faintly with the low-frequency resonance of the powerful engine, and the massive tires compress the soil. Metal parts reflect the shifting light, while dust settles into the tire treads. Emphasize continuous micro-movements: the turning reel on the header, grains spilling downward, and the slight sway of the cab. Bright midday lighting casts hard but short shadows from the header and cab. Golden highlights slide across the painted metal as the machine changes course. The warm, saturated autumn color grade highlights the scarlet body and amber wheat, with a light layer of film grain for a tactile finish. The wind blows in the same direction as the combine, pressing uncut wheat toward the ground and leaving a strip of flattened, darkened stubble behind the machine. \u003cAUDCAP>A low, rhythmic hum from the combine's running diesel engine at medium volume, blending with the steady rustle of golden-copper ears of wheat in the wind.\u003cENDAUDCAP>", "clips": [{"label": "Base", "video": "assets/videos/comparison/RL/98.mp4", "poster": "assets/posters/comparison/rl/98.jpg"}, {"label": "Distill", "video": "assets/videos/comparison/Distill/98.mp4", "poster": "assets/posters/comparison/distill/98.jpg"}, {"label": "NVFP4 QAD", "video": "assets/videos/comparison/Quant_Distill/98.mp4", "poster": "assets/posters/comparison/quant/98.jpg"}]}]</script>
<script type="application/json" data-fk6="metrics">{"gpu": "NVIDIA GeForce RTX 5090", "frames": 121, "fps": 24, "seed": 575431, "bf16": {"steps": 10, "forwards": 10, "guidance_scale": 1.0, "dit": 186.9609965120908, "per_forward": 18.696099651209078, "total": 205.39850421459414, "components": {"transformer": 186.9609965120908, "vae": 17.091872333432548, "other": 1.3456353690708056}, "timing_scope": "generation", "measured_runs": 2, "peak_memory": {"ram": 65.46033096313477, "vram": 26.284517288208008}, "width": 768, "height": 512, "offload_dit": true, "resident": 16, "lazy": false, "attention": "FlashAttention-2", "linear": "bf16", "compile_dit": true, "backends": {"clip": "FlashAttention-2", "qwen": "flash_attention_2"}}, "fp4": {"steps": 8, "forwards": 8, "guidance_scale": 1.0, "dit": 42.85914997605141, "per_forward": 5.357393747006427, "total": 61.31029370194301, "components": {"transformer": 42.85914997605141, "vae": 17.100271246396005, "other": 1.3508724794955924}, "timing_scope": "generation", "measured_runs": 2, "peak_memory": {"ram": 44.046836853027344, "vram": 16.0275821685791}, "width": 768, "height": 512, "offload_dit": true, "resident": 16, "lazy": false, "attention": "ATTN_QAT_INFER/cutlass_sm12x", "linear": "nvfp4", "compile_dit": true, "backends": {"clip": "FlashAttention-2", "qwen": "flash_attention_2"}}, "source": "k6-distill-compiled-1w3-001/results.json", "torch": "2.12.0+cu130", "cuda": "13.0", "timing_scope": "generation", "pro": {"steps": 50, "forwards": 100, "guidance_scale": 5.0, "dit": 1869.6099651209079, "per_forward": 18.696099651209078, "total": 1888.0474728234112, "width": 768, "height": 512, "attention": "FlashAttention-2", "peak_memory": {"ram": 65.46033096313477, "vram": 26.284517288208008}, "timing_source": "extrapolated_from_distill_bf16"}, "speedup": 3.4897751657055216, "generation_speedup": 3.3501471255892024}</script>
<script>
(() => {
  const root = document.currentScript.closest('.fk6-article');
  const slot = name => root.querySelector(`[data-fk6-slot="${name}"]`);
  const readData = name => JSON.parse(root.querySelector(`script[data-fk6="${name}"]`).textContent);
  const format = value => value == null ? '—' : value.toFixed(2);
  const base = root.dataset.mediaBase || '';
  const media = path => !path || /^(?:[a-z]+:)?\/\//i.test(path) || path.startsWith('/') ? path : base + path;
  const metrics = readData('metrics');
  function appendText(parent, tag, text, className) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    parent.append(node);
    return node;
  }
  function addVideo(parent, path, label, poster) {
    if (!path) {
      appendText(parent, 'div', 'Clip pending', 'fk6-missing');
      return;
    }
    const video = document.createElement('video');
    video.className = 'fk6-video';
    video.src = media(path);
    video.controls = true;
    video.playsInline = true;
    video.preload = poster ? 'none' : 'metadata';
    if (poster) video.poster = media(poster);
    video.setAttribute('aria-label', label);
    parent.append(video);
  }
  function addPrompt(parent, text) {
    const details = appendText(parent, 'details', '', 'fk6-prompt');
    const summary = appendText(details, 'summary', '');
    appendText(summary, 'span', 'Prompt', 'fk6-prompt-label');
    const preview = text.length > 145 ? text.slice(0, 145).replace(/\s+\S*$/, '') + '…' : text;
    appendText(summary, 'span', preview, 'fk6-prompt-preview');
    appendText(details, 'p', text);
  }
  // Opening clip, samples and side-by-side comparisons.
  const intro = readData('intro');
  addVideo(slot('intro'), intro.video, 'Opening generation', intro.poster);
  for (const sample of readData('gallery')) {
    const figure = appendText(slot('gallery'), 'figure', '');
    addVideo(figure, sample.video, `NVFP4 sample ${sample.id}`, sample.poster);
    addPrompt(appendText(figure, 'figcaption', ''), sample.prompt);
  }
  const comparisons = readData('comparisons');
  slot('compare-count').textContent = `${comparisons.length} prompts`;
  for (const comparison of comparisons) {
    const figure = appendText(slot('compare'), 'figure', '', 'fk6-compare-item');
    const columns = appendText(figure, 'div', '', 'fk6-compare-grid');
    for (const clip of comparison.clips) {
      const column = appendText(columns, 'div', '', clip.label === 'NVFP4 QAD' ? 'ours' : '');
      appendText(column, 'h3', clip.label);
      addVideo(column, clip.video, `${clip.label} generation ${comparison.id}`, clip.poster);
    }
    addPrompt(appendText(figure, 'figcaption', ''), comparison.prompt);
  }
  // Headline numbers.
  const {pro, bf16, fp4} = metrics;
  const warmTiming = metrics.timing_scope === 'generation';
  const gpu = metrics.gpu.replace(/^NVIDIA GeForce /, '');
  const setupText = `${gpu} · ${fp4.width} × ${fp4.height} · ${metrics.frames} frames · ${metrics.fps} fps`;
  const speedup = pro.total / fp4.total;
  const vramSaving = pro.peak_memory?.vram && fp4.peak_memory?.vram ? 1 - fp4.peak_memory.vram / pro.peak_memory.vram : null;
  // The tagline sits in the page header, outside this block; in a blog post the post description takes its place.
  const tagline = document.getElementById('fk6-tagline');
  if (tagline && vramSaving != null) {
    tagline.replaceChildren(`${Math.floor(speedup)}× faster. `);
    appendText(tagline, 'span', `${Math.floor(vramSaving * 100)}% lighter.`);
  }
  slot('lead').textContent = `FastKandinsky 6 QAD generates a ${metrics.frames}-frame ${fp4.width} × ${fp4.height} video with audio `
    + `in ${format(fp4.total)} s ${warmTiming ? 'end to end' : 'of component time'} on ${gpu}: `
    + `${format(speedup)}× faster than the Kandinsky 6 Pro base model in BF16`
    + (vramSaving != null ? `, with ${Math.floor(vramSaving * 100)}% less peak VRAM.` : '.');
  // Inference speed stats and table.
  slot('stat-sft').textContent = format(speedup) + '×';
  slot('stat-distill').textContent = format(metrics.generation_speedup) + '×';
  slot('stat-dit').textContent = format(metrics.speedup) + '×';
  if (!warmTiming) {
    slot('stat-sft-label').textContent = 'NVFP4 component time vs. Pro Base BF16';
    slot('stat-distill-label').textContent = 'NVFP4 component time vs. Distill BF16';
    slot('total-heading').textContent = 'Components (s)';
  }
  slot('setup').textContent = setupText;
  const setupNotes = [];
  if (bf16.offload_dit && fp4.offload_dit) setupNotes.push('DiT offload');
  if (bf16.compile_dit && fp4.compile_dit) setupNotes.push('DiT compiled');
  slot('setup-note').textContent = setupNotes.join(' · ');
  slot('setup-note').hidden = !setupNotes.length;
  const models = [
    [pro, 'Pro · Base BF16', 'Base BF16', pro.attention, false],
    [bf16, 'Pro · Distill BF16', 'Distill BF16', bf16.attention, false],
    [fp4, 'Pro · NVFP4 QAD', 'NVFP4 QAD', 'Native FP4 attention · FlashInfer', true]
  ].map(([config, label, short, attention, isOurs]) => {
    const callsPerStep = config.forwards / config.steps;
    const steps = callsPerStep === 1 ? `${config.steps} steps` : `${config.steps} × ${callsPerStep} · CFG ${config.guidance_scale}`;
    return {config, label, short, details: `${steps} · ${attention}`, isOurs};
  });
  for (const {config, label, details, isOurs} of models) {
    const row = slot('metrics').insertRow();
    row.classList.toggle('ours', isOurs);
    const model = appendText(row, 'th', label);
    model.scope = 'row';
    appendText(model, 'span', details, 'fk6-model-details');
    for (const value of [config.total, config.per_forward, config.peak_memory?.vram]) {
      appendText(row, 'td', format(value), isOurs ? 'win' : '');
    }
  }
  // Figure 2: end-to-end time as columns, with a vertical arrow for the speedup.
  const panel = slot('speed');
  const tip = appendText(panel, 'div', '', 'fk6-tip');
  tip.hidden = true;
  tip.setAttribute('aria-hidden', 'true');
  const hideTip = () => { tip.hidden = true; };
  function showTip(col, model, value, unit, oursValue) {
    tip.replaceChildren();
    appendText(tip, 'strong', `${format(value)} ${unit}`);
    appendText(tip, 'div', model.label, 'fk6-tip-model');
    appendText(tip, 'div', model.details);
    if (!model.isOurs && value != null && oursValue) appendText(tip, 'div', `${format(value / oursValue)}× vs. NVFP4`);
    tip.hidden = false;
    const box = panel.getBoundingClientRect();
    const bar = col.querySelector('.fk6-bar').getBoundingClientRect();
    const half = tip.offsetWidth / 2;
    const x = bar.left + bar.width / 2 - box.left;
    tip.style.left = Math.min(Math.max(x, half), box.width - half) + 'px';
    tip.style.top = (bar.top - box.top - 22) + 'px';
  }
  const unit = 's';
  const chartTitle = warmTiming ? 'End-to-end time' : 'Component time';
  const values = models.map(model => model.config.total);
  const max = Math.max(...values.filter(value => value != null));
  const oursIndex = models.findIndex(model => model.isOurs);
  const oursValue = values[oursIndex];
  const speedHead = appendText(panel, 'div', '', 'fk6-panel-head');
  const speedTitle = appendText(speedHead, 'div', `${chartTitle} · Kandinsky 6 Pro`, 'fk6-panel-title');
  appendText(speedTitle, 'span', `${setupText} · seconds, lower is better`, 'fk6-panel-sub');
  const plot = appendText(panel, 'div', '', 'fk6-plot');
  const xlabels = appendText(panel, 'div', '', 'fk6-xlabels');
  xlabels.setAttribute('aria-hidden', 'true');
  const shares = values.map(value => value == null || !(max > 0) ? 0 : value / max * 100);
  models.forEach((model, i) => {
    const value = values[i];
    const col = appendText(plot, 'div', '', model.isOurs ? 'fk6-col ours' : 'fk6-col');
    col.tabIndex = 0;
    col.setAttribute('role', 'img');
    col.setAttribute('aria-label', `${model.label}: ${format(value)} ${unit}`);
    const bar = appendText(col, 'span', '', 'fk6-bar');
    bar.style.height = shares[i] + '%';
    bar.hidden = value == null;
    const valueLabel = appendText(col, 'span', format(value), 'fk6-value');
    valueLabel.style.bottom = `calc(${shares[i]}% + 6px)`;
    valueLabel.setAttribute('aria-hidden', 'true');
    appendText(xlabels, 'span', model.short, model.isOurs ? 'ours' : '');
    const show = () => showTip(col, model, value, unit, oursValue);
    col.addEventListener('pointerenter', show);
    col.addEventListener('focus', show);
    col.addEventListener('pointerleave', hideTip);
    col.addEventListener('blur', hideTip);
  });
  const slowIndex = values.indexOf(max);
  let speedCaption = `Figure 2. ${chartTitle} on ${gpu}, in seconds.`;
  if (oursValue > 0 && slowIndex < oursIndex) {
    const ratio = max / oursValue;
    const center = i => (i + .5) / models.length * 100;
    const arrowOffset = 30;
    const refRight = `calc(${100 - center(oursIndex)}% - ${arrowOffset}px)`;
    for (const [from, share] of [[slowIndex, shares[slowIndex]], [oursIndex, shares[oursIndex]]]) {
      const ref = appendText(plot, 'span', '', 'fk6-ref');
      ref.setAttribute('aria-hidden', 'true');
      ref.style.left = `calc(${center(from)}% + 12px)`;
      ref.style.right = refRight;
      ref.style.bottom = share + '%';
    }
    const arrow = appendText(plot, 'div', '', 'fk6-arrow');
    arrow.setAttribute('aria-hidden', 'true');
    arrow.style.left = `calc(${center(oursIndex)}% + ${arrowOffset}px)`;
    arrow.style.top = `${100 - shares[slowIndex]}%`;
    arrow.style.bottom = shares[oursIndex] + '%';
    appendText(arrow, 'span', '', 'fk6-arrow-line');
    appendText(arrow, 'span', `${Math.floor(ratio)}×`, 'fk6-arrow-label');
    appendText(arrow, 'span', '', 'fk6-arrow-line');
    speedCaption += ` ${models[oursIndex].label} is ${format(ratio)}× faster than ${models[slowIndex].label}.`;
    slot('speed-figure').setAttribute('aria-label',
      `${chartTitle}: ${models[oursIndex].label} is ${format(ratio)}× faster than ${models[slowIndex].label}`);
  }
  slot('speed-caption').textContent = speedCaption;
  // Figure 1: where one NVFP4 generation spends its time.
  const {transformer, vae, other, text_encoder, text_encoder_2, audio_vae, vocoder} = fp4.components;
  const stages = [
    ['transformer', warmTiming ? 'DiT denoising' : 'DiT', transformer],
    ['vae', warmTiming ? 'Video decode' : 'Video VAE', vae],
    ['other', 'Other', other ?? (text_encoder + text_encoder_2 + audio_vae + vocoder),
      warmTiming ? 'Qwen, CLIP, audio decode, request overhead' : 'Qwen, CLIP, Audio VAE, Vocoder'],
  ];
  const stageSeconds = seconds => seconds < 1 ? seconds.toFixed(3) : format(seconds);
  const stagesHead = slot('stages-head');
  appendText(stagesHead, 'div', `NVFP4 · ${fp4.forwards} DiT forwards`, 'fk6-panel-title');
  appendText(stagesHead, 'div', `${format(fp4.total)} s`, 'fk6-panel-total');
  const stageBar = slot('stages-bar');
  stageBar.style.gridTemplateRows = stages.map(([, , seconds]) => `minmax(0, ${seconds}fr)`).join(' ');
  const segmentLabels = {transformer: 'DiT', vae: 'Decode'};
  for (const [key, label, seconds] of stages) {
    const segment = appendText(stageBar, 'span', segmentLabels[key] ?? '', `fk6-segment stage-${key}`);
    segment.title = `${label} · ${stageSeconds(seconds)} s`;
  }
  for (const [key, label, seconds, detail] of stages) {
    const row = slot('stages-table').insertRow();
    const heading = appendText(row, 'th', '');
    heading.scope = 'row';
    const rail = appendText(heading, 'span', '', `fk6-component stage-${key}`);
    appendText(rail, 'span', '', 'fk6-leader').setAttribute('aria-hidden', 'true');
    appendText(rail, 'span', '', 'fk6-swatch').setAttribute('aria-hidden', 'true');
    const name = appendText(rail, 'span', label);
    if (detail) appendText(name, 'small', detail, 'fk6-component-detail');
    appendText(row, 'td', stageSeconds(seconds));
  }
  slot('stages-caption').textContent =
    `Figure 1. Where one NVFP4 generation spends its ${format(fp4.total)} s, by FastVideo component. Weight loading excluded.`;
})();
</script>
</div>
