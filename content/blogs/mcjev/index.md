+++
title = "MCJev: A 24 ms Jev for Minecraft PvP on NVIDIA Blackwell GPUs"
date = 2026-10-05T00:00:00-07:00
authors = ["Minshen (Alex) Zhang", "Junda Chen", "Yuanbo Yang", "Yuxuan Zhang", "Yi Sun", "Shaoxiong Duan", "Lanxiang Hu", "Jiaqi Leng", "Yulun Wu", "Will Lin", "Hao Zhang"]
author = "Minshen (Alex) Zhang, Junda Chen, Yuanbo Yang, Yuxuan Zhang, Yi Sun, Shaoxiong Duan, Lanxiang Hu, Jiaqi Leng, Yulun Wu, Will Lin, Hao Zhang"
ShowReadingTime = true
authorOnNewLine = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://x.com/haoailab/status/2104302648786919643?s=20"
[cover]
      image = "img/mcjev_overview_v2.png"
      alt = "MCJev: the fight as text, one forward pass of DJev, a distribution over moves; 24 ms per decision"
      caption = "MCJev: a 24 ms Jev for Minecraft PvP on NVIDIA Blackwell GPUs"
      hidden = true
+++

{{< socialBadges github="alexzms/MCJev" demo="https://mc.alexzms.com" x="https://x.com/haoailab/status/2104302648786919643?s=20" >}}

{{< justify >}}

**TL;DR**

- **What we built.** Minecraft PvP bots that real people fight on a public server, where every move is picked by a [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)-style model: it scores every option in one forward pass, with no tokens generated. We used [DJev](https://github.com/mmastrac/djev) on Google's DiffusionGemma, unchanged.
- **Fast enough to fight.** We rebuilt inference around that single pass: DJev-serve is 2.5× faster than upstream vLLM, and a decision takes ~24 ms in real rounds, inside one 50 ms server tick.
- **Built on NVIDIA Blackwell.** We developed DJev-serve on NVIDIA GB200 (NVL72) and B200 GPUs and serve the public arena from both: with batching, one B200 makes 58–64 decisions a second, and eight GB200 GPUs behind our gateway handle about 360 a second.
- **Agent evolution: iterate in hours, training free.** Teams that use Jev-style models for classification and decision jobs need to iterate fast, and training makes every iteration slow. We trained nothing: an LLM reflecting on real rounds took our Block UHC harness from v1 to the v10 that fought the public in about nine hours. Agent evolution is a practical, training-free way to adapt Jev to a new business scenario quickly, and the loop is not specific to games.
<!-- - **Future Path.** Minecraft is our testbed; at the end we look at what may carry over to physical AI. -->

We collaborated closely with the NVIDIA Enterprise Products team (Pen Chung Li) on MCJev. MCJev runs on **NVIDIA GB200 NVL72** and **B200** GPUs provided by NVIDIA. We are deeply grateful for their support.


{{< /justify >}}

## What We Want to Achieve

{{< justify >}}

Many real decisions have to be made while the world keeps moving: flagging a payment before it clears, holding back a chat message before other people see it, routing a support call, choosing what a game character or a robot does next. Each is the same job: read the situation, pick one of a fixed set of options, and do it within tens of milliseconds, often millions of times a day. A large model that reasons before it answers is too slow and too costly per decision for that; a classifier trained for the job is fast, but every change means new data and a new training run. We want a third option: a lightweight, low-latency decision model, used as is, with a harness that can be iterated in hours.

We test that in the hardest setting we could build cheaply: Minecraft PvP against real people. A server ticks 20 times a second, a person reacts in about a quarter of a second, and a full-power arrow crosses the arena in half a second, so an agent that takes a second to decide is a target, not an opponent. Rounds cost nothing, one server runs dozens at once, and the opponents adapt (Figure 1).

{{< /justify >}}

{{< image src="img/applications_v2.svg" alt="Real-time decision jobs and the Minecraft PvP test" width="100%" title="Figure 1. Many real-time decision jobs share one shape: the situation comes in, one of a few options must go out within tens of milliseconds, and the rules change often. MCJev tests that shape in Minecraft PvP, where the deadline is a 50 ms server tick, the rules are a harness an LLM iterates, and the other side is a person who adapts.">}}

## Why Jev

{{< justify >}}

A [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)-style model decides without writing. It scores every option in one forward pass and returns a full distribution over them, so each answer comes with its own confidence: act on the sure calls, and hand the unsure ones to a person or a bigger model. An agent that writes a chain of thought first spends far more than a tick on every move: at 200 tokens a second, 200 tokens of reasoning take a full second. DJev, the Jev-style model we use, is also light: DiffusionGemma-26B-A4B activates about 4B of its 26B parameters per token, and on DJev-serve a decision takes about 24 ms, inside one server tick and about ten times faster than a person can react (Figure 2). How one pass makes a decision is in [Decide Without Decoding](#decide-without-decoding).

{{< /justify >}}

{{< image src="img/how_fast.png" alt="How fast is that" width="100%" title="Figure 2. One Jev decision in real rounds (median) against one server tick, a person's reaction to a visual cue, and an LLM that reasons before it answers (estimated: 200 reasoning tokens at 200 tokens/s, prompt processing not counted).">}}

## How It Works

{{< justify >}}

Each bot is a deciding head over an even faster actioner:

- The **actioner** is a bot client built on [mineflayer](https://github.com/PrismarineJS/mineflayer). Every 50 ms tick it reads what a player could see and runs the reflexes that cannot wait: aiming, leading arrows through Minecraft's drag and gravity plus the measured network delay, sidestepping arrows it sees coming, never walking into lava or off a ledge.
- The **head** is [DJev](https://github.com/mmastrac/djev), served by **DJev-serve**, our inference engine. Given the fight as text, it picks the tactic.

{{< /justify >}}

{{< image src="img/mcjev_overview_v2.png" alt="MCJev at a glance" width="100%" title="Figure 3. MCJev at a glance. The actioner reads the game through the mineflayer API every tick, and the harness writes the fight down as text with one answer slot left masked; DJev fills the slot in a single forward pass, with zero tokens generated; the logits at the slot become a distribution over the moves offered, and Jev presses the top one (the purple loop, ~24 ms). The gold loop is reflection: we read the logs of what Jev did, edit the harness and reload it, v1 to v10 in about nine hours with no training (next section). Bottom: one decision in real rounds (median), upstream vLLM against DJev-serve.">}}

{{< justify >}}

The split matters because a model call, however fast, is not a tick: the actioner keeps aiming while the head thinks. It also decides *when* the head is asked. While a bow is drawn, asking every step adds up many small chances of letting go too early, so the head is asked only at the draw's real decision points. Robotics uses the same split, a slower policy over fast low-level controllers, and MCJev is a game-sized version of it in which every part can be measured.

The text the head reads is the harness, and most of Jev's gains came from it rather than from the model. The next section shows how we iterated it, with an LLM in the loop.

{{< /justify >}}

## How It Evolves

{{< justify >}}

Training turns every change into days or weeks of collecting data, training and evaluating. We changed the prompt instead, and let an LLM do the reflecting (Figure 4).

{{< /justify >}}

{{< image src="img/reflection_loop_llm.svg" alt="The reflection loop" width="100%" title="Figure 4. Agent reflection. The harness writes the prompt, Jev plays, every round is observed, and an LLM rewrites the prompt from what it sees; the bots reload it in place. The model that plays never changes.">}}

{{< justify >}}

1. **Prompt.** Every tick, the actioner reads the game through the mineflayer API and works out the facts in code (distances, reach, damage, cover). The harness writes them into a prompt of ~1,760 tokens and offers only the moves that make sense right now, each saying what it would do.
2. **Play and observe.** Jev plays real rounds. Each round leaves a recording of both sides and a log of every decision: the prompt Jev read, DJev's full distribution over the options, and what happened next.
3. **Iterate with an LLM.** An LLM (Opus 5.5 or Astra) reads the logs, finds where Jev went wrong (a fight it misread, a move it was not offered, a question asked at the wrong moment, a call that should have been a reflex) and rewrites the prompt, and the actioner's reflexes where needed.
4. **Reload.** The bots load the new harness in place, with no restart and no retraining, and the loop runs again.

{{< /justify >}}

{{< image src="img/harness_timeline_v2.png" alt="Harness versions over time" width="100%" title="Figure 5. The Block UHC harness over time (git commit times). Eight versions in the first hour; v10, the version that fought the public, about nine hours after v1; the next version is being iterated now.">}}

{{< justify >}}

**What it cost.** v1 to v8 took 54 minutes, v10 came about nine hours after v1, and the agent went from its first line of code to a public release in about a day (Figure 5). Every change was a text edit and a reload: no data collection, no labels, no backpropagation, and the same off-the-shelf model throughout. Training is still the long-run path, and these logs are its data ([What's Next](#whats-next)); but to get a Jev-style model doing a new job well, an LLM reflecting on the prompt was by far the cheaper and faster loop. We are now iterating on the next version, and it will be fast: since the server opened, the loop has the rounds and decisions of every player who fought Jev to learn from.

{{< /justify >}}

## Decide Without Decoding

{{< justify >}}

Chat models answer by writing: one token, then the next. For a decision among sixteen moves that is wasted work, and on a 26B model it is far too slow for a fight where an arrow is in the air for half a second. DJev asks a diffusion language model for the answer in a different way.

DiffusionGemma pairs a causal **encoder**, which reads the prompt into a KV cache, with a bidirectional **decoder** that denoises a "canvas" (up to 256 tokens; we serve 64) conditioned on it; normal generation starts from a noised canvas and denoises it over dozens of refinement steps. A Jev-like read skips all that:

- Write an *answer scaffold* into the canvas (`1: ▒`, one slot per question) and leave only the slots noised.
- Run a single decoder pass that denoises every slot at once.
- At each slot, take the logits, keep only that question's candidate tokens (` A`, ` B`, …, or ` yes`/` no`), and renormalise.

That is the full probability distribution over the options, with no token ever sampled. This is how [DJev](https://github.com/mmastrac/djev) works: [mmastrac](https://github.com/mmastrac)'s Jev-style decision server for DiffusionGemma, built on the structured reads he added to vLLM ([#57250](https://github.com/vllm-project/vllm/pull/57250), [#58216](https://github.com/vllm-project/vllm/pull/58216)).

{{< /justify >}}

{{< image src="img/djev_readout.svg" alt="How DJev reads a decision" width="100%" title="Figure 6. One decision. The state and the question go into the prompt; the canvas holds the answer scaffold with the slot left noised; one decoder pass fills it; the logits at the slot, restricted to the options' tokens, are the distribution Jev acts on.">}}

{{< justify >}}

This is what TypeSafe calls a [*System One*](https://typesafe.ai/blog/introducing-system-one-models-and-jev) decision: fast and structured, with no reasoning tokens. DiffusionGemma used this way is very good at reading a situation and choosing (in our probes, device selection, maze safety from a small view and email triage all scored 0.95–1.0), and it will not do arithmetic for you; the harness does the numbers (distances, arrow flight times, cooldowns) and hands the model the conclusions.

{{< /justify >}}

{{< image src="img/decisions_round_v2.png" alt="A real round" width="100%" title="Figure 7. Four seconds of a real round: Jev2 made 131 decisions in 4.4 s. Each bar is one decision; its height is the probability DJev gave the move it took, its colour the move. Runs of 0.99 are firm commitments (here, taking cover); the lower, varied bars are the close calls.">}}

## Inference Engine

{{< justify >}}

**Our contribution on NVIDIA Blackwell:**

- **DJev-serve**: a Jev-style read as one forward pass replayed from a CUDA graph, 2.5× faster than upstream vLLM: ~24 ms a decision in real rounds, against ~60 ms.
- **Blackwell kernel choices for DiffusionGemma**: FlashAttention-2 for the sliding-window layers, a tuned Triton kernel for the global layers, and CUTLASS grouped GEMMs through FlashInfer for the 128-expert MoE.
- **A public serving pool on GB200 NVL72 and B200**: about 360 decisions a second on eight GB200 GPUs, around the clock.
- **All of it open source**, with NVFP4 as the next step.

Three stages of serving the same model:

| | One decision | What changed |
|---|---|---|
| Hugging Face transformers | 174 ms | the first working version |
| upstream vLLM with the structured read (DJev) | ~60 ms | no generation loop; vLLM's model code and kernels |
| **DJev-serve** | **~24 ms** | one forward pass per decision, captured in a CUDA graph |

(174 ms: the first prototype on one NVIDIA GB200, a short test request; it never served real rounds. ~60 and ~24 ms: medians in real rounds (Figure 12). On a controlled benchmark, the same recorded request with the same weights on one NVIDIA B200, DJev-serve is the same 2.5× faster.)

**Where the time went.** The first version launched **10,340 kernels** per forward, mostly tiny glue around 30 layers, and kept the GPU busy for only a third of the time; 8 requests batched together took barely longer than one. DJev on upstream vLLM removes most of that, but still runs prefill and denoise as two scheduled steps over the full canvas width.

**DJev-serve.** We built it by autoresearch: an AI coding agent ran most of the profile, change and benchmark loop, while we chose what to try next and checked every number. It keeps vLLM's model code, paged KV cache and attention kernels, drops the scheduler, and computes the same read in exactly one forward pass:

- **One CUDA graph from token ids to answer log-probs**, one per prompt-length bucket; per decision we fill two small index buffers and replay.
- **A prefix cache inside the graph**: each request reuses the KV of the longest block-aligned prefix it shares with a recent prompt, and the prefix length is just data in the index buffers.
- **FlashAttention-2 for the 25 sliding-window layers** (about 4× faster per layer than vLLM's Triton attention on Blackwell); the 5 global layers (head size 512) stay on a tuned Triton kernel.
- **NVIDIA kernels for the MoE**, the largest share of each forward (Figure 8): CUTLASS grouped GEMMs through FlashInfer, 7% faster than vLLM's Triton MoE kernel.

The result is **~24 ms a decision** in real rounds, request to answer, most of it the GPU forward itself, so the bottleneck is now the GPU. The raw benchmark numbers, versions and the full layout are in [serving/docs/PERFORMANCE.md](https://github.com/alexzms/MCJev/blob/main/serving/docs/PERFORMANCE.md).

**NVFP4.** NVIDIA publishes an NVFP4 checkpoint of the same model, [nvidia/diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4), in Blackwell's native 4-bit format. Through vLLM on GB200 it was not yet faster than BF16 at our batch size (about 9% slower on a short read), but with GEMMs more than half of every forward, bringing it into DJev-serve's graph is our next step, along with FlashAttention-4, which we have not benchmarked yet.

{{< /justify >}}

{{< image src="img/gpu_breakdown_v2.png" alt="GPU time breakdown" width="100%" title="Figure 8. Where the GPU time of one forward goes (a real MCJev step after the prefix cache, kernel times summed from an eager profile): the MoE experts (128 of them, 8 per token) dominate, then the other projections and attention.">}}

{{< justify >}}

**What a decision costs.** About **6 ms fixed per forward** plus **~10.4 ms per 1,000 tokens** not in the prefix cache, which turns prompt design into a systems problem: of a bot's ~1,760-token prompt only ~480 hit the cache, because the line that changes every step sat before a fixed section, and each bot's name came in the first sentence. Harness authors and serving engineers have to design the prompt together.

**Batching** up to four requests in one forward gives 1.5× the throughput without slowing a single bot down, and takes a GPU from 43 to 58–64 steps a second.

{{< /justify >}}

{{< image src="img/scaling_batching_v2.png" alt="Scaling and batching" width="100%" title="Figure 9. Left: a forward costs ~6 ms plus ~10.4 ms per 1,000 uncached tokens. Right: batching requests into one forward.">}}

## Serving Many Jevs

{{< justify >}}

A public server needs many Jevs at once: we now run a pool of 24, and at busy moments many of them fight at once. Each GPU runs its own copy of DJev-serve; a small **gateway** gives the agents one address, sends each request to the healthy engine with the fewest requests in flight, checks every engine with a tiny evaluate every two seconds, and retries on the next one if an engine dies mid-request. Adding a node is one line in a file.

Because a Jev asks for its next decision only when the last one has answered, load is closed-loop: the worst case is every Jev in a fight at once. We replayed real decisions from the logs against the live gateway (eight NVIDIA GB200 GPUs at the time):

- a single stream gets 39 decisions a second;
- 8 streams get 324 a second, with no rise in latency;
- 24 streams flat out saturate at 362 a second, and the median latency climbs to 66 ms.

In real rounds a Jev averages about 6 decisions a second, so 24 fighting Jevs need ~140 a second, well inside the envelope.

{{< /justify >}}

{{< image src="img/gateway_load.png" alt="Gateway load test" width="80%" title="Figure 10. Closed-loop load on the gateway with real decisions (8 GPUs): throughput (bars) and latency (line).">}}

{{< justify >}}

The last few tens of milliseconds were not in the model at all. Our first agents ran on a laptop and reached the model through a Cloudflare tunnel: **~330 ms** per decision, most of it spent crossing the internet. Moving the agents onto the GPU node brought a decision down to the model's own latency, and later we moved the Minecraft server itself onto the same cluster, behind a login proxy, so a Jev's eyes, head and actioner all live on one cluster, with no hop across the internet anywhere on the path.

That also mattered for aim: when every observation of the opponent is ~150–300 ms stale, a moving target is somewhere else by the time the arrow arrives, and no amount of ballistics fixes that. A robot faces the same choice between a model in a distant data center and a model next to its actuators; our numbers are a small data point for keeping the head close to the body when the loop is tight.

{{< /justify >}}

{{< image src="img/network_path.png" alt="Network path" width="85%" title="Figure 11. What one decision cost end to end, as the bots saw it, as the pieces moved closer together (medians in real rounds).">}}

{{< image src="img/latency_timeline.png" alt="Latency over the release" width="100%" title="Figure 12. Every decision Jev made in real rounds during the release (94k decisions of the text harnesses, 5-minute medians and p90), request to answer as the bots saw it: ~60 ms on upstream vLLM, ~24 ms after the switch to DJev-serve. These are production medians over different hours and loads, not a controlled comparison; on a controlled same-request benchmark the speedup is the same 2.5×.">}}

## Play

{{< justify >}}

Minecraft: Java Edition 1.9 or newer, a paid game (the server checks accounts, so you need a Microsoft account that owns it): **Multiplayer → Add Server → `mc.alexzms.com`**. In the lobby, right-click the compass and pick a mode, or run down the red carpet onto the iron pads. If you win, you will be on the leaderboard in the lobby.

{{< /justify >}}

<video controls preload="metadata" poster="img/trailer_poster.jpg" style="width: 100%; border-radius: 6px;">
  <source src="img/trailer.mp4" type="video/mp4">
</video>
<p style="text-align:center; color:#808080; font-size:16px;">Figure 13. 41 seconds of real rounds: a lava trap, a dash across lava at half a heart, a comeback at 4 HP.</p>

## How It Performs

{{< justify >}}

On September 26 we opened the server to anyone with Minecraft, with the v10 harness. Players pick one of three modes in the lobby:

- **Block UHC**: both fighters get a sword, a bow, lava and water buckets and blocks. Damage doubles at 60 seconds, and 30 seconds later both are dropped onto a small platform ringed by lava.
- **Sumo**: a single platform and no weapons. Knock the other one off.
- **2v2**: you and a friend against two Jevs.

36 UHC arenas run side by side, and every round is recorded. In the first ~32 hours (September 26, 19:05 UTC to September 28, 03:29 UTC, across our two servers):

- **54 people** fought Jev in the lobby's one-on-one modes (counted by Minecraft account; the numbers below include our team's own accounts).
- **1,795 rounds** against Jev: 870 of Block UHC, which **Jev won 70%** of (the fastest anyone killed it: 1.2 seconds; the median winning round for a person: 21.5 s), and 925 of Sumo, which **people won 88%** of. Jev is a much better duelist than a sumo wrestler, and the Sumo rounds are the clearest to-do list we have.
- **1,757 recorded matches**, 10.2 hours in all: both sides every tick (position, view, movement, held item, health) and every swing, hit, arrow, block and death, 12.3 MB compressed.
- **More than 130,000 logged decisions** on our main cluster alone (our own test fights included), each with the full text state Jev read and DJev's complete distribution over the options, plus how the move turned out.

{{< /justify >}}

{{< image src="img/release_results.png" alt="Who won" width="80%" title="Figure 14. Lobby one-on-one rounds against Jev in the first 32 hours, by mode.">}}

<style>
.mcjev-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 10px 0 4px; }
.mcjev-grid video { width: 100%; height: auto; border-radius: 4px; margin: 0; display: block; }
@media (max-width: 700px) { .mcjev-grid { grid-template-columns: 1fr; } }
</style>
<div class="mcjev-grid">
  <video autoplay muted loop playsinline preload="metadata" poster="img/loop_counter_poster.jpg" src="img/loop_counter.mp4" aria-label="Jev fights from behind cover and wins"></video>
  <video autoplay muted loop playsinline preload="metadata" poster="img/loop_neon_poster.jpg" src="img/loop_neon.mp4" aria-label="Jev comes through the player's lava and wins"></video>
</div>
<p style="text-align:center; color:#808080; font-size:16px;">Figure 15. Two real Block UHC rounds, players' view (looping clips). Left: Jev fights from behind its cover, trades hits and wins. Right: the player pours lava around a pillar; Jev comes through it, sword first, and wins in 19.4 seconds.</p>

## What's Next

{{< justify >}}

MCJev started as a latency experiment. The release turned it into an arena where every decision is on disk with its outcome. Three directions follow from that:

**Evolve.** Jev plays with zero training today. The logs are training data: supervised fine-tuning on the side that won, then reinforcement learning with the round's outcome as the reward. Each decision already stores DJev's full distribution over the options from one forward pass, which keeps the RL simple.

**Harness.** Keep the reflection loop, but judge each version on a varied suite of fights rather than one opponent, put the fixed parts of the context first for the prefix cache, and eventually let Jev see the screen instead of a description of it.

**Arena.** Jevs with different harnesses and checkpoints playing each other around the clock, as a league where old versions stay in and human rounds anchor the ratings. 2v2 already runs, with two Jevs sharing what they see.

Underneath, the systems work continues: batching across Jevs on one GPU, lower-precision weights and newer attention kernels to push a decision under 20 ms.

{{< /justify >}}

<!-- Toward Physical AI: hidden for now, restore by removing this comment wrapper.

## Toward Physical AI

{{< justify >}}

Minecraft was never the end goal. A world that keeps moving while the model thinks is also a central constraint of physical AI: a robot arm picking parts off a conveyor cannot pause the belt while it decides. We do not know yet how physical AI is best built, and a game is not a robot, but MCJev let us practise one slice of that setting, end to end and in front of real opponents.

{{< /justify >}}

{{< image src="img/robot_dojo.svg" alt="From a Minecraft dojo to a robot dojo" width="100%" title="Figure 16. From a Minecraft dojo to a robot dojo. Right: an imagined dynamic manipulation task, where the conveyor speeds up while the arm is reaching, so where to grasp has to be re-decided on the move.">}}

{{< justify >}}

Some of what we learned may carry over: deciding in one forward pass, a slower head over a fast actioner, a latency budget that runs from kernels to the network, and every decision logged with its outcome. Which of these hold up on a robot, for example on manipulation benchmarks such as [RoboDojo](https://robodojo-benchmark.com/), whose simulated tasks run on NVIDIA Isaac Sim, or in scenes that change while the arm moves, and how far the head can shrink when it has to run on the robot itself, on a computer like NVIDIA Jetson Thor, is what we want to explore next.

{{< /justify >}}

-->

## Acknowledgement

{{< justify >}}

We especially thank **NVIDIA**, and every NVIDIA team that provided hardware support for this work: MCJev was built on and runs on NVIDIA **GB200 NVL72** and **B200** GPUs, and a pool of Jevs fighting the public around the clock, each asking for a decision every few tens of milliseconds, is only possible on that hardware. We thank the NVIDIA Enterprise Products team (Pen Chung Li) for collaborating on MCJev. We thank TypeSafe AI, whose [Jev and System One models](https://typesafe.ai/blog/introducing-system-one-models-and-jev) this work takes its idea and its name from; Google for releasing DiffusionGemma with open weights; the vLLM project that DJev-serve builds on; [mmastrac](https://github.com/mmastrac), for [DJev](https://github.com/mmastrac/djev) and the structured reads in vLLM it runs on ([#57250](https://github.com/vllm-project/vllm/pull/57250), [#58216](https://github.com/vllm-project/vllm/pull/58216)); and everyone who logged in and fought Jev.

{{< /justify >}}

## Contributors

[Minshen (Alex) Zhang](https://alexzms.github.io/), [Junda Chen](https://github.com/GindaChen), [Yuanbo Yang](https://freemty.github.io/), [Yuxuan Zhang](https://yuxuandexter.github.io/), [Yi Sun](https://www.linkedin.com/in/yi-sun-mlsys), [Shaoxiong Duan](https://github.com/shaoxiongduan), [Lanxiang Hu](https://snyhlxde1.github.io/), [Jiaqi Leng](https://jacky-leng.github.io/), [Yulun Wu](https://memset0.github.io), [Will Lin](https://github.com/SolitaryThinker), [Hao Zhang](https://haozhang.ai)

**Cite us:**

```bibtex
@misc{mcjev2026,
  title        = {{MCJev}: A 24 ms {Jev} for {Minecraft} {PvP} on {NVIDIA} {Blackwell} {GPUs}},
  author       = {Zhang, Minshen and Chen, Junda and Yang, Yuanbo and Zhang, Yuxuan and Sun, Yi and Duan, Shaoxiong and Hu, Lanxiang and Leng, Jiaqi and Wu, Yulun and Lin, Will and Zhang, Hao},
  year         = {2026},
  month        = oct,
  howpublished = {Hao AI Lab blog},
  url          = {https://haoailab.com/blogs/mcjev/}
}
```
