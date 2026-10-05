+++
title = "Real-Time Decisions with Jev: Self-Evolution and a 24 ms Engine on NVIDIA Blackwell"
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
      image = "img/mcjev_overview_v3.png"
      alt = "MCJev: the fight as text, one forward pass of DJev, a distribution over moves; 24 ms per decision"
      caption = "MCJev: real-time decisions with Jev, self-evolving and at 24 ms on NVIDIA Blackwell"
      hidden = true
+++

{{< socialBadges github="alexzms/MCJev" demo="https://mc.alexzms.com" x="https://x.com/haoailab/status/2104302648786919643?s=20" >}}

{{< justify >}}

**TL;DR**

- **What we built.** MCJev: an agent that makes every move with a [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)-style model ([DJev](https://github.com/mmastrac/djev)) in one forward pass, tested against real people in Minecraft PvP.
- **Self-evolves on a complex task, no training.** With a SOTA LLM agent and human supervision, we evolve the prompt rapidly from the bot's own logs. In under two hours, the bot went from losing every bench round against a scripted fighter at full skill to winning every one.
- **24 ms per decision on NVIDIA Blackwell.** Our engine, DJev-serve, is 2.5× faster than a leading inference engine.
- **Tested against real people, at scale.** 54 people played 1,795 public rounds in 32 hours, and Jev won 70% in Block UHC. A first step toward real-time AI and physical AI.

We collaborated closely with Pen Chung Li at NVIDIA on MCJev. MCJev runs on **NVIDIA GB200 NVL72** and **B200** GPUs provided by NVIDIA. We are deeply grateful for their support.

{{< /justify >}}

## The Problem: Real-Time Decisions

{{< justify >}}

Many real decisions have to be made while the world keeps moving: flagging a payment before it clears, holding back a chat message before other people see it, routing a support call, choosing what a game character or a robot does next. Each is the same job: read the situation, pick one of a fixed set of options, and do it within tens of milliseconds, often millions of times a day. A large model that reasons before it answers is too slow and too costly per decision for that; a classifier trained for the job is fast, but every change means new data and a new training run. We want a third option: a lightweight, low-latency decision model, used as is, with a harness that can be iterated in hours (Figure 1).

{{< /justify >}}

{{< image src="img/applications_v2.svg" alt="Real-time decision jobs and the Minecraft PvP test" width="100%" title="Figure 1. Many real-time decision jobs share one shape: the situation comes in, one of a few options must go out within tens of milliseconds, and the rules change often. MCJev tests that shape in Minecraft PvP, where the deadline is a 50 ms server tick, the rules are a prompt we evolve with an LLM agent, and the other side is a person who adapts.">}}

{{< justify >}}

A [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)-style model fits that job. It decides without writing: it scores every option in one forward pass and returns a full distribution over them, so each answer comes with its own confidence: act on the sure calls, and hand the unsure ones to a person or a bigger model. An agent that writes a chain of thought first cannot meet a budget of tens of milliseconds: at 200 tokens a second, 200 tokens of reasoning take a full second. DJev, the Jev-style model we use, is also light: DiffusionGemma-26B-A4B activates about 4B of its 26B parameters per token, and on DJev-serve a decision takes about 24 ms, about ten times faster than a person can react (Figure 2).

{{< /justify >}}

{{< image src="img/how_fast.png" alt="How fast is that" width="100%" title="Figure 2. One Jev decision in real rounds (median) against one server tick, a person's reaction to a visual cue, and an LLM that reasons before it answers (estimated: 200 reasoning tokens at 200 tokens/s, prompt processing not counted).">}}

{{< justify >}}

We needed a test where decisions cannot wait, the other side adapts, and every decision is scored, and that we could still run cheaply and at scale. Minecraft PvP against real people is that test. A server ticks 20 times a second and a person reacts in about a quarter of a second, so an agent that takes a second to decide is a target, not an opponent. Rounds cost nothing, one server runs dozens at once, every decision is logged with how it turned out, and the opponents are people who learn the agent's habits. The main mode, Block UHC, is also a complex task rather than one simple choice: a sword, a bow, lava and blocks, and more than a dozen moves to pick from at every step. The game itself, and how to play it, are in [Play](#play).

{{< /justify >}}

## How It Evolves

{{< justify >}}

Jev decides from a harness: the prompt that describes the situation and the options it is offered. Most of Jev's gains came from the harness rather than from the model, so the harness is what we evolve. Training turns every change into days or weeks of collecting data, training and evaluating. Instead, we evolve the prompt rapidly with a SOTA LLM agent under human supervision: the agent reads the logs of real rounds and drafts the changes, and we decide what to keep (Figure 3).

{{< /justify >}}

{{< image src="img/reflection_loop_master.svg" alt="The reflection loop" width="100%" title="Figure 3. Agent reflection. The harness writes the prompt, Jev plays, every round is observed, and we evolve the prompt with an LLM agent from what we see; the bots reload it in place. The model that plays never changes.">}}

{{< justify >}}

1. **Prompt.** Every tick, the bot reads the game and works out the facts in code (distances, reach, damage, cover). The harness writes them into a prompt of ~1,760 tokens and offers only the moves that make sense right now, each saying what it would do.
2. **Play and observe.** Jev plays real rounds. Each round leaves a recording of both sides and a log of every decision: the prompt Jev read, DJev's full distribution over the options, and what happened next.
3. **Evolve the prompt.** We go through the logs with a SOTA LLM agent (Claude Opus 5.5 or GPT-6 Astra). It finds where Jev went wrong (a fight it misread, a move it was not offered, a question asked at the wrong moment, a call that should have been a reflex) and drafts a new prompt, and new reflexes for the bot where needed; we review each change and choose what to play next.
4. **Reload.** The bots load the new harness in place, with no restart and no retraining, and the loop runs again.

{{< /justify >}}

{{< image src="img/harness_timeline_v2.png" alt="Harness versions over time" width="100%" title="Figure 4. The Block UHC harness over time (git commit times). Eight versions in the first hour; v10, the version that fought the public, about nine hours after v1; the next version is being iterated now.">}}

{{< justify >}}

**Did it get better?** Block UHC is complex enough that the first harness lost. We benchmarked versions against the same scripted opponent, a fighter bot at full skill, three rounds per version: v1 lost all three, v6 won two, and v9, under two hours after v1, won all three. Against people, v10 went on to win 70% of 870 Block UHC rounds ([How It Performs](#how-it-performs)). The bench is small, and not everything improved: against a scripted archer, v9 still won only 2 of 14.

**What it cost.** v1 to v8 took 54 minutes, v10 came about nine hours after v1, and the agent went from its first line of code to a public release in about a day (Figure 4). Every change was a text edit and a reload: no data collection, no labels, no backpropagation, and the same off-the-shelf model throughout. Nothing in the loop is specific to games: any job where a Jev-style model reads a state, picks from options and leaves a log of how its choices turned out can be iterated the same way. We are now iterating on the next version, and it will be fast: since the server opened, the loop has the rounds and decisions of every player who fought Jev to learn from.

{{< /justify >}}

## How It Works

{{< justify >}}

Each bot is a deciding head over an even faster actioner (Figure 5). The **actioner**, a [mineflayer](https://github.com/PrismarineJS/mineflayer) client, reads the game every 50 ms tick and runs the reflexes that cannot wait: aiming, dodging arrows, never walking into lava. The **head**, [DJev](https://github.com/mmastrac/djev) served by **DJev-serve**, reads the fight as text and picks the tactic. The split is general: fast code for what cannot wait and a model for the judgement call, the same shape as a slower policy over fast low-level controllers in robotics.

{{< /justify >}}

{{< image src="img/mcjev_overview_v3.png" alt="MCJev at a glance" width="100%" title="Figure 5. MCJev at a glance. The actioner reads the game through the mineflayer API every tick, and the harness writes the fight down as text with one answer slot left masked; DJev fills the slot in a single forward pass, with zero tokens generated; the logits at the slot become a distribution over the moves offered, and Jev presses the top one (the purple loop, ~24 ms). The gold loop is reflection: we read the logs of what Jev did, edit the harness and reload it, v1 to v10 in about nine hours with no training (How It Evolves). Bottom: one decision in real rounds (median), a leading inference engine against DJev-serve.">}}

## How Fast It Runs

{{< justify >}}

A real-time decision has a budget, and every part of the path counts against it: how the model is asked, the engine that runs it, and the network around it.

{{< /justify >}}

### Decide without decoding

{{< justify >}}

Chat models answer by writing, one token at a time; for a choice among sixteen options that is wasted work. DJev asks a diffusion language model for the answer differently. DiffusionGemma pairs a causal **encoder**, which reads the prompt into a KV cache, with a bidirectional **decoder** that denoises a "canvas" (we serve 64 tokens of it). A Jev-like read writes an answer scaffold into the canvas (`1: ▒`, one slot per question), runs a single decoder pass, and at each slot keeps only that question's candidate tokens (` A`, ` B`, …) and renormalises (Figure 6). That is the full probability distribution over the options, with no token ever sampled. This is how [DJev](https://github.com/mmastrac/djev) works: [mmastrac](https://github.com/mmastrac)'s Jev-style decision server for DiffusionGemma, built on the structured reads he added to vLLM ([#57250](https://github.com/vllm-project/vllm/pull/57250), [#58216](https://github.com/vllm-project/vllm/pull/58216)).

DiffusionGemma used this way is very good at reading a situation and choosing (in our probes, device selection, maze safety from a small view and email triage all scored 0.95–1.0), but it will not do arithmetic for you; the harness does the numbers and hands the model the conclusions.

{{< /justify >}}

{{< image src="img/djev_readout.svg" alt="How DJev reads a decision" width="100%" title="Figure 6. One decision. The state and the question go into the prompt; the canvas holds the answer scaffold with the slot left noised; one decoder pass fills it; the logits at the slot, restricted to the options' tokens, are the distribution Jev acts on.">}}

### The inference engine

{{< justify >}}

**Our contribution on NVIDIA Blackwell:**

- **DJev-serve**: a Jev-style read as one forward pass replayed from a CUDA graph, 2.5× faster than a leading inference engine: ~24 ms a decision in real rounds, against ~60 ms.
- **Blackwell kernel choices for DiffusionGemma**: FlashAttention-2 for the sliding-window layers, a tuned Triton kernel for the global layers, and CUTLASS grouped GEMMs through FlashInfer for the 128-expert MoE.
- **A public serving pool on GB200 NVL72 and B200**: about 360 decisions a second on eight GB200 GPUs, around the clock.
- **All of it open source**, with NVFP4 as the next step.

Three stages of serving the same model:

| | One decision | What changed |
|---|---|---|
| Hugging Face transformers | 174 ms | the first working version |
| a leading inference engine with the structured read (DJev) | ~60 ms | no generation loop, but general-purpose scheduling |
| **DJev-serve** | **~24 ms** | one forward pass per decision, captured in a CUDA graph |

(174 ms: the first prototype on one NVIDIA GB200, a short test request; it never served real rounds. ~60 and ~24 ms: medians in real rounds (Figure 12). On a controlled benchmark, the same recorded request with the same weights on one NVIDIA B200, DJev-serve is the same 2.5× faster.)

**Where the time went.** The first version launched **10,340 kernels** per forward, mostly tiny glue around 30 layers, and kept the GPU busy for only a third of the time; 8 requests batched together took barely longer than one. DJev on the leading inference engine removes most of that, but still runs prefill and denoise as two scheduled steps over the full canvas width.

**DJev-serve.** We built it by autoresearch: an AI coding agent ran most of the profile, change and benchmark loop, while we chose what to try next and checked every number. It keeps vLLM's model code, paged KV cache and attention kernels, drops the scheduler, and computes the same read in exactly one forward pass:

- **One CUDA graph from token ids to answer log-probs**, one per prompt-length bucket; per decision we fill two small index buffers and replay.
- **A prefix cache inside the graph**: each request reuses the KV of the longest block-aligned prefix it shares with a recent prompt, and the prefix length is just data in the index buffers.
- **FlashAttention-2 for the 25 sliding-window layers** (about 4× faster per layer than the Triton attention kernel on Blackwell); the 5 global layers (head size 512) stay on a tuned Triton kernel.
- **NVIDIA kernels for the MoE**, the largest share of each forward (Figure 7): CUTLASS grouped GEMMs through FlashInfer, 7% faster than the Triton MoE kernel.

The result is **~24 ms a decision** in real rounds, request to answer, most of it the GPU forward itself, so the bottleneck is now the GPU. The raw benchmark numbers, versions and the full layout are in [serving/docs/PERFORMANCE.md](https://github.com/alexzms/MCJev/blob/main/serving/docs/PERFORMANCE.md).

**NVFP4.** NVIDIA publishes an NVFP4 checkpoint of the same model, [nvidia/diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4), in Blackwell's native 4-bit format. In our tests on GB200 it was not yet faster than BF16 at our batch size (about 9% slower on a short read), but with GEMMs more than half of every forward, bringing it into DJev-serve's graph is our next step, along with FlashAttention-4, which we have not benchmarked yet.

{{< /justify >}}

{{< image src="img/gpu_breakdown_v2.png" alt="GPU time breakdown" width="100%" title="Figure 7. Where the GPU time of one forward goes (a real MCJev step after the prefix cache, kernel times summed from an eager profile): the MoE experts (128 of them, 8 per token) dominate, then the other projections and attention.">}}

{{< justify >}}

**What a decision costs.** About **6 ms fixed per forward** plus **~10.4 ms per 1,000 tokens** not in the prefix cache, which turns prompt design into a systems problem: of a bot's ~1,760-token prompt only ~480 hit the cache, because the line that changes every step sat before a fixed section, and each bot's name came in the first sentence. Harness authors and serving engineers have to design the prompt together.

**Batching** up to four requests in one forward gives 1.5× the throughput without slowing a single bot down, and takes a GPU from 43 to 58–64 steps a second.

{{< /justify >}}

{{< image src="img/scaling_batching_v2.png" alt="Scaling and batching" width="100%" title="Figure 8. Left: a forward costs ~6 ms plus ~10.4 ms per 1,000 uncached tokens. Right: batching requests into one forward.">}}

### Serving many Jevs

{{< justify >}}

A public server needs many Jevs at once; we run a pool of 24. Each GPU runs its own copy of DJev-serve behind a small **gateway**, which sends each request to the healthy engine with the fewest requests in flight and retries on another if one dies mid-request. Replaying real decisions against the live gateway on eight NVIDIA GB200 GPUs, 8 streams got 324 decisions a second with no rise in latency, and 24 streams flat out saturated at 362 a second (Figure 9). A fighting Jev averages about 6 decisions a second, so 24 of them need only ~140.

{{< /justify >}}

{{< image src="img/gateway_load.png" alt="Gateway load test" width="80%" title="Figure 9. Closed-loop load on the gateway with real decisions (8 GPUs): throughput (bars) and latency (line).">}}

{{< justify >}}

The network is part of the budget too. Our first agents reached the model through a tunnel from a laptop: **~330 ms** per decision, most of it spent crossing the internet. Moving the agents, and then the Minecraft server itself, onto the GPU cluster brought a decision down to the model's own latency (Figure 10). A robot faces the same choice between a model in a distant data center and one next to its actuators; our numbers are a small data point for keeping the head close to the body when the loop is tight.

{{< /justify >}}

{{< image src="img/network_path_v2.png" alt="Network path" width="85%" title="Figure 10. What one decision cost end to end, as the bots saw it, as the pieces moved closer together (medians in real rounds).">}}

## How It Performs

{{< justify >}}

On September 26 we opened the server to the public, with the v10 harness. In the first ~32 hours (September 26, 19:05 UTC to September 28, 03:29 UTC), **54 people** played **1,795 rounds** against Jev in one-on-one modes (counted by Minecraft account, our team's own accounts included):

- In **Block UHC**, a full fight with sword, bow, lava and blocks, **Jev won 70%** of 870 rounds (Figure 11).
- In **Sumo**, where the only move is to push, **people won 88%** of 925. Jev is a much better duelist than a sumo wrestler, and the Sumo rounds are the clearest to-do list we have.
- Through the release, a decision took **~24 ms** request to answer (Figure 12), fast enough for Jev to make 131 decisions in 4.4 seconds of one fight (Figure 13).
- Every decision left a log: **more than 130,000** on our main cluster alone (our own test fights included), each with the state Jev read, DJev's full distribution over the options and how the move turned out. This is the data the next version learns from.

{{< /justify >}}

{{< image src="img/release_results.png" alt="Who won" width="80%" title="Figure 11. Lobby one-on-one rounds against Jev in the first 32 hours, by mode.">}}

{{< image src="img/latency_timeline_v2.png" alt="Latency over the release" width="100%" title="Figure 12. Every decision Jev made in real rounds during the release (94k decisions of the text harnesses, 5-minute medians and p90), request to answer as the bots saw it: ~60 ms on the leading inference engine, ~24 ms after the switch to DJev-serve. These are production medians over different hours and loads, not a controlled comparison; on a controlled same-request benchmark the speedup is the same 2.5×.">}}

{{< image src="img/decisions_round_v2.png" alt="A real round" width="100%" title="Figure 13. Four seconds of a real round: Jev2 made 131 decisions in 4.4 s. Each bar is one decision; its height is the probability DJev gave the move it took, its colour the move. Runs of 0.99 are firm commitments (here, taking cover); the lower, varied bars are the close calls.">}}

## Beyond the Game

{{< justify >}}

MCJev is a game, but the constraint it tests is not: decide within tens of milliseconds while the world keeps moving, against a counterpart that adapts. Real-time response jobs face it, from flagging a payment before it clears to holding back a message before others see it, and so does physical AI: a robot arm picking parts off a conveyor cannot pause the belt while it decides. A game is neither a business nor a robot, and we do not know yet how far our results carry, but some of what we learned may: decide in one forward pass, put a slower head over a fast actioner, budget latency from the kernels to the network, log every decision with its outcome, and evolve the prompt with an LLM agent and human supervision in hours.

Next, we want to:

- **Train on the logs.** Jev plays with zero training today, and the logs are training data: supervised fine-tuning on the side that won, then reinforcement learning with the round's outcome as the reward. Each decision already stores DJev's full distribution over the options, which keeps the RL simple.
- **Evolve against more than one opponent.** Judge each harness version on a varied suite of fights, and run Jevs with different harnesses and checkpoints against each other around the clock, with human rounds anchoring the ratings.
- **Go faster.** NVFP4 weights, newer attention kernels and batching across Jevs on one GPU, to push a decision under 20 ms.
- **Try physical tasks.** Test which of these lessons hold up on a robot, for example on manipulation benchmarks such as [RoboDojo](https://robodojo-benchmark.com/), whose simulated tasks run on NVIDIA Isaac Sim, and how far the head can shrink to run on the robot itself, on a computer like NVIDIA Jetson Thor.

{{< /justify >}}

## Play

{{< justify >}}

Minecraft: Java Edition 1.9 or newer, a paid game (the server checks accounts, so you need a Microsoft account that owns it): **Multiplayer → Add Server → `mc.alexzms.com`**. In the lobby, right-click the compass and pick a mode, or run down the red carpet onto the iron pads. If you win, you will be on the leaderboard in the lobby.

- **Block UHC**: both fighters get a sword, a bow, lava and water buckets and blocks. Damage doubles at 60 seconds, and 30 seconds later both are dropped onto a small platform ringed by lava. 36 arenas run side by side.
- **Sumo**: a single platform and no weapons. Knock the other one off.
- **2v2**: you and a friend against two Jevs, which share what they see.

Some records from the first 32 hours: the fastest anyone killed Jev in Block UHC took 1.2 seconds, and the median winning round for a person took 21.5 s. All 1,757 matches were recorded, 10.2 hours in all: both sides every tick and every swing, hit, arrow, block and death, 12.3 MB compressed.

{{< /justify >}}

<video controls preload="metadata" poster="img/trailer_poster.jpg" style="width: 100%; border-radius: 6px;">
  <source src="img/trailer.mp4" type="video/mp4">
</video>
<p style="text-align:center; color:#808080; font-size:16px;">Figure 14. 41 seconds of real rounds: a lava trap, a dash across lava at half a heart, a comeback at 4 HP.</p>

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

## Acknowledgement

{{< justify >}}

We especially thank **NVIDIA**, and every NVIDIA team that provided hardware support for this work: MCJev was built on and runs on NVIDIA **GB200 NVL72** and **B200** GPUs, and a pool of Jevs fighting the public around the clock, each asking for a decision every few tens of milliseconds, is only possible on that hardware. We thank Pen Chung Li at NVIDIA for collaborating on MCJev. We thank TypeSafe AI, whose [Jev and System One models](https://typesafe.ai/blog/introducing-system-one-models-and-jev) this work takes its idea and its name from; Google for releasing DiffusionGemma with open weights; the vLLM project that DJev-serve builds on; [mmastrac](https://github.com/mmastrac), for [DJev](https://github.com/mmastrac/djev) and the structured reads in vLLM it runs on ([#57250](https://github.com/vllm-project/vllm/pull/57250), [#58216](https://github.com/vllm-project/vllm/pull/58216)); and everyone who logged in and fought Jev.

{{< /justify >}}

## Contributors

[Minshen (Alex) Zhang](https://alexzms.github.io/), [Junda Chen](https://github.com/GindaChen), [Yuanbo Yang](https://freemty.github.io/), [Yuxuan Zhang](https://yuxuandexter.github.io/), [Yi Sun](https://www.linkedin.com/in/yi-sun-mlsys), [Shaoxiong Duan](https://github.com/shaoxiongduan), [Lanxiang Hu](https://snyhlxde1.github.io/), [Jiaqi Leng](https://jacky-leng.github.io/), [Yulun Wu](https://memset0.github.io), [Will Lin](https://github.com/SolitaryThinker), [Hao Zhang](https://haozhang.ai)

**Cite us:**

```bibtex
@misc{mcjev2026,
  title        = {Real-Time Decisions with {Jev}: Self-Evolution and a 24 ms Engine on {NVIDIA} {Blackwell}},
  author       = {Zhang, Minshen and Chen, Junda and Yang, Yuanbo and Zhang, Yuxuan and Sun, Yi and Duan, Shaoxiong and Hu, Lanxiang and Leng, Jiaqi and Wu, Yulun and Lin, Will and Zhang, Hao},
  year         = {2026},
  month        = oct,
  howpublished = {Hao AI Lab blog},
  url          = {https://haoailab.com/blogs/mcjev/}
}
```
