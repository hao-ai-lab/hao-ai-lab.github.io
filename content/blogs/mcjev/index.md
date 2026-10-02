+++
title = "MCJev: A 24 ms Jev for Minecraft PvP"
date = 2026-09-27T12:00:00-07:00
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
      image = "img/cover_mcjev.png"
      alt = "MCJev: the fight as text, one forward pass of DJev, a distribution over moves; 24 ms per decision"
      caption = "MCJev: a 24 ms Jev for Minecraft PvP"
      hidden = true
+++

{{< socialBadges github="alexzms/MCJev" demo="https://mc.alexzms.com" x="https://x.com/haoailab/status/2104302648786919643?s=20" >}}

{{< justify >}}

**TL;DR**

- **Decisions without decoding.** [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)-style models answer a multiple-choice question by scoring every option in one forward pass, with no tokens generated. We took one off the shelf, [DJev](https://github.com/mmastrac/djev) on Google's DiffusionGemma, and trained nothing: everything in this post is the harness around the model and the inference under it.
- **Fast enough to fight.** We rebuilt inference around that single pass: on the same request, 56 ms on upstream vLLM becomes 22 ms on DJev-serve, and a decision takes ~24 ms in real rounds, inside one 50 ms server tick.
- **It wins at Block UHC.** In the first 32 hours after release, 54 people played 1,795 rounds against Jev; it won **70% of the Block UHC rounds**.
- **Open source, open to play.** The code is at [github.com/alexzms/MCJev](https://github.com/alexzms/MCJev), and the arena is at **`mc.alexzms.com`** (Minecraft: Java Edition, which needs a paid account).

Minecraft is our testbed; at the end we look at what may carry over to physical AI.

*MCJev runs on **NVIDIA B200** GPUs provided by NVIDIA. We are deeply grateful for their support.*

{{< /justify >}}

<video controls preload="metadata" poster="img/trailer_poster.jpg" style="width: 100%; border-radius: 6px;">
  <source src="img/trailer.mp4" type="video/mp4">
</video>
<p style="text-align:center; color:#808080; font-size:16px;">Figure 1. 41 seconds of real rounds: a lava trap, a dash across lava at half a heart, a comeback at 4 HP.</p>

## Thinking Slowly Loses Fights

{{< justify >}}

LLM agents in games usually get to think: a move every few seconds, sometimes minutes, in turn-based or slow worlds. PvP gives you none of that. A Minecraft server ticks 20 times a second, and a person reacts to what they see in roughly a quarter of a second. An agent that takes a second to decide is a target, not an opponent.

An agent that decides by writing, a chain of thought and then an answer, one token at a time, spends far more than a tick on every move: at 200 tokens a second, even 200 tokens of reasoning take a full second. Jev does not write. It scores every move it could make in one forward pass ([Decide Without Decoding](#decide-without-decoding)), so a decision takes about 24 ms: inside a single server tick, and about ten times faster than a person can react (Figure 2).

{{< /justify >}}

{{< image src="img/how_fast.png" alt="How fast is that" width="100%" title="Figure 2. One Jev decision in real rounds (median) against one server tick, a person's reaction to a visual cue, and an LLM that reasons before it answers (estimated: 200 reasoning tokens at 200 tokens/s, prompt processing not counted).">}}

{{< justify >}}

Block UHC is not only a sword fight, either; much of it is a bow duel. Two players trade arrows across the arena, and a full-power arrow takes about half a second to cross it, so every shot has to be aimed where the opponent *will* be. Fighters sidestep arrows they see coming, duck behind cover and peek out, hold a drawn bow while the other one hides and let go the moment they show, then close in with the sword, lava and water buckets and blocks to finish. Each of those is a split-second call.

That makes PvP a clean test of *decision latency* as a first-class capability. The model does not need to write an essay; it needs to pick the right one of about twenty moves (draw the bow, hold the draw behind cover, let go, take cover, jump the crit, zigzag in, pour lava, block the water…) many times a second, from a text description of the fight, against people who are trying hard to win.

{{< /justify >}}

{{< justify >}}

The same pressure, a world that keeps moving while the model thinks, is what makes physical AI hard; we come back to that at the end.

{{< /justify >}}

## 1,795 Rounds Against Real Players

{{< justify >}}

Players pick a mode in the lobby:

- **Block UHC**: both fighters get a sword, a bow, lava and water buckets and blocks. Damage doubles at 60 seconds, and 30 seconds later both are dropped onto a small platform ringed by lava.
- **Sumo**: a single platform and no weapons. Knock the other one off.
- **2v2**: you and a friend against two Jevs.

36 UHC arenas run side by side, and every round is recorded. We opened the server on September 26. In the first ~32 hours (September 26, 19:05 UTC to September 28, 03:29 UTC, across our two servers):

- **54 people** fought Jev in the lobby's one-on-one modes (counted by Minecraft account; the numbers below include our team's own accounts).
- **1,795 rounds** against Jev: 870 of Block UHC, which **Jev won 70%** of (the fastest anyone killed it: 1.2 seconds; the median winning round for a person: 21.5 s), and 925 of Sumo, which **people won 88%** of. Jev is a much better duelist than a sumo wrestler, and the Sumo rounds are the clearest to-do list we have.
- **1,757 recorded matches**, 10.2 hours in all: both sides every tick (position, view, movement, held item, health) and every swing, hit, arrow, block and death, 12.3 MB compressed.
- **More than 130,000 logged decisions** on our main cluster alone (our own test fights included), each with the full text state Jev read and DJev's complete distribution over the options, plus how the move turned out.

{{< /justify >}}

{{< image src="img/release_results.png" alt="Who won" width="80%" title="Figure 3. Lobby one-on-one rounds against Jev in the first 32 hours, by mode.">}}

<style>
.mcjev-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 10px 0 4px; }
.mcjev-grid video { width: 100%; height: auto; border-radius: 4px; margin: 0; display: block; }
@media (max-width: 700px) { .mcjev-grid { grid-template-columns: 1fr; } }
</style>
<div class="mcjev-grid">
  <video autoplay muted loop playsinline preload="metadata" poster="img/loop_counter_poster.jpg" src="img/loop_counter.mp4" aria-label="Jev fights from behind cover and wins"></video>
  <video autoplay muted loop playsinline preload="metadata" poster="img/loop_neon_poster.jpg" src="img/loop_neon.mp4" aria-label="Jev comes through the player's lava and wins"></video>
</div>
<p style="text-align:center; color:#808080; font-size:16px;">Figure 4. Two real Block UHC rounds, players' view (looping clips). Left: Jev fights from behind its cover, trades hits and wins. Right: the player pours lava around a pillar; Jev comes through it, sword first, and wins in 19.4 seconds.</p>

## A Deciding Head, an Even Faster Actioner

{{< justify >}}

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe AI's *System One* model: instead of writing an answer token by token, it fills in a structured answer in one parallel pass, with calibrated probabilities. Our bots are Jev-style: every move is picked by **[DJev](https://github.com/mmastrac/djev)**, served by **DJev-serve**, our own inference engine.

{{< /justify >}}

{{< image src="img/cover_mcjev.png" alt="MCJev at a glance" width="100%" title="Figure 5. MCJev at a glance. The fight is written down as text with one answer slot left masked; DJev fills the slot in a single forward pass, with zero tokens generated; the logits at the slot become a distribution over the moves offered, and Jev presses the top one. Bottom: the same recorded request on one NVIDIA B200, upstream vLLM against DJev-serve.">}}

{{< justify >}}

A Jev is two pieces:

- Its **actioner** is a bot client on the server. Every 50 ms tick it reads what a player could see and writes the fight down as text (where you are and how you move, its health and yours, the blocks around it, what is in both your hands, what just happened). It also runs the reflexes that must happen every tick no matter what: aiming, leading arrows through Minecraft's ballistics (0.99 drag and 0.05 gravity a tick, plus the measured network delay), sidestepping an arrow it can see coming, never walking into lava or off a ledge.
- Its **head** is DJev: given the text, it picks the tactic.

{{< /justify >}}

{{< image src="img/system_loop.svg" alt="The Jev loop" width="100%" title="Figure 6. The loop. The actioner (a bot client) turns the world into text and runs the per-tick reflexes; DJev scores the tactics; the chosen one becomes key presses. A gateway spreads the decisions over engines on several GPU nodes.">}}

{{< justify >}}

The split matters because a model call, however fast, is not a tick: the actioner has to keep aiming while the head thinks. It also changes *when* the head should be asked. While a bow is drawn, for instance, asking every step invites a small chance of letting go early each time, and those chances add up over a one-second draw; so the harness asks at the draw's real decision points (full power, a gap in the opponent's cover, an arrow in the air, a hit either way) and lets the actioner hold steady in between. Making decisions faster and making them at the right moments turned out to be the same project.

Robotics arrived at this structure long ago: a slower policy chooses what to do, and fast low-level controllers keep the body stable, on target and safe at their own rate. Many robot foundation models use a similar split today, a large model running at a few to tens of hertz on top of control loops running at hundreds. MCJev is a small, game-sized version of that structure in which every part can be measured.

The text the head reads went through eleven harness versions: from a flat description to a layered context (fixed rules and techniques, then the match so far, this round, the last few seconds as timed events, and "now"), with the opponent's movement and intent summarised by rules and the actioner's reflexes reported as events. Most of Jev's gains in the arena came from here, not from the model.

{{< /justify >}}

## Decide Without Decoding

{{< justify >}}

Chat models answer by writing: one token, then the next. For a decision among twenty moves that is wasted work, and on a 26B model it is far too slow for a fight where an arrow is in the air for half a second. DJev asks a diffusion language model for the answer in a different way.

DiffusionGemma pairs a causal **encoder**, which reads the prompt into a KV cache, with a bidirectional **decoder** that denoises a "canvas" (up to 256 tokens; we serve 64) conditioned on it; normal generation starts from a noised canvas and denoises it over dozens of refinement steps. A Jev-like read skips all that:

- Write an *answer scaffold* into the canvas (`1: ▒`, one slot per question) and leave only the slots noised.
- Run a single decoder pass that denoises every slot at once.
- At each slot, take the logits, keep only that question's candidate tokens (` A`, ` B`, …, or ` yes`/` no`), and renormalise.

That is the full probability distribution over the options, with no token ever sampled. This is how [DJev](https://github.com/mmastrac/djev) works: [mmastrac](https://github.com/mmastrac)'s Jev-style decision server for DiffusionGemma, built on the structured reads he added to vLLM ([#57250](https://github.com/vllm-project/vllm/pull/57250), [#58216](https://github.com/vllm-project/vllm/pull/58216)).

{{< /justify >}}

{{< image src="img/djev_readout.svg" alt="How DJev reads a decision" width="100%" title="Figure 7. One decision. The state and the question go into the prompt; the canvas holds the answer scaffold with the slot left noised; one decoder pass fills it; the logits at the slot, restricted to the options' tokens, are the distribution Jev acts on.">}}

{{< justify >}}

This is what TypeSafe calls a [*System One*](https://typesafe.ai/blog/introducing-system-one-models-and-jev) decision: fast and structured, with no reasoning tokens. DiffusionGemma used this way is very good at reading a situation and choosing (in our probes, device selection, maze safety from a small view and email triage all scored 0.95–1.0), and it will not do arithmetic for you; the harness does the numbers (distances, arrow flight times, cooldowns) and hands the model the conclusions.

{{< /justify >}}

{{< image src="img/decisions_round.png" alt="A real round" width="100%" title="Figure 8. Four seconds of a real round: Jev2 made 131 decisions in 4.4 s, a median of 23 ms each. Each bar is one decision; its height is the probability DJev gave the move it took, its colour the move. Runs of 0.99 are firm commitments (here, taking cover); the lower, varied bars are the close calls.">}}

## Inference Engine

{{< justify >}}

Three stages of serving the same model, each timed for one decision on one GPU:

| | One decision | What changed |
|---|---|---|
| Hugging Face transformers | 174 ms | the first working version |
| upstream vLLM with the structured read (DJev) | 56 ms | no generation loop; vLLM's model code and kernels |
| **DJev-serve** | **22 ms** | one forward pass per decision, captured in a CUDA graph |

(The 174 ms was measured on a short test request: a 292-token prompt with three questions and a 256-token canvas. The 56 and 22 ms are the same recorded 2.7k-character MCJev request, the same bf16 weights, one NVIDIA B200.)

**Why the first version was slow.**

- The GPU was busy for only 55 of the 174 ms.
- One forward pass launched **10,340 kernels**, mostly tiny element-wise, copy and reduce kernels from the glue around 30 layers of routing, normalisation and rotary embeddings.
- So the pass was bound by kernel launches, not by compute: batching 8 requests only moved it from 174 to 206 ms.

**What upstream vLLM still spends.** DJev on vLLM removes most of that overhead, but a read still runs through a general-purpose engine:

- the encoder prefill and the single decoder denoise run as two scheduled steps, each with its own input preparation and kernel launches;
- the read is computed over the full served canvas width.

**DJev-serve.** We built it by autoresearch: an AI coding agent ran most of the profile, change and benchmark loop, while we chose what to try next and checked every number. The engine sits on top of vLLM, keeps vLLM's model code, paged KV cache and attention kernels, drops its scheduler, and computes the same read (prefill plus denoise, the same math) as exactly one forward pass per decision:

- **One CUDA graph from token ids to answer log-probs.** Prompt lengths are bucketed, one graph per bucket. Per decision we fill two small index buffers, replay the graph, and copy back a slots × candidates array.
- **A prefix cache that fits inside a graph.** A Jev sends a long, mostly fixed preamble every step. Each request reuses the KV of the longest block-aligned prefix it shares with a recent prompt and computes only the rest; the prefix length is data in the index buffers, so the same graphs serve any prefix.
- **Faster attention where it counted.** On Blackwell, vLLM's Triton attention takes ~520 µs per sliding-window layer at 1.6k tokens, against ~130 µs for FlashAttention-2. The 25 sliding-window layers run FlashAttention-2; the 5 global layers (head size 512, which FlashAttention-2 lacks) stay on a tuned Triton kernel. This choice alone is worth about 10 ms of the gap to a Triton baseline.

The full layout, the attention details and the exact versions we benchmarked are in [serving/docs/PERFORMANCE.md](https://github.com/alexzms/MCJev/blob/main/serving/docs/PERFORMANCE.md).

**The result.**

- **~26 ms per step end to end**: 22.8 ms GPU forward, 1.0 ms parsing and tokenising, under 1 ms moving data between processes.
- On the same recorded request, upstream vLLM takes **56 ms** and DJev-serve **22 ms**: **2.5× faster** (DJev-serve on vLLM 0.30.1 nightly, commit 5840d95, with a 64-token canvas; vLLM can also run DiffusionGemma on FlashAttention 4, which we have not benchmarked).
- The bottleneck is now the GPU itself.

{{< /justify >}}

{{< image src="img/same_request.png" alt="Upstream vLLM 56 ms, DJev-serve 22 ms" width="80%" title="Figure 9. One decision, the same recorded MCJev request (2.7k characters), the same bf16 weights, one NVIDIA B200: upstream vLLM with the structured read against DJev-serve.">}}

{{< image src="img/gpu_breakdown.png" alt="GPU time breakdown" width="100%" title="Figure 10. Where the GPU time of one forward goes (a real MCJev step with 1,408 tokens left to compute after the prefix cache, kernel times summed from an eager profile; a longer request than Figure 9's, which is timed end to end): the MoE experts (128 of them, 8 per token) dominate, then the other projections and attention.">}}

{{< justify >}}

**What a decision costs now.** Once launches are gone, the cost is easy to model: about **6 ms fixed per forward** (mostly reading the mixture-of-experts weights once) plus **~10.4 ms per 1,000 tokens** that are not in the prefix cache. That turns prompt design into a systems problem. An MCJev bot's prompt is ~1,760 tokens, yet only ~480 of them hit the cache, for two reasons we would not have noticed from the model's side:

- the line that changes every step ("Now: alexzms is 3 blocks away…") sat at the end of the techniques section, so the fixed common-sense section after it could never be reused;
- each bot's name came in the first sentence ("You are Jev1…"), so no two bots shared any prefix at all.

Harness authors and serving engineers have to design the prompt together.

**Batching.** DJev-serve started by computing one request per forward, so a GPU saturated at ~43 steps a second and every extra bot just queued. Batching up to four requests in one forward:

- cuts the GPU time per request from 23.1 ms to 15.1 ms (1.53× throughput), while a single bot still sees 26 ms;
- with a 4 ms batching window, takes a GPU from 43 to 58–64 steps a second.

{{< /justify >}}

{{< image src="img/scaling_batching.png" alt="Scaling and batching" width="100%" title="Figure 11. Left: a forward costs ~6 ms plus ~10.4 ms per 1,000 uncached tokens. Right: batching requests into one forward.">}}

## Serving Many Jevs

{{< justify >}}

A public server needs many Jevs at once: we now run a pool of 24, and at busy moments many of them fight at once. Each GPU runs its own copy of DJev-serve; a small **gateway** gives the agents one address, sends each request to the healthy engine with the fewest requests in flight, checks every engine with a tiny evaluate every two seconds, and retries on the next one if an engine dies mid-request. Adding a node is one line in a file.

Because a Jev asks for its next decision only when the last one has answered, load is closed-loop: the worst case is every Jev in a fight at once. We replayed real decisions from the logs against the live gateway (8 GPUs at the time):

- a single stream gets 39 decisions a second at a 25 ms median;
- 8 streams get 324 a second at the same 25 ms;
- 24 streams flat out saturate at 362 a second with a 66 ms median.

In real rounds a Jev averages about 6 decisions a second, so 24 fighting Jevs need ~140 a second, well inside the envelope.

{{< /justify >}}

{{< image src="img/gateway_load.png" alt="Gateway load test" width="80%" title="Figure 12. Closed-loop load on the gateway with real decisions (8 GPUs): throughput (bars) and latency (line).">}}

{{< justify >}}

The last few tens of milliseconds were not in the model at all. Our first agents ran on a laptop and reached the model through a Cloudflare tunnel: **~330 ms** per decision, most of it spent crossing the internet. Moving the agents onto the GPU node brought a decision down to the model's own latency, and later we moved the Minecraft server itself onto the same cluster, behind a login proxy, so a Jev's eyes, head and actioner all live on one cluster, with no hop across the internet anywhere on the path.

That also mattered for aim: when every observation of the opponent is ~150–300 ms stale, a moving target is somewhere else by the time the arrow arrives, and no amount of ballistics fixes that. A robot faces the same choice between a model in a distant data center and a model next to its actuators; our numbers are a small data point for keeping the head close to the body when the loop is tight.

{{< /justify >}}

{{< image src="img/network_path.png" alt="Network path" width="85%" title="Figure 13. What one decision cost end to end, as the bots saw it, as the pieces moved closer together (medians in real rounds).">}}

{{< image src="img/latency_timeline.png" alt="Latency over the release" width="100%" title="Figure 14. Every decision Jev made in real rounds during the release (94k decisions of the text harnesses, 5-minute medians and p90), request to answer as the bots saw it: ~60 ms on upstream vLLM, ~24 ms after the switch to DJev-serve. These are production medians over different hours and loads, not a controlled comparison; Figure 9 is the controlled one.">}}

## Play

{{< justify >}}

Minecraft: Java Edition 1.9 or newer, a paid game (the server checks accounts, so you need a Microsoft account that owns it): **Multiplayer → Add Server → `mc.alexzms.com`**. In the lobby, right-click the compass and pick a mode, or run down the red carpet onto the iron pads. If you win, you will be on the leaderboard in the lobby.

{{< /justify >}}

{{< image src="img/sumo_dojo.jpg" alt="Sumo" width="100%" title="Figure 15. Sumo, the mode where people win 88% of rounds. Come and help us make that number go down.">}}

## What's Next

{{< justify >}}

MCJev started as a latency experiment. The release turned it into something more interesting: a place where a model plays thousands of short, scored, recorded games against real people, and where every decision it made is on disk with its outcome. We see three directions inside the game; the last section is about where they lead outside it.

**Evolve.** DJev plays with zero training today. Now that every round is scored and on disk, Jev can improve in two loops. The first leaves the model alone and evolves the harness: propose a change to what Jev reads or is offered, let it play a few hundred rounds, keep it if it wins more. The second loop is the weights. Every round is training data: the winner's decisions, human or Jev, with the state each was made in and how the round ended. We start with supervised fine-tuning (SFT) on the side that won, then move to reinforcement learning (RL) with the round's outcome as the reward. The logs suit RL unusually well: each decision stores DJev's full distribution over the options, not just the move it took, which is exactly what off-policy correction needs; and since a move's probability comes from one forward pass rather than a sampled sequence of tokens, the policy gradient is as simple as it gets. Each new model goes straight back into the arena against the same people. The loop we want is a model that gets measurably better from its own public games.

**Harness.** Most of Jev's gains came from the harness, the text it reads and the options it is offered, not from the model, and eleven hand-written versions taught us what that costs. We want harness design to become searchable: versions judged on a varied suite of fights rather than one opponent, context written so the fixed parts come first and the prefix cache can do its job, and eventually a Jev that sees the screen instead of a description of it, which is also the step most likely to matter beyond Minecraft.

**Arena: agents that evolve together.** People are the best opponents we have, but there are only so many of them, and they sleep. The arena can host many agents at once: Jevs with different harnesses, different checkpoints, eventually different models, playing each other around the clock (our pool of 24 alone can play thousands of rounds a day). We want them to co-evolve as a population rather than one model chasing its own reflection: each agent trains against the current field, past versions stay in the league so a new strategy cannot win by forgetting how to beat an old one, and ratings across the whole league tell us whether the field is getting stronger or just going in circles. Human rounds stay in the mix as the anchor, so the league cannot drift away from how people actually play. Teams fit the same picture: 2v2 already runs with two Jevs sharing what they see and choose, and in a co-evolving population, coordination becomes something partners learn together instead of something we script.

Underneath all three, the systems work continues: batching across Jevs on one GPU, lower-precision weights and newer attention kernels to push a decision well under 20 ms, and prompts designed for the cache.

{{< /justify >}}

## Toward Physical AI

{{< justify >}}

Minecraft was never the end goal. A world that keeps moving while the model thinks is also a central constraint of physical AI: a robot arm picking parts off a conveyor cannot pause the belt while it decides. We do not know yet how physical AI is best built, and a game is not a robot, but MCJev let us practise one slice of that setting, end to end and in front of real opponents.

{{< /justify >}}

{{< image src="img/robot_dojo.svg" alt="From a Minecraft dojo to a robot dojo" width="100%" title="Figure 16. From a Minecraft dojo to a robot dojo. Right: an imagined dynamic manipulation task, where the conveyor speeds up while the arm is reaching, so where to grasp has to be re-decided on the move.">}}

{{< justify >}}

Some of what we learned may carry over: deciding in one forward pass, a slower head over a fast actioner, a latency budget that runs from kernels to the network, and every decision logged with its outcome. Which of these hold up on a robot, for example on manipulation benchmarks such as [RoboDojo](https://robodojo-benchmark.com/) or in scenes that change while the arm moves, is what we want to explore next.

{{< /justify >}}

## Acknowledgement

{{< justify >}}

We especially thank **NVIDIA** for providing the **B200** GPUs that MCJev runs on. A pool of Jevs fighting the public around the clock, each asking for a decision every few tens of milliseconds, is only possible on that hardware. We thank TypeSafe AI, whose [Jev and System One models](https://typesafe.ai/blog/introducing-system-one-models-and-jev) this work takes its idea and its name from; Google for releasing DiffusionGemma with open weights; the vLLM project that DJev-serve builds on; [mmastrac](https://github.com/mmastrac), for [DJev](https://github.com/mmastrac/djev) and the structured reads in vLLM it runs on ([#57250](https://github.com/vllm-project/vllm/pull/57250), [#58216](https://github.com/vllm-project/vllm/pull/58216)); and everyone who logged in and fought Jev.

{{< /justify >}}

## Contributors

[Minshen (Alex) Zhang](https://alexzms.github.io/), [Junda Chen](https://github.com/GindaChen), [Yuanbo Yang](https://freemty.github.io/), [Yuxuan Zhang](https://yuxuandexter.github.io/), [Yi Sun](https://www.linkedin.com/in/yi-sun-mlsys), [Shaoxiong Duan](https://github.com/shaoxiongduan), [Lanxiang Hu](https://snyhlxde1.github.io/), [Jiaqi Leng](https://jacky-leng.github.io/), [Yulun Wu](https://memset0.github.io), [Will Lin](https://github.com/SolitaryThinker), [Hao Zhang](https://haozhang.ai)

**Cite us:**

```bibtex
@misc{mcjev2026,
  title        = {{MCJev}: A 24 ms {Jev} for {Minecraft} {PvP}},
  author       = {Zhang, Minshen and Chen, Junda and Yang, Yuanbo and Zhang, Yuxuan and Sun, Yi and Duan, Shaoxiong and Hu, Lanxiang and Leng, Jiaqi and Wu, Yulun and Lin, Will and Zhang, Hao},
  year         = {2026},
  month        = sep,
  howpublished = {Hao AI Lab blog},
  url          = {https://haoailab.com/blogs/mcjev/}
}
```
