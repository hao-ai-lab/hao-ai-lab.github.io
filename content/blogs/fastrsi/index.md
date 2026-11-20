+++
title = "Towards Live-Editable LLM Infra for RSI"
date = 2026-11-20T00:00:00-08:00
authors = ["Junda Chen", "Yi Sun", "Yuxuan Zhang", "Yuanbo Yang", "Junda Su", "Will Lin", "Hao Zhang"]
author = "Junda Chen, Yi Sun, Yuxuan Zhang, Yuanbo Yang, Junda Su, Will Lin, Hao Zhang"
ShowReadingTime = true
authorOnNewLine = true
draft = false
math = true
contentClass = "post-content-justified"
summary = "In recursive self-improvement (RSI), coding agents spend 83% to 90% of wall-clock time waiting on cold restarts and full benchmarks. We introduce FastRSI, a live-editable framework for GPU engines that cuts vLLM edit testing from 228 s to 2.4 s (95x), generalizes across six engines (vLLM, SGLang, Megatron-LM, FastVideo, veRL, and Miles) with 11/11 conformance, and delivers up to 1.52x bit-exact speedups on FastVideo and +7.1% throughput on Megatron-LM with zero cold restarts."

[cover]
  image = "img/cover.png"
  relative = true
  alt = "Towards Live-Editable LLM Infra for RSI"
  caption = "One edit, two test loops: 228 s cold restart vs. 2.4 s FastRSI live test."
  hidden = true

[socialIcons]
  [[socialIcons.icon]]
    name = "twitter"
    url = "https://twitter.com/haoailab"
+++

<div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 4px 0 18px;">
  <a href="https://github.com/hao-ai-lab/fastrsi" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/GitHub-FastRSI-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub: FastRSI" style="height: 28px; margin: 0;">
  </a>
  <a href="https://x.com/haoailab" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Follow-@haoailab-000000?style=for-the-badge&logo=x&logoColor=white" alt="X: @haoailab" style="height: 28px; margin: 0;">
  </a>
</div>

{{< image src="img/cover.png" alt="One edit, two test loops: cold restart vs. FastRSI live test" width="100%" title="Cover Figure. One edit, two test loops. Top (baseline): edit → cold restart → full benchmark (228 s). Bottom (FastRSI): edit → hot reload → targeted probe (2.4 s, 95x faster)." >}}

**TL;DR:** In a recursive self-improvement (RSI) loop for AI systems, coding agents optimize the underlying infrastructure itself by iteratively editing an engine and testing each change against correctness and performance benchmarks. While an agent can write an edit in seconds, testing that edit on the engine takes minutes, leaving **83% to 90%** of every optimization run spent waiting on evaluation.

Why does testing an edit take so long? **Because testing a code change today requires restarting the engine from scratch, and cold-starting a stateful GPU engine carries massive startup overhead.** Even when an agent makes a one-line edit, the engine must tear down and rebuild its entire runtime state before running a full benchmark. As a result, **testing an edit pays the engine's full cold-start overhead, rather than the cost of what the edit actually touches.**

We introduce **FastRSI**, a framework that makes GPU engines live-editable. Instead of restarting the process, FastRSI keeps untouched GPU state warm, hot-reloads only what the edit affects, screens the change with a fast targeted probe, and resets the engine to a clean state for the next test. We show that FastRSI can cut the time to test a vLLM edit from **228 s to 2.4 s (95x)** and reach the same optimization target **1.4–2.3x sooner** in end-to-end RSI loops. We generalize this approach across six GPU systems spanning inference, training, video generation, and RL: vLLM, SGLang, Megatron-LM, FastVideo, veRL, and Miles.

---

## Evaluation Is the Bottleneck of RSI

Recursive self-improvement (RSI) for AI systems uses coding agents to optimize the GPU infrastructure that runs them, from request schedulers to custom GPU kernels. During an optimization run, the loop repeats two very different phases over and over for hours. In the first phase, a coding agent inspects the codebase and writes an edit. In the second phase, the system tests that edit on the target engine: it loads the modified code, runs a benchmark, and measures whether the change is correct and whether it runs faster. Based on those results, the agent keeps or discards the edit and moves on to its next idea.

These two phases are improving at very different speeds. Coding models generate tokens faster with every new release, steadily shrinking the time needed to write an edit. Testing that edit, however, depends on the system under test. Starting up and benchmarking a distributed GPU engine like vLLM takes just as long today as it did before coding agents existed.

As a result, self-improving agents spend almost all of their time waiting. Across our agent optimization runs, **83% to 90% of total wall-clock time is spent waiting on evaluation**.

{{< image src="img/fig1-eval-bottleneck.png" alt="Wall-clock breakdown of an agent optimization run" width="100%" title="Figure 1. Wall-clock breakdown of an agent optimization run: time the agent spends writing edits vs. time spent waiting on evaluation (83% to 90%). In a trace replay of a 5.9-hour vLLM optimization run, speeding up model token generation 30x to 1,500 tok/s trims total runtime by only 6% (to 5.53 hours) because evaluation dominates." >}}

This imbalance creates a tight Amdahl's Law bottleneck. In a trace replay of a recorded 5.9-hour vLLM optimization run with a fixed sequence of agent edits, accelerating coding-model token generation 30x to `1,500 tok/s` trims total wall-clock time by only 6% (from 5.9 hours to 5.53 hours), because 83% of the run is spent waiting on engine restarts and benchmarks. When 83% to 90% of wall-clock time goes to testing edits on the engine and only 10% to 17% goes to the agent writing code, making the coding model faster barely shrinks total run time. Even if a future model could generate every edit instantly, the end-to-end loop would speed up by only a small fraction because nearly the entire run is still spent restarting the engine and running benchmarks. To make RSI meaningfully faster, we have to make testing an edit faster.

---

## Why Is Testing an Edit So Slow?

### What happens when an agent changes one line

Take vLLM as an example, running `Nemotron 3.5 Super NVFP4` on 1× NVIDIA GB200 with warm on-disk caches, and suppose the agent changes one line in the request scheduler. To test this edit today, the testing setup performs a **cold restart** (killing the engine process and starting it again from scratch), forcing the engine through five startup stages before the first request can run:

1. **Load weights** from disk into GPU memory.
2. **Initialize NCCL** groups for communication across GPUs.
3. **Autotune kernels** for the model and the hardware.
4. **Allocate the KV cache.**
5. **Capture CUDA graphs**, one for each shape the engine will run.

Only after all five stages finish does the benchmark start. Testing that single edit takes **228 s**.

{{< image src="img/fig2-vllm-ladder.png" alt="Timeline and cost ladder of one cold test in vLLM vs. FastRSI" width="100%" title="Figure 2. Cost ladder of testing a one-line vLLM scheduler edit on 1× GB200 (warm on-disk caches). Of the 228 s baseline check, 204.4 s (nearly 90%) is cold-start setup that the scheduler edit does not invalidate." >}}

### Paying for the engine, not for the edit

None of the five stages above depends on the changed line in the scheduler. The weights are the same, the NCCL groups are the same, the tuned kernels are the same, the KV cache is the same, and the CUDA graphs are the same. Yet a cold restart tears down and rebuilds all of this state simply because the engine has no way to load modified code in place.

This is the core mismatch. Most edits in an iterative RSI loop are short and local: a scheduler rule, a few lines of one kernel, or a helper function. Yet because every test pays the engine's full cold-start overhead rather than the cost of what the edit touches, a one-line edit and a full rewrite of the engine both cost the same 228 s.

Testing an edit carries a second unnecessary cost. After the restart finishes, every edit runs the full benchmark, even though most edits have bugs or hurt performance and could be screened out in seconds by a **targeted probe** (a short test that exercises the edited code before the full benchmark runs).

Removing these two costs in turn changes the time required to test our one-line scheduler edit:

| How the edit is tested | Time per test | Speedup |
| --- | --- | --- |
| Cold restart, then the full benchmark | 228 s | 1.0x |
| Hot reload into a warm engine, then the full benchmark | 23.6 s | 9.6x |
| Hot reload into a warm engine, then a targeted probe | 2.4 s (4.8 s with logprob check kept) | 95x (48x with logprob check kept) |

The first jump shows where most of the time goes: **204.4 s of the original 228 s (nearly 90%) is pure cold-start overhead** before the benchmark even begins. Using **hot reload** to load the edit in place inside a **warm engine** (a running engine whose weights, NCCL groups, tuned kernels, KV cache, and CUDA graphs are already in GPU memory) drops the test time on the exact same full check to **23.6 s (9.6x)**. Skipping the redundant logprob check reduces that to **14.1 s**, and screening the edit first with a small-batch **targeted probe** cuts the test time down to **2.4 s (95x)**, or **4.8 s (48x)** when the logprob check is kept.

---

## Live Editing: Pay for the Edit, Not for the Engine

The idea is simple. Web developers have been live-editing their code for over a decade: they save a file, and the running page updates in place while the application keeps its state. We want the same for GPU engines. **Testing an edit should cost only what the edit affects, and everything else should stay warm.**

At first glance, this looks like a matter of keeping the process alive or calling Python's `importlib.reload()` (or notebook `%autoreload`). In a distributed GPU engine like vLLM, naive module reloading silently fails: existing worker instances keep stale function and method references, compiled Triton cubin binaries ignore edits to their Python source text, and C++ CUDA graphs continue replaying the old kernel addresses captured at startup. Making a GPU engine truly live-editable is hard for three reasons:

1. **State is expensive and spread across the engine.** The state worth keeping is built by different stages of startup and lives in different places across the runtime, distributed worker processes, and GPU memory.
2. **Code is baked into state.** A CUDA graph is a recording of the kernels that ran when it was captured, and a `@triton.jit` kernel is compiled and cached as a binary. If a kernel changes, the captured graph and cached binary still run the old code.
3. **Most edits are bad.** An edit that crashes or hangs a GPU kernel can take the warm engine down with it, forcing the loop back into a cold restart.

FastRSI addresses these three problems in turn.

{{< image src="img/fig3-architecture.png" alt="FastRSI architecture: warm GPU state, boundary-tiered hot reload, execution witnesses, and recovery" width="100%" title="Figure 3. FastRSI architecture: untouched state stays warm in GPU memory while edits commit at the step gap, route to the cheapest reload tier that covers their state boundary, prove execution via witnesses, and recover via millisecond code rollback or second-scale warm worker respawn." >}}

### 3.1 Keep the untouched state warm

We start by sampling a few evaluation runs and tracing where startup time goes. The trace points to the five stages listed earlier: weight loading, NCCL initialization, kernel autotuning, KV cache allocation, and CUDA graph capture. We factor these out of the restart path one by one, so that each stays warm in GPU memory while the code around it changes.

Keeping this state warm accounts for the first jump in our table (from 228 s to 23.6 s): by avoiding the 204.4 s cold-start rebuild, a scheduler edit is tested on a warm engine in 23.6 s even when running the full benchmark.

### 3.2 Hot-reload what the edit affects

Keeping startup state warm is not enough on its own. If the warm components are frozen, the agent can edit only Python glue code that does not touch them, and that rules out Triton kernels, CUDA graphs, and stateful constructors, where much of the performance lies.

Rather than relying on fragile monkeypatches that silently leave long-lived state out of sync, FastRSI routes each edit along a **boundary-tiered reload ladder**. While the engine is running, FastRSI parses the diff and maps it to the smallest state boundary that the edit invalidates. In the quiet gap between two inference or training steps (taking only `0.5–1.2 ms` inside the vLLM engine loop), a two-phase coordinator commits the change across all GPU ranks at once:

| Reload Boundary | Mechanism | What Edits It Handles | Measured Reload Cost |
| --- | --- | --- | --- |
| **In-place Python swap** | Atomic code-object and method rebinding across all ranks with journaled undo | Pure Python functions, schedulers, and method bodies | **0.05–0.06 s** (undo **0.3–1.6 ms**) |
| **In-place Triton rebuild** | Invalidates `@triton.jit` caches and dependent callers; recompiles in place | Triton GPU kernel bodies | **0.07–0.3 s** (undo **< 1 ms**, zero recompile) |
| **Selective CUDA graph recapture** | Rebinds the op and recaptures only the affected batch or layer graphs in the live worker | Edits inside CUDA-graph-captured regions | **0.41–5.6 s** |
| **In-process module / constructor re-init** | Re-imports the module or re-runs `__init__` for newly added fields while keeping GPU allocations | Constructor field additions, module constants, and local objects | **0.19–1.41 s** (undo **1.58 ms**) |
| **Warm Zygote fork + CUDA IPC weights** | Forks from a pre-imported Zygote server and maps weights over CUDA IPC without disk reads | Launch flags, startup-only code, or edits refused by in-place tiers | **4.78–16.5 s** (`7.37 s` on vLLM Qwen3-8B) |
| **Full cold restart** | New process from scratch with warm on-disk caches | Parallelism topology changes or native C++ builds | **228 s** (vLLM baseline) |

Take a GPU kernel edit. Without FastRSI, an agent that edits a Triton kernel has to cold-restart vLLM. With FastRSI, the kernel service instead does three things in place:

1. **Recompile** only the edited `@triton.jit` kernel and invalidate its callers' autotuner caches.
2. **Rebind** the operator in place so that the running engine calls the newly compiled binary.
3. **Recapture** only the CUDA graphs that the edited kernel affects, leaving all other graphs and GPU state intact.

When an edit touches unsupported startup state or native C++ extensions that cannot be safely patched in place, FastRSI refuses the in-place swap in `0.07–0.60 s` with an explicit reason code and names the cheapest warm restart boundary that can run it. In our manual audit of 80 real vLLM performance pull requests sampled from 487 candidates, **`63/80 (79%)` avoid a full cold restart** (20% via in-place Python swap, 51% via Triton or CUDA graph reload, and 8% via buffer or component rebuild). Static AST coverage scans across all six engines show that **49.3% to 84.3%** of Python functions qualify for live in-process reloading under our checked rules.

### 3.3 Prove the edit ran and recover when it fails

Even when an edit installs cleanly, a subtle danger remains: **what if the modified code never actually executed?** Suppose an agent edits a file, runs a probe, and sees that the output matches the baseline with zero errors. Did the optimization preserve equivalence, or did the workload never reach that branch? To eliminate silent false positives, FastRSI attaches an **execution witness** to every modified function, counting invocations across both eager execution and CUDA graph captures on every GPU rank. If any rank records zero calls after the probe finishes, FastRSI marks the test as `zero_witness` (`proven: false`) rather than reporting a false pass.

Conversely, in an RSI loop, most candidate edits have bugs, hurt performance, or crash a GPU kernel. FastRSI recovers at two levels and uses the cheaper one whenever it can:

- **Journaled code rollback, in milliseconds.** Every install records its exact inverse in an append-only rollback journal. If an edit raises an exception, fails a witness check, or regresses performance while leaving the GPU workers healthy, `uninstall()` restores the original `__code__` pointers, compiled Triton binaries, and CUDA graph handles in **0.3–1.6 ms** without recompiling, and verifies that every function's fingerprint matches the base state.
- **Partial warm restart, in seconds.** If a bad kernel crashes, hangs, or desynchronizes a GPU worker, FastRSI marks the session as `poisoned` and refuses further in-place installs until it respawns the affected worker group from a pre-imported **Zygote** process and reattaches weights held in GPU memory via **CUDA IPC** (taking **4.78–7.37 s** on FastVideo and vLLM instead of minutes for a cold disk reload).

### 3.4 Putting it together: testing one edit with FastRSI

With these pieces in place, one iteration of the FastRSI loop runs through five steps:

1. **Write:** The agent writes an edit.
2. **Hot-reload (`install`):** FastRSI stages the diff while the engine runs and commits it at the next step gap, updating only what the edit affects.
3. **Probe (`check`):** A targeted probe exercises the edited path in seconds and verifies that the execution witness fired on every GPU rank.
4. **Recover or promote (`uninstall`):** If the edit fails the probe, FastRSI rolls it back in milliseconds or warm-restarts the worker in seconds. If the edit passes the targeted probe, it is promoted to a full benchmark and verified in a fresh cold process before being kept.
5. **Clean state (`reset`):** FastRSI drains in-flight work and restores volatile caches, KV/activation buffers, RNG states, and data iterators (plus coupled model/optimizer `S0` snapshots in training and RL), so the next test starts from a clean, reproducible baseline.

---

## Generalizing Beyond vLLM

Nothing in this approach is specific to vLLM. However, distributed GPU systems differ widely in their process trees, step boundaries, and state holders: vLLM and SGLang run asynchronous inference schedulers with CUDA graphs, Megatron-LM runs synchronous tensor- and sequence-parallel training steps, FastVideo drives multi-step diffusion denoising workers, and veRL and Miles orchestrate multi-actor Ray reinforcement learning loops across colocated trainers and rollout servers.

### The three-stage workflow from vLLM to five other engines

To generalize FastRSI across all six systems without building a bloated abstraction layer, we followed a disciplined three-stage workflow:

1. **Start engine-local (Offline Toolsmith).** For each new engine, we first profile its startup cost ladder, map its process tree and state holders, and mine past performance PRs for edit hot spots. We then bootstrap `repos/<engine>/livelab/` from our starter scaffold (`template/livelab/`) and implement the five-call `livelab` interface (`install`, `check`, `reset`, `uninstall`, `status`) specialized to that engine's step boundary, with zero cross-engine imports.
2. **Distill shared primitives between waves (Rule of Two).** We never invent generic base classes upfront. Only after at least two engines independently solve the same concrete problem on real upstream PRs do we extract the pure mechanism into the zero-dependency, standard-library-only `core/livelab/` runtime (`12,673` lines across `39` files, including the `10,183`-line transaction kernel). This distillation produced our shared two-phase step-gap coordinator, execution witnesses, rollback journal, pre-fork CUDA guards (`forksafe.py`), and `ray_participants.py` (`182` lines), which solves a classic Ray pitfall where `@ray.remote` workers deserialize separate copies of module globals.
3. **Certify on real GPUs with 11 conformance cases (`c0`–`c10`).** Every engine integration must pass the same 11-case fault-injection conformance suite (`core/livelab/conformance/`) on real GPUs before running agent loops: workload repeatability (`c0`), no-op edit (`c1`), visible edit with rank witnesses (`c2`), bit-exact equivalence against a fresh cold process (`c3`), exact undo (`c4`), mid-install fault rollback across ranks (`c5`), refusal of unsupported edits (`c6`), chained edits (`c7`), crash/hang/tamper poisoning and recovery (`c8`), a 20-cycle install/undo soak test (`c9`), and `zero_witness` detection on unreached code (`c10`).

### Brief Engine-by-Engine Summary

Because the transaction kernel, witnesses, rollback journal, pre-fork guards, and Ray actor helpers are shared in `core/livelab/`, onboarding later engines required only a few hundred lines of adapter code and zero patches to upstream engine source files:

| Engine | Workload Domain | Adapter Non-Test LOC | Upstream Engine Patches | GPU Conformance | Key Live Reload & Reset Measurements |
| --- | --- | --- | --- | --- | --- |
| **vLLM** | LLM serving | 7,848 (adapter + boundary services) | 0 | **11/11** (1× B200) | `110 ms` install / `74 ms` undo; `0.3 s` Triton rebuild; `0.41 s` single-bucket graph recapture; `7.37 s` Zygote+IPC warm restart |
| **FastVideo** | Video diffusion | 2,206 (adapter + rebuild/zygote) | 0 | **11/11** (2× GB200) | `60 ms` install (`0.72 s` edit-to-run); `0.19–1.27 s` in-process module rebuild; `4.78–5.62 s` Zygote+IPC respawn (vs. `~72 s` warm / `~507 s` cold) |
| **Megatron-LM** | Distributed pretraining | 2,465 (adapter + `S0` restore/zygote) | 0 | **11/11** (2× GB200 TP=2) | `50 ms` install; `0.21–0.27 s` model/optimizer `S0` restore (vs. `13.69 s` rebuild and `57.5 s` full restart) |
| **SGLang** | LLM serving | 784 (`534` adapter + `250` zygote) | 0 | **11/11** (2× GB200 TP=2) | `91 ms` install / `8 ms` undo; `0.84 s` decode CUDA graph recapture vs. `24.97 s` cold load (**29.8x**) |
| **veRL** | Multi-actor RL (GRPO) | 827 (`595` adapter + `232` probe CLI) | 0 | **11/11** (4× GB200, 13 procs) | `9.22 s` live advantage swap (**24.3x** vs. `224.05 s` boot; `1,022 s` on NFS); `2.46 s` coupled 4-GPU trainer+rollout `S0` restore |
| **Miles** | Multi-actor RL | 337 (reusing `ray_participants.py`) | 0 | **11/11** (2× GB200) | `69 ms` multi-actor live swap; `EpochDataBuffer` `code_epoch` rollout queue drain; coupled `S0` weight sync |

- **vLLM:** Hooks into unmodified vLLM at runtime via a worker extension class and an engine-core step-gap hook. After distilling our 45k-line initial prototype down to `7,848` lines of boundary-tiered services atop `core/livelab/`, the vLLM lab passes `11/11` conformance on 1× B200 (`110 ms` install, `74 ms` undo) and replays 19 real upstream PRs with zero full restarts (8 live in place and 11 via forked warm restarts).
- **FastVideo:** Intercepts `Executor.get_class` at runtime so diffusion workers attach to the transaction coordinator without modifying FastVideo source. Passes `11/11` conformance on 2× GB200 with `60 ms` live swaps, `0.19–1.27 s` in-process module rebuilds for constructor edits, and `4.78–5.62 s` Zygote+IPC respawns.
- **Megatron-LM:** Wraps `pretrain()` around the training step gap without modifying `megatron-core`. Passes `11/11` conformance on 2× GB200 (TP=2) with `50 ms` code swaps and `0.21–0.27 s` `S0` state restoration of weights, optimizer states, and RNG across ranks, compared to `57.5 s` for a full restart.
- **SGLang:** Required only `784` non-test Python lines (`534` adapter + `250` zygote) with zero upstream patches, hooking into `Scheduler` event-loop batch gaps and `CudaGraphRunner`. Passes `11/11` conformance on 2× GB200 (TP=2), cutting decode CUDA graph recapture to `0.84 s` versus `24.97 s` for a cold load (**29.8x**).
- **veRL:** Required `827` non-test lines reusing `core/livelab/ray_participants.py` with zero upstream patches. On 4× GB200 across **13 warm Ray processes** (trainer, 4× vLLM rollout engines, and 8× reward workers), it cuts live advantage function testing from `224.05 s` (`1,022 s` on shared NFS) to `9.22 s` (**24.3x**), live reward swaps to `24.91 s` (**9.0x**), and synchronizes coupled trainer and vLLM rollout weights to a shared `S0` checkpoint in `2.46 s`.
- **Miles:** Onboarded in just **`337` non-test lines** by reusing `ray_participants.py` with zero upstream patches. Passes `11/11` conformance on 2× GB200 with `69 ms` multi-actor swaps and uses an `EpochDataBuffer` `code_epoch` drain so stale asynchronous rollout samples generated by old code are automatically discarded after a live edit.

---

## Results and Case Studies

### 5.1 Single-Edit Testing Speedup Across Six Engines

We evaluated FastRSI on NVIDIA GB200 and B200 GPUs across all six engines, using `GPT-6 Astra`, `Claude`, and `Codex` as coding agents under 6-to-8-hour run budgets for end-to-end optimization loops. Across every engine, replacing cold restarts with boundary-tiered hot reloading and targeted probes cuts the edit-to-result feedback loop from minutes down to seconds or milliseconds:

{{< image src="img/fig4-cross-engine.png" alt="Single-edit testing latency across six GPU engines: cold restart baseline vs. FastRSI" width="100%" title="Figure 4. Single-edit testing time across six GPU systems on NVIDIA GB200 and B200 GPUs, comparing standard cold/warm restarts against FastRSI live hot reload and targeted probing." >}}

| Engine & Representative Edit | Baseline Restart Time | FastRSI Live Test Time | Speedup | Hardware & Workload Setup |
| --- | --- | --- | --- | --- |
| **vLLM** (1-line scheduler edit + targeted probe) | 228 s | **2.4 s** (`4.8 s` with logprob) | **95x** (`48x`) | 1× GB200, Nemotron 3.5 Super NVFP4 |
| **vLLM** (Qwen3-8B sampler/scheduler live edit) | 62.98–63.37 s | **1.07–1.18 s** | **54x–59x** | 1× GB200, Qwen3-8B TP=1, CUDA graphs |
| **vLLM** (Triton kernel bugfix, PR `#62046`) | 104.96 s | **2.72 s** (`0.3 s` install) | **38.6x** | 1× GB200, Qwen3.6-27B, `torch.compile` + graphs |
| **vLLM** (`Scheduler.__init__` new field, PR `#62872`) | 42.93 s | **1.41 s** (undo `1.58 ms`) | **30.4x** | 1× B200, Qwen3-8B, CUDA graphs |
| **vLLM** (selective CUDA graph recapture) | 35.8 s | **0.41 s** | **87x** | 1× B200, Qwen3-8B single bucket |
| **SGLang** (decode CUDA graph recapture) | 24.97 s | **0.84 s** (`91 ms` swap) | **29.8x** | 2× GB200 TP=2, Qwen3-0.6B |
| **Megatron-LM** (live swap + 3-step probe; `S0` restore) | 57.5 s (`13.69 s` rebuild) | **0.50 s** (`0.05 s` install + `0.45 s` 3-step probe); **0.21–0.27 s** `S0` restore | **115x** (probe vs. `57.5 s`); **50.7x** (`S0` vs. `13.69 s` rebuild) | 2× GB200 TP=2 |
| **FastVideo** (MiniMax H3 attention edit + 1 video check) | ~72 s warm (~507 s cold) | **6.08–6.7 s** (`0.72 s` edit-to-run) | **11.2x** warm (**75x** cold) | 1–2× GB200, MiniMax H3 |
| **veRL** (13-process GRPO live advantage swap) | 224.05 s (`1,022 s` NFS) | **9.22 s** (`2.46 s` coupled `S0`) | **24.3x** (**111x** NFS) | 4× GB200, Qwen3-0.6B GSM8K GRPO |
| **Miles** (multi-actor RL live swap + epoch drain) | 60–95 s | **0.069 s** (`69 ms` install) | **>800x** (install) | 2× GB200, Miles 0.2.0 |

### 5.2 End-to-End Autonomous Optimization Results

In end-to-end RSI loops, agents equipped with FastRSI reach the same target system performance **1.4–2.3x sooner** in wall-clock time than baseline runs that cold-restart on every edit, and discover real multi-patch optimizations with **zero cold restarts** during the climb.

This 1.4–2.3x end-to-end speedup directly reflects the Amdahl's Law bottleneck from Section 1. On runs where 83% to 90% of wall-clock time goes to testing edits, a faster coding model hits a tight ceiling because startup and benchmarking still dominate the loop. FastRSI removes that bottleneck directly, reaching the target 1.4–2.3x sooner with the exact same coding model. Why does a 95x speedup per probe test translate to 1.4–2.3x across a full multi-hour run? Once hot-reloading removes the 204.4 s cold-start wait and targeted probing cuts the remaining 23.6 s test down to 2.4 s, the rest of the run goes to the agent thinking and writing code, full benchmarks on promising edits that pass the probe, and occasional partial warm restarts after kernel crashes.

To ensure that performance gains measured inside a long-running live engine are never artifacts of warm caches or state drift, **every final speedup we report is re-measured from scratch in fresh cold processes**, alternating base and patched runs on the exact same GPU placement:

{{< image src="img/fig5-hillclimb.png" alt="End-to-end autonomous optimization trajectories on FastVideo MiniMax H3 and Megatron-LM Qwen3" width="100%" title="Figure 5. End-to-end autonomous optimization trajectories with zero cold restarts during the climb, verified in fresh cold processes: 1.45x (1 GPU) and 1.52x (2 GPUs, SP=2) bit-exact video generation speedup on FastVideo MiniMax H3 across 19 kept patches, and +5.1% (Qwen3-14B TP=2) to +7.1% (Qwen3-32B TP=4) training throughput on Megatron-LM across 82 iterations." >}}

- **FastVideo (MiniMax H3, 1.45x on 1× GB200 and 1.52x on 2× GB200 SP=2, bit-exact):** Across an overnight autonomous climb on 1× GB200 followed by a 2-GPU sequence-parallel climb (`SP=2`), `Claude` and `Codex` tested over 230 candidate edits and kept **19 patches** (fusing Triton QK-norm/RoPE reductions, gated merge and untile kernels, Ulysses sequence-parallel remote-only all-to-all packing, and per-layer NVFP4 CUDA graph replay). Every kept edit required witness verification, at least a 1% latency reduction over the rolling baseline, and bit-identical video frames and audio (`sha256` match). When re-measured in **fresh cold processes** without the live lab (1 cold + 5 warm videos per run, alternating base and patched twice), the 19-patch stack cut per-video generation latency from **`6.10/6.12 s` to `4.20/4.23 s` (1.45x)** on 1× GB200 and from **`4.41/4.42 s` to `2.93/2.88 s` (1.52x)** on 2× GB200 (`SP=2`), with every video and audio stream bit-identical to the unpatched baseline. As an operational caveat uncovered by our multi-shape gate, patches 15–17 cache per-layer CUDA graphs assuming a fixed sequence length; switching from a 124-frame request to a 243-frame request in the same process exhausts GPU memory (`74.9 GiB` held in private graph pools), so mixed-resolution deployment requires an eviction budget flag.
- **Megatron-LM (Qwen3-14B & Qwen3-32B, +5.1% and +7.1% tokens/s/GPU across 82 zero-restart iterations):** On 2× GB200 (`TP=2`), `Codex` ran **82 consecutive hill-climb iterations** on the Qwen3-14B architecture with **zero restarts between edits**. Each candidate edit was hot-swapped in `50 ms`, restored to the exact `S0` weight/optimizer/RNG state in `0.21–0.27 s`, and gated on keeping training loss within `2.2e-3` and gradient norm within `3.5e-3` of the baseline. Re-measured in fresh processes across 4 paired alternating rounds, the three kept edits delivered **`+5.1%` tokens/s/GPU on Qwen3-14B (`TP=2`, `6,973–7,360` to `7,337–7,872` tok/s/GPU)** and transferred to **`+7.1%` tokens/s/GPU on Qwen3-32B (`TP=4`, `2,306–2,389` to `2,475–2,521` tok/s/GPU)**. Two of the three kept climb edits (iterations 11 and 61) are clean source patches to `megatron-core` (`+1.24%` and `+2.97%` standalone in 4-pair fresh-process medians), while the third (iteration 12, `fuse_wgrad_accumulation`) toggled an attribute on live Transformer Engine modules in a private runner outside the public adapter; a third clean source patch from the same campaign (`residual-plain-add-without-bias-dropout`, packaged alongside iterations 11 and 61 in `repos/megatron/patches/`) delivers `+5.0%` on 14B and `+6.0%` on 32B.

### 5.3 Case Studies: Real Edits Hot-Reloaded and Optimized

To make concrete how FastRSI handles real production code across Python, Triton kernels, diffusion attention, and distributed autograd, we highlight four short diffs from our evaluation.

#### Case 1: Catching a silent wrong-file edit in vLLM (`zero_witness`, `63 s → 1.07 s`)

Anyone who has hacked on vLLM has hit this trap: vLLM contains two similarly named sampler files, `vllm/v1/sample/sampler.py` (an older entry point) and `vllm/v1/worker/gpu/sample/sampler.py` (the path actually executed by the V2 GPU worker). Suppose an agent edits the first file to mask out the top-1 logit and pick the second-best token:

```diff
# Wrong file (unreached on V2 GPU runner): vllm/v1/sample/sampler.py
-        return logits.argmax(dim=-1).view(-1)
+        return logits.topk(2, dim=-1).indices[:, 1].view(-1)

# Active file: vllm/v1/worker/gpu/sample/sampler.py
+        processed_logits = processed_logits.scatter(-1, processed_logits.argmax(-1, keepdim=True), float("-inf"))
```

With a traditional workflow on 1× GB200 (`Qwen3-8B`, 8 prompts × 32 tokens), editing the wrong file costs **`63.08 s`** for a cold `vllm serve` restart and silently returns the unchanged baseline output hash (`db291a3a797916ec`) with zero warnings, tricking an agent into thinking its edit had no effect or introduced no regression. FastRSI hot-swaps and checks the edit in **`1.07 s`** (`30 ms` install, `0.5 ms` engine-loop pause) and immediately flags `proven: false, findings: ["zero_witness"] (0 calls)`. When the edit is placed in the active worker file, FastRSI completes the test in **`1.07 s`**, records **`768` witness hits** (8 requests × 32 tokens × 3 sampler stages), produces an output hash (`e8aa4a6b469cd55b`) bit-identical to a fresh cold restart, and rolls back to base in **`0.3 ms`**.

#### Case 2: Hot-reloading a 1-line Triton kernel bugfix (`vLLM #62046`, `104.96 s → 2.72 s`, `38.6x`)

Swapping a `@triton.jit` GPU kernel in a running serving engine is notoriously tricky because Triton compiles the kernel from source into a cached binary and callers hold references to the compiled handle. In upstream PR `vllm-project/vllm#62046`, a one-line fix in `_topp_sb_mask_kernel` updates `num_preceding` across multi-block loop iterations so split top-p sampling does not miscount duplicate boundary probabilities:

```diff
# vllm/v1/sample/ops/topk_topp_triton.py (@triton.jit _topp_sb_mask_kernel)
             cum = tl.cumsum(dmask.to(tl.int32)).to(tl.float32) + num_preceding
             keep_dup = dmask & (cum <= numkeep)
             keep = keep & (~dmask | keep_dup)
+            num_preceding += tl.sum(dmask.to(tl.float32))
         tl.store(ROW + i + offs, tl.where(keep, x, MASK_VALUE), mask=mask_n)
```

On 1× GB200 serving `Qwen3.6-27B` with `torch.compile` and CUDA graphs enabled, testing this single line via a restart takes **`104.96 s`** (`32.2 s` graph capture, `24.7 s` startup imports, `15.1 s` warmup). FastRSI invalidates the `@triton.jit` cache, recompiles the kernel in place in **`0.3 s`**, runs the 14-request verification workload in **`2.72 s` total (38.6x faster)** while recording **`43` kernel witness hits**, and rolls back in **`< 1 ms`** by restoring the snapshot's `CompiledKernel` pointer without invoking the Triton compiler again. Similarly, when an upstream PR adds a new state attribute inside `Scheduler.__init__` (`vLLM #62872`), FastRSI's `constructor_reinit` service updates the live scheduler instance in **`1.41 s`** vs. **`42.93 s` (30.4x)** with a **`1.58 ms`** undo, while live runtime reconfiguration of `--max-num-seqs` takes **`2.2 ms`** vs. **`27.6 s`**.

#### Case 3: FastVideo MiniMax H3: enabling heads-first Triton scatter (`01-vsa-heads-first-tile`, `67.9 s → 6.08 s`, `11.2x`)

During the FastVideo MiniMax H3 optimization run, the very first patch kept by the agent (`01-vsa-heads-first-tile` in `video_sparse_attn_h3.py`, `+4 / -1` lines) noticed that FastVideo's heads-first Triton tile scatter was gated behind an off-by-default environment variable, forcing every attention layer through a slow `index_put` and three tensor transpose copies:

```diff
# fastvideo/attention/backends/video_sparse_attn_h3.py (MiniMaxH3VSAImpl.tile)
         if (not compiling and attn_metadata.tile_elems in _PTX_BLACKWELL_TILE_ELEMS
-                and envs.FASTVIDEO_H3_VSA_HEADS_FIRST_TILE.get()
                 and supports_heads_first_scatter(x, attn_metadata.untile_combined_index)):
             return self._tile_heads_first(x, attn_metadata, holder, kernel_tiles, needs_ptx_blackwell_pair)
```

FastRSI hot-swapped this edit into the warm diffusion worker (`0.96 s` edit-to-run) and verified a full generated video in **`6.08 s`** instead of **`67.9 s` (11.2x faster)**. That change alone cut single-video generation latency from **`6.32 s` to `5.27 s` (-16.7%)** with bit-identical `sha256` output, launching the 19-patch climb to **1.52x**.

#### Case 4: Megatron-LM: overlapping sequence-parallel all-gather and gradient cast (`lmhead-sp-allgather-before-cast`, `44.4 s → 5.4 s`, `8.2x`)

During the 82-iteration zero-restart climb on Megatron-LM (`Qwen3-14B`, `TP=2`), `Codex` inspected `LinearWithGradAccumulationAndAsyncCommunication.backward` in `megatron/core/tensor_parallel/layers.py` (`+6 / -3` lines). Previously, the backward pass synchronously cast the large FP32 logit gradient `grad_output.to(ctx.input_dtype)` *before* launching the asynchronous sequence-parallel input `all_gather`. Retaining the cast in the deferred `grad_output_buffer` branch while delaying the main dtype conversion until immediately after the asynchronous `all_gather` launch overlaps communication with memory-bound casting:

```diff
# megatron/core/tensor_parallel/layers.py (LinearWithGradAccumulationAndAsyncCommunication.backward)
         input, weight = ctx.saved_tensors
         main_grad = ctx.main_grad
         use_bias = ctx.use_bias
-        # TE owns only the forward GEMM here; Megatron retains the backward contract.
-        # Cast dY to the input dtype to match the legacy FP32-logit-cast backward path.
-        grad_output = grad_output.to(ctx.input_dtype)
         ...
         wgrad_compute = True
         if grad_output_buffer is not None:
+            # Deferred weight gradients must retain the original compute dtype.
+            grad_output = grad_output.to(ctx.input_dtype)
             if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                 grad_output_buffer.append(grad_output)
                 wgrad_compute = False
         ...
+        # Launch the input all-gather before the potentially large FP32-logit
+        # gradient conversion, allowing communication to overlap this conversion.
+        # Preserve the legacy FP32-logit-cast backward compute dtype.
+        grad_output = grad_output.to(ctx.input_dtype)
         grad_input = grad_output.matmul(weight)
```

With FastRSI, testing this autograd edit and restoring `S0` state took **`5.4 s`** instead of **`44.4 s` (8.2x faster)**, contributing a standalone **`+1.24%` tokens/s/GPU** gain (median of 4 paired fresh-process rounds) toward the overall **`+5.1%` (14B) / `+7.1%` (32B)** throughput improvement.

---

## FastRSI Today

We are open-sourcing **FastRSI**, including the zero-dependency `core/livelab/` transaction kernel, the 11-case fault-injection conformance suite (`c0`–`c10`), the onboarding template (`template/`), the toolsmith agent prompts (`core/agents/`), and all six engine integrations (`vLLM`, `SGLang`, `Megatron-LM`, `FastVideo`, `veRL`, and `Miles`):

- **Code & Documentation:** [github.com/hao-ai-lab/fastrsi](https://github.com/hao-ai-lab/fastrsi)

You can try the transaction kernel and run all 11 conformance cases on a laptop in under 3 seconds with Python 3.11+ and zero third-party packages or GPUs required:

```sh
git clone https://github.com/hao-ai-lab/fastrsi.git && cd fastrsi
PYTHONPATH=core/livelab python3 -m conformance --lab kernel.toy_lab:ToyLab
```

### What's next

We are extending FastRSI along two directions:

1. **Live in-memory `TP ↔ EP` parallelism reconfiguration without disk weight reads:** Changing distributed parallelism strategies (for example, switching between tensor parallelism and expert parallelism on MoE models) normally requires killing the server and reloading tens of gigabytes of safetensors weights from disk. Our `reshard.py` service transposes `FusedMoE` shards in memory across a CUDA IPC `WeightHolder` without touching disk; it is verified on CPU against vLLM's MoE weight layouts today, with multi-GPU B200 benchmarks upcoming.
2. **Shared read-only weight sandboxes for parallel multi-branch testing on one GPU:** In a prototype sandbox on 1× B200 (`Qwen3-8B`), two concurrent `vllm serve` instances map all 291 model parameters onto a single shared `15.27 GiB` read-only GPU memory allocation, allowing an agent to hot-edit, benchmark, or even `SIGKILL` candidate worker A while worker B continues serving unaffected. Integrating shared-weight sandboxes with FastRSI will let coding agents evaluate multiple candidate edits in parallel on a single GPU.

---

## Acknowledgement

We thank NVIDIA for supporting our development and evaluation on NVIDIA GB200 and B200 GPUs. We also thank the open-source communities behind vLLM, SGLang, Megatron-LM, FastVideo, veRL, and Miles for building the engines and upstream PR ecosystems that made this work possible.

## Contributors

[Junda Chen](https://github.com/GindaChen), [Yi Sun](https://www.linkedin.com/in/yi-sun-mlsys), [Yuxuan Zhang](https://yuxuandexter.github.io/), [Yuanbo Yang](https://freemty.github.io/), [Junda Su](https://davids048.github.io/), [Will Lin](https://github.com/SolitaryThinker), [Hao Zhang](https://haozhang.ai)

## Citation

```bibtex
@misc{fastrsi2026,
  title        = {Towards Live-Editable {LLM} Infra for {RSI}},
  author       = {Chen, Junda and Sun, Yi and Zhang, Yuxuan and Yang, Yuanbo and Su, Junda and Lin, Will and Zhang, Hao},
  year         = {2026},
  month        = nov,
  howpublished = {Hao AI Lab Blog},
  url          = {https://haoailab.com/blogs/fastrsi/}
}
```
