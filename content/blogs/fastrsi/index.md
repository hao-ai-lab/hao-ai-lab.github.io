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
summary = "Improving AI systems through recursive self-improvement (RSI) is bottlenecked by evaluation: agents spend 60% to 95% of their wall-clock time waiting for GPU tests to finish. Today, testing even a one-line code change in a GPU engine like vLLM requires killing the process and booting the entire system from scratch, waiting several minutes to reload model weights and re-initialize the GPUs before a single test prompt can run. We introduce RSIBox, a live-editable GPU sandbox environment that keeps heavy GPU state alive in memory and hot-reloads only the small piece of the system that an edit actually changed, while isolating and rolling back faulty mutations in milliseconds. Paired with rapid targeted testing, RSIBox shrinks the time to test an edit in vLLM from 228 s to 2.4 s (95x faster), lets an agent reach a 6-hour vLLM optimization target 2.3x sooner, and generalizes across six major AI engines (vLLM, SGLang, Megatron-LM, FastVideo, veRL, and Miles) to deliver up to 1.52x end-to-end workload speedups with near-zero cold restarts."

[cover]
  image = "/img/fastrsi/cover.png"
  relative = false
  alt = "Cold restart every check vs. RSIBox live hot reload"
  caption = "Testing a one-line vLLM edit: cold restart vs. RSIBox live hot reload."
  hidden = true

[socialIcons]
  [[socialIcons.icon]]
    name = "twitter"
    url = "https://twitter.com/haoailab"
+++

<div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 4px 0 18px;">
  <a href="https://github.com/hao-ai-lab/rsibox" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/GitHub-RSIBox-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub: RSIBox" style="height: 28px; margin: 0;">
  </a>
  <a href="https://x.com/haoailab" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Follow-@haoailab-000000?style=for-the-badge&logo=x&logoColor=white" alt="X: @haoailab" style="height: 28px; margin: 0;">
  </a>
</div>

{{< image src="/img/fastrsi/cover.png" alt="Cold restart every check vs. RSIBox live hot reload" width="100%" title="Cover Figure. Testing a one-line vLLM scheduler edit. Left: a cold restart rebuilds tuned kernels, model weights, CUDA graphs, and the engine process before every test, 228 s in total, while RSIBox keeps that GPU state warm inside a live-editable sandbox and swaps the one function, finishing a targeted test in 2.4 s (95x). Right: with the faster test, the agent reaches the same speedup 2.3x sooner in a 6-hour vLLM optimization run." >}}


**TL;DR.** Improving AI systems through recursive self-improvement (RSI) is bottlenecked by evaluation: agents spend 60% to 95% of their wall-clock time waiting for GPU tests to finish. Today, testing even a one-line code change in a GPU engine like vLLM requires killing the process and booting the entire system from scratch, waiting several minutes to reload model weights and re-initialize the GPUs before a single test prompt can run.

We introduce **RSIBox**, a runtime sandbox and execution environment that makes GPU systems **live-editable**. Instead of restarting the engine on every edit, RSIBox keeps the heavy GPU state alive inside a fault-isolated sandbox, hot-reloads only the small piece of the system that the edit actually changed, and rolls back or recovers from broken edits without losing warm GPU state. Paired with rapid targeted testing, RSIBox shrinks the time to test an edit in **vLLM** from `228 s` to `2.4 s` (**up tp 95x faster**), lets an agent reach a 6-hour vLLM optimization target **2.3x sooner**, and generalizes across six major AI engines (**vLLM**, **SGLang**, **Megatron-LM**, **FastVideo**, **veRL**, and **Miles**) to deliver up to **1.52x** end-to-end workload speedups with near-zero cold restarts.

---

## Evaluation Is the Bottleneck of RSI

Imagine you are a developer sitting at your keyboard fixing a bug or tuning a kernel in an AI engine. You tweak a few lines of code, run a quick check to see whether it helps, and either keep the change or revert and try something else. As a project grows larger, however, evaluating even a single line of code against the full system can take several minutes to warm up. As a result, you can wait for several minutes (or even longer!) to test your one line of change actually runs and perform as expected in the real system. 

Recursive self-improvement (RSI) for AI systems follows that exact loop. Each iteration has two steps: (1) the model **generates candidate edits** (token generation), and (2) the runtime **evaluates each edit on physical GPUs** for correctness and speed.

As frontier coding models generate tokens faster and faster, GPU evaluation becomes the dominant wall-clock bottleneck. Booting a distributed engine, loading weights from storage, and compiling kernels still take just as long today as they did before coding agents existed.

As a concrete example, we ran a **6-hour** experiment to optimize vLLM on a particular metric, and replayed it while speeding up model's token generation. As shown in the figure, even as we scale the token generation to 1500 tps, the total runtime only shrink by **6%** because the evaluation dominated the clock. In other words, faster code generation will quickly hit an Amdahl's Law wall unless we make evaluation itself faster.

{{< image src="img/fig1-eval-bottleneck.png" alt="Wall-clock breakdown of an agent optimization run" width="100%" title="Figure 1. Replay of a 6-hour vLLM optimization trace at 1x, 5x, 15x, and 30x token generation speed. A 30x faster coding model shortens the run by only 6%: the time saved on generation turns into waiting on evaluation." >}}

---

## Reloading the System can Dominate Evaluation Time

In a standard agentic workflow, each code change is evaluated by running a pre-defined test script from a clean process, which forces the system to reload from scratch on every trial. To see how much overhead this adds, consider optimizing vLLM to serve Nemotron 3.5 Super on an NVIDIA GB200 GPU. Even when an agent modifies just a single line in the request scheduler, testing that change triggers a full **cold restart** through four heavy startup stages before a single test prompt can run:

1. **Kernel autotuning (`~73 s`):** Resolving and warming up GPU kernel launch configurations.
2. **Model weight loading (`~57 s`):** Streaming multi-gigabyte checkpoints into GPU memory.
3. **CUDA graph capture (`~40 s`):** Recording decode kernel launch graphs across batch sizes.
4. **Process, library, and device initialization (`~30 s`):** Importing the Python/C++ software stack (`torch`, `triton`, `vllm`) and initializing device contexts, communication groups, and KV cache pools.

Repeating this initialization sequence before running the `~28 s` test suite means evaluating a single one-line scheduler edit takes **228 s**, nearly **90%** of which is pure system reload overhead.

{{< image src="img/fig2-vllm-ladder.png" alt="Breakdown of testing a one-line edit in vLLM: cold restart vs. RSIBox" width="100%" title="Figure 2. Where time goes when testing a one-line vLLM scheduler edit on 1x NVIDIA GB200. Roughly 200 s (nearly 90%) is reload overhead that the scheduler edit never invalidates." >}}

Notice that none of those four startup stages depends on the modified line in the scheduler. In general, most exploratory edits are localized tweaks (to a scheduling rule, a kernel loop bound, or a communication call) that only invalidate a tiny fraction of the running system. As the upper steps of Figure 2 show, keeping the engine process and GPU state warm in memory while **hot-reloading** only the modified code eliminates the `~200 s` reload tax completely, cutting a full evaluation from **228 s down to 23.6 s (9.6x faster)**.

Once that `~200 s` reload wall is out of the way, the remaining `23.6 s` is spent on the test suite itself: a full test batch to measure throughput (`14.1 s`) followed by a log-probability check (`9.5 s`, a separate reference-prompt pass that compares output token log-probability distributions against a golden baseline to verify numerical accuracy without adding logprob copy overhead to the throughput run). Because most exploratory edits fail correctness checks or regress latency right away, we no longer need to run the full test batch on every trial: screening first with a small-batch **targeted test** on the warm engine (the bottom steps of Figure 2) compresses the evaluation loop all the way down to **2.4 s (95x faster)**, or 4.8 s when including the log-probability check pass. This motivates making GPU engines **live-editable** rather than restarting them from scratch for every experiment.

---

## How to Make vLLM Live-Editable

So how do we make a complex system like vLLM live-editable? Our guiding principle is simple: **reload only the minimal subset of the system that an edit actually invalidates, and keep all untouched GPU state warm in memory.**

Think about how you work inside a **Jupyter Notebook**. Once you have imported PyTorch and loaded a multi-gigabyte model into GPU memory in cell 1, you never restart the kernel just to tweak a helper function in cell 5. You simply re-run the cell that defines that one function, and your next forward pass immediately uses the new code while the model weights, CUDA context, and allocated buffers stay right where they are. That notebook workflow makes live editing look effortless.

In a production system like vLLM, however, changing a function in place immediately hits four system-level caveats:

1. **Compiled and captured execution layers:** What if the function is a JIT-compiled GPU kernel (such as Triton or C++/CUDA) or is already baked into a captured CUDA graph? Updating a host-level function pointer does nothing if the GPU keeps replaying stale compiled binaries or recorded driver graphs.
2. **Stateful mutations and native extensions:** What if the edit changes class attributes, reshapes pre-allocated KV or activation buffers, or modifies a compiled C++/CUDA extension rather than a pure stateless function?
3. **Multi-process and multi-node distributed workers:** What if the system runs across multiple processes or distributed worker nodes over a network (across tensor-, pipeline-, or expert-parallel ranks, or remote Ray actors)? If rank 0 updates a collective call mid-step while rank 1 runs the old version, the cluster deadlocks.
4. **System-breaking faults and state contamination:** What if a submitted edit crashes the process, triggers an illegal GPU memory access that corrupts the CUDA context, or silently mutates model weights, KV caches, or random seeds so that every subsequent test runs on contaminated state (or never even executes the edited path)? How do we recover from fault states?

Existing process checkpointing tools (`CRIU`, `cuda-checkpoint`) and inference memory/snapshot systems such as NVIDIA Dynamo already make great strides on accelerating inference startup and unchanged-code recovery. For LLM inference, RSIBox builds directly on top of them—and adds a standalone application-layer runtime for training, diffusion, and RL—separating persistent GPU state (loaded weights, communication groups, KV pools, and unaffected CUDA graphs) from mutable execution layers so code edits take effect in milliseconds without restarting the engine.

{{< image src="img/fig3-architecture.png" alt="Two-plane live-editable runtime architecture: mutable worker plane vs. persistent out-of-process GPU and host state" width="100%" title="Figure 3a. Live-editable runtime architecture. Top: staging, AST/call-graph routing, and atomic two-phase commit between steps. Middle: the mutable worker plane, separating the in-process live swap zone (89% of PRs) and verification/rollback guards from the 5 to 12 s process respawn zone. Bottom: the persistent out-of-process warm foundation (CUDA IPC weight sentinel, pre-CUDA Zygote fork server, and baseline snapshots) that survives across edits and worker crashes." >}}

### Reloading Only What Changed

To turn the minimal-reload principle into a concrete system rather than a collection of ad-hoc patches, we first asked an empirical question: **what do real code changes in a production system like vLLM actually modify?**

We surveyed the vLLM repository's pull-request history, scanning roughly **5,000 PRs** related to logic and performance changes and selecting **487 representative candidates** (with deep execution-path and state-boundary audits across **80 performance PRs**). Across these real-world changes, we found that only **~5% of deep engineering changes** (mostly large multi-file refactorings, file movements, and distributed topology changes) truly require a full cold restart. The remaining **90% to 95% of practical optimization and bugfix edits** fall cleanly into four modification patterns, reaching progressively deeper into the runtime stack (Figure 3b). We designed our live-reload mechanisms directly around these four cases:

**Case 1: Editing a pure host function (`~20%` of audited PRs; reload in `~0.05 to 0.15 s`).**
Roughly a fifth of performance and logic PRs modify pure host-level orchestration code, such as tweaking a batch-ordering heuristic in vLLM's request scheduler (`Scheduler.schedule`) or a logit-filtering rule in the token sampler. Standard module reloaders (`importlib.reload`) fail here because they only rebind the top-level module attribute, while the running `LLMEngine`, worker loops, bound methods, and closures keep pointing to the old function object in memory. Instead, we locate the live function object via its active referrers (`gc.get_referrers`) and swap its compiled bytecode (`__code__`), default arguments, and closure cells in place, so the very next step runs the new logic in about `0.1 s`.

**Case 2: Editing a JIT-compiled GPU kernel (`~0.1 to 0.3 s`).**
What if we edit a custom `@triton.jit` GPU kernel, such as fixing duplicate boundary probability accumulation in vLLM's split top-p sampling kernel or tuning tile sizes in a fused MoE helper? Here, swapping the Python function object alone has zero effect on the GPU. Triton compiles each kernel into a native GPU binary (`cubin`) on first launch and caches it by specialization signature; worse, when an outer Triton kernel calls an inner `@triton.jit` helper, the compiler inlines that helper directly into the caller's binary. To make a kernel edit take effect, we trace the cross-module Triton call graph, clear the compiled binary and autotuner caches on both the edited kernel and all of its transitive callers, and recompile only those affected kernels in `~0.1 to 0.3 s`.

**Case 3: Editing code or kernels inside captured CUDA graphs (`~0.4 to 1.0 s`; together with Case 2 covering `~51%` of audited PRs).**
Over half of vLLM performance PRs touch GPU kernels or operators that execute inside vLLM's decode loop, which records sequences of kernel launches into **captured CUDA graphs** across dozens of padded batch sizes at startup. During decode, replaying a CUDA graph dispatches the recorded GPU kernel pointers directly from C++ inside the CUDA driver without re-entering Python or Triton caches at all, meaning even a freshly recompiled Triton kernel would be completely bypassed. After updating the host function or recompiling the Triton kernel, we trace which captured graphs invoke that operator and selectively recapture only the affected batch-size or piecewise layer CUDA graphs in `~0.4 to 1.0 s`, leaving all other graphs untouched.

<!-- <details>
<summary><b>View implementation: In-place hot reload across Cases 1-3 (host functions, Triton kernels, and CUDA graphs)</b></summary>

```python
def hot_reload_target(old_fn, new_fn, triton_call_graph, cuda_graphs, journal):
    # Case 1: Swap bytecode & defaults in place across all live references
    journal.record_fn(old_fn, old_fn.__code__, old_fn.__defaults__)
    old_fn.__code__, old_fn.__defaults__ = new_fn.__code__, new_fn.__defaults__

    # Case 2: If editing a @triton.jit kernel/helper, invalidate all transitive callers
    if isinstance(old_fn, triton.runtime.JITFunction):
        for caller in triton_call_graph.transitive_callers_of(old_fn):
            journal.record_triton_cache(caller, caller.cache.copy())
            caller.cache.clear()          # Drop cached cubin / PTX binaries
            caller.autotuner_cache.clear()

    # Case 3: Selectively recapture only the CUDA graphs that invoke old_fn
    for graph_key in cuda_graphs.graphs_touching(old_fn):
        journal.record_graph(graph_key, cuda_graphs[graph_key])
        cuda_graphs[graph_key] = cuda_graphs.recapture_single(graph_key)
```
</details> -->



<!-- {{< image src="img/fig3b-execution-layers.png" alt="Three execution layers of a GPU system: host bindings, compiled GPU kernel caches, and captured CUDA graphs" width="100%" title="Figure 3b. Why naive module reloading fails across host code, compiled GPU kernel caches, and captured CUDA graphs, and how our runtime updates each level in place." >}} -->

**Case 4: Editing stateful objects, native C++/CUDA kernels, or worker startup flags (`~18%` in-process state/C++ + `~5%` warm respawn; reload in `0.2 to 12 s`).**
Not every optimization is a stateless function or Triton kernel tweak. About `8%` of audited PRs add a new state attribute or scratch tensor inside a component's `__init__` (for example, pre-allocating a reusable position-metadata buffer on the attention runner so it is not rebuilt on every layer); for these, we re-initialize only that modified sub-module or restore its saved component snapshot in `~0.2 to 1.8 s` while keeping GPU model weights and KV caches intact. Another `10%` modify a native C++/CUDA `.cu` kernel (such as vectorizing vLLM's fused RMSNorm or rotary embedding C++ kernel); we incrementally compile the `.cu` source into a versioned shared library loaded via `ctypes.CDLL(..., RTLD_LOCAL)` and rebind the operator on the PyTorch C++ dispatcher (`torch.library.Library`) in `~7 to 12 s`, followed by selective CUDA graph recapture.


<!-- <details>
<summary><b>View implementation: Native C++/CUDA side-library reload and zero-copy CUDA IPC weight attach</b></summary>

```python
# Native C++/CUDA side-library reload via PyTorch dispatcher override:
def hot_reload_cuda_op(so_path: str, op_ns: str, op_name: str, sym: str, journal):
    cdll = ctypes.CDLL(so_path, mode=ctypes.RTLD_LOCAL)
    lib = torch.library.Library(op_ns, "IMPL", "CUDA")
    lib.impl(op_name, getattr(cdll, sym))
    journal.on_rollback(lambda: lib._destroy())  # 2 ms rollback restores previous C++ kernel

# Zero-copy CUDA IPC weight attach in a fresh Zygote-forked worker:
# (Sentinel exports: ipc_handles = {k: v.untyped_storage()._share_cuda_() for k, v in params.items()})
def attach_shared_weights(ipc_handles, model_params):
    torch.cuda.init()  # Initialize a clean CUDA context after fork()
    for name, (device, handle, size, offset, meta) in ipc_handles.items():
        storage = torch.UntypedStorage._new_shared_cuda(device, handle, size, offset)
        model_params[name].set_(storage, *meta)  # Reattaches in ~2 s with 0 B disk read
```
</details> -->

Even when an edit changes a process-level startup flag or global registry that requires a clean worker process, we can still avoid re-reading checkpoints from disk or re-importing the Python stack. We hold model weights in an out-of-process **CUDA IPC weight sentinel** and fork fresh workers from a pre-CUDA **Zygote fork server** that has already imported `torch`, `triton`, and `vllm` upfront. The forked worker initializes a clean CUDA context and reattaches multi-gigabyte model weights over zero-copy CUDA IPC in **`5 to 12 s`** with zero disk reads.



Finally, in a multi-GPU run (such as tensor-parallel decoding across 4x GB200 GPUs), all workers must switch to the new code at the exact same step boundary or not at all, lest rank 0 and rank 1 diverge on a collective call. While the system runs, the runtime classifies the edit and finishes any compilation in the background. During the brief (`~1 ms`) pause between two steps, a **two-phase coordinator** (`PREPARE` -> `COMMIT` / `ABORT`) applies the update atomically across all GPU ranks. The table below summarizes the full hierarchy of reload boundaries:

| Reload Boundary | How the Runtime Applies the Edit | What Edits It Handles | Reload Time |
| --- | --- | --- | --- |
| **In-place function swap** | Updates the function's code object and closure cells in place across all live references (`gc.get_referrers`) and distributed workers | Pure host functions, schedulers, samplers, and method bodies | **~0.05 to 0.15 s** |
| **Compiled GPU kernel rebuild** | Extracts updated kernel source, walks the cross-module `@triton.jit` call graph to clear binary and autotuner caches on all transitive callers, and recompiles | Triton GPU kernels and inlined `@triton.jit` helper functions | **~0.1 to 0.3 s** |
| **Selective CUDA graph recapture** | Applies the code or kernel update and selectively recaptures only the affected batch-size or piecewise layer CUDA graphs | Functions and GPU kernels executed inside captured CUDA graphs | **~0.4 to 1.0 s** |
| **In-process object re-initialization** | Re-initializes modified sub-modules or restores saved component snapshots while keeping GPU model weights and KV caches intact | Edits to `__init__`, adding new state attributes, or resizing local scratch buffers | **~0.2 to 1.8 s** |
| **Native C++/CUDA extension hot-reload** | Incrementally compiles edited `.cu`/C++ sources into a versioned shared library loaded via `ctypes.CDLL(..., RTLD_LOCAL)`, rebinds the PyTorch dispatcher (`_Cv<key>` `torch.library.Library`) with selective CUDA graph recapture, and rolls back in 2 to 3 ms via journaled `lib._destroy()` | Native C++ and raw CUDA `.cu` kernel extensions (e.g., fused RMSNorm, RoPE, custom attention/activation kernels) | **~7 to 12 s** |
| **Warm fork server respawn** | Forks a clean worker from the pre-warmed Zygote fork server and reattaches GPU weights over zero-copy CUDA IPC | Startup CLI flags, global registries, or clean recovery after a GPU kernel crash | **~5 to 12 s** |
| **Full cold restart** | Launches a fresh worker group from scratch with warm on-disk compiler caches | Cross-node or world-size topology changes and unrecoverable driver faults | **~1 to 4 min** |

### Recovering from Bad Edits

In an exploratory search, most candidate edits fail correctness checks, regress latency, or crash a GPU kernel. Keeping a warm session alive across hundreds of experiments requires handling four distinct failure situations without letting any failed trial pollute the next one:

**Case 1: The edit touches unswappable startup state (safe refusal and fallback in `< 0.2 s`).**
Suppose an edit alters the tensor-parallel world size or breaks a compiled kernel wrapper signature. Before mutating any live state, our static analyzer catches the violation in under `0.2 s`, refuses the in-place swap with an explicit diagnostic reason, and automatically routes the change to a warm fork respawn or a cold restart.

**Case 2: The edit fails correctness or regresses latency on a healthy worker (`1 to 3 ms` journaled rollback + state reset).**
Suppose a tweaked scheduler rule or Triton tile config runs without crashing, but shifts output log-probabilities or runs 5% slower. Because every transaction records its exact inverse in an append-only in-memory journal (saving prior bytecode references, compiled Triton GPU binary pointers, autotuner winners, `torch.library` handles, and CUDA graph objects), the runtime unwinds the edit in **`1 to 3 ms` with zero recompilation** and verifies by SHA-256 hash that every symbol is back at baseline. After every trial, it also drains in-flight requests and restores pre-test snapshots of volatile KV caches, scratch buffers, random seeds, data iterators, and model parameters so state never leaks across tests.

**Case 3: The edited code never actually ran during the test (per-rank live call-counter gate).**
Large systems like vLLM often retain legacy or conditional execution paths alongside the active GPU pipeline (such as an older token-sampling branch that active GPU workers no longer call). If an edit modifies an inactive path, the test finishes with unchanged outputs and zero warnings, which a standard harness mistakes for a valid pass. We catch this in **`~1 s`** by tracking **per-rank call counters** on every modified function and kernel, explicitly counting **live evaluation replays** separately from **CUDA graph-capture passes** so an edit is accepted only if the live workload executes it on every GPU rank.

**Case 4: A buggy GPU kernel crashes or deadlocks the worker (`5 to 12 s` warm respawn via Zygote + CUDA IPC).**
Suppose an out-of-bounds indexing bug in a custom Triton or CUDA `.cu` kernel triggers an illegal memory access (`CUDA_ERROR_ILLEGAL_ADDRESS`), a device-side assert, or an NCCL collective timeout that poisons the CUDA context. Once a CUDA context is corrupted, in-memory rollback is unsafe: the runtime immediately marks the worker session **tainted**, kills the poisoned worker group, and forks a clean replacement from the pre-warmed **Zygote fork server**, reattaching model weights over **CUDA IPC** in **`5 to 12 s`** without touching disk.

<!-- <details>
<summary><b>View implementation: Step-boundary atomic evaluation, live call-counter verification, and 1-3 ms journaled rollback</b></summary>

```python
def evaluate_edit_at_step_boundary(patch, dist_group, engine_state):
    with step_boundary_barrier(dist_group), Journal() as journal:
        hot_reload_target(*patch, journal=journal)
        call_witness.reset(mode="LIVE_REPLAY")  # Ignore calls made during CUDA graph capture

        metrics, ok = run_targeted_probe(engine_state)
        ran_on_all_ranks = dist_group.all_reduce_min(call_witness.count) > 0

        if not (ok and ran_on_all_ranks):
            journal.unwind()                    # 1-3 ms pointer restore, zero recompilation
            engine_state.restore_snapshot()     # Reset KV cache, RNG seeds, and buffers
            return RejectResult(ran_on_all_ranks=ran_on_all_ranks)
        return PassResult(metrics=metrics)
```
</details> -->

### Evaluating on vLLM

We evaluated this live-reload runtime on **vLLM** across per-category micro-benchmarks, continuous zero-restart stress tests, and a 6-hour end-to-end autonomous optimization race.

**1. Per-edit evaluation latency across vLLM edit categories.**
The table below compares a standard cold restart against live evaluation across representative vLLM edits on NVIDIA GB200 and B200 GPUs (hover over each edit category for model and hardware details):

| vLLM Edit Category | Cold Restart | Live Eval | Speedup & Breakdown |
| --- | --- | --- | --- |
| <span title="4-bit Nemotron 3.5 Super, 1x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**In-place scheduler edit + full evaluation**</span> | 228 s | **23.6 s** | **9.6x** (`~0.1 s` swap, `~204 s` reload overhead removed) |
| <span title="4-bit Nemotron 3.5 Super, 1x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**In-place scheduler edit + targeted test**</span> | 228 s | **2.4 s** (`4.8 s` with logprobs) | **95x** (`48x` with full log-probability distribution check) |
| <span title="Qwen3-8B, 1x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**Active vs. dead-code sampler edit**</span> | ~63 s | **~1.1 s** | **~57x** (`~0.14 s` install, `< 1 ms` undo; flags 0-call dead code in `1.1 s`) |
| <span title="Split top-p sampling bugfix, Qwen3.6-27B, 1x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**Compiled Triton kernel edit**</span> | ~105 s | **~2.7 s** | **~39x** (`0.3 s` transitive Triton kernel recompile, `< 1 ms` binary pointer undo) |
| <span title="Upstream PR #49750 fused RMSNorm, Qwen3-8B, 2x NVIDIA B200 (TP=2)" style="border-bottom: 1px dotted currentColor; cursor: help;">**Native CUDA `.cu` kernel edit + selective graph recapture**</span> | ~47 to 55 s | **~13.6 s** | **Up to 3.9x** (`11.8 s` parallel `.so` build, `0.7 s` graph recapture, `9 ms` cascade undo) |
| <span title="Zygote fork + CUDA IPC weights, Qwen3-8B, 1x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**Worker respawn / CLI flag reconfigure**</span> | 51.9 s | **7.4 s** | **7.0x** (`2.2 s` CUDA IPC weight attach, `0 B` read from disk) |

**2. Continuous zero-restart replay and fault-injection stress tests.**
Can a single long-lived vLLM session survive dozens of back-to-back mutations, rollbacks, and deliberately broken edits without drifting or needing a cold restart? We replayed **19 real upstream vLLM optimization pull requests** sequentially and ran **43 end-to-end step and fault-injection scenarios** (including syntax errors, bad tensor shapes, NaN injection, zero-call dead paths, and multi-function transaction rollbacks) inside a single warm vLLM session. The engine completed the entire suite with **zero cold restarts**, maintaining bit-exact greedily decoded tokens and matching top-5 log-probabilities within `1.2e-4` against a fresh cold process on every accepted edit.

**3. End-to-end 6-hour vLLM optimization race.**
Finally, we measured how eliminating system reload overhead changes an end-to-end autonomous optimization run on **vLLM** serving **DeepSeek-V4.1 Flash** decode on **4x NVIDIA GB200 GPUs** (`8` concurrent requests per GPU at `32K` context length) over a fixed **6-hour wall-clock budget**, comparing a standard **Cold-Restart Baseline** against our **Live-Editable Runtime** using the same coding model and prompt harness:

| Metric (<span title="vLLM serving DeepSeek-V4.1 Flash decode on 4x NVIDIA GB200, 8 reqs/GPU at 32K context, 6-hour wall-clock budget" style="border-bottom: 1px dotted currentColor; cursor: help;">6-Hour vLLM Optimization Race</span>) | Cold-Restart Baseline | Live-Editable Runtime (Hot Reload + Targeted Test) | Advantage |
| --- | --- | --- | --- |
| **Time per GPU check** | **227.5 s** (`~199.5 s` reload + `28.0 s` test) | **2.4 s** targeted probe / **23.6 s** full eval | **95x faster** screening / **9.6x faster** full check |
| **Fastest check-to-check turnaround in race** | **183.6 s** (>3 min wait between checks) | **13.7 s** (including agent reasoning between probes) | **13.4x faster** minimum iteration turnaround |
| **GPU evaluation throughput capacity** | **~16 checks / hour** of GPU eval time | **~150 full checks** or **~1,500 targeted checks / hour** of GPU eval time | **9.6x to 95x more checks** per hour of GPU evaluation |
| **Speedup at 1.0 hour** | `1.050x` (+5.0% at `1.06 h`) | **`1.080x` (+8.0% at `1.02 h`)** | Reaches +7.8% in `0.98 h` vs. `4.1 h` (**>4.1x sooner**) |
| **Time to match Baseline's final 6-hour speedup (`+9.8%`)** | **5.49 hours** (`1.098x`) | **2.44 hours** (`1.101x`) | **2.3x sooner** (saves >3 hours of wall-clock time) |
| **Final 6-hour decode throughput speedup** | `1.098x` (+9.8%, repeat: `+10.6%`) | **`1.112x` (+11.2%, repeat: `+11.3%`)** | Higher final speedup across deeper multi-kernel optimizations |

Because live reload and targeted testing drop per-check GPU evaluation from nearly four minutes (`227.5 s`) to `2.4 s` for a targeted probe and `23.6 s` for a full confirmation, the agent no longer spends its 6-hour budget waiting for vLLM to reload weights, autotune kernels, and recapture CUDA graphs. Instead, it screens and discards faulty kernel variants in seconds (`13.7 s` minimum check-to-check turnaround vs. `183.6 s`), crosses the baseline's 6-hour finish line (`+9.8%`) in just **2.44 hours (2.3x sooner)**, and uses the remaining three and a half hours to stack five verified custom Triton and CUDA kernel optimizations (parallel hidden-tile mHC post kernels, reused mega-attention position/sparse metadata across layers, single-warp MegaMoE staging tiles for `block_m == 1`, fused valid-row mHC normalization, and rounded-division Triton statistics) to reach **1.112x to 1.113x** decode throughput.

{{< image src="img/fig3c-vllm-eval.png" alt="Evaluating live reload on vLLM: per-edit evaluation time across six categories and 6-hour autonomous optimization race" width="100%" title="Figure 3c. Evaluating live reload on vLLM. Left: single-edit evaluation latency across six representative vLLM edit categories (3.9x to 95x faster). Right: 6-hour autonomous optimization race on 4x NVIDIA GB200 GPUs serving DeepSeek-V4.1 Flash decode, where live reload reaches the cold-restart baseline's final 6-hour speedup 2.3x sooner (2.44 h vs. 5.49 h) and climbs to 1.112x (+11.2%)." >}}

---

## RSIBox: Turning AI Systems into Live-Editable Sandboxes

So far, we built live editability specifically for vLLM. Next, we show how we design **RSIBox** to generalize live editability to other AI systems, including **SGLang**, **FastVideo**, **Megatron-LM**, **veRL**, and **Miles**. We observe that generalizing across codebases mostly requires answering two questions: **what uniform interface should a live-editable sandbox expose**, and **how do we wire those application-level hooks into a new repository automatically?**

### Design

To support live editing of AI applications rather than forcing a cold restart on every shell command, **RSIBox** wraps a warm GPU process behind a minimal four-operation protocol (Figure 4a, right) that abstracts the reload, verification, and rollback mechanisms from Section 3:

- **`install(patch)`:** Applies a candidate diff at the narrowest valid reload boundary without restarting the engine.
- **`check(probe)`:** Runs a targeted probe on the warm process and returns live execution counts, numerical parity, and latency.
- **`reset()`:** Restores volatile application state after a test so the next trial starts from an identical baseline.
- **`uninstall()`:** Reverts a rejected edit cleanly (or triggers a warm worker respawn if the edit faulted the GPU context).

At run time (`①` to `⑥` in Figure 4a), any standard coding agent (such as Codex or Claude Code) plans an edit (`① Think`), writes a patch (`② Edit`), and calls `③ install` and `④ check` for hardware feedback in seconds. Based on the verdict (`⑤ Decide`), it either keeps the change (`reset`) or reverts it (`uninstall`), while promising wins are asynchronously re-measured in **fresh, alternating cold processes** (`⑥`) to confirm that the speedup holds from a clean boot.

{{< image src="img/fig4a-rsibox-agent-loop.png" alt="RSIBox architecture: offline Toolsmith builds the live-editable sandbox, and a coding agent optimizes inside it" width="100%" title="Figure 4a. The RSIBox architecture. (a) Offline: a Toolsmith agent scans a target repository, wires shared reload primitives into a warm RSIBox sandbox exposing install, check, reset, and uninstall, and verifies parity against a fresh cold run. (b) At run time: a standard coding agent (e.g., Codex or Claude) runs its edit-and-test loop against the sandbox API, while the Toolsmith stays on call if an edit hits an unhooked path." >}}

Implementing the API `install`, `check`, and `reset` inside each codebase then becomes an engine-specific task because every AI system has its own step loop and mutable state. To bring live editability across diverse codebases without hand-coding every adapter, RSIBox pairs the sandbox with an automated **Toolsmith agent** (Figure 4a, left). First, as a one-time offline effort, the Toolsmith scans the target repository and past PRs (`Mine`), hooks our shared primitives from Section 3 into `install / check / reset / uninstall` (`Build`), and verifies on physical GPUs that hot-reloaded edits match a fresh cold-process ground truth (`Verify`). Then, during an active RSI loop, if the coding agent edits an unhooked path that triggers a slow restart (`⑦`), the Toolsmith wakes up on call (`⑧`) to add and verify a new reload hook on the fly.

**Take FastVideo as an example.** When pointed at the FastVideo repository, the Toolsmith scans the codebase and commit history (`Mine`), identifying the 3D diffusion denoising loop and its Triton sparse-attention kernels as the primary hotspots while noting that loading the multi-stage checkpoints from disk takes ~8.5 minutes. It then wires our shared primitives into the four sandbox hooks (`Build`): keeping the foundation weights in an out-of-process CUDA IPC sentinel, attaching `install` and `uninstall` at the denoising step boundary, configuring `check` to run a short video probe with frame and audio parity checks, and configuring `reset` to clear latent buffers and restore the noise seed (`Verify`). Later during the optimization run, when the coding agent introduces per-layer CUDA graph caching inside the transformer blocks (a new stateful path that initially falls back to a slow reload, `⑦`), the on-call Toolsmith wakes up (`⑧`), adds a selective graph-recapture and cache-reset hook, verifies it against a fresh cold run, and hands the updated sandbox back to the coding agent.

### Generalization to different systems

To see why case-by-case adaptation matters, we applied RSIBox across six production systems spanning four distinct workload domains (**requiring zero upstream source changes** across vLLM, SGLang, veRL, and Miles):

| System | Workload Domain | Case-by-Case Live-Edit Challenge Solved in RSIBox |
| --- | --- | --- |
| **vLLM** | LLM inference serving | Coordinating in-place Python and Triton kernel rebuilds across transitive inlined callers alongside selective piecewise CUDA graph recapture. |
| **SGLang** | LLM inference serving | Selectively recapturing decode CUDA graphs in `0.84 s` inside an asynchronous scheduler event loop across tensor-parallel workers. |
| **FastVideo** | Video-audio diffusion | Sharing multi-stage diffusion weights over CUDA IPC for 5-second crash respawns and hot-swapping kernels inside the denoising loop. |
| **Megatron-LM** | Distributed pretraining | Wrapping synchronous training steps to pair `50 ms` code swaps with `< 0.3 s` in-VRAM restoration of coupled weights and optimizer states. |
| **veRL** | RL | Rebinding live remote Ray actor instances in place across a distributed RL cluster and resetting coupled trainer and rollout weights in `~2.5 s`. |
| **Miles** | RL | Synchronizing `< 0.1 s` code swaps across decoupled Ray rollout actors while tagging rollout trajectories by code version to drain stale samples. |

Below, we walk through how RSIBox handles the unique state, process, and execution-loop challenges in **FastVideo**, **Megatron-LM**, and **veRL / Miles**.

#### 1. FastVideo: Multi-stage diffusion pipelines and per-layer CUDA graphs

In **FastVideo (MiniMax H3)**, cold-starting the multi-stage video and audio pipeline takes **~8.5 minutes** (and **~68 to 72 s** even for a warm restart), while each generation request steps through dozens of synchronous denoising iterations across sequence-parallel GPUs. With all model stages anchored in the CUDA IPC sentinel for **~5 s** crash recovery, RSIBox's `reset` hook clears latent buffers, restores the initial noise seed, and enforces a **multi-shape memory gate** (running two distinct video resolutions in the same warm worker) to catch per-layer CUDA graph cache leaks before they can cause out-of-memory errors on variable-length requests.

During our optimization run, the coding agent inspected FastVideo's sparse 3D video attention backend and noticed that an optimized heads-first Blackwell Triton tile-scatter path was already implemented in the repository but disabled by default, forcing every attention layer to fall back to generic indexing and transpose copies. Inside RSIBox, the agent hot-swapped a 4-line condition to enable the heads-first Triton tiling path and evaluated a complete generated video clip on the warm worker in **~6 to 7 s** (versus ~70 s for a warm restart and ~8.5 min cold, an **~11x to ~75x** speedup), verifying bit-identical frames and audio while cutting end-to-end generation latency by **17%**.

#### 2. Megatron-LM: Stateful distributed pretraining with coupled optimizer snapshots

In **Megatron-LM**, every distributed training step mutates model weights, gradients, Adam optimizer states, and data-loader cursors in place. If a kernel edit is hot-swapped on step $t$ and tested over three steps, step $t+3$ leaves the model and optimizer on a modified trajectory for the next candidate edit. Instead of restarting the training job across tensor- and sequence-parallel GPUs (which takes **~44 to 58 s**, or over **4 minutes** with full warmup), RSIBox wraps `train_step` to snapshot those coupled training states directly in GPU memory before each probe and restores them in place across all ranks in **under `0.3 s`**, paired with a **`50 ms` code swap**.

<!-- <details>
<summary><b>View implementation: Megatron-LM coupled weight, Adam optimizer, and RNG snapshot restore (&lt; 0.3 s)</b></summary>

```python
class MegatronStateSnapshot:
    def capture(self, model_chunks, optimizer, rng_tracker, dataloader):
        self.weights = [p.data.clone() for c in model_chunks for p in c.parameters()]
        self.adam = {id(p): (st["exp_avg"].clone(), st["exp_avg_sq"].clone())
                     for st in optimizer.state.values() for p in [st] if "exp_avg" in st}
        self.rng_states = rng_tracker.get_states()
        self.data_cursor = dataloader.state_dict()

    def restore_in_place(self, model_chunks, optimizer, rng_tracker, dataloader):
        with torch.no_grad():
            for p, saved_w in zip((p for c in model_chunks for p in c.parameters()), self.weights):
                p.data.copy_(saved_w)                 # In-VRAM tensor copy (< 0.3 s total)
            for p_id, (m1, m2) in self.adam.items():
                 optimizer.state_by_id[p_id]["exp_avg"].copy_(m1)
                 optimizer.state_by_id[p_id]["exp_avg_sq"].copy_(m2)
            for c in model_chunks:
                c.zero_grad_buffer()                  # Clear main_grad views cleanly
        rng_tracker.set_states(self.rng_states)
        dataloader.load_state_dict(self.data_cursor)
```
</details> -->

For example, while optimizing **Qwen3-14B** pretraining across 2x GB200 GPUs, Codex found that the backward pass of the sequence-parallel linear layer synchronously cast a high-precision gradient tensor down to the activation dtype *before* launching the asynchronous `all-gather`, even though only the deferred weight-gradient GEMM needed that cast. Delaying the cast until right after launching the `all-gather` overlapped memory-bound casting with multi-GPU communication. With RSIBox, a 3-step loss-equivalence probe finished in **`0.5 s`** (and a multi-step throughput check in **`~5.4 s`** vs. `~44 to 58 s` for a restart, **~8x to >100x faster**), confirming identical loss and gradient norms within `1e-5`.

#### 3. veRL and Miles: Multi-actor Ray reinforcement learning clusters

Distributed reinforcement learning frameworks like **veRL** and **Miles** couple policy trainers and colocated vLLM rollout servers inside a multi-process **Ray cluster**, taking **~224 s** on local storage (**~17 minutes** on network storage) to restart. Live-editing a Ray RL cluster introduces three consistency traps:
1. **Detached Ray actor instances:** Ray serializes `@ray.remote` actor classes into isolated worker processes at startup, so standard module reloaders never update the live actor objects already running in memory.
2. **Coupled trainer and rollout weight drift:** An RL step updates policy weights on the trainer and broadcasts them to the rollout servers. Resetting only the trainer leaves the rollout servers generating trajectories from mutated weights and stale KV caches.
3. **Pre-queued stale trajectories (in asynchronous RL like Miles):** Decoupled rollout actors populate an experience buffer asynchronously, leaving trajectories generated by older code versions queued in the buffer after an edit.

Instead of restarting the Ray cluster, RSIBox broadcasts each patch to all Ray workers, locates the active Ray actor instances directly in each worker's Python heap (`gc.get_objects()`), and rebinds their methods in place. During `reset`, RSIBox restores the trainer's parameter snapshot and re-syncs weights to the colocated vLLM rollout engines while flushing their KV caches in **`~2.5 s`**, cutting a full advantage-estimation check in **veRL** from **~224 s down to ~9 s (~24x faster locally, >100x on NFS)**. In **Miles**, RSIBox pairs **`< 0.1 s` multi-actor code swaps** with a version-tagged rollout buffer that automatically drains trajectories stamped by older code versions upon commit.

<!-- <details>
<summary><b>View implementation: In-heap Ray actor method rebinding and version-tagged rollout buffer drain</b></summary>

```python
def hot_reload_ray_actor_method(actor_cls_name: str, method_name: str, new_fn, journal):
    # Locate live detached Ray actor instances directly in the worker's Python heap
    for obj in gc.get_objects():
        if type(obj).__name__ == actor_cls_name:
            old_fn = getattr(type(obj), method_name)
            journal.record_fn(old_fn, old_fn.__code__, old_fn.__defaults__)
            old_fn.__code__, old_fn.__defaults__ = new_fn.__code__, new_fn.__defaults__

def drain_stale_rollouts_on_commit(rollout_buffer, new_code_version: int):
    # In asynchronous RL (Miles), discard trajectories generated before the committed edit
    rollout_buffer.retain_only(lambda sample: sample.code_version == new_code_version)
```
</details> -->

### Cross-System Evaluation and End-to-End Optimization Results

Figure 4b and the table below summarize how RSIBox compresses single-edit evaluation latency across all six systems on NVIDIA GB200 and B200 GPUs (hover over each row for model and hardware details):

{{< image src="img/fig4-cross-engine.png" alt="Single-edit testing latency across six GPU systems: baseline restart vs. RSIBox" width="100%" title="Figure 4b. Single-edit evaluation time across six AI systems on NVIDIA GB200 and B200 GPUs, comparing standard cold or warm restarts against RSIBox live hot reload and targeted testing." >}}

| System & Representative Edit | Baseline Restart | RSIBox Live Check | Speedup |
| --- | --- | --- | --- |
| <span title="4-bit Nemotron 3.5 Super, 1x NVIDIA GB200 (full breakdown in Section 3.3)" style="border-bottom: 1px dotted currentColor; cursor: help;">**vLLM:** Scheduler / Triton / CUDA kernel edits</span> | 52 to 228 s | **1.1 to 23.6 s** (`2.4 s` probe) | **Up to 95x** |
| <span title="Selective decode CUDA graph recapture across TP=2 workers, 2x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**SGLang:** Decode CUDA graph edit and selective recapture</span> | ~25 s | **0.84 s** | **~30x** |
| <span title="Live code swap + 3-step training test and GPU state snapshot restore, 2x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**Megatron-LM:** Live swap + 3-step training check + state restore</span> | ~58 s | **0.5 s** (`~5.4 s` full pass) | **> 100x** (`~8x` vs. warm restart) |
| <span title="MiniMax H3 sparse attention edit + full video generation check, 1 to 2x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**FastVideo:** Attention / Triton kernel edit + video generation check</span> | ~68 to 72 s warm / ~8.5 min cold | **~6 to 7 s** | **~11x** warm / **~75x** cold |
| <span title="Distributed Ray RL cluster advantage edit + coupled trainer/rollout weight reset, 4x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**veRL:** Distributed Ray RL advantage edit + coupled weight reset</span> | ~224 s local / ~17 min NFS | **~9 s** | **~24x** local / **> 100x** NFS |
| <span title="Multi-actor Ray RL live code swap (< 0.1 s) + 1-step GRPO rollout/train/weight-sync check, Qwen3-0.6B on 4x NVIDIA GB200" style="border-bottom: 1px dotted currentColor; cursor: help;">**Miles:** Multi-actor Ray RL live swap + 1-step GRPO check</span> | 60 to 95 s (`2 to 3.5 min` on 8B+) | **2.5 to 8.0 s** (`< 0.1 s` actor swap) | **~12x to 24x** |

Finally, Figure 5 compares head-to-head autonomous optimization races across five AI systems (six workloads) under identical coding agents, prompts, and GPU budgets. In every race, every promoted patch must preserve exact output or numerical equivalence, and final winners are re-verified from scratch in fresh cold processes without RSIBox attached:

{{< image src="img/fig5-hillclimb.png" alt="RSIBox End-to-End Evaluation across five AI systems: vLLM, SGLang, Megatron-LM, FastVideo T2VA and Ref2VA, and Miles" width="100%" title="Figure 5. Head-to-head autonomous optimization races across five AI systems (six workloads), plotting verified speedup over starting code (y-axis) against wall-clock hours (x-axis) under RSIBox versus a restart baseline. Arrows show how much sooner RSIBox reaches the best baseline's final speedup; dashed lines show replicate runs." >}}

- **LLM serving (`vLLM` and `SGLang`):** On 6-hour GB200 decode races, both RSIBox runs beat both baseline runs on **vLLM** (`1.112x` to `1.113x` vs. `1.098x` to `1.106x`, reaching the best baseline's final speedup **`1.5x` sooner** and `2.3x` sooner against race 1). On **SGLang**, RSIBox reaches the baseline's final speedup **`3.1x` sooner** and finishes at **`1.76x` vs. `1.37x`** cold-verified.
- **Distributed training (`Megatron-LM`):** On 6-hour **Qwen3-30B-A3B** (`4x GB200`) pretraining, both RSIBox runs outperform both baseline runs (`+9.8%` and `+11.2%` vs. `+8.4%` and `+8.7%`), passing the best baseline's final throughput gain **`1.9x` sooner**. In standalone zero-restart GB200 campaigns, discovered patches also lifted cold-verified throughput by **`+5.1%`** on **Qwen3-14B** and transferred **`+7.1%`** to **Qwen3-32B**.
- **Video generation (`FastVideo`):** Across 8-hour **MiniMax H3** races on DGX Spark, RSIBox reaches the baseline's final speedup **`1.4x` sooner** on both **T2VA** (`1.95x` vs. `1.91x`) and **Ref2VA** (`1.38x` vs. `1.26x`). In standalone GB200 campaigns (>230 edits, zero cold restarts), cold-verified clip latency dropped **`1.45x` on 1x GB200** and **`1.52x` on 2x GB200** with bit-identical frames and audio.
- **Reinforcement learning (`Miles`):** Across 6-hour **Qwen3-0.6B GRPO** races on `2x GB200`, the best RSIBox run reaches the best baseline's final `2.56x` speedup **`1.3x` sooner** and finishes at **`2.73x`** (with replicate runs shown as dashed curves).

---

## RSIBox: Open Source and What's Next

We are open-sourcing **RSIBox** at [github.com/hao-ai-lab/rsibox](https://github.com/hao-ai-lab/rsibox), including:

1. **Live-editable sandboxes for six production AI systems:** **vLLM** and **SGLang** (LLM serving), **Megatron-LM** (distributed training), **FastVideo** (video generation), and **veRL** and **Miles** (distributed reinforcement learning).
2. **Uniform sandbox runtime and safety layer:** The `install`, `check`, `reset`, and `uninstall` interface covering live code swapping, selective CUDA graph recapture, GPU state snapshots, automatic rollback, and warm crash recovery.
3. **Toolsmith onboarding workflow:** Agent skills and verification harnesses for mining edit boundaries and building new RSIBox adapters on other GPU codebases.

### Supporting NVIDIA Dynamo Snapshot

Making GPU infrastructure warm and fault-tolerant is a broader open-source effort, and for LLM inference, **NVIDIA Dynamo** already provides strong foundations that are very close to what live-editable LLM inference systems need: GPU Memory Service (`GMS`) keeps model weights resident in GPU memory across restarts, and Dynamo Snapshot checkpoints and restores multi-GPU inference engines across crashes.

RSIBox natively supports NVIDIA Dynamo and builds directly on top of it. Combining NVIDIA Dynamo (`GMS` + `Snapshot`) with RSIBox's millisecond code hot-swapping, live execution checks, and rollback lets researchers run large-scale, stateful experiments on frontier models—applying live code edits in `~50–200 ms`, while relying on Dynamo to keep weights warm and handle fast process recovery (`~2.5–15 s` when a fault happens). Looking ahead, we are excited to keep working with the NVIDIA Dynamo team and the open-source community to deepen this integration and turn it into a shared infrastructure foundation for large-scale RSI experimentation.

{{< image src="img/fig6-dynamo-snapshot.png" alt="Building RSIBox on top of NVIDIA Dynamo (Snapshot + warm GMS) across four frontier TP4 LLMs on NVIDIA B200" width="100%" title="Figure 6. RSIBox built on top of NVIDIA Dynamo (Snapshot + warm GMS) across four frontier TP4 LLMs on NVIDIA B200 (151.5–491.1 GiB; CUDA Graphs and FlashInfer autotuning enabled). Dynamo cuts multi-GPU recovery from 8–31 min cold start down to 9.3–15.4 s; built on top, RSIBox applies live 1-line code edits in 90–99 ms." >}}

### What's Next

Making existing GPU engines live-editable is only the first step. Going forward, we are focusing on four directions:

- **Service-decomposed, agent-native AI infra:** Today RSIBox retrofits live editability onto engines built around monolithic batch processes. As coding agents become primary contributors to AI infrastructure, future engines should natively separate long-lived GPU state services from hot-swappable execution layers. We already have follow-up work underway on this direction and will share more soon.
- **Swarm-scale parallel evaluation sandboxes:** RSIBox speeds up a single edit-test loop by 10x–100x, and we have already built an initial shared-weight sandbox that lets multiple isolated workers share read-only weights on the same GPU. Scaling to large agent swarms still brings real fault-tolerance and resource-contention challenges when processes share hardware, though many of them can be mitigated specifically for development workloads. We are actively refining this design and testing it on large-scale swarms to harden it in practice.
- **Full-stack coverage across distributed boundaries:** Live hot-swapping paired with second-level snapshots already covers in-process edits and fast crash recovery in inference. Bringing the same fast fallback guarantees to multi-node training and distributed RL boundaries will let agents sustain multi-day autonomous optimization across the entire AI stack.
- **Beyond GPU engines to other fields:** Slow startup and heavy persistent state bottleneck iterative development far outside LLM infrastructure, from robotics simulation to formal verification and database systems. We are actively generalizing RSIBox across these domains while polishing and co-evolving both the framework and the coding agents together to see how far live-editable loops can push autonomous discovery.

**More live-editable AI infrastructure for RSI is coming soon, stay tuned.**

---

## Contributors

[Junda Chen](https://github.com/GindaChen), [Yi Sun](https://www.linkedin.com/in/yi-sun-mlsys), [Yuxuan Zhang](https://yuxuandexter.github.io/), [Yuanbo Yang](https://freemty.github.io/), [Junda Su](https://davids048.github.io/), [Will Lin](https://github.com/SolitaryThinker), [Hao Zhang](https://haozhang.ai)

## Citation

```bibtex
@misc{rsibox2026,
  title        = {Towards Live-Editable {LLM} Infra for {RSI}},
  author       = {Chen, Junda and Sun, Yi and Zhang, Yuxuan and Yang, Yuanbo and Su, Junda and Lin, Will and Zhang, Hao},
  year         = {2026},
  month        = nov,
  howpublished = {Hao AI Lab Blog},
  url          = {https://haoailab.com/blogs/rsibox/}
}
```
