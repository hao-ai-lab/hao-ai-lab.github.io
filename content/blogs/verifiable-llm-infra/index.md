+++
title = "Towards Verifiable LLM Infra"
date = 2026-11-10T00:00:00-08:00
authors = ["Yuxuan Zhang", "Minshen Zhang", "Junda Su", "Yi Sun", "Junda Chen", "Hao Zhang"]
author = "Yuxuan Zhang, Minshen Zhang, Junda Su, Yi Sun, Junda Chen, Hao Zhang"
ShowReadingTime = true
authorOnNewLine = true
draft = false
math = true
contentClass = "post-content-justified"
summary = "Suppose an AI agent is strong enough to build a fast inference engine from scratch; in this second half of the problem, the bottleneck shifts from capability to verification. We introduce vstack, a 5-stage formal verification workflow for coding agents, and veriserve, a prototype verifiable LLM serving engine with PD disaggregation built around a formally verified KV-page lifecycle design model, uncovering real RDMA concurrency bugs across SGLang and vLLM while adding only 0.18% mean p50 TPOT overhead on NVIDIA GB200."

[cover]
  image = "img/cover.png"
  relative = true
  alt = "Towards Verifiable LLM Infra"
  caption = "Towards Verifiable LLM Infra"
  hidden = true

[socialIcons]
  [[socialIcons.icon]]
    name = "twitter"
    url = "https://twitter.com/haoailab"
+++

<div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 4px 0 18px;">
  <a href="https://github.com/hao-ai-lab/vstack" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/GitHub-vstack-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub: vstack" style="height: 28px; margin: 0;">
  </a>
  <a href="https://github.com/hao-ai-lab/veriserve" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/GitHub-veriserve-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub: veriserve" style="height: 28px; margin: 0;">
  </a>
  <a href="https://x.com/haoailab" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Follow-@haoailab-000000?style=for-the-badge&logo=x&logoColor=white" alt="X: @haoailab" style="height: 28px; margin: 0;">
  </a>
</div>

<figure style="margin: 14px 0 28px;">
  <video controls preload="metadata" poster="img/poster.png" style="width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; display: block;">
    <source src="https://github.com/hao-ai-lab/hao-ai-lab.github.io/releases/download/website-demo-media-v2/verifiable-llm-infra--img--overview.mp4" type="video/mp4">
  </video>
</figure>

Coding agents are getting remarkably good at writing LLM infrastructure, from GPU kernels and schedulers to custom serving engines. Suppose an AI agent is strong enough to build a fast inference engine from scratch; in this [second half](https://ysymyth.github.io/The-Second-Half/) of the problem, the bottleneck shifts from capability to **verification**: how do we know the engine it wrote is actually correct? Passing latency and throughput benchmarks still leaves rare concurrency bugs hidden. As a single engineer begins orchestrating 100, 1,000, or even 10,000 coding agents in parallel, human code review hits a hard cognitive wall. Having agents review other agents does not solve this either; as [`pstack`](https://github.com/cursor/plugins/tree/main/pstack) author [Lauren Tan noted](https://x.com/poteto/status/2117437607940681849), adversarial reviews are mostly a waste of tokens, and it is far better to spend those tokens on empirical and formal verification.

In this post, we introduce [**`vstack`**](https://github.com/hao-ai-lab/vstack), a 5-stage formal verification workflow for coding agents, and [**`veriserve`**](https://github.com/hao-ai-lab/veriserve), a prototype **verifiable LLM serving engine** with PD disaggregation built around a formally verified KV-page lifecycle design model. Applying `vstack` to production Prefill-Decode serving uncovered three classes of real RDMA and cache concurrency bugs across SGLang and vLLM, resulting in **6 proposed upstream pull requests**. Meanwhile, `veriserve` backs its `PdLedger` design model with a **226-module, 516,973-obligation [`TLAPS`](https://proofs.tlapl.us/) safety proof** while adding only **`0.18%` mean p50 `TPOT` overhead** on NVIDIA GB200. **We believe verifiable serving defines the second half of LLM infra as autonomous coding agents write the deepest layers of production systems.**

---

## Why We Need Verifiable Serving

Why is verifying an LLM serving engine so hard in the first place? Over the past three years, inference engines like [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) have evolved from single-node executors into asynchronous distributed systems, layering [**paged KV memory**](https://arxiv.org/abs/2309.06180), [**shared prefix caching**](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html), and [**disaggregated prefill-decode RDMA transfers**](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) over the same pool of physical GPU KV pages. When a request times out or aborts, its cleanup now races across RDMA network cards, CPU cache trees, and GPU attention kernels simultaneously. Because [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/) writes directly into GPU memory without involving the target CPU, a lifecycle mismatch often does not crash the server, but returns `HTTP 200 OK` while silently mixing one user's KV cache into another user's response. Recent [LMSYS](https://www.lmsys.org/blog/2026-10-26-cf-security/) and [Cloudflare](https://blog.cloudflare.com/kv-cache-exposure-remediation/) disclosures highlight this risk in production, documenting both cross-node RDMA abort races and a rejected-request KV handoff bug that exposed data across **155 user responses**.

Worse yet, **catching these silent concurrency bugs on real hardware is slow and prohibitively expensive**. Standard CI unit tests run on a single machine and only cover known paths, while subtle distributed races under rare asynchronous interleavings rarely surface in CI. Reproducing an RDMA lifecycle race requires multi-GPU RDMA hardware like H100 or GB200, separate prefill and decode instances, and tight cross-worker timing. Flushing them out with long stress tests on GPU clusters burns scarce compute and still cannot guarantee that all race conditions are gone.

To catch these deep concurrency bugs without burning GPU hours, we turn to **formal verification**: modeling the system as a state machine and combining `TLA+` model checking with [`Z3`](https://github.com/Z3Prover/z3) and [`TLAPS`](https://proofs.tlapl.us/) theorem provers to exhaustively explore millions of asynchronous schedules and prove safety invariants on CPUs. In the past, formal verification was impractical for AI infrastructure because engines like vLLM and SGLang merge new pull requests continuously, while manually writing and updating formal specifications and proofs takes months of human effort. **Today, coding agents give us a practical way to bridge this gap.** Because LLMs can read large systems codebases and generate thousands of lines of formal specifications and proofs far faster than manual proof engineering, we can guide coding agents to specify, model-check, and prove critical subsystems, making **verifiable serving** practical in fast-moving open-source projects.

---

## How We Verify: Formal Verification with `vstack`

Turning agent-driven formal verification into a reliable workflow requires solving two immediate challenges. First, the hardest part of formal verification is not running the solver, but writing an accurate **Specification**: a requirement the software must never violate under any asynchronous schedule. Thanks to open-source [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang), we do not invent these requirements in a vacuum, but ground them directly in real user issues, bug-fix PRs, and GPU kernel preconditions.

Second, even with a clear specification, simply asking a coding agent to write a formal model and prove it rarely works out of the box. When a model checker reports a race condition, **a naive coding agent will often reward-hack the verifier**: instead of surfacing the real bug, it weakens the invariant, deletes the timeout branch from the model, or adds an unrealistic `ASSUME` statement so the checker returns `0 errors`.

{{< image src="https://github.com/hao-ai-lab/hao-ai-lab.github.io/releases/download/website-demo-media-v2/verifiable-llm-infra--img--fig1.gif" alt="The vstack 5-stage verification workflow" width="100%" title="Figure 1. The 5-stage vstack verification workflow. Common agent reward-hacks such as weakening the rule, deleting the timeout branch, or assuming the race away are blocked by pinned specifications, fidelity checks, and proof gates until a real fix passes." >}}

To prevent reward hacking and make agent-driven verification reliable, we built [**`vstack`**](https://github.com/hao-ai-lab/vstack), an open-source plugin for Claude Code and Cursor that enforces a **5-stage verification workflow**:

1. **`Specification`:** Encode software requirements as formal safety invariants and pin the specification files by hash so unapproved edits to the proof goal are detected and rejected.
2. **`Abstraction`:** Build `TLA+` and `Z3` models where every state transition maps to an exact `file:line@commit` in the production codebase, splitting non-atomic cross-thread and cross-node operations into separate steps.
3. **`Validation`:** Before trusting any proof, verify model fidelity against real code: the historical buggy code must fail on the model, the patched code must pass, removing a guard from the fix must fail again under nonredundant mutation checks, and recorded execution traces must replay cleanly through the model.
4. **`Verification`:** Run [`TLC`](https://docs.tlapl.us/using:tlc:start) breadth-first search to find the shortest counterexample trace within the bounded model, then use [`Apalache`](https://apalache-mc.org/), [`Z3`](https://github.com/Z3Prover/z3), and [`TLAPS`](https://proofs.tlapl.us/) to prove that the modeled state machine and GPU KV-buffer address arithmetic hold under our explicit configuration assumptions.
5. **`Continuous Checking`:** Re-run the specification suite on new pull requests on CPUs, and mirror the proven preconditions as lightweight runtime checks in the engine.

---

## Case Study: Formally Verifying Prefill-Decode Disaggregated Serving

As a prototype, we applied `vstack` to **Prefill-Decode Disaggregated Serving**, where a Prefill node computes the prompt's KV cache and transfers it over one-sided RDMA to a Decode node that shares cached prefixes and runs the Decode attention kernel, as shown in **Figure 2**. For this core KV page subsystem, we define **four specifications**:

{{< image src="https://github.com/hao-ai-lab/hao-ai-lab.github.io/releases/download/website-demo-media-v2/verifiable-llm-infra--img--fig2.gif" alt="Silent KV corruption under disaggregated Prefill-Decode RDMA" width="100%" title="Figure 2. Silent KV corruption under disaggregated Prefill-Decode RDMA. If Decode reclaims request A's GPU page on a local timeout and assigns it to request B while Prefill's one-sided RDMA write is still in flight, A's late payload overwrites B's KV page and the server still returns HTTP 200." >}}

1. **`Isolation`:** In-flight RDMA transfers or late control messages from an old request must never read or overwrite a new request's memory or state.
2. **`Integrity`:** Every KV page read by the Decode kernel or published into the global prefix cache must contain complete, uncorrupted data from the intended request generation.
3. **`Finality`:** Every active request and draining memory page must reach a terminal state in finite steps rather than hanging forever.
4. **`Conservation`:** Every physical page is strictly accounted for: a page exposed to in-flight RDMA returns to `FREE` only after hardware stop evidence arrives, or else stays permanently isolated in `QUARANTINED` accounting.

### Uncovering Real Concurrency Issues in SGLang and vLLM

Checking the production Prefill-Decode implementations of [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm) against these four specifications on CPUs uncovered **three classes of concurrency issues**, for which we reproduced selected RDMA failure paths on NVIDIA GB200 and control-plane deadlocks on CPUs, and submitted **6 proposed upstream pull requests**:

- **Reclaiming Transfer Slots or KV Blocks While RDMA Is Still In Flight:** When a request aborts or times out, one-sided RDMA transfers already queued on the NIC cannot be canceled in hardware on the studied transport paths; yet both engines allowed new requests to reuse the same transfer identity or reallocate the expired GPU page before the hardware transfer stopped, violating **`Isolation`**. On GB200, we reproduced this race in both SGLang and vLLM, where late RDMA accesses silently read or overwrote another request's KV cache under `HTTP 200 OK`. We proposed upstream fixes in SGLang [`#48883`](https://github.com/sgl-project/sglang/pull/48883), [`#48889`](https://github.com/sgl-project/sglang/pull/48889), [`#48960`](https://github.com/sgl-project/sglang/pull/48960), [`#48979`](https://github.com/sgl-project/sglang/pull/48979), and [`#48982`](https://github.com/sgl-project/sglang/pull/48982), alongside a related vLLM pull-validation PR [`#62065`](https://github.com/vllm-project/vllm/pull/62065) by [`@vadiklyutiy`](https://github.com/vadiklyutiy).
- **Publishing Race-Corrupted KV Pages into the Global Prefix Cache:** When an RDMA transfer finishes, the Decode node can insert the received KV pages into the global prefix cache without checking whether the page's lifecycle generation was invalidated mid-flight, violating **`Integrity`**. In our vLLM GB200 experiment, race-corrupted KV pages entered the prefix cache and corrupted a subsequent request that reused the cached prefix. This motivates tracking versioned page ownership consistently across the RDMA transfer and prefix-cache boundary.
- **Unrenewed Transfer Leases and Hung Decode Requests:** When a remote transfer stalls or an abort notification is dropped, unrenewed leases and missing watchdogs leave Decode requests waiting indefinitely while holding their allocated GPU pages, violating **`Finality`** and **`Conservation`**. We reproduced stalled RDMA leases on GB200 and scheduler capacity deadlocks in CPU control-plane replay, and proposed an upstream lease-renewal and receive-timeout fix in vLLM [`#66443`](https://github.com/vllm-project/vllm/pull/66443).

**Why Local Timeouts Fail:** The deepest lifecycle failures above share a fundamental root cause: **using local CPU timeouts or error codes to guess when asynchronous hardware RDMA has finished**. Using [`TLAPS`](https://proofs.tlapl.us/) and [`Z3`](https://github.com/Z3Prover/z3), we formally proved that under unbounded asynchronous delay and indistinguishable local histories without trustworthy stop or revocation evidence, *any* local-only page reclamation policy cannot guarantee both **`Isolation`** if it reclaims the page early and eventual page reclamation under **page-level `Finality`** if it holds the page forever.

### Building `veriserve`: A Verifiable LLM Serving Engine

Because no local-only timeout policy can escape this trade-off, current engines like SGLang and vLLM remain vulnerable when their RDMA connector, prefix cache, and decode scheduler maintain separate local counters over raw `page_id` integers and guess when hardware RDMA is done. To fix this by design, we built [**`veriserve`**](https://github.com/hao-ai-lab/veriserve), a verifiable LLM serving engine with PD disaggregation on [`mini-sglang`](https://github.com/sgl-project/mini-sglang) and [Mooncake](https://github.com/kvcache-ai/Mooncake) GPUDirect RDMA centered on a single capability state machine, **`PdLedger`**, which tracks every physical page via a versioned ticket `(page, gen)` across six explicit implementation states: **`FREE`**, **`PRIVATE`**, **`SHARED`**, **`DRAINING`**, **`QUARANTINED`**, and **`DUP`**, with **Figure 3** illustrating the four states on the RDMA timeout and cancellation paths.

{{< image src="https://github.com/hao-ai-lab/hao-ai-lab.github.io/releases/download/website-demo-media-v2/verifiable-llm-infra--img--fig3.gif" alt="PdLedger page lifecycle resolving the local-timeout dilemma" width="100%" title="Figure 3. How PdLedger resolves the local-timeout dilemma on the RDMA transfer path. Without stop evidence, a timed-out page is isolated in Quarantined so late RDMA writes never touch new requests; with stop evidence, a canceled page waits in Draining and returns to Free only after the remote transfer stops. Prefix-cache states Shared and Dup are omitted for clarity." >}}

Our key design delta is using **`DRAINING`** and **`QUARANTINED`** to decouple *request cancellation* from *hardware DMA completion*, resolving the request-hang-versus-corruption trade-off:

- **Release with Stop Evidence → `FREE`:** Releasing a private page with outstanding RDMA or kernel accesses moves it to `DRAINING` first. Once hardware completion or peer acknowledgement confirms all tracked accesses have stopped, `PdLedger` returns the page to `FREE`, while each new request incarnation receives a fresh generation identifier `gen`, preserving **`Conservation`**.
- **Timeout without Stop Evidence → `QUARANTINED`:** When a transfer times out without stop evidence, `PdLedger` terminates the affected waiting request so it does not hang, while permanently isolating that exposed physical page in **`QUARANTINED`** for the lifetime of the process so any late RDMA packet lands harmlessly outside `FREE` and the prefix cache, preserving **`Isolation`** and **`Integrity`**.

Trusting the [`TLAPS`](https://proofs.tlapl.us/) and [`Z3`](https://github.com/Z3Prover/z3) verifiers, a formal proof establishes safety for the abstract design model under its explicit assumptions, not that our Python implementation, GPU kernels, or RDMA providers are bug-free. To keep the Python engine aligned with the model, `veriserve` checks `PdLedger` preconditions synchronously at runtime, rejects injected state-transition mutations in our mutation gates, and logs Prefill and Decode boundary transitions for step-by-step `TLC` trace validation.

---

## Evaluation

We evaluated `veriserve` across both CPU formal verification of the `PdLedger` design model and end-to-end GPUDirect RDMA serving on a 4-GPU NVIDIA GB200 node with `Qwen3-0.6B` and `Qwen3-32B`, running separate Prefill and Decode instances on one GPU each over Mooncake:

- **Formal Safety Proof on CPUs:** [`TLC`](https://docs.tlapl.us/using:tlc:start) model checking explored **21M+ reachable states** on bounded configurations with zero safety violations. Under the explicit `Sizes` and `Design` assumptions of our specification, we completed a **226-module, 516,973-obligation [`TLAPS`](https://proofs.tlapl.us/) induction proof** across all 64 protocol actions and 54 inductive invariants, paired with [`Z3`](https://github.com/Z3Prover/z3) proofs of the modeled 6D GPU KV-buffer address arithmetic across page sizes 1, 16, and 64.
- **100% Trace Validation Across `22,104` Boundary Events:** By causally merging asynchronous Prefill and Decode runtime logs into versioned page steps, **100% of `22,104` mapped boundary events across eight normal serving runs** replayed cleanly through `TLC`.
- **Zero Checksum Errors Under Cache & Fault Tests:** Across eight single-page prefix-sharing and cache-eviction runs, `PdLedger` verified **540 KV checksums with zero mismatches and zero ledger violations**, while runtime checks caught **all three injected glue faults**.
- **Under `0.7%` Mean p50 `TPOT` Overhead on GB200:** On [Alibaba `qwen-bailian`](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) workloads with `Qwen3-32B`, Decode-side `PdLedger` checks take **`0.33 ms` of CPU time per decode step at p50** against a **`13.6 ms` median HTTP `TPOT`**. Across six paired runs against an unverified bypass baseline, `PdLedger` adds **`+0.18%` mean p50 `TPOT` overhead**, with a **95% confidence interval from `-0.32%` to `+0.68%`**.

---

## Open Source and What's Next

We are open-sourcing all three parts of our work:

1. [**`vstack`**](https://github.com/hao-ai-lab/vstack): Our 5-stage formal verification plugin for Claude Code and Cursor.
2. [**`veriserve`**](https://github.com/hao-ai-lab/veriserve): Our verifiable LLM serving engine on [`mini-sglang`](https://github.com/sgl-project/mini-sglang) and [Mooncake](https://github.com/kvcache-ai/Mooncake) RDMA, including `PdLedger`, the `TLA+`, [`TLAPS`](https://proofs.tlapl.us/), and [`Z3`](https://github.com/Z3Prover/z3) models and proofs, and the trace validation suite.
3. **Upstream Fixes for [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm)**: Concurrency fixes proposed across SGLang [`#48883`](https://github.com/sgl-project/sglang/pull/48883), [`#48889`](https://github.com/sgl-project/sglang/pull/48889), [`#48960`](https://github.com/sgl-project/sglang/pull/48960), [`#48979`](https://github.com/sgl-project/sglang/pull/48979), [`#48982`](https://github.com/sgl-project/sglang/pull/48982) and vLLM [`#66443`](https://github.com/vllm-project/vllm/pull/66443).

Verifying the KV page lifecycle of Prefill-Decode disaggregated serving is just the first step. Our formal verification work on hierarchical KV caching such as [SGLang HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/), multi-page prefix trees, and low-level GPU kernels is next. **More verifiable serving is coming soon, stay tuned.**

---

## Acknowledgement

We thank NVIDIA for supporting our development and GPUDirect RDMA evaluation on NVIDIA GB200. We thank the SGLang, mini-sglang, vLLM, and Mooncake communities for building the open-source disaggregated serving stack and reviewing our upstream bug reports and fixes, as well as the Qwen team and Alibaba for releasing the Qwen3 models and `qwen-bailian` traces used in our evaluation. Our verification workflow builds on the open-source TLA+, TLC, Apalache, TLAPS, and Z3 ecosystems, and draws inspiration from Lauren Tan's `pstack`.

## Contributors

[Yuxuan Zhang](https://yuxuandexter.github.io/), [Minshen Zhang](https://alexzms.github.io/), [Junda Su](https://davids048.github.io/), [Yi Sun](https://www.linkedin.com/in/yi-sun-mlsys), [Junda Chen](https://github.com/GindaChen), [Hao Zhang](https://haozhang.ai)

## Citation

```bibtex
@misc{vstack2026,
  title        = {Towards Verifiable {LLM} Infra},
  author       = {Zhang, Yuxuan and Zhang, Minshen and Su, Junda and Sun, Yi and Chen, Junda and Zhang, Hao},
  year         = {2026},
  month        = nov,
  howpublished = {Hao AI Lab Blog},
  url          = {https://haoailab.com/blogs/verifiable-llm-infra/}
}
```
