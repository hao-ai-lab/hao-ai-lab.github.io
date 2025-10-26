+++
title = "Disaggregated Inference: The Past, Present and Future"
date = 2024-02-22T00:00:00-08:00
authors = ["Junda Chen", "Yonghao Zhuang", "Hao Zhang"]
author = "Junda Chen, Yonghao Zhuang, Hao Zhang"
ShowReadingTime = true
draft = true
[cover]
    image = "/img/distserve_anime-crop.gif"
    alt = "DistServe"
    caption = "A request going through an LLM serving engine with disaggregated prefill and decode"

+++

# Disaggregated Inference: The Past, Present and Future

**Disaggregated inference** has become one of the core technologies powering modern large language model (LLM) serving systems. By decoupling different stages of inference into independently scalable components, it enables **low latency, high throughput, and efficient resource utilization** in large-scale distributed deployments. Production-grade frameworks such as NVIDIA Dynamo, llm-d, SGLang, vLLM, LMCache, MoonCake have demonstrated the power of this approach in real-world environments, and many more continue to push the boundaries of what disaggregation can achieve. 

Behind the scenes, much of this architectural shift can be traced back to a simple but powerful idea we explored in our research system [**DistServe**](https://hao-ai-lab.github.io/blogs/distserve/): that the two fundamental phases of LLM inference - *prefill* and *decode* - should be treated as separate services. By introducing and formalizing **prefill-decode disaggregation**, DistServe showed how isolating these stages could dramatically reduce tail latency, improve utilization, and unlock independent scaling. That concept, once a research prototype, has since become the architectural foundation underpinning many of today’s production serving frameworks.

In this blog, we delve into the past, present, and future of disaggregated inference. We first revisit how the idea of disaggregated inference emerged, then examine how it has evolved into the infrastructure that powers today’s large-scale systems, and finally explore where it is heading next as LLM workloads continue to grow in scale and complexity.



## **Past: From Monolithic Serving to Prefill-Decode Disaggregation**

The shift toward disaggregated inference was a response to fundamental limitations in the way LLM serving systems were originally built. DistServe, our research system, was among the first to formalize prefill-decode disaggregation and transform the LLM serving stack from a monolithic execution path into a composition of specialized services, each optimized and scaled for its unique workload characteristics.


### **DistServe: A New Architecture for Inference**

[DistServe](https://hao-ai-lab.github.io/blogs/distserve/) starts from a simple observation: LLM inference is not a single homogeneous workload. Every request goes through two fundamentally different phases - a **prefill** phase, which processes the input prompt in one dense forward pass, and a **decode** phase, which generates output tokens one by one. These two phases have fundamentally different characteristics: prefill is **compute-bound** and saturates the GPU even with a small batch, while decode is **memory-bound**, requiring much larger batches to approach peak efficiency.

Instead of forcing prefill and decode to share the same GPU, DistServe disaggregates them into different sets of GPUs. This delivers two critical benefits:

- **Eliminates prefill-decode interference:** Prefill bursts no longer delay ongoing decoding, and long-lived decode sessions no longer block new requests from entering the system.
- **Enables independent scaling:** Resource allocator can provision resources to match the temporal dynamics of each phase - scaling prefill capacity to absorb traffic spikes without wasting decode resources, or expanding decode capacity to support thousands of concurrent sessions without impacting prefill latency.



### **Bottlenecks in Conventional LLM Serving**

Before DistServe, most serving frameworks relied on **continuous batching**, a technique popularized by systems like [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), [vLLM](https://dl.acm.org/doi/10.1145/3600006.3613165), and [Sarathi](https://www.usenix.org/conference/osdi24/presentation/agrawal). Multiple requests were grouped into a single forward pass to improve GPU utilization - an effective solution when workloads were simple and latency targets were loose. But as deployment scales grew and strict user-facing SLOs emerged, two inherent bottlenecks became impossible to ignore.

#### **1. Prefill-Decode Interference**

Colocating both stages on the same GPU inevitably led to **resource contention**. Compute-intensive prefills arriving during long decode sessions stalled token generation, inflating per-token latency. Conversely, memory-heavy decoding workloads blocked new prefills from being scheduled promptly. The result was a cascade of idle gaps and wasted cycles - a system may fully utilize the GPU but suffered from degraded responsiveness in practice. 

DistServe fundamentally solved prefill–decode interference by disaggregating the two phases onto separate GPU pools, allowing each to progress independently. Prefill bursts no longer interrupt ongoing decoding, and decode sessions can sustain steady throughput without being preempted by new prefills.

{{< image src="img/01-interference.png" alt="interference" width="100%" title="Figure 2. Colocation leads to interference between prefill and decode. In contrast, disaggregation allows each phase to progress independently, thereby eliminating interference.">}}



#### **2. Enabling Independent Resource Scaling**

A single monolithic service also meant a coupled single resource pool for both prefill and decode phases. Because prefill and decode have very different scaling demands, resource allocators were forced to choose between (a) **overprovisioning** resources to handle peak bursts, with the cost of wasting capacity and inflating cost, or (b) **underprovisioning** resources, with the risk of SLO violations. The inability to scale each stage independently was a structural inefficiency that no scheduling or kernel optimization could fundamentally overcome.

{{< image src="img/02-resource-colocate.png" alt="scaling" width="100%" title="Figure 3(a). Colocation forces resource allocators to choose between over- or under-provisioning, leading to inefficiencies.">}}


DistServe fundamentally solves this problem by isolating prefill and decode into independently scalable services, enabling fine-grained resource allocation that matches each phase’s unique demand profile and eliminates wasteful over- or under-provisioning. On top of this, DistServe also finds the optimal configuration of available compute and network hierarchies to maximize throughput under SLO constraints, making large-scale LLM serving both more efficient and more predictable in production environments.

{{< image src="img/02-resource-disagg.png" alt="scaling" width="100%" title="Figure 3(b). Disaggregation enables independent scaling of prefill and decode, making resource allocation strategy tailored for each phase.">}}



In summary, by disaggregating prefill and decode into independently scalable services, DistServe fundamentally solved both performance interference and resource coupling. This architectural shift marked a clean break from monolithic serving - and quickly proved transformative for large-scale LLM deployments.

## **Present: Disaggregated Inference as the Major Trend**

What began as a radical architectural idea has now become the standard playbook for large-scale LLM serving. Virtually all production-grade frameworks - spanning orchestration layers, inference engines, storage systems, and even emerging hardware architectures - now embrace some form of prefill-decode (P/D) disaggregation, making it one of the defining principles of the modern inference stack.

**Orchestration Layer**

[**NVIDIA Dynamo**](https://www.nvidia.com/en-us/ai/dynamo/)**.** Announced at NVIDIA GTC 2025, Dynamo is one of the most advanced and mature data center scale distributed inference frameworks for p/d disaggregation. [Dynamo](https://github.com/ai-dynamo/dynamo) is open source and supports popular inference engines like [TensorRT-LLM](https://docs.nvidia.com/tensorrt-llm/index.html), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang). Dynamo introduces prefill and decode workers as first-class citizens, and connects these workers using a KV-aware router to efficiently route requests between p/d instances. It treats prefill and decode workers as first-class services, linked by a KV-aware Router for efficient scheduling and routing. A centralized GPU Planner continuously profiles GPUs to auto-scale resources and balance workloads. NVIDIA Dynamo also includes NVIDIA [**NIXL**, ](https://github.com/ai-dynamo/nixl)a low latency point-to-point inference transfer library that unifies NVLink, InfiniBand, PCIe, and SSD fabrics under a single abstraction layer. Dynamo achieved state-of-the-art results on the [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) rack scale architecture at the [SemiAnalysis InferenceMax](https://developer.nvidia.com/blog/nvidia-blackwell-leads-on-new-semianalysis-inferencemax-benchmarks/) and [MLPerf Inference](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) benchmarks. More recently, NVIDIA announced [Rubin CPX](https://developer.nvidia.com/blog/nvidia-rubin-cpx-accelerates-inference-performance-and-efficiency-for-1m-token-context-workloads/), a new accelerated compute hardware architecture that fully embraces p/d disaggregation to serve long-context inference workload. This is the testimony that shows disaggregation has influenced both software and hardware architectures for LLM inference. 

{{< image src="img/nvidia-dynamo.png" alt="nvidia-dynamo" width="100%" title="Figure 4(a). [NVIDIA Dynamo](https://www.nvidia.com/en-us/ai/dynamo/) architecture diagram.">}}

[**llm-d**](https://github.com/llm-d/llm-d)**.** Built with Kubernetes-native primitives like Gateway API Inference Extension and LeaderWorkerSet, llm-d integrates disaggregated serving into the Kubernetes operational model. It separates prefill and decode variants as independently scalable services with strict SLO control, enabling elastic deployment across heterogeneous GPU clusters. llm-d focuses on policy-driven routing and autoscaling and failure isolation within containerized, multi-tenant environments - a step toward bringing disaggregation to general cloud inference platforms. Together, llm-d brings amazing performance on [llama4 moe models](https://llm-d.ai/blog/llm-d-v0.2-our-first-well-lit-paths#improved-benchmarking-suite) and [models with wide EP](https://llm-d.ai/blog/llm-d-v0.3-expanded-hardware-faster-perf-and-igw-ga).

{{< image src="img/llm-d.png" alt="afd" width="100%" title="Figure 4(b). [llm-d](https://github.com/llm-d/llm-d) architecture diagram.">}}


**Storage Layer**

[**LMCache**](https://github.com/LMCache/LMCache), developed by the University of Chicago team, focuses on accelerating the storage layer performance for KV cache centric LLM inference applications. LMCache optimizes p/d disaggregation by optimizing the KV cache movement from prefill to decode instance, including batched data movement operations, compute and I/O pipelining (see the [tech report](https://arxiv.org/abs/2510.09665)).

{{< image src="img/lmcache.png" alt="afd" width="100%" title="Figure 4(c). [LMCache](https://github.com/LMCache/LMCache) architecture diagram.">}}

[**MoonCake**](https://github.com/kvcache-ai/Mooncake), developed by the **Kimi AI** team as both an academic project ([FAST’25 best paper](https://www.usenix.org/conference/fast25/presentation/qin)) and now another industry standard open-source project, features a KVCache-centric p/d disaggregated platform for LLM serving. Mooncake pools the underexploited storage mediums as a centralized KV cache abstraction such that any prefill can hand off to any decode anywhere in the cluster. 

{{< image src="img/mooncake.png" alt="afd" width="100%" title="Figure 4(d). [Mooncake](https://github.com/kvcache-ai/Mooncake) architecture diagram.">}}

Today, both LMCache and Mooncake have become one of the standard storage backends for large-scale LLM inference. 

**Core Engine Layer**

P/D disaggregation also rely on the core inference engine to support most of the major features, and most of the major open-source LLM inference engines including [**vLLM**](https://docs.vllm.ai/en/stable/features/disagg_prefill.html) ([with NVIDIA Dynamo + LMCache](https://blog.lmcache.ai/2025-04-29-pdbench/)) and [**SGLang**](https://lmsys.org/blog/2025-05-05-large-scale-ep/) ([Deepseek-R1 on 96*H100](https://lmsys.org/blog/2025-05-05-large-scale-ep/), on [Deepseek on GB200](https://lmsys.org/blog/2025-09-25-gb200-part-2/)) have support for disaggregated inference. Both vLLM and SGLang has integration with most of the orchestration frameworks and storage frameworks in the community. 

{{< image src="img/04-deepseek-sglang-pd-arch.png" alt="afd" width="100%" title="Figure 4(e). Deepseek-V3/R1 reference p/d architecture from [SGLang](https://lmsys.org/blog/2025-05-05-large-scale-ep/).">}}

**Deepseek** is also an early adopter of [p/d disaggregation](https://arxiv.org/abs/2412.19437) as their core inference stack. Deepseek-v3 uses prefill-decode disaggregation combined with different parallelisms for prefill and decoding instances. Amazingly, they also open source many useful frameworks to the open source community. For example, their [3FS](https://github.com/deepseek-ai/3FS) framework is a cost-effective SSD-based distributed storage as the centralized KV cache store for inference. [DeepEP](https://github.com/deepseek-ai/DeepEP) optimizes expert-parallel communication by offering high-throughput, low-latency all-to-all GPU kernels with support for FP8, and efficiently overlaps communication and computation without consuming SM resources. ￼These contributions effectively lowered the barrier for practitioners to build high-performance disaggregated inference systems at scale, with a much lower entry cost to hardware and network. 

In addition, many companies including [Meta](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/), [Amazon](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/disaggregated-inference.html), [Modular](https://docs.modular.com/mammoth/disaggregated-inference/) also have their own LLM serving framework supporting p/d disaggregation. The widespread adoption of prefill-decode disaggregation and its remarkable performance gains in production over the past year stand as the strongest testament to the power of disaggregation in the LLM serving era.

### **Academic Works and the Generalization of Disaggregation**

Perhaps the closest concurrent academic work with DistServe is [Splitwise](https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/) and [ShuffleInfer](https://dl.acm.org/doi/10.1145/3732941). [Splitwise](https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/) also focuses on the architectural perspective that separates P/D, and uses heterogeneous hardware (H100 vs A100) to achieve better energy efficiency. [ShuffleInfer](https://dl.acm.org/doi/10.1145/3732941) also studies p/p and d/d interference, and proposes to group requests of different lengths into buckets to further isolate the interference. 

Other academic work also extend or improve disaggregation to optimize KV cache transfer ([CacheGen](https://dl.acm.org/doi/10.1145/3651890.3672274), [MemServe](https://arxiv.org/abs/2406.17565)), parallelism configurations ([Mitra et al](https://research.nvidia.com/publication/2025-06_beyond-buzz-pragmatic-take-inference-disaggregation)), scheduling ([SLO-Serve](https://arxiv.org/abs/2504.08784)), disaggregation/aggregation fusion ([TaiChi](https://arxiv.org/abs/2508.01989)), multimodal serving ([ModServe](https://arxiv.org/abs/2502.00937)), heterogeneous hardwares ([Helix](https://dl.acm.org/doi/abs/10.1145/3669940.3707215), [HexGen-2](https://arxiv.org/abs/2502.07903), [CENT](https://dl.acm.org/doi/abs/10.1145/3676641.3716267)), network ([FuseLink](https://www.usenix.org/conference/osdi25/presentation/ren)), RL ([StreamRL](https://arxiv.org/pdf/2504.15930)), power efficiency ([EcoServe](https://arxiv.org/abs/2502.05043), [GreenLLM](https://arxiv.org/abs/2412.20322)), and more. Together, these works demonstrate the rapid expansion of disaggregation as a general systems principle that continues to evolve across diverse contexts in LLM inference and beyond.

## **Future - A/F Disaggregation: A New direction of Disaggregated Inference**

Beyond prefill-decode disaggregation, a quieter but equally intriguing frontier is attention-FFN (A/F) disaggregation. Traditionally, A/F disaggregation has been considered impractical: unlike P/D, which transfers the KV cache only once, A/F separation requires frequent back-and-forth communication across every transformer layer - incurring prohibitive network overhead.

[MegaScale-Infer](https://dl.acm.org/doi/10.1145/3718958.3750506) and its concurrent work in [Stepfun](https://arxiv.org/abs/2507.19427) challenges this long-held assumption. By leveraging Mixture-of-Experts (MoE) architectures, where each FFN already communicates via an all-to-all pattern, the system reinterprets the cost model of disaggregation: as long as the A/F split communication is faster than the existing MoE all-to-all, the additional overhead becomes tolerable - even advantageous. MegaScale-Infer exploits this by disaggregating attention and MLP within each layer, scaling them independently according to their compute-memory ratios, and overlapping them in a ping-pong pipeline to hide latency. In doing so, MegaScale-Infer exposes both the opportunity and the current limitation of A/F disaggregation: its efficiency hinges on architectures that already tolerate intensive communication - which makes MoE a natural fit, but not dense models which still face prohibitive synchronization costs.

In principle, A/F disaggregation is the next logical step beyond P/D: an attempt to decouple the structural components of a transformer based on the characteristics of each component, rather than its temporal phases. To overcome the overhead, we believe that future systems may need to drastically reduce communication cost through model or even architectural & hardware innovations, or develop scheduling strategies that exploit latency-insensitive or throughput-bound workloads. 

{{< image src="img/03-stepfun_afd_timeline.png" alt="a-f-disagg" width="100%" title="Figure 5. A/F disaggregation timeline from [StepFun-ai/StepMesh](https://github.com/stepfun-ai/StepMesh/) library.">}}

## **Conclusion**

From its early conception in DistServe, disaggregation has evolved from a radical architectural experiment into a defining principle of modern LLM inference. DistServe opened the door - demonstrating that disaggregating prefill and decode into independent services could eliminate long-standing inefficiencies in monolithic serving. In the years since, both academia and industry have carried this idea forward, and continue to expand what disaggregation can mean across different layers of the stack, delivering over 10x speedup for many applications and use cases.As we look ahead, new forms such as A/F disaggregation point toward an even more fine-grained future - one where compute is not merely scaled, but re-architected around the semantics of each component. 

The journey of disaggregated inference is far from complete: there remain open questions about scheduling, hardware co-design, caching hierarchy, and latency tolerance. Yet, the trajectory is clear - disaggregation is no longer a niche optimization, but a guiding paradigm for scalable, efficient, and adaptable AI infrastructure.
