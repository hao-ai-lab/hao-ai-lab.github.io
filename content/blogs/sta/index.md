+++
title = "Fast Video Generation with Sliding Tile Attention"
date = 2025-02-17T12:00:00-08:00
authors = ["Peiyuan Zhang", "Yongqi Chen", "Runlong Su", "Hangliang Ding", "Ion Stoica", "Zhengzhong Liu", "Hao Zhang"]
author = "Peiyuan Zhang, Yongqi Chen*, Runlong Su*, Hangliang Ding, Ion Stoica, Zhengzhong Liu, Hao Zhang"
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/FastVideo"
[cover]
      image = "img/attention_visualization.jpg"
      alt = "Attention visualization"
      caption = "Visualization of attention locality and the Sliding Tile Attention mechanism."
+++

{{< socialBadges arxiv-index="2502.04507" github="https://github.com/hao-ai-lab/FastVideo" >}}

{{< justify >}}

**TL;DR:** Video generation is **painfully slow**—HunyuanVideo takes 16 minutes to generate a 5-second video on an H100 GPU. We cut this down to **5 minutes** without sacrificing quality, all without training.

**Abstract:**  The bottleneck? Attention. It accounts for 13 of the 16 minutes required for video generation. We observe that attention scores in video diffusion models are highly localized in 3D space, with different heads needing different window sizes. This makes sliding window attention (SWA) an attractive alternative to retain full attention's expressiveness while reducing computational cost. However, 3D SWA is GPU-unfriendly and fails to translate FLOP savings into real speedups. We introduce **Sliding Tile Attention (STA)**, the first higher-order local attention with *efficient hardware implementation*.  Unlike SWA, STA operates tile-by-tile with a novel hardware-aware sliding window design. STA accelerates attention by **2.8–17×** over FlashAttention-2 (FA2) and **1.6–10×** over FlashAttention-3 (FA3). We design a calibration stategy to determine the optimal window size for each attention head. With STA and other optimizations, our solution boosts end-to-end generation speed by **2.98×** compared to the FA3 full attention baseline, without quality loss or the need for training.

Can you guess which of the videos below are from the original HunyuanVideo, and which of them are from our accelerated infernece solutions？ 


{{< /justify >}}



## Attention in Video DiTs

{{< justify >}}

State-of-the-art Video DiTs use 3D full attention to model relationships across tokens, allowing each token to attend to any other token across spatial and temporal dimensions. However, modern video models generate an extremely large number of tokens. For instance, in HunyuanVideo, a 5-second 720P video corresponds to 115K tokens. The problem worsens as video resolution or duration increases. Assuming a video of shape (L, L, L) (where temporal and spatial dimensions are equal for simplicity), a slight increase in L leads to a cubic growth in the number of tokens. Since attention has a quadratic complexity in sequence length, this rapid expansion makes attention the primary computational bottleneck. As illustrated in Figure 1 (a), attention computation dominates the overall inference cost.

{{< /justify >}}

{{< image src="img/attn_latancy.png" alt="Attn Latency" width="70%" title="Figure 1: (a) Generating a 5s 720P clip in Hunyuan involves processing 115K tokens, making attention the dominant cost. (b) Attention latency comparison: existing methods fail to translate FLOP reduction into wall-clock speedup; STA is hardware-efficient and achieves proportional speedup with sparsity">}}

{{< justify >}}
We hypothesize that significant redundancy exists in the 3D full attention pattern, which, if leveraged effectively, can greatly accelerate inference. To test this, we visualize the attention scores of HunyuanVideo in Figure 2 (left). The results reveal a strong 3D locality pattern: queries tend to focus their attention on spatially and temporally nearby keys. To quantify this effect, we compute attention recall—-the fraction of total attention scores concentrated within a local window. As shown in Figure 2 (middle), despite being trained with full 3D attention, HunyuanVideo exhibits strong locality: a local window covering only 15.52% of the total space captures 70% of the total attention score. Interestingly, while different attention heads show varying degrees of locality, their bevarious are generally static across different prompts. In Figure 2 (right), we evaluate 10 different prompts and compute the standard deviation of attention recall across prompts for each head. The results show consistently low variance, indicating that the locality pattern for each head remains stable.


{{< /justify >}}

{{< image src="img/attn_is_sparse.png" alt="Attn Sparsity" width="100%" title="Figure 2. Left: Instead of attending to the entire image, the query (green dot)’ only attends to keys within a local window. Mid: Attention scores within the local window accouts for mojority of the entire attention. Right: Despite the different recall across heads, the standard deviation across prompts remains low.">}}


{{< justify >}}
The analysis above suggests a clear strategy to make attention more efficient: replacing full 3D attention with localized attention in video generation models. This approach, known as sliding window attention (SWA), has been widely explored in natural language processing for 1D sequences. However, the challenge is far from solved—-there is no efficient 2D or 3D implementation of SWA! As shown in Figure 1 (right), existing sliding window implementations like CLEAR and NATTEN fail to convert FLOP reductions into actual speedups, due to their limited hardware utilization. We argue that higher-order sliding window attentino is inherrently imcompatible to FlashAttention (FA) and unfriendly to GPU, and we will give our arguments below.

{{< /justify >}}


## Inefficiency of Sliding Window Attention
{{< justify >}}
To understand why SWA is imcompatible with FlashAttention, one key concept to grasp is  FA's **block-by-block** computation pattern. FA splits the input sequence (Q, K, V) into small blocks, typically of size (128, 64). We assume square block size in this blog for simplicity. Instead of processing tokens individually, FA loads an entire block of Q, K, and V into GPU SRAM, performs the necessary computations, and writes back only the output matrix O to HBM—without storing intermediate values like attention masks or scores. As shown in Figure 3, we can think of FA as splitting the attention map into smaller blocks, where a single block is the minimum computation unit of attention. GPUs are essentially large matrix multiplication machines—they don’t handle scalar or even vector operations efficiently -- they like *block-by-block* computation, not *token-by-token* computation.
{{< /justify >}}

{{< image src="img/attn_map.png" alt="Attn Map" width="100%" title="Figure 3. The attention map of NATTEN, Tiled NATTEN, and STA. We plot with an image size 24×24 and a 12×12 local window. The tile size is set to 4×4. Note that we mainly use STA in 3D scenarios for video generation in this paper, but for better illustration, we present the 2D scenario in this plot.">}}

{{< justify >}}
Implementing 2D/3D SWA with FlashAttention requires defining its attention mask. Based on how the mask is applied within a block, we categorize attention blocks into three types: dense (with all
attention scores retained), empty (mask out all values), and mixed (with some scores removed). Dense and empty blocks are efficient: dense blocks are highly efficient, while empty blocks can be skipped entirely.

However, mixed blocks are problematic because they introduce extra computation steps: 1. Compute the entire block. 2. Compute the mask for this block. 3. Apply the mask to filter out unwanted attention scores. This masking process introduces significant overhead for two reasons. First, Since a block is the minimum computation unit, FA must compute the entire block before masking out parts of it—wasting compute. Second, the intra-block mask’s value depends on the user-defined attention pattern and the block’s location within the attention map. This mask calculation is GPU-unfriendly and cannot be precomputed (precomputing mask would lead to quadratic memory overhead). For reference, even a simple causal mask in FlexAttention introduces a 15% overhead. In 3D SWA, which has a far more complex masking pattern, the overhead can exceed the cost of computing the block itself.

That is why higher-order SWA is unfriendly to GPUs -- they produce too many mixed blocks! We plot the attention map of NATTEN in Figure 3 (a), an improved sliding window variant that shift window centers at image and video boundaries to ensure each query attends to a constant number of keys. In NATTEN, each query attends to a local window centered around it, resulting in different queries attending to distinct key groups. This lack of shared attention key groups is the root cause of irregularities in SWA’s attention map, creating mixed blocks. To mitigate this issue, Tiled NATTEN attempts to reorder inputs to increase the number of dense blocks (Figure 3(b)). However, a significant portion of blocks still remain mixed.


{{< /justify >}}


## Sliding Tile Attention

{{< justify >}}
The idea behind *Sliding Tile Attention (STA)* is simple: GPUs work best with *block-by-block* computations, but SWA slides its window *token-by-token*, which is inefficient. Our proposed STA fixes this by sliding *tile-by-tile*. In 3D scenarios, we define a tile as a contiguous group of tokens forming a spatial-temporal cube, with its size determined
by the block size in FlashAttention. This small change eliminates *mixed blocks* in the attention map and significantly improves computational efficiency.

- **SWA**: Moves *token-by-token*, creating irregular attention maps that GPUs struggle with.
- **STA**: Moves *tile-by-tile*, forming dense and empty attention blocks that are GPU-friendly.

Specifically, 
1. A video of size \(L, L, L\) is divided into non-overlapping tiles of size \(T, T, T\). Assuming Flash Attention's block size is \(B, B\), T should satisfy the condition  B = T^3.
2. Tokens within each tile are **flattend consecutively**. The window size should also be integer multiple of the tile size.
3. The attention window **moves tile-by-tile** with a step size of \(T, T, T\). For each local window, the central query tiles attend to keys within the window. 
4. This results in only dense and mixed blocks in the attention map, completely eliminating inefficient mixed blocks, as shown in Figure 3 (c).
{{< /justify >}}

The video below demonstrates how STA works and how it differs from SWA. For better illustration, we use a 2D scenario. In this example, we apply SWA and STA to a 12×12 image. SWA uses a (5,5) window size, while STA operates with (2,2) tiles and a (6,6) window.
{{<youtube AGXIt0DWfyM>}}

{{< justify >}}
 STA can be efficiently implemented with FlexAttention, which provides enough functionality to skip all empty blocks and avoid adding unnecessary *intra-block* mask on the dense blocks. We can further optimize the sparse attention masks by *disaggregating* the *inter-block* mask logic from the compute kernels. Thus, we implement our attention kernels based on ThunderKittens and FlashAttention3 . 
 
 Our implementation split the threadblock into compute warpgroups and data warpgroups, and the inter-block mask is completely managed by the data warpgroups. Each compute warpgroup is responsible for calculating one query block, which always resides in the SRAM (Split-Q). The data warpgroup is responsible for asynchronously loading the KV blocks from HBM to SRAM. For each block of query, the data warpgroup needs to decide which key and value blocks the query block will attend to in STA and only load those blocks. Since the data warpgroups are asynchronous, the overhead of calculating the inter-block mask in STA and deciding which data to load can be hidden with overlapping. On the other hand, the compute worker is completely oblivious of the sparse attention pattern. It performs attention computation with the key value blocks in shared memory loaded by data workers, and once all data is consumed in the circular cache, the computation is finished.

{{< /justify >}}
{{< image src="img/kernel_speed.png" alt="Kernel Speed" width="90%" title="Table 1. Forward speed of sparse attention kernels in a setup aligned with HunyuanVideo's inference configuration (bf16, 720P, 5s, 115.2K seq len, dhead = 128, # heads = 24). Config controls the window size of each sparse attention.">}}

We report our kernel performance in Table 1. The results show that existing local attention methods struggle with efficiency. For example, while CLEAR reduces FLOPs to 15.65, it actually slows down inference by 14%. NATTEN also falls short—despite achieving 91% sparsity, its basic version is 15% slower than full attention, and even the optimized tiled variant in FlexAttention only speeds things up by 1.27×. Among current options, Swin is the only kernel with a memory utilization factor (MFU) above 40% and kernel efficiency above 60%, but it sacrifices flexibility in the attention mechanism. But Swin is not a strictly local attention variant, and we will show in the next section that applying swin the video generation models significantly degrades performance. 

In contrast, when tested in FlexAttention, STA improves MFU from 8.20% to 41.03% compared to Tiled NATTEN. With our further kernel optimizations, STA achieves a 10.45× speedup over full attention. Even at 58.33% sparsity, it still delivers 2.37× faster processing. This means STA can handle larger attention windows while still outperforming NATTEN. To our knowledge, STA is the first method to combine efficient 3D sparse attention with real-world speed improvements.

## Window Size Calibration Enables Training-free Speedup
As shown ealier in Figure 2, video diffusion models exhibit strong 3D locality and head specialization. While different attention heads capture information at different scales, their locality patterns remain consistent across prompts. This allows us to search for an optimal window size per head using a small set of prompts and generalize the results to others. Specifically, for each (s, l, h) tuple—where s is the inference step index, l is the layer index, and h is the head index—we determine the best attention mask.

Since early sampling steps are crucial for global structure, we retain full attention for the first XX steps. For the remaining steps, we pick candidate masks from a predefined set by computing the L2 distance between their outputs and full attention outputs, selecting the mask with the lowest distance. Our video generation setup uses a 117×768×1280 resolution, translating to a DiT shape of 30×48×80. We set the tile size to 6×8×8 and select from window sizes [(30, 24, 24), (18, 24, 40), (6, 48, 80), (30, 48, 8), (30, 8, 80)]. We calibrate on 18 prompts, averaging the L2 distance across them to determine the best mask strategy per head. The entire search process completes in under 18 hours on a single H100 GPU.



We evaluate on 200 random prompts from MovieGen Bench. Below, we show **uncherry-picked** results (videos are strictly randomly sampled from the 200 generated videos).


We also quantitatively meature how close 

STA accelerates attention by exploiting redundancy in 3D full attention. Another approach to speeding up video generation focuses on caching, leveraging redundancy across diffusion steps. We demonstrate that STA is compatible with these methods, achieving even greater speedups when combined with TeaCache, a state-of-the-art diffusion acceleration technique based on caching.



## Train with STA Unlocks Greater Speedup



## Before We Finish...
{{< justify >}}
It might seem surprising that there was no efficient 2D/3D sliding window attention before STA. After all, sliding window attention is a fundamental algorithm—widely used in 1D contexts. Why hasn’t an efficient 2D/3D implementation existed until now?

To make you believe this claim, consider the Swin Transformer. The authors recognized that sliding window attention lacked an efficient 2D implementation. Their solution? Simply avoid it. Instead of true sliding windows, Swin uses non-overlapping and static window partitions. However, this prevents queries and keys from attending across window boundaries, breaking the 3D locality crucial for video tasks. Since Swin is used in a pretraining setup, this limitation is addressed by using different window partitions across different layers and force the model to learn such pattern. However, in training-free or fine-tuning scenarios like ours, Swin performs suboptimally. Their proposed solution win won the Marr Prize at ICCV 2021. 

{{< /justify >}}

{{< image src="img/swin.png" alt="Swin" width="50%" title="Figure 4. An illustration of the shifted window approach for computing self-attention in the Swin Transformer.">}}

{{< justify >}}



{{< /justify >}}
## Conclusion 

This work makes the following contributions: (1) We identify and quantify 3D locality and head specialization in stateof-the-art video DiTs, revealing substantial redundancy in full 3D attention. (2) We introduce Sliding Tile Attention, a tile-based sliding window attention mechanism. Our optimized kernel achieves minimum overhead compared to
FlashAttention 3 with an MFU of 58.79%. (3) STA accelerates attention by > 10× and end-to-end video generation by up to 3.53× with no or minimum quality loss.

*We believe STA’s potential extends far beyond accelerating video diffusion models.* It can be applied in pretraining and generalized to other high-order data. Locality is a universal property across almost all data modalities. We hope STA inspires new, more efficient models across various domains.
## Acknowledgements

This work is greatly motivated by FlexAtteniton and NATEN. Our implementation is based on ThunderKitten's H100 attention kernel. We thank Yichao Fu, Junda Chen, and Lanxiang Hu for their feedback on this blog. 

## Citation

```
@misc{zhang2025sta,
      title={Fast Video Generation with Sliding Tile Attention},
      author={Peiyuan Zhang and Yongqi Chen and Runlong Su and Hangliang Ding and Ion Stoica and Zhengzhong Liu and Hao Zhang},
      year={2025},
      eprint={2502.04507},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

