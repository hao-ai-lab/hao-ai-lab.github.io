+++
title = "Fast Video Generation with Sliding Tile Attention"
date = 2025-02-11T12:00:00-08:00
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

**TL;DR:** Video generation is **painfully slow**—HunyuanVideo takes 16 minutes to generate a 5-second video on an H100 GPU. We cut this down to XX minutes without sacrificing quality, all without requiring retraining.

**Abstract:**  The bottleneck? Attention. It accounts for 13 of the 16 minutes required for video generation. Our key observation: attention scores in video diffusion models are highly localized in 3D space, with different heads needing different window sizes. This makes sliding window attention (SWA) an attractive alternative to retain full attention's expressiveness while reducing computational cost. However, 3D SWA is GPU-unfriendly and fails to translate FLOP savings into real speedups. We introduce **Sliding Tile Attention (STA)**, the first higher-order local attention with **efficient hardware implementation**.  Unlike SWA, STA operates tile-by-tile with a novel hardware-aware sliding window design. STA accelerates attention by **2.8–17×** over FlashAttention-2 (FA2) and **1.6–10×** over FlashAttention-3 (FA3). We design a calibration stategy to determine the optimal window size for each attention head. With that, our solution increase the generation speed by **XX**  without quality loss, requiring no training.

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
The analysis above suggests a clear strategy to make attention more efficient: replacing full 3D attention with a localized attention mechanism. This approach, known as sliding window attention (SWA), has been widely explored in natural language processing for 1D sequences. However, the challenge is far from solved—-there is no efficient 2D or 3D implementation of SWA! As shown in Figure 1 (right), existing sliding window implementations like CLEAR and NATTEN fail to convert FLOP reductions into actual speedups, due to their poor hardware utilization. I will actually argue that sliding window attentino is inherrently unfriendly to GPU, and I will give my arguments below from the perspective of Flash Attention.

{{< /justify >}}


## Inefficiency of Sliding Window Attention
### Some Intuition on Flash Attention
We do not expect you to fully understand Flash Atteniton to understand what we are doing. But what you do need to understand is flash attention's block by block computation pattern: FA splits the input sequence (Q, K, V) into many small segments, and the size is typicall 64 or 128. To compute attention, FA loads a single segment of Q, K, V into GPU SRAM, perform computation, and only write back the output matrix O to HBM without any intermediate values (masks, attention scores..). Intuitively, you can treat the computation of such segment as the minimum computation unit for FA


Implementing 2D/3D SWA with FlashAttention requires defining its attention mask. (Yes, we made the assumption that we have to use FA, it is 2025, if you kernel does not run as fast as FA, no one is going to use your kernel). 


### Sliding Tile Attention

{{< justify >}}
STA overcomes these limitations by leveraging a **tile-based attention** mechanism that aligns with modern GPU architectures. Instead of computing attention for individual tokens, STA **groups tokens into spatial-temporal tiles** and slides over these tiles in a structured manner. This approach eliminates expensive masking operations, reducing memory overhead and improving hardware utilization.
{{< /justify >}}

{{< image src="img/sta_vs_natten.png" alt="STA vs NATTEN" width="70%" title="Figure 2: Comparison of Sliding Tile Attention and prior methods. STA eliminates inefficient mixed blocks, leading to substantial speedups.">}}
## Window Size Calibration for Each Head

{{< justify >}}
One of STA’s advantages is its **plug-and-play** nature: it can replace full attention in pretrained video DiTs **without requiring retraining**. By simply swapping full attention with STA, we achieve **1.36× to 2.59× speedup** without degrading video quality. Further fine-tuning STA under more aggressive sparsity constraints yields up to **3.53× speedup** with minimal perceptual quality loss.
{{< /justify >}}

## Experimental Results

### Speedup on Video Generation
{{< justify >}}
We benchmark STA on **HunyuanVideo**, a state-of-the-art open-source video DiT, comparing against FlashAttention-3, Tiled NATTEN, and Swin attention. Results show that **STA achieves up to 10.45× acceleration over full attention** while maintaining video quality comparable to the original model.
{{< /justify >}}

{{< image src="img/sta_speedup.png" alt="STA speedup" width="70%" title="Figure 3: STA achieves significant speedup in attention computation while preserving quality.">}}

### Human Evaluation and Quality Metrics
{{< justify >}}
We conducted a large-scale human study on 200 prompts from **MovieGen Bench**, evaluating video quality preferences between STA and baseline methods. The results indicate **STA is preferred over Δ-DiT and Tiled NATTEN while achieving superior efficiency**. Additionally, VBench scores confirm that STA preserves fidelity even under aggressive sparsification.
{{< /justify >}}

## Get Started

{{< justify >}}
For more details, check out our **[paper](https://arxiv.org/abs/2502.04507)** and explore our **[GitHub repository](https://github.com/hao-ai-lab/SlidingTileAttention)**. Try our pre-trained STA-enabled video generation models on **[Hugging Face](https://huggingface.co/sta-video-models)**!
{{< /justify >}}

## Before We Finish...
{{< justify >}}
I know the claim that there is no efficient 2D/3D sliding window attention before STA may still sound to be very sub-real, after all, sliding window attention seems to be such a simple/basic algorithm, how come it is wildely used in 1D but no efficient 2D/3D implementation? To make you really believe, I can provide another evidense: There is a paper called swin transformer that win the Marr Prize in ICCV 2021. Basically they find that sliding window attention has no efficient implementation on 2D, and the solution they propose is simply not use sliding window atteniton. As shown in Figure
{{< /justify >}}

{{< image src="img/swin.png" alt="Swin" width="70%" title="An illustration of the shifted window approach for computing self-attention in the Swin Transformer.">}}



## Acknowledgements

This work is greatly motivated by FlexAtteniton and NATEN. Our implementation is based on ThunderKitten's FA3 kernel. We thank XXX 
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

