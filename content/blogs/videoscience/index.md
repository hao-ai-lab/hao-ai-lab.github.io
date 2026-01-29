+++
title = "From Physical Commonsense to Scientific Reasoning: Why World Modeling in Video Matters"
date = 2026-01-19T12:00:00-08:00
authors = ["Lanxiang Hu", "Abhilash Shankarampeta", "Yixin Huang", "Zilin Dai", "Haoyang Yu", "Yujie Zhao", "Haoqiang Kang", "Daniel Zhao", "Tajana Rosing", "Hao Zhang"]

ShowReadingTime = true
draft = false

[cover]
image = "img/header.png"
alt = "scientific reasoning in video world models"
caption = "Video generations are getting impressively realistic, but scientific correctness is a different bar. VideoScience-Bench evaluates whether video models behave like faithful simulators, not just good renderers."

[[socialIcons.icon]]
name = "twitter"
url = "https://twitter.com"

[[socialIcons.icon]]
name = "github"
url = "https://github.com/hao-ai-lab/VideoScience"
+++

<style>
.post-cover,
.post-cover img,
figure.post-cover img,
.cover-image,
img[alt="scientific reasoning in video world models"] {
  margin-left: auto !important;
  margin-right: auto !important;
  display: block !important;
  text-align: center !important;
}

.post-cover {
  text-align: center !important;
}
</style>

{{< socialBadges arxiv-index="2512.02942" github="hao-ai-lab/VideoScience" huggingface="TBD" >}}

{{< justify >}}
TL;DR: The golden age of AI video has mastered the "look" of reality, but it has yet to learn the laws of reality. Without adhering to rigorous scientific principles, even the most photorealistic model remains a high-fidelity hallucination engine rather than a reliable world simulator. To bridge this gap, we introduce VideoScience-Bench: the first benchmark specifically designed to move beyond "physical commonsense" and evaluate undergraduate-level scientific reasoning in video models.

This blog post introduces VideoScience-Bench, a benchmark designed to evaluate undergraduate-level scientific understanding in video models, and VideoScience-Judge, a scalable VLM-as-a-judge pipeline that evaluates generated videos against rigorous scientific criteria. Correlation analysis shows that VideoScience-Judge achieves the strongest alignment with expert-rated rankings and best captures a video model’s scientific reasoning capability in comparison with existing benchmarks.
{{< /justify >}}

{{< two_images
src1="img/phygenbench_3-4.gif"
src2="img/vid_087_run_2.gif"
alt1="physical_commonsense_world_modeling"
alt2="scientific_reasoning_world_modeling"
width1="50%"
width2="50%"
title="Figure 1: Left: A video model generating a physically plausible scene based on everyday commonsense ('A balloon is floating over a serene and mirror-like ocean.', Source: PhyGenBench). Right: A video generation task that requires scientific reasoning, where correct outcomes depend on multiple interacting laws rather than visual plausibility alone. ('A clear plastic water bottle has a small hole in its side, from which a smooth, laminar stream of water is flowing. A red laser pointer is aimed from the other side of the bottle, directly through the water and into the hole.', Source: VideoScience-Bench)."
>}}

## Video Model Reasoning and World Modeling

{{< justify >}}
What makes this moment particularly exciting is that video models are exhibiting zero-shot reasoning abilities, tackling tasks that involve scientific, mathematical, and spatial reasoning ([Veo 3](https://arxiv.org/abs/2509.20328)). Unlike their predecessors, which often exhibited hallucinations of object permanence, these models are beginning to demonstrate "Chain-of-Frames" reasoning, materializing thought as a sequence of frame-by-frame visual representations.
{{< /justify >}}

### Spatial Reasoning and Puzzle Solving

{{< justify >}}
This reasoning capability is most evident in spatial and puzzle-solving domains. Recent benchmarks, such as [VR-Bench](https://imyangc7.github.io/VRBench_Web/) and [VideoThinkBench](https://thinking-with-video.github.io/), have demonstrated that video models can now solve mazes, mentally rotate objects, and handle spatial obstacle avoidance with surprising accuracy.

<!-- - Maze solving examples: https://imyangc7.github.io/VRBench_Web/ -->
<!-- - Puzzle reasoning examples: https://thinking-with-video.github.io/ -->

<!-- {{< image src="https://raw.githubusercontent.com/giusha12i/Thinking-with-Video/refs/heads/main/assets/main_picture.png" alt="vrbench" width="85%" title="Examples from VRBench.">}} -->

{{< image src="https://thinking-with-video.github.io/assets/main_picture.png" alt="thinking_with_video" width="75%" title="Thinking with Video: Examples from VideoThinkBench.">}}

These achievements mark a pivotal shift: video models are no longer just generators, they're becoming reasoners.
{{< /justify >}}

### Simulation and Robotics Applications

{{< justify >}}
In recent context of [World Model Roadmap](https://world-model-roadmap.github.io/), [WorldSimBench](https://iranqin.github.io/WorldSimBench.github.io/), video generation is increasingly framed as: implicit world model (physics + dynamics) + renderer (pixels). In this view, video models aren’t only content engines, they could be simulation engines. If the simulator is scientifically wrong, downstream systems trained on it can inherit those failures.

The stakes for scientific accuracy are highest in robotics, where models must evolve from simple visual generators into reliable world simulators. Industry leaders like 1X and NVIDIA are developing world models, such as [1X-WorldModel](https://www.1x.tech/discover/1x-world-model) and [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/), that function as virtual simulators, leveraging raw sensor data to predict complex material interactions and envision potential futures. Because these systems generate the massive datasets used to train physical AI at scale, their adherence to scientific laws is a critical prerequisite for the safety and effectiveness of robots in the real world.
{{< /justify >}}

## Scientific Reasoning as the Next Step

{{< justify >}}
Spatial reasoning and puzzles are an important leap forward, but to be broadly useful, video models must evolve from Physical Commonsense (everyday intuition) to Scientific Reasoning (the rigorous application of multiple interacting principles).

Physical commonsense tests often ask: “Does this look plausible?”. On the other hand, scientific reasoning asks: “Does this obey the laws that govern the system, even when multiple concepts interact?”
{{< /justify >}}

### The Current Gap: from Physical Commonsense to Scientific Reasoning

{{< justify >}}
Existing benchmarks primarily assess physical commonsense, the kind of basic, everyday intuition that allows us to recognize a bouncing ball or a falling stream of water. While these tests ensure a model understands simple gravity and reflections, they only scratch the surface of true world modeling. To be truly useful for scientific discovery or robotic safety, a model must move beyond these single-principle scenarios and demonstrate an undergraduate-level understanding of science.
{{< /justify >}}

## VideoScience-Bench

{{< justify >}}
VideoScience-Bench is (to our knowledge) the first benchmark designed to evaluate whether video models can faithfully simulate multi-concept scientific phenomena, not just “look realistic.”

Unlike commonsense-based evaluations, each challenge in VideoScience-Bench requires:

- Multiple interacting concepts: Understanding how specific heat capacity and heat transfer principles work together to explain why a water-filled balloon doesn't pop when exposed to flame
- Undergraduate-level knowledge: Spanning 103 concepts across 14 topics in physics and chemistry
- Complex reasoning: Moving beyond single-principle scenarios to cascading effects where second-order dynamics matter
  {{< /justify >}}

{{< image src="img/science_atlas_category.png" alt="videoscience_categories" width="70%" title="Subcategory frequency of questions From VideoScience-Bench.">}}

</div><div style="display: flex; justify-content: space-between; gap: 10px; text-align: center;">
  <div style="flex: 1;">
   <img src="img/vid_094_run_2.gif" alt="Prince Rupert's Drop Tail Break" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> A teardrop-shaped piece of tempered glass is held at its bulbous head. Small pliers gently snip the thin tail end.<br><strong>Expected:</strong> The entire drop explosively shatters into powder as internal tension is released.</p>
  </div>
  <div style="flex: 1;">
    <img src="img/vid_137_run_3.gif" alt="The Ball and Cart" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> A clear plastic ruler is placed between two crossed polarizing filters and illuminated by a bright white light.<br><strong>Expected:</strong> The stressed plastic causes rotation of the light's polarization plane in a wavelength-dependent way, yielding colored interference fringes.</p>
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666;">
  Examples from VideoScience-Bench showing model-generated videos and expected phenomena.
</p>

<!-- <div style="display: flex; justify-content: space-between; gap: 10px; text-align: center;">
  <div style="flex: 1;">
    <img src="img/vid_094_run_2.gif" alt="Prince Rupert's Drop Tail Break" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> A teardrop-shaped piece of tempered glass is held at its bulbous head. Small pliers gently snip the thin tail end.<br><strong>Expected:</strong> The entire drop explosively shatters into powder as internal tension is released.</p>
  </div>
  <div style="flex: 1;">
    <img src="img/vid_177_run_2.gif" alt="Polarized Plastic Fringes" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> A clear plastic ruler is placed between two crossed polarizing filters and illuminated by a bright white light.<br><strong>Expected:</strong> The stressed plastic causes rotation of the light's polarization plane in a wavelength-dependent way, yielding colored interference fringes.
</p>
  </div>
</div>
<p style="text-align: center; font-style: italic; color: #666;">
  Examples from VideoScience-Bench showing model-generated videos and expected phenomena.
</p> -->


### Evaluation Metrics

{{< justify >}}
We don't just ask "Does the video look real?" We treat the video model as a simulation engine and evaluate it against five rigorous dimensions:

- Prompt Consistency: Does the experimental setup match the description?
- Phenomenon Congruency: Are the observed outcomes scientifically accurate?
- Correct Dynamism: Do motion and object interactions follow physical laws?
- Immutability: Do objects remain unchanged when they should?
- Spatio-Temporal Coherence: Are frame transitions smooth and temporally consistent?

This multidimensional framework enables us to pinpoint precisely where models succeed and where they fail, distinguishing between models that produce visually appealing but scientifically inaccurate videos and those that genuinely comprehend the underlying physics.
{{< /justify >}}

### Key Findings: Scientific Understanding and Reasoning are still Lacking

{{< justify >}}
Across seven state-of-the-art models Sora-2, Veo-3, Kling-v2.5-Turbo-Pro, Wan-2.5-T2V-Preview, Seedance 1.0 Pro, Hailuo 2.3, Ray2, we often see a recurring pattern:

While these models have mastered the aesthetics of reality, they often fail to grasp the physics of reality. We tested top models on VideoScience-Bench, a benchmark of 200 undergraduate-level science experiments, and found that even the best models frequently “hallucinate” physics.

{{< /justify >}}

#### The "Invisible Field" Failure: Great Visuals, Bad Physics

{{< justify >}}
The video looks perfect. The lighting is realistic, the motion is almost smooth (high Spatio-Temporal Coherence), and the object remains consistent (high Immutability). Yet, the scientific outcome is completely wrong. This is a failure of Phenomenon Congruency, the model knows what the objects are, but not how they interact.
{{< /justify >}}

<!-- </div><div style="display: flex; justify-content: space-between; gap: 10px; text-align: center;">
  <div style="flex: 1;">
    <img src="img/vid_095_run_1.gif" alt="The Spaghetti Mystery" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> "A dry spaghetti stick is held at both ends and slowly bent until it breaks."<br><strong>Expected:</strong> The spaghetti breaks into three or more pieces rather than two, because stress waves from the first fracture cause additional breaks before the fragments separate.</p>
  </div>
  <div style="flex: 1;">
    <img src="img/vid_137_run_3.gif" alt="The Ball and Cart" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> "A cart moves forward at a constant speed and launches a ball straight upward from its top."<br><strong>Expected:</strong> The ball travels upward and then downward in a parabolic path, but lands back on the moving cart because both the ball and the cart have the same horizontal velocity.</p>
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666;">
  Failure Examples on Violations of Phenomenon Congruency.
</p> -->
<div style="display: flex; flex-direction: column; gap: 20px;">
  <div style="text-align: center;">
    <img src="img/vid_095_run_3.gif" alt="The Spaghetti Mystery" style="width: 70%; max-width: 600px; display: block; margin: 0 auto;">
    <p style="font-size: 0.85em; margin-top: 10px;"><strong>Prompt:</strong> A dry spaghetti stick is held at both ends and slowly bent until it breaks.<br><strong>Expected:</strong> The spaghetti breaks into three or more pieces rather than two, because stress waves from the first fracture cause additional breaks before the fragments separate.</p>
  </div>
  <div style="text-align: center;">
    <img src="img/vid_137_run_3.gif" alt="The Ball and Cart" style="width: 70%; max-width: 600px; display: block; margin: 0 auto;">
    <p style="font-size: 0.85em; margin-top: 10px;"><strong>Prompt:</strong> A cart moves forward at a constant speed and launches a ball straight upward from its top."<br><strong>Expected:</strong> The ball travels upward and then downward in a parabolic path, but lands back on the moving cart because both the ball and the cart have the same horizontal velocity.</p>
  </div>
</div>
<p style="text-align: center; font-style: italic; color: #666; margin-top: 20px;">
  Failure Examples on Violations of Phenomenon Congruency.
</p>

{{< justify >}}
These two examples perfectly illustrate the current "uncanny valley" of AI video generation: High Visual Fidelity, Low Physical Logic. In both the spaghetti and the cart simulations, the model achieves impressive "Spatio-Temporal Coherence"—the lighting is realistic, the motion is smooth, and object identity is stable. However, both fail primarily in Phenomenon Congruency. The video generation model operates as a pattern-matcher rather than a physics engine; it knows what a breaking stick or a launching ball looks like in isolation, but it lacks the underlying understanding of material science (the "snap-back" effect in spaghetti) or Newtonian mechanics (conservation of momentum in the cart). The result is a video that looks perfect at a glance but falls apart under scientific scrutiny, revealing that the model is hallucinating motion rather than simulating reality.
{{< /justify >}}

#### The Universal Failure: When No One Gets It Right

{{< justify >}}
Some scenarios are so complex that they trigger a total collapse of reasoning across all models tested. These represent the current ceiling of zero-shot scientific reasoning.
{{< /justify >}}

<!-- <div style="display: flex; justify-content: space-between; gap: 10px; text-align: center;">
  <div style="flex: 1;">
    <img src="img/vid_124_run_1.gif" alt="The Chemical Traffic Light " style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> "A flask containing a yellow solution of glucose, sodium hydroxide, and the indicator indigocarmine is shown. The person lifts and gently shakes the flask."<br><strong>Expected:</strong> The yellow solution rapidly shifts toward green as the flask is shaken, showing the indicator’s partial oxidation by oxygen introduced from the air. Continued shaking drives the oxidation further and the color moves from green to red. When agitation stops, dissolved glucose reduces the indicator and the solution relaxes back to yellow.</p>
  </div>
  <div style="flex: 1;">
    <img src="img/vid_087_run_2.gif" alt="The Laser Fiber Optic " style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> "A clear plastic water bottle has a small hole in its side, from which a smooth, laminar stream of water is flowing. A red laser pointer is aimed from the other side of the bottle, directly through the water and into the hole."<br><strong>Expected:</strong> The laser beam enters the stream and becomes "trapped." It reflects repeatedly off the inner surface of the water stream, causing the entire parabolic arc of the falling water to glow red as if it were a fiber optic cable.</p>
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666;">
  Failure Examples on Violations of Prompt Conistency.
</p> -->

<div style="display: flex; flex-direction: column; gap: 20px;">
  <div style="text-align: center;">
    <img src="img/vid_124_run_1.gif" alt="The Chemical Traffic Light " style="width: 70%; max-width: 600px; display: block; margin: 0 auto;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> A flask containing a yellow solution of glucose, sodium hydroxide, and the indicator indigocarmine is shown. The person lifts and gently shakes the flask.<br><strong>Expected:</strong> The yellow solution rapidly shifts toward green as the flask is shaken, showing the indicator’s partial oxidation by oxygen introduced from the air. Continued shaking drives the oxidation further and the color moves from green to red. When agitation stops, dissolved glucose reduces the indicator and the solution relaxes back to yellow.</p>
  </div>
  <div style="text-align: center;">
    <img src="img/vid_087_run_2.gif" alt="The Laser Fiber Optic " style="width: 70%; max-width: 600px; display: block; margin: 0 auto;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> A clear plastic water bottle has a small hole in its side, from which a smooth, laminar stream of water is flowing. A red laser pointer is aimed from the other side of the bottle, directly through the water and into the hole.<br><strong>Expected:</strong> The laser beam enters the stream and becomes "trapped." It reflects repeatedly off the inner surface of the water stream, causing the entire parabolic arc of the falling water to glow red as if it were a fiber optic cable.</p>
  </div>
</div>

{{< justify >}}
The failure of these generated clips represents more than just a visual glitch; it is a total collapse of scientific causality. In the 'Traffic Light' reaction, the model successfully renders the liquid's texture but fails to link physical agitation to the chemical redox cycle—ignoring the oxygen-driven transition from yellow to green. Similarly, in the laser experiment, the model treats light as a 2D overlay rather than a physical entity governed by refractive indices. It fails to 'trap' the beam within the water stream via Total Internal Reflection, proving that while AI can mimic the appearance of reality, it still lacks a foundational world model of the laws of physics and chemistry.
{{< /justify >}}

<!-- <p style="text-align: center; font-style: italic; color: #666; margin-top: 20px;">
  Failure Examples on Violations of Phenomenon Congruency.
</p> -->

#### The "Complexity Collapse": Failing the Setup

{{< justify >}}
Before a model can simulate physics, it must build the setup. In sophisticated prompts, models often fail Prompt Consistency, they cannot even construct the experimental setup correctly, making the result inevitably wrong.
{{< /justify >}}

<!-- </div><div style="display: flex; justify-content: space-between; gap: 10px; text-align: center;">
  <div style="flex: 1;">
    <img src="img/vid_226_run_2.gif" alt="The Heart Motor" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> "Position a battery vertically on top of the three neodymium magnets so that the magnets contact the battery’s negative terminal. Place a heart-shaped copper wire so that it can touch both the top of the battery (positive terminal) and the sides of the magnets simultaneously."<br><strong>Expected:</strong> When the copper wire touches both ends of the circuit, it begins to spin or move continuously, creating a small, self-turning “heart motor.</p>
  </div>
  <div style="flex: 1;">
    <img src="img/vid_212_run_1.gif" alt="The Polymer Trick" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> "Put Superabsorbent polymer powder into one of the three opaque cups, pour water in them, and flip the three cups one after another."<br><strong>Expected:</strong> Only the cup containing the polymer retains the liquid, demonstrating the strong water-absorbing and gel-forming property of the superabsorbent polymer.</p>
  </div>
</div> -->

<!-- <p style="text-align: center; font-style: italic; color: #666;">
  Failure Examples on Violations of Prompt Congruency.
</p> -->

<div style="display: flex; flex-direction: column; gap: 20px;">
  <div style="text-align: center;">
    <img src="img/vid_226_run_2.gif" alt="The Heart Motor" style="width: 70%; max-width: 600px; display: block; margin: 0 auto;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> Position a battery vertically on top of the three neodymium magnets so that the magnets contact the battery’s negative terminal. Place a heart-shaped copper wire so that it can touch both the top of the battery (positive terminal) and the sides of the magnets simultaneously.<br><strong>Expected:</strong> When the copper wire touches both ends of the circuit, it begins to spin or move continuously, creating a small, self-turning “heart motor.</p>
  </div>
  <div style="flex: 1;">
    <img src="img/vid_212_run_1.gif" alt="The Polymer Trick" style="width: 70%; max-width: 600px; display: block; margin: 0 auto;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> Put Superabsorbent polymer powder into one of the three opaque cups, pour water in them, and flip the three cups one after another.<br><strong>Expected:</strong> Only the cup containing the polymer retains the liquid, demonstrating the strong water-absorbing and gel-forming property of the superabsorbent polymer.</p>
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666; margin-top: 20px;">
  Failure Examples on Violations of Prompt Congruency.
</p>

<!-- #### Example Failure: Rotating Cups with Balls

{{< justify >}}
Sora-2 failed to correctly understand centrifugal force and was unable to simulate the expected phenomenon in the rotating cups with balls experiment testing for centrifugal force understanding. When the system is spun around its center, both balls are expected to move outward and press against the sides of their respective cups, which are incorrectly portrayed in the image above.
{{< /justify >}}

{{< image src="img/videoscience/centrifugal_failure.png" alt="centrifugal_failure" width="85%" title="Figure 2: Example failure on centrifugal-force based generation.">}}

#### Example Failure: Electrolysis of Copper Sulfate

{{< justify >}}
While Sora-2 rendered realistic textures, it failed to adhere to the underlying electrochemical laws governing the electrolysis of copper sulfate. The model erroneously conflated oxidation and reduction at one electrode, leading to a nonphysical overproduction of solid copper and a failure to respect standard reaction kinetics.
{{< /justify >}}

{{< image src="img/videoscience/electrolysis_failure.png" alt="electrolysis_failure" width="85%" title="Figure 3: Example failure on electrochemistry.">}}

#### Example Failure: Setup Failure

Unable to correctly simulate physical commonsense (correct dynamism) in some weaker models

{{< image src="img/videoscience/pretty_but_wrong_basic_physical_commonsense.png" alt="pretty_but_wrong" width="95%" title="Figure 4: “Pretty but wrong physical commonsense” examples.">}} -->

#### Cross-Model Case Study: The Spectrum of Scientific Hallucination

{{< justify >}}
Example 107
{{</ justify >}}

<div style="display: flex; justify-content: space-between; gap: 10px; text-align: center;">
  <div style="flex: 1;">
    <img src="img/videoscience/sora_centrifugal.gif" alt="Sora-2 Failure" style="width: 100%;">
    <p><strong>Sora-2</strong><br><small>Static/No Motion</small></p>
  </div>
  <div style="flex: 1;">
    <img src="img/videoscience/veo_centrifugal.gif" alt="Veo-3 Failure" style="width: 100%;">
    <p><strong>Veo-3</strong><br><small>Inverse Physics</small></p>
  </div>
  <div style="flex: 1;">
    <img src="img/videoscience/kling_centrifugal.gif" alt="Kling Failure" style="width: 100%;">
    <p><strong>Kling-v2.5</strong><br><small>Boundary Violation</small></p>
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666;">
  Qualitative comparison across models for the same centrifugal force prompt.
</p>

#### Takeaways: The “Pretty but Wrong” Problem

{{< justify >}}
A central takeaway is that today’s best video generators can be photorealistic and temporally smooth while still being scientifically incorrect.

That mismatch is exactly what VideoScience-Bench is designed to reveal:

- models can imitate visual surface statistics
- without reliably internalizing the causal structure and constraints of scientific systems
  {{< /justify >}}

{{< image src="img/sciencecompass_modelwise_radar_dense_v9-1.png" alt="expert_label_ranking" width="95%" title="Expert annotated model performance on VideoScienceBench.">}}

### VideoScience-Judge: Scalable Expert-level Evaluation

{{< justify >}}
Manual evaluation of scientific accuracy is labor-intensive and requires domain expertise. To enable scalable yet rigorous assessment, we developed VideoScience-Judge, a VLM-as-a-Judge framework that emulates expert evaluation through:

- Checklist Generation: An LLM generates a specific rubric for the prompt (e.g., "Check if the laser bends downward," "Check if the liquid turns blue").
- Evidence Extraction: We utilize computer vision tools (such as GroundingDINO, ByteTrack, and RAFT optical flow) to extract "hard evidence," including tracking object trajectories and color changes, frame by frame.
- VLM Grading: A reasoning-capable VLM acts as the final judge, reviewing the checklist against the visual evidence to assign a score.

Our experiments demonstrate that this method achieves a correlation of 0.89 with human expert ratings, significantly outperforming other evaluation methods.
{{< /justify >}}

{{< image src="img/vsci_judge_correlation_blog.png" alt="correlation_charts" width="90%" title="VideoScience-Judge (CL+CV) achieves a Spearman correlation of 0.96, significantly outperforming all other baselines. This indicates our logic-grounded verification aligns almost perfectly with human scientific evaluation..">}}

### The Bottom Line

{{< justify >}}
Modern models excel at producing high-quality, photorealistic, and temporally coherent videos that are visually stunning, VideoScience-Bench reveals fundamental limitations in their ability to comprehend complex scientific phenomena.

Among the evaluated models, Sora-2 and Veo-3 demonstrate notably stronger performance. These models show an emerging ability to handle multi-concept scenarios and maintain physical consistency, suggesting that the path toward scientifically literate video generation, while challenging, is not out of reach.
{{< /justify >}}

## The path forward

{{< justify >}}
Video models are at an inflection point. If we want them to become reliable world simulators, we need to measure and then train for scientific correctness. VideoScience-Bench provides a testbed to track this progress. Looking ahead, scientifically grounded video world models could:

- accelerate scientific discovery via accurate simulation
- enable safer and more capable robotics
- power education and training tools
- support rapid engineering prototyping and testing

We hope VideoScience-Bench helps push video models toward being not only compelling generators, but faithful simulators that can reason about the laws governing our world.
{{< /justify >}}

## Get started

- GitHub: https://github.com/hao-ai-lab/VideoScience
- Dataset: HF LINK TBD
- Leaderboard: HF LINK TBD

## Citation

```bibtex
@article{hu2025benchmarking,
  title={Benchmarking Scientific Understanding and Reasoning for Video Generation using VideoScience-Bench},
  author={Hu, Lanxiang and Shankarampeta, Abhilash and Huang, Yixin and Dai, Zilin and Yu, Haoyang and Zhao, Yujie and Kang, Haoqiang and Zhao, Daniel and Rosing, Tajana and Zhang, Hao},
  journal={arXiv preprint arXiv:2512.02942},
  year={2025}
}
```
