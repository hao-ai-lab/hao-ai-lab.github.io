+++
title = "From Physical Commonsense to Scientific Reasoning: Why World Modeling in Video Matters"
date = 2026-02-12T12:00:00-08:00
authors = ["Lanxiang Hu", "Abhilash Shankarampeta", "Yixin Huang", "Zilin Dai", "Haoyang Yu", "Yujie Zhao", "Haoqiang Kang", "Daniel Zhao", "Tajana Rosing", "Hao Zhang"]
ShowReadingTime = true
draft = false
[socialIcons]
    [[socialIcons.icon]]
      name = "twitter"
      url = "https://twitter.com"
    [[socialIcons.icon]]
      name = "github"
      url = "https://github.com/hao-ai-lab/VideoScience"
[cover]
  image = "img/videoscience/header.png"
  alt = "scientific reasoning in video world models"
  caption = "Video generations are getting impressively realistic, but scientific correctness is a different bar. VideoScience-Bench evaluates whether video models behave like faithful simulators, not just good renderers."
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
  max-width: 100% !important;
  width: 100% !important;
  height: auto !important;
}

.post-cover {
  text-align: center !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}
</style>

{{< socialBadges arxiv-index="2512.02942" github="hao-ai-lab/VideoScience" huggingface="https://huggingface.co/datasets/lmgame/VideoScienceBench" >}}

{{< justify >}}
TL;DR: The golden age of AI video has mastered the "look" of reality, but it has yet to learn the laws of reality. Without adhering to rigorous scientific principles, even the most photorealistic model remains a high-fidelity hallucination engine rather than a reliable world simulator. To bridge this gap, we introduce VideoScience-Bench: the first benchmark specifically designed to move beyond "physical commonsense" and evaluate undergraduate-level scientific reasoning in video models.

We also introduce VideoScience-Judge, a scalable VLM-as-a-judge pipeline that evaluates generated videos against rigorous scientific criteria. Correlation analysis shows that VideoScience-Judge achieves the strongest alignment with expert-rated rankings and best captures a video model’s scientific reasoning capability in comparison with existing benchmarks.
{{< /justify >}}

{{< two_images
src1="img/phygenbench_3-2.gif"
src2="videos/vid_087_run_2.gif"
alt1="physical_commonsense_world_modeling"
alt2="scientific_reasoning_world_modeling"
width1="50%"
width2="50%"
title="Figure 1: Left: A video model generating a physically plausible scene based on everyday commonsense ('A stone is gently placed on the surface of a pool filled with water.', Source: PhyGenBench). Right: A video generation task that requires scientific reasoning, where correct outcomes depend on multiple interacting laws rather than visual plausibility alone. ('A clear plastic water bottle has a small hole in its side, from which a smooth, laminar stream of water is flowing. A red laser pointer is aimed from the other side of the bottle, directly through the water and into the hole.', Source: VideoScience-Bench)."
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

{{< image src="https://thinking-with-video.github.io/assets/main_picture.png" alt="thinking_with_video" width="150%" title="Thinking with Video: Examples from VideoThinkBench.">}}

These achievements mark a pivotal shift: video models are no longer just generators, they're becoming reasoners.
{{< /justify >}}

### Simulation and Robotics Applications

{{< justify >}}
In recent context of [World Model Roadmap](https://world-model-roadmap.github.io/), [WorldSimBench](https://iranqin.github.io/WorldSimBench.github.io/), video generation is increasingly framed as: implicit world model (physics + dynamics) + renderer (pixels). In this view, video models aren’t only content engines, they could be simulation engines. If the simulator is scientifically wrong, downstream systems trained on it can inherit those failures.

The stakes for scientific accuracy are highest in robotics, where models must evolve from simple visual generators into reliable world simulators. Industry leaders like 1X and NVIDIA are developing world models, such as [1X-WorldModel](https://www.1x.tech/discover/1x-world-model) and [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/), that function as virtual simulators, leveraging raw sensor data to predict complex material interactions and envision potential futures. Because these systems generate the massive datasets used to train physical AI at scale, their adherence to scientific laws is a critical prerequisite for the safety and effectiveness of robots in the real world.
{{< /justify >}}

## Scientific Reasoning as the Foundation of World Modeling

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

<div style="display: flex; justify-content: space-between; gap: 10px; text-align: center;">
  <div style="flex: 1;">
   <img src="videos/vid_094_run_2.gif" alt="Prince Rupert's Drop Tail Break" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> A teardrop-shaped piece of tempered glass is held at its bulbous head. Small pliers gently snip the thin tail end.<br><strong>Expected:</strong> The entire drop explosively shatters into powder as internal tension is released.</p>
  </div>
  <div style="flex: 1;">
    <img src="videos/vid_177_run_2.gif" alt="Polarized Plastic Fringes" style="width: 100%;">
    <p style="font-size: 0.85em;"><strong>Prompt:</strong> A clear plastic ruler is placed between two crossed polarizing filters and illuminated by a bright white light.<br><strong>Expected:</strong> The stressed plastic causes rotation of the light's polarization plane in a wavelength-dependent way, yielding colored interference fringes.</p>
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666;">
  Examples from VideoScience-Bench showing Sora-2 generated videos and expected phenomena.
</p>


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

<div style="display: flex; flex-direction: column; gap: 20px;">
  <div style="display: flex; gap: 20px; align-items: stretch;">
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_095_run_3.gif" alt="The Spaghetti Mystery" style="width: 100%; height: 300px; object-fit: contain; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Generated Video</strong></p>
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_095_ref.gif" alt="The Spaghetti Mystery" style="width: 100%; height: 300px; object-fit: contain; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Reference Video</strong></p>
    </div>
  </div>
  <p style="font-size: 0.85em; text-align: center;"><strong>Prompt:</strong> A dry spaghetti stick is held at both ends and slowly bent until it breaks.<br><strong>Expected:</strong> The spaghetti breaks into three or more pieces rather than two, because stress waves from the first fracture cause additional breaks before the fragments separate.</p>
  
  <div style="display: flex; gap: 20px; align-items: stretch; margin-top: 30px;">
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_137_run_3.gif" alt="The Ball and Cart" style="width: 100%; height: 300px; object-fit: contain; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Generated Video</strong></p>
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_137_ref.gif" alt="The Ball and Cart" style="width: 100%; height: 300px; object-fit: contain; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Reference Video</strong></p>
    </div>
  </div>
  <p style="font-size: 0.85em; text-align: center;"><strong>Prompt:</strong> A cart moves forward at a constant speed and launches a ball straight upward from its top."<br><strong>Expected:</strong> The ball travels upward and then downward in a parabolic path, but lands back on the moving cart because both the ball and the cart have the same horizontal velocity.</p>
</div>
<p style="text-align: center; font-style: italic; color: #666; margin-top: 20px;">
  Failure Examples on Violations of Phenomenon Congruency generated using Sora-2.
</p>

{{< justify >}}
These two examples perfectly illustrate the current "uncanny valley" of AI video generation: High Visual Fidelity, Low Physical Logic. In both the spaghetti and the cart simulations, the model achieves impressive "Spatio-Temporal Coherence"—the lighting is realistic, the motion is smooth, and object identity is stable. However, both fail primarily in Phenomenon Congruency. The video generation model operates as a pattern-matcher rather than a physics engine; it knows what a breaking stick or a launching ball looks like in isolation, but it lacks the underlying understanding of material science (the "snap-back" effect in spaghetti) or Newtonian mechanics (conservation of momentum in the cart). The result is a video that looks perfect at a glance but falls apart under scientific scrutiny, revealing that the model is hallucinating motion rather than simulating reality.
{{< /justify >}}

#### The Universal Failure: When No One Gets It Right

{{< justify >}}
Some scenarios are so complex that they trigger a total collapse of reasoning across all models tested. These represent the current ceiling of zero-shot scientific reasoning.
{{< /justify >}}

<div style="display: flex; flex-direction: column; gap: 20px;">
  <div style="display: flex; gap: 20px; align-items: flex-start;">
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_124_run_1.gif" alt="The Chemical Traffic Light " style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Generated Video</strong></p>
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_124_ref.gif" alt="The Chemical Traffic Light " style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Reference Video</strong></p>
    </div>
  </div>
  <p style="font-size: 0.85em; text-align: center;"><strong>Prompt:</strong> A flask containing a yellow solution of glucose, sodium hydroxide, and the indicator indigocarmine is shown. The person lifts and gently shakes the flask.<br><strong>Expected:</strong> The yellow solution rapidly shifts toward green as the flask is shaken, showing the indicator’s partial oxidation by oxygen introduced from the air. Continued shaking drives the oxidation further and the color moves from green to red. When agitation stops, dissolved glucose reduces the indicator and the solution relaxes back to yellow.</p>
  
  <div style="display: flex; gap: 20px; align-items: flex-start; margin-top: 30px;">
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_087_run_2.gif" alt="The Laser Fiber Optic " style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Generated Video</strong></p>
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_087_ref.gif" alt="The Laser Fiber Optic " style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Reference Video</strong></p>
    </div>
  </div>
  <p style="font-size: 0.85em; text-align: center;"><strong>Prompt:</strong> A clear plastic water bottle has a small hole in its side, from which a smooth, laminar stream of water is flowing. A red laser pointer is aimed from the other side of the bottle, directly through the water and into the hole.<br><strong>Expected:</strong> The laser beam enters the stream and becomes "trapped." It reflects repeatedly off the inner surface of the water stream, causing the entire parabolic arc of the falling water to glow red as if it were a fiber optic cable.</p>
</div>
<p style="text-align: center; font-style: italic; color: #666; margin-top: 20px;">
  Failure Examples generated using Sora-2 on multi-concept scientific phenomena.
</p>

{{< justify >}}
The failure of these generated clips represents more than just a visual glitch; it is a total collapse of scientific causality. In the 'Traffic Light' reaction, the model successfully renders the liquid's texture but fails to link physical agitation to the chemical redox cycle—ignoring the oxygen-driven transition from yellow to green. Similarly, in the laser experiment, the model treats light as a 2D overlay rather than a physical entity governed by refractive indices. It fails to 'trap' the beam within the water stream via Total Internal Reflection, proving that while AI can mimic the appearance of reality, it still lacks a foundational world model of the laws of physics and chemistry.
{{< /justify >}}


#### The "Complexity Collapse": Failing the Setup

{{< justify >}}
Current video generation models are masters of aesthetic mimicry, but they are often functionally illiterate when it comes to the laws of physics. Before a model can simulate gravity, electromagnetism, or optics, it must first "build" the experimental setup correctly. In sophisticated prompts, models often fail Prompt Consistency, they cannot even construct the experimental setup correctly, making the result inevitably wrong.
{{< /justify >}}

<div style="display: flex; flex-direction: column; gap: 20px;">
  <div style="display: flex; gap: 20px; align-items: flex-start;">
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_226_run_2.gif" alt="The Heart Motor" style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Generated Video</strong></p>
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_226_ref.gif" alt="The Heart Motor" style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Reference Video</strong></p>
    </div>
  </div>
  <p style="font-size: 0.85em; text-align: center;"><strong>Prompt:</strong> Position a battery vertically on top of the three neodymium magnets so that the magnets contact the battery’s negative terminal. Place a heart-shaped copper wire so that it can touch both the top of the battery (positive terminal) and the sides of the magnets simultaneously.<br><strong>Expected:</strong> When the copper wire touches both ends of the circuit, it begins to spin or move continuously, creating a small, self-turning “heart motor.</p>
  <div style="display: flex; gap: 20px; align-items: flex-start; margin-top: 30px;">
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_173_run_1.gif" alt="Polarized Film Colors" style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Generated Video</strong></p>
    </div>
    <div style="flex: 1; text-align: center;">
      <img src="videos/vid_173_ref.gif" alt="Polarized Film Colors" style="width: 100%; max-width: 600px; display: block; margin: 0 auto;">
      <p style="font-size: 0.85em; margin-top: 10px;"><strong>Reference Video</strong></p>
    </div>
  </div>
  <p style="font-size: 0.85em; text-align: center;"><strong>Prompt:</strong> A plastic film is placed between two crossed polarizing filters and illuminated by a white flashlight; the film is slowly twisted while being recorded.<br><strong>Expected:</strong> The film’s birefringence splits light into components that interfere after passing through the analyzer. Rotation changes retardation, producing colored interference fringes dependent on polarization angle.</p>
</div>
<p style="text-align: center; font-style: italic; color: #666; margin-top: 20px;">
  Failure Examples on Violations of Prompt Congruency generated using Sora-2.
</p>

{{< justify >}}
While these AI models can render the "textures" of a laboratory—the glint of copper or the shimmer of a film—they fundamentally fail the "construction phase" of the experiment. In the first video, it renders a motor that cannot run because the model doesn't understand that a circuit must be closed; it places the wire near the battery rather than connecting it to the terminals. While the prompt asks for the copper wire to touch both the positive terminal and the magnets, the model renders the wire floating or balanced on the "shoulders" of the battery.

In the second, it treats "birefringence" as a decorative skin on the plastic rather than an optical result of light passing through a specific sequence of filters. To see birefringence, the plastic must be sandwiched between two filters. Here, the film is held above two side-by-side filters that aren't even overlapping.
{{< /justify >}}


#### Cross-Model Case Study: The Spectrum of Scientific Hallucination

{{< justify >}}
The three state-of-the-art AI video models Sora 2, Veo 3 and Kling-v2.5, when tasked with simulating a controlled electrochemical experiment: copper sulfate electrolysis. The prompt: "Two copper wires connected to a battery are placed in a blue copper sulfate solution." Beyond visual aesthetics, the objective is to test adherence to electrochemical laws. In a real-world scenario, this setup drives electrolysis, which should result in specific observable phenomena: bubbles appearing on the anode, metallic copper slowly depositing on the cathode, and the blue solution becoming lighter near the cathode as Cu²⁺ ions are depleted. The following critiques detail how each model fails to accurately simulate these foundational physical and chemical interactions.
{{</ justify >}}

<div style="display: flex; justify-content: space-between; gap: 10px; text-align: center;">
  <div style="flex: 1;">
    <img src="videos/vid_138_run_2_sora.gif" alt="Sora-2 Failure" style="width: 100%;">
    <p><strong>Sora-2</strong><br><small>Wrong Redox</small></p>
  </div>
  <div style="flex: 1;">
    <img src="videos/vid_138_run_2_veo.gif" alt="Veo-3 Failure" style="width: 100%;">
    <p><strong>Veo-3</strong><br><small>Physical Morphing</small></p>
  </div>
  <div style="flex: 1;">
    <img src="videos/vid_138_run_2_kling.gif" alt="Kling Failure" style="width: 100%;">
    <p><strong>Kling-v2.5</strong><br><small>Object Hallucination</small></p>
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666;">
  Qualitative comparison across models for "Electrolysis of Copper Sulfate" prompt.
</p>

{{< justify >}}
While Sora-2 rendered realistic textures, it failed to adhere to the underlying electrochemical laws governing the electrolysis of copper sulfate. The model erroneously conflated oxidation and reduction at one electrode, leading to a nonphysical overproduction of solid copper and a failure to respect standard reaction kinetics.

Kling achieved high fidelity in lighting and fluid rendering, it fundamentally misunderstood the prompt's physical setup. Instead of submerging copper wires acting as electrodes, the model hallucinated entire cylindrical batteries (or capacitors) and submerged them directly into the solution.

Veo failed to maintain the physical rigidity required for solid electrodes and ignored the chemical reaction entirely. The generated video exhibits "dream-like" physics where the copper rods morph, bend, and intersect with the glass beaker impossible ways, violating basic object permanence and solidity.

Ultimately, this experiment demonstrates that current video models operate on visual association rather than physical causation; while they can successfully render the di`stinct components of the setup—copper, glass, and solution—they fail to simulate the underlying electrochemical circuit, ignoring the flow of electrons and ions required to drive the specific redox reactions defined by the prompt.
{{< /justify >}}


#### Takeaways: The “Pretty but Wrong” Problem

{{< justify >}}
A central takeaway is that today’s best video generators can be photorealistic and temporally smooth while still being scientifically incorrect.

That mismatch is exactly what VideoScience-Bench is designed to reveal:

- Models can imitate visual surface statistics
- Without reliably internalizing the causal structure and constraints of scientific systems
  {{< /justify >}}

{{< image src="img/sciencecompass_modelwise_radar_dense_v9-1.png" alt="expert_label_ranking" width="75%" title="Expert annotated model performance on VideoScienceBench.">}}

### VideoScience-Judge: Scalable Expert-level Evaluation

{{< justify >}}
Manual evaluation of scientific accuracy is labor-intensive and requires domain expertise. To enable scalable yet rigorous assessment, we developed VideoScience-Judge, a VLM-as-a-Judge framework that emulates expert evaluation through:

- Checklist Generation: An LLM generates a specific rubric for the prompt (e.g., "Check if the laser bends downward," "Check if the liquid turns blue").
- Evidence Extraction: We utilize computer vision tools (such as GroundingDINO, ByteTrack, and RAFT optical flow) to extract "hard evidence," including tracking object trajectories and color changes, frame by frame.
- VLM Grading: A reasoning-capable VLM acts as the final judge, reviewing the checklist against the visual evidence to assign a score.

Our experiments demonstrate that this method achieves a correlation of 0.89 with human expert ratings, significantly outperforming other evaluation methods.
{{< /justify >}}

{{< image src="img/vsci_judge_correlation_blog.png" alt="correlation_charts" width="90%" title="VideoScience-Judge (CL+CV) achieves a Spearman correlation of 0.96, significantly outperforming all other baselines. This indicates our logic-grounded verification aligns almost perfectly with human scientific evaluation..">}}

### Qualitative Comparison: VideoScience-Bench vs. Baselines

| Evaluation Framework | Core Mechanism | Limitations on VideoScience-Bench |
| :--- | :--- | :--- |
| **T2V-CompBench** | **Attribute Binding**<br>Checks for the presence and consistent binding of object attributes. | • **Surface-Level:** Focuses on static attribute correctness rather than dynamic scientific validity.<br>• **Physics Blind:** Misses critical failures in temporal coherence, momentum, and physical plausibility. |
| **LMArena-T2V** | **Crowdsourced ELO**<br>Aggregates human votes from a general user base. | • **Visual Bias:** Voters prioritize visual fidelity over logical correctness, often ignoring scientific inaccuracies.<br>• **Irrelevance:** Most prompts test daily scenarios rather than scientific reasoning. |
| **PhyGenEval**<br>*(PhyGenBench)* | **Rule-Based / Binary**<br>Checks for discrete violations (e.g., "Did the object fall?"). | • **Lack of Nuance:** Fails to capture holistic physical realism like acceleration patterns, momentum, and causal relationships.<br>• **Rigidity:** Binary judgment cannot evaluate the "cascading effects" of multi-concept phenomena. |
| **VideoScore2** | **General-Purpose VLM**<br>Trained on real-world prompts to assess "plausibility". | • **Everyday Bias:** Prioritizes "looking plausible" (everyday common sense) over rigorous domain-specific physical laws.<br>• **Data Gap:** Training data lacks complex scientific scenarios, causing it to accept scientifically inaccurate but visually "normal" videos. |
| **VideoScience-Judge**<br>*(Ours)* | **Evidence-Based VLM**<br>Uses domain-specific checklists + Computer Vision (CV) tools. | • **Holistic Reasoning:** Explicitly evaluates *Phenomenon Congruency* and *Correct Dynamism* using extracted key frames.<br>• **Expert Alignment:** Achieves the highest correlation with human experts by grounding scores in concrete physical evidence. |

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

{{< youtube Coy2TyBcT4g>}}

## Get started

- GitHub: https://github.com/hao-ai-lab/VideoScience
- Paper: https://arxiv.org/abs/2512.02942
- Dataset: https://huggingface.co/datasets/lmgame/VideoScienceBench
- Leaderboard: https://huggingface.co/spaces/lmgame/videoscience-bench
  
## Citation

```bibtex
@article{hu2025benchmarking,
  title={Benchmarking Scientific Understanding and Reasoning for Video Generation using VideoScience-Bench},
  author={Hu, Lanxiang and Shankarampeta, Abhilash and Huang, Yixin and Dai, Zilin and Yu, Haoyang and Zhao, Yujie and Kang, Haoqiang and Zhao, Daniel and Rosing, Tajana and Zhang, Hao},
  journal={arXiv preprint arXiv:2512.02942},
  year={2025}
}
```
