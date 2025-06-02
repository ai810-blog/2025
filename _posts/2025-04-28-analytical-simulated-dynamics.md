---
layout: distill
title: "GenMol: A Drug Discovery Generalist with Discrete Diffusion"
description: 
  GenMol introduces a unified and versatile framework for molecular design using masked discrete diffusion on fragment-based molecular representations. By integrating bidirectional parallel decoding, fragment remasking, and molecular context guidance, GenMol tackles diverse drug discovery tasks such as de novo generation, fragment-constrained design, hit generation, and lead optimization. The model leverages non-autoregressive generation to improve efficiency and adaptivity, significantly outperforming task-specific baselines across multiple benchmarks.



date: 2025-06-01
future: true
htmlwidgets: true
hidden: false

bibliography: 2025-04-28-genmol.bib  

toc:
- name: "Introduction"
- name: "Related Work"
  subsections:
  - name: Discrete Diffusion Models
  - name: Fragment-based Molecular Generation
- name: "Background"
  subsections:
  - name: Masked Discrete Diffusion
  - name: SAFE Molecular Representation
- name: "GenMol Framework"
  subsections:
  - name: Masked Diffusion Architecture
  - name: Fragment Remasking
  - name: Molecular Context Guidance
- name: "Experimental Evaluation"
  subsections:
  - name: De Novo Generation
  - name: Fragment-Constrained Generation
  - name: Goal-Directed Hit Generation
  - name: Lead Optimization
  - name: Ablation Studies
- name: "Conclusion and Future Work"
---

# Introduction

Drug discovery is a multifaceted process that involves stages such as de novo generation, fragment-constrained design, hit identification, and lead optimization. Traditional generative models often specialize in specific tasks, requiring task-specific architectures or expensive retraining for new objectives. **GenMol** addresses this limitation by offering a **generalist molecular generation framework** based on **masked discrete diffusion** over **fragment-based molecular representations (SAFE)**. This approach allows GenMol to serve as a unified model across different phases of drug discovery with enhanced efficiency, flexibility, and chemical fidelity. Unlike prior autoregressive methods, GenMol enables **bidirectional parallel decoding** and effective **contextual guidance**, improving sample quality and reducing inference time.

# Related Work

**Discrete Diffusion Models:** Discrete diffusion has recently emerged as a compelling approach for generative modeling of categorical data. D3PM<d-cite key="peng2022d3pm"></d-cite> introduces the notion of forward masking via transition matrices and backward denoising conditioned on time. MDLM<d-cite key="sahoo2024simple"></d-cite>, adopted by GenMol, simplifies this by formulating the training as a time-weighted sum of masked language modeling (MLM) losses. This framework has shown promise in NLP but remains underexplored in molecular generation.

**Fragment-based Molecular Generation:** Fragment-based drug design (FBDD) focuses on assembling known chemical fragments to build novel molecules. Earlier methods like graph-based VAEs<d-cite key="jin2020multi"></d-cite>, RL-based molecule completion<d-cite key="yang2021hit"></d-cite>, and genetic algorithms<d-cite key="jensen2019graph"></d-cite> have utilized fragments as building blocks. However, these models often suffer from limited diversity or require extensive retraining for new tasks. More recent work such as f-RAG enhances fragment utility through retrieval-augmented generation, but still lacks the unification across multiple generation tasks.

# Background

## Masked Discrete Diffusion

In masked discrete diffusion, a clean sequence $\mathbf{x}$ is progressively corrupted into a fully masked sequence $\mathbf{z}^t$ through a forward process:

$$
q(\mathbf{z}^t_l | \mathbf{x}_l) = \text{Cat}(\alpha_t \mathbf{x}_l + (1 - \alpha_t) \mathbf{m})
$$

Here, $\alpha_t$ denotes the retention probability of the original token. The reverse process is trained to recover $\mathbf{x}$ by predicting token distributions from masked sequences. The training objective integrates cross-entropy losses over all time steps:

$$
\mathcal{L}_{\text{NELBO}} = \mathbb{E}_q \int_0^1 \frac{\alpha'_t}{1 - \alpha_t} \sum_l \log \langle \mathbf{x}_{\theta, l}(\mathbf{z}^t, t), \mathbf{x}_l \rangle \, dt
$$

## SAFE Molecular Representation

SAFE encodes molecules as an unordered sequence of fragments, each represented by a contiguous block of non-canonical SMILES tokens. These fragments are extracted using the BRICS algorithm and joined with attachment point markers. Unlike traditional SMILES or graph representations, SAFE maintains permutation invariance on fragment order, making it ideal for **non-autoregressive** generation. This also enables intuitive operations like fragment masking and replacement for goal-directed tasks.

# GenMol Framework

## Masked Diffusion Architecture

GenMol utilizes a BERT-based transformer trained using the masked discrete diffusion objective over SAFE sequences. The key advantages are:

1. **Parallel Decoding:** Tokens are predicted simultaneously, accelerating generation.
2. **Bidirectional Attention:** The model conditions predictions on full context, improving coherence.
3. **Fragment-Level Semantics:** By operating on fragments, GenMol better aligns with chemists' reasoning.

The reverse sampling from the model is defined using temperature-controlled confidence sampling:

$$
p^l_{\theta, i} = \frac{\exp(\log x^l_{\theta, i}(\mathbf{z}^t, t)/\tau)}{\sum_j \exp(\log x^l_{\theta, j}(\mathbf{z}^t, t)/\tau)}
$$

## Fragment Remasking

To optimize molecules, GenMol introduces **fragment remasking**: a fragment in a candidate molecule is randomly selected, masked, and regenerated via diffusion. This facilitates local exploration:

$$
y(f_k) = \frac{1}{|\mathcal{S}(f_k)|} \sum_{\mathbf{x} \in \mathcal{S}(f_k)} y(\mathbf{x})
$$

Fragments with high average scores are favored for vocabulary updates, enabling iterative refinement. The remasking mechanism is analogous to Gibbs sampling over fragment configurations, promoting controlled diversity and optimization.

## Molecular Context Guidance (MCG)

MCG provides **classifier-free guidance** for discrete diffusion. It interpolates between predictions on a given input and its noisier variant:

$$
\log x^{(w)}_{\theta, l, i} = w \log x_{\theta, l, i} + (1 - w) \log x'_{\theta, l, i}
$$

Here, $x'_{\theta}$ is the model's output on a more degraded (further masked) version of the input. This guidance boosts sample quality in tasks requiring conditional generation.

# Experimental Evaluation

## De Novo Generation

GenMol generates 1,000 molecules and measures validity, uniqueness, diversity, and a composite **quality** metric (valid + QED ≥ 0.6 + SA ≤ 4). Results show:

* **Validity:** 100%
* **Uniqueness:** 99.7%
* **Quality:** 84.6%
* **Sampling Speed:** Faster than SAFE-GPT by 2.5x using $N=3$ parallel decoding

GenMol demonstrates a tunable trade-off between quality and diversity through temperature ($\tau$) and randomness ($r$).

## Fragment-Constrained Generation

Using 10 drug scaffolds, GenMol outperforms SAFE-GPT in five sub-tasks: linker design, scaffold morphing, motif extension, scaffold decoration, and superstructure generation. It maintains high **constraint adherence** while improving novelty and structural diversity. For example:

* **Motif Extension Quality:** 30.1% (vs. 18.6% in SAFE-GPT)
* **Superstructure Diversity:** 0.599 (vs. 0.573)

## Goal-Directed Hit Generation

Using the PMO benchmark (23 tasks), GenMol applies fragment remasking to optimize molecular properties. It achieves the best performance in 19/23 tasks and the highest total AUC:

$$
\text{AUC}_{\text{top-10}}^{\text{sum}} = 18.362
$$

Significantly surpassing f-RAG, Genetic GFN, and other leading baselines, GenMol's improvements highlight the benefit of fragment-level exploration.

## Lead Optimization

GenMol performs constrained lead optimization under similarity thresholds $\delta = 0.4, 0.6$. On targets like JAK2, PARP1, and 5HT1B, it produces leads with:

* Lower binding affinity (more negative docking scores)
* High QED and low SA
* Preserved structural similarity

It succeeds in 26/30 tasks, whereas other baselines frequently fail to meet all criteria.

## Ablation Studies

* **Token-level remasking** underperforms due to limited exploration.
* **Autoregressive GPT-style remasking** degrades performance due to ordering constraints.
* **Removing MCG** reduces goal-directed generation quality.

Together, **fragment remasking + MCG** yields the best outcomes across diverse settings.

# Conclusion and Future Work

**GenMol** introduces a generalist molecular generation framework that unifies diverse drug discovery tasks within a single model. By combining masked discrete diffusion, SAFE representation, and context-aware guidance, it outperforms task-specific baselines in efficiency, sample quality, and task generalization.

Future work may include:

* **Integrating multimodal data**, such as protein-ligand complexes.
* **Enhancing safety controls** by excluding toxic fragments.
* **Scaling to larger molecular libraries** and industrial pipelines.

GenMol represents a promising step toward flexible and effective AI-powered drug discovery.
