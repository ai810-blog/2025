---
layout: distill
title: "Molecule Generation with Fragment Retrieval Augmentation"
description: 
  This paper introduces f-RAG, a novel framework for molecular generation that enhances fragment-based drug discovery through retrieval-augmented generation. By combining hard and soft fragment retrieval with genetic fragment modification, f-RAG guides a pre-trained molecular language model (SAFE-GPT) to generate novel, diverse, and synthesizable molecules. Hard fragments serve as structural constraints, soft fragments provide contextual guidance via cross-attention, and genetic operations enable exploration beyond known chemical space. Extensive experiments on benchmarks demonstrate that f-RAG achieves state-of-the-art performance in optimizing molecular properties while maintaining high novelty and synthesizability.


  
date: 2025-06-01
future: true
htmlwidgets: true
hidden: false

# must be the exact same name as your blogpost
bibliography: 2025-04-28-f-rag.bib  

toc:
- name: "Introduction"
- name: "Related Work"
  subsections:
    - name: Fragment-based Molecule Generation
    - name: Retrieval-Augmented Molecule Generation
- name: "The f-RAG Framework"
  subsections: 
    - name: Hard Fragment Retrieval
    - name: Soft Fragment Retrieval
    - name: Genetic Fragment Modification
- name: "Training the f-RAG Model"
- name: "Experimental Evaluation"
  subsections: 
    - name: Results and Discussion
    - name: Ablation Studies
- name: "Conclusion and Future Directions"
---

# Molecule Generation with Fragment Retrieval Augmentation (f-RAG)

## Introduction

Fragment-based drug discovery (FBDD) is a powerful approach in modern pharmaceutical research, enabling the construction of new molecules by assembling known molecular fragments. Despite its success, traditional FBDD methods often limit exploration to known chemical spaces, constraining the novelty and diversity of generated molecules. The research paper **"Molecule Generation with Fragment Retrieval Augmentation (f-RAG)"** addresses this limitation by proposing a novel framework that integrates retrieval-augmented generation to significantly enhance the exploration-exploitation trade-off in molecule generation.

## Related Work

Fragment-based molecule generation typically recombines existing molecular fragments to build new molecules. Notable methods include Junction Tree VAE (JT-VAE)<d-cite key="jin2018junction"></d-cite>, which structures molecules by combining large substructures with smaller fragments. Techniques like MARS<d-cite key="xie2021mars"></d-cite> and reinforcement learning-based methods<d-cite key="yang2021hit"></d-cite> also iteratively add or remove fragments guided by chemical properties. Genetic algorithm-based methods<d-cite key="jensen2019graph"></d-cite> further enhance fragment assembly using crossover and mutation, yet remain limited by initial fragment pools.

Retrieval-augmented generation (RAG)<d-cite key="borgeaud2022improving"></d-cite>, popularized in natural language processing, incorporates external contextual data to guide generative models. In molecular applications, retrieval augmentation methods often use entire molecules as retrieval units to influence generation<d-cite key="liu2024conversational"></d-cite>. This contrasts with f-RAG, which innovatively retrieves fragments to guide molecular construction more precisely, significantly enhancing chemical novelty and diversity.

## The f-RAG Framework

The f-RAG framework strategically employs two fragment retrieval methods:

### Hard Fragment Retrieval

Hard fragments explicitly become building blocks of newly generated molecules. Fragments are derived from known molecular structures using an arm-linker-arm decomposition approach. Each fragment's relevance is quantified by assessing its contribution to desired molecular properties:

$$
\text{score}(F_j) = \frac{1}{|S(F_j)|} \sum_{(x, y) \in S(F_j)} y,
$$

where $S(F_j)$ denotes molecules containing fragment $F_j$, and $y$ is a measure of the desired property.

The selected fragments form partial sequences, guiding generation via motif extensions or linker insertions. This direct inclusion ensures generated molecules retain meaningful chemical properties derived from existing fragments.

### Soft Fragment Retrieval

Soft fragments indirectly guide molecule generation without direct inclusion in the final structure. They provide context to the generative model, aiding the prediction of novel molecular fragments through embedding augmentation. The embeddings for hard and soft fragments are computed separately and integrated using a cross-attention mechanism:

$$
h_{\text{input}} = \text{LM}_{0:L}(x_{\text{input}}), \quad h_k^{\text{soft}} = \text{LM}_{0:L}(F_k^{\text{soft}}),
$$

followed by:

$$
h = \text{softmax}\left(\frac{\text{Query}(h_{\text{input}}) \cdot \text{Key}(H_{\text{soft}})^\top}{\sqrt{d_{\text{Key}}}}\right) \cdot \text{Value}(H_{\text{soft}}).
$$

This integration significantly boosts the generation of novel and chemically plausible fragments, promoting exploration beyond existing fragment spaces.

### Genetic Fragment Modification

To further enhance molecular novelty, genetic algorithms (GA) are employed post-generation. Using crossover and mutation operations, the GA continuously diversifies the fragment population, allowing exploration into previously uncharted chemical spaces.

## Training the f-RAG Model

f-RAG relies on the pre-trained molecular language model SAFE-GPT, keeping it frozen to preserve its extensive chemical knowledge. Training focuses exclusively on a lightweight fragment injection module with a self-supervised objective. This training involves decomposing molecules into fragments and predicting the most similar fragment based on embeddings, efficiently teaching the model to utilize soft fragments for informative guidance.

## Experimental Evaluation

f-RAG was extensively evaluated on multiple tasks, including the Practical Molecular Optimization (PMO) benchmark and tasks focused on optimizing docking scores, quantitative estimates of drug-likeness (QED), synthetic accessibility (SA), and molecular novelty.

### Results and Discussion

f-RAG consistently outperformed existing methods on several metrics:

* **Optimization Performance:** Achieved state-of-the-art performance across 12 out of 23 PMO benchmark tasks, significantly surpassing traditional and recent baselines.
* **Diversity, Novelty, Synthesizability:** Maintained superior balance among these criteria, demonstrating an effective exploration-exploitation trade-off and generating high-quality drug candidates.
* **Docking Score Optimization:** Produced molecules with higher binding affinities and improved drug-likeness and synthesizability across various protein targets.

### Ablation Studies

Detailed ablation studies underscored the essential contributions of each f-RAG component (hard retrieval, soft retrieval, genetic modifications). Particularly, soft fragment retrieval was crucial in achieving high molecular diversity and novelty without sacrificing performance.

## Conclusion and Future Directions

f-RAG significantly advances fragment-based drug discovery through innovative retrieval augmentation and genetic modification techniques. Its robust performance across diverse evaluation criteria confirms its potential in practical drug discovery applications.

Future research directions include:

* **Adaptive Retrieval Strategies:** Dynamically adjusting retrieval strategies based on ongoing molecular population performance.
* **Multi-modal Data Integration:** Incorporating additional modalities such as textual or structural biological data into the fragment retrieval process.
* **Scalability:** Extending f-RAG to larger chemical libraries and more complex molecular scenarios, validating its applicability and robustness on broader scales.


## Code availability

The official code written by the authors for reproduction can be found at
[https://github.com/NVlabs/f-RAG](https://github.com/NVlabs/f-RAG).

