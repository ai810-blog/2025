---
layout: about
title: about
permalink: /about/
nav: true
nav_order: 1
# subtitle: 

profile:
  align: right
  image: 
  image_circular: false # crops the image to make it circular
  address: 

news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false  # includes social icons at the bottom of the page
---

In this blog post, I’ll be reviewing two recent advances in molecular generative modeling for drug discovery:

1. Molecule Generation with Fragment Retrieval Augmentation (NeurIPS 2024)
- Seul Lee, Karsten Kreis, Srimukh Prasad Veccham, Meng Liu, Danny Reidenbach, Saee Paliwal, Arash Vahdat, Weili Nie
2. GenMol: A Drug Discovery Generalist with Discrete Diffusion (ICML 2025)
- Seul Lee, Karsten Kreis, Srimukh Prasad Veccham, Meng Liu, Danny Reidenbach, Yuxing Peng, Saee Paliwal, Weili Nie, Arash Vahdat

Both works tackle the challenge of exploring vast chemical spaces to generate novel, synthesizable drug-like molecules, but they approach it through distinct generative paradigms.

The f-RAG framework enhances fragment-based drug discovery by augmenting a pretrained molecular language model (SAFE-GPT) with a dynamic retrieval system. It retrieves hard fragments for direct inclusion in the molecule and soft fragments to guide generation via a trainable fragment injection module. To escape the limitations of fixed fragment libraries, it also applies genetic modification and iterative vocabulary updates, leading to a better exploration-exploitation trade-off across optimization, novelty, diversity, and synthesizability.

In contrast, GenMol proposes a generalist molecular generation model using a masked discrete diffusion process, allowing for non-autoregressive, permutation-invariant generation of fragment-based molecule sequences. Its design makes it task-agnostic: it supports de novo generation, fragment-constrained tasks, and goal-directed optimization (like hit generation and lead optimization) within a unified framework. GenMol introduces fragment remasking—a strategy to iteratively replace and regenerate molecular fragments—and molecular context guidance to improve control over generation.

Together, these papers illustrate two complementary trends in molecular design: augmenting generative models with external knowledge and building universal, diffusion-based frameworks that generalize across diverse drug discovery tasks.