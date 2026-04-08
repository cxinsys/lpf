<img src="assets/logo.png" alt="Drawing" width="200px"/>


## Introduction
- LPF represents **Ladybird Pattern Formation**.
- LPF is a framework for exploring the biological pattern formation exemplified by ladybird beetles.
- LPF can be utilized as an educational framework for understanding biological dynamics, pattern formation, and complexity.


<p align="center">
  <img src="assets/ladybird.gif" alt="Drawing" width="500px"/>
</p>

<p align="center">
  <img src="assets/pattern.gif" alt="Drawing" width="500px"/>
</p>

## Features
- Reaction-diffusion PDE models for the color pattern formation of ladybird beetles.
- Evolutionary search based on [PyGMO](https://esa.github.io/pygmo2/) providing the concept of [island](https://esa.github.io/pygmo2/tutorials/using_island.html) and [archipelago](https://esa.github.io/pygmo2/tutorials/using_archipelago.html).
- GPU optimization of the reaction-diffusion PDE solver for a batch of parameter sets based on [CuPy](https://cupy.dev/).

## Installation

See [INSTALL.md](INSTALL.md) for the full installation guide, which covers:

- Pre-built CUDA wheels (Linux / Windows, CUDA 12.6 – 13.2)
- From source with `uv` for development
- Optional PyTorch and JAX backends
- Conda-based setup for evolutionary search (pygmo)
- AOT kernel build

## Getting Started
- [Tutorials](https://github.com/cxinsys/lpf/tree/main/tutorials).
  
## Citation
- Daewon Lee, "**LPF: a framework for exploring the wing color pattern formation of ladybird beetles in Python**", [_Bioinformatics_, 39(7), btad430, July 2023](https://academic.oup.com/bioinformatics/article/39/7/btad430/7221539).
