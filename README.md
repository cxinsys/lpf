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

- :snake: [Anaconda](https://www.anaconda.com) is recommended to use and develop LPF.
- :penguin: Linux distros are tested and recommended to use and develop LPF.

### Anaconda virtual environment

After installing anaconda, create a conda virtual environment for LPF.
In the following command, you can change the Python version
(e.g.,`python=3.7` or `python=3.9`).

```
conda create -n lpf python=3.9
```

Now, we can activate our virtual environment for LPF as follows.

```
conda activate lpf
```

### Package installation

See [INSTALL.md](INSTALL.md) for the full installation guide (pre-built CUDA
wheels, `uv sync` from source, PyTorch backend, AOT kernel build).

Quick paths:

```bash
# Pre-built CUDA wheel (replace cu132 with your CUDA version)
pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu132-py3-none-linux_x86_64.whl

# From source with uv (recommended for development)
git clone https://github.com/cxinsys/lpf.git
cd lpf
uv sync --extra cu130 --extra evosearch
```

#### 3. Install PyGMO

Install PyGMO as follows.

```
conda install pygmo -c conda-forge
```

## Getting Started
- [Tutorials](https://github.com/cxinsys/lpf/tree/main/tutorials).
  
## Citation
- Daewon Lee, "**LPF: a framework for exploring the wing color pattern formation of ladybird beetles in Python**", [_Bioinformatics_, 39(7), btad430, July 2023](https://academic.oup.com/bioinformatics/article/39/7/btad430/7221539).
