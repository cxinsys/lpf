<img src="assets/logo.png" alt="Drawing" width="200px"/>


## Introduction
- LPF represents **Ladybird Pattern Formation**.
- LPF is a framework for analyzing the biological pattern formation exemplified by ladybird beetles.
- LPF can be utilized as an educational framework for understanding biological dynamics, pattern formation, and complexity.


<p style="text-align:center;">
<img src="assets/ladybird.gif" alt="Drawing" width="700px"/>

<img src="assets/pattern.gif" alt="Drawing" width="700px"/>
</p>

## Features
- Reaction-diffusion PDE models for the color pattern formation of ladybird bettles.
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

### Dependency of the PDE solver

:bulb: To optimize the solver for a batch of parameter sets based on GPU computing, install CuPy. However, if you want to use only the cpu, you can omit it.

- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pillow](https://pillow.readthedocs.io/en/stable/)
- [pyyaml](https://pyyaml.org/)
- [cupy](https://cupy.dev/) (optional)


### Dependency of the evolutionary search 
The order of installing the following packages is important to avoid version conflicts.


#### 1. Install PyTorch
Install PyTorch following the [official documentation](https://pytorch.org/).


#### 2. Install the packages in the requirements.
Install the packages in [requirements](https://github.com/cxinsys/lpf/blob/main/requirements.txt). 

```
pip install -r requirements.txt
```

#### 3. Install PyGMO

Install PyGMO as follows.

```
conda install pygmo -c conda-forge
```

## Getting Started
