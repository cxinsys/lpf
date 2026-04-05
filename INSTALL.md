# Installation Guide

## Quick Install

CUDA 13.2 example:

```bash
pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu132-py3-none-linux_x86_64.whl
```

This installs `lpf`, CuPy, and all other dependencies automatically.
Replace `cu132` with your CUDA version from the table below.

### Available wheels

Check your CUDA version with `nvidia-smi` and pick the matching wheel.

| CUDA | lpf wheel | CuPy (auto-installed) |
|------|-----------|-----------------------|
| 13.2 | `lpf-0.2.0+cu132` | `cupy-cuda13x` |
| 13.0 | `lpf-0.2.0+cu130` | `cupy-cuda13x` |
| 12.8 | `lpf-0.2.0+cu128` | `cupy-cuda12x` |
| 12.6 | `lpf-0.2.0+cu126` | `cupy-cuda12x` |

All wheels: [GitHub Releases](https://github.com/cxinsys/lpf/releases/tag/v0.2.0)

### PyTorch (optional)

Only needed for `device="torch:gpu:0"`. Install separately with the matching CUDA version:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu132
```

Replace `cu132` to match your CUDA version (e.g. `cu128`, `cu130`).

---

## Manual Installation

### 1. Create an environment

#### conda

```bash
conda create -n lpf python=3.11 -y

conda activate lpf
```

#### venv

```bash
python -m venv .venv

source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 2. Install lpf

```bash
# From source (development):
pip install -e .

# From wheel (production):
pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu130-py3-none-linux_x86_64.whl
```

### 3. Install GPU packages

```bash
# CuPy (required for GPU):
pip install cupy-cuda13x    # CUDA 13.x
pip install cupy-cuda12x    # CUDA 12.x

# PyTorch (optional, for device="torch:gpu:0"):
pip install torch --extra-index-url https://download.pytorch.org/whl/cu130
```

Check your CUDA version with `nvidia-smi`.

> **Note:** CuPy is required for CUDA kernel acceleration regardless of whether
> you use `device="cuda:0"` or `device="torch:gpu:0"`. When using PyTorch,
> `lpf` internally bridges PyTorch tensors to CuPy via DLPack (zero-copy)
> for kernel execution.

### Optional dependencies

```bash
pip install -e ".[viz]"    # lpips, opencv-python, torchmetrics

pip install -e ".[test]"   # pytest
```

---

## GPU Usage

```python
model = LiawModel(..., device="cuda:0")      # CuPy backend

model = LiawModel(..., device="torch:gpu:0") # PyTorch backend

solver = EulerSolver(dt=0.01, n_iters=500000)
solver.solve(model)  # CUDA kernels activate automatically
```

---

## CUDA Kernel Compilation Modes

`lpf` provides two ways to compile CUDA kernels: **JIT** and **AOT**.
Both produce identical results. The system selects the best available option automatically.

### JIT (Just-In-Time) — Default, zero setup

Kernels are compiled at runtime on first use via CuPy's NVRTC compiler.

- **No `nvcc` needed** — CuPy includes NVRTC
- **First-run overhead:** ~2-3 seconds (cached afterwards)
- **Supported dtypes:** float32, float64

### AOT (Ahead-Of-Time) — Pre-built wheels or manual build

Pre-built wheels from GitHub Releases already include AOT binaries.
To build manually:

```bash
# Install nvcc:
conda install -c nvidia cuda-nvcc                          # conda

pip install nvidia-cuda-nvcc-cu13                           # pip

# or install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
```

```bash
# Build AOT kernels:
python -m lpf.kernels.aot.build                # Auto-detect GPU architecture

python -m lpf.kernels.aot.build --arch sm_90   # Specific architecture

python -m lpf.kernels.aot.build --all-arch     # All common architectures (portable)
```

### How auto-selection works

```
1. Native .so exists?  →  C solve loop (zero Python during iteration)

2. .fatbin exists?     →  AOT kernels + Python loop

3. Neither?            →  JIT compile via NVRTC + Python loop
```

All three paths produce the same numerical results.

---

## Building Wheels

```bash
python build_wheel.py                 # AOT compile + wheel (auto-detect GPU)

python build_wheel.py --all-arch      # All common architectures (sm_80~90)

python build_wheel.py --arch sm_90    # Specific architecture

python build_wheel.py --skip-aot      # Skip AOT, use existing binaries

python build_wheel.py --cpu           # CPU-only wheel
```

---

## Verifying the Installation

```bash
python -m pytest tests/ -v
```

Quick GPU smoke test:

```python
python -c "
from lpf.models import LiawModel
from lpf.solvers import EulerSolver
from lpf.initializers import LiawInitializer
import numpy as np

model = LiawModel(
    initializer=LiawInitializer(
        init_pts=np.array([[[16, 16]]], dtype=np.uint32),
        init_states=np.array([[0.5, 0.5]], dtype=np.float32)),
    n_init_pts=1,
    params=np.array([[1e-3, 1e-2, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]], dtype=np.float32),
    width=64, height=64, dx=0.1, device='cuda:0')

EulerSolver(dt=0.01, n_iters=1000).solve(model, verbose=1)
print('GPU acceleration working!')
"
```

---

## Summary

| Setup level | Command | What you get |
|-------------|---------|--------------|
| **Quick (GPU)** | `python install.py` | lpf + CuPy + PyTorch, CUDA auto-detected |
| **CPU only** | `python install.py --cpu` | NumPy solver (~100 it/s) |
| **GPU (JIT)** | manual CuPy install | CUDA kernels, auto-compiled (~150K it/s) |
| **GPU (AOT)** | pre-built wheel or `python -m lpf.kernels.aot.build` | Pre-compiled kernels + native C loop |
| **PyTorch GPU** | manual torch install | Same CUDA kernels via DLPack bridge |
