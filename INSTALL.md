# Installation Guide

## 1. Basic Installation

```bash
# Clone the repository
git clone https://github.com/cxinsys/lpf.git
cd lpf
```

### Option A: conda

```bash
conda create -n lpf python=3.13 -y
conda activate lpf
pip install -e .
```

### Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
pip install -e .
```

Core dependencies installed automatically:

- `numpy`
- `scipy`
- `pillow`
- `tqdm`
- `pyyaml`
- `xxhash`

### Optional dependencies

```bash
pip install -e ".[viz]"    # Visualization & objectives
pip install -e ".[test]"   # Testing
```

`[viz]` installs:

- `lpips`
- `opencv-python`
- `torchmetrics`

---

## 2. GPU Acceleration

`lpf` automatically uses CUDA kernels when a GPU-capable backend is detected.
No code changes needed — just set `device` when creating a model:

```python
model = LiawModel(..., device="cuda:0")      # CuPy backend
model = LiawModel(..., device="torch:gpu:0") # PyTorch backend

solver = EulerSolver(dt=0.01, n_iters=500000)
solver.solve(model)  # CUDA kernels activate automatically
```

### Required packages for GPU

| Package | Install command | Purpose |
|---------|----------------|---------|
| **CuPy** | `pip install cupy-cuda13x` | GPU array operations + JIT kernel compilation |
| **PyTorch** | `pip install torch torchvision` | Alternative GPU backend |

Choose the CuPy package matching your CUDA driver version:

| CUDA driver version | CuPy package |
|---------------------|--------------|
| CUDA 13.x | `pip install cupy-cuda13x` |
| CUDA 12.x | `pip install cupy-cuda12x` |
| CUDA 11.x | `pip install cupy-cuda11x` |

Check your CUDA version with `nvidia-smi`.

> **Note:** CuPy is required for CUDA kernel acceleration regardless of whether
> you use `device="cuda:0"` or `device="torch:gpu:0"`. When using PyTorch,
> `lpf` internally bridges PyTorch tensors to CuPy via DLPack (zero-copy)
> for kernel execution.

---

## 3. CUDA Kernel Compilation Modes

lpf provides two ways to compile CUDA kernels: **JIT** and **AOT**.
Both produce identical results. The system selects the best available option automatically.

### JIT (Just-In-Time) — Default, zero setup

Kernels are compiled at runtime on first use via CuPy's NVRTC compiler.

- **Requirements:** `cupy-cuda1Xx` only (includes NVRTC)
- **No `nvcc` needed**
- **First-run overhead:** ~2-3 seconds (cached afterwards by CuPy)
- **Supported dtypes:** float32, float64

```bash
# This is all you need:
pip install cupy-cuda13x
```

### AOT (Ahead-Of-Time) — Optional, faster startup

Kernels are pre-compiled into a binary (`.fatbin` + `.so`) using `nvcc`.
Eliminates first-run compilation overhead and enables float16 support.

- **Requirements:** `nvcc` (CUDA Toolkit compiler)
- **Supported dtypes:** float16, float32, float64

```bash
# Install nvcc (choose one):
conda install -c nvidia cuda-nvcc                          # conda
pip install nvidia-cuda-nvcc-cu13                           # pip (venv)
# or install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads

# Build the AOT kernels:
python -m lpf.csrc.build

# Verify:
ls lpf/csrc/kernels.fatbin lpf/csrc/libsolver.so
```

Build options:

```bash
python -m lpf.csrc.build                # Auto-detect GPU architecture
python -m lpf.csrc.build --arch sm_90   # Specific architecture
python -m lpf.csrc.build --all-arch     # All common architectures (portable)
python -m lpf.csrc.build -v             # Verbose (show nvcc commands)
```

### How auto-selection works

When you call `solver.solve(model)` with a CUDA device:

```
1. Native .so exists?  →  C solve loop (zero Python during iteration)
2. .fatbin exists?     →  AOT kernels + Python loop
3. Neither?            →  JIT compile via NVRTC + Python loop
```

All three paths produce the same numerical results.

---

## 4. Verifying the Installation

```bash
# Run all tests
python -m pytest tests/ -v

# Quick GPU smoke test
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

## 5. Summary

| Setup level | What you install | What you get |
|-------------|-----------------|--------------|
| **CPU only** | `pip install -e .` | NumPy solver (~100 it/s) |
| **GPU (JIT)** | + `pip install cupy-cuda13x` | CUDA kernels, auto-compiled (~150K it/s) |
| **GPU (AOT)** | + `conda install cuda-nvcc` + `python -m lpf.csrc.build` | Pre-compiled kernels + native C loop |
| **PyTorch GPU** | + `pip install torch` | Same CUDA kernels via DLPack bridge |
