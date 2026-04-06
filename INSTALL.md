# Installation Guide

## Quick Install (pre-built wheel)

For most users, grab the pre-built wheel for your CUDA version. CUDA 13.2
example:

```bash
pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu132-py3-none-linux_x86_64.whl
```

This installs LPF, CuPy, and all core dependencies automatically. AOT
CUDA kernels are bundled inside the wheel — no compilation needed at
import time. Replace `cu132` with your CUDA version from the table below.

To also pull in the perceptual / image-based objectives used by
evolutionary search:

```bash
pip install "lpf[evosearch] @ https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu132-py3-none-linux_x86_64.whl"
```

### Available wheels

Check your CUDA version with `nvidia-smi` and pick the matching wheel.

| CUDA | LPF wheel | CuPy (auto-installed) |
|------|-----------|-----------------------|
| 13.2 | `lpf-0.2.0+cu132` | `cupy-cuda13x` |
| 13.0 | `lpf-0.2.0+cu130` | `cupy-cuda13x` |
| 12.8 | `lpf-0.2.0+cu128` | `cupy-cuda12x` |
| 12.6 | `lpf-0.2.0+cu126` | `cupy-cuda12x` |

All wheels: [GitHub Releases](https://github.com/cxinsys/lpf/releases/tag/v0.2.0)

### PyTorch (optional)

LPF uses CuPy as the default GPU backend, so PyTorch is not required for
basic GPU usage.

If you want to integrate LPF into a PyTorch-based workflow (for example,
using LPF simulation results as inputs to a neural network, or running
LPF alongside other PyTorch models on the same GPU), set
`device="torch:cuda:<gpu_id>"` (e.g. `"torch:cuda:0"`) to keep all data
as PyTorch tensors. This avoids unnecessary copies between frameworks.

In this mode, LPF internally bridges PyTorch tensors to CuPy via DLPack
(zero-copy) for CUDA kernel execution, and returns the results back as
PyTorch tensors.

Install PyTorch from the matching CUDA wheel index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Replace `cu130` to match your CUDA version (`cu126`, `cu128`, `cu130`).
CUDA 13.2 has no dedicated PyTorch wheel — use the cu130 index, which is
forward-compatible with CUDA 13.x runtimes.

If you cloned the source tree, the easier path is `uv sync --extra cuXXX
--extra torch-cuXXX` (see [From Source](#from-source-uv-recommended-for-development)
below).

---

## From Source (uv, recommended for development)

LPF uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. A single `uv sync` command creates the virtualenv, installs
the right Python (3.12 by default, see `.python-version`), pulls every
dependency from the locked versions in `uv.lock`, and installs LPF in
editable mode.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux / macOS
# Windows: see https://docs.astral.sh/uv/getting-started/installation/
```

### 2. Clone and sync

```bash
git clone https://github.com/cxinsys/lpf.git
cd lpf
```

Pick the extras that match your CUDA toolkit (`nvidia-smi`) and the
features you need:

```bash
# Most common: GPU + perceptual objectives (CUDA 13.0 example)
uv sync --extra cu130 --extra evosearch

# Add the optional PyTorch backend (matching CUDA index is auto-selected)
uv sync --extra cu130 --extra evosearch --extra torch-cu130

# CPU only (no CuPy, no GPU)
uv sync
```

### Available extras

| Extra | What it pulls in |
|-------|------------------|
| `cu126` / `cu128` / `cu130` / `cu132` | Matching CuPy (`cupy-cuda12x` or `cupy-cuda13x`) |
| `torch-cu126` / `torch-cu128` / `torch-cu130` / `torch-cu132` | PyTorch + torchvision from the matching `download.pytorch.org/whl/cuXXX` index |
| `evosearch` | `lpips`, `opencv-python`, `torchmetrics[image]` (perceptual / SSIM objectives) |

CUDA extras are mutually exclusive — pick exactly one. Same for `torch-cu*`.
Use the `cuXXX` digits that match your CUDA toolkit. CUDA 13.2 has no
dedicated PyTorch wheel: `torch-cu132` falls back to the cu130 build,
which is forward-compatible with CUDA 13.x runtimes.

### Running things

`uv run` executes a command inside the synced env without needing to
activate it manually:

```bash
uv run pytest tests/ -v
uv run python benchmarks/bench_kernel_vs_python.py
```

Or activate it the traditional way:

```bash
source .venv/bin/activate
```

### Updating dependencies

```bash
uv lock --upgrade        # refresh uv.lock to the newest compatible versions
uv sync                  # apply the refreshed lock to the env
```

`uv.lock` is committed to the repo so every contributor (and CI) gets a
bit-identical environment.

> **Note:** CuPy is required for CUDA kernel acceleration regardless of
> whether you use `device="cuda:0"` or `device="torch:gpu:0"`. When using
> PyTorch, LPF internally bridges PyTorch tensors to CuPy via DLPack
> (zero-copy) for kernel execution.

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

LPF provides two ways to compile CUDA kernels: **JIT** and **AOT**.
Both produce identical results. The system selects the best available option automatically.

### JIT (Just-In-Time): Default, zero setup

Kernels are compiled at runtime on first use via CuPy's NVRTC compiler.

- **No `nvcc` needed** (CuPy includes NVRTC)
- **First-run overhead:** ~2-3 seconds (cached afterwards)
- **Supported dtypes:** float32, float64

### AOT (Ahead-Of-Time): Pre-built wheels or manual build

Pre-built wheels from GitHub Releases already include AOT binaries.
To build manually:

```bash
# Install nvcc (choose one):
conda install -c nvidia cuda-nvcc     # conda

# or install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
```

```bash
# Build AOT kernels:
python -m lpf.kernels.aot.build                # Auto-detect GPU architecture

python -m lpf.kernels.aot.build --arch sm_90   # Specific architecture

python -m lpf.kernels.aot.build --all-arch     # All common architectures (portable)
```

### How auto-selection works

When you call `solver.solve(model)` with a CUDA device, LPF selects the fastest
available path:

1. If the native solver library exists (`libsolver.so` on Linux, `libsolver.dll` on Windows): C solve loop (zero Python overhead during iteration)
2. If `kernels.fatbin` exists: AOT kernels with Python loop
3. Otherwise: JIT compile via NVRTC with Python loop

All three paths produce the same numerical results.

### Why separate wheels for each CUDA version?

LPF follows PyTorch's convention of publishing separate wheels for each CUDA minor version
(cu126, cu128, cu130, cu132), but for a different reason.

**PyTorch** must do this because:

- CUDA minor versions can break ABI compatibility (e.g. 12.1 vs 12.4)
- Bundled libraries (cuDNN, NCCL) are tied to specific CUDA minor versions
- C extension modules are linked directly against the CUDA runtime
- Each minor version requires a different minimum driver version

**LPF** has none of these constraints:

- No C extension modules (loads `.so` via ctypes)
- No cuDNN/NCCL dependencies
- `libsolver.so` and `kernels.fatbin` include PTX, which is forward-compatible across GPU architectures

LPF uses per-minor-version wheels primarily to **pin the matching CuPy package automatically**.
For example, `lpf-0.2.0+cu132` declares `cupy-cuda13x` as a dependency so that
`pip install <wheel>` installs the correct CuPy without extra steps.
If CuPy later splits into finer-grained packages, or if LPF adds tighter CUDA dependencies
in the future, the per-version wheel structure is already in place.

---

## Building Wheels

`build_wheel.py` runs the AOT kernel compilation and then builds a
platform-tagged wheel with the `+cuXXX` local version segment baked in.

```bash
python build_wheel.py                 # AOT compile + wheel (auto-detect GPU)

python build_wheel.py --all-arch      # All common architectures (sm_80~90)

python build_wheel.py --arch sm_90    # Specific architecture

python build_wheel.py --skip-aot      # Skip AOT, use existing binaries

python build_wheel.py --cpu           # CPU-only wheel (no AOT)
```

The script invokes Python's `build` module under the hood, so make sure
it is available (`pip install build`, or it is already in the `dev`
group when you `uv sync`).

---

## Verifying the Installation

If you installed via `uv sync`:

```bash
uv run pytest tests/ -v
```

If you installed a wheel into a regular venv:

```bash
python -m pytest tests/ -v
```

Quick GPU smoke test:

```bash
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
| **GPU (wheel)** | `pip install lpf-0.2.0+cuXXX-....whl` | LPF + CuPy + AOT kernels (zero setup) |
| **GPU (uv, source)** | `uv sync --extra cu130 --extra evosearch` | Same as wheel, plus editable source + locked deps |
| **GPU + PyTorch (uv)** | `uv sync --extra cu130 --extra evosearch --extra torch-cu130` | DLPack bridge to a matching PyTorch CUDA build |
| **CPU only (uv)** | `uv sync` | NumPy solver, no GPU deps (~100 it/s) |
| **AOT rebuild** | `python -m lpf.kernels.aot.build --all-arch` | Recompile fused kernels + native C loop |
