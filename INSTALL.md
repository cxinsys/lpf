# Installation Guide

LPF is distributed as pre-built CUDA wheels and as a source tree. The
recommended path for almost everyone is **a conda environment** —
either with the wheel (for users) or with an editable source install
(for developers). A separate section at the end shows how to do the
same things with `uv` if you prefer that toolchain.

| You want to… | Go to |
|---|---|
| Use LPF as a library | [Quick Install (conda + wheel)](#quick-install-conda--wheel) |
| Run evolutionary search (`lpf.search.evosearch`) | [Evolutionary search](#evolutionary-search) |
| Edit / develop LPF source | [Editable Source Install (conda + `pip install -e`)](#editable-source-install-conda--pip-install--e) |
| Use `uv` instead of conda | [Using uv instead of conda](#using-uv-instead-of-conda) |

---

## Checking your CUDA driver

Run `nvidia-smi` to see what CUDA your **driver** supports:

```
NVIDIA-SMI 580.82       Driver Version: 580.82       CUDA Version: 13.0
```

A few things to know before you pick a wheel:

- You do **NOT** need to install the CUDA Toolkit or `nvcc` separately.
  LPF wheels bundle all the CUDA runtime libraries they need (via the
  matching CuPy build).
- What you **do** need is an NVIDIA driver. `nvidia-smi` confirms it
  is installed and tells you the highest CUDA version it supports.
- LPF ships wheels for **CUDA 12.x** (cu126, cu128) and **CUDA 13.x**
  (cu130, cu132). Within the same major version (12.x or 13.x), wheels
  are forward-compatible with the driver — for example, a driver that
  reports CUDA 13.0 can run the cu132 wheel just fine.
- If your driver only reports CUDA 12.x, stick with cu126 / cu128.
  Crossing the 12 → 13 boundary requires a newer driver.

---

## Quick Install (conda + wheel)

The simplest path: a conda environment plus a single `pip install` of
a pre-built LPF wheel. The wheel includes LPF, the matching CuPy
build, and AOT-compiled CUDA kernels. No source clone, no `nvcc`, no
compiler setup.

```bash
# 1. Create a conda env (Python 3.12 recommended) and activate it
conda create -n lpf python=3.12
conda activate lpf

# 2. (Optional) CUDA PyTorch from the matching index, if you want the
#    torch backend (device="torch:cuda:0"). Install this *before* the
#    LPF wheel so pip reuses the CUDA build instead of pulling the
#    CPU build later.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 3. LPF wheel (CUDA 13.0 example — see the table below for other versions)
pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu130-py3-none-linux_x86_64.whl
```

After this, `conda activate lpf` from any directory gives you a Python
that can `import lpf` and run simulations.

### Available wheels

Pick the row matching your OS and copy the command into the activated
conda env.

#### CUDA 13.2 (auto-installs `cupy-cuda13x`)

| OS      | Command |
|---------|---------|
| Linux   | `pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu132-py3-none-linux_x86_64.whl` |
| Windows | `pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu132-py3-none-win_amd64.whl` |

#### CUDA 13.0 (auto-installs `cupy-cuda13x`)

| OS      | Command |
|---------|---------|
| Linux   | `pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu130-py3-none-linux_x86_64.whl` |
| Windows | `pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu130-py3-none-win_amd64.whl` |

#### CUDA 12.8 (auto-installs `cupy-cuda12x`)

| OS      | Command |
|---------|---------|
| Linux   | `pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu128-py3-none-linux_x86_64.whl` |
| Windows | `pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu128-py3-none-win_amd64.whl` |

#### CUDA 12.6 (auto-installs `cupy-cuda12x`)

| OS      | Command |
|---------|---------|
| Linux   | `pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu126-py3-none-linux_x86_64.whl` |
| Windows | `pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu126-py3-none-win_amd64.whl` |

All wheels: [GitHub Releases](https://github.com/cxinsys/lpf/releases/tag/v0.2.0).

> **CUDA 13.2 + PyTorch**: PyTorch does not publish a dedicated cu132
> wheel. Use the `--index-url .../whl/cu130` PyTorch install (it is
> forward-compatible with CUDA 13.x runtimes) together with the cu132
> LPF wheel. The two `cu*` numbers being different here is expected.

---

## Evolutionary search

`lpf.search.evosearch` drives the optimizer through `pygmo`, and the
fitness functions use perceptual / image-similarity metrics (`lpips`,
`torchmetrics[image]`, `opencv-python`). pygmo's PyPI wheels are
uneven across OS / Python combinations, while conda-forge ships
maintained binaries for Linux, Windows, and macOS — so the
evolutionary-search stack is **only** supported inside a conda env.

This extends the conda setup from the previous section: same env, plus
two extra steps (pygmo from conda-forge, then the perceptual
objectives via pip).

```bash
# 1. Create a conda env and activate it
conda create -n lpf python=3.12
conda activate lpf

# 2. Install pygmo from conda-forge
conda install -c conda-forge pygmo

# 3. CUDA PyTorch from the matching PyTorch index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. Perceptual / image-similarity objectives
pip install lpips torchmetrics[image] opencv-python

# 5. LPF wheel (CUDA 13.0 example)
pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu130-py3-none-linux_x86_64.whl
```

Replace `cu130` with the variant matching your CUDA driver (`cu126`,
`cu128`, `cu130`, `cu132`). The CUDA 13.2 PyTorch fallback note above
applies here too.

If you want an editable source install instead of the wheel, replace
step 5 with the developer recipe in the next section
(`pip install -e ".[cu130]"` after `git clone` and `cd lpf`).

### Full install (everything, user path)

The maximal setup for users: every optional backend (PyTorch, JAX)
plus the full evolutionary-search stack, all in one conda env, on top
of the pre-built LPF wheel. Use this when you want to exercise every
feature without building from source.

```bash
# 1. conda env
conda create -n lpf python=3.12
conda activate lpf

# 2. pygmo (evolutionary search optimizer)
conda install -c conda-forge pygmo

# 3. CUDA PyTorch backend (device="torch:cuda:0")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. CUDA JAX backend (device="jax:gpu:0")
#    cu13x drivers → jax[cuda13]; cu12x drivers → jax[cuda12]
pip install "jax[cuda13]"

# 5. Evolutionary-search fitness objectives
pip install lpips torchmetrics[image] opencv-python

# 6. LPF wheel (pulls in CuPy automatically; CUDA 13.0 example)
pip install https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu130-py3-none-linux_x86_64.whl
```

Replace `cu130` / `cuda13` with the variant matching your CUDA driver.
The CUDA 13.2 PyTorch fallback note in the previous section still
applies (keep the PyTorch index on cu130 but use the cu132 LPF wheel).

---

## Editable Source Install (conda + `pip install -e`)

Use this if you want to **edit LPF source** while running it from your
own scripts, notebooks, or projects elsewhere on disk. The recipe
installs LPF in editable mode into a conda env, so:

- Edits to `~/repos/lpf/...` are picked up immediately by anything
  that runs in the activated env.
- You do **not** have to stay inside the LPF directory to use it.
  Activate the env from any directory and `import lpf` works.

```bash
# 1. Create a dedicated dev env and activate it
conda create -n lpf-dev python=3.12
conda activate lpf-dev

# 2. Clone the source somewhere convenient
git clone https://github.com/cxinsys/lpf.git ~/repos/lpf
cd ~/repos/lpf

# 3. (Optional) CUDA PyTorch from the matching index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. LPF in editable mode + the matching CuPy (CUDA 13.0 example)
#    Chain extras as needed: ".[cu130,jax-cu130]" pulls in JAX too.
pip install -e ".[cu130]"
```

You can now `cd` anywhere, leave the env activated, and use LPF in
your own code:

```bash
cd ~/my-experiment
python sim.py        # imports lpf from ~/repos/lpf in editable mode
jupyter lab          # same env, any working directory
pytest .             # your project tests
```

Pick the `[cuXXX]` extra that matches your CUDA major version
(`cu126` / `cu128` for CUDA 12.x, `cu130` / `cu132` for CUDA 13.x).
See [Available extras](#available-extras) for the full list.

> **`[torch-cuXXX]` extras under `pip install`**: the
> `[torch-cuXXX]` extras rely on `[tool.uv.sources]` to route PyTorch
> to its CUDA-specific wheel index, and that routing is **only
> honored by `uv sync`**. Under `pip install` (or `uv pip install`)
> the same extras silently pull the PyPI default — which is the
> **CPU** PyTorch build. To get a CUDA PyTorch in this option,
> install `torch` from the matching PyTorch index manually first
> (step 3 above) and skip the `[torch-cuXXX]` extra.

> **`pip` vs `uv pip`**: `uv pip install -e ".[cu130]"` is a faster
> drop-in replacement for `pip install -e ".[cu130]"` inside the same
> active conda env. Same resolver inputs, same outputs, same caveats.

### Running tests inside the dev env

```bash
conda activate lpf-dev
cd ~/repos/lpf
pytest tests/ -v
```

### Full install (everything, developer path)

The maximal setup for LPF developers: editable source + every
optional backend + the full evolutionary-search stack, in one conda
env. Same shape as the user full install above, except step 6
replaces the wheel with `pip install -e ".[cu130,jax-cu130]"` so
LPF and JAX are installed together as extras of the source tree.

```bash
# 1. Dev conda env
conda create -n lpf-dev python=3.12
conda activate lpf-dev

# 2. pygmo (evolutionary search optimizer)
conda install -c conda-forge pygmo

# 3. CUDA PyTorch backend (device="torch:cuda:0")
#    Install this BEFORE step 6 so pip reuses the CUDA build instead
#    of pulling the CPU build when LPF's extras resolve.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. Evolutionary-search fitness objectives
pip install lpips torchmetrics[image] opencv-python

# 5. Clone the source
git clone https://github.com/cxinsys/lpf.git ~/repos/lpf
cd ~/repos/lpf

# 6. LPF editable + CuPy + JAX (CUDA 13.0 example)
pip install -e ".[cu130,jax-cu130]"
```

Notes:

- The `[torch-cu130]` extra is intentionally **not** chained in step 6
  because `pip install` does not honor `[tool.uv.sources]`; step 3
  already installed the CUDA PyTorch build manually.
- Swap `cu130` / `jax-cu130` for the variants matching your driver
  (`cu126` / `jax-cu126`, `cu128` / `jax-cu128`, `cu132` / `jax-cu132`).
- Edits to `~/repos/lpf/...` are picked up immediately by anything
  that runs in the `lpf-dev` env, from any working directory.

---

## Using uv instead of conda

[`uv`](https://docs.astral.sh/uv/) is an alternative Python toolchain
that resolves and installs dependencies much faster than pip and pins
them via a `uv.lock` file. Everything below is optional — the conda
recipes above remain the recommended path. Use uv if you already
manage your other projects with it or want lockfile-reproducible
environments.

> **uv cannot install pygmo** reliably across platforms, so the
> evolutionary search stack still requires a conda env. The recipes
> below cover plain simulation and LPF development only.

### Install uv

```bash
pip install uv
# or, if you do not have a Python yet:
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux / macOS
# Windows: see https://docs.astral.sh/uv/getting-started/installation/
```

### Pattern 1 — Add LPF as a dependency in your own uv project

If you manage your own project with uv and just want to use LPF as a
library, add the wheel to your project's dependencies. uv will install
it into your project's `.venv/`.

```bash
mkdir my-experiment
cd my-experiment

uv init
uv add "lpf @ https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu130-py3-none-linux_x86_64.whl"

# Run your code through uv (no manual activation needed)
uv run python sim.py
uv run jupyter lab
```

Or activate the venv the traditional way and use `python` directly:

```bash
source .venv/bin/activate                # Linux / macOS
.venv\Scripts\activate                   # Windows
```

For the optional PyTorch backend, install CUDA PyTorch into the same
venv first (uv project's `[tool.uv.sources]` would otherwise route
through PyPI default):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv add "lpf @ https://github.com/cxinsys/lpf/releases/download/v0.2.0/lpf-0.2.0+cu130-py3-none-linux_x86_64.whl"
```

### Pattern 2 — Develop LPF inside its own checkout (`uv sync`)

This is the analog of the developer recipe above, but using uv instead
of conda. uv reads the `pyproject.toml` and `uv.lock` shipped with the
LPF source tree and creates a project-local `.venv/` with all the
right versions pinned.

```bash
git clone https://github.com/cxinsys/lpf.git
cd lpf

# Pick the extras matching your CUDA toolkit and the features you want
uv sync --extra cu130                              # GPU simulation only
uv sync --extra cu130 --extra torch-cu130          # + PyTorch backend
uv sync --extra cu130 --extra jax-cu130            # + JAX/XLA backend
uv sync --extra cu130 --extra torch-cu130 --extra jax-cu130   # everything
uv sync                                            # CPU only
```

Unlike Pattern 1, the `[torch-cuXXX]` extras **do** work here, because
`uv sync` honors `[tool.uv.sources]` and routes PyTorch to the
matching CUDA wheel index.

`uv sync` always installs into `./.venv/` regardless of any active
conda or virtualenv. Run code through it like this:

```bash
uv run pytest tests/ -v                  # run tests inside .venv
uv run python benchmarks/bench_kernel_vs_python.py
```

Or activate the venv directly:

```bash
source .venv/bin/activate                # Linux / macOS
.venv\Scripts\activate                   # Windows
```

> **Caveat for users who want to combine LPF with their own code**:
> the `.venv/` lives inside the LPF directory, so `uv run` only finds
> it when your current working directory is the LPF tree. If you want
> to use LPF *from your own project elsewhere on disk*, use Pattern 1
> instead, or install LPF editable into your own project with
> `uv add --editable /path/to/lpf` (note: this works for the cuXXX
> extras but the `[torch-cuXXX]` routing is still `uv sync`-only).

#### Updating dependencies in the LPF checkout

```bash
uv lock --upgrade        # refresh uv.lock to the newest compatible versions
uv sync                  # apply the refreshed lock to the env
```

`uv.lock` is committed to the repo so every contributor (and CI) gets
a bit-identical environment.

---

## Available extras

| Extra | What it pulls in |
|-------|------------------|
| `cu126` / `cu128` / `cu130` / `cu132` | Matching CuPy (`cupy-cuda12x` for cu12x, `cupy-cuda13x` for cu13x) |
| `torch-cu126` / `torch-cu128` / `torch-cu130` / `torch-cu132` | PyTorch + torchvision from the matching `download.pytorch.org/whl/cuXXX` index. **Only effective with `uv sync`** — see notes above. |
| `jax` | JAX (CPU only) — for `device="jax:cpu"` |
| `jax-cu126` / `jax-cu128` | `jax[cuda12]` — pulls `jax-cuda12-plugin` + `jax-cuda12-pjrt` |
| `jax-cu130` / `jax-cu132` | `jax[cuda13]` — pulls `jax-cuda13-plugin` + `jax-cuda13-pjrt` |
| `jax-tpu` | `jax[tpu]` — for `device="jax:tpu:0"` |

CUDA extras are mutually exclusive — pick exactly one per family
(`cu*`, `torch-cu*`, `jax*`). CUDA 13.2 has no dedicated PyTorch wheel,
so `torch-cu132` falls back to the cu130 build (forward-compatible
with CUDA 13.x). JAX bundles its CUDA support as `cuda12` / `cuda13`
(not per-minor), so `jax-cu126`/`jax-cu128` both pull `jax[cuda12]`
and `jax-cu130`/`jax-cu132` both pull `jax[cuda13]`.

> **Note**: CuPy is required for CUDA kernel acceleration regardless
> of whether you use `device="cuda:0"` or `device="torch:cuda:0"`. When
> using PyTorch, LPF internally bridges PyTorch tensors to CuPy via
> DLPack (zero-copy) for kernel execution.

---

## GPU Usage

```python
model = LiawModel(..., device="cuda:0")       # CuPy backend (default GPU)

model = LiawModel(..., device="torch:cuda:0") # PyTorch backend (DLPack → CuPy kernels)

model = LiawModel(..., device="jax:gpu:0")    # JAX/XLA backend
# or "jax:cpu", or "jax:tpu:0" if you have a TPU

solver = EulerSolver(dt=0.01, n_iters=500000)
solver.solve(model)  # solver auto-dispatches to the matching backend
```

LPF picks the fastest path per backend automatically:

- `cuda:*` / `torch:cuda:*` → AOT-compiled fused CUDA `.so` (PyTorch is
  bridged via DLPack, zero-copy).
- `jax:gpu:*` / `jax:tpu:*` → `@jax.jit` + `lax.fori_loop`/`scan`
  whole-loop XLA compilation. See
  [benchmarks/benchmark_results_backends.md](benchmarks/benchmark_results_backends.md)
  for the launch-bound vs compute-bound trade-off between these paths.

---

## CUDA Kernel Compilation Modes

LPF provides two ways to compile CUDA kernels: **JIT** and **AOT**.
Both produce identical results. The system selects the best available
option automatically.

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

When you call `solver.solve(model)` with a CUDA device, LPF selects
the fastest available path:

1. If the native solver library exists (`libsolver.so` on Linux,
   `libsolver.dll` on Windows): C solve loop (zero Python overhead
   during iteration)
2. If `kernels.fatbin` exists: AOT kernels with Python loop
3. Otherwise: JIT compile via NVRTC with Python loop

All three paths produce the same numerical results.

### Why separate wheels for each CUDA version?

LPF follows PyTorch's convention of publishing separate wheels for
each CUDA minor version (cu126, cu128, cu130, cu132), but for a
different reason.

**PyTorch** must do this because:

- CUDA minor versions can break ABI compatibility (e.g. 12.1 vs 12.4)
- Bundled libraries (cuDNN, NCCL) are tied to specific CUDA minor versions
- C extension modules are linked directly against the CUDA runtime
- Each minor version requires a different minimum driver version

**LPF** has none of these constraints:

- No C extension modules (loads `.so` via ctypes)
- No cuDNN/NCCL dependencies
- `libsolver.so` and `kernels.fatbin` include PTX, which is forward-compatible across GPU architectures

LPF uses per-minor-version wheels primarily to **pin the matching CuPy
package automatically**. For example, `lpf-0.2.0+cu132` declares
`cupy-cuda13x` as a dependency so that `pip install <wheel>` installs
the correct CuPy without extra steps.

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

The script invokes Python's `build` module under the hood, so make
sure it is available (`pip install build`, or it is already in the
`dev` group when you `uv sync`).

---

## Verifying the Installation

If you installed into a conda env:

```bash
conda activate lpf            # or lpf-dev
python -m pytest tests/ -v    # only meaningful from a source checkout
```

If you installed via `uv sync`:

```bash
uv run pytest tests/ -v
```

Quick GPU smoke test (works in any of the above):

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

| Setup | Command | What you get |
|-------|---------|--------------|
| **conda + wheel** (users) | `conda activate lpf` → `pip install <wheel URL>` | LPF + CuPy + AOT kernels in a conda env |
| **conda + wheel + evosearch** | `conda install -c conda-forge pygmo` → `pip install lpips torchmetrics[image] opencv-python` → `pip install <wheel URL>` | Full evolutionary-search stack |
| **conda + editable source** (developers) | `pip install -e ".[cu130]"` after `conda activate` and `git clone` | Editable LPF source inside a conda dev env |
| **uv project + wheel** | `uv add "lpf @ <wheel URL>"` in your own project | LPF in your project's `.venv/` |
| **uv sync in LPF checkout** | `uv sync --extra cu130` | Editable source + locked deps in `lpf/.venv/` |
| **uv sync + PyTorch backend** | `uv sync --extra cu130 --extra torch-cu130` | DLPack bridge to a matching PyTorch CUDA build |
| **uv sync + JAX backend** | `uv sync --extra cu130 --extra jax-cu130` | JAX/XLA backend (`device="jax:gpu:0"`) |
| **AOT rebuild** | `python -m lpf.kernels.aot.build --all-arch` | Recompile fused kernels + native C loop |
