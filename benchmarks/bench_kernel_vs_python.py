"""Benchmark: Fused CUDA Kernels vs Python Element-wise (CuPy) on GPU.

Compares two GPU execution paths:
  1. Fused CUDA kernel  — single kernel launch: Laplacian + Reaction fused
  2. Python element-wise — Python loop calling CuPy array ops (laplacian, reaction separately)

Both paths run on GPU with CuPy arrays. The difference is kernel fusion vs
multiple element-wise CuPy operations orchestrated by a Python loop.
"""

import os
import time

import numpy as np

from lpf.initializers import LiawInitializer
from lpf.models import LiawModel
from lpf.solvers import EulerSolver, RungeKuttaSolver
from lpf.data import load_model_dicts

# ── Parameters ──────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dpath_pop = os.path.join(REPO_ROOT, "population", "init_pop_axyridis")

dt = 0.01
dx = 0.1
width = 128
height = 128
n_iters = 500_000

model_dicts = load_model_dicts(dpath_pop)
print(f"Batch size : {len(model_dicts)} models")
print(f"Grid       : {height}x{width}, dt={dt}")
print(f"Iterations : {n_iters:,}")
print()


# ── Helpers ─────────────────────────────────────────────────────────────
def create_model(device):
    initializer = LiawInitializer()
    initializer.update(model_dicts)
    params = LiawModel.parse_params(model_dicts)
    return LiawModel(
        initializer=initializer, params=params,
        dx=dx, width=width, height=height, device=device,
    )


def sync_gpu():
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


# ── Python element-wise solver (bypasses fused CUDA kernels) ────────────
class PythonEulerSolver(EulerSolver):
    """Forces the Python element-wise path even on CUDA models.

    Overrides _is_cuda() to return False so that solver.solve() uses
    the generic Python loop (line 201 of solver.py) which calls
    model.pdefunc() — a sequence of CuPy element-wise operations.
    """

    @staticmethod
    def _is_cuda(model):
        return False  # Force Python loop path


class PythonRungeKuttaSolver(RungeKuttaSolver):
    """Forces the Python element-wise path for RK4."""

    @staticmethod
    def _is_cuda(model):
        return False


# ── Benchmark function ──────────────────────────────────────────────────
def benchmark(solver, model, label, n_iters, n_warmup=500):
    # Warm-up (JIT compilation, memory allocation, etc.)
    solver.solve(model=model, dt=dt, n_iters=n_warmup, init_model=True, verbose=0)
    sync_gpu()

    # Measure
    model.initialize()
    sync_gpu()
    t0 = time.perf_counter()
    solver.solve(model=model, dt=dt, n_iters=n_iters, init_model=False, verbose=0)
    sync_gpu()
    elapsed = time.perf_counter() - t0
    ips = n_iters / elapsed
    print(f"  [{label:<45s}] {elapsed:7.2f}s  ({ips:>10,.0f} iters/s)")
    return elapsed, ips


# ══════════════════════════════════════════════════════════════════════════
#   Euler Solver
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print(f"  Euler Solver  ({len(model_dicts)} models, {n_iters:,} iters)")
print("=" * 72)

# 1) Fused CUDA kernel (normal path)
e_kernel, ips_kernel = benchmark(
    EulerSolver(), create_model("cuda:0"),
    "Fused CUDA kernel", n_iters)

# 2) Python element-wise on GPU (CuPy array ops, no fusion)
e_python, ips_python = benchmark(
    PythonEulerSolver(), create_model("cuda:0"),
    "Python element-wise (CuPy on GPU)", n_iters)

# 3) PyTorch DLPack -> fused CUDA kernel
e_torch, ips_torch = benchmark(
    EulerSolver(), create_model("torch:cuda:0"),
    "PyTorch -> DLPack -> Fused CUDA kernel", n_iters)

print()
speedup_euler = ips_kernel / ips_python
print(f"  >>> Fused kernel is {speedup_euler:.1f}x faster than Python element-wise (Euler)")
print()


# ══════════════════════════════════════════════════════════════════════════
#   RK4 Solver  (4 PDE evaluations per step -> more kernel launches)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print(f"  RK4 Solver  ({len(model_dicts)} models, {n_iters:,} iters)")
print("=" * 72)

# 1) Fused CUDA kernel
_, ips_rk4_kernel = benchmark(
    RungeKuttaSolver(), create_model("cuda:0"),
    "Fused CUDA kernel", n_iters)

# 2) Python element-wise on GPU
_, ips_rk4_python = benchmark(
    PythonRungeKuttaSolver(), create_model("cuda:0"),
    "Python element-wise (CuPy on GPU)", n_iters)

# 3) PyTorch DLPack
_, ips_rk4_torch = benchmark(
    RungeKuttaSolver(), create_model("torch:cuda:0"),
    "PyTorch -> DLPack -> Fused CUDA kernel", n_iters)

print()
speedup_rk4 = ips_rk4_kernel / ips_rk4_python
print(f"  >>> Fused kernel is {speedup_rk4:.1f}x faster than Python element-wise (RK4)")
print()


# ══════════════════════════════════════════════════════════════════════════
#   fast_math comparison (Euler, fused kernel)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  fast_math comparison (Euler, Fused CUDA kernel)")
print("=" * 72)

_, ips_normal = benchmark(
    EulerSolver(fast_math=False), create_model("cuda:0"),
    "fast_math=False", n_iters)
_, ips_fast = benchmark(
    EulerSolver(fast_math=True), create_model("cuda:0"),
    "fast_math=True", n_iters)
print(f"\n  fast_math speedup: {ips_fast / ips_normal:.2f}x")
print()


# ══════════════════════════════════════════════════════════════════════════
#   Summary
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print(f"  {'Method':<45s} {'iters/s':>12s} {'vs Python':>10s}")
print("-" * 72)

all_results = {
    "Euler: Fused CUDA kernel":      ips_kernel,
    "Euler: Python element-wise":    ips_python,
    "Euler: PyTorch DLPack":         ips_torch,
    "RK4:   Fused CUDA kernel":      ips_rk4_kernel,
    "RK4:   Python element-wise":    ips_rk4_python,
    "RK4:   PyTorch DLPack":         ips_rk4_torch,
}

for solver_name, baseline_key in [("Euler", "Euler: Python element-wise"),
                                   ("RK4", "RK4:   Python element-wise")]:
    base_ips = all_results[baseline_key]
    for key, ips in all_results.items():
        if key.startswith(solver_name):
            speedup = ips / base_ips
            print(f"  {key:<45s} {ips:>12,.0f} {speedup:>9.1f}x")
    print()


# ══════════════════════════════════════════════════════════════════════════
#   Numerical verification
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  Numerical verification (1000 iters)")
print("=" * 72)

n_verify = 1000

# Fused kernel path
m_kernel = create_model("cuda:0")
EulerSolver().solve(model=m_kernel, dt=dt, n_iters=n_verify, verbose=0)
u_kernel = m_kernel.am.get(m_kernel.u)

# Python element-wise path (same CuPy arrays, no fusion)
m_python = create_model("cuda:0")
PythonEulerSolver().solve(model=m_python, dt=dt, n_iters=n_verify, verbose=0)
u_python = m_python.am.get(m_python.u)

# PyTorch DLPack path
m_torch = create_model("torch:cuda:0")
EulerSolver().solve(model=m_torch, dt=dt, n_iters=n_verify, verbose=0)
u_torch = m_torch.am.get(m_torch.u)

print(f"  Fused kernel vs Python element-wise:")
print(f"    max|diff| = {np.max(np.abs(u_kernel - u_python)):.2e}")
print(f"    allclose  = {np.allclose(u_kernel, u_python, rtol=1e-4)}")
print()
print(f"  Fused kernel vs PyTorch DLPack:")
print(f"    max|diff| = {np.max(np.abs(u_kernel - u_torch)):.2e}")
print(f"    allclose  = {np.allclose(u_kernel, u_torch)}")
print()
print("Done.")
