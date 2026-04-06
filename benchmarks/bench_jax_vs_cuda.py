#!/usr/bin/env python3
"""
Benchmark: JAX kernel vs CUDA fused kernel.

Compares the two production GPU paths in LPF for every solver:

  - **CuPy CUDA**  — LPF's hand-written fused CUDA kernel
                     (NVRTC-compiled stencil + reaction + boundary).
                     1 host dispatch per outer step.
  - **PyTorch CUDA** — same fused kernel, reached via DLPack zero-copy
                       bridge from torch tensors.
  - **JAX (CUDA, fori_loop)** — XLA-fused integration loop wrapped in
                                ``jax.jit`` + ``lax.fori_loop``.  Entire
                                ``n_iters`` loop reduces to 1 host dispatch.

The same reaction-diffusion solve is run through every backend with the
same model parameters; correctness is verified against the NumPy
baseline before timing.

Usage:
    python benchmarks/bench_jax_vs_cuda.py [--n_iters N] [--batch B]
"""

import argparse
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Stable Liaw morphs from population/init_pop_01
# ---------------------------------------------------------------------------

_MORPH_CACHE = None


def _load_init_pop_01():
    global _MORPH_CACHE
    if _MORPH_CACHE is not None:
        return _MORPH_CACHE

    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    pop_dir = os.path.normpath(os.path.join(here, "..", "population", "init_pop_01"))
    fnames = sorted(f for f in os.listdir(pop_dir) if f.endswith(".json"))

    morphs = []
    for fn in fnames:
        with open(os.path.join(pop_dir, fn), "r") as fin:
            morphs.append(json.load(fin))

    n = len(morphs)
    params = np.zeros((n, 8), dtype=np.float32)
    init_states = np.zeros((n, 2), dtype=np.float32)
    init_pts_list = []

    for i, m in enumerate(morphs):
        params[i] = [m["Du"], m["Dv"], m["ru"], m["rv"],
                     m["k"], m["su"], m["sv"], m["mu"]]
        init_states[i] = [m["u0"], m["v0"]]
        pts = []
        for k, v in m.items():
            if k.startswith("init_pts_"):
                pts.append((int(v[0]), int(v[1])))
        init_pts_list.append(pts)

    max_pts = max(len(p) for p in init_pts_list)
    init_pts = np.zeros((n, max_pts, 2), dtype=np.uint32)
    for i, pts in enumerate(init_pts_list):
        for j, (r, c) in enumerate(pts):
            init_pts[i, j] = (r, c)

    _MORPH_CACHE = (params, init_states, init_pts)
    return _MORPH_CACHE


def make_model(batch_size, device, grid_size=128, dtype=np.float32):
    from lpf.models import LiawModel
    from lpf.initializers import LiawInitializer

    base_params, base_states, base_pts = _load_init_pop_01()
    n_morphs = base_params.shape[0]
    idx = np.arange(batch_size) % n_morphs

    params = base_params[idx].astype(dtype)
    init_states = base_states[idx].astype(dtype)
    init_pts = base_pts[idx].copy()

    if grid_size != 128:
        scale = grid_size / 128.0
        init_pts = (init_pts.astype(np.float64) * scale).astype(np.uint32)
        init_pts = np.clip(init_pts, 0, grid_size - 1)

    n_init_pts = init_pts.shape[1]

    model = LiawModel(
        initializer=LiawInitializer(init_pts=init_pts, init_states=init_states),
        n_init_pts=n_init_pts,
        params=params,
        width=grid_size, height=grid_size, dx=0.1,
        device=device, dtype=dtype,
    )
    model.initialize()
    return model


def get_numpy(model):
    arr = model.y_mesh
    if hasattr(arr, "block_until_ready"):
        arr.block_until_ready()
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def sync_device(model):
    am_name = type(model.am).__name__
    arr = model.y_mesh
    if am_name == "JaxModule" and hasattr(arr, "block_until_ready"):
        arr.block_until_ready()
    elif am_name == "CupyModule":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif am_name == "TorchModule":
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Backends to compare
# ---------------------------------------------------------------------------

def get_backends():
    """Return a list of (label, device) for every available GPU backend."""
    bs = []

    # JAX GPU
    try:
        import jax
        try:
            if jax.devices("gpu"):
                bs.append(("JAX (CUDA)", "jax:gpu:0"))
        except RuntimeError:
            pass
    except ImportError:
        pass

    # CuPy CUDA fused kernel
    try:
        import cupy  # noqa: F401
        bs.append(("CuPy (CUDA fused)", "cuda:0"))
    except ImportError:
        pass

    # PyTorch CUDA (uses LPF fused kernel via DLPack bridge)
    try:
        import torch
        if torch.cuda.is_available():
            bs.append(("PyTorch (CUDA fused)", "torch:cuda:0"))
    except ImportError:
        pass

    return bs


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def benchmark_solver(solver_cls, device, n_iters, dt, batch_size, grid_size,
                      repeats=3, warmup_iters=None):
    """Run solver and return best wall time of `repeats` measurements."""
    if warmup_iters is None:
        warmup_iters = min(100, n_iters)

    # Warmup pass — also triggers JIT/AOT compile
    m = make_model(batch_size, device, grid_size)
    solver_cls(dt=dt, n_iters=warmup_iters).solve(m, init_model=False, verbose=0)
    sync_device(m)

    times = []
    for _ in range(repeats):
        m = make_model(batch_size, device, grid_size)
        sync_device(m)

        t0 = time.perf_counter()
        solver_cls(dt=dt, n_iters=n_iters).solve(m, init_model=False, verbose=0)
        sync_device(m)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times), m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(args):
    from lpf.solvers import (EulerSolver, HeunSolver, RungeKuttaSolver,
                              RK23Solver, AdamsBashforth2Solver,
                              AdaptiveRKF45Solver, DOPRI5Solver)

    backends = get_backends()
    if not backends:
        print("No GPU backends available — install CuPy / PyTorch / JAX with CUDA.")
        return

    print("=" * 80)
    print(" LPF JAX kernel vs CUDA fused kernel")
    print("=" * 80)
    print(f" Backends: {', '.join(label for label, _ in backends)}")
    print(f" Default: batch_size={args.batch_size}, grid={args.grid_size}x{args.grid_size}, "
          f"n_iters={args.n_iters}, dt={args.dt}")
    print(f" Repeats: {args.repeats} (best-of-{args.repeats})")
    print("=" * 80)

    fixed_solvers = [
        ("EulerSolver", EulerSolver),
        ("HeunSolver", HeunSolver),
        ("RungeKuttaSolver", RungeKuttaSolver),
        ("RK23Solver", RK23Solver),
        ("AdamsBashforth2Solver", AdamsBashforth2Solver),
    ]

    adaptive_solvers = [
        ("AdaptiveRKF45Solver", AdaptiveRKF45Solver),
        ("DOPRI5Solver", DOPRI5Solver),
    ]

    all_results = {}

    # ------------------------------------------------------------------
    # Section 1: Fixed-step solvers
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print(" 1. Fixed-step solvers")
    print("=" * 80)

    for solver_name, solver_cls in fixed_solvers:
        print(f"\n--- {solver_name}  "
              f"(n_iters={args.n_iters}, batch={args.batch_size}, "
              f"{args.grid_size}x{args.grid_size}) ---")
        row = {}
        for label, device in backends:
            try:
                t, _ = benchmark_solver(
                    solver_cls, device, args.n_iters, args.dt,
                    args.batch_size, args.grid_size, args.repeats)
                row[label] = t
                print(f"  {label:24s}  {t*1000:8.2f} ms  "
                      f"({args.n_iters/t:>10,.0f} iter/s)")
            except Exception as e:
                print(f"  {label:24s}  FAIL — {e}")
        all_results[solver_name] = row

        # Speedup of JAX vs CuPy
        jax_t = row.get("JAX (CUDA)")
        cupy_t = row.get("CuPy (CUDA fused)")
        if jax_t is not None and cupy_t is not None:
            speedup = cupy_t / jax_t
            verb = "faster" if speedup > 1 else "slower"
            print(f"\n  >> JAX vs CuPy: {speedup:.2f}x {verb}")

    # ------------------------------------------------------------------
    # Section 2: Adaptive solvers (smaller iter count)
    # ------------------------------------------------------------------
    n_iters_adapt = max(100, args.n_iters // 5)
    print()
    print("=" * 80)
    print(f" 2. Adaptive solvers  (n_iters={n_iters_adapt})")
    print("=" * 80)
    print(" Note: CuPy/PyTorch adaptive solvers use the eager CupyPdefuncMixin")
    print(" path (no fused kernel for adaptive sub-stepping). JAX uses jit+while_loop.")

    for solver_name, solver_cls in adaptive_solvers:
        print(f"\n--- {solver_name}  "
              f"(n_iters={n_iters_adapt}, batch={args.batch_size}, "
              f"{args.grid_size}x{args.grid_size}) ---")
        row = {}
        for label, device in backends:
            try:
                t, _ = benchmark_solver(
                    solver_cls, device, n_iters_adapt, args.dt,
                    args.batch_size, args.grid_size, args.repeats)
                row[label] = t
                print(f"  {label:24s}  {t*1000:8.2f} ms  "
                      f"({n_iters_adapt/t:>10,.0f} iter/s)")
            except Exception as e:
                print(f"  {label:24s}  FAIL — {e}")
        all_results[solver_name] = row

        jax_t = row.get("JAX (CUDA)")
        cupy_t = row.get("CuPy (CUDA fused)")
        if jax_t is not None and cupy_t is not None:
            speedup = cupy_t / jax_t
            verb = "faster" if speedup > 1 else "slower"
            print(f"\n  >> JAX vs CuPy: {speedup:.2f}x {verb}")

    # ------------------------------------------------------------------
    # Section 3: Correctness check (Euler @ default settings)
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print(" 3. Correctness check (EulerSolver, vs NumPy CPU baseline)")
    print("=" * 80)

    m_ref = make_model(args.batch_size, "cpu", args.grid_size)
    EulerSolver(dt=args.dt, n_iters=args.n_iters).solve(
        m_ref, init_model=False, verbose=0)
    y_ref = get_numpy(m_ref)

    for label, device in backends:
        try:
            m = make_model(args.batch_size, device, args.grid_size)
            EulerSolver(dt=args.dt, n_iters=args.n_iters).solve(
                m, init_model=False, verbose=0)
            y = get_numpy(m)
            max_diff = float(np.max(np.abs(y_ref - y)))
            ok = np.allclose(y_ref, y, atol=1e-4, rtol=1e-3)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label:24s}  max_diff={max_diff:.6e}")
        except Exception as e:
            print(f"  [ERR ] {label:24s}  {e}")

    # ------------------------------------------------------------------
    # Section 4: Grid scaling (Euler only)
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print(f" 4. Grid scaling — EulerSolver x{args.n_iters}  (batch={args.batch_size})")
    print("=" * 80)

    grid_sizes = args.grid_sizes
    header = f"  {'Grid':>9s}"
    for label, _ in backends:
        header += f"  {label:>22s}"
    print(header)
    print("  " + "-" * (9 + 24 * len(backends)))
    for gs in grid_sizes:
        line = f"  {gs:>4d}x{gs:<3d}"
        for label, device in backends:
            try:
                t, _ = benchmark_solver(
                    EulerSolver, device, args.n_iters, args.dt,
                    args.batch_size, gs, args.repeats)
                line += f"  {t*1000:>18.2f} ms   "
            except Exception:
                line += f"  {'SKIP':>22s}  "
        print(line)

    # ------------------------------------------------------------------
    # Section 5: Batch scaling
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print(f" 5. Batch scaling — EulerSolver x{args.n_iters}  "
          f"(grid={args.grid_size}x{args.grid_size})")
    print("=" * 80)

    header = f"  {'Batch':>8s}"
    for label, _ in backends:
        header += f"  {label:>22s}"
    print(header)
    print("  " + "-" * (8 + 24 * len(backends)))
    for bs in args.batch_sizes:
        line = f"  {bs:>8d}"
        for label, device in backends:
            try:
                t, _ = benchmark_solver(
                    EulerSolver, device, args.n_iters, args.dt,
                    bs, args.grid_size, args.repeats)
                line += f"  {t*1000:>18.2f} ms   "
            except Exception:
                line += f"  {'SKIP':>22s}  "
        print(line)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print(" SUMMARY: best time per solver / backend")
    print("=" * 80)

    backend_labels = [label for label, _ in backends]
    header = f"  {'Solver':<22s}"
    for lbl in backend_labels:
        header += f"  {lbl:>22s}"
    header += f"  {'JAX/CuPy':>10s}"
    print(header)
    print("  " + "-" * (22 + 24 * len(backend_labels) + 12))

    for solver_name, _ in fixed_solvers + adaptive_solvers:
        row = all_results.get(solver_name, {})
        line = f"  {solver_name:<22s}"
        for lbl in backend_labels:
            t = row.get(lbl)
            if t is None:
                line += f"  {'-':>22s}"
            else:
                line += f"  {t*1000:>18.2f} ms   "
        jax_t = row.get("JAX (CUDA)")
        cupy_t = row.get("CuPy (CUDA fused)")
        if jax_t and cupy_t:
            line += f"  {cupy_t/jax_t:>9.2f}x"
        else:
            line += f"  {'-':>10s}"
        print(line)

    print()
    print("=" * 80)
    print(" Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX kernel vs CUDA fused kernel")
    parser.add_argument("--n_iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grid_sizes", type=int, nargs="+",
                        default=[64, 128, 256])
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                        default=[1, 4, 16])
    args = parser.parse_args()
    run_benchmark(args)
