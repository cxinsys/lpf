#!/usr/bin/env python3
"""
Benchmark: cross-backend performance comparison.

Compares NumPy, JAX (CPU), PyTorch (CPU/CUDA), and CuPy (CUDA) across
multiple solvers, grid sizes, and batch sizes.

Usage:
    python benchmarks/bench_backends.py [--n_iters N] [--batch_size B]
"""

import argparse
import time
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def create_model(batch_size, device, grid_size=128, dtype=np.float32):
    from lpf.models import LiawModel
    from lpf.initializers import LiawInitializer

    height = width = grid_size
    n_init_pts = 20

    np.random.seed(42)
    init_pts = np.random.randint(0, min(height, width),
                                 size=(batch_size, n_init_pts, 2)).astype(np.uint32)
    init_states = np.random.uniform(0.1, 1.0, size=(batch_size, 2)).astype(dtype)
    log_params = np.random.uniform(-3, 0, size=(batch_size, 8)).astype(dtype)
    params = 10.0 ** log_params

    model = LiawModel(
        initializer=LiawInitializer(init_pts=init_pts, init_states=init_states),
        n_init_pts=n_init_pts,
        params=params,
        width=width, height=height, dx=0.1,
        device=device, dtype=dtype,
    )
    return model


def get_numpy(model):
    arr = model.y_mesh
    if hasattr(arr, 'get'):
        return arr.get()
    if hasattr(arr, 'cpu'):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def sync_device(model):
    """Synchronize GPU to get accurate timing."""
    am = model.am
    am_name = type(am).__name__
    if am_name == "CupyModule":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif am_name == "TorchModule":
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------

def get_available_backends():
    backends = [("numpy", "cpu", "NumPy (CPU)")]

    try:
        import jax  # noqa: F401
        backends.append(("jax:cpu", "jax:cpu", "JAX (CPU)"))
    except ImportError:
        pass

    try:
        import torch  # noqa: F401
        backends.append(("torch:cpu", "torch:cpu", "PyTorch (CPU)"))
        if torch.cuda.is_available():
            backends.append(("torch:cuda:0", "torch:cuda:0", "PyTorch (CUDA)"))
    except ImportError:
        pass

    try:
        import cupy  # noqa: F401
        backends.append(("cuda:0", "cuda:0", "CuPy (CUDA)"))
    except ImportError:
        pass

    return backends


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_solve(solver_cls, device, n_iters, dt, batch_size, grid_size,
                    warmup_iters=None, repeats=3, solver_kwargs=None):
    """Create model, warm up, then time `repeats` runs. Returns best time."""
    if warmup_iters is None:
        warmup_iters = min(100, n_iters)
    if solver_kwargs is None:
        solver_kwargs = {}

    model = create_model(batch_size, device, grid_size)
    model.initialize()

    # Warmup
    solver = solver_cls(dt=dt, n_iters=warmup_iters, **solver_kwargs)
    solver.solve(model, init_model=True, verbose=0)
    sync_device(model)

    # Timed runs
    times = []
    for _ in range(repeats):
        model.initialize()
        sync_device(model)

        solver = solver_cls(dt=dt, n_iters=n_iters, **solver_kwargs)
        t0 = time.perf_counter()
        solver.solve(model, init_model=False, verbose=0)
        sync_device(model)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    best = min(times)
    return best, model


def benchmark_pdefunc(device, n_calls, batch_size, grid_size,
                      warmup_calls=50, repeats=3):
    """Time raw pdefunc calls."""
    model = create_model(batch_size, device, grid_size)
    model.initialize()

    # Warmup
    for _ in range(warmup_calls):
        model.pdefunc(t=0.0, y_mesh=model.y_mesh)
    sync_device(model)

    times = []
    for _ in range(repeats):
        sync_device(model)
        t0 = time.perf_counter()
        for _ in range(n_calls):
            model.pdefunc(t=0.0, y_mesh=model.y_mesh)
        sync_device(model)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(args):
    from lpf.solvers import (EulerSolver, HeunSolver, RungeKuttaSolver,
                              RK23Solver, AdamsBashforth2Solver)

    backends = get_available_backends()
    backend_names = [b[2] for b in backends]

    print("=" * 78)
    print(" LPF Cross-Backend Benchmark")
    print("=" * 78)
    print(f" Backends: {', '.join(backend_names)}")
    print(f" Default: batch_size={args.batch_size}, grid={args.grid_size}x{args.grid_size}, "
          f"n_iters={args.n_iters}, dt={args.dt}")
    print(f" Repeats: {args.repeats} (best-of-{args.repeats})")
    print("=" * 78)

    all_results = {}

    # ------------------------------------------------------------------
    # 1. pdefunc micro-benchmark
    # ------------------------------------------------------------------
    n_pdefunc_calls = 500
    print(f"\n{'─' * 78}")
    print(f" 1. pdefunc() ×{n_pdefunc_calls}  "
          f"(batch={args.batch_size}, grid={args.grid_size}×{args.grid_size})")
    print(f"{'─' * 78}")

    pdefunc_results = {}
    for device_key, device_str, label in backends:
        try:
            elapsed = benchmark_pdefunc(
                device_str, n_pdefunc_calls, args.batch_size, args.grid_size,
                repeats=args.repeats)
            per_call = elapsed / n_pdefunc_calls * 1000  # ms
            pdefunc_results[label] = elapsed
            print(f"  {label:24s}  {elapsed:8.4f}s  ({per_call:.3f} ms/call)")
        except Exception as e:
            print(f"  {label:24s}  SKIP ({e})")

    if pdefunc_results:
        ref = pdefunc_results.get("NumPy (CPU)")
        if ref:
            print()
            for label, t in pdefunc_results.items():
                if label == "NumPy (CPU)":
                    continue
                ratio = ref / t
                faster = "faster" if ratio > 1 else "slower"
                print(f"    {label} vs NumPy: {ratio:.2f}x {faster}")

    # ------------------------------------------------------------------
    # 2. Solver benchmarks
    # ------------------------------------------------------------------
    solvers = [
        ("EulerSolver", EulerSolver, {}),
        ("HeunSolver", HeunSolver, {}),
        ("RungeKuttaSolver", RungeKuttaSolver, {}),
        ("RK23Solver", RK23Solver, {}),
        ("AB2Solver", AdamsBashforth2Solver, {}),
    ]

    solver_results = {}

    for solver_name, solver_cls, solver_kwargs in solvers:
        print(f"\n{'─' * 78}")
        print(f" 2. {solver_name}  "
              f"(n_iters={args.n_iters}, batch={args.batch_size}, "
              f"grid={args.grid_size}×{args.grid_size})")
        print(f"{'─' * 78}")

        row = {}
        for device_key, device_str, label in backends:
            try:
                elapsed, model = benchmark_solve(
                    solver_cls, device_str, args.n_iters, args.dt,
                    args.batch_size, args.grid_size,
                    repeats=args.repeats, solver_kwargs=solver_kwargs)
                iters_s = args.n_iters / elapsed
                row[label] = elapsed
                print(f"  {label:24s}  {elapsed:8.4f}s  ({iters_s:>10,.0f} iter/s)")
            except Exception as e:
                print(f"  {label:24s}  SKIP ({e})")

        solver_results[solver_name] = row

        # Speedup vs NumPy
        ref = row.get("NumPy (CPU)")
        if ref and len(row) > 1:
            print()
            for label, t in row.items():
                if label == "NumPy (CPU)":
                    continue
                ratio = ref / t
                faster = "faster" if ratio > 1 else "slower"
                print(f"    {label} vs NumPy: {ratio:.2f}x {faster}")

    all_results["solvers"] = solver_results

    # ------------------------------------------------------------------
    # 3. Grid size scaling (Euler only)
    # ------------------------------------------------------------------
    grid_sizes = args.grid_sizes
    euler_iters = args.n_iters

    print(f"\n{'─' * 78}")
    print(f" 3. Grid scaling — EulerSolver ×{euler_iters}  (batch={args.batch_size})")
    print(f"{'─' * 78}")

    header = f"  {'Grid':>8s}"
    for _, _, label in backends:
        header += f"  {label:>16s}"
    print(header)
    print("  " + "─" * (8 + 18 * len(backends)))

    grid_results = {}
    for gs in grid_sizes:
        line = f"  {gs:>4d}×{gs:<3d}"
        row = {}
        for device_key, device_str, label in backends:
            try:
                elapsed, _ = benchmark_solve(
                    EulerSolver, device_str, euler_iters, args.dt,
                    args.batch_size, gs, repeats=args.repeats)
                iters_s = euler_iters / elapsed
                row[label] = elapsed
                line += f"  {elapsed:>12.4f}s   "
            except Exception:
                line += f"  {'SKIP':>12s}    "
        print(line)
        grid_results[gs] = row

    all_results["grid_scaling"] = grid_results

    # ------------------------------------------------------------------
    # 4. Batch size scaling (Euler, 128x128)
    # ------------------------------------------------------------------
    batch_sizes = args.batch_sizes

    print(f"\n{'─' * 78}")
    print(f" 4. Batch scaling — EulerSolver ×{euler_iters}  (grid={args.grid_size}×{args.grid_size})")
    print(f"{'─' * 78}")

    header = f"  {'Batch':>8s}"
    for _, _, label in backends:
        header += f"  {label:>16s}"
    print(header)
    print("  " + "─" * (8 + 18 * len(backends)))

    batch_results = {}
    for bs in batch_sizes:
        line = f"  {bs:>8d}"
        row = {}
        for device_key, device_str, label in backends:
            try:
                elapsed, _ = benchmark_solve(
                    EulerSolver, device_str, euler_iters, args.dt,
                    bs, args.grid_size, repeats=args.repeats)
                row[label] = elapsed
                line += f"  {elapsed:>12.4f}s   "
            except Exception:
                line += f"  {'SKIP':>12s}    "
        print(line)
        batch_results[bs] = row

    all_results["batch_scaling"] = batch_results

    # ------------------------------------------------------------------
    # 5. Correctness verification
    # ------------------------------------------------------------------
    print(f"\n{'─' * 78}")
    print(f" 5. Correctness verification (EulerSolver ×{args.n_iters})")
    print(f"{'─' * 78}")

    ref_model = create_model(args.batch_size, "cpu", args.grid_size)
    ref_model.initialize()
    EulerSolver(dt=args.dt, n_iters=args.n_iters).solve(
        ref_model, init_model=False, verbose=0)
    y_ref = get_numpy(ref_model)

    for device_key, device_str, label in backends:
        if device_key == "numpy":
            continue
        try:
            m = create_model(args.batch_size, device_str, args.grid_size)
            m.initialize()
            EulerSolver(dt=args.dt, n_iters=args.n_iters).solve(
                m, init_model=False, verbose=0)
            y_other = get_numpy(m)

            max_diff = np.max(np.abs(y_ref - y_other))
            match = np.allclose(y_ref, y_other, atol=1e-4, rtol=1e-3)
            status = "PASS" if match else "FAIL"
            print(f"  [{status}] {label:24s}  max_diff={max_diff:.6e}")
        except Exception as e:
            print(f"  [SKIP] {label:24s}  {e}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f" Summary: Speedup vs NumPy (CPU)  — EulerSolver ×{args.n_iters}")
    print(f"{'=' * 78}")

    euler_row = solver_results.get("EulerSolver", {})
    np_time = euler_row.get("NumPy (CPU)")
    if np_time:
        print(f"  {'Backend':24s}  {'Time':>10s}  {'Speedup':>10s}  {'iter/s':>12s}")
        print(f"  {'─' * 60}")
        for _, _, label in backends:
            t = euler_row.get(label)
            if t is not None:
                speedup = np_time / t
                iters_s = args.n_iters / t
                marker = " (baseline)" if label == "NumPy (CPU)" else ""
                print(f"  {label:24s}  {t:>9.4f}s  {speedup:>9.2f}x  {iters_s:>11,.0f}{marker}")

    print(f"\n{'=' * 78}")
    print(" Benchmark complete.")
    print(f"{'=' * 78}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPF Cross-Backend Benchmark")
    parser.add_argument("--n_iters", type=int, default=2000,
                        help="Number of time steps (default: 2000)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--grid_size", type=int, default=128,
                        help="Grid height/width (default: 128)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step (default: 0.01)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeats per measurement, report best (default: 3)")
    parser.add_argument("--grid_sizes", type=int, nargs="+",
                        default=[32, 64, 128],
                        help="Grid sizes for scaling test (default: 32 64 128)")
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                        default=[1, 4, 16],
                        help="Batch sizes for scaling test (default: 1 4 16)")
    args = parser.parse_args()
    run_benchmark(args)
