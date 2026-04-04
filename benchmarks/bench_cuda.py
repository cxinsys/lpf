#!/usr/bin/env python3
"""
Benchmark: Python (CuPy) vs CUDA Fused Kernel solvers.

Compares performance of:
  1. EulerSolver   (CuPy backend - existing Python implementation)
  2. CuEulerSolver (fused CUDA kernel - new implementation)
  3. RungeKuttaSolver (CuPy backend)
  4. CuRungekuttaSolver (fused CUDA kernel)

Also verifies numerical correctness by comparing final states.

Usage:
    python benchmarks/bench_cuda.py [--n_iters N] [--batch_size B]
"""

import argparse
import time
import sys
import os

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_model(model_class, batch_size, device, dtype=np.float32):
    """Create a model with random parameters and initialization."""
    from lpf.initializers import LiawInitializer

    height, width = 128, 128
    n_init_pts = 25

    # Random initial points
    np.random.seed(42)
    init_pts = np.random.randint(0, height, size=(batch_size, n_init_pts, 2))
    init_pts = init_pts.astype(np.uint32)

    # Random initial states
    init_states = np.random.uniform(0.1, 1.0, size=(batch_size, 2)).astype(dtype)

    initializer = LiawInitializer(
        init_pts=init_pts,
        init_states=init_states,
    )

    # Model-specific random parameters (log-uniform)
    if model_class.__name__ == "LiawModel":
        # params: Du, Dv, ru, rv, k, su, sv, mu
        log_params = np.random.uniform(-3, 0, size=(batch_size, 8)).astype(dtype)
        params = 10.0 ** log_params
    elif model_class.__name__ == "GrayScottModel":
        # params: Du, Dv, F, k
        log_params = np.random.uniform(-3, 0, size=(batch_size, 4)).astype(dtype)
        params = 10.0 ** log_params
    elif model_class.__name__ == "SchnakenbergModel":
        log_params = np.random.uniform(-3, 0, size=(batch_size, 6)).astype(dtype)
        params = 10.0 ** log_params
    elif model_class.__name__ == "GiererMeinhardtModel":
        log_params = np.random.uniform(-3, 0, size=(batch_size, 6)).astype(dtype)
        params = 10.0 ** log_params
    else:
        raise ValueError(f"Unknown model class: {model_class.__name__}")

    model = model_class(
        initializer=initializer,
        n_init_pts=n_init_pts,
        params=params,
        width=width,
        height=height,
        dx=0.1,
        device=device,
        dtype=dtype,
    )

    return model


def benchmark_solver(solver, model, n_iters, dt=0.01, warmup_iters=100, label=""):
    """Run solver and measure elapsed time."""
    import cupy as cp

    # Warmup
    model.initialize()
    warmup_solver = solver.__class__(dt=dt, n_iters=warmup_iters)
    if hasattr(warmup_solver, '_fast_math'):
        warmup_solver._fast_math = getattr(solver, '_fast_math', False)
    warmup_solver.solve(model, dt=dt, n_iters=warmup_iters, init_model=True, verbose=0)
    cp.cuda.Stream.null.synchronize()

    # Actual benchmark
    model.initialize()
    cp.cuda.Stream.null.synchronize()

    t_start = time.perf_counter()
    solver.solve(model, dt=dt, n_iters=n_iters, init_model=False, verbose=0)
    cp.cuda.Stream.null.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    iters_per_sec = n_iters / elapsed

    print(f"  {label:40s}  {elapsed:8.3f}s  ({iters_per_sec:,.0f} iter/s)")

    return elapsed, model


def verify_correctness(model_a, model_b, label_a, label_b, atol=1e-3, rtol=1e-3):
    """Compare final states of two models for numerical agreement.
    Handles NaN values that may occur with unstable random parameters."""
    import cupy as cp

    y_a = cp.asnumpy(model_a.y_mesh)
    y_b = cp.asnumpy(model_b.y_mesh)

    # Check NaN agreement (both should have NaN in same locations)
    nan_a = np.isnan(y_a)
    nan_b = np.isnan(y_b)
    nan_match = np.array_equal(nan_a, nan_b)

    # Compare non-NaN values
    valid = ~(nan_a | nan_b)
    if valid.any():
        diff = np.abs(y_a[valid] - y_b[valid])
        max_diff = np.max(diff)
        rel_diff = max_diff / (np.max(np.abs(y_a[valid])) + 1e-30)
        values_match = np.allclose(y_a[valid], y_b[valid], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        rel_diff = 0.0
        values_match = True

    match = nan_match and values_match
    status = "PASS" if match else "FAIL"
    nan_info = f", NaN count: {nan_a.sum()}" if nan_a.any() else ""

    print(f"  [{status}] {label_a} vs {label_b}: "
          f"max_abs_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}{nan_info}")

    return match


def run_benchmark(args):
    from lpf.models import LiawModel, GrayScottModel
    from lpf.solvers import EulerSolver, RungeKuttaSolver

    print("=" * 72)
    print(f" LPF CUDA Kernel Benchmark")
    print(f" Model: {args.model}, Batch: {args.batch_size}, "
          f"Grid: 128x128, Iters: {args.n_iters:,}")
    print("=" * 72)

    model_classes = {
        "Liaw": LiawModel,
        "GrayScott": GrayScottModel,
    }
    model_class = model_classes[args.model]
    device = "cuda:0"

    # ---- Euler Solvers ----
    print("\n--- Euler Method ---")

    solver_py = EulerSolver(dt=args.dt, n_iters=args.n_iters)
    solver_cuda = EulerSolver(dt=args.dt, n_iters=args.n_iters,
                              backend="cuda")
    solver_cuda_fm = EulerSolver(dt=args.dt, n_iters=args.n_iters,
                                 backend="cuda", fast_math=True)

    model_py = create_model(model_class, args.batch_size, device)
    t_py, model_py = benchmark_solver(
        solver_py, model_py, args.n_iters, args.dt,
        label="EulerSolver (python)")

    model_cuda = create_model(model_class, args.batch_size, device)
    t_cuda, model_cuda = benchmark_solver(
        solver_cuda, model_cuda, args.n_iters, args.dt,
        label="EulerSolver (backend='cuda')")

    model_cuda_fm = create_model(model_class, args.batch_size, device)
    t_cuda_fm, model_cuda_fm = benchmark_solver(
        solver_cuda_fm, model_cuda_fm, args.n_iters, args.dt,
        label="EulerSolver (cuda + fast_math)")

    print(f"\n  Speedup (cuda vs python):        {t_py / t_cuda:.2f}x")
    print(f"  Speedup (cuda+fm vs python):     {t_py / t_cuda_fm:.2f}x")

    # ---- Verify Euler correctness ----
    print("\n--- Correctness Check (Euler) ---")
    verify_correctness(model_py, model_cuda, "python", "cuda")
    verify_correctness(model_py, model_cuda_fm, "python", "cuda+fm")

    # ---- RK4 Solvers ----
    if args.rk4:
        print("\n--- Runge-Kutta 4th Order ---")

        solver_rk_py = RungeKuttaSolver(dt=args.dt, n_iters=args.n_iters)
        solver_rk_cuda = RungeKuttaSolver(dt=args.dt, n_iters=args.n_iters,
                                          backend="cuda")

        model_rk_py = create_model(model_class, args.batch_size, device)
        t_rk_py, model_rk_py = benchmark_solver(
            solver_rk_py, model_rk_py, args.n_iters, args.dt,
            label="RungeKuttaSolver (python)")

        model_rk_cuda = create_model(model_class, args.batch_size, device)
        t_rk_cuda, model_rk_cuda = benchmark_solver(
            solver_rk_cuda, model_rk_cuda, args.n_iters, args.dt,
            label="RungeKuttaSolver (backend='cuda')")

        print(f"\n  Speedup (cuda RK4 vs python RK4): {t_rk_py / t_rk_cuda:.2f}x")

        print("\n--- Correctness Check (RK4) ---")
        verify_correctness(model_rk_py, model_rk_cuda, "python RK4", "cuda RK4")

    print("\n" + "=" * 72)
    print(" Benchmark complete.")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPF CUDA Kernel Benchmark")
    parser.add_argument("--n_iters", type=int, default=10000,
                        help="Number of time steps (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step (default: 0.01)")
    parser.add_argument("--model", type=str, default="Liaw",
                        choices=["Liaw", "GrayScott"],
                        help="Model type (default: Liaw)")
    parser.add_argument("--rk4", action="store_true",
                        help="Also benchmark RK4 solvers")
    args = parser.parse_args()
    run_benchmark(args)
