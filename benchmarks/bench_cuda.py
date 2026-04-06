#!/usr/bin/env python3
"""
Benchmark: Python (CuPy element-wise) vs Fused CUDA Kernel solvers.

Compares performance of:
  1. EulerSolver with fused CUDA kernel (auto-dispatched for CUDA models)
  2. EulerSolver forced to Python element-wise path (CuPy ops, no fusion)
  3. EulerSolver with fast_math enabled
  4. Optionally: RungeKuttaSolver (same comparison)

Also verifies numerical correctness by comparing final states.

Usage:
    python benchmarks/bench_cuda.py [--n_iters N] [--batch_size B] [--rk4]
"""

import argparse
import time

import numpy as np


def create_model(model_class, batch_size, device, dtype=np.float32):
    """Create a model with random parameters and initialization."""
    from lpf.initializers import LiawInitializer

    height, width = 128, 128
    n_init_pts = 25

    np.random.seed(42)
    init_pts = np.random.randint(0, height, size=(batch_size, n_init_pts, 2))
    init_pts = init_pts.astype(np.uint32)
    init_states = np.random.uniform(0.1, 1.0, size=(batch_size, 2)).astype(dtype)

    initializer = LiawInitializer(
        init_pts=init_pts,
        init_states=init_states,
    )

    name = model_class.__name__
    param_cols = {
        "LiawModel": 8,            # Du, Dv, ru, rv, k, su, sv, mu
        "GrayScottModel": 4,       # Du, Dv, F, k
        "SchnakenbergModel": 6,
        "GiererMeinhardtModel": 6,
    }
    if name not in param_cols:
        raise ValueError(f"Unknown model class: {name}")

    log_params = np.random.uniform(-3, 0, size=(batch_size, param_cols[name])).astype(dtype)
    params = 10.0 ** log_params

    return model_class(
        initializer=initializer,
        n_init_pts=n_init_pts,
        params=params,
        width=width,
        height=height,
        dx=0.1,
        device=device,
        dtype=dtype,
    )


# -- Force-Python solvers (bypass fused CUDA kernels) ---------------------

class PythonEulerSolver:
    """EulerSolver that always takes the Python element-wise path."""
    _cls = None

    @classmethod
    def _get_cls(cls):
        if cls._cls is None:
            from lpf.solvers import EulerSolver

            class _Impl(EulerSolver):
                @staticmethod
                def _is_cuda(model):
                    return False
            cls._cls = _Impl
        return cls._cls

    def __new__(cls, **kwargs):
        return cls._get_cls()(**kwargs)


class PythonRungeKuttaSolver:
    """RungeKuttaSolver that always takes the Python element-wise path."""
    _cls = None

    @classmethod
    def _get_cls(cls):
        if cls._cls is None:
            from lpf.solvers import RungeKuttaSolver

            class _Impl(RungeKuttaSolver):
                @staticmethod
                def _is_cuda(model):
                    return False
            cls._cls = _Impl
        return cls._cls

    def __new__(cls, **kwargs):
        return cls._get_cls()(**kwargs)


# -- Benchmark helpers ----------------------------------------------------

def benchmark_solver(solver, model, n_iters, dt=0.01, warmup_iters=100, label=""):
    """Run solver and measure elapsed time."""
    import cupy as cp

    # Warmup
    model.initialize()
    warmup_solver = solver.__class__(dt=dt, n_iters=warmup_iters)
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

    print(f"  {label:45s}  {elapsed:8.3f}s  ({iters_per_sec:,.0f} iter/s)")
    return elapsed, model


def verify_correctness(model_a, model_b, label_a, label_b, atol=1e-3, rtol=1e-3):
    """Compare final states of two models for numerical agreement."""
    import cupy as cp

    y_a = cp.asnumpy(model_a.y_mesh)
    y_b = cp.asnumpy(model_b.y_mesh)

    nan_a = np.isnan(y_a)
    nan_b = np.isnan(y_b)
    nan_match = np.array_equal(nan_a, nan_b)

    valid = ~(nan_a | nan_b)
    if valid.any():
        max_diff = np.max(np.abs(y_a[valid] - y_b[valid]))
        rel_diff = max_diff / (np.max(np.abs(y_a[valid])) + 1e-30)
        values_match = np.allclose(y_a[valid], y_b[valid], atol=atol, rtol=rtol)
    else:
        max_diff = rel_diff = 0.0
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

    model_classes = {"Liaw": LiawModel, "GrayScott": GrayScottModel}
    model_class = model_classes[args.model]
    device = "cuda:0"

    print("=" * 72)
    print(f" LPF CUDA Kernel Benchmark")
    print(f" Model: {args.model}, Batch: {args.batch_size}, "
          f"Grid: 128x128, Iters: {args.n_iters:,}")
    print("=" * 72)

    # ---- Euler ----
    print("\n--- Euler Method ---")

    model_fused = create_model(model_class, args.batch_size, device)
    t_fused, model_fused = benchmark_solver(
        EulerSolver(dt=args.dt, n_iters=args.n_iters),
        model_fused, args.n_iters, args.dt,
        label="Fused CUDA kernel")

    model_py = create_model(model_class, args.batch_size, device)
    t_py, model_py = benchmark_solver(
        PythonEulerSolver(dt=args.dt, n_iters=args.n_iters),
        model_py, args.n_iters, args.dt,
        label="Python element-wise (CuPy on GPU)")

    model_fm = create_model(model_class, args.batch_size, device)
    t_fm, model_fm = benchmark_solver(
        EulerSolver(dt=args.dt, n_iters=args.n_iters, fast_math=True),
        model_fm, args.n_iters, args.dt,
        label="Fused CUDA kernel (fast_math)")

    print(f"\n  Speedup (fused vs python):      {t_py / t_fused:.2f}x")
    print(f"  Speedup (fast_math vs python):   {t_py / t_fm:.2f}x")

    print("\n--- Correctness Check (Euler) ---")
    verify_correctness(model_fused, model_py, "fused", "python")
    verify_correctness(model_fused, model_fm, "fused", "fast_math")

    # ---- RK4 (optional) ----
    if args.rk4:
        print("\n--- Runge-Kutta 4th Order ---")

        model_rk_fused = create_model(model_class, args.batch_size, device)
        t_rk_fused, model_rk_fused = benchmark_solver(
            RungeKuttaSolver(dt=args.dt, n_iters=args.n_iters),
            model_rk_fused, args.n_iters, args.dt,
            label="Fused CUDA kernel")

        model_rk_py = create_model(model_class, args.batch_size, device)
        t_rk_py, model_rk_py = benchmark_solver(
            PythonRungeKuttaSolver(dt=args.dt, n_iters=args.n_iters),
            model_rk_py, args.n_iters, args.dt,
            label="Python element-wise (CuPy on GPU)")

        print(f"\n  Speedup (fused RK4 vs python RK4): {t_rk_py / t_rk_fused:.2f}x")

        print("\n--- Correctness Check (RK4) ---")
        verify_correctness(model_rk_fused, model_rk_py, "fused RK4", "python RK4")

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
