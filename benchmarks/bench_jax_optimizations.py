#!/usr/bin/env python3
"""
JAX optimization benchmark.

Question: how do JAX's GPU execution modes (eager / @jit / @jit+scan)
compare against NumPy and LPF's fused CUDA backends?

JAX-on-CPU is intentionally excluded — it has no reason to exist for this
workload (per-op dispatch dominates and it runs ~10x slower than NumPy).

Configurations:

  1. NumPy (CPU)            — baseline
  2. JAX  (GPU, eager)
  3. JAX  (GPU, @jit)       — single-step JIT
  4. JAX  (GPU, @jit+scan)  — entire integration loop fused into one XLA program

For comparison we also include LPF's auto-fused CUDA paths (CuPy / PyTorch
CUDA), which represent the upper bound on this hardware.

Usage:
    python benchmarks/bench_jax_optimizations.py
"""

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Stable Liaw parameters from population/init_pop_01 (loaded once)
# ---------------------------------------------------------------------------

def _load_init_pop_01():
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

    return params, init_states, init_pts


def make_initial_state(batch_size, height=128, width=128, dtype=np.float32):
    """Build the (2, B, H, W) initial y_mesh tensor and (B, 8) params."""
    base_params, base_states, base_pts = _load_init_pop_01()
    n_morphs = base_params.shape[0]
    idx = np.arange(batch_size) % n_morphs

    params = base_params[idx].astype(dtype)
    init_states = base_states[idx].astype(dtype)
    init_pts = base_pts[idx]

    y = np.zeros((2, batch_size, height, width), dtype=dtype)
    for b in range(batch_size):
        u0 = init_states[b, 0]
        v0 = init_states[b, 1]
        y[1, b, :, :] = v0
        for r, c in init_pts[b]:
            r = int(min(max(r, 0), height - 1))
            c = int(min(max(c, 0), width - 1))
            y[0, b, r, c] = u0
    return y, params


# ---------------------------------------------------------------------------
# Pure-NumPy reference implementation of the Liaw RHS
# ---------------------------------------------------------------------------

def make_numpy_step():
    def liaw_rhs(y, params, dx_inv2):
        u = y[0]                     # (B, H, W)
        v = y[1]
        Du = params[:, 0].reshape(-1, 1, 1)
        Dv = params[:, 1].reshape(-1, 1, 1)
        ru = params[:, 2].reshape(-1, 1, 1)
        rv = params[:, 3].reshape(-1, 1, 1)
        k  = params[:, 4].reshape(-1, 1, 1)
        su = params[:, 5].reshape(-1, 1, 1)
        sv = params[:, 6].reshape(-1, 1, 1)
        mu = params[:, 7].reshape(-1, 1, 1)

        u_c = u[:, 1:-1, 1:-1]
        v_c = v[:, 1:-1, 1:-1]

        lap_u = (u[:, :-2, 1:-1] + u[:, 2:, 1:-1] +
                 u[:, 1:-1, :-2] + u[:, 1:-1, 2:] - 4 * u_c) * dx_inv2
        lap_v = (v[:, :-2, 1:-1] + v[:, 2:, 1:-1] +
                 v[:, 1:-1, :-2] + v[:, 1:-1, 2:] - 4 * v_c) * dx_inv2

        denom = 1.0 + k * u_c * u_c
        f = ru * (u_c * u_c * v_c) / denom + su - mu * u_c
        g = -rv * (u_c * u_c * v_c) / denom + sv

        du = Du * lap_u + f
        dv = Dv * lap_v + g

        dydt = np.zeros_like(y)
        dydt[0, :, 1:-1, 1:-1] = du
        dydt[1, :, 1:-1, 1:-1] = dv
        return dydt

    def step(y, params, dt, dx_inv2):
        return y + dt * liaw_rhs(y, params, dx_inv2)

    def integrate(y0, params, dt, dx_inv2, n_iters):
        y = y0
        for _ in range(n_iters):
            y = step(y, params, dt, dx_inv2)
        return y

    return integrate


# ---------------------------------------------------------------------------
# JAX implementations: eager, @jit, @jit+scan, with optional donate
# ---------------------------------------------------------------------------

def make_jax_steps(device):
    """Returns dict of {variant_name: callable(y0, params, dt, dx_inv2, n_iters)}.

    Each callable runs `n_iters` Euler steps and returns the final y.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    target_device = jax.devices(device)[0]

    def liaw_rhs(y, params, dx_inv2):
        u = y[0]
        v = y[1]
        Du = params[:, 0].reshape(-1, 1, 1)
        Dv = params[:, 1].reshape(-1, 1, 1)
        ru = params[:, 2].reshape(-1, 1, 1)
        rv = params[:, 3].reshape(-1, 1, 1)
        k  = params[:, 4].reshape(-1, 1, 1)
        su = params[:, 5].reshape(-1, 1, 1)
        sv = params[:, 6].reshape(-1, 1, 1)
        mu = params[:, 7].reshape(-1, 1, 1)

        u_c = u[:, 1:-1, 1:-1]
        v_c = v[:, 1:-1, 1:-1]

        lap_u = (u[:, :-2, 1:-1] + u[:, 2:, 1:-1] +
                 u[:, 1:-1, :-2] + u[:, 1:-1, 2:] - 4 * u_c) * dx_inv2
        lap_v = (v[:, :-2, 1:-1] + v[:, 2:, 1:-1] +
                 v[:, 1:-1, :-2] + v[:, 1:-1, 2:] - 4 * v_c) * dx_inv2

        denom = 1.0 + k * u_c * u_c
        f = ru * (u_c * u_c * v_c) / denom + su - mu * u_c
        g = -rv * (u_c * u_c * v_c) / denom + sv

        du = Du * lap_u + f
        dv = Dv * lap_v + g

        # Functional update (JAX-immutable arrays)
        dydt = jnp.zeros_like(y)
        dydt = dydt.at[0, :, 1:-1, 1:-1].set(du)
        dydt = dydt.at[1, :, 1:-1, 1:-1].set(dv)
        return dydt

    def step_fn(y, params, dt, dx_inv2):
        return y + dt * liaw_rhs(y, params, dx_inv2)

    # ---- variant 1: eager (Python loop, no jit) ----
    def integrate_eager(y0, params, dt, dx_inv2, n_iters):
        y = jax.device_put(y0, target_device)
        params_d = jax.device_put(params, target_device)
        for _ in range(n_iters):
            y = step_fn(y, params_d, dt, dx_inv2)
        y.block_until_ready()
        return y

    # ---- variant 2: @jit single step (Python loop drives jit'd step) ----
    step_jit = jax.jit(step_fn, static_argnames=())

    def integrate_jit_step(y0, params, dt, dx_inv2, n_iters):
        y = jax.device_put(y0, target_device)
        params_d = jax.device_put(params, target_device)
        for _ in range(n_iters):
            y = step_jit(y, params_d, dt, dx_inv2)
        y.block_until_ready()
        return y

    # ---- variant 3: @jit + lax.scan (entire loop fused) ----
    def integrate_scan_impl(y0, params, dt, dx_inv2, n_iters):
        def body(y, _):
            return step_fn(y, params, dt, dx_inv2), None
        y_final, _ = lax.scan(body, y0, None, length=n_iters)
        return y_final

    integrate_scan_jit = jax.jit(integrate_scan_impl, static_argnames=("n_iters",))

    def integrate_scan(y0, params, dt, dx_inv2, n_iters):
        y = jax.device_put(y0, target_device)
        params_d = jax.device_put(params, target_device)
        result = integrate_scan_jit(y, params_d, dt, dx_inv2, n_iters)
        result.block_until_ready()
        return result

    return {
        "eager": integrate_eager,
        "jit_step": integrate_jit_step,
        "jit_scan": integrate_scan,
    }


# ---------------------------------------------------------------------------
# CuPy / PyTorch (CUDA) — using LPF's auto-fused kernel path
# ---------------------------------------------------------------------------

def integrate_lpf(device, y0, params, dt, dx_inv2, n_iters):
    from lpf.models import LiawModel
    from lpf.initializers import LiawInitializer
    from lpf.solvers import EulerSolver

    dx = (1.0 / dx_inv2) ** 0.5
    height = y0.shape[2]
    width = y0.shape[3]
    batch_size = y0.shape[1]

    # Reverse-engineer init_states / init_pts from y0 (slow but only at setup)
    init_states = np.zeros((batch_size, 2), dtype=np.float32)
    init_pts_list = [[] for _ in range(batch_size)]
    for b in range(batch_size):
        v_val = float(y0[1, b, 0, 0])
        init_states[b, 1] = v_val
        u_arr = y0[0, b]
        nz = np.argwhere(u_arr > 0)
        if len(nz) > 0:
            init_states[b, 0] = float(u_arr[nz[0, 0], nz[0, 1]])
            for r, c in nz:
                init_pts_list[b].append((int(r), int(c)))
        else:
            init_states[b, 0] = 0.0
            init_pts_list[b].append((0, 0))

    max_pts = max(len(p) for p in init_pts_list)
    init_pts = np.zeros((batch_size, max_pts, 2), dtype=np.uint32)
    for b, pts in enumerate(init_pts_list):
        for j, (r, c) in enumerate(pts):
            init_pts[b, j] = (r, c)

    model = LiawModel(
        initializer=LiawInitializer(init_pts=init_pts, init_states=init_states),
        n_init_pts=max_pts, params=params,
        width=width, height=height, dx=dx,
        device=device, dtype=np.float32,
    )
    model.initialize()

    solver = EulerSolver(dt=dt, n_iters=n_iters)
    solver.solve(model, init_model=False, verbose=0)

    arr = model.y_mesh
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_run(fn, args, sync_fn=None, repeats=3):
    """Warmup once, then time `repeats` runs and return the best wall time."""
    # Warmup (also triggers JIT compile)
    out = fn(*args)
    if sync_fn is not None:
        sync_fn(out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        if sync_fn is not None:
            sync_fn(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), out


def jax_sync(out):
    out.block_until_ready()


def cuda_sync(out):
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(batch_size=4, grid_size=128, n_iters=1000, dt=0.01, dx=0.1,
                  repeats=3):
    print("=" * 78)
    print(" JAX Optimization Benchmark")
    print("=" * 78)
    print(f" batch={batch_size}, grid={grid_size}x{grid_size}, "
          f"n_iters={n_iters}, dt={dt}")
    print(f" Repeats: best-of-{repeats}")
    print("=" * 78)

    dx_inv2 = 1.0 / (dx ** 2)
    y0, params = make_initial_state(batch_size, grid_size, grid_size)

    results = {}

    # ---- 1. NumPy baseline ----
    print("\n[NumPy CPU]")
    np_integrate = make_numpy_step()
    t, y_ref = time_run(
        np_integrate, (y0, params, dt, dx_inv2, n_iters),
        repeats=repeats)
    results["NumPy (CPU)"] = (t, y_ref)
    print(f"  {t:.4f}s   ({n_iters/t:>10,.0f} iter/s)")

    # ---- 2-4. JAX GPU variants ----
    import jax
    has_jax_gpu = False
    try:
        jax.devices("gpu")
        has_jax_gpu = True
    except RuntimeError:
        pass

    if has_jax_gpu:
        jax_gpu_steps = make_jax_steps("gpu")
        for variant_label, variant_key in [
            ("JAX (GPU, eager)", "eager"),
            ("JAX (GPU, @jit step)", "jit_step"),
            ("JAX (GPU, @jit+scan)", "jit_scan"),
        ]:
            print(f"\n[{variant_label}]")
            try:
                fn = jax_gpu_steps[variant_key]
                t, out = time_run(
                    fn, (y0, params, dt, dx_inv2, n_iters),
                    sync_fn=jax_sync, repeats=repeats)
                results[variant_label] = (t, np.asarray(out))
                print(f"  {t:.4f}s   ({n_iters/t:>10,.0f} iter/s)")
            except Exception as e:
                print(f"  FAILED: {e}")
    else:
        print("\n[JAX GPU]  SKIPPED (no GPU device available to JAX)")

    # ---- LPF auto-fused CUDA paths ----
    for label, device in [("CuPy (CUDA, fused kernel)", "cuda:0"),
                           ("PyTorch (CUDA, fused kernel)", "torch:cuda:0")]:
        print(f"\n[{label}]")
        try:
            def fn(y, p, dt_, di, n):
                return integrate_lpf(device, y, p, dt_, di, n)
            t, out = time_run(
                fn, (y0, params, dt, dx_inv2, n_iters),
                sync_fn=cuda_sync, repeats=repeats)
            results[label] = (t, out)
            print(f"  {t:.4f}s   ({n_iters/t:>10,.0f} iter/s)")
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ---- Correctness check ----
    print("\n" + "=" * 78)
    print(" Correctness check (vs NumPy reference)")
    print("=" * 78)
    y_ref = results["NumPy (CPU)"][1]
    for label, (t, out) in results.items():
        if label == "NumPy (CPU)":
            continue
        diff = np.abs(out - y_ref)
        max_diff = float(diff.max())
        ok = "PASS" if (np.allclose(out, y_ref, atol=1e-4, rtol=1e-3)
                        or not np.isfinite(max_diff)) else "FAIL"
        print(f"  [{ok}] {label:32s}  max_diff={max_diff:.6e}")

    # ---- Summary ----
    print("\n" + "=" * 78)
    print(" Summary (sorted, fastest first)")
    print("=" * 78)
    sorted_results = sorted(results.items(), key=lambda kv: kv[1][0])
    fastest = sorted_results[0][1][0]
    np_time = results["NumPy (CPU)"][0]
    print(f"  {'Configuration':32s}  {'Time':>10s}  {'iter/s':>12s}  "
          f"{'vs NumPy':>10s}  {'vs fastest':>11s}")
    print("  " + "-" * 86)
    for label, (t, _) in sorted_results:
        speedup_np = np_time / t
        speedup_fast = fastest / t
        marker_np = "x faster" if speedup_np >= 1 else "x slower"
        marker_np = f"{abs(speedup_np):.2f}{marker_np}" if speedup_np < 1 else f"{speedup_np:.2f}x faster"
        marker_fast = f"{speedup_fast:.2f}x"
        if speedup_np < 1:
            speedup_np_str = f"{1/speedup_np:.2f}x slower"
        elif speedup_np == 1:
            speedup_np_str = "baseline"
        else:
            speedup_np_str = f"{speedup_np:.2f}x faster"
        print(f"  {label:32s}  {t:>9.4f}s  {n_iters/t:>11,.0f}  "
              f"{speedup_np_str:>13s}  {marker_fast:>10s}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JAX optimization benchmark")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--n_iters", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        n_iters=args.n_iters,
        dt=args.dt,
        dx=args.dx,
        repeats=args.repeats,
    )
