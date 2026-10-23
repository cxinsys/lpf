#!/usr/bin/env python3
"""
Benchmark: the fused CUDA kernels on every geometry.

Per-step time of the fixed-step CUDA solvers on the full grid, on polygon
domains (lpf.domains) and on triangle-mesh surfaces (lpf.surfaces), for the
native solve loop (libsolver) and the Python loop (CuPy launches).

The fixed cost of solve() (initialization, buffers, the final copy) is
removed by timing an n-step and an n/10-step solve and dividing the
difference by 0.9 n.

Usage:
    python benchmarks/bench_geometry.py [--paths native python]
                                        [--cases grid128x16 mesh6x4 ...]
"""

import argparse
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np

warnings.filterwarnings("ignore")

LIAW = {"u0": 2.0, "v0": 1.0, "Du": 0.0005, "Dv": 0.075, "ru": 0.18,
        "rv": 0.0287, "k": 0.2, "su": 0.001, "sv": 0.025, "mu": 0.08}

# name: (geometry, size, batch size). size is the grid side, or the
# subdivision level of an icosphere (level 6: 40,962 vertices).
CASES = {
    "grid128x16": ("grid", 128, 16),
    "grid512x16": ("grid", 512, 16),
    "domains128x16": ("domains", 128, 16),
    "domain512x16": ("domain", 512, 16),
    "mesh6x4": ("mesh", 6, 4),
    "mesh6x16": ("mesh", 6, 16),
    "mesh7x4": ("mesh", 7, 4),
}

FIXED_STEP = ["EulerSolver", "HeunSolver", "RungeKuttaSolver", "RK23Solver",
              "AdamsBashforth2Solver"]


def icosphere(level):
    """Unit sphere: an icosahedron subdivided level times."""
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
         (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
         (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    v = [np.array(p) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        mids, faces = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in mids:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                mids[key] = len(v) - 1
            return mids[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = faces
    return np.array(v), np.array(f)


def star(n, r):
    a = np.arange(10) * np.pi / 5
    s = np.where(np.arange(10) % 2 == 0, 1.0, 0.45)
    return np.stack([n / 2 + r * s * np.sin(a), n / 2 - r * s * np.cos(a)], 1)


def ellipse(n, r):
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    return np.stack([n / 2 + r * np.cos(t), n / 2 + 0.8 * r * np.sin(t)], 1)


def build(case):
    from lpf.domains import PolygonDomain
    from lpf.initializers import LiawInitializer
    from lpf.models import LiawModel
    from lpf.surfaces import SurfaceMesh

    kind, n, batch = case
    geometry = {}
    if kind == "domain":
        geometry = {"domain": PolygonDomain(star(n, 0.45 * n))}
    elif kind == "domains":
        geometry = {"domain": [PolygonDomain(star(n, 0.45 * n)) if i % 2 == 0
                               else PolygonDomain(ellipse(n, 0.4 * n))
                               for i in range(batch)]}
    elif kind == "mesh":
        v, f = icosphere(n)
        # Scaled so that the edges are about 0.1 long, like dx on the grid.
        geometry = {"mesh": SurfaceMesh(v, f, scale=5.3 * 2 ** (n - 6),
                                        reorder=True)}
    rng = np.random.default_rng(0)
    dicts = []
    for _ in range(batch):
        n2v = dict(LIAW)
        if kind == "mesh":
            pts = rng.choice(geometry["mesh"].n_vertices, 30, replace=False)
            for j, p in enumerate(pts):
                n2v["init_pts_%d" % j] = [int(p)]
        else:
            for j, p in enumerate(rng.integers(n // 4, 3 * n // 4, size=(30, 2))):
                n2v["init_pts_%d" % j] = [int(p[0]), int(p[1])]
        dicts.append(n2v)
    initializer = LiawInitializer()
    initializer.update(dicts)
    return LiawModel(initializer=initializer, params=LiawModel.parse_params(dicts),
                     width=n, height=n, dx=0.1, device="cuda:0", **geometry)


def per_step_us(model, solver, n_iters, repeats):
    import cupy as cp
    solver.solve(model=model, dt=0.01, n_iters=200)  # warm-up
    cp.cuda.Device().synchronize()

    def best(n):
        t = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            solver.solve(model=model, dt=0.01, n_iters=n)
            cp.cuda.Device().synchronize()
            t = min(t, time.perf_counter() - t0)
        return t

    n0 = n_iters // 10
    return 1e6 * (best(n_iters) - best(n0)) / (n_iters - n0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paths", nargs="+", default=["native", "python"],
                        choices=["native", "python"])
    parser.add_argument("--cases", nargs="+", default=list(CASES),
                        choices=list(CASES))
    parser.add_argument("--solvers", nargs="+", default=FIXED_STEP,
                        choices=FIXED_STEP)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError
    except Exception:
        sys.exit("bench_geometry.py needs CuPy and a CUDA device.")

    import lpf.solvers as S
    import lpf.solvers._cuda.base as cuda_base

    print("%-8s %-14s %-22s %10s" % ("path", "case", "solver", "us/step"))
    for path in args.paths:
        if path == "python":  # force the CuPy launch loop
            cuda_base._native_solver = SimpleNamespace(is_available=lambda: False)
        for name in args.cases:
            big = "512" in name or "mesh7" in name
            n_iters = {"native": 2000 if big else 20000,
                       "python": 1000 if big else 5000}[path]
            for solver_name in args.solvers:
                us = per_step_us(build(CASES[name]), getattr(S, solver_name)(),
                                 n_iters, args.repeats)
                print("%-8s %-14s %-22s %10.2f" % (path, name, solver_name, us),
                      flush=True)


if __name__ == "__main__":
    main()
