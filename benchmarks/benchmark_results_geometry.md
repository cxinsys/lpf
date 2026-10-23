# Geometry benchmark: fused CUDA kernels on grids, polygons and meshes

Per-step time (µs) of the fixed-step CUDA solvers before and after the
kernel rewrite of LPF 0.3.0.dev0, measured with
[`bench_geometry.py`](bench_geometry.py). "Before" is the first 0.3.0.dev0
geometry commit (one dy/dt kernel per Runge-Kutta stage plus separate
elementwise update kernels); "after" is the current code.

## Environment

| Component | Detail |
|-----------|--------|
| OS | Linux 7.0 (Ubuntu) |
| CPU | AMD Ryzen 9 5900X (12 cores) |
| GPU | NVIDIA GeForce RTX 3090 Ti (sm_86), driver 595.91 |
| CUDA | 13.2 |
| Python | 3.13.17 |
| NumPy / CuPy | 2.5.4 / 14.2.0 |

Model: `LiawModel`, float32, dt = 0.01. Fixed `solve()` overhead removed
(difference of an n-step and an n/10-step solve, best of 3).

## What changed

All CUDA paths (the native loop in `libsolver`, the AOT fatbin and NVRTC
JIT) now compile one source, `lpf/kernels/csrc/include/rd_kernels.cuh`:

- **Fused stages.** A kernel evaluates dy/dt and applies the solver's
  update in the same pass (Euler step, Runge-Kutta stage, Heun/RK23/RK4
  final combination, AB2). An RK4 step is 4 launches instead of 9 and moves
  ~30% less memory. Every expression is evaluated in the same order as the
  unfused sequence, so results are bit-identical.
- **Domain cell flags.** One byte per cell (inactive / all weights 1 / cut
  cell); full interior cells no longer read the five stencil arrays.
- **Several models per thread on meshes.** With a batch of B > 1, a thread
  computes four models of its vertex and reads the ELL operator once.
- **CUDA graphs.** The native loop replays graphs of 64 steps, which helps
  small, launch-bound problems.

## Native solve loop

| case | Euler | Heun | RK4 | RK23 | AB2 |
|---|---|---|---|---|---|
| grid 128² × 16 | 6.7 → 5.9 (1.14×) | 28.8 → 15.8 (1.83×) | 57.9 → 41.7 (1.39×) | 43.6 → 28.8 (1.52×) | 19.3 → 10.8 (1.79×) |
| grid 512² × 16 | 83.4 → 82.8 (1.01×) | 496.1 → 266.1 (1.86×) | 945.2 → 635.4 (1.49×) | 719.3 → 452.5 (1.59×) | 308.3 → 157.1 (1.96×) |
| 16 polygons 128² (one per model) | 7.5 → 6.0 (1.25×) | 34.7 → 19.0 (1.83×) | 68.9 → 48.0 (1.44×) | 51.6 → 33.5 (1.54×) | 21.0 → 11.8 (1.78×) |
| polygon 512² × 16 | 116.1 → 82.7 (1.40×) | 567.1 → 278.2 (2.04×) | 1089.2 → 669.0 (1.63×) | 828.8 → 472.7 (1.75×) | 344.0 → 160.9 (2.14×) |
| mesh 40,962 vertices × 4 | 7.2 → 5.2 (1.39×) | 25.7 → 14.1 (1.83×) | 49.7 → 36.0 (1.38×) | 38.2 → 25.0 (1.53×) | 16.4 → 9.1 (1.81×) |
| mesh 40,962 vertices × 16 | 23.5 → 19.6 (1.20×) | 101.5 → 57.0 (1.78×) | 195.9 → 134.7 (1.45×) | 148.7 → 94.0 (1.58×) | 60.4 → 31.2 (1.93×) |
| mesh 163,842 vertices × 4 | 50.2 → 23.6 (2.13×) | 155.2 → 69.1 (2.25×) | 295.6 → 161.2 (1.83×) | 225.8 → 111.8 (2.02×) | 86.7 → 40.2 (2.16×) |

## Python loop (CuPy launches, native library disabled)

| case | Euler | RK4 |
|---|---|---|
| grid 128² × 16 | 6.9 → 6.4 (1.08×) | 59.3 → 47.8 (1.24×) |
| grid 512² × 16 | 86.4 → 82.1 (1.05×) | 942.7 → 644.3 (1.46×) |
| 16 polygons 128² (one per model) | 8.6 → 8.0 (1.08×) | 68.5 → 51.8 (1.32×) |
| polygon 512² × 16 | 125.3 → 87.5 (1.43×) | 1087.2 → 672.2 (1.62×) |
| mesh 40,962 vertices × 4 | 7.5 → 5.6 (1.34×) | 50.0 → 43.3 (1.16×) |
| mesh 40,962 vertices × 16 | 25.2 → 20.9 (1.20×) | 194.3 → 141.9 (1.37×) |
| mesh 163,842 vertices × 4 | 48.8 → 28.1 (1.74×) | 294.1 → 165.8 (1.77×) |


## Application: the Blender lizard

Lizard mesh (32,049 vertices, 64,094 triangles, RCM order), 4 models:

| solver | steps | before | after |
|---|---|---|---|
| Euler | 500,000 | 3.20 s | 2.61 s (1.23×) |
| RK4 | 100,000 | 4.47 s | 3.31 s (1.35×) |

Euler on a large grid (512² × 16) was already at over 80% of the memory
bandwidth and is unchanged; the gains grow with the number of stages and
for domains and meshes.
