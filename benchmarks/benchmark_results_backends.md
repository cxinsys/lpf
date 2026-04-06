# Cross-Backend Benchmark: NumPy vs JAX (CUDA) vs PyTorch vs CuPy

Performance comparison of all GPU-capable array backends LPF supports,
running the same solver loops on identical data.

> **JAX (CPU) is intentionally excluded.** Running JAX on CPU for this
> workload is dominated by per-op dispatch and runs ~10x slower than
> NumPy. JAX exists in LPF for its **GPU/XLA path** — that is the only
> configuration benchmarked here.

## Environment

| Component | Detail |
|-----------|--------|
| OS | Linux 6.17 (Ubuntu) |
| CPU | AMD Ryzen 9 5900X (12 cores) |
| GPU | NVIDIA GeForce RTX 5090 |
| CUDA | 13.0 |
| Python | 3.13.12 |
| NumPy | 2.4.4 |
| JAX | 0.9.2 (`jax[cuda]`) |
| PyTorch | 2.11.0+cu130 |
| CuPy | 14.0.1 |

## Configuration

- **Model**: `LiawModel` with the 16 morphs from `population/init_pop_01`
  (cycled to fill the requested batch size).
- **Grid**: 128 x 128, dx = 0.1
- **Time step**: dt = 0.01
- **Repeats**: 3 (best-of-3 reported)
- **Warmup**: solver-loop runs are warmed up to keep JIT/native caches hot

LPF auto-dispatches each backend to its fastest path:

| Backend | Solve loop runs as |
|---|---|
| NumPy | Pure Python loop |
| PyTorch (CPU) | Pure Python loop |
| **CuPy (CUDA)** | LPF AOT-compiled fused CUDA kernel (`.so`) |
| **PyTorch (CUDA)** | Same `.so` as CuPy via DLPack zero-copy bridge |
| **JAX (CUDA)** | XLA `@jit` + `lax.fori_loop`/`scan` over the entire integration |

## pdefunc() micro-benchmark (500 calls, batch=4, grid=128x128)

| Backend | Time | per call | vs NumPy |
|---|---|---|---|
| NumPy (CPU) | 0.179 s | 0.357 ms | 1.00x |
| JAX (CUDA) | 5.395 s | 10.791 ms | **0.03x (30x slower)** |
| PyTorch (CPU) | 0.263 s | 0.526 ms | 0.68x |
| PyTorch (CUDA) | 0.173 s | 0.346 ms | 1.03x |
| CuPy (CUDA) | 0.225 s | 0.450 ms | 0.79x |

A bare `pdefunc()` call is the *worst case* for JAX because each call
re-traces the function and re-launches XLA. JAX's strength is the
**compiled solver loop** below — never the per-call dispatch.

## Solver loops (n_iters=1000, batch=4, grid=128x128)

| Solver | NumPy | JAX (CUDA) | PyTorch (CPU) | PyTorch (CUDA) | CuPy (CUDA) |
|---|---|---|---|---|---|
| Euler | 0.459 s | **0.0069 s** | 0.563 s | 0.0511 s | 0.0502 s |
| Heun | 0.879 s | **0.0089 s** | 1.076 s | 0.0591 s | 0.0578 s |
| RK4 | 1.777 s | **0.0119 s** | 2.254 s | 0.0623 s | 0.0637 s |
| RK23 | 1.301 s | **0.0087 s** | 1.573 s | 0.0597 s | 0.0573 s |
| AB2 | 0.495 s | **0.0073 s** | 0.595 s | 0.0513 s | 0.0514 s |

### Speedup vs NumPy (CPU baseline)

| Solver | JAX (CUDA) | PyTorch (CPU) | PyTorch (CUDA) | CuPy (CUDA) |
|---|---|---|---|---|
| Euler | **66.7x** | 0.81x | 8.97x | 9.14x |
| Heun | **99.0x** | 0.82x | 14.87x | 15.20x |
| RK4 | **149.5x** | 0.79x | 28.54x | 27.91x |
| RK23 | **149.9x** | 0.83x | 21.78x | 22.69x |
| AB2 | **68.1x** | 0.83x | 9.65x | 9.63x |

For `n_iters=1000` JAX (CUDA) is the fastest backend by a wide margin.
This is the regime where XLA's whole-loop fusion shines: a 1,000-step
integration becomes a single GPU kernel launch, while LPF's hand-fused
kernels still pay one launch per step (~50 us overhead × 1000 = 50 ms
floor).

## Grid scaling (Euler, batch=4, n_iters=1000)

| Grid | NumPy | JAX (CUDA) | PyTorch (CPU) | PyTorch (CUDA) | CuPy (CUDA) |
|---|---|---|---|---|---|
| 64x64 | 0.205 s | 0.0067 s | 0.282 s | 0.0486 s | 0.0470 s |
| 128x128 | 0.451 s | 0.0067 s | 0.554 s | 0.0484 s | 0.0471 s |

GPU paths are essentially flat over this range — they are launch-bound
on small grids. NumPy/PyTorch (CPU) scale linearly with grid area.

## Batch scaling (Euler, grid=128x128, n_iters=1000)

| Batch | NumPy | JAX (CUDA) | PyTorch (CPU) | PyTorch (CUDA) | CuPy (CUDA) |
|---|---|---|---|---|---|
| 1 | 0.177 s | 0.0068 s | 0.276 s | 0.0485 s | 0.0486 s |
| 4 | 0.448 s | 0.0068 s | 0.541 s | 0.0497 s | 0.0487 s |
| 16 | 1.414 s | 0.0067 s | 0.736 s | 0.0616 s | 0.0648 s |

JAX (CUDA) is essentially constant from batch=1 to batch=16 — XLA fuses
the batch dimension into the same compiled kernel.

## Numerical correctness (EulerSolver, n_iters=1000)

All non-NumPy backends compared against the NumPy baseline:

| Backend | max abs diff | result |
|---|---|---|
| JAX (CUDA) | 1.53e-05 | passes (rtol=1e-3, atol=1e-4) |
| PyTorch (CPU) | 0.0 | bit-identical |
| PyTorch (CUDA) | 1.53e-05 | passes (rtol=1e-3, atol=1e-4) |
| CuPy (CUDA) | 1.53e-05 | passes (rtol=1e-3, atol=1e-4) |

CPU PyTorch matches NumPy bit-for-bit. The CUDA paths differ by ~1.5e-05
due to fused-kernel operation reordering — well within float32 ULP for a
1,000-step trajectory.

## Long-run benchmark: tutorial03 settings (n_iters=500,000)

The 1,000-iteration table above measures the *launch-overhead-dominated*
regime. The realistic LPF workload is 100k–500k iterations, which
flips the picture:

**Configuration**: Liaw, 128×128, batch=16, n_iters=500,000, dt=0.01,
float32. File I/O disabled, warmup before measurement, best-of-3.

| Backend | Time | vs CuPy CUDA |
|---|---:|---:|
| **CuPy CUDA** (LPF native `.so`) | **1.66 s** | 1.00× (baseline) |
| **PyTorch CUDA** (LPF native `.so`) | 1.66 s | 1.00× |
| JAX (CUDA, jit + fori_loop) | 3.11 s | 0.53× (1.87x slower) |
| NumPy (CPU) | 707.65 s | 0.0023× |

### Why the cross-over?

For **short runs** (< ~10k iterations) JAX (CUDA) is unbeatable because
it amortizes ~50 us of per-step launch overhead across the entire
compiled loop — LPF native pays that once per step.

For **long runs** the per-step launch overhead becomes negligible
(50 us × 500k = 25 s, but actual time is 1.66 s, so launches account for
≤ 5%). At that point what matters is **steady-state arithmetic
throughput** of the inner kernel — and LPF's hand-fused CUDA kernel
[`lpf/kernels/aot/`](../lpf/kernels/aot/) is closer to the memory
bandwidth ceiling than what XLA generates from generic JAX primitives.
LPF's kernel uses domain knowledge (Liaw RHS shape, stencil layout)
that XLA does not have.

| Regime | Bottleneck | Winner |
|---|---|---|
| Short loops (≪ 10k iters) | Per-step kernel launch latency | **JAX (CUDA)** — single fused launch |
| Long loops (≫ 100k iters) | Steady-state memory bandwidth | **LPF native CUDA** (CuPy/PyTorch) |

## Practical recommendation

| Use case | Recommended backend |
|---|---|
| Interactive prototyping, short solves | **JAX (CUDA)** — fastest below ~10k iters |
| Production runs, 100k+ iterations | **CuPy (CUDA)** or **PyTorch (CUDA)** |
| CPU-only environment | **NumPy** (PyTorch CPU is comparable) |
| Mixed CPU/GPU portability without code changes | Any of the above — LPF auto-dispatches |

JAX is also valuable when you need:
- TPU support (`device="jax:tpu:0"`)
- `jax.grad` for differentiable simulations
- Whole-program XLA optimization across LPF + downstream code

## Reproducing

```bash
# Short-run table (n_iters=1000)
conda run -n lpf python benchmarks/bench_backends.py \
    --n_iters 1000 --batch_size 4 --grid_size 128 --repeats 3 \
    --grid_sizes 64 128 --batch_sizes 1 4 16

# Long-run tutorial03 comparison (n_iters=500000)
# See bench_jax_optimizations.py for the JAX optimization breakdown.
```

The benchmark auto-detects which backends are installed and skips any
that are unavailable. JAX (CPU) is no longer registered as a target —
it is intentionally excluded.
