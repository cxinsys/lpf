# LPF benchmarks

Three focused benchmark scripts and one results document.  Each script
answers exactly one question; pick the one that matches what you want
to know.

| Script | Question it answers |
|---|---|
| [`bench_backends.py`](bench_backends.py) | How do all of LPF's array backends (NumPy / JAX (CUDA) / PyTorch CPU+CUDA / CuPy CUDA) compare on the same solver loop, across grid sizes, batch sizes, and solver families? |
| [`bench_kernel_vs_python.py`](bench_kernel_vs_python.py) | How much does LPF's hand-written **fused CUDA kernel** beat a generic Python loop driving CuPy element-wise ops on the *same* GPU memory? |
| [`bench_jax_optimizations.py`](bench_jax_optimizations.py) | What does each layer of JAX optimization buy on **GPU**? Compares eager → `@jit` → `@jit + lax.scan` against NumPy and LPF's fused CUDA paths. |

The most up-to-date numbers and the full discussion of when each
backend wins live in [`benchmark_results_backends.md`](benchmark_results_backends.md).

## Conventions

- All scripts measure **best-of-N** wall time after a warm-up pass that
  primes JIT/AOT caches.
- All scripts auto-detect installed backends and skip ones that are not
  available.
- **JAX (CPU) is intentionally excluded everywhere** — it has no reason
  to exist for this workload (per-op dispatch dominates and it runs
  ~10x slower than NumPy). LPF still ships a JAX solver for the GPU
  path; if you really want to test it on CPU, instantiate a model with
  `device="jax:cpu"` directly.

## Quick start

```bash
# Cross-backend comparison (the main benchmark)
conda run -n lpf python benchmarks/bench_backends.py \
    --n_iters 1000 --batch_size 4 --grid_size 128 --repeats 3 \
    --grid_sizes 64 128 --batch_sizes 1 4 16

# How much does kernel fusion matter?
conda run -n lpf python benchmarks/bench_kernel_vs_python.py

# What does each JAX optimization layer buy?
conda run -n lpf python benchmarks/bench_jax_optimizations.py
```
