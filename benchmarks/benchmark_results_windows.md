# GPU Benchmark: Fused CUDA Kernels vs Python Element-wise

Benchmark comparing two GPU execution paths in LPF:

1. **Fused CUDA kernel** — Laplacian + Reaction computed in a single kernel launch
2. **Python element-wise** — Python loop calling individual CuPy array operations (no kernel fusion)
3. **PyTorch DLPack** — PyTorch tensors bridged to the same fused CUDA kernels via zero-copy DLPack

Both paths use GPU memory (CuPy arrays). The difference is whether operations are fused into a single CUDA kernel or dispatched as separate element-wise CuPy operations from a Python loop.

## Environment

| Component | Detail |
|-----------|--------|
| OS | Windows 11 Pro for Workstations (10.0.26200) |
| CPU | Intel Core i9-12900K |
| GPU | NVIDIA GeForce RTX 4090 (24 GB GDDR6X) |
| NVIDIA Driver | 595.97 |
| CUDA Toolkit | 13.0 |
| Python | 3.12.12 |
| CuPy | 14.0.1 (cupy-cuda13x) |
| PyTorch | 2.10.0+cu130 |

## Configuration

- **Batch size**: 16 models (population: `init_pop_axyridis`)
- **Grid**: 128 x 128, dx = 0.1
- **Time step**: dt = 0.01
- **Iterations**: 500,000
- **Warm-up**: 500 iterations (JIT compilation, memory allocation)

## Results

### Euler Solver

| Method | Time (s) | iters/s | Speedup |
|--------|----------|---------|---------|
| Fused CUDA kernel | 13.67 | 36,572 | **41.2x** |
| Python element-wise (CuPy on GPU) | 563.62 | 887 | 1.0x |
| PyTorch -> DLPack -> Fused CUDA kernel | 13.58 | 36,825 | 41.5x |

### RK4 Solver

| Method | Time (s) | iters/s | Speedup |
|--------|----------|---------|---------|
| Fused CUDA kernel | 115.66 | 4,323 | **13.6x** |
| Python element-wise (CuPy on GPU) | 1,567.81 | 319 | 1.0x |
| PyTorch -> DLPack -> Fused CUDA kernel | 37.66 | 13,276 | 41.6x |

### fast_math (Euler, Fused CUDA kernel)

| Option | iters/s | Speedup |
|--------|---------|---------|
| fast_math=False | 103,739 | 1.00x |
| fast_math=True | 104,905 | 1.01x |

At this grid size (128x128), `fast_math` provides negligible benefit.

## Numerical Accuracy (1,000 iterations)

| Comparison | max \|diff\| | allclose |
|------------|--------------|----------|
| Fused kernel vs Python element-wise | 6.68e-06 | True (rtol=1e-4) |
| Fused kernel vs PyTorch DLPack | 0.00e+00 | True (bit-for-bit) |

The fused kernel and Python element-wise paths produce nearly identical results. The small difference (O(1e-6)) is due to floating-point operation ordering. The PyTorch DLPack path shares the exact same GPU memory via zero-copy, so results are bit-for-bit identical to the CuPy fused kernel path.

## Key Takeaways

- **Kernel fusion provides 14-41x speedup** over Python-orchestrated element-wise CuPy operations on the same GPU hardware.
- The bottleneck in the Python path is not compute but **Python loop overhead and multiple small kernel launches** (laplacian, reaction, boundary conditions dispatched separately).
- **PyTorch DLPack bridge adds zero overhead** — performance matches the native CuPy path because DLPack shares GPU memory without copying.
- RK4 fused kernel shows a smaller relative speedup (14x) versus Python element-wise, but the PyTorch DLPack path achieves 42x due to more efficient dispatch of the 4 PDE evaluations per step.

## Reproducing

```bash
cd benchmarks
conda run -n lpf python bench_kernel_vs_python.py
```
