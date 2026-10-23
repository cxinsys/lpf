/**
 * lpf AOT CUDA kernels — the fused reaction-diffusion kernels of
 * include/rd_kernels.cuh for every model and dtype (the NVRTC JIT path of
 * lpf/kernels/compiler.py compiles the same entry points for one model).
 *
 * All extern "C" entry points accept scalar args as double and convert
 * internally, so the Python caller always passes np.float64 for scalars.
 *
 * Compile:
 *   nvcc --fatbin -std=c++17 -O3 --fmad=false \
 *        -I include/ kernels.cu -o kernels.fatbin
 *
 * Why --fmad=false (and no --use_fast_math):
 *   The fused stencil performs `D*lap + f` and `c + dt*(D*lap + f)` inside
 *   one function, so by default nvcc contracts each pair into a single
 *   `fma` instruction (one rounding instead of two). That is mathematically
 *   *more* accurate, but it produces results that differ — bit by bit —
 *   from the historical PyTorch-eager reference path, where each elementwise
 *   op lives in its own ATen kernel and FMA contraction is impossible.
 *   Disabling --fmad makes this fused kernel reproduce the legacy
 *   ladybirdmnist dataset bit-exactly: verified on all 70,000 items
 *   (du_max = dv_max = 0 for every item, 0 mismatches).
 *
 *   --use_fast_math is also intentionally omitted: it enables FTZ/DAZ and
 *   approximate div/sqrt/transcendentals, breaking IEEE-754 and
 *   reproducibility, while giving negligible speedup for our reaction-
 *   diffusion kernels (no transcendentals, only one division per pixel).
 *
 *   Performance cost of --fmad=false on this kernel: none measurable.
 *   The 5-point stencil is memory-bound (~0.5 FLOP/byte arithmetic
 *   intensity, two orders of magnitude below the GPU's compute roofline),
 *   so the extra ALU instruction sits in slack between memory loads.
 *   Single-batch benchmark on RTX 5080 (sm_120), batch=128, 500k iters,
 *   128x128 grid, Liaw, fp32: fmad=false 6.74s median, fmad=true 6.79s
 *   median — well inside run-to-run variation.
 */

#include "include/rd_kernels.cuh"

// lpf_<epilogue>_<space>_<Model>_<dtype> for every space and epilogue.
#define LPF_ALL_MODELS(T, SFX) \
    LPF_ENTRY_POINTS(GrayScott,       T, SFX) \
    LPF_ENTRY_POINTS(Liaw,            T, SFX) \
    LPF_ENTRY_POINTS(Schnakenberg,    T, SFX) \
    LPF_ENTRY_POINTS(GiererMeinhardt, T, SFX) \
    LPF_ENTRY_POINTS(Brusselator,     T, SFX) \
    LPF_ENTRY_POINTS(FitzHughNagumo,  T, SFX) \
    LPF_ENTRY_POINTS(LengyelEpstein,  T, SFX) \
    LPF_ENTRY_POINTS(Thomas,          T, SFX) \
    LPF_ENTRY_POINTS(Barkley,         T, SFX)

LPF_ALL_MODELS(__half, f16)
LPF_ALL_MODELS(float,  f32)
LPF_ALL_MODELS(double, f64)
