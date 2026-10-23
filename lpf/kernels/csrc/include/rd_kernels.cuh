#pragma once
#include "spaces.cuh"
#include "epilogues.cuh"

/**
 * Fused reaction-diffusion kernels: a space (spaces.cuh) evaluates dy/dt at
 * the thread's point and an epilogue (epilogues.cuh) writes the result.
 *
 * rd_kernel is used directly by the native solve loops (solver_loop.cu).
 * For CuPy (the AOT fatbin and NVRTC JIT, see lpf/kernels/compiler.py) the
 * LPF_ENTRY_POINTS macro below defines extern "C" kernels named
 *
 *     lpf_<epilogue>_<space>_<Model>_<dtype>, e.g. lpf_stage_mesh_Liaw_f32
 *
 * with the arguments (<space arguments>, y, params, <epilogue arguments>):
 *
 *     grid    B, H, W, dx2_inv
 *     domain  B, H, W, dx2_inv, stencil, stride, flags
 *     mesh    B, N, K, nbr, wts   (one batch element per thread)
 *     meshb   B, N, K, nbr, wts   (four batch elements per thread)
 *
 *     euler   y_next, dt                   y_next = y + dt f(y)
 *     rhs     dydt                         dydt = f(y)
 *     stage   k, y_next, base, alpha       k = f(y), y_next = base + alpha k
 *     heun    y_next, base, k1, a, b       y_next = base + (a k1 + b f(y))
 *     rk23    y_next, base, k1, k2, a, b, c
 *                                          y_next = base + (a k1 + b k2 + c f(y))
 *     rk4     y_next, base, k1, k2, k3, d  y_next = base + d (k1 + 2 k2 + 2 k3 + f(y))
 *     ab2     cur, y_next, prev, a, b      cur = f(y), y_next = y + (a cur + b prev)
 *
 * Each kernel takes only the arguments it uses: CuPy's launch cost grows
 * with the argument count, and small problems are launch-bound.
 */

template<typename T, typename R, typename Space, typename Epi>
__global__ void rd_kernel(const Space sp, const T* __restrict__ y,
                          const T* __restrict__ params, const Epi epi)
{
    sp.template run<R>(y, params, epi);
}

template<typename R, typename Space, typename T, typename Epi>
__device__ __forceinline__ void lpf_run(const Space& sp, const T* y,
                                        const T* params, const Epi& epi)
{
    sp.template run<R>(y, params, epi);
}

#define LPF_SPACE_ARGS_grid(T)   int B, int H, int W, double dx2_inv
#define LPF_SPACE_grid(T)        GridSpace<T>{B, H, W, dx2_inv}
#define LPF_SPACE_ARGS_domain(T) int B, int H, int W, double dx2_inv, \
                                 const T* stencil, long long stride, \
                                 const unsigned char* flags
#define LPF_SPACE_domain(T)      DomainSpace<T>{B, H, W, dx2_inv, stencil, stride, flags}
#define LPF_SPACE_ARGS_mesh(T)   int B, int N, int K, const int* nbr, const T* wts
#define LPF_SPACE_mesh(T)        MeshSpace<T>{B, N, K, nbr, wts}
#define LPF_SPACE_ARGS_meshb(T)  LPF_SPACE_ARGS_mesh(T)
#define LPF_SPACE_meshb(T)       MeshSpace<T, 256, 4>{B, N, K, nbr, wts}

#define LPF_EPI_ARGS_euler(T) T* y_next, double dt
#define LPF_EPI_euler(T)      EulerEpi<T>{y_next, dt}
#define LPF_EPI_ARGS_rhs(T)   T* dydt
#define LPF_EPI_rhs(T)        RhsEpi<T>{dydt}
#define LPF_EPI_ARGS_stage(T) T* k, T* y_next, const T* base, double alpha
#define LPF_EPI_stage(T)      StageEpi<T>{k, y_next, base, base == y, alpha}
#define LPF_EPI_ARGS_heun(T)  T* y_next, const T* base, const T* k1, double a, double b
#define LPF_EPI_heun(T)       HeunEpi<T>{y_next, base, k1, a, b}
#define LPF_EPI_ARGS_rk23(T)  T* y_next, const T* base, const T* k1, const T* k2, \
                              double a, double b, double c
#define LPF_EPI_rk23(T)       RK23Epi<T>{y_next, base, k1, k2, a, b, c}
#define LPF_EPI_ARGS_rk4(T)   T* y_next, const T* base, const T* k1, const T* k2, \
                              const T* k3, double d
#define LPF_EPI_rk4(T)        RK4Epi<T>{y_next, base, k1, k2, k3, d}
#define LPF_EPI_ARGS_ab2(T)   T* cur, T* y_next, const T* prev, double a, double b
#define LPF_EPI_ab2(T)        AB2Epi<T>{cur, y_next, prev, a, b}

#define LPF_ENTRY(EPI, SPACE, MODEL, T, SFX) \
extern "C" __global__ void lpf_##EPI##_##SPACE##_##MODEL##_##SFX( \
    LPF_SPACE_ARGS_##SPACE(T), \
    const T* __restrict__ y, const T* __restrict__ params, \
    LPF_EPI_ARGS_##EPI(T)) \
{ \
    lpf_run<MODEL##Reaction<T>>(LPF_SPACE_##SPACE(T), y, params, \
                                LPF_EPI_##EPI(T)); \
}

#define LPF_SPACE_ENTRY_POINTS(SPACE, MODEL, T, SFX) \
    LPF_ENTRY(euler, SPACE, MODEL, T, SFX) \
    LPF_ENTRY(rhs,   SPACE, MODEL, T, SFX) \
    LPF_ENTRY(stage, SPACE, MODEL, T, SFX) \
    LPF_ENTRY(heun,  SPACE, MODEL, T, SFX) \
    LPF_ENTRY(rk23,  SPACE, MODEL, T, SFX) \
    LPF_ENTRY(rk4,   SPACE, MODEL, T, SFX) \
    LPF_ENTRY(ab2,   SPACE, MODEL, T, SFX)

#define LPF_ENTRY_POINTS(MODEL, T, SFX) \
    LPF_SPACE_ENTRY_POINTS(grid,   MODEL, T, SFX) \
    LPF_SPACE_ENTRY_POINTS(domain, MODEL, T, SFX) \
    LPF_SPACE_ENTRY_POINTS(mesh,   MODEL, T, SFX) \
    LPF_SPACE_ENTRY_POINTS(meshb,  MODEL, T, SFX)
