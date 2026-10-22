#pragma once
#include "types.cuh"
#include "reactions.cuh"

/**
 * Reaction-diffusion on the vertices of a triangle mesh (lpf.surfaces).
 *
 * The Laplace-Beltrami operator is stored in transposed ELL format, shared
 * by the batch: for vertex i and slot k < K,
 *   j = nbr[k*N + i],  w = wts[k*N + i]   (w = cot weight / vertex area)
 * Rows shorter than K are padded with j = i, w = 0. The transposed layout
 * makes consecutive threads (vertices) read consecutive addresses.
 *
 *   lap = sum_k w_k * (u_j - u_i)
 *
 * One thread per (vertex, batch). Grid: (ceil(N/BLOCK), B).
 * The arithmetic order matches the JIT templates in lpf/kernels/sources.py
 * (SURFACE_*_TEMPLATE). Must be compiled with --fmad=false.
 */
template<typename T, typename Reaction, KernelMode MODE>
__device__ void surface_core(
    const T*   __restrict__ y_in,
    T*         __restrict__ out,
    const T*   __restrict__ params,
    const int* __restrict__ nbr,
    const T*   __restrict__ wts,
    const int K, const int B, const int N,
    const T dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (i >= N) return;

    const int BN = B * N;
    const T* __restrict__ u = y_in + b * N;
    const T* __restrict__ v = y_in + BN + b * N;
    const T u_c = u[i], v_c = v[i];

    T lap_u = make_real<T>(0.0), lap_v = make_real<T>(0.0);
    for (int k = 0; k < K; ++k) {
        const int j = ldg(&nbr[k * N + i]);
        const T w = ldg(&wts[k * N + i]);
        lap_u += w * (u[j] - u_c);
        lap_v += w * (v[j] - v_c);
    }

    const int pb = b * Reaction::N_PARAMS;
    const T Du = ldg(&params[pb]), Dv = ldg(&params[pb+1]);
    T f, g;
    Reaction::compute(u_c, v_c, params, pb, f, g);

    if constexpr (MODE == EULER_STEP) {
        out[b * N + i]      = u_c + dt * (Du * lap_u + f);
        out[BN + b * N + i] = v_c + dt * (Dv * lap_v + g);
    } else {
        out[b * N + i]      = Du * lap_u + f;
        out[BN + b * N + i] = Dv * lap_v + g;
    }
}
