#pragma once
#include "types.cuh"

/* ---- template __global__ kernels for utility operations ---- */

template<typename T, typename Reaction, int TX, int TY>
__global__ void euler_kernel_tpl(
    const T* __restrict__ y_in, T* __restrict__ y_out,
    const T* __restrict__ params,
    int B, int H, int W, double dt_d, double dx2_inv_d)
{
    stencil_core<T, Reaction, TX, TY, EULER_STEP>(
        y_in, y_out, params, B, H, W,
        make_real<T>(dt_d), make_real<T>(dx2_inv_d));
}

template<typename T, typename Reaction, int TX, int TY>
__global__ void pdefunc_kernel_tpl(
    const T* __restrict__ y_in, T* __restrict__ dydt,
    const T* __restrict__ params,
    int B, int H, int W, double dx2_inv_d)
{
    stencil_core<T, Reaction, TX, TY, PDEFUNC>(
        y_in, dydt, params, B, H, W,
        make_real<T>(0.0), make_real<T>(dx2_inv_d));
}

template<typename T>
__global__ void rk_stage_tpl(
    const T* __restrict__ y, const T* __restrict__ k,
    T* __restrict__ out, double alpha_d, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    T a = make_real<T>(alpha_d);
    out[i] = y[i] + a * k[i];
}

template<typename T>
__global__ void linear_combine2_tpl(
    const T* __restrict__ x, const T* __restrict__ y,
    T* __restrict__ out, double a_d, double b_d, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    T a = make_real<T>(a_d), b = make_real<T>(b_d);
    out[i] = a * x[i] + b * y[i];
}

template<typename T>
__global__ void linear_combine3_tpl(
    const T* __restrict__ x, const T* __restrict__ y,
    const T* __restrict__ z, T* __restrict__ out,
    double a_d, double b_d, double c_d, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    T a = make_real<T>(a_d), b = make_real<T>(b_d), c = make_real<T>(c_d);
    out[i] = a * x[i] + b * y[i] + c * z[i];
}

template<typename T>
__global__ void rk4_combine_tpl(
    T* __restrict__ delta,
    const T* __restrict__ k1, const T* __restrict__ k2,
    const T* __restrict__ k3, const T* __restrict__ k4,
    double dt6_d, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    T d = make_real<T>(dt6_d);
    delta[i] = d * (k1[i] + make_real<T>(2.0)*k2[i]
                          + make_real<T>(2.0)*k3[i] + k4[i]);
}

/* ---- convergence check (single-block reduction) ---- */
template<typename T>
__global__ void convergence_reduce(
    const T* __restrict__ cur, const T* __restrict__ prev,
    double* __restrict__ out, int N)
{
    __shared__ double sd[1024], ss[1024];
    int tid = threadIdx.x;
    double md = 0.0, ms = 0.0;
    for (int i = tid; i < N; i += 1024) {
        double d = fabs((double)cur[i] - (double)prev[i]);
        double s = fabs((double)prev[i]);
        if (d > md) md = d;
        if (s > ms) ms = s;
    }
    sd[tid] = md; ss[tid] = ms;
    __syncthreads();
    for (int s = 512; s > 0; s >>= 1) {
        if (tid < s) {
            if (sd[tid+s] > sd[tid]) sd[tid] = sd[tid+s];
            if (ss[tid+s] > ss[tid]) ss[tid] = ss[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) { out[0] = sd[0]; out[1] = ss[0]; }
}
