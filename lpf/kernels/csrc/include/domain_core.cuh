#pragma once
#include "types.cuh"
#include "reactions.cuh"

/**
 * Shared-memory tiled stencil restricted to a spatial domain (lpf.domains).
 *
 * `stencil` has shape (5, H, W) and is shared by the batch:
 *   [w_north, w_south, w_west, w_east, active]
 * Active cells:   lap = (wN*(uN-u) + wS*(uS-u) + wW*(uW-u) + wE*(uE-u)) * dx2_inv
 * Inactive cells: keep their value (EULER_STEP) / dydt = 0 (PDEFUNC).
 * Unlike stencil_core, the outermost grid cells are solved like any other
 * cell; their missing neighbors have zero weight.
 *
 * The arithmetic order matches the JIT templates in lpf/kernels/sources.py
 * (_DOMAIN_STENCIL_BODY), so AOT, JIT and the native loop agree bit for bit.
 * Must be compiled with --fmad=false, like stencil_core.
 */
template<typename T, typename Reaction, int TILE_X, int TILE_Y, KernelMode MODE>
__device__ void domain_core(
    const T* __restrict__ y_in,
    T*       __restrict__ out,
    const T* __restrict__ params,
    const T* __restrict__ stencil,
    const int B, const int H, const int W,
    const T dt, const T dx2_inv)
{
    constexpr int SW = TILE_X + 2;
    __shared__ T s_u[(TILE_Y + 2) * SW];
    __shared__ T s_v[(TILE_Y + 2) * SW];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_X + tx;
    const int gy = blockIdx.y * TILE_Y + ty;
    const int b  = blockIdx.z;
    const int HW = H * W, BHW = B * HW;
    const int gidx = b * HW + gy * W + gx;
    const int si = (ty + 1) * SW + (tx + 1);
    const bool valid = (gx < W && gy < H);

    const T u_val = valid ? y_in[gidx]       : make_real<T>(0.0);
    const T v_val = valid ? y_in[BHW + gidx] : make_real<T>(0.0);
    s_u[si] = u_val;  s_v[si] = v_val;

    // halos (out-of-grid halos repeat the center; their weight is zero)
    if (ty == 0) {
        const int h = tx + 1, h_gy = gy - 1;
        if (h_gy >= 0 && gx < W) { int hi = b*HW + h_gy*W + gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (ty == TILE_Y - 1) {
        const int h = (TILE_Y+1)*SW + (tx+1), h_gy = gy + 1;
        if (h_gy < H && gx < W) { int hi = b*HW + h_gy*W + gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (tx == 0) {
        const int h = (ty+1)*SW, h_gx = gx - 1;
        if (h_gx >= 0 && gy < H) { int hi = b*HW + gy*W + h_gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    if (tx == TILE_X - 1) {
        const int h = (ty+1)*SW + (TILE_X+1), h_gx = gx + 1;
        if (h_gx < W && gy < H) { int hi = b*HW + gy*W + h_gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
        else { s_u[h] = u_val; s_v[h] = v_val; }
    }
    __syncthreads();

    if (!valid) return;

    const int cidx = gy * W + gx;
    if (ldg(&stencil[4*HW + cidx]) == make_real<T>(0.0)) {
        if constexpr (MODE == EULER_STEP) { out[gidx] = u_val; out[BHW+gidx] = v_val; }
        else { out[gidx] = make_real<T>(0.0); out[BHW+gidx] = make_real<T>(0.0); }
        return;
    }

    const T wn = ldg(&stencil[cidx]),        ws = ldg(&stencil[HW + cidx]);
    const T ww = ldg(&stencil[2*HW + cidx]), we = ldg(&stencil[3*HW + cidx]);

    const T u_c = s_u[si], v_c = s_v[si];
    const T lap_u = (wn * (s_u[si-SW] - u_c) + ws * (s_u[si+SW] - u_c)
                     + ww * (s_u[si-1] - u_c) + we * (s_u[si+1] - u_c)) * dx2_inv;
    const T lap_v = (wn * (s_v[si-SW] - v_c) + ws * (s_v[si+SW] - v_c)
                     + ww * (s_v[si-1] - v_c) + we * (s_v[si+1] - v_c)) * dx2_inv;

    const int pb = b * Reaction::N_PARAMS;
    const T Du = ldg(&params[pb]), Dv = ldg(&params[pb+1]);
    T f, g;
    Reaction::compute(u_c, v_c, params, pb, f, g);

    if constexpr (MODE == EULER_STEP) {
        out[gidx]     = u_c + dt * (Du*lap_u + f);
        out[BHW+gidx] = v_c + dt * (Dv*lap_v + g);
    } else {
        out[gidx]     = Du*lap_u + f;
        out[BHW+gidx] = Dv*lap_v + g;
    }
}
