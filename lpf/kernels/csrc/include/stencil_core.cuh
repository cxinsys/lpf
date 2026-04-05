#pragma once
#include "types.cuh"
#include "reactions.cuh"

/**
 * Shared-memory tiled 5-point stencil core.
 * MODE = EULER_STEP: out = y + dt*(D*lap + reaction)
 * MODE = PDEFUNC:    out = D*lap + reaction  (boundary = 0)
 */
template<typename T, typename Reaction, int TILE_X, int TILE_Y, KernelMode MODE>
__device__ void stencil_core(
    const T* __restrict__ y_in,
    T*       __restrict__ out,
    const T* __restrict__ params,
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

    // halos
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
    if (gx == 0 || gx == W-1 || gy == 0 || gy == H-1) {
        if constexpr (MODE == EULER_STEP) { out[gidx] = u_val; out[BHW+gidx] = v_val; }
        else { out[gidx] = make_real<T>(0.0); out[BHW+gidx] = make_real<T>(0.0); }
        return;
    }

    const T u_c = s_u[si], v_c = s_v[si];
    const T lap_u = (s_u[si-SW]+s_u[si-1]+s_u[si+SW]+s_u[si+1] - make_real<T>(4.0)*u_c) * dx2_inv;
    const T lap_v = (s_v[si-SW]+s_v[si-1]+s_v[si+SW]+s_v[si+1] - make_real<T>(4.0)*v_c) * dx2_inv;

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
