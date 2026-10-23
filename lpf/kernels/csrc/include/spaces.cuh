#pragma once
#include "types.cuh"
#include "reactions.cuh"

/**
 * Spatial discretizations of the two-component state y = (u, v), each
 * component of shape (B, *space) and the two stored one after the other.
 *
 * A space evaluates, for the point of the calling thread,
 *
 *     du = Du * lap(u) + f(u, v),   dv = Dv * lap(v) + g(u, v)
 *
 * and hands the result to an epilogue (epilogues.cuh), which decides what
 * is written: an Euler step, dy/dt, a Runge-Kutta stage, ... Points that are
 * not solved (the fixed border of the grid, cells outside a domain) go to
 * epi.passive() instead of epi.active().
 *
 *     GridSpace    the full H x W grid, outermost cells fixed, 5-point stencil
 *     DomainSpace  a domain on the grid: cut-cell face weights (lpf.domains)
 *     MeshSpace    the vertices of a triangle mesh, operator in ELL format
 *                  (MeshSpace<T, 256, 4>: four batch elements per thread)
 *
 * Every space is a small POD struct passed to the kernel by value. The
 * arithmetic (operation order, no FMA contraction: compile with
 * --fmad=false) is that of the reference NumPy path, so all backends agree
 * bit for bit.
 */

// Shared-memory tile of u and v with a one-cell halo. Out-of-grid halo cells
// repeat the center value (their weight is zero or the cell is fixed).
template<typename T, int TX, int TY>
struct Tile {
    static constexpr int SW = TX + 2;
    T* s_u;
    T* s_v;
    int si, gidx;
    T u_val, v_val;
    bool valid;

    __device__ __forceinline__ void load(const T* __restrict__ y_in,
                                         int b, int gx, int gy,
                                         int H, int W, int HW, int BHW) {
        const int tx = threadIdx.x, ty = threadIdx.y;
        gidx = b * HW + gy * W + gx;
        si = (ty + 1) * SW + (tx + 1);
        valid = (gx < W && gy < H);

        u_val = valid ? y_in[gidx]       : make_real<T>(0.0);
        v_val = valid ? y_in[BHW + gidx] : make_real<T>(0.0);
        s_u[si] = u_val;  s_v[si] = v_val;

        if (ty == 0) {
            const int h = tx + 1, h_gy = gy - 1;
            if (h_gy >= 0 && gx < W) { int hi = b*HW + h_gy*W + gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
            else { s_u[h] = u_val; s_v[h] = v_val; }
        }
        if (ty == TY - 1) {
            const int h = (TY+1)*SW + (tx+1), h_gy = gy + 1;
            if (h_gy < H && gx < W) { int hi = b*HW + h_gy*W + gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
            else { s_u[h] = u_val; s_v[h] = v_val; }
        }
        if (tx == 0) {
            const int h = (ty+1)*SW, h_gx = gx - 1;
            if (h_gx >= 0 && gy < H) { int hi = b*HW + gy*W + h_gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
            else { s_u[h] = u_val; s_v[h] = v_val; }
        }
        if (tx == TX - 1) {
            const int h = (ty+1)*SW + (TX+1), h_gx = gx + 1;
            if (h_gx < W && gy < H) { int hi = b*HW + gy*W + h_gx; s_u[h] = y_in[hi]; s_v[h] = y_in[BHW+hi]; }
            else { s_u[h] = u_val; s_v[h] = v_val; }
        }
    }
};

// Reaction and diffusion coefficients of batch element b, then the epilogue.
template<typename T, typename R, typename Epi>
__device__ __forceinline__ void finish(
    const T* __restrict__ params, int b, int i, int off,
    T u_c, T v_c, T lap_u, T lap_v, const Epi& epi)
{
    const int pb = b * R::N_PARAMS;
    const T Du = ldg(&params[pb]), Dv = ldg(&params[pb+1]);
    T f, g;
    R::compute(u_c, v_c, params, pb, f, g);
    epi.active(i, off, u_c, v_c, Du*lap_u + f, Dv*lap_v + g);
}


// ---------------------------------------------------------------------------
// Full grid. Block (TX, TY), grid (ceil(W/TX), ceil(H/TY), B).
// ---------------------------------------------------------------------------
template<typename T, int TX = 32, int TY = 8>
struct GridSpace {
    int B, H, W;
    double dx2_inv;

    template<typename R, typename Epi>
    __device__ __forceinline__ void run(const T* __restrict__ y_in,
                                        const T* __restrict__ params,
                                        const Epi& epi) const {
        constexpr int SW = TX + 2;
        __shared__ T s_u[(TY + 2) * SW];
        __shared__ T s_v[(TY + 2) * SW];

        const int gx = blockIdx.x * TX + threadIdx.x;
        const int gy = blockIdx.y * TY + threadIdx.y;
        const int b  = blockIdx.z;
        const int HW = H * W, BHW = B * HW;

        Tile<T, TX, TY> t{s_u, s_v};
        t.load(y_in, b, gx, gy, H, W, HW, BHW);
        __syncthreads();

        if (!t.valid) return;
        if (gx == 0 || gx == W-1 || gy == 0 || gy == H-1) {
            epi.passive(t.gidx, BHW, t.u_val, t.v_val);
            return;
        }

        const int si = t.si;
        const T dx2i = make_real<T>(dx2_inv);
        const T u_c = s_u[si], v_c = s_v[si];
        const T lap_u = (s_u[si-SW]+s_u[si-1]+s_u[si+SW]+s_u[si+1] - make_real<T>(4.0)*u_c) * dx2i;
        const T lap_v = (s_v[si-SW]+s_v[si-1]+s_v[si+SW]+s_v[si+1] - make_real<T>(4.0)*v_c) * dx2i;
        finish<T, R>(params, b, t.gidx, BHW, u_c, v_c, lap_u, lap_v, epi);
    }
};


// ---------------------------------------------------------------------------
// Domain on the grid (lpf.domains). Same launch shape as GridSpace.
//
// stencil holds (5, H, W) per batch element, [w_north, w_south, w_west,
// w_east, active]; element b uses stencil + b * stride (stride 0: one
// domain shared by the batch).
//   active cells:   lap = (wN*(uN-u) + wS*(uS-u) + wW*(uW-u) + wE*(uE-u)) * dx2_inv
//   inactive cells: passive (they keep their values)
// flags (H, W) per batch element (at b * H * W, or shared when stride is 0)
// classify the cells so that most of them skip the weights: 0 inactive,
// 1 all four weights exactly 1 (w * x == x, so the result is the same),
// 2 any other active cell. Unlike GridSpace, the outermost cells are solved
// like any other; their missing neighbors have zero weight.
// ---------------------------------------------------------------------------
enum DomainCell : unsigned char { CELL_INACTIVE = 0, CELL_FULL = 1, CELL_CUT = 2 };

template<typename T, int TX = 32, int TY = 8>
struct DomainSpace {
    int B, H, W;
    double dx2_inv;
    const T* stencil;
    long long stride;
    const unsigned char* flags;

    template<typename R, typename Epi>
    __device__ __forceinline__ void run(const T* __restrict__ y_in,
                                        const T* __restrict__ params,
                                        const Epi& epi) const {
        constexpr int SW = TX + 2;
        __shared__ T s_u[(TY + 2) * SW];
        __shared__ T s_v[(TY + 2) * SW];

        const int gx = blockIdx.x * TX + threadIdx.x;
        const int gy = blockIdx.y * TY + threadIdx.y;
        const int b  = blockIdx.z;
        const int HW = H * W, BHW = B * HW;

        Tile<T, TX, TY> t{s_u, s_v};
        t.load(y_in, b, gx, gy, H, W, HW, BHW);
        __syncthreads();

        if (!t.valid) return;

        const int cidx = gy * W + gx;
        const unsigned char cell =
            __ldg(&flags[(stride ? (long long)b * HW : 0) + cidx]);
        if (cell == CELL_INACTIVE) {
            epi.passive(t.gidx, BHW, t.u_val, t.v_val);
            return;
        }

        const int si = t.si;
        const T dx2i = make_real<T>(dx2_inv);
        const T u_c = s_u[si], v_c = s_v[si];
        T lap_u, lap_v;
        if (cell == CELL_FULL) {
            lap_u = ((s_u[si-SW] - u_c) + (s_u[si+SW] - u_c)
                     + (s_u[si-1] - u_c) + (s_u[si+1] - u_c)) * dx2i;
            lap_v = ((s_v[si-SW] - v_c) + (s_v[si+SW] - v_c)
                     + (s_v[si-1] - v_c) + (s_v[si+1] - v_c)) * dx2i;
        } else {
            const T* __restrict__ st = stencil + b * stride;
            const T wn = ldg(&st[cidx]),        ws = ldg(&st[HW + cidx]);
            const T ww = ldg(&st[2*HW + cidx]), we = ldg(&st[3*HW + cidx]);
            lap_u = (wn * (s_u[si-SW] - u_c) + ws * (s_u[si+SW] - u_c)
                     + ww * (s_u[si-1] - u_c) + we * (s_u[si+1] - u_c)) * dx2i;
            lap_v = (wn * (s_v[si-SW] - v_c) + ws * (s_v[si+SW] - v_c)
                     + ww * (s_v[si-1] - v_c) + we * (s_v[si+1] - v_c)) * dx2i;
        }
        finish<T, R>(params, b, t.gidx, BHW, u_c, v_c, lap_u, lap_v, epi);
    }
};


// ---------------------------------------------------------------------------
// Triangle-mesh surface (lpf.surfaces).
// Block (BLOCK), grid (ceil(N/BLOCK), ceil(B/BCHUNK)).
//
// The Laplace-Beltrami operator is shared by the batch, in transposed ELL
// format: for vertex i and slot k < K, j = nbr[k*N + i], w = wts[k*N + i]
// (w = cot weight / vertex area; short rows padded with j = i, w = 0), and
//   lap = sum_k w_k * (u_j - u_i).
// The transposed layout makes consecutive threads read consecutive slots.
// A thread handles BCHUNK batch elements of its vertex, so the operator is
// read once for all of them (up to 2x faster from B = 4 on; BCHUNK = 1 is
// best for a single model). The sum order per element does not change.
// ---------------------------------------------------------------------------
template<typename T, int BLOCK = 256, int BCHUNK = 1>
struct MeshSpace {
    int B, N, K;
    const int* nbr;
    const T* wts;

    template<typename R, typename Epi>
    __device__ __forceinline__ void run(const T* __restrict__ y_in,
                                        const T* __restrict__ params,
                                        const Epi& epi) const {
        const int i = blockIdx.x * BLOCK + threadIdx.x;
        const int b0 = blockIdx.y * BCHUNK;
        if (i >= N) return;

        const int BN = B * N;
        const int n_b = min(BCHUNK, B - b0);
        const int* __restrict__ nb = nbr;
        const T* __restrict__ wt = wts;

        T u_c[BCHUNK], v_c[BCHUNK], lap_u[BCHUNK], lap_v[BCHUNK];
#pragma unroll
        for (int c = 0; c < BCHUNK; ++c) {
            lap_u[c] = make_real<T>(0.0);
            lap_v[c] = make_real<T>(0.0);
            if (c < n_b) {
                u_c[c] = y_in[(b0 + c) * N + i];
                v_c[c] = y_in[BN + (b0 + c) * N + i];
            }
        }
        for (int k = 0; k < K; ++k) {
            const int j = ldg(&nb[k * N + i]);
            const T w = ldg(&wt[k * N + i]);
#pragma unroll
            for (int c = 0; c < BCHUNK; ++c) {
                if (c < n_b) {
                    lap_u[c] += w * (y_in[(b0 + c) * N + j] - u_c[c]);
                    lap_v[c] += w * (y_in[BN + (b0 + c) * N + j] - v_c[c]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < BCHUNK; ++c)
            if (c < n_b)
                finish<T, R>(params, b0 + c, (b0 + c) * N + i, BN,
                             u_c[c], v_c[c], lap_u[c], lap_v[c], epi);
    }
};
