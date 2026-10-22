/**
 * lpf native solve loops — entire solve() in C/CUDA, zero Python during iteration.
 *
 * Features supported:
 *   - All 5 solver types (Euler, Heun, RK4, RK23, AB2)
 *   - All 9 model types × 3 dtypes (f16/f32/f64)
 *   - Three spatial operators: the full grid (lpf_native_solve), a domain
 *     on the grid (lpf_native_solve_domain, lpf.domains) and a triangle-mesh
 *     surface (lpf_native_solve_surface, lpf.surfaces)
 *   - Trajectory capture at arbitrary waypoints
 *   - Early stopping (convergence check every N iters)
 *   - Ping-pong buffers
 *
 * Compile:
 *   nvcc -shared -Xcompiler -fPIC -std=c++17 -O3 --fmad=false \
 *        -I include/ solver_loop.cu -o libsolver.so
 *
 * --fmad=false (and no --use_fast_math) is required for bit-exact
 * reproduction of the legacy PyTorch-eager dataset; see kernels.cu for
 * the full rationale and benchmark numbers.
 */

#include "include/stencil_core.cuh"
#include "include/domain_core.cuh"
#include "include/surface_core.cuh"
#include "include/utility_tpl.cuh"
#include <cstdio>

// ============================================================
// Convergence helper
// ============================================================
template<typename T>
static bool check_convergence(
    const T* cur, const T* prev, int N, double rtol,
    double* d_buf, double h_buf[2], cudaStream_t s)
{
    double zero[2] = {0.0, 0.0};
    cudaMemcpyAsync(d_buf, zero, 16, cudaMemcpyHostToDevice, s);
    convergence_reduce<T><<<1, 1024, 0, s>>>(cur, prev, d_buf, N);
    cudaMemcpyAsync(h_buf, d_buf, 16, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return h_buf[0] / (h_buf[1] + 1e-30) < rtol;
}

// ============================================================
// Trajectory helper — device-to-device async copy
// ============================================================
template<typename T>
static inline void try_capture_trj(
    T* trj_buf, const int* trj_iters, int n_trj, int& trj_idx,
    const T* state, int elem_count, int iter_1based, cudaStream_t s)
{
    if (!trj_buf || trj_idx >= n_trj) return;
    if (trj_iters[trj_idx] != iter_1based) return;
    cudaMemcpyAsync(trj_buf + (long long)trj_idx * elem_count,
                    state, elem_count * sizeof(T),
                    cudaMemcpyDeviceToDevice, s);
    trj_idx++;
}

// ============================================================
// Spatial operators
// ============================================================
// An operator launches one fused Euler step (euler) or one dy/dt
// evaluation (pdefunc) over the whole state of size(). The solve loops
// below are written against this interface, so every solver runs on the
// full grid, on a domain and on a mesh surface. GridOp launches exactly
// the kernels (and arguments) the full-grid loops always launched.

template<typename T, typename Reaction, int TX, int TY, KernelMode MODE>
__global__ void domain_kernel_tpl(
    const T* __restrict__ y_in, T* __restrict__ out,
    const T* __restrict__ params, const T* __restrict__ stencil,
    int stencil_stride, int B, int H, int W, double dt_d, double dx2_inv_d)
{
    domain_core<T, Reaction, TX, TY, MODE>(
        y_in, out, params, stencil, stencil_stride, B, H, W,
        make_real<T>(dt_d), make_real<T>(dx2_inv_d));
}

template<typename T, typename Reaction, KernelMode MODE>
__global__ void surface_kernel_tpl(
    const T* __restrict__ y_in, T* __restrict__ out,
    const T* __restrict__ params,
    const int* __restrict__ nbr, const T* __restrict__ wts,
    int K, int B, int N, double dt_d)
{
    surface_core<T, Reaction, MODE>(
        y_in, out, params, nbr, wts, K, B, N, make_real<T>(dt_d));
}

template<typename T, typename Reaction, int TX, int TY>
struct GridOp {
    int B, H, W;
    double dx2_inv;

    int size() const { return 2*B*H*W; }
    dim3 grid() const { return dim3((W+TX-1)/TX, (H+TY-1)/TY, B); }

    void euler(const T* src, T* dst, const T* params, double dt, cudaStream_t s) const {
        euler_kernel_tpl<T,Reaction,TX,TY><<<grid(),dim3(TX,TY),0,s>>>(
            src, dst, params, B, H, W, dt, dx2_inv);
    }
    void pdefunc(const T* src, T* dydt, const T* params, cudaStream_t s) const {
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<grid(),dim3(TX,TY),0,s>>>(
            src, dydt, params, B, H, W, dx2_inv);
    }
};

template<typename T, typename Reaction, int TX, int TY>
struct DomainOp {
    int B, H, W;
    double dx2_inv;
    const T* stencil;
    int stencil_stride;  // 0: one domain for the batch, 5*H*W: one each

    int size() const { return 2*B*H*W; }
    dim3 grid() const { return dim3((W+TX-1)/TX, (H+TY-1)/TY, B); }

    void euler(const T* src, T* dst, const T* params, double dt, cudaStream_t s) const {
        domain_kernel_tpl<T,Reaction,TX,TY,EULER_STEP><<<grid(),dim3(TX,TY),0,s>>>(
            src, dst, params, stencil, stencil_stride, B, H, W, dt, dx2_inv);
    }
    void pdefunc(const T* src, T* dydt, const T* params, cudaStream_t s) const {
        domain_kernel_tpl<T,Reaction,TX,TY,PDEFUNC><<<grid(),dim3(TX,TY),0,s>>>(
            src, dydt, params, stencil, stencil_stride, B, H, W, 0.0, dx2_inv);
    }
};

template<typename T, typename Reaction, int BLOCK = 256>
struct SurfaceOp {
    int B, N, K;
    const int* nbr;
    const T* wts;

    int size() const { return 2*B*N; }
    dim3 grid() const { return dim3((N+BLOCK-1)/BLOCK, B); }

    void euler(const T* src, T* dst, const T* params, double dt, cudaStream_t s) const {
        surface_kernel_tpl<T,Reaction,EULER_STEP><<<grid(),BLOCK,0,s>>>(
            src, dst, params, nbr, wts, K, B, N, dt);
    }
    void pdefunc(const T* src, T* dydt, const T* params, cudaStream_t s) const {
        surface_kernel_tpl<T,Reaction,PDEFUNC><<<grid(),BLOCK,0,s>>>(
            src, dydt, params, nbr, wts, K, B, N, 0.0);
    }
};

// Arguments shared by every solve loop (after the operator-specific ones).
#define LOOP_TAIL_PARAMS \
    T* trj_buf, const int* trj_iters, int n_trj, \
    double rtol, int rtol_period, int* stopped_at, \
    double* d_conv, double h_conv[2], cudaStream_t s

#define CHECK_STOP(i) \
    if (rtol>0 && rtol_period>0 && (i+1)%rtol_period==0) \
        if (check_convergence(dst,src,N,rtol,d_conv,h_conv,s)) \
            { *stopped_at=i+1; cudaStreamSynchronize(s); return (i+1)%2; }

// ============================================================
// EULER solve loop
// ============================================================
template<typename T, typename Op>
static int solve_euler(
    T* y_a, T* y_b, const T* params, const Op& op, double dt, int n_iters,
    LOOP_TAIL_PARAMS)
{
    int N = op.size(), trj_idx = 0;

    for (int i = 0; i < n_iters; i++) {
        T *src = (i%2==0)?y_a:y_b, *dst = (i%2==0)?y_b:y_a;
        op.euler(src, dst, params, dt, s);

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        CHECK_STOP(i)
    }
    *stopped_at = -1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// HEUN solve loop
// ============================================================
template<typename T, typename Op>
static int solve_heun(
    T* y_a, T* y_b, const T* params, const Op& op, double dt, int n_iters,
    T* k1, T* k2, T* y_temp,
    LOOP_TAIL_PARAMS)
{
    int N = op.size(), ub = 256, ug = (N+ub-1)/ub, trj_idx = 0;

    for (int i = 0; i < n_iters; i++) {
        T *src=(i%2==0)?y_a:y_b, *dst=(i%2==0)?y_b:y_a;
        op.pdefunc(src,k1,params,s);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k1,y_temp,dt,N);
        op.pdefunc(y_temp,k2,params,s);
        linear_combine2_tpl<T><<<ug,ub,0,s>>>(k1,k2,y_temp,0.5*dt,0.5*dt,N);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,y_temp,dst,1.0,N);

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        CHECK_STOP(i)
    }
    *stopped_at=-1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// RK4 solve loop
// ============================================================
template<typename T, typename Op>
static int solve_rk4(
    T* y_a, T* y_b, const T* params, const Op& op, double dt, int n_iters,
    T* k1, T* k2, T* k3, T* k4, T* y_temp, T* delta,
    LOOP_TAIL_PARAMS)
{
    int N=op.size(), ub=256, ug=(N+ub-1)/ub, trj_idx=0;

    for (int i = 0; i < n_iters; i++) {
        T *src=(i%2==0)?y_a:y_b, *dst=(i%2==0)?y_b:y_a;
        op.pdefunc(src,k1,params,s);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k1,y_temp,0.5*dt,N);
        op.pdefunc(y_temp,k2,params,s);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k2,y_temp,0.5*dt,N);
        op.pdefunc(y_temp,k3,params,s);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k3,y_temp,dt,N);
        op.pdefunc(y_temp,k4,params,s);
        rk4_combine_tpl<T><<<ug,ub,0,s>>>(delta,k1,k2,k3,k4,dt/6.0,N);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,delta,dst,1.0,N);

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        CHECK_STOP(i)
    }
    *stopped_at=-1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// RK23 solve loop
// ============================================================
template<typename T, typename Op>
static int solve_rk23(
    T* y_a, T* y_b, const T* params, const Op& op, double dt, int n_iters,
    T* k1, T* k2, T* k3, T* y_temp, T* delta,
    LOOP_TAIL_PARAMS)
{
    int N=op.size(), ub=256, ug=(N+ub-1)/ub, trj_idx=0;

    for (int i = 0; i < n_iters; i++) {
        T *src=(i%2==0)?y_a:y_b, *dst=(i%2==0)?y_b:y_a;
        op.pdefunc(src,k1,params,s);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k1,y_temp,0.5*dt,N);
        op.pdefunc(y_temp,k2,params,s);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k2,y_temp,0.75*dt,N);
        op.pdefunc(y_temp,k3,params,s);
        linear_combine3_tpl<T><<<ug,ub,0,s>>>(k1,k2,k3,delta,
            dt*(2.0/9.0), dt*(1.0/3.0), dt*(4.0/9.0), N);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,delta,dst,1.0,N);

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        CHECK_STOP(i)
    }
    *stopped_at=-1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// AB2 solve loop
// ============================================================
template<typename T, typename Op>
static int solve_ab2(
    T* y_a, T* y_b, const T* params, const Op& op, double dt, int n_iters,
    T* dydt_a, T* dydt_b, T* delta,
    LOOP_TAIL_PARAMS)
{
    int N=op.size(), ub=256, ug=(N+ub-1)/ub, trj_idx=0;
    T* dydt_cur = dydt_a;
    T* dydt_prev = dydt_b;

    for (int i = 0; i < n_iters; i++) {
        T *src=(i%2==0)?y_a:y_b, *dst=(i%2==0)?y_b:y_a;
        op.pdefunc(src,dydt_cur,params,s);

        if (i == 0) {
            rk_stage_tpl<T><<<ug,ub,0,s>>>(src,dydt_cur,dst,dt,N);
        } else {
            linear_combine2_tpl<T><<<ug,ub,0,s>>>(dydt_cur,dydt_prev,delta,1.5*dt,-0.5*dt,N);
            rk_stage_tpl<T><<<ug,ub,0,s>>>(src,delta,dst,1.0,N);
        }
        T* tmp = dydt_cur; dydt_cur = dydt_prev; dydt_prev = tmp;

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        CHECK_STOP(i)
    }
    *stopped_at=-1; cudaStreamSynchronize(s); return n_iters%2;
}

#undef CHECK_STOP
#undef LOOP_TAIL_PARAMS

// ============================================================
// Dispatch
// ============================================================

/*
 * work[] provides solver-specific buffers:
 *   Euler: (none needed)
 *   Heun:  work[0]=k1, [1]=k2, [2]=y_temp
 *   RK4:   work[0]=k1, [1]=k2, [2]=k3, [3]=k4, [4]=y_temp, [5]=delta
 *   RK23:  work[0]=k1, [1]=k2, [2]=k3, [3]=y_temp, [4]=delta
 *   AB2:   work[0]=dydt_a, [1]=dydt_b, [2]=delta
 * Solver type: 0=Euler, 1=Heun, 2=RK4, 3=RK23, 4=AB2.
 * Each entry point returns the buffer holding the final state (0: y_a, 1: y_b).
 */

struct LoopArgs {
    void* y_a; void* y_b; const void* params;
    double dt; int n_iters; int solver_type;
    void** work;
    void* trj_buf; const int* trj_iters; int n_trj;
    double rtol; int rtol_period; int* stopped_at;
    double* d_conv; double* h_conv; cudaStream_t stream;
};

template<typename T, typename Op>
static int run_solver(const LoopArgs& a, const Op& op)
{
    T* y_a = (T*)a.y_a;
    T* y_b = (T*)a.y_b;
    const T* p = (const T*)a.params;
    #define WK(k) ((T*)a.work[k])
    #define TAIL (T*)a.trj_buf, a.trj_iters, a.n_trj, a.rtol, a.rtol_period, \
                 a.stopped_at, a.d_conv, a.h_conv, a.stream
    switch (a.solver_type) {
        case 0: return solve_euler<T>(y_a, y_b, p, op, a.dt, a.n_iters, TAIL);
        case 1: return solve_heun<T>(y_a, y_b, p, op, a.dt, a.n_iters,
                                     WK(0), WK(1), WK(2), TAIL);
        case 2: return solve_rk4<T>(y_a, y_b, p, op, a.dt, a.n_iters,
                                    WK(0), WK(1), WK(2), WK(3), WK(4), WK(5), TAIL);
        case 3: return solve_rk23<T>(y_a, y_b, p, op, a.dt, a.n_iters,
                                     WK(0), WK(1), WK(2), WK(3), WK(4), TAIL);
        case 4: return solve_ab2<T>(y_a, y_b, p, op, a.dt, a.n_iters,
                                    WK(0), WK(1), WK(2), TAIL);
    }
    #undef TAIL
    #undef WK
    return 0;
}

template<typename X> struct Tag { using type = X; };

// dtype code -> scalar type: 0=f16, 1=f32, 2=f64
template<typename F>
static int with_dtype(int dtype_code, F&& f)
{
    switch (dtype_code) {
        case 0: return f(Tag<__half>{});
        case 1: return f(Tag<float>{});
        case 2: return f(Tag<double>{});
    }
    return 0;
}

// model code -> reaction struct (order matches lpf/kernels/aot/native.py)
template<typename T, typename F>
static int with_reaction(int model_type, F&& f)
{
    switch (model_type) {
        case 0: return f(Tag<GrayScottReaction<T>>{});
        case 1: return f(Tag<LiawReaction<T>>{});
        case 2: return f(Tag<SchnakenbergReaction<T>>{});
        case 3: return f(Tag<GiererMeinhardtReaction<T>>{});
        case 4: return f(Tag<BrusselatorReaction<T>>{});
        case 5: return f(Tag<FitzHughNagumoReaction<T>>{});
        case 6: return f(Tag<LengyelEpsteinReaction<T>>{});
        case 7: return f(Tag<ThomasReaction<T>>{});
        case 8: return f(Tag<BarkleyReaction<T>>{});
    }
    return 0;
}

// Stream and convergence buffer for one solve() call.
struct LoopContext {
    cudaStream_t stream;
    double* d_conv = nullptr;
    double h_conv[2] = {0, 0};

    explicit LoopContext(double rtol) {
        cudaStreamCreate(&stream);
        if (rtol > 0) cudaMalloc(&d_conv, 16);
    }
    ~LoopContext() {
        if (d_conv) cudaFree(d_conv);
        cudaStreamDestroy(stream);
    }
};

#define LOOP_ENTRY_TAIL \
    int n_iters, int solver_type, int model_type, int dtype_code, \
    void** work,          /* solver-specific buffers (see above) */ \
    void* trj_buf,        /* NULL if no trajectory */ \
    const int* trj_iters, /* host array of 1-based iter numbers to capture */ \
    int n_trj, \
    double rtol,          /* 0 = disabled */ \
    int rtol_period,      /* check every N iters */ \
    int* stopped_at       /* output: iter where stopped, -1 if completed */

#define MAKE_LOOP_ARGS(ctx) LoopArgs{ \
    y_a, y_b, params, dt, n_iters, solver_type, work, \
    trj_buf, trj_iters, n_trj, rtol, rtol_period, stopped_at, \
    ctx.d_conv, ctx.h_conv, ctx.stream}

/* Full rectangular grid; the outermost cells are held fixed. */
extern "C" int lpf_native_solve(
    void* y_a, void* y_b, const void* params,
    int B, int H, int W, double dt, double dx2_inv,
    LOOP_ENTRY_TAIL)
{
    LoopContext ctx(rtol);
    const LoopArgs a = MAKE_LOOP_ARGS(ctx);
    return with_dtype(dtype_code, [&](auto dtag) {
        using T = typename decltype(dtag)::type;
        return with_reaction<T>(model_type, [&](auto rtag) {
            using R = typename decltype(rtag)::type;
            return run_solver<T>(a, GridOp<T, R, 32, 8>{B, H, W, dx2_inv});
        });
    });
}

/* Domain on the grid (lpf.domains): stencil is (5, H, W) per batch element
 * at a stride of stencil_stride elements (0: shared), see domain_core. */
extern "C" int lpf_native_solve_domain(
    void* y_a, void* y_b, const void* params,
    const void* stencil, int stencil_stride,
    int B, int H, int W, double dt, double dx2_inv,
    LOOP_ENTRY_TAIL)
{
    LoopContext ctx(rtol);
    const LoopArgs a = MAKE_LOOP_ARGS(ctx);
    return with_dtype(dtype_code, [&](auto dtag) {
        using T = typename decltype(dtag)::type;
        return with_reaction<T>(model_type, [&](auto rtag) {
            using R = typename decltype(rtag)::type;
            return run_solver<T>(a, DomainOp<T, R, 32, 8>{
                B, H, W, dx2_inv, (const T*)stencil, stencil_stride});
        });
    });
}

/* Triangle-mesh surface (lpf.surfaces): transposed ELL operator nbr/wts of
 * shape (K, N), see surface_core. The state is (2, B, N). */
extern "C" int lpf_native_solve_surface(
    void* y_a, void* y_b, const void* params,
    const int* nbr, const void* wts, int K, int B, int N, double dt,
    LOOP_ENTRY_TAIL)
{
    LoopContext ctx(rtol);
    const LoopArgs a = MAKE_LOOP_ARGS(ctx);
    return with_dtype(dtype_code, [&](auto dtag) {
        using T = typename decltype(dtag)::type;
        return with_reaction<T>(model_type, [&](auto rtag) {
            using R = typename decltype(rtag)::type;
            return run_solver<T>(a, SurfaceOp<T, R>{B, N, K, nbr, (const T*)wts});
        });
    });
}
