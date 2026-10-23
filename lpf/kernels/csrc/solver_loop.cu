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
 * Every launch is a fused kernel (rd_kernels.cuh): a space evaluates dy/dt
 * and an epilogue applies the solver's update in the same pass, so e.g. an
 * RK4 step is 4 kernels instead of 9 and reads/writes the state ~1.5x less.
 * The results are bit-identical to the unfused sequences.
 *
 * Compile:
 *   nvcc -shared -Xcompiler -fPIC -std=c++17 -O3 --fmad=false \
 *        -I include/ solver_loop.cu -o libsolver.so
 *
 * --fmad=false (and no --use_fast_math) is required for bit-exact
 * reproduction of the legacy PyTorch-eager dataset; see kernels.cu for
 * the full rationale and benchmark numbers.
 */

#include "include/rd_kernels.cuh"
#include <algorithm>
#include <utility>

// ============================================================
// Convergence check (single-block reduction)
// ============================================================
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
// Operator: one space of one model with its launch shape
// ============================================================
// op(y, params, epi, s) launches the fused kernel that evaluates dy/dt at y
// and applies epi. size is the number of elements of the state (2 * B * ...).
template<typename T, typename R, typename Space>
struct Operator {
    Space space;
    dim3 grid, block;
    int size;

    template<typename Epi>
    void operator()(const T* y, const T* params, const Epi& epi,
                    cudaStream_t s) const {
        rd_kernel<T, R, Space, Epi><<<grid, block, 0, s>>>(space, y, params, epi);
    }
};

// ============================================================
// CUDA graphs of consecutive steps
// ============================================================
// Small problems are bound by the launch rate. A step's launches depend on
// the step only through its parity (the ping-pong buffers), so STEPS
// consecutive steps (an even number) are captured once per parity and
// replayed. If graphs are unavailable the steps are launched directly.
struct StepGraphs {
    static constexpr int STEPS = 64;
    cudaGraphExec_t exec[2] = {nullptr, nullptr};
    bool off = false;

    ~StepGraphs() {
        for (cudaGraphExec_t e : exec) if (e) cudaGraphExecDestroy(e);
    }

    // Runs steps [i, end); returns end.
    template<typename Step>
    int run(int i, int end, Step& step, cudaStream_t s) {
        while (!off && end - i >= STEPS) {
            const int p = i % 2;
            if (!exec[p] && !capture(p, i, step, s)) break;
            cudaGraphLaunch(exec[p], s);
            i += STEPS;
        }
        for (; i < end; ++i) step(i);
        return end;
    }

    template<typename Step>
    bool capture(int p, int i, Step& step, cudaStream_t s) {
        cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            cudaGetLastError();
            off = true;
            return false;
        }
        for (int k = 0; k < STEPS; ++k) step(i + k);  // recorded, not run
        cudaError_t err = cudaStreamEndCapture(s, &g);
        if (err == cudaSuccess) err = cudaGraphInstantiate(&exec[p], g, 0);
        if (g) cudaGraphDestroy(g);
        if (err != cudaSuccess) {
            cudaGetLastError();
            exec[p] = nullptr;
            off = true;
            return false;
        }
        return true;
    }
};

// ============================================================
// Loop: trajectory capture and early stopping around the steps
// ============================================================
// run(step) performs step(0) .. step(n_iters - 1), where step i reads y_a
// (even i) or y_b (odd i) and writes the other. After step i, the state is
// copied to the trajectory if i + 1 is the next waypoint, and every
// rtol_period steps the relative change is checked against rtol. From step
// first_graph on, step(i) may depend on i only through i % 2.
// Returns the buffer holding the final state (0: y_a, 1: y_b).
template<typename T>
struct Loop {
    T* y_a; T* y_b; int N; int n_iters;
    T* trj_buf; const int* trj_iters; int n_trj;
    double rtol; int rtol_period; int* stopped_at;
    double* d_conv; double* h_conv; cudaStream_t s;

    template<typename Step>
    int run(Step&& step, int first_graph = 0) const {
        StepGraphs graphs;
        const bool check = rtol > 0 && rtol_period > 0;
        int trj_idx = 0, i = 0;
        while (i < n_iters) {
            while (trj_buf && trj_idx < n_trj && trj_iters[trj_idx] <= i) ++trj_idx;
            int end = n_iters;
            if (trj_buf && trj_idx < n_trj) end = std::min(end, trj_iters[trj_idx]);
            if (check) end = std::min(end, (i / rtol_period + 1) * rtol_period);

            for (; i < end && i < first_graph; ++i) step(i);
            i = graphs.run(i, end, step, s);

            // After step end - 1:
            T* dst = (end % 2 == 1) ? y_b : y_a;
            T* src = (end % 2 == 1) ? y_a : y_b;
            if (trj_buf && trj_idx < n_trj && trj_iters[trj_idx] == end) {
                cudaMemcpyAsync(trj_buf + (long long)trj_idx * N, dst,
                                (size_t)N * sizeof(T), cudaMemcpyDeviceToDevice, s);
                ++trj_idx;
            }
            if (check && end % rtol_period == 0
                    && check_convergence(dst, src, N, rtol, d_conv, h_conv, s)) {
                *stopped_at = end;
                cudaStreamSynchronize(s);
                return end % 2;
            }
        }
        *stopped_at = -1;
        cudaStreamSynchronize(s);
        return n_iters % 2;
    }
};

#define SRC_DST(i) T* src = ((i) % 2 == 0) ? L.y_a : L.y_b; \
                   T* dst = ((i) % 2 == 0) ? L.y_b : L.y_a

// ============================================================
// EULER: y1 = y + dt f(y)
// ============================================================
template<typename T, typename Op>
static int solve_euler(const Loop<T>& L, const T* p, const Op& op, double dt)
{
    return L.run([&](int i) {
        SRC_DST(i);
        op(src, p, EulerEpi<T>{dst, dt}, L.s);
    });
}

// ============================================================
// HEUN: k1 = f(y), y1 = y + dt/2 (k1 + f(y + dt k1))
// ============================================================
template<typename T, typename Op>
static int solve_heun(const Loop<T>& L, const T* p, const Op& op, double dt,
                      T* k1, T* y_temp)
{
    return L.run([&](int i) {
        SRC_DST(i);
        op(src, p, StageEpi<T>{k1, y_temp, src, true, dt}, L.s);
        op(y_temp, p, HeunEpi<T>{dst, src, k1, 0.5*dt, 0.5*dt}, L.s);
    });
}

// ============================================================
// RK4 (the stage inputs alternate between two buffers)
// ============================================================
template<typename T, typename Op>
static int solve_rk4(const Loop<T>& L, const T* p, const Op& op, double dt,
                     T* k1, T* k2, T* k3, T* y_t1, T* y_t2)
{
    return L.run([&](int i) {
        SRC_DST(i);
        op(src,  p, StageEpi<T>{k1, y_t1, src, true,  0.5*dt}, L.s);
        op(y_t1, p, StageEpi<T>{k2, y_t2, src, false, 0.5*dt}, L.s);
        op(y_t2, p, StageEpi<T>{k3, y_t1, src, false, dt}, L.s);
        op(y_t1, p, RK4Epi<T>{dst, src, k1, k2, k3, dt/6.0}, L.s);
    });
}

// ============================================================
// RK23 (Bogacki-Shampine, third-order solution)
// ============================================================
template<typename T, typename Op>
static int solve_rk23(const Loop<T>& L, const T* p, const Op& op, double dt,
                      T* k1, T* k2, T* y_t1, T* y_t2)
{
    return L.run([&](int i) {
        SRC_DST(i);
        op(src,  p, StageEpi<T>{k1, y_t1, src, true,  0.5*dt}, L.s);
        op(y_t1, p, StageEpi<T>{k2, y_t2, src, false, 0.75*dt}, L.s);
        op(y_t2, p, RK23Epi<T>{dst, src, k1, k2,
                               dt*(2.0/9.0), dt*(1.0/3.0), dt*(4.0/9.0)}, L.s);
    });
}

// ============================================================
// AB2 (the first step is an Euler step; f(y) alternates between buffers)
// ============================================================
template<typename T, typename Op>
static int solve_ab2(const Loop<T>& L, const T* p, const Op& op, double dt,
                     T* dydt_a, T* dydt_b)
{
    T* dydt_cur = dydt_a;
    T* dydt_prev = dydt_b;
    return L.run([&](int i) {
        SRC_DST(i);
        if (i == 0)
            op(src, p, StageEpi<T>{dydt_cur, dst, src, true, dt}, L.s);
        else
            op(src, p, AB2Epi<T>{dydt_cur, dst, dydt_prev, 1.5*dt, -0.5*dt}, L.s);
        std::swap(dydt_cur, dydt_prev);
    }, /*first_graph=*/1);
}

#undef SRC_DST

// ============================================================
// Dispatch
// ============================================================

/*
 * work[] provides solver-specific buffers, each the size of the state:
 *   Euler: (none needed)
 *   Heun:  work[0]=k1, [1]=y_temp
 *   RK4:   work[0]=k1, [1]=k2, [2]=k3, [3]=y_t1, [4]=y_t2
 *   RK23:  work[0]=k1, [1]=k2, [2]=y_t1, [3]=y_t2
 *   AB2:   work[0]=dydt_a, [1]=dydt_b
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
    const Loop<T> L{(T*)a.y_a, (T*)a.y_b, op.size, a.n_iters,
                    (T*)a.trj_buf, a.trj_iters, a.n_trj,
                    a.rtol, a.rtol_period, a.stopped_at,
                    a.d_conv, a.h_conv, a.stream};
    const T* p = (const T*)a.params;
    #define WK(k) ((T*)a.work[k])
    switch (a.solver_type) {
        case 0: return solve_euler<T>(L, p, op, a.dt);
        case 1: return solve_heun<T>(L, p, op, a.dt, WK(0), WK(1));
        case 2: return solve_rk4<T>(L, p, op, a.dt, WK(0), WK(1), WK(2), WK(3), WK(4));
        case 3: return solve_rk23<T>(L, p, op, a.dt, WK(0), WK(1), WK(2), WK(3));
        case 4: return solve_ab2<T>(L, p, op, a.dt, WK(0), WK(1));
    }
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

// Runs the solver for the dtype and model codes with the operator that
// make_op<T, R>() builds.
#define DISPATCH(ctx, make_op) \
    with_dtype(dtype_code, [&](auto dtag) { \
        using T = typename decltype(dtag)::type; \
        return with_reaction<T>(model_type, [&](auto rtag) { \
            using R = typename decltype(rtag)::type; \
            return run_solver<T>(MAKE_LOOP_ARGS(ctx), make_op); \
        }); \
    })

static dim3 tiled_grid(int B, int H, int W) {
    return dim3((W + 31) / 32, (H + 7) / 8, B);
}

/* Full rectangular grid; the outermost cells are held fixed. */
extern "C" int lpf_native_solve(
    void* y_a, void* y_b, const void* params,
    int B, int H, int W, double dt, double dx2_inv,
    LOOP_ENTRY_TAIL)
{
    LoopContext ctx(rtol);
    return DISPATCH(ctx, (Operator<T, R, GridSpace<T>>{
        GridSpace<T>{B, H, W, dx2_inv},
        tiled_grid(B, H, W), dim3(32, 8), 2 * B * H * W}));
}

/* Domain on the grid (lpf.domains): stencil is (5, H, W) per batch element
 * at a stride of stencil_stride elements (0: shared), flags the (H, W) cell
 * classes at the same sharing, see DomainSpace. */
extern "C" int lpf_native_solve_domain(
    void* y_a, void* y_b, const void* params,
    const void* stencil, int stencil_stride, const unsigned char* flags,
    int B, int H, int W, double dt, double dx2_inv,
    LOOP_ENTRY_TAIL)
{
    LoopContext ctx(rtol);
    return DISPATCH(ctx, (Operator<T, R, DomainSpace<T>>{
        DomainSpace<T>{B, H, W, dx2_inv, (const T*)stencil, stencil_stride, flags},
        tiled_grid(B, H, W), dim3(32, 8), 2 * B * H * W}));
}

/* Triangle-mesh surface (lpf.surfaces): transposed ELL operator nbr/wts of
 * shape (K, N), see MeshSpace. The state is (2, B, N). A batch of several
 * models is processed four per thread. */
extern "C" int lpf_native_solve_surface(
    void* y_a, void* y_b, const void* params,
    const int* nbr, const void* wts, int K, int B, int N, double dt,
    LOOP_ENTRY_TAIL)
{
    LoopContext ctx(rtol);
    const dim3 block(256), grid1((N + 255) / 256, B), grid4((N + 255) / 256, (B + 3) / 4);
    if (B == 1)
        return DISPATCH(ctx, (Operator<T, R, MeshSpace<T>>{
            MeshSpace<T>{B, N, K, nbr, (const T*)wts}, grid1, block, 2 * B * N}));
    return DISPATCH(ctx, (Operator<T, R, MeshSpace<T, 256, 4>>{
        MeshSpace<T, 256, 4>{B, N, K, nbr, (const T*)wts}, grid4, block, 2 * B * N}));
}
