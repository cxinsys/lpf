/**
 * lpf native solve loops — entire solve() in C/CUDA, zero Python during iteration.
 *
 * Features supported:
 *   - All 5 solver types (Euler, Heun, RK4, RK23, AB2)
 *   - All 4 model types × 3 dtypes (f16/f32/f64)
 *   - Trajectory capture at arbitrary waypoints
 *   - Early stopping (convergence check every N iters)
 *   - Ping-pong buffers
 *
 * Compile:
 *   nvcc -shared -Xcompiler -fPIC -std=c++17 -O3 --use_fast_math \
 *        -I include/ solver_loop.cu -o libsolver.so
 */

#include "include/stencil_core.cuh"
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
// EULER solve loop
// ============================================================
template<typename T, typename Reaction, int TX, int TY>
static int solve_euler(
    T* y_a, T* y_b, const T* params,
    int B, int H, int W, double dt, double dx2_inv, int n_iters,
    T* trj_buf, const int* trj_iters, int n_trj,
    double rtol, int rtol_period, int* stopped_at,
    double* d_conv, double h_conv[2], cudaStream_t s)
{
    dim3 grid((W+TX-1)/TX, (H+TY-1)/TY, B);
    dim3 block(TX, TY);
    int N = 2*B*H*W, trj_idx = 0;

    for (int i = 0; i < n_iters; i++) {
        T *src = (i%2==0)?y_a:y_b, *dst = (i%2==0)?y_b:y_a;
        euler_kernel_tpl<T,Reaction,TX,TY><<<grid,block,0,s>>>(
            src, dst, params, B, H, W, dt, dx2_inv);

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);

        if (rtol > 0 && rtol_period > 0 && (i+1) % rtol_period == 0) {
            if (check_convergence(dst, src, N, rtol, d_conv, h_conv, s)) {
                *stopped_at = i+1; cudaStreamSynchronize(s); return (i+1)%2;
            }
        }
    }
    *stopped_at = -1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// HEUN solve loop
// ============================================================
template<typename T, typename Reaction, int TX, int TY>
static int solve_heun(
    T* y_a, T* y_b, const T* params,
    int B, int H, int W, double dt, double dx2_inv, int n_iters,
    T* k1, T* k2, T* y_temp,
    T* trj_buf, const int* trj_iters, int n_trj,
    double rtol, int rtol_period, int* stopped_at,
    double* d_conv, double h_conv[2], cudaStream_t s)
{
    dim3 sg((W+TX-1)/TX,(H+TY-1)/TY,B), sb(TX,TY);
    int N = 2*B*H*W, ub = 256, ug = (N+ub-1)/ub, trj_idx = 0;

    for (int i = 0; i < n_iters; i++) {
        T *src=(i%2==0)?y_a:y_b, *dst=(i%2==0)?y_b:y_a;
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(src,k1,params,B,H,W,dx2_inv);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k1,y_temp,dt,N);
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(y_temp,k2,params,B,H,W,dx2_inv);
        linear_combine2_tpl<T><<<ug,ub,0,s>>>(k1,k2,y_temp,0.5*dt,0.5*dt,N);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,y_temp,dst,1.0,N);

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        if (rtol>0 && rtol_period>0 && (i+1)%rtol_period==0)
            if (check_convergence(dst,src,N,rtol,d_conv,h_conv,s))
                { *stopped_at=i+1; cudaStreamSynchronize(s); return (i+1)%2; }
    }
    *stopped_at=-1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// RK4 solve loop
// ============================================================
template<typename T, typename Reaction, int TX, int TY>
static int solve_rk4(
    T* y_a, T* y_b, const T* params,
    int B, int H, int W, double dt, double dx2_inv, int n_iters,
    T* k1, T* k2, T* k3, T* k4, T* y_temp, T* delta,
    T* trj_buf, const int* trj_iters, int n_trj,
    double rtol, int rtol_period, int* stopped_at,
    double* d_conv, double h_conv[2], cudaStream_t s)
{
    dim3 sg((W+TX-1)/TX,(H+TY-1)/TY,B), sb(TX,TY);
    int N=2*B*H*W, ub=256, ug=(N+ub-1)/ub, trj_idx=0;

    for (int i = 0; i < n_iters; i++) {
        T *src=(i%2==0)?y_a:y_b, *dst=(i%2==0)?y_b:y_a;
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(src,k1,params,B,H,W,dx2_inv);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k1,y_temp,0.5*dt,N);
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(y_temp,k2,params,B,H,W,dx2_inv);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k2,y_temp,0.5*dt,N);
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(y_temp,k3,params,B,H,W,dx2_inv);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k3,y_temp,dt,N);
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(y_temp,k4,params,B,H,W,dx2_inv);
        rk4_combine_tpl<T><<<ug,ub,0,s>>>(delta,k1,k2,k3,k4,dt/6.0,N);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,delta,dst,1.0,N);

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        if (rtol>0 && rtol_period>0 && (i+1)%rtol_period==0)
            if (check_convergence(dst,src,N,rtol,d_conv,h_conv,s))
                { *stopped_at=i+1; cudaStreamSynchronize(s); return (i+1)%2; }
    }
    *stopped_at=-1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// RK23 solve loop
// ============================================================
template<typename T, typename Reaction, int TX, int TY>
static int solve_rk23(
    T* y_a, T* y_b, const T* params,
    int B, int H, int W, double dt, double dx2_inv, int n_iters,
    T* k1, T* k2, T* k3, T* y_temp, T* delta,
    T* trj_buf, const int* trj_iters, int n_trj,
    double rtol, int rtol_period, int* stopped_at,
    double* d_conv, double h_conv[2], cudaStream_t s)
{
    dim3 sg((W+TX-1)/TX,(H+TY-1)/TY,B), sb(TX,TY);
    int N=2*B*H*W, ub=256, ug=(N+ub-1)/ub, trj_idx=0;

    for (int i = 0; i < n_iters; i++) {
        T *src=(i%2==0)?y_a:y_b, *dst=(i%2==0)?y_b:y_a;
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(src,k1,params,B,H,W,dx2_inv);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k1,y_temp,0.5*dt,N);
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(y_temp,k2,params,B,H,W,dx2_inv);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,k2,y_temp,0.75*dt,N);
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(y_temp,k3,params,B,H,W,dx2_inv);
        linear_combine3_tpl<T><<<ug,ub,0,s>>>(k1,k2,k3,delta,
            dt*(2.0/9.0), dt*(1.0/3.0), dt*(4.0/9.0), N);
        rk_stage_tpl<T><<<ug,ub,0,s>>>(src,delta,dst,1.0,N);

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        if (rtol>0 && rtol_period>0 && (i+1)%rtol_period==0)
            if (check_convergence(dst,src,N,rtol,d_conv,h_conv,s))
                { *stopped_at=i+1; cudaStreamSynchronize(s); return (i+1)%2; }
    }
    *stopped_at=-1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// AB2 solve loop
// ============================================================
template<typename T, typename Reaction, int TX, int TY>
static int solve_ab2(
    T* y_a, T* y_b, const T* params,
    int B, int H, int W, double dt, double dx2_inv, int n_iters,
    T* dydt_a, T* dydt_b, T* delta,
    T* trj_buf, const int* trj_iters, int n_trj,
    double rtol, int rtol_period, int* stopped_at,
    double* d_conv, double h_conv[2], cudaStream_t s)
{
    dim3 sg((W+TX-1)/TX,(H+TY-1)/TY,B), sb(TX,TY);
    int N=2*B*H*W, ub=256, ug=(N+ub-1)/ub, trj_idx=0;
    T* dydt_cur = dydt_a;
    T* dydt_prev = dydt_b;

    for (int i = 0; i < n_iters; i++) {
        T *src=(i%2==0)?y_a:y_b, *dst=(i%2==0)?y_b:y_a;
        pdefunc_kernel_tpl<T,Reaction,TX,TY><<<sg,sb,0,s>>>(src,dydt_cur,params,B,H,W,dx2_inv);

        if (i == 0) {
            rk_stage_tpl<T><<<ug,ub,0,s>>>(src,dydt_cur,dst,dt,N);
        } else {
            linear_combine2_tpl<T><<<ug,ub,0,s>>>(dydt_cur,dydt_prev,delta,1.5*dt,-0.5*dt,N);
            rk_stage_tpl<T><<<ug,ub,0,s>>>(src,delta,dst,1.0,N);
        }
        T* tmp = dydt_cur; dydt_cur = dydt_prev; dydt_prev = tmp;

        try_capture_trj(trj_buf, trj_iters, n_trj, trj_idx, dst, N, i+1, s);
        if (rtol>0 && rtol_period>0 && (i+1)%rtol_period==0)
            if (check_convergence(dst,src,N,rtol,d_conv,h_conv,s))
                { *stopped_at=i+1; cudaStreamSynchronize(s); return (i+1)%2; }
    }
    *stopped_at=-1; cudaStreamSynchronize(s); return n_iters%2;
}

// ============================================================
// Dispatch macros — expand all (model × dtype) combinations
// ============================================================

// Solver type → enum
// 0=Euler, 1=Heun, 2=RK4, 3=RK23, 4=AB2

/*
 * The single extern "C" entry point. work[] provides solver-specific buffers:
 *   Euler: (none needed)
 *   Heun:  work[0]=k1, [1]=k2, [2]=y_temp
 *   RK4:   work[0]=k1, [1]=k2, [2]=k3, [3]=k4, [4]=y_temp, [5]=delta
 *   RK23:  work[0]=k1, [1]=k2, [2]=k3, [3]=y_temp, [4]=delta
 *   AB2:   work[0]=dydt_a, [1]=dydt_b, [2]=delta
 */

#define DISPATCH_CASE(SOLVER_CALL, MODEL, DTYPE, TX, TY) \
    SOLVER_CALL<DTYPE, MODEL##Reaction<DTYPE>, TX, TY>

#define DISPATCH_MODEL(SOLVER_CALL, model_t, DTYPE, TX, TY, ...) \
    switch (model_t) { \
        case 0: result = DISPATCH_CASE(SOLVER_CALL,GrayScott,DTYPE,TX,TY)(__VA_ARGS__); break; \
        case 1: result = DISPATCH_CASE(SOLVER_CALL,Liaw,DTYPE,TX,TY)(__VA_ARGS__); break; \
        case 2: result = DISPATCH_CASE(SOLVER_CALL,Schnakenberg,DTYPE,TX,TY)(__VA_ARGS__); break; \
        case 3: result = DISPATCH_CASE(SOLVER_CALL,GiererMeinhardt,DTYPE,TX,TY)(__VA_ARGS__); break; \
        case 4: result = DISPATCH_CASE(SOLVER_CALL,Brusselator,DTYPE,TX,TY)(__VA_ARGS__); break; \
        case 5: result = DISPATCH_CASE(SOLVER_CALL,FitzHughNagumo,DTYPE,TX,TY)(__VA_ARGS__); break; \
        case 6: result = DISPATCH_CASE(SOLVER_CALL,LengyelEpstein,DTYPE,TX,TY)(__VA_ARGS__); break; \
        case 7: result = DISPATCH_CASE(SOLVER_CALL,Thomas,DTYPE,TX,TY)(__VA_ARGS__); break; \
        case 8: result = DISPATCH_CASE(SOLVER_CALL,Barkley,DTYPE,TX,TY)(__VA_ARGS__); break; \
    }

extern "C" int lpf_native_solve(
    void* y_a, void* y_b, const void* params,
    int B, int H, int W, double dt, double dx2_inv,
    int n_iters,
    int solver_type, int model_type, int dtype_code,
    void** work,          /* solver-specific buffers (see above) */
    void* trj_buf,        /* NULL if no trajectory */
    const int* trj_iters, /* host array of 1-based iter numbers to capture */
    int n_trj,
    double rtol,          /* 0 = disabled */
    int rtol_period,      /* check every N iters */
    int* stopped_at       /* output: iter where stopped, -1 if completed */
)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    double *d_conv = nullptr, h_conv[2] = {0,0};
    if (rtol > 0) cudaMalloc(&d_conv, 16);

    int result = 0;

    #define BASE(D) (D*)y_a, (D*)y_b, (const D*)params, \
                    B, H, W, dt, dx2_inv, n_iters
    #define TAIL(D) (D*)trj_buf, trj_iters, n_trj, \
                    rtol, rtol_period, stopped_at, d_conv, h_conv, stream
    #define W0(D)   (D*)work[0]
    #define W1(D)   (D*)work[1]
    #define W2(D)   (D*)work[2]
    #define W3(D)   (D*)work[3]
    #define W4(D)   (D*)work[4]
    #define W5(D)   (D*)work[5]

    #define D_EULER(D) \
        DISPATCH_MODEL(solve_euler, model_type, D, 32, 8, BASE(D), TAIL(D))

    #define D_HEUN(D) \
        DISPATCH_MODEL(solve_heun, model_type, D, 32, 8, BASE(D), \
            W0(D), W1(D), W2(D), TAIL(D))

    #define D_RK4(D) \
        DISPATCH_MODEL(solve_rk4, model_type, D, 32, 8, BASE(D), \
            W0(D), W1(D), W2(D), W3(D), W4(D), W5(D), TAIL(D))

    #define D_RK23(D) \
        DISPATCH_MODEL(solve_rk23, model_type, D, 32, 8, BASE(D), \
            W0(D), W1(D), W2(D), W3(D), W4(D), TAIL(D))

    #define D_AB2(D) \
        DISPATCH_MODEL(solve_ab2, model_type, D, 32, 8, BASE(D), \
            W0(D), W1(D), W2(D), TAIL(D))

    #define DO_SOLVER(DTYPE) \
        switch (solver_type) { \
            case 0: D_EULER(DTYPE); break; \
            case 1: D_HEUN(DTYPE);  break; \
            case 2: D_RK4(DTYPE);   break; \
            case 3: D_RK23(DTYPE);  break; \
            case 4: D_AB2(DTYPE);   break; \
        }

    switch (dtype_code) {
        case 0: DO_SOLVER(__half);  break;
        case 1: DO_SOLVER(float);   break;
        case 2: DO_SOLVER(double);  break;
    }

    #undef DO_SOLVER
    #undef D_AB2
    #undef D_RK23
    #undef D_RK4
    #undef D_HEUN
    #undef D_EULER
    #undef W5
    #undef W4
    #undef W3
    #undef W2
    #undef W1
    #undef W0
    #undef TAIL
    #undef BASE

    if (d_conv) cudaFree(d_conv);
    cudaStreamDestroy(stream);
    return result;
}
