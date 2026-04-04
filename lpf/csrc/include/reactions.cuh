#pragma once
#include "types.cuh"

// ============================================================
// Reaction-kinetics structs (templated on scalar type T).
// Each struct provides:
//   N_PARAMS  - number of kinetic parameters per batch element
//   compute() - evaluate f(u,v) and g(u,v) reaction terms
// ============================================================

template<typename T>
struct GrayScottReaction {
    static constexpr int N_PARAMS = 4;

    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g
    ) {
        const T F_p = ldg(&params[base + 2]);
        const T k_p = ldg(&params[base + 3]);
        const T u_vsq = u * v * v;
        f = -u_vsq + F_p * (make_real<T>(1.0) - u);
        g =  u_vsq - (F_p + k_p) * v;
    }
};

template<typename T>
struct LiawReaction {
    static constexpr int N_PARAMS = 8;

    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g
    ) {
        const T ru = ldg(&params[base + 2]);
        const T rv = ldg(&params[base + 3]);
        const T k  = ldg(&params[base + 4]);
        const T su = ldg(&params[base + 5]);
        const T sv = ldg(&params[base + 6]);
        const T mu = ldg(&params[base + 7]);
        const T u_sq = u * u;
        const T u2v  = u_sq * v / (make_real<T>(1.0) + k * u_sq);
        f =  ru * u2v + su - mu * u;
        g = -rv * u2v + sv;
    }
};

template<typename T>
struct SchnakenbergReaction {
    static constexpr int N_PARAMS = 6;

    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g
    ) {
        const T rho = ldg(&params[base + 2]);
        const T su  = ldg(&params[base + 3]);
        const T sv  = ldg(&params[base + 4]);
        const T mu  = ldg(&params[base + 5]);
        const T usq_v = u * u * v;
        f = su - mu * u + rho * usq_v;
        g = sv - rho * usq_v;
    }
};

template<typename T>
struct GiererMeinhardtReaction {
    static constexpr int N_PARAMS = 6;

    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g
    ) {
        const T ru = ldg(&params[base + 2]);
        const T rv = ldg(&params[base + 3]);
        const T mu = ldg(&params[base + 4]);
        const T nu = ldg(&params[base + 5]);
        const T usq = u * u;
        f = ru * usq / v - mu * u;
        g = rv * usq - nu * v;
    }
};

template<typename T>
struct BrusselatorReaction {
    static constexpr int N_PARAMS = 4;
    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g) {
        const T A = ldg(&params[base + 2]), B = ldg(&params[base + 3]);
        const T usq_v = u * u * v;
        f = A - (B + make_real<T>(1.0)) * u + usq_v;
        g = B * u - usq_v;
    }
};

template<typename T>
struct FitzHughNagumoReaction {
    static constexpr int N_PARAMS = 5;
    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g) {
        const T eps = ldg(&params[base+2]), gamma = ldg(&params[base+3]), beta = ldg(&params[base+4]);
        f = u - u*u*u - v;
        g = eps * (u - gamma*v + beta);
    }
};

template<typename T>
struct LengyelEpsteinReaction {
    static constexpr int N_PARAMS = 4;
    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g) {
        const T a = ldg(&params[base+2]), b = ldg(&params[base+3]);
        const T uv_sat = u * v / (make_real<T>(1.0) + u*u);
        f = a - u - make_real<T>(4.0) * uv_sat;
        g = b * (u - uv_sat);
    }
};

template<typename T>
struct ThomasReaction {
    static constexpr int N_PARAMS = 7;
    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g) {
        const T a=ldg(&params[base+2]), b=ldg(&params[base+3]);
        const T rho=ldg(&params[base+4]), alpha=ldg(&params[base+5]), K=ldg(&params[base+6]);
        const T uv_term = rho*u*v / (make_real<T>(1.0) + u + K*u*u);
        f = a - u - uv_term;
        g = alpha*(b - v) - uv_term;
    }
};

template<typename T>
struct BarkleyReaction {
    static constexpr int N_PARAMS = 5;
    __device__ __forceinline__ static void compute(
        T u, T v, const T* __restrict__ params, int base, T& f, T& g) {
        const T eps=ldg(&params[base+2]), a=ldg(&params[base+3]), b=ldg(&params[base+4]);
        f = (make_real<T>(1.0)/eps) * u * (make_real<T>(1.0)-u) * (u - (v+b)/a);
        g = u - v;
    }
};
