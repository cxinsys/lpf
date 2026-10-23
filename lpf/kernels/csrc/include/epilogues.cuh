#pragma once
#include "types.cuh"

/**
 * Epilogues: what a fused kernel writes once a space (spaces.cuh) has
 * evaluated du = dy/dt for a point.
 *
 *   active(i, off, u, v, du, dv)   a solved point
 *   passive(i, off, u, v)          a point that is not solved (dy/dt = 0)
 *
 * i indexes the u component and off is the size of one component, so the
 * v component of the point is at off + i.
 *
 * The multi-stage epilogues fuse a dy/dt evaluation with the elementwise
 * update that follows it in an explicit Runge-Kutta or Adams-Bashforth step,
 * so a step reads and writes the state far less often and launches fewer
 * kernels. Each one evaluates exactly the expressions of the unfused
 * sequence (dy/dt kernel, then the update kernel), in the same order and
 * with the same coefficient conversions, so fused and unfused steps agree
 * bit for bit (with --fmad=false); passive points have dy/dt = 0.
 * Coefficients are passed as double and converted on the device, like the
 * unfused kernels did.
 */

// y_out = y + dt * dy/dt; passive points are copied.
template<typename T>
struct EulerEpi {
    T* out;
    double dt;

    __device__ __forceinline__ void active(int i, int off, T u, T v, T du, T dv) const {
        const T dt = make_real<T>(this->dt);
        out[i]       = u + dt * du;
        out[off + i] = v + dt * dv;
    }
    __device__ __forceinline__ void passive(int i, int off, T u, T v) const {
        out[i] = u;  out[off + i] = v;
    }
};

// dydt = dy/dt.
template<typename T>
struct RhsEpi {
    T* dydt;

    __device__ __forceinline__ void active(int i, int off, T u, T v, T du, T dv) const {
        dydt[i] = du;  dydt[off + i] = dv;
    }
    __device__ __forceinline__ void passive(int i, int off, T, T) const {
        dydt[i] = make_real<T>(0.0);  dydt[off + i] = make_real<T>(0.0);
    }
};

// Base value of a point: the input state itself (base_is_y) or another array.
template<typename T>
__device__ __forceinline__ T base_of(const T* __restrict__ base, bool base_is_y,
                                     int j, T y) {
    return base_is_y ? y : base[j];
}

// Runge-Kutta stage: k = dy/dt, y_next = base + alpha * k.
template<typename T>
struct StageEpi {
    T* k;
    T* y_next;
    const T* base;
    bool base_is_y;
    double alpha;

    __device__ __forceinline__ void active(int i, int off, T u, T v, T du, T dv) const {
        const T alpha = make_real<T>(this->alpha);
        k[i] = du;  k[off + i] = dv;
        y_next[i]       = base_of(base, base_is_y, i, u)       + alpha * du;
        y_next[off + i] = base_of(base, base_is_y, off + i, v) + alpha * dv;
    }
    __device__ __forceinline__ void passive(int i, int off, T u, T v) const {
        active(i, off, u, v, make_real<T>(0.0), make_real<T>(0.0));
    }
};

// Last Heun stage: y_next = base + 1 * (a * k1 + b * dy/dt).
template<typename T>
struct HeunEpi {
    T* y_next;
    const T* base;
    const T* k1;
    double a, b;

    __device__ __forceinline__ T step(int j, T d) const {
        const T a = make_real<T>(this->a), b = make_real<T>(this->b);
        const T delta = a * k1[j] + b * d;
        return base[j] + make_real<T>(1.0) * delta;
    }
    __device__ __forceinline__ void active(int i, int off, T, T, T du, T dv) const {
        y_next[i] = step(i, du);  y_next[off + i] = step(off + i, dv);
    }
    __device__ __forceinline__ void passive(int i, int off, T u, T v) const {
        active(i, off, u, v, make_real<T>(0.0), make_real<T>(0.0));
    }
};

// Last RK23 stage: y_next = base + 1 * (a * k1 + b * k2 + c * dy/dt).
template<typename T>
struct RK23Epi {
    T* y_next;
    const T* base;
    const T* k1;
    const T* k2;
    double a, b, c;

    __device__ __forceinline__ T step(int j, T d) const {
        const T a = make_real<T>(this->a), b = make_real<T>(this->b);
        const T c = make_real<T>(this->c);
        const T delta = a * k1[j] + b * k2[j] + c * d;
        return base[j] + make_real<T>(1.0) * delta;
    }
    __device__ __forceinline__ void active(int i, int off, T, T, T du, T dv) const {
        y_next[i] = step(i, du);  y_next[off + i] = step(off + i, dv);
    }
    __device__ __forceinline__ void passive(int i, int off, T u, T v) const {
        active(i, off, u, v, make_real<T>(0.0), make_real<T>(0.0));
    }
};

// Last RK4 stage: y_next = base + 1 * (d * (k1 + 2 k2 + 2 k3 + dy/dt)).
template<typename T>
struct RK4Epi {
    T* y_next;
    const T* base;
    const T* k1;
    const T* k2;
    const T* k3;
    double d;

    __device__ __forceinline__ T step(int j, T k4) const {
        const T d = make_real<T>(this->d), two = make_real<T>(2.0);
        const T delta = d * (k1[j] + two * k2[j] + two * k3[j] + k4);
        return base[j] + make_real<T>(1.0) * delta;
    }
    __device__ __forceinline__ void active(int i, int off, T, T, T du, T dv) const {
        y_next[i] = step(i, du);  y_next[off + i] = step(off + i, dv);
    }
    __device__ __forceinline__ void passive(int i, int off, T u, T v) const {
        active(i, off, u, v, make_real<T>(0.0), make_real<T>(0.0));
    }
};

// Adams-Bashforth 2 (after the first step): cur = dy/dt,
// y_next = y + 1 * (a * cur + b * prev).
template<typename T>
struct AB2Epi {
    T* cur;
    T* y_next;
    const T* prev;
    double a, b;

    __device__ __forceinline__ void active(int i, int off, T u, T v, T du, T dv) const {
        const T a = make_real<T>(this->a), b = make_real<T>(this->b);
        cur[i] = du;  cur[off + i] = dv;
        y_next[i]       = u + make_real<T>(1.0) * (a * du + b * prev[i]);
        y_next[off + i] = v + make_real<T>(1.0) * (a * dv + b * prev[off + i]);
    }
    __device__ __forceinline__ void passive(int i, int off, T u, T v) const {
        active(i, off, u, v, make_real<T>(0.0), make_real<T>(0.0));
    }
};
