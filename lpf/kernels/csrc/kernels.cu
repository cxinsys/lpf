/**
 * lpf AOT CUDA kernels — shared-memory tiled stencil + vectorised RK utils.
 *
 * All extern "C" entry points accept scalar args as double and convert
 * internally, so the Python caller always passes np.float64 for scalars.
 *
 * Compile:
 *   nvcc --fatbin -std=c++17 -O3 --use_fast_math \
 *        -I include/ kernels.cu -o kernels.fatbin
 */

#include "include/stencil_core.cuh"

// kernels.cu now uses the shared stencil_core from the header.
// The following dummy reference prevents "unused" warnings.
// (stencil_core is used via the INST_EULER / INST_PDEFUNC macros below.)

// ============================================================
// Instantiation macros  (scalar params always double)
// ============================================================

#define INST_EULER(MODEL, DTYPE, SUFFIX, TX, TY) \
extern "C" __global__ void euler_##MODEL##_##SUFFIX( \
    const DTYPE* __restrict__ y_in, DTYPE* __restrict__ y_out, \
    const DTYPE* __restrict__ params, \
    int B, int H, int W, double dt_d, double dx2_inv_d \
) { \
    stencil_core<DTYPE, MODEL##Reaction<DTYPE>, TX, TY, EULER_STEP>( \
        y_in, y_out, params, B, H, W, \
        make_real<DTYPE>(dt_d), make_real<DTYPE>(dx2_inv_d)); \
}

#define INST_PDEFUNC(MODEL, DTYPE, SUFFIX, TX, TY) \
extern "C" __global__ void pdefunc_##MODEL##_##SUFFIX( \
    const DTYPE* __restrict__ y_in, DTYPE* __restrict__ dydt, \
    const DTYPE* __restrict__ params, \
    int B, int H, int W, double dx2_inv_d \
) { \
    stencil_core<DTYPE, MODEL##Reaction<DTYPE>, TX, TY, PDEFUNC>( \
        y_in, dydt, params, B, H, W, \
        make_real<DTYPE>(0.0), make_real<DTYPE>(dx2_inv_d)); \
}

#define INST_ALL_MODELS(DTYPE, SUFFIX, TX, TY) \
    INST_EULER(GrayScott,        DTYPE, SUFFIX, TX, TY) \
    INST_EULER(Liaw,             DTYPE, SUFFIX, TX, TY) \
    INST_EULER(Schnakenberg,     DTYPE, SUFFIX, TX, TY) \
    INST_EULER(GiererMeinhardt,  DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(GrayScott,      DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(Liaw,           DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(Schnakenberg,   DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(GiererMeinhardt,DTYPE, SUFFIX, TX, TY) \
    INST_EULER(Brusselator,      DTYPE, SUFFIX, TX, TY) \
    INST_EULER(FitzHughNagumo,   DTYPE, SUFFIX, TX, TY) \
    INST_EULER(LengyelEpstein,   DTYPE, SUFFIX, TX, TY) \
    INST_EULER(Thomas,           DTYPE, SUFFIX, TX, TY) \
    INST_EULER(Barkley,          DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(Brusselator,    DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(FitzHughNagumo, DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(LengyelEpstein, DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(Thomas,         DTYPE, SUFFIX, TX, TY) \
    INST_PDEFUNC(Barkley,        DTYPE, SUFFIX, TX, TY)

// float16 / float32 / float64
INST_ALL_MODELS(__half,  f16, 32, 8)
INST_ALL_MODELS(float,   f32, 32, 8)
INST_ALL_MODELS(double,  f64, 32, 8)


// ============================================================
// Vectorised RK utility kernels
// ============================================================

// ---- rk_stage : y_out = y + alpha * k ----

extern "C" __global__ void rk_stage_f32(
    const float* __restrict__ y, const float* __restrict__ k,
    float* __restrict__ y_out, double alpha_d, int N)
{
    const float alpha = (float)alpha_d;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 yv = *reinterpret_cast<const float4*>(&y[i]);
        float4 kv = *reinterpret_cast<const float4*>(&k[i]);
        float4 o;
        o.x = fmaf(alpha, kv.x, yv.x);
        o.y = fmaf(alpha, kv.y, yv.y);
        o.z = fmaf(alpha, kv.z, yv.z);
        o.w = fmaf(alpha, kv.w, yv.w);
        *reinterpret_cast<float4*>(&y_out[i]) = o;
    } else {
        for (int j = i; j < min(i + 4, N); ++j)
            y_out[j] = fmaf(alpha, k[j], y[j]);
    }
}

extern "C" __global__ void rk_stage_f64(
    const double* __restrict__ y, const double* __restrict__ k,
    double* __restrict__ y_out, double alpha, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        double2 yv = *reinterpret_cast<const double2*>(&y[i]);
        double2 kv = *reinterpret_cast<const double2*>(&k[i]);
        double2 o;
        o.x = fma(alpha, kv.x, yv.x);
        o.y = fma(alpha, kv.y, yv.y);
        *reinterpret_cast<double2*>(&y_out[i]) = o;
    } else if (i < N) {
        y_out[i] = fma(alpha, k[i], y[i]);
    }
}

extern "C" __global__ void rk_stage_f16(
    const __half* __restrict__ y, const __half* __restrict__ k,
    __half* __restrict__ y_out, double alpha_d, int N)
{
    const __half2 a2 = __float2half2_rn((float)alpha_d);
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        __half2 yv = *reinterpret_cast<const __half2*>(&y[i]);
        __half2 kv = *reinterpret_cast<const __half2*>(&k[i]);
        *reinterpret_cast<__half2*>(&y_out[i]) = __hadd2(yv, __hmul2(a2, kv));
    } else if (i < N) {
        __half a = __double2half(alpha_d);
        y_out[i] = __hadd(y[i], __hmul(a, k[i]));
    }
}

// ---- rk4_combine : delta = (dt/6)*(k1 + 2*k2 + 2*k3 + k4) ----

extern "C" __global__ void rk4_combine_f32(
    float* __restrict__ delta,
    const float* __restrict__ k1, const float* __restrict__ k2,
    const float* __restrict__ k3, const float* __restrict__ k4,
    double dt6_d, int N)
{
    const float d = (float)dt6_d;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 a = *reinterpret_cast<const float4*>(&k1[i]);
        float4 b = *reinterpret_cast<const float4*>(&k2[i]);
        float4 c = *reinterpret_cast<const float4*>(&k3[i]);
        float4 e = *reinterpret_cast<const float4*>(&k4[i]);
        float4 o;
        o.x = d * (a.x + 2.f*b.x + 2.f*c.x + e.x);
        o.y = d * (a.y + 2.f*b.y + 2.f*c.y + e.y);
        o.z = d * (a.z + 2.f*b.z + 2.f*c.z + e.z);
        o.w = d * (a.w + 2.f*b.w + 2.f*c.w + e.w);
        *reinterpret_cast<float4*>(&delta[i]) = o;
    } else {
        for (int j = i; j < min(i + 4, N); ++j)
            delta[j] = d * (k1[j] + 2.f*k2[j] + 2.f*k3[j] + k4[j]);
    }
}

extern "C" __global__ void rk4_combine_f64(
    double* __restrict__ delta,
    const double* __restrict__ k1, const double* __restrict__ k2,
    const double* __restrict__ k3, const double* __restrict__ k4,
    double dt6, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        double2 a = *reinterpret_cast<const double2*>(&k1[i]);
        double2 b = *reinterpret_cast<const double2*>(&k2[i]);
        double2 c = *reinterpret_cast<const double2*>(&k3[i]);
        double2 e = *reinterpret_cast<const double2*>(&k4[i]);
        double2 o;
        o.x = dt6 * (a.x + 2.*b.x + 2.*c.x + e.x);
        o.y = dt6 * (a.y + 2.*b.y + 2.*c.y + e.y);
        *reinterpret_cast<double2*>(&delta[i]) = o;
    } else if (i < N) {
        delta[i] = dt6 * (k1[i] + 2.*k2[i] + 2.*k3[i] + k4[i]);
    }
}

extern "C" __global__ void rk4_combine_f16(
    __half* __restrict__ delta,
    const __half* __restrict__ k1, const __half* __restrict__ k2,
    const __half* __restrict__ k3, const __half* __restrict__ k4,
    double dt6_d, int N)
{
    const __half2 d2  = __float2half2_rn((float)dt6_d);
    const __half2 tw  = __float2half2_rn(2.0f);
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        __half2 a = *reinterpret_cast<const __half2*>(&k1[i]);
        __half2 b = *reinterpret_cast<const __half2*>(&k2[i]);
        __half2 c = *reinterpret_cast<const __half2*>(&k3[i]);
        __half2 e = *reinterpret_cast<const __half2*>(&k4[i]);
        __half2 s = __hadd2(a, __hadd2(__hmul2(tw, b),
                   __hadd2(__hmul2(tw, c), e)));
        *reinterpret_cast<__half2*>(&delta[i]) = __hmul2(d2, s);
    } else if (i < N) {
        __half d = __double2half(dt6_d);
        __half t = make_real<__half>(2.0);
        delta[i] = d * (k1[i] + t * k2[i] + t * k3[i] + k4[i]);
    }
}

// ---- heun_combine : delta = 0.5 * (k1 + k2) ----

extern "C" __global__ void heun_combine_f32(
    float* __restrict__ delta,
    const float* __restrict__ k1, const float* __restrict__ k2,
    int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 a = *reinterpret_cast<const float4*>(&k1[i]);
        float4 b = *reinterpret_cast<const float4*>(&k2[i]);
        float4 o;
        o.x = 0.5f * (a.x + b.x);
        o.y = 0.5f * (a.y + b.y);
        o.z = 0.5f * (a.z + b.z);
        o.w = 0.5f * (a.w + b.w);
        *reinterpret_cast<float4*>(&delta[i]) = o;
    } else {
        for (int j = i; j < min(i + 4, N); ++j)
            delta[j] = 0.5f * (k1[j] + k2[j]);
    }
}

extern "C" __global__ void heun_combine_f64(
    double* __restrict__ delta,
    const double* __restrict__ k1, const double* __restrict__ k2,
    int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        double2 a = *reinterpret_cast<const double2*>(&k1[i]);
        double2 b = *reinterpret_cast<const double2*>(&k2[i]);
        double2 o;
        o.x = 0.5 * (a.x + b.x);
        o.y = 0.5 * (a.y + b.y);
        *reinterpret_cast<double2*>(&delta[i]) = o;
    } else if (i < N) {
        delta[i] = 0.5 * (k1[i] + k2[i]);
    }
}

extern "C" __global__ void heun_combine_f16(
    __half* __restrict__ delta,
    const __half* __restrict__ k1, const __half* __restrict__ k2,
    int N)
{
    const __half2 half2 = __float2half2_rn(0.5f);
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        __half2 a = *reinterpret_cast<const __half2*>(&k1[i]);
        __half2 b = *reinterpret_cast<const __half2*>(&k2[i]);
        *reinterpret_cast<__half2*>(&delta[i]) = __hmul2(half2, __hadd2(a, b));
    } else if (i < N) {
        delta[i] = __hmul(make_real<__half>(0.5), __hadd(k1[i], k2[i]));
    }
}


// ---- scale : out = alpha * x ----

extern "C" __global__ void scale_f32(
    const float* __restrict__ x, float* __restrict__ out,
    double alpha_d, int N)
{
    const float a = (float)alpha_d;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 xv = *reinterpret_cast<const float4*>(&x[i]);
        float4 o;
        o.x = a * xv.x; o.y = a * xv.y; o.z = a * xv.z; o.w = a * xv.w;
        *reinterpret_cast<float4*>(&out[i]) = o;
    } else {
        for (int j = i; j < min(i + 4, N); ++j) out[j] = a * x[j];
    }
}

extern "C" __global__ void scale_f64(
    const double* __restrict__ x, double* __restrict__ out,
    double alpha, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        double2 xv = *reinterpret_cast<const double2*>(&x[i]);
        double2 o;
        o.x = alpha * xv.x; o.y = alpha * xv.y;
        *reinterpret_cast<double2*>(&out[i]) = o;
    } else if (i < N) {
        out[i] = alpha * x[i];
    }
}

extern "C" __global__ void scale_f16(
    const __half* __restrict__ x, __half* __restrict__ out,
    double alpha_d, int N)
{
    const __half2 a2 = __float2half2_rn((float)alpha_d);
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        __half2 xv = *reinterpret_cast<const __half2*>(&x[i]);
        *reinterpret_cast<__half2*>(&out[i]) = __hmul2(a2, xv);
    } else if (i < N) {
        out[i] = __hmul(__double2half(alpha_d), x[i]);
    }
}


// ---- linear_combine2 : out = a*x + b*y ----

extern "C" __global__ void linear_combine2_f32(
    const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ out, double a_d, double b_d, int N)
{
    const float a = (float)a_d, b = (float)b_d;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 xv = *reinterpret_cast<const float4*>(&x[i]);
        float4 yv = *reinterpret_cast<const float4*>(&y[i]);
        float4 o;
        o.x = fmaf(a, xv.x, b * yv.x);
        o.y = fmaf(a, xv.y, b * yv.y);
        o.z = fmaf(a, xv.z, b * yv.z);
        o.w = fmaf(a, xv.w, b * yv.w);
        *reinterpret_cast<float4*>(&out[i]) = o;
    } else {
        for (int j = i; j < min(i + 4, N); ++j)
            out[j] = fmaf(a, x[j], b * y[j]);
    }
}

extern "C" __global__ void linear_combine2_f64(
    const double* __restrict__ x, const double* __restrict__ y,
    double* __restrict__ out, double a, double b, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        double2 xv = *reinterpret_cast<const double2*>(&x[i]);
        double2 yv = *reinterpret_cast<const double2*>(&y[i]);
        double2 o;
        o.x = fma(a, xv.x, b * yv.x);
        o.y = fma(a, xv.y, b * yv.y);
        *reinterpret_cast<double2*>(&out[i]) = o;
    } else if (i < N) {
        out[i] = fma(a, x[i], b * y[i]);
    }
}

extern "C" __global__ void linear_combine2_f16(
    const __half* __restrict__ x, const __half* __restrict__ y,
    __half* __restrict__ out, double a_d, double b_d, int N)
{
    const __half2 a2 = __float2half2_rn((float)a_d);
    const __half2 b2 = __float2half2_rn((float)b_d);
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        __half2 xv = *reinterpret_cast<const __half2*>(&x[i]);
        __half2 yv = *reinterpret_cast<const __half2*>(&y[i]);
        *reinterpret_cast<__half2*>(&out[i]) = __hadd2(__hmul2(a2, xv), __hmul2(b2, yv));
    } else if (i < N) {
        __half a = __double2half(a_d), b = __double2half(b_d);
        out[i] = __hadd(__hmul(a, x[i]), __hmul(b, y[i]));
    }
}


// ---- linear_combine3 : out = a*x + b*y + c*z ----

extern "C" __global__ void linear_combine3_f32(
    const float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ z, float* __restrict__ out,
    double a_d, double b_d, double c_d, int N)
{
    const float a = (float)a_d, b = (float)b_d, c = (float)c_d;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 xv = *reinterpret_cast<const float4*>(&x[i]);
        float4 yv = *reinterpret_cast<const float4*>(&y[i]);
        float4 zv = *reinterpret_cast<const float4*>(&z[i]);
        float4 o;
        o.x = fmaf(a, xv.x, fmaf(b, yv.x, c * zv.x));
        o.y = fmaf(a, xv.y, fmaf(b, yv.y, c * zv.y));
        o.z = fmaf(a, xv.z, fmaf(b, yv.z, c * zv.z));
        o.w = fmaf(a, xv.w, fmaf(b, yv.w, c * zv.w));
        *reinterpret_cast<float4*>(&out[i]) = o;
    } else {
        for (int j = i; j < min(i + 4, N); ++j)
            out[j] = fmaf(a, x[j], fmaf(b, y[j], c * z[j]));
    }
}

extern "C" __global__ void linear_combine3_f64(
    const double* __restrict__ x, const double* __restrict__ y,
    const double* __restrict__ z, double* __restrict__ out,
    double a, double b, double c, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        double2 xv = *reinterpret_cast<const double2*>(&x[i]);
        double2 yv = *reinterpret_cast<const double2*>(&y[i]);
        double2 zv = *reinterpret_cast<const double2*>(&z[i]);
        double2 o;
        o.x = fma(a, xv.x, fma(b, yv.x, c * zv.x));
        o.y = fma(a, xv.y, fma(b, yv.y, c * zv.y));
        *reinterpret_cast<double2*>(&out[i]) = o;
    } else if (i < N) {
        out[i] = fma(a, x[i], fma(b, y[i], c * z[i]));
    }
}

extern "C" __global__ void linear_combine3_f16(
    const __half* __restrict__ x, const __half* __restrict__ y,
    const __half* __restrict__ z, __half* __restrict__ out,
    double a_d, double b_d, double c_d, int N)
{
    const __half2 a2 = __float2half2_rn((float)a_d);
    const __half2 b2 = __float2half2_rn((float)b_d);
    const __half2 c2 = __float2half2_rn((float)c_d);
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < N) {
        __half2 xv = *reinterpret_cast<const __half2*>(&x[i]);
        __half2 yv = *reinterpret_cast<const __half2*>(&y[i]);
        __half2 zv = *reinterpret_cast<const __half2*>(&z[i]);
        *reinterpret_cast<__half2*>(&out[i]) =
            __hadd2(__hmul2(a2, xv), __hadd2(__hmul2(b2, yv), __hmul2(c2, zv)));
    } else if (i < N) {
        __half a = __double2half(a_d), b = __double2half(b_d), c = __double2half(c_d);
        out[i] = __hadd(__hmul(a, x[i]), __hadd(__hmul(b, y[i]), __hmul(c, z[i])));
    }
}
