#pragma once
#include <cuda_fp16.h>

// ============================================================
// Type conversion: double literal -> target type
// ============================================================
template<typename T>
__device__ __forceinline__ T make_real(double v);

template<> __device__ __forceinline__ __half  make_real<__half>(double v)  { return __double2half(v); }
template<> __device__ __forceinline__ float   make_real<float>(double v)   { return static_cast<float>(v); }
template<> __device__ __forceinline__ double  make_real<double>(double v)  { return v; }

// ============================================================
// Read-only cache hint via __ldg
// ============================================================
template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) { return __ldg(ptr); }

template<>
__device__ __forceinline__ __half ldg<__half>(const __half* ptr) {
    return __ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(ptr)));
}

// ============================================================
// Kernel execution modes
// ============================================================
enum KernelMode { EULER_STEP, PDEFUNC };
