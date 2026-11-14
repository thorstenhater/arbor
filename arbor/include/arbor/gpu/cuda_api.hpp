#include <utility>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <arbor/export.hpp>

namespace arb {
namespace gpu {

/// Device queries

using DeviceProp = cudaDeviceProp;

struct ARB_SYMBOL_VISIBLE api_error_type {
    cudaError_t value;
    api_error_type(cudaError_t e): value(e) {}

    operator bool() const {
        return value==cudaSuccess;
    }

    bool is_invalid_device() const {
        return value == cudaErrorInvalidDevice;
    }

    std::string name() const {
        std::string s = cudaGetErrorName(value);
        return s;
    }

    std::string description() const {
        std::string s = cudaGetErrorString(value);
        return s;
    }
};

inline api_error_type get_last_error() {
    return cudaGetLastError();
}

inline api_error_type device_synchronize() {
    return cudaDeviceSynchronize();
}

/// Streams

// Stream handle as used by the GPU backend. It crosses the API-free library
// layers (memory:: gpu_memcpy_*(stream, ...), arb_mechanism_ppack.stream) as
// an opaque void*; cast with static_cast<stream_type> to use the functions
// below.
using stream_type = cudaStream_t;

// The legacy default stream ("stream 0").
inline constexpr stream_type null_stream = nullptr;

// Create a stream that does not synchronize with the legacy default stream:
// work on such streams only orders against work on the same stream.
inline api_error_type create_stream(stream_type* s) {
    return cudaStreamCreateWithFlags(s, cudaStreamNonBlocking);
}

inline api_error_type destroy_stream(stream_type s) {
    return cudaStreamDestroy(s);
}

// Block the calling host thread until all work previously submitted on s has
// completed. On the legacy stream (null_stream) this waits for the legacy
// stream only, not for other (non-blocking) streams in the process.
inline api_error_type stream_synchronize(stream_type s) {
    return cudaStreamSynchronize(s);
}

constexpr auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr auto gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
constexpr auto gpuHostRegisterPortable = cudaHostRegisterPortable;

template <typename... ARGS>
inline api_error_type get_device_properties(ARGS &&... args) {
    return cudaGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type set_device(ARGS &&... args) {
    return cudaSetDevice(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_memcpy(ARGS &&... args) {
    return cudaMemcpy(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_memcpy_async(ARGS &&... args) {
    return cudaMemcpyAsync(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type host_register(ARGS &&... args) {
    return cudaHostRegister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type host_unregister(ARGS &&... args) {
    return cudaHostUnregister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_malloc(ARGS &&... args) {
    return cudaMalloc(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_free(ARGS &&... args) {
    return cudaFree(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_mem_get_info(ARGS &&... args) {
    return cudaMemGetInfo(std::forward<ARGS>(args)...);
}

// Lane mask type: 32-bit on CUDA (warp size always 32).
using lane_mask_type = unsigned;

#ifdef __CUDACC__
/// Atomics

// Wrappers around CUDA addition functions.
// CUDA 8 introduced support for atomicAdd with double precision, but only for
// Pascal GPUs (__CUDA_ARCH__ >= 600). We assume no one is running anything older.
__device__
inline double gpu_atomic_add(double* address, double val) {
    return atomicAdd(address, val);
}

__device__
inline double gpu_atomic_sub(double* address, double val) {
    return gpu_atomic_add(address, -val);
}

__device__
inline float gpu_atomic_add(float* address, float val) {
    return atomicAdd(address, val);
}

__device__
inline float gpu_atomic_sub(float* address, float val) {
    return atomicAdd(address, -val);
}

/// Warp-Level Primitives
__device__ __inline__ unsigned ballot(unsigned mask, unsigned is_root) {
    return __ballot_sync(mask, is_root);
}

__device__ __inline__ unsigned active_mask() {
    return __activemask();
}

__device__ __inline__ unsigned any(unsigned mask, unsigned width) {
    return __any_sync(mask, width);
}

template<typename T>
__device__ __inline__ T shfl_up(unsigned mask, T var, unsigned lane_id, unsigned shift) {
    return __shfl_up_sync(mask, var, shift);
}

template<typename T>
__device__ __inline__ T shfl_down(unsigned mask, T var, unsigned lane_id, unsigned shift) {
    return __shfl_down_sync(mask, var, shift);
}

// Count leading zeros / find first set on a lane mask. lane_mask_type is 32-bit
// on CUDA (warp size is always 32), so the 32-bit __clz/__ffs variants are used.
__device__ __inline__ unsigned count_leading_zeros(lane_mask_type mask) {
    return __clz(mask);
}

__device__ __inline__ unsigned find_first_set(lane_mask_type mask) {
    return __ffs(mask);
}
#endif

} // namespace gpu
} // namespace arb
