#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

//
// prototypes for compiled wrappers around fill kernels for GPU memory
//

namespace arb {
namespace gpu {

void fill8(uint8_t* v, uint8_t value, std::size_t n);
void fill16(uint16_t* v, uint16_t value, std::size_t n);
void fill32(uint32_t* v, uint32_t value, std::size_t n);
void fill64(uint64_t* v, uint64_t value, std::size_t n);

// Brief:
// Perform type punning to pass arbitrary POD types to the GPU backend
// without polluting the library front end with CUDA kernels that would
// require compilation with nvcc.
//
// Detail:
// The implementation takes advantage of 4 fill functions that fill GPU
// memory with a {8, 16, 32, 64} bit unsigned integer. These these functions
// are used to fill a block of GPU memory with _any_ 8, 16, 32 or 64 bit POD
// value. e.g. for a 64-bit double, first convert the double into a 64-bit
// unsigned integer (with the same bits, not the same value), then call the
// 64-bit fill kernel precompiled using nvcc in the gpu library. This
// technique of converting from one type to another is called type-punning.
// There are some subtle challenges, due to C++'s strict aliasing rules,
// that require memcpy of single bytes if alignment of the two types does
// not match.

#define FILL(N) \
template <typename T> \
std::enable_if_t<sizeof(T)==sizeof(uint ## N ## _t)> \
fill(T* ptr, T value, std::size_t n) { \
    using I = uint ## N ## _t; \
    I v; \
    std::copy_n( \
        reinterpret_cast<char*>(&value), \
        sizeof(T), \
        reinterpret_cast<char*>(&v) \
    ); \
    arb::gpu::fill ## N(reinterpret_cast<I*>(ptr), v, n); \
}

FILL(8)
FILL(16)
FILL(32)
FILL(64)

// Stream-ordered variants: fill GPU memory on the given stream instead of
// the legacy default stream. 'stream' is a raw arb::gpu::stream_type handle
// passed as void*, so that this header stays free of the CUDA/HIP API (see
// memory/gpu_wrappers.hpp).

void fill8(void* stream, uint8_t* v, uint8_t value, std::size_t n);
void fill16(void* stream, uint16_t* v, uint16_t value, std::size_t n);
void fill32(void* stream, uint32_t* v, uint32_t value, std::size_t n);
void fill64(void* stream, uint64_t* v, uint64_t value, std::size_t n);

#define FILL_STREAM(N) \
template <typename T> \
std::enable_if_t<sizeof(T)==sizeof(uint ## N ## _t)> \
fill(void* stream, T* ptr, T value, std::size_t n) { \
    using I = uint ## N ## _t; \
    I v; \
    std::copy_n( \
        reinterpret_cast<char*>(&value), \
        sizeof(T), \
        reinterpret_cast<char*>(&v) \
    ); \
    arb::gpu::fill ## N(stream, reinterpret_cast<I*>(ptr), v, n); \
}

FILL_STREAM(8)
FILL_STREAM(16)
FILL_STREAM(32)
FILL_STREAM(64)

} // namespace gpu

namespace memory {

// Fill device memory of a container (anything with data()/size() and a
// value_type member) on the given raw GPU stream handle; the type-punning
// fill kernels of gpu:: are used. See gpu_wrappers.hpp for stream semantics.
template <typename C, typename T>
void fill(void* stream, C&& target, T value) {
    using value_type = typename std::decay_t<C>::value_type;
    gpu::fill(stream, target.data(), static_cast<value_type>(value), target.size());
}

} // namespace memory
} // namespace arb
