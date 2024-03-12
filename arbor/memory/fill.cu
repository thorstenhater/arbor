#include <arbor/gpu/gpu_common.hpp>

#include <cstdint>
#include <iostream>

namespace arb {
namespace gpu {

template <typename T, typename I>
__global__
void fill_kernel(T* __restrict__ const v, T value, I n) {
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid < n) {
        v[tid] = value;
    }
}

__global__
void fill_kernel_vec(std::uint64_t* __restrict__ const v, std::uint64_t value, std::size_t n) {
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;
    auto v4 = reinterpret_cast<ulong4*>(v);
    ulong4 value4 = make_ulong4(value, value, value, value);
    size_t n4 = n/4;
    // Bulk write
    if(tid < n4) v4[tid] = value4;
    // First thread handles the remainder at the tail end
    if(tid == 0) {
        for (int ix = 0; ix < n%4; ++ix) {
            v[ix + 4*n4] = value;
        }
    }
}

constexpr static int block_size = 128;

void fill8(uint8_t* v, uint8_t value, std::size_t n) {
    launch_1d(n, block_size, fill_kernel<uint8_t, std::size_t>, v, value, n);
}

void fill16(uint16_t* v, uint16_t value, std::size_t n) {
    launch_1d(n, block_size, fill_kernel<uint16_t, std::size_t>, v, value, n);
}

void fill32(uint32_t* v, uint32_t value, std::size_t n) {
    launch_1d(n, block_size, fill_kernel<uint32_t, std::size_t>, v, value, n);
}

void fill64(uint64_t* v, uint64_t value, std::size_t n) {
    if (n >= 4) {
        auto len = (n + 3)/4;
        std::cout << "[VEC] val=" << value << " ptr=" << v << " cnt=" << n << '\n';
        launch_1d(len, block_size, fill_kernel_vec, v, value, n);
    }
    else {
        std::cout << "[SCL] val=" << value << " ptr=" << v << " cnt=" << n << '\n';
        launch_1d(n, block_size, fill_kernel_vec, v, value, n);
    }
}

} // namespace gpu
} // namespace arb
