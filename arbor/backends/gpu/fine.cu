#include <arbor/fvm_types.hpp>
#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

#include "fine.hpp"

namespace arb {
namespace gpu {
namespace kernels {

template <typename T, typename I>
__global__
void gather(const T* __restrict__ const from,
            T* __restrict__ const to,
            const I* __restrict__ const p,
            unsigned n) {
    unsigned i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i<n) to[i] = from[p[i]];
}

template <typename T, typename I>
__global__
void gather_compact(const T* __restrict__ const from,
            T* __restrict__ const to,
            const I* __restrict__ const p,
            unsigned n) {
    unsigned i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i<n) to[i] = from[2*p[i]];
}


template <typename T, typename I>
__global__
void scatter(const T* __restrict__ const from,
             T* __restrict__ const to,
             const I* __restrict__ const p,
             unsigned n) {
    unsigned i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<n) to[p[i]] = from[i];
}

} // namespace kernels

ARB_ARBOR_API void gather(const arb_value_type* from,
                          arb_value_type* to,
                          const arb_index_type* p,
                          unsigned n,
                          bool compact) {
    if (compact) {
        launch_1d(n, 128, kernels::gather_compact<arb_value_type, arb_index_type>, from, to, p, n);
    }
    else {
        launch_1d(n, 128, kernels::gather<arb_value_type, arb_index_type>, from, to, p, n);
    }
}

ARB_ARBOR_API void scatter(const arb_value_type* from,
                           arb_value_type* to,
                           const arb_index_type* p,
                           unsigned n) {
    launch_1d(n, 128, kernels::scatter<arb_value_type, arb_index_type>, from, to, p, n);
}

} // namespace gpu
} // namespace arb
