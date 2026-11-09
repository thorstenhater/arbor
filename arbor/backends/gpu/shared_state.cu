// GPU kernels and wrappers for shared state methods.

#include <backends/event.hpp>
#include <backends/event_stream_state.hpp>

#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

#include "backends/gpu/point_partials.hpp"

namespace arb {
namespace gpu {

namespace kernel {

// Vector/scalar addition: x[i] += v ∀i
template <typename T>
__global__ void add_scalar(unsigned n,
                           T* __restrict__ const x,
                           arb_value_type v) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        x[i] += v;
    }
}

__global__ void take_samples_impl(const raw_probe_info* __restrict__ const begin_marked,
                                  const raw_probe_info* __restrict__ const end_marked,
                                  const arb_value_type time,
                                  arb_value_type* __restrict__ const sample_time,
                                  arb_value_type* __restrict__ const sample_value) {
    const unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    const unsigned nsamples = end_marked - begin_marked;
    if (i<nsamples) {
        const auto p = begin_marked+i;
        sample_time[p->offset] = time;
        sample_value[p->offset] = p->handle? *p->handle: 0;
    }
}

// Add the parked point-process partials into conductivity and current_density.
//
// One thread per CV, instance groups folded in instantiation order and each
// CV's slots in ascending order. Slot order is warp order, so the sum no longer
// depends on which warp reached memory first. Only a CV whose run crossed a
// warp boundary owns any slots, so a mechanism spread thinly over CVs does
// almost no work here. The scratch is re-zeroed as it is read, so a group that
// did not run in this step cannot feed stale partials into the next one.
__global__
void fold_point_partials_impl(point_fold_desc* __restrict__ const desc,
                              const unsigned n_desc,
                              const arb_size_type n_cv,
                              arb_value_type* __restrict__ const vec_g,
                              arb_value_type* __restrict__ const vec_i) {
    const unsigned cv = threadIdx.x + blockIdx.x*blockDim.x;
    if (cv >= n_cv) return;

    for (unsigned d = 0; d < n_desc; ++d) {
        const auto e = desc[d];
        if (cv >= e.n_cv) continue;
        const auto first = e.part_begin[cv];
        if (first < 0) continue;
        const auto last = e.part_end[cv];

        arb_value_type g = 0;
        arb_value_type i = 0;
        for (auto s = first; s < last; ++s) {
            g += e.partial[2*s];
            i += e.partial[2*s + 1];
            e.partial[2*s] = 0;
            e.partial[2*s + 1] = 0;
        }
        vec_g[cv] += g;
        vec_i[cv] += i;
    }
}

} // namespace kernel

unsigned warp_lanes() {
    return impl::threads_per_warp();
}

void launch_fold_point_partials(point_fold_desc* desc,
                                unsigned n_desc,
                                arb_size_type n_cv,
                                arb_value_type* vec_g,
                                arb_value_type* vec_i) {
    if (!n_desc || !n_cv) return;
    launch_1d(n_cv, 128, kernel::fold_point_partials_impl, desc, n_desc, n_cv, vec_g, vec_i);
}

void add_scalar(std::size_t n, arb_value_type* data, arb_value_type v) {
    launch_1d(n, 128, kernel::add_scalar<arb_value_type>, n, data, v);
}

void take_samples_impl(const event_stream_state<raw_probe_info>& s,
                       const arb_value_type& time, arb_value_type* sample_time, arb_value_type* sample_value) {
    launch_1d(s.size(), 128, kernel::take_samples_impl, s.begin_marked, s.end_marked, time, sample_time, sample_value);
}

} // namespace gpu
} // namespace arb
