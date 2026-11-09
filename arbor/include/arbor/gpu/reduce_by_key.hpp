#pragma once

#include <cstdint>
#include "gpu_api.hpp"
#include "gpu_common.hpp"

namespace arb {
namespace gpu {

// key_set_pos stores information required by a thread to calculate its
// contribution to a reduce by key operation.
//
// In reduce_by_key each thread performs a reduction with threads that have the same
// index in a sorted index array.
// The algorithm for reduce_by_key implemented here requires that each thread
// knows how far it is from the end of the key set, and whether it is the first
// thread in the warp with the key.
//
// As currently implemented, this could be performed inline inside the reduce_by_key
// function, however it is in a separate data type as a first step towards reusing
// the same key set information for multipe reduction kernel calls.

struct key_set_pos {
    unsigned width;         // distance to one past the end of this run
    unsigned lane_id;       // id of this warp lane
    lane_mask_type key_mask; // warp mask of threads participating in reduction
    unsigned is_root;       // if this lane is the first in the run

    // The constructor takes the index of each thread and a mask of which threads
    // in the warp are participating in the reduction, and the threads work cooperatively
    // using warp intrinsics and bit twiddling, so that each thread will have unique
    // information that describes its position in the key set.
    __device__
    key_set_pos(int idx, lane_mask_type mask) {
        key_mask = mask;
        lane_id = threadIdx.x%impl::threads_per_warp();
        // count_leading_zeros counts from the top bit of lane_mask_type, so the
        // lane count is measured against its full bit width: 64 on HIP (ballot is
        // wave64-wide on CDNA, wave32 on RDNA) and 32 on CUDA. On a wave32 HIP
        // mask the upper 32 bits are zero, so 64 - count_leading_zeros still
        // yields the correct count.
        unsigned num_lanes = 8*sizeof(lane_mask_type) - count_leading_zeros(key_mask);

        // Determine if this thread is the root (i.e. first thread with this key).
        int left_idx  = shfl_up(key_mask, idx, lane_id, lane_id? 1: 0);

        is_root = lane_id? left_idx!=idx: 1;

        // Determine the range this thread contributes to.
        lane_mask_type roots = ballot(key_mask, is_root);

        // Find the distance to the lane id one past the end of the run.
        // Take care if this is the last run in the warp.
        width = find_first_set(roots>>(lane_id+1));
        if (!width) width = num_lanes-lane_id;
    }
};

// The in-warp part of the reduction: after this, every lane that starts a run
// segment inside this warp holds the sum over that segment. The order of these
// adds is fixed. What is not fixed is the order the segments reach memory.
template <typename T>
__device__ __inline__
void butterfly_in_run(T& contribution, const key_set_pos& run) {
    unsigned shift = 1;
    const unsigned width = run.width;

    unsigned w = shift<width? shift: 0;

    while (any(run.key_mask, w)) {
        T source_value = shfl_down(run.key_mask, contribution, run.lane_id, w);

        if (w) contribution += source_value;

        shift <<= 1;
        w = shift<width? shift: 0;
    }
}

template <typename T, typename I>
__device__ __inline__
void reduce_by_key(T contribution, T* target, I i, lane_mask_type mask) {
    key_set_pos run(i, mask);
    butterfly_in_run(contribution, run);

    if(run.is_root) {
        // The update must be atomic, because the run may span multiple warps.
        gpu_atomic_add(target+i, contribution);
    }
}

// Same reduction as reduce_by_key, but a warp whose run continues in another
// warp hands its partial to scratch instead of adding it to the target.
//
// Point instances are laid out in ascending CV order, so a CV occupies one
// contiguous run of instances. A run that stays inside a warp contributes a
// single add per step, which is the same value whenever it lands; a run that
// straddles a warp boundary contributes one add per warp it touches, and the
// order those adds arrive in is decided by the memory system rather than by the
// program, which makes the accumulated conductance and current irreproducible.
// Those partials are stored here, in slot order, which is ascending warp
// order, and gpu::fold_point_partials adds them once every mechanism is done.
//
// partial, part_begin and warp_base are the scratch tables described in
// arbor/mechanism_abi.h. target_ix selects the half of a slot that belongs to
// this target (0 = vec_g, 1 = vec_i). With part_begin == nullptr, or for a CV
// whose run stays in one warp, this does exactly what reduce_by_key does.
template <typename T, typename I>
__device__ __inline__
void accumulate_point(T contribution, T* target, I i, lane_mask_type mask,
                      T* partial, const I* part_begin, const I* warp_base,
                      unsigned target_ix) {
    key_set_pos run(i, mask);
    butterfly_in_run(contribution, run);

    if (run.is_root) {
        const I begin = part_begin? part_begin[i]: -1;
        if (begin >= 0) {
            const unsigned tid  = threadIdx.x + blockDim.x*blockIdx.x;
            const unsigned warp = tid / impl::threads_per_warp();
            const unsigned rank = unsigned(begin - warp_base[i]) + warp; // 0 within the run
            partial[2*rank + target_ix] = contribution;
        }
        else {
            gpu_atomic_add(target+i, contribution);
        }
    }
}

} // namespace gpu
} // namespace arb
