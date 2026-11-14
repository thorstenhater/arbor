#pragma once

// Scratch layout for ordered accumulation of point-process contributions.
//
// A point mechanism that accumulates into vec_g or vec_i runs one thread per
// instance. Each warp reduces the instances it covers with a butterfly
// (gpu/reduce_by_key.hpp) and its root thread adds the result to the target.
// Point instances are laid out in ascending CV order, so a CV's instances form
// one contiguous run; when that run crosses a warp boundary the CV takes one
// add per warp it touches, and the arrival order of those adds is decided by
// the memory system, not by the program. That is the source of the run-to-run
// variation reported in gpu-spike-count-variation.org.
//
// The fix routes the partials of a multi-warp run to scratch, to be added in a
// fixed order once every mechanism is done. A CV whose run stays inside one
// warp keeps the direct add: it contributes a single add per step, which is the
// same value whoever wins the bus. That distinction is what keeps the fold
// small — it touches only CVs that actually need it.
//
// This header is host-only on purpose: the table construction is plain integer
// arithmetic, so it can be built and tested without a device.

#include <cstddef>
#include <vector>

#include <arbor/mechanism_abi.h>
#include <arbor/assert.hpp>

namespace arb {
namespace gpu {

// Number of lanes in a warp, as assumed by the reduction in reduce_by_key.hpp.
constexpr unsigned partial_lanes = 32;

struct point_partial_tables {
    // False if the instance layout cannot use the ordered path, in which case
    // the mechanism keeps the direct atomic add and behaves as before.
    bool valid = true;

    // Indexed by CV. First scratch slot of that CV's partials, or -1 if the CV
    // takes the direct add (single-warp run, or no instance of this mechanism).
    std::vector<arb_index_type> part_begin;

    // Indexed by CV. Warp index of that CV's first instance, so a root thread
    // can number its own partial from 0 within the run.
    std::vector<arb_index_type> warp_base;

    // Indexed by CV. One past the last slot of that CV's partials, or -1. Only
    // the fold needs this, so it stays out of the mechanism ABI.
    std::vector<arb_index_type> part_end;

    // Scratch slots in total; each holds one (g, i) pair, so the scratch array
    // is 2*n_slots doubles.
    std::size_t n_slots = 0;

    // How many CVs take the fold path.
    std::size_t n_fold_cv = 0;

    // How many CVs have at least one instance of the mechanism.
    std::size_t n_cv_with_inst = 0;
};

// node_index is the mechanism's instance-to-CV map, exactly as it is handed to
// the device: one entry per instance, in instance order.
inline point_partial_tables build_point_partial_tables(
    arb_size_type n_cv,
    const arb_index_type* node_index,
    std::size_t n_inst,
    unsigned lanes = partial_lanes)
{
    point_partial_tables t;
    t.part_begin.assign(n_cv, -1);
    t.warp_base.assign(n_cv, 0);
    t.part_end.assign(n_cv, -1);

    // The fold relies on each CV occupying one contiguous run of instances.
    // fvm_layout permutes point instances into ascending CV order, so this
    // holds; anything else falls back to the old path rather than mis-folding.
    for (std::size_t i = 1; i < n_inst; ++i) {
        if (node_index[i] < node_index[i - 1]) {
            t.valid = false;
            t.part_begin.assign(n_cv, -1);
            t.warp_base.assign(n_cv, 0);
            t.part_end.assign(n_cv, -1);
            return t;
        }
    }

    std::size_t i = 0;
    while (i < n_inst) {
        const arb_index_type cv = node_index[i];
        if (cv < 0 || arb_size_type(cv) >= n_cv) {
            t.valid = false;
            t.part_begin.assign(n_cv, -1);
            t.warp_base.assign(n_cv, 0);
            t.part_end.assign(n_cv, -1);
            return t;
        }
        std::size_t j = i;
        while (j < n_inst && node_index[j] == cv) ++j;
        const std::size_t begin = i;
        const std::size_t end = j; // half-open [begin, end)

        const unsigned first_warp = unsigned(begin / lanes);
        const unsigned last_warp = unsigned((end - 1) / lanes);
        const unsigned n_warp = last_warp - first_warp + 1;

        t.warp_base[cv] = arb_index_type(first_warp);
        ++t.n_cv_with_inst;

        // A run in a single warp contributes one add, so nothing to order.
        if (n_warp > 1) {
            arb_assert(t.part_begin[cv] == -1); // one run per CV, by sortedness
            t.part_begin[cv] = arb_index_type(t.n_slots);
            t.n_slots += n_warp;
            t.part_end[cv] = arb_index_type(t.n_slots);
            ++t.n_fold_cv;
        }
        i = end;
    }
    return t;
}

// What the fold needs to know about one mechanism instance group: its scratch
// and the per-CV slot ranges inside it. Written at instantiation time, read by
// the fold kernel every step.
struct point_fold_desc {
    arb_value_type*  partial;    // 2*n_slots doubles, zeroed between folds
    arb_index_type*  part_begin; // n_cv entries, -1 where the CV needs no fold
    arb_index_type*  part_end;   // n_cv entries
    arb_size_type    n_cv;
};

// Lanes in a warp on the device this library was built for. Must be asked of
// the GPU toolchain: a wave64 (CDNA) build reports 64 from device code and 32
// from a host compiler that has no view of the target.
unsigned warp_lanes();

// Add every parked partial into conductivity and current_density, one thread
// per CV, mechanisms in instantiation order, slots in ascending order. Launched
// after all mechanisms have contributed their currents for the step.
void launch_fold_point_partials(point_fold_desc* desc,
                                unsigned n_desc,
                                arb_size_type n_cv,
                                arb_value_type* vec_g,
                                arb_value_type* vec_i);

} // namespace gpu
} // namespace arb
