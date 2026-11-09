#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <arbor/fvm_types.hpp>

#include "backends/gpu/point_partials.hpp"

using namespace arb;
using namespace arb::gpu;

// The scratch layout that lets a GPU point mechanism accumulate a CV's partials
// in a fixed order. The tables are built on the host and the device only indexes
// them, so the whole contract can be checked without a device: build tables from
// a layout, have the roots write where the tables say, then fold as the fold
// kernel does, and require the outcome not to depend on the order the roots
// arrived in.
//
// gpu/reduce_by_key.hpp is what decides who writes: lane 0 of a warp is always a
// root, and any other lane is a root when the lane to its left holds a different
// CV. A root writes the sum of its own warp's instances of that CV.

namespace {

constexpr unsigned lanes = partial_lanes;

struct root_write {
    arb_index_type cv;
    unsigned warp;
    arb_value_type value; // this warp's instances of this CV, summed
};

// Faithful to the device: one thread per instance, in instance order.
std::vector<root_write> find_roots(const std::vector<arb_index_type>& node_index,
                                   const std::vector<arb_value_type>& contrib) {
    std::vector<root_write> writes;
    for (std::size_t i = 0; i < node_index.size(); ++i) {
        const unsigned lane = unsigned(i % lanes);
        const bool left_differs = lane == 0 || node_index[i] != node_index[i - 1];
        if (!left_differs) continue;
        const auto cv = node_index[i];
        const unsigned warp = unsigned(i / lanes);
        auto sum = contrib[i];
        for (std::size_t j = i + 1; j % lanes != 0 && j < node_index.size(); ++j) {
            if (node_index[j] != cv) break;
            sum += contrib[j];
        }
        writes.push_back({cv, warp, sum});
    }
    return writes;
}

// Where a root puts its partial, exactly as accumulate_point computes it.
std::size_t slot_of(const point_partial_tables& t, const root_write& w) {
    return std::size_t(t.part_begin[w.cv] + (arb_index_type(w.warp) - t.warp_base[w.cv]));
}

// Replica of gpu::kernel::fold_point_partials_impl for one instance group: one
// thread per CV, slots ascending, scratch re-zeroed as it is read. Adds into
// `out`, which already holds the contributions that went straight to the target.
void fold(const point_partial_tables& t,
          std::vector<arb_value_type>& partial,
          arb_size_type n_cv,
          std::vector<arb_value_type>& out) {
    for (arb_size_type cv = 0; cv < n_cv; ++cv) {
        const auto first = t.part_begin[cv];
        if (first < 0) continue;
        arb_value_type g = 0;
        for (auto s = first; s < t.part_end[cv]; ++s) {
            g += partial[2*s];
            partial[2*s] = 0;
        }
        out[cv] += g;
    }
}

// Everything a step's worth of roots leaves behind: the scratch, and the CVs
// that took a single direct add. `order` is the arrival order of the roots.
std::vector<arb_value_type> apply_roots(const point_partial_tables& t,
                                        const std::vector<root_write>& writes,
                                        const std::vector<std::size_t>& order,
                                        arb_size_type n_cv,
                                        std::size_t n_slots) {
    std::vector<arb_value_type> partial(2*n_slots, 0);
    std::vector<arb_value_type> out(n_cv, 0);
    for (auto i: order) {
        const auto& w = writes[i];
        if (t.part_begin[w.cv] >= 0) partial[2*slot_of(t, w)] = w.value;
        else out[w.cv] += w.value; // one root only, so when it lands cannot matter
    }
    fold(t, partial, n_cv, out);
    return out;
}

// A layout: how many instances sit on each CV, in CV order, so the instance list
// is sorted, which is what fvm_layout hands the backend.
struct layout {
    std::vector<arb_index_type> node_index;
    std::vector<arb_value_type> contrib;
    arb_size_type n_cv;
};

layout make_layout(std::mt19937& rng, arb_size_type n_cv, unsigned max_per_cv) {
    layout l;
    l.n_cv = n_cv;
    std::uniform_int_distribution<unsigned> inst_count(0, max_per_cv);
    // Spread the first run off the start of a warp, which is what decides
    // whether a run straddles.
    std::uniform_int_distribution<unsigned> lead(0, lanes - 1);
    for (unsigned pad = 0; pad < lead(rng) % 7; ++pad) {
        l.node_index.push_back(0);
        l.contrib.push_back(0);
    }
    for (arb_size_type cv = 0; cv < n_cv; ++cv) {
        const auto k = inst_count(rng);
        for (unsigned j = 0; j < k; ++j) {
            l.node_index.push_back(arb_index_type(cv));
            // A mix of magnitudes so that a change of order is a change of bits.
            static constexpr arb_value_type pool[] = {1.0, 3.7, 1e-13, -2.5, 1e7, 0.125, -1e-11, 6.0};
            std::uniform_int_distribution<unsigned> pick(0, 7);
            l.contrib.push_back(pool[pick(rng)]);
        }
    }
    // The leading pad above landed on CV 0, so drop it if CV 0 has no instances.
    return l;
}

// Roots writing into a scalar in arrival order: what the code did before the
// scratch existed. Used to prove the test data can tell orders apart at all.
std::vector<arb_value_type> atomic_add_in_order(const layout& l,
                                                const std::vector<root_write>& writes,
                                                const std::vector<std::size_t>& order) {
    std::vector<arb_value_type> out(l.n_cv, 0);
    for (auto i: order) out[writes[i].cv] += writes[i].value;
    return out;
}

class point_partials: public ::testing::Test {
public:
    point_partials(): rng(12345) {}
protected:
    std::mt19937 rng;
};

TEST_F(point_partials, empty_layout_needs_no_scratch) {
    auto t = build_point_partial_tables(4, nullptr, 0, lanes);
    EXPECT_TRUE(t.valid);
    EXPECT_EQ(0u, t.n_slots);
    EXPECT_EQ(0u, t.n_fold_cv);
    ASSERT_EQ(4u, t.part_begin.size());
    for (auto b: t.part_begin) EXPECT_EQ(-1, b);
}

TEST_F(point_partials, run_inside_one_warp_keeps_the_direct_add) {
    std::vector<arb_index_type> node_index(8, 3);
    auto t = build_point_partial_tables(6, node_index.data(), node_index.size(), lanes);
    EXPECT_TRUE(t.valid);
    EXPECT_EQ(0u, t.n_slots);
    EXPECT_EQ(1u, t.n_cv_with_inst);
    EXPECT_EQ(-1, t.part_begin[3]);
    EXPECT_EQ(-1, t.part_end[3]);
    EXPECT_EQ(0, t.warp_base[3]);
}

TEST_F(point_partials, run_across_one_boundary_gets_two_slots) {
    std::vector<arb_index_type> node_index(40, 0);
    auto t = build_point_partial_tables(1, node_index.data(), node_index.size(), lanes);
    EXPECT_TRUE(t.valid);
    EXPECT_EQ(1u, t.n_fold_cv);
    EXPECT_EQ(2u, t.n_slots);
    EXPECT_EQ(0, t.part_begin[0]);
    EXPECT_EQ(2, t.part_end[0]);
    EXPECT_EQ(0, t.warp_base[0]);
}

TEST_F(point_partials, run_starting_mid_warp_is_numbered_from_its_own_warp) {
    // 20 instances of CV 0 fill lanes 0..19 of warp 0; CV 1 then runs from index
    // 20 to 39, so it starts in warp 0 and continues into warp 1.
    std::vector<arb_index_type> node_index(20, 0);
    for (unsigned i = 0; i < 20; ++i) node_index.push_back(1);

    auto t = build_point_partial_tables(2, node_index.data(), node_index.size(), lanes);
    EXPECT_TRUE(t.valid);
    EXPECT_EQ(-1, t.part_begin[0]);   // CV 0 stays inside warp 0
    EXPECT_EQ(0, t.part_begin[1]);
    EXPECT_EQ(2, t.part_end[1]);
    EXPECT_EQ(0, t.warp_base[1]);
    EXPECT_EQ(2u, t.n_slots);
}

TEST_F(point_partials, long_run_gets_one_slot_per_warp_it_touches) {
    std::vector<arb_index_type> node_index;
    for (unsigned i = 0; i < 32; ++i) node_index.push_back(0); // fills warp 0
    for (unsigned i = 0; i < 100; ++i) node_index.push_back(1); // indices 32..131

    auto t = build_point_partial_tables(2, node_index.data(), node_index.size(), lanes);
    EXPECT_TRUE(t.valid);
    EXPECT_EQ(-1, t.part_begin[0]);
    EXPECT_EQ(0, t.part_begin[1]);
    EXPECT_EQ(4, t.part_end[1]);   // warps 1, 2, 3, 4
    EXPECT_EQ(1, t.warp_base[1]);
    EXPECT_EQ(4u, t.n_slots);
}

TEST_F(point_partials, cvs_without_instances_are_skipped) {
    std::vector<arb_index_type> node_index(50, 2);
    auto t = build_point_partial_tables(5, node_index.data(), node_index.size(), lanes);
    EXPECT_TRUE(t.valid);
    EXPECT_EQ(2u, t.n_slots);      // indices 0..49 span warps 0 and 1
    EXPECT_EQ(0, t.part_begin[2]);
    for (auto cv: {0u, 1u, 3u, 4u}) {
        EXPECT_EQ(-1, t.part_begin[cv]);
        EXPECT_EQ(-1, t.part_end[cv]);
    }
}

TEST_F(point_partials, unsorted_layout_is_rejected_not_mis_folded) {
    std::vector<arb_index_type> node_index = {0, 0, 1, 0};
    auto t = build_point_partial_tables(2, node_index.data(), node_index.size(), lanes);
    EXPECT_FALSE(t.valid);
    EXPECT_EQ(0u, t.n_slots);
    for (auto b: t.part_begin) EXPECT_EQ(-1, b);
    for (auto e: t.part_end) EXPECT_EQ(-1, e);
}

TEST_F(point_partials, cv_index_out_of_range_is_rejected) {
    std::vector<arb_index_type> node_index(40, 7);
    auto t = build_point_partial_tables(4, node_index.data(), node_index.size(), lanes);
    EXPECT_FALSE(t.valid);
    EXPECT_EQ(0u, t.n_slots);
    for (auto b: t.part_begin) EXPECT_EQ(-1, b);
}

// The tables must hand each straddling CV exactly the warps its run covers, and
// no two CVs may share a slot.
TEST_F(point_partials, slot_ranges_are_the_warps_the_run_covers) {
    for (unsigned trial = 0; trial < 200; ++trial) {
        auto l = make_layout(rng, arb_size_type(1 + rng()%40), 70);
        auto t = build_point_partial_tables(l.n_cv, l.node_index.data(), l.node_index.size(), lanes);
        ASSERT_TRUE(t.valid);

        std::set<std::size_t> used;
        for (arb_size_type cv = 0; cv < l.n_cv; ++cv) {
            std::set<unsigned> warps;
            for (std::size_t i = 0; i < l.node_index.size(); ++i) {
                if (l.node_index[i] == arb_index_type(cv)) warps.insert(unsigned(i/lanes));
            }
            if (warps.size() < 2) {
                EXPECT_EQ(-1, t.part_begin[cv]) << "cv " << cv << " in trial " << trial;
                continue;
            }
            EXPECT_EQ(arb_index_type(warps.size()), t.part_end[cv] - t.part_begin[cv])
                << "cv " << cv << " in trial " << trial;
            EXPECT_EQ(arb_index_type(*warps.begin()), t.warp_base[cv]);
            for (auto s = t.part_begin[cv]; s < t.part_end[cv]; ++s) {
                ASSERT_TRUE(used.insert(std::size_t(s)).second) << "slot reuse in trial " << trial;
            }
        }
        EXPECT_EQ(t.n_slots, used.size());
    }
}

// The point of the whole exercise: whichever order the roots reach memory, the
// CV ends up with the same conductance.
TEST_F(point_partials, fold_does_not_depend_on_arrival_order) {
    unsigned old_scheme_varied = 0;
    unsigned straddling_trials = 0;

    for (unsigned trial = 0; trial < 100; ++trial) {
        auto l = make_layout(rng, arb_size_type(1 + rng()%30), 60);
        auto t = build_point_partial_tables(l.n_cv, l.node_index.data(), l.node_index.size(), lanes);
        ASSERT_TRUE(t.valid);
        if (!t.n_slots) continue;
        ++straddling_trials;

        auto writes = find_roots(l.node_index, l.contrib);
        ASSERT_FALSE(writes.empty());
        for (const auto& w: writes) {
            if (t.part_begin[w.cv] >= 0) { ASSERT_LT(slot_of(t, w), t.n_slots); }
        }

        std::vector<std::size_t> order(writes.size());
        for (std::size_t i = 0; i < writes.size(); ++i) order[i] = i;

        // What the CVs should hold when the step is done, computed straight from
        // the layout: one direct add for a single-warp CV, and for the rest the
        // partials added in ascending warp order.
        std::vector<arb_value_type> want(l.n_cv, 0);
        for (arb_size_type cv = 0; cv < l.n_cv; ++cv) {
            std::vector<std::pair<unsigned, arb_value_type>> by_warp;
            for (const auto& w: writes) if (w.cv == arb_index_type(cv)) by_warp.push_back({w.warp, w.value});
            if (by_warp.empty()) continue;
            if (t.part_begin[cv] < 0) {
                ASSERT_EQ(1u, by_warp.size());
                want[cv] = by_warp.front().second;
                continue;
            }
            std::sort(by_warp.begin(), by_warp.end());
            for (auto& w: by_warp) want[cv] += w.second;
        }

        bool old_varied_here = false;
        for (unsigned round = 0; round < 8; ++round) {
            std::shuffle(order.begin(), order.end(), rng);
            auto got = apply_roots(t, writes, order, l.n_cv, t.n_slots);
            EXPECT_EQ(want, got) << "arrival order changed the result, trial " << trial;

            // The pre-fix scheme, same data, does depend on the order, or this
            // test would pass for a layout that could never have varied.
            if (atomic_add_in_order(l, writes, order) != want) old_varied_here = true;
        }
        if (old_varied_here) ++old_scheme_varied;
    }
    EXPECT_LT(0u, straddling_trials);
    EXPECT_LT(0u, old_scheme_varied) << "test data cannot tell an ordered sum from a shuffled one";
}

} // namespace
