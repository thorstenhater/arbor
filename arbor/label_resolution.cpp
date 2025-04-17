#include <iterator>
#include <vector>
#include <numeric>
#include <algorithm>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/expected.hpp>
#include <arbor/util/hash_def.hpp>

#include "label_resolution.hpp"
#include "util/span.hpp"

namespace arb {

// cell_label_range methods
cell_label_range::cell_label_range(std::vector<cell_size_type> size_vec,
                                   const std::vector<cell_tag_type>& label_vec,
                                   std::vector<lid_range> range_vec):
    sizes(std::move(size_vec)), ranges(std::move(range_vec)) {
    std::transform(label_vec.begin(), label_vec.end(),
                   std::back_inserter(labels),
                   hash_value<const std::string&>);
    arb_assert(check_invariant());
};

cell_label_range::cell_label_range(std::vector<cell_size_type> size_vec,
                                   std::vector<hash_type> label_vec,
                                   std::vector<lid_range> range_vec):
    sizes(std::move(size_vec)), labels(std::move(label_vec)), ranges(std::move(range_vec)) {
    arb_assert(check_invariant());
};


void cell_label_range::add_cell() { sizes.push_back(0); }

void cell_label_range::add_label(hash_type label, lid_range range) {
    if (sizes.empty()) throw arbor_internal_error("adding label to cell_label_range without cell");
    ++sizes.back();
    labels.push_back(label);
    ranges.push_back(std::move(range));
}

void cell_label_range::append(cell_label_range other) {
    using std::make_move_iterator;
    sizes.insert(sizes.end(),   make_move_iterator(other.sizes.begin()),  make_move_iterator(other.sizes.end()));
    labels.insert(labels.end(), make_move_iterator(other.labels.begin()), make_move_iterator(other.labels.end()));
    ranges.insert(ranges.end(), make_move_iterator(other.ranges.begin()), make_move_iterator(other.ranges.end()));
}

bool cell_label_range::check_invariant() const {
    const cell_size_type count = std::accumulate(sizes.begin(), sizes.end(), cell_size_type(0));
    return count==labels.size() && count==ranges.size();
}

// cell_labels_and_gids methods
cell_labels_and_gids::cell_labels_and_gids(cell_label_range lr, std::vector<cell_gid_type> gid):
    label_range(std::move(lr)), gids(std::move(gid)) {
    if (label_range.sizes.size()!=gids.size()) throw arbor_internal_error("cell_label_range and gid count mismatch");
}

void cell_labels_and_gids::append(cell_labels_and_gids other) {
    label_range.append(other.label_range);
    gids.insert(gids.end(), make_move_iterator(other.gids.begin()), make_move_iterator(other.gids.end()));
}

bool cell_labels_and_gids::check_invariant() const {
    return label_range.check_invariant() && label_range.sizes.size()==gids.size();
}

/* The n-th local item (by index) to its identifier (lid). The lids are organised
   in potentially discontiguous ranges.

   idx --------           len0 <= idx < len0 + len1
                \
                 v
   | [s0, e0) [s1, e1), ... [sk, ek), ... |
       len0     len1
*/
cell_lid_type label_resolution_map::range_set::at(unsigned idx) const {
    if (idx >= size) throw arbor_internal_error("invalid lid");
    for (const auto& [beg, end]: ranges) {
        auto len = end - beg;
        if (idx < len) return idx + beg;
        idx -= len;
    }
    ARB_UNREACHABLE;
}

const label_resolution_map::range_set& label_resolution_map::at(cell_gid_type gid, hash_type hash) const { return map.at(Key{gid, hash}); }

std::size_t label_resolution_map::count(cell_gid_type gid, hash_type hash) const { return map.count(Key{gid, hash}); }

label_resolution_map::label_resolution_map(const cell_labels_and_gids& clg) {
    arb_assert(clg.label_range.check_invariant());
    const auto& gids = clg.gids;
    const auto& labels = clg.label_range.labels;
    const auto& ranges = clg.label_range.ranges;
    const auto& sizes = clg.label_range.sizes;

    map.reserve(labels.size());
    auto lo = 0;
    Key key;
    for (auto idx: util::count_along(gids)) {
        auto gid = gids[idx];
        auto size = sizes[idx];
        key.f = gid;
        for (auto label_idx: util::make_span(lo, lo + size)) {
            auto range = ranges[label_idx];
            key.s = labels[label_idx];
            auto& range_set = map[key];
            range_set.ranges.push_back(range);
            range_set.size += range.end - range.begin;
        }
        lo += size;
    }
}

cell_lid_type resolver::resolve(const cell_global_label_type& iden) { return resolve(iden.gid, iden.label); }

cell_lid_type resolver::resolve(cell_gid_type gid, const cell_local_label_type& label) {
    const auto& [tag, pol] = label;
    auto hash = hash_value(tag);
    auto it = label_map_->find(gid, hash);
    if (it == label_map_->end()) throw arb::bad_connection_label(gid, tag, "label does not exist");
    auto rs_size = it->second.size;
    if (rs_size <= 0) throw arb::bad_connection_label(gid, tag, "no valid lids");
    if (pol == lid_selection_policy::assert_univalent) {
        // must have single-entry range_set
        if (rs_size != 1) throw arb::bad_connection_label(gid, tag, "range is not univalent");
        return it->second.at(0);
    }
    else if (pol == lid_selection_policy::round_robin) {
        // cycle through range_set
        auto idx = rr_state_map_[gid][hash];
        rr_state_map_[gid][hash] = (idx + 1) % rs_size;
        return it->second.at(idx);
    }
    else if (pol == lid_selection_policy::round_robin_halt) {
        // use previous state of round_robin policy
        auto idx = rr_state_map_[gid][hash];
        return it->second.at(idx);
    }
    ARB_UNREACHABLE;
}
} // namespace arb
