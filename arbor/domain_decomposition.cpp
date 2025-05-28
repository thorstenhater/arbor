#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/domdecexcept.hpp>
#include <arbor/recipe.hpp>

#include "execution_context.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"
#include "util/span.hpp"

#include <ankerl/unordered_dense.h>

namespace arb {
domain_decomposition::domain_decomposition(const recipe& rec,
                                           context ctx,
                                           const std::vector<group_description>& groups) {
    const auto* dist = ctx->distributed.get();
    unsigned num_domains = dist->size();
    int domain_id = dist->id();
    const auto num_global_cells = rec.num_cells();
    const bool has_gpu = ctx->gpu->has_gpu();

    std::vector<cell_gid_type> local_gids;
    for (const auto& g: groups) {
        if (g.backend == backend_kind::gpu && !has_gpu) throw invalid_backend(domain_id);
        if (g.backend == backend_kind::gpu && g.kind != cell_kind::cable) throw incompatible_backend(domain_id, g.kind);

        ankerl::unordered_dense::set<cell_gid_type> gid_set(g.gids.begin(), g.gids.end());
        for (const auto& gid: g.gids) {
            if (gid >= num_global_cells) throw out_of_bounds(gid, num_global_cells);
            for (const auto& gj: rec.gap_junctions_on(gid)) {
                if (!gid_set.count(gj.peer.gid)) throw invalid_gj_cell_group(gid, gj.peer.gid);
            }
            local_gids.push_back(gid);
        }
    }
    cell_size_type num_local_cells = local_gids.size();

    auto global_gids = dist->gather_gids(local_gids);
    if (global_gids.size() != num_global_cells) throw invalid_sum_local_cells(global_gids.size(), num_global_cells);

    // SANITY checks
    // for a given cell count N:
    // 1. check that we have N gids in the list
    // 2. sum up gids, which must run 0..N-1, thus the sum S must match
    //    N*(N-1)/2 via Gauss'.
    // If any check fails, try to find the misbehaving gid.
    {
        const auto& gids = global_gids.values();
        auto N = std::size_t(num_global_cells);
        auto S = std::size_t(0);
        for (auto gid: gids) S += gid;
        auto expected = N*(N - 1)/2;
        if ((N != num_global_cells) || (S != expected)) {
            ankerl::unordered_dense::set<cell_gid_type> seen;
            seen.reserve(gids.size());
            for (auto gid: gids) {
                if (!seen.insert(gid).second) throw duplicate_gid(gid);
            }
            throw arbor_internal_error{util::pprintf("Unknown gid error occured. size {} /= {} OR gid_sum {} /= {}", N, num_global_cells, S, expected)};
        }
    }

    num_domains_ = num_domains;
    domain_id_ = domain_id;
    num_local_cells_ = num_local_cells;
    num_global_cells_ = num_global_cells;
    groups_ = groups;

    // write tables gid -> (rank, offset)
    gid_domain_.resize(global_gids.size());
    gid_index_.resize(global_gids.size());
    auto rank_part = util::partition_view(global_gids.partition());
    for (auto rank: util::count_along(rank_part)) {
        cell_size_type index_on_domain = 0;
        for (auto gid: util::subrange_view(global_gids.values(), rank_part[rank])) {
            gid_domain_[gid] = rank;
            gid_index_[gid] = index_on_domain;
            ++index_on_domain;
        }
    }
}

int domain_decomposition::num_domains() const { return num_domains_; }
int domain_decomposition::domain_id() const { return domain_id_; }
cell_size_type domain_decomposition::num_local_cells() const { return num_local_cells_; }
cell_size_type domain_decomposition::num_global_cells() const { return num_global_cells_; }
cell_size_type domain_decomposition::num_groups() const { return groups_.size(); }
const std::vector<group_description>& domain_decomposition::groups() const { return groups_; }

const group_description& domain_decomposition::group(unsigned idx) const {
    arb_assert(idx<num_groups());
    return groups_[idx];
}

} // namespace arb

