#pragma once

#include <iostream>

#include <arbor/export.hpp>
#include <arbor/fvm_types.hpp>

namespace arb {
namespace gpu {


// Helper type for branch meta data in setup phase of fine grained
// matrix storage+solver.
//
//      leaf
//      .
//      .
//      .
//  -   *
//      |
//  l   *
//  e   |
//  n   *
//  g   |
//  t   *
//  h   |
//  -   start_idx
//      |
//      parent_idx
//      |
//      .
//      .
//      .
//      root
struct branch {
    unsigned id;         // branch id
    unsigned parent_id;  // parent branch id
    unsigned parent_idx; //
    unsigned start_idx;  // the index of the first node in the input parent index
    unsigned length;     // the number of nodes in the branch

    bool operator==(const branch&) const = default;
};

// order branches by:
//  - descending length
//  - ascending id
inline
bool operator<(const branch& lhs, const branch& rhs) {
    if (lhs.length!=rhs.length) {
        return lhs.length>rhs.length;
    } else {
        return lhs.id<rhs.id;
    }
}

inline
std::ostream& operator<<(std::ostream& o, branch b) {
    return o << "[" << b.id
        << ", len " << b.length
        << ", pid " << b.parent_idx
        << ", sta " << b.start_idx
        << "]";
}

// Value of level_metadata::merge_data_index when a level needs no
// parent-merge table: either no branch has a parent (the root level of a
// block), or every parent branch on the next level has at most one
// contributor on this level. In the latter case the solve_matrix_fine kernel
// merges the single addend directly, with no ordering ambiguity.
inline constexpr unsigned no_merge_csr = ~0u;

struct level_metadata {
    unsigned num_branches = 0; // Number of branches in a level
    unsigned max_length = 0;   // Length of the longest branch
    unsigned matrix_data_index = 0;   // Index into data values (d, u, rhs) of the first branch
    unsigned level_data_index  = 0;   // Index into data values (lengths, parents) of each level
    // Index into the parent-merge CSR table of the solve_matrix_fine kernel,
    // or no_merge_csr (see above). Where a table exists, it holds per level
    //   [num_branches(next level)+1 start offsets][contributor column index ...]
    // and the contributors of one parent branch are listed in ascending order
    // of their start index (a global CV index), so that folding the staged
    // addends in table order reproduces the merged sums bit for bit across
    // runs and under any regrouping of cells into blocks.
    unsigned merge_data_index = no_merge_csr;
};

// C wrappers around kernels
ARB_ARBOR_API void gather(
    const arb_value_type* from,
    arb_value_type* to,
    const arb_index_type* p,
    unsigned n);

ARB_ARBOR_API void scatter(
    const arb_value_type* from,
    arb_value_type* to,
    const arb_index_type* p,
    unsigned n);

} // gpu
} // arb
