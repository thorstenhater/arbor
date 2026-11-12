#include <arbor/fvm_types.hpp>
#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

#include "matrix_common.hpp"
#include "matrix_fine.hpp"

namespace arb {
namespace gpu {
namespace kernels {

/// GPU implementation of Hines matrix assembly.
/// Fine layout.
/// For a given time step size dt:
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve.
template <typename T, typename I>
__global__
void assemble_matrix_fine(
        T* __restrict__ const d,
        T* __restrict__ const rhs,
        const T* __restrict__ const invariant_d,
        const T* __restrict__ const voltage,
        const T* __restrict__ const current,
        const T* __restrict__ const conductivity,
        const T* __restrict__ const cv_capacitance,
        const T* __restrict__ const area,
        const T dt,
        const I* __restrict__ const perm,
        unsigned n)
{
    const unsigned tid = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid < n) {
        // The 1e-3 is a constant of proportionality required to ensure that the
        // conductance (gi) values have units μS (micro-Siemens).
        // See the model documentation in docs/model for more information.
        const auto pid = perm[tid];
        const auto area_factor = T(1e-3)*area[tid];
        const auto gi = T(1e-3)*cv_capacitance[tid]/dt + area_factor*conductivity[tid];
        d[pid] = gi + invariant_d[tid];
        rhs[pid] = gi*voltage[tid] - area_factor*current[tid];
    }
}

/// GPU implementation of Hines Matrix solver.
/// Fine-grained tree based solver.
/// Each block solves a set of matricesb iterating over the levels of matrix
/// and perfoming a backward and forward substitution. On each level one thread
/// gets assigned to one branch on this level of a matrix and solves and
/// performs the substitution. Afterwards all threads continue on the next
/// level.
/// To avoid idle threads, one should try that on each level, there is a similar
/// number of branches.
///
/// The backward pass merges a branch's final addend into its parent branch
/// without the unordered atomic accumulation that made results depend on
/// thread arrival order (Arbor issue #1059). Levels where every parent has
/// at most one contributor merge directly, which cannot race. All other
/// levels stage the addends in shared memory, then one thread per parent
/// branch folds its whole contributor list in a fixed order given by a
/// per-level CSR table (contributors sorted by ascending start CV index).
/// Merged sums therefore depend only on the matrix, not on scheduling nor on
/// how cells were packed into blocks.
template <typename T>
__global__
void solve_matrix_fine(
    T* __restrict__ const rhs,
    T* __restrict__ const d,
    const T* __restrict__ const u,
    const level_metadata* __restrict__ const level_meta,
    const arb_index_type* __restrict__ const level_lengths,
    const arb_index_type* __restrict__ const level_parents,
    const arb_index_type* __restrict__ const block_index,
    const arb_index_type* __restrict__ const num_matrix, // number of packed matrices = number of cells
    const arb_index_type* __restrict__ const merge_data) // CSR tables, one per non-root level
{
    // Upper bound on level width: max_branches_per_level, hard-coded at
    // construction (matrix_state_fine.hpp), equals the launch block size.
    constexpr unsigned max_branches_per_block = 128u;
    // Parent-merge addends staged by the backward pass, indexed by the
    // contributing branch's column index (its tid) on the current level.
    __shared__ T s_add_d[max_branches_per_block];
    __shared__ T s_add_rhs[max_branches_per_block];

    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    const auto first_level = block_index[bid];
    const auto num_levels  = block_index[bid + 1] - first_level;

    const auto block_level_meta = &level_meta[first_level];

    // backward substitution

    for (unsigned l=0; l<num_levels-1; ++l) {
        // Metadata for this level and the next level
        const auto& lvl_meta = block_level_meta[l];
        const auto& next_lvl_meta = block_level_meta[l+1];

        // Addresses of the first elements of level_lengths and level_parents
        // that belong to this level
        const auto lvl_lengths = level_lengths + lvl_meta.level_data_index;
        const auto lvl_parents = level_parents + lvl_meta.level_data_index;

        const unsigned width = lvl_meta.num_branches;

        // A level with a CSR table stages its merge addends in shared memory
        // and folds them in phase B. A level without one (no_merge_csr) has,
        // by construction, at most one contributor per parent branch, so
        // phase A may merge into the parent directly: no two threads touch
        // the same slot, hence no ordering ambiguity and no second barrier.
        const unsigned merge_index = lvl_meta.merge_data_index;

        // ---- Phase A: Perform backward substitution for each branch on this
        // level. One thread per branch.
        if (tid < width) {
            const unsigned len = lvl_lengths[tid];
            unsigned pos = lvl_meta.matrix_data_index + tid;

            // Zero diagonal term implies dt==0; just leave rhs (for whole matrix)
            // alone in that case.

            // Each cell has a different `dt`, because we choose time step size
            // according to when the next event is arriving at a cell. So, some
            // cells require more time steps than others, but we have to solve
            // all the matrices at the same time. When a cell finishes, we put a
            // `0` on the diagonal to mark that it should not be solved for.
            if (d[pos]!=0) {
                // each branch perform substitution
                for (unsigned i=0; i<len-1; ++i) {
                    const unsigned next_pos = pos + width;
                    const auto d_next = d[next_pos];
                    const auto rhs_next = rhs[next_pos];
                    const T factor = -u[pos]/d[pos];
                    d[next_pos] = fma(factor, u[pos], d_next);
                    rhs[next_pos] = fma(factor, rhs[pos], rhs_next);
                    pos = next_pos;
                }
                // The update for the parent node of this branch.
                const T factor = -u[pos] / d[pos];
                const T add_d   = factor*u[pos];
                const T add_rhs = factor*rhs[pos];
                if (merge_index == no_merge_csr) {
                    // Single-contributor parent: merge in place.
                    const unsigned p = next_lvl_meta.matrix_data_index + lvl_parents[tid];
                    d[p]   += add_d;
                    rhs[p] += add_rhs;
                }
                else {
                    s_add_d[tid]   = add_d;
                    s_add_rhs[tid] = add_rhs;
                }
            }
            else if (merge_index != no_merge_csr) {
                // A skipped branch contributes nothing, and phase B adds
                // exactly what it finds staged here.
                s_add_d[tid]   = T(0);
                s_add_rhs[tid] = T(0);
            }
        }
        __syncthreads();

        // ---- Phase B: fold the staged addends into row 0 of each parent
        // branch. One thread per branch of the next level; each folds the
        // complete contributor list of its own branch in CSR order, so the
        // folded sums are reproducible by construction. Rows 0 of the next
        // level have no writer other than this fold (contributors no longer
        // touch them), so plain loads and stores are race free.
        if (merge_index != no_merge_csr) { // uniform across the block
            const unsigned next_width = next_lvl_meta.num_branches;
            if (tid < next_width) {
                const arb_index_type* starts = merge_data + merge_index;
                const arb_index_type src_begin = starts[tid];
                const arb_index_type src_end   = starts[tid+1];
                if (src_end > src_begin) {
                    const arb_index_type* srcs = starts + next_width + 1;
                    const unsigned p = next_lvl_meta.matrix_data_index + tid;
                    T acc_d   = d[p];
                    T acc_rhs = rhs[p];
                    for (arb_index_type j = src_begin; j < src_end; ++j) {
                        const unsigned s = (unsigned)srcs[j];
                        acc_d   += s_add_d[s];
                        acc_rhs += s_add_rhs[s];
                    }
                    d[p]   = acc_d;
                    rhs[p] = acc_rhs;
                }
            }
            // The next level's phase A reads d and rhs at its row 0 — the
            // slots this fold has just written — so this barrier must not be
            // skipped when a merge table exists. Like the one above, it
            // sits outside every tid guard.
            __syncthreads();
        }
    }

    // Solve the root
    {
        // The levels are sorted such that the root is the last level
        const auto& last_lvl_meta = block_level_meta[num_levels-1];
        const auto lvl_lengths = level_lengths + last_lvl_meta.level_data_index;

        const unsigned width = num_matrix[bid];

        if (tid < width) {
            const unsigned len = lvl_lengths[tid];
            unsigned pos = last_lvl_meta.matrix_data_index + tid;

            if (d[pos]!=0) {
                // backward
                for (unsigned i=0; i<len-1; ++i) {
                    const unsigned next_pos = pos + width;
                    const T factor = -u[pos] / d[pos];
                    const auto rhs_next = rhs[next_pos];
                    const auto d_next = d[next_pos];
                    d[next_pos]   = fma(factor, u[pos], d_next);
                    rhs[next_pos] = fma(factor, rhs[pos], rhs_next);
                    pos = next_pos;
                }

                auto rhsp = rhs[pos] / d[pos];
                rhs[pos] = rhsp;
                pos -= width;

                // forward
                for (unsigned i=0; i<len-1; ++i) {
                    rhsp = rhs[pos] - u[pos]*rhsp;
                    rhsp /= d[pos];
                    rhs[pos] = rhsp;
                    pos -= width;
                }
            }
        }
    }

    // forward substitution

    // take great care with loop limits decrementing unsigned counter l
    for (unsigned l=num_levels-1; l>0; --l) {
        const auto& lvl_meta = block_level_meta[l-1];

        // Addresses of the first elements of level_lengths and level_parents
        // that belong to this level
        const auto lvl_lengths = level_lengths + lvl_meta.level_data_index;
        const auto lvl_parents = level_parents + lvl_meta.level_data_index;

        const unsigned width = lvl_meta.num_branches;
        const unsigned parent_index = block_level_meta[l].matrix_data_index;

        __syncthreads();

        // Perform forward-substitution for each branch on this level.
        // One thread per branch.
        if (tid < width) {
            // Find the index of the first node in this branch.
            const unsigned len = lvl_lengths[tid];
            unsigned pos = lvl_meta.matrix_data_index + (len-1)*width + tid;

            if (d[pos]!=0) {
                // Load the rhs value for the parent node of this branch.
                const unsigned p = parent_index + lvl_parents[tid];
                T rhsp = rhs[p];
                // each branch perform substitution
                for (unsigned i=0; i<len; ++i) {
                    rhsp = rhs[pos] - u[pos]*rhsp;
                    rhsp /= d[pos];
                    rhs[pos] = rhsp;
                    pos -= width;
                }
            }
        }
    }
}

} // namespace kernels

ARB_ARBOR_API void assemble_matrix_fine(
    arb_value_type* d,
    arb_value_type* rhs,
    const arb_value_type* invariant_d,
    const arb_value_type* voltage,
    const arb_value_type* current,
    const arb_value_type* conductivity,
    const arb_value_type* cv_capacitance,
    const arb_value_type* area,
    const arb_value_type  dt,
    const arb_index_type* perm,
    unsigned n)
{
    launch_1d(n, 128, kernels::assemble_matrix_fine<arb_value_type, arb_index_type>,
        d, rhs, invariant_d, voltage, current, conductivity, cv_capacitance, area,
        dt, perm, n);
}

// Example:
//
//         block 0                  block 1              block 2
// .~~~~~~~~~~~~~~~~~~.  .~~~~~~~~~~~~~~~~~~~~~~~~.  .~~~~~~~~~~~ ~ ~
//
//  L0 \  /                                           L5    \  /
//      \/                                                   \/
//  L1   \   /   \   /    L3 \   /   \ | /   \   /    L6 \   /  . . .
//        \ /     \ /         \ /     \|/     \ /         \ /
//  L2     |       |      L4   |       |       |      L7   |
//         |       |           |       |       |           |
//
// levels       = [L0, L1, L2, L3, L4, L5, L6, L7, ... ]
// block_index  = [0, 3, 5, 8, ...]
// num_levels   = [3, 2, 3, ...]
// num_cells    = [2, 3, ...]
// num_blocks   = level_start.size() - 1 = num_levels.size() = num_cells.size()
ARB_ARBOR_API void solve_matrix_fine(
    arb_value_type* rhs,
    arb_value_type* d,                     // diagonal values
    const arb_value_type* u,               // upper diagonal (and lower diagonal as the matrix is SPD)
    const level_metadata* level_meta,      // information pertaining to each level
    const arb_index_type* level_lengths,   // lengths of branches of every level concatenated
    const arb_index_type* level_parents,   // parents of branches of every level concatenated
    const arb_index_type* block_index,     // start index into levels for each gpu block
    arb_index_type* num_cells,             // the number of cells packed into this single matrix
    arb_index_type* padded_size,           // length of rhs, d, u, including padding
    const arb_index_type* merge_data,      // parent-merge CSR tables, indexed by level_metadata::merge_data_index
    unsigned num_blocks,                   // number of blocks
    unsigned blocksize)                    // size of each block
{
    launch(num_blocks, blocksize, kernels::solve_matrix_fine<arb_value_type>,
        rhs, d, u, level_meta, level_lengths, level_parents, block_index,
        num_cells, merge_data);
}

} // namespace gpu
} // namespace arb
