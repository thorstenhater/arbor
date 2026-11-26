#pragma once

#include <util/partition.hpp>
#include <util/span.hpp>
#include <arbor/simd/simd.hpp>

#include <memory/memory.hpp>

#include "multicore_common.hpp"

namespace arb {
namespace multicore {

namespace {
    inline void
    do_solve(arb_index_type const * const ARB_NO_ALIAS s_,
             arb_index_type const ncells,
             arb_value_type       * const ARB_NO_ALIAS r_,
             arb_value_type const dt,
             arb_value_type       * const ARB_NO_ALIAS d_,
             arb_value_type const * const ARB_NO_ALIAS i_,
             arb_value_type const * const ARB_NO_ALIAS inv_,
             arb_value_type const * const ARB_NO_ALIAS c_,
             arb_value_type const * const ARB_NO_ALIAS g_,
             arb_value_type const * const ARB_NO_ALIAS a_,
             arb_value_type const * const ARB_NO_ALIAS u_,
             arb_index_type const * const ARB_NO_ALIAS p_) {
        const arb_value_type oodt = 1e-3/dt;         // [1/µs]
        for (int idx = 0; idx < ncells; ++idx) {
            auto lo = s_[idx];
            auto hi = s_[idx + 1];
            // Assemble; loop over submatrices
            // Afterwards the diagonal and RHS will have been set given dt, voltage and current.
            //   dt              [ms]
            //   voltage         [mV]      (per control volume)
            //   current density [A.m^-2]  (per control volume)
            //   conductivity    [kS.m^-2] (per control volume)
            for(int ix = lo; ix < hi; ++ix) {
                const auto area = 1e-3*a_[ix];                // [1e-9·m²]
                const auto gi   = oodt*c_[ix] + area*g_[ix];  // [μS]
                d_[ix] = gi + inv_[ix];                       // [μS]
                r_[ix] = gi*r_[ix] - area*i_[ix];             // [nA]
            }
        }
        for (int idx = 0; idx < ncells; ++idx) {
            auto lo = s_[idx];
            auto hi = s_[idx + 1];
            if (d_[lo] == 0) continue;
            // Solve system
            // backward sweep
            for(int ix = hi - 1; ix > lo; --ix) {
                const auto factor = u_[ix] / d_[ix];
                const auto pi     = p_[ix];
                d_[pi] -= factor * u_[ix];
                r_[pi] -= factor * r_[ix];
            }
        }
        for (int idx = 0; idx < ncells; ++idx) {
            auto lo = s_[idx];
            auto hi = s_[idx + 1];
            if (d_[lo] == 0) continue;
            // solve root
            r_[lo] /= d_[lo];
            // forward sweep
            for(int ix = lo + 1; ix < hi; ++ix) {
                r_[ix] -= u_[ix] * r_[p_[ix]];
                r_[ix] /= d_[ix];
            }
        }
    }

    inline void
    do_solve_simd(arb_index_type const * const ARB_NO_ALIAS s_,
                  arb_index_type const ncells,
                  arb_value_type       * const ARB_NO_ALIAS r_,
                  arb_value_type const dt,
                  arb_value_type       * const ARB_NO_ALIAS d_,
                  arb_value_type const * const ARB_NO_ALIAS i_,
                  arb_value_type const * const ARB_NO_ALIAS inv_,
                  arb_value_type const * const ARB_NO_ALIAS c_,
                  arb_value_type const * const ARB_NO_ALIAS g_,
                  arb_value_type const * const ARB_NO_ALIAS a_,
                  arb_value_type const * const ARB_NO_ALIAS u_,
                  arb_index_type const * const ARB_NO_ALIAS p_) {
        namespace abi = simd::simd_abi;
        constexpr auto simd_width = abi::native_width<arb_value_type>::value;
        using simd_value = simd::simd<arb_value_type, simd_width, abi::native>;
        using simd_index = simd::simd<arb_index_type, simd_width, abi::native>;
        using simd::simd_cast;

        auto to_simd_index = [](arb_index_type const * const ARB_NO_ALIAS base, int idx) {
            return simd_cast<simd_index>(simd::indirect(base + idx, simd_width));
        };
        auto gather = [](arb_value_type const * const ARB_NO_ALIAS base, simd_index idx) {
            return simd_cast<simd_value>(simd::indirect(base, idx, simd_width));
        };
        auto igather = [](arb_index_type const * const ARB_NO_ALIAS base, simd_index idx) {
            return simd_cast<simd_index>(simd::indirect(base, idx, simd_width));
        };
        auto scatter = [](arb_value_type * const ARB_NO_ALIAS base, simd_index idx, simd_value val) {
            simd::indirect(base, idx, simd_width) = val;
        };

        auto ones = simd_cast<simd_index>(1);
        auto scale = simd_cast<simd_value>(1e-3);

        const auto voodt = simd_cast<simd_value>(1e-3/dt);
        int idx = 0;
        while (idx + simd_width < ncells) { // we can do a full lane still
            int max_count = 0;
            for (int ix = idx; ix < idx + simd_width; ++ix) {
                auto len = s_[ix + 1] - s_[ix];
                if (len > max_count) max_count = len;
            }

            auto lo = to_simd_index(s_,     idx);
            auto hi = to_simd_index(s_ + 1, idx);

            auto cur = lo;
            for(int ix = 0; ix < max_count; ++ix) {
                auto msk  = simd::cmp_lt(cur, hi);
                auto a    = gather(a_, cur);
                auto g    = gather(g_, cur);
                auto c    = gather(c_, cur);
                auto r    = gather(r_, cur);
                auto d    = gather(d_, cur);
                auto i    = gather(i_, cur);
                auto inv  = gather(inv_, cur);
                auto area = simd::mul(scale, a);
                auto gi   = simd::add(simd::mul(voodt, c), simd::mul(area, g));
                simd::where(msk, d) = simd::add(gi, inv);
                scatter(d_, cur, d);
                simd::where(msk, r) = simd::sub(simd::mul(gi, r), simd::mul(area, i));
                scatter(r_, cur, r);
                simd::where(msk, cur) = simd::add(cur, ones);
            }
            // backward sweep
            cur = simd::sub(hi, ones);
            for(int ix = 0; ix < max_count; ++ix) {
                auto msk = simd::cmp_gt(cur, lo);
                auto u   = gather(u_, cur);
                auto d   = gather(d_, cur);
                auto fac = simd::div(u, d);
                auto r   = gather(r_, cur);
                auto pi  = igather(p_, cur);
                auto dp  = gather(d_, pi);
                auto rp  = gather(r_, pi);
                simd::where(msk, dp) = simd::sub(dp, simd::mul(fac, u));
                simd::where(msk, rp) = simd::sub(rp, simd::mul(fac, r));
                scatter(d_, pi, dp);
                scatter(r_, pi, rp);
                simd::where(msk, cur) = simd::sub(cur, ones);
            }
            // solve root
            auto d = gather(d_, lo);
            auto r = gather(r_, lo);
            r = simd::div(r, d);
            scatter(r_, lo, r);
            // forward sweep
            cur = simd::add(lo, ones);
            for(int ix = 0; ix < max_count; ++ix) {
                auto msk = simd::cmp_lt(cur, hi);
                auto u   =  gather(u_, cur);
                auto d   =  gather(d_, cur);
                auto r   =  gather(r_, cur);
                auto pi  = igather(p_, cur);
                auto rp  =  gather(r_, pi);
                simd::where(msk, r) = simd::div(simd::sub(r, simd::mul(u, rp)), d);
                scatter(r_, cur, r);
                simd::where(msk, cur) = simd::add(cur, ones);
            }
            idx += simd_width;
        } // while (idx < ncells)
        const arb_value_type oodt = 1e-3/dt;         // [1/µs]
        while (idx < ncells) {
            auto lo = s_[idx];
            auto hi = s_[idx + 1];
            // Assemble; loop over submatrices
            // Afterwards the diagonal and RHS will have been set given dt, voltage and current.
            //   dt              [ms]
            //   voltage         [mV]      (per control volume)
            //   current density [A.m^-2]  (per control volume)
            //   conductivity    [kS.m^-2] (per control volume)
            for (int ix = lo; ix < hi; ++ix) {
                const auto area = 1e-3 * a_[ix];                 // [1e-9·m²]
                const auto gi   = oodt * c_[ix] + area * g_[ix]; // [μS]
                d_[ix]          = gi + inv_[ix];                 // [μS]
                r_[ix]          = gi * r_[ix] - area * i_[ix];   // [nA]
            }
            // Solve system
            // backward sweep
            for (int ix =  hi - 1; ix > lo; --ix) {
                const auto factor = u_[ix] / d_[ix];
                const auto pi     = p_[ix];
                d_[pi] -= factor * u_[ix];
                r_[pi] -= factor * r_[ix];
            }
            // solve root
            r_[lo] /= d_[lo];
            // forward sweep
            for (int ix = lo + 1; ix < hi; ++ix) {
                r_[ix] -= u_[ix] * r_[p_[ix]];
                r_[ix] /= d_[ix];
            }
            idx += 1;
        }
    }
} // namespace

struct cable_solver {
    using value_type = arb_value_type;
    using index_type = arb_index_type;
    using array      = padded_vector<value_type>;
    using const_view = const array&;
    using iarray     = padded_vector<index_type>;

    iarray parent_index;
    iarray cell_cv_divs;

    array d;              // [μS]
    array u;              // [μS]
    array cv_capacitance; // [pF]
    array invariant_d;    // [μS] invariant part of matrix diagonal

    cable_solver() = default;
    cable_solver(const cable_solver&) = default;
    cable_solver(cable_solver&&) = default;

    cable_solver& operator=(const cable_solver&) = default;
    cable_solver& operator=(cable_solver&&) = default;

    cable_solver(const std::vector<index_type>& p,
                 const std::vector<index_type>& cell_part,
                 const std::vector<value_type>& cap,
                 const std::vector<value_type>& cond):
        parent_index(p.begin(), p.end()),
        d(size(), 0), u(size(), 0),
        cv_capacitance(cap.begin(), cap.end()),
        invariant_d(size(), 0)
    {
        // TODO We really should push this up to FVM layout, no?
        if (cell_part.size() > 0) {
            cell_cv_divs.push_back(cell_part[0]);
            for (std::size_t ix = 1; ix < cell_part.size(); ++ix) {
                auto p = cell_part[ix];
                if (cell_cv_divs.back() < p) cell_cv_divs.push_back(p);
            }
        }

        // Sanity check
        arb_assert(cap.size() == size());
        arb_assert(cond.size() == size());
        arb_assert(cell_cv_divs.back() == (index_type)size());

        // Build invariant parts
        if (size() >= 1) {
            for (auto i: util::make_span(1u, size())) {
                const auto gij = cond[i];
                u[i] = -gij;
                invariant_d[i] += gij;
                if (auto pi= p[i]; pi != -1) invariant_d[pi] += gij;
            }
        }
    }

    // Setup and solve the cable equation
    // * expects the voltage from its first argument
    // * will likewise overwrite the first argument with the solution
    template<typename T>
    void solve(T& rhs, const value_type dt, const_view current, const_view conductivity, const_view cv_area) {
#ifdef  ARB_VECTORIZE_ENABLED
        do_solve_simd(cell_cv_divs.data(),
                      cell_cv_divs.size() - 1,
                      rhs.data(),
                      dt,
                      d.data(),
                      current.data(),
                      invariant_d.data(),
                      cv_capacitance.data(),
                      conductivity.data(),
                      cv_area.data(),
                      u.data(),
                      parent_index.data());
#else
        do_solve(cell_cv_divs.data(),
                 cell_cv_divs.size() - 1,
                 rhs.data(),
                 dt,
                 d.data(),
                 current.data(),
                 invariant_d.data(),
                 cv_capacitance.data(),
                 conductivity.data(),
                 cv_area.data(),
                 u.data(),
                 parent_index.data());
#endif
    }

    // Solve; loop over submatrices
    // Afterwards rhs will contain the solution.
    // NOTE: This exists separately only to cater to the tests
    template<typename T>
    void solve(T& rhs) {
        value_type * const ARB_NO_ALIAS r_ = rhs.data();
        value_type * const ARB_NO_ALIAS d_ = d.data();

        const value_type * const ARB_NO_ALIAS u_ = u.data();
        const index_type * const ARB_NO_ALIAS p_ = parent_index.data();

        const auto cell_cv_part = util::partition_view(cell_cv_divs);
        for (const auto& [first, last]: cell_cv_part) {
            // skip vacuous cells
            if (d_[first] == 0) continue;
            // backward sweep
            for(int ix = last - 1; ix > first; --ix) {
                const auto factor = u_[ix] / d_[ix];
                const auto pi = p_[ix];
                d_[pi] -= factor * u_[ix];
                r_[pi] -= factor * r_[ix];
            }
            // solve root
            r_[first] /= d_[first];
            // forward sweep
            for(int ix = first + 1; ix < last; ++ix) {
                r_[ix] -= u_[ix] * r_[p_[ix]];
                r_[ix] /= d_[ix];
            }
        }
    }

    std::size_t num_cells() const { return cell_cv_divs.size() - 1; }
    std::size_t size() const { return parent_index.size(); }
};

} // namespace multicore
} // namespace arb
