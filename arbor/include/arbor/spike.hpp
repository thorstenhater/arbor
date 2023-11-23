#pragma once

#include <arbor/cable_cell_param.hpp>
#include <ostream>
#include <type_traits>
#include <functional>

#include <arbor/common_types.hpp>
#include <arbor/serdes.hpp>

namespace arb {

template <typename I>
struct basic_spike {
    using id_type = I;

    id_type source = id_type{};
    time_type time = -1;

    basic_spike() = default;

    basic_spike(id_type s, time_type t):
        source(std::move(s)), time(t)
    {}

    ARB_SERDES_ENABLE(basic_spike<I>, source, time);
};

/// Standard specialization:
using spike = basic_spike<cell_member_type<threshold_detector>>;

using spike_predicate = std::function<bool(const spike&)>;

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(spike, (a.source, a.time), (b.source, b.time));

// Custom stream operator for printing arb::spike<> values.
template <typename I>
std::ostream& operator<<(std::ostream& o, basic_spike<I> const& s) {
    return o << "S[src " << s.source << ", t " << s.time << "]";
}

} // namespace arb
