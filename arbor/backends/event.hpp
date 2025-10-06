#pragma once

#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/serdes.hpp>
#include <arbor/mechanism_abi.h>

namespace arb {

struct deliverable_event {
    time_type time = 0;
    float weight = 0;
    target_handle handle;

    deliverable_event() = default;
    constexpr deliverable_event(time_type time,
                                target_handle handle,
                                float weight) noexcept:
        time(time), weight(weight), handle(std::move(handle)) {}

    ARB_SERDES_ENABLE(deliverable_event, time, weight, handle);
};

// Subset of event information required for mechanism delivery.
struct deliverable_event_data {
    cell_local_size_type mech_index = 0; // same as target_handle::mech_index
    float weight = 0;

    auto operator<=>(deliverable_event_data const&) const noexcept = default;

    deliverable_event_data(cell_local_size_type idx, float w) noexcept:
        mech_index(idx),
        weight(w) {}
    ARB_SERDES_ENABLE(deliverable_event_data,
                      mech_index,
                      weight);
};

// Delivery data accessor function for multi_event_stream:
inline arb_deliverable_event_data event_data(const deliverable_event& ev) {
    return {ev.handle.index, ev.weight};
}

inline arb_deliverable_event_stream make_event_stream_state(arb_deliverable_event_data* begin,
                                                            arb_deliverable_event_data* end) {
    return {begin, end};
}

// Sample events (raw values from back-end state).

using probe_handle = const arb_value_type*;

struct raw_probe_info {
    probe_handle handle;      // where the to-be-probed value sits
    sample_size_type offset;  // offset into array to store raw probed value
};

struct sample_event {
    time_type time;
    raw_probe_info raw;           // event payload: what gets put where on sample
};

inline raw_probe_info event_data(const sample_event& ev) {
    return ev.raw;
}

} // namespace arb
