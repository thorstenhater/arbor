#pragma once

#include <iosfwd>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/spike_event.hpp>

#include "profile/profiler_macro.hpp"
#include "util/range.hpp"

// Merge a collection of sorted event sequences into a sorted output sequence.

namespace arb {

using event_span = util::range<const spike_event*>;

void ARB_ARBOR_API linear_merge_events(std::vector<event_span>& sources, pse_vector& out);
void ARB_ARBOR_API pqueue_merge_events(std::vector<event_span>& sources, pse_vector& out);

void ARB_ARBOR_API merge_events(std::vector<event_span>& sources, pse_vector& out);

} // namespace arb
