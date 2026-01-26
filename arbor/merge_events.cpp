#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>

#include <arbor/common_types.hpp>

#include "merge_events.hpp"
#include "util/tourney_tree.hpp"

namespace arb {

// k-way linear merge:
// Pick stream with the minimum element, pop that and push into output.
// Repeat.
void ARB_ARBOR_API linear_merge_events(std::vector<event_span>& sources, pse_vector& out) {
    // Consume all events.
    for (;;) {
        // Now find the minimum
        auto mevt =  spike_event{0, terminal_time, 0};
        auto midx = -1;
        for (auto idx = 0ul; idx < sources.size(); ++idx) {
            auto& source = sources[idx];
            if (!source.empty()) {
                auto& evt = source.front();
                if (evt < mevt) {
                    mevt = evt;
                    midx = idx;
                }
            }
        }
        if (midx == -1) break;
        // Take event: bump chosen stream and stuff event into output.
        sources[midx].left++;
        out.emplace_back(mevt);
    }
}

template <typename T>
std::vector<T> merge_pair(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    std::vector<T> res;
    res.reserve(lhs.size() + rhs.size());
    std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(res));
    return res;
}
    
// merge streams in pairs
void ARB_ARBOR_API pairwise_merge_events(std::vector<event_span>& sources, pse_vector& out) {
    // spans are lightweight to sort and manipulate
    std::erase_if(sources, [](auto& src) { return src.empty(); });
    if (sources.empty()) return;
    std::sort(sources.begin(), sources.end(),
              [](auto& a, auto& b) { return a.size() < b.size(); });

    std::vector<std::vector<spike_event>> tmp;
    for(auto& src: sources) tmp.emplace_back(src.begin(), src.end());

    while (tmp.size() > 1) {
        std::vector<std::vector<spike_event>> next;
        for (size_t ix = 0; ix + 1 < tmp.size(); ix += 2) {
            next.emplace_back(merge_pair(tmp[ix], tmp[ix + 1]));
        }
        if (tmp.size() & 1) next.emplace_back(std::move(tmp.back()));
        tmp = std::move(next);
    }
    out.reserve(out.size() + tmp.size());
    for(const auto& evt: tmp.back()) out.push_back(evt);
}


// priority-queue based merge.
void ARB_ARBOR_API pqueue_merge_events(std::vector<event_span>& sources, pse_vector& out) {
    // Min heap tracking the minimum element from each span
    using kv_type = std::pair<spike_event, int>;
    std::priority_queue<kv_type, std::vector<kv_type>, std::greater<>> heap;

    // Add the first element from each sorted vector to the min heap
    for (std::size_t ix = 0; ix < sources.size(); ++ix) {
        auto& source = sources[ix];
        if (source.empty()) continue;
        heap.emplace(source.front(), ix);
        source.left++;
    }

    // Merge by continually popping the minimum element from the min heap
    while (!heap.empty()) {
        auto [value, ix] = heap.top();
        heap.pop();
        out.emplace_back(value);

        // If the sorted vector from which the minimum element was taken still
        // has elements, add the next smallest element to the heap
        auto& source = sources[ix];
        if (source.empty()) continue;
        heap.emplace(source.front(), ix);
        source.left++;
    }
}

void ARB_ARBOR_API merge_events(std::vector<event_span>& sources, pse_vector &out, std::size_t n_evts) {
    out.reserve(out.size() + n_evts);
    auto n_queues = sources.size();
    // pqueue_merge_events(sources, out);
    // if (n_queues < 20) { // NOTE: MAGIC NUMBER, found by ubench/merge
        linear_merge_events(sources, out);
    // }
    // else {
    // }
}

void ARB_ARBOR_API merge_events(std::vector<event_span>& sources, pse_vector &out) {
    // Count events, bail if none; else allocate enough space to store them.
    auto n_evts = std::accumulate(sources.begin(), sources.end(),
                                  0,
                                  [] (auto acc, const auto& rng) { return acc + rng.size(); });
    merge_events(sources, out, n_evts);
}


} // namespace arb
