#pragma once

#include <arbor/arbexcept.hpp>
#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>

#include "backends/common_types.hpp"
#include "execution_context.hpp"
#include "memory/memory.hpp"
#include "memory/gpu_copy.hpp"
#include "util/span.hpp"

#include "backends/threshold_crossing.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/stack.hpp"

#include "stack.hpp"

namespace arb {
namespace gpu {

// CUDA implementation entry points; all take the owning cell group's stream
// (see backends/gpu/stream.hpp), on which their launches must be enqueued.

void test_thresholds_impl(
    stream_type stream,
    int size,
    arb_value_type t_after,
    arb_value_type t_before,
    const arb_index_type* src_to_spike,
    arb_value_type* time_since_spike,
    stack_storage<threshold_crossing>& stack,
    arb_index_type* is_crossed,
    arb_value_type* prev_values,
    const arb_index_type* cv_index,
    const arb_value_type* values,
    const arb_value_type* thresholds,
    bool record);

void reset_crossed_impl(
    stream_type stream,
    int size,
    arb_index_type* is_crossed,
    const arb_index_type* cv_index,
    const arb_value_type* values,
    const arb_value_type* thresholds);


class threshold_watcher {
public:
    using stack_type = stack<threshold_crossing>;

    threshold_watcher() = default;
    threshold_watcher(threshold_watcher&& other) = default;
    threshold_watcher& operator=(threshold_watcher&& other) = default;

    threshold_watcher(const execution_context& ctx): stack_(ctx.gpu) {}

    threshold_watcher(const arb_size_type num_cv,
                      const arb_index_type* src_to_spike,
                      const fvm_detector_info& info):
        threshold_watcher{num_cv, src_to_spike, info.cv, info.threshold, info.ctx}
    {}

    threshold_watcher(const arb_size_type num_cv,
                      const arb_index_type* src_to_spike,
                      const std::vector<arb_index_type>& cv_index,
                      const std::vector<arb_value_type>& thresholds,
                      const execution_context& context):
        src_to_spike_(src_to_spike),
        n_detectors_(cv_index.size()),
        cv_index_(memory::make_const_view(cv_index)),
        is_crossed_(n_detectors_),
        thresholds_(memory::make_const_view(thresholds)),
        v_prev_(num_cv),
        // TODO: allocates enough space for 10 spikes per watch.
        // A more robust approach might be needed to avoid overflows.
        stack_(100*size(), context.gpu)
    {
        crossings_.reserve(stack_.capacity());
        // reset() needs to be called before this is ready for use
    }

    /// Remove all stored crossings that were detected in previous calls to test()
    /// Stream parameters order the stack copies/kernels on the owning cell
    //  group's GPU stream; the default (legacy stream) preserves the
    /// behaviour for direct users outside a cell group (unit tests etc.).
    void clear_crossings(stream_type s = null_stream) {
        stack_.update_host(s);
        stack_.clear(s);
    }

    /// Reset state machine for each detector.
    /// Assume that the values in values_ have been set correctly before
    /// calling, because the values are used to determine the initial state
    void reset(const array& values, stream_type s = null_stream) {
        values_ = values.data();
        memory::copy_async(s, values, v_prev_);
        clear_crossings(s);
        if (size()>0) {
            reset_crossed_impl(s, (int)size(), is_crossed_.data(), cv_index_.data(), values_, thresholds_.data());
        }
    }

    // Testing-only interface.
    bool is_crossed(int i) const {
        return is_crossed_[i];
    }

    const std::vector<threshold_crossing>& crossings(stream_type s = null_stream) const {
        stack_.update_host(s);

        if (stack_.overflow()) {
            throw arbor_internal_error("gpu/threshold_watcher: gpu spike buffer overflow");
        }

        crossings_.clear();
        crossings_.insert(crossings_.end(), stack_.begin(), stack_.end());
        return crossings_;
    }

    /// Tests each target for changed threshold state.
    /// Crossing events are recorded for each threshold that has been
    /// crossed since current time t, and the last time the test was
    /// performed.
    void test(array& time_since_spike, const arb_value_type& t_before, const arb_value_type& t_after, stream_type s = null_stream) {
        arb_assert(values_);

        if (size()>0) {
            test_thresholds_impl(
                s,
                (int)size(),
                t_after, t_before,
                src_to_spike_, time_since_spike.data(),
                stack_.storage(),
                is_crossed_.data(), v_prev_.data(),
                cv_index_.data(), values_, thresholds_.data(),
                !time_since_spike.empty());

            // Check that the number of spikes has not exceeded capacity.
            arb_assert(!stack_.overflow());
        }
    }

    /// the number of threshold values that are being monitored
    std::size_t size() const {
        return cv_index_.size();
    }

private:
    // Non-owning pointers
    const arb_value_type* values_ = nullptr;
    const arb_index_type* src_to_spike_ = nullptr;

    // Threshold watch state, with data on gpu:
    arb_size_type n_detectors_ = 0;
    iarray cv_index_;           // Compartment indexes of values to watch.
    iarray is_crossed_;         // Boolean flag for state of each watch.
    array thresholds_;          // Threshold for each watch.
    array v_prev_;              // Values at previous sample time.

    // Hybrid host/gpu data structure for accumulating threshold crossings.
    mutable stack_type stack_;

    // host side storage for the crossings
    mutable std::vector<threshold_crossing> crossings_;
};

} // namespace gpu
} // namespace arb
