#pragma once

// A move-only, lazily created GPU stream handle, used to run all device work
// of one cable cell group (kernels, copies, fills) on its own stream so that
// groups overlap on the device while each group's work stays strictly
// ordered (stream FIFO = submission order; see the stream-per-group design).
//
// Compiled only in GPU builds; the multicore backend has its own no-op stubs.

#include <utility>

#include <arbor/gpu/gpu_api.hpp>

namespace arb {
namespace gpu {

struct stream {
    stream_type handle_ = null_stream;

    stream() = default;

    stream(const stream&) = delete;
    stream& operator=(const stream&) = delete;

    stream(stream&& o) noexcept:
        handle_(std::exchange(o.handle_, null_stream))
    {}

    stream& operator=(stream&& o) noexcept {
        if (this != &o) {
            reset();
            handle_ = std::exchange(o.handle_, null_stream);
        }
        return *this;
    }

    ~stream() {
        // Ignore API errors: we are in a destructor; at teardown a CUDA/HIP
        // error here is fatal anyway.
        if (handle_ != null_stream) {
            stream_synchronize(handle_);
            destroy_stream(handle_);
            handle_ = null_stream;
        }
    }

    // Create the stream if it does not exist yet. Binds to the device that is
    // current on the calling thread (which, for cell-group streams, must be
    // the group's execution_context device: the simulation always calls
    // set_gpu() before any group device work).
    void ensure() {
        if (handle_ == null_stream) {
            ARB_GPU_CHECK(create_stream(&handle_));
        }
    }

    // Ensure the stream exists and return its handle.
    stream_type handle() {
        ensure();
        return handle_;
    }

    // Block the calling thread until all work submitted on this stream has
    // completed. A no-op if the stream was never created.
    void synchronize() {
        if (handle_ != null_stream) {
            ARB_GPU_CHECK(stream_synchronize(handle_));
        }
    }

    // Synchronize and destroy the stream.
    void reset() {
        if (handle_ != null_stream) {
            ARB_GPU_CHECK(stream_synchronize(handle_));
            ARB_GPU_CHECK(destroy_stream(handle_));
            handle_ = null_stream;
        }
    }

    explicit operator bool() const {
        return handle_ != null_stream;
    }
};

// Block the calling host thread until all work submitted on the legacy
// default stream has completed (but not work on non-blocking streams).
//
// Construction of group-owned device memory launches fill kernels on the
// legacy stream through API-free layers (memory::array's value constructor,
// chunk_writer) that are stream-unaware. Every construction entry for group
// state ends with this call, so those kernels are complete before any work is
// submitted on the group's own stream; the group's host thread is the single
// submitter, so host program order gives the required happens-before.
inline void drain_legacy_stream() {
    ARB_GPU_CHECK(stream_synchronize(null_stream));
}

} // namespace gpu
} // namespace arb
