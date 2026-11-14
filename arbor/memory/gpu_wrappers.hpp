#pragma once

#include <cstddef>

namespace arb {
namespace memory {

void gpu_memcpy_d2d(void* dest, const void* src, std::size_t n);
void gpu_memcpy_d2h(void* dest, const void* src, std::size_t n);
void gpu_memcpy_h2d(void* dest, const void* src, std::size_t n);
void gpu_memcpy_h2d_async(void* dest, const void* src, std::size_t n);
void* gpu_host_register(void* ptr, std::size_t size);
void gpu_host_unregister(void* ptr);
void* gpu_malloc(std::size_t n);
void gpu_free(void* ptr);

// Stream-ordered variants. 'stream' is a raw GPU stream handle
// (arb::gpu::stream_type passed as void*, so that this layer stays free of
// the CUDA/HIP API). These are the only copies used by the GPU backend of
// cable cell groups, which run all of their device work on a per-group
// stream; the stream-less variants above stay for device work that is not
// owned by a group (they order against the legacy default stream).
//
// The async variants require that the host source buffer (for H2D) stays
// alive and unmodified until the stream has passed the copy; callers in the
// cell-group backend guarantee this by host ordering plus the end-of-epoch
// stream synchronization.
//
// gpu_memcpy_d2h_sync makes its result host-visible before it returns; the
// other three only enqueue on the given stream.

void gpu_memcpy_h2d_async(void* stream, void* dest, const void* src, std::size_t n);
void gpu_memcpy_d2d_async(void* stream, void* dest, const void* src, std::size_t n);
void gpu_memcpy_d2h_sync(void* stream, void* dest, const void* src, std::size_t n);

// Block the calling host thread until the device has completed all
// outstanding work on every stream. Host-side code that observes device data
// produced outside the legacy stream (e.g. a test intercepting data that a
// cell group's kernels produced on the group's own stream) must synchronize
// against that stream first; this is the blunt instrument that needs no
// handle to it.
void gpu_device_synchronize();

} // namespace memory
} // namespace arb
