#pragma once

#include <cstddef>
#include <type_traits>

#include <arbor/assert.hpp>

#include "wrappers.hpp"
#include "gpu_wrappers.hpp"

namespace arb {
namespace memory {

// ---------------------------------------------------------------------
// Stream-ordered copies between host and device memory.
//
// These are used by the GPU backend of cable cell groups, which runs all of
// its device work on a per-group stream. 'stream' is a raw GPU stream handle
// (arb::gpu::stream_type passed as void*, see gpu_wrappers.hpp); the
// coordinator layer is bypassed on purpose, containers only need
// data()/size(). The host-side buffer of an async copy must stay alive and
// unmodified until the stream has passed the copy; the cell-group backend
// guarantees this by host ordering plus the end-of-epoch stream sync.
//
// This lives in its own header rather than copy.hpp because copy.hpp is
// pulled in transitively by wrappers.hpp (via memory.hpp), i.e. before the
// is_on_host/is_on_gpu predicates that these templates name are declared.
// ---------------------------------------------------------------------

// Host-to-device or device-to-device copy, enqueued on 'stream' and not
// waited for.
template <typename LHS, typename RHS>
void copy_async(void* stream, LHS&& from, RHS&& to) {
    arb_assert(from.size() == to.size());
    using src_type = std::decay_t<LHS>;
    using dst_type = std::decay_t<RHS>;
    auto nbytes = from.size()*sizeof(typename src_type::value_type);
    if constexpr (util::is_on_host<src_type>::value) {
        static_assert(util::is_on_gpu<dst_type>::value, "copy_async: host source, non-device target");
        gpu_memcpy_h2d_async(stream, to.data(), from.data(), nbytes);
    }
    else {
        static_assert(util::is_on_gpu<src_type>::value && util::is_on_gpu<dst_type>::value,
                      "copy_async: expected device-to-device copy");
        gpu_memcpy_d2d_async(stream, to.data(), from.data(), nbytes);
    }
}

// Device-to-host copy; the data is host-visible when this call returns.
template <typename LHS, typename RHS>
void copy_sync(void* stream, LHS&& from, RHS&& to) {
    arb_assert(from.size() == to.size());
    using src_type = std::decay_t<LHS>;
    using dst_type = std::decay_t<RHS>;
    static_assert(util::is_on_gpu<src_type>::value, "copy_sync: expected device source");
    static_assert(util::is_on_host<dst_type>::value, "copy_sync: expected host destination");
    gpu_memcpy_d2h_sync(stream, to.data(), from.data(),
                        from.size()*sizeof(typename src_type::value_type));
}

} // namespace memory
} // namespace arb
