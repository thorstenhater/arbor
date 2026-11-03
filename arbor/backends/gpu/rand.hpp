#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <arbor/mechanism.hpp>
#include <arbor/serdes.hpp>

#include <util/pimpl.hpp>
#include <backends/rand_fwd.hpp>
#include <backends/gpu/gpu_store_types.hpp>

#include <arbor/gpu/gpu_api.hpp>

namespace arb {
namespace gpu {

class random_numbers {
public:
    void instantiate(mechanism& m, std::size_t value_width_padded, const mechanism_layout& pos_data,
        arb_seed_type seed);

    // Selects the next random number cache and, when the cache wraps,
    // regenerates the numbers; all device work runs on `s`, the stream of the
    // cell group owning this store (pass `null_stream` outside a cell group).
    void update(stream_type s, mechanism& m);

    ARB_SERDES_ENABLE(random_numbers, data_, random_number_update_counter_);

  private:
    // random number device storage
    array data_;
    std::array<std::vector<arb_value_type*>, cbprng::cache_size()>           random_numbers_;
    std::array<memory::device_vector<arb_value_type*>, cbprng::cache_size()> random_numbers_d_;

    // time step counter
    cbprng::counter_type random_number_update_counter_ = 0u;

    // pimpl
    struct impl;
    util::pimpl<impl> impl_;
};

} // namespace gpu
} // namespace arb
