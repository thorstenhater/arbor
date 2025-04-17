#include <arbor/symmetric_recipe.hpp>

namespace arb {

cell_size_type symmetric_recipe::num_cells() const {
    return tiled_recipe_->num_cells() * tiled_recipe_->num_tiles();
}

util::unique_any symmetric_recipe::get_cell_description(cell_gid_type i) const {
    return tiled_recipe_->get_cell_description(i % tiled_recipe_->num_cells());
}

cell_kind symmetric_recipe::get_cell_kind(cell_gid_type i) const {
    return tiled_recipe_->get_cell_kind(i % tiled_recipe_->num_cells());
}

// Only function that calls the underlying tile's function on the same gid.
// This is because applying transformations to event generators is not straightforward.
std::vector<event_generator> symmetric_recipe::event_generators(cell_gid_type i) const {
    return tiled_recipe_->event_generators(i);
}

// Take connections_on from the original tile recipe for the cell we are duplicating.
// Translate the source and destination gids
connections symmetric_recipe::connections_on(cell_gid_type i) const {
    int n_local = tiled_recipe_->num_cells();
    int n_global = num_cells();
    int offset = (i / n_local) * n_local;

    auto conns = tiled_recipe_->connections_on(i % n_local);

    for (unsigned j = 0; j < conns.size(); j++) {
        if (std::holds_alternative<cell_connection>(conns[j])) {
            auto& cn = std::get<cell_connection>(conns[j]);
            cn.source.gid  = (cn.source.gid + offset) % n_global;
        }
        else if (std::holds_alternative<raw_cell_connection>(conns[j])) {
            auto& cn = std::get<raw_cell_connection>(conns[j]);
            cn.source.gid  = (cn.source.gid + offset) % n_global;
        }
        else {
            ARB_UNREACHABLE;
        }
    }
    return conns;
}

std::vector<probe_info> symmetric_recipe::get_probes(cell_gid_type i) const {
    i %= tiled_recipe_->num_cells();
    return tiled_recipe_->get_probes(i);
}

std::any symmetric_recipe::get_global_properties(cell_kind ck) const {
    return tiled_recipe_->get_global_properties(ck);
};

} //namespace arb
