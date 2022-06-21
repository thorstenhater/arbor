#pragma once

// Implementations for fvm_lowered_cell are parameterized
// on the back-end class.
//
// Classes here are exposed in a header only so that
// implementation details may be tested in the unit tests.
// It should otherwise only be used in `fvm_lowered_cell.cpp`.

#include <array>
#include <cmath>
#include <iterator>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unordered_set>


#include <fstream> //todo
#include <iostream> //todo

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/recipe.hpp>
#include <arbor/util/any_visitor.hpp>

#include "execution_context.hpp"
#include "fvm_layout.hpp"
#include "fvm_lowered_cell.hpp"
#include "label_resolution.hpp"
#include "matrix.hpp"
#include "profile/profiler_macro.hpp"
#include "sampler_map.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"
#include "util/transform.hpp"

#include <communication/mpi.hpp>

namespace arb {

template <class Backend>
class fvm_lowered_cell_impl: public fvm_lowered_cell {
public:
    using backend = Backend;
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type = fvm_size_type;

    cell_gid_type cell_group;

    fvm_lowered_cell_impl(execution_context ctx, cell_gid_type cg): context_(ctx), threshold_watcher_(ctx) {cell_group = context_.distributed->id();};

    void reset() override;

    fvm_initialization_data initialize(
        const std::vector<cell_gid_type>& gids,
        const recipe& rec) override;

    fvm_integration_result integrate(
        value_type tfinal,
        value_type max_dt,
        std::vector<deliverable_event> staged_events,
        std::vector<sample_event> staged_samples) override;

    // Generates indom index for every gid, guarantees that gids belonging to the same supercell are in the same intdom
    // Fills cell_to_intdom map; returns number of intdoms
    fvm_size_type fvm_intdom(
        const recipe& rec,
        const std::vector<cell_gid_type>& gids,
        std::vector<fvm_index_type>& cell_to_intdom);

    value_type time() const override { return tmin_; }

    //Exposed for testing purposes
    std::vector<mechanism_ptr>& mechanisms() {
        return mechanisms_;
    }

private:
    // Host or GPU-side back-end dependent storage.x
    using array = typename backend::array;
    using shared_state = typename backend::shared_state;
    using sample_event_stream = typename backend::sample_event_stream;
    using threshold_watcher = typename backend::threshold_watcher;

    execution_context context_;

    std::unique_ptr<shared_state> state_; // Cell state shared across mechanisms.

    // TODO: Can we move the backend-dependent data structures below into state_?
    sample_event_stream sample_events_;
    array sample_time_;
    array sample_value_;
    matrix<backend> matrix_;
    threshold_watcher threshold_watcher_;

    value_type tmin_ = 0;
    std::vector<mechanism_ptr> mechanisms_; // excludes reversal potential calculators.
    std::vector<mechanism_ptr> revpot_mechanisms_;

    // Non-physical voltage check threshold, 0 => no check.
    value_type check_voltage_mV_ = 0;

    // Flag indicating that at least one of the mechanisms implements the post_events procedure
    bool post_events_ = false;

    // Host-side views/copies and local state.
    decltype(backend::host_view(sample_time_)) sample_time_host_;
    decltype(backend::host_view(sample_value_)) sample_value_host_;

    void update_ion_state();

    // Throw if absolute value of membrane voltage exceeds bounds.
    void assert_voltage_bounded(fvm_value_type bound);

    // Throw if any cell time not equal to tmin_
    void assert_tmin();

    // Assign tmin_ and call assert_tmin() if assertions on.
    void set_tmin(value_type t) {
        tmin_ = t;
        arb_assert((assert_tmin(), true));
    }

    static unsigned dt_steps(value_type t0, value_type t1, value_type dt) {
        return t0>=t1? 0: 1+(unsigned)((t1-t0)/dt);
    }

    // Sets the GPU used for CUDA calls from the thread that calls it.
    // The GPU will be the one in the execution context context_.
    // If not called, the thread may attempt to launch on a different GPU,
    // leading to crashes.
    void set_gpu() {
        if (context_.gpu->has_gpu()) context_.gpu->set_gpu();
    }

    // Translate cell probe descriptions into probe handles etc.
    void resolve_probe_address(
        std::vector<fvm_probe_data>& probe_data, // out parameter
        const std::vector<cable_cell>& cells,
        std::size_t cell_idx,
        const std::any& paddr,
        const fvm_cv_discretization& D,
        const fvm_mechanism_data& M,
        const std::vector<target_handle>& handles,
        const std::unordered_map<std::string, mechanism*>& mech_instance_by_name);
};

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::assert_tmin() {
    auto time_minmax = state_->time_bounds();
    if (time_minmax.first != time_minmax.second) {
        throw arbor_internal_error("fvm_lowered_cell: inconsistent times across cells");
    }
    if (time_minmax.first != tmin_) {
        throw arbor_internal_error("fvm_lowered_cell: out of synchronziation with cell state time");
    }
}

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::reset() {
    state_->reset();
    set_tmin(0);

    for (auto& m: revpot_mechanisms_) {
        m->initialize();
    }

    for (auto& m: mechanisms_) {
        m->initialize();
    }

    update_ion_state();

    state_->zero_currents();

    // Note: mechanisms must be initialized again after the ion state is updated,
    // as mechanisms can read/write the ion_state within the initialize block
    for (auto& m: revpot_mechanisms_) {
        m->initialize();
    }

    for (auto& m: mechanisms_) {
        m->initialize();
    }

    // NOTE: Threshold watcher reset must come after the voltage values are set,
    // as voltage is implicitly read by watcher to set initial state.
    threshold_watcher_.reset(state_->voltage);
}


template <typename Backend>
fvm_integration_result fvm_lowered_cell_impl<Backend>::integrate(
    value_type tfinal,
    value_type dt_max,
    std::vector<deliverable_event> staged_events,
    std::vector<sample_event> staged_samples)
{
    set_gpu();

    // Integration setup
    PE(advance_integrate_setup);
    threshold_watcher_.clear_crossings();

    auto n_samples = staged_samples.size();
    if (sample_time_.size() < n_samples) {
        sample_time_ = array(n_samples);
        sample_value_ = array(n_samples);
    }

    state_->deliverable_events.init(std::move(staged_events));
    sample_events_.init(std::move(staged_samples));

    arb_assert((assert_tmin(), true));
    unsigned remaining_steps = dt_steps(tmin_, tfinal, dt_max);
    PL();

    // TODO: Consider devolving more of this to back-end routines (e.g.
    // per-compartment dt probably not a win on GPU), possibly rumbling
    // complete fvm state into shared state object.

    // Reset state
    multicore::iarray cv_to_intdom_r = state_->cv_to_intdom;
    multicore::iarray cv_to_cell_r   = state_->cv_to_cell;

    array time_r            = state_->time;
    array time_to_r         = state_->time_to;
    array dt_intdom_r       = state_->dt_intdom;
    array dt_cv_r           = state_->dt_cv;
    array voltage_r         = state_->voltage;
    array current_density_r = state_->current_density;
    array conductivity_r    = state_->conductivity;

    array time_since_spike_r = state_->time_since_spike;
    multicore::deliverable_event_stream deliverable_events_r = state_->deliverable_events;
    multicore::iarray src_to_spike_r = state_->src_to_spike;
    
    multicore::istim_state stim_data_r = state_->stim_data;
    std::unordered_map<std::string, multicore::ion_state> ion_data_r = state_->ion_data;
    
    std::unordered_map<unsigned, multicore::shared_state::mech_storage> storage_r;
    std::unordered_map<unsigned, std::vector<arb_value_type>>           parameters_r;
    std::unordered_map<unsigned, std::vector<arb_value_type>>           state_vars_r;
    
    for (auto store: state_->storage) {
        storage_r[store.first].data_        = store.second.data_;
        storage_r[store.first].indices_     = store.second.indices_;
        //storage_r[store.first].constraints_ = store.second.constraints_;
        storage_r[store.first].ion_states_  = store.second.ion_states_;
        for (auto i = 0; i<store.second.parameters_.size(); ++i) {
            parameters_r[store.first].push_back(*store.second.parameters_[i]);
        }
        for (auto j = 0; j<store.second.state_vars_.size(); ++j) {
            state_vars_r[store.first].push_back(*store.second.state_vars_[j]);
        }
    }

    // trace vectors for recording voltages
    // data layout
    //  mechanism id -> [[U]]
    //                   ^  ^-- steps
    //                    \---- GJ CV voltage, once per connection from this CV
    std::unordered_map<arb_index_type, std::vector<std::vector<arb_value_type>>> trace, trace_prev;

    threshold_watcher threshold_watcher_reset = threshold_watcher_;

    // Save starting times
    unsigned max_steps = remaining_steps;
    value_type tmin_reset = tmin_;
    value_type tfinal_reset = tfinal;
    value_type dt_max_reset = dt_max;

    std::ofstream file0;

    std::vector<arb_index_type> peer_ix;
    std::vector<arb_index_type> peer_index_reset;
    
    std::vector<int> node_arr;
    for (auto& m: mechanisms_) {
        if (m->kind() == arb_mechanism_kind_gap_junction) {
            for(int i = 0; i<m->ppack_.width; ++i) {
                node_arr.push_back(m->ppack_.node_index[i]);
            }
        }
    }
    
    //WR iterations
    int wr_max = 1;
    auto eps = 1e-7;
    for (int wr_it = 0; wr_it < wr_max; wr_it++) {
        // Reset remaining_steps
        if (wr_it > 0) {
            remaining_steps = dt_steps(tmin_reset, tfinal_reset, dt_max_reset);
        }

        // Reset error variables for WR break condition
        auto max_err = 0.;

        // Threshold_watcher reset
        threshold_watcher_ = threshold_watcher_reset;

        while (remaining_steps) {
            auto step = max_steps - remaining_steps;
            
            // Update any required reversal potentials based on ionic concs.
            for (auto& m: revpot_mechanisms_) {
                m->update_current();
            }

            // Deliver events and accumulate mechanism current contributions.

            PE(advance_integrate_events);
            state_->deliverable_events.mark_until_after(state_->time);
            PL();

            PE(advance_integrate_current_zero);
            state_->zero_currents();
            PL();
            for (auto& m: mechanisms_) {
                auto state = state_->deliverable_events.marked_events();
                arb_deliverable_event_stream events;
                events.n_streams = state.n;
                events.begin     = state.begin_offset;
                events.end       = state.end_offset;
                events.events    = (arb_deliverable_event_data*) state.ev_data; // FIXME(TH): This relies on bit-castability
                m->deliver_events(events);

                // index_to_cell: translate global CV index to cell gid

                if (m->kind() == arb_mechanism_kind_gap_junction) {

                    auto& ppack = m->ppack_;
                    auto& pidx  = ppack.peer_index;
                    auto& nidx  = ppack.node_index;

                    std::vector<arb_index_type> peer_ix(ppack.width, 0);

                    if (wr_it == 0 && step == 0) {
                        std::cerr << "Setting up index mappings #p=" << peer_ix.size() << " #pg=" << peer_ix_group.size() << '\n';
                        for (int i = 0; i<ppack.width; ++i) {
                            node_ix_group.push_back(std::get<2>(index_to_cell[nidx[i]]));
                        }

                        for (int i = 0; i<ppack.width; ++i) {
                            auto cg_node = std::get<1>(index_to_cell[nidx[i]]);
                            auto cg_peer = std::get<1>(index_to_cell[pidx[i]]);
                            std::cout << "cell_group node = " << cg_node << " cell_group peer = " << cg_peer << std::endl;
                            std::cout << " tmin = " << tmin_ << std::endl;
                            if (cg_peer == cg_node) { // We are local to local
                                peer_ix_group.push_back(std::get<2>(index_to_cell[pidx[i]]));
                            }
                            else {
                                peer_ix_group.push_back(std::get<2>(index_to_cell[nidx[i]]));
                            }
                        }

                        // Setup peer indices
                        //  * peer_ix:       indices into the trace, to be consumed after the first WR iteration has recorded one
                        //  * peer_ix_group: indices into the *local* voltage array, s.t. the i'th CV get its own voltage back since
                        //                   we assume zero differential for the first iteration
                        peer_ix.clear();
                        peer_ix.resize(ppack.width);

                        for (int ix = 0; ix < ppack.width; ++ix) {
                            if (true /* TODO decide whether we are local-local */) {
                                const auto& nodes = ppack.node_index;
                                auto len    = ppack.width;
                                // TODO use the correct index arrays here
                                auto peer   = ppack.peer_index[ix];
                                auto node   = std::find(nodes, nodes + len, peer);
                                auto offset = std::distance(nodes, node);
                                if (offset >= len) throw std::runtime_error("Connection not found.");
                                peer_ix[ix] = offset;
                            }
                            else {
                                throw std::runtime_error("Non-local connectivity is not implemented yet.");
                            }
                        }

                        std::cerr << "m_id=" << m->mechanism_id() << '\n';
                        for (int ix = 0; ix < ppack.width; ++ix) std::cerr << " * " << peer_ix[ix] << '\n';
                    }

                    // TH: Why only step=0
                    if (step == 0) {
                        // in iteration=0 we do not have a trace yet
                        if (wr_it == 0) {
                            // Peer index in this group?
                            // Q: Didn't we want to have the identity here, ie i --> i, instead of the peer index?
                            // A: Yes, indeed peer_ix_group is either the local peer index or the identity, if not local
                            pidx = peer_ix_group.data();
                        }
                        else {
                            auto gj = m->mechanism_id();
                            // Here we expect peer_ix to be the index into the trace structure, regardlees of local
                            // or not. However, the selection of the trace must take locality into account.
                            // Q: How to select?
                            pidx = peer_ix.data();
                            ppack.vec_v_peer = trace_prev[gj][step].data();
                        }
                    }
                }
                
                m->update_current();

                //update traces
                if (m->kind() == arb_mechanism_kind_gap_junction) {
                    //write trace
                    auto gj = m->mechanism_id();
                    std::vector<value_type> v_step(m->ppack_.width, 0.);
    
                    for(int ix = 0; ix < m->ppack_.width; ++ix) {
                        auto node_cv = m->ppack_.node_index[ix];
                        v_step[ix] = state_->voltage[node_cv];
                    }
                    trace[gj].push_back(v_step);
                }
            }

            PE(advance_integrate_events);
            state_->deliverable_events.drop_marked_events();

            // Update event list and integration step times.

            state_->update_time_to(dt_max, tfinal);
            state_->deliverable_events.event_time_if_before(state_->time_to);
            state_->set_dt();
            PL();

            // Add stimulus current contributions.
            // (Note: performed after dt, time_to calculation, in case we
            // want to use mean current contributions as opposed to point
            // sample.)

            PE(advance_integrate_stimuli)
            state_->add_stimulus_current();
            PL();

            // Take samples at cell time if sample time in this step interval.

            PE(advance_integrate_samples);
            sample_events_.mark_until(state_->time_to);
            state_->take_samples(sample_events_.marked_events(), sample_time_, sample_value_);
            sample_events_.drop_marked_events();
            PL();

            // Integrate voltage by matrix solve.

            PE(advance_integrate_matrix_build);
            matrix_.assemble(state_->dt_intdom, state_->voltage, state_->current_density, state_->conductivity);
            PL();
            PE(advance_integrate_matrix_solve);
            matrix_.solve(state_->voltage);
            PL();

            // Integrate mechanism state.

            for (auto& m: mechanisms_) {
                m->update_state();
            }

            // Update ion concentrations.

            PE(advance_integrate_ionupdate);
            update_ion_state();
            PL();

            // Update time and test for spike threshold crossings.

            PE(advance_integrate_threshold);
            threshold_watcher_.test(&state_->time_since_spike);
            PL();

            PE(advance_integrate_post)
            if (post_events_) {
                for (auto& m: mechanisms_) {
                    m->post_event();
                }
            }
            PL();

            std::swap(state_->time_to, state_->time);
            state_->time_ptr = state_->time.data();

            // Check for non-physical solutions:

            if (check_voltage_mV_>0) {
                PE(advance_integrate_physicalcheck);
                assert_voltage_bounded(check_voltage_mV_);
                PL();
            }

            // Check for end of integration.

            PE(advance_integrate_stepsupdate);
            if (!--remaining_steps) {
                tmin_ = state_->time_bounds().first;
                remaining_steps = dt_steps(tmin_, tfinal, dt_max);
            }
            PL();
        }

        // Break if no CV has an err > eps or maximum WR iterations reached
        if ((wr_it > 1 && max_err < eps) or wr_max == 1 || wr_it == wr_max) {
            //std::cout << "break at it = " << wr_it << std::endl;
            break;
        }

        // Reset traces
        trace_prev = trace;
        trace = {};

        // Reset state
        std::copy(dt_intdom_r.begin(), dt_intdom_r.end(), state_->dt_intdom.begin());
        std::copy(dt_cv_r.begin(), dt_cv_r.end(), state_->dt_cv.begin());

        std::copy(time_r.begin(), time_r.end(), state_->time.begin());
        std::copy(time_to_r.begin(), time_to_r.end(), state_->time_to.begin());
        state_->dt_intdom = dt_intdom_r;
        state_->dt_cv = dt_cv_r;
        state_->current_density = current_density_r;
        state_->conductivity = conductivity_r;

        std::copy(voltage_r.begin(), voltage_r.end(), state_->voltage.begin());

        std::copy(cv_to_intdom_r.begin(), cv_to_intdom_r.end(), state_->cv_to_intdom.begin());
        std::copy(cv_to_cell_r.begin(), cv_to_cell_r.end(), state_->cv_to_cell.begin());            
            
        std::copy(current_density_r.begin(), current_density_r.end(), state_->current_density.begin());
        std::copy(conductivity_r.begin(), conductivity_r.end(), state_->conductivity.begin());
            
        std::copy(time_since_spike_r.begin(), time_since_spike_r.end(), state_->time_since_spike.begin());
        std::copy(src_to_spike_r.begin(), src_to_spike_r.end(), state_->src_to_spike.begin());
            
        //state_->stim_data = stim_data_r;
        //state_->ion_data = ion_data_r;
        state_->deliverable_events = deliverable_events_r; 
         
        for (auto store: state_->storage) {
            std::copy(storage_r[store.first].data_.begin(), storage_r[store.first].data_.end(), store.second.data_.begin());
            std::copy(storage_r[store.first].indices_.begin(), storage_r[store.first].indices_.end(), store.second.indices_.begin());
            //store.second.constraints_ = storage_r[store.first].constraints_;
            std::copy(storage_r[store.first].ion_states_.begin(), storage_r[store.first].ion_states_.end(), store.second.ion_states_.begin());
            for (auto i = 0; i<store.second.parameters_.size(); ++i) {
                *store.second.parameters_[i] = parameters_r[store.first][i];
            }
            for (auto j = 0; j<store.second.state_vars_.size(); ++j) {
                *store.second.state_vars_[j] = state_vars_r[store.first][j];
            }
        } 
    }
    
    set_tmin(tfinal);

    const auto& crossings = threshold_watcher_.crossings();
    sample_time_host_ = backend::host_view(sample_time_);
    sample_value_host_ = backend::host_view(sample_value_);

    return fvm_integration_result{
        util::range_pointer_view(crossings),
        util::range_pointer_view(sample_time_host_),
        util::range_pointer_view(sample_value_host_)
    };
}

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::update_ion_state() {
    state_->ions_init_concentration();
    for (auto& m: mechanisms_) {
        m->update_ions();
    }
}

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::assert_voltage_bounded(fvm_value_type bound) {
    auto v_minmax = state_->voltage_bounds();
    if (v_minmax.first>=-bound && v_minmax.second<=bound) {
        return;
    }

    auto t_minmax = state_->time_bounds();
    throw range_check_failure(
        util::pprintf("voltage solution out of bounds for t in [{}, {}]", t_minmax.first, t_minmax.second),
        v_minmax.first<-bound? v_minmax.first: v_minmax.second);
}

template <typename Backend>
fvm_initialization_data fvm_lowered_cell_impl<Backend>::initialize(
    const std::vector<cell_gid_type>& gids,
    const recipe& rec)
{
    using std::any_cast;
    using util::count_along;
    using util::make_span;
    using util::value_by_key;
    using util::keys;

    fvm_initialization_data fvm_info;

    set_gpu();

    std::vector<cable_cell> cells;
    const std::size_t ncell = gids.size();

    cells.resize(ncell);
    threading::parallel_for::apply(0, gids.size(), context_.thread_pool.get(),
           [&](cell_size_type i) {
               auto gid = gids[i];
               try {
                   cells[i] = any_cast<cable_cell&&>(rec.get_cell_description(gid));
               }
               catch (std::bad_any_cast&) {
                   throw bad_cell_description(rec.get_cell_kind(gid), gid);
               }
           });

    // Populate source, target and gap_junction data vectors.
    for (auto i : util::make_span(ncell)) {
        auto gid = gids[i];
        const auto& c = cells[i];

        fvm_info.source_data.add_cell();
        fvm_info.target_data.add_cell();
        fvm_info.gap_junction_data.add_cell();

        unsigned count = 0;
        for (const auto& [label, range]: c.detector_ranges()) {
            fvm_info.source_data.add_label(label, range);
            count+=(range.end - range.begin);
        }
        fvm_info.num_sources[gid] = count;

        count = 0;
        for (const auto& [label, range]: c.synapse_ranges()) {
            fvm_info.target_data.add_label(label, range);
            count+=(range.end - range.begin);
        }
        fvm_info.num_targets[gid] = count;

        for (const auto& [label, range]: c.junction_ranges()) {
            fvm_info.gap_junction_data.add_label(label, range);
        }
    }

    cable_cell_global_properties global_props;
    try {
        std::any rec_props = rec.get_global_properties(cell_kind::cable);
        if (rec_props.has_value()) {
            global_props = any_cast<cable_cell_global_properties>(rec_props);
        }
    }
    catch (std::bad_any_cast&) {
        throw bad_global_property(cell_kind::cable);
    }

    // Assert that all global default parameters have been set.
    // (Throws cable_cell_error on failure.)
    check_global_properties(global_props);

    const auto& catalogue = global_props.catalogue;

    // Mechanism instantiator helper.
    auto mech_instance = [&catalogue](const std::string& name) {
        return catalogue.instance(backend::kind, name);
    };

    // Check for physically reasonable membrane volages?

    check_voltage_mV_ = global_props.membrane_voltage_limit_mV;

    auto nintdom = fvm_intdom(rec, gids, fvm_info.cell_to_intdom);

    // Discretize cells, build matrix.

    fvm_cv_discretization D = fvm_cv_discretize(cells, global_props.default_parameters, context_);

    std::vector<index_type> cv_to_intdom(D.size());
    std::transform(D.geometry.cv_to_cell.begin(), D.geometry.cv_to_cell.end(), cv_to_intdom.begin(),
                   [&fvm_info](index_type i){ return fvm_info.cell_to_intdom[i]; });

    arb_assert(D.n_cell() == ncell);
    matrix_ = matrix<backend>(D.geometry.cv_parent, D.geometry.cell_cv_divs,
                              D.cv_capacitance, D.face_conductance, D.cv_area, fvm_info.cell_to_intdom);
    sample_events_ = sample_event_stream(nintdom);

    // Discretize and build gap junction info.
    //std::unordered_map<cell_member_type, fvm_size_type> gj_cvs = fvm_build_gap_junction_cv_map(cells, gids, D);
    //std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = fvm_resolve_gj_connections(gids, fvm_info.gap_junction_data, gj_cvs, rec);

    // Build {gid, lid} -> {cg, cv} map
    std::unordered_map<cell_member_type, cell_member_type> cg_map = fvm_build_gap_junction_cg_cv_map(cells, gids, cell_group, D);

    std::vector<int> cvs;
    for (auto element: cg_map){
        cvs.push_back(int(element.second.index));
    }
    int max_cv = 0;
    for (int i = 0; i<cvs.size(); ++i){
        if (cvs[i] > max_cv){
            max_cv = cvs[i];
        }
    }    

    //Todo: make this more elegant
    std::vector<int> num_cvs_group = {int(max_cv)+1};
    std::vector<int> num_cvs_global = context_.distributed->gather_cg_cv_map(num_cvs_group);

    auto gid_lid_gcv = fvm_convert_cv(cg_map, num_cvs_global);
    auto gid_lid_gcv_gathered = context_.distributed->gather_cg_cv_map(gid_lid_gcv);

    auto gcv_map = fvm_convert_cv_to_map(gid_lid_gcv_gathered);
    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = fvm_resolve_gj_connections(gids, fvm_info.gap_junction_data, gcv_map, rec);

    // Discretize mechanism data.

    fvm_mechanism_data mech_data = fvm_build_mechanism_data(global_props, cells, gids, gj_conns, D, context_);

    // Fill src_to_spike and cv_to_cell vectors only if mechanisms with post_events implemented are present.
    post_events_ = mech_data.post_events;
    auto max_detector = 0;
    if (post_events_) {
        auto it = util::max_element_by(fvm_info.num_sources, [](auto elem) {return util::second(elem);});
        max_detector = it->second;
    }
    std::vector<fvm_index_type> src_to_spike, cv_to_cell;

    if (post_events_) {
        for (auto cell_idx: make_span(ncell)) {
            for (auto lid: make_span(fvm_info.num_sources[gids[cell_idx]])) {
                src_to_spike.push_back(cell_idx * max_detector + lid);
            }
        }
        src_to_spike.shrink_to_fit();
        cv_to_cell = D.geometry.cv_to_cell;
    }

    // Create shared cell state.
    // Shared state vectors should accommodate each mechanism's data alignment requests.

    unsigned data_alignment = util::max_value(
        util::transform_view(keys(mech_data.mechanisms),
            [&](const std::string& name) { return mech_instance(name).mech->data_alignment(); }));

    state_ = std::make_unique<shared_state>(
                nintdom, ncell, max_detector, cv_to_intdom, std::move(cv_to_cell),
                D.init_membrane_potential, D.temperature_K, D.diam_um, std::move(src_to_spike),
                data_alignment? data_alignment: 1u);

    // Instantiate mechanisms, ions, and stimuli.

    for (const auto& [ion, data]: mech_data.ions) {
        if (auto charge = value_by_key(global_props.ion_species, ion)) {
            state_->add_ion(ion, *charge, data);
        }
        else {
            throw cable_cell_error("unrecognized ion '"+ion+"' in mechanism");
        }
    }

    if (!mech_data.stimuli.cv.empty()) {
        state_->configure_stimulus(mech_data.stimuli);
    }

    fvm_info.target_handles.resize(mech_data.n_target);

    // Keep track of mechanisms by name for probe lookup.
    std::unordered_map<std::string, mechanism*> mechptr_by_name;

    unsigned mech_id = 0;
    for (auto& m: mech_data.mechanisms) {
        auto& name = m.first;
        auto& config = m.second;

        mechanism_layout layout;
        layout.cv = config.cv;
        layout.multiplicity = config.multiplicity;
        layout.peer_cv = config.peer_cv;
        layout.weight.resize(layout.cv.size());

        std::vector<fvm_index_type> multiplicity_divs;
        auto multiplicity_part = util::make_partition(multiplicity_divs, layout.multiplicity);

        // Mechanism weights are F·α where α ∈ [0, 1] is the proportional
        // contribution in the CV, and F is the scaling factor required
        // to convert from the mechanism current contribution units to A/m².

        switch (config.kind) {
        case arb_mechanism_kind_point:
            // Point mechanism contributions are in [nA]; CV area A in [µm^2].
            // F = 1/A * [nA/µm²] / [A/m²] = 1000/A.

            for (auto i: count_along(config.cv)) {
                auto cv = layout.cv[i];
                layout.weight[i] = 1000/D.cv_area[cv];

                if (config.target.empty()) continue;

                target_handle handle(mech_id, i, cv_to_intdom[cv]);
                if (config.multiplicity.empty()) {
                    fvm_info.target_handles[config.target[i]] = handle;
                }
                else {
                    for (auto j: make_span(multiplicity_part[i])) {
                        fvm_info.target_handles[config.target[j]] = handle;
                    }
                }
            }
            break;
        case arb_mechanism_kind_gap_junction:
            // Junction mechanism contributions are in [nA] (µS * mV); CV area A in [µm^2].
            // F = 1/A * [nA/µm²] / [A/m²] = 1000/A.

            for (auto i: count_along(layout.cv)) {
                auto cv = layout.cv[i];
                layout.weight[i] = config.local_weight[i] * 1000/D.cv_area[cv];
            }
            break;
        case arb_mechanism_kind_density:
            // Current density contributions from mechanism are already in [A/m²].

            for (auto i: count_along(layout.cv)) {
                layout.weight[i] = config.norm_area[i];
            }
            break;
        case arb_mechanism_kind_reversal_potential:
            // Mechanisms that set reversal potential should not be contributing
            // to any currents, so leave weights as zero.
            break;
        }

        auto minst = mech_instance(name);
        state_->instantiate(*minst.mech, mech_id++, minst.overrides, layout);
        mechptr_by_name[name] = minst.mech.get();

        for (auto& pv: config.param_values) {
            state_->set_parameter(*minst.mech, pv.first, pv.second);
        }

        if (config.kind==arb_mechanism_kind_reversal_potential) {
            revpot_mechanisms_.push_back(mechanism_ptr(minst.mech.release()));
        }
        else {
            mechanisms_.push_back(mechanism_ptr(minst.mech.release()));
        }
    }


    std::vector<index_type> detector_cv;
    std::vector<value_type> detector_threshold;
    std::vector<fvm_probe_data> probe_data;

    for (auto cell_idx: make_span(ncell)) {
        cell_gid_type gid = gids[cell_idx];

        // Collect detectors, probe handles.
        for (auto entry: cells[cell_idx].detectors()) {
            detector_cv.push_back(D.geometry.location_cv(cell_idx, entry.loc, cv_prefer::cv_empty));
            detector_threshold.push_back(entry.item.threshold);
        }

        std::vector<probe_info> rec_probes = rec.get_probes(gid);
        for (cell_lid_type i: count_along(rec_probes)) {
            probe_info& pi = rec_probes[i];
            resolve_probe_address(probe_data, cells, cell_idx, std::move(pi.address),
                D, mech_data, fvm_info.target_handles, mechptr_by_name);

            if (!probe_data.empty()) {
                cell_member_type probe_id{gid, i};
                fvm_info.probe_map.tag[probe_id] = pi.tag;

                for (auto& data: probe_data) {
                    fvm_info.probe_map.data.insert({probe_id, std::move(data)});
                }
            }
        }
    }

    threshold_watcher_ = backend::voltage_watcher(*state_, detector_cv, detector_threshold, context_);

    reset();

    return fvm_info;
}

template <typename Backend>
fvm_size_type fvm_lowered_cell_impl<Backend>::fvm_intdom(
        const recipe& rec,
        const std::vector<cell_gid_type>& gids,
        std::vector<fvm_index_type>& cell_to_intdom) {

    cell_to_intdom.resize(gids.size());

    std::unordered_map<cell_gid_type, cell_size_type> gid_to_loc;
    for (auto i: util::count_along(gids)) {
        gid_to_loc[gids[i]] = i;
    }

    std::unordered_set<cell_gid_type> visited;
    std::queue<cell_gid_type> intdomq;
    cell_size_type intdom_id = 0;

    for (auto gid: gids) {
        if (visited.count(gid)) continue;
        visited.insert(gid);

        intdomq.push(gid);
        while (!intdomq.empty()) {
            auto g = intdomq.front();
            intdomq.pop();

            cell_to_intdom[gid_to_loc[g]] = intdom_id;

            for (const auto& gj: rec.gap_junctions_on(g)) {
                if (!visited.count(gj.peer.gid)) {
                    visited.insert(gj.peer.gid);
                    intdomq.push(gj.peer.gid);
                }
            }
        }
        intdom_id++;
    }

    return intdom_id;
}

// Resolution of probe addresses into a specific fvm_probe_data draws upon data
// from the cable cell, the discretization, the target handle map, and the
// back-end shared state.
//
// `resolve_probe_address` collates this data into a `probe_resolution_data`
// struct which is then passed on to the specific resolution procedure
// determined by the type of the user-supplied probe address.

template <typename Backend>
struct probe_resolution_data {
    std::vector<fvm_probe_data>& result;
    typename Backend::shared_state* state;
    const cable_cell& cell;
    const std::size_t cell_idx;
    const fvm_cv_discretization& D;
    const fvm_mechanism_data& M;
    const std::vector<target_handle>& handles;
    const std::unordered_map<std::string, mechanism*>& mech_instance_by_name;

    // Backend state data for a given mechanism and state variable.
    const fvm_value_type* mechanism_state(const std::string& name, const std::string& state_var) const {
        mechanism* m = util::value_by_key(mech_instance_by_name, name).value_or(nullptr);
        if (!m) return nullptr;

        const fvm_value_type* data = state->mechanism_state_data(*m, state_var);
        if (!data) throw cable_cell_error("no state variable '"+state_var+"' in mechanism '"+name+"'");

        return data;
    }

    // Extent of density mechanism on cell.
    mextent mechanism_support(const std::string& name) const {
        auto& mech_map = cell.region_assignments().template get<density>();
        auto opt_mm = util::value_by_key(mech_map, name);

        return opt_mm? opt_mm->support(): mextent{};
    };

    // Index into ion data from location.
    std::optional<fvm_index_type> ion_location_index(const std::string& ion, mlocation loc) const {
        if (state->ion_data.count(ion)) {
            return util::binary_search_index(M.ions.at(ion).cv,
                fvm_index_type(D.geometry.location_cv(cell_idx, loc, cv_prefer::cv_nonempty)));
        }
        return std::nullopt;
    }
};

template <typename Backend>
void fvm_lowered_cell_impl<Backend>::resolve_probe_address(
    std::vector<fvm_probe_data>& probe_data,
    const std::vector<cable_cell>& cells,
    std::size_t cell_idx,
    const std::any& paddr,
    const fvm_cv_discretization& D,
    const fvm_mechanism_data& M,
    const std::vector<target_handle>& handles,
    const std::unordered_map<std::string, mechanism*>& mech_instance_by_name)
{
    probe_data.clear();
    probe_resolution_data<Backend> prd{
        probe_data, state_.get(), cells[cell_idx], cell_idx, D, M, handles, mech_instance_by_name};

    using V = util::any_visitor<
        cable_probe_membrane_voltage,
        cable_probe_membrane_voltage_cell,
        cable_probe_axial_current,
        cable_probe_total_ion_current_density,
        cable_probe_total_ion_current_cell,
        cable_probe_total_current_cell,
        cable_probe_stimulus_current_cell,
        cable_probe_density_state,
        cable_probe_density_state_cell,
        cable_probe_point_state,
        cable_probe_point_state_cell,
        cable_probe_ion_current_density,
        cable_probe_ion_current_cell,
        cable_probe_ion_int_concentration,
        cable_probe_ion_int_concentration_cell,
        cable_probe_ion_ext_concentration,
        cable_probe_ion_ext_concentration_cell>;

    auto visitor = util::overload(
        [&prd](auto& probe_addr) { resolve_probe(probe_addr, prd); },
        [] { throw cable_cell_error("unrecognized probe type"), fvm_probe_data{}; });

    return V::visit(visitor, paddr);
}

template <typename B>
void resolve_probe(const cable_probe_membrane_voltage& p, probe_resolution_data<B>& R) {
    const fvm_value_type* data = R.state->voltage.data();

    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        fvm_voltage_interpolant in = fvm_interpolate_voltage(R.cell, R.D, R.cell_idx, loc);

        R.result.push_back(fvm_probe_interpolated{
            {data+in.proximal_cv, data+in.distal_cv},
            {in.proximal_coef, in.distal_coef},
            loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_membrane_voltage_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_multi r;
    mcable_list cables;

    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        const double* ptr = R.state->voltage.data()+cv;
        for (auto cable: R.D.geometry.cables(cv)) {
            r.raw_handles.push_back(ptr);
            cables.push_back(cable);
        }
    }
    r.metadata = std::move(cables);
    r.shrink_to_fit();

    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_axial_current& p, probe_resolution_data<B>& R) {
    const fvm_value_type* data = R.state->voltage.data();

    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        fvm_voltage_interpolant in = fvm_axial_current(R.cell, R.D, R.cell_idx, loc);

        R.result.push_back(fvm_probe_interpolated{
            {data+in.proximal_cv, data+in.distal_cv},
            {in.proximal_coef, in.distal_coef},
            loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_total_ion_current_density& p, probe_resolution_data<B>& R) {
    // Use interpolated probe with coeffs 1, -1 to represent difference between accumulated current density and stimulus.
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        fvm_index_type cv = R.D.geometry.location_cv(R.cell_idx, loc, cv_prefer::cv_nonempty);
        const double* current_cv_ptr = R.state->current_density.data() + cv;

        auto opt_i = util::binary_search_index(R.M.stimuli.cv_unique, cv);
        const double* stim_cv_ptr = opt_i? R.state->stim_data.accu_stim_.data()+*opt_i: nullptr;

        R.result.push_back(fvm_probe_interpolated{
            {current_cv_ptr, stim_cv_ptr},
            {1., -1.},
            loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_total_ion_current_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_interpolated_multi r;
    std::vector<const double*> stim_handles;

    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        const double* current_cv_ptr = R.state->current_density.data()+cv;
        auto opt_i = util::binary_search_index(R.M.stimuli.cv_unique, cv);
        const double* stim_cv_ptr = opt_i? R.state->stim_data.accu_stim_.data()+*opt_i: nullptr;

        for (auto cable: R.D.geometry.cables(cv)) {
            double area = R.cell.embedding().integrate_area(cable); // [µm²]
            if (area>0) {
                r.raw_handles.push_back(current_cv_ptr);
                stim_handles.push_back(stim_cv_ptr);
                r.coef[0].push_back(0.001*area); // Scale from [µm²·A/m²] to [nA].
                r.coef[1].push_back(-r.coef[0].back());
                r.metadata.push_back(cable);
            }
        }
    }

    util::append(r.raw_handles, stim_handles);
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_total_current_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_membrane_currents r;

    auto cell_cv_ival = R.D.geometry.cell_cv_interval(R.cell_idx);
    auto cv0 = cell_cv_ival.first;

    util::assign(r.cv_parent, util::transform_view(util::subrange_view(R.D.geometry.cv_parent, cell_cv_ival),
        [cv0](auto cv) { return cv+1==0? cv: cv-cv0; }));
    util::assign(r.cv_parent_cond, util::subrange_view(R.D.face_conductance, cell_cv_ival));

    const auto& stim_cvs = R.M.stimuli.cv_unique;
    const fvm_value_type* stim_src = R.state->stim_data.accu_stim_.data();

    r.cv_cables_divs = {0};
    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        r.raw_handles.push_back(R.state->voltage.data()+cv);
        double oo_cv_area = R.D.cv_area[cv]>0? 1./R.D.cv_area[cv]: 0;

        for (auto cable: R.D.geometry.cables(cv)) {
            double area = R.cell.embedding().integrate_area(cable); // [µm²]
            if (area>0) {
                r.weight.push_back(area*oo_cv_area);
                r.metadata.push_back(cable);
            }
        }
        r.cv_cables_divs.push_back(r.metadata.size());
    }
    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        auto opt_i = util::binary_search_index(stim_cvs, cv);
        if (!opt_i) continue;

        r.raw_handles.push_back(stim_src+*opt_i);
        r.stim_cv.push_back(cv-cv0);
        r.stim_scale.push_back(0.001*R.D.cv_area[cv]); // Scale from [µm²·A/m²] to [nA].
    }
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_stimulus_current_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_weighted_multi r;

    const auto& stim_cvs = R.M.stimuli.cv_unique;
    const fvm_value_type* src = R.state->stim_data.accu_stim_.data();

    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        auto opt_i = util::binary_search_index(stim_cvs, cv);
        const double* ptr = opt_i? src+*opt_i: nullptr;

        for (auto cable: R.D.geometry.cables(cv)) {
            double area = R.cell.embedding().integrate_area(cable); // [µm²]
            if (area>0) {
                r.raw_handles.push_back(ptr);
                r.weight.push_back(0.001*area); // Scale from [µm²·A/m²] to [nA].
                r.metadata.push_back(cable);
            }
        }
    }

    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_density_state& p, probe_resolution_data<B>& R) {
    const fvm_value_type* data = R.mechanism_state(p.mechanism, p.state);
    if (!data) return;

    auto support = R.mechanism_support(p.mechanism);
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        if (!support.intersects(loc)) continue;

        fvm_index_type cv = R.D.geometry.location_cv(R.cell_idx, loc, cv_prefer::cv_nonempty);
        auto opt_i = util::binary_search_index(R.M.mechanisms.at(p.mechanism).cv, cv);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{data+*opt_i}, loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_density_state_cell& p, probe_resolution_data<B>& R) {
    fvm_probe_multi r;

    const fvm_value_type* data = R.mechanism_state(p.mechanism, p.state);
    if (!data) return;

    mextent support = R.mechanism_support(p.mechanism);
    auto& mech_cvs = R.M.mechanisms.at(p.mechanism).cv;
    mcable_list cables;

    for (auto i: util::count_along(mech_cvs)) {
        auto cv = mech_cvs[i];
        auto cv_cables = R.D.geometry.cables(cv);
        mextent cv_extent = mcable_list(cv_cables.begin(), cv_cables.end());
        for (auto cable: intersect(cv_extent, support)) {
            if (cable.prox_pos==cable.dist_pos) continue;

            r.raw_handles.push_back(data+i);
            cables.push_back(cable);
        }
    }
    r.metadata = std::move(cables);
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_point_state& p, probe_resolution_data<B>& R) {
    arb_assert(R.handles.size()==R.M.target_divs.back());
    arb_assert(R.handles.size()==R.M.n_target);

    const fvm_value_type* data = R.mechanism_state(p.mechanism, p.state);
    if (!data) return;

    // Convert cell-local target number to cellgroup target number.
    auto cg_target = p.target + R.M.target_divs[R.cell_idx];
    if (cg_target>=R.M.target_divs.at(R.cell_idx+1)) return;

    if (R.handles[cg_target].mech_id!=R.mech_instance_by_name.at(p.mechanism)->mechanism_id()) return;
    auto mech_index = R.handles[cg_target].mech_index;

    auto& multiplicity = R.M.mechanisms.at(p.mechanism).multiplicity;
    auto& placed_instances = R.cell.synapses().at(p.mechanism);

    auto opt_i = util::binary_search_index(placed_instances, p.target, [](auto& item) { return item.lid; });
    if (!opt_i) throw arbor_internal_error("inconsistent mechanism state");
    mlocation loc = placed_instances[*opt_i].loc;

    cable_probe_point_info metadata{p.target, multiplicity.empty()? 1u: multiplicity.at(mech_index), loc};

    R.result.push_back(fvm_probe_scalar{{data+mech_index}, metadata});
}

template <typename B>
void resolve_probe(const cable_probe_point_state_cell& p, probe_resolution_data<B>& R) {
    const fvm_value_type* data = R.mechanism_state(p.mechanism, p.state);
    if (!data) return;

    unsigned id = R.mech_instance_by_name.at(p.mechanism)->mechanism_id();
    auto& multiplicity = R.M.mechanisms.at(p.mechanism).multiplicity;
    auto& placed_instances = R.cell.synapses().at(p.mechanism);

    std::size_t cell_targets_base = R.M.target_divs[R.cell_idx];
    std::size_t cell_targets_end = R.M.target_divs[R.cell_idx+1];

    fvm_probe_multi r;
    std::vector<cable_probe_point_info> metadata;

    for (auto target: util::make_span(cell_targets_base, cell_targets_end)) {
        if (R.handles[target].mech_id!=id) continue;

        auto mech_index = R.handles[target].mech_index;
        r.raw_handles.push_back(data+mech_index);

        auto cell_target = target-cell_targets_base; // Convert to cell-local target index.

        auto opt_i = util::binary_search_index(placed_instances, cell_target, [](auto& item) { return item.lid; });
        if (!opt_i) throw arbor_internal_error("inconsistent mechanism state");
        mlocation loc = placed_instances[*opt_i].loc;

        metadata.push_back(cable_probe_point_info{
            cell_lid_type(cell_target), multiplicity.empty()? 1u: multiplicity.at(mech_index), loc});
    }

    r.metadata = std::move(metadata);
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_ion_current_density& p, probe_resolution_data<B>& R) {
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        auto opt_i = R.ion_location_index(p.ion, loc);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{R.state->ion_data.at(p.ion).iX_.data()+*opt_i}, loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_ion_current_cell& p, probe_resolution_data<B>& R) {
    if (!R.state->ion_data.count(p.ion)) return;

    auto& ion_cvs = R.M.ions.at(p.ion).cv;
    const fvm_value_type* src = R.state->ion_data.at(p.ion).iX_.data();

    fvm_probe_weighted_multi r;
    for (auto cv: R.D.geometry.cell_cvs(R.cell_idx)) {
        auto opt_i = util::binary_search_index(ion_cvs, cv);
        if (!opt_i) continue;

        const double* ptr = src+*opt_i;
        for (auto cable: R.D.geometry.cables(cv)) {
            double area = R.cell.embedding().integrate_area(cable); // [µm²]
            if (area>0) {
                r.raw_handles.push_back(ptr);
                r.weight.push_back(0.001*area); // Scale from [µm²·A/m²] to [nA].
                r.metadata.push_back(cable);
            }
        }
    }
    r.metadata.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_ion_int_concentration& p, probe_resolution_data<B>& R) {
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        auto opt_i = R.ion_location_index(p.ion, loc);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{R.state->ion_data.at(p.ion).Xi_.data()+*opt_i}, loc});
    }
}

template <typename B>
void resolve_probe(const cable_probe_ion_ext_concentration& p, probe_resolution_data<B>& R) {
    for (mlocation loc: thingify(p.locations, R.cell.provider())) {
        auto opt_i = R.ion_location_index(p.ion, loc);
        if (!opt_i) continue;

        R.result.push_back(fvm_probe_scalar{{R.state->ion_data.at(p.ion).Xo_.data()+*opt_i}, loc});
    }
}

// Common implementation for int and ext concentrations across whole cell:
template <typename B>
void resolve_ion_conc_common(const std::vector<fvm_index_type>& ion_cvs, const fvm_value_type* src, probe_resolution_data<B>& R) {
    fvm_probe_multi r;
    mcable_list cables;

    for (auto i: util::count_along(ion_cvs)) {
        for (auto cable: R.D.geometry.cables(ion_cvs[i])) {
            if (cable.prox_pos!=cable.dist_pos) {
                r.raw_handles.push_back(src+i);
                cables.push_back(cable);
            }
        }
    }
    r.metadata = std::move(cables);
    r.shrink_to_fit();
    R.result.push_back(std::move(r));
}

template <typename B>
void resolve_probe(const cable_probe_ion_int_concentration_cell& p, probe_resolution_data<B>& R) {
    if (!R.state->ion_data.count(p.ion)) return;
    resolve_ion_conc_common<B>(R.M.ions.at(p.ion).cv, R.state->ion_data.at(p.ion).Xi_.data(), R);
}

template <typename B>
void resolve_probe(const cable_probe_ion_ext_concentration_cell& p, probe_resolution_data<B>& R) {
    if (!R.state->ion_data.count(p.ion)) return;
    resolve_ion_conc_common<B>(R.M.ions.at(p.ion).cv, R.state->ion_data.at(p.ion).Xo_.data(), R);
}

} // namespace arb
