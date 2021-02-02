#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/common_types.hpp>
#include <arbor/constants.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/mcable_map.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/util/hash_def.hpp>
#include <arbor/util/typed_map.hpp>

namespace arb {

// Pair of indexes that describe range of local indices.
// Returned by cable_cell::place() calls, so that the caller can
// refer to targets, detectors, etc on the cell.
struct lid_range {
    cell_lid_type begin;
    cell_lid_type end;
    lid_range(cell_lid_type b, cell_lid_type e):
        begin(b), end(e) {}
};

// `cable_sample_range` is simply a pair of `const double*` pointers describing the sequence
// of double values associated with the cell-wide sample.

using cable_sample_range = std::pair<const double*, const double*>;


// Each kind of probe has its own type for representing its address, as below.
//
// Probe address specifications can be for _scalar_ data, associated with a fixed
// location or synapse on a cell, or _vector_ data, associated with multiple
// sites or sub-sections of a cell.
//
// Sampler functions receive an `any_ptr` to sampled data. The underlying pointer
// type is a const pointer to:
//     * `double` for scalar data;
//     * `cable_sample_range*` for vector data (see definition above).
//
// The metadata associated with a probe is also passed to a sampler via an `any_ptr`;
// the underlying pointer will be a const pointer to one of the following metadata types:
//     * `mlocation` for most scalar queries;
//     * `cable_probe_point_info` for point mechanism state queries;
//     * `mcable_list` for most vector queries;
//     * `std::vector<cable_probe_point_info>` for cell-wide point mechanism state queries.
//
// Scalar probes which are described by a locset expression will generate multiple
// calls to an attached sampler, one per valid location matched by the expression.
//
// Metadata for point process probes.
struct cable_probe_point_info {
    cell_lid_type target;   // Target number of point process instance on cell.
    unsigned multiplicity;  // Number of combined instances at this site.
    mlocation loc;          // Point on cell morphology where instance is placed.
};

// Voltage estimate [mV] at `location`, interpolated.
// Sample value type: `double`
// Sample metadata type: `mlocation`
struct cable_probe_membrane_voltage {
    locset locations;
};

// Voltage estimate [mV], reported against each cable in each control volume. Not interpolated.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mcable_list`
struct cable_probe_membrane_voltage_cell {};

// Axial current estimate [nA] at `location`, interpolated.
// Sample value type: `double`
// Sample metadata type: `mlocation`
struct cable_probe_axial_current {
    locset locations;
};

// Total current density [A/m²] across membrane _excluding_ capacitive current at `location`.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mlocation`
struct cable_probe_total_ion_current_density {
    locset locations;
};

// Total ionic current [nA] across membrane _excluding_ capacitive current across components of the cell.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mcable_list`
struct cable_probe_total_ion_current_cell {};

// Total membrane current [nA] across components of the cell.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mcable_list`
struct cable_probe_total_current_cell {};

// Value of state variable `state` in density mechanism `mechanism` in CV at `location`.
// Sample value type: `double`
// Sample metadata type: `mlocation`
struct cable_probe_density_state {
    locset locations;
    std::string mechanism;
    std::string state;
};

// Value of state variable `state` in density mechanism `mechanism` across components of the cell.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mcable_list`
struct cable_probe_density_state_cell {
    std::string mechanism;
    std::string state;
};

// Value of state variable `key` in point mechanism `source` at target `target`.
// Sample value type: `double`
// Sample metadata type: `cable_probe_point_info`
struct cable_probe_point_state {
    cell_lid_type target;
    std::string mechanism;
    std::string state;
};

// Value of state variable `key` in point mechanism `source` at every target with this mechanism.
// Metadata has one entry of type cable_probe_point_info for each matched (possibly coalesced) instance.
// Sample value type: `cable_sample_range`
// Sample metadata type: `std::vector<cable_probe_point_info>`
struct cable_probe_point_state_cell {
    std::string mechanism;
    std::string state;
};

// Current density [A/m²] across membrane attributed to the ion `source` at `location`.
// Sample value type: `double`
// Sample metadata type: `mlocation`
struct cable_probe_ion_current_density {
    locset locations;
    std::string ion;
};

// Total ionic current [nA] attributed to the ion `source` across components of the cell.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mcable_list`
struct cable_probe_ion_current_cell {
    std::string ion;
};

// Ionic internal concentration [mmol/L] of ion `source` at `location`.
// Sample value type: `double`
// Sample metadata type: `mlocation`
struct cable_probe_ion_int_concentration {
    locset locations;
    std::string ion;
};

// Ionic internal concentration [mmol/L] of ion `source` across components of the cell.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mcable_list`
struct cable_probe_ion_int_concentration_cell {
    std::string ion;
};

// Ionic external concentration [mmol/L] of ion `source` at `location`.
// Sample value type: `double`
// Sample metadata type: `mlocation`
struct cable_probe_ion_ext_concentration {
    locset locations;
    std::string ion;
};

// Ionic external concentration [mmol/L] of ion `source` across components of the cell.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mcable_list`
struct cable_probe_ion_ext_concentration_cell {
    std::string ion;
};

// Forward declare the implementation, for PIMPL.
struct cable_cell_impl;


// Typed maps for access to painted and placed assignments:
//
// Mechanisms and initial ion data are further keyed by
// mechanism name and ion name respectively.

template <typename T>
using region_assignment =
    std::conditional_t<
        std::is_same<T, mechanism_desc>::value || std::is_same<T, init_int_concentration>::value ||
        std::is_same<T, init_ext_concentration>::value || std::is_same<T, init_reversal_potential>::value,
        std::unordered_map<std::string, mcable_map<T>>,
        mcable_map<T>>;

template <typename T>
struct placed {
    mlocation loc;
    cell_lid_type lid;
    T item;
};

// Note: lid fields of elements of mlocation_map used in cable_cell are strictly increasing.
template <typename T>
using mlocation_map = std::vector<placed<T>>;

template <typename T>
using location_assignment =
    std::conditional_t<
        std::is_same<T, mechanism_desc>::value,
        std::unordered_map<std::string, mlocation_map<T>>,
        mlocation_map<T>>;

using cable_cell_region_map = static_typed_map<region_assignment,
    mechanism_desc, init_membrane_potential, axial_resistivity,
    temperature_K, membrane_capacitance, init_int_concentration,
    init_ext_concentration, init_reversal_potential>;

using cable_cell_location_map = static_typed_map<location_assignment,
    mechanism_desc, i_clamp, gap_junction_site, threshold_detector>;

// High-level abstract representation of a cell.
class cable_cell {
public:
    using index_type = cell_lid_type;
    using size_type = cell_local_size_type;
    using value_type = double;

    using gap_junction_instance = mlocation;

    // Default constructor.
    cable_cell();

    // Copy and move constructors.
    cable_cell(const cable_cell& other);
    cable_cell(cable_cell&& other) = default;

    // Copy and move assignment operators.
    cable_cell& operator=(cable_cell&&) = default;
    cable_cell& operator=(const cable_cell& other) {
        return *this = cable_cell(other);
    }

    /// Construct from morphology, label and decoration descriptions.
    cable_cell(const class morphology&, const label_dict&, const decor&);
    cable_cell(const class morphology& m):
        cable_cell(m, {}, {})
    {}

    /// Access to morphology and embedding
    const concrete_embedding& embedding() const;
    const arb::morphology& morphology() const;
    const mprovider& provider() const;

    // Convenience access to placed items.

    const std::unordered_map<std::string, mlocation_map<mechanism_desc>>& synapses() const {
        return location_assignments().get<mechanism_desc>();
    }

    const mlocation_map<gap_junction_site>& gap_junction_sites() const {
        return location_assignments().get<gap_junction_site>();
    }

    const mlocation_map<threshold_detector>& detectors() const {
        return location_assignments().get<threshold_detector>();
    }

    const mlocation_map<i_clamp>& stimuli() const {
        return location_assignments().get<i_clamp>();
    }

    // Access to a concrete list of locations for a locset.
    mlocation_list concrete_locset(const locset&) const;

    // Access to a concrete list of cable segments for a region.
    mextent concrete_region(const region&) const;

    // Generic access to painted and placed items.
    const cable_cell_region_map& region_assignments() const;
    const cable_cell_location_map& location_assignments() const;

    // The decorations on the cell.
    const decor& decorations() const;

    // The default parameter and ion settings on the cell.
    const cable_cell_parameter_set& default_parameters() const;

    // The range of lids assigned to the items with placement index idx, where
    // the placement index is the value returned by calling decor::place().
    lid_range placed_lid_range(unsigned idx) const;

private:
    std::unique_ptr<cable_cell_impl, void (*)(cable_cell_impl*)> impl_;
};

} // namespace arb

ARB_DEFINE_HASH(arb::cable_probe_point_info, a.target, a.multiplicity, a.loc);
