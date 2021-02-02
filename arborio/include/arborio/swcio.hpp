#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/morph/morphology.hpp>

namespace arborio {

// SWC exceptions are thrown by `parse_swc`, and correspond
// to inconsistent, or in `strict` mode, dubious SWC data.

struct swc_error: public arb::arbor_exception {
    swc_error(const std::string& msg, int record_id);
    int record_id;
};

// Parent id in record has no corresponding SWC record,
// nor is the record the root record with parent id -1.
struct swc_no_such_parent: swc_error {
    explicit swc_no_such_parent(int record_id);
};

// Parent id is greater than or equal to record id.
struct swc_record_precedes_parent: swc_error {
    explicit swc_record_precedes_parent(int record_id);
};

// Multiple records cannot have the same id.
struct swc_duplicate_record_id: swc_error {
    explicit swc_duplicate_record_id(int record_id);
};

// Smells like a spherical soma.
struct swc_spherical_soma: swc_error {
    explicit swc_spherical_soma(int record_id);
};

// Smells like a non-spherical soma.
struct swc_non_spherical_soma: swc_error {
    explicit swc_non_spherical_soma(int record_id);
};

// Missing soma.
struct swc_no_soma: swc_error {
    explicit swc_no_soma(int record_id);
};

// Non-consecutive soma samples.
struct swc_non_consecutive_soma: swc_error {
    explicit swc_non_consecutive_soma(int record_id);
};

// Non-serial soma samples.
struct swc_non_serial_soma: swc_error {
    explicit swc_non_serial_soma(int record_id);
};

// Sample connecting to the middle of a soma causing an unsupported branch.
struct swc_branchy_soma: swc_error {
    explicit swc_branchy_soma(int record_id);
};

// Sample connecting to the middle of a soma causing an unsupported branch.
struct swc_collocated_soma: swc_error {
    explicit swc_collocated_soma(int record_id);
};

// Sample is not part of a segment
struct swc_single_sample_segment: swc_error {
    explicit swc_single_sample_segment(int record_id);
};

// Segment cannot have samples with different tags
struct swc_mismatched_tags: swc_error {
    explicit swc_mismatched_tags(int record_id);
};

// Only tags 1, 2, 3, 4 supported
struct swc_unsupported_tag: swc_error {
    explicit swc_unsupported_tag(int record_id);
};

// No gaps allowed
struct swc_unsupported_gaps: swc_error {
    explicit swc_unsupported_gaps(int record_id);
};

// Can't form a segment from a single sample
struct swc_bad_description: swc_error {
    explicit swc_bad_description(int record_id);
};

struct swc_record {
    int id = 0;          // sample number
    int tag = 0;         // structure identifier (tag)
    double x = 0;        // sample coordinates
    double y = 0;
    double z = 0;
    double r = 0;        // sample adius
    int parent_id= -1;   // record parent's sample number

    swc_record() = default;
    swc_record(int id, int tag, double x, double y, double z, double r, int parent_id):
        id(id), tag(tag), x(x), y(y), z(z), r(r), parent_id(parent_id)
    {}

    bool operator==(const swc_record& other) const {
        return id == other.id &&
            x == other.x &&
            y == other.y &&
            z == other.z &&
            r == other.r &&
            parent_id == other.parent_id;
    }

    bool operator!=(const swc_record& other) const {
        return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream&, const swc_record&);
    friend std::istream& operator>>(std::istream&, swc_record&);
};

struct swc_data {
private:
    std::string metadata_;
    std::vector<swc_record> records_;

public:
    swc_data() = delete;
    swc_data(std::vector<arborio::swc_record>);
    swc_data(std::string, std::vector<arborio::swc_record>);

    const std::vector<swc_record>& records() const {return records_;};
    std::string metadata() const {return metadata_;};
};

// Read SWC records from stream, collecting any initial metadata represented
// in comments (stripping initial '#' and subsequent whitespace).
// Stops at EOF or after reading the first line that does not parse as SWC.
//
// In `relaxed` mode, it will check that:
//     * There are no duplicate record ids.
//     * All record ids are positive.
//     * There are no records whose parent id is not less than the record id.
//     * Only one record has parent id -1; all other parent ids correspond to records.
//
// In `strict` mode, it will additionally check that the data cannot be interpreted
// as a 'spherical soma' SWC file:
//     * The root record must share its tag with at least one other record
//       which has the root as parent. This implies that there must be at least
//       two SWC records.
//
// Throws a corresponding exception of type derived from `swc_error` if any of the
// conditions above are encountered.
//
// SWC records are returned in id order.

swc_data parse_swc(std::istream&);
swc_data parse_swc(const std::string&);

// Convert a valid, ordered sequence of SWC records into a morphology.
//
// Note that 'one-point soma' SWC files are explicitly not supported.
//
// The segments of the generated morphology  will be contiguous. There will be
// one segment for each SWC record after the first: this record defines the tag
// and distal point of the segment, while the proximal point is taken from the
// parent record.

arb::morphology load_swc_arbor(const swc_data& data);

// As above, will convert a valid, ordered sequence of SWC records into a morphology
//
// Note that 'one-point soma' SWC files are supported here
//
// These functions comply with inferred SWC rules from the Allen institute and Neuron.
// These rules are explicitly listed in the docs.

arb::morphology load_swc_neuron(const swc_data& data);
arb::morphology load_swc_allen(const swc_data& data, bool no_gaps=false);

} // namespace arborio
