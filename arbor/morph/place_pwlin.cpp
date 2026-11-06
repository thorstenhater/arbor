#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/math.hpp>

#include "util/piecewise.hpp"
#include "util/ratelem.hpp"
#include "util/span.hpp"

namespace arb {

using util::rat_element;

namespace {

// Arithmetic on the cartesian parts of a sample point. A radius plays no part in
// finding the nearest point, so these leave it at zero; the arithmetic is that of
// arb::distance, spelled out here so that it stays inlined.
inline mpoint operator-(const mpoint& a, const mpoint& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, 0};
}

inline mpoint operator+(const mpoint& a, const mpoint& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, 0};
}

inline mpoint operator*(double u, const mpoint& a) {
    return {u*a.x, u*a.y, u*a.z, 0};
}

inline double dot(const mpoint& a, const mpoint& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline double norm(const mpoint& a) {
    return std::sqrt(dot(a, a));
}

} // namespace

// Distance from p to a segment, and the corresponding position along its branch.
// `lo` and `hi` are the segment's bounds in branch coordinates.
//
// This is all_closest()'s arithmetic and nothing else, and it is the only place
// that arithmetic lives, so the full scan and the index cannot drift apart.
static inline std::pair<double, double> segment_distance(const msegment& seg, double lo, double hi, const mpoint& p) {
    const mpoint v = seg.prox;
    const mpoint w = seg.dist;
    const mpoint vw = w - v;
    const double wvs = dot(vw, vw);
    if (wvs==0.) { // zero length segment is a special case
        return {norm(p - v), lo};
    }

    // Find the relative position of the orthogonal projection onto the line segment
    // that along the axis of the segment:
    //   t=0 -> proximal end of the segment
    //   t=1 -> distal end of the segment
    // values are clamped to the range [0, 1]
    const double t = std::max(0., std::min(1., dot(vw, p - v)/wvs));
    const double distance =
        t<=0.? norm(p - v):
        t>=1.? norm(p - w):
               norm(p - (v + t*vw));
    return {distance, math::lerp(lo, hi, t)};
}

namespace {

// An axis-aligned bounding volume hierarchy over the placed segments, so that
// closest() need not look at every one of them. Built only when asked for;
// immutable once built, so it is shared by every copy of the owning place_pwlin.
//
// The index stores no coordinates: an entry names a segment of
// place_pwlin_data::segments, and the endpoints are read from there. Those are
// already isometry-applied, so an index and a scan describe the same placed
// object.
//
// A query returns exactly what a full scan returns, tie policy included; see the
// argument on pwlin_index::closest().

struct pwlin_node {
    std::array<double, 3> lo, hi; // bounds of everything below, widened outward
    std::uint32_t left = 0;       // children of an internal node
    std::uint32_t right = 0;
    std::uint32_t first = 0;      // first entry of a leaf, into pwlin_index::entries
    std::uint32_t count = 0;      // entries in a leaf; zero marks an internal node
};

// One segment as the index sees it: which segment of place_pwlin_data::segments
// it stands for, and where on its branch it lies.
struct pwlin_entry {
    std::uint32_t seq = 0;  // position in place_pwlin_data::segments
    msize_t branch = 0;
    double lo = 0, hi = 0;
};

struct pwlin_index {
    using node = pwlin_node;
    using entry = pwlin_entry;

    std::vector<node> nodes;
    std::vector<entry> entries; // permuted, so that a leaf is a contiguous range
    std::uint32_t root = 0;

    static constexpr unsigned leaf_size = 8;
    static constexpr unsigned max_depth = 32;

    bool empty() const { return nodes.empty(); }

    static double coord(int k, const mpoint& p) {
        return k==0? p.x: k==1? p.y: p.z;
    }
    static double mid_coord(int k, const msegment& s) {
        return 0.5*(coord(k, s.prox) + coord(k, s.dist));
    }
    static const msegment& segment_of(const std::vector<msegment>& segments,
                                      const std::vector<entry>& in, std::uint32_t i) {
        return segments[in[i].seq];
    }

    // Bounds of the segments at order[b, e), and bounds of their centroids.
    static void bounds_of(const std::vector<msegment>& segments, const std::vector<entry>& in,
                          const std::vector<std::uint32_t>& order, std::uint32_t b, std::uint32_t e,
                          std::array<double, 3>& lo, std::array<double, 3>& hi,
                          std::array<double, 3>& clo, std::array<double, 3>& chi) {
        const auto maxd = std::numeric_limits<double>::max();
        lo  = { maxd, maxd, maxd};
        hi  = {-maxd,-maxd,-maxd};
        clo = { maxd, maxd, maxd};
        chi = {-maxd,-maxd,-maxd};

        for (auto i = b; i<e; ++i) {
            const msegment& s = segment_of(segments, in, order[i]);
            for (int k = 0; k<3; ++k) {
                lo[k] = std::min({lo[k], coord(k, s.prox), coord(k, s.dist)});
                hi[k] = std::max({hi[k], coord(k, s.prox), coord(k, s.dist)});
                clo[k] = std::min(clo[k], mid_coord(k, s));
                chi[k] = std::max(chi[k], mid_coord(k, s));
            }
        }
    }

    // One ulp of slack outwards on each face. Rounding in the bounds above could
    // otherwise leave a box fractionally inside its own segments, and a lower
    // bound that is not one could discard the answer.
    static double widen_lo(double v) {
        return std::nextafter(v, -std::numeric_limits<double>::infinity());
    }
    static double widen_hi(double v) {
        return std::nextafter(v, std::numeric_limits<double>::infinity());
    }

    static double box_distance2(const node& nd, const mpoint& p) {
        const double dx = std::max(0., std::max(nd.lo[0] - p.x, p.x - nd.hi[0]));
        const double dy = std::max(0., std::max(nd.lo[1] - p.y, p.y - nd.hi[1]));
        const double dz = std::max(0., std::max(nd.lo[2] - p.z, p.z - nd.hi[2]));
        return dx*dx + dy*dy + dz*dz;
    }

    // Top-down median split along the widest centroid axis. `in` stays in
    // traversal order; the recursion permutes an index vector only.
    std::uint32_t build_node(const std::vector<msegment>& segments,
                             const std::vector<entry>& in,
                             std::vector<std::uint32_t>& order,
                             std::uint32_t b,
                             std::uint32_t e,
                             unsigned depth) {
        const std::uint32_t idx = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back({});

        std::array<double, 3> lo, hi, clo, chi;
        bounds_of(segments, in, order, b, e, lo, hi, clo, chi);

        node nd;
        for (int k = 0; k<3; ++k) {
            nd.lo[k] = widen_lo(lo[k]);
            nd.hi[k] = widen_hi(hi[k]);
        }

        const std::uint32_t n = e - b;
        int axis = -1;
        if (n > leaf_size && depth < max_depth) {
            double widest = 0;
            for (int k = 0; k<3; ++k) {
                if (const double extent = chi[k] - clo[k]; extent>widest) {
                    widest = extent;
                    axis = k;
                }
            }
            // Coincident centroids still have to split, or the recursion never
            // terminates on a bundle of zero-length or coincident segments.
            if (widest<=0) axis = static_cast<int>(depth % 3);
        }

        if (axis<0) {
            nd.first = b;
            nd.count = n;
            nodes[idx] = nd;
            return idx;
        }

        const int ax = axis;
        const std::uint32_t mid = b + n/2;
        std::nth_element(order.begin() + b, order.begin() + mid, order.begin() + e,
            [&](std::uint32_t l, std::uint32_t r) {
                return mid_coord(ax, segment_of(segments, in, l)) < mid_coord(ax, segment_of(segments, in, r));
            });

        nodes[idx] = nd;
        const std::uint32_t left = build_node(segments, in, order, b, mid, depth + 1);
        const std::uint32_t right = build_node(segments, in, order, mid, e, depth + 1);
        nodes[idx].left = left;
        nodes[idx].right = right;
        nodes[idx].count = 0;
        return idx;
    }

    // `in` must be in traversal order (branch-major over place_pwlin_data), which
    // is the order a full scan resolves ties by.
    void build(std::vector<entry> in, const std::vector<msegment>& segments) {
        if (in.empty()) return;

        std::vector<std::uint32_t> order(in.size());
        std::iota(order.begin(), order.end(), 0u);

        nodes.reserve(2*(in.size()/leaf_size + 2));
        root = build_node(segments, in, order, 0, static_cast<std::uint32_t>(order.size()), 0);

        entries.resize(order.size());
        for (std::size_t i = 0; i<order.size(); ++i) entries[i] = in[order[i]];
    }

    // Closest location to (x, y, z), identical to a scan of every segment.
    //
    // A node's box contains its segments, so the distance to the box bounds below
    // the distance to any of them. The incumbent is always the distance of some
    // real segment, so it is never below the true minimum d*. Skipping a node
    // therefore needs box > incumbent, which makes every segment below it farther
    // than the incumbent >= d*: none is nearest, and none ties with one. The
    // comparison has to be strict, because closest() prefers the most proximal
    // location among equidistant ones, and a non-strict test could skip the one a
    // scan lists first.
    //
    // Comparing squared distances is equivalent only up to the rounding of sqrt:
    // two distinct squared distances can share a square root, and the scan treats
    // those as a tie. A relative margin of four epsilons makes this test strictly
    // more conservative than the unsquared one, so it cannot skip a tie.
    //
    // Ties go to the lowest seq. A scan takes the first minimum in traversal
    // order, and traversal order is the order segments were filled, so the lowest
    // seq is what a scan returns however this traversal visits them.
    std::pair<mlocation, double> closest(const std::vector<msegment>& segments, double x, double y, double z) const {
        mpoint p{x, y, z, 0};
        double best = std::numeric_limits<double>::max();
        double best2 = best;
        mlocation best_loc;
        std::uint32_t best_seq = std::numeric_limits<std::uint32_t>::max();
        if (empty()) return {best_loc, best};

        // Depth is capped at build time, so a fixed stack suffices: no recursion
        // and no allocation per query.
        std::array<std::uint32_t, 2*max_depth + 2> stack;
        int sp = 0;
        stack[sp++] = root;

        constexpr double margin = 1.0 + 4.0*std::numeric_limits<double>::epsilon();
        while (sp) {
            const node& nd = nodes[stack[--sp]];
            if (box_distance2(nd, p)>best2*margin) continue;

            if (nd.count) {
                for (auto i = nd.first, e = nd.first + nd.count; i<e; ++i) {
                    const entry& s = entries[i];
                    const auto [distance, pos] = segment_distance(segments[s.seq], s.lo, s.hi, p);
                    if (distance<best || (distance == best && s.seq<best_seq)) {
                        best = distance;
                        best2 = distance*distance;
                        best_loc = mlocation{.branch=s.branch, .pos=pos};
                        best_seq = s.seq;
                    }
                }
            }
            else {
                // The nearer child first: a tighter incumbent earlier prunes more
                // of the tree.
                const std::uint32_t l = nd.left, r = nd.right;
                if (box_distance2(nodes[l], p) < box_distance2(nodes[r], p)) {
                    stack[sp++] = r;
                    stack[sp++] = l;
                }
                else {
                    stack[sp++] = l;
                    stack[sp++] = r;
                }
            }
        }
        return {best_loc, best};
    }
};

} // namespace

struct place_pwlin_data {
    // Piecewise-constant indices into segment data, by branch.
    std::vector<util::pw_elements<std::size_t>> segment_index;

    // Segments from segment tree, after isometry is applied.
    std::vector<msegment> segments;

    // Spatial index over segments, used by closest() when built.
    pwlin_index index;

    explicit place_pwlin_data(msize_t n_branch):
        segment_index(n_branch)
    {}

    // Entries in traversal order, which is also the order a full scan resolves
    // ties by, so the position of an entry in this vector is its tie-break key.
    void make_index() {
        std::vector<pwlin_entry> in;
        in.reserve(segments.size());
        for (msize_t bid: util::count_along(segment_index)) {
            for (auto sidx: segment_index[bid]) {
                in.push_back(pwlin_entry{
                    .seq = static_cast<std::uint32_t>(sidx.value),
                    .branch = bid,
                    .lo = sidx.lower_bound(),
                    .hi = sidx.upper_bound()});
            }
        }
        index.build(std::move(in), segments);
    }
};

static mpoint interpolate_segment(const std::pair<double, double>& bounds, const msegment& seg, double pos) {
    if (bounds.first==bounds.second) {
        return seg.prox;
    }
    else {
        double u = (pos-bounds.first)/(bounds.second-bounds.first);

        util::rat_element<1, 0> x{seg.prox.x, seg.dist.x};
        util::rat_element<1, 0> y{seg.prox.y, seg.dist.y};
        util::rat_element<1, 0> z{seg.prox.z, seg.dist.z};
        util::rat_element<1, 0> r{seg.prox.radius, seg.dist.radius};

        return {x(u), y(u), z(u), r(u)};
    }
}

template <typename Elem>
static bool is_degenerate(const util::pw_elements<Elem>& pw) {
    return  pw.bounds().second==0;
}

mpoint place_pwlin::at(mlocation loc) const {
    const auto& pw_index = data_->segment_index.at(loc.branch);
    double pos = is_degenerate(pw_index)? 0: loc.pos;

    auto index = pw_index(pos);
    return interpolate_segment(index.extent, data_->segments.at(index), pos);
}

std::vector<mpoint> place_pwlin::all_at(mlocation loc) const {
    std::vector<mpoint> result;
    const auto& pw_index = data_->segment_index.at(loc.branch);
    double pos = is_degenerate(pw_index)? 0: loc.pos;

    for (auto index: util::make_range(pw_index.equal_range(pos))) {
        auto bounds = index.extent;
        auto seg = data_->segments.at(index);

        // Add both ends of zero length segment, if they differ.
        if (bounds.first==bounds.second && seg.prox!=seg.dist) {
            result.push_back(seg.prox);
            result.push_back(seg.dist);
        }
        else {
            result.push_back(interpolate_segment(bounds, seg, pos));
        }
    }
    return result;
}

template <bool exclude_trivial>
static std::vector<msegment> extent_segments_impl(const place_pwlin_data& data, const mextent& extent) {
    std::vector<msegment> result;

    for (mcable c: extent) {
        const auto& pw_index = data.segment_index.at(c.branch);
        if (is_degenerate(pw_index)) {
            c.prox_pos = c.dist_pos = 0;
        }

        auto b = pw_index.equal_range(c.prox_pos).first;
        auto e = pw_index.equal_range(c.dist_pos).second;

        for (const auto [bounds, index]: util::make_range(b, e)) {
            const msegment& seg = data.segments.at(index);

            auto partial_bounds = bounds;
            msegment partial = seg;

            if (c.prox_pos>bounds.first) {
                arb_assert(c.prox_pos<=bounds.second);
                partial.prox = interpolate_segment(bounds, seg, c.prox_pos);
                partial_bounds.first = c.prox_pos;
            }
            if (c.dist_pos<bounds.second) {
                arb_assert(c.dist_pos>=bounds.first);
                partial.dist = interpolate_segment(bounds, seg, c.dist_pos);
                partial_bounds.second = c.dist_pos;
            }

            // With exclude_trivial set, skip zero-length segments if cable is non-trivial.
            if (exclude_trivial && partial_bounds.first==partial_bounds.second && c.prox_pos!=c.dist_pos) {
                continue;
            }

            result.push_back(partial);

            // With exclude_trivial set, keep only one zero-length (partial) segment if cable is trivial.
            if (exclude_trivial && c.prox_pos==c.dist_pos) {
                break;
            }
        }
    }

    return result;
}

std::vector<msegment> place_pwlin::all_segments(const mextent& extent) const {
    return extent_segments_impl<false>(*data_, extent);
}

std::vector<msegment> place_pwlin::segments(const mextent& extent) const {
    return extent_segments_impl<true>(*data_, extent);
}

place_pwlin::place_pwlin(const arb::morphology& m, const isometry& iso, bool with_index) {
    msize_t n_branch = m.num_branches();
    data_ = std::make_shared<place_pwlin_data>(n_branch);

    if (!n_branch) return;

    std::vector<double> seg_pos;
    for (msize_t bid = 0; bid<n_branch; ++bid) {
        auto& segments = m.branch_segments(bid);
        arb_assert(!segments.empty());

        seg_pos.reserve(segments.size()+1);
        seg_pos.clear();
        seg_pos.push_back(0);
        for (auto& seg: segments) {
            seg_pos.push_back(seg_pos.back()+distance(seg.prox, seg.dist));
        }

        double branch_length = seg_pos.back();
        if (branch_length!=0) {
            for (auto& x: seg_pos) {
                x /= branch_length;
            }
        }

        for (auto i: util::count_along(segments)) {
            msegment seg = segments[i];
            double p0 = seg_pos[i];
            double p1 = seg_pos[i+1];

            seg.prox = iso.apply(seg.prox);
            seg.dist = iso.apply(seg.dist);

            data_->segment_index[bid].push_back(p0, p1, data_->segments.size());
            data_->segments.push_back(seg);
        }
    }

    if (with_index) data_->make_index();
};

std::pair<std::vector<mlocation>, double> place_pwlin::all_closest(double x, double y, double z) const {
    double mind = std::numeric_limits<double>::max();
    mpoint p{x, y, z, 0};
    std::vector<mlocation> locs;

    // loop over each branch
    for (msize_t bid: util::count_along(data_->segment_index)) {
        const auto& branch = data_->segment_index[bid];
        // loop over the segments in the branch
        for (auto sidx: branch) {
            const auto [distance, pos] = segment_distance(data_->segments[sidx.value],
                                                          sidx.lower_bound(), sidx.upper_bound(), p);
            mlocation loc{.branch=bid, .pos=pos};
            if (distance<mind) {
                mind = distance;
                locs = {loc};
            }
            else if (distance == mind) {
                locs.push_back(loc);
            }
        }
    }
    return {locs, mind};
}

// Policy:
//  If two collated points are equidistant from the input point, take the
//  proximal location.
// Rationale:
//  if the location is on a fork point, it makes sense to take the proximal
//  location, which corresponds to the end of the parent branch.
std::pair<mlocation, double> place_pwlin::closest(double x, double y, double z) const {
    if (!data_->index.empty()) return data_->index.closest(data_->segments, x, y, z);

    const auto& [locs, delta] = all_closest(x, y, z);
    return {locs.front(), delta};
}

} // namespace arb
