#pragma once

#include <cstddef>
#include <algorithm>
#include <vector>

#include <arbor/assert.hpp>

namespace arb {

template <typename T>
class partitioned_vector {
public:
    using value_type = T;
    using count_type = unsigned;

    partitioned_vector() = default;
    
    partitioned_vector(std::vector<value_type>&& v, std::vector<count_type>&& p) :
        values_(std::move(v)),
        partition_(std::move(p)) {
        arb_assert(std::is_sorted(partition_.begin(), partition_.end()));
        arb_assert(partition_.back() == values_.size());
    }

    /// extract value by partition and offset
    const T& value(count_type part, count_type off) const {
        arb_assert(part + 1 < partition_.size());
        auto lo = partition_[part];
        arb_assert(off + lo < partition_[part + 1]);
        return values_[lo + off];
    }
    
    /// the partition of distribution
    const std::vector<count_type>& partition() const { return partition_; }

    /// the number of entries in the gathered vector in partition i
    count_type count(std::size_t i) const { return partition_[i+1] - partition_[i]; }

    /// the values in the gathered vector
    const std::vector<value_type>& values() const { return values_; }

    /// the size of the gathered vector
    std::size_t size() const { return values_.size(); }

    std::size_t partition_size() const { return partition_.empty() ? 0 : partition_.size() - 1; }

    /// append another part-vector
    void append(const partitioned_vector<T>& rhs) {
        for(const auto& val: rhs.values_) values_.push_back(val);
        if (partition_.empty()) partition_.push_back(0);
        auto off = partition_.back();
        for (auto idx = 1ul; idx < rhs.partition_.size(); ++idx) {
            partition_.push_back(rhs.partition_[idx] + off);
        }
    }

    void clear() { values_.clear(); partition_.clear(); }
    
    
private:
    std::vector<value_type> values_;
    std::vector<count_type> partition_;
};

} // namespace arb
