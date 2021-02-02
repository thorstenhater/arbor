#include <optional>
#include <unordered_map>
#include <utility>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

namespace arb {

size_t label_dict::size() const {
    return locsets_.size() + regions_.size();
}

void label_dict::set(const std::string& name, arb::locset ls) {
    if (regions_.count(name)) {
        throw label_type_mismatch(name);
    }
    locsets_[name] = std::move(ls);
}

void label_dict::set(const std::string& name, arb::region reg) {
    if (locsets_.count(name)) {
        throw label_type_mismatch(name);
    }
    regions_[name] = std::move(reg);
}

void label_dict::import(const label_dict& other, const std::string& prefix) {
    for (const auto& entry: other.locsets()) {
        set(prefix+entry.first, entry.second);
    }
    for (const auto& entry: other.regions()) {
        set(prefix+entry.first, entry.second);
    }
}

std::optional<region> label_dict::region(const std::string& name) const {
    auto it = regions_.find(name);
    if (it==regions_.end()) return {};
    return it->second;
}

std::optional<locset> label_dict::locset(const std::string& name) const {
    auto it = locsets_.find(name);
    if (it==locsets_.end()) return {};
    return it->second;
}

const std::unordered_map<std::string, locset>& label_dict::locsets() const {
    return locsets_;
}

const std::unordered_map<std::string, region>& label_dict::regions() const {
    return regions_;
}

} // namespace arb
