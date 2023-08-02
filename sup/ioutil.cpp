#include <fstream>
#include <utility>

#include <sup/ioutil.hpp>
#include <sup/strsub.hpp>

namespace sup {

ARB_SUP_API std::fstream open_or_throw(const std::filesystem::path& p, std::ios_base::openmode mode, bool exclusive) {
    if (exclusive && std::filesystem::exists(p)) throw std::runtime_error(strsub("file % already exists", p));
    std::fstream file;
    file.open(p, mode);
    if (!file) throw std::runtime_error(strsub("unable to open file %", p));
    return file;
}

} // namespace sup

