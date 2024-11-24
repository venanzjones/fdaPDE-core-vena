// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __GEOFRAME_UTILS_H__
#define __GEOFRAME_UTILS_H__

namespace fdapde {

// generates a sequence of strings as {"base_0", "base_1", \ldots, "base_{n-1}"}
std::vector<std::string> seq(const std::string& base, int n) {
    std::vector<std::string> vec;
    vec.reserve(n);
    for (int i = 0; i < n; ++i) { vec.emplace_back(base + std::to_string(i)); }
    return vec;
}

// generates a sequence of arithmetic values as {base + 0, base + 1, \ldots, base + n-1}
template <typename T>
    requires(std::is_arithmetic_v<T>)
std::vector<int> seq(T begin, int n, int by = 1) {
    std::vector<T> vec;
    vec.reserve(n);
    for (int i = 0; i < n; i += by) { vec.emplace_back(begin + i); }
    return vec;
}

namespace layer_t {

struct point_t { } point;
struct areal_t { } areal;

}   // namespace layer_t

namespace internals {
enum ltype { point = 0, areal = 1 };

inline void throw_geoframe_error(const std::string& msg) { throw std::runtime_error("GeoFrame: " + msg); }

#define geoframe_assert(condition, msg)                                                                                \
    if (!(condition)) { internals::throw_geoframe_error(msg); }

}   // namespace internals
}   // namespace fdapde

#endif   // __GEOFRAME_UTILS_H__
