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

#ifndef __POINT_LAYER_H__
#define __POINT_LAYER_H__

#include <utility>

#include "../utils/traits.h"
#include "data_layer.h"
#include "utils.h"

namespace fdapde {
namespace internals {

template <typename GeoFrame_> struct point_layer {
    using GeoFrame = typename std::decay_t<GeoFrame_>;
    using storage_t = plain_data_layer<GeoFrame::Order>;
    using index_t = typename storage_t::index_t;
    using size_t = typename storage_t::size_t;
    using layer_category = layer_t::point_t;
    template <typename T> using reference = typename storage_t::template reference<T>;
    template <typename T> using const_reference = typename storage_t::template const_reference<T>;
    static constexpr int Order = GeoFrame::Order;
    static constexpr int local_dim = GeoFrame::local_dim;
    static constexpr int embed_dim = GeoFrame::embed_dim;

    point_layer() : geoframe_(nullptr), coords_() { }
    point_layer(const std::string& layer_name, GeoFrame_* geoframe) noexcept :
        geoframe_(geoframe), coords_() { }
    template <typename CoordsType>
        requires(std::is_convertible_v<CoordsType, DMatrix<double>>)
    point_layer(const std::string& layer_name, GeoFrame_* geoframe, const std::shared_ptr<CoordsType>& coords) noexcept
        :
        geoframe_(geoframe), coords_(coords) { }

    // observers
    size_t rows() { data_.rows(); }
    size_t cols() { data_.cols(); }
    size_t size() const { return rows() * cols(); }
    const auto& field_descriptor(const std::string& colname) const { return data_.field_descriptor(colname); }
    const auto& field_descriptors() const { return data_.field_descriptors(); }
    Eigen::Matrix<double, embed_dim, 1> geometry(int i) const { return coordinates().row(i); }
    bool contains(const std::string& column) const { return data_.contains(column); }

    template <typename... DataT> void set_data(DataT&&... data) {
        data_ = storage_t(std::forward<DataT>(data)...);
    }
    storage_t& data() { return data_; }
    const storage_t& data() const { return data_; }
    // geometry
    const DMatrix<double>& coordinates() const { return coords_ ? *coords_ : triangulation().nodes(); }
    Triangulation<local_dim, embed_dim>& triangulation() { return geoframe_->triangulation(); }
    const Triangulation<local_dim, embed_dim>& triangulation() const { return geoframe_->triangulation(); }
    DVector<int> locate() const { return triangulation().locate(*coords_); }
   private:
    GeoFrame_* geoframe_;
    std::shared_ptr<DMatrix<double>> coords_;
    storage_t data_;
};
  
}   // namespace internals
}   // namespace fdapde

#endif // __POINT_LAYER_H__
