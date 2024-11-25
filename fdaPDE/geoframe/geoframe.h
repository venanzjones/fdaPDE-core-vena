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

#ifndef __GEOFRAME_H__
#define __GEOFRAME_H__

#include <filesystem>
#include <memory>
#include <optional>

#include "../geometry/polygon.h"
#include "../geometry/triangulation.h"
#include "../linear_algebra/mdarray.h"
#include "../linear_algebra/binary_matrix.h"
#include "../utils/symbols.h"
#include "../utils/traits.h"
#include "csv.h"
#include "shp.h"
#include "data_layer.h"
#include "areal_layer.h"
#include "point_layer.h"
#include "utils.h"

namespace fdapde {

template <typename Triangulation_, int Order_ = 2> struct GeoFrame {
    fdapde_static_assert(Order_ > 1, GEOFRAME_MUST_HAVE_ORDER_TWO_OR_HIGHER);
   private:
    using This = GeoFrame<Triangulation_, Order_>;
    using point_layer_t = internals::point_layer<This>;
    using areal_layer_t = internals::areal_layer<This>;
    using layers_t = std::tuple<point_layer_t, areal_layer_t>;
    template <typename... Ts> using LayerMap_ = std::tuple<std::unordered_map<std::string, Ts>...>;
    template <typename T, typename U> auto& fetch_(U& u) { return std::get<index_of_type<T, layers_t>::index>(u); }
    template <typename T, typename U> const auto& fetch_(const U& u) const {
        return std::get<index_of_type<T, layers_t>::index>(u);
    }
    template <typename Layer_>
        requires(requires(Layer_ l) { typename Layer_::layer_category; })
    struct is_supported_layer {
        using Layer = std::decay_t<Layer_>;
        static constexpr bool value = std::is_same_v<typename Layer::layer_category, layer_t::point_t> ||
                                      std::is_same_v<typename Layer::layer_category, layer_t::areal_t>;
    };
    template <typename Layer_> static constexpr bool is_supported_layer_v = is_supported_layer<Layer_>::value;
   public:
    static constexpr int local_dim = Triangulation_::local_dim;
    static constexpr int embed_dim = Triangulation_::embed_dim;
    static constexpr int Order = Order_;
    using LayerMap  = typename fdapde::strip_tuple_into<LayerMap_, layers_t>::type;
    using index_t = int;
    using size_t  = std::size_t;
    using Triangulation = Triangulation_;

    // constructors
    GeoFrame() noexcept : triangulation_(nullptr), layers_(), n_layers_(0) { }
    explicit GeoFrame(Triangulation_& triangulation) noexcept :
        triangulation_(std::addressof(triangulation)), layers_(), n_layers_(0) { }

    // modifiers
    // multipoint layer with geometrical locations at mesh nodes
    void push(const std::string& name, layer_t::point_t) {
        geoframe_assert(!name.empty() && !has_layer(name), "empty or duplicated name.");
        using layer_t = internals::point_layer<This>;
        fetch_<layer_t>(layers_).insert({name, layer_t(name, this)});
        idx_to_layer_name_[n_layers_] = name;
        n_layers_++;
        return;
    }
    template <typename ColnamesContainer>
    void push(const ColnamesContainer& layers, layer_t::point_t) {
        for (const auto& name : layers) { push(name, layer_t::point); }
    }
    // multipoint layer with specified locations
    template <typename T>
        requires(std::is_convertible_v<T, std::string>)
    void push(const std::initializer_list<T>& layers, layer_t::point_t) {
        for (auto it = layers.begin(); it != layers.end(); ++it) { push(*it, layer_t::point); }
    }
    template <typename CoordsType>
        requires(fdapde::is_eigen_dense_v<CoordsType> || std::contiguous_iterator<typename CoordsType::iterator>)
    void push(const std::string& name, layer_t::point_t, const CoordsType& coords) {
        geoframe_assert(!name.empty() && !has_layer(name), "empty or duplicated name.");
	using layer_t = internals::point_layer<This>;
        if constexpr (fdapde::is_eigen_dense_v<CoordsType>) {
            geoframe_assert(
              coords.cols() == embed_dim && coords.rows() > 0, "empty or wrongly sized coordinate matrix.");
            using Scalar__ = typename CoordsType::Scalar;
            std::shared_ptr<DMatrix<Scalar__>> coords_ptr = std::make_shared<DMatrix<Scalar__>>(coords);
            fetch_<layer_t>(layers_).insert({name, layer_t(name, this, coords_ptr)});
        } else {
            geoframe_assert(
              coords.size() > 0 && coords.size() % embed_dim == 0, "empty or wrongly sized coordinate matrix.");
            using Scalar__ = typename CoordsType::value_type;
            int n_rows = coords.size() / embed_dim;
            int n_cols = embed_dim;
            std::shared_ptr<DMatrix<Scalar__>> coords_ptr =
              std::make_shared<DMatrix<Scalar__>>(Eigen::Map<const DMatrix<Scalar__>>(coords.data(), n_rows, n_cols));
	    fetch_<layer_t>(layers_).insert({name, layer_t(name, this, coords_ptr)});
        }
        idx_to_layer_name_[n_layers_] = name;
        n_layers_++;
        return;
    }
    // packed multipoint layers sharing the same locations
    template <typename T, typename CoordsType>
        requires(
          (fdapde::is_eigen_dense_v<CoordsType> || std::contiguous_iterator<typename CoordsType::iterator>) &&
          std::is_convertible_v<T, std::string>)
    void push(const std::initializer_list<T>& layers, layer_t::point_t, const CoordsType& coords) {
        using layer_t = internals::point_layer<This>;
        // layers share the same locations, allocate here once
        using Scalar__ = decltype([]() {
            if constexpr (fdapde::is_eigen_dense_v<CoordsType>) return typename CoordsType::Scalar();
            else return typename CoordsType::value_type();
        }());
        std::shared_ptr<DMatrix<Scalar__>> coords_ptr;
        if constexpr (fdapde::is_eigen_dense_v<CoordsType>) {
            coords_ptr = std::make_shared<DMatrix<Scalar__>>(coords);
        } else {
            int n_rows = coords.size() / embed_dim;
            int n_cols = embed_dim;
            coords_ptr =
              std::make_shared<DMatrix<Scalar__>>(Eigen::Map<const DMatrix<Scalar__>>(coords.data(), n_rows, n_cols));
        }
        for (auto it = layers.begin(); it != layers.end(); ++it) {
            std::string name(*it);
            geoframe_assert(!name.empty() && !has_layer(name), "empty or duplicated name.");
            fetch_<layer_t>(layers_).insert({name, layer_t(name, this, coords_ptr)});
            idx_to_layer_name_[n_layers_] = name;
            n_layers_++;
        }
    }

    // areal layer
    void push(
      const std::string& name, layer_t::areal_t, const std::vector<MultiPolygon<local_dim, embed_dim>>& regions) {
        geoframe_assert(!name.empty() && !has_layer(name), "empty or duplicated name.");
        using layer_t = internals::areal_layer<This>;
        using areal_t = std::vector<MultiPolygon<local_dim, embed_dim>>;
        std::shared_ptr<areal_t> regions_ptr = std::make_shared<areal_t>(regions);
        fetch_<layer_t>(layers_).insert({name, layer_t(name, this, regions_ptr)});
        idx_to_layer_name_[n_layers_] = name;
	n_layers_++;
        return;
    }
    // packed areal layers sharing the same regions
    template <typename T>
        requires(std::is_convertible_v<T, std::string>)
    void push(
      const std::initializer_list<T>& layers, layer_t::areal_t,
      const std::vector<MultiPolygon<local_dim, embed_dim>>& regions) {
        using layer_t = internals::areal_layer<This>;
	using areal_t = std::vector<MultiPolygon<local_dim, embed_dim>>;
        // allocate shared memory
        std::shared_ptr<areal_t> regions_ptr = std::make_shared<areal_t>(regions);
        for (auto it = layers.begin(); it != layers.end(); ++it) {
            std::string name(*it);
            geoframe_assert(!it->empty() && !has_layer(name), "empty or duplicated name.");
            fetch_<layer_t>(layers_).insert({name, layer_t(name, this, regions_ptr)});
            idx_to_layer_name_[n_layers_] = name;
	    n_layers_++;
        }
        return;
    }
    // construct layer from a subset of another layer
    template <typename T>
        requires(std::is_convertible_v<T, std::string> || std::is_convertible_v<T, index_t>)
    void push(
      const std::string& name, layer_t::areal_t,
      const internals::plain_row_filter<internals::areal_layer<This>>& filter, const std::vector<T>& cols) {
        std::vector<MultiPolygon<local_dim, embed_dim>> regions;
        regions.reserve(filter.rows());
	for (int i = 0, n = filter.rows(); i < n; ++i) { regions.emplace_back(filter(i).geometry()); }
        push(name, layer_t::areal, regions);
        if constexpr (std::is_same_v<T, std::string>) {
            get_as(layer_t::areal, name).data() = typename internals::areal_layer<This>::storage_t(filter, cols);
        }
        if constexpr (std::is_convertible_v<T, index_t>) {
            auto field_descriptors = filter.field_descriptors();
            std::vector<std::string> cols_;
            for (index_t i : cols) { cols_.push_back(field_descriptors[i].colname); }
            get_as(layer_t::areal, name).data() = typename internals::areal_layer<This>::storage_t(filter, cols_);
        }
    }
    template <typename T>
        requires(std::is_convertible_v<T, std::string> || std::is_convertible_v<T, index_t>)
    void push(
      const std::string& name, layer_t::areal_t,
      const internals::plain_row_filter<internals::areal_layer<This>>& filter, const std::initializer_list<T>& cols) {
        std::vector<std::string> cols_;
        if constexpr (std::is_convertible_v<T, std::string>) { cols_.insert(cols_.begin(), cols.begin(), cols.end()); }
        if constexpr (std::is_convertible_v<T, index_t>) {
            auto field_descriptors = filter.field_descriptors();
            for (index_t i : cols) { cols_.push_back(field_descriptors[i].colname); }
        }
        push(name, layer_t::areal_t {}, filter, cols_);
    }
    void push(
      const std::string& name, layer_t::areal_t,
      const internals::plain_row_filter<internals::areal_layer<This>>& filter) {
        push(name, layer_t::areal_t {}, filter, filter.colnames());
    }

    // directly push an externally created layer
    template <typename Layer>
        requires(is_supported_layer_v<std::decay_t<Layer>>)
    void push(const std::string& name, Layer&& layer) {
        using layer_t = std::decay_t<Layer>;
        fetch_<layer_t>(layers_).insert({name, layer});
        idx_to_layer_name_[n_layers_] = name;
	n_layers_++;
        return;
    }
    void erase(const std::string& layer_name) {
        // search for layer_name (if no layer found does nothing)
        if (!has_layer(layer_name)) return;
        std::apply(
          [&](auto&&... layer) {
              (std::erase_if(
                 layer,
                 [&](const auto& item) {
                     auto const& [k, v] = item;
                     return k == layer_name;
                 }),
               ...);
          },
          layers_);
	// update idx - layer_name mapping
	int i = 0;
        for (; i < n_layers_; ++i) {
            if (idx_to_layer_name_.at(i) == layer_name) {
                idx_to_layer_name_.erase(i);
                break;
            }
        }
        for (; i < n_layers_ - 1; ++i) { idx_to_layer_name_[i] = idx_to_layer_name_.at(i + 1); }
	idx_to_layer_name_.erase(n_layers_ - 1);
	n_layers_--;
        return;
    }
  
    // file import
    // TODO: supply some of the columns as coordinates
    template <typename Scalar>
    void load_csv(const std::string& name, layer_t::point_t, const std::string& file_name) {
        geoframe_assert(std::filesystem::exists(file_name), "file " + file_name + " not found.");
        auto csv = fdapde::read_csv<Scalar>(file_name);
        geoframe_assert(csv.rows() == triangulation_->n_nodes(), "wrong csv size.");
        push(name, layer_t::point, csv.cols());
        // move data in memory buffer
        get_as(layer_t::point, name).data().assign_inplace_from(csv.data());
	get_as(layer_t::point, name).set_colnames(csv.colnames());
	idx_to_layer_name_[n_layers_] = name;
	n_layers_++;
        return;
    }
    template <typename Scalar, typename CoordsType>
        requires(fdapde::is_eigen_dense_v<CoordsType> || std::contiguous_iterator<typename CoordsType::iterator>)
    void load_csv(const std::string& name, layer_t::point_t, const std::string& file_name, const CoordsType& coords) {
        geoframe_assert(std::filesystem::exists(file_name), "file " + file_name + " not found.");
        auto csv = fdapde::read_csv<Scalar>(file_name);
        if constexpr (fdapde::is_eigen_dense_v<CoordsType>) {
            geoframe_assert(csv.rows() == coords.rows(), "wrong csv size.");
        } else {
            geoframe_assert(csv.rows() == coords.size() / embed_dim, "wrong csv size.");
        }
        push(name, layer_t::point, coords, csv.cols());
        // move data in memory buffer
        get_as(layer_t::point, name).data().assign_inplace_from(csv.data());
	get_as(layer_t::point, name).set_colnames(csv.colnames());
	idx_to_layer_name_[n_layers_] = name;
	n_layers_++;
        return;
    }
  
    void load_shp(const std::string& name, const std::string& file_name) {
        std::string file_name_ = std::filesystem::current_path().string() + "/" + file_name;
        geoframe_assert(std::filesystem::exists(file_name_), "file " + file_name_ + " not found.");
        ShapeFile shp(file_name_);
        // dispatch to processing logic
        switch (shp.shape_type()) {
        case shp_reader::Polygon: {
            // load polygon as areal layer
            std::vector<MultiPolygon<local_dim, embed_dim>> regions;
            regions.reserve(shp.n_records());
            for (int i = 0, n = shp.n_records(); i < n; ++i) { regions.emplace_back(shp.polygon(i).nodes()); }
            push(name, layer_t::areal, regions);
	    break;
        }
        }
        std::vector<std::pair<std::string, std::vector<int        >>> int_data;
        std::vector<std::pair<std::string, std::vector<double     >>> dbl_data;
        std::vector<std::pair<std::string, std::vector<std::string>>> str_data;
        // TODO: std::vector<std::pair<std::string, std::vector<bool       >>> bin_data;
        for (const auto& [name, field_type] : shp.field_descriptors()) {
            if (field_type == 'N') { dbl_data.emplace_back(name, shp.get<double>(name)); }
            if (field_type == 'C') { str_data.emplace_back(name, shp.get<std::string>(name)); }
        }
        get_as(layer_t::areal, name).set_data(int_data, dbl_data, str_data/*, bin_data*/);
	idx_to_layer_name_[n_layers_] = name;
	n_layers_++;
	
        // TODO: we should reorder the fields in the same order they come from the shp
    }
    // layer access
    template <typename Tag> auto& get_as(Tag, const std::string& name) {
        return get_as_<layer_type_from_tag<Tag>>(name);
    }
    template <typename Tag> auto& get_as(Tag t, int idx) { return get_as(t, idx_to_layer_name_.at(idx)); }
    template <typename Tag> const auto& get_as(Tag, const std::string& name) const {
        return get_as_<layer_type_from_tag<Tag>>(name);
    }
    template <typename Tag> const auto& get_as(Tag t, int idx) const { return get_as(t, idx_to_layer_name_.at(idx)); }
    // observers
    bool has_layer(const std::string& name) const {
        // search for layer_name in each layer type
        bool found_ = false;
        std::apply([&](auto&&... layer) { ([&]() { found_ |= layer.contains(name); }(), ...); }, layers_);
        return found_;
    }
    bool contains(const std::string& column) const {   // true if column is in geoframe
        bool found_ = false;
        std::apply(
          [&](auto&&... layer) {
              (
                [&]() {
		  for (const auto& [name, data] : layer) { found_ |= data.contains(column); }
                }(),
                ...);
          },
          layers_);
        return found_;
    }
    std::optional<internals::ltype> layer_category(const std::string& name) const {
        geoframe_assert(has_layer(name), std::string("key " + name + " not found."));
        if (fetch_<point_layer_t>(layers_).contains(name)) return internals::ltype::point;
        if (fetch_<areal_layer_t>(layers_).contains(name)) return internals::ltype::areal;
	return std::nullopt;
    }
    std::optional<internals::ltype> layer_category(int idx) const { return layer_category(idx_to_layer_name_.at(idx)); }
    int n_layers() const { return n_layers_; }
    // indexed access
    const internals::plain_data_layer<Order>& operator[](int idx) const {
        geoframe_assert(idx < n_layers_, "out of bound access.");
        auto get_plain_data_ =
          [&, this]<typename LayerType>([[maybe_unused]] LayerType l, const std::string& layer_name) -> const void* {
            if (fetch_<LayerType>(layers_).contains(layer_name)) {
                return std::addressof(fetch_<LayerType>(layers_).at(layer_name).data());
            }
            return nullptr;
        };
        std::string name = idx_to_layer_name_.at(idx);
        const void* layer_ptr;
	// as idx < n_layers_, it is guaranteed that one of the branches will be taken
        if (fetch_<point_layer_t>(layers_).contains(name)) layer_ptr = get_plain_data_(point_layer_t {}, name);
        if (fetch_<areal_layer_t>(layers_).contains(name)) layer_ptr = get_plain_data_(areal_layer_t {}, name);
        return *reinterpret_cast<const internals::plain_data_layer<Order>*>(layer_ptr);
    }
    // iterator
    class iterator {
        const GeoFrame* gf_;
        int index_;
       public:
        using value_type = internals::plain_data_layer<Order>;
        using pointer = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator(const GeoFrame* gf, int index) : gf_(gf), index_(index) { }
        const value_type* operator->() const { return std::addressof(gf_->operator[](index_)); }
        const value_type& operator*() const { return gf_->operator[](index_); }
        iterator& operator++() {
            index_++;
            return *this;
        }
        internals::ltype category() const { return *(gf_->layer_category(index_)); }
        const reference data() const { return operator*(); }
        template <typename Tag> const auto& as(Tag t) const {
            return gf_->get_as(t, gf_->idx_to_layer_name_.at(index_));
        }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
        friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
    };
    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, n_layers_); }
    // geometry access
    Triangulation_& triangulation() { return *triangulation_; }
    const Triangulation_& triangulation() const { return *triangulation_; }
    int n_cells() const { return triangulation_->n_cells(); }
    int n_nodes() const { return triangulation_->n_nodes(); }
    const DMatrix<double>& nodes() const { return triangulation_->nodes(); }
    const DMatrix<int, Eigen::RowMajor>& cells() const { return triangulation_->cells(); }
   private:
    // internal utilities
    template <typename LayerType> decltype(auto) get_as_(const std::string& name) {
        geoframe_assert(fetch_<LayerType>(layers_).contains(name), std::string("key " + name + " not found."));
        return fetch_<LayerType>(layers_).at(name);
    }
    template <typename Tag> class layer_type_from_tag_impl {
        static auto layer_type_from_tag_(Tag t) {
            if constexpr (std::is_same_v<Tag, layer_t::point_t>) return internals::point_layer<This> {};
            if constexpr (std::is_same_v<Tag, layer_t::areal_t>) return internals::areal_layer<This> {};
        }
       public:
        using type = decltype(layer_type_from_tag_(std::declval<Tag>()));
    };
    template <typename Tag> using layer_type_from_tag = layer_type_from_tag_impl<Tag>::type;

    // data members
    Triangulation_* triangulation_ = nullptr;
    LayerMap layers_ {};
    int n_layers_ = 0;
    std::unordered_map<int, std::string> idx_to_layer_name_;
};

// // regions is a vector with the same number of elements as number of cells
// template <typename LayerType_, typename SubregionsType_, typename F_>
//     requires(requires(SubregionsType_ s, int i) {
//                 { s.operator[](i) } -> std::convertible_to<int>;
//             }) && (std::is_same_v<typename LayerType_::layer_category, layer_t::point_t>)
// auto aggregate(const LayerType_& layer, const SubregionsType_& regions, F_&& f) {
//     geoframe_assert(
//       regions.size() == layer.triangulation().n_cells(), "Number of rows does not match number of cells.");
//     if constexpr (fdapde::is_eigen_dense_v<SubregionsType_>) {
//         geoframe_assert(regions.rows() > 0 && regions.cols() == 1, "Not a vector.");
//     }
//     using layer_t = internals::areal_layer<typename LayerType_::GeoFrame>;
//     using geoframe_t = typename LayerType_::GeoFrame;
//     layer_t res {};
//     // compute number of regions
//     std::decay_t<SubregionsType_> regions_ = regions;
//     std::sort(regions_.begin(), regions_.end());
//     int n_regions = 1;
//     for (int i = 1, n = regions_.size(); i < n; ++i) {
//         if (regions_[i] != regions_[i - 1]) { n_regions++; }
//     }
//     res.resize(n_regions, layer.cols());
//     res.set_colnames(layer.colnames());
//     DVector<int> coords_to_cell = layer.triangulation().locate(layer.coordinates());
//     // aggregate data
//     std::vector<std::vector<int>> bucket_list;
//     bucket_list.resize(n_regions);
//     for (std::size_t i = 0, n = layer.rows(); i < n; ++i) { bucket_list[regions[coords_to_cell[i]]].push_back(i); }
//     // apply functor to aggregated data
//     for (int i = 0; i < n_regions; ++i) { res.row(i) = f(layer(bucket_list[i])); }
//     return res;
// }

  
}   // namespace fdapde

#endif // __GEOFRAME_H__
