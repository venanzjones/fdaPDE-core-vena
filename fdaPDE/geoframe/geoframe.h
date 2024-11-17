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

namespace fdapde {

// generates a sequence of strings as {"base_0", "base_1", \ldots, "base_{n-1}"}
std::vector<std::string> seq(const std::string& base, int n) {
    std::vector<std::string> vec;
    vec.reserve(n);
    for(int i = 0; i < n; ++i) { vec.emplace_back(base + std::to_string(i)); }
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

template <typename T>
concept is_contiguous_memory = requires(T t) {
    typename T::value_type;
    { t.size() } -> std::same_as<std::size_t>;
    { t.data() } -> std::same_as<typename T::value_type*>;
} && std::contiguous_iterator<typename T::iterator>;

inline void throw_geoframe_error(const std::string& msg) { throw std::runtime_error("GeoFrame: " + msg); }

#define geoframe_assert(condition, msg)                                                                                \
    if (!(condition)) { internals::throw_geoframe_error(msg); }

// geometric specific layers
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

    template <typename... DataT> void set_data(DataT&&... data) {
        data_ = storage_t(std::forward<DataT>(data)...);
    }
    storage_t& data() { return data_; }
    const storage_t& data() const { return data_; }
  
    const DMatrix<double>& coordinates() const { return coords_ ? *coords_ : triangulation().nodes(); }
    Triangulation<local_dim, embed_dim>& triangulation() { return geoframe_->triangulation(); }
    const Triangulation<local_dim, embed_dim>& triangulation() const { return geoframe_->triangulation(); }
    DVector<int> locate() const { return triangulation().locate(*coords_); }
   private:
    GeoFrame_* geoframe_;
    std::shared_ptr<DMatrix<double>> coords_;
    storage_t data_;
};

template <typename GeoFrame_> struct areal_layer {
    using GeoFrame = typename std::decay_t<GeoFrame_>;
    using storage_t = plain_data_layer<GeoFrame::Order>;
    using index_t = typename storage_t::index_t;
    using size_t = typename storage_t::size_t;
    using layer_category = layer_t::areal_t;
    template <typename T> using reference = typename storage_t::template reference<T>;
    template <typename T> using const_reference = typename storage_t::template const_reference<T>;
    static constexpr int Order = GeoFrame::Order;
    static constexpr int local_dim = GeoFrame::local_dim;
    static constexpr int embed_dim = GeoFrame::embed_dim;

    areal_layer() : geoframe_(nullptr), regions_() { }
    template <typename SubregionsType>
        requires(std::is_same_v<SubregionsType, std::vector<MultiPolygon<local_dim, embed_dim>>>)
    areal_layer(
      const std::string& layer_name, GeoFrame_* geoframe, const std::shared_ptr<SubregionsType>& regions) noexcept :
        geoframe_(geoframe), regions_(regions) { }
    // observers
    size_t n_regions() const { return regions_->size(); }
    size_t rows() const { return n_regions(); }
    size_t cols() const { return data_.cols(); }
    size_t size() const { return rows() * cols(); }
    const auto& field_descriptor(const std::string& colname) const { return data_.field_descriptor(colname); }
    const auto& field_descriptors() const { return data_.field_descriptors(); }

    const MultiPolygon<local_dim, embed_dim>& geometry(int i) const { return regions_->operator[](i); }
    // computes measures of subdomains
    std::vector<double> measure() const {
        std::vector<double> m_(regions_.size(), 0);
	for(int i = 0; i < regions_.size(); ++i) { m_[i] = regions_->operator[](i).measure(); }
        return m_;
    }
    // computes matrix [M]_{ij} : [M]_{ij} == 1 \iff cell j is inside region i, 0 otherwise
    BinaryMatrix<Dynamic, Dynamic> incidence_matrix() const {
        BinaryMatrix<Dynamic, Dynamic> m(regions_.size(), triangulation().n_cells());
        // for each cell, check in which region its barycenter lies
        for (auto it = triangulation().cells_begin(); it != triangulation().cells_end(); ++it) {
            Eigen::Matrix<double, embed_dim, 1> barycenter = it->barycenter();
            for (int i = 0, n = regions_.size(); i < n; ++i) {
                if (regions_[i].contains(barycenter)) { m(i, it->id()).set(); }
            }
        }
        return m;
    }
    template <typename... DataT> void set_data(DataT&&... data) {
        data_ = storage_t(std::forward<DataT>(data)...);
    }
    storage_t& data() { return data_; }
    const storage_t& data() const { return data_; }
  
    struct areal_row_view : public plain_row_view<storage_t> {
        using Base = plain_row_view<storage_t>;

        areal_row_view() noexcept = default;
        areal_row_view(areal_layer<GeoFrame>* data, int row) :
            Base(std::addressof(data->data()), row), data_(data), row_(row) { }
        // observers
        const MultiPolygon<local_dim, embed_dim>& geometry() const { return data_->geometry(row_); }
       private:
        areal_layer<GeoFrame>* data_;
        index_t row_;
    };
    using row_view = areal_row_view;
    using const_row_view = areal_row_view;

    // row access
    row_view row(size_t row) {
        geoframe_assert(row < n_regions(), "out of bound access.");
        return areal_row_view(this, row);
    }
    const_row_view row(size_t row) const {
        geoframe_assert(row < n_regions(), "out of bound access.");
        return areal_row_view(this, row);
    }
    // row filtering operations
    template <typename Iterator>
        requires(internals::is_integer_v<typename Iterator::value_type>)
    plain_row_filter<areal_layer<GeoFrame>> operator()(Iterator begin, Iterator end) {
        return plain_row_filter<areal_layer<GeoFrame>>(this, begin, end);
    }
    template <typename T>
        requires(std::is_convertible_v<T, index_t>)
    plain_row_filter<areal_layer<GeoFrame>> operator()(const std::initializer_list<T>& idxs) {
        return plain_row_filter<areal_layer<GeoFrame>>(this, idxs.begin(), idxs.end());
    }
    template <typename Filter>
        requires(requires(Filter f, index_t i) {
            { f(i) } -> std::convertible_to<bool>;
        })
    plain_row_filter<areal_layer<GeoFrame>> operator()(Filter&& f) {
        return plain_row_filter<areal_layer<GeoFrame>>(this, std::forward<Filter>(f));
    }
    template <typename LogicalVec>
        requires(requires(LogicalVec vec, index_t i) {
            { vec[i] } -> std::convertible_to<bool>;
        })
    plain_row_filter<areal_layer<GeoFrame>> operator()(LogicalVec&& logical_vec) {
        return plain_row_filter<areal_layer<GeoFrame>>(this, logical_vec);
    }
   private:
    // internal utils
    typename GeoFrame::Triangulation& triangulation() { return geoframe_->triangulation(); }
    const Triangulation<local_dim, embed_dim>& triangulation() const { return geoframe_->triangulation(); }
    // data members
    GeoFrame_* geoframe_;
    std::shared_ptr<std::vector<MultiPolygon<local_dim, embed_dim>>> regions_;
    storage_t data_;
};

}   // namespace internals

// regions is a vector with the same number of elements as number of cells
template <typename LayerType_, typename SubregionsType_, typename F_>
    requires(requires(SubregionsType_ s, int i) {
                { s.operator[](i) } -> std::convertible_to<int>;
            }) && (std::is_same_v<typename LayerType_::layer_category, layer_t::point_t>)
auto aggregate(const LayerType_& layer, const SubregionsType_& regions, F_&& f) {
    geoframe_assert(
      regions.size() == layer.triangulation().n_cells(), "Number of rows does not match number of cells.");
    if constexpr (fdapde::is_eigen_dense_v<SubregionsType_>) {
        geoframe_assert(regions.rows() > 0 && regions.cols() == 1, "Not a vector.");
    }
    using layer_t = internals::areal_layer<typename LayerType_::GeoFrame>;
    using geoframe_t = typename LayerType_::GeoFrame;
    layer_t res {};
    // compute number of regions
    std::decay_t<SubregionsType_> regions_ = regions;
    std::sort(regions_.begin(), regions_.end());
    int n_regions = 1;
    for (int i = 1, n = regions_.size(); i < n; ++i) {
        if (regions_[i] != regions_[i - 1]) { n_regions++; }
    }
    res.resize(n_regions, layer.cols());
    res.set_colnames(layer.colnames());
    DVector<int> coords_to_cell = layer.triangulation().locate(layer.coordinates());
    // aggregate data
    std::vector<std::vector<int>> bucket_list;
    bucket_list.resize(n_regions);
    for (std::size_t i = 0, n = layer.rows(); i < n; ++i) { bucket_list[regions[coords_to_cell[i]]].push_back(i); }
    // apply functor to aggregated data
    for (int i = 0; i < n_regions; ++i) { res.row(i) = f(layer(bucket_list[i])); }
    return res;
}

template <typename Triangulation_, int Order_ = 2> struct GeoFrame {
    fdapde_static_assert(Order_ > 1, GEOFRAME_MUST_HAVE_ORDER_TWO_OR_HIGHER);
   private:
    using This = GeoFrame<Triangulation_, Order_>;
    using layers_t = std::tuple<internals::point_layer<This>, internals::areal_layer<This>>;
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
    GeoFrame() noexcept : triangulation_(nullptr), layers_() { }
    explicit GeoFrame(Triangulation_& triangulation) noexcept :
        triangulation_(std::addressof(triangulation)), layers_() { }
  
    // modifiers
    // multipoint layer with geometrical locations at mesh nodes
    void push(const std::string& name, layer_t::point_t, size_t n) {
        geoframe_assert(!name.empty() && !has_layer(name), "empty or duplicated name.");
        using layer_t = internals::point_layer<This>;
        fetch_<layer_t>(layers_).insert({name, layer_t(name, this, triangulation_->n_nodes(), n)});
        return;
    }
    // multipoint layer with specified locations
    template <typename ColnamesContainer>
    void push(const ColnamesContainer& colnames, layer_t::point_t, size_t n) {
        for (const auto& name : colnames) { push(name, layer_t::point, n); }
    }
    template <typename T>
        requires(std::is_convertible_v<T, std::string>)
    void push(const std::initializer_list<T>& colnames, layer_t::point_t, size_t n) {
        for (auto it = colnames.begin(); it != colnames.end(); ++it) { push(*it, layer_t::point, n); }
    }
    template <typename CoordsType>
        requires(
          (fdapde::is_eigen_dense_v<CoordsType> && (CoordsType::Cols == Dynamic || CoordsType::Cols == embed_dim) &&
           std::is_floating_point_v<typename CoordsType::Scalar>) ||
          (internals::is_contiguous_memory<CoordsType> && std::is_floating_point_v<typename CoordsType::value_type>))
    void push(const std::string& name, layer_t::point_t, const CoordsType& coords, size_t n) {
        geoframe_assert(!name.empty() && !has_layer(name), "empty or duplicated name.");
	using layer_t = internals::point_layer<This>;
        if constexpr (fdapde::is_eigen_dense_v<CoordsType>) {
            geoframe_assert(
              coords.cols() == embed_dim && coords.rows() > 0, "empty or wrongly sized coordinate matrix.");
            using Scalar__ = typename CoordsType::Scalar;
            std::shared_ptr<DMatrix<Scalar__>> coords_ptr = std::make_shared<DMatrix<Scalar__>>(coords);
            fetch_<layer_t>(layers_).insert({name, layer_t(name, this, coords_ptr, coords.rows(), n)});
        } else {
            geoframe_assert(
              coords.size() > 0 && coords.size() % embed_dim == 0, "empty or wrongly sized coordinate matrix.");
            using Scalar__ = typename CoordsType::value_type;
            int n_rows = coords.size() / embed_dim;
            int n_cols = embed_dim;
            std::shared_ptr<DMatrix<Scalar__>> coords_ptr =
              std::make_shared<DMatrix<Scalar__>>(Eigen::Map<DMatrix<Scalar__>>(coords.data(), n_rows, n_cols));
	    fetch_<layer_t>(layers_).insert({name, layer_t(name, this, coords_ptr, n_rows, n)});
        }
        return;
    }
    // packed multipoint layers sharing the same locations
    template <typename T, typename CoordsType>
        requires(
          (fdapde::is_eigen_dense_v<CoordsType> || internals::is_contiguous_memory<CoordsType>) &&
          std::is_convertible_v<T, std::string>)
    void push(const std::initializer_list<T>& colnames, layer_t::point_t, const CoordsType& coords, size_t n) {
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
              std::make_shared<DMatrix<Scalar__>>(Eigen::Map<DMatrix<Scalar__>>(coords.data(), n_rows, n_cols));
        }
        for (auto it = colnames.begin(); it != colnames.end(); ++it) {
            std::string name(*it);
            geoframe_assert(!name.empty() && !has_layer(name), "empty or duplicated name.");
            fetch_<layer_t>(layers_).insert({name, layer_t(name, this, coords_ptr, coords_ptr->rows(), n)});
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
            geoframe_assert(!it->empty() && !has_layer(*it), "empty or duplicated name.");
            fetch_<layer_t>(layers_).insert({*it, layer_t(*it, this, regions_ptr)});
        }
        return;
    }
    void push(
      const std::string& name, layer_t::areal_t,
      const internals::plain_row_filter<internals::areal_layer<This>>& filter) {
        std::vector<MultiPolygon<local_dim, embed_dim>> regions;
        regions.reserve(filter.rows());
        for (int i = 0, n = filter.rows(); i < n; ++i) { regions.emplace_back(filter(i).geometry()); }
        push(name, layer_t::areal, regions);
	get_as(layer_t::areal, name).data() = typename internals::areal_layer<This>::storage_t(filter);
    }

    // directly push an externally created layer
    template <typename Layer>
        requires(is_supported_layer_v<std::decay_t<Layer>>)
    void push(const std::string& name, Layer&& layer) {
        using layer_t = std::decay_t<Layer>;
        fetch_<layer_t>(layers_).insert({name, layer});
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
        return;
    }

  // count number of points in a region
  
    // file import
   // ---- supply some of the columns as coordinates
    template <typename Scalar>
    void load_csv(const std::string& name, layer_t::point_t, const std::string& file_name) {
        geoframe_assert(std::filesystem::exists(file_name), "file " + file_name + " not found.");
        auto csv = fdapde::read_csv<Scalar>(file_name);
        geoframe_assert(csv.rows() == triangulation_->n_nodes(), "wrong csv size.");
        push(name, layer_t::point, csv.cols());
        // move data in memory buffer
        get_as(layer_t::point, name).data().assign_inplace_from(csv.data());
	get_as(layer_t::point, name).set_colnames(csv.colnames());
        return;
    }
    template <typename Scalar, typename CoordsType>
        requires(fdapde::is_eigen_dense_v<CoordsType> || internals::is_contiguous_memory<CoordsType>)
    void
    load_csv(const std::string& name, layer_t::point_t, const std::string& file_name, const CoordsType& coords) {
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
	    std::cout << shp.n_records() << std::endl;
            regions.reserve(shp.n_records());
            for (int i = 0, n = shp.n_records(); i < n; ++i) { regions.emplace_back(shp.polygon(i).nodes()); }
            push(name, layer_t::areal, regions);
	    break;
        }
        }
        std::vector<std::pair<std::string, std::vector<int        >>> int_data;
        std::vector<std::pair<std::string, std::vector<double     >>> dbl_data;
        std::vector<std::pair<std::string, std::vector<std::string>>> str_data;
        //std::vector<std::pair<std::string, std::vector<bool       >>> bin_data;
        for (const auto& [name, field_type] : shp.field_descriptors()) {
            if (field_type == 'N') { dbl_data.emplace_back(name, shp.get<double>(name)); }
            if (field_type == 'C') { str_data.emplace_back(name, shp.get<std::string>(name)); }
        }
        get_as(layer_t::areal, name).set_data(int_data, dbl_data, str_data/*, bin_data*/);

        // we should reorder the fields in the same order they come from the shp
    }
    // iterators
    auto begin(layer_t::point_t) { return fetch_<layer_type_from_tag<layer_t::point_t>>(layers_).begin(); }
    auto end(layer_t::point_t) { return fetch_<layer_type_from_tag<layer_t::point_t>>(layers_).end(); }
    // layer access
    template <typename Tag> auto& get_as(Tag, const std::string& name) {
        return get_as_<layer_type_from_tag<Tag>>(name);
    }
    template <typename Tag> const auto& get_as(Tag, const std::string& name) const {
        return get_as_<layer_type_from_tag<Tag>>(name);
    }
    bool has_layer(const std::string& name) const {
        // search for layer_name in each layer type
        bool found_ = false;
        std::apply([&](auto&&... layer) { ([&]() { found_ |= layer.contains(name); }(), ...); }, layers_);
        return found_;
    }
    std::optional<internals::ltype> layer_type(const std::string& name) const {
        geoframe_assert(has_layer(name), std::string("key " + name + " not found."));
        if (fetch_<layer_type_from_tag<layer_t::point_t>>(layers_).contains(name)) return internals::ltype::point;
        if (fetch_<layer_type_from_tag<layer_t::areal_t>>(layers_).contains(name)) return internals::ltype::areal;
	return std::nullopt;
    }
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
};

}   // namespace fdapde

#endif // __GEOFRAME_H__
