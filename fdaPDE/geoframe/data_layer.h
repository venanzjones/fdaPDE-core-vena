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

#ifndef __PLAIN_DATA_LAYER_H__
#define __PLAIN_DATA_LAYER_H__

#include <utility>

#include "../utils/traits.h"
#include "parsing.h"

namespace fdapde {
namespace internals {
    enum dtype { flt64 = 0, flt32 = 1, int64 = 2, int32 = 3, bin = 4, str = 5 };    // runtime data type identification
}   // namespace internals

namespace data_t {

struct flt64_ : std::type_identity<double      > { internals::dtype type_id = internals::dtype::flt64; } flt64;
struct flt32_ : std::type_identity<float       > { internals::dtype type_id = internals::dtype::flt32; } flt32;
struct int64_ : std::type_identity<std::int64_t> { internals::dtype type_id = internals::dtype::int64; } int64;
struct int32_ : std::type_identity<std::int32_t> { internals::dtype type_id = internals::dtype::int32; } int32;
struct bin_   : std::type_identity<bool        > { internals::dtype type_id = internals::dtype::bin;   } bin;
struct str_   : std::type_identity<std::string > { internals::dtype type_id = internals::dtype::str;   } str;
  
}   // namespace data_t

namespace internals {

template <typename FieldDescriptor_, typename F_, typename... Args>
void dispatch_to_dtype(const FieldDescriptor_& t, F_&& f, Args&&... args) {
    using dtypes =
      std::tuple<data_t::flt64_, data_t::flt32_, data_t::int64_, data_t::int32_, data_t::bin_, data_t::str_>;
    std::apply(
      [&](const auto&... ts) {
          (
            [&]() {
                if (t.type_id == ts.type_id) {
                    f.template operator()<typename std::decay_t<decltype(ts)>::type>(std::forward<Args>(args)...);
                }
            }(),
            ...);
      },
      dtypes {});
}

template <typename DataLayer> struct plain_row_view {
    using index_t = typename DataLayer::index_t;
    using size_t = typename DataLayer::size_t;
    using storage_t = std::conditional_t<
      std::is_const_v<DataLayer>, std::add_const_t<typename DataLayer::storage_t>, typename DataLayer::storage_t>;
    static constexpr int Order = DataLayer::Order;
    template <typename T> using reference = typename DataLayer::reference<T>;
    template <typename T> using const_reference = typename DataLayer::const_reference<T>;
  
    plain_row_view() noexcept = default;
    plain_row_view(DataLayer* data, index_t row) noexcept : data_(data), row_(row) { }
    // observers
    size_t rows() const { return 1; }
    size_t cols() const { return data_->cols(); }
    size_t size() const { return data_->cols(); }
    index_t id() const { return row_; }
    const auto& field_descriptors() const { return data_->field_descriptors(); }
    // accessors
    template <typename T> reference<T> get(const std::string& colname) {
        return data_->template data<T>()(row_, data_->field_descriptor(colname).col_id);
    }
  
    template <typename T> const_reference<T> get(const std::string& colname) const {
        return data_->template data<T>()(row_, data_->field_descriptor(colname).col_id);
    }
    template <typename T> reference<T> get(size_t col) { return data_->template data<T>()(row_, col); }
    template <typename T> const_reference<T> get(size_t col) const { return data_->template data<T>()(row_, col); }
    // modifiers
    plain_row_view& operator=(const plain_row_view& src) {
        for (const auto& f : field_descriptors()) {
            internals::dispatch_to_dtype(
              f, [&]<typename T>(plain_row_view& dst) mutable { dst.get<T>(f.colname) = src.get<T>(f.colname); },
              *this);
        }
        return *this;
    }
    template <typename... Ts> plain_row_view& operator=(const std::tuple<Ts...>& src) {
        int i = 0;
        std::apply(
          [&](const Ts&... args) {
              (
                [&]() {
                    const auto& f = field_descriptors()[i];
                    if (f.type_id == dtype::flt64) { get<double>(f.colname) = parse_<double>(args); }
                    if (f.type_id == dtype::flt32) { get<float >(f.colname) = parse_<float >(args); }
                    if (f.type_id == dtype::int64) { get<std::int64_t>(f.colname) = parse_<std::int64_t>(args); }
                    if (f.type_id == dtype::int32) { get<std::int32_t>(f.colname) = parse_<std::int32_t>(args); }
                    if (f.type_id == dtype::bin) { get<bool>(f.colname) = parse_<bool>(args); }
                    if (f.type_id == dtype::str) {
                        if constexpr (std::is_convertible_v<Ts, std::string>) {
                            get<std::string>(f.colname) = args;
                        } else {
                            get<std::string>(f.colname) = std::to_string(args);
                        }
                    }
		    i++;
                }(),
                ...);
          },
          src);
        return *this;
    }
   private:
    template <typename T> struct mapped_type {
        using type = std::decay_t<decltype([](T) {
            if constexpr (internals::is_integer_v<T> && !std::is_same_v<T, std::int64_t>) {
                return std::int32_t();   // map integral, not 64 bit and not boolean types, to 32 bit int
            } else {
                return T();   // do not map other types
            }
        }(T()))>;
    };  
    template <typename T, typename U_> T parse_(U_&& u) const {
        using U = std::decay_t<U_>;
        if constexpr (std::is_same_v<U, std::string> || std::is_same_v<U, const char*>) {
            // string to integral conversion
            if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) { return T(std::stod(u)); }
            if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::int32_t>) {
                return T(std::stoi(u));
            }
            if constexpr (std::is_same_v<T, bool>) {
	      if (u == "true"  || u == "TRUE"  || u == "True" ) return true;
	      if (u == "false" || u == "FALSE" || u == "False") return false;
	      throw std::runtime_error("GeoFrame: parsing error.");
            }
        } else {
            if constexpr (std::is_same_v<T, U>) {
                return u;
            } else {
                return T(u);
            }
        }
    }
    template <typename T> using mapped_type_t = typename mapped_type<T>::type;
    DataLayer* data_;
    index_t row_;
};

template <typename Scalar_, typename DataLayer> struct plain_col_view {
    using Scalar = Scalar_;
    using index_t = typename DataLayer::index_t;
    using size_t = typename DataLayer::size_t;
    static constexpr int Order = DataLayer::Order;
    using storage_t = std::conditional_t<
      std::is_const_v<DataLayer>, MdArraySlice<std::add_const_t<MdArray<Scalar, full_dynamic_extent_t<Order>>>, 1>,
      MdArraySlice<MdArray<Scalar, full_dynamic_extent_t<Order>>, 1>>;
    using reference = typename DataLayer::reference<Scalar>;
    using const_reference = typename DataLayer::const_reference<Scalar>;

    plain_col_view() noexcept = default;
    template <typename FieldDescriptor>
    plain_col_view(DataLayer* data, const FieldDescriptor& desc) noexcept :
        slice_(data->template data<Scalar>().template slice<1>(desc.col_id)),
        data_(data),
        col_(desc.col_id),
        type_id_(desc.type_id),
        colname_(desc.colname) { }
    // observers
    size_t rows() const { return data_->rows(); }
    size_t cols() const { return 1; }
    size_t size() const { return data_->rows(); }
    index_t id() const { return col_; }
    const auto& field_descriptor() const { return data_->field_descriptor(colname_); }
    const std::string& colname() const { return colname_; }
    internals::dtype type_id() const { return type_id_; }
    // accessors
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) &&
                (sizeof...(Idxs) == Order - 1 && !std::is_const_v<DataLayer>)
    reference operator()(Idxs&&... idxs) {
        return slice_(static_cast<index_t>(idxs)...);
    }
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order - 1)
    const_reference operator()(Idxs&&... idxs) const {
        return slice_(static_cast<index_t>(idxs)...);
    }
    // logical comparison
    std::vector<bool> operator==(const Scalar& rhs) const { return logical_apply_(rhs, std::equal_to<Scalar>      {}); }
    std::vector<bool> operator!=(const Scalar& rhs) const { return logical_apply_(rhs, std::not_equal_to<Scalar>  {}); }
    std::vector<bool> operator< (const Scalar& rhs) const { return logical_apply_(rhs, std::less<Scalar>          {}); }
    std::vector<bool> operator> (const Scalar& rhs) const { return logical_apply_(rhs, std::greater<Scalar>       {}); }
    std::vector<bool> operator<=(const Scalar& rhs) const { return logical_apply_(rhs, std::less_equal<Scalar>    {}); }
    std::vector<bool> operator>=(const Scalar& rhs) const { return logical_apply_(rhs, std::greater_equal<Scalar> {}); }
   protected:
    template <typename Functor> std::vector<bool> logical_apply_(const Scalar& rhs, Functor&& f) const {
        std::vector<bool> mask(rows(), false);
        for (int i = 0, n = rows(); i < n; ++i) {
	  if (f(operator()(i), rhs)) { mask[i] = true; }
        }
        return mask;
    }
    storage_t slice_;
    DataLayer* data_;
    index_t col_;
    internals::dtype type_id_;
    std::string colname_;
};

template <typename Scalar_, typename DataLayer> struct filtered_col_view : public plain_col_view<Scalar_, DataLayer> {
    using Scalar = Scalar_;
    using Base = plain_col_view<Scalar_, DataLayer>;
    using index_t = typename Base::index_t;
    using size_t = typename Base::size_t;
    static constexpr int Order = Base::Order;
    using reference = typename Base::reference;
    using const_reference = typename Base::const_reference;

    filtered_col_view() noexcept = default;
    template <typename FieldDescriptor>
    filtered_col_view(DataLayer* data, const FieldDescriptor& desc, const std::vector<index_t>& idxs) noexcept :
        Base(data, desc), idxs_(idxs) { }
    // observers
    size_t rows() const { return idxs_.size(); }
    size_t cols() const { return 1; }
    size_t size() const { return idxs_.size(); }
    index_t id() const { return Base::col_; }
    // accessors
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) &&
                (sizeof...(Idxs) == Order - 1 && !std::is_const_v<DataLayer>)
    reference operator()(Idxs&&... idxs) {
        return Base::operator()(static_cast<index_t>(idxs_[idxs])...);
    }
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order - 1)
    const_reference operator()(Idxs&&... idxs) const {
        return Base::operator()(static_cast<index_t>(idxs_[idxs])...);
    }
   private:
    std::vector<index_t> idxs_ {};
};

// an indexed set of rows
template <typename DataLayer> struct plain_row_filter {
    using index_t = typename DataLayer::index_t;
    using size_t = typename DataLayer::size_t;
    using row_view = typename DataLayer::row_view;
    using const_row_view = typename DataLayer::const_row_view;
    using storage_t = std::conditional_t<
      std::is_const_v<DataLayer>, std::add_const<typename DataLayer::storage_t>, typename DataLayer::storage_t>;
    static constexpr int Order = DataLayer::Order;

    plain_row_filter() noexcept = default;
    template <typename Iterator>
    plain_row_filter(DataLayer* data, Iterator begin, Iterator end) : data_(data), idxs_(begin, end) {
      fdapde_assert(*begin >= 0 && *begin < data->rows() && *(end - 1) >= *begin && *(end - 1) < data->rows());
    }
    template <typename Filter>
        requires(requires(Filter f, index_t i) {
            { f(i) } -> std::same_as<bool>;
        })
    plain_row_filter(DataLayer* data, Filter&& f) : data_(data) {
        for (size_t i = 0, n = data_->rows(); i < n; ++i) {
            if (f(i)) idxs_.push_back(i);
        }
    }
    template <typename LogicalVec>
        requires(requires(LogicalVec vec, int i) {
            { vec.size() } -> std::convertible_to<size_t>;
            { vec[i] } -> std::convertible_to<bool>;
        })
    plain_row_filter(DataLayer* data, const LogicalVec& vec) : data_(data) {
        fdapde_assert(vec.size() == data_->rows());
        for (size_t i = 0, n = vec.size(); i < n; ++i) {
	  if (vec[i]) { idxs_.push_back(i); }
        }
    }
    // observers
    size_t rows() const { return idxs_.size(); }
    size_t cols() const { return data_->cols(); }
    size_t size() const { return rows() * cols(); }
    const auto& field_descriptors() const { return data_->field_descriptors(); }
    const auto& field_descriptor(const std::string& colname) const { return data_->field_descriptor(colname); }
    std::vector<std::string> colnames() const { return data_->colnames(); }
    // accessors
    row_view operator()(index_t i) {
        fdapde_assert(i < idxs_.size());
        return data_->row(idxs_[i]);
    }
    const_row_view operator()(index_t i) const {
        fdapde_assert(i < idxs_.size());
        return data_->row(idxs_[i]);
    }
    template <typename T> filtered_col_view<T, DataLayer> get(const std::string& colname) {
        return filtered_col_view<T, DataLayer>(data_, field_descriptor(colname), idxs_);
    }
    template <typename T> filtered_col_view<T, const DataLayer> get(const std::string& colname) const {
        return filtered_col_view<T, const DataLayer>(data_, field_descriptor(colname), idxs_);
    }
    // iterator support
    class iterator {
       public:
        using value_type = plain_row_view<DataLayer>;
        using pointer_t = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(int index, plain_row_filter* accessor) : index_(index), accessor_(accessor), val_() {
            if (index_ != accessor_->rows()) { val_ = accessor_->operator()(index); }
            index_++;
        }
        reference operator*() { return val_; }
        const reference operator*() const { return val_; }
        pointer_t operator->() { return std::addressof(val_); }
        const pointer_t operator->() const { return std::addressof(val_); }
        iterator& operator++() {
            if (index_ != accessor_->rows()) { [[likely]] val_ = accessor_->operator()(index_); }
            index_++;
            return *this;
        }
        friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        index_t index_;
        plain_row_filter* accessor_;
        value_type val_;
    };
    iterator begin() { return iterator(0, this); }
    iterator end() { return iterator(idxs_.size(), this); }
   private:
    DataLayer* data_;
    std::vector<index_t> idxs_;
};

// heterogeneous container
template <int Order_> class plain_data_layer {
    fdapde_static_assert(Order_ > 1, ORDER_MUST_BE_STRICTLY_GREATER_THAN_ONE);
    using types = std::tuple<
      double,         // data_t::flt64
      float,          // data_t::flt32
      std::int64_t,   // data_t::int64
      std::int32_t,   // data_t::int32
      bool,           // data_t::bin
      std::string     // data_t::str
      >;
    template <typename T> struct mapped_type {
        using type = std::decay_t<decltype([](T) {
            if constexpr (internals::is_integer_v<T> && !std::is_same_v<T, std::int64_t>) {
                return std::int32_t();   // map integral, not 64 bit and not boolean types, to 32 bit int
            } else {
                return T();   // do not map other types
            }
        }(T()))>;
    };
    template <typename T> using mapped_type_t = typename mapped_type<T>::type;
    template <typename T> static constexpr bool is_type_supported_v  = has_type_v<T, types>;
    template <typename T> static constexpr bool is_dtype_supported_v = is_type_supported_v<typename T::type>;
    template <typename T> constexpr auto dtype_from_static_type() const {
        using T_ = std::decay_t<T>;
        if constexpr (std::is_same_v<T_, double      >) return data_t::flt64;
        if constexpr (std::is_same_v<T_, float       >) return data_t::flt32;
        if constexpr (std::is_same_v<T_, std::int64_t>) return data_t::int64;
        if constexpr (std::is_same_v<T_, std::int32_t>) return data_t::int32;
        if constexpr (std::is_same_v<T_, bool        >) return data_t::bin;
        if constexpr (std::is_same_v<T_, std::string >) return data_t::str;
    }
    template <typename T> using data_table = MdArray<T, full_dynamic_extent_t<Order_>>;
    template <typename... Ts> using data_map_ = std::tuple<MdArray<Ts, full_dynamic_extent_t<Order_>>...>;
    using This = plain_data_layer<Order_>;
    template <typename T, typename U>
        requires(is_type_supported_v<T>)
    auto& fetch_(U& u) {
        return std::get<index_of_type<T, types>::index>(u);
    }
    template <typename T, typename U>
        requires(is_type_supported_v<T>)
    const auto& fetch_(const U& u) const {
        return std::get<index_of_type<T, types>::index>(u);
    }
    bool has_column_(const std::string& colname) const { return colname_to_field_.contains(colname); }
    struct field {
        std::string colname;
        int col_id;   // column position in MdArray
        internals::dtype type_id;

        field(const std::string& colname_, int col_id_, internals::dtype type_id_) :
            colname(colname_), col_id(col_id_), type_id(type_id_) { }
    };
    template <typename T>
    static constexpr bool is_valid_pair_v =
      (std::is_same_v<std::tuple_element_t<0, std::decay_t<T>>, std::string> ||
       std::is_same_v<std::tuple_element_t<0, std::decay_t<T>>, const char*>) &&
      fdapde::is_subscriptable<std::tuple_element_t<1, std::decay_t<T>>, int> &&
      is_type_supported_v<mapped_type_t<
        std::decay_t<decltype(std::declval<std::tuple_element_t<1, std::decay_t<T>>>()[std::declval<int>()])>>>;
   public:
    template <typename T> using reference = typename data_table<T>::reference;
    template <typename T> using const_reference = typename data_table<T>::const_reference;
    using storage_t = typename fdapde::strip_tuple_into<data_map_, types>::type;
    using index_t = int;
    using size_t = std::size_t;
    using row_view = plain_row_view<plain_data_layer<Order_>>;
    using const_row_view = plain_row_view<const plain_data_layer<Order_>>;
    static constexpr int Order = Order_;

    plain_data_layer() noexcept : rows_(0), cols_(0) { }
    template <typename... DataT>
        requires(
          ([]() {
              if constexpr (internals::is_pair_v<std::decay_t<DataT>>) {   // DataT is a pair-like structure
                  return is_valid_pair_v<std::decay_t<std::decay_t<DataT>>>;
              } else if constexpr (requires(DataT data, index_t i) {   // DataT is a vector of pairs
                                       { data[i] };
                                   }) {
                  if constexpr (internals::is_pair_v<   // check if subscripting DataT returns a tuple
                                  std::decay_t<decltype(std::declval<DataT>()[std::declval<index_t>()])>>) {
                      // check the tuple is indeed ammisible for plain_data_layer
                      return is_valid_pair_v<decltype(std::declval<std::decay_t<DataT>>()[std::declval<index_t>()])>;
                  } else {
                      return false;
                  }
              } else {
                  return false;
              }
          }()) &&
          ...)
    plain_data_layer(DataT&&... data) {
        // for each type id, the number of columns of that type
        std::unordered_map<dtype, int> type_id_cnt {
          {dtype::flt64, 0},
          {dtype::flt32, 0},
          {dtype::int64, 0},
          {dtype::int32, 0},
          {dtype::bin,   0},
          {dtype::str,   0}
        };
        std::unordered_map<dtype, int> type_id_col = type_id_cnt;
        auto push_column_descriptor = [&, this]<typename T>(const std::string& colname, const T& t) mutable {
            if (has_column_(colname) || colname.size() == 0) {
                throw std::runtime_error("GeoFrame: duplicated or empty column names.");
            }
            if (rows_ == 0) {
                rows_ = t.size();
            } else if (rows_ != 0 && rows_ != t.size()) {
                throw std::runtime_error("GeoFrame: columns of different size.");
            }
            using MappedT = mapped_type_t<std::decay_t<decltype(std::declval<T>()[std::declval<index_t>()])>>;
            // add field descriptor
            auto dtype_ = dtype_from_static_type<MappedT>();
            fields_.emplace_back(colname, type_id_cnt[dtype_.type_id], dtype_.type_id);
            type_id_cnt[dtype_.type_id]++;
            colname_to_field_[colname] = fields_.size() - 1;
        };
	// push column descriptors
        internals::for_each_index_and_args<sizeof...(DataT)>(
          [&]<int Ns_, typename T>(T t) {
              if constexpr (internals::is_pair_v<T>) {
                  push_column_descriptor(std::get<0>(t), std::get<1>(t));
		  cols_++;
              } else {   // vector of pairs
                  for (const auto& pair : t) {
                      push_column_descriptor(std::get<0>(pair), std::get<1>(pair));
                      cols_++;
                  }
              }
          },
          data...);
        // reserve space for all types in pack, copy data in internal storage
        internals::for_each_index_and_args<sizeof...(DataT)>(
          [&]<int Ns_, typename T>(T t) {
              using MappedT = mapped_type_t<std::decay_t<decltype([t]() {
                  if constexpr (internals::is_pair_v<T>) {
                      return std::get<1>(t);
                  } else {
                      return std::get<1>(t[0]);
                  }
              }().operator[](std::declval<index_t>()))>>;
              auto dtype_ = dtype_from_static_type<MappedT>();
              fetch_<MappedT>(data_).resize(rows_, type_id_cnt[dtype_.type_id]);
              if constexpr (internals::is_pair_v<T>) {
                  fetch_<MappedT>(data_)
                    .template slice<1>(type_id_col[dtype_.type_id])
                    .assign_inplace_from(std::get<1>(t));
                  type_id_col[dtype_.type_id]++;
              } else {   // map-like object
                  for (const auto& [colname, data] : t) {
                      fetch_<MappedT>(data_).template slice<1>(type_id_col[dtype_.type_id]).assign_inplace_from(data);
                      type_id_col[dtype_.type_id]++;
                  }
              }
          },
          data...);
    }
    template <typename LayerType>
    plain_data_layer(const plain_row_filter<LayerType>& row_filter, const std::vector<std::string>& cols) {
        fdapde_assert(cols.size() > 0);
        // for each type id, the number of columns of that type
        std::unordered_map<dtype, int> type_id_cnt {
          {dtype::flt64, 0},
          {dtype::flt32, 0},
          {dtype::int64, 0},
          {dtype::int32, 0},
          {dtype::bin,   0},
          {dtype::str,   0}
        };
        // push column descriptors
        rows_ = row_filter.rows();	
	cols_ = cols.size();
        for (int i = 0; i < cols_; ++i) {
            internals::dtype type_id = row_filter.field_descriptor(cols[i]).type_id;
            fields_.emplace_back(cols[i], type_id_cnt[type_id], type_id);
            type_id_cnt[fields_.back().type_id]++;
            colname_to_field_[fields_.back().colname] = fields_.size() - 1;
        }	
        // reserve space, copy data in internal storage
        std::apply(
          [&](const auto&... ts) {
              (
                [&]() {
                    using T = std::decay_t<decltype(ts)>;
                    auto dtype_ = dtype_from_static_type<T>();
		    int col_id_ = 0;
                    if (type_id_cnt[dtype_.type_id] != 0) {
                        fetch_<T>(data_).resize(rows_, type_id_cnt[dtype_.type_id]);
                        for (const auto& colname : cols) {
                            auto desc = row_filter.field_descriptor(colname);
                            if (desc.type_id == dtype_.type_id) {
                                auto dst = fetch_<T>(data_).template slice<1>(col_id_);
                                auto src = row_filter.template get<T>(desc.colname);
                                for (int i = 0; i < rows_; ++i) { dst(i) = src(i); }
                                col_id_++;
                            }
                        }
                    }
                }(),
                ...);
          },
          types {});
    }
    template <typename LayerType>
    plain_data_layer(const plain_row_filter<LayerType>& row_filter) :
        plain_data_layer(row_filter, row_filter.colnames()) { }

    // observers
    const field& field_descriptor(const std::string& colname) const { return fields_[colname_to_field_.at(colname)]; }
    const std::vector<field>& field_descriptors() const { return fields_; }
    std::vector<std::string> colnames() const {
        std::vector<std::string> colnames_;
        for (int i = 0, n = fields_.size(); i < n; ++i) { colnames_.push_back(fields_[i].colname); }
        return colnames_;
    }
    bool contains(const std::string& column) const {   // true if this layer contains column
        for (int i = 0, n = fields_.size(); i < n; ++i) {
            if (fields_[i].colname == column) { return true; }
        }
        return false;
    }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    // accessors
    // column access
    template <typename T> plain_col_view<T, plain_data_layer<Order>> col(size_t col) {
        if (col > cols_) { throw std::runtime_error("GeoFrame: out of bound access."); }
        return plain_col_view<T, plain_data_layer<Order>>(this, fields_[col]);
    }
    template <typename T> plain_col_view<T, const plain_data_layer<Order>> col(size_t col) const {
        if (col > cols_) { throw std::runtime_error("GeoFrame: out of bound access."); }
        return plain_col_view<T, const plain_data_layer<Order>>(this, fields_[col]);
    }
    template <typename T> plain_col_view<T, plain_data_layer<Order>> col(const std::string& colname) {
        if (!has_column_(colname)) { throw std::runtime_error("GeoFrame: column not found."); }
        return col<T>(colname_to_field_.at(colname));
    }
    template <typename T> plain_col_view<T, const plain_data_layer<Order>> col(const std::string& colname) const {
        if (!has_column_(colname)) { throw std::runtime_error("GeoFrame: column not found."); }
	return col<T>(colname_to_field_.at(colname));
    }
    // row access
    plain_row_view<plain_data_layer<Order>> row(size_t row) {
        if (row > rows_) { throw std::runtime_error("GeoFrame: out of bound access."); }
        return plain_row_view<plain_data_layer<Order>>(this, row);
    }
    plain_row_view<const plain_data_layer<Order>> row(size_t row) const {
        if (row > rows_) { throw std::runtime_error("GeoFrame: out of bound access."); }
        return plain_row_view<const plain_data_layer<Order>>(this, row);
    }
    // row filtering operations
    template <typename Iterator>
        requires(internals::is_integer_v<typename Iterator::value_type>)
    plain_row_filter<plain_data_layer<Order>> operator()(Iterator begin, Iterator end) {
        return plain_row_filter<plain_data_layer<Order>>(this, begin, end);
    }
    template <typename T>
        requires(std::is_convertible_v<T, index_t>)
    plain_row_filter<plain_data_layer<Order>> operator()(const std::initializer_list<T>& idxs) {
        return plain_row_filter<plain_data_layer<Order>>(this, idxs.begin(), idxs.end());
    }
    template <typename Filter>
        requires(requires(Filter f, index_t i) {
            { f(i) } -> std::same_as<bool>;
        })
    plain_row_filter<plain_data_layer<Order>> operator()(Filter&& f) {
        return plain_row_filter<plain_data_layer<Order>>(this, std::forward<Filter>(f));
    }
    template <typename LogicalVec>
        requires(requires(LogicalVec vec, int i) {
            { vec.size() } -> std::convertible_to<size_t>;
            { vec[i] } -> std::convertible_to<bool>;
        })
    plain_row_filter<plain_data_layer<Order>> operator()(LogicalVec&& logical_vec) {
        return plain_row_filter<plain_data_layer<Order>>(this, logical_vec);
    }
  
    template <typename T> const data_table<T>& data() const { return fetch_<mapped_type_t<T>>(data_); }
    template <typename T> data_table<T>& data() { return fetch_<mapped_type_t<T>>(data_); }
    // modifiers
    void set_colnames(const std::vector<std::string>& colnames) {
        if (colnames.size() != fields_.size()) { throw std::runtime_error("GeoFrame: wrong number of columns."); }
        std::vector<std::string> tmp = colnames;
        std::sort(tmp.begin(), tmp.end());
        if (
          colnames.size() != cols_ || std::unique(tmp.begin(), tmp.end()) != tmp.end() ||
          !std::accumulate(
            colnames.begin(), colnames.end(), true, [](bool v, const auto& name) { return (v & !name.empty()); })) {
            throw std::runtime_error("GeoFrame: not unique or empty column names.");
        }
        for (size_t i = 0; i < colnames.size(); ++i) { fields_[i].colname = colnames[i]; }
        return;
    }
    template <typename Src>
        requires(
          (std::is_pointer_v<Src> && is_type_supported_v<std::remove_pointer_t<Src>>) ||
          (fdapde::is_subscriptable<Src, index_t> &&
           requires(Src src) {
               { src.size() } -> std::convertible_to<size_t>;
           } && is_type_supported_v<std::decay_t<decltype(std::declval<Src>()[index_t()])>>))
    void add_column(const std::string& name, const Src& src) {
        using SrcType = std::conditional_t<
          std::is_pointer_v<Src>, std::remove_pointer_t<Src>, std::decay_t<decltype(std::declval<Src>()[index_t()])>>;
	using MappedSrcType = mapped_type_t<SrcType>;
	if constexpr (!std::is_pointer_v<Src>) { fdapde_assert(src.size() == fetch_<MappedSrcType>(data_).extent(0)); }
        // add field descriptor
        auto dtype_ = dtype_from_static_type<MappedSrcType>();
        int col = 0;
        for (const field& f : fields_) {
            if (f.type_id == dtype_.type_id) { col++; }
        }
        fields_.emplace_back(name, col, dtype_.type_id);
	colname_to_field_[name] = fields_.size() - 1;
        // resize space if column doesn't fit current size (double number of columns, amortized constant time insertion)
        if (col == fetch_<SrcType>(data_).extent(1)) {
            internals::apply_index_pack<Order>([&]<int... Ns_>() {
                conservative_resize(
                  dtype_,
                  (Ns_ == 1 ? (2 * fetch_<SrcType>(data_).extent(Ns_)) : fetch_<SrcType>(data_).extent(Ns_))...);
            });
        }
        // copy src into data_
        fetch_<SrcType>(data_).template slice<1>(col).assign_inplace_from(src);
	cols_++;
        return;
    }  
    // output stream
    friend std::ostream& operator<<(std::ostream& os, const plain_data_layer& data) {
        std::vector<std::vector<std::string>> out;
        std::vector<std::size_t> max_size(data.field_descriptors().size(), 0);
	int n_rows = std::min(size_t(8), data.rows());
        out.resize(data.cols());
        auto print =
          [&]<typename T>(std::vector<std::string>& out, const std::string& typestring, const std::string& colname) {
              out.push_back(colname);
              out.push_back(typestring);
              auto col = data.col<T>(colname);
              if constexpr (!std::is_same_v<T, std::string>) {
                  for (int i = 0; i < n_rows; ++i) { out.push_back(std::to_string(col(i))); }
              } else {
                  for (int i = 0; i < n_rows; ++i) { out.push_back(col(i)); }
              }
          };
        for (int i = 0, n = data.cols(); i < n; ++i) {
            const std::string& colname = data.field_descriptors()[i].colname;
            dtype coltype = data.field_descriptors()[i].type_id;
            if (coltype == dtype::flt64) { print.template operator()<double      >(out[i], "<flt64>", colname); }
            if (coltype == dtype::flt32) { print.template operator()<float       >(out[i], "<flt32>", colname); }
            if (coltype == dtype::int64) { print.template operator()<std::int64_t>(out[i], "<int64>", colname); }
            if (coltype == dtype::int32) { print.template operator()<std::int32_t>(out[i], "<int32>", colname); }
            if (coltype == dtype::bin)   { print.template operator()<bool        >(out[i], "<bin>"  , colname); }
            if (coltype == dtype::str)   { print.template operator()<std::string >(out[i], "<str>"  , colname); }
        }
        // pretty format
        for (int i = 0, n = data.cols(); i < n; ++i) {
            for (int j = 0, m = out[i].size(); j < m; ++j) { max_size[i] = std::max(max_size[i], out[i][j].size()); }
        }
        for (int i = 0, n = data.cols(); i < n; ++i) {
            for (int j = 0, m = out[i].size(); j < m; ++j) {
                out[i][j].append(max_size[i] - out[i][j].size() + 1, ' ');
            }
        }
	// send to output stream
        for (int j = 0, m = out[0].size(); j < m - 1; ++j) {
            for (int i = 0, n = data.cols(); i < n; ++i) { os << out[i][j]; }
            os << std::endl;
        }
        for (int i = 0, n = data.cols(); i < n; ++i) { os << out[i][out[0].size() - 1]; }
        return os;
    }
   private:
    // resize mdarray storage, preserving old values
    template <typename dtype, typename... Extents_>
        requires(std::is_convertible_v<Extents_, index_t> && ...) &&
                (sizeof...(Extents_) == Order && is_type_supported_v<typename dtype::type>)
    void conservative_resize(dtype, Extents_... exts) {
        using mem_t = MdArray<typename dtype::type, full_dynamic_extent_t<Order_>>;
        mem_t& data = fetch_<typename dtype::type>(data_);
        // exts coincide with current size, skip resizing
        if (internals::apply_index_pack<Order>([&]<int... Ns_>() { return ((exts == data.extent(Ns_)) && ...); })) {
            return;
        }
        auto tmp = data.block(static_cast<index_t>(exts)...);
        data.resize(static_cast<index_t>(exts)...);
        data = tmp;
        return;
    }  
    storage_t data_;
    std::vector<field> fields_;
    std::unordered_map<std::string, int> colname_to_field_;
    int rows_ = 0, cols_ = 0;
};

}   // namespace internals
}   // namespace fdapde

#endif // __PLAIN_DATA_LAYER_H__
