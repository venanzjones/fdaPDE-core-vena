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

#ifndef __BS_DOF_HANDLER_H__
#define __BS_DOF_HANDLER_H__

#include "../geometry/triangulation.h"
#include "../utils/symbols.h"

namespace fdapde {
  
class BsDofHandler {
   public:
    using TriangulationType = Triangulation<1, 1>;
    static constexpr int local_dim = TriangulationType::local_dim;
    static constexpr int embed_dim = TriangulationType::embed_dim;

    // a geometrical segment with attached dofs
    struct CellType : public Segment<TriangulationType> {
        using Base = Segment<TriangulationType>;
        const BsDofHandler* dof_handler_;
       public:
        static constexpr int local_dim = 1;
        static constexpr int embed_dim = 1;

        CellType() : dof_handler_(nullptr) { }
        CellType(int cell_id, const BsDofHandler* dof_handler) :
            Base(cell_id, dof_handler->triangulation()), dof_handler_(dof_handler) { }
        std::vector<int> dofs() const { return dof_handler_->active_dofs(Base::id()); }
        std::vector<int> dofs_markers() const {
            std::vector<int> dofs_ = dofs();
            std::vector<int> dofs_markers_(dofs_.size());
            for (int i = 0, n = dofs_.size(); i < n; ++i) { dofs_markers_[i] = dof_handler_->dof_marker(dofs_[i]); }
	    return dofs_markers_;
        }
        BinaryVector<Dynamic> boundary_dofs() const {
            std::vector<int> dofs_ = dofs();
            BinaryVector<fdapde::Dynamic> boundary(dofs_.size());
            int i = 0;
            for (int dof : dofs_) {
                if (dof_handler_->is_dof_on_boundary(dof)) boundary.set(i);
                ++i;
            }
            return boundary;
        }
    };
    // constructor
    BsDofHandler() = default;
    BsDofHandler(const TriangulationType& triangulation) : triangulation_(std::addressof(triangulation)) { }
    // getters
    CellType cell(int id) const { return CellType(id, this); }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> dofs() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          dofs_.data(), dofs_.size() / 2, 2);
    }
    int n_dofs() const { return n_dofs_; }
    bool is_dof_on_boundary(int i) const { return boundary_dofs_[i]; }
    const std::vector<int>& dofs_markers() const { return dofs_markers_; }
    int dof_marker(int dof) const { return dofs_markers_[dof]; }
    const TriangulationType* triangulation() const { return triangulation_; }
    const std::vector<double>& dofs_coords() const { return dofs_coords_; }
    int n_boundary_dofs() const { return boundary_dofs_.count(); }
    int n_boundary_dofs(int marker) const {
        int i = 0, sum = 0;
        for (int dof_marker : dofs_markers_) { sum += (dof_marker == marker && boundary_dofs_[i++]) ? 1 : 0; }
        return sum;
    }
    std::vector<int> filter_dofs_by_marker(int marker) const {
        std::vector<int> result;
        for (int i = 0; i < n_dofs_; ++i) {
            if (dofs_markers_[i] == marker) result.push_back(i);
        }
        return result;
    }
    std::vector<int> active_dofs(int i) const {
        int cell_id = 2 * i;
	int n_dofs = dofs_[cell_id + 1] - dofs_[cell_id] + 1;
        std::vector<int> tmp(n_dofs);
        for (int j = 0; j <= n_dofs; ++j) { tmp[j] = dofs_[cell_id] + j; }
        return tmp;
    }
    template <typename ContainerT> void active_dofs(int cell_id, ContainerT& dst) const {
        for (int i = dofs_[cell_id], n = dofs_[cell_id + 1]; i <= n; ++i) { dst.push_back(i); }
    }
    operator bool() const { return n_dofs_ != 0; }

    // iterates over geometric cells coupled with dofs informations (possibly filtered by marker)
    class cell_iterator : public internals::filtering_iterator<cell_iterator, CellType> {
        using Base = internals::filtering_iterator<cell_iterator, CellType>;
        using Base::index_;
        friend Base;
        const BsDofHandler* dof_handler_;
        int marker_;
        cell_iterator& operator()(int i) {
            Base::val_ = dof_handler_->cell(i);
            return *this;
        }
       public:
        cell_iterator() = default;
        cell_iterator(
          int index, const BsDofHandler* dof_handler, const BinaryVector<fdapde::Dynamic>& filter, int marker) :
            Base(index, 0, dof_handler->triangulation()->n_cells(), filter),
            dof_handler_(dof_handler),
            marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        cell_iterator(int index, const BsDofHandler* dof_handler, int marker) :
            cell_iterator(
              index, dof_handler,
              marker == TriangulationAll ?
                BinaryVector<fdapde::Dynamic>::Ones(dof_handler->triangulation()->n_cells()) :   // apply no filter
                fdapde::make_binary_vector(
                  dof_handler->triangulation()->cells_markers().begin(),
                  dof_handler->triangulation()->cells_markers().end(), marker),
              marker) { }
        int marker() const { return marker_; }
    };  
    cell_iterator cells_begin(int marker = TriangulationAll) const {
        const std::vector<int>& cells_markers = triangulation_->cells_markers();
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers.size() != 0));
        return cell_iterator(0, this, marker);
    }
    cell_iterator cells_end(int marker = TriangulationAll) const {
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && triangulation_->cells_markers().size() != 0));
        return cell_iterator(triangulation_->n_cells(), this, marker);
    }

    // class boundary_dofs_iterator {};

    template <typename BsType> void enumerate(BsType&& bs) {
        n_dofs_ = bs.size();
        dofs_coords_.resize(n_dofs_);
        for (int i = 0; i < n_dofs_; ++i) { dofs_coords_[i] = bs[i].knot(); }
        int i = 0, n_cells = triangulation()->n_cells();
        for (int j = 0; j < n_cells; ++j) {
            dofs_.push_back(i);
            while (i < n_dofs_ && dofs_coords_[i] == dofs_coords_[i + 1]) { i++; }
            i++;
            while (i < n_dofs_ && dofs_coords_[i] == dofs_coords_[i + 1]) { i++; }
            dofs_.push_back(i);
        }
        // update boundary and inherit markers from geometry
        boundary_dofs_.resize(n_dofs_);
        dofs_markers_ = std::vector<int>(n_dofs_, Unmarked);
        for (int i = dofs_[0]; i < dofs_[1]; ++i) { boundary_dofs_.set(i); }
        for (int i = dofs_.back() - 1; i <= dofs_.back(); ++i) { boundary_dofs_.set(i); }
        dofs_markers_ = triangulation_->nodes_markers();
        return;
    }
   private:
    std::vector<double> dofs_coords_;       // physical knots vector
    BinaryVector<Dynamic> boundary_dofs_;   // whether the i-th dof is on boundary or not
    std::vector<int> dofs_;
    int n_dofs_;
    std::vector<int> dofs_markers_;
    const Triangulation<1, 1>* triangulation_;
};

}   // namespace fdapde

#endif   // __BS_DOF_HANDLER_H__
