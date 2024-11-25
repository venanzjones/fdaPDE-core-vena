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

#ifndef __FE_SPACE_H__
#define __FE_SPACE_H__

#include "../utils/symbols.h"
#include "dof_handler.h"

namespace fdapde {

template <typename Triangulation_, typename FeType_> class FeSpace {
   public:
    using Triangulation = std::decay_t<Triangulation_>;
    using FeType = std::decay_t<FeType_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using face_dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using ReferenceCell = typename cell_dof_descriptor::ReferenceCell;
    using BasisType = typename cell_dof_descriptor::BasisType;
    using DofHandlerType = DofHandler<local_dim, embed_dim>;
    // vector finite element descriptors
    static constexpr int n_components  = FeType::n_components;
    static constexpr bool is_vector_fe = (n_components > 1);
  
    FeSpace() = default;
    FeSpace(const Triangulation_& triangulation, FeType_ fe) :
        triangulation_(&triangulation), dof_handler_(triangulation) {
        dof_handler_.enumerate(fe);
        cell_basis_ = BasisType(unit_cell_dofs_.dofs_phys_coords());
        face_basis_ = typename face_dof_descriptor::BasisType(unit_face_dofs_.dofs_phys_coords());
    }
    // observers
    const Triangulation& triangulation() const { return *triangulation_; }
    const DofHandlerType& dof_handler() const { return dof_handler_; }
    DofHandlerType& dof_handler() { return dof_handler_; }
    constexpr int n_shape_functions() const { return n_components * cell_basis_.size(); }
    constexpr int n_shape_functions_face() const { return n_components * face_basis_.size(); }
    int n_dofs() const { return dof_handler_.n_dofs(); }
    const BasisType& cell_basis() const { return cell_basis_; }
    const typename face_dof_descriptor::BasisType& face_basis() const { return face_basis_; }

    // evaluation
    template <typename T> constexpr auto eval_shape_value(int i, const T& p) const { return cell_basis_[i](p); }
    template <typename T> constexpr auto eval_shape_grad(int i, const T& p) const {
        return cell_basis_[i].gradient()(p);
    }
    template <typename T> constexpr auto eval_shape_div(int i, const T& p) const {
        fdapde_static_assert(n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENTS_ONLY);
        return cell_basis_[i].divergence()(p);
    }
    template <typename T> constexpr auto eval_face_shape_value(int i, const T& p) const { return face_basis_[i](p); }
    template <typename T> constexpr auto eval_face_shape_grad(int i, const T& p) const {
        return face_basis_[i].gradient()(p);
    }
    template <typename T> constexpr auto eval_face_shape_div(int i, const T& p) const {
        fdapde_static_assert(n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENTS_ONLY);
        return face_basis_[i].divergence()(p);
    }
    template <typename T> auto eval_cell_value(int i, const T& p) const {
        // localize p in physical domain
        int e_id = triangulation_->locate(p);
        if (e_id == -1) return std::numeric_limits<double>::quiet_NaN();
        // map p to reference cell and evaluate
        typename DofHandler<local_dim, embed_dim>::CellType cell = dof_handler_.cell(e_id);
        T ref_p = cell.invJ() * (p - cell.node(0));
        return eval_shape_value(i, ref_p);
    }
   private:
    const Triangulation* triangulation_;
    DofHandlerType dof_handler_;
    cell_dof_descriptor unit_cell_dofs_;
    face_dof_descriptor unit_face_dofs_;
    BasisType cell_basis_;
    typename face_dof_descriptor::BasisType face_basis_;
};
  
}   // namespace fdapde

#endif   // __FE_SPACE_H__
