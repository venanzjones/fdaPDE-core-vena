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

template <typename FeSpace_> class FeFunction;

template <typename Triangulation_, typename FeType_> class FeSpace {
    // internals
    template <typename T> struct subscript_type_of_impl {
        using type = std::decay_t<decltype(std::declval<T>().operator[](std::declval<int>()))>;
    };
    template <typename T> using subscript_type_of = typename subscript_type_of_impl<T>::type;
   public:
    using Triangulation = std::decay_t<Triangulation_>;
    using FeType = std::decay_t<FeType_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using face_dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using ReferenceCell = typename cell_dof_descriptor::ReferenceCell;
    using BasisType = typename cell_dof_descriptor::BasisType;
    using ShapeFunctionType = subscript_type_of<BasisType>;
    using FaceBasisType = typename face_dof_descriptor::BasisType;
    using FaceShapeFunctionType = subscript_type_of<FaceBasisType>;
    using DofHandlerType = DofHandler<local_dim, embed_dim>;
    // vector finite element descriptors
    static constexpr int n_components  = FeType::n_components;
    static constexpr bool is_vector_fe = (n_components > 1);
  
    FeSpace() = default;
    FeSpace(const Triangulation_& triangulation, FeType_ fe) :
        triangulation_(std::addressof(triangulation)), dof_handler_(triangulation) {
        dof_handler_.enumerate(fe);
        if constexpr (requires(cell_dof_descriptor c) { c.dofs_phys_coords(); }) {
            cell_basis_ = BasisType(unit_cell_dofs_.dofs_phys_coords());
        } else {
            cell_basis_ = BasisType();
        }
        if constexpr (requires(face_dof_descriptor c) { c.dofs_phys_coords(); }) {
            face_basis_ = FaceBasisType(unit_face_dofs_.dofs_phys_coords());
        } else {
            face_basis_ = FaceBasisType();
        }
    }
    // observers
    const Triangulation& triangulation() const { return *triangulation_; }
    const DofHandlerType& dof_handler() const { return dof_handler_; }
    DofHandlerType& dof_handler() { return dof_handler_; }
    constexpr int n_shape_functions() const { return n_components * cell_basis_.size(); }
    constexpr int n_shape_functions_face() const { return n_components * face_basis_.size(); }
    int n_dofs() const { return dof_handler_.n_dofs(); }
    const BasisType& cell_basis() const { return cell_basis_; }
    const FaceBasisType& face_basis() const { return face_basis_; }
    // evaluation
    template <typename InputType>
        requires(std::is_invocable_v<ShapeFunctionType, InputType>)
    constexpr auto eval_shape_value(int i, const InputType& p) const {
        return cell_basis_[i](p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<ShapeFunctionType>().gradient()), InputType>)
    constexpr auto eval_shape_grad(int i, const InputType& p) const {
        return cell_basis_[i].gradient()(p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<ShapeFunctionType>().divergence()), InputType>)
    constexpr auto eval_shape_div(int i, const InputType& p) const {
        fdapde_static_assert(n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENTS_ONLY);
        return cell_basis_[i].divergence()(p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<FaceShapeFunctionType, InputType>)
    constexpr auto eval_face_shape_value(int i, const InputType& p) const {
        return face_basis_[i](p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<FaceShapeFunctionType>().gradient()), InputType>)    
    constexpr auto eval_face_shape_grad(int i, const InputType& p) const {
        return face_basis_[i].gradient()(p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<FaceShapeFunctionType>().divergence()), InputType>)
    constexpr auto eval_face_shape_div(int i, const InputType& p) const {
        fdapde_static_assert(n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENTS_ONLY);
        return face_basis_[i].divergence()(p);
    }
    // evaluation on physical domain
    // shape function evaluation, skip point location step (cell_id provided as input)
    template <typename InputType> auto eval_cell_value(int i, int cell_id, const InputType& p) const {
        // map p to reference cell and evaluate
        typename DofHandler<local_dim, embed_dim>::CellType cell = dof_handler_.cell(cell_id);
        InputType ref_p = cell.invJ() * (p - cell.node(0));
        return eval_shape_value(i, ref_p);
    }
    // evaluate value of the i-th shape function defined on the physical cell containing point p
    template <typename InputType> auto eval_cell_value(int i, const InputType& p) const {
        // localize p in physical domain
        int cell_id = triangulation_->locate(p);
        if (cell_id == -1) return std::numeric_limits<double>::quiet_NaN();
	return eval_cell_value(i, cell_id, p);
    }
  
  // need to return something which represent a basis function on the whole physical domain
  
    // generate fe_function bounded to this finite element space
    FeFunction<FeSpace<Triangulation_, FeType_>> make_fe_function(const Eigen::Matrix<double, Dynamic, 1>& coeff_vec) {
        return FeFunction<FeSpace<Triangulation_, FeType_>>(*this, coeff_vec);
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
