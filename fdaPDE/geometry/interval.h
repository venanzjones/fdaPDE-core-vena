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

#ifndef __INTERVAL_H__
#define __INTERVAL_H__

#include "segment.h"
#include "../linear_algebra/binary_matrix.h"

namespace fdapde {
  
// template specialization for 1D meshes (bounded intervals)
template <int M, int N> class Triangulation;
template <> class Triangulation<1, 1> : public TriangulationBase<1, 1, Triangulation<1, 1>> {
   public:
    using Base = TriangulationBase<1, 1, Triangulation<1, 1>>;
    using Base::cells_;       // N \times 2 matrix of nodes identifiers of each segment
    using Base::n_cells_;     // N: number of segments
    using Base::n_nodes_;     // number of nodes (N + 1)
    using Base::neighbors_;   // N \times 2 matrix of neighboring cells identifiers
    using Base::nodes_;       // physical coordinates of nodes

    Triangulation() = default;
    Triangulation(const Eigen::Matrix<double, Dynamic, 1>& nodes) : Base() {
        fdapde_assert(nodes.rows() > 1 && nodes.cols() == 1);
        nodes_ = nodes;
        // store number of nodes and elements
        n_nodes_ = nodes_.rows();
        n_cells_ = n_nodes_ - 1;
        // compute mesh limits
        Base::range_[0] = nodes[0];
        Base::range_[1] = nodes[n_nodes_ - 1];
        // build elements and neighboring structure
        cells_.resize(n_cells_, 2);
        for (int i = 0; i < n_nodes_ - 1; ++i) {
            cells_(i, 0) = i;
            cells_(i, 1) = i + 1;
        }
        neighbors_ = Eigen::Matrix<int, Dynamic, Dynamic>::Constant(n_cells_, n_neighbors_per_cell, -1);
        neighbors_(0, 1) = 1;
        for (int i = 1; i < n_cells_ - 1; ++i) {
            neighbors_(i, 0) = i - 1;
            neighbors_(i, 1) = i + 1;
        }
        neighbors_(n_cells_ - 1, 0) = n_cells_ - 2;
        // set first and last nodes as boundary nodes
        Base::boundary_markers_.resize(n_nodes_);
        Base::boundary_markers_.set(0);
        Base::boundary_markers_.set(n_nodes_ - 1);
    };
    // construct from interval's bounds [a, b] and the number of equidistant nodes n used to split [a, b]
    Triangulation(double a, double b, int n) : Triangulation(DVector<double>::LinSpaced(n, a, b)) { }

    // getters
    const typename Base::CellType& cell(int id) const {
        if (Base::flags_ & cache_cells) {   // cell caching enabled
            return cell_cache_[id];
        } else {
            cell_ = typename Base::CellType(id, this);
            return cell_;
        }
    }
    bool is_node_on_boundary(int id) const { return (id == 0 || id == (n_nodes_ - 1)); }
    constexpr int n_boundary_nodes() const { return 2; }
    double measure() const { return std::abs(range_[1] - range_[0]); }
    // boundary iterator
    using boundary_iterator = Base::boundary_node_iterator;
    BoundaryIterator<Triangulation<1, 1>> boundary_begin(int marker = BoundaryAll) const {
        return BoundaryIterator<Triangulation<1, 1>>(0, this, marker);
    }
    BoundaryIterator<Triangulation<1, 1>> boundary_end(int marker = BoundaryAll) const {
        return BoundaryIterator<Triangulation<1, 1>>(n_boundary_nodes(), this, marker);
    }
    std::pair<BoundaryIterator<Triangulation<1, 1>>, BoundaryIterator<Triangulation<1, 1>>>
    boundary(int marker = BoundaryAll) const {
        return std::make_pair(boundary_begin(marker), boundary_end(marker));
    }
    // point location
    template <int Rows, int Cols>
    std::conditional_t<Rows == Dynamic || Cols == Dynamic, DVector<int>, int>
    locate(const Eigen::Matrix<double, Rows, Cols>& p) const {
        fdapde_static_assert(
          (Cols == 1 && Rows == 1) || (Cols == Dynamic && Rows == Dynamic),
          YOU_PASSED_A_MATRIX_OF_POINTS_TO_LOCATE_OF_WRONG_DIMENSIONS);
        if constexpr (Cols == 1) {
            return locate_(p[0]);
        } else {
            fdapde_assert(p.rows() > 0 && p.cols() == 1);
            DVector<int> result;
            result.resize(p.rows());
            // start search
            for (int i = 0; i < p.rows(); ++i) { result[i] = locate_(p(i, 0)); }
            return result;
        }
    }
    std::vector<int> node_patch(int node_id) const {   // set of cells have node with id node_id as vertex, O(1)
        return is_node_on_boundary(node_id) ?
	  std::vector<int>({node_id == 0 ? 0 : (n_cells_ - 1)}) : std::vector<int>({node_id - 1, node_id});
    }

    std::vector<int> node_one_ring(int node_id) const {   // nodes connected to the node with id node_id, O(1)
        return is_node_on_boundary(node_id) ?
	  std::vector<int>({node_id == 0 ? 1 : (n_nodes_ - 2)}) : std::vector<int>({node_id - 1, node_id + 1});
    }
   private:
      // localize element containing point using a O(log(n)) binary search strategy
    int locate_(double p) const {
        // check if point is inside
        if (p < range_[0] || p > range_[1]) {
            return -1;
        } else {
            // binary search strategy
            int h_min = 0, h_max = n_nodes_;
            while (true) {
                int j = h_min + std::floor((h_max - h_min) / 2);
                if (p >= nodes_(j, 0) && p < nodes_(j + 1, 0)) {
                    return j;
                } else {
                    if (p < nodes_(j, 0)) {
                        h_max = j;
                    } else {
                        h_min = j;
                    }
                }
            }
        }
        return -1;
    }
    // cell caching
    std::vector<typename Base::CellType> cell_cache_;
    mutable typename Base::CellType cell_;   // used in case cell caching is off
};

}   // namespace fdapde

#endif   // __INTERVAL_H__
