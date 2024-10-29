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

#ifndef __POLYGON_H__
#define __POLYGON_H__

#include "primitives.h"
#include "dcel.h"
#include <deque>
#include <set>

namespace fdapde {

// a polygon is a connected list of vertices
template <int LocalDim, int EmbedDim> class Polygon {
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;

    // constructors
    Polygon() noexcept = default;
    Polygon(const DMatrix<double>& nodes) noexcept : triangulation_() { triangulate_(nodes); }
    Polygon(const Polygon&) noexcept = default;
    Polygon(Polygon&&) noexcept = default;  
    // observers
    const DMatrix<double>& nodes() const { return triangulation_.nodes(); }
    const DMatrix<int, Eigen::RowMajor>& cells() const { return triangulation_.cells(); }
    const Triangulation<local_dim, embed_dim>& triangulation() const { return triangulation_; }
    double measure() const { return triangulation_.measure(); }
    int n_nodes() const { return triangulation_.n_nodes(); }
    int n_cells() const { return triangulation_.n_cells(); }
    int n_edges() const { return triangulation_.n_edges(); }
    // point location
    template <int Rows, int Cols>
    std::conditional_t<Rows == Dynamic || Cols == Dynamic, DVector<int>, int>
    locate(const Eigen::Matrix<double, Rows, Cols>& p) const {
        return triangulation_.locate(p);
    }
    // test whether point p is contained in polygon
    template <int Rows, int Cols>
        requires((Rows == embed_dim && Cols == 1) || (Cols == embed_dim && Rows == 1))
    bool contains(const Eigen::Matrix<double, Rows, Cols>& p) const {
        return triangulation_.locate(p) != -1;
    }
   private:
    // perform polygon triangulation
    void triangulate_(const DMatrix<double>& nodes) {
        std::vector<int> cells;
        // perform monotone partitioning
        std::vector<std::vector<int>> poly_partition = monotone_partition_(nodes);
        // triangulate each monotone polygon
        for (const std::vector<int>& poly : poly_partition) {
            std::vector<int> local_cells = triangulate_monotone_(nodes(poly, Eigen::all));
            // move local node numbering to global node numbering
            for (std::size_t i = 0; i < local_cells.size(); ++i) { local_cells[i] = poly[local_cells[i]]; }
            cells.insert(cells.end(), local_cells.begin(), local_cells.end());
        }
        // set-up face-based data structure
        triangulation_ = Triangulation<LocalDim, EmbedDim>(
          nodes, Eigen::Map<DMatrix<int, Eigen::RowMajor>>(cells.data(), cells.size() / 3, 3),
          DVector<int>::Ones(nodes.rows()));
    }

    // partition an arbitrary polygon P into a set of monotone polygons (plane sweep approach, section 3.2 of De Berg,
    // M. (2000). Computational geometry: algorithms and applications. Springer Science & Business Media.)
    std::vector<std::vector<int>> monotone_partition_(const DMatrix<double>& nodes) {
        using poly_t = DCEL<local_dim, embed_dim>;
        using node_t = typename poly_t::node_t;
        using edge_t = typename poly_t::edge_t;
        using node_ptr_t = std::add_pointer_t<node_t>;
        using edge_ptr_t = std::add_pointer_t<edge_t>;
        // data structure inducing an ad-hoc edge ordering for monotone partitioning
        struct edge_t_ {
            double p1x, p1y;   // p1 coordinates
            double p2x, p2y;   // p2 coordinates
            int id;
            // constructor
            edge_t_(double p1x_, double p1y_, double p2x_, double p2y_) noexcept :
                p1x(p1x_), p1y(p1y_), p2x(p2x_), p2y(p2y_) { }
            edge_t_(int id_, double p1x_, double p1y_, double p2x_, double p2y_) noexcept :
                p1x(p1x_), p1y(p1y_), p2x(p2x_), p2y(p2y_), id(id_) { }
            // right-to-left edge ordering relation along x-coordinate
            bool operator<(const edge_t_& rhs) const {   // for monotone partitioning, rhs is always below the p1 point
                if (rhs.p1y == rhs.p2y) {               // rhs is horizontal
                    if (p1y == p2y) { return (p1x < rhs.p1x); }   // both edges are horizontal lines
                    return internals::orientation(
                             std::array {p2x, p2y}, std::array {p1x, p1y}, std::array {rhs.p1x, rhs.p1y}) ==
                           internals::Orientation::LEFT;
                } else if (p1y <= p2y) {
                    return internals::orientation(
                             std::array {rhs.p2x, rhs.p2y}, std::array {rhs.p1x, rhs.p1y}, std::array {p1x, p1y}) !=
                           internals::Orientation::LEFT;
                } else {
                    return internals::orientation(
                             std::array {p2x, p2y}, std::array {rhs.p1x, rhs.p1y}, std::array {p1x, p1y}) ==
                           internals::Orientation::LEFT;
                }
            }
        };
        // O(n) polygon construction as Doubly Connected Edge List
        poly_t poly = DCEL<local_dim, embed_dim>::make_polygon(nodes);
        int n_nodes = poly.n_nodes();
        int n_edges = poly.n_edges();
        std::set<edge_t_> sweep_line;   // edges pierced by sweep line, sorted by x-coord
        std::vector<node_ptr_t> helper(n_edges, nullptr);
        std::vector<node_ptr_t> node_ptr(n_nodes);
        for (auto it = std::make_pair(poly.nodes_begin(), 0); it.first != poly.nodes_end(); ++it.first, ++it.second) {
            node_ptr[it.second] = std::addressof(*it.first);
        }
        // given node id i, return the previous and next node ids on the counterclockwise sorted polygon border
        auto node_prev = [&](int node_id) { return node_id == 0 ? n_nodes - 1 : node_id - 1; };
        auto node_next = [&](int node_id) { return node_id == n_nodes - 1 ? 0 : node_id + 1; };

        // O(n) node type detection
        enum node_category_t { start = 0, split = 1, end = 2, merge = 3, regular = 4 };
        std::vector<node_category_t> node_category(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {   // counterclocwise order loop, the interior of P is always on the left
            auto prev = nodes.row(node_prev(i));
            auto curr = nodes.row(i);
            auto next = nodes.row(node_next(i));
            double m_signed = internals::signed_measure_2d_tri(prev, curr, next);
            if (m_signed >= 0) {   // interior angle less or equal than \pi
                if (prev[1] < curr[1] && next[1] < curr[1]) {
                    node_category[i] = node_category_t::start;
                } else if (prev[1] > curr[1] && next[1] > curr[1]) {
                    node_category[i] = node_category_t::end;
                } else {
                    node_category[i] = node_category_t::regular;
                }
            } else if (m_signed < 0) {   // interior angle grater than \pi
                if (prev[1] < curr[1] && next[1] < curr[1]) {
                    node_category[i] = node_category_t::split;
                } else if (prev[1] > curr[1] && next[1] > curr[1]) {
                    node_category[i] = node_category_t::merge;
                } else {
                    node_category[i] = node_category_t::regular;
                }
            }
        }
        // O(nlog(n)) y-coordinate sort (break tiles using x-coordinate)
        std::sort(node_ptr.begin(), node_ptr.end(), [&](node_ptr_t n, node_ptr_t m) {
            int i = n->id(), j = m->id();
            return nodes(i, 1) > nodes(j, 1) || (nodes(i, 1) == nodes(j, 1) && nodes(i, 0) < nodes(j, 0));
        });
        // monotone partitioning algorithm (details in section 3.2 of De Berg, M. (2000). Computational geometry:
        // algorithms and applications. Springer Science & Business Media.)
	std::vector<edge_ptr_t> diagonals;
        std::vector<typename std::set<edge_t_>::iterator> sweep_line_it(n_edges);
        sweep_line_it.resize(n_edges);
        // given edge {p1, p2}, asserts true if the polygon interior is on the left of {p1, p2}
        auto interior_on_left = []<typename point_t>(const point_t& p1, const point_t& p2) {
            if (p1[1] < p2[1]) {
                return true;
            } else if (p1[1] == p2[1]) {
                if (p1[0] < p2[0]) { return true; }
            }
            return false;
        };
	
        for (node_ptr_t v : node_ptr) {   // loops in decreasing y-coordinate order
            int v_i = v->id();
	    // process i-th node
            switch (node_category[v_i]) {
            case node_category_t::start: {
                auto ref = sweep_line.emplace(
                  v_i, nodes(v_i, 0), nodes(v_i, 1), nodes(node_next(v_i), 0), nodes(node_next(v_i), 1));
                sweep_line_it[v_i] = ref.first;
                helper[v_i] = v;
                break;
            }
            case node_category_t::end: {
                if (node_category[helper[node_prev(v_i)]->id()] == node_category_t::merge) {
                    diagonals.push_back(poly.insert_edge(v, helper[node_prev(v_i)]));
                }
                sweep_line.erase(sweep_line_it[node_prev(v_i)]);
                break;
            }
            case node_category_t::split: {
                // search edge in sweep line directly left to v_i
                edge_t_ edge(nodes(v_i, 0), nodes(v_i, 1), nodes(v_i, 0), nodes(v_i, 1));
                auto it = sweep_line.lower_bound(edge);
                // insert diagonal
                diagonals.push_back(poly.insert_edge(v, helper[it->id]));
                helper[it->id] = v;
                auto ref = sweep_line.emplace(
                  v_i, nodes(v_i, 0), nodes(v_i, 1), nodes(node_next(v_i), 0), nodes(node_next(v_i), 1));
                sweep_line_it[v_i] = ref.first;
                helper[v_i] = v;
                break;
            }
            case node_category_t::merge: {
                if (node_category[helper[node_prev(v_i)]->id()] == node_category_t::merge) {   // insert diagonal
                    diagonals.push_back(poly.insert_edge(v, helper[node_prev(v_i)]));
                }
                sweep_line.erase(sweep_line_it[node_prev(v_i)]);
                // search edge in sweep line directly left to v_i
                edge_t_ edge(nodes(v_i, 0), nodes(v_i, 1), nodes(v_i, 0), nodes(v_i, 1));
                auto it = sweep_line.lower_bound(edge);
                if (node_category[it->id] == node_category_t::merge) {
                    diagonals.push_back(poly.insert_edge(v, helper[it->id]));
                }
                helper[it->id] = v;
                break;
            }
            case node_category_t::regular: {
                if (!interior_on_left(nodes.row(v_i), nodes.row(node_next(v_i)))) {
                    if (node_category[helper[node_prev(v_i)]->id()] == node_category_t::merge) {   // insert diagonal
                        diagonals.push_back(poly.insert_edge(v, helper[node_prev(v_i)]));
                    }
                    sweep_line.erase(sweep_line_it[node_prev(v_i)]);
                    auto ref = sweep_line.emplace(
                      v_i, nodes(v_i, 0), nodes(v_i, 1), nodes(node_next(v_i), 0), nodes(node_next(v_i), 1));
                    sweep_line_it[v_i] = ref.first;
                    helper[v_i] = v;
                } else {
		    // search edge in sweep line directly left to v_i
                    edge_t_ edge(nodes(v_i, 0), nodes(v_i, 1), nodes(v_i, 0), nodes(v_i, 1));
                    auto it = sweep_line.lower_bound(edge);
                    if (node_category[helper[it->id]->id()] == node_category_t::merge) {
                        diagonals.push_back(poly.insert_edge(v, helper[it->id]));
                    }
                    helper[it->id] = v;
                }
                break;
            }
            }
        }
        // recover from the DCEL structure the node numbering of each monotone polygon
        std::vector<bool> visited(poly.n_halfedges(), false);
        std::vector<std::vector<int>> monotone_partition;
        auto collect_polygon_node_id_from = [&]<typename halfedge_t>(halfedge_t* chain) mutable {
            auto& it = monotone_partition.emplace_back();
            for (auto jt = poly.halfedge_begin(chain); jt != poly.halfedge_end(chain); ++jt) {
                int node_id = jt->node()->id();
                it.push_back(node_id);
                visited[jt->id()] = true;
            }
        };
        for (edge_ptr_t diagonal : diagonals) {
            // follow the chains from the halfedges composing this edge, if halfedge not already visited
            if (!visited[diagonal->first() ->id()]) { collect_polygon_node_id_from(diagonal->first ()); }
            if (!visited[diagonal->second()->id()]) { collect_polygon_node_id_from(diagonal->second()); }
        }
        return monotone_partition;
    }

    // triangulate monotone polygon (returns a RowMajor ordered matrix of cells).
    std::vector<int> triangulate_monotone_(const DMatrix<double>& nodes) {
        // every triangulation of a polygon of n points has n - 2 triangles (lemma 1.2.2 of (1))
        std::vector<int> cells;
	int n_nodes = nodes.rows();
        cells.reserve(3 * (n_nodes - 2));
        auto push_cell = [&](int i, int j, int k) {   // convinient lambda to add a triangle
            cells.push_back(i);
            cells.push_back(j);
            cells.push_back(k);
        };
        if (nodes.rows() == 3) {   // already a triangle, stop
            push_cell(0, 1, 2);
            return cells;
        }
        // O(n) minimum and maximum coordinate search
        int min = nodes(0, 1) < nodes(1, 1) ? 0 : 1, max = nodes(0, 1) < nodes(1, 1) ? 1 : 0;
        double y_min = nodes(min, 1), y_max = nodes(max, 1);
        for (int i = 1; i < n_nodes; ++i) {
            if (nodes(i, 1) < y_min) {
                y_min = nodes(i, 1);
                min = i;
            } else if (nodes(i, 1) > y_max) {
                y_max = nodes(i, 1);
                max = i;
            }
        }
        // build right and left chains (assume counterclockwise sorting)
	std::vector<int> node(n_nodes);
	std::unordered_set<int> r_chain, l_chain;
        {
            std::vector<int> r_chain_, l_chain_;
            for (int j = max; j != min; j = ((j + 1) % n_nodes + n_nodes) % n_nodes) { l_chain_.push_back(j); }
            l_chain_.push_back(min);
            for (int j = ((max - 1) % n_nodes + n_nodes) % n_nodes; j != min;
                 j = ((j - 1) % n_nodes + n_nodes) % n_nodes) {
                r_chain_.push_back(j);
            }
            // O(n) sorted list merge
            std::merge(
              l_chain_.begin(), l_chain_.end(), r_chain_.begin(), r_chain_.end(), node.begin(), [&](int a, int b) {
                  return nodes(a, 1) > nodes(b, 1) || (nodes(a, 1) == nodes(b, 1) && nodes(a, 0) > nodes(b, 0));
              });
            l_chain.insert(l_chain_.begin(), l_chain_.end());
            r_chain.insert(r_chain_.begin(), r_chain_.end());
        }
        auto is_l_chain = [&](int i) { return l_chain.contains(i); };
        auto is_r_chain = [&](int i) { return r_chain.contains(i); };
        auto are_in_opposite_chains = [&](int i, int j) -> bool {
            return (is_l_chain(i) && is_r_chain(j)) || (is_r_chain(i) && is_l_chain(j));
        };
        std::deque<int> reflex_chain;   // queue of reflex nodes
        reflex_chain.push_back(node[0]);
        reflex_chain.push_back(node[1]);
        int node_i, node_j, node_k;
        bool on_left = is_l_chain(node[1]);   // whether the currently pointed node is on the left or right chain
        // start triangulating
        for (std::size_t j = 2, n = node.size(); j < n; ++j) {
            node_i = *(reflex_chain.end() - 1);
            node_j = node[j];
	    
            if (are_in_opposite_chains(node_i, node_j)) {   // triangulate
                node_i = *reflex_chain.begin();
                on_left = !on_left;
                while (reflex_chain.size() > 1) {
                    reflex_chain.pop_front();
                    node_k = reflex_chain.front();
                    // add triangle
		    push_cell(node_i, node_j, node_k);
                    node_i = node_k;
                }
                reflex_chain.push_back(node_j);
            } else {   // check if the triplet (node_i, node_j, node_k) makes a reflex turn or not
                node_k = *(reflex_chain.end() - 2);
                double m_signed =
                  internals::signed_measure_2d_tri(nodes.row(node_j), nodes.row(node_k), nodes.row(node_i));
                if ((on_left && m_signed < 0) || (!on_left && m_signed > 0)) {    // reflex turn
                    reflex_chain.push_back(node_j);
                } else {
                    do {
                        // add triangle
                        push_cell(node_i, node_j, node_k);
                        // triangulate until convex turn is found
                        reflex_chain.pop_back();
                        if (reflex_chain.size() > 1) {
                            node_i = *(reflex_chain.end() - 1);
                            node_k = *(reflex_chain.end() - 2);
                            m_signed =
                              internals::signed_measure_2d_tri(nodes.row(node_j), nodes.row(node_k), nodes.row(node_i));
                        }
                    } while (reflex_chain.size() > 1 && ((on_left && m_signed > 0) || (!on_left && m_signed < 0)));
                    reflex_chain.push_back(node_j);
                }
            }
        }
        return cells;
    }
    // internal face-based storage
    Triangulation<LocalDim, EmbedDim> triangulation_;
};

}   // namespace fdapde

#endif // __POLYGON_H__
