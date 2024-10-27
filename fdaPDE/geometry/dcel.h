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

#ifndef __DCEL_H__
#define __DCEL_H__

namespace fdapde {

// implementation of the Double Connected Edge List data structure (also known as DCEL or half-edge)
template <int LocalDim, int EmbedDim> class DCEL {
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    // forward decl
    struct node_t;
    struct halfedge_t;
  
    // internal data structures
    struct node_t {
       private:
        using coords_t = Eigen::Matrix<double, embed_dim, 1>;
        int id_;                  // global node index
        halfedge_t* halfedge_;    // any edge having this node as its origin
        bool boundary_;           // asserted true if node is on boundary
        coords_t coords_;
       public:
        node_t() : coords_(), halfedge_(nullptr), boundary_(false) { }
        template <typename CoordsType>
            requires(fdapde::is_eigen_dense_v<CoordsType>)
        node_t(int id, halfedge_t* halfedge, bool boundary, const CoordsType& coords) :
            id_(id), halfedge_(halfedge), boundary_(boundary), coords_() {
            fdapde_assert(
              (coords.rows() == 1 && coords.cols() == embed_dim) || (coords.rows() == embed_dim && coords.cols() == 1));
            if (coords.rows() == 1) {
                coords_ = coords.transpose();
            } else {
                coords_ = coords;
            }
        }
        template <typename CoordsType>
            requires(fdapde::is_eigen_dense_v<CoordsType>)
        node_t(int id, bool boundary, const CoordsType& coords) : node_t(id, nullptr, boundary, coords) { }
        template <typename... CoordsType>
            requires(std::is_floating_point_v<CoordsType> && ...) && (sizeof...(CoordsType) == embed_dim)
        node_t(int id, halfedge_t* halfedge, bool boundary, CoordsType&&... coords) :
            id_(id), halfedge_(halfedge), boundary_(boundary), coords_(coords...) { }
        template <typename... CoordsType>
            requires(std::is_floating_point_v<CoordsType> && ...) && (sizeof...(CoordsType) == embed_dim)
        node_t(int id, bool boundary, CoordsType&&... coords) :
            node_t(id, nullptr, boundary, coords...) { }
        // observers
        const Eigen::Matrix<double, embed_dim, 1>& coords() const { return coords_; }
        halfedge_t* halfedge() const { return halfedge_; }
        void set_halfedge(halfedge_t* halfedge) { halfedge_ = halfedge; }
        int id() const { return id_; }
        bool on_boundary() const { return boundary_; }
    };

    struct halfedge_t {
       private:
        int id_;   // global halfedge index
        halfedge_t *prev_, *next_, *twin_;
        node_t* node_;
       public:
        halfedge_t() : node_(nullptr), prev_(nullptr), next_(nullptr), twin_(nullptr) { }
        halfedge_t(int id, halfedge_t* prev, halfedge_t* next, halfedge_t* twin, node_t* node) :
            id_(id), prev_(prev), next_(next), twin_(twin), node_(node) { }
        // no twin constructors
        halfedge_t(int id, halfedge_t* prev, halfedge_t* next, node_t* node) :
            halfedge_t(id, prev, next, nullptr, node) { }
        // minimal constructor
        halfedge_t(int id, node_t* node) : halfedge_t(id, nullptr, nullptr, nullptr, node) { }

        // observers
        halfedge_t* prev() const { return prev_; }
        halfedge_t* next() const { return next_; }
        halfedge_t* twin() const { return twin_; }
        node_t* node() const { return node_; }
        int id() const { return id_; }
        bool on_boundary() const { return node_->on_boundary() && twin_->node()->on_boundary(); }
        // modifiers
        void set_prev(halfedge_t* prev) { prev_ = prev; }
        void set_next(halfedge_t* next) { next_ = next; }
        void set_twin(halfedge_t* twin) { twin_ = twin; }

        // iterator (follows the chain of directed edges until no next valid edge or this edge is found)
        struct iterator {
            using value_type = halfedge_t;
            using pointer = std::add_pointer_t<value_type>;
            using reference = std::add_lvalue_reference_t<value_type>;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;

            iterator(halfedge_t* halfedge) :
                halfedge_(halfedge), end_(halfedge == nullptr ? nullptr : halfedge->prev()) {
            }
            iterator& operator++() {
                if (last_) { [[unlikely]]
                    end_ = nullptr;
                } else {
                    halfedge_ = halfedge_->next();
                    if (halfedge_ == end_) { last_ = true; }   // implement cyclic structure
                }
                return *this;
            }
            // access
            pointer operator->() { return halfedge_; }
            const pointer operator->() const { return halfedge_; }
            reference operator*() { return *halfedge_; }
            const reference operator*() const { return *halfedge_; }
            // comparison
            friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.end_ == rhs.end_; }
            friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.end_ != rhs.end_; }
           private:
            bool last_ = false;
            pointer halfedge_, end_;
        };
    };

    // an edge is an indexed pair of halfedges which are twins one another
    struct edge_t {
        edge_t() : h1_(nullptr), h2_(nullptr) { }
        edge_t(int id, halfedge_t* h1, halfedge_t* h2) : id_(id), h1_(h1), h2_(h2) {
            fdapde_assert(h1->twin() == h2 && h2->twin() == h1);
        }
        // observers
        int id() const { return id_; }
        halfedge_t* first() { return h1_; }
        halfedge_t* second() { return h2_; }
        bool on_boundary() const { return h1_->on_boundary(); }
        Eigen::Matrix<int, 2, 1> node_ids() const {
            Eigen::Matrix<int, 2, 1> ids;
            ids[0] = h1_->node()->id();
            ids[1] = h2_->node()->id();
            return ids;
        }
       private:
        int id_;
        halfedge_t *h1_, *h2_;
    };

    DCEL() : nodes_(), halfedges_(), n_nodes_(0), n_halfedges_(0), n_edges_(0) { }
    // constructs a closed loop structure linking nodes one after the other
    static DCEL<local_dim, embed_dim> make_polygon(const Eigen::Matrix<double, Dynamic, Dynamic>& nodes) {
        fdapde_assert(nodes.cols() == embed_dim);
        int n_nodes = nodes.rows();
        int n_halfedges = 2 * (n_nodes + 1);
        DCEL<local_dim, embed_dim> dcel;
        // push nodes
        for (int i = 0; i < n_nodes; ++i) {
            node_t* n = dcel.insert_node(node_t(i, /* boundary = */ true, nodes.row(i)));
            halfedge_t* h = dcel.emplace_halfedge_(n);
            n->set_halfedge(h);
        }
        // create twin edges
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
            node_t* n1 = std::addressof(*it);
            node_t* n2 = std::addressof((it->id() == n_nodes - 1) ? *dcel.nodes_begin() : *it);
            halfedge_t* h1 = n1->halfedge();
            halfedge_t* h2 = dcel.emplace_halfedge_(n2);   // push twin edge
            h2->set_twin(h1);
            h1->set_twin(h2);
	    dcel.insert_edge(edge_t(it->id(), h1, h2));
        }
        // finalize next-prev pointer pairs
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
            halfedge_t* h1 = it->halfedge();
            halfedge_t* h2 = ((it->id() == n_nodes - 1) ? dcel.nodes_begin() : it)->halfedge();
            h1->set_next(h2);
            h2->set_prev(h1);
            h1->twin()->set_prev(h2->twin());
            h2->twin()->set_next(h1->twin());
        }
        return dcel;
    }
    // modifiers
    node_t* insert_node(const node_t& node) {
        nodes_.push_back(node);
	n_nodes_++;
        return std::addressof(nodes_.back());
    }
    // insert edge between nodes with id {id1, id2} (assumes DCEL already in a valid state)
    edge_t* insert_edge(node_t* n1, node_t* n2) {
        // create a pair of twin half-edges
        halfedge_t* h1 = emplace_halfedge_(n1);
        halfedge_t* h2 = emplace_halfedge_(n2);
        h1->set_twin(h2);
        h2->set_twin(h1);
        // get exiting halfedges from n1 and n2
        halfedge_t* h3 = n1->halfedge();
        halfedge_t* h4 = n2->halfedge();
        // update next and prev pointers
        h1->set_next(h4);
        h2->set_next(h3);
	h1->set_prev(h3->twin()->next()->twin());
        h2->set_prev(h4->twin()->next()->twin());
	
	n_halfedges_ = n_halfedges_ + 2;
	edges_.push_back(edge_t(n_edges_++, h1, h2));
        return std::addressof(edges_.back());
    }
    edge_t* insert_edge(const edge_t& edge) {
        edges_.push_back(edge);
	n_edges_++;
        return std::addressof(edges_.back());
    }  
    // observers
    Eigen::Matrix<double, Dynamic, Dynamic> nodes() const {   // matrix of nodes coordinates
        Eigen::Matrix<double, Dynamic, Dynamic> coords(n_nodes_, embed_dim);
        for (int i = 0; i < n_nodes_; ++i) { coords.row(nodes[i].id()) = nodes[i].coords(); }
        return coords;
    }
    int n_nodes() const { return n_nodes_; }
    int n_halfedges() const { return n_halfedges_; }
    int n_edges() const { return n_edges_; }
    // cyclic iteration over half-edge chain
    typename halfedge_t::iterator halfedge_begin(halfedge_t* halfedge) {
        return typename halfedge_t::iterator(std::addressof(halfedges_[halfedge->id()]));
    }
    typename halfedge_t::iterator halfedge_end([[maybe_unused]] halfedge_t* halfedge) {
        return typename halfedge_t::iterator(nullptr);
    }
    // sequential iterator over all halfedges
    typename std::list<halfedge_t>::iterator halfedges_begin() { return halfedges_.begin(); }
    typename std::list<halfedge_t>::iterator halfedges_end() { return halfedges_.end(); }
    // sequential iterator over all nodes
    typename std::list<node_t>::iterator nodes_begin() { return nodes_.begin(); }
    typename std::list<node_t>::iterator nodes_end() { return nodes_.end(); }
   private:
    template <typename... Args> halfedge_t* emplace_halfedge_(Args&&... args) {
        halfedges_.emplace_back(n_halfedges_++, std::forward<Args>(args)...);
        return std::addressof(halfedges_.back());
    }
    // internal storage (use list to avoid reallocations)
    std::list<node_t> nodes_;
    std::list<halfedge_t> halfedges_;
    std::list<edge_t> edges_;
    int n_nodes_, n_halfedges_, n_edges_;
};
  
}   // namespace fdapde

#endif // __DCEL_H__
