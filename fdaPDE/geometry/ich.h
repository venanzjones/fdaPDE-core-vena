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


#ifndef __GEODESIC_SOVLER_H__
#define __GEODESIC_SOVLER_H__

#include <queue>
#include <set>
#include <map>
#include <vector>
#include <float.h>
#include <Eigen\core>
#include <limits>
#include <numbers>
#include "geodesic_utils.h"

typedef fdapde::core::Triangulation<2, 3> MeshType;
double INFINITY = std::numeric_limits<double>::max;
double PI_EXACT = std::numbers::pi;

namespace fdapde {
namespace core {

    struct node_t
{
    bool parent_is_a_pseudosource;
    char birth_time;
    int direct_parent_id;
    int root_node_of_direct_parent_id;
    long seq_tree_level;
    int ancestor_id;
    double updated_distance;
    double entry_proportion;
    // Constructor
    node_t()
    {
        parent_is_a_pseudosource = true;
        seq_tree_level = -1;
        birth_time = -1;
        direct_parent_id = -1;
        ancestor_id = -1;
        updated_distance = DBL_MAX;
    }
};

struct priority_node_t
{
    char birth_time;
    int node_id;
    double updated_distance;
    bool operator<(const priority_node_t& rhs) const
    {
        return updated_distance > rhs.updated_distance;
    }
    priority_node_t() {}
    priority_node_t(char birth_time, int node_id, double updated_distance)
    {
        this->birth_time = birth_time;
        this->node_id = node_id;
        this->updated_distance = updated_distance;
    }
};
struct edge_t
{
    char birth_time;
    double updated_distance;
    double entry_proportion;
    edge_t()
    {
        birth_time = -1;
        updated_distance = DBL_MAX;
    }
};

struct window_t
{
    bool is_on_left_subtree;
    bool brach_parent_is_pseudo_source;
    bool direct_parent_edge_on_left; //may removed
    bool direct_parent_is_pseudo_source; //may removed
    char parent_birth_time;
    int brach_parent_id;
    int root_node_id;
    int current_edge_id;
    long seq_tree_level; //may removed
    int ancestor_id;
    double distance_to_root;
    double proportions[2]; // [a,b]
    double parent_entry_proportion;
    Eigen::Vector2d ps_coordinates;
};

struct priority_window_t
{
    window_t* w_pointer;
    double updated_distance;
    bool operator<(const priority_window_t& rhs) const
    {
        return updated_distance > rhs.updated_distance;
    }
};

class GeodesicSolver
{
protected:
    std::vector<double> scalar_field_; // scalar field
    const MeshType& mesh_; // mesh object
    std::map<int, double> sources_; // sources, distances
    std::set<int> destinations_; // set of destinations
    std::priority_queue<priority_window_t> Q_; // window queue
	std::priority_queue<priority_node_t> P_; // pseudosources queue
    std::vector<edge_t> edge_aux_; // auxiliary structure for the egdes
    std::vector<node_t> node_aux_; // auxiliary structure for the nodes
    std::unordered_map<int,std::vector<int>> nodes_adjacencies_; // adjacencies, contains vector of one-ring

protected:  
    void init(); // commented
    void dispose(); // commented
    void propagate(); 
    void add_to_windows_queue(priority_window_t& w_quote); // commented
    bool is_too_small(const window_t& w) const; // commented
    void compute_children_of_pseudosource(int parent_node_id); // commented
    void compute_children_of_pseudosource_from_pseudosource(int parent_node_id); // commented
    void compute_children_of_pseudosource_from_window(int parent_node_id); // commented
    void create_interval_child_of_pseudosource(int src, int incident_edge_subindex, double prop_left = 0, double prop_right = 1);
    void extend_pseudosource(int src, int neigh_id);
    void compute_sources_children();
    void compute_source_children(int src_node_id, double dis);
    void compute_left_child(const window_t& w);
    void compute_only_left_trimmed_child(const window_t& w);
    void compute_left_trimmed_child_with_parent(const window_t& w);
    void compute_right_child(const window_t& w);
    void compute_only_right_trimmed_child(const window_t& w);
    void compute_right_trimmed_child_with_parent(const window_t& w);
    void compute_window_children(priority_window_t& parent_w_quote);
    bool check_window(window_t& w) const;
    bool update_tree_depth_with_choice();
    double min_distance_of_windows(const window_t& w) const;

public:
    // Constructors
    // init) sources distances are automatically set to 0 in the constructors
    GeodesicSolver(const MeshType& mesh, int src) : mesh_(mesh){sources_[src] = 0;}
    GeodesicSolver(const MeshType& mesh, const std::map<int, double>& sources) : mesh_(mesh), sources_(sources){}
    GeodesicSolver(const MeshType& mesh, const std::map<int, double>& sources, const std::set<int> &destinations) : mesh_(mesh), sources_(sources), destinations_(destinations){}
    GeodesicSolver(const MeshType& mesh, const std::set<int>& sources) : mesh_(mesh)
    {
	for (auto it = sources.begin(); it != sources.end(); ++it)
		sources_[*it] = 0;
    }
    GeodesicSolver(const MeshType& mesh, const std::set<int>& sources, const std::set<int>& destinations) : mesh_(mesh), destinations_(destinations)
    {
	for (auto it = sources.begin(); it != sources.end(); ++it)
		sources_[*it] = 0;
    }
    // Public methods
    void run();
    const std::vector<double>& get_distance_field() const;
};

// Method to be called to run the algorithm
void GeodesicSolver::run()
{
	init();
	propagate();
	dispose();
}

// Initialize the variables needed for the algorithm
void GeodesicSolver::init() // O(n log n) for the adjacencies
{
        // Init the scalar_field distances to infinity  
    	scalar_field_.resize(mesh_.n_nodes(), DBL_MAX);
    	node_aux_.resize(mesh_.n_nodes());
        nodes_adjacency_.resize(mesh_.n_nodes());
        // Initialize the adjacencies 
        for(size_t i = 0; i<mesh_.n_nodes(); ++i ){
            nodes_adjacencies_[i] = mesh_.node_one_ring(i);
        }
            
}

// Deletes what is not neeeded after the algorithm execution
void GeodesicSolver::dispose()
{
    while (!Q_.empty())
    {
        delete Q_.top().w_pointer;
        Q_.pop();
    }
    P_ = priority_queue<priority_node_t>();
}

// Main method
void GeodesicSolver::propagate()
{
    // Used to track when all desired destination nodes have been reached during the propagation
    std::set<int> tmp_destinations(destinations_);
    // init) create a pseudo-source window w for s
    for (std::map<int, double>::const_iterator it = sources_.begin();
        it != sources_.end(); ++it){
            compute_source_children(it->first, it->second);}

    // Decide where to take the next window/ps
    bool from_ps_queue = update_tree_depth_with_choice();

    // 1) Main loop:
    while (!P_.empty() || !Q_.empty())
    {
        // 4)
        // If we take from ps queue:
        if (from_ps_queue)
        {
            int node_id = P_.top().node_id;
            P_.pop();
            tmp_destinations.erase(node_id);
            if (!destinations_.empty() && tmp_destinations.empty())
                return;
            propagate_ps(node_id);
        }
        else // We are in window queue:
        {
            priority_window_t w_quote = Q_.top();
            Q_.pop();
            compute_window_children(w_quote);
            delete w_quote.w_pointer;
        }
        from_ps_queue = update_tree_depth_with_choice();
    }
}

// Given a src and a distance, extends the information related to 
// its one-ring, then, creates child intervals
void GeodesicSolver::compute_source_children(int src_node_id, double dis)
{
    // Update the aux struct 
    ++node_aux_[src_node_id].birth_time;
    node_aux_[src_node_id].seq_tree_level = 0;
    node_aux_[src_node_id].ancestor_id = src_node_id;
    node_aux_[src_node_id].updated_distance = dis;
    // Get the number of neighbours
    int n_neigh = this->nodes_adjacencies_[src_node_id].size();
    // Extend the pseudosource to each of the neghbouring nodes
    for (int i = 0; i < n_neigh; ++i)
    {
        extend_pseudosource(src_node_id, i);
    }
    // Compute the children
    for (int i = 0; i < n_neigh; ++i)
    {
        create_interval_child_of_pseudosource(src_node_id, i);
    }
}
// Extends the pseudo-source influence to a neighboring node
void GeodesicSolver::extend_pseudosource(int src, int neigh_id)
{
    // FIX: voglio il lato che connette src e neigh_id
    int edge_id = this->nodes_adjacencies_[src][neigh_id];
    const EdgeType& edge = mesh_.edges().row(edge_id);
    // FIX: quale dei due è right_node_id?
    int index = edge.right_node_id;
    // La nuova candidata distanza è quella attuale + misura edge
    double dis = node_aux_[src].updated_distance + edge.measure();
    
    if (dis >= node_aux_[index].updated_distance - EPSILON)
        return;
    // Se è minore di quella che arriva a index:
    // Update the aux struct 
    node_aux_[index].parent_is_a_pseudosource = true;
    ++node_aux_[index].birth_time;
    node_aux_[index].direct_parent_id = src;
    node_aux_[index].seq_tree_level = node_aux_[src].seq_tree_level + 1;
    node_aux_[index].ancestor_id = node_aux_[src].ancestor_id;
    node_aux_[index].updated_distance = dis;
    // Se il nodo non è strettamente convesso, lo pusho su P
    if (!mesh_.is_node_strongly_convex(index))
        P_.push(priority_node_t(node_aux_[index].birth_time,
            index, dis));
}

// Creates an interval window [a, b] on an incident edge to a pseudo-source node, 
// representing part of the wavefront propagation through that edge.
// Adds the resulting window to the queue if valid.
void create_interval_child_of_pseudosource(int src, int incident_edge_subindex, double prop_left= 0, double prop_right = 1)
{
    int incident_edge_id = this->nodes_adjacencies_[src][incident_edge_subindex];
    if (mesh_.is_edge_on_boundary(incident_edge_id))
        return;
    const EdgeType& edge = mesh_.edges().row(incident_edge_id);
    // FIX: right edge id?
    const int edge_id = edge.right_edge_id;
    // Check whether the edge is a boundary edge, in that case stop propagation
    if (mesh_.is_edge_on_boundary(edge_id))
        return;
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    w_quote.w_pointer->proportions[0] = prop_left;
    w_quote.w_pointer->proportions[1] = prop_right;
    // If the window became too small:
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    // Update window information
    w_quote.w_pointer->brach_parent_is_pseudo_source = true;
    w_quote.w_pointer->direct_parent_is_pseudo_source = true;
    w_quote.w_pointer->parent_birth_time = node_aux_[src].birth_time;
    w_quote.w_pointer->brach_parent_id = src;
    w_quote.w_pointer->root_node_id = src;
    w_quote.w_pointer->current_edge_id = edge_id;
    w_quote.w_pointer->seq_tree_level = node_aux_[src].seq_tree_level + 1;
    w_quote.w_pointer->ancestor_id = node_aux_[src].ancestor_id;
    w_quote.w_pointer->distance_to_root = node_aux_[src].updated_distance;
    w_quote.w_pointer->parent_entry_proportion;
    //TODO: reverse edge id e Edge -> edges() + altra roba sotto
    int reverse_edge = mesh_.edges().row(edge_id).reverse_edge_id;
    w_quote.w_pointer->ps_coordinates = reposotion_wrt_edge(reverse_edge,
        mesh_.edges().row(reverse_edge).coordOfOppositeVert,mesh_);
    add_to_windows_queue(w_quote);
}

// The following methods are to compute windows children:

// Computes and creates the window that results from continuing propagation to the “left child” edge 
// of the current triangle. Adds this new window to the queue if valid
void compute_left_child(const window_t& w)
{
    //  FIX: questo if va rivisto in luce del paper
    if (is_edge_on_boundary(mesh_.is_edge_on_boundary(mesh_.Edge(w.current_edge_id).indexOfLeftEdge)))
        return;
	
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    w_quote.w_pointer->proportions[0] = mesh_.ProportionOnLeftEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[0]);
    w_quote.w_pointer->proportions[0] = max(0.0, w_quote.w_pointer->proportions[0]);
    w_quote.w_pointer->proportions[1] = mesh_.ProportionOnLeftEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[1]);
    w_quote.w_pointer->proportions[1] = min(1.0, w_quote.w_pointer->proportions[1]);
    // If too small
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.w_pointer->brach_parent_is_pseudo_source = w.brach_parent_is_pseudo_source;
    w_quote.w_pointer->direct_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_edge_on_left = true;
    w_quote.w_pointer->current_edge_id = mesh_.Edge(w.current_edge_id).indexOfLeftEdge;
    w_quote.w_pointer->distance_to_root = w.distance_to_root;
    // Rotation due to unfolding
    w_quote.w_pointer->ps_coordinates = mesh_.rotate_around_left_child_edge(w.current_edge_id, w.ps_coordinates);
    w_quote.w_pointer->is_on_left_subtree = w.is_on_left_subtree;
    w_quote.w_pointer->seq_tree_level = w.seq_tree_level + 1;
    w_quote.w_pointer->ancestor_id = w.ancestor_id;
    w_quote.w_pointer->parent_entry_proportion = w.parent_entry_proportion;
    w_quote.w_pointer->parent_birth_time = w.parent_birth_time;
    w_quote.w_pointer->brach_parent_id = w.brach_parent_id;
    w_quote.w_pointer->root_node_id = w.root_node_id;
    add_to_windows_queue(w_quote);
}
// Same as above but for right child
void compute_right_child(const window_t& w)
{
    if (mesh_.is_edge_on_boundary(mesh_.Edge(w.current_edge_id).right_edge_id))
        return;
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    w_quote.w_pointer->proportions[0] = mesh_.ProportionOnRightEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[0]);
    w_quote.w_pointer->proportions[0] = max(0., w_quote.w_pointer->proportions[0]);
    w_quote.w_pointer->proportions[1] = mesh_.ProportionOnRightEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[1]);
    w_quote.w_pointer->proportions[1] = min(1., w_quote.w_pointer->proportions[1]);
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.w_pointer->brach_parent_is_pseudo_source = w.brach_parent_is_pseudo_source;
    w_quote.w_pointer->direct_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_edge_on_left = false;
    w_quote.w_pointer->current_edge_id = mesh_.Edge(w.current_edge_id).right_edge_id;
    w_quote.w_pointer->distance_to_root = w.distance_to_root;
    w_quote.w_pointer->ps_coordinates = rotate_around_right_child_edge(w.current_edge_id, w.ps_coordinates, mesh_);
    w_quote.w_pointer->seq_tree_level = w.seq_tree_level + 1;
    w_quote.w_pointer->ancestor_id = w.ancestor_id;
    w_quote.w_pointer->parent_birth_time = w.parent_birth_time;
    w_quote.w_pointer->brach_parent_id = w.brach_parent_id;
    w_quote.w_pointer->root_node_id = w.root_node_id;
    w_quote.w_pointer->is_on_left_subtree = w.is_on_left_subtree;
    w_quote.w_pointer->parent_entry_proportion = w.parent_entry_proportion;
    add_to_windows_queue(w_quote);
}


// Similar to compute_left_child but for the case where only a portion (trimmed interval) of the left edge is considered. 
// This happens when geometric conditions prevent the full interval from being used.
void compute_only_left_trimmed_child(const window_t& w)
{
    if (mesh_.is_edge_on_boundary(mesh_.Edge(w.current_edge_id).indexOfLeftEdge))
        return;
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    w_quote.w_pointer->proportions[0] = mesh_.ProportionOnLeftEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[0]);
    w_quote.w_pointer->proportions[0] = std::max(0., w_quote.w_pointer->proportions[0]);
    w_quote.w_pointer->proportions[1] = 1;
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.w_pointer->brach_parent_is_pseudo_source = w.brach_parent_is_pseudo_source;
    w_quote.w_pointer->direct_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_edge_on_left = true;
    w_quote.w_pointer->current_edge_id = mesh_.Edge(w.current_edge_id).indexOfLeftEdge;
    w_quote.w_pointer->distance_to_root = w.distance_to_root;
    w_quote.w_pointer->ps_coordinates = mesh_.rotate_around_left_child_edge(w.current_edge_id, w.ps_coordinates);

    w_quote.w_pointer->seq_tree_level = w.seq_tree_level + 1;
    w_quote.w_pointer->ancestor_id = w.ancestor_id;
    w_quote.w_pointer->parent_birth_time = w.parent_birth_time;
    w_quote.w_pointer->brach_parent_id = w.brach_parent_id;
    w_quote.w_pointer->root_node_id = w.root_node_id;
    w_quote.w_pointer->is_on_left_subtree = w.is_on_left_subtree;
    w_quote.w_pointer->parent_entry_proportion = w.parent_entry_proportion;

    add_to_windows_queue(w_quote);
} 


// Like above
void compute_only_right_trimmed_child(const window_t& w)
{
    if (mesh_.is_edge_on_boundary(mesh_.Edge(w.current_edge_id).right_edge_id))
        return;
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    w_quote.w_pointer->proportions[0] = 0;
    w_quote.w_pointer->proportions[1] = mesh_.ProportionOnRightEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[1]);
    w_quote.w_pointer->proportions[1] = std::min(1., w_quote.w_pointer->proportions[1]);
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.w_pointer->brach_parent_is_pseudo_source = w.brach_parent_is_pseudo_source;
    w_quote.w_pointer->direct_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_edge_on_left = false;
    w_quote.w_pointer->current_edge_id = mesh_.Edge(w.current_edge_id).right_edge_id;
    w_quote.w_pointer->distance_to_root = w.distance_to_root;
    w_quote.w_pointer->ps_coordinates = rotate_around_right_child_edge(w.current_edge_id, w.ps_coordinates,mesh_);

    w_quote.w_pointer->seq_tree_level = w.seq_tree_level + 1;
    w_quote.w_pointer->ancestor_id = w.ancestor_id;
    w_quote.w_pointer->parent_birth_time = w.parent_birth_time;
    w_quote.w_pointer->brach_parent_id = w.brach_parent_id;
    w_quote.w_pointer->root_node_id = w.root_node_id;
    w_quote.w_pointer->is_on_left_subtree = w.is_on_left_subtree;
    w_quote.w_pointer->parent_entry_proportion = w.parent_entry_proportion;

    add_to_windows_queue(w_quote);
}

// Creates a left child interval window considering the presence of a parent window’s influence
void compute_left_trimmed_child_with_parent(const window_t& w)
{
    if (mesh_.is_edge_on_boundary(mesh_.Edge(w.current_edge_id).indexOfLeftEdge))
        return;
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    w_quote.w_pointer->proportions[0] = mesh_.ProportionOnLeftEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[0]);
    w_quote.w_pointer->proportions[0] = std::max(0., w_quote.w_pointer->proportions[0]);
    w_quote.w_pointer->proportions[1] = 1;
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.w_pointer->brach_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_edge_on_left = true;
    w_quote.w_pointer->current_edge_id = mesh_.Edge(w.current_edge_id).indexOfLeftEdge;
    w_quote.w_pointer->distance_to_root = w.distance_to_root;
    w_quote.w_pointer->ps_coordinates = mesh_.rotate_around_left_child_edge(w.current_edge_id, w.ps_coordinates);

    w_quote.w_pointer->seq_tree_level = w.seq_tree_level + 1;
    w_quote.w_pointer->ancestor_id = w.ancestor_id;
    w_quote.w_pointer->parent_birth_time = edge_aux_[w.current_edge_id].birth_time;
    w_quote.w_pointer->brach_parent_id = w.current_edge_id;
    w_quote.w_pointer->root_node_id = w.root_node_id;
    w_quote.w_pointer->is_on_left_subtree = true;
    w_quote.w_pointer->parent_entry_proportion = edge_aux_[w.current_edge_id].entry_proportion;

    add_to_windows_queue(w_quote);
}

// Same as above
void compute_right_trimmed_child_with_parent(const window_t& w)
{
    if (mesh_.is_edge_on_boundary(mesh_.Edge(w.current_edge_id).right_edge_id))
        return;
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    w_quote.w_pointer->proportions[0] = 0;
    w_quote.w_pointer->proportions[1] = mesh_.ProportionOnRightEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[1]);
    w_quote.w_pointer->proportions[1] = std::min(1., w_quote.w_pointer->proportions[1]);
    w_quote.w_pointer->proportions[1] = std::max(w_quote.w_pointer->proportions[1], w_quote.w_pointer->proportions[0]);
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.w_pointer->brach_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_edge_on_left = false;
    w_quote.w_pointer->current_edge_id = mesh_.Edge(w.current_edge_id).right_edge_id;
    w_quote.w_pointer->distance_to_root = w.distance_to_root;
    w_quote.w_pointer->ps_coordinates = rotate_around_right_child_edge(w.current_edge_id, w.ps_coordinates,mesh_);

    w_quote.w_pointer->is_on_left_subtree = false;
    w_quote.w_pointer->parent_birth_time = edge_aux_[w.current_edge_id].birth_time;
    w_quote.w_pointer->brach_parent_id = w.current_edge_id;
    w_quote.w_pointer->root_node_id = w.root_node_id;
    w_quote.w_pointer->seq_tree_level = w.seq_tree_level + 1;
    w_quote.w_pointer->ancestor_id = w.ancestor_id;
    w_quote.w_pointer->parent_entry_proportion = edge_aux_[w.current_edge_id].entry_proportion;

    add_to_windows_queue(w_quote);
}

// Given a parent window, determines the appropriate child windows (left, right, trimmed, etc.) and creates them.
// Incapsulate the above logic in a unique method
void compute_window_children(QuoteWindow& quoteParentWindow)
	{
		const Window& w = *quoteParentWindow.pWindow;
		const auto& edge = mesh_.Edge(w.indexOfCurEdge);
		double entryProp = mesh_.ProportionOnEdgeByImage(w.indexOfCurEdge, w.coordOfPseudoSource);

		if (entryProp >= w.proportions[1]
			|| entryProp >= 1 - LENGTH_TOL)
		{
			compute_left_child(w);
			return;
		}

		if (entryProp <= w.proportions[0]
			|| entryProp <= LENGTH_TOL)
		{
			compute_right_child(w);
			return;
		}
		double disToAngle = mesh_.DistanceToOppositeAngle(w.indexOfCurEdge, w.coordOfPseudoSource);
		int incidentVertex = edge.indexOfOppositeVert;
		bool fLeftChildToCompute(false), fRightChildToCompute(false);
		bool fWIsWinning(false);
		double totalDis = w.disToRoot + disToAngle;

		if (m_InfoAtAngles[w.indexOfCurEdge].birthTime == -1)
		{
			fLeftChildToCompute = fRightChildToCompute = true;
			fWIsWinning = true;
		}
		else
		{
			if (totalDis < m_InfoAtAngles[w.indexOfCurEdge].disUptodate
				- 2 * LENGTH_TOL)
			{
				fLeftChildToCompute = fRightChildToCompute = true;
				fWIsWinning = true;
			}
			else if (totalDis < m_InfoAtAngles[w.indexOfCurEdge].disUptodate
				+ 2 * LENGTH_TOL)
			{
				fLeftChildToCompute = fRightChildToCompute = true;
				fWIsWinning = false;
			}
			else
			{
				fLeftChildToCompute = entryProp < m_InfoAtAngles[w.indexOfCurEdge].entryProp;
				fRightChildToCompute = !fLeftChildToCompute;
				fWIsWinning = false;
			}

		}
		if (!fWIsWinning)
		{
			if (fLeftChildToCompute)
			{
				compute_only_left_trimmed_child(w);
			}
			if (fRightChildToCompute)
			{
				compute_only_right_trimmed_child(w);
			}
			return;
		}

		m_InfoAtAngles[w.indexOfCurEdge].disUptodate = totalDis;
		m_InfoAtAngles[w.indexOfCurEdge].entryProp = entryProp;
		++m_InfoAtAngles[w.indexOfCurEdge].birthTime;

		compute_left_trimmed_child_with_parent(w);
		compute_right_trimmed_child_with_parent(w);
		if (totalDis < m_InfoAtVertices[incidentVertex].disUptodate - LENGTH_TOL)
		{
			m_InfoAtVertices[incidentVertex].fParentIsPseudoSource = false;
			++m_InfoAtVertices[incidentVertex].birthTimeForCheckingValidity;
			m_InfoAtVertices[incidentVertex].indexOfDirectParent = w.indexOfCurEdge;
			m_InfoAtVertices[incidentVertex].indexOfRootVertOfDirectParent = w.indexOfRootVertex;
			m_InfoAtVertices[incidentVertex].levelOnSequenceTree = w.levelOnSequenceTree + 1;
			m_InfoAtVertices[incidentVertex].indexOfAncestor = w.indexOfAncestor;
			m_InfoAtVertices[incidentVertex].disUptodate = totalDis;
			m_InfoAtVertices[incidentVertex].entryProp = entryProp;

			if (!mesh_.is_node_strongly_convex(incidentVertex))
				add_to_windows_queue(QuoteInfoAtVertex(m_InfoAtVertices[incidentVertex].birthTimeForCheckingValidity,
					incidentVertex, totalDis));
		}
	}

// Computes the minimal distance from the pseudo-source associated with a window to any point in that window’s interval.
double min_distance_of_windows(const window_t& w) const
	{
        // Gets the length of the considered edge and the lengtgh of the projection of the pseudo source on it
        double edge_length = mesh_.edges().row(w.current_edge_id).measure();
		double projection_proportion = w.ps_coordinates(0) / edge_length;

        // If the projection is outside the edge, the minimum distance is the distance from the pseudo source to the root
		if (projection_proportion <= w.proportions[0])
		{
			double dx = w.ps_coordinates(0) - w.proportions[0] * edge_length;
			return w.distance_to_root + sqrt(dx * dx + w.ps_coordinates(1) * w.ps_coordinates(1));
		} 
		if (projection_proportion >= w.proportions[1])
		{
			double dx = w.ps_coordinates(0) - w.proportions[1] * edge_length;
			return w.distance_to_root + sqrt(dx * dx + w.ps_coordinates(1) * w.ps_coordinates(1));
		}
        // Otherwise, the minimum distance is just the one from the pseudo source
		return w.distance_to_root - w.ps_coordinates(1);
	}

// Check if a window is too small to be propagated ( b-a < eps )
bool is_too_small(const window_t& w) const
{
    return w.proportions[1] - w.proportions[0] < EPSILON;
}

// Decides which priority queue (pseudosources or windows) should be processed next 
// based on their sequence tree levels and birth times. 
// Helps maintain a consistent global ordering so that nodes and windows are expanded in the correct order.

bool update_tree_depth_with_choice()
{
    while (!P_.empty()
    && P_.front().birth_time != node_aux_[P_.front().node_id].birth_time)
    P_.pop();

    while (!m_QueueForWindows.empty())
    {
        const QuoteWindow& quoteW = m_QueueForWindows.top();

        if (quoteW.pWindow->fBrachParentIsPseudoSource)
        {
            if (quoteW.pWindow->birthTimeOfParent !=
                m_InfoAtVertices[quoteW.pWindow->indexOfBrachParent].birthTimeForCheckingValidity)
            {
                delete quoteW.pWindow;
                m_QueueForWindows.pop();
            }
            else
                break;
        }
        else
        {
            if (quoteW.pWindow->birthTimeOfParent ==
                m_InfoAtAngles[quoteW.pWindow->indexOfBrachParent].birthTime)
                break;
            else if (quoteW.pWindow->fIsOnLeftSubtree ==
                (quoteW.pWindow->entryPropOfParent < m_InfoAtAngles[quoteW.pWindow->indexOfBrachParent].entryProp))
                break;
            else
            {
                delete quoteW.pWindow;
                m_QueueForWindows.pop();
            }
        }
    }

    bool from_ps_q(false);
    if (m_QueueForWindows.empty())
    {
        if (!m_QueueForPseudoSources.empty())
        {
            const InfoAtVertex& infoOfHeadElemOfPseudoSources = m_InfoAtVertices[m_QueueForPseudoSources.top().indexOfVert];
            m_depthOfResultingTree = max(m_depthOfResultingTree,
                infoOfHeadElemOfPseudoSources.levelOnSequenceTree);
            from_ps_q = true;
        }
    }
    else
    {
        if (m_QueueForPseudoSources.empty())
        {
            const Window& infoOfHeadElemOfWindows = *m_QueueForWindows.top().pWindow;
            m_depthOfResultingTree = max(m_depthOfResultingTree,
                infoOfHeadElemOfWindows.levelOnSequenceTree);
            from_ps_q = false;
        }
        else
        {
            const QuoteInfoAtVertex& headElemOfPseudoSources = m_QueueForPseudoSources.top();
            const QuoteWindow& headElemOfWindows = m_QueueForWindows.top();
            if (headElemOfPseudoSources.disUptodate <=
                headElemOfWindows.disUptodate)
            {
                m_depthOfResultingTree = max(m_depthOfResultingTree,
                    m_InfoAtVertices[headElemOfPseudoSources.indexOfVert].levelOnSequenceTree);
                from_ps_q = true;
            }
            else
            {
                m_depthOfResultingTree = max(m_depthOfResultingTree,
                    headElemOfWindows.pWindow->levelOnSequenceTree);
                from_ps_q = false;
            }
        }
    }
    return from_ps_q;
}





// Verifies if a given window is still valid and can contribute to a shorter geodesic path
bool check_window(window_t& w) const
{
    if (w.direct_parent_is_pseudo_source)
        return true;
    const double mesh_range = (mesh_.range.row(1)-mesh_.range.row(1)).squadredNorm;
    const EdgeType& edge = mesh_.edges().row(w.current_edge_id); 
    int leftVert = edge.indexOfLeftVert;
    double dx = w.ps_coordinates(0) - w.proportions[1] * edge.measure();
    double rightLen = sqrt(dx * dx + w.ps_coordinates(1) * w.ps_coordinates(1));
    if (node_aux_[leftVert].updated_distance < 10000 * mesh_range && node_aux_[leftVert].updated_distance + w.proportions[1] * edge.measure()
        < w.distance_to_root + rightLen)
    {
        return false;
    }
    int rightVert = edge.right_node_id;

   dx = w.ps_coordinates(0) - w.proportions[0] * edge.measure();
    double leftLen = sqrt(dx * dx + w.ps_coordinates(1) * w.ps_coordinates(1));
    if (node_aux_[rightVert].updated_distance < 10000 * mesh_range && node_aux_[rightVert].updated_distance + (1 - w.proportions[0]) * edge.measure()
        < w.distance_to_root + leftLen)
    {
        return false;
    }
    const auto& oppositeEdge = mesh_.Edge(edge.reverse_edge_id);
    double xOfVert = edge.measure() - oppositeEdge.coordOfOppositeVert(0);
    double yOfVert = -oppositeEdge.coordOfOppositeVert(1);
    if (node_aux_[oppositeEdge.indexOfOppositeVert].updated_distance < 10000 * mesh_range)
    {
        if (w.direct_parent_edge_on_left)
        {
            double deta = w.distance_to_root + leftLen - node_aux_[oppositeEdge.indexOfOppositeVert].updated_distance;
            if (deta <= 0)
                return true;
            dx = xOfVert - w.proportions[0] * edge.measure();
            if (dx * dx + yOfVert * yOfVert < deta * deta)
                return false;
        }
        else
        {
            double deta = w.distance_to_root + rightLen - node_aux_[oppositeEdge.indexOfOppositeVert].updated_distance;
            if (deta <= 0)
                return true;
            dx = xOfVert - w.proportions[1] * edge.measure();
            if (dx * dx + yOfVert * yOfVert < deta * deta)
                return false;
        }
    }
    return true;
}

// Compute the ps children based on the parent: ps or window?
void propagate_ps(int parent_node_id)
{
    if (node_aux_[parent_node_id].parent_is_a_pseudosource)
        propagate_ps_from_ps(parent_node_id);
    else
        propagate_ps_from_window(parent_node_id);
}

// Compute the ps children when a ps parent is again a ps
void propagate_ps_from_ps(int parent_node_id)
{
    int degree = (int)mesh_.Neigh(parent_node_id).size();
    const std::vector<std::pair<int, double> >& neighs = mesh_.Neigh(parent_node_id);
    int indexOfParentOfParent = node_aux_[parent_node_id].direct_parent_id;
    int subIndex = mesh_.GetSubindexToVert(parent_node_id, indexOfParentOfParent);
    double angleSumPlus(0);
    int indexPlus;
    for (indexPlus = subIndex; indexPlus != (subIndex - 1 + degree) % degree; indexPlus = (indexPlus + 1) % degree)
    {
        angleSumPlus += neighs[indexPlus].second;
        if (angleSumPlus > EXACT_PI - 3 * AngleTolerance)
            break;
    }
    double angleSumMinus = 0;
    int indexMinus;
    for (indexMinus = (subIndex - 1 + degree) % degree;
        indexMinus == (subIndex - 1 + degree) % degree || indexMinus != (indexPlus - 1 + degree) % degree;
        indexMinus = (indexMinus - 1 + degree) % degree)
    {
        angleSumMinus += neighs[indexMinus].second;
        if (angleSumMinus > EXACT_PI - 3 * AngleTolerance)
            break;
    }
    if (indexMinus == (indexPlus - 1 + degree) % degree)
        return;
    //vertices;
    for (int i = (indexPlus + 1) % degree; i != (indexMinus + 1) % degree; i = (i + 1) % degree)
    {
        extend_pseudosource(parent_node_id, i);
    }

    //windows
    double propPlus = 0;
    double propMinus = 1;

    double devirationAngle = 20. * EXACT_PI / 180.0;
    double anglePlus = mesh_.Neigh(parent_node_id)[indexPlus].second - (angleSumPlus - EXACT_PI);

    if (mesh_.Edge(mesh_.Neigh(parent_node_id)[indexPlus].first).indexOfOppositeVert != -1)
    {
        if (mesh_.Neigh(parent_node_id)[indexPlus].second
            < 2 * devirationAngle)
        {
            propPlus = 0;
        }
        else if (angleSumPlus - EXACT_PI < devirationAngle
            || abs(anglePlus - mesh_.Neigh(parent_node_id)[indexPlus].second)
            < devirationAngle
            || mesh_.Neigh(parent_node_id)[indexPlus].second
            > EXACT_PI - devirationAngle)
        {
            propPlus = (mesh_.Neigh(parent_node_id)[indexPlus].second - (angleSumPlus - EXACT_PI))
                / mesh_.Neigh(parent_node_id)[indexPlus].second
                - devirationAngle / mesh_.Neigh(parent_node_id)[indexPlus].second;
            if (propPlus >= 1)
            {
                std::cerr << "propPlus " << __LINE__ << ": " << propPlus << std::endl;
            }
        }
        else
        {
            double angleTmp =
                mesh_.Edge(mesh_.Edge(mesh_.Edge(mesh_.Neigh(parent_node_id)[indexPlus].first).indexOfLeftEdge).reverse_edge_id).angleOpposite;
            propPlus = sin(anglePlus) / sin(anglePlus + angleTmp) * mesh_.Edge(mesh_.Neigh(parent_node_id)[indexPlus].first).measure() / mesh_.Edge(mesh_.Edge(mesh_.Neigh(parent_node_id)[indexPlus].first).right_edge_id).measure()
                - devirationAngle / mesh_.Neigh(parent_node_id)[indexPlus].second;
            if (propPlus >= 1)
            {
                std::cerr << "propPlus " << __LINE__ << ": " << propPlus << std::endl;
            }
        }
        propPlus = std::max(0., propPlus);
        propPlus = std::min(1., propPlus);
    }

    double angleMinus = angleSumMinus - EXACT_PI;
    if (mesh_.Edge(mesh_.Neigh(parent_node_id)[indexMinus].first).indexOfOppositeVert != -1)
    {
        if (mesh_.Neigh(parent_node_id)[indexMinus].second
            < 2 * devirationAngle)
        {
            propMinus = 1;
        }
        else if (angleSumMinus - EXACT_PI < devirationAngle
            || abs(angleSumMinus - EXACT_PI - mesh_.Neigh(parent_node_id)[indexMinus].second)
            < devirationAngle
            || mesh_.Neigh(parent_node_id)[indexMinus].second
            > EXACT_PI - devirationAngle)
        {
            propMinus = (angleSumMinus - EXACT_PI) / mesh_.Neigh(parent_node_id)[indexMinus].second
                + devirationAngle / mesh_.Neigh(parent_node_id)[indexMinus].second;
            if (propMinus <= 0)
            {
                std::cerr << "propMinus " << __LINE__ << ": " << propMinus << std::endl;
            }
        }
        else
        {
            double angleTmp =
                mesh_.Edge(mesh_.Edge(mesh_.Edge(mesh_.Neigh(parent_node_id)[indexMinus].first).indexOfLeftEdge).reverse_edge_id).angleOpposite;
            propMinus = sin(angleMinus) / sin(angleMinus + angleTmp) * mesh_.Edge(mesh_.Neigh(parent_node_id)[indexMinus].first).measure() / mesh_.Edge(mesh_.Edge(mesh_.Neigh(parent_node_id)[indexMinus].first).right_edge_id).measure()
                + devirationAngle / mesh_.Neigh(parent_node_id)[indexMinus].second;
            if (propMinus <= 0)
            {
                std::cerr << "propMinus " << __LINE__ << ": " << propMinus << std::endl;
            }
        }
        propMinus = max(0., propMinus);
        propMinus = min(1., propMinus);
    }

    for (int i = indexPlus; i != (indexMinus + 1) % degree; i = (i + 1) % degree)
    {
        if (mesh_.Edge(mesh_.Neigh(parent_node_id)[i].first).indexOfOppositeVert == -1)
            continue;
        double prop_left = 0;
        double prop_right = 1;
        if (indexPlus == i)
        {
            prop_right = 1 - propPlus;
        }
        if (indexMinus == i)
        {
            prop_left = 1 - propMinus;
        }

        create_interval_child_of_pseudosource(parent_node_id, i, prop_left, prop_right);
    }
}
// Compute the ps children when a ps parent is a window
void propagate_ps_from_window(int parent_node_id)
	{
		int degree = (int)mesh_.Neigh(parent_node_id).size();
		const std::vector<std::pair<int, double>> & neighs = mesh_.Neigh(parent_node_id);
		int indexOfParentOfParent = node_aux_[parent_node_id].direct_parent_id;
		int leftVert = mesh_.Edge(indexOfParentOfParent).indexOfLeftVert;
		int rightVert = mesh_.Edge(indexOfParentOfParent).right_node_id;
		int subIndexLeft = mesh_.GetSubindexToVert(parent_node_id, leftVert);
		int subIndexRight = (subIndexLeft + 1) % degree;
		double x1 = node_aux_[parent_node_id].entry_proportion * mesh_.Edge(indexOfParentOfParent).measure();
		double y1 = 0;
		double x2 = mesh_.Edge(indexOfParentOfParent).measure();
		double y2 = 0;
		x1 -= mesh_.Edge(indexOfParentOfParent).coordOfOppositeVert(0);
		y1 -= mesh_.Edge(indexOfParentOfParent).coordOfOppositeVert(1);
		x2 -= mesh_.Edge(indexOfParentOfParent).coordOfOppositeVert(0);
		y2 -= mesh_.Edge(indexOfParentOfParent).coordOfOppositeVert(1);

		double anglePlus = acos((x1 * x2 + y1 * y2) / sqrt((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2)));
		double angleSumR(anglePlus);
		int indexPlus;
		for (indexPlus = subIndexRight; indexPlus != subIndexLeft; indexPlus = (indexPlus + 1) % degree)
		{
			angleSumR += neighs[indexPlus].second;
			if (angleSumR > EXACT_PI - 4 * AngleTolerance)
				break;
		}
		double angleSumL = neighs[subIndexLeft].second - anglePlus;
		int indexMinus;
		for (indexMinus = (subIndexLeft - 1 + degree) % degree; indexMinus != (indexPlus - 1 + degree) % degree; indexMinus = (indexMinus - 1 + degree) % degree)
		{
			angleSumL += neighs[indexMinus].second;
			if (angleSumL > EXACT_PI - 4 * AngleTolerance)
				break;
		}
		if (indexMinus == (indexPlus - 1 + degree) % degree)
			return;
		for (int i = (indexPlus + 1) % degree; i != (indexMinus + 1) % degree; i = (i + 1) % degree)
		{
			extend_pseudosource(parent_node_id, i);
		}
		double propMinus = 1;
		double propPlus = 0;

		double devirationAngle = 20. * EXACT_PI / 180.;
		int minusEdge = mesh_.Neigh(parent_node_id)[indexMinus].first;
		if (mesh_.Edge(minusEdge).indexOfOppositeVert != -1)
		{
			double angleRemaining = angleSumL - EXACT_PI;
			if (mesh_.Neigh(parent_node_id)[indexMinus].second
				< 2 * devirationAngle)
			{
				propMinus = 1;
			}
			else if (angleSumL - EXACT_PI < devirationAngle
				|| abs(angleRemaining - mesh_.Neigh(parent_node_id)[indexMinus].second)
				< devirationAngle
				|| mesh_.Neigh(parent_node_id)[indexMinus].second
				> EXACT_PI - devirationAngle)
			{
				propMinus = angleRemaining / mesh_.Neigh(parent_node_id)[indexMinus].second
					+ devirationAngle / mesh_.Neigh(parent_node_id)[indexMinus].second;
				if (propMinus <= 0)
				{
					std::cerr << "propMinus " << __LINE__ << ": " << propMinus << "\n";
					assert(0);
				}
			}
			else
			{
				propMinus = sin(angleRemaining) * mesh_.Edge(minusEdge).measure()
					/ sin(angleRemaining +
						mesh_.Edge(mesh_.Edge(mesh_.Edge(minusEdge).indexOfLeftEdge).reverse_edge_id).angleOpposite)
					/ mesh_.Edge(mesh_.Edge(minusEdge).right_edge_id).measure()
					+ devirationAngle / mesh_.Neigh(parent_node_id)[indexMinus].second;
				if (propMinus <= 0)
				{
					std::cerr << "propMinus " << __LINE__ << ": " << propMinus << "\n";
					assert(0);
				}
			}

			propMinus = max(0., propMinus);
			propMinus = min(1., propMinus);
		}

		int rightEdge = mesh_.Neigh(parent_node_id)[indexPlus].first;
		if (mesh_.Edge(rightEdge).indexOfOppositeVert != -1)
		{
			double angleRemaining = mesh_.Neigh(parent_node_id)[indexPlus].second - (angleSumR - EXACT_PI);
			if (mesh_.Neigh(parent_node_id)[indexPlus].second
				< 2 * devirationAngle)
			{
				propPlus = 0;
			}
			else if (angleSumR - EXACT_PI < devirationAngle
				|| abs(angleRemaining) < devirationAngle
				|| mesh_.Neigh(parent_node_id)[indexPlus].second
				> EXACT_PI - devirationAngle)
			{
				propPlus = angleRemaining / mesh_.Neigh(parent_node_id)[indexPlus].second
					- devirationAngle / mesh_.Neigh(parent_node_id)[indexPlus].second;
				if (propPlus >= 1)
				{
					std::cerr << "propPlus " << __LINE__ << ": " << propPlus << "\n";
					assert(0);
				}
			}
			else
			{
				propPlus = sin(angleRemaining)
					* mesh_.Edge(rightEdge).measure()
					/ sin(angleRemaining + mesh_.Edge(mesh_.Edge(mesh_.Edge(rightEdge).indexOfLeftEdge).reverse_edge_id).angleOpposite)
					/ mesh_.Edge(mesh_.Edge(rightEdge).right_edge_id).measure()
					- devirationAngle / mesh_.Neigh(parent_node_id)[indexPlus].second;
				if (propPlus >= 1)
				{
					std::cerr << "propPlus " << __LINE__ << ":= " << propPlus << "\n";
					assert(0);
				}
			}

			propPlus = max(0., propPlus);
			propPlus = min(1., propPlus);
		}

		for (int i = indexPlus; i != (indexMinus + 1) % degree; i = (i + 1) % degree)
		{
			if (mesh_.Edge(mesh_.Neigh(parent_node_id)[i].first).indexOfOppositeVert == -1)
				continue;
			double prop_left = 0;
			double prop_right = 1;

			if (indexPlus == i)
			{
				prop_right = 1 - propPlus;
			}
			if (indexMinus == i)
			{
				prop_left = 1 - propMinus;
			}
			create_interval_child_of_pseudosource(parent_node_id, i, prop_left, prop_right);
		}
	}

// Method to add a valid window to windows queue
// The distance is updated via min_distance_of_windows()
void add_to_windows_queue(priority_window_t& w_quote)
{
    if (!check_window(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.updated_distance = min_distance_of_windows(*w_quote.w_pointer);
    Q_.push(w_quote);
}

} // namespace core
} // namespace fdapde

#endif   // __DISTANCE_SOVLER_H__