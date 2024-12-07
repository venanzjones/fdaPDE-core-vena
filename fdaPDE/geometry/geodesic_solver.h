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


#ifndef __DISTANCE_SOVLER_H__
#define __DISTANCE_SOVLER_H__

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

// Auxiliary structures
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
    long seq_tree_level;//may removed
    int ancestor_id;
    double distance_to_root;
    double proportions[2];
    double parent_entry_proportion;
    // TODO: capire perchè 2d e non 3d
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

class MyModel
{
protected:
    std::vector<double> scalar_field;
    const MeshType& mesh_;
    std::map<int, double> sources_;
    std::set<int> destinations_;
    std::priority_queue<priority_window_t> Q_; // window queue
	std::priority_queue<priority_node_t> P_; // pseudosources queue
    std::vector<edge_t> edge_aux_;
    std::vector<node_t> node_aux_;
    std::map<int,std::unordered_map<int>> nodes_adjacency; // da implementare

protected:
    void init();
    void dispose();
    void propagate();
    void push_into_pseudosources_queue(const priority_node_t& psuedo_source_quote);
    void add_to_windows_queue(priority_window_t& w_quote);
    bool is_too_small(const window_t& w) const;
    void compute_children_of_pseudosource(int parent_node_id);
    void compute_children_of_pseudosource_from_pseudosource(int parent_node_id);
    void compute_children_of_pseudosource_from_window(int parent_node_id);
    void create_interval_child_of_pseudosource(int source, int incident_edge_subindex, double prop_left = 0, double prop_right = 1);
    void fill_node_child_of_pseudosource(int source, int subnode_id);
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
    // Will choose only one in the end
    BaseModel(const MeshType& mesh, int source) : mesh_(mesh){sources_[source] = 0;}
    BaseModel(const MeshType& mesh, const std::map<int, double>& sources) : mesh_(mesh), sources_(sources){}
    BaseModel(const MeshType& mesh, const std::map<int, double>& sources, const std::set<int> &destinations) : mesh_(mesh), sources_(sources), destinations_(destinations){}
    BaseModel(const MeshType& mesh, const std::set<int>& sources) : mesh_(mesh)
    {
	for (auto it = sources.begin(); it != sources.end(); ++it)
		sources_[*it] = 0;
    }
    BaseModel(const MeshType& mesh, const std::set<int>& sources, const std::set<int>& destinations) : mesh_(mesh), destinations_(destinations)
    {
	for (auto it = sources.begin(); it != sources.end(); ++it)
		sources_[*it] = 0;
    }
    // Public methods
    void run();
    std::vector<EdgePoint> backtrace_shortest_path(int end) const;
    int get_ancestor(int node_id) const; // TODO: serve veramente?
    const std::vector<double>& get_distance_field() const;
};

void ICH::run()
{
	init();
	propagate();
	dispose();
}

void MyModel::init()
{
    	scalar_field.resize(mesh.n_nodes(), DBL_MAX);
    	node_aux_.resize(mesh.n_nodes());
}

void ICH::dispose()
{
    while (!Q_.empty())
    {
        delete Q_.top().w_pointer;
        Q_.pop();
    }
    P_ = priority_queue<priority_node_t>();
}

void MyModel::propagate()
{
    std::set<int> temp_destinations(destinations_);
    compute_sources_children();
    bool from_pseudosources_queue = update_tree_depth_with_choice();
    
    while (!P_.empty() || !Q_.empty())
    {
        if (from_pseudosources_queue)
        {
            int node_id = P_.top().node_id;
            P_.pop();
            temp_destinations.erase(node_id);
            if (!destinations_.empty() && temp_destinations.empty())
                return;
            compute_children_of_pseudosource(node_id);
        }
        else
        {
            priority_window_t w_quote = Q_.top();
            Q_.pop();
            compute_window_children(w_quote);
            delete w_quote.w_pointer;
        }
        from_pseudosources_queue = update_tree_depth_with_choice();
    }
}

void ICH::compute_sources_children()
{
    for (std::map<int, double>::const_iterator it = sources_.begin();
        it != sources_.end(); ++it)
    {
        compute_source_children(it->first, it->second);
    }
}

// TODO
void MyModel::compute_source_children(int src_node_id, double dis)
    {
		++node_aux_[src_node_id].birth_time;
		node_aux_[src_node_id].seq_tree_level = 0;
		node_aux_[src_node_id].ancestor_id = src_node_id;
		node_aux_[src_node_id].updated_distance = dis;

        // TODO: questo forse lo avevo già fatto in VTP, checkare
		int degree = (int)mesh_.Neigh(src_node_id).size();
		for (int i = 0; i < degree; ++i)
		{
			fill_node_child_of_pseudosource(src_node_id, i);
		}

		for (int i = 0; i < degree; ++i)
		{
			create_interval_child_of_pseudosource(src_node_id, i);
		}
	}

// TODO:
void MyModel::fill_node_child_of_pseudosource(int source, int subnode_id)
	{
        const EdgeType& edge = mesh_.edges().row(mesh_.Neigh(source)[subnode_id].first);
        // TODO: right vert cos'è?
		int index = edge.right_node_id;

		double dis = node_aux_[source].updated_distance + edge.measure();
        // TODO: EPSILON definition e edge.length?
		if (dis >= node_aux_[index].updated_distance - EPSILON)
			return;
		node_aux_[index].parent_is_a_pseudosource = true;
		++node_aux_[index].birth_time;
		node_aux_[index].direct_parent_id = source;

		node_aux_[index].seq_tree_level = node_aux_[source].seq_tree_level + 1;
		node_aux_[index].ancestor_id = node_aux_[source].ancestor_id;
		node_aux_[index].updated_distance = dis;
		if (!mesh_.is_node_strongly_convex(index))
			push_into_pseudosources_queue(priority_node_t(node_aux_[index].birth_time,
				index, dis));
	}
// TODO: 
void MyModel::create_interval_child_of_pseudosource(int source, int incident_edge_subindex, double prop_left= 0, double prop_right = 1)
{
    // TODO: 
    int incident_edge_id = mesh_.Neigh(source)[incident_edge_subindex].first;
    if (mesh_.is_edge_on_boundary(incident_edge_id))
        return;
    const EdgeType& edge = mesh_.edges().row(incident_edge_id);
    // TODO: right edge id
    const int edge_id = edge.right_edge_id;
    if (mesh_.is_edge_on_boundary(edge_id))
        return;
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    w_quote.w_pointer->proportions[0] = prop_left;
    w_quote.w_pointer->proportions[1] = prop_right;
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.w_pointer->brach_parent_is_pseudo_source = true;
    w_quote.w_pointer->direct_parent_is_pseudo_source = true;
    w_quote.w_pointer->parent_birth_time = node_aux_[source].birth_time;
    w_quote.w_pointer->brach_parent_id = source;
    w_quote.w_pointer->root_node_id = source;
    w_quote.w_pointer->current_edge_id = edge_id;
    w_quote.w_pointer->seq_tree_level = node_aux_[source].seq_tree_level + 1;
    w_quote.w_pointer->ancestor_id = node_aux_[source].ancestor_id;
    w_quote.w_pointer->distance_to_root = node_aux_[source].updated_distance;
    w_quote.w_pointer->parent_entry_proportion;
    //TODO: reverse edge id e Edge -> edges() + altra roba sotto
    int reverse_edge = mesh_.Edge(edge_id).reverse_edge_id;
    w_quote.w_pointer->ps_coordinates = reposotion_wrt_edge(reverse_edge,
        mesh_.Edge(reverse_edge).coordOfOppositeVert,mesh_);
    add_to_windows_queue(w_quote);
}

void MyModel::compute_left_child(const window_t& w)
{
    //  TODO: serve un metodo che mi dica se l'edge è estremo, ma non dentro triangulation, quindi
    //  bool is_extreme_edge(mesh,edge)
    if (is_edge_on_boundary(mesh_, mesh_.Edge(w.current_edge_id).indexOfLeftEdge))
        return;
	
    priority_window_t w_quote;
    w_quote.w_pointer = new window_t;
    // TODO: check ProportionOnLeftEdgeByImage
    w_quote.w_pointer->proportions[0] = mesh_.ProportionOnLeftEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[0]);
    w_quote.w_pointer->proportions[0] = max(0., w_quote.w_pointer->proportions[0]);
    w_quote.w_pointer->proportions[1] = mesh_.ProportionOnLeftEdgeByImage(w.current_edge_id,
        w.ps_coordinates, w.proportions[1]);
    w_quote.w_pointer->proportions[1] = min(1., w_quote.w_pointer->proportions[1]);
    if (is_too_small(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.w_pointer->brach_parent_is_pseudo_source = w.brach_parent_is_pseudo_source;
    w_quote.w_pointer->direct_parent_is_pseudo_source = false;
    w_quote.w_pointer->direct_parent_edge_on_left = true;
    // TODO: use fdapde
    w_quote.w_pointer->current_edge_id = mesh_.Edge(w.current_edge_id).indexOfLeftEdge;
    w_quote.w_pointer->distance_to_root = w.distance_to_root;

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
// TODO: change accordingly to function above
void MyModel::compute_right_child(const window_t& w)
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

void MyModel::compute_only_left_trimmed_child(const window_t& w)
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

void MyModel::compute_only_right_trimmed_child(const window_t& w)
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

void MyModel::compute_left_trimmed_child_with_parent(const window_t& w)
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


void MyModel::compute_right_trimmed_child_with_parent(const window_t& w)
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

double ICH::min_distance_of_windows(const window_t& w) const
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

bool ICH::is_too_small(const window_t& w) const
{
    return w.proportions[1] - w.proportions[0] < EPSILON;
}

bool MyModel::update_tree_depth_with_choice()
{
    while (!P_.empty()
        && P_.front().birth_time != node_aux_[P_.front().node_id].birth_time)
        P_.pop();

    while (!Q_.empty())
    {
        const priority_window_t& w_quote = Q_.front();
        if (w_quote.w_pointer->brach_parent_is_pseudo_source)
        {
            if (w_quote.w_pointer->parent_birth_time !=
                node_aux_[w_quote.w_pointer->brach_parent_id].birth_time)
            {
                delete w_quote.w_pointer;
                Q_.pop();
            }
            else
                break;
        }
        else
        {
            if (w_quote.w_pointer->parent_birth_time ==
                edge_aux_[w_quote.w_pointer->brach_parent_id].birth_time)
                break;
            else if (w_quote.w_pointer->is_on_left_subtree ==
                (w_quote.w_pointer->parent_entry_proportion < edge_aux_[w_quote.w_pointer->brach_parent_id].entry_proportion))
                break;
            else
            {
                delete w_quote.w_pointer;
                Q_.pop();
            }
        }
    }

    bool ICH::from_pseudosources_queue(false);
    if (Q_.empty())
    {
        if (!P_.empty())
        {
            const node_t& infoOfHeadElemOfPseudoSources = node_aux_[P_.front().node_id];
            from_pseudosources_queue = true;
        }
    }
    else
    {
        if (P_.empty())
        {
            const window_t& infoOfHeadElemOfWindows = *Q_.front().w_pointer;
            from_pseudosources_queue = false;
        }
        else
        {
            const node_t& infoOfHeadElemOfPseudoSources = node_aux_[P_.front().node_id];
            const window_t& infoOfHeadElemOfWindows = *Q_.front().w_pointer;
            if (infoOfHeadElemOfPseudoSources.seq_tree_level <=
                infoOfHeadElemOfWindows.seq_tree_level)
            {
                from_pseudosources_queue = true;
            }
            else
            {
                from_pseudosources_queue = false;
            }
        }
    }
    return from_pseudosources_queue;
}

bool MyModel::check_window(window_t& w) const
{
    if (w.direct_parent_is_pseudo_source)
        return true;
    const EdgeType& edge = mesh_.edges().row(w.current_edge_id); 
    int leftVert = edge.indexOfLeftVert;
    double dx = w.ps_coordinates(0) - w.proportions[1] * edge.measure();
    double rightLen = sqrt(dx * dx + w.ps_coordinates(1) * w.ps_coordinates(1));
    if (node_aux_[leftVert].updated_distance < 10000 * mesh_.GetScale() && node_aux_[leftVert].updated_distance + w.proportions[1] * edge.measure()
        < w.distance_to_root + rightLen)
    {
        return false;
    }
    int rightVert = edge.right_node_id;
    // int  rightVert = edgeNodes(1);
    // go on
   dx = w.ps_coordinates(0) - w.proportions[0] * edge.measure();
    double leftLen = sqrt(dx * dx + w.ps_coordinates(1) * w.ps_coordinates(1));
    if (node_aux_[rightVert].updated_distance < 10000 * mesh_.GetScale() && node_aux_[rightVert].updated_distance + (1 - w.proportions[0]) * edge.measure()
        < w.distance_to_root + leftLen)
    {
        return false;
    }
    const CRichModel::CEdge& oppositeEdge = mesh_.Edge(edge.reverse_edge_id);
    double xOfVert = edge.measure() - oppositeEdge.coordOfOppositeVert(0);
    double yOfVert = -oppositeEdge.coordOfOppositeVert(1);
    if (node_aux_[oppositeEdge.indexOfOppositeVert].updated_distance < 10000 * mesh_.GetScale())
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

void MyModel::compute_children_of_pseudosource(int parent_node_id)
{
    if (node_aux_[parent_node_id].parent_is_a_pseudosource)
        compute_children_of_pseudosource_from_pseudosource(parent_node_id);
    else
        compute_children_of_pseudosource_from_window(parent_node_id);
}
void MyModel::compute_children_of_pseudosource_from_pseudosource(int parent_node_id)
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
        fill_node_child_of_pseudosource(parent_node_id, i);
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

void MyModel::compute_children_of_pseudosource_from_window(int parent_node_id)
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
			fill_node_child_of_pseudosource(parent_node_id, i);
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

void ICH::push_into_pseudosources_queue(const priority_node_t& psuedo_source_quote)
{
    P_.push(psuedo_source_quote);
}

void ICH::add_to_windows_queue(priority_window_t& w_quote)
{
    if (!check_window(*w_quote.w_pointer))
    {
        delete w_quote.w_pointer;
        return;
    }
    w_quote.updated_distance = min_distance_of_windows(*w_quote.w_pointer);
    Q_.push(w_quote);
}

std::vector<EdgePoint> MyModel::BacktraceShortestPath(int end) const
{
    if (m_InfoAtVertices[end].birth_time == -1
        || m_InfoAtVertices[end].direct_parent_id == -1)
    {
        assert(mesh.GetNumOfComponents() != 1 || mesh.Neigh(end).empty());
        return std::vector<EdgePoint>();
    }
    std::vector<EdgePoint> path;
    std::vector<int> vertexNodes;
    int index = end;
    vertexNodes.push_back(index);
    while (m_InfoAtVertices[index].direct_parent_id != -1)
    {
        int indexOfParent = m_InfoAtVertices[index].direct_parent_id;
        if (m_InfoAtVertices[index].parent_is_a_pseudosource)
        {
            index = indexOfParent;
        }
        else
        {
            index = m_InfoAtVertices[index].root_node_of_direct_parent_id;
        }
        vertexNodes.push_back(index);
    };
    int src_node_id = index;

    for (int i = 0; i < (int)vertexNodes.size() - 1; ++i)
    {
        int lastVert = vertexNodes[i];
        path.push_back(EdgePoint(lastVert));
        if (m_InfoAtVertices[lastVert].parent_is_a_pseudosource)
        {
            continue;
        }
        int parentedge_id = m_InfoAtVertices[lastVert].direct_parent_id;
        int edge_id = mesh.Edge(parentedge_id).reverse_edge_id;
        Eigen::Vector2d coord(reposotion_wrt_edge(parentedge_id, mesh.Edge(parentedge_id).coordOfOppositeVert,mesh_));

        double proportion = 1 - m_InfoAtVertices[lastVert].entry_proportion;
        while (true)
        {
            path.push_back(EdgePoint(edge_id, proportion));
            if (mesh.Edge(edge_id).indexOfOppositeVert == vertexNodes[i + 1])
                break;
            double oldprop_rightotion = proportion;
            proportion = mesh.ProportionOnLeftEdgeByImage(edge_id, coord, oldprop_rightotion);
            if (abs(proportion - 1) < 1e-2)
            {
                vector<EdgePoint> path2 = BacktraceShortestPath(mesh.Edge(edge_id).indexOfOppositeVert);
                reverse(path.begin(), path.end());
                copy(path.begin(), path.end(), back_inserter(path2));
                return path2;
            }
            else if (proportion >= 0 && proportion <= 1)
            {
                proportion = max(proportion, 0);
                coord = mesh.rotate_around_left_child_edge(edge_id, coord);
                edge_id = mesh.Edge(edge_id).indexOfLeftEdge;
            }
            else
            {
                proportion = mesh.ProportionOnRightEdgeByImage(edge_id, coord, oldprop_rightotion);
                proportion = max(proportion, 0);
                proportion = min(proportion, 1);
                coord = rotate_around_right_child_edge(edge_id, coord,mesh_);
                edge_id = mesh.Edge(edge_id).right_edge_id;
            }
        };
    }
    path.push_back(EdgePoint(src_node_id));
    reverse(path.begin(), path.end());
    return path;
}


int MyModel::GetAncestor(int vertex) const
{
    return m_InfoAtVertices[vertex].ancestor_id;
}

}   // namespace core
}   // namespace fdapde

#endif   // __DISTANCE_SOVLER_H__
