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

#ifndef __OVERPROPAGATION_H__
#define __OVERPROPAGATION_H__

#include <vector> 
#include <set>
#include <map> 
#include <Eigen/Dense> 
#include "geodesic_solver.h"

namespace fdapde {
namespace core {

struct OverPropagation : public MyModel
{
    // Room for the candidate faces for each source
    std::map<int, std::set<int>> candidates;

    // Constructor: inserts the faces containing each src to the 
    // src_id -> set<face_id> map, trivial inizialization
    OverPropagation(const MeshType& mesh, const std::set<int>& sources)
        : MyModel(mesh, sources) {

        // Initialize the candidates faces for each source
        for (auto src : sources)
        {   
            // Con node patch estraggo le facce contenenti src,
            // che chiaramente vanno inizializzate come candidate
            const auto& src_patch = mesh.node_patch(src);
            for (auto face_id : src_patch)
            {
                candidates[src].insert(face_id);
            }
        }
    }

    // This snippet follows the same propagation as the one in the mark-and-sweep solver
    // but adds the candidate cells for each source, through (★) in the paper.
    // For the propagation logic, check geodesic_solver.h -> MyModel::propagate()
    void propagate()
    {
        std::set<int> temp_destinations(destinations_);
        compute_sources_children();
        bool from_pseudosources_queue = update_tree_depth_with_choice();

        while (!P_.empty() || !Q_.empty())
        {
            if (from_pseudosources_queue)
            {
                int node_id = P_.top().node_id;
                const auto& src_patch = this->mesh_.node_patch(node_id);
                for (auto face_id : src_patch)
                {
                    candidates[node_aux_[node_id].ancestor_id].insert(face_id);
                }
                P_.pop();
                temp_destinations.erase(node_id);

                if (!destinations_.empty() && temp_destinations.empty())
                    return;
                compute_children_of_pseudosource(node_id);
            }
            else
            {
                priority_window_t window_quote = Q_.top();
                int e = window_quote.window_pointer->current_edge_id;
                // Add the faces containing edge e, the use of the set handles eventual duplicates
                const auto face_ids = mesh_.edges_to_cells().row(e);
                candidates[window_quote.window_pointer->ancestor_id].insert(face_ids[0]);
                candidates[window_quote.window_pointer->ancestor_id].insert(face_ids[1]);
                Q_.pop();
                compute_window_children(window_quote);
                delete window_quote.window_pointer;
            }
            from_pseudosources_queue = update_tree_depth_with_choice();
        }
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __OVERPROPAGATION_H__
