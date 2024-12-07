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

#ifndef __SURFACE_VORONOI_H__
#define __SURFACEVORONOI_H__

#include <vector> 
#include <map> 
#include <Eigen/Dense> 
#include "prism.h"
#include "overpropagation.h"

namespace fdapde {
namespace core {

typedef Tringulation<2,3> = MeshType;
typedef TriangularPrism::CellSection CellSection;
typedef std::vector<CellSection> VoronoiCellType;
typedef std::map<int,VoronoiCellType> VoronoiPartitionType;
typedef std::map<int,double> WeightsType;
typedef TriangularPrism::Facet Facet;
typedef std::vector<std::tuple<int, double, double, double>> DistanceType;
typedef std::vector<DistanceType> DistanceFieldType;
double INFINITY_DISTANCE = std::numeric_limits<double>::max;

// Main class for the algorithm
class SurfaceVoronoi {
    public:
        void compute_voronoi_partition(const std::set<int>& sources);
        double compute_voronoi_cell_area(const VoronoiCellType& voronoi_cell);
        WeightsType compute__voronoi_weights(const VoronoiPartitionType& voronoi_partition);
    
        // Constructor
        SurfaceVoronoi(const MeshType& mesh) : mesh_(mesh) {}
        // Destructor
        ~SurfaceVoronoi() {}

    private:
        MeshType mesh_;
        VoronoiPartitionType voronoi_partition_;
        void assembly_VD_in_triangle(VoronoiPartitionType & domain_vd_partition);
        DistanfceFieldType compute_overpropagation_field(const std::set<int>& sources)

}


// Method to compute the area of a voronoi cell
double SurfaceVoronoi::compute_voronoi_cell_area(const VoronoiCellType& voronoi_cell) {
    double area = 0.0;
    for (const auto& cell_section : voronoi_cell) {
	// if the restricted VD in the cell coincides with the cell boundary:
	if (cell_section.covers_entire_cell) {
	    // Get the vertices of the triangular cell
	    const auto& triangular_cell = this->mesh_.cells()[cell_section.cell_id];
	    total_area += triangular_cell.area();
	} else {
	    // else, calculate the area directly from the vertices convex polygon
        // TODO: ConvexPolygon, metodo area() che usa shoelace formula
	    area += cell_section.section.area();
	}
    }
    return total_area;
}

// Method to compute the area (weights for the integration) of all the voronoi cells
WeightsType SurfaceVoronoi::compute_voronoi_weights(VoronoiPartitionType& voronoi_partition) {
	WeightsType weights;
	for (const auto& [id, cell]: voronoi_partition){
		weights[id] = compute_voronoi_cell_area(cell);
	}
	return weights;
}

// Method to run the SurfaceVoronoi algorithm and compute the Voronoi Diagram given the sources
VoronoiCellType SurfaceVoronoi::compute_voronoi_partition(const std::set<int>& sources){
    VoronoiPartitionType voronoi_partition;
    // Infer the distance field using the overpropagation algorithm
    auto distance_field = compute_overpropagation_field(sources);
    // Loop over all the cells
    for (int cell_id = 0; cell_id < this->mesh_.n_cells(); ++cell_id)
    {
        // If only one source contributes to the VD in the triangle
        if (distance_field[cell_id].size() <= 1)
        {
            // The VD in the cell is just the triangle itself associated with the single source
            CellSection facet;
            facet.cell_id = cell_id;
            // The bool avoids us to specify the actual section and in the future,
            // will simplify the voronoi cell computation by just computing triangle area
            facet.covers_entire_cell = true;
            // Add the cell to the corresponding voronoi partition  
            voronoi_partition[std::get<0>(distance_field[cell_id][0])].push_back(facet);
        }
        else // If the cell is affected by multiple sources
        {
            // Implement the incremental half-edge cutting algorithm
            TriangularPrism prism(this->mesh_, cell_id);
            // Loop over all the sources that contribute to the VD in the given cell
            for (int j = 0; j < distance_field[cell_id].size(); ++j)
            {
                // Construct the hyperplane passing through the points
                // (x1,y1, ||p - v1||^2), (x2,y2, ||p - v2||^2), (x3,y3,||p - v3||^2)
                // as justified by the lifting scheme proposed in the papers

                // Check if we can simplify this code!
                // Moreover, check wheher the tuple is still in our returntype from dist
                Facet new_facet(&prism, std::get<1>(distance_field[cell_id][j]) * std::get<1>(distance_field[cell_id][j]),
                            std::get<2>(distance_field[cell_id][j]) * std::get<2>(distance_field[cell_id][j]),
                            std::get<3>(distance_field[cell_id][j]) * std::get<3>(distance_field[cell_id][j]),
                            std::get<0>(distance_field[cell_id][j]));
                // Cut the prism with the new facet
                prism.incremental_cutting(new_facet);
            }
            prism.assembly_VD_in_triangle(voronoi_partition);
        }
    }
    return voronoi_partition;
}

void SurfaceVoronoi::assembly_VD_in_triangle(VoronoiPartitionType& domain_vd_partition) const
{
    std::map<int, std::vector<SVector<2>>> source_to_vertices;
    // Loop over the surviving nodes
    for (auto vertex_id : surviving_nodes)
    {
        auto vertex = prism_nodes[vertex_id];
        if (vertex.is_bottom())
            continue;
        if (!pi[vertex.facet1].is_wall)
        {
            source_to_vertices[pi[vertex.facet1].ancestor].emplace_back(SVector<2>(vertex.x, vertex.y));
        }
        if (!pi[vertex.facet2].is_wall)
        {
            source_to_vertices[pi[vertex.facet2].ancestor].emplace_back(SVector<2>(vertex.x, vertex.y));
        }
        if (!pi[vertex.facet3].is_wall)
        {
            source_to_vertices[pi[vertex.facet3].ancestor].emplace_back(SVector<2>(vertex.x, vertex.y));
        }
    }
    // id_to_vertices itera su una std::map<int, std::vector<SVector<2>>>
    // quindi è un std::pair<int, std::vector<SVector<2>>> e accedo
    // al source_id con id_to_vertices.first e al vettore di vertici con id_to_vertices.second
    for (auto id_to_vertices : source_to_vertices)
    {
        int source_id = id_to_vertices.first;
        CellSection restricted_vd;
        restricted_vd.base_id = base_id;
        // Degenerate case: same as only one source
        // We have 3 nodes in the basis + 3 obtained from cutting the prism, in this case their projection 
        // coincides with the triangle vertices and thus the restricted VD coincide with the cell boundary 
        restricted_vd.covers_entire_cell = (surviving_nodes.size() <= 6);
        // Make sure ConvexPolygon has a proper constructor for ConvexPolygon(std::vector<SVector<2>>)
        ConvexPolygon vd_boundary(id_to_vertices.second);
        domain_vd_partition[source_id].push_back(facet);
    }
}

DistanfceFieldType SurfaceVoronoi::compute_overpropagation_field(const std::set<int>& sources)
{
    // Run the overpropagation algorithm (inherits from ICH)
    // This is needed to get the candidates faces for each src!
    OverPropagation alg(this->mesh_, sources);
    alg.Execute();

    // Make room for the distance field
    DistanfceFieldType distance_field(this->mesh.n_cells());
    for (auto src : sources)
    {
        // This time we have a sinle source at a time, but constructor takes set as input
        std::set<int> single_src;
        single_src.insert(src);
        // Destination are only the nodes of the candidate cells
        std::set<int> destinations;
        // Loop over the candidates cells for the source
        for (auto face_id : alg.candidates[src])
        {
            // Extract the nodes of the face
            const auto& nodes = this->mesh_.cells().row(src_cell_id);
            // Loop over the nodes of the cell
            for (int k = 0; k < 3; ++k)
            {
                destinations.insert(nodes[k]);
            }
        }
        // Now I am ready to run the algorithm
        DistanceSolver alg_single_src(model, single_src, destinations);
        alg_single_src.execute();

        // Store the distances in the distance field
        for (auto face_id : alg.candidates[src])
        {
            // TODO: check 
            distance_field[face_id].push_back(make_tuple(src,
                alg_single_src.get_distance_field()[model.Face(face_id)[0]],
                alg_single_src.get_distance_field()[model.Face(face_id)[1]],
                alg_single_src.get_distance_field()[model.Face(face_id)[2]]));
        }
    }
    return distance_field;
}


}   // namespace core
}   // namespace fdapde

#endif   // __SURFACE_VORONOI_H__






//-------------------

// EUCLIDEAN DISTANCE

//-------------------



DistanfceFieldType SurfaceVoronoi::compute_euclidean_overpropagation_field(const std::vector<int>& sources)
{
    // Room for the overpropagated field, one tuple for each cell
    DistanfceFieldType final_distances(this->mesh_.n_cells());
    // Keeps track of the smallest distances to each face's (by that time)
    // Init the distances to infinity and the source_id to -1
    DistanceType best_distances(this->mesh_.n_cells(), std::make_tuple(-1, INFINITY_DISTANCE, INFINITY_DISTANCE, INFINITY_DISTANCE));

    // Lambda function to compute the average distance in a distance tuple
    double avg_tuple = [](const tuple<int, double, double, double>& dist_tuple)
    {
        const auto& [id, d1, d2, d3] = dist_tuple; 
        return (d1 + d2 + d3) / 3.0;
    };

    // Event struct
    struct Event
    {
        int from_source, to_face;
        double d1, d2, d3;
        const bool operator>(const Event& rhs) const
        {
            return d1 + d2 + d3 > rhs.d1 + rhs.d2 + rhs.d3;
        }
        const double avg() const
        {
            return (d1 + d2 + d3) / 3.0;
        }
    };

    // Initialize the event queue and the visited faces for each source
    std::vector<std::unordered_set<int>> visited(sources.size());
    std::priority_queue<Event> pending;
    Event Event;

    // Initialize the propagation by creating an event for each source at its starting cell
    // For each source, calculate distances (d1, d2, d3) from the source to the three nodes of its cell
    // Create an Event with these distances and push it into the queue, and finally mark as visited for that source
    for (int i = 0; i < sources.size(); ++i)
    {
        // Get source_id
        Event.from_source = i;
        // Get src info using structured binding
        const auto& [coord1, coord2, coord3, src_cell_id] = sources[i]; 
        Event.to_face = src_cell_id;
        // Get the node coordinates
        src_coordinates NodeType(coord1, coord2, coord3);
        const auto& cell_nodes = this->mesh_.cells().row(src_cell_id);
        // Calculate the distances from the src to the three nodes of the cell
        Event.d1 = (src_coordinates - this->mesh_.nodes().row(cell_nodes[0]).squadredNorm());
        Event.d2 = (src_coordinates - this->mesh_.nodes().row(cell_nodes[1]).squadredNorm());
        Event.d3 = (src_coordinates - this->mesh_.nodes().row(cell_nodes[2]).squadredNorm());
        // Push the event in the queue
        pending.push(Event);
        // update the visited faces of source i
        visited[i].insert(src_cell_id);
    }

    // Initialize an empty vector to store cell neighbours
    std::vector<int> neighbours;
    NodeType src_coordinates;

    // While the queue is not empty
    while (!pending.empty())
    {
        Event& Event = pending.front();
        if (Event.avg() > avg_tuple(best_distances[Event.to_face]))
        {
            // Check if worth updating the best distances
            if (Event.d1 < std::get<1>(best_distances[Event.to_face])
                || Event.d2 < std::get<2>(best_distances[Event.to_face])
                || Event.d3 < std::get<3>(best_distances[Event.to_face]))
            {
                final_distances[Event.to_face].emplace_back(Event.from_source, Event.d1, Event.d2, Event.d3);
            }
            else {
                pending.pop();
                continue;
            }
        }
        else
        {
            best_distances[Event.to_face] = std::make_tuple(Event.from_source, Event.d1, Event.d2, Event.d3);
            final_distances[Event.to_face].emplace_back(Event.from_source, Event.d1, Event.d2, Event.d3);
        }

        // TODO: check if neighbours is always <= 3
        auto& neighbours = this->mesh_.neighbours(Event.to_face);

        const auto& [coord1, coord2, coord3, src_cell_id] = sources[Event.from_source]; 
        src_coordinates NodeType(coord1, coord2, coord3);

        for (int i = 0; i < neighbours.size(); ++i)
        {
            if (neighbours[i] == -1 || visited[Event.from_source].find(neighbours[i]) != visited[Event.from_source].end())
                continue;

            Event.to_face = neighbours[i];
            const auto& cell_nodes = this->mesh_.cells().row(Event.to_face);
            // Calculate the distances from the src to the three nodes of the cell
            Event.d1 = (src_coordinates - this->mesh_.nodes().row(cell_nodes[0]).squadredNorm());
            Event.d2 = (src_coordinates - this->mesh_.nodes().row(cell_nodes[1]).squadredNorm());
            Event.d3 = (src_coordinates - this->mesh_.nodes().row(cell_nodes[2]).squadredNorm());
            pending.push(Event);
            visited[Event.from_source].insert(Event.to_face);
        }
        pending.pop();
    }

    return final_distances;
}