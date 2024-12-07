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

#ifndef __GEODESIC_UTILS_H__
#define __GEODESIC_UTILS_H__

#include<cmath> // for acos()
typedef fdapde::core::triangulation<2,3> MeshType;

namespace fdapde {
namespace core {

// Checks if the sum of the incident angles is < than 2*pi,
// it is needed for the insertion in pseudosource queue
// Shall be optimized!
const bool is_node_strongly_convex(MeshType& mesh, int node_id)
{
	double angle_sum = 0.0;
	std::vector<int> patch = mesh.node_patch(node_id);
	for(auto cell : patch){
		std::nodes = cell.node_ids();
		std::vector<int> a_id(nodes[0]), b_id(nodes[1]), c_id(nodes[2]);
		NodeType a(mesh.node(a_id)), b(mesh.node(b_id)), c(mesh.node(c_id));
		// Compute edge lengths
		double ab = (a-b).squaredNorm();
		double bc = (b-c).squaredNorm();
		double ca = (c-a).squaredNorm();
		// Use Carnot theorem to compute the proper angle
		if(node_id == a_id){angle_sum += acos((ca * ca + ab * ab - bc * bc) / (2 * ca * ab));}
		else if(node_id == b_id){angle_sum += acos((ab * ab + bc * bc - ca * ca) / (2 * ab * bc));}
		else if(node_id == c_id){angle_sum += acos((bc * bc + ca * ca - ab * ab) / (2 * bc * ca));}
	}
	return (angle_sum < 2 * EXACT_PI);
}
// Computes a translation vector for repositioning a point or vector in the 2D space relative to a mesh edge
// Usage: pseudo_source_coordinates = reposotion_wrt_edge(e, e.opposite_vertex().coordinates(),mesh_);
// Input: mesh, edge_id, coordinates of the node opposite to edge_id
Eigen::Vector2d reposotion_wrt_edge(int edge_id, const Eigen::Vector2d& vec, MeshType& mesh) const
{
    return Eigen::Vector2d(mesh.edges().row(edgde_id).measure() - vec(0), -vec(1));
}
// Rotates the vector vec around the left child edge of the edge specified by edge_id
Eigen::Vector2d rotate_around_left_child_edge(int edge_id, const Eigen::Vector2d& vec,MeshType& mesh) const
{   
    const auto coord1 = Edge(edge_id).matrixRotatedToLeftEdge(0) * vec(0) - Edge(edge_id).matrixRotatedToLeftEdge(1) * vec(1);
    const auto coord2 = Edge(edge_id).matrixRotatedToLeftEdge(1) * vec(0) + Edge(edge_id).matrixRotatedToLeftEdge(0) * vec(1);
    return Eigen::Vector2d(coord1,coord2);
}

// Rotates the vector vec around the right child edge of the edge specified by edge_id.
Eigen::Vector2d rotate_around_right_child_edge(int edge_id, const Eigen::Vector2d& vec,MeshType& mesh) const
{
	int reverseEdge = Edge(Edge(edge_id).indexOfRightEdge).indexOfReverseEdge;
	Eigen::Vector2d coordOfLeftEnd = reposotion_wrt_edge(reverseEdge, Edge(reverseEdge).coordOfOppositeVert, mesh);
	return Eigen::Vector2d(Edge(edge_id).matrixRotatedToRightEdge(0) * vec(0) - Edge(edge_id).matrixRotatedToRightEdge(1) * vec(1) + coordOfLeftEnd(0),
		Edge(edge_id).matrixRotatedToRightEdge(1) * vec(0) + Edge(edge_id).matrixRotatedToRightEdge(0) * vec(1) + coordOfLeftEnd(1));
}

}   // namespace core
}   // namespace fdapde

#endif   // __GEODESIC_UTILS_H__