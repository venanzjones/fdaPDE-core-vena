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

#ifndef PRISM_H
#define PRISM_H

// put in config file
constexpr double MAX_HEIGHT = 100000.0;

// To be decided where should be put
template <typename T>
T convex_combination(double lambda, T val1, T val2) {
    return (1 - lambda) * val1 + lambda * val2;}

namespace fdapde
{
namespace core
{
// We need to define a struct that stores a prism with triangular base and a conve polytope at top.
struct TriangularPrism
{
    MeshType mesh; // mesh
    Eigen::Vector2d node_1, node_2, node_3; // triangle vertex projected in 2D
    int base_id; // id of the triangular base in the mesh
    std::vector<Facet> pi; // vector of facets composing the top and the lateral edges
    std::vector<PrismVertex> prism_nodes; // vector of prism nodes
    std::set<int> surviving_nodes; // id of vertices currently part of the prism
    std::vector<PrismEdge> prism_edges; // vector of prism edges
    std::set<int> surviving_edges; // id of nodes currently part of the prism

// Auxiliary struct that stores a general facet
struct Facet
{
    bool is_wall; // bool to know if it is a lateral facet
    int ancestor; // source index associated with the facet 
    int wall_id; // lateral facet id
    double a, b, c; // coefficientes of plane equation z = ax + by + c passing through the facet
    // default constructor for wall facets
    Facet()
    {
        is_wall = true;
        wall_id = -1;
        ancestor = -1; 
    }
    // constructor for non-wall facets
    Facet(const TriangularPrism* prism, double h1, double h2, double h3, int ancestor)
    : ancestor(ancestor), is_wall(false)
    {
        Eigen::Matrix3d m;
        m << prism->node_1.x(), prism->node_1.y(), 1.0,
            prism->node_2.x(), prism->node_2.y(), 1.0,
            prism->node_3.x(), prism->node_3.y(), 1.0;

        Eigen::Vector3d rhs(h1, h2, h3);

        Eigen::FullPivLU<Eigen::Matrix3d> lu(m);
        if(!lu.isInvertible()) {
            throw std::runtime_error("cannot compute plane");
        }
        Eigen::Vector3d res = lu.solve(rhs);

        a = res(0);
        b = res(1);
        c = res(2);
        wall_id = -1;
    }
    // Aux function 
    bool is_above(double x, double y, double z) const noexcept
    {
        return (z > (a*x + b*y + c));
    }
};
// Auxiliary structure to store prism vertices
struct PrismVertex {
    int facet1, facet2, facet3; // facet intersecting in v
    double x, y, h;
    // Constructor to simplify the prism initalization
    PrismVertex(int facet1, int facet2, int facet3, double x, double y, double h)
    : facet1(facet1),facet2(facet2),facet3(facet3),x(x),y(y),h(h){}
    // Auxiliary function to check if the vertex is a bottom vertex
    bool is_bottom() const
    {
        return facet3 == -1 || facet1 == -1 || facet2 == -1;
    }
};
// Auxiliary struct to store prism edges
struct PrismEdge
{
    int vertex1, vertex2;
    int facet1, facet2;
    PrismEdge(int vertex1, int vertex2, int facet1, int facet2)
    : vertex1(vertex1),vertex2(vertex2),facet1(facet1),facet2(facet2){}
};
// Sturcure to save Restricted Voronoi one each triangles
struct CellSection
{
    int base_id;
    bool covers_entire_cell; // Useful to compute cell area later
    ConvexPolygon section;   // If !covers_entire_cell: mi serve per calcolare l'area
};

struct ConvexPolygon
{
	std::vector<Eigen::Vector2d> vertices; // Ordered vertices

	// Compute the area of the polygon
	double area() const
	{
		double a = 0.0;
		size_t n = vertices.size();
		for (size_t i = 0; i < n; ++i)
		{
			const Eigen::Vector2d p1 = vertices[i];
			const Eigen::Vector2d p2 = vertices[(i + 1) % n];
			a += (p1.x() * p2.y()) - (p2.x() * p1.y());
		}
		// Se è negativa dovrei fare reverse dei vertici ma non è strettamente necessario
		return std::abs(a) / 2.0;
	}
}
// Triangular prism constructor, from coordinates of the base triangle and infinite height
TriangularPrism(double x1, double y1,
			    double x2, double y2,
			    double x3, double y3) : 
                base_id(-1)
{
    node_1 = Eigen::Vector2d(x1, y1);
    node_2 = Eigen::Vector2d(x2, y2);
    node_3 = Eigen::Vector2d(x3, y3);
    Facet facet;
    // Add the 3 wall facets (id: 0,1,2)
    facet.is_wall = true;
    for (int i = 0; i < 3; ++i)
    {
        facet.wall_id = i;
        pi.push_back(facet);
    }
    // Add the base (id: 3)
    pi.push_back(Facet(this, MAX_HEIGHT, MAX_HEIGHT, MAX_HEIGHT, -1));

    // Now the tedious part to incorporate the connecitivty
    // Aux vector to have a clean construction
    std::vector<std::pair<double, double>> v_coords = {{x1, y1}, {x2, y2}, {x3, y3}};

    // Add the prism vertices
    // Top vertices with coordinates (x_i,y_i, MAX_HEIGHT)
    // Bottom vertices with coordinates (x_i,y_i, -MAX_HEIGHT)
    double z = MAX_HEIGHT;
    for (int h = 0; h < 2; ++h) {  // Loop for top (h=0) and bottom (h=1)
        int id = (h == 0) ? 3 : -1;  // Top (3) or bottom (-1)
        for (int i = 0; i < 3; ++i) {
            prism_nodes.push_back(PrismVertex(
                (i + 2) % 3,  // First index 
                i % 3,        // Second index
                id,                          // id
                v_coords[i].first,           // x-coordinate
                v_coords[i].second,          // y-coordinate
                z                            // z-coordinate
            ));
            surviving_nodes.insert(prism_nodes.size() - 1);
        }
        z = -z; // For the second iteration, h becomes - MAX_HEIGHT
    }

    // Now the edges: 0-1, 1-2, 2-0, 0-3, 1-4, 2-5 
    // The induced labeling:  0    1    2    3    4    5

    // Add the base edges    
    for (int i = 0; i < 3; ++i) {  
        prism_edges.push_back(PrismEdge(
            i,                                 
            (i+1) % 3,                               
            i,        
            3        
        ));
        surviving_edges.insert(prism_edges.size() - 1);
    }
    // Add the vertical edges
    for (int i = 0; i < 3; ++i) {  // Loop over vertices
        prism_edges.push_back(PrismEdge(
            i,                                 
            i+3,                               
            (i+2) % 3,        
            i        
        ));
        surviving_edges.insert(prism_edges.size() - 1);
    }


// Triangular prism constructor from 
TriangularPrism(const MeshType& mesh, int base_id) : base_id(base_id), mesh(mesh)
{   
    // Build the triangular base in 2D: (0,0) (x2,0) (x3,y3)
    double x1(0), y1(0);
    node_1 = Eigen::Vector2d(x1, y1);
    // Take the base_edge: will connect (0,0) to (x2,0)
    double x2 = mesh.cells(base_id).edges(0).measure();
    node_2 = Eigen::Vector2d(x2, 0);
    // TODO: fix this
    double x3 = mesh.Edge(edge_id).coordOfOppositeVert.x();
    double y3 = mesh.Edge(edge_id).coordOfOppositeVert.y();
    node_3 = Eigen::Vector2d(x3, y3);

    Facet facet;
    // Add the 3 wall facets (id: 0,1,2)
    facet.is_wall = true;
    for (int i = 0; i < 3; ++i)
    {
        facet.wall_id = i;
        pi.push_back(facet);
    }
    double MAX_HEIGHT = 100000;
    // Same as before
    pi.push_back(Facet(this, MAX_HEIGHT, MAX_HEIGHT, MAX_HEIGHT, -1));

    // Now the tedious part to incorporate the connecitivty
    // Aux vector to have a clean construction
    std::vector<std::pair<double, double>> v_coords = {{x1, y1}, {x2, y2}, {x3, y3}};

    // Add the prism vertices
    // Top vertices with coordinates (x_i,y_i, MAX_HEIGHT)
    // Bottom vertices with coordinates (x_i,y_i, -MAX_HEIGHT)
    double z = MAX_HEIGHT;
    for (int h = 0; h < 2; ++h) {  // Loop for top (h=0) and bottom (h=1)
        int id = (h == 0) ? 3 : -1;  // Top (3) or bottom (-1)
        for (int i = 0; i < 3; ++i) {
            prism_nodes.push_back(PrismVertex(
                (i + 2) % 3,  // First index 
                i % 3,        // Second index
                id,                          // id
                v_coords[i].first,           // x-coordinate
                v_coords[i].second,          // y-coordinate
                z                            // z-coordinate
            ));
            surviving_nodes.insert(prism_nodes.size() - 1);
        }
        z = -z; // for the second iteration, h becomes - MAX_HEIGHT
    }

    // Now the edges: 0-1, 1-2, 2-0, 0-3, 1-4, 2-5 
    // The induced labeling:  0    1    2    3    4    5

    // Add the base edges    for (int i = 0; i < 3; ++i) {  
        prism_edges.push_back(PrismEdge(
            i,                                 
            (i+1) % 3,                               
            i,        
            3        
        ));
        surviving_edges.insert(prism_edges.size() - 1);
    }
    // Add the vertical edges
    for (int i = 0; i < 3; ++i) {  // Loop over vertices
        prism_edges.push_back(PrismEdge(
            i,                                 
            i+3,                               
            (i+2) % 3,        
            i        
        ));
        surviving_edges.insert(prism_edges.size() - 1);
    }
}
};

// Now that everything is set up, we provide an
// implementation of the incremental cutting method used in the paper
// Input: new_facet to cut the prism with
// Output: internally updates the prism structure
void incremental_cutting(const Facet& new_facet)
{
    // Init a set to store the vertex to remove, for fast insertion/deletion
    std::set<int> v_to_remove;
    // Loop over all the surviving nodes
    for (auto v : surviving_nodes)
    {
        int v_id = prism_nodes[v];
        // Check if: h > a * x + b * y + c
        if (new_facet.is_above(v_id->x, v_id->y, v_id->h))
            v_to_remove.insert(v);
    }
    if (v_to_remove.empty())
        return;
    for (auto v : v_to_remove)
        surviving_nodes.erase(v);

    // Append the new facet
    pi.push_back(new_facet);
    int new_facet_id = pi.size() - 1;

    // map to store the intersections of the new facet with the existing facets
    std::map<int, std::set<int>> facet_intersections; 
    // edges to be removed
    std::set<int> edges_to_remove;
    // loop over the surviving edges
    for (auto edge_id : surviving_edges)
    {
        // check whether v1 and v2 of the edge_id survive
        bool v1_survives = (surviving_nodes.find(prism_edges[edge_id].vertex1) != surviving_nodes.end());
        bool v2_survives = (surviving_nodes.find(prism_edges[edge_id].vertex2) != surviving_nodes.end());
        // if they both survive, go to the next edge
        if (v1_survives && v2_survives)
            continue;

        // if only v1 survives
        if (v1_survives && !v2_survives)
        {
            PrismVertex vertex_new;
            PrismVertex v1 = prism_nodes[prism_edges[edge_id].vertex1];
            PrismVertex v2 = prism_nodes[prism_edges[edge_id].vertex2];
            
            vertex_new.facet1 = prism_edges[edge_id].facet1;
            vertex_new.facet2 = prism_edges[edge_id].facet2;
            vertex_new.facet3 = new_facet_id;

            double delta1 = new_facet.a *v1.x
                + new_facet.b *v1.y
                + new_facet.c -v1.h;

            double delta2 = new_facet.a * v2.x
                + new_facet.b * v2.y
                + new_facet.c - v2.h;

            double lambda = (delta1) / (delta1 - delta2);
            vertex_new.x = convex_combination(lambda, v1.x, v2.x);
            vertex_new.y = convex_combination(lambda, v1.y, v2.y);
            vertex_new.z = convex_combination(lambda, v1.z, v2.z);
            prism_nodes.push_back(vertex_new);
            int new_vertex_id = prism_nodes.size() - 1;
            
            surviving_nodes.insert(new_vertex_id);
            prism_edges[edge_id].vertex2 = new_vertex_id;
            facet_intersections[prism_edges[edge_id].facet1].insert(new_vertex_id);
            facet_intersections[prism_edges[edge_id].facet2].insert(new_vertex_id);
        }
        // if only v2 survives
        else if (!v1_survives && v2_survives)
        {
            PrismVertex vertex_new;
            PrismVertex v1 = prism_nodes[prism_edges[edge_id].vertex1];
            PrismVertex v2 = prism_nodes[prism_edges[edge_id].vertex2];

            vertex_new.facet1 = prism_edges[edge_id].facet1;
            vertex_new.facet2 = prism_edges[edge_id].facet2;
            vertex_new.facet3 = new_facet_id;
            double delta1 = new_facet.a *v1.x
                + new_facet.b *v1.y
                + new_facet.c -v1.h;
            double delta2 = new_facet.a * v2.x
                + new_facet.b * v2.y
                + new_facet.c - v2.h;

            double lambda = (delta1) / (delta1 - delta2);
            vertex_new.x = convex_combination(lambda, v1.x, v2.x);
            vertex_new.y = convex_combination(lambda, v1.y, v2.y);
            vertex_new.z = convex_combination(lambda, v1.z, v2.z);
            prism_nodes.push_back(vertex_new);
            
            int new_vertex_id = prism_nodes.size() - 1;
            surviving_nodes.insert(new_vertex_id);
            prism_edges[edge_id].vertex1 = new_vertex_id;
            facet_intersections[prism_edges[edge_id].facet1].insert(new_vertex_id);
            facet_intersections[prism_edges[edge_id].facet2].insert(new_vertex_id);
        }
        // If none of the survives, remove the edge
        else
        {
            edges_to_remove.insert(edge_id);
        }
    }
    for (auto edge_id : edges_to_remove)
        surviving_edges.erase(edge_id);


    for (auto mypair : facet_intersections)
    {
        PrismEdge new_edge;
        new_edge.vertex1 = *mypair.second.begin();
        new_edge.vertex2 = *mypair.second.rbegin();
        new_edge.facet2 = new_facet_id;
        if (prism_nodes[new_edge.vertex1].facet1 == prism_nodes[new_edge.vertex2].facet1)
        {
            new_edge.facet1 = prism_nodes[new_edge.vertex1].facet1;
        }
        else if (prism_nodes[new_edge.vertex1].facet1 == prism_nodes[new_edge.vertex2].facet2)
        {
            new_edge.facet1 = prism_nodes[new_edge.vertex1].facet1;
        }
        else if (prism_nodes[new_edge.vertex1].facet2 == prism_nodes[new_edge.vertex2].facet1)
        {
            new_edge.facet1 = prism_nodes[new_edge.vertex1].facet2;
        }
        else 
        {
            new_edge.facet1 = prism_nodes[new_edge.vertex1].facet2;
        }
        prism_edges.push_back(new_edge);
        surviving_edges.insert(prism_edges.size() - 1);
    }
}
}   // namespace core
}   // namespace fdapde


#endif // PRISM_H
