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
double MAX_HEIGHT = 100000;

namespace fdapde
{
namespace core
{
// We need to define a struct that stores a prism with triangular base and a conve polytope at top.
struct TriangularPrism
{
    MeshType mesh; // mesh
    Eigen::Svector<2> node_1, node_2, node_3; // triangle vertex projected in 2D
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
        : ancestor(ancestor)
    {
        is_wall = false;
        Eigen::Matrix3d m;
        m << prism->x1, prism->y1, 1,
            prism->x2, prism->y2, 1,
            prism->x3, prism->y3, 1;
        Eigen::Vector3d rhs;
        rhs << h1, h2, h3;
        auto res = m.inverse() * rhs;
        this->a = res(0);
        this->b = res(1);
        this->c = res(2);
    }
    // Aux function 
    bool is_above(double x, double y, double z) const
    {
        return z > a * x + b * y + c;
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
	std::vector<Eigen::SVector<2>> vertices; // Ordered vertices

	// Compute the area of the polygon
	double area() const
	{
		double a = 0.0;
		size_t n = vertices.size();
		for (size_t i = 0; i < n; ++i)
		{
			const Eigen::SVector<2> p1 = vertices[i];
			const Eigen::SVector<2> p2 = vertices[(i + 1) % n];
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
                mesh(// TODO), base_id(-1)
{
    node_1 = Eigen::Svector<2>(x1, y1);
    node_2 = Eigen::Svector<2>(x2, y2);
    node_3 = Eigen::Svector<2>(x3, y3);
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

    // Add the 6 vertices of the prism: 0,1,2 are the top vertices, 3,4,5 are the bottom vertices

    // Top vertices with coordinates (x_i,y_i,MAX_HEIGHT)
    prism_nodes.push_back(PrismVertex(2, 0, 3, x1, y1, MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    prism_nodes.push_back(PrismVertex(0, 1, 3, x2, y2, MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    prism_nodes.push_back(PristVertex(1, 2, 3, x3, y3, MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);

    // Top vertices with coordinates (x_i,y_i, -MAX_HEIGHT)
    prism_nodes.push_back(PrismVertex(2, 0, -1, x1, y1, -MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    prism_nodes.push_back(PrismVertex(0, 1, -1, x2, y2, -MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    prism_nodes.push_back(PrismVertex(1, 2, -1, x3, y3, -MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);

    // Now we add the edges: 0-1, 1-2, 2-0, 0-3, 1-4, 2-5 
    // The induced labeling:  0    1    2    3    4    5
    prism_edges.push_back(PrismEdge(0, 1, 0, 3));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(1, 2, 1, 3));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(2, 0, 2, 3));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(0, 3, 2, 0));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(1, 4, 0, 1));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(2, 5, 1, 2));
    surviving_edges.insert(prism_edges.size() - 1);
}
// Triangular prism constructor from 
TriangularPrism(const MeshType& mesh, int base_id) : base_id(base_id), mesh(mesh)
{   
    // Build the triangular base in 2D: (0,0) (x2,0) (x3,y3)
    double x1(0), y1(0);
    node_1 = Eigen::Svector<2>(x1, y1);
    // Take the base_edge: will connect (0,0) to (x2,0)
    double x2 = mesh.cells(base_id).edges(0).measure();
    node_2 = Eigen::SVector<2>(x2, 0);
    // TODO: fix this
    double x3 = mesh.Edge(edge_id).coordOfOppositeVert.x();
    double y3 = mesh.Edge(edge_id).coordOfOppositeVert.y();
    node_3 = Eigen::Svector<2>(x3, y3);

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

    // Add the 6 vertices of the prism: 0,1,2 are the top vertices, 3,4,5 are the bottom vertices
    // Top vertices with coordinates (x_i,y_i,MAX_HEIGHT)
    prism_nodes.push_back(PrismVertex(2, 0, 3, x1, y1, MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    prism_nodes.push_back(PrismVertex(0, 1, 3, x2, y2, MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    prism_nodes.push_back(PristVertex(1, 2, 3, x3, y3, MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    // Top vertices with coordinates (x_i,y_i, -MAX_HEIGHT)
    prism_nodes.push_back(PrismVertex(2, 0, -1, x1, y1, -MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    prism_nodes.push_back(PrismVertex(0, 1, -1, x2, y2, -MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);
    prism_nodes.push_back(PrismVertex(1, 2, -1, x3, y3, -MAX_HEIGHT));
    surviving_nodes.insert(prism_nodes.size() - 1);

    // Now we add the edges: 0-1, 1-2, 2-0, 0-3, 1-4, 2-5 
    // The induced labeling:  0    1    2    3    4    5
    prism_edges.push_back(PrismEdge(0, 1, 0, 3));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(1, 2, 1, 3));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(2, 0, 2, 3));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(0, 3, 2, 0));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(1, 4, 0, 1));
    surviving_edges.insert(prism_edges.size() - 1);
    prism_edges.push_back(PrismEdge(2, 5, 1, 2));
    surviving_edges.insert(prism_edges.size() - 1);
};

// Now that everything is set up, we provide an
// implementation of the incremental cutting method used in the paper
// Input: new_facet to cut the prism with
// Output: internally updates the prism structure
void incremental_cutting(const Facet& new_facet)
{
    // Init a set to store the vertex to remove, for fast insertion/deletion
    std::set<int> vertex_to_remove;
    // Loop over all the surviving nodes
    for (auto v : surviving_nodes)
    {
        auto v_id = prism_nodes[v];
        // Check if: h > a * x + b * y + c
        if (new_facet.is_above(v_id->x, v_id->y, v_id->h))
            vertex_to_remove.insert(v);
    }
    if (vertex_to_remove.empty())
        return;
    for (auto v : vertex_to_remove)
        surviving_nodes.erase(v);

    // Append the new facet
    pi.push_back(new_facet);
    int new_facet_id = pi.size() - 1;

    // map to store the intersections of the new facet with the existing facets
    std::map<int, std::setset<int>> facet_intersections; 
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
            vertex_new.facet1 = prism_edges[edge_id].facet1;
            vertex_new.facet2 = prism_edges[edge_id].facet2;
            vertex_new.facet3 = new_facet_id;

            double delta1 = new_facet.a * prism_nodes[prism_edges[edge_id].vertex1].x
                + new_facet.b * prism_nodes[prism_edges[edge_id].vertex1].y
                + new_facet.c - prism_nodes[prism_edges[edge_id].vertex1].h;

            double delta2 = new_facet.a * prism_nodes[prism_edges[edge_id].vertex2].x
                + new_facet.b * prism_nodes[prism_edges[edge_id].vertex2].y
                + new_facet.c - prism_nodes[prism_edges[edge_id].vertex2].h;

            double lambda = (delta1 - 0) / (delta1 - delta2);
            vertex_new.x = (1 - lambda) * prism_nodes[prism_edges[edge_id].vertex1].x
                + lambda * prism_nodes[prism_edges[edge_id].vertex2].x;
            vertex_new.y = (1 - lambda) * prism_nodes[prism_edges[edge_id].vertex1].y
                + lambda * prism_nodes[prism_edges[edge_id].vertex2].y;
            vertex_new.h = (1 - lambda) * prism_nodes[prism_edges[edge_id].vertex1].h
                + lambda * prism_nodes[prism_edges[edge_id].vertex2].h;
            prism_nodes.push_back(vertex_new);
	    /*
            if (isnan(vertex_new.x) || isnan(vertex_new.y) || isnan(vertex_new.h)) {
                cout << "Algorithm failed" << endl;
            }
	    */
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
            vertex_new.facet1 = prism_edges[edge_id].facet1;
            vertex_new.facet2 = prism_edges[edge_id].facet2;
            vertex_new.facet3 = new_facet_id;
            double delta1 = new_facet.a * prism_nodes[prism_edges[edge_id].vertex1].x
                + new_facet.b * prism_nodes[prism_edges[edge_id].vertex1].y
                + new_facet.c - prism_nodes[prism_edges[edge_id].vertex1].h;
            double delta2 = new_facet.a * prism_nodes[prism_edges[edge_id].vertex2].x
                + new_facet.b * prism_nodes[prism_edges[edge_id].vertex2].y
                + new_facet.c - prism_nodes[prism_edges[edge_id].vertex2].h;

            double lambda = (delta1 - 0) / (delta1 - delta2);
            vertex_new.x = (1 - lambda) * prism_nodes[prism_edges[edge_id].vertex1].x
                + lambda * prism_nodes[prism_edges[edge_id].vertex2].x;
            vertex_new.y = (1 - lambda) * prism_nodes[prism_edges[edge_id].vertex1].y
                + lambda * prism_nodes[prism_edges[edge_id].vertex2].y;
            vertex_new.h = (1 - lambda) * prism_nodes[prism_edges[edge_id].vertex1].h
                + lambda * prism_nodes[prism_edges[edge_id].vertex2].h;


            if (isnan(vertex_new.x) || isnan(vertex_new.y) || isnan(vertex_new.h)) {
                cout << "incremental_cutting produces nan" << endl;
                exit(-1);
            }

            prism_nodes.push_back(vertex_new);
            int new_vertex_id = prism_nodes.size() - 1;
            surviving_nodes.insert(new_vertex_id);
            prism_edges[edge_id].vertex1 = new_vertex_id;
            facet_intersections[prism_edges[edge_id].facet1].insert(new_vertex_id);
            facet_intersections[prism_edges[edge_id].facet2].insert(new_vertex_id);
        }
        // if none of the survives, remove the edge
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
};
}; 
} // namespace core
} // namespace fdapde


#endif // PRISM_H
