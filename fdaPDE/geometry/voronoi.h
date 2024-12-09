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

#ifndef __VORONOI_H__
#define __VORONOI_H__

#include "triangulation.h"

// per ora senza namespace, todo later
namespace fdaPDE{
namespace core{

using MeshType = Triangulation<2, 3>;
using DistanceType = std::vector<std::tuple<int, double, double, double>>;
using DistanceFieldType = std::vector<DistanceType>;

template <> class Voronoi<Triangulation<2, 3>> {
    
   public:

    static constexpr int local_dim = Triangulation<2,3>::local_dim;
    static constexpr int embed_dim = Triangulation<2,3>::embed_dim;

    // If sources are not specified, we compute the Voronoi wrt 
    // all the nodes, i.e. the dual of the triangulation
    Voronoi(const MeshType& mesh) : mesh_(&mesh) 
    {
        for(size_t i = 0; i <mesh_.n_nodes(); ++i)
        {
            sources_.insert(i) // dato che i node_id vanno da 0 a n_nodes-1
        }
        this->voronoi_partition_ = this->compute_voronoi_partition(this->sources_);
    }

    // If sources are specified, we compute the Voronoi wrt the sources 
    Voronoi(const MeshType& mesh, std::set<int> sources) : mesh_(&mesh), sources_(sources)
    {
        this->voronoi_partition_ = this->compute_voronoi_partition(this->sources_);
    }

    // Cell data structure
    class VoronoiCell {
       private:
        const Voronoi* v_;
        int id_ = 0;
        std::vector<polygon> subcells_;
       public:
        VoronoiCell() = default;
        bool contains(const SVector<embed_dim>& p) const { return v_->locate(p)[0] == id_; }
        double measure() const {
            double m=0;
            for(size_t i =0; i)
        }
    };



};
} // namespace core
} // namespace fdapde


#endif // __VORONOI_H__