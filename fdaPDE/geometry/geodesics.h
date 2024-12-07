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

#ifndef __GEODESICS_H__
#define __GEODESICS_H__

#include <vector> 
#include <Eigen/Dense> 

namespace fdapde {
namespace core {

template<int N>
class Geodesic {
    private:
    
        SVector<N> S; // Source 
        SVector<N> D; // Destination 
        std::vector<SVector<N>> G; // Vector of intermediate points

    public:
        // Default constructor
        Geodesic() {}
        // Specified constructor 
        Geodesic(const SVector<N>& source, const SVector<N>& destination, const std::vector<SVector<N>>& intermediates)
            : S(source), D(destination), G(intermediates) {}
        // Getters
        SVector<N> source() const { return S; }
        SVector<N> destination() const { return D; }

        void set_source(const SVector<N>& s) { S = s; }
        void set_destination(const SVector<N>& d) { D = d; }

        // Add a point to the geodesic 
        void push_back(const SVector<N>& point) {
            G.push_back(point);
        }

        // Calculate the length of the geodesic
        double length() const {

            double temp_length = 0.0;

            if (!G.empty()) {

                temp_length += (S - G.front()).squaredNorm();

                for (size_t j = 1; j < G.size(); ++j) {
                    temp_length += (G[j] - G[j - 1]).squaredNorm();
                }

                temp_length += (D - G.back()).squaredNorm();

            } else {
                // Handle degenerate case
                temp_length = (S - D).squaredNorm();
            }

            return temp_length;
        }
};

}   // namespace core
}   // namespace fdapde

#endif   // __GEODESICS_H__