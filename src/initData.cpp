#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "geometry/mesh.h"
#include "geometry/shapes.h"
#include "initData.h"
#include "io/reader.h"
#include "utils.h"

void FLB::initFreeSurfaceFlags(std::vector<uint8_t> &h_flags, const bool is3D, const unsigned int numVelocities, const unsigned int idx, const  unsigned int x, const unsigned int y, const unsigned int z, const unsigned int h_Nx, const unsigned int h_Ny, const unsigned h_Nz)
{
  uint8_t& flag = h_flags[idx];
  if (flag & ~(FLB::TypesNodes::FLUID | FLB::TypesNodes::INLET | FLB::TypesNodes::OUTLET)) return;
  if (flag & FLB::TypesNodes::INLET) flag |= FLB::TypesNodes::FLUID;
  else if (flag & FLB::TypesNodes::OUTLET) flag |= FLB::TypesNodes::GAS;
  //std::cout <<x  << " " << y<< " IDX "  << idx <<" AAA\n";
  if (is3D)
  {

  }
  else
  {
    std::vector<uint32_t> neighborsIdx;
    neighborsIdx.reserve(9);
    calculateNeighborsIndexD2Q9(neighborsIdx, idx, h_Nx, h_Ny, x, y);
    // the boundary are not included by defnition of the domain (the flag cannot be fluid)
    for (int i = 1; i < neighborsIdx.size(); i++)
    {  
      uint8_t& neighborFlag = h_flags[neighborsIdx[i]];
      //std::cout << (int)h_flags[neighborsIdx[i]]<< " IDX "  << neighborsIdx[i]<< " \n";
      if (neighborFlag == FLB::TypesNodes::GAS && flag & FLB::FLUID) neighborFlag = FLB::TypesNodes::INTERFACE;
    }
  }
}

void FLB::calculateNeighborsIndexD2Q9(std::vector<uint32_t>& neighborsIdx, const unsigned int idx, const unsigned int h_Nx, const unsigned int h_Ny, const unsigned int x, const unsigned int y)
{
  /* 
     8 3 5  
      ***
     2*0*1
      ***
     6 4 7
  */

  neighborsIdx[0] = idx; //Own cell 
  
  if (x == 0) // left boundary and corners
  {
    neighborsIdx.push_back(idx + 1); // 1 
    if (y == 0)
    {
      neighborsIdx.push_back(idx + h_Nx); // 3
      neighborsIdx.push_back(idx + 1 + h_Nx); // 5
    }
    else if (y == (h_Ny - 1))
    {
      neighborsIdx.push_back(idx - h_Nx); // 4
      neighborsIdx.push_back(idx + 1 - h_Nx); // 7
    }
    else
    {
      neighborsIdx.push_back(idx + h_Nx); // 3
      neighborsIdx.push_back(idx - h_Nx); // 4
      neighborsIdx.push_back(idx + 1 + h_Nx); // 5
      neighborsIdx.push_back(idx + 1 - h_Nx); // 7
    }
  }

  else if (x == (h_Nx - 1)) // right boundary and corners
  {
    neighborsIdx.push_back(idx - 1); // 2
    if (y == 0)
    {
      neighborsIdx.push_back(idx + h_Nx); // 3
      neighborsIdx.push_back(idx - 1 + h_Nx); // 8
    }
    else if (y == (h_Ny - 1))
    {
      neighborsIdx.push_back(idx - h_Nx); // 4
      neighborsIdx.push_back(idx - 1 - h_Nx); // 6 
    }
    else
    {
      neighborsIdx.push_back(idx + h_Nx); // 3
      neighborsIdx.push_back(idx - h_Nx); // 4
      neighborsIdx.push_back(idx - 1 - h_Nx); // 6 
      neighborsIdx.push_back(idx - 1 + h_Nx); // 8
    }
  }

  else if (y == 0) // lower boundary
  {
    neighborsIdx.push_back(idx + 1); // 1 
    neighborsIdx.push_back(idx - 1); // 2
    neighborsIdx.push_back(idx + h_Nx); // 3
    neighborsIdx.push_back(idx + 1 + h_Nx); // 5
    neighborsIdx.push_back(idx - 1 + h_Nx); // 8
  }

  else if (y == (h_Ny - 1)) // upper boundary
  {
    neighborsIdx.push_back(idx + 1); // 1 
    neighborsIdx.push_back(idx - 1); // 2
    neighborsIdx.push_back(idx - h_Nx); // 4
    neighborsIdx.push_back(idx - 1 - h_Nx); // 6 
    neighborsIdx.push_back(idx + 1 - h_Nx); // 7
  }

  else // general case not in boundary 
  {
    neighborsIdx.push_back(idx + 1); // 1 
    neighborsIdx.push_back(idx - 1); // 2
    neighborsIdx.push_back(idx + h_Nx); // 3
    neighborsIdx.push_back(idx - h_Nx); // 4
    neighborsIdx.push_back(idx + 1 + h_Nx); // 5
    neighborsIdx.push_back(idx - 1 - h_Nx); // 6 
    neighborsIdx.push_back(idx + 1 - h_Nx); // 7
    neighborsIdx.push_back(idx - 1 + h_Nx); // 8
  }
}

