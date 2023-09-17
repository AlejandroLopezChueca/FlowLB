#include "mesh.h"
#include <cmath>
#include <iostream>


FLB::Mesh::Mesh(): xMax(0), xMin(0), yMax(0), yMin(0), zMax(0), zMin(0), is3D(false), separationCDW(0.0f), isFreeSurface(false) {};

void FLB::Mesh::getCoordinatesPoints()
{
  unsigned int idx; 
  int count = 0;

  for (unsigned int z = 0; z < Nz; z++)
  {
    for (unsigned int y = 0; y < Ny; y++)
    {
      for (unsigned int x = 0; x < Nx; x++)
      {
	//idx = x + (y + z * Ny) * Nx;
	//coordinatesPoints[idx + 2 * count] = x * sizeInterval;
	coordinatesPoints.push_back(x * sizeInterval);
	//coordinatesPoints[idx + 2 * count + 1] = y * sizeInterval;
	coordinatesPoints.push_back(y * sizeInterval);
	//coordinatesPoints[idx + 2 * count + 2] = z * sizeInterval; 
	coordinatesPoints.push_back(z * sizeInterval); 
	//count += 1; 
      }
    }
  }
};

void FLB::Mesh::getNumberPointsMesh()
{
  Nx = std::floor((xMax - xMin)/sizeInterval) + 1;
  Ny = std::floor((yMax - yMin)/sizeInterval) + 1; 
  Nz = zMax == 0 ? 1 : std::floor((zMax - zMin)/sizeInterval) + 1;
  // It is assumed taht the domain is a rectangular cuboid
  numPointsMesh = Nx * Ny * Nz;
};

void FLB::Mesh::getIndicesCorners()
{
  m_indicesCorners[0] = 0;
  m_indicesCorners[1] = Nx;
  m_indicesCorners[2] = Nx * Ny;
  m_indicesCorners[3] = numPointsMesh; 
}

void FLB::Mesh::reserveMemory()
{
  domainFlags.reserve(numPointsMesh);
  coordinatesPoints.reserve(3 * numPointsMesh);
}

void FLB::Mesh::clear()
{
  obstacles.clear();
  CDWs.clear();
  points.clear();
  domainFlags.clear();
  coordinatesPoints.clear();
}



