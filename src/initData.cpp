#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "geometry/mesh.h"
#include "geometry/shapes.h"
#include "initData.h"
#include "io/reader.h"
#include "utils.h"



template<typename PRECISION, FLB::initVAndf<PRECISION> initVelocityAndDistributionFuntion>
void FLB::initData(unsigned int numVelocities, FLB::Mesh* mesh, PRECISION* h_f, PRECISION* h_ux, PRECISION* h_uy, PRECISION* h_weights, PRECISION* h_uz)
{
  auto& h_flags = mesh -> domainFlags;

  std::vector<FLB::Point>& points = mesh -> points; // points that define the line of the cross drainge works 
  auto& cdws = mesh -> CDWs; //cross drainage works
  std::vector<FLB::Shape> obstacles = mesh -> obstacles;
			     //
  float initPointCDWxSI = -1.0f;
  float initPointCDWySI = -1.0f;
  float initPointCDWzSI = -1.0f;
  float endPointCDWxSI = -1.0f;
  float endPointCDWySI = -1.0f;
  float endPointCDWzSI = -1.0f;
  // If there are points, the true values are taken
  if (points.size() > 0)
  {
    float initPointCDWxSI = points[mesh -> initPointCDW].x;
    float initPointCDWySI = points[mesh -> initPointCDW].y;
    float initPointCDWzSI = points[mesh -> initPointCDW].z;
    float endPointCDWxSI = points[mesh -> endPointCDW].x;
    float endPointCDWySI = points[mesh -> endPointCDW].y;
    float endPointCDWzSI = points[mesh -> endPointCDW].z;
  }

  unsigned int h_Nx = mesh -> Nx; 
  unsigned int h_Ny = mesh -> Ny; 
  unsigned int h_Nz = mesh -> Nz;
  float sizeInterval = mesh -> sizeInterval;
  float xSI = 0.0f, ySI = 0.0f, zSI = 0.0f; //coordinates in SI units;

  unsigned int idx;
  unsigned int idxEndPointBase;
  float baseCoordinateySI = 0.0f;
  bool is3D = mesh -> is3D;
  int defaulValueNode;
  if (mesh -> isFreeSurface) defaulValueNode = FLB::TypesNodes::GAS;
  else defaulValueNode = FLB::TypesNodes::FLUID;
  for (unsigned int z = 0; z < h_Nz; z++)
  {
    for (unsigned int y = 0; y < h_Ny; y++)
    {
      for (unsigned int x = 0; x < h_Nx; x++)
      {
	idx = x + (y + z * h_Ny) * h_Nx;
	h_flags.push_back(defaulValueNode);
	std::cout<<idx<<"\n";
	initVelocityAndDistributionFuntion(h_f, h_ux, h_uy, h_uz, h_weights, is3D, numVelocities, idx, x, y, z, h_Nx, h_Ny);

	xSI = x * sizeInterval;
	ySI = y * sizeInterval;
	zSI = z * sizeInterval;
	//h_ux[idx] = 0;
	//h_uy[idx] = 0;
	//if (is3D) h_uz[idx] = 0;
	//// Init distribution function as a SoA for tthe GPU, to achive coalescense read
	//for (int i = 0; i < numVelocities; i++)
	//{
	//  h_f[(x + h_Nx * y + z * h_Nx * h_Ny) * numVelocities + i] = h_weights[i]; 

	//}

	// Initialization of the flags in the CDWs
	// Check if node is inside the fluid domain of the cross drainage works

        // Only if the node is between the points that mark the beginning and the end of the CDW
	if (xSI >= initPointCDWxSI  && xSI <= endPointCDWxSI && ySI >= initPointCDWySI && ySI <= endPointCDWySI && zSI >= initPointCDWzSI && zSI <= endPointCDWzSI) 
	{	
	//for (std::unique_ptr(FLB::CDW)&  cdw : cdws) 
	//{
	//  if (cdw.isNodeInside(yDiff, zDiff)) h_flags[idx] = FLB::TypesNodes::GAS;
	//}
	}
	//Initialization of the flags in the domain that are not inside the CDWs, before the first point of tyhe CDW or after the end point of the CDWs
	else	
	{
	  idxEndPointBase = getEndPointBaseCoordinate(xSI, points, sizeInterval);
	  baseCoordinateySI = getBaseCoordinatey(xSI,idxEndPointBase,points);
	  if (ySI <= baseCoordinateySI) h_flags[idx] = FLB::TypesNodes::WALL;
	}


      // initialization of the nodes inside the obstacles
	for (FLB::Shape& obstacle : obstacles)
	{

	}

	//Apply input boundary
      }
    }
  }
  // initialization of the nodes inside the obstacles
  //for (auto obstacle : obstacles) obstacle -> initDomainShape(h_flags, FLB::TypesNodes::WALL);
};

template void FLB::initVelocityAndf<float>(float* h_f, float* h_ux, float* h_uy, float* h_uz, float* h_weights, bool is3D, unsigned int numVelocities, const unsigned int idx, unsigned int x, unsigned int y, unsigned int z, unsigned int h_Nx, unsigned int h_Ny);

template void FLB::initData<float, FLB::initVelocityAndf<float>>(unsigned int numVelocities, FLB::Mesh *mesh, float* h_f, float* h_ux, float* h_uy, float* h_weights, float* h_uz);

template void FLB::initData<float, FLB::voidInitVelocityAndf<float>>(unsigned int numVelocities, FLB::Mesh *mesh, float* h_f = nullptr, float* h_ux = nullptr, float* h_uy = nullptr, float* h_weights = nullptr, float* h_uz = nullptr);



