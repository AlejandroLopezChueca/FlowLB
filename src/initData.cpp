#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "geometry/mesh.h"
#include "geometry/shapes.h"
#include "initData.h"
#include "io/reader.h"
#include "utils.h"



template<typename PRECISION, FLB::initFieldsData<PRECISION> initFieldsValues>
void FLB::initData(const unsigned int numVelocities, FLB::Mesh* mesh, std::vector<PRECISION>& h_f, std::vector<PRECISION>& h_u, const std::vector<PRECISION>& h_weights, std::vector<PRECISION>& h_rho, FLB::OptionsCalculation* optionsCalculation)
{
  std::vector<uint8_t>& h_flags = mesh -> getDomainFlags();

  std::vector<FLB::Point>& points = mesh -> getPoints(); // points that define the line of the cross drainge works 
  const std::vector<std::unique_ptr<FLB::CDW>>& cdws = mesh -> getCDWs(); //cross drainage works
  const std::vector<std::unique_ptr<FLB::Shape>>& obstacles = mesh -> getObstacles();

  // to prevent error due to precision
  double epsilon = 1e-12;
			     
  double initPointCDWxSI = -1.0;
  double initPointCDWySI = -1.0;
  double initPointCDWzSI = -1.0;
  double endPointCDWxSI = -1.0;
  double endPointCDWySI = -1.0;
  double endPointCDWzSI = -1.0;
  // If there are points, the true values are taken
  if (points.size() > 0)
  {
    initPointCDWxSI = points[mesh -> getIdxInitPointCDW()].x - epsilon;
    initPointCDWySI = points[mesh -> getIdxInitPointCDW()].y - epsilon;
    initPointCDWzSI = points[mesh -> getIdxInitPointCDW()].z - epsilon;
    endPointCDWxSI = points[mesh -> getIdxEndPointCDW()].x + epsilon;
    endPointCDWySI = points[mesh -> getIdxEndPointCDW()].y + epsilon;
    endPointCDWzSI = points[mesh -> getIdxEndPointCDW()].z + epsilon;
  }

  unsigned int h_Nx = mesh -> getNx(); 
  unsigned int h_Ny = mesh -> getNy(); 
  unsigned int h_Nz = mesh -> getNz();
  double sizeInterval = mesh -> getSizeInterval();
  double xSI = 0.0, ySI = 0.0, zSI = 0.0; //coordinates in SI units;
  
  // Reserve flags with number nodes in domain
  unsigned int countNodes = mesh -> is3D() ? h_Nx * h_Ny * h_Nz : h_Nx * h_Ny;
  h_flags.reserve(countNodes);

  double xMin = mesh -> getxMin();
  double yMin = mesh -> getyMin();
  double zMin = mesh -> getzMin();

  uint32_t idx;
  uint32_t idxEndPointBase;
  double baseCoordinateySI = 0.0;
  bool is3D = mesh -> is3D();
  uint8_t defaulValueNode = mesh -> isFreeSurface() ? FLB::TypesNodes::GAS : FLB::TypesNodes::FLUID;
  for (unsigned int z = 0; z < h_Nz; z++)
  {
    for (unsigned int y = 0; y < h_Ny; y++)
    {
      for (unsigned int x = 0; x < h_Nx; x++)
      {
	idx = x + (y + z * h_Ny) * h_Nx;
	h_flags.push_back(defaulValueNode);

	// not using the coordinates calculated of the mesh because they are type float (to load in VRAM)
	xSI = xMin + x * sizeInterval;
	ySI = yMin + y * sizeInterval;
	zSI = zMin + z * sizeInterval;

	// end index of the two points beetween which the node is located
	idxEndPointBase = getEndPointBaseCoordinate(xSI, points);
	baseCoordinateySI = getBaseCoordinatey(xSI, idxEndPointBase, points);

	// Check if node is inside the fluid domain of the cross drainage works
        // Only if the node is between the points that mark the beginning and the end of the CDW
      // TODO Correct for 3D
	//if (xSI >= initPointCDWxSI  && xSI <= endPointCDWxSI && ySI >= initPointCDWySI && ySI <= endPointCDWySI && zSI >= initPointCDWzSI && zSI <= endPointCDWzSI) 
	if (xSI >= initPointCDWxSI  && xSI <= endPointCDWxSI) 
	{	
	  for (const std::unique_ptr<FLB::CDW>&  cdw : cdws) 
	  {
	    if (cdw -> isNodeInside(xSI, ySI, zSI, points, idxEndPointBase)) h_flags[idx] = defaulValueNode;
	    else h_flags[idx] = FLB::TypesNodes::WALL;
	  }
	}
	//Initialization of the flags in the domain that are not inside the CDWs, before the first point of the CDW or after the end point of the CDWs
	//TODO Correct applying boundary conditions
	else	
	{
	  //if (x == 0) h_flags[idx] = optionsCalculation -> boundaryLeft;
	  //else if (x == (h_Nx - 1)) h_flags[idx] = optionsCalculation -> boundaryRight;
	  if (ySI <= (baseCoordinateySI + epsilon)) h_flags[idx] = optionsCalculation -> boundaryDown;
	  else if (ySI >= (mesh ->getyMax() - epsilon)) h_flags[idx] = optionsCalculation -> boundaryUp;
	}
	// Coorect flgas in boundary. TODO Revise to generalice better
	if (ySI <= (baseCoordinateySI + epsilon) && h_flags[idx] == defaulValueNode) h_flags[idx] = optionsCalculation -> boundaryDown;
	else if (ySI >= (mesh ->getyMax() - epsilon)&& h_flags[idx] == defaulValueNode) h_flags[idx] = optionsCalculation -> boundaryUp;
	else if (x == 0 && h_flags[idx] == defaulValueNode) h_flags[idx] = optionsCalculation -> boundaryLeft;
	else if (x == (h_Nx - 1) && h_flags[idx] == defaulValueNode) h_flags[idx] = optionsCalculation -> boundaryRight;


	// initialization of the nodes inside the obstacles
	for (const std::unique_ptr<FLB::Shape>& obstacle : obstacles)
	{
	  if (obstacle -> isNodeInside(xSI, ySI, zSI)) h_flags[idx] = FLB::TypesNodes::WALL;
	}

	// init fiedls when all the flags are indicated
	
	initFieldsValues(h_f, h_u, h_weights, h_flags, h_rho, is3D, numVelocities, idx, x, y, z, countNodes, optionsCalculation);
      }
    }
  }
};

//template void FLB::initFields<float>(std::vector<float>& h_f, std::vector<float>& h_ux, std::vector<float>& h_uy, std::vector<float>& h_uz, const std::vector<float>& h_weights, std::vector<float>& h_rho, const bool is3D, const unsigned int numVelocities, const unsigned int idx, const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int h_Nx, const unsigned int h_Ny, const float flow);

template void FLB::initData<float, FLB::initFields<float>>(const unsigned int numVelocities, FLB::Mesh* mesh, std::vector<float>& h_f, std::vector<float>& h_u, const std::vector<float>& h_weights, std::vector<float>& h_rho, FLB::OptionsCalculation* optionsCalculation);

template void FLB::initData<float, FLB::voidInitFields<float>>(const unsigned int numVelocities, FLB::Mesh* mesh, std::vector<float>& h_f, std::vector<float>& h_ux, const std::vector<float>& h_weights, std::vector<float>& h_rho, FLB::OptionsCalculation* optionsCalculation = nullptr);


