#pragma once
//#include "cuda/initData.cuh"
#include "utils.h"
#include "geometry/shapes.h"
//#include "geometry/mesh.h"
#include "io/reader.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <memory>


namespace FLB
{ 
  /**
   * @brief Obtaining the index of the point with a x coordinate equal or superior to the XSI coordinate
   * @param[in]  xSi           coordinate x in SI units
   * @param[in]  points        vectors whith all the points that define the domain 
   * @param[in]  sizeInterval  vectors whith all the points that define the domain 
   * @return     Index of the point
   */
  inline uint32_t getEndPointBaseCoordinate(double xSI, std::vector<FLB::Point>& points)
  {
    // The first point is avoided as it is the initial point
    uint32_t cont = 1;
    //for (FLB::Point& point : points)
    for (std::vector<FLB::Point>::iterator point = points.begin() + 1; point != points.end(); ++point)
    {
      if (point -> x >= (xSI - 1e-12)) return cont;
      cont += 1;
    }
    return cont;
  };

  /**
   * @brief Obtaining the y coordinate of the axis which defines the domain
   * @param[in]  xSi              coordinate x in SI units
   * @param[in]  idxEndPointBase  index of the second point that defines the line where is the coordinate to obtain
   * @param[in]  points           vectors whith all the points that define the domain 
   * @return     The y coordinate of the axis that defines the domain
   */
  inline double getBaseCoordinatey(double xSI, unsigned int idxEndPointBase, std::vector<FLB::Point>& points)
  {
    double slope = (points[idxEndPointBase].y - points[idxEndPointBase - 1].y)/(points[idxEndPointBase].x - points[idxEndPointBase - 1].x);
    return points[idxEndPointBase - 1].y + (xSI - points[idxEndPointBase - 1].x) * slope;
  };

  /**
   *
    * @brief Initialization o the DDF with the equilibrium distribution function for D2Q9 
   *
   *
   */
  template<typename PRECISION>
  inline void initEquilibriumDDF2Q9(std::vector<PRECISION>& h_f, const std::vector<PRECISION>& h_weights, PRECISION ux, PRECISION uy, const double rho, const unsigned int idx, const unsigned numberPoints)
  {
    const double u2 = 3.0 * (ux * ux + uy * uy);
    const double rhow14 = h_weights[1] * rho;
    const double rhow58 = h_weights[5] * rho;
    ux *= 3.0;
    uy *= 3.0;
    const double u0 = ux - uy;
    const double u1 = ux + uy;
    
    h_f[idx] = h_weights[0] * rho * (1.0 - 0.5 * u2);

    h_f[idx + numberPoints * 1] = rhow14 * (1.0 + ux - 0.5 * u2 + 0.5 * ux * ux);
    h_f[idx + numberPoints * 2] = rhow14 * (1.0 - ux - 0.5 * u2 + 0.5 * ux * ux);
    h_f[idx + numberPoints * 3] = rhow14 * (1.0 + uy - 0.5 * u2 + 0.5 * uy * uy);
    h_f[idx + numberPoints * 4] = rhow14 * (1.0 - uy - 0.5 * u2 + 0.5 * uy * uy);
    h_f[idx + numberPoints * 5] = rhow58 * (1.0 + u1 - 0.5 * u2 + 0.5 * u1 * u1);
    h_f[idx + numberPoints * 6] = rhow58 * (1.0 - u1 - 0.5 * u2 + 0.5 * u1 * u1);
    h_f[idx + numberPoints * 7] = rhow58 * (1.0 + u0 - 0.5 * u2 + 0.5 * u0 * u0);
    h_f[idx + numberPoints * 8] = rhow58 * (1.0 - u0 - 0.5 * u2 + 0.5 * u0 * u0);
  }
  
  template<typename PRECISION>
  using initFieldsData = void(*)(std::vector<PRECISION>&, std::vector<PRECISION>&, const std::vector<PRECISION>&, const std::vector<uint8_t>&, std::vector<PRECISION>&, const bool, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, FLB::OptionsCalculation*);
  
  
  /**
   * @brief Initialization of the fields used in the calculation 
   *
   * @param[in]
   *
   *
   */
  template<typename PRECISION>
  inline void initFields(std::vector<PRECISION>& h_f, std::vector<PRECISION>& h_u, const std::vector<PRECISION>& h_weights, const std::vector<uint8_t>& h_flags, std::vector<PRECISION>& h_rho, const bool is3D, const unsigned int numVelocities, const unsigned int idx, const  unsigned int x, const unsigned int y, const unsigned int z, const unsigned int h_N, FLB::OptionsCalculation* optionsCalculation)
  {
    // Apply input boundary
    if (x == 0 && h_flags[idx] == FLB::TypesNodes::INLET)
    {
      h_u[idx] = optionsCalculation -> LBVelocity.x;
    }
    else
    {
      if (h_flags[idx] == FLB::TypesNodes::OUTLET) h_u[idx] = optionsCalculation -> LBVelocity.x;
      else h_u[idx] = 0; // x value
    }
    h_u[idx + h_N] = 0; // y value
    if (is3D) h_u[idx + 2 * h_N] = 0; // z value
    // Init distribution function as a SoA for the GPU, to achive coalescense read
    double rho = 1.0; 
    /*for (int i = 0; i < numVelocities; i++)
    {
      //int idx2 = (x + h_Nx * y + z * h_Nx * h_Ny) * numVelocities + i; 
      //int idx2 = idx * numVelocities + i; 
      //h_f[idx2] = h_weights[i];
      //rho += h_weights[i];
    }*/

    h_rho[idx] = rho;

    if (optionsCalculation ->typeProblem == FLB::TypeProblem::FREE_SURFACE)
    {

    }

    // TODO Correct for 3D Analysis
    initEquilibriumDDF2Q9(h_f, h_weights, h_u[idx], h_u[idx + h_N], rho, idx, h_N);
  };

  /**
   * @brief
   *
   */ 
  void initFreeSurfaceFlags(std::vector<uint8_t>& h_flags, const bool is3D, const unsigned int numVelocities, const unsigned int idx, const  unsigned int x, const unsigned int y, const unsigned int z, const unsigned int h_Nx, const unsigned int h_Ny, const unsigned h_Nz);

  /**
   *
   *
   */
  void calculateNeighborsIndexD2Q9(std::vector<uint32_t>& neighborsIdx, const unsigned int idx, const unsigned int h_Nx, const unsigned int h_Ny, const unsigned int x, const unsigned int y);
  
  template<typename PRECISION>
  using initFreeSurfaceFieldsData = void(*) (const std::vector<uint8_t>&, std::vector<PRECISION>&, std::vector<PRECISION>&, std::vector<PRECISION>&, const unsigned int);
  
  /**
   * @brief Initialization of the fields used in a free surface domain
   *
   */
  template<typename PRECISION>
  inline void initFreeSurfaceFields(const std::vector<uint8_t>& h_flags, std::vector<PRECISION>& h_phi, std::vector<PRECISION>& h_mass, std::vector<PRECISION>& h_excessMass, const unsigned int idx)
  {
    switch (h_flags[idx]) 
    {
      case FLB::TypesNodes::FLUID:
      case FLB::TypesNodes::FLUID_INLET:
      case FLB::TypesNodes::FLUID_OUTLET:
      {
	h_phi[idx] = 1.0;
	h_mass[idx] = 1.0; // it is assumed that the density is 1.0
	break;
      }

      case FLB::TypesNodes::GAS:
      case FLB::TypesNodes::GAS_OUTLET:
      {
	h_phi[idx] = 0.0;
	h_mass[idx] = 0.0;
	break;
      }
      
      case FLB::TypesNodes::INTERFACE:
      {
	h_phi[idx] = 0.5;
	h_mass[idx] = 0.5;
	break;
      }
    }

    h_excessMass[idx] = 0.0;
  }

  /**
   * @brief initialization of the domain 
   *
   * @param[in, out]  h_ux           velocity in the x direction
   * @param[in, out]  h_uy           velocity in the y direction
   * @param[in, out]  h_Nx           Number of nodes in the x direction
   * @param[in, out]  h_Ny           Number of nodes in the y direction
   * @param[in]       numVelocities  Number of directions of velocities
   * @param[in]       h_weights      Weights epmployess in the LBM
   * @param[in]       mesh           Mesh of the domain
   * @param[in, out]  h_f             probability density function of the particle
   * @param[in]       h_Nz           Number of nodes in the z direction. 1 default for 2D cases
   * @param[in, out]  h_uz          velocity in the z direction. null pointer deafult for 2D cases
   */
  template<typename PRECISION, initFieldsData<PRECISION> initFieldsValues, initFreeSurfaceFieldsData<PRECISION> initFreeSurfaceFieldsValues>
  void initData(const unsigned int numVelocities, FLB::Mesh* mesh, std::vector<PRECISION>& h_f, std::vector<PRECISION>& h_u, const std::vector<PRECISION>& h_weights, std::vector<PRECISION>& h_rho, std::vector<PRECISION>& h_phi, std::vector<PRECISION>& h_mass, std::vector<PRECISION>& h_excessMass, FLB::OptionsCalculation* optionsCalculation = nullptr)
  {
    std::vector<uint8_t>& h_flags = mesh -> getDomainFlags();

    std::vector<FLB::Point>& points = mesh -> getPoints(); // points that define the line of the cross drainage works 
    const std::vector<std::unique_ptr<FLB::CDW>>& cdws = mesh -> getCDWs(); //cross drainage works
    const std::vector<std::unique_ptr<FLB::Shape>>& obstacles = mesh -> getObstacles();
    const std::vector<std::unique_ptr<FLB::Shape>>& initialConditions = mesh -> getInitialConditionShapes();

    // to prevent error due to precision
    double epsilon = 1e-12;
			       
    // If there are points, the true values are taken 
    double initPointCDWxSI = points.size() > 0 ? points[mesh -> getIdxInitPointCDW()].x - epsilon : -1.0;
    double initPointCDWySI = points.size() > 0 ? points[mesh -> getIdxInitPointCDW()].y - epsilon : -1.0;
    double initPointCDWzSI = points.size() > 0 ? points[mesh -> getIdxInitPointCDW()].z - epsilon: -1.0;
    double endPointCDWxSI = points.size() > 0 ? points[mesh -> getIdxEndPointCDW()].x + epsilon : -1.0;
    double endPointCDWySI = points.size() > 0 ? points[mesh -> getIdxEndPointCDW()].y + epsilon : -1.0;
    double endPointCDWzSI = points.size() > 0 ? points[mesh -> getIdxEndPointCDW()].z + epsilon : -1.0;
    
    unsigned int h_Nx = mesh -> getNx(); 
    unsigned int h_Ny = mesh -> getNy(); 
    unsigned int h_Nz = mesh -> getNz();
    double sizeInterval = mesh -> getSizeInterval();
    double xSI = 0.0, ySI = 0.0, zSI = 0.0; //coordinates in SI units;
    
    const unsigned int countNodes = mesh -> is3D() ? h_Nx * h_Ny * h_Nz : h_Nx * h_Ny;

    double xMin = mesh -> getxMin();
    double yMin = mesh -> getyMin();
    double zMin = mesh -> getzMin();

    double baseCoordinateySI = 0.0;
    const bool is3D = mesh -> is3D();
    const uint8_t defaulValueNode = optionsCalculation -> typeProblem == FLB::TypeProblem::FREE_SURFACE ? FLB::TypesNodes::GAS : FLB::TypesNodes::FLUID;
    for (unsigned int z = 0; z < h_Nz; z++)
    {
      for (unsigned int y = 0; y < h_Ny; y++)
      {
	for (unsigned int x = 0; x < h_Nx; x++)
	{
	  uint32_t idx = x + (y + z * h_Ny) * h_Nx;
	  h_flags.push_back(idx);

	  // not using the coordinates calculated of the mesh because they are type float (to load in VRAM)
	  xSI = xMin + x * sizeInterval;
	  ySI = yMin + y * sizeInterval;
	  zSI = zMin + z * sizeInterval;

	  // end index of the two points beetween which the node is located
	  uint32_t idxEndPointBase = getEndPointBaseCoordinate(xSI, points);
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
	  // Correct flags in boundary. TODO Revise to generalice better
	  if (ySI <= (baseCoordinateySI + epsilon) && h_flags[idx] == defaulValueNode) h_flags[idx] = optionsCalculation -> boundaryDown;
	  else if (ySI >= (mesh ->getyMax() - epsilon)&& h_flags[idx] == defaulValueNode) h_flags[idx] = optionsCalculation -> boundaryUp;
	  else if (x == 0 && h_flags[idx] == defaulValueNode) h_flags[idx] = optionsCalculation -> boundaryLeft;
	  else if (x == (h_Nx - 1) && h_flags[idx] == defaulValueNode) h_flags[idx] = optionsCalculation -> boundaryRight;

	  // initialization of the nodes inside the obstacles
	  for (const std::unique_ptr<FLB::Shape>& obstacle : obstacles)
	  {
	    if (obstacle -> isNodeInside(xSI, ySI, zSI)) h_flags[idx] = FLB::TypesNodes::WALL;
	  }

	  // initialization of the nodes inside the initial conditions
	  for (const std::unique_ptr<FLB::Shape>& initialCondition : initialConditions)
	  {
	    if (initialCondition -> isNodeInside(xSI, ySI, zSI)) h_flags[idx] = FLB::TypesNodes::FLUID;
	  }

	  // init fields when all the flags are indicated	
	  initFieldsValues(h_f, h_u, h_weights, h_flags, h_rho, is3D, numVelocities, idx, x, y, z, countNodes, optionsCalculation);
	}
      }
    }

    // init free surface variables
    if (optionsCalculation -> typeProblem != FLB::TypeProblem::FREE_SURFACE) return;
    for (unsigned int z = 0; z < h_Nz; z++)
    {
      for (unsigned int y = 0; y < h_Ny; y++)
      {
	for (unsigned int x = 0; x < h_Nx; x++)
	{
	  uint32_t idx = x + (y + z * h_Ny) * h_Nx;
	  initFreeSurfaceFlags(h_flags, is3D, numVelocities, idx, x, y, z, h_Nx, h_Ny, h_Nz);
	  initFreeSurfaceFieldsValues(h_flags, h_phi, h_mass, h_excessMass, idx);
	}
      }
    }
  }

  //inline float getBaseCoordinatez(float xSI, float ySi, float zSi, unsigned int idxEndPointBase, std::vector<Point>& points)
  //{


  //};

  template<typename PRECISION>
  inline void voidInitFields(std::vector<PRECISION>& h_f, std::vector<PRECISION>& h_u, const std::vector<PRECISION>& h_weights, const std::vector<uint8_t>& h_flags, std::vector<PRECISION>& h_rho, const bool is3D, const unsigned int numVelocities, const unsigned int idx, const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int h_N, FLB::OptionsCalculation* optionsCalculation)
  {
  };
  
  template<typename PRECISION>
  inline void voidInitFreeSurfaceFields(const std::vector<uint8_t>& h_flags, std::vector<PRECISION>& h_phi, std::vector<PRECISION>& h_mass, std::vector<PRECISION>& h_excessMass, const unsigned int idx)
  {

  };
}

