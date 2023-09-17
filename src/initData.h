#pragma once
//#include "cuda/initData.cuh"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "utils.h"
#include "geometry/shapes.h"
#include "geometry/mesh.h"

namespace FLB
{ 
  template<typename PRECISION>
  using initVAndf = void(*)(PRECISION*, PRECISION*, PRECISION*, PRECISION*, PRECISION*, bool, unsigned int, const unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);

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
  template<typename PRECISION, initVAndf<PRECISION> initVelocityAndDistributionFuntion>
  void initData(unsigned int numVelocities, FLB::Mesh* mesh,PRECISION* h_f = nullptr, PRECISION* h_ux = nullptr, PRECISION* h_uy = nullptr, PRECISION* h_weights = nullptr, PRECISION* h_uz = nullptr);


  /**
   * @brief 
   *
   * @param[in]
   *
   *
   */
  template<typename PRECISION>
  inline void initVelocityAndf(PRECISION* h_f, PRECISION* h_ux, PRECISION* h_uy, PRECISION* h_uz, PRECISION* h_weights, bool is3D, unsigned int numVelocities, const unsigned int idx, unsigned int x, unsigned int y, unsigned int z, unsigned int h_Nx, unsigned int h_Ny)
  {
    std::cout<<numVelocities<<"\n";
    h_ux[idx] = 0;
    h_uy[idx] = 0;
    if (is3D) h_uz[idx] = 0;
    // Init distribution function as a SoA for the GPU, to achive coalescense read
    for (int i = 0; i < numVelocities; i++)
    {
      h_f[(x + h_Nx * y + z * h_Nx * h_Ny) * numVelocities + i] = h_weights[i]; 
    }
  };

  /**
   * @brief Obtaining the index of the point with a x coordinate equal or superior to the XSI coordinate
   * @param[in]  xSi           coordinate x in SI units
   * @param[in]  points        vectors whith all the points that define the domain 
   * @param[in]  sizeInterval  vectors whith all the points that define the domain 
   * @return     Index of the point
   */
  inline unsigned int getEndPointBaseCoordinate(float xSI, std::vector<FLB::Point>& points, float sizeInterval)
  {
    // The first point si avoided as it is the initial point
    unsigned int cont = 1;
    //for (FLB::Point& point : points)
    for (std::vector<FLB::Point>::iterator point = points.begin() + 1; point != points.end(); ++point)
    {
      if (sizeInterval * point -> x >= xSI) return cont;
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
  inline float getBaseCoordinatey(float xSI, unsigned int idxEndPointBase, std::vector<FLB::Point>& points)
  {
    float slope = (points[idxEndPointBase].y - points[idxEndPointBase - 1].y)/(points[idxEndPointBase].x - points[idxEndPointBase - 1].x);
    return points[idxEndPointBase - 1].y + (xSI - points[idxEndPointBase - 1].x) * slope;
  };

  //inline float getBaseCoordinatez(float xSI, float ySi, float zSi, unsigned int idxEndPointBase, std::vector<Point>& points)
  //{


  //};

  template<typename PRECISION>
  inline void voidInitVelocityAndf(PRECISION* h_f, PRECISION* h_ux, PRECISION* h_uy, PRECISION* h_uz, PRECISION* h_weights, bool is3D, unsigned int numVelocities, const unsigned int idx, unsigned int x, unsigned int y, unsigned int z, unsigned int h_Nx, unsigned int h_Ny){};
}

