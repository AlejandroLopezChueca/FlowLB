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
  template<typename PRECISION>
  using initFieldsData = void(*)(std::vector<PRECISION>&, std::vector<PRECISION>&, const std::vector<PRECISION>&, const std::vector<uint8_t>&, std::vector<PRECISION>&, const bool, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, FLB::OptionsCalculation*);

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
  template<typename PRECISION, initFieldsData<PRECISION> initFieldsValues>
  void initData(const unsigned int numVelocities, FLB::Mesh* mesh, std::vector<PRECISION>& h_f, std::vector<PRECISION>& h_u, const std::vector<PRECISION>& h_weights, std::vector<PRECISION>& h_rho, FLB::OptionsCalculation* optionsCalculation = nullptr);

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
    //printf("11111 local0 = %6.4f local1 = %6.4f local2 = %6.4f local3 = %6.4f local4 = %6.4f local5 = %6.4f local6 = %6.4f local7 = %6.4f local8 = %6.4f \n",h_f[idx],  h_f[idx +numberPoints* 1], h_f[idx + numberPoints*2], h_f[idx + numberPoints*3], h_f[idx+numberPoints*4], h_f[idx+numberPoints*5], h_f[idx+numberPoints*6], h_f[idx+numberPoints*7], h_f[idx+numberPoints*8]);
  }

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
      h_u[idx] = optionsCalculation -> LBVelocity;

    }
    else
    {
      if (h_flags[idx] == FLB::TypesNodes::FLUID || h_flags[idx] == FLB::TypesNodes::OUTLET) h_u[idx] = optionsCalculation ->LBVelocity;
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
    // TODO Correct for 3D Analysis
    initEquilibriumDDF2Q9(h_f, h_weights, h_u[idx], h_u[idx + h_N], rho, idx, h_N);
  };

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

  //inline float getBaseCoordinatez(float xSI, float ySi, float zSi, unsigned int idxEndPointBase, std::vector<Point>& points)
  //{


  //};

  template<typename PRECISION>
  inline void voidInitFields(std::vector<PRECISION>& h_f, std::vector<PRECISION>& h_u, const std::vector<PRECISION>& h_weights, const std::vector<uint8_t>& h_flags, std::vector<PRECISION>& h_rho, const bool is3D, const unsigned int numVelocities, const unsigned int idx, const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int h_N, FLB::OptionsCalculation* optionsCalculation)
  {
  };
}

