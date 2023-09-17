#pragma once

#include <cstddef>
#include <cstdint>


#include "io/reader.h"

namespace FLB 
{
  // Lattice velocities
  template<int SIZE>
  __constant__ int d_cx[SIZE];
  template<int SIZE>
  __constant__ int d_cy[SIZE];

  extern template __constant__ int d_cx<9>[9];
  extern template __constant__ int d_cy<9>[9];
  extern __constant__ size_t d_numPointsMesh;

  //Lattice weights
  template<typename PRECISION, int SIZE>
  __constant__ PRECISION d_weights[SIZE];

  extern template __constant__ float d_weights<float, 9>[9];
  extern template __constant__ double d_weights<double, 9>[9];

  // Cell types, boundary conditions and inner conditions 
  extern __constant__ uint8_t CT_FLUID; 
  extern __constant__ uint8_t CT_GAS;
  extern __constant__ uint8_t CT_INTERFACE;
  extern __constant__ uint8_t CT_INTERFACE_FLUID; // Convert interface to fluid
  extern __constant__ uint8_t CT_INTERFACE_GAS; // Convert interface to gas
  extern __constant__ uint8_t CT_GAS_INTERFACE; // Convert gas to interface
  extern __constant__ uint8_t CT_WALL; 
  extern __constant__ uint8_t CT_MWALL; // Moving wall
  extern __constant__ uint8_t CT_OPEN;  // Input or output

  // Constant variables
  extern __constant__ float d_g; //Value of the gravity acceleration in lattice units
  extern __constant__ float d_invTau; // inverse of the relaxation time

  // Size of the domain, number of points in each direction and the total number
  extern __constant__ unsigned int d_Nx; //Size x direction
  extern __constant__ unsigned int d_Ny;
  extern __constant__ unsigned int d_Nz;
  extern __constant__ unsigned int d_N; // Total number of nodes

  // Number of lattice velocities
  //extern __constant__ unsigned int d_numVelocities;

  // Type of collision operator -> 0 (SRT), 1(TRT)
  extern __constant__ uint8_t d_collisionOperator;

  // Simulate surface tneion or not
  extern __constant__ bool d_surfaceTension;

  // Force terms
  extern __constant__ float d_Fx, d_Fy, d_Fz;
 
  /**
    * @brief Initialization of all the constant data needed for a 2D analysis in the GPU.
    *
    * @param[in]
    * @param[in]
    *
    *
    *
    * 
    * @param[in]  h_g   value of the gravity acceleration in lattice units
    * @param[in]  h_nu  kinematic viscosity in lattice units
    */
  template<typename PRECISION>
  void h_initConstantDataCuda2D(FLB::OptionsCalculation& optionsCalc, PRECISION* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, uint8_t h_collisionOperator, float h_g, float h_nu);
  
  /** Initialization of all the non constant data needed for the analysis 2D or 3D in the GPU.
     *
     */  
  template<typename PRECISION>
  void h_initDataCuda2D(FLB::OptionsCalculation& optionsCalc, size_t numPointsMesh, unsigned int numDimensions, unsigned int numVelocities, PRECISION* d_f, PRECISION* h_f, PRECISION* d_vx, PRECISION* h_vx, PRECISION* d_vy, PRECISION* h_vy, uint8_t* d_flags, uint8_t* h_flags, PRECISION* d_mass, PRECISION* h_mass);


//extern template void FLB::h_initConstantDataCuda2D<float>(FLB::OptionsCalculation& OptionsCalc, float* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, unsigned int h_numVelocities, uint8_t h_collisionOperator, float h_g, float h_nu);

}
