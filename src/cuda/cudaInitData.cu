#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iterator>
#include "cudaInitData.cuh"
#include "cudaUtils.cuh"
#include "math/math.h"


template __constant__ int FLB::d_cx<9>[9];
template __constant__ int FLB::d_cy<9>[9];

template __constant__ float FLB::d_weights<float, 9>[9];
template __constant__ double FLB::d_weights<double, 9>[9];

__constant__ size_t FLB::d_numPointsMesh;
__constant__ unsigned int FLB::d_Nx;
__constant__ unsigned int FLB::d_Ny;
__constant__ unsigned int FLB::d_Nz;
__constant__ unsigned int FLB::d_N;
//__constant__ unsigned int FLB::d_numVelocities;
__constant__ uint8_t FLB::d_collisionOperator;

// Types of nodes
__constant__ uint8_t FLB::CT_FLUID = 1 << 0;        // 0000 0001
__constant__ uint8_t FLB::CT_GAS = 1 << 1;          // 0000 0010
__constant__ uint8_t FLB::CT_INTERFACE = 1 << 2;    // 0000 0100
__constant__ uint8_t FLB::CT_FLUID_GAS_INTERFACE = 0x7;  // 0000 0111
__constant__ uint8_t FLB::CT_WALL = 1 << 3;         // 0000 1000  Static wall
__constant__ uint8_t FLB::CT_MWALL = 1 << 4;        // 0001 0000 Moving wall
__constant__ uint8_t FLB::CT_INLET = 1 << 5;        // 0010 0000
__constant__ uint8_t FLB::CT_OUTLET = 1 << 6;       // 0100 0000
__constant__ uint8_t FLB::CT_INLET_FLUID = 0x21;    // 0010 0001
__constant__ uint8_t FLB::CT_INTERFACE_FLUID = 0x5; // 0000 0101
__constant__ uint8_t FLB::CT_INTERFACE_GAS = 0x6;   // 0000 0110
//__constant__ uint8_t FLB::CT_FLUID_INLET =;   // 0000 0110
//__constant__ uint8_t FLB::CT_FLUID_OUTLET =;   // 0000 0110
//__constant__ uint8_t FLB::CT_GAS_OUTLET =;   // 0000 0110
__constant__ uint8_t FLB::CT_GAS_INTERFACE = 1<<7;  // 1000 0000

// constant variables
//__constant__ float FLB::d_g;
__constant__ float FLB::d_invTau;

// Use of grvity in the simulation
__constant__ bool FLB::d_useVolumetricForce;

// Use subgrid model for turbulence in the simulation
__constant__ bool FLB::d_useSubgridModel;

// Simulate surface tneion or not
__constant__ bool FLB::d_surfaceTension;

// Force terms
__constant__ float FLB::d_Fx, FLB::d_Fy, FLB::d_Fz;

template<typename PRECISION>
void FLB::h_initConstantDataCuda2D(FLB::OptionsCalculation& optionsCalc, PRECISION* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, FLB::Units& unitsConverter)
{
  uint8_t h_collisionOperator = optionsCalc.collisionOperator;
  bool optionsHasRelaxationTime = FLB::Math::essentiallyEqual<double>(optionsCalc.relaxationTime, 0.0, 1e-6) ? false : true;
  float h_nu = optionsHasRelaxationTime ? unitsConverter.nuLatticeUnitsFromRelaxationTime(optionsCalc.relaxationTime) : unitsConverter.nuToLatticeUnits(optionsCalc.kinematicViscosity);

  //std::cout << unitsConverter.lenghtToLatticeUnits(0.2f) << " " << h_nu <<" "<< unitsConverter.velocitySIToVelocityLB(optionsCalc.SIVelocity).x <<" KINEMA " << optionsCalc.kinematicViscosity <<" RE\n";

  float h_Fy = 0.0f, h_Fx = 0.0f;
  // If a value other than 0 is indicated in the acceleration, this takes precedence over the value of the volumetric force,
  if (optionsCalc.useVolumetricForce)
  {
    h_Fx = FLB::Math::essentiallyEqual<double>(optionsCalc.acceleration.x, 0.0, 1e-6) ?  unitsConverter.volumeForceToLatticeUnits(optionsCalc.volumetricForce.x) : unitsConverter.volumeForceToLatticeUnits(optionsCalc.acceleration.x, optionsCalc.density);
    h_Fy = FLB::Math::essentiallyEqual<double>(optionsCalc.acceleration.y, 0.0, 1e-6) ?  unitsConverter.volumeForceToLatticeUnits(optionsCalc.volumetricForce.y) : unitsConverter.volumeForceToLatticeUnits(optionsCalc.acceleration.y, optionsCalc.density);
  }

  // Lattice velocities
  //int h_cx[9] = {0, 1, -1, 0,  0, 1, -1, -1,  1};
  int h_cx[9] = {0, 1, -1, 0,  0, 1, -1,  1, -1};
  //int h_cy[9] = {0, 0,  0, 1, -1, 1, -1,  1, -1};
  int h_cy[9] = {0, 0,  0, 1, -1, 1, -1, -1,  1};

  //relaxation time
  float h_tau = optionsHasRelaxationTime ? optionsCalc.relaxationTime : 3.0f * h_nu + 0.5f;
  float h_invTau = 1.0f/h_tau;
  std::cout << h_tau << " TAU " << h_invTau << " INVTAU\n";
  std::cout << h_Fx << " FORCE " << optionsCalc.volumetricForce.x << " FORCE\n";

  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_cx<9>, &h_cx, 9*sizeof(int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_cy<9>, &h_cy, 9*sizeof(int), 0, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_invTau, &h_invTau, sizeof(float), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_Nx, &h_Nx, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_Ny, &h_Ny, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_N, &h_N, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaMemcpyToSymbol(FLB::d_numVelocities, &h_numVelocities, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_collisionOperator, &h_collisionOperator, sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
  if (optionsCalc.precision == 32)
  {
    checkCudaErrors(cudaMemcpyToSymbol(FLB::d_weights<float, 9>, h_weights, 9*sizeof(float), 0, cudaMemcpyHostToDevice));
  }
  else if (optionsCalc.precision == 64)
  {
checkCudaErrors(cudaMemcpyToSymbol(FLB::d_weights<double, 9>, h_weights, 9*sizeof(double), 0, cudaMemcpyHostToDevice));
  }
  
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_Fx, &h_Fx, sizeof(float), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_Fy, &h_Fy, sizeof(float), 0, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_useVolumetricForce, &optionsCalc.useVolumetricForce, sizeof(bool), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_useSubgridModel, &optionsCalc.useSubgridModel, sizeof(bool), 0, cudaMemcpyHostToDevice));
}

// Explicit instantation of the functions because the templates functions are instantiated in a diferent compilation unit

template void FLB::h_initConstantDataCuda2D<float>(FLB::OptionsCalculation& OptionsCalc, float* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, FLB::Units& unitsConverter);

template void FLB::h_initConstantDataCuda2D<double>(FLB::OptionsCalculation& OptionsCalc, double* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, FLB::Units& unitsConverter);

