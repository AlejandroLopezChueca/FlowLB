#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iterator>
#include "cudaInitData.cuh"
#include "cudaUtils.cuh"


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
__constant__ uint8_t FLB::CT_FLUID = 0;
__constant__ uint8_t FLB::CT_GAS = 1;
__constant__ uint8_t FLB::CT_INTERFACE = 2;
__constant__ uint8_t FLB::CT_INTERFACE_FLUID = 3;
__constant__ uint8_t FLB::CT_INTERFACE_GAS = 4;
__constant__ uint8_t FLB::CT_GAS_INTERFACE = 5;
__constant__ uint8_t FLB::CT_WALL = 6;  //Static wall
__constant__ uint8_t FLB::CT_MWALL = 7; // Moving wall
__constant__ uint8_t FLB::CT_OPEN = 8; // Input or Output

// constant variables
__constant__ float FLB::d_g;
__constant__ float FLB::d_invTau;

// Use of grvity in the simulation
__constant__ bool FLB::d_useGravity;

// Simulate surface tneion or not
__constant__ bool FLB::d_surfaceTension;

// Force terms
__constant__ float FLB::d_Fx, FLB::d_Fy, FLB::d_Fz;

template<typename PRECISION>
void FLB::h_initConstantDataCuda2D(FLB::OptionsCalculation& optionsCalc, PRECISION* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, uint8_t h_collisionOperator, float h_g, float h_nu)
{
  // Lattice velocities
  int h_cx[9] = {0, 1, -1, 0,  0, 1, -1, -1,  1};
  int h_cy[9] = {0, 0,  0, 1, -1, 1, -1,  1, -1};

  //relaxation time
  float h_tau = 3.0 * h_nu + 0.5f; 
  float h_invTau = 1.0f/h_tau;

  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_cx<9>, &h_cx, 9*sizeof(int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_cy<9>, &h_cy, 9*sizeof(int), 0, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_g, &h_g, sizeof(float), 0, cudaMemcpyHostToDevice));
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
checkCudaErrors(cudaMemcpyToSymbol(FLB::d_weights<double, 9>, &h_weights, 9*sizeof(double), 0, cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_useGravity, &optionsCalc.useGravity, sizeof(bool), 0, cudaMemcpyHostToDevice));
}

// Explicit instantation of the functions because the templates functions are instantiated in a diferent compilation unit

template void FLB::h_initConstantDataCuda2D<float>(FLB::OptionsCalculation& OptionsCalc, float* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, uint8_t h_collisionOperator, float h_g, float h_nu);

template void FLB::h_initConstantDataCuda2D<double>(FLB::OptionsCalculation& OptionsCalc, double* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, uint8_t h_collisionOperator, float h_g, float h_nu);

