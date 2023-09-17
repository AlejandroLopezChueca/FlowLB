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

// Simulate surface tneion or not
__constant__ bool FLB::d_surfaceTension;

// Force terms
__constant__ float FLB::d_Fx, FLB::d_Fy, FLB::d_Fz;

template<typename PRECISION>
void FLB::h_initConstantDataCuda2D(FLB::OptionsCalculation& OptionsCalc, PRECISION* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, uint8_t h_collisionOperator, float h_g, float h_nu)
{
  // Lattice velocities
  int h_cx[9] = {0, 1, -1, 0,  0, 1, -1, -1,  1};
  int h_cy[9] = {0, 0,  0, 1, -1, 1, -1,  1, -1};

  //relaxation time
  float h_tau = 3 * h_nu +0.5f; 
  float h_invTau = 1/h_tau;

  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_cx<9>, &h_cx, 9*sizeof(int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_cy<9>, &h_cy, 9*sizeof(int), 0, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_g, &h_g, sizeof(float), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_invTau, &h_invTau, sizeof(float), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_Nx, &h_Nx, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_Ny, &h_Ny, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_N, &h_N, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaMemcpyToSymbol(FLB::d_numVelocities, &h_numVelocities, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_collisionOperator, &h_collisionOperator, sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
  if (OptionsCalc.precision == 32)
  {
    checkCudaErrors(cudaMemcpyToSymbol(FLB::d_weights<float, 9>, &h_weights, 9*sizeof(float), 0, cudaMemcpyHostToDevice));
  }
  else if (OptionsCalc.precision == 64)
  {
checkCudaErrors(cudaMemcpyToSymbol(FLB::d_weights<double, 9>, &h_weights, 9*sizeof(double), 0, cudaMemcpyHostToDevice));
  }
}

template<typename PRECISION>
void FLB::h_initDataCuda2D(FLB::OptionsCalculation& optionsCalc, size_t numPointsMesh, unsigned int numDimensions, unsigned int numVelocities, PRECISION* d_f, PRECISION* h_f, PRECISION* d_vx, PRECISION* h_vx, PRECISION* d_vy, PRECISION* h_vy, uint8_t* d_flags, uint8_t* h_flags, PRECISION* d_mass, PRECISION* h_mass)
{
  size_t fieldSize = numVelocities * numPointsMesh * sizeof(PRECISION);
  checkCudaErrors(cudaMalloc((void **)&d_f, fieldSize));
  checkCudaErrors(cudaMemcpy(d_f, h_f, fieldSize, cudaMemcpyHostToDevice));
  
  fieldSize = numPointsMesh * sizeof(PRECISION);
  checkCudaErrors(cudaMalloc((void **)&d_mass, fieldSize));
  checkCudaErrors(cudaMemcpy(d_mass, h_mass, fieldSize, cudaMemcpyHostToDevice));

  // if the results are going to be renderer,  the velocity is allocated with the API used
  if (optionsCalc.graphicsAPI != FLB::API::NONE)
  {
    checkCudaErrors(cudaMalloc((void **)&d_vx, fieldSize));
    checkCudaErrors(cudaMemcpy(d_vx, h_vx, fieldSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_vy, fieldSize));
    checkCudaErrors(cudaMemcpy(d_vy, h_vy, fieldSize, cudaMemcpyHostToDevice));
  }
 
  if (optionsCalc.typeProblem == FLB::TypeProblem::FREE_SURFACE)
  {
    fieldSize = numPointsMesh * sizeof(uint8_t);
    checkCudaErrors(cudaMalloc((void **)&d_flags, fieldSize));
    checkCudaErrors(cudaMemcpy(d_flags, h_flags, fieldSize, cudaMemcpyHostToDevice)); 
  }
}

// Explicit instantation of the functions because the templates functions are instantiated in a diferent compilation unit

template void FLB::h_initConstantDataCuda2D<float>(FLB::OptionsCalculation& OptionsCalc, float* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, uint8_t h_collisionOperator, float h_g, float h_nu);

template void FLB::h_initConstantDataCuda2D<double>(FLB::OptionsCalculation& OptionsCalc, double* h_weights, unsigned int h_Nx, unsigned int h_Ny, unsigned int h_N, uint8_t h_collisionOperator, float h_g, float h_nu);

template void FLB::h_initDataCuda2D<float>(FLB::OptionsCalculation& optionsCalc, size_t numPointsMesh, unsigned int numDimensions, unsigned int numVelocities, float* d_f, float* h_f, float* d_vx, float* h_vx, float* d_vy, float* h_vy, uint8_t* d_flags, uint8_t* h_flags, float* d_mass, float* h_mass);

template void FLB::h_initDataCuda2D<double>(FLB::OptionsCalculation& optionsCalc, size_t numPointsMesh, unsigned int numDimensions, unsigned int numVelocities, double* d_f, double* h_f, double* d_vx, double* h_vx, double* d_vy, double* h_vy, uint8_t* d_flags, uint8_t* h_flags, double* d_mass, double* h_mass);
