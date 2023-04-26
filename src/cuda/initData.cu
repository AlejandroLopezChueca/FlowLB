#include <cstddef>
#include <cuda_runtime.h>
#include "initData.cuh"
#include "cudaUtils.h"


template __constant__ int FLB::d_cx<9>[9];
template __constant__ int FLB::d_cy<9>[9];

template __constant__ float FLB::d_weights<float, 9>[9];
template __constant__ double FLB::d_weights<double, 9>[9];


void FLB::h_initConstantData2D(FLB::OptionsCalculation &OptionsCalc, size_t numPointsMesh)
{
  // Lattice velocities
  int h_cx[9] = {0, 1, 0, -1,  0, 1, -1, -1,  1};
  int h_cy[9] = {0, 0, 1,  0, -1, 1,  1, -1, -1};

  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_cx<9>, &h_cx, 9*sizeof(int), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(FLB::d_cy<9>, &h_cy, 9*sizeof(int), 0, cudaMemcpyHostToDevice));
  
  if (OptionsCalc.precision == 32)
  {
    float h_weights[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
    checkCudaErrors(cudaMemcpyToSymbol(FLB::d_weights<float, 9>, &h_weights, 9*sizeof(float), 0, cudaMemcpyHostToDevice));
  }
  else if (OptionsCalc.precision == 64)
  {
    double h_weights[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
    checkCudaErrors(cudaMemcpyToSymbol(FLB::d_weights<double, 9>, &h_weights, 9*sizeof(double), 0, cudaMemcpyHostToDevice));
  }

}
template<typename PRECISION>
void FLB::h_initData2D()
{
  
}



