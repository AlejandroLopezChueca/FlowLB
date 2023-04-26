#include <cstddef>
#include <cuda_runtime.h>
#include <stdio.h>
#include "initData.cuh"
#include "lbmKernel2D.cuh"


template<typename PRECISION>
__global__ void d_print()
{

  printf("Thread number %d. f = %f\n", threadIdx.x,FLB::d_weights<PRECISION, 9>[0]);
  PRECISION m = FLB::d_weights<PRECISION, 9>[8] * FLB::d_weights<PRECISION, 9>[8];
  printf("%.16f\n", m);
}

template<typename PRECISION>
__global__ void FLB::d_collision()
{

}

void FLB::h_runCalculationsGPU2D(FLB::OptionsCalculation& OptionsCalc, size_t numPointsMesh)
{

  if (OptionsCalc.precision == 32)
  {
    float* d_f;  // distribution function

  }
  
  else if (OptionsCalc.precision == 64)
  {
    double* d_f;  // distribution function

  
  }



  int devID;
  cudaDeviceProp props;

  // Get GPU information
  cudaGetDevice(&devID);
  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);


  FLB::h_initData2D(OptionsCalc, numPointsMesh);
  printf("ASA\n");
  int blockSize = 2;
  dim3 block(blockSize);
  dim3 grid((0/block.x)+1);

  if (OptionsCalc.precision == 32)
  {

  d_print<float><<<grid, block>>>();
  }
  
  else if (OptionsCalc.precision == 64)
  {

  d_print<double><<<grid, block>>>();
  }
  cudaDeviceSynchronize();

}
