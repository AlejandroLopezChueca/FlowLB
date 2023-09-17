#include "cudaUtils.cuh"

template<typename PRECISION>
void FLB::copyDataFromDevice(size_t numPointsMesh, unsigned int numDimensions, unsigned int numVelocities, PRECISION* d_v, PRECISION* h_v)
{
  size_t fieldSize = numVelocities * numPointsMesh * sizeof(PRECISION);
  checkCudaErrors(cudaMemcpy(h_v, d_v, fieldSize, cudaMemcpyDeviceToHost));
}

void FLB::getInfoDevice(int& devID, cudaDeviceProp& props)
{
  //int devID;
  //cudaDeviceProp props;

  // Get GPU information
  cudaGetDevice(&devID);
  cudaGetDeviceProperties(&props, devID);

}
