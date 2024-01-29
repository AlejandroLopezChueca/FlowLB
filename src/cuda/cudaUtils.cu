#include "cudaUtils.cuh"
#include "cudaInitData.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
//#include "cudaUtils.h"

void FLB::CudaUtils::printInfoDevice(Fl_Simple_Terminal* terminal)
{
  int deviceCount = 0;
  int deviceID;
  int major = 0;
  int minor = 0;
  cudaDeviceProp deviceProps;

  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0)
  {
    terminal -> printf("[GPU ERROR] No devices supporting CUDA detected\n");
    return;
  }
  else if (deviceCount == 1)
  {
    terminal -> printf("[GPU INFO] One device that supports CUDA has been detected\n");
  }
  else
  {
    terminal -> printf("[GPU INFO] More than one device that support CUDA has been detected\n");
  }
  
  checkCudaErrors(cudaGetDevice(&deviceID));

  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceID));
  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceID));

  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, deviceID));

  terminal -> printf("[GPU INFO] CUDA device [%s] has %d Multi-Processors with compute capability %d.%d\n", deviceProps.name, deviceProps.multiProcessorCount, major, minor);

  // memory 
  size_t freeBytes, totalBytes;
  checkCudaErrors(cudaMemGetInfo(&freeBytes, &totalBytes));
  size_t usedBytes = totalBytes - freeBytes;
  terminal -> printf("[GPU INFO] Total Memory (MB): %.2f\n", totalBytes / 1024.0 / 1024.0);
  terminal -> printf("[GPU INFO] Free Memory (MB): %.2f\n", freeBytes / 1024.0 / 1024.0);
  terminal -> printf("[GPU INFO] Used Memory (MB): %.2f\n", usedBytes / 1024.0 / 1024.0);

  uint32_t maxThreadsPerBlock = getMaxThreadsPerBlock();
  terminal -> printf("[GPU INFO] The maximum number of threads per block is %d\n", maxThreadsPerBlock);
}

uint32_t FLB::CudaUtils::getMaxThreadsPerBlock()
{
  int deviceID;
  checkCudaErrors(cudaGetDevice(&deviceID));

  // first get the maximum number of threads per block
  int maxThreadsPerBlock;
  checkCudaErrors(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceID));
  return maxThreadsPerBlock;

}

dim3 FLB::CudaUtils::getBlockSize(int numDimensions)
{
  int deviceID;
  checkCudaErrors(cudaGetDevice(&deviceID));

  // first it is necesary to know the maximum dmmesion of a block
  int maxBlockDimX, maxBlockDimY, maxBlockDimZ;
  checkCudaErrors(cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, deviceID));
  checkCudaErrors(cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, deviceID));
  checkCudaErrors(cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, deviceID));


}
dim3 FLB::CudaUtils::getGridSize(const int numDimensions, const FLB::Mesh* mesh, const dim3& blockSize)
{
  int deviceID;
  checkCudaErrors(cudaGetDevice(&deviceID));

  // first it is necesary to know the maximum dmmesion of the grid
  int maxGridDimX, maxGridDimY, maxGridDimZ;
  checkCudaErrors(cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, deviceID));
  checkCudaErrors(cudaDeviceGetAttribute(&maxGridDimY, cudaDevAttrMaxGridDimY, deviceID));
  checkCudaErrors(cudaDeviceGetAttribute(&maxGridDimZ, cudaDevAttrMaxGridDimZ, deviceID));
 

  dim3 gridSize;
  if (numDimensions == 2) gridSize = {(mesh -> getNx() + blockSize.x - 1)/blockSize.x, (mesh -> getNy() + blockSize.y - 1)/blockSize.y};

  return gridSize;
}

template <typename PRECISION>
__global__ void FLB::CudaUtils::save2DDataToOpenGL(PRECISION* v, cudaSurfaceObject_t d_SurfaceTexture)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  const unsigned int idx = x + y * FLB::d_Nx;
  
  if (x >= FLB::d_Nx || y >= FLB::d_Ny) return;
  float2 data = make_float2(v[idx], v[idx + FLB::d_N]);
  //printf("OPENGL vx = %6.4f  y = %6.4f\n", v[idx], v[idx + d_N]);
  surf2Dwrite(data, d_SurfaceTexture, x * sizeof(float2), y);
}

template __global__ void FLB::CudaUtils::save2DDataToOpenGL<float>(float *v, cudaSurfaceObject_t d_SurfaceTexture);

template __global__ void FLB::CudaUtils::save2DDataToOpenGL<double>(double *v, cudaSurfaceObject_t d_SurfaceTexture);

