#pragma once

#include "geometry/mesh.h"

#include <cstdint>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "FL/Fl_Simple_Terminal.H"
#include "surface_types.h"


template <typename T>
void checkCuda(T result, char const *const func, const char *const file, int const line) 
{
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

namespace FLB::CudaUtils
{
  template<typename PRECISION>
  void copyDataFromDevice(size_t numPointsMesh, unsigned int numDimensions, PRECISION* d_u, PRECISION* h_u)
  {
    size_t fieldSize = numDimensions * numPointsMesh * sizeof(PRECISION);
    checkCudaErrors(cudaMemcpy(h_u, d_u, fieldSize, cudaMemcpyDeviceToHost));
  }
 
  void printInfoDevice(Fl_Simple_Terminal* terminal);


  /**
   * @brief Get maximum number of threads per block 
   *
   *
   **/
  uint32_t getMaxThreadsPerBlock();

  /**
   * @brief Get number of blocks per grid 
   *
   *
   **/
  dim3 getBlockSize(int numDimensions);

  /**
   * @brief Get number of grids
   *
   *
   **/
  dim3 getGridSize(const int numDimensions, const FLB::Mesh* mesh, const dim3& blockSize);

  /**
    * @brief Save data calculated with Cuda to OpenGL texture
    *
    */
  template <typename PRECISION>
  __global__ void save2DDataToOpenGL(PRECISION* v, cudaSurfaceObject_t d_SurfaceTexture);
}


