#pragma once

#include "geometry/mesh.h"
#include "graphics/texture.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "FL/Fl_Simple_Terminal.H"
#include "surface_types.h"
#include <cuda_gl_interop.h>


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
  struct copyDataElement
  {
    const unsigned int numDimensions;
    const unsigned int sizeData;
    PRECISION* h_ptr;
    PRECISION* d_ptr;

    copyDataElement<PRECISION>(const unsigned int numberDimensions, const unsigned int size, PRECISION* host_ptr, PRECISION* device_ptr) 
      : numDimensions(numberDimensions), sizeData(size), h_ptr(host_ptr), d_ptr(device_ptr) {}
  };

  template<typename PRECISION>
  void copyDataFromDevice(size_t numPointsMesh, const std::initializer_list<copyDataElement<PRECISION>>& listElements, uint8_t* h_flags, uint8_t* d_flags)
  {
    for (auto& element : listElements)
    {
      size_t fieldSize = element.numDimensions * numPointsMesh * element.sizeData;
    checkCudaErrors(cudaMemcpy(static_cast<void*>(element.h_ptr), static_cast<const void*>(element.d_ptr), fieldSize, cudaMemcpyDeviceToHost));
    }

    size_t fieldSize = 1 * numPointsMesh * sizeof(uint8_t);
    checkCudaErrors(cudaMemcpy(static_cast<void*>(h_flags), static_cast<const void*>(d_flags), fieldSize, cudaMemcpyDeviceToHost));
  }
 
  void printInfoDevice(Fl_Simple_Terminal* terminal);

  std::array<float, 2> getUsedFreeMemory(Fl_Simple_Terminal* terminal);


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

  void mapTexture2DToOpenGL(struct cudaGraphicsResource** cudaResource, const FLB::Texture2D* texture2D, unsigned int flags);
}


