#pragma once

#include <stdlib.h>
#include <stdio.h>

template <typename T>
void checkCuda(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)


namespace FLB
{

}

