//#include <iostream>
//#include <cuda_runtime.h>
//#include <GLFW/glfw3.h>
//#include "cuda/lbm_kernel.cuh"

#include "runCalculations.h"
#include "geometry/createGeometry.h"

int main(){
  FLB::createGeometry();
  FLB::runCalculations();
  return 0;
}
