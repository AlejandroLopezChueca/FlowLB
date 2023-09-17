//#include <iostream>
//#include <cuda_runtime.h>
//#include <GLFW/glfw3.h>
//#include "cuda/lbm_kernel.cuh"

//#include "runCalculations.h"
#include "ui/app.h"
#include <iostream>
//#include "geometry/createMesh.h"

int main()
{
  FLB::App app;
  if (FLB::App::appCreated) return 0;
  app.startApp();
  //FLB::runCalculations();
  //FLB::Mesh mesh;
  return 0;
}
