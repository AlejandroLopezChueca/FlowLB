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
  if (FLB::App::s_AppCreated) return 0;
  FLB::App app;
  app.startApp();
  return 0;
}
