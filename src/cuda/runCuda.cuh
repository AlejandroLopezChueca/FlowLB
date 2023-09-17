#pragma once 

#include "graphics/window.h"
#include "graphics/buffer.h"
#include "graphics/vertexArray.h"
#include "graphics/shader.h"
//#include "graphics/cameraController.h"
#include "io/reader.h"
#include <memory>

namespace FLB
{
  /** Run all the calculation in the GPU using Cuda, both 2D and 3D.
    *
    *
    * @param[in]

    */
  template<typename PRECISION>
  void h_runCudaCalculations2D(FLB::OptionsCalculation& optionsCalc, FLB::Window& renderWindow, FLB::VertexArray& vertexArray, FLB::Shader& textureShader, FLB::OrthographicCameraController& orthographicCameraController, PRECISION* h_weights, FLB::Mesh* mesh, size_t maxIterations,unsigned int numDimensions, unsigned int numVelocities,Fl_Simple_Terminal* terminal);

}
