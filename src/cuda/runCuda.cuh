#pragma once 

#include "graphics/window.h"
#include "graphics/buffer.h"
#include "graphics/vertexArray.h"
#include "geometry/mesh.h"
#include "graphics/shader.h"
#include "utils.h"
//#include "graphics/cameraController.h"
#include "io/reader.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace FLB
{
  class RenderLayer;
  /** Run all the 2D calculation in the GPU using Cuda
    *
    *
    * @param[in]

    */
  template<typename PRECISION>
  void h_launchCudaCalculations2D(FLB::OptionsCalculation& optionsCalc, std::vector<PRECISION>& h_weights, FLB::Mesh* mesh, size_t maxIterations,unsigned int numDimensions, unsigned int numVelocities, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, std::filesystem::path& directorySave);

  template<typename PRECISION>
  void h_runCudaMonoFluidOpenGL2D(FLB::OptionsCalculation& optionsCalc, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, FLB::Units& unitsConverter, std::vector<PRECISION>& h_u, PRECISION* d_f, PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass);

  template<typename PRECISION>
  void h_runCudaFreeSurfaceOpenGL2D(FLB::OptionsCalculation& optionsCalc, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, FLB::RenderLayer* renderLayer, FLB::Units& unitsConverter, std::vector<PRECISION>& h_u, PRECISION* d_f, PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass, PRECISION* d_excessMass, PRECISION* d_phi);
  
  template<typename PRECISION>
  void h_runCudaMonoFluidNoGraphics2D(FLB::OptionsCalculation& optionsCalc, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal, FLB::Units& unitsConverter, PRECISION* d_u, std::vector<PRECISION>& h_u, PRECISION* d_f, PRECISION* d_rho, std::vector<PRECISION>& h_rho, uint8_t* d_flags, PRECISION* d_mass, std::filesystem::path& directorySave);

  template<typename PRECISION>
  void h_runCudaFreeSurfaceNoGraphics2D(FLB::OptionsCalculation &optionsCalc, FLB::Mesh *mesh, Fl_Simple_Terminal *terminal, FLB::Units& unitsConverter, PRECISION* d_u, std::vector<PRECISION> &h_u, PRECISION *d_f, PRECISION *d_rho, std::vector<PRECISION>& h_rho, uint8_t *d_flags, PRECISION *d_mass, PRECISION *d_excessMass, PRECISION *d_phi, std::vector<PRECISION>& h_phi, std::filesystem::path& directorySave);

}
