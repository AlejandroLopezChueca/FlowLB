#pragma once

#include "FL/Fl_Simple_Terminal.H"
#include "io/reader.h"
#include "geometry/mesh.h"
#include "ui/app.h"

namespace FLB 
{ 
  /**
   * @brief Create the mesh and initialization of the domain
   *
   * @param[in]  pathFiles  path of the directory with the geometry and the calculation options
   * @param[in]  mesh       mesh of the domain
   * @param[in]  terminal   terminal of the app to print information
   */
  void createDomain(const char* pathFiles, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal);

  /** 
   * @brief Start all the calculations.
   *
   * @param[in]  pathFiles  path of the directory with the geometry and the calculation options
   * @param[in]  mesh       mesh of the domain
   * @param[in]  terminal   terminal of the app to print information
   */
  void runCalculations(const char* pathFiles, FLB::Mesh* mesh, Fl_Simple_Terminal* terminal);

  void viewResults(const char* pathFiles, Fl_Simple_Terminal* terminal);
}

