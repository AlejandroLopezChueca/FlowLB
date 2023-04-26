#pragma once


#include "../io/reader.h"

namespace FLB 
{

  template<typename PRECISION>
  __global__ void d_stream();

  __global__ void d_swap();
  
  template<typename PRECISION>
  __global__ void d_collision();

  template<typename PRECISION>
  __global__ void d_boundaryConditions();
  /** Run all the 2D calculations.
    *
    */
  void h_runCalculationsGPU2D(FLB::OptionsCalculation& OptionsCalc);
}
