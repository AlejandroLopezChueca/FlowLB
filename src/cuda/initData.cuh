#pragma once

#include "../io/reader.h"
#include "initData2D.cuh"
#include <cstddef>

namespace FLB 
{
  // Lattice velocities
  template<int SIZE>
  __constant__ int d_cx[SIZE];
  template<int SIZE>
  __constant__ int d_cy[SIZE];

  extern template __constant__ int d_cx<9>[9];
  extern template __constant__ int d_cy<9>[9];

  //Lattice weights
  template<typename PRECISION, int SIZE>
  __constant__ PRECISION d_weights[SIZE];

  extern template __constant__ float d_weights<float, 9>[9];
  extern template __constant__ double d_weights<double, 9>[9];
  
  /** Initialization of all the constant data needed for a 2D analysis in the GPU.
    *
    */
  void h_initConstantData2D(FLB::OptionsCalculation& OptionsCalc, size_t numPointsMesh);
  
  template<typename PRECISION>
  void h_initData2D();

}
