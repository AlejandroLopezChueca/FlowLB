#pragma once

//#include "io/reader.h"
#include <cstdint>

namespace FLB 
{
 
  /** 
    * @brief  Approximate the free surface for interface nodes  Redistribution of excess mass for interface and fluid nodes to conserve mass
    *
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __global__ void d_FreeSurface2D_1(PRECISION* d_f, const PRECISION* d_rho, const PRECISION* d_ux, const PRECISION* d_uy, const uint8_t* d_flags, PRECISION* d_mass, PRECISION* d_excessMass, PRECISION* d_phi, const unsigned long int t);
  
  /** 
    * @brief  Prevent neighbors nodes from converting to gas or being gas when the node is interface and it is going to be transformed to fluid.
    *
    */
  template<int NUMVELOCITIES>
  __global__ void d_FreeSurface2D_2(uint8_t* d_flags);

  /** 
    * @brief  Reconstruct probabilty density function for nodes thar are goint to be transformed from gas to interface. It also prevent neighbors nodes from converting to gas or being fluid when the node is interface and it is going to be transformed to gas.
    *
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __global__ void d_FreeSurface2D_3(PRECISION* d_f, PRECISION* d_rho, PRECISION* d_ux, PRECISION* d_uy, uint8_t* d_flags, const unsigned long int t);
  
  /** 
    * @brief  Change label for nodes that are going to be transformed, INTERFACE_FLUID -> FLUID, INTERFACE_GAS -> GAS and GAS_INTERFACE -> GAS. It is also necessary to satisfy the conservation of the mass
    *
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __global__ void d_FreeSurface2D_4(PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass, PRECISION* d_excessMass, PRECISION* d_phi);

  /** 
    * @brief  Single Kernel performing streaming before collision.
    *
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __global__ void d_streamCollide2D(PRECISION* d_f, PRECISION* d_rho, PRECISION* d_ux, PRECISION* d_uy, uint8_t* d_flags, PRECISION* d_mass, const unsigned long int t);
}
