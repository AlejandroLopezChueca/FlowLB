#pragma once
//#include "io/reader.h"
#include <cstdint>

namespace FLB 
{

  /**
   * @brief Class to convert units between SI and lattice units
   */
  class Units
  {
    public:
      /*
       *
       * #include ""geometry/mesh.h"*
       

#include ""geometry/mesh.h"*  @brief Calculate the basic parametersto transform between Si units and lattice units
       *
       *  @param[in]  lenght    lenght of any dimension in SI units
       *  @param[in]  LBlenght  length of the same dimension in lattice units  
       */
      void setConversionParameters(float lenght, float LBlenght, float u, float LBu, float rho, float LBrho);
      /**
       * @brief Convert the acceleration of gravity from SI to lattice units 
       *
       * @param[in]  g    acceleration of gravity in SI units
       * @return     LBg  acceleration of gravity in lattice units
       */
      float gToLatticeUnits(float g);
      float nuToLatticeUnits(float nu);
      float rhoToLatticeUnits(float rho);
      /**
       * @brief Get the time value in SI (seconds) of a interval in the simulation
       *
       */
      float timeToSIUnits();

    private:
      float massParameter = 1.0f; 
      float lengthParameter = 1.0f;
      float timeParameter = 1.0f;

  };

  extern Units units;

  /**
   * @brief Types of nodes inside the domain
   */
  enum TypesNodes: uint8_t
  {
    GAS = 0,
    FLUID = 1,
    INTERFACE = 2,
    WALL = 3,
    MOVING_WALL = 4,
    INLET = 5,
    OUTLET = 6
  };
}
