#pragma once
//#include "io/reader.h"
#include <cstddef>
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
       *  @brief Calculate the basic parameters used to transform between Si units and lattice units
       *
       *  @param[in]  lenght    lenght of any dimension in SI units
       *  @param[in]  LBlenght  length of the same dimension in lattice units 
       *  @param[in]  rho       kinematic viscosity in SI units
       *  @param[in]  LBrho     kinematic viscosity in lattice units
       */
      void setConversionParameters(double lenght, double LBlenght, double u, double LBu, double rho, double LBrho);
      /**
       * @brief Convert the acceleration of gravity from SI to lattice units 
       *
       * @param[in]  g    acceleration of gravity in SI units
       * @return     LBg  acceleration of gravity in lattice units
       */
      float gToLatticeUnits(float g);

      /**
       *
       *
       */
      float nuToLatticeUnits(float nu);

      /**
       *
       *
       *
       */
      float rhoToLatticeUnits(float rho);

      /**
       * @brief Get the time value in SI (seconds) of a interval in the simulation
       *
       */
      float timeToSIUnits(size_t LBtime);

      float timeToLatticeUnits(float time);

    private:
      double m_MassParameter = 1.0; 
      double m_LengthParameter = 1.0;
      double m_TimeParameter = 1.0;

  };

  //extern Units units;

  /**
   * @brief Types of nodes inside the domain
   */
  enum TypesNodes: uint8_t
  {
    // This has to be the same than graphics constants
    FLUID = 0,
    GAS = 1,
    INTERFACE = 2,
    INTERFACE_FLUID = 3,
    INTERFACE_GAS = 4,
    GAS_INTERFACE = 5,
    WALL = 6,
    MOVING_WALL = 7,
    INLET = 8,
    OUTLET = 8,

    NONE = 100

  };
}
