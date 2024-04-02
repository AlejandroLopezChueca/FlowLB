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
      void setConversionParameters(const double lenght, const double LBlenght, const double u, const double LBu, const double rho, const double LBrho);
      /**
       * @brief Convert the acceleration of gravity from SI to lattice units 
       *
       * @param[in]  g    acceleration of gravity in SI units
       * @return     LBg  acceleration of gravity in lattice units
       */
      float gToLatticeUnits(const float g);

      /**
       *
       *
       */
      float nuToLatticeUnits(const float nu);

      /**
       *
       *
       *
       */
      float rhoToLatticeUnits(const float rho);

      /**
       * @brief Get the time value in SI (seconds) of a interval in the simulation
       *
       */
      float timeToSIUnits(const size_t LBtime);

      float timeToLatticeUnits(const float time);

      /**
       *  @brief Calculate the volume force in lattice units form the value of the gravity and density in SI units
       *
       */

      float calculateVolumeForceLatticeUnits(const float volumeForce, const float rho);

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
    FLUID = 1 << 0,                       // 0000 0001
    GAS = 1 << 1,                         // 0000 0010
    INTERFACE = 1 << 2,                   // 0000 0100
    WALL = 1 << 3,                        // 0000 1000
    MOVING_WALL = 1 << 4,                 // 0001 0000
    INLET = 1 << 5,                       // 0010 0000
    OUTLET = 1 << 6,                      // 0100 0000
    INTERFACE_FLUID = INTERFACE | FLUID,  // 0000 0101
    INTERFACE_GAS = INTERFACE | GAS,      // 0000 0110
    FLUID_INLET = FLUID | INLET,          // 0010 0001
    FLUID_OUTLET = FLUID | OUTLET,        // 0100 0001
    GAS_OUTLET = GAS | OUTLET,            // 0100 0010
    GAS_INTERFACE = 0xff,                 // 1111 1111
    NONE = 100                            // 0110 0100
  };
}
