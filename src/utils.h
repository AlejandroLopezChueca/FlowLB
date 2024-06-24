#pragma once
//#include "io/reader.h"
#include <cstddef>
#include <cstdint>

namespace FLB 
{
  // fordard declaration
  struct vector3D;

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
      float gToLatticeUnits(const float g) const;

      /**
       *
       *
       */
      float nuToLatticeUnits(const float nu) const;

      /**
       *
       *
       *
       */
      float rhoToLatticeUnits(const float rho) const;
      
      /**
       *
       *
       *
       */
      float lenghtToLatticeUnits(const float lenght) const;

      /**
       * @brief Get the time value in SI (seconds) of a interval in the simulation
       */
      float timeToSIUnits(const size_t LBtime) const;

      float timeToLatticeUnits(const float time) const;

      /**
       *  @brief Calculate the volume force in lattice units from the value of the acceleration and density in SI units
       */

      float volumeForceToLatticeUnits(const double acceleration, const double rho) const ;
      
      /**
       *  @brief Calculate the volume force in lattice units from the value of the volume force in Si Units
       */
      
      float volumeForceToLatticeUnits(const double volumeForce) const;

      vector3D velocitySIToVelocityLB(const vector3D velocitySI) const;

      float nuLatticeUnitsFromRelaxationTime(const double relaxationTime) const;

      double getVelocityParameterToSIUnits() const;
      double getRhoParameterToSIUnits() const;

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
    GAS_INTERFACE = 1<<7,                 // 1000 0000
    NONE = 100                            // 0110 0100
  };
}
