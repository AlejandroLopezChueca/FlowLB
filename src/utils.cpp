#include "utils.h"
#include <io/reader.h>

#include <iostream>


void FLB::Units::setConversionParameters(const double length, const double LBlenght, const double u, const double LBu, const double rho, const double LBrho)
{
  m_LengthParameter = length / LBlenght; //lenght (SI) = lengthParameter * LBlenght
  m_TimeParameter = m_LengthParameter * LBu / u; // u (SI) = LBu * lengthParameter / timeParameter; 
  m_MassParameter = rho / LBrho * (m_LengthParameter * m_LengthParameter * m_LengthParameter); // rho (SI) = LBrho * massParameter / lengthParameter^3
  std::cout <<m_TimeParameter<< " " << m_LengthParameter*m_LengthParameter  <<" DIVI =   "<<m_TimeParameter/(m_LengthParameter*m_LengthParameter)<< " DIFUSSIVE\n";
}

float FLB::Units::gToLatticeUnits(const float g) const
{
  return g * m_TimeParameter * m_TimeParameter / m_LengthParameter;
}

float FLB::Units::nuToLatticeUnits(const float nu) const
{
  return nu * m_TimeParameter / (m_LengthParameter * m_LengthParameter); 
}

float FLB::Units::rhoToLatticeUnits(const float rho) const
{
  return rho * m_LengthParameter * m_LengthParameter * m_LengthParameter / m_MassParameter;
}

float FLB::Units::lenghtToLatticeUnits(const float lenght) const
{
  return lenght / m_LengthParameter;
}

float FLB::Units::timeToSIUnits(const size_t LBtime) const
{
  // the lattice time is 1
  return LBtime * m_TimeParameter;
}

float FLB::Units::timeToLatticeUnits(const float time) const
{
  return time/m_TimeParameter;
}

float FLB::Units::volumeForceToLatticeUnits(const double acceleration, const double rho) const
{
  // f(SI) = rho(SI) * g(SI)
  // f(LB) = rho / (m_MassParameter / m_LengthParameter^3) * g / (m_LengthParameter / m_TimeParameter^2) = rho * g * m_LengthParameter^2 * m_TimeParameter^2 / m_MassParameter
  return rho * acceleration * m_LengthParameter * m_LengthParameter * m_TimeParameter * m_TimeParameter / m_MassParameter;
}

float FLB::Units::volumeForceToLatticeUnits(const double volumeForce) const
{
  // volumeForce (SI) = Newton / m^3 = kg * m / (s^2 * m^3) = kg / (s^2 * m^2)
  // volumeForce (LB) = volumeForce (SI) * m_TimeParameter^2 * m_LengthParameter^2 / m_MassParameter
  return volumeForce * m_TimeParameter * m_TimeParameter * m_LengthParameter * m_LengthParameter / m_MassParameter;
}

FLB::vector3D FLB::Units::velocitySIToVelocityLB(const FLB::vector3D velocitySI) const
{
  FLB::vector3D velocityLB{velocitySI.x / getVelocityParameterToSIUnits(), velocitySI.y / getVelocityParameterToSIUnits(), velocitySI.z / getVelocityParameterToSIUnits()};
  std::cout <<velocitySI.x<< " " << velocityLB.x  <<" " << " CONVERT V_Si_LB\n";

  return velocityLB;
}

float FLB::Units::nuLatticeUnitsFromRelaxationTime(const double relaxationTime) const
{
  return (relaxationTime - 0.5)/3.0;
}

double FLB::Units::getVelocityParameterToSIUnits() const
{
  return m_LengthParameter / m_TimeParameter;
}

double FLB::Units::getRhoParameterToSIUnits() const
{
  return m_MassParameter / (m_LengthParameter * m_LengthParameter * m_LengthParameter);
}

