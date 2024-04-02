#include "utils.h"


void FLB::Units::setConversionParameters(const double length, const double LBlenght, const double u, const double LBu, const double rho, const double LBrho)
{
  m_LengthParameter = length / LBlenght; //lenght = lengthParameter * LBlenght
  m_TimeParameter = m_LengthParameter * LBu / u; // u = LBu * lengthParameter / timeParameter; 
  m_MassParameter = rho / LBrho * (m_LengthParameter * m_LengthParameter * m_LengthParameter); // rho = LBrho * massParameter / lengthParameter^3 
}

float FLB::Units::gToLatticeUnits(const float g)
{
  return g * m_TimeParameter * m_TimeParameter / m_LengthParameter;
}

float FLB::Units::nuToLatticeUnits(const float nu)
{
  return nu * m_TimeParameter / (m_LengthParameter * m_LengthParameter); 
}

float FLB::Units::rhoToLatticeUnits(const float rho)
{
  return rho * m_LengthParameter * m_LengthParameter * m_LengthParameter / m_MassParameter;
}

float FLB::Units::timeToSIUnits(const size_t LBtime)
{
  // the lattice time is 1
  return LBtime * m_TimeParameter;
}

float FLB::Units::timeToLatticeUnits(const float time)
{
  return time/m_TimeParameter;
}

float FLB::Units::calculateVolumeForceLatticeUnits(const float volumeForce, const float rho)
{
  // f(SI) = rho(SI) * g(SI)
  // f = rho / (m_MassParameter / m_LengthParameter^3) * g / (m_LengthParameter / m_TimeParameter^2) = rho * g * m_LengthParameter^2 * m_TimeParameter^2 / m_MassParameter
  return rho * volumeForce * m_LengthParameter * m_LengthParameter * m_TimeParameter * m_TimeParameter / m_MassParameter;

}

