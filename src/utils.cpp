#include "utils.h"


void FLB::Units::setConversionParameters(double length, double LBlenght, double u, double LBu, double rho, double LBrho)
{
  m_LengthParameter = length / LBlenght; //lenght = lengthParameter * LBlenght
  m_TimeParameter = m_LengthParameter * LBu / u; // u = LBu * lengthParameter / timeParameter; 
  m_MassParameter = rho / LBrho * (m_LengthParameter * m_LengthParameter * m_LengthParameter); // rho = LBrho * massParameter / lengthParameter^3 
}

float FLB::Units::gToLatticeUnits(float g)
{
  return g * m_TimeParameter * m_TimeParameter / m_LengthParameter;
}

float FLB::Units::nuToLatticeUnits(float nu)
{
  return nu * m_TimeParameter / (m_LengthParameter * m_LengthParameter); 
}

float FLB::Units::rhoToLatticeUnits(float rho)
{
  return rho * m_LengthParameter * m_LengthParameter * m_LengthParameter / m_MassParameter;
}

float FLB::Units::timeToSIUnits(size_t LBtime)
{
  // the lattice time is 1
  return LBtime * m_TimeParameter;
}

float FLB::Units::timeToLatticeUnits(float time)
{
  return time/m_TimeParameter;
}

