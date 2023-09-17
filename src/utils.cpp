#include "utils.h"

void FLB::Units::setConversionParameters(float length, float LBlenght, float u, float LBu, float rho, float LBrho)
{

  lengthParameter = length / LBlenght; //lenght = lengthParameter * LBlenght
  timeParameter = lengthParameter * LBu / u; // u = LBu * lengthParameter / timeParameter; 
  massParameter = rho / LBrho * (lengthParameter * lengthParameter * lengthParameter); // rho = LBrho * massParameter / lengthParameter^3 
}

float FLB::Units::gToLatticeUnits(float g)
{
  return g * timeParameter * timeParameter / lengthParameter;
}

float FLB::Units::nuToLatticeUnits(float nu)
{
  return nu * timeParameter / (lengthParameter * lengthParameter); 
}

float FLB::Units::rhoToLatticeUnits(float rho)
{
  return rho * lengthParameter * lengthParameter * lengthParameter / massParameter;
}
FLB::Units units;
