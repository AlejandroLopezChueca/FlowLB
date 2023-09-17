#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/types.h>

#include "lbmKernel2D.cuh"
#include "cellCalculations.cuh"
//#include "initData.cuh"


template<typename PRECISION, int NUMVELOCITIES>
__global__ void FLB::d_FreeSurface2D_1(PRECISION* d_f, const PRECISION* d_rho, const PRECISION* d_ux, const PRECISION* d_uy, const uint8_t* d_flags, PRECISION* d_mass, PRECISION* d_excessMass, PRECISION* d_phi, const unsigned long int t)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  const unsigned int idx = x + y * d_Nx;

  if (x >= d_Nx || y >= d_Ny) return; // Prevent threads outside domain 
  // Extract if the node cell is located in the input or output
  bool d_isOpenBoundary = (x == 0) || (x == d_Nx);

  uint8_t d_localFlag = d_flags[idx];
  if (d_localFlag == FLB::CT_GAS || d_localFlag == FLB::CT_WALL) return; // if cell is gas or static wall return
  
  unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
  FLB::calculateNeighborsIdxD2Q9(idx, neighborsIdx); // get neighbors indices

  PRECISION d_incomingLocalf[NUMVELOCITIES];
  FLB::loadf(idx, d_f, d_incomingLocalf, neighborsIdx); // First part of the esoteric pull. Load incomoing particle distribution function 
  
  PRECISION d_outgoingLocalf[NUMVELOCITIES];
  d_outgoingLocalf[0] = d_incomingLocalf[0]; // Value of local node is already loaded
  FLB::loadOutgoingf(idx, d_f, d_outgoingLocalf, neighborsIdx, t); 

  PRECISION d_massl = d_mass[idx]; // local mass
  for (int i = 1; i < NUMVELOCITIES; i++) d_massl += d_excessMass[neighborsIdx[i]]; // Redistribute excess mass from last iteration coming from flags conversion to ensure mass conservation

  if (d_localFlag == FLB::CT_FLUID)
  {
    // Calculate the change of mass that is streamed for each direction of the node
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      //Substract outgoing mass and sum incoming mass
      d_massl += d_incomingLocalf[i + 1] - d_outgoingLocalf[i];
      d_massl += d_incomingLocalf[i]     - d_outgoingLocalf[i + 1];
    //d_massl += d_incomingLocalf[i] - d_outgoingLocalf[i];// Substract outgoing mass and sum incoming mass
    }
  }
  else if (d_localFlag == FLB::CT_INTERFACE)
  {
    PRECISION d_rhol, d_uxl, d_uyl, d_rhoYoungLaplace;

    if (d_isOpenBoundary)
    {
      d_rhol = d_rho[idx];
      d_uxl = d_ux[idx];
      d_uyl = d_uy[idx];
    }
    else
    {
      FLB::calculateDensity(d_outgoingLocalf, d_rhol);
      FLB::calculateVelocityD2Q9(d_outgoingLocalf, d_rhol, d_uxl, d_uxl);
    }
    // Calculate the fill level of the node and its neighbors
    PRECISION d_neighborsPhi[NUMVELOCITIES]; // fill level of the neighbors nodes
    for (int i = 1; i < NUMVELOCITIES; i++) d_neighborsPhi[i] = d_phi[neighborsIdx[i]];
    d_neighborsPhi[0] = FLB::calculatePhi(d_massl, d_rhol); // fill level of the local node

    // If the option of surface tension is selected it is necessary to calculate the local mean curvature contained in the Young-Laplace pressure
    if (FLB::d_surfaceTension) ;
    else d_rhoYoungLaplace = 0.0f;

    // VOLUME FORCES (Guo Forcing Scheme)
    const PRECISION d_rhol2 = 0.5f / d_rhol;
    // Correct local velocities with volume forces
    d_uxl = d_uxl + d_rhol2 * FLB::d_Fx;
    d_uyl = d_uyl + d_rhol2 * FLB::d_Fy;

    // calculate the atmospheric equilibrium distribution function
    PRECISION d_feq[NUMVELOCITIES];
    FLB::calculateEquilibriumDFD2Q9(1.0f - d_rhoYoungLaplace, d_uxl, d_uyl, d_feq);
     
    // Load neighbors flags
    uint8_t d_neighborsFlag[NUMVELOCITIES];
    for (int i = 1; i < NUMVELOCITIES; i++) uint8_t d_neighborFlag = d_flags[neighborsIdx[i]]; //neighbor flag

    // Calculate the change of mass that is streamed for each direction of the node
    uint8_t neighborFlag;
    for ( int i = 1; i < NUMVELOCITIES; i += 2)
    {
      neighborFlag = d_neighborsFlag[i];
      // If the neighborFlag is not fluid or interface the change of mass to sum is 0
      if (neighborFlag == FLB::CT_FLUID) d_massl += d_incomingLocalf[i + 1] - d_outgoingLocalf[i];
      else if (neighborFlag == FLB::CT_INTERFACE) d_massl += (d_incomingLocalf[i + 1] - d_outgoingLocalf[i]) * 0.5f * (d_neighborsPhi[neighborsIdx[i]] + d_neighborsPhi[0]);

      neighborFlag = d_neighborsFlag[i + 1];
      if (neighborFlag == FLB::CT_FLUID) d_massl += d_incomingLocalf[i] - d_outgoingLocalf[i + 1];
      else if (neighborFlag == FLB::CT_INTERFACE) d_massl +=(d_incomingLocalf[i] - d_outgoingLocalf[i + 1]) * 0.5f * (d_neighborsPhi[neighborsIdx[i]] + d_neighborsPhi[0]);;
    }

    // Calculate reconstructed particle distribution function
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      d_incomingLocalf[i] = d_feq[i] - d_outgoingLocalf[i + 1] + d_feq[i + 1];
      d_incomingLocalf[i + 1] = d_feq[i +1] - d_outgoingLocalf[i] + d_feq[1];
    }
    // Save reconstructed particle distribution function for gas type to use in the main kernel
    FLB::storeReconstructedGasf(idx, d_f, d_incomingLocalf, d_neighborsFlag);
  }

  d_mass[idx] = d_massl;
}

template<int NUMVELOCITIES>
__global__ void FLB::d_FreeSurface2D_2(uint8_t* d_flags)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  const unsigned int idx = x + y * d_Nx;

  if (x >= d_Nx || y >= d_Ny) return; // Prevent threads outside domain 
  uint8_t d_localFlag = d_flags[idx]; //flag of the node
  if (d_localFlag == FLB::CT_INTERFACE_FLUID)
  {
    unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
    FLB::calculateNeighborsIdxD2Q9(idx, neighborsIdx); // get neighbors indices

    for (int i = 1; i < NUMVELOCITIES; i++)
    {
      uint8_t d_neighborFlag = d_flags[neighborsIdx[i]]; //neighbor flag
      if (d_neighborFlag == FLB::CT_INTERFACE_GAS) d_flags[neighborsIdx[i]] = FLB::CT_INTERFACE;
      else if (d_neighborFlag == FLB::CT_GAS) d_flags[neighborsIdx[i]] = FLB::CT_GAS_INTERFACE; 
    }
  }
}

template<typename PRECISION, int NUMVELOCITIES>
__global__ void FLB::d_FreeSurface2D_3(PRECISION* d_f, PRECISION* d_rho, PRECISION* d_ux, PRECISION* d_uy, uint8_t* d_flags, const unsigned long int t)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  const unsigned int idx = x + y * FLB::d_Nx;

  if (x >= d_Nx || y >= d_Ny) return; // Prevent threads outside domain 
  uint8_t d_localFlag = d_flags[idx]; //flag of the node
  if (d_localFlag == FLB::CT_GAS_INTERFACE)
  {
    unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
    FLB::calculateNeighborsIdxD2Q9(idx, neighborsIdx); // get neighbors indices
    PRECISION d_rhol, d_uxl, d_uyl;
    FLB::calculateAverageValuesNeigbors2D(idx, d_flags, d_rhol, d_uxl, d_uyl, d_rho, d_ux, d_uy, neighborsIdx);
    // Calulate equilibrium function
    PRECISION d_feq[NUMVELOCITIES];
    // VOLUME FORCES (Guo Forcing Scheme)
    const PRECISION d_rhol2 = 0.5f / d_rhol;
    // Correct local velocities with volume forces
    d_uxl = d_uxl + d_rhol2 * d_Fx;
    d_uyl = d_uyl + d_rhol2 * d_Fy;
    FLB::calculateEquilibriumDFD2Q9(d_rhol, d_uxl, d_uyl, d_feq);
    // Write thte equilibrium function  
    FLB::storef(idx, d_f, d_feq, neighborsIdx, t);
  }
  else if (d_localFlag == FLB::CT_INTERFACE_GAS)
  {
    unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
    FLB::calculateNeighborsIdxD2Q9(idx, neighborsIdx); // get neighbors indices
    for (int i = 1; i < NUMVELOCITIES; i++)
    {
      uint8_t d_neighborFlag = d_flags[neighborsIdx[i]]; //neighbor flag
      if (d_neighborFlag == CT_FLUID || d_neighborFlag == CT_INTERFACE_FLUID) d_flags[neighborsIdx[i]] = FLB::CT_INTERFACE;
    }
  }
}

template<typename PRECISION, int NUMVELOCITIES>
 __global__ void FLB::d_FreeSurface2D_4(PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass, PRECISION* d_excessMass, PRECISION* d_phi)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  const unsigned int idx = x + y * FLB::d_Nx;

  if (x >= FLB::d_Nx || y >= FLB::d_Ny) return; // Prevent threads outside domain 
  uint8_t d_localFlag = d_flags[idx]; //flag of the node
  if (d_localFlag == FLB::CT_WALL or d_localFlag == FLB::CT_MWALL) return;
  PRECISION d_rhol = d_rho[idx]; // local density
  PRECISION d_massl = d_mass[idx]; // local mass
  PRECISION d_excessMassl = 0.0f; // local excess mass
  PRECISION d_phil = 0.0f; // local fill level
  if (d_localFlag == FLB::CT_FLUID)
  {
    d_excessMassl = d_massl - d_rhol; // save excess local mass
    d_massl = d_rhol; //fluid mass has to be equal to local density
    d_phil = 1.0f;
  }
  else if (d_localFlag == FLB::CT_GAS)
  {
    d_excessMassl = d_massl; // excess mass is equal to all the local mass
    d_massl = 0.0f; // local mass has to be 0
    d_phil = 0.0f;
  }
  else if (d_localFlag == FLB::CT_INTERFACE)
  {
    if (d_massl > d_rhol)
    {
      d_excessMassl = d_massl - d_rhol;
      d_massl = d_rhol;
    }
    else if (d_massl < 0.0f) 
    {
      d_excessMassl = d_massl;
      d_massl = 0.0f;
    }
    else d_excessMassl = 0.0f; // Without excess mass the local mass doesn't change
    
    d_phil = FLB::calculatePhi(d_excessMassl, d_rhol);
  }
  else if (d_localFlag == FLB::CT_INTERFACE_FLUID)
  {
    d_flags[idx] = FLB::CT_FLUID; // Change node to fluid type
    d_excessMassl = d_massl - d_rhol; // save excess local mass
    d_massl = d_rhol; //fluid mass has to be equal to local density
    d_phil = 1.0f;
  }
  else if (d_localFlag == FLB::CT_INTERFACE_GAS)
  {
    d_flags[idx] = FLB::CT_GAS; // change node to gas type
    d_excessMassl = d_massl; // excess mass is equal to all the local mass
    d_massl = 0.0f; // local mass has to be 0
    d_phil = 0.0f;
  }
  else if (d_localFlag == FLB::CT_GAS_INTERFACE)
  {
    d_flags[idx] = FLB::CT_INTERFACE; // change to iterface type
    if (d_massl > d_rhol)
    {
      d_excessMassl = d_massl - d_rhol;
      d_massl = d_rhol;
    }
    else if (d_massl < 0.0f) 
    {
      d_excessMassl = d_massl;
      d_massl = 0.0f;
    }
    else d_excessMassl = 0.0f; // Without excess mass the local mass doesn't change
    
    d_phil = FLB::calculatePhi(d_excessMassl, d_rhol);
  }

  unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
  FLB::calculateNeighborsIdxD2Q9(idx, neighborsIdx); // get neighbors indices
  uint8_t cont = 0; // Count neighboring nodes that are fluid or interface
  // The excess mass is evenly distributed for all neighboring nodes that are fluid or interface
  for (int i = 1; i < NUMVELOCITIES; i++)
  {
    uint8_t d_neighborFlag = d_flags[neighborsIdx[i]]; //neighbor flag
    cont += (uint8_t)(d_neighborFlag == FLB::CT_FLUID || d_neighborFlag == FLB::CT_INTERFACE || d_neighborFlag == FLB::CT_INTERFACE_FLUID || d_neighborFlag == FLB::CT_GAS_INTERFACE);
  }

  d_massl += cont > 0u ? 0.0f : d_excessMassl; // If there are not neighbors nodes of type fluid or interface save mass in local node to conserve mass
  d_excessMassl = cont > 0u ? d_excessMassl/(float)cont : 0.0f; // equal distribution of the mass for all neighboring nodes that are fluid or interface

  d_mass[idx] = d_massl;
  d_excessMass[idx] = d_excessMassl;
  d_phi[idx] = d_phil;
}

template<typename PRECISION, int NUMVELOCITIES>
__global__ void FLB::d_streamCollide2D(PRECISION* d_f, PRECISION* d_rho, PRECISION* d_ux, PRECISION* d_uy, uint8_t* d_flags, PRECISION* d_mass, const unsigned long int t)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  const unsigned int idx = x + y * d_Nx;

  if (x >= d_Nx || y >= d_Ny) return; // Prevent threads outside domain 
  
  uint8_t d_localFlag = d_flags[idx];
  if (d_localFlag == FLB::CT_GAS || d_localFlag == FLB::CT_WALL) return; // if cell is gas or static wall return

  unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
  FLB::calculateNeighborsIdxD2Q9(idx, neighborsIdx); // get neighbors indices

  PRECISION d_localf[NUMVELOCITIES]; // Local distribution function
  
  FLB::loadf<PRECISION, 9>(idx, d_f, d_localf, neighborsIdx, t); // First part of the esoteric pull. 

  PRECISION d_rhol, d_uxl, d_uyl; // Local density and velocities

  // Apply Boundary conditions in input and output 
  if (d_localFlag == FLB::CT_OPEN) 
  {
    d_rhol = d_rho[idx];
    d_uxl = d_ux[idx];
    d_uyl = d_uy[idx];
  } 
  else
  { //Calculate density and velocity
    FLB::calculateDensity<PRECISION, 9>(d_localf, d_rhol);
    FLB::calculateVelocityD2Q9<PRECISION>(d_localf, d_rhol, d_uxl, d_uxl);
  }

  PRECISION d_Fi[NUMVELOCITIES]; // Forcing terms
  //PRECISION d_Fx, d_Fy; //Force terms

  // After collision it is neccesary to check if the interface flag needs to change
  if (d_localFlag == FLB::CT_INTERFACE)
  {
    PRECISION d_massl = d_mass[idx]; //local mass
    bool d_flagIF = false, d_flagIG = false;
    for (int i = 1; i < NUMVELOCITIES; i++)
    {
      uint8_t d_neighborFlag = d_flags[neighborsIdx[i]]; //neighbor flag
      d_flagIF = d_flagIF || (d_neighborFlag == FLB::CT_FLUID);
      d_flagIG = d_flagIG || (d_neighborFlag == FLB::CT_GAS);
    }
    // (1 + epsilon) * d_rhol to prevents the unwanted effect of fast oscillations between cell type
    if (d_massl > 1.001f * d_rhol || d_flagIF) d_flags[idx] = FLB::CT_INTERFACE_FLUID; // change flag to interface - fluid
    else if (d_massl < -0.001f || d_flagIG) d_flags[idx] = FLB::CT_INTERFACE_GAS; // change flag to interface - gas
  }
  
  // VOLUME FORCES (Guo Forcing Scheme)
  const PRECISION d_rhol2 = 0.5f / d_rhol;
  // Correct local velocities with volume forces
  d_uxl = d_uxl + d_rhol2 * FLB::d_Fx;
  d_uyl = d_uyl + d_rhol2 * FLB::d_Fy;
  FLB::calculateForcingTerms2D<PRECISION>(d_Fi, d_Fx, d_Fy, d_uxl, d_uyl);

  // Update fields except if flag is input or output (boundary)
  if (d_localFlag != FLB::CT_OPEN)
  {
    d_rho[idx] = d_rhol;
    d_ux[idx] = d_uxl;
    d_uy[idx] = d_uyl;
  }

  // Calculate equilibrium function
  PRECISION d_feq[NUMVELOCITIES];
  FLB::calculateEquilibriumDFD2Q9(d_rhol, d_uxl, d_uyl, d_feq);

  switch (d_collisionOperator) //Collision
  {
    case 0:// SRT
    {

      PRECISION d_taul = 1 - 0.5f * d_invTau; //Relaxation of the forcing terms	
      // Perform  collision
      if (d_localFlag != FLB::CT_OPEN)
	for (int i = 0; i < NUMVELOCITIES; i++)
	{
	  d_localf[i] = (1.0f - d_invTau) * d_localf[i] + d_invTau * d_feq[i] + d_taul * d_Fi[i];
	}
      // Boundary condition: Local distribution function in input or output
      else
      {
	for (int i = 0; i < NUMVELOCITIES; i++) d_localf[i] = d_feq[i];
      }
      break;
    }

    case 1: // TRT
      {
	break;
      }
  
  }

  FLB::storef<PRECISION, 9>(idx, d_f, d_localf, neighborsIdx, t); // second part of esoteric pull.
}


// Explicit instantation of the functions because the templates functions are instantiated in a diferent compilation unit

template __global__ void FLB::d_streamCollide2D<float, 9>(float* d_f, float* d_rho, float* d_ux, float* d_uy, uint8_t *d_flags, float* d_mass, const unsigned long t);

template __global__ void FLB::d_streamCollide2D<double, 9>(double* d_f, double* d_rho, double* d_ux, double* d_uy, uint8_t *d_flags, double* d_mass, const unsigned long t); 
