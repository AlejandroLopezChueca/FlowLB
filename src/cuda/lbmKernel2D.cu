#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/types.h>

#include "cudaInitData.cuh"
#include "lbmKernel2D.cuh"
#include "cellCalculations.cuh"
//#include "initData.cuh"


template<typename PRECISION, int NUMVELOCITIES>
__global__ void FLB::d_FreeSurface2D_1(PRECISION* d_f, const PRECISION* d_rho, const PRECISION* d_u, const uint8_t* d_flags, PRECISION* d_mass, PRECISION* d_excessMass, PRECISION* d_phi, const unsigned long int t)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;

  if (x >= d_Nx || y >= d_Ny) return; // Prevent threads outside domain 
  const unsigned int idx = x + y * d_Nx;

  uint8_t localFlag = d_flags[idx];
  // for the inlet the local mass is constant with a value of 1.0
  //if (idx == 43 || idx == 62 || idx == 61 || idx == 155 || idx == 114) printf("11XXXSTREAM local_flag = %d idx = %d localMass = %6.4f localRho = %6.4f localPhi = %6.4f Ux = %6.4f Uy = %6.4f\n", +localFlag, idx, d_mass[idx], d_rho[idx], d_phi[idx], d_u[idx], d_u[idx + FLB::d_N]);
  if (idx == 43 || idx == 62 || idx == 61 || idx == 155 || idx == 114) printf("11XXXSTREAM Forcex = %6.4f Forcey = %6.4f\n", FLB::d_Fx, FLB::d_Fy);
  if (localFlag & (FLB::CT_GAS | FLB::CT_WALL | FLB::CT_INLET)) return; // if cell is gas, inlet or static wall return
   
  unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
  FLB::calculateNeighborsIdxD2Q9(idx, x, y, neighborsIdx); // get neighbors indices

  PRECISION incomingLocalf[NUMVELOCITIES];
  FLB::loadf<PRECISION, 9>(idx, d_f, incomingLocalf, neighborsIdx, t); // First part of the esoteric pull. Load incoming particle distribution function 
  
  PRECISION outgoingLocalf[NUMVELOCITIES];
  outgoingLocalf[0] = incomingLocalf[0]; // Value of local node is already loaded
  FLB::loadOutgoingf<PRECISION, 9>(idx, d_f, outgoingLocalf, neighborsIdx, t); 

  PRECISION localMass = d_mass[idx]; // local mass
  for (int i = 1; i < NUMVELOCITIES; i++) localMass += d_excessMass[neighborsIdx[i]]; // Redistribute excess mass from last iteration coming from flags conversion to ensure mass conservation

  uint8_t localFlagExtracted = localFlag & FLB::CT_FLUID_GAS_INTERFACE;
  if (localFlagExtracted == FLB::CT_FLUID)
  {
    // Calculate the change of mass that is streamed for each direction of the node
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      //Substract outgoing mass and sum incoming mass
      localMass += incomingLocalf[i + 1] - outgoingLocalf[i];
      localMass += incomingLocalf[i]     - outgoingLocalf[i + 1];
     //localMass += incomingLocalf[i] - outgoingLocalf[i];// Substract outgoing mass and sum incoming mass
    }
  }
  else if (localFlagExtracted == FLB::CT_INTERFACE)
  {
    PRECISION localRho, localUx, localUy, localRhoYoungLaplace = 0.0f;

    if (localFlag & (FLB::CT_INLET | FLB::CT_OUTLET))
    {
      localRho = d_rho[idx];
      localUx = d_u[idx];
      localUy = d_u[idx + FLB::d_N];
    }
    else
    {
      FLB::calculateDensity<PRECISION, 9>(outgoingLocalf, localRho);
      FLB::calculateVelocityD2Q9(outgoingLocalf, localRho, localUx, localUy);
    }
    // Calculate the fill level of the node and its neighbors
    PRECISION neighborsPhi[NUMVELOCITIES]; // fill level of the neighbors nodes
    for (int i = 1; i < NUMVELOCITIES; i++) neighborsPhi[i] = d_phi[neighborsIdx[i]];

    neighborsPhi[0] = FLB::calculatePhi(localMass, localRho); // fill level of the local node

    // If the option of surface tension is selected it is necessary to calculate the local mean curvature contained in the Young-Laplace pressure
    // TODO calculate tension
    if (FLB::d_surfaceTension) localRhoYoungLaplace = 0.0f;

    // VOLUME FORCES (Guo Forcing Scheme)
    const PRECISION localRho2 = 0.5f / localRho;
    // Correct local velocities with volume forces
    localUx = localUx + localRho2 * FLB::d_Fx;
    localUy = localUy + localRho2 * FLB::d_Fy;
    if (idx == 43 || idx == 62 || idx == 61 || idx == 155 || idx == 124) printf("11XXXSTREAM localUx = %6.4f localUy = %6.4f localRho = %6.4f\n", localUx, localUy, localRho);

    // calculate the atmospheric equilibrium distribution function
    PRECISION localFeq[NUMVELOCITIES];
    FLB::calculateEquilibriumDDF2Q9(1.0f - localRhoYoungLaplace, localUx, localUy, localFeq);
     
    // Load neighbors flags
    uint8_t neighborsFlag[NUMVELOCITIES];
    for (int i = 1; i < NUMVELOCITIES; i++) neighborsFlag[i] = d_flags[neighborsIdx[i]] & FLB::CT_FLUID_GAS_INTERFACE; //neighbor flag

    // Calculate the change of mass that is streamed for each direction of the node
    for ( int i = 1; i < NUMVELOCITIES; i += 2)
    {
      uint8_t neighborFlagExtracted = neighborsFlag[i];
      // If the neighborFlag is not fluid or interface the change of mass to sum is 0
      if (neighborFlagExtracted == FLB::CT_FLUID) localMass += incomingLocalf[i + 1] - outgoingLocalf[i];
      else if (neighborFlagExtracted == FLB::CT_INTERFACE) localMass += (incomingLocalf[i + 1] - outgoingLocalf[i]) * 0.5f * (neighborsPhi[i] + neighborsPhi[0]);
      if (idx == 124) printf("111 local_flag = %d idx = %d  neighFlag = %d i = %d incoming = %6.4f out = %6.4f localPhi = %6.4f neighborsPhi = %6.4f neighborsIdx = %d\n", +localFlag, idx, +neighborsFlag[i], i, incomingLocalf[i+1], outgoingLocalf[i], neighborsPhi[0], neighborsPhi[i], neighborsIdx[i]);

      neighborFlagExtracted = neighborsFlag[i + 1];
      if (neighborFlagExtracted == FLB::CT_FLUID) localMass += incomingLocalf[i] - outgoingLocalf[i + 1];
      else if (neighborFlagExtracted == FLB::CT_INTERFACE) localMass += (incomingLocalf[i] - outgoingLocalf[i + 1]) * 0.5f * (neighborsPhi[i + 1] + neighborsPhi[0]);;
      if (idx == 124) printf("111 local_flag = %d idx = %d  neighFlag = %d i = %d incoming = %6.4f out = %6.4f localPhi = %6.4f neighborsPhi = %6.4f neighborsIdx = %d\n", +localFlag, idx, +neighborsFlag[i+1], i+1, incomingLocalf[i], outgoingLocalf[i+1], neighborsPhi[0], neighborsPhi[i + 1], neighborsIdx[i+1]);
    }

    // Calculate reconstructed particle distribution function
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      incomingLocalf[i] = localFeq[i] - outgoingLocalf[i + 1] + localFeq[i + 1];
      incomingLocalf[i + 1] = localFeq[i + 1] - outgoingLocalf[i] + localFeq[i];
      if (idx == 124) printf("111 local_flag = %d idx = %d incoming1 = %6.4f incoming2 = %6.4f \n", +localFlag, idx, incomingLocalf[i], incomingLocalf[i+1]);
    }
    // Save reconstructed particle distribution function for gas type only to use in the main kernel
    FLB::storeReconstructedGasf<PRECISION, 9>(idx, d_f, incomingLocalf, neighborsFlag, neighborsIdx, t);
  if (idx == 114) printf("11XXXSTREAM local_flag = %d idx = %d localMass = %6.4f localRho = %6.4f\n", +localFlag, idx, localMass, localRho);
  }

  d_mass[idx] = localMass;
}

template<int NUMVELOCITIES>
__global__ void FLB::d_FreeSurface2D_2(uint8_t* d_flags)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  if (x >= d_Nx || y >= d_Ny) return; // Prevent threads outside domain 
  
  const unsigned int idx = x + y * d_Nx;
  if (idx ==43) printf("22XXXSTREAM local_flag = %d idx = %d \n", +d_flags[idx], idx);
  uint8_t localFlagExtracted = d_flags[idx] & FLB::CT_INTERFACE_FLUID; //  it can be fluid, interface or interface-fluid

  if (localFlagExtracted == FLB::CT_INTERFACE_FLUID)
  {
    unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
    FLB::calculateNeighborsIdxD2Q9(idx, x, y, neighborsIdx); // get neighbors indices

    for (int i = 1; i < NUMVELOCITIES; i++)
    {
      uint8_t neighborFlag = d_flags[neighborsIdx[i]]; //neighbor flag
      uint8_t neighborFlagExtracted = neighborFlag & FLB::CT_INTERFACE_GAS; // the extraction can be interface-gas, gas or interface
      if (neighborFlagExtracted == FLB::CT_INTERFACE_GAS) d_flags[neighborsIdx[i]] = neighborFlag & ~FLB::CT_GAS; // first delete gas flag and change to interface flag
      else if (neighborFlagExtracted == FLB::CT_GAS) d_flags[neighborsIdx[i]] = (neighborFlag & ~FLB::CT_GAS) | FLB::CT_GAS_INTERFACE; // first delete the gas flag and change to interface if it is gas
    }
  if (idx ==43) printf("22XXXSTREAM local_flag = %d flag44 = %d idx = %d \n", +d_flags[idx], d_flags[neighborsIdx[1]], idx);
  }
}

template<typename PRECISION, int NUMVELOCITIES>
__global__ void FLB::d_FreeSurface2D_3(PRECISION* d_f, PRECISION* d_rho, PRECISION* d_u, uint8_t* d_flags, const unsigned long int t)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  if (x >= d_Nx || y >= d_Ny) return; // Prevent threads outside domain 
  
  const unsigned int idx = x + y * FLB::d_Nx;
  uint8_t localFlagExtracted = d_flags[idx] & (FLB::CT_GAS_INTERFACE | FLB::CT_INTERFACE_GAS);
  
  if (localFlagExtracted == FLB::CT_GAS_INTERFACE)
  {
    unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
    FLB::calculateNeighborsIdxD2Q9(idx, x, y, neighborsIdx); // get neighbors indices
    PRECISION localRho, localUx, localUy;
    FLB::calculateAverageValuesNeighbors2D<PRECISION, 9>(idx, d_flags, localRho, localUx, localUy, d_rho, d_u, neighborsIdx);
    // Calulate equilibrium function
    PRECISION localFeq[NUMVELOCITIES];
    // VOLUME FORCES (Guo Forcing Scheme)
    const PRECISION localRho2 = 0.5f / localRho;
    // Correct local velocities with volume forces
    localUx = localUx + localRho2 * FLB::d_Fx;
    localUy = localUy + localRho2 * FLB::d_Fy;
    FLB::calculateEquilibriumDDF2Q9(localRho, localUx, localUy, localFeq);
    // Write thte equilibrium function  
    FLB::storef<PRECISION, 9>(idx, d_f, localFeq, neighborsIdx, t);
    //if (idx == 44) printf("333XXXSTREAM local_flag = %d idx = %d localf0 = %6.4f localf1 = %6.4f localf2 = %6.4f localf3 = %6.4f localf4 = %6.4f localf5 = %6.4f localf6 = %6.4f local7 = %6.4f localRho = %6.4f\n", +d_flags[idx], idx, d_f[0], localFeq[1], localFeq[2],localFeq[3] ,localFeq[4] ,localFeq[5] ,localFeq[6], localFeq[7] , localRho);
  }
  else if (localFlagExtracted == FLB::CT_INTERFACE_GAS)
  {
    unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
    FLB::calculateNeighborsIdxD2Q9(idx, x, y, neighborsIdx); // get neighbors indices
    for (int i = 1; i < NUMVELOCITIES; i++)
    {
      uint8_t neighborFlag = d_flags[neighborsIdx[i]];
      uint8_t neighborFlagExtracted = neighborFlag & (FLB::CT_INLET_FLUID | FLB::CT_INTERFACE);
      if (neighborFlagExtracted != FLB::CT_INLET_FLUID && (neighborFlagExtracted == FLB::CT_FLUID || neighborFlagExtracted == FLB::CT_INTERFACE_FLUID)) d_flags[neighborsIdx[i]] = (neighborFlag & ~FLB::CT_INTERFACE_FLUID) | FLB::CT_INTERFACE; // delete interface_fluid flag and change to interface
    }
  }
  if (idx == 43) printf("333XXXSTREAM local_flag = %d idx = %d \n", +d_flags[idx], idx);
}

template<typename PRECISION, int NUMVELOCITIES>
 __global__ void FLB::d_FreeSurface2D_4(PRECISION* d_rho, uint8_t* d_flags, PRECISION* d_mass, PRECISION* d_excessMass, PRECISION* d_phi)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
  if (x >= FLB::d_Nx || y >= FLB::d_Ny) return; // Prevent threads outside domain 
  
  const unsigned int idx = x + y * FLB::d_Nx;
  uint8_t localFlag = d_flags[idx]; //flag of the node
  // for the inlet the local mass is constant with a value of 1.0 and phi equal to 1.0
  if (localFlag & (FLB::CT_WALL | FLB::CT_MWALL | FLB::CT_INLET)) return;
  
  PRECISION localRho = d_rho[idx]; // local density
  PRECISION localMass = d_mass[idx]; // local mass
  PRECISION localExcessMass = 0.0f; // local excess mass
  PRECISION localPhi = 0.0f; // local fill level

  uint8_t localFlagExtracted = localFlag & (FLB::CT_FLUID_GAS_INTERFACE | FLB::CT_GAS_INTERFACE);
  if (idx == 43 || idx == 62 || idx ==63 || idx == 124) printf("444XXXSTREAM local_flag = %d localFlagExtracted = %d idx = %d localPhi = %6.4f localMass = %6.4f localRho = %6.4f\n", +d_flags[idx], localFlagExtracted, idx, d_phi[idx], d_mass[idx], d_rho[idx]);
  switch (localFlagExtracted) 
  {
    case 1: // FLB::CT_FLUID
    {
      localExcessMass = localMass - localRho; // save excess local mass
      localMass = localRho; //fluid mass has to be equal to local density
      localPhi = 1.0f;
      break;
    }

    case 2: // FLB::CT_GAS
    {
      localExcessMass = localMass; // excess mass is equal to all the local mass
      localMass = 0.0f; // local mass has to be 0
      localPhi = 0.0f;
      break;
    }
    
    case 4: // FLB::CT_INTERFACE
    {
      if (localMass > localRho)
      {
	localExcessMass = localMass - localRho;
	localMass = localRho;
      }
      else if (localMass < 0.0f) 
      {
	localExcessMass = localMass;
	localMass = 0.0f;
      }
      else localExcessMass = 0.0f; // Without excess mass the local mass doesn't change
      
      localPhi = FLB::calculatePhi(localMass, localRho);
      break;
    }
    
    case 5: // FLB::CT_INTERFACE_FLUID
    {
      d_flags[idx] = localFlag & ~FLB::CT_INTERFACE; // first delete flag of interface and Change node to fluid type (the fluid flag remains after the deletion)
      localExcessMass = localMass - localRho; // save excess local mass
      localMass = localRho; //fluid mass has to be equal to local density
      localPhi = 1.0f;
      break;
    }
    
    case 6: // FLB::CT_INTERFACE_GAS
    {
      d_flags[idx] = localFlag & ~FLB::CT_INTERFACE; // first delete flag of interface and change node to gas type (the inteface flag remains after the deletion)
      localExcessMass = localMass; // excess mass is equal to all the local mass
      localMass = 0.0f; // local mass has to be 0
      localPhi = 0.0f;
      break;
    }

    case 128: // FLB::CT_GAS_INTERFACE
    {
      d_flags[idx] = (localFlag & ~FLB::CT_GAS_INTERFACE) | FLB::CT_INTERFACE; // first delete flag of gas_interface and change to iterface type
      if (localMass > localRho)
      {
	localExcessMass = localMass - localRho;
	localMass = localRho;
      }
      else if (localMass < 0.0f) 
      {
	localExcessMass = localMass;
	localMass = 0.0f;
      }
      else localExcessMass = 0.0f; // Without excess mass the local mass doesn't change
      
      localPhi = FLB::calculatePhi(localMass, localRho);
      break;
    }  
    if (idx == 43 || idx == 44 || idx == 62) printf("444XXXSTREAM local_flag = %d idx = %d localPhi = %6.4f\n", +d_flags[idx], idx, d_phi[idx]);
  }
  
  unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
  FLB::calculateNeighborsIdxD2Q9(idx, x, y, neighborsIdx); // get neighbors indices
  int cont = 0; // Count neighboring nodes that are fluid or interface
  // The excess mass is evenly distributed for all neighboring nodes that are fluid or interface
  for (int i = 1; i < NUMVELOCITIES; i++)
  {
    uint8_t neighborFlagExtracted = d_flags[neighborsIdx[i]] & (FLB::CT_INTERFACE_FLUID | FLB::CT_GAS_INTERFACE); //neighbor flag
    cont += (int)(neighborFlagExtracted == FLB::CT_FLUID || neighborFlagExtracted == FLB::CT_INTERFACE || neighborFlagExtracted == FLB::CT_INTERFACE_FLUID || neighborFlagExtracted == FLB::CT_GAS_INTERFACE);
  }

  localMass += cont > 0 ? 0.0f : localExcessMass; // If there are not neighbors nodes of type fluid or interface save mass in local node to conserve mass
  localExcessMass = cont > 0 ? localExcessMass/(float)cont : 0.0f; // equal distribution of the mass for all neighboring nodes that are fluid or interface

  d_mass[idx] = localMass;
  d_excessMass[idx] = localExcessMass;
  d_phi[idx] = localPhi;
  if (idx == 43 || idx == 62) printf("444XXXSTREAM local_flag = %d idx = %d localMass = %6.4f localPhi = %6.4f\n", +d_flags[idx], idx, localMass, localPhi);
}

template<typename PRECISION, int NUMVELOCITIES>
__global__ void FLB::d_StreamCollide2D(PRECISION* d_f, PRECISION* d_rho, PRECISION* d_u, uint8_t* d_flags, PRECISION* d_mass, const unsigned long int t)
{
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;

  if (x >= d_Nx || y >= d_Ny) return; // Prevent threads outside domain 
  const unsigned int idx = x + y * FLB::d_Nx;
  
  const uint8_t localFlag = d_flags[idx];
  if (localFlag & (FLB::CT_GAS | FLB::CT_WALL)) return; // if cell is gas or static wall return

  unsigned int neighborsIdx[NUMVELOCITIES]; // neighbors indices
  FLB::calculateNeighborsIdxD2Q9(idx, x, y, neighborsIdx); // get neighbors indices

  PRECISION localf[NUMVELOCITIES]; // Local distribution function
  
  FLB::loadf<PRECISION, 9>(idx, d_f, localf, neighborsIdx, t); // First part of the esoteric pull. 
  PRECISION localRho, localUx, localUy; // Local density and velocities

  // Apply Boundary conditions in input and output
  const uint8_t localFlagInletOutlet = localFlag & (FLB::CT_INLET | FLB::CT_OUTLET); // extract inlet - outlet flag
  /*if (localFlagInletOutlet) 
  {
    localRho = d_rho[idx];
    localUx = d_u[idx];
    localUy = d_u[idx + FLB::d_N];
  }*/

  if (localFlagInletOutlet & FLB::CT_INLET) 
  {
    localRho = d_rho[idx+1];
    //FLB::calculateDensity<PRECISION, 9>(localf, localRho);
    localUx = d_u[idx];
    localUy = d_u[idx + FLB::d_N];
  }
  else if (localFlag & FLB::CT_OUTLET)
  {
    localRho = d_rho[idx];
    localUx = d_u[idx - 1];
    localUy = d_u[idx + FLB::d_N - 1];
  }
  else
  { //Calculate density and velocity
    FLB::calculateDensity<PRECISION, 9>(localf, localRho);
    FLB::calculateVelocityD2Q9<PRECISION>(localf, localRho, localUx, localUy);
  }
  // After collision it is neccesary to check if the interface flag needs to change
  const uint8_t localFlagInterface = localFlag & FLB::CT_INTERFACE; // extract interface flag
  if (localFlagInterface == FLB::CT_INTERFACE)
  {
    const PRECISION localMass = d_mass[idx]; //local mass
    bool neighborsNotFluid = true, neighborsNotGas = true; // the node doesn't have a neighbor with fluid or gas
    for (int i = 1; i < NUMVELOCITIES; i++)
    {
      uint8_t neighborFlagExtracted = d_flags[neighborsIdx[i]] & FLB::CT_FLUID_GAS_INTERFACE; //neighbor flag
      neighborsNotFluid = neighborsNotFluid && (neighborFlagExtracted != FLB::CT_FLUID);
      neighborsNotGas = neighborsNotGas && (neighborFlagExtracted != FLB::CT_GAS);
    }
    // (1 + epsilon) * d_rhol to prevents the unwanted effect of fast oscillations between cell type
    if ((localMass > 1.001f * localRho) || neighborsNotGas) d_flags[idx] = (localFlag & ~FLB::CT_FLUID_GAS_INTERFACE) | FLB::CT_INTERFACE_FLUID; // change flag to interface - fluid
    else if ((localMass < -0.001f) || neighborsNotFluid) d_flags[idx] = (localFlag & ~FLB::CT_FLUID_GAS_INTERFACE) | FLB::CT_INTERFACE_GAS; // change flag to interface - gas  
  }

    
  // VOLUME FORCES (Guo Forcing Scheme)
  PRECISION localFi[NUMVELOCITIES]; // Forcing terms
  if (FLB::d_useVolumetricForce)
  {
    const PRECISION rhol2 = 0.5f / localRho;
    // Correct local velocities with volume forces
    localUx = localUx + rhol2 * FLB::d_Fx;
    localUy = localUy + rhol2 * FLB::d_Fy;
    FLB::calculateForcingTerms2D<PRECISION>(localFi, localUx, localUy);
    if (idx == 90013 && (t == 10000) ) printf("ZZZZZZZSTREAM fx = %10.6f localUx = %10.6f\n", FLB::d_Fx, localUx);
    if (idx == 90013 && (t == 10000)) printf("STREAM local_flag = %d idx = %d localf0 = %10.6f localf1 = %10.6f localf2 = %10.6f localf3 = %10.6f localf4 = %10.6f localf5 = %10.6f localf6 = %10.6f local7 = %10.6f localFi8 = %10.6f\n", +d_flags[idx], idx, localFi[0], localFi[1], localFi[2],localFi[3] ,localFi[4] ,localFi[5] ,localFi[6], localFi[7], localFi[8] );
  }
    //if (idx == 124) printf("STREAM local_flag = %d idx = %d localf0 = %6.4f localf1 = %6.4f localf2 = %6.4f localf3 = %6.4f localf4 = %6.4f localf5 = %6.4f localf6 = %6.4f local7 = %6.4f localFi\n", +d_flags[idx], idx, localFi[0], localFi[1], localFi[2],localFi[3] ,localFi[4] ,localFi[5] ,localFi[6], localFi[7] );

  // Update fields except if flag is input or output (boundary)
  if (!localFlagInletOutlet)
  {
    d_rho[idx] = localRho;
    d_u[idx] = localUx;
    d_u[idx + FLB::d_N] = localUy;

  }
  else if (localFlag & FLB::CT_OUTLET)
  {
    d_u[idx] = localUx;
    d_u[idx + FLB::d_N] = localUy;
  }
  // TODO
  d_rho[idx] = localRho;


  // Calculate equilibrium function
  PRECISION localFeq[NUMVELOCITIES];
  FLB::calculateEquilibriumDDF2Q9<PRECISION>(localRho, localUx, localUy, localFeq);
  if ((idx == 80020 || idx == 80021) && (t==0 || t== 1 || t == 2 || t == 3 || t==400)) 
  {
      //printf("idx = %d  localf1 = %6.4f localf2 = %6.4f 3 = %6.4f 4 = %6.4f 5 = %6.4f 6 = %6.4f  7 = %6.4f  8 = %6.4f t = %d\n",idx, localf[1], localf[2], localf[3], localf[4], localf[5],localf[6],localf[7],localf[8], t);
      printf("idx = %d  localFeq1 = %6.4f localFeq2 = %6.4f 3 = %6.4f 4 = %6.4f 5 = %6.4f 6 = %6.4f  7 = %6.4f  8 = %6.4f t = %d\n",idx, localFeq[1], localFeq[2], localFeq[3], localFeq[4], localf[5],localFeq[6],localFeq[7],localFeq[8], t);
    //float rhoRho = localf[0];
    //for (int i = 1; i < NUMVELOCITIES; i++) rhoRho += localf[i];
    //float uxl = (localf[1] - localf[2] + localf[5] - localf[6] + localf[7] - localf[8]) / localRho;
    //float uxlEq = (localFeq[1] - localFeq[2] + localFeq[5] - localFeq[6] + localFeq[7] - localFeq[8]) / localRho;
    printf("idx = %d localUx = %6.4f localRho = %6.4f t = %d\n", idx, localUx, localRho, t);
  }

  float invTau = FLB::d_invTau; //Inverse relaxation time

  if (FLB::d_useSubgridModel)
  {
    // Source : A Lattice Boltzman Subgrid Model for High Reynolds Number Flows 
    // Smagorinsky subgrid turbulence model
    // local nonequilibrium stress tensor
    PRECISION HNonEqxx = 0.0f, HNonEqyy = 0.0f;
    PRECISION HNonEqxy = 0.0f; // the tensor is symmetric
    for (int i = 1; i < NUMVELOCITIES; i++)
    {
      const PRECISION fNonEq = localf[i] - localFeq[i];
      const PRECISION cx = FLB::d_cx<NUMVELOCITIES>[i];
      const PRECISION cy = FLB::d_cy<NUMVELOCITIES>[i];
      HNonEqxx += cx * cx * fNonEq; 
      HNonEqyy += cy * cy * fNonEq; 
      HNonEqxy += cx * cy * fNonEq; 
    }
    // calculate the norm (squared) of the stress tensor
    const PRECISION Q = HNonEqxx * HNonEqxx + HNonEqyy * HNonEqyy + 2.0f * HNonEqxy * HNonEqxy;

    const float tau = 1.0f / invTau;
    // correct relaxation time with subgrid viscosity by increasing the total viscosity
    // The value of the Smagorinsky Coefficient is from Lilly (1966)
    // C = 1 / pi * (2 / (3 * Ck))^(3/4) with Ck = 1.5 (Kolmogorov Constant)
    // 0.54037965 = 18 * (C * Delta) ^ 2 . Delta = 1 (distance between nodes)
    invTau = 2.0f / (tau + sqrtf(tau * tau + 0.54037965f * sqrtf(Q) / localRho));
  }

  switch (FLB::d_collisionOperator) //Collision
  {
    case 0:// SRT
    {
      // Perform  collision
      if (!localFlagInletOutlet) // not inlet or outlet
      {
	if (FLB::d_useVolumetricForce)
	{
	  const float taul = 1.0f - 0.5f * invTau; //Relaxation of the forcing terms	
  //if (idx == 124) printf("STREAM local_flag = %d idx = %d localf0 = %6.4f localf1 = %6.4f localf2 = %6.4f localf3 = %6.4f localf4 = %6.4f localf5 = %6.4f localf6 = %6.4f local7 = %6.4f taul = %6.4f\n", +d_flags[idx], idx, localf[0], localf[1], localf[2],localf[3] ,localf[4] ,localf[5] ,localf[6], localf[7] , taul);
  
  //if (idx == 124) printf("idx = 2  valor = %6.4f valor2 = %6.4f, valor3 = %6.4f localFi = %6.4f\n", (1.0f - invTau)*localf[2], invTau*localFeq[2], taul*localFi[2], localFi[2]);

	  for (int i = 0; i < NUMVELOCITIES; i++) localf[i] = (1.0f - invTau) * localf[i] + invTau * localFeq[i] + taul * localFi[i];

  //if (idx == 124) printf("STREAM local_flag = %d idx = %d localf0 = %6.4f localf1 = %6.4f localf2 = %6.4f localf3 = %6.4f localf4 = %6.4f localf5 = %6.4f localf6 = %6.4f local7 = %6.4f\n", +d_flags[idx], idx, localf[0], localf[1], localf[2],localf[3] ,localf[4] ,localf[5] ,localf[6], localf[7] );
	}
	else
	{
	  for (int i = 0; i < NUMVELOCITIES; i++) localf[i] = (1.0f - invTau) * localf[i] + invTau * localFeq[i];
	}
      }
      // Boundary condition: Local distribution function in input or output
      else
      {
	for (int i = 0; i < NUMVELOCITIES; i++) localf[i] = localFeq[i];
      }
      break;
    }

    case 1: // TRT
    {
      const float invTauPos = invTau; // inverse of the positive relaxation time 
      const float invTauNeg = 1.0f/(0.25f/(1.0f/invTau - 0.5f) + 0.5f); // inverse of the negative relaxation time. Value of magic A for best stability A = 1/4
      // Perform  collision
      if (!localFlagInletOutlet)
      {

      // decompose the population in the symmetric and antisymmetric parts
	// TODO
	if (FLB::d_useVolumetricForce)
	{

	}
	else 
	{ 
	  //d_localf[0] = d_localf[0] - 0.5f * invTauPos * (d_localf[0] - d_feq[0] + d_localf[0] - d_feq[0]) - 0.5f * invTauNeg * (d_localf[0] - d_localf[0] + d_feq[0] - d_feq[0]);
	  localf[0] = localf[0] - invTauPos * (localf[0] - localFeq[0]);
	  for (int i = 1; i < NUMVELOCITIES; i += 2)
	  {
	    float fMinusFeq = localf[i] - localFeq[i] + localf[i + 1] - localFeq[i + 1]; // subtration of the symmetry populations
	    float fAntisymmetric = localf[i] - localf[i + 1]; // antisymmetric part of the populations
	    float feqAntisymmetric = localFeq[i] - localFeq[i + 1];

	    localf[i]     = localf[i] - 0.5f * invTauPos * fMinusFeq - 0.5f * invTauNeg * (fAntisymmetric - feqAntisymmetric);
	    localf[i + 1] = localf[i + 1] - 0.5f * invTauPos * fMinusFeq - 0.5f * invTauNeg * (feqAntisymmetric - fAntisymmetric);
	  }
	}
      }
      // Boundary condition: Local distribution function in input or output
      else
      {
	for (int i = 0; i < NUMVELOCITIES; i++) localf[i] = localFeq[i];
      }
      break;
    } 
  }
  if ((idx == 80020 || idx == 80021) && (t==0 || t== 1 || t ==2 || t ==3 || t==400)) 
  {
      //printf("idx = %d neighbor1 = %d neighbor2 = %d neighbor3 = %d neighbor4 = %d neighbor5 = %d neighbor6 = %d neighbor7 = %d neighbor8 = %d \n", idx, neighborsIdx[1], neighborsIdx[2], neighborsIdx[3], neighborsIdx[4], neighborsIdx[5], neighborsIdx[6], neighborsIdx[7], neighborsIdx[8]);
      printf("idx = %d  localf1 = %6.4f 2 = %6.4f 3 = %6.4f 4 = %6.4f 5 = %6.4f 6 = %6.4f  7 = %6.4f  8 = %6.4f localFlagInletOutlet = %d localFlag = %d t = %d\n\n",idx, localf[1], localf[2], localf[3], localf[4], localf[5],localf[6],localf[7],localf[8], localFlagInletOutlet, localFlag, t);
  }

  FLB::storef<PRECISION, 9>(idx, d_f, localf, neighborsIdx, t); // second part of esoteric pull.
}


// Explicit instantation of the functions because the templates functions are instantiated in a diferent compilation unit
template __global__ void FLB::d_FreeSurface2D_1<float, 9>(float* d_f, const float* d_rho, const float* d_u, const uint8_t* d_flags, float* d_mass, float* d_excessMass, float* d_phi, const unsigned long int t);

template __global__ void FLB::d_FreeSurface2D_1<double, 9>(double* d_f, const double* d_rho, const double* d_u, const uint8_t* d_flags, double* d_mass, double* d_excessMass, double* d_phi, const unsigned long int t);

template __global__ void FLB::d_FreeSurface2D_2<9>(uint8_t* d_flags);

template __global__ void FLB::d_FreeSurface2D_3<float, 9>(float* d_f, float* d_rho, float* d_u, uint8_t* d_flags, const unsigned long int t);

template __global__ void FLB::d_FreeSurface2D_3<double, 9>(double* d_f, double* d_rho, double* d_u, uint8_t* d_flags, const unsigned long int t);

template __global__ void FLB::d_FreeSurface2D_4<float, 9>(float* d_rho, uint8_t* d_flags, float* d_mass, float* d_excessMass, float* d_phi);

template __global__ void FLB::d_FreeSurface2D_4<double, 9>(double* d_rho, uint8_t* d_flags, double* d_mass, double* d_excessMass, double* d_phi);

template __global__ void FLB::d_StreamCollide2D<float, 9>(float* d_f, float* d_rho, float* d_u, uint8_t *d_flags, float* d_mass, const unsigned long int t);

template __global__ void FLB::d_StreamCollide2D<double, 9>(double* d_f, double* d_rho, double* d_u, uint8_t *d_flags, double* d_mass, const unsigned long int t); 
