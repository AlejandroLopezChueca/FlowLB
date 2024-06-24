#pragma once

#include <cstdint>
#include <sys/types.h>

#include "cudaInitData.cuh"

namespace FLB
{
  /**
    * @brief Calculate local density for a cell
    *
    * @param[in]       d_localf  Local distribution function
    * @param[in, out]  d_rhol    Local density
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __device__ void calculateDensity(const PRECISION* localf, PRECISION& localRho)
  {
    PRECISION rho = localf[0];
    for (int i = 1; i < NUMVELOCITIES; i++) rho += localf[i];
    localRho = rho;
  }

  /**
    * @brief Calculate the equilibrium distribution function for D2Q9 
    *
    * @param[in]       d_rhol  Local density of the node
    * @param[in]       d_uxl   Local velocity in x direction
    * @param[in]       d_uyl   Local velocity in y direction
    * @param[in, out]  d_feq   equilibrium distribution function of the node
    */
  template<typename PRECISION>
  __device__ void calculateEquilibriumDDF2Q9(PRECISION localRho, PRECISION localUx, PRECISION localUy, PRECISION* localFeq)
  {
    const PRECISION u2 = 3.0f * (localUx * localUx + localUy * localUy);
    const PRECISION rhow14 = FLB::d_weights<PRECISION, 9>[1] * localRho;
    const PRECISION rhow58 = FLB::d_weights<PRECISION, 9>[5] * localRho;
    localUx *= 3.0f;
    localUy *= 3.0f;
    const PRECISION u0 = localUx - localUy;
    const PRECISION u1 = localUx + localUy;

    localFeq[0] = FLB::d_weights<PRECISION, 9>[0] * localRho * (1.0f - 0.5f * u2);
    localFeq[1] = rhow14 * (1.0f + localUx - 0.5f * u2 + 0.5f * localUx * localUx);
    localFeq[2] = rhow14 * (1.0f - localUx - 0.5f * u2 + 0.5f * localUx * localUx);
    localFeq[3] = rhow14 * (1.0f + localUy - 0.5f * u2 + 0.5f * localUy * localUy);
    localFeq[4] = rhow14 * (1.0f - localUy - 0.5f * u2 + 0.5f * localUy * localUy);
    localFeq[5] = rhow58 * (1.0f + u1 - 0.5f * u2 + 0.5f * u1 * u1);
    localFeq[6] = rhow58 * (1.0f - u1 - 0.5f * u2 + 0.5f * u1 * u1);
    localFeq[7] = rhow58 * (1.0f + u0 - 0.5f * u2 + 0.5f * u0 * u0);
    localFeq[8] = rhow58 * (1.0f - u0 - 0.5f * u2 + 0.5f * u0 * u0);
  }

  /**
    * @brief Calculate the forcing tems for a volume force using the Guo Scheme in 2D. This forcing terms needs to be relaxed according to the respective collision operator
    *
    * @param[in, out]  d_Fi  Forcing terms
    * @param[in]       d_Fx  Force in x direction
    * @param[in]       d_Fy  Force in y directino
    * @param[in]       d_uxl Local velocity in x direction
    * @param[in]       d_uyl Local velocity in y direction
    * @param[in]
    */
  template<typename PRECISION>
  __device__ void calculateForcingTerms2D(PRECISION* localFi, PRECISION localUx, PRECISION localUy)
  {
    const PRECISION uF = -0.33333334f * (localUx * FLB::d_Fx + localUy * FLB::d_Fy);
    localFi[0] = 9.0f * FLB::d_weights<PRECISION, 9>[0] * uF;
    for (int i = 1; i < 9; i++)
    {
      localFi[i] = 9.0f * FLB::d_weights<PRECISION, 9>[i] * ((FLB::d_cx<9>[i] * FLB::d_Fx + FLB::d_cy<9>[i] * FLB::d_Fy) * (FLB::d_cx<9>[i] * localUx + FLB::d_cy<9>[i] * localUy + 0.33333334f) + uF);
    }
  }

  /** 
    * @brief Calculate the neighbors indices of a cell for D2Q9.
    *
    * @param[in]       idx            Index of the cell
    * @param[in, out]  neighborsIdx   Neighbor Indices
    * @param[in]       t              Lattice time 
    */
  __device__ void calculateNeighborsIdxD2Q9(const unsigned int idx, const unsigned int x, const unsigned y, unsigned int* neighborsIdx)
  {
    // Directions based in cuda (0, 0) in the upper left corner
    /* 
       8 3 5  
        ***
       2*0*1
        ***
       6 4 7
       */
    // The index are calculated for a periodic boundary by default
    const unsigned int yCenter = y * FLB::d_Nx;
    //const unsigned int yUp = ((y + FLB::d_Ny - 1u) % FLB::d_Ny) * FLB::d_Nx;
    const unsigned int yUp = ((y + 1u) % FLB::d_Ny) * FLB::d_Nx;
    const unsigned int yDown = ((y + FLB::d_Ny - 1u) % FLB::d_Ny) * FLB::d_Nx;
    //const unsigned int yDown = ((y + 1u) % FLB::d_Ny) * FLB::d_Nx;
    const unsigned int xRight = (x + 1u) % FLB::d_Nx;
    const unsigned int xLeft = (x + FLB::d_Nx - 1u) % FLB::d_Nx;
    neighborsIdx[0] = idx; //Own cell 
    neighborsIdx[1] = xRight + yCenter; 
    neighborsIdx[2] = xLeft + yCenter; 
    neighborsIdx[3] = x + yUp; 
    neighborsIdx[4] = x + yDown; 
    neighborsIdx[5] = xRight + yUp;
    neighborsIdx[6] = xLeft + yDown; 
    neighborsIdx[7] = xRight + yDown; 
    neighborsIdx[8] = xLeft + yUp;
  }

  /**
    * @brief Calculate the average values of the density and velocities from the values of the neighbors nodes that are fluid, interface or interface -> fluid.
    *
    * @param[in]       idx           Global index of the node
    * @param[in]       d_flags       Flags of all the domain
    * @param[in, out]  d_rhol        Local density of the node
    * @param[in, out]  d_uxl         Local velocity of the node in the x direction
    * @param[in, out]  d_uyl         Local velocity of the node in the y direction
    * @param[in]       d_rho         Density of all the domain
    * @param[in]       d_ux          Velocity in the x direction of all the domain
    * @param[in]       d_uy          Velocity in the y direction of all the domain
    * @param[in]       neighborsIdx  Indexes of the neighbors nodes
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __device__ void calculateAverageValuesNeighbors2D(const unsigned int idx, uint8_t* d_flags, PRECISION& rhol, PRECISION& uxl, PRECISION& uyl, PRECISION* d_rho, PRECISION* d_u, const unsigned int* neighborsIdx)
    {
      float count = 0.0f, localRho = 0.0f, localUx = 0.0f, localUy = 0.0f;
      const unsigned int numberNodes = FLB::d_N;

      for (int i = 1; i < NUMVELOCITIES; i++)
      {
	const uint8_t neighborFlag = d_flags[neighborsIdx[i]]; //neighbor flag
	//if (neighborFlag == FLB::CT_FLUID || neighborFlag == FLB::CT_INTERFACE || neighborFlag == FLB::CT_INTERFACE_FLUID)
	if (neighborFlag & FLB::CT_INTERFACE_FLUID) // if the neighbor is fluid, interface or interface_fluid
	{
	  count += 1.0f;
	  localRho += d_rho[neighborsIdx[i]];
	  localUx += d_u[neighborsIdx[i]];
	  localUy += d_u[neighborsIdx[i] + numberNodes];
	}
      }
      rhol = count > 0.0f ? localRho/count : 0.0f;
      uxl = count > 0.0f ? localUx/count : 0.0f;
      uyl = count > 0.0f ? localUy/count : 0.0f;
    }

  /**
    * @brief Calculate the fill level of local node when is interface or it is going to be interface
    * 
    * @param[in]   localMass    mass of the local node
    * @param[in]   localRho     Local density
    */
  template<typename PRECISION>
  __device__ PRECISION calculatePhi(const PRECISION localMass, const PRECISION localRho)
  {
    if (localRho > 0.0f) return __saturatef(localMass/localRho); // clamp to [0.0, 1.0]
    else return 0.5f;
  }

  /**
    * @brief Calculate local velocities for a cell
    *
    * @param[in]       localf  Local distribution function
    * @param[in]       rhol    Local density
    * @param[in, out]  uxl     Local velocity in x direction
    * @param[in, out]  uyl     Local velocity in y direction  
    */
  template<typename PRECISION>
  __device__ void calculateVelocityD2Q9(const PRECISION* localf, const PRECISION rhol, PRECISION& uxl, PRECISION& uyl)
  {
    // Alternating to reduce absolute values of intermediate sum and achieve better precision
    uxl = (localf[1] - localf[2] + localf[5] - localf[6] + localf[7] - localf[8]) / rhol;
    uyl = (localf[3] - localf[4] + localf[5] - localf[6] + localf[8] - localf[7]) / rhol;
  }

  /**
    * @brief Load data (distribution function) from actual node and from neighbors nodes in local distribution function of the current node.
    *
    * @param[in]       idx            Index of the cell
    * @param[in]       d_f            General particle dstribution function
    * @param[in, out]  d_localf       Local particle distribution fucntion of the node
    * @param[in]       neighborsIdx   Neighbors Indices
    * @param[in]       t              Lattice time 
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __device__ void loadf(const unsigned int idx, const PRECISION* d_f, PRECISION* localf, const unsigned int* neighborsIdx, const unsigned long int t)
  { 
    // Esoteric pull
    localf[0] = d_f[idx]; 
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      localf[i]     = d_f[idx + FLB::d_N * (t%2ul ? i : i + 1)]; // ternary operator to distinguis beetween odd and even time step
      localf[i + 1] = d_f[neighborsIdx[i] + FLB::d_N * (t%2ul ? i + 1 : i)];
      //if (idx == 4809 && (t== 0 || t == 1)) printf("i = %d, localF i + 1 = %6.4f i = %d neighbor = %d\n", i, localf[i+1], t%2?i:i+1, neighborsIdx[i]);
    }
  }

  /**
    * @brief Load data of the particle distribution function that is going to be streamed from the local node to the neighbors node 
    *
    * @param[in]       idx               Index of the cell
    * @param[in]       d_f               General particle dstribution function
    * @param[in, out]  d_outgoingLocalf  Local outgoing particle distribution fucntion of the node
    * @param[in]       neighborsIdx      Neighbors Indices
    * @param[in]       t                 Lattice time 
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __device__ void loadOutgoingf(const unsigned int idx, PRECISION* d_f, PRECISION* d_outgoingLocalf, const unsigned int* neighborsIdx, const unsigned long int t)
  { // esoteric pull
    // TODO correct index
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      d_outgoingLocalf[i]     = d_f[neighborsIdx[i] + FLB::d_N * (t%2ul ? i : i + 1)];
      d_outgoingLocalf[i + 1] = d_f[idx + FLB::d_N * (t%2ul ? i + 1: i)];
      //if (idx == 43) printf("INCO2 f1 = %6.4f f2 = %6.4f, nei = %d\n", d_outgoingLocalf[i], d_outgoingLocalf[i+1], neighborsIdx[i]);
    }
  }

  /**
    * @brief Store the particle distribution function after collision
    *
    * @param[in]       idx            Index of the cell
    * @param[in, out]  d_f            General particle dstribution function
    * @param[in]       localf         Local particle distribution function to the cell
    * @param[in]       neighborsIdx   Neighbors Indices
    * @param[in]       t              Lattice time 
    */ 
  template<typename PRECISION, int NUMVELOCITIES>
  __device__ void storef(const unsigned int idx, PRECISION* d_f, const PRECISION* localf, const unsigned int* neighborsIdx, const unsigned long int t)
  {
    // esoteric pull
    d_f[idx] = localf[0];
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      d_f[neighborsIdx[i] + FLB::d_N * (t%2ul ? i + 1 : i)] = localf[i]; 
      d_f[idx + FLB::d_N * (t%2ul ? i : i + 1)] = localf[i + 1]; 
    }
  }

  /**
    * @brief Store the reconstructed particle distribution function for the neighbors nodes only if they are gas type
    *
    * @param[in]       idx               Index of the node
    * @param[in, out]  d_f               General particle dstribution function
    * @param[in]       incomingLocalf    Particle distribution function incoming in the next step
    * @param[in]       neighborsFlag     Neighbors flags
    * @param[in]       neighborsIdx      Neighbors Indices
    * @param[in]       t                 Lattice time 
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __device__ void storeReconstructedGasf(const unsigned int idx, PRECISION* d_f, PRECISION* incomingLocalf, uint8_t* neighborsFlag, const unsigned int* neighborsIdx, const unsigned long int t)
  {
    // TODO correct index
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      if (neighborsFlag[i] == FLB::CT_GAS) d_f[idx + FLB::d_N * (t%2ul ? i : i + 1)] = incomingLocalf[i];
      if (neighborsFlag[i + 1] == FLB::CT_GAS) d_f[neighborsIdx[i] + FLB::d_N * (t%2ul ? i + 1 : i)] = incomingLocalf[i + 1];
    }
  }
}
