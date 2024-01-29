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
  __device__ void calculateDensity(const PRECISION* d_localf, PRECISION& d_rhol)
  { 
    d_rhol = d_localf[0];
    for (int i = 1; i < NUMVELOCITIES; i++) d_rhol += d_localf[i];
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
  __device__ void calculateEquilibriumDDF2Q9(const PRECISION d_rhol, PRECISION d_uxl, PRECISION d_uyl, PRECISION* d_feq)
  {
    const PRECISION u2 = 3.0f * (d_uxl * d_uxl + d_uyl * d_uyl);
    const PRECISION rhow14 = FLB::d_weights<PRECISION, 9>[1] * d_rhol;
    const PRECISION rhow58 = FLB::d_weights<PRECISION, 9>[5] * d_rhol;
    d_uxl *= 3.0f;
    d_uyl *= 3.0f;
    const PRECISION u0 = d_uxl - d_uyl;
    const PRECISION u1 = d_uxl + d_uyl;

    d_feq[0] = FLB::d_weights<PRECISION, 9>[0] * d_rhol * (1.0f - 0.5f * u2);

    d_feq[1] = rhow14 * (1.0f + d_uxl - 0.5f * u2 + 0.5f * d_uxl * d_uxl);
    d_feq[2] = rhow14 * (1.0f - d_uxl - 0.5f * u2 + 0.5f * d_uxl * d_uxl);
    d_feq[3] = rhow14 * (1.0f + d_uyl - 0.5f * u2 + 0.5f * d_uyl * d_uyl);
    d_feq[4] = rhow14 * (1.0f - d_uyl - 0.5f * u2 + 0.5f * d_uyl * d_uyl);
    d_feq[5] = rhow58 * (1.0f + u1   - 0.5f * u2 + 0.5f * u1 * u1);
    d_feq[6] = rhow58 * (1.0f - u1   - 0.5f * u2 + 0.5f * u1 * u1);
    d_feq[7] = rhow58 * (1.0f + u0   - 0.5f * u2 + 0.5f * u0 * u0);
    d_feq[8] = rhow58 * (1.0f - u0   - 0.5f * u2 + 0.5f * u0 * u0);
    //printf("feq0 = %6.4f feq1 = %6.4f feq2 = %6.4f feq3 = %6.4f\n", d_feq[0], d_feq[1], d_feq[2], d_feq[3]);
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
  __device__ void calculateForcingTerms2D(PRECISION* d_Fi, PRECISION d_Fx, PRECISION d_Fy, PRECISION d_uxl, PRECISION d_uyl)
  {
    const PRECISION uF = -0.33333334f * (d_uxl * d_Fx + d_uyl * d_Fy);
    d_Fi[0] = 9.0f * FLB::d_weights<PRECISION, 9>[0] * uF;
    for (int i = 1; i < 9; i++)
    {
      d_Fi[1] = 9.0f * FLB::d_weights<PRECISION, 9>[i] * ((FLB::d_cx<9>[i] * FLB::d_Fx + FLB::d_cy<9>[i] * FLB::d_Fy) * (FLB::d_cx<9>[i] * d_uxl + FLB::d_cy<9>[i] * d_uyl + 0.33333334f) + uF);
    }
  }

  /** 
    * @brief Calculate the neighbors indices of a cell for D2Q9.
    *
    * @param[in]       idx            Index of the cell
    * @param[in, out]  neighborsIdx   Neighbor Indices
    * @param[in]       t              Lattice time 
    */
  __device__ void calculateNeighborsIdxD2Q9(const unsigned int idx, unsigned int* neighborsIdx)
  {
    // Directions based in cuda (0, 0) in the upper left corner
    /* 
       8 3 5  
        ***
       2*0*1
        ***
       6 4 7
       */ 
    neighborsIdx[0] = idx; //Own cell 
    neighborsIdx[1] = idx + 1; 
    neighborsIdx[2] = idx - 1; 
    neighborsIdx[3] = idx - d_Nx; 
    neighborsIdx[4] = idx + d_Nx; 
    neighborsIdx[5] = idx + 1 - d_Nx;
    neighborsIdx[6] = idx - 1 + d_Nx; 
    neighborsIdx[7] = idx + 1 + d_Nx; 
    neighborsIdx[8] = idx - 1 - d_Nx;
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
  __device__ void calculateAverageValuesNeigbors2D(const unsigned int idx,uint8_t* d_flags, PRECISION& d_rhol, PRECISION& d_uxl, PRECISION& d_uyl, PRECISION* d_rho, PRECISION* d_ux, PRECISION* d_uy, PRECISION* neighborsIdx)
    {
      float count = 0.0f;
      for (int i = 1; i < NUMVELOCITIES; i++)
      {
	uint8_t d_neighborFlag = d_flags[neighborsIdx[i]]; //neighbor flag
	if (d_neighborFlag == FLB::CT_FLUID || d_neighborFlag == FLB::CT_INTERFACE || d_neighborFlag == FLB::CT_INTERFACE_FLUID)
	{
	  count += 1.0f;
	  d_rhol += d_rho[idx];
	  d_uxl += d_ux[idx];
	  d_uxl += d_uy[idx];
	}	
      }
      d_rhol = count > 0.0f ? d_rhol/count : 0.0f;
      d_uxl = count > 0.0f ? d_uxl/count : 0.0f;
      d_uyl = count > 0.0f ? d_uyl/count : 0.0f;
    }

  /**
    * @brief Calculate the fill level of local node when is interface or it is going to be interface
    * 
    * @param[in]   d_excessMassl  Excess mass of the local node
    * @param[in]   d_rhol         Local density
    * @param[out]  d_phil         Local fill level
    */
  template<typename PRECISION>
  __device__ PRECISION calculatePhi(PRECISION d_excessMassl, PRECISION d_rhol)
  {
    // TODO 
    //if (d_rhol > 0.0f)
    {
      PRECISION d_phil = d_excessMassl/d_rhol;
      if (d_phil > 0.0f) return d_phil < 1.0f ? d_phil : 1.0f;  
      else return 0.0f;
    }
  }

  /**
    * @brief Calculate local velocities for a cell
    *
    * @param[in]       d_localf  Local distribution function
    * @param[in]       d_rhol    Local density
    * @param[in, out]  d_uxl     Local velocity in x direction
    * @param[in, out]  d_uyl     Local velocity in y direction  
    */
  template<typename PRECISION>
  __device__ void calculateVelocityD2Q9(const PRECISION* d_localf, const PRECISION d_rhol, PRECISION& d_uxl, PRECISION& d_uyl)
  {
    // Alternating to reduce absolute values of intermediate sum and achive better precision
    d_uxl = (d_localf[1] - d_localf[2] + d_localf[5] - d_localf[8] + d_localf[7] - d_localf[6]) / d_rhol;
    d_uyl = (d_localf[3] - d_localf[4] + d_localf[5] - d_localf[7] + d_localf[8] - d_localf[6]) / d_rhol;
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
  __device__ void loadf(const unsigned int idx, const PRECISION* d_f, PRECISION* d_localf, const unsigned int* neighborsIdx, const unsigned long int t)
  { 
    // Esoteric pull
    d_localf[0] = d_f[idx]; 
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      d_localf[i]     = d_f[idx + FLB::d_N * (t%2ul ? i + 1 : i)]; // ternary operator to distinguis beetween odd and even time step
      d_localf[i + 1] = d_f[neighborsIdx[i] + FLB::d_N * (t%2ul ? i : i + 1)];
      //printf("IDX = %6.4f\n", d_f[idx + FLB::d_N * (t%2ul ? i + 1 : i)]);
    }
    //printf("d_localf LOAD F local1 = %6.4f local2 = %6.4f local5 = %6.4f local8 = %6.4f local7 = %6.4f local6 = %6.4f\n", d_localf[1], d_localf[2], d_localf[5], d_localf[8], d_localf[7], d_localf[6]);
    //printf("d_f LOAD F local1 = %6.4f local2 = %6.4f local5 = %6.4f local8 = %6.4f local7 = %6.4f local6 = %6.4f\n", d_f[1], d_f[2], d_f[5], d_f[8], d_f[7], d_f[6]);
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
    for (int i = 1; i < NUMVELOCITIES; i +=2)
    {
      d_outgoingLocalf[i]     = d_f[neighborsIdx[i] + FLB::d_N * (t%2ul ? i +1 : i)];
      d_outgoingLocalf[i + 1] = d_f[idx + FLB::d_N * (t%2ul ? i : i + 1)];
    }
  }

  /**
    * @brief Store the particle distribution function after collision
    *
    * @param[in]       idx            Index of the cell
    * @param[in, out]  d_f            General particle dstribution function
    * @param[in]       d_localf       Local particle distribution function to the cell
    * @param[in]       neighborsIdx   Neighbors Indices
    * @param[in]       t              Lattice time 
    */ 
  template<typename PRECISION, int NUMVELOCITIES>
  __device__ void storef(const unsigned int idx, PRECISION* d_f, const PRECISION* d_localf, const unsigned int* neighborsIdx, const unsigned long int t)
  {
    // esoteric pull
    d_f[idx] = d_localf[0];
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      d_f[neighborsIdx[i] + FLB::d_N * (t%2ul ? i : i + 1)] = d_localf[i]; 
      d_f[idx + FLB::d_N * (t%2ul ? i + 1 : i)] = d_localf[i + 1];
      
      //printf("LOCAL_FFFFF = %6.4f LOCAL_F2 = %6.4f idx = %d i = %d\n", d_localf[i], d_localf[i+1], idx, i);
    }
    //printf("LOCAL_F0 = %6.4f idx = %d \n", d_localf[0], idx);
  }

  /**
    * @brief Store the reconstructed particle distribution function for the neighbors nodes only if they are gas type
    *
    * @param[in]       idx               Index of the node
    * @param[in, out]  d_f               General particle dstribution function
    * @param[in]       d_incomingLocalf  Particle distribution function incoming in the next step
    * @param[in]       d_neighborsFlag   Neighbors flags
    * @param[in]       neighborsIdx      Neighbors Indices
    * @param[in]       t                 Lattice time 
    */
  template<typename PRECISION, int NUMVELOCITIES>
  __device__ void storeReconstructedGasf(const unsigned int idx, PRECISION* d_f, PRECISION* d_incomingLocalf, uint8_t* d_neighborsFlag, const unsigned int* neighborsIdx, const unsigned long int t)
  {
    for (int i = 1; i < NUMVELOCITIES; i += 2)
    {
      if (d_neighborsFlag[i] == FLB::CT_GAS) d_f[idx + FLB::d_N * (t%2ul ? i : i + 1)] = d_incomingLocalf[i];
      if (d_neighborsFlag[i + 1] == FLB::CT_GAS) d_f[neighborsIdx[i] + FLB::d_N * (t%2ul ? i + 1 : i)] = d_incomingLocalf[i + 1];
    }
  }
}
