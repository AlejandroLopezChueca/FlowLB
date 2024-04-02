#pragma once

#include <cstdint>

#include "cudaInitData.cuh"
#include "ui/renderLayer.h"

namespace FLB 
{
  /**
    * @brief Interpolate point in line using a IsoValue.
    *
    */
  __device__ float2 getInterpolatedPoint(const float2 p0, const float2 p1, const float value0, const float value1, const float IsoValue);

  /**
    *
    *
    *
    */
  __device__ void drawLine(float2 p0, float2 p1, float2 pixelsTextPerNode, const float xOffset, const float yOffset, cudaSurfaceObject_t d_SurfaceTexture, const unsigned int height);

  /**
    * @brief
    *
    * @param[in]
    * @return
    *
    */

  __device__ unsigned int marchingSquares(const float* values, const float isoValue);

  template <typename PRECISION>
  __global__ void d_CreateIsoSurfaceVector2D(const PRECISION* d_field, cudaSurfaceObject_t d_SurfaceTexture, const unsigned int width, const unsigned int height, const float isoValue, const unsigned int xMinCameraBounds, const unsigned int xMaxCameraBounds, const unsigned int yMinCameraBounds, const unsigned int yMaxCameraBounds)
  {
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y *blockDim.y;
    if (x >= (d_Nx - 1) || y >= (d_Ny - 1)) return; // Prevent threads outside domain and in the edges of the domain there are not squares
  
    if (x < xMinCameraBounds || x >= xMaxCameraBounds || y < yMinCameraBounds || y >= yMaxCameraBounds) return;

    // (FLB::d_Ny - y - 1) -> used to flip y coordinate because CUDA has the origin in the upper left corner and OpenGL in the lower left corner
    const unsigned int idx = x + (FLB::d_Ny - y - 1) * FLB::d_Nx;

    /* scheme vertices
       0 x 1
       x   x
       3 x 2
    */
    float verticesValues[4];
    verticesValues[0] = hypotf(d_field[idx], d_field[idx + FLB::d_N]);
    verticesValues[1] = hypotf(d_field[idx + 1], d_field[idx + FLB::d_N + 1]);
    // substract FLB::d_Ny because the y origin of cuda is different from OpenGL 
    verticesValues[2] = hypotf(d_field[idx - FLB::d_Nx + 1], d_field[idx - FLB::d_Nx + 1 + FLB::d_N]);
    verticesValues[3] = hypotf(d_field[idx - FLB::d_Nx], d_field[idx - FLB::d_Nx + FLB::d_N]);
    
    unsigned int squareIndex = FLB::marchingSquares(verticesValues, isoValue);
  
    float2 pointsSquare[4];
    pointsSquare[0] = {0.0f, 0.0f};
    pointsSquare[1] = {1.0f, 0.0f};
    pointsSquare[2] = {1.0f, 1.0f};
    pointsSquare[3] = {0.0f, 1.0f};

    const float2 pixelsTextPerNode = {(float)width / (xMaxCameraBounds - xMinCameraBounds), (float)height / (yMaxCameraBounds - yMinCameraBounds)};

    const float xOffset = x - xMinCameraBounds;
    const float yOffset = y - yMinCameraBounds;
    
    //printf("x = %u y = %u xMIN = %u YMIN = %u xOffset = %6.2f yOffset = %6.2f pixelsTextPerNode.x = %6.4f pixelsTextPerNode.y = %6.4f\n", x, y, xMinCameraBounds, yMinCameraBounds, xOffset, yOffset,pixelsTextPerNode.x, pixelsTextPerNode.y);

    //if (x == 0) printf("y = %d INDEX = %d\n", y, squareIndex);
    
    switch (squareIndex) {
      /*case 0: // the square is empty
      case 15: // the square is full
      {
	return;
	break;
      }*/
      /* scheme case
	 x  v0 x
	 v1    x
	 x  x  x
      */
      case 1:
      case 14:
      {
	const float2 vertice0 = getInterpolatedPoint(pointsSquare[0], pointsSquare[1], verticesValues[0], verticesValues[1], isoValue);
	const float2 vertice1 = getInterpolatedPoint(pointsSquare[0], pointsSquare[3], verticesValues[0], verticesValues[3], isoValue);
	drawLine(vertice0, vertice1, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	break;
      }

      /* scheme case
	 x  v0 x
	 x     v1
	 x  x  x
      */
      case 2:
      case 13:
      {
	const float2 vertice0 = getInterpolatedPoint(pointsSquare[0], pointsSquare[1], verticesValues[0], verticesValues[1], isoValue);
	const float2 vertice1 = getInterpolatedPoint(pointsSquare[1], pointsSquare[2], verticesValues[1], verticesValues[2], isoValue);
	drawLine(vertice0, vertice1, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	break;
      }
      
      
      /* scheme case
	 x  x  x
	 v0    v1
	 x  x  x
      */
      case 3:
      case 12:
      {
	const float2 vertice0 = getInterpolatedPoint(pointsSquare[0], pointsSquare[3], verticesValues[0], verticesValues[3], isoValue);
	const float2 vertice1 = getInterpolatedPoint(pointsSquare[1], pointsSquare[2], verticesValues[1], verticesValues[2], isoValue);
	drawLine(vertice0, vertice1, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	break;
      }
      
      /* scheme case
	 x  x   x 
	 x      v0
	 x  v1  x
      */
      case 4:
      case 11:
      {
	const float2 vertice0 = getInterpolatedPoint(pointsSquare[1], pointsSquare[2], verticesValues[1], verticesValues[2], isoValue);
	const float2 vertice1 = getInterpolatedPoint(pointsSquare[3], pointsSquare[2], verticesValues[3], verticesValues[2], isoValue);
	drawLine(vertice0, vertice1, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	break;
      }

      /* scheme case
	 x  v0  x 
	 v1     v3
	 x  v2  x
      */
      case 5:
      {
	const float2 vertice0 = getInterpolatedPoint(pointsSquare[0], pointsSquare[1], verticesValues[0], verticesValues[1], isoValue);
	const float2 vertice1 = getInterpolatedPoint(pointsSquare[0], pointsSquare[3], verticesValues[0], verticesValues[3], isoValue);
	const float2 vertice2 = getInterpolatedPoint(pointsSquare[3], pointsSquare[2], verticesValues[3], verticesValues[2], isoValue);
	const float2 vertice3 = getInterpolatedPoint(pointsSquare[1], pointsSquare[2], verticesValues[1], verticesValues[2], isoValue);
	drawLine(vertice0, vertice1, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	drawLine(vertice2, vertice3, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	break;
      }

      /* scheme case
	 x  v0  x 
	 x      x
	 x  v1  x
      */
      case 6:
      case 9:
      {
	const float2 vertice0 = getInterpolatedPoint(pointsSquare[0], pointsSquare[1], verticesValues[0], verticesValues[1], isoValue);
	const float2 vertice1 = getInterpolatedPoint(pointsSquare[3], pointsSquare[2], verticesValues[3], verticesValues[2], isoValue);
	drawLine(vertice0, vertice1, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	break;
      }

      /* scheme case
	 x  x   x 
	 v0     x
	 x  v1  x
      */
      case 7:
      case 8:
      {
	const float2 vertice0 = getInterpolatedPoint(pointsSquare[0], pointsSquare[3], verticesValues[0], verticesValues[3], isoValue);
	const float2 vertice1 = getInterpolatedPoint(pointsSquare[3], pointsSquare[2], verticesValues[3], verticesValues[2], isoValue);
	drawLine(vertice0, vertice1, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	break;
      }

      /* scheme case
	 x  v2  x 
	 v0     v3
	 x  v1  x
      */
      case 10:
      {
	const float2 vertice0 = getInterpolatedPoint(pointsSquare[0], pointsSquare[3], verticesValues[0], verticesValues[3], isoValue);
	const float2 vertice1 = getInterpolatedPoint(pointsSquare[3], pointsSquare[2], verticesValues[3], verticesValues[2], isoValue);
	const float2 vertice2 = getInterpolatedPoint(pointsSquare[0], pointsSquare[1], verticesValues[0], verticesValues[1], isoValue);
	const float2 vertice3 = getInterpolatedPoint(pointsSquare[1], pointsSquare[2], verticesValues[1], verticesValues[2], isoValue);
	drawLine(vertice0, vertice1, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	drawLine(vertice2, vertice3, pixelsTextPerNode, xOffset, yOffset, d_SurfaceTexture, height);
	break;
      } 
    }
  } 
}
