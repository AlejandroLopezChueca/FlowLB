#include "cudaIsosurface.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace FLB
{
  /*__constant__ uint8_t d_EdgeTable[16] = 
  {
    0x0,

  };*/

}

__device__ float2 FLB::getInterpolatedPoint(const float2 p0, const float2 p1, const float value0, const float value1, const float isoValue)
{
  const float aux = (isoValue - value0) / (value1 - value0);
  const float aux2 = (1.0f - aux);
  return  {aux2 * p0.x + aux * p1.x, aux2 * p0.y + aux * p1.y};
}

__device__ void FLB::drawLine(float2 p0, float2 p1, float2 pixelsTextPerNode, const float xOffset, const float yOffset, cudaSurfaceObject_t d_SurfaceTexture, const unsigned int height)
{
  float xOffsetp0 = p0.x + xOffset, xOffsetp1 = p1.x + xOffset;
  float yOffsetp0 = p0.y + yOffset, yOffsetp1 = p1.y + yOffset;
  // traform coordinates points to pixels in texture
  int x0 = pixelsTextPerNode.x * xOffsetp0  - 1.0f * ((int)xOffsetp0 != 0) + 0.5f; // max is width - 1 of texture
  const int x1 = pixelsTextPerNode.x * xOffsetp1 - 1.0f * ((int)xOffsetp1 != 0) + 0.5f; // max is width of texture

  // -0.5 = 0.5 (used to round number) - 1.0 (because the firts index is 0, substract to height)
  int y0 = height - pixelsTextPerNode.y * yOffsetp0 - 0.5f; // max is height -1 of texture
  const int y1 = height - pixelsTextPerNode.y * yOffsetp1 - 0.5f; // max is  height of texture
  
  // Bresenham's line algorithm implementation from "Zingl, Alois (2016) A Rasterizing Algorithm for Drawing Curves"

  const int dx = abs(x1 - x0);
  const int dy = -abs(y1 - y0);

  int error = dx + dy; // error value e_xy

  const int xIncrement = 2 * (x0 < x1) - 1;
  const int yIncrement = 2 * (y0 < y1) - 1;
  /*if (x0 <1)
  {
    printf("X0 = %d Y0 =  %d xOffset = %6.2f yOffset = %6.2f \n", x0, y0, xOffset, yOffset);
  }*/

  uint8_t data = 255;
  while (x0 != x1 || y0 != y1)
  {
    surf2Dwrite(data, d_SurfaceTexture, x0 * sizeof(uint8_t), y0); 
    int error2 = 2 * error;
    if (error2 >= dy) // e_xy + e_x > 0 
    {
      error += dy;
      x0 += xIncrement;
    }
    if (error2 <= dx) // e_xy + e_y < 0
    {
      error += dx;
      y0 += yIncrement;
    }
  } 
}

__device__ unsigned int FLB::marchingSquares(const float* values, const float isoValue)
{
  unsigned int squareIndex = 0;
  squareIndex |= (values[0] > isoValue);
  squareIndex |= (values[1] > isoValue) << 1;
  squareIndex |= (values[2] > isoValue) << 2;
  squareIndex |= (values[3] > isoValue) << 3;

  return squareIndex;
};



