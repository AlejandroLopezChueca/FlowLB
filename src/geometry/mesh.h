#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "shapes.h"

namespace FLB
{
  struct Point
  {
    float x, y, z;
  };

  /**
   * @brief Mesh of the the domain
   *
   */
  class Mesh
  {
    public:
      Mesh();
      //~Mesh();
      void buifloat();
      void getCoordinatesPoints();
      void getOpenGLCoordinatesPoints();
      void getNumberPointsMesh();
      void getIndicesCorners();
      void reserveMemory();
      void clear();

      std::vector<FLB::Shape> obstacles;
      std::vector<std::unique_ptr<FLB::CDW>> CDWs;
      std::vector<Point> points;
      std::vector<uint8_t> domainFlags;
      std::vector<float> coordinatesPoints;

      //  Max and min values of the domain
      float xMax, xMin, yMax, yMin, zMax, zMin;

      //size beetween two consecutives points
      float sizeInterval;

      // number of point in each direction
      unsigned int Nx, Ny, Nz;
      size_t numPointsMesh; //Total number of points
      // Domain is 2D or 3D
      bool is3D;

      // The problem is Free surface or it is with only one fluid
      bool isFreeSurface;

      // Initial and end points where the cross drainage works start and end
      unsigned int initPointCDW, endPointCDW;

      // Separation (m) between CDWs. It is the distance between consecutive centers
      float separationCDW;

      // Indices of the points i the corners of the domain
      uint32_t m_indicesCorners[4];

  };
 }
