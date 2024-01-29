#pragma once

#include "graphics/API.h"
#include "shapes.h"
#include "graphics/vertexArray.h"
#include "graphics/buffer.h"

#include "FL/Fl_Simple_Terminal.H"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>


namespace FLB
{
  struct Point
  {
    double x, y, z;
  };

  struct BBox
  {
    double m_xMaxCDW = -1.0, m_xMinCDW = -1.0; 
    double m_yMaxCDW = -1.0, m_yMinCDW = -1.0;
    double m_zMaxCDW = -1.0, m_zMinCDW = -1.0;
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
      void addVertexBuffer(FLB::VertexBuffer* const vertexBuffer) {m_VertexArray ->addVertexBuffer(vertexBuffer);}
      void init();
      void calculateCoordinatesPoints();
      void calculateNumberPointsMesh();
      void calculateIndicesCorners();
      void reserveMemory();
      void clear();

      size_t getNumberPointsMesh() const {return m_NumPointsMesh;}
      std::vector<std::unique_ptr<FLB::Shape>>& getObstacles() {return m_Obstacles;}
      const std::vector<uint8_t>& getDomainFlags() const {return m_DomainFlags;}
      std::vector<uint8_t>& getDomainFlags() {return m_DomainFlags;}
      std::vector<std::unique_ptr<FLB::CDW>>& getCDWs() {return m_CDWs;}
      std::vector<Point>& getPoints() {return m_Points;}
      const std::vector<float>& getCoordinatesPoints() const {return m_CoordinatesPoints;}
      double& getSizeInterval() {return m_SizeInterval;}
      uint32_t getNx() const {return m_Nx;}
      uint32_t getNy() const {return m_Ny;}
      uint32_t getNz() const {return m_Nz;}
      double getxMax() const {return m_xMax;}
      double& getxMax() {return m_xMax;}
      double getxMin() const {return m_xMin;}
      double& getxMin() {return m_xMin;}
      double getyMax() const {return m_yMax;}
      double& getyMax() {return m_yMax;}
      double getyMin() const {return m_yMin;}
      double& getyMin() {return m_yMin;}
      double getzMax() const {return m_zMax;}
      double& getzMax() {return m_zMax;}
      double getzMin() const {return m_zMin;}
      double& getzMin() {return m_zMin;}
      uint32_t getIdxInitPointCDW() const {return m_IdxInitPointCDW;}
      uint32_t& getIdxInitPointCDW() {return m_IdxInitPointCDW;}
      uint32_t getIdxEndPointCDW() const {return m_IdxEndPointCDW;}
      uint32_t& getIdxEndPointCDW() {return m_IdxEndPointCDW;}
      const std::array<uint32_t, 4>& getIndicesCorners() const {return m_IndicesCorners;}
      FLB::VertexArray* getVertexArray() const {return m_VertexArray.get();}
      double getSeparationCDW() const {return m_SeparationCDW;}
      const BBox& getBoundingBoxCDWs() const {return m_BBoxCDWs;}

      bool& getIs3D() {return m_Is3D;}
      bool is3D() const {return m_Is3D;}
      bool isFreeSurface() const {return m_IsFreeSurface;}

      void setupGraphicsOptions(FLB::API api, Fl_Simple_Terminal* terminal);

    private:

      void calculateBoundingBoxCDWs();

      std::unique_ptr<FLB::VertexArray> m_VertexArray;
      std::unique_ptr<FLB::VertexBuffer> m_VertexBufferDomain;
      
      std::vector<std::unique_ptr<FLB::Shape>> m_Obstacles;
      std::vector<std::unique_ptr<FLB::CDW>> m_CDWs;
      std::vector<Point> m_Points; //points of the CDW
      std::vector<uint8_t> m_DomainFlags;
      std::vector<float> m_CoordinatesPoints;
      
      //size beetween two consecutives points
      double m_SizeInterval;

      // number of point in each direction
      unsigned int m_Nx, m_Ny, m_Nz;
      size_t m_NumPointsMesh; //Total number of points
      // Domain is 2D or 3D
      bool m_Is3D;
      
      // The problem is Free surface or it is with only one fluid
      bool m_IsFreeSurface;
      
      // Indices of the points i the corners of the domain
      std::array<uint32_t, 4> m_IndicesCorners;
      
      // Initial and end points where the cross drainage works start and end
      unsigned int m_IdxInitPointCDW, m_IdxEndPointCDW;
      
      // Separation (m) between CDWs. It is the distance between consecutive centers
      double m_SeparationCDW;
      

      //  Max and min values of the domain
      double m_xMax, m_xMin, m_yMax, m_yMin, m_zMax, m_zMin;

      BBox m_BBoxCDWs;

  };
 }
