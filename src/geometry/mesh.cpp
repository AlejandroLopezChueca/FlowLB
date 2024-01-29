#include "mesh.h"
#include "graphics/API.h"
#include <cmath>
#include <iostream>


FLB::Mesh::Mesh()
  : m_xMax(0), m_xMin(0), m_yMax(0), m_yMin(0), m_zMax(0), m_zMin(0), m_Is3D(false), m_SeparationCDW(0.0f), m_IsFreeSurface(false) {};


void FLB::Mesh::init()
{
  calculateNumberPointsMesh();
  reserveMemory();
  calculateCoordinatesPoints();
  calculateIndicesCorners();
  calculateBoundingBoxCDWs();
}

void FLB::Mesh::calculateCoordinatesPoints()
{
  unsigned int idx; 
  int count = 0;

  for (unsigned int z = 0; z < m_Nz; z++)
  {
    for (unsigned int y = 0; y < m_Ny; y++)
    {
      for (unsigned int x = 0; x < m_Nx; x++)
      {
	//idx = x + (y + z * Ny) * Nx;
	//coordinatesPoints[idx + 2 * count] = x * sizeInterval;
	m_CoordinatesPoints.push_back(m_xMin + x * m_SizeInterval);
	//coordinatesPoints[idx + 2 * count + 1] = y * sizeInterval;
	m_CoordinatesPoints.push_back(m_yMin + y * m_SizeInterval);
	//coordinatesPoints[idx + 2 * count + 2] = z * sizeInterval; 
	m_CoordinatesPoints.push_back(m_zMin + z * m_SizeInterval); 
	//count += 1; 
      }
    }
  }
};

void FLB::Mesh::calculateNumberPointsMesh()
{
  m_Nx = std::floor((m_xMax - m_xMin)/m_SizeInterval) + 1;
  m_Ny = std::floor((m_yMax - m_yMin)/m_SizeInterval) + 1; 
  m_Nz = m_zMax == 0 ? 1 : std::floor((m_zMax - m_zMin)/m_SizeInterval) + 1;
  // It is assumed taht the domain is a rectangular cuboid
  m_NumPointsMesh = m_Nx * m_Ny * m_Nz;
};

void FLB::Mesh::calculateIndicesCorners()
{
  m_IndicesCorners[0] = 0;
  m_IndicesCorners[1] = m_Nx;
  m_IndicesCorners[2] = m_Nx * m_Ny;
  m_IndicesCorners[3] = m_NumPointsMesh; 
}

void FLB::Mesh::reserveMemory()
{
  m_DomainFlags.reserve(m_NumPointsMesh);
  m_CoordinatesPoints.reserve(3 * m_NumPointsMesh);
}

void FLB::Mesh::clear()
{
  m_Obstacles.clear();
  m_CDWs.clear();
  m_Points.clear();
  m_DomainFlags.clear();
  m_CoordinatesPoints.clear();
}

void FLB::Mesh::setupGraphicsOptions(FLB::API api, Fl_Simple_Terminal* terminal)
{
  m_VertexArray = FLB::VertexArray::create(api);
  m_VertexBufferDomain = FLB::VertexBuffer::create(api, terminal, m_CoordinatesPoints.data(), 3 * m_NumPointsMesh * sizeof(float));
  FLB::BufferLayout layout = {
    {ShaderDataType::Float3, "a_PointsPosition"}
  };

  m_VertexBufferDomain -> setLayout(layout);
  m_VertexArray -> addVertexBuffer(m_VertexBufferDomain.get());
}

void FLB::Mesh::calculateBoundingBoxCDWs()
{


  for (const std::unique_ptr<FLB::CDW>&  cdw : m_CDWs) 
  {

  }
}



