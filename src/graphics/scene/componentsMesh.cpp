#include "componentsMesh.h"

#include "glm/ext/matrix_transform.hpp"
#include "glm/fwd.hpp"
#include "glm/gtx/quaternion.hpp"
#include <cstdint>
#include <vector>
#include <iostream>

//#include <iostream>
#include <glm/gtx/io.hpp>
//////////////////////// ComponentMesh ////////////////////////

FLB::ComponentMesh::ComponentMesh(FLB::API api)
  : m_Api(api) {}

std::vector<glm::mat4> FLB::ComponentMesh::calculateTransformIntanceData()
{
  glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
  glm::vec3 scale = {1.0f, 1.0f, 1.0f};
  glm::mat4 identity = glm::mat4(1.0f);

  std::vector<glm::mat4> transformData;

  for (int i = 0; i < m_TotalInstanceCount; i++)
  {
    glm::mat4 rotationMatrix = glm::toMat4(glm::quat(rotation));
    transformData.push_back(glm::translate(identity, m_TranslationData[i]) * rotationMatrix * glm::scale(identity, scale));
  }

  return transformData;
}
//////////////////////// RenctangleMesh ////////////////////////

bool FLB::RectangleMesh::s_Created = false;

FLB::RectangleMesh::RectangleMesh(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh)
  : ComponentMesh(api)
{
  init(terminal, mesh);
}

void FLB::RectangleMesh::init(Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh)
{
  m_CenterCoordinates.x = 0.5f * (mesh -> getxMax() - mesh ->getxMin());
  m_CenterCoordinates.y = 0.5f * (mesh -> getyMax() - mesh ->getyMin());

  m_VertexArray = FLB::VertexArray::create(m_Api);
  
  // create vertices to use
  std::vector<float> vertices =
  {
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
     0.5f, 0.5f, 0.0f,
    -0.5, 0.5f, 0.0f,
  };
  
  m_VertexBuffer = FLB::VertexBuffer::create(m_Api, terminal, vertices.data(), sizeof(float) * vertices.size());

  FLB::BufferLayout layout;
  layout = {{FLB::ShaderDataType::Float3, "a_Pos"}};
  m_VertexBuffer -> setLayout(layout);
  m_VertexArray -> addVertexBuffer(m_VertexBuffer.get());

  m_ColorData.push_back({1.0f, 0.0f, 0.0f, 1.0f});
  m_ColorBuffer = FLB::VertexBuffer::create(m_Api, terminal, m_ColorData.data(), sizeof(glm::vec4) * m_ColorData.size());
  layout = {{FLB::ShaderDataType::Float4, "a_Color", true}};
  m_ColorBuffer -> setLayout(layout);
  m_VertexArray -> addVertexBuffer(m_ColorBuffer.get());

  m_TranslationData.push_back({m_CenterCoordinates.x, m_CenterCoordinates.y, 0.0f});
  m_TotalInstanceCount = 1;
  std::vector<glm::mat4> transformData = calculateTransformIntanceData();
  m_TransformBuffer = FLB::VertexBuffer::create(m_Api, terminal, transformData.data(), sizeof(glm::mat4) * transformData.size());
  layout = {{FLB::ShaderDataType::Mat4, "a_Transform", true}};
  m_TransformBuffer -> setLayout(layout);
  m_VertexArray -> addVertexBuffer(m_TransformBuffer.get());
  
  s_Created = true;
}

void FLB::RectangleMesh::addInstance()
{ 
  m_ColorData.push_back({1.0f, 0.0f, 0.0f, 1.0f});
  m_ColorBuffer -> resize(m_ColorData.data(), sizeof(glm::vec4) * m_ColorData.size());

  m_TranslationData.push_back({m_CenterCoordinates.x, m_CenterCoordinates.y, 0.0f});
  m_TotalInstanceCount += 1;
  std::vector<glm::mat4> transformData = calculateTransformIntanceData();
  m_TransformBuffer -> resize(transformData.data(), sizeof(glm::mat4) * transformData.size());

  // Correct vertex array, because the buffer are recreated
  m_VertexArray -> recreate();
  m_VertexArray -> addVertexBuffer(m_VertexBuffer.get());
  m_VertexArray -> addVertexBuffer(m_ColorBuffer.get());
  m_VertexArray -> addVertexBuffer(m_TransformBuffer.get());
}

void FLB::RectangleMesh::removeInstance(uint32_t idx)
{
  m_ColorData.erase(m_ColorData.begin() + idx, m_ColorData.begin() + idx + 1);
  m_ColorBuffer -> resize(m_ColorData.data(), sizeof(glm::vec4) * m_ColorData.size());

  m_TranslationData.erase(m_TranslationData.begin() + idx, m_TranslationData.begin() + idx + 1);
  m_TotalInstanceCount -= 1;
  std::vector<glm::mat4> transformData = calculateTransformIntanceData();
  m_TransformBuffer -> resize(transformData.data(), sizeof(glm::mat4) * transformData.size()); 
  
  // Correct vertex array, because the buffer are recreated
  m_VertexArray -> recreate();
  m_VertexArray -> addVertexBuffer(m_VertexBuffer.get());
  m_VertexArray -> addVertexBuffer(m_ColorBuffer.get());
  m_VertexArray -> addVertexBuffer(m_TransformBuffer.get());
}

//////////////////////// Arrow2DMesh ////////////////////////

FLB::Arrow2DMesh::Arrow2DMesh(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh)
  : ComponentMesh(api), m_XMaxDomain(mesh -> getxMax()), m_XMinDomain(mesh -> getxMin()), m_YMaxDomain(mesh -> getyMax()), m_YMinDomain(mesh -> getyMin())
{
  init(terminal);
}

void FLB::Arrow2DMesh::getDistributionPoints()
{
  m_TranslationData.clear();
  m_TotalInstanceCount = m_NumberPointsX * m_NumberPointsY;

  float xSeparation = m_xLongitude/((float)m_NumberPointsX - 1.0f);
  float ySeparation = m_yLongitude/((float)m_NumberPointsY - 1.0f);

  for (int i = 0; i < m_NumberPointsY; i++)
  {
    for (int j = 0; j < m_NumberPointsX; j++)
    {
      float xPoint = -0.5f + j * xSeparation;
      float yPoint = -0.5f + i * ySeparation;
      //if (xPoint < m_XMinDomain || xPoint > m_XMaxDomain || yPoint < m_YMinDomain || yPoint > m_YMaxDomain) continue;
      m_TranslationData.push_back({xPoint, yPoint, 0.0f});
      //m_TotalInstanceCount += 1;

    }
  }
}

void FLB::Arrow2DMesh::resize(const glm::ivec2& numberPoints)
{
  m_NumberPointsX = numberPoints.x;
  m_NumberPointsY = numberPoints.y;
  // recreate transform buffer
  getDistributionPoints();
  m_OffsetBuffer -> resize(m_TranslationData.data(), sizeof(glm::vec3) * m_TranslationData.size());

  // Correct vertex array, because the offset buffer are recreated
  m_VertexArray -> recreate();
  m_VertexArray -> addVertexBuffer(m_VertexBuffer.get());
  m_VertexArray -> addVertexBuffer(m_OffsetBuffer.get());
  m_VertexArray -> setIndexBuffer(m_IndexBuffer.get());
}

void FLB::Arrow2DMesh::setVelocityVertexBuffer(FLB::VertexBuffer* vertexBuffer)
{
  m_VelocityVertexBuffer = vertexBuffer;
  m_VertexArray ->addVertexBuffer(m_VelocityVertexBuffer);
}

void FLB::Arrow2DMesh::init(Fl_Simple_Terminal* terminal)
{
  m_VertexArray = FLB::VertexArray::create(m_Api);

  std::vector<float> vertices = 
  {
    0.000f, 0.048f, 0.0f,
    0.847f, 0.048f, 0.0f,
    0.761f, 0.132f, 0.0f,
    0.761f, 0.237f, 0.0f,
    1.000f, 0.000f, 0.0f,
    0.000f, -0.048f, 0.0f,
    0.761f, -0.237f, 0.0f,
    0.761f, -0.132f, 0.0f,
    0.847f, -0.048f, 0.0f,
  };
  
  std::vector<uint32_t> indices = 
  {
    0, 8, 1,
    1, 8, 4,
    8, 6, 4,
    1, 3, 2,
    0, 5, 8,
    8, 7, 6,
    1, 4, 3
  };

  m_IndexCount = indices.size();
 
  // vertex buffer
  m_VertexBuffer = FLB::VertexBuffer::create(m_Api, terminal, vertices.data(), sizeof(float) * vertices.size());

  FLB::BufferLayout layout;
  layout = {{FLB::ShaderDataType::Float3, "a_Pos"}};
  m_VertexBuffer -> setLayout(layout);
  m_VertexArray -> addVertexBuffer(m_VertexBuffer.get());
 
  // Offset buffer
  getDistributionPoints();
  m_OffsetBuffer = FLB::VertexBuffer::create(m_Api, terminal, m_TranslationData.data(), sizeof(glm::vec3) * m_TranslationData.size());
  layout = {{FLB::ShaderDataType::Float3, "a_Offset", true}};
  m_OffsetBuffer -> setLayout(layout);
  m_VertexArray -> addVertexBuffer(m_OffsetBuffer.get());
  
  // index buffer
  m_IndexBuffer = FLB::IndexBuffer::create(m_Api, terminal, indices.data(), indices.size());
  m_VertexArray -> setIndexBuffer(m_IndexBuffer.get());
}
