#pragma once

#include "graphics/texture.h"
#include "graphics/vertexArray.h"
#include "graphics/buffer.h"
#include "graphics/API.h"
#include <array>
#include <geometry/mesh.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "FL/Fl_Simple_Terminal.H"

namespace FLB 
{

  class ComponentMesh
  {
    public:
      ComponentMesh(FLB::API api);

      const uint32_t getInstanceCount() const {return m_TotalInstanceCount;}

      FLB::VertexArray* getVertexArray() const {return m_VertexArray.get();}
      
      size_t getIndexCount() const {return m_IndexCount;}

    protected:

      std::vector<glm::mat4> calculateTransformIntanceData();

      FLB::API m_Api;
      std::vector<float> m_VertexData;
      std::vector<glm::vec3> m_TranslationData;
      std::vector<glm::vec4> m_ColorData;

      std::unique_ptr<FLB::VertexArray> m_VertexArray;
      std::unique_ptr<FLB::VertexBuffer> m_VertexBuffer;
      std::unique_ptr<FLB::VertexBuffer> m_TransformBuffer;
      std::unique_ptr<FLB::VertexBuffer> m_OffsetBuffer;
      std::unique_ptr<FLB::IndexBuffer> m_IndexBuffer;
      std::unique_ptr<FLB::VertexBuffer> m_ColorBuffer;

      uint32_t m_TotalInstanceCount;
      uint32_t m_AliveInstanceCount;
      size_t m_IndexCount;
  };

  class RectangleMesh: public ComponentMesh
  {
    public:
      RectangleMesh(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh);

      const bool isCreated() const {return s_Created;}
      void addInstance();
      void removeInstance(uint32_t idx);

      const glm::vec2& getCenterCoordinates() const {return m_CenterCoordinates;}

    private:
      void init(Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh);

      glm::vec2 dimensions = {0.0f, 0.0f};
      glm::vec2 m_CenterCoordinates = {0.0f, 0.0f};

      static bool s_Created;

  };

  class Arrow2DMesh: public ComponentMesh
  {
    public:
      Arrow2DMesh(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh);

      void getDistributionPoints();
      void resize(const glm::ivec2& numberPoints);
      
      void setVelocityVertexBuffer(FLB::VertexBuffer* vertexBuffer);

    private:
      void init(Fl_Simple_Terminal* terminal);

      FLB::VertexBuffer* m_VelocityVertexBuffer;

      const float m_XMinDomain, m_XMaxDomain;
      const float m_YMinDomain, m_YMaxDomain;

      float m_xLongitude = 1.0f;
      float m_yLongitude = 1.0f;

      uint32_t m_NumberPointsX = 2;
      uint32_t m_NumberPointsY = 2;
  };
  
  enum TypeIsosurface
  {
    Phi = 0, Velocity // phi only for free surface
  };

  class IsoSurfaceMesh
  {
    public:
      IsoSurfaceMesh(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh, uint32_t width, uint32_t height, std::array<float, 8>& vertices);
      
      FLB::VertexArray* getVertexArray() const {return m_VertexArray.get();}
      FLB::Texture2D* getTexture() const {return m_Texture.get();}
      TypeIsosurface& getTypeIsosurface()  {return m_TypeIsosurface;}
      void resize(const unsigned int width, const unsigned int height);

    private:
      void init(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh, uint32_t width, uint32_t height, std::array<float, 8>& vertices);

      std::unique_ptr<FLB::VertexArray> m_VertexArray;
      std::unique_ptr<FLB::Texture2D> m_Texture;
      std::unique_ptr<FLB::VertexBuffer> m_VertexBufferVertices;
      std::unique_ptr<FLB::VertexBuffer> m_VertexBufferTexture;

      TypeIsosurface m_TypeIsosurface;
  };

}
