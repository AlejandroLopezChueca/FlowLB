#pragma once

#include "graphics/rendererAPI.h"
#include <cstddef>
#include <cstdint>

namespace FLB
{

  class OpenGLRendererAPI: public RendererAPI
  {
    public:
      OpenGLRendererAPI(uint32_t indices[4]);

      void setClearColor(const glm::vec4& color) override;
      void clear() override;

      void drawElements(const FLB::VertexArray& vertexArray, size_t count) override;
      void drawPoints(const FLB::VertexArray& vertexArray, size_t count) override;

    private:
      uint32_t m_Indices[4];
  };

  class OpenGLRendererAPI2D
  {

  };

}
