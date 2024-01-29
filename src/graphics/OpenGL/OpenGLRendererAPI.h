#pragma once

#include "graphics/rendererAPI.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace FLB
{

  class OpenGLRendererAPI: public RendererAPI
  {
    public:
      OpenGLRendererAPI(const std::array<uint32_t, 4> indices);

      void setClearColor(const glm::vec4& color) override;
      void clear() override;

      void drawElements() const override;
      void drawInstancedElements(size_t indexCount, uint32_t instanceCount) const override;
      void drawInstancedLines(size_t vertexCount, uint32_t instanceCount) const override;
      void drawPoints(size_t count, float pointSize) const override;

    private:
      std::array<uint32_t, 4> m_Indices;
  };

  class OpenGLRendererAPI2D
  {

  };

}
