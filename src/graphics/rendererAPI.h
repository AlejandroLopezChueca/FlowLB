#pragma once

#include "API.h"
#include "vertexArray.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <glm/glm.hpp>


namespace FLB
{
  /*enum class API
  {
    NONE = 0, OPENGL = 1
  };*/
  
  class RendererAPI
  {
    public:
      virtual ~RendererAPI() = default;

      virtual void setClearColor(const glm::vec4& color) = 0;
      virtual void clear() = 0;

      virtual void drawElements() const = 0;
      virtual void drawInstancedElements(size_t indexCount, uint32_t instanceCount) const = 0;
      virtual void drawInstancedLines(size_t vertexCount, uint32_t instanceCount) const = 0;
      virtual void drawPoints(size_t count, float pointSize) const = 0;

      static API getAPI() {return s_API;}
      static void setAPI(API api) {s_API = api;}
      static std::unique_ptr<RendererAPI> create(const std::array<uint32_t,4> indices = {});

    private:
      static API s_API;
  };

}
