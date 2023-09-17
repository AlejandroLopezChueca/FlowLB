#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <glm/glm.hpp>

#include "API.h"
#include "vertexArray.h"

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

      virtual void drawElements(const FLB::VertexArray& vertexArray, size_t count) = 0;
      virtual void drawPoints(const FLB::VertexArray& vertexArray, size_t count) = 0;

      static API getAPI() {return s_API;}
      static void setAPI(API api) {s_API = api;}
      static std::unique_ptr<RendererAPI> create(uint32_t indices[4] = {});

    private:
      static API s_API;
  };

}
