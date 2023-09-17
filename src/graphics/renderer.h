#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <FL/Fl_Simple_Terminal.H>

#include "buffer.h"
#include "glm/fwd.hpp"
#include "graphics/shader.h"
#include "graphics/frameBuffer.h"
#include "vertexArray.h"
#include "rendererAPI.h"
#include "camera.h"
namespace FLB
{
  class Renderer
  {
    public:
      static void beginScene();
      static void endScene();
      
      static void setClearColor(const glm::vec4& color) {s_RendererAPI -> setClearColor(color);}
      static void clear() {s_RendererAPI -> clear();}
      static void submit(const FLB::VertexArray& vertexArray, const FLB::Shader& shader, size_t count, const glm::mat4& viewProjectionMatrix);
      static void setRendererAPI(uint32_t indices[4]) {s_RendererAPI = FLB::RendererAPI::create(indices);}
      static void createQuad(FLB::API api, Fl_Simple_Terminal* terminal);
      static void drawQuad();

      static bool s_UpdateRender;
      static bool s_VelocityPointMode;
      static float s_PointSize;
      static bool s_VelocityTextureMode2D;

      static std::unique_ptr<FLB::VertexBuffer> s_VertexBufferQuad; 

      static FLB::FrameBuffer* s_TextureToUse;
      static FLB::FrameBuffer* s_TextureVelocity;

      static FLB::Shader* s_shaderToUse;
      static FLB::Shader* s_shaderPointsVelocity;
      static FLB::Shader* s_shaderTextureVelocity2D;
   
    private:
      static std::unique_ptr<FLB::RendererAPI> s_RendererAPI;
  };

  class Renderer2D
  {
    public:
      static void beginScene(FLB::OrthographicCamera& camera);
      static void endScene();

      static void submit(FLB::VertexArray* const vertexArray, FLB::Shader* const shader);

  };
}
