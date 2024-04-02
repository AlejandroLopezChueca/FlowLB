#pragma once

#include "API.h"
#include "buffer.h"
#include "graphics/shader.h"
#include "graphics/frameBuffer.h"
#include "io/reader.h"
#include "texture.h"
#include "vertexArray.h"
#include "rendererAPI.h"
#include "camera.h"
#include "font.h"
#include "graphics/scene/components.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <FL/Fl_Simple_Terminal.H>

namespace FLB
{
  enum class ScalarVectorialVisualization
  {
    xVelocity = 0,
    yVelocity,
    zVelocity,
    velocityMagnitude
  };

  class Renderer
  {
    public:

      static void init(const FLB::OptionsCalculation& optionsCalc, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh);

      static void resetData();

      static void addInstanceRectangles(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh);
      static void removeInstanceRectangles(uint32_t idx);

      static void beginScene(const FLB::OrthographicCamera& camera);
      static void endScene();
 
      static void setClearColor(const glm::vec4& color) {s_RendererAPI -> setClearColor(color);}

      static void setShaderInUse(uint32_t idx);
      static void setArrowShader();
      static void clear() {s_RendererAPI -> clear();}
      
      static void drawScalarVectorialField(const FLB::ScalarVectorialFieldComponent& scalarVectorialFieldsTexture, size_t numPointsMesh);
      static void drawIsoSurface(const FLB::IsoSurfaceComponent& IsoSurfaceComponent, const std::array<float, 8>& cameraDomainCornersSI, bool updateIsoSurfaceBounds);
      static void drawInstancedRectangles(int idx, bool updateValues, glm::vec4* color, glm::mat4* transform);
      static void drawInstancedArrows(const FLB::Arrow2DComponent& arrow2DComponent, const FLB::TransformComponent& transformComponent);
      static void drawString(const std::string& string);

      static void setRendererAPI(const std::array<uint32_t,4>& indices) {s_RendererAPI = FLB::RendererAPI::create(indices);}
      static void createQuad(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh);

      static uint32_t getInstanceCountRectangles();

      static FLB::VertexBuffer* getVertexBufferQuad();
      static FLB::Texture2D* getScalarFieldsTexture();
      static FLB::RectangleMesh* getRectangleMesh();

      static bool s_UpdateRender;

      // font used to render strings of text
      static std::unique_ptr<FLB::Font> s_Font; 
      static std::unique_ptr<Texture2D> s_FontTexture;
      static std::unique_ptr<FLB::Shader> s_TextShader;
   
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
