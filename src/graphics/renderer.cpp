
#include "renderer.h"
#include "buffer.h"
#include "font.h"
#include "frameBuffer.h"
#include "geometry/mesh.h"
#include "io/reader.h"
#include "rendererAPI.h"
#include "shader.h"
#include "texture.h"
#include "colorMaps.h"
#include "vertexArray.h"
#include "graphics/scene/componentsMesh.h"

#include "glm/fwd.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <iostream>
#include <glad/glad.h>


//#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#include <glm/gtx/io.hpp>

namespace FLB 
{
  struct RendererData
  {
    std::unique_ptr<FLB::VertexArray> vertexArrayQuad;
    std::unique_ptr<FLB::VertexBuffer> vertexBufferQuad;
     
    std::unique_ptr<FLB::Shader> pointsXVelocityShader;
    std::unique_ptr<FLB::Shader> pointsYVelocityShader;
    std::unique_ptr<FLB::Shader> pointsMagVelocityShader;
    std::unique_ptr<FLB::Shader> textureXVelocity2DShader;
    std::unique_ptr<FLB::Shader> textureYVelocity2DShader;
    std::unique_ptr<FLB::Shader> textureMagVelocity2DShader;
    
    std::unique_ptr<FLB::Shader> instancedArrowShader;
    std::unique_ptr<FLB::Shader> instancedLinesShader;

    std::unique_ptr<FLB::Font> font;
    std::unique_ptr<FLB::Texture2D> fontTexture;
    std::unique_ptr<FLB::Texture2D> scalarVectorialFieldsTexture;

    std::array<std::unique_ptr<FLB::Texture1D>, 4> colorMaps;

    FLB::Shader* shaderScalarVectorialFieldInUse;

    std::unique_ptr<FLB::RectangleMesh> rectangleMesh;

    // Uniform buffer of the view x projection matrix of the camera
    std::unique_ptr<FLB::UniformBuffer> viewProjectionUniformBuffer;

    glm::vec4 maxMinCoordDomain;
  };

  static RendererData s_RendererData;

}

//////////////////////////// Renderer ////////////////////////////

//Null initialization of renderer api
std::unique_ptr<FLB::RendererAPI> FLB::Renderer::s_RendererAPI = FLB::RendererAPI::create();

bool FLB::Renderer::s_UpdateRender = false;

std::unique_ptr<FLB::Font> FLB::Renderer::s_Font = nullptr;
std::unique_ptr<FLB::Texture2D> FLB::Renderer::s_FontTexture = nullptr;


void FLB::Renderer::init(const FLB::OptionsCalculation& optionsCalc,  Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh)
{
  // Shaders to use
  s_RendererData.pointsXVelocityShader = FLB::Shader::create("assets/shaders/pointsXVelocity.glsl", optionsCalc.graphicsAPI, terminal);
  s_RendererData.pointsYVelocityShader = FLB::Shader::create("assets/shaders/pointsYVelocity.glsl", optionsCalc.graphicsAPI, terminal);
  s_RendererData.pointsMagVelocityShader = FLB::Shader::create("assets/shaders/pointsMagnitudeVelocity.glsl", optionsCalc.graphicsAPI, terminal);
  
  s_RendererData.textureXVelocity2DShader = FLB::Shader::create("assets/shaders/textureXVelocity2D.glsl", optionsCalc.graphicsAPI, terminal);
  s_RendererData.textureYVelocity2DShader = FLB::Shader::create("assets/shaders/textureYVelocity2D.glsl", optionsCalc.graphicsAPI, terminal);
  s_RendererData.textureMagVelocity2DShader = FLB::Shader::create("assets/shaders/textureMagnitudeVelocity2D.glsl", optionsCalc.graphicsAPI, terminal);

  s_RendererData.instancedArrowShader = FLB::Shader::create("assets/shaders/arrowInstancing.glsl", optionsCalc.graphicsAPI, terminal);
  
  s_RendererData.instancedLinesShader = FLB::Shader::create("assets/shaders/lineInstancing.glsl", optionsCalc.graphicsAPI, terminal);

  // create the colormaps
  for (int i = 0; i < FLB::s_RendererData.colorMaps.size(); i++) FLB::s_RendererData.colorMaps[i] = FLB::Texture1D::create(optionsCalc.graphicsAPI, FLB::ColorMaps::colorMaps[i]);

  // size of the ViewProjection matrix (glm::mat4)
  size_t size = 4 * 4 * sizeof(float);
  s_RendererData.viewProjectionUniformBuffer = FLB::UniformBuffer::create(optionsCalc.graphicsAPI, terminal, size, 0);
  s_RendererData.viewProjectionUniformBuffer -> setData(0); //default

  // Default options
  s_RendererData.shaderScalarVectorialFieldInUse = s_RendererData.pointsMagVelocityShader.get();

  
  if (optionsCalc.typeAnalysis == 0) 
  {
    createQuad(optionsCalc.graphicsAPI, terminal, mesh);
    s_RendererData.scalarVectorialFieldsTexture = FLB::Texture2D::create(optionsCalc.graphicsAPI, mesh -> getNx() , mesh -> getNy(), FLB::ImageFormat::RG32F);
  }

  s_RendererData.maxMinCoordDomain = {mesh -> getxMax(), mesh -> getyMax(), mesh -> getxMin(), mesh -> getyMin()};
}

void FLB::Renderer::addInstanceRectangles(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh)
{
  if (s_RendererData.rectangleMesh -> isCreated()) s_RendererData.rectangleMesh -> addInstance();
  else s_RendererData.rectangleMesh = std::make_unique<FLB::RectangleMesh>(api, terminal, mesh);
}

void FLB::Renderer::removeInstanceRectangles(uint32_t idx)
{
  s_RendererData.rectangleMesh -> removeInstance(idx);
}

void FLB::Renderer::beginScene(const FLB::OrthographicCamera& camera)
{
  s_RendererData.viewProjectionUniformBuffer -> setData((const void*)&camera.getViewProjectionMatrix());
}

void FLB::Renderer::endScene()
{
  //s_RendererData.frameBufferInUse -> unbind();
  //setClearColor({0.5f, 0.5f, 0.5f, 1});
  //clear();

}

void FLB::Renderer::setShaderInUse(uint32_t idx)
{
  switch (idx) 
  {
    case 0:
    {
      s_RendererData.shaderScalarVectorialFieldInUse = s_RendererData.pointsMagVelocityShader.get();
      break;
    }
    case 1:
    {
      s_RendererData.shaderScalarVectorialFieldInUse = s_RendererData.pointsXVelocityShader.get();
      break;
    }
    case 2:
    {
      s_RendererData.shaderScalarVectorialFieldInUse = s_RendererData.pointsYVelocityShader.get();
      break;
    }
    case 10:
    {
      s_RendererData.shaderScalarVectorialFieldInUse = s_RendererData.textureMagVelocity2DShader.get(); 
      break;
    }
    case 11:
    {
      s_RendererData.shaderScalarVectorialFieldInUse = s_RendererData.textureXVelocity2DShader.get(); 
      break;
    }
    case 12:
    {
      s_RendererData.shaderScalarVectorialFieldInUse = s_RendererData.textureYVelocity2DShader.get(); 
      break;
    }
  }
}

void FLB::Renderer::setArrowShader()
{
  s_RendererData.instancedArrowShader -> bind();
  s_RendererData.instancedArrowShader -> setFloat4("u_MaxMinCoordDomain", s_RendererData.maxMinCoordDomain);
  s_RendererData.scalarVectorialFieldsTexture -> bind(1);
}

void FLB::Renderer::drawScalarVectorialField(const FLB::ScalarVectorialFieldComponent& scalarVectorialFieldComponent, size_t numPointsMesh)
{
  s_RendererData.colorMaps[scalarVectorialFieldComponent.currentIdxColorMap] -> bind(0);
  if (scalarVectorialFieldComponent.type == 0)
  {
    s_RendererData.shaderScalarVectorialFieldInUse -> bind();
    s_RendererData.shaderScalarVectorialFieldInUse -> setFloat2("u_MaxMinValues", scalarVectorialFieldComponent.maxMinValues);
    scalarVectorialFieldComponent.meshVertexArray -> bind();
    s_RendererAPI -> drawPoints(numPointsMesh, scalarVectorialFieldComponent.sizePoints);
  }
  else 
  {
    s_RendererData.shaderScalarVectorialFieldInUse -> bind();
    s_RendererData.shaderScalarVectorialFieldInUse -> setFloat2("u_MaxMinValues", scalarVectorialFieldComponent.maxMinValues);

    s_RendererData.scalarVectorialFieldsTexture -> bind(1);
    s_RendererData.vertexArrayQuad -> bind();
    s_RendererAPI -> drawElements();
  }
}

void FLB::Renderer::drawInstancedRectangles(int idx, bool updateValues, glm::vec4* color, glm::mat4* transform)
{
  s_RendererData.instancedLinesShader -> bind();

  FLB::VertexArray* vertexArray = s_RendererData.rectangleMesh -> getVertexArray();
  vertexArray -> bind();

  if (updateValues)
  {
    // update color
    uint32_t size = sizeof(glm::vec4);
    vertexArray -> updateMemberBufferData(1, size * idx, size, (const void*)color);

    // update transform
    size = sizeof(glm::mat4);
    vertexArray -> updateMemberBufferData(2, size * idx, size, (const void*)transform);
  }
 
  s_RendererAPI -> drawInstancedLines(4, s_RendererData.rectangleMesh -> getInstanceCount());
}

void FLB::Renderer::drawInstancedArrows(const FLB::Arrow2DComponent& arrow2DComponent, const FLB::TransformComponent& transformComponent)
{
  s_RendererData.colorMaps[arrow2DComponent.currentIdxColorMap] -> bind(0);
  if (arrow2DComponent.updateMesh) arrow2DComponent.mesh -> resize(arrow2DComponent.numberPoints);

  arrow2DComponent.vertexArray -> bind();
  glm::mat4 transform = transformComponent.getTransform();

  s_RendererData.instancedArrowShader -> setMat4("u_RecTransform", transform);
  s_RendererData.instancedArrowShader -> setFloat("u_Scale", arrow2DComponent.scale);

  s_RendererAPI -> drawInstancedElements(arrow2DComponent.mesh -> getIndexCount(), arrow2DComponent.mesh -> getInstanceCount());

}

void FLB::Renderer::drawString(const std::string& string)
{
  const auto& metrics = s_Font -> getFontGeometry() -> getMetrics();

  // font advance
  float x =0.0f;
  float fsScale = 1.0/(metrics.ascenderY - metrics.descenderY);
  float y = 0.0f;

  //In this way we cna treat every character as as char32_t
}

void FLB::Renderer::createQuad(FLB::API api, Fl_Simple_Terminal* terminal, const FLB::Mesh* mesh)
{
  float xMin = mesh -> getxMin();
  float xMax = mesh -> getxMax();
  float yMin = mesh -> getyMin();
  float yMax = mesh -> getyMax();
  float vertices[] =
  {
    xMin, yMin, 0.0f, 0.0f, 0.0f,
    xMax, yMin, 0.0f, 1.0f, 0.0f,
    xMax, yMax, 0.0f, 1.0f, 1.0f,
    xMin, yMax, 0.0f, 0.0f, 1.0f
  };
  
  s_RendererData.vertexArrayQuad = FLB::VertexArray::create(api);

  s_RendererData.vertexBufferQuad = FLB::VertexBuffer::create(api, terminal, vertices, sizeof(vertices));
  FLB::BufferLayout layout = 
  {
    {ShaderDataType::Float3, "a_QuadPoints"},
    {ShaderDataType::Float2, "a_QuadTextCoord"}
  };
  s_RendererData.vertexBufferQuad -> setLayout(layout);
  s_RendererData.vertexArrayQuad -> addVertexBuffer(s_RendererData.vertexBufferQuad.get());
}

uint32_t FLB::Renderer::getInstanceCountRectangles()
{
  return s_RendererData.rectangleMesh -> getInstanceCount();
}

FLB::VertexBuffer* FLB::Renderer::getVertexBufferQuad()
{
  return s_RendererData.vertexBufferQuad.get();
}

FLB::Texture2D* FLB::Renderer::getScalarFieldsTexture()
{
  return s_RendererData.scalarVectorialFieldsTexture.get();
}
FLB::RectangleMesh* FLB::Renderer::getRectangleMesh()
{
  return s_RendererData.rectangleMesh.get();
}
