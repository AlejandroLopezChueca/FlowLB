
#include "renderer.h"
#include "FL/Fl_Simple_Terminal.H"
#include "buffer.h"
#include "frameBuffer.h"
#include "glm/fwd.hpp"
#include "rendererAPI.h"
#include "shader.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <iostream>
#include <glad/glad.h>


//#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020

//////////////////////////// Renderer ////////////////////////////

//Null initialization
std::unique_ptr<FLB::RendererAPI> FLB::Renderer::s_RendererAPI = FLB::RendererAPI::create();

std::unique_ptr<FLB::VertexBuffer> FLB::Renderer::s_VertexBufferQuad = nullptr;

bool FLB::Renderer::s_UpdateRender = false;
bool FLB::Renderer::s_VelocityPointMode = true;
float FLB::Renderer::s_PointSize = 2.0f;
bool FLB::Renderer::s_VelocityTextureMode2D = false;

FLB::Shader* FLB::Renderer::s_shaderToUse = nullptr;
FLB::Shader* FLB::Renderer::s_shaderPointsVelocity = nullptr;
FLB::Shader* FLB::Renderer::s_shaderTextureVelocity2D = nullptr;

FLB::FrameBuffer* FLB::Renderer::s_TextureToUse = nullptr;
FLB::FrameBuffer* FLB::Renderer::s_TextureVelocity = nullptr;

void FLB::Renderer::beginScene()
{

}


void FLB::Renderer::endScene()
{

}

void FLB::Renderer::submit(const FLB::VertexArray& vertexArray, const FLB::Shader& shader, size_t count, const glm::mat4& viewProjectionMatrix)
{
  //if (!FLB::Renderer::s_VelocityPointMode) s_TextureToUse->bind();
  
  s_shaderToUse -> bind();
  s_shaderToUse -> setMat4("u_ViewProjection", viewProjectionMatrix);
  vertexArray.bind();
  if (FLB::Renderer::s_VelocityPointMode) s_RendererAPI -> drawPoints(vertexArray,count);
  else 
  {
    s_TextureToUse -> bindTexture();
    //s_shaderToUse -> setInt("u_ColorMap", 0);
    //s_shaderToUse -> setInt("u_Texture", 0);
    //s_TextureToUse -> unbind();
    //s_shaderToUse -> bind();
    //vertexArray.bind();
    //s_TextureToUse -> bindTexture();
    s_RendererAPI -> drawElements(vertexArray, count);
  }
}

void FLB::Renderer::createQuad(FLB::API api, Fl_Simple_Terminal* terminal)
{    
  float vertices[] =
    {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
	1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
	0.0f, 1.0f, 0.0f, 0.0f, 1.0f

       /*-1.0f, -1.0f,  0.0f,
	1.0f, -1.0f,  0.0f,
	1.0f,  1.0f,  0.0f, 
       -1.0f,  1.0f,  0.0f, */
    };

  s_VertexBufferQuad = FLB::VertexBuffer::create(api, terminal, vertices, sizeof(vertices));
  FLB::BufferLayout layout = {
    {ShaderDataType::Float3, "a_QuadPoints"},
    {ShaderDataType::Float2, "a_QuadTextCoord"}
  };
  s_VertexBufferQuad -> setLayout(layout);
}
