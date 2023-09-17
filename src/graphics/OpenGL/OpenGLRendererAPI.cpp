#include <cstdint>
#include <glad/glad.h>

#include "OpenGLRendererAPI.h"

#include "graphics/renderer.h"

#include "graphics/rendererAPI.h"
#include "iostream"

#define GL_POINT_SMOOTH 0x0B10

FLB::OpenGLRendererAPI::OpenGLRendererAPI(uint32_t indices[4])
  : m_Indices{indices[0], indices[1], indices[2], indices[3]} {}

void FLB::OpenGLRendererAPI::setClearColor(const glm::vec4& color)
{
  glClearColor(color.r, color.g, color.b, color.a);
}

void FLB::OpenGLRendererAPI::clear()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void FLB::OpenGLRendererAPI::drawElements(const FLB::VertexArray& vertexArray, size_t count)
{
  unsigned int indices[6] = {0, 1, 3, 3, 1, 2};
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)&indices);
}

void FLB::OpenGLRendererAPI::drawPoints(const FLB::VertexArray& vertexArray, size_t count)
{
  glPointSize(FLB::Renderer::s_PointSize);
  glDrawArrays(GL_POINTS, 0, count);
}
