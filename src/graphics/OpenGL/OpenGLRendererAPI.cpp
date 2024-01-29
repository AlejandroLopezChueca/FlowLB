#include <array>
#include <cstddef>
#include <cstdint>
#include <glad/glad.h>

#include "OpenGLRendererAPI.h"

#include "graphics/renderer.h"

#include "graphics/rendererAPI.h"
#include "iostream"

#define GL_POINT_SMOOTH 0x0B10

FLB::OpenGLRendererAPI::OpenGLRendererAPI(const std::array<uint32_t, 4> indices)
  : m_Indices{indices} {}

void FLB::OpenGLRendererAPI::setClearColor(const glm::vec4& color)
{
  glClearColor(color.r, color.g, color.b, color.a);
}

void FLB::OpenGLRendererAPI::clear()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void FLB::OpenGLRendererAPI::drawElements() const
{
  static unsigned int indices[6] = {0, 1, 3, 3, 1, 2};
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)&indices);
}


void FLB::OpenGLRendererAPI::drawInstancedElements(size_t indexCount, uint32_t instanceCount) const
{
  glDrawElementsInstanced(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0, instanceCount);
}

void FLB::OpenGLRendererAPI::drawInstancedLines(size_t vertexCount, uint32_t instanceCount) const
{
  glDrawArraysInstanced(GL_LINE_LOOP, 0, vertexCount, instanceCount);
}

void FLB::OpenGLRendererAPI::drawPoints(size_t count, float pointSize) const
{
  glPointSize(pointSize);
  glDrawArrays(GL_POINTS, 0, count);
}
