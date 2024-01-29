#include <cstddef>
#include <cstdint>

#include "OpenGLBuffer.h"
#include <iostream>

///////////////////////// VertexBuffer ///////////////////////////////
///
FLB::OpenGLVertexBuffer::OpenGLVertexBuffer(void* vertices, size_t size)
{
  resize(vertices, size);
}

FLB::OpenGLVertexBuffer::~OpenGLVertexBuffer()
{
  glDeleteBuffers(1, &m_RendererID);
}

void FLB::OpenGLVertexBuffer::bind() const
{
  glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
}

void FLB::OpenGLVertexBuffer::unbind() const
{
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void FLB::OpenGLVertexBuffer::resize(void* vertices, size_t size)
{
  if (m_RendererID) glDeleteBuffers(1, &m_RendererID);
  glCreateBuffers(1, &m_RendererID);
  glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
  glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);
}

///////////////////////// IndexBuffer ///////////////////////////////

FLB::OpenGLIndexBuffer::OpenGLIndexBuffer(uint32_t* indices, uint32_t count)
{
  glCreateBuffers(1, &m_RendererID);
  // GL_ELEMENT_ARRAY_BUFFER is not valid without an actively bound VAO
 // Binding with GL_ARRAY_BUFFER allows the data to be loaded regardless of VAO state. 
  glBindBuffer(GL_ARRAY_BUFFER, m_RendererID);
  glBufferData(GL_ARRAY_BUFFER, count * sizeof(uint32_t), indices, GL_STATIC_DRAW);

}

FLB::OpenGLIndexBuffer::~OpenGLIndexBuffer()
{
  glDeleteBuffers(1, &m_RendererID);
}

void FLB::OpenGLIndexBuffer::bind() const
{
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID);
}

void FLB::OpenGLIndexBuffer::unbind() const
{
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

///////////////////////// UniformBuffer ///////////////////////////////

FLB::OpenGLUniformBuffer::OpenGLUniformBuffer(size_t size, uint32_t bindingPoint)
  : m_Size(size)
{
  glCreateBuffers(1, &m_RendererID);
  glBindBuffer(GL_UNIFORM_BUFFER, m_RendererID);
  glBufferData(GL_UNIFORM_BUFFER, size, nullptr, GL_STATIC_DRAW);
  glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, m_RendererID);
}

FLB::OpenGLUniformBuffer::~OpenGLUniformBuffer()
{
  glDeleteBuffers(1, &m_RendererID);
}

void FLB::OpenGLUniformBuffer::bind() const
{
  glBindBuffer(GL_UNIFORM_BUFFER, m_RendererID);
}

void FLB::OpenGLUniformBuffer::unbind() const
{
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void FLB::OpenGLUniformBuffer::setData(const void *data) const
{
  glBindBuffer(GL_UNIFORM_BUFFER, m_RendererID);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, m_Size, data);
}
