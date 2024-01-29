#include "OpenGLVertexArray.h"
#include "graphics/buffer.h"
#include <GL/gl.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace FLB
{
  static GLenum ShaderDataTypeToOpenGLBaseType(FLB::ShaderDataType type)
  {
    switch (type)
    {
      case ShaderDataType::Float: return GL_FLOAT;
      case ShaderDataType::Float2: return GL_FLOAT;
      case ShaderDataType::SOAFloat2: return GL_FLOAT;
      case ShaderDataType::Float3: return GL_FLOAT;
      case ShaderDataType::Float4: return GL_FLOAT;
      case ShaderDataType::Mat4: return GL_FLOAT;
      default: throw std::invalid_argument("VertexArray: Unknown ShaderDataType\n");
    }
    return 0;
  }
}

/////////////////////////////// OpenGLVertexBuffer ///////////////////////////////

FLB::OpenGLVertexArray::OpenGLVertexArray()
{
  glCreateVertexArrays(1, &m_RendererID);
}

FLB::OpenGLVertexArray::~OpenGLVertexArray()
{
  glDeleteVertexArrays(1, &m_RendererID);
}

void FLB::OpenGLVertexArray::bind() const
{
  glBindVertexArray(m_RendererID);
}

void FLB::OpenGLVertexArray::unbind() const
{
  glBindVertexArray(0);
}

void FLB::OpenGLVertexArray::addVertexBuffer(VertexBuffer* const vertexBuffer)
{
  if (vertexBuffer -> getLayout().getElements().size() == 0) throw std::invalid_argument("Vertex Buffer has no layout");
  // make sure that the vertex array is bound
  glBindVertexArray(m_RendererID);
  vertexBuffer -> bind();

  const auto& layout = vertexBuffer -> getLayout();
  for (const auto& element : layout)
  {
    switch (element.type)
    {
      case ShaderDataType::Float:
      case ShaderDataType::Float2:
      case ShaderDataType::Float3:
      case ShaderDataType::Float4:
      {
	glEnableVertexAttribArray(m_VertexBufferIndex);
	glVertexAttribPointer(m_VertexBufferIndex,
	    element.getComponentCount(),
	    ShaderDataTypeToOpenGLBaseType(element.type),
	    element.normalized ? GL_TRUE : GL_FALSE,
	    layout.getStride(),
	    reinterpret_cast<const void*>(element.offset));
	  
	if (element.instanced) glVertexAttribDivisor(m_VertexBufferIndex, 1);
	m_VertexBufferIndex += 1;
	break;
      }
      case ShaderDataType::SOAFloat2:
      {
	uint32_t count = element.getComponentCount();
	for (int i = 0; i < count; i++)
	{
	  glEnableVertexAttribArray(m_VertexBufferIndex);
	  glVertexAttribPointer(m_VertexBufferIndex,
	      1, // 1 count due to SOA scheme
	      ShaderDataTypeToOpenGLBaseType(element.type),
	      element.normalized ? GL_TRUE : GL_FALSE,
	      layout.getStride(),
	      reinterpret_cast<const void*>(element.offset + sizeof(float) * element.countStrideSOA * i));
	  if (element.instanced) glVertexAttribDivisor(m_VertexBufferIndex, 1);
	  m_VertexBufferIndex += 1;
	}
	break;
      }
      case ShaderDataType::Mat4:
      {
	uint32_t count = element.getComponentCount();
	for (int i = 0; i < count; i++)
	{
	  glEnableVertexAttribArray(m_VertexBufferIndex);
	  glVertexAttribPointer(m_VertexBufferIndex,
	      count,
	      ShaderDataTypeToOpenGLBaseType(element.type),
	      element.normalized ? GL_TRUE : GL_FALSE,
	      layout.getStride(),
	      reinterpret_cast<const void*>(element.offset + sizeof(float) * count * i));
	  if (element.instanced) glVertexAttribDivisor(m_VertexBufferIndex, 1);
	  m_VertexBufferIndex += 1;
	}
	break;
      }
      default:
	throw std::invalid_argument("Vertex Array: Unknown ShaderDataType\n");
    
    }
  }
  m_VertexBuffers.push_back(vertexBuffer);
}

void FLB::OpenGLVertexArray::setIndexBuffer(const FLB::IndexBuffer *indexBuffer)
{
  glBindVertexArray(m_RendererID);
  indexBuffer -> bind();
}

void FLB::OpenGLVertexArray::updateMemberBufferData(const uint32_t idxBuffer, const uint32_t offset, const uint32_t size, const void* data)
{
  glNamedBufferSubData(m_VertexBuffers[idxBuffer] -> getVertexBufferID(), offset, size, data);
}

void FLB::OpenGLVertexArray::recreate()
{
  if (m_RendererID) glDeleteVertexArrays(1, &m_RendererID);
  glCreateVertexArrays(1, &m_RendererID);
  m_VertexBufferIndex = 0;
  m_VertexBuffers.clear();
}
