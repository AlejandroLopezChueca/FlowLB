#include "OpenGLVertexArray.h"
#include "graphics/buffer.h"
#include <GL/gl.h>
#include <stdexcept>

namespace FLB
{
  static GLenum ShaderDataTypeToOpenGLBaseType(FLB::ShaderDataType type)
  {
    switch (type)
    {
      case ShaderDataType::Float: return GL_FLOAT;
      case ShaderDataType::Float2: return GL_FLOAT;
      case ShaderDataType::Float3: return GL_FLOAT;
    }
    return 0;
  }
}

/////////////////////////////// OpenGLVertexBuffer ///////////////////////////////

FLB::OpenGLVertexArray::OpenGLVertexArray()
{
  glCreateVertexArrays(1, &m_rendererID);
}

void FLB::OpenGLVertexArray::bind() const
{
  glBindVertexArray(m_rendererID);
}

void FLB::OpenGLVertexArray::unbind() const
{
  glBindVertexArray(0);
}

void FLB::OpenGLVertexArray::addVertexBuffer(VertexBuffer* const vertexBuffer)
{
  if (vertexBuffer -> getLayout().getElements().size() == 0) throw std::invalid_argument("Vertex Buffer has no layout");
  // make sure that the vertex array is bound
  glBindVertexArray(m_rendererID);
  vertexBuffer -> bind();

  const auto& layout = vertexBuffer -> getLayout();
  for (const auto& element : layout)
  {
    switch (element.type)
    {
      case ShaderDataType::Float:
      case ShaderDataType::Float2:
      case ShaderDataType::Float3:
	glEnableVertexAttribArray(m_vertexBufferIndex);
	glVertexAttribPointer(m_vertexBufferIndex,
	    element.getComponentCount(),
	    ShaderDataTypeToOpenGLBaseType(element.type),
	    element.normalized ? GL_TRUE : GL_FALSE,
	    layout.getStride(),
	    (const void*)element.offset);
	m_vertexBufferIndex += 1; 
	break;
      default:
	throw std::invalid_argument("Unknown ShaderDataType");
    
    }
  }
  vertexBuffers.push_back(vertexBuffer);
}
