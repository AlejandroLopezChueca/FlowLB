#include "buffer.h"
#include "OpenGL/OpenGLBuffer.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>


static uint32_t FLB::ShaderDataTypeSize(ShaderDataType type)
  {
    switch (type) 
    { 
      case ShaderDataType::Float:     return 4;
      case ShaderDataType::Float2:    return 4 * 2;
      case ShaderDataType::SOAFloat2: return 4;
      case ShaderDataType::Float3:    return 4 * 3; 
      case ShaderDataType::Float4:    return 4 * 4; 
      case ShaderDataType::Mat4:      return 4 * 4 * 4; 
      default: throw std::invalid_argument("Unknown ShaderDataType");
    }
    return 0;
}

////////////////////////// BufferElement ////////////////////////// 

FLB::BufferElement::BufferElement(FLB::ShaderDataType type_ , const std::string name_, bool instanced_, size_t countStrideSOA_, bool normalized_)
  : name(name_), type(type_), size(FLB::ShaderDataTypeSize(type_)), offset(0), instanced(instanced_), countStrideSOA(countStrideSOA_), normalized(normalized_) {}

uint32_t FLB::BufferElement::getComponentCount() const
{
  switch (type)
  {
    case ShaderDataType::Float:      return 1;
    case ShaderDataType::Float2:     return 2;
    case ShaderDataType::SOAFloat2:  return 2;
    case ShaderDataType::Float3:     return 3;
    case ShaderDataType::Float4:     return 4;
    case ShaderDataType::Mat4:       return 4; //max amount of data per vertex aatribute is equal to vec4
    default: throw std::invalid_argument("Unknown ShaderDataType");
  }
  return 0;
}


////////////////////////// BufferLayout ////////////////////////// 

FLB::BufferLayout::BufferLayout(const std::initializer_list<BufferElement>& elements)
  : m_Elements(elements) {calculateOffsetsAndStride();}

void FLB::BufferLayout::calculateOffsetsAndStride()
{
  size_t offset = 0;
  m_Stride = 0;
  for (auto& element : m_Elements)
  {
      element.offset = offset;
      offset += element.size;
      m_Stride += element.size;
  }
}

////////////////////////// VertexBuffer ////////////////////////// 

std::unique_ptr<FLB::VertexBuffer> FLB::VertexBuffer::create(FLB::API api, Fl_Simple_Terminal* terminal, void* vertices, size_t size)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLVertexBuffer>(vertices, size);
    case FLB::API::NONE:
      return nullptr;
  };
  return nullptr;
}

////////////////////////// IndexBuffer ////////////////////////// 

std::unique_ptr<FLB::IndexBuffer> FLB::IndexBuffer::create(FLB::API api, Fl_Simple_Terminal* terminal, uint32_t* indices, uint32_t count)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLIndexBuffer>(indices, count);
    case FLB::API::NONE:
      return nullptr; 
  };
  return nullptr;
}

////////////////////////// UniformBuffer ////////////////////////// 

std::unique_ptr<FLB::UniformBuffer> FLB::UniformBuffer::create(FLB::API api, Fl_Simple_Terminal *terminal, size_t size, uint32_t bindingPoint)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLUniformBuffer>(size, bindingPoint);
    case FLB::API::NONE:
      return nullptr; 
  };
  return nullptr;

}
