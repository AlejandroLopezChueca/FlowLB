#include "buffer.h"
#include "OpenGL/OpenGLBuffer.h"
#include <cstdint>


static uint32_t FLB::ShaderDataTypeSyze(ShaderDataType type)
  {
    switch (type) 
    { 
      case ShaderDataType::Float:  return 4;
      case ShaderDataType::Float2: return 4 * 2;
      case ShaderDataType::Float3: return 4 * 3; 
    }
    return 0;
}

////////////////////////// BufferElement ////////////////////////// 

FLB::BufferElement::BufferElement(FLB::ShaderDataType type_ ,const std::string name_, bool normalized_)
  : name(name_), type(type_), size(FLB::ShaderDataTypeSyze(type_)), offset(0), normalized(normalized_) {}

uint32_t FLB::BufferElement::getComponentCount() const
{
  switch (type)
  {
    case ShaderDataType::Float:   return 1;
    case ShaderDataType::Float2:  return 2;
    case ShaderDataType::Float3:  return 3;
  }
  return 0;
}


////////////////////////// BufferLayout ////////////////////////// 

FLB::BufferLayout::BufferLayout(const std::initializer_list<BufferElement>& elements_)
  : elements(elements_) {calculateOffsetsAndStride();}

void FLB::BufferLayout::calculateOffsetsAndStride()
{
  size_t offset = 0;
  stride = 0;
  for (auto& element : elements)
  {
	element.offset = offset;
	offset += element.size;
	stride += element.size;
  }
}

////////////////////////// VertexBuffer ////////////////////////// 

std::unique_ptr<FLB::VertexBuffer> FLB::VertexBuffer::create(FLB::API api, Fl_Simple_Terminal* terminal, float* vertices, int size)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLVertexBuffer>(vertices, size);
    case FLB::API::NONE:
      return nullptr;
  
  };
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
}
