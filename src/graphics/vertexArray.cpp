#include <memory>

#include "vertexArray.h"
#include "graphics/buffer.h"
#include "OpenGL/OpenGLVertexArray.h"

std::unique_ptr<FLB::VertexArray> FLB::VertexArray::create(FLB::API api)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLVertexArray>();
    case FLB::API::NONE:
      return nullptr;
  
  };
}

