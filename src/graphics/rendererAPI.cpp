

#include "rendererAPI.h"
#include "OpenGL/OpenGLRendererAPI.h"
#include <array>
#include <cstdint>

// default API
FLB::API FLB::RendererAPI::s_API = FLB::API::NONE;

std::unique_ptr<FLB::RendererAPI> FLB::RendererAPI::create(const std::array<uint32_t,4> indices)

{
  switch (s_API) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLRendererAPI>(indices);
    case FLB::API::NONE:
      return nullptr; 
  }

}


