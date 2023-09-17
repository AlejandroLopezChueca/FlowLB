#include "texture.h"
#include "OpenGL/OpenGLTexture.h"
#include "colorMaps.h"
#include <cstdint>
#include <vector>

std::unique_ptr<FLB::Texture1D> FLB::Texture1D::create(FLB::API api)
{
  std::vector<glm::vec3> defaultColor = FLB::ColorMaps::colorMaps["imola"];
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLTexture1D>(defaultColor);
    case FLB::API::NONE:
      return nullptr; 
  };
}

std::unique_ptr<FLB::Texture2D> FLB::Texture2D::create(FLB::API api, uint32_t width, uint32_t height)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLTexture2D>(width, height);
    case FLB::API::NONE:
      return nullptr; 
  };
}
