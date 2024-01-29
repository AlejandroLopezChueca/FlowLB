#include "texture.h"
#include "OpenGL/OpenGLTexture.h"
#include "colorMaps.h"
#include <cstdint>
#include <vector>

/////////////////////////////////// Texture1D ///////////////////////////////////

std::unique_ptr<FLB::Texture1D> FLB::Texture1D::create(FLB::API api, const std::vector<glm::vec3>& colors)
{
  //std::vector<glm::vec3> defaultColor = FLB::ColorMaps::colorMaps["imola"];
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLTexture1D>(colors);
    case FLB::API::NONE:
      return nullptr; 
  }
  return nullptr;
}

/////////////////////////////////// Texture2D ///////////////////////////////////

std::unique_ptr<FLB::Texture2D> FLB::Texture2D::create(FLB::API api, uint32_t width, uint32_t height, FLB::ImageFormat format)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLTexture2D>(width, height, format);
    case FLB::API::NONE:
      return nullptr; 
  };
}

std::unique_ptr<FLB::Texture2D> FLB::Texture2D::create(FLB::API api, const std::filesystem::path& path)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLTexture2D>(path);
    case FLB::API::NONE:
      return nullptr; 
  };

}
