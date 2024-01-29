#include <cstdint>
#include <glad/glad.h>
#include <iostream>
#include <vector>

#include "stb_image.h"
#include "OpenGLTexture.h"
#include "glm/ext/vector_float3.hpp"
#include "graphics/texture.h"

namespace FLB::GLTextureUtils
{
  static GLenum FLBImageFormatToInternalGL(FLB::ImageFormat format)
  {
    switch (format) 
    {
      case FLB::ImageFormat::R8:      return GL_R8;
      case FLB::ImageFormat::RGB8:    return GL_RGB;
      case FLB::ImageFormat::RGBA8:   return GL_RGBA;
      case FLB::ImageFormat::R32F:    return GL_R32F;
      case FLB::ImageFormat::RG32F:   return GL_RG32F;
      case FLB::ImageFormat::RGB32F:  return GL_RGB32F;
      case FLB::ImageFormat::RGBA32F: return GL_RGBA32F; 
    }
  }
}

/////////////////////////////////// OpenGLTexture1D ///////////////////////////////////
///
FLB::OpenGLTexture1D::OpenGLTexture1D(const std::vector<glm::vec3>& colors)
{
  glCreateTextures(GL_TEXTURE_1D, 1, &m_RendererID);
  glBindTexture(GL_TEXTURE_1D, m_RendererID);

  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, colors.size(), 0, GL_RGB, GL_FLOAT, colors.data());
  glBindTexture(GL_TEXTURE_1D, 0);
}

FLB::OpenGLTexture1D::~OpenGLTexture1D()
{
  glDeleteTextures(1, &m_RendererID);

}

void FLB::OpenGLTexture1D::bind(uint32_t slot) const
{
  glBindTextureUnit(slot, m_RendererID);
}

void FLB::OpenGLTexture1D::setColors(const std::vector<glm::vec3> &colors)
{
  //glBindTexture(GL_TEXTURE_1D, m_RendererID);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, colors.size(), 0, GL_RGB, GL_FLOAT, colors.data());
}

void FLB::OpenGLTexture1D::setData(void *data, uint32_t size)
{

}


/////////////////////////////////// OpenGLTexture2D ///////////////////////////////////

FLB::OpenGLTexture2D::OpenGLTexture2D(uint32_t width, uint32_t height, FLB::ImageFormat format)
  : m_Width(width), m_Height(height)
{
  m_InternalFormat = FLB::GLTextureUtils::FLBImageFormatToInternalGL(format);

  glCreateTextures(GL_TEXTURE_2D, 1, &m_RendererID);
  glBindTexture(GL_TEXTURE_2D, m_RendererID);

  glTexStorage2D(GL_TEXTURE_2D, 1, m_InternalFormat, m_Width, m_Height);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
}

FLB::OpenGLTexture2D::OpenGLTexture2D(const std::filesystem::path& path)
{
  int width, height, channels;
  // for OpenGL it is necesary to flip
  stbi_set_flip_vertically_on_load(1);
  stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
  if (data)
  {
    m_Width = width;
    m_Height = height;
    GLenum internalFormat = 0, dataFormat = 0;
    if (channels == 4)
    {
      internalFormat = GL_RGBA8;
      dataFormat = GL_RGBA;
    }
    else if (channels == 3)
    {
      internalFormat = GL_RGB8;
      dataFormat = GL_RGB;
    }
    else if (channels == 1)
    {
      internalFormat = GL_R8;
      dataFormat = GL_RED;
    }
    m_InternalFormat = internalFormat;
    m_DataFormat = dataFormat;
    // check formats supperted
    if (!internalFormat || !dataFormat)
    {
      std::string  message = "Format not supported in texture: " + path.string() + "\n";
      std::cout << message;
      //GV::ImGuiLogPanel::addLog(message, LogCategory::Error);
    }

    glCreateTextures(GL_TEXTURE_2D, 1, &m_RendererID);
    glBindTexture(GL_TEXTURE_2D, m_RendererID);
    glTexStorage2D(GL_TEXTURE_2D, 1, m_InternalFormat, m_Width, m_Height);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTextureSubImage2D(m_RendererID, 0, 0, 0, m_Width, m_Height, m_DataFormat, GL_UNSIGNED_BYTE, data);

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  else 
  {
    std::string  message = "failed to load texture: " + path.string() + "\n";
    std::cout << message;
    //GV::ImGuiLogPanel::addLog(message, LogCategory::Error);
  }
}

FLB::OpenGLTexture2D::~OpenGLTexture2D()
{
  glDeleteTextures(1, &m_RendererID);
}

void FLB::OpenGLTexture2D::bind(uint32_t slot) const
{
  glBindTextureUnit(slot, m_RendererID);
}

void FLB::OpenGLTexture2D::setData(void *data, uint32_t size)
{
  glTextureSubImage2D(m_RendererID, 0, 0, 0, m_Width, m_Height, m_DataFormat, GL_UNSIGNED_BYTE, data);
}
