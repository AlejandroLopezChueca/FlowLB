#include <cstdint>
#include <glad/glad.h>
#include <vector>

#include "OpenGLTexture.h"
#include "glm/ext/vector_float3.hpp"

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


/////////////////////////////////// OpenGLTexture2D ///////////////////////////////////

FLB::OpenGLTexture2D::OpenGLTexture2D(uint32_t width, uint32_t height)
  : m_Width(width), m_Height(height)
{
  glCreateTextures(GL_TEXTURE_2D, 1, &m_RendererID);
  glBindTexture(GL_TEXTURE_2D, m_RendererID);

  glTexStorage2D(GL_TEXTURE_2D, 1,GL_RGBA16F, m_Width, m_Height);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

}

FLB::OpenGLTexture2D::~OpenGLTexture2D()
{
  glDeleteTextures(1, &m_RendererID);
}

void FLB::OpenGLTexture2D::bind(uint32_t slot) const
{
  glBindTextureUnit(slot, m_RendererID);
}
