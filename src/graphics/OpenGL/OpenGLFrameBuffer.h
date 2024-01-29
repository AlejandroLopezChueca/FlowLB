#pragma once

#include "graphics/frameBuffer.h"
#include <cstdint>

namespace FLB
{
  class OpenGLFrameBuffer: public FrameBuffer
  {
    public:
      OpenGLFrameBuffer(const FrameBufferSpecifications& specs);
      ~OpenGLFrameBuffer();

      void create();

      void bind() override;
      void bindTexture(uint32_t binding) override;
      void unbind() override;
      
      void resize(uint32_t width, uint32_t height) override;

      uint32_t getTextureColorID() const override {return m_ColorAttachment;}
      FrameBufferSpecifications& getSpecifications() override {return m_Specifications;}


    private:
      uint32_t m_RendererID;
      uint32_t m_ColorAttachment;
      uint32_t m_DepthAttachment;
      FrameBufferSpecifications m_Specifications;

  };
}
