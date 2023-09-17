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

      void resize();

      void bind() override;
      void bindTexture() override;
      void unbind() override;

      uint32_t getTextureColorID() const override {return m_ColorAttachment;}
      const FrameBufferSpecifications& getSpecifications() const override {return m_Specifications;}


    private:
      uint32_t m_RendererID;
      uint32_t m_ColorAttachment;
      uint32_t m_DepthAttachment;
      FrameBufferSpecifications m_Specifications;

  };
}
