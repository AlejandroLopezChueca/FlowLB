#pragma once

#include <cstdint>
#include <glad/glad.h>
#include "graphics/buffer.h"

namespace FLB
{
  class OpenGLVertexBuffer: public VertexBuffer
  {
    public:
      OpenGLVertexBuffer(float* vertices, int size);
      ~OpenGLVertexBuffer();

      void bind() const override;
      void unbind() const override;
      const BufferLayout& getLayout() const override {return m_layout;}
      void setLayout(const BufferLayout& layout) override {m_layout = layout;}
      unsigned int getVertexBufferID() const override {return m_rendererID;}
    
    private:
      unsigned int m_rendererID;
      BufferLayout m_layout;
  };
  
  class OpenGLIndexBuffer: public IndexBuffer
  {
    public:
      OpenGLIndexBuffer(uint32_t* indices, uint32_t count);
      ~OpenGLIndexBuffer();

      void bind() const override;
      void unbind() const override;
    
    private:
      uint32_t m_rendererID;
  };
}
