#pragma once

#include <cstddef>
#include <cstdint>
#include <glad/glad.h>
#include "graphics/buffer.h"

namespace FLB
{
  class OpenGLVertexBuffer: public VertexBuffer
  {
    public:
      OpenGLVertexBuffer(void* vertices, size_t size);
      ~OpenGLVertexBuffer();

      void bind() const override;
      void unbind() const override;
      const BufferLayout& getLayout() const override {return m_layout;}
      void setLayout(const BufferLayout& layout) override {m_layout = layout;}
      unsigned int getVertexBufferID() const override {return m_RendererID;}
      void resize(void* vertices, size_t size) override;
    
    private:
      unsigned int m_RendererID;
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
      uint32_t m_RendererID;
  };

  class OpenGLUniformBuffer: public UniformBuffer
  {
    public:
      OpenGLUniformBuffer(size_t size, uint32_t bindingPoint);
      ~OpenGLUniformBuffer();

      void bind() const override;
      void unbind() const override;
      void setData(const void* data) const override;
    private:
      uint32_t m_RendererID;
      size_t m_Size;

  };
}
