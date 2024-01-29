#pragma once

#include <cstdint>
#include <vector>
#include <glad/glad.h>

#include "graphics/vertexArray.h"
#include "graphics/buffer.h"
#include "OpenGLVertexArray.h"

namespace FLB
{
//static GLenum ShaderDataTypeToOpenGLBaseType(FLB::ShaderDataType type); 

  class OpenGLVertexArray: public VertexArray
  {
    public:
      OpenGLVertexArray();
      ~OpenGLVertexArray();

      virtual void bind() const override;
      virtual void unbind() const override;


      void addVertexBuffer(VertexBuffer* const vertexBuffer) override;
      void setIndexBuffer(const FLB::IndexBuffer* indexBuffer) override;
      const std::vector<VertexBuffer*>& getVertexBuffers() const override {return m_VertexBuffers;}; 
    
      void updateMemberBufferData(const uint32_t idxBuffer, const uint32_t offset, const uint32_t size, const void* data) override;

      void recreate() override;

    private:
      uint32_t m_RendererID;
      uint32_t m_VertexBufferIndex = 0;
      std::vector<VertexBuffer*> m_VertexBuffers;

  };

}
