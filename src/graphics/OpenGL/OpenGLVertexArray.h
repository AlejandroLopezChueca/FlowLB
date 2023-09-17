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

      virtual void bind() const override;
      virtual void unbind() const override;


      void addVertexBuffer(VertexBuffer* const vertexBuffer) override;
      const std::vector<VertexBuffer*>& getVertexBuffers() const override {return vertexBuffers;}; 

    private:
      uint32_t m_rendererID;
      uint32_t m_vertexBufferIndex = 0;
      std::vector<VertexBuffer*> vertexBuffers;

  };

}
