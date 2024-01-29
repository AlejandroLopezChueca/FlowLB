#pragma once

#include "API.h"
#include "buffer.h"

#include <cstdint>
#include <memory>
#include <vector>

//#include "rendererAPI.h"

namespace FLB 
{
  class VertexArray
  {

  public:
    virtual ~VertexArray() = default;

    virtual void bind() const = 0;
    virtual void unbind() const = 0;

    virtual void addVertexBuffer(FLB::VertexBuffer* const vertexBuffer) = 0;
    virtual void setIndexBuffer(const FLB::IndexBuffer* indexBuffer) = 0;
    virtual const std::vector<FLB::VertexBuffer*>& getVertexBuffers() const = 0; 

    virtual void updateMemberBufferData(const uint32_t idxBuffer, const uint32_t offset, const uint32_t size, const void* data) = 0;

    virtual void recreate() = 0;

    static std::unique_ptr<VertexArray> create(FLB::API api);
  };


}
