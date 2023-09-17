#pragma once

#include <memory>
#include <vector>

//#include "rendererAPI.h"
#include "API.h"
#include "buffer.h"

namespace FLB 
{
  class VertexArray
  {

  public:
    virtual ~VertexArray() = default;

    virtual void bind() const = 0;
    virtual void unbind() const = 0;

    virtual void addVertexBuffer(FLB::VertexBuffer* const vertexBuffer) = 0;
    virtual const std::vector<FLB::VertexBuffer*>& getVertexBuffers() const = 0; 

    static std::unique_ptr<VertexArray> create(FLB::API api);
  };


}
