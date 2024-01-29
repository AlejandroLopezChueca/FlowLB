#pragma once

#include <cstdint>
#include <memory>

#include "graphics/API.h"
namespace FLB
{
  struct FrameBufferSpecifications
  {
    uint32_t width;
    uint32_t height;

    // if we are rendering to screen or not
    bool swapChainTarget = false;
    bool is3D = false;

  };

  class FrameBuffer
  {
    public:

      virtual ~FrameBuffer() = default;

      virtual void bind() = 0;
      virtual void bindTexture(uint32_t binding) = 0;
      virtual void unbind() = 0;

      virtual void resize(uint32_t width, uint32_t height) = 0;

      virtual uint32_t getTextureColorID() const = 0;

      virtual FrameBufferSpecifications& getSpecifications() = 0;

      static std::unique_ptr<FrameBuffer> create(FLB::API api, const FrameBufferSpecifications& specs);

  };

}
