#include "frameBuffer.h"
#include "OpenGL/OpenGLFrameBuffer.h"
#include "graphics/API.h"


std::unique_ptr<FLB::FrameBuffer> FLB::FrameBuffer::create(FLB::API api, const FrameBufferSpecifications& specs)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLFrameBuffer>(specs);

    case FLB::API::NONE:
      return nullptr;
  }

}
