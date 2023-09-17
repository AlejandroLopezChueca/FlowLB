#pragma once

#include "graphics/API.h"
#include <cstdint>
#include <memory>
#include <glm/glm.hpp>
#include <sys/types.h>
#include <vector>

namespace FLB
{
  class Texture
  {
    public:
      virtual ~Texture() = default;

      virtual uint32_t getWidth() const = 0;
      virtual uint32_t getHeight() const = 0;
      virtual uint32_t getID() const = 0;

      virtual void bind(uint32_t slot) const = 0;
  };

  class Texture1D: public Texture
  {
    public:
      virtual void setColors(const std::vector<glm::vec3>& colors) = 0;
      static std::unique_ptr<Texture1D> create(FLB::API api); 
  };

  class Texture2D: public Texture
  {
    public:
      static std::unique_ptr<Texture2D> create(FLB::API api, uint32_t width, uint32_t height); 
  };

}
