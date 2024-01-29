#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "FL/Fl_Simple_Terminal.H"

#include <glm/glm.hpp>

#include "API.h"

namespace FLB 
{
  class Shader
  {
    public:
      virtual ~Shader() = default;

      virtual void bind() const = 0;
      virtual void unbind() const = 0;

      virtual void setInt(const std::string& name, int value) const = 0;
      virtual void setFloat(const std::string& name, const float& value) const = 0;
      virtual void setFloat2(const std::string& name, const glm::vec2& vector) const = 0;
      virtual void setFloat3(const std::string& name, const glm::vec3& vector) const = 0;
      virtual void setFloat4(const std::string& name, const glm::vec4& vector) const = 0;
      virtual void setMat4(const std::string& name, const glm::mat4& matrix) const = 0;

      static std::unique_ptr<Shader> create(const std::string& filePath, FLB::API api, Fl_Simple_Terminal* terminal);
      static std::unique_ptr<Shader> create(const std::string& vertexSrc, const std::string& fragmentSrc, FLB::API api, Fl_Simple_Terminal* terminal);
  };

}
