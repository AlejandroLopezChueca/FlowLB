#pragma once

#include "graphics/shader.h"
#include "FL/Fl_Simple_Terminal.H"
#include <glad/glad.h>
#include <cstdint>
#include <unordered_map>

namespace FLB
{
  class OpenGLShader: public Shader
  {
    public:
      OpenGLShader(const std::string& filePath, Fl_Simple_Terminal* terminal);
      /*
       * @brief Create a shader object in OpenGL
       *
       * @param[in] vertexSrc    Source code for vertex shader
       * @param[in] fragmentSrc  Source code for fragment shader
       */
      OpenGLShader(const std::string& vertexSrc, const std::string& fragmentSrc, Fl_Simple_Terminal* terminal);
      ~OpenGLShader();


      void bind() const override;
      void unbind() const override;
      
      void setInt(const std::string& name, int value) const override;
      void setFloat(const std::string& name, const float& value) const override;
      void setFloat2(const std::string& name, const glm::vec2& vector) const override;
      void setFloat3(const std::string& name, const glm::vec3& vector) const override;
      void setFloat4(const std::string& name, const glm::vec4& vector) const override;
      void setMat4(const std::string& name, const glm::mat4& matrix) const override;

    private:
      std::string readFile(const std::string& filePath, Fl_Simple_Terminal* terminal);
      std::unordered_map<GLenum, std::string> preProcess(const std::string& source, Fl_Simple_Terminal* terminal);

      void compile(const std::unordered_map<GLenum, std::string>& shaderSources, Fl_Simple_Terminal* terminal);

      uint32_t m_RendererID;
      std::string m_FilePath;

  };
}
