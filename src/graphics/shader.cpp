#include <glad/glad.h>
#include <memory>

#include "API.h"
#include "FL/Fl_Simple_Terminal.H"
#include "OpenGL/OpenGLShader.h"
#include "shader.h"


std::unique_ptr<FLB::Shader> FLB::Shader::create(const std::string& filePath, FLB::API api, Fl_Simple_Terminal* terminal)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLShader>(filePath, terminal);
    case FLB::API::NONE:
      return nullptr; 
  };
}
std::unique_ptr<FLB::Shader> FLB::Shader::create(const std::string& vertexSrc, const std::string& fragmentSrc, FLB::API api, Fl_Simple_Terminal* terminal)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      return std::make_unique<FLB::OpenGLShader>(vertexSrc, fragmentSrc, terminal);
    case FLB::API::NONE:
      return nullptr; 
  };
}

