#include <memory>

#include "OpenGL/OpenGLWindow.h"
#include "window.h"

template<typename T>
std::unique_ptr<FLB::Window> FLB::Window::create(FLB::API api, FLB::OrthographicCameraController* orthographicCameraController, bool is3D, Fl_Simple_Terminal* terminal)
{
  switch (api) 
  {
    case FLB::API::OPENGL:
      if (!is3D) return std::make_unique<FLB::OpenGLWindow<T>>(terminal, orthographicCameraController);
      //return std::make_unique<FLB::OpenGLWindow<T>>(terminal);
    case FLB::API::NONE:
      return nullptr; 
  };
}

//bool FLB::Window::render = true;
template std::unique_ptr<FLB::Window> FLB::Window::create<FLB::OrthographicCameraController>(FLB::API api, FLB::OrthographicCameraController* orthographicCameraController, bool is3D, Fl_Simple_Terminal *terminal);
