#pragma once

//#include <GL/gl.h>
//#include <GL/glext.h>
#include <cstdint>
#include <iostream>

#include <GLFW/glfw3.h>
#include "FL/Fl_Simple_Terminal.H"

#include "graphics/window.h"
//#include "graphics/cameraController.h"

namespace FLB
{
  template<typename T>
  class OpenGLWindow: public Window
  {

    public:
      OpenGLWindow(Fl_Simple_Terminal* terminal, T* cameraController, bool enableVSync = false);
      ~OpenGLWindow();

      void setVSync(bool enabled) override;
      void update() override;

      inline void* getWindow() const override {return m_renderWindow;};
      
      void setGizmoOperation(int* gizmoOperation) override {s_GizmoOperation = gizmoOperation;}

      //bool isKeyPressed(int keyCode) override;

      //bool render;
      //bool a = render;

    private:
      static void setScrollCallback(GLFWwindow* window,double xOffset, double yOffset);

      static void setKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

      static void debugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);

      GLFWwindow* m_renderWindow;
      static bool s_GLFWInitialized;
      static T* s_cameraController;
      static int* s_GizmoOperation;
      

  };

}
