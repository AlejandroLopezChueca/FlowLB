#pragma once

#include <cstdint>
#include <iostream>

#include <GLFW/glfw3.h>
#include "FL/Fl_Simple_Terminal.H"

#include "graphics/window.h"

namespace FLB
{
  template<typename T>
  class OpenGLWindow: public Window
  {

    public:
      OpenGLWindow(Fl_Simple_Terminal* terminal, T* cameraController);
      ~OpenGLWindow();

      void setVSync(bool enabled) override;
      void update() override;

      inline void* getWindow() const override {return m_renderWindow;};

      //bool isKeyPressed(int keyCode) override;

      //bool render;
      //bool a = render;

    private:
      static void setScrollCallback(GLFWwindow* window,double xOffset, double yOffset);

      static void setKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

      GLFWwindow* m_renderWindow;
      static bool s_GLFWInitialized;
      static T* s_cameraController;
      /*struct WindowData
      {
	bool& isRendering = render;
      };

      WindowData windowData;*/

  };

}
