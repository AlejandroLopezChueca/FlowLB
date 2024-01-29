#pragma once

#include "graphics/window.h"

#include "GLFW/glfw3.h"
#include <cstdint>



namespace FLB 
{
  class ImGuiLayer
  {
    public:
      ImGuiLayer() = default;
      ~ImGuiLayer();

      void onAttach();
      void onImGuiRender();

      void begin();
      void end();

      void setContext(FLB::Window* window);
      void SetDarkThemeColors();

      private:

      FLB::Window* m_Window;
  };

}
