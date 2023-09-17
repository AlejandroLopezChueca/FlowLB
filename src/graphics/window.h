#pragma once

#include <cstdint>
#include <memory>
#include "FL/Fl_Simple_Terminal.H"

#include "renderer.h"
#include "cameraController.h"

namespace FLB 
{
  class Window
  {
    public:

      virtual ~Window() = default;

      virtual void setVSync(bool enabled) = 0;
      virtual void update() = 0;

      virtual void* getWindow() const = 0;
      uint32_t getWidth() const {return m_Width;}
      uint32_t getHeight() const {return m_Height;}

      //virtual bool isKeyPressed(int keyCode) = 0;

      template<typename T>
      static Window* create(FLB::API api, FLB::OrthographicCameraController* orthographicCameraController, bool is3D, Fl_Simple_Terminal* terminal);

      //static bool render;

    protected:
      std::string title = "FLowLB 0.1";
      uint32_t m_Width = 1600;
      uint32_t m_Height = 900;
  };
    
}
