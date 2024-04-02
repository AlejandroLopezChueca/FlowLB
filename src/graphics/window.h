#pragma once

#include "cameraController.h"

#include "API.h"
#include "glm/fwd.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <FL/Fl_Simple_Terminal.H>

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
      virtual bool getMousePos(glm::dvec2& mousePos) const = 0;
      virtual bool isLeftButtonMouseClicked() const = 0;

      //virtual bool isKeyPressed(int keyCode) = 0;
      virtual void setGizmoOperation(int* gizmoOperation) = 0;

      virtual bool isInitialized() const = 0;

      template<typename T>
      static std::unique_ptr<Window> create(FLB::API api, FLB::OrthographicCameraController* orthographicCameraController, bool is3D, Fl_Simple_Terminal* terminal);

      //static bool render;

    protected:
      std::string title = "FLowLB 0.1";
      uint32_t m_Width = 1600;
      uint32_t m_Height = 900;
  };
    
}
