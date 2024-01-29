#pragma once

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include "graphics/camera.h"
#include <memory>
#include <utility>

namespace FLB
{
  class OrthographicCameraController
  {
    public:
      OrthographicCameraController(float width, float height);
      ~OrthographicCameraController() = default;

      void GLFWUpdate(GLFWwindow* window);
     // void resize(float widht, float height);
      void onMouseScrolled(float xOffset, float yOffset);
      void mousePan(const glm::vec2& delta);

      const FLB::OrthographicCamera& getCamera() const {return m_Camera;}

      void setZoomLevel(float zoomLevel) {m_ZoomLevel = zoomLevel;}
      void setViewportSize(float width, float height);

      std::pair<float, float> getPanSpeed() const;


    private:
      float m_AspectRatio;
      float m_ViewportWidth;
      float m_ViewportHeight;
      float m_ZoomLevel = 1.0f;
      OrthographicCamera m_Camera;

      glm::vec3 m_CameraPosition = {0.0f, 0.0f, 0.0f};
      glm::vec2 m_InitialMousePosition = {0.0f, 0.0f};
      float m_CameraRotation = 0.0f; // in degrees, in the anti-clock direction
      float m_CameraTranslationSpeed = 0.05f;
      float m_CameraRotationSpeed = 0.5f;
  };

}
