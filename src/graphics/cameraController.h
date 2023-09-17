
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include "graphics/camera.h"
#include <memory>

namespace FLB
{
  class OrthographicCameraController
  {
    public:
      OrthographicCameraController(float width, float height);

      void GLFWUpdate(GLFWwindow* window);
      void Resize(float widht, float height);
      void onMouseScrolled(float xOffset, float yOffset);

      const FLB::OrthographicCamera& getCamera() const {return m_Camera;}

      void setZoomLevel(float zoomLevel) {m_ZoomLevel = zoomLevel;}


    private:
      float m_AspectRatio;
      float m_Width;
      float m_Height;
      float m_ZoomLevel = 1.0f;
      OrthographicCamera m_Camera;

      glm::vec3 m_CameraPosition = {0.0f, 0.0f, 0.0f};
      float m_CameraRotation = 0.0f; // in degrees, in the anti-clock direction
      float m_CameraTranslationSpeed = 0.05f;
      float m_CameraRotationSpeed = 10.0f;
  };

}
