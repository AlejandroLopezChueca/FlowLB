#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include "geometry/mesh.h"
#include "graphics/camera.h"
#include <memory>
#include <utility>
#include <vector>

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
      float& getCameraTranslationSpeed() {return m_CameraTranslationSpeedFactor;}

      void setDomainData(FLB::Mesh* mesh);


      /**
       * @brief Get the bounds of the camera with respect of the domain. Extract the max and the min (indices) of the lattice nodes that are been seen by the camera
       *
       */
      void getCameraDomainBounds(unsigned int* cameraBounds);
      
      /**
       * @brief Get the bounds of the camera with respect of the domain. Extract the max and the min (SI values in global space) of the lattice nodes that are been seen by the camera
       *
       */
      std::array<float, 8>& getCameraDomainBoundsSI();
      bool getUpdateCameraDomainCornersSI() const {return m_UpdateCameraDomainCornersSI;}

      const glm::vec4& getViewUpperLeftCornerDomain() const {return m_ViewUpperLeftCornerDomain;}
      const glm::vec4& getViewLowerRightCornerDomain() const {return m_ViewLowerRightCornerDomain;}
      const float getxMaxViewCamera() const {return m_AspectRatio * m_ZoomLevel;}
      const float getyMaxViewCamera() const {return m_ZoomLevel;}

    private:

      void recalculateDomainCoordViewSpace();

      float m_AspectRatio;
      float m_ViewportWidth;
      float m_ViewportHeight;
      float m_ZoomLevel = 1.0f;
      OrthographicCamera m_Camera;

      //glm::vec3 m_CameraPosition = {0.0f, 0.0f, 0.0f};
      glm::vec2 m_InitialMousePosition = {0.0f, 0.0f};
      float m_CameraRotation = 0.0f; // in degrees, in the anti-clock direction
      float m_CameraTranslationSpeed = 0.1f;
      float m_CameraRotationSpeed = 0.5f;

      float m_CameraTranslationSpeedFactor = 2.0f;
      //  Max and min values of the domain in SI units
      glm::vec4 m_UpperLeftCornerDomain;
      glm::vec4 m_LowerRightCornerDomain;

      //  Max and min values of the domain in SI units in the view space
      glm::vec4 m_ViewUpperLeftCornerDomain;
      glm::vec4 m_ViewLowerRightCornerDomain;

      // max and min lattice nodes of the domain corners visibel by the camera (it include a border of one lattice around the camera view)
      unsigned int m_CameraDomainCornersNodes[4];
      bool m_UpdateCameraDomainCornersNodes = true;

      // x, y max and min values of the domain's corners visible by the camera (it include the border of one lattice around the camera view)
      std::array<float, 8> m_CameraDomainCornersSI;
      bool m_UpdateCameraDomainCornersSI = true;

      // number of intervals in each direction
      unsigned int m_NxIntervalsDomain, m_NyIntervalsDomain;
  };

}
