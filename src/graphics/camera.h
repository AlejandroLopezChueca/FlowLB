#pragma once

//#include "cameraController.h"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>


namespace FLB
{
  class OrthographicCamera
  {
    public:
      OrthographicCamera(float left, float right, float bottom, float top);

      void setProjection(float left, float right, float bottom, float top);
      const glm::vec3& getPosition() const {return  m_Position;}
      float getRotation() const {return  m_Rotation;}

      glm::vec3 getRightDirection() const;
      glm::vec3 getUpDirection() const;
      glm::quat getOrientation() const;;

      const glm::mat4& getProjectionMatrix() const { return m_ProjectionMatrix; }
      const glm::mat4& getViewMatrix() const { return m_ViewMatrix; }
      const glm::mat4& getViewProjectionMatrix() const { return m_ViewProjectionMatrix;}
    private:

      void recalculateViewMatrix();

      glm::mat4 m_ProjectionMatrix;
      glm::mat4 m_ViewMatrix;
      glm::mat4 m_InverseViewMatrix;
      glm::mat4 m_ViewProjectionMatrix;

      glm::mat4 m_IdentityMatrix;

      glm::vec3 m_Position = {0.0f, 0.0f, 0.0f};
      // ratation in z axis
      float m_Rotation = 0.0f; // in degress, anti-clock direction

      friend class OrthographicCameraController;

  };
}
