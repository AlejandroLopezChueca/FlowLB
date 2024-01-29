#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "camera.h"
#include "glm/gtx/quaternion.hpp"
#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>

FLB::OrthographicCamera::OrthographicCamera(float left, float right, float bottom, float top)
  : m_ProjectionMatrix(glm::ortho(left, right, bottom, top, -1.0f, 1.0f)), m_ViewMatrix(1.0f)
{
  m_IdentityMatrix = glm::mat4(1.0f);
  m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
}

void FLB::OrthographicCamera::setProjection(float left, float right, float bottom, float top)
{
  m_ProjectionMatrix = glm::ortho(left, right, bottom, top, -1.0f, 1.0f);
  m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
}

glm::vec3 FLB::OrthographicCamera::getRightDirection() const
{
  return glm::rotate(getOrientation(), glm::vec3(1.0f, 0.0f, 0.0f)); 
}

glm::vec3 FLB::OrthographicCamera::getUpDirection() const
{
  return glm::rotate(getOrientation(), glm::vec3(0.0f, 1.0f, 0.0f)); 
}

glm::quat FLB::OrthographicCamera::getOrientation() const
{
  return glm::quat(glm::vec3(0.0f, 0.0f, -m_Rotation));
}

void FLB::OrthographicCamera::recalculateViewMatrix()
{
  glm::mat4 transform = glm::translate(m_IdentityMatrix, m_Position) * glm::toMat4(getOrientation());

  m_ViewMatrix = glm::inverse(transform);
  // the order of multiplication is due to glm, which is column major
  m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
}
