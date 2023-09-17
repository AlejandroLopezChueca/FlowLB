#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "camera.h"
#include "glm/matrix.hpp"
#include "glm/trigonometric.hpp"

FLB::OrthographicCamera::OrthographicCamera(float left, float right, float bottom, float top)
  : m_ProjectionMatrix(glm::ortho(left, right, bottom, top, -1.0f, 1.0f)), m_ViewMatrix(1.0f)
{
  m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
}

void FLB::OrthographicCamera::setProjection(float left, float right, float bottom, float top)
{
  m_ProjectionMatrix = glm::ortho(left, right, bottom, top, -1.0f, 1.0f);
  m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
}

void FLB::OrthographicCamera::recalculateViewMatrix()
{
  glm::mat4 transform = glm::translate(glm::mat4(1.0f), m_Position) * glm::rotate(glm::mat4(1.0f), glm::radians(m_Rotation), glm::vec3(0, 0, 1));

  m_ViewMatrix = glm::inverse(transform);
  // the order of multiplication is due to glm, which is column major
  m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
}
