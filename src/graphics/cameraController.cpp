#include "cameraController.h"
#include "GLFW/glfw3.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <utility>
#include <cmath>

FLB::OrthographicCameraController::OrthographicCameraController(float width, float height)
  : m_ViewportWidth(width), m_ViewportHeight(height), m_AspectRatio(width/height), m_Camera(-m_AspectRatio * m_ZoomLevel, m_AspectRatio * m_ZoomLevel, -m_ZoomLevel, m_ZoomLevel)
{
}

void FLB::OrthographicCameraController::GLFWUpdate(GLFWwindow* window)
{
  if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
  {
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);
    const glm::vec2 mouse{xPos, yPos};
    const glm::vec2 delta = (mouse - m_InitialMousePosition) * 0.003f;
    m_InitialMousePosition = mouse;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
    {
      mousePan(delta);
      m_Camera.recalculateViewMatrix();
      recalculateDomainCoordViewSpace();
    }
  }

  else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
  {
    m_Camera.m_Rotation += m_CameraRotationSpeed;
    m_Camera.recalculateViewMatrix();
    recalculateDomainCoordViewSpace();
  }

  else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
  {
    m_Camera.m_Rotation -= m_CameraRotationSpeed;
    m_Camera.recalculateViewMatrix();
    recalculateDomainCoordViewSpace();
  }
}

void FLB::OrthographicCameraController::onMouseScrolled(float xOffset, float yOffset)
{
  m_ZoomLevel -= yOffset * 0.1f;
  m_ZoomLevel = std::max(m_ZoomLevel, 0.05f);
  m_Camera.setProjection(-m_AspectRatio * m_ZoomLevel, m_AspectRatio * m_ZoomLevel, -m_ZoomLevel, m_ZoomLevel);
  m_CameraTranslationSpeed = 0.1f * m_ZoomLevel;
  m_UpdateCameraDomainCornersSI = true;
  m_UpdateCameraDomainCornersNodes = true;
}

void FLB::OrthographicCameraController::mousePan(const glm::vec2& delta)
{
  auto [xSpeed, ySpeed] = getPanSpeed();
  m_Camera.m_Position += - m_Camera.getRightDirection() * xSpeed * delta.x * m_CameraTranslationSpeedFactor;
  m_Camera.m_Position += m_Camera.getUpDirection() * ySpeed * delta.y * m_CameraTranslationSpeedFactor;
  m_UpdateCameraDomainCornersSI = true;
  m_UpdateCameraDomainCornersNodes = true;
}

void FLB::OrthographicCameraController::setViewportSize(float width, float height)
{
  m_ViewportWidth = width;
  m_ViewportHeight = height;
  m_AspectRatio = m_ViewportWidth / m_ViewportHeight;
  m_Camera.setProjection(-m_AspectRatio * m_ZoomLevel, m_AspectRatio * m_ZoomLevel, -m_ZoomLevel, m_ZoomLevel);
  m_UpdateCameraDomainCornersSI = true;
  m_UpdateCameraDomainCornersNodes = true;
}

std::pair<float, float> FLB::OrthographicCameraController::getPanSpeed() const
{
  float x = std::min(m_ViewportWidth / 1000.0f, 2.4f); //  max of 2.4
  float xFactor = 0.036f * (x * x) - 0.1778f * x + 0.32021;
  
  float y = std::min(m_ViewportHeight / 1000.0f, 2.4f); //  max of 2.4
  float yFactor = 0.036f * (y * y) - 0.1778f * y + 0.32021;
  return {xFactor, yFactor};
}

void FLB::OrthographicCameraController::setDomainData(FLB::Mesh *mesh)
{
  m_UpperLeftCornerDomain = {mesh -> getxMin(), mesh-> getyMax(), 0.0f, 1.0f};
  m_LowerRightCornerDomain = {mesh -> getxMax(), mesh-> getyMin(), 0.0f, 1.0f};
  
  // Transform coordinates of the domain to view space
  m_ViewUpperLeftCornerDomain = m_Camera.m_ViewMatrix * m_UpperLeftCornerDomain;
  m_ViewLowerRightCornerDomain = m_Camera.m_ViewMatrix * m_LowerRightCornerDomain;

  // set number of intervals beetween nodes in each direction
  m_NxIntervalsDomain = mesh -> getNx() - 1;
  m_NyIntervalsDomain = mesh -> getNy() - 1;

  // initialization of the bounds
  unsigned int dummy[4];
  getCameraDomainBounds(dummy);
  getCameraDomainBoundsSI();
}

void FLB::OrthographicCameraController::getCameraDomainBounds(unsigned int *cameraBounds)
{
  if (m_UpdateCameraDomainCornersNodes)
  {
    float xMinViewCamera = - m_AspectRatio * m_ZoomLevel;
    //float xMaxViewCamera = - xMinViewCamera;

    float xDistanceDomain = m_ViewLowerRightCornerDomain.x - m_ViewUpperLeftCornerDomain.x;
    float yDistanceDomain = m_ViewUpperLeftCornerDomain.y - m_ViewLowerRightCornerDomain.y;
    
    m_CameraDomainCornersNodes[0] = m_ViewUpperLeftCornerDomain.x > xMinViewCamera ? 0 : m_NxIntervalsDomain * (xMinViewCamera - m_ViewUpperLeftCornerDomain.x) / xDistanceDomain;
    m_CameraDomainCornersNodes[1] = m_ViewLowerRightCornerDomain.x > (-xMinViewCamera) ? m_NxIntervalsDomain * (-xMinViewCamera - m_ViewUpperLeftCornerDomain.x) / xDistanceDomain + 1 : m_NxIntervalsDomain;
    
    // TODO For CUDA because the origin is in the upper left corner
    m_CameraDomainCornersNodes[2] = m_ViewUpperLeftCornerDomain.y > m_ZoomLevel ? m_NyIntervalsDomain * (m_ViewUpperLeftCornerDomain.y - m_ZoomLevel) / yDistanceDomain : 0;
    m_CameraDomainCornersNodes[3] = m_ViewLowerRightCornerDomain.y > (-m_ZoomLevel) ? m_NyIntervalsDomain : m_NyIntervalsDomain * (m_ViewUpperLeftCornerDomain.y + m_ZoomLevel) / yDistanceDomain + 1;
    m_UpdateCameraDomainCornersNodes = false;	
  }
  cameraBounds[0] = m_CameraDomainCornersNodes[0];
  cameraBounds[1] = m_CameraDomainCornersNodes[1];
  cameraBounds[2] = m_CameraDomainCornersNodes[2];
  cameraBounds[3] = m_CameraDomainCornersNodes[3];
}

std::array<float, 8>& FLB::OrthographicCameraController::getCameraDomainBoundsSI()
{
  if (!m_UpdateCameraDomainCornersSI) return m_CameraDomainCornersSI;
  float xMaxViewCamera = m_AspectRatio * m_ZoomLevel;
    
  float xDistanceDomainNode = (m_ViewLowerRightCornerDomain.x - m_ViewUpperLeftCornerDomain.x) / m_NxIntervalsDomain;
  float yDistanceDomainNode = (m_ViewUpperLeftCornerDomain.y - m_ViewLowerRightCornerDomain.y) / m_NyIntervalsDomain;
  
  float xMin = m_ViewUpperLeftCornerDomain.x > (-xMaxViewCamera) ? m_UpperLeftCornerDomain.x : m_UpperLeftCornerDomain.x + xDistanceDomainNode * std::floor((-xMaxViewCamera - m_ViewUpperLeftCornerDomain.x) / xDistanceDomainNode); // sum of the count of distances beetween nodes

  float xMax = m_ViewLowerRightCornerDomain.x > xMaxViewCamera ? m_LowerRightCornerDomain.x - xDistanceDomainNode * std::floor((m_ViewLowerRightCornerDomain.x - xMaxViewCamera) / xDistanceDomainNode) : m_LowerRightCornerDomain.x;
  
  float yMin = m_ViewLowerRightCornerDomain.y > (-m_ZoomLevel) ? m_LowerRightCornerDomain.y : m_LowerRightCornerDomain.y + yDistanceDomainNode * std::floor((-m_ZoomLevel - m_ViewLowerRightCornerDomain.y) / yDistanceDomainNode);
  float yMax = m_ViewUpperLeftCornerDomain.y > m_ZoomLevel ? m_UpperLeftCornerDomain.y - yDistanceDomainNode * std::floor((m_ViewUpperLeftCornerDomain.y - m_ZoomLevel) / yDistanceDomainNode) : m_UpperLeftCornerDomain.y;

  m_CameraDomainCornersSI[0] = xMin; m_CameraDomainCornersSI[1] = yMin;
  m_CameraDomainCornersSI[2] = xMax; m_CameraDomainCornersSI[3] = yMin;
  m_CameraDomainCornersSI[4] = xMax; m_CameraDomainCornersSI[5] = yMax;
  m_CameraDomainCornersSI[6] = xMin; m_CameraDomainCornersSI[7] = yMax;

  m_UpdateCameraDomainCornersSI = false;

  return m_CameraDomainCornersSI;
}

void FLB::OrthographicCameraController::recalculateDomainCoordViewSpace()
{
  // Transform coordinates of the domain to view space
  m_ViewUpperLeftCornerDomain = m_Camera.m_ViewMatrix * m_UpperLeftCornerDomain;
  m_ViewLowerRightCornerDomain = m_Camera.m_ViewMatrix * m_LowerRightCornerDomain;
}
