#include "cameraController.h"
#include "GLFW/glfw3.h"

#include <algorithm>
#include <iostream>
#include <utility>

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
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) mousePan(delta);
    m_Camera.recalculateViewMatrix();
  }

  else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {m_Camera.m_Rotation += m_CameraRotationSpeed; m_Camera.recalculateViewMatrix();}

  else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {m_Camera.m_Rotation -= m_CameraRotationSpeed; m_Camera.recalculateViewMatrix();}
}

void FLB::OrthographicCameraController::onMouseScrolled(float xOffset, float yOffset)
{
  m_ZoomLevel -= yOffset * 0.1f;
  m_ZoomLevel = std::max(m_ZoomLevel, 0.1f);
  m_Camera.setProjection(-m_AspectRatio * m_ZoomLevel, m_AspectRatio * m_ZoomLevel, -m_ZoomLevel, m_ZoomLevel);
  m_CameraTranslationSpeed = 0.1f * m_ZoomLevel;
}

void FLB::OrthographicCameraController::mousePan(const glm::vec2& delta)
{
  auto [xSpeed, ySpeed] = getPanSpeed();
  m_Camera.m_Position += - m_Camera.getRightDirection() * xSpeed * delta.x;
  m_Camera.m_Position += m_Camera.getUpDirection() * ySpeed * delta.y;
}

void FLB::OrthographicCameraController::setViewportSize(float width, float height)
{
  m_ViewportWidth = width;
  m_ViewportHeight = height;
  m_AspectRatio = m_ViewportWidth / m_ViewportHeight;
  m_Camera.setProjection(-m_AspectRatio * m_ZoomLevel, m_AspectRatio * m_ZoomLevel, -m_ZoomLevel, m_ZoomLevel);
}

std::pair<float, float> FLB::OrthographicCameraController::getPanSpeed() const
{
  float x = std::min(m_ViewportWidth / 1000.0f, 2.4f); //  max of 2.4
  float xFactor = 0.036f * (x * x) - 0.1778f * x + 0.32021;
  
  float y = std::min(m_ViewportHeight / 1000.0f, 2.4f); //  max of 2.4
  float yFactor = 0.036f * (y * y) - 0.1778f * y + 0.32021;
  return {xFactor, yFactor};

}
