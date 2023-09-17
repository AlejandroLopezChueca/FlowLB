#include "cameraController.h"
#include "GLFW/glfw3.h"

#include <algorithm>
#include <iostream>

FLB::OrthographicCameraController::OrthographicCameraController(float width, float height)
  : m_Width(width), m_Height(height), m_AspectRatio(width/height), m_Camera(-m_AspectRatio * m_ZoomLevel, m_AspectRatio * m_ZoomLevel, -m_ZoomLevel, m_ZoomLevel)
{
}

void FLB::OrthographicCameraController::GLFWUpdate(GLFWwindow* window)
{

  if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {m_CameraPosition.x -= m_CameraTranslationSpeed;} 
  else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {m_CameraPosition.x += m_CameraTranslationSpeed;}
  if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {m_CameraPosition.y += m_CameraTranslationSpeed;}
  else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {m_CameraPosition.y -= m_CameraTranslationSpeed;}
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {m_CameraRotation += m_CameraRotationSpeed;}
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {m_CameraRotation -= m_CameraRotationSpeed;}
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
  {
    double mouseX;
    double mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    m_CameraPosition.x = - (float)((mouseX - m_Width/2)/(m_Width/2));
    m_CameraPosition.y = (float)((mouseY - m_Height/2)/(m_Height/2));
    //std::cout<<m_CameraPosition.x<<" , "<<m_CameraPosition.y<<std::endl;
  }
  else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
  {
  }
  m_Camera.setPosition(m_CameraPosition);
  m_Camera.setRotation(m_CameraRotation);
}

void FLB::OrthographicCameraController::onMouseScrolled(float xOffset, float yOffset)
{
  m_ZoomLevel -= yOffset * 0.1f;
  m_ZoomLevel = std::max(m_ZoomLevel, 0.1f);
  m_Camera.setProjection(-m_AspectRatio * m_ZoomLevel, m_AspectRatio * m_ZoomLevel, -m_ZoomLevel, m_ZoomLevel);
  m_CameraTranslationSpeed = 0.1f * m_ZoomLevel;
}
