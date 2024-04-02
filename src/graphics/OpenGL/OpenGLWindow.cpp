#include <cstdio>
#include <glad/glad.h>
#include "GLFW/glfw3.h"
#include "OpenGLWindow.h"

#include "graphics/renderer.h"
#include "ui/app.h"

#include <iostream>
#include <imgui.h>
#include <ImGuizmo.h>

template<typename T>
T* FLB::OpenGLWindow<T>::s_cameraController = nullptr;

template<typename T>
int* FLB::OpenGLWindow<T>::s_GizmoOperation = nullptr;

template<typename T>
FLB::OpenGLWindow<T>::OpenGLWindow(Fl_Simple_Terminal* terminal, T* cameraController, bool enableVSync)
{
  s_cameraController = cameraController;
  if (!glfwInit())
  {
    terminal -> printf("Failed to initialize GLFW\n");
    glfwTerminate();
    return;
  }
  // Min version 4.5
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  m_renderWindow = glfwCreateWindow(m_Width, m_Height, title.c_str(), nullptr, nullptr);
  if (!m_renderWindow)
  {
    terminal -> printf("Window or context creation failed for OpenGL\n");
    //throw std::runtime_error("Error in the creation of the window for OpenGL");
  }
  glfwMakeContextCurrent(m_renderWindow);
  //OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    terminal -> printf("Failed to initialize GLAD\n");
    if (GLVersion.major < 4 || (GLVersion.major == 4 && GLVersion.minor < 5)) terminal -> printf("It is required al least OpenGL version 4.5\n");
    glfwDestroyWindow(m_renderWindow);
    glfwTerminate();
    return;
  }
  glViewport(0, 0, m_Width, m_Height);
  setVSync(enableVSync); 
  //glEnable(GL_DEPTH_TEST)

  // Set GLFW callbacks
  glfwSetFramebufferSizeCallback(m_renderWindow, [](GLFWwindow* window, int width, int height) {glViewport(0, 0, width, height);});

  glfwSetWindowCloseCallback(m_renderWindow, [](GLFWwindow* window)
      {
	FLB::App::closeGraphics();
      });

  glfwSetKeyCallback(m_renderWindow, FLB::OpenGLWindow<T>::setKeyCallback);
  /*glfwSetKeyCallback(m_renderWindow, [](GLFWwindow* window, int key, int scancode, int action, int mods)

      {
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) m_pauseRendering = !m_pauseRendering;
      });*/

  glfwSetScrollCallback(m_renderWindow, FLB::OpenGLWindow<T>::setScrollCallback);

#ifdef DEBUG_MODE
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(debugMessageCallback, 0);
#endif 
  m_IsInitialized = true;
}

template<typename T>
FLB::OpenGLWindow<T>::~OpenGLWindow()
{
  glfwDestroyWindow(m_renderWindow);
  glfwTerminate();
  m_IsInitialized = false;
}

template<typename T>
void FLB::OpenGLWindow<T>::setVSync(bool enabled)
{
  if (enabled) glfwSwapInterval(1);
  else glfwSwapInterval(0);
}

template<typename T>
void FLB::OpenGLWindow<T>::update()
{
  // Pool for and process events
  glfwPollEvents();
  // swap front and back buffers
  glfwSwapBuffers(m_renderWindow);
}

template<typename T>
void FLB::OpenGLWindow<T>::setKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) FLB::Renderer::s_UpdateRender = !FLB::Renderer::s_UpdateRender;
  // Gizmos in render Layer
  else if (key == GLFW_KEY_Q && GLFW_PRESS && !ImGuizmo::IsUsing()) *s_GizmoOperation = -1;
  else if (key == GLFW_KEY_W && GLFW_PRESS && !ImGuizmo::IsUsing()) *s_GizmoOperation = ImGuizmo::OPERATION::TRANSLATE;
  else if (key == GLFW_KEY_E && GLFW_PRESS && !ImGuizmo::IsUsing()) *s_GizmoOperation = ImGuizmo::OPERATION::ROTATE;
  else if (key == GLFW_KEY_R && GLFW_PRESS && !ImGuizmo::IsUsing()) *s_GizmoOperation =ImGuizmo::OPERATION::SCALE;
}

template<typename T>
void FLB::OpenGLWindow<T>::setScrollCallback(GLFWwindow* window, double xOffset, double yOffset)
{
  s_cameraController -> onMouseScrolled(xOffset, yOffset);
}

template<typename T>
void FLB::OpenGLWindow<T>::debugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
  if (type == GL_DEBUG_TYPE_ERROR) fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n", (type == GL_DEBUG_TYPE_ERROR ? "** GL_ERROR **" : ""), type, severity, message);
}

template<typename T>
bool FLB::OpenGLWindow<T>::getMousePos(glm::dvec2& mousePos) const
{
  bool state = glfwGetMouseButton(m_renderWindow, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  if (state) glfwGetCursorPos(m_renderWindow, &mousePos.x, &mousePos.y);
  return state;
}

template <typename T>
bool FLB::OpenGLWindow<T>::isLeftButtonMouseClicked() const
{
  return glfwGetMouseButton(m_renderWindow, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
}

template class FLB::OpenGLWindow<FLB::OrthographicCameraController>; 
