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
bool FLB::OpenGLWindow<T>::s_GLFWInitialized = false;

template<typename T>
T* FLB::OpenGLWindow<T>::s_cameraController = nullptr;

template<typename T>
int* FLB::OpenGLWindow<T>::s_GizmoOperation = nullptr;

template<typename T>
FLB::OpenGLWindow<T>::OpenGLWindow(Fl_Simple_Terminal* terminal, T* cameraController, bool enableVSync)
{
  s_cameraController = cameraController;
  if (!s_GLFWInitialized)
  {
    glfwInit();
    s_GLFWInitialized = true;
  }
  // Min version 4.5
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  m_renderWindow = glfwCreateWindow(m_Width, m_Height, title.c_str(), nullptr, nullptr);
  if (!m_renderWindow)
  {
    terminal -> printf("Window or context creation failed for OpenGL\n");
    throw std::runtime_error("Error in the creation of the window for OpenGL");
  }
  glfwMakeContextCurrent(m_renderWindow);
  //OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    terminal -> printf("Failed to initialize GLAD\n");
    if (GLVersion.major < 4 || (GLVersion.major == 4 && GLVersion.minor < 5)) terminal -> printf("It is required al least OpenGL version 4.5\n");
    throw std::runtime_error("Failed to initialize GLAD");
  }
  glViewport(0, 0, m_Width, m_Height);
  //glfwSetWindowUserPointer(renderWindow, &windowData);
  setVSync(enableVSync); 

  // Set GLFW callbacks
  glfwSetFramebufferSizeCallback(m_renderWindow, [](GLFWwindow* window, int width, int height) {glViewport(0, 0, width, height);});

  glfwSetWindowCloseCallback(m_renderWindow, [](GLFWwindow* window)
      {
	FLB::App::closeGraphics();
      });

  /*glfwSetMouseButtonCallback(m_renderWindow, [](GLFWwindow* window, int button, int action, int mods)
      {
	if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) std::cout<<"BUTTON"<<std::endl;
      });*/

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
}

template<typename T>
FLB::OpenGLWindow<T>::~OpenGLWindow()
{
  glfwDestroyWindow(m_renderWindow);
  glfwTerminate();
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
  // swap front and bahk buffers
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

template class FLB::OpenGLWindow<FLB::OrthographicCameraController>; 
