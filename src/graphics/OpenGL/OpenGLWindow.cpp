#include <glad/glad.h>
#include "GLFW/glfw3.h"
#include "OpenGLWindow.h"

#include "graphics/renderer.h"

#include <iostream>

template<typename T>
bool FLB::OpenGLWindow<T>::s_GLFWInitialized = false;

template<typename T>
T* FLB::OpenGLWindow<T>::s_cameraController = nullptr;

template<typename T>
FLB::OpenGLWindow<T>::OpenGLWindow(Fl_Simple_Terminal* terminal, T* cameraController)
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
  setVSync(true); 

  // Set GLFW callbacks
  glfwSetFramebufferSizeCallback(m_renderWindow, [](GLFWwindow* window, int width, int height) {glViewport(0, 0, width, height);});

  glfwSetWindowCloseCallback(m_renderWindow, [](GLFWwindow* window)
      {
	//WindowData& windowData = *(WindowData*)glfwGetWindowUserPointer(window);
	//windowData.isRendering = false;
	glfwSetWindowShouldClose(window,true);
      });

  glfwSetMouseButtonCallback(m_renderWindow, [](GLFWwindow* window, int button, int action, int mods)
      {
	if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) std::cout<<"BUTTON"<<std::endl;
      });

  glfwSetKeyCallback(m_renderWindow, FLB::OpenGLWindow<T>::setKeyCallback);
  /*glfwSetKeyCallback(m_renderWindow, [](GLFWwindow* window, int key, int scancode, int action, int mods)

      {
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) m_pauseRendering = !m_pauseRendering;
      });*/

  glfwSetScrollCallback(m_renderWindow, FLB::OpenGLWindow<T>::setScrollCallback);
  
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
  else if (FLB::Renderer::s_VelocityPointMode && key == GLFW_KEY_KP_ADD && (action == GLFW_PRESS || action == GLFW_REPEAT)) FLB::Renderer::s_PointSize += 0.2f;
  else if (FLB::Renderer::s_VelocityPointMode && key == GLFW_KEY_KP_SUBTRACT && (action == GLFW_PRESS || action == GLFW_REPEAT)) FLB::Renderer::s_PointSize -= 0.2f;
  else if (key == GLFW_KEY_1 && action == GLFW_PRESS)
  {
    //if (FLB::Renderer::s_VelocityPointMode) return;
    FLB::Renderer::s_VelocityPointMode = true;
    FLB::Renderer::s_PointSize = 2.0f;
    FLB::Renderer::s_shaderToUse = FLB::Renderer::s_shaderPointsVelocity;
    //FLB::Renderer::s_VelocityTextureMode = false;
  }
  else if (key == GLFW_KEY_2 && action == GLFW_PRESS)
  {
    FLB::Renderer::s_VelocityPointMode = false;
    FLB::Renderer::s_shaderToUse = FLB::Renderer::s_shaderTextureVelocity2D;
    FLB::Renderer::s_TextureToUse = FLB::Renderer::s_TextureVelocity;
    //FLB::Renderer::s_VelocityTextureMode = true;
  }
}


template<typename T>
void FLB::OpenGLWindow<T>::setScrollCallback(GLFWwindow* window, double xOffset, double yOffset)
{
  s_cameraController -> onMouseScrolled(xOffset, yOffset);
}

template class FLB::OpenGLWindow<FLB::OrthographicCameraController>; 
