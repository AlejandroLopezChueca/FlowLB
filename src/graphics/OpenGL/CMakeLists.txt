
target_sources(OpenGL_graphics
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/OpenGLRenderer.cpp
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/OpenGLRenderer.h)

  target_include_directories(OpenGL_graphics PUBLIC ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_SOURCE_DIR}/src/io)
