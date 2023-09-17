target_sources(ui
  PUBLIC  
    ${CMAKE_CURRENT_LIST_DIR}/app.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/app.h)

target_include_directories(ui 
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/dependencies/fltk
    ${CMAKE_SOURCE_DIR}/dependencies/glm
    ${CMAKE_SOURCE_DIR}/dependencies/glfw_3_3_8/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
