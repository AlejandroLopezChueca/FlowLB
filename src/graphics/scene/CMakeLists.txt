
target_sources(scene
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/scene.cpp
  ${CMAKE_CURRENT_LIST_DIR}/entity.cpp
  ${CMAKE_CURRENT_LIST_DIR}/componentsMesh.cpp
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/scene.h
  ${CMAKE_CURRENT_LIST_DIR}/entity.h
  ${CMAKE_CURRENT_LIST_DIR}/componentsMesh.h
  ${CMAKE_CURRENT_LIST_DIR}/components.h)

  target_include_directories(scene 
    PUBLIC
      ${CMAKE_CURRENT_LIST_DIR}
      ${CMAKE_SOURCE_DIR}/dependencies/glad/include
      ${CMAKE_SOURCE_DIR}/dependencies/glfw_3_3_8/include
      ${CMAKE_SOURCE_DIR}/dependencies/fltk
      ${CMAKE_SOURCE_DIR}/dependencies/glm
      ${CMAKE_SOURCE_DIR}/dependencies/entt/include
      ${CMAKE_SOURCE_DIR}/dependencies/msdf-atlas-gen/msdfgen
      ${CMAKE_SOURCE_DIR}/dependencies/msdf-atlas-gen/msdf-atlas-gen
      ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

