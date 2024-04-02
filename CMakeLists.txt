cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "Vcpkg toolchain file")

project("FlowLB" VERSION 1.0 LANGUAGES CXX C CUDA)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  add_compile_definitions(__OS_LINUX__)
elseif ((${CMAKE_SYSTEM_NAME} STREQUAL "Windows"))
  add_compile_definitions(__OS_WINDOWS__)
else()
  message("The Operating System is not found")
endif()

set(USE_DEBUG ON CACHE BOOL "Debug mode" FORCE)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Debug mode" FORCE)

if(USE_DEBUG)
  add_definitions(-DDEBUG_MODE)
endif()

unset(USE_DEBUG CACHE)

###################################################################################################################
# MAIN
add_executable(${PROJECT_NAME} src/main.cpp)

#******************************************************************************************************************
# EXTERNAL LIBRARIES
#******************************************************************************************************************

###################################################################################################################
#FLTK
add_subdirectory(dependencies/fltk)
target_link_libraries(${PROJECT_NAME} PUBLIC fltk)

###################################################################################################################
#GLFW

add_subdirectory(dependencies/glfw_3_3_8)
#Not build the GLFW tests, examples and documentation
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

#target_include_directories(${PROJECT_NAME} PUBLIC dependencies/glfw_3_3_8/include)
target_link_libraries(${PROJECT_NAME} PUBLIC glfw)

###################################################################################################################
#GLAD
add_library(s_glad)
target_sources(s_glad
  PUBLIC
    dependencies/glad/src/glad.c
    dependencies/glad/include/glad/glad.h)

target_include_directories(s_glad PUBLIC dependencies/glad/include)
target_link_libraries(${PROJECT_NAME} PUBLIC s_glad)

###################################################################################################################
#GLM
add_subdirectory(dependencies/glm)
target_link_libraries(${PROJECT_NAME} PUBLIC glm)

###################################################################################################################
#MSDF-ATLAS-GEM and MSDF

set(MSDF_ATLAS_BUILD_STANDALONE OFF CACHE BOOL "Build the msdf-atlas-gen standalone executable" FORCE)
#set(MSDF_ATLAS_USE_VCPKG OFF CACHE BOOL "Use vcpkg package manager to link project dependencies" FORCE)
set(MSDF_ATLAS_USE_SKIA OFF CACHE BOOL "Build with the Skia library" FORCE)
#set(MSDF_ATLAS_MSDFGEN_EXTERNAL OFF CACHE BOOL "Do not build the msdfgen submodule but find it as an external package" FORCE)

set(MSDFGEN_BUILD_STANDALONE OFF CACHE BOOL "Build the msdfgen standalone executable" FORCE)
set(MSDFGEN_USE_SKIA OFF CACHE BOOL "Build with the Skia library" FORCE)

# If it not required to lad svg files and saving PNG images
set(MSDFGEN_DISABLE_SVG ON CACHE BOOL "" FORCE)
set(MSDFGEN_DISABLE_PNG OFF CACHE BOOL "" FORCE)

add_subdirectory(dependencies/msdf-atlas-gen)

target_link_libraries(${PROJECT_NAME} PUBLIC msdf-atlas-gen)

###################################################################################################################
#Lean-VTK
#add_subdirectory(dependencies/lean-vtk)
#target_link_libraries(${PROJECT_NAME} PUBLIC LeanVTK)

###################################################################################################################
#CUDA libraries

find_package(CUDAToolkit)

# We need to explicitly state that we need all CUDA files 
# library to be built with -dc as the member functions could be called by
# other libraries and executables

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)

###################################################################################################################
#ImGui
add_library(s_ImGui)
target_sources(s_ImGui
  PUBLIC
    dependencies/imgui/imgui.cpp
    dependencies/imgui/imgui_demo.cpp
    dependencies/imgui/imgui_draw.cpp
    dependencies/imgui/imgui_tables.cpp
    dependencies/imgui/imgui_widgets.cpp
    #backends
    dependencies/imgui/backends/imgui_impl_glfw.cpp
    dependencies/imgui/backends/imgui_impl_opengl3.cpp)

  target_include_directories(s_ImGui
    PUBLIC
      dependencies/imgui
      dependencies/imgui/backends
      dependencies/glfw_3_3_8/include)

target_link_libraries(${PROJECT_NAME} PUBLIC s_ImGui)

###################################################################################################################
#Tiny File Dialogs
add_library(s_tiny)
target_sources(s_tiny
  PUBLIC
    dependencies/tinyfiledialogs/tinyfiledialogs.c
    dependencies/tinyfiledialogs/tinyfiledialogs.h)

target_include_directories(s_tiny PUBLIC dependencies/tinyfiledialogs)
target_link_libraries(${PROJECT_NAME} PUBLIC s_tiny)

###################################################################################################################
#STB
add_library(s_stb)
target_sources(s_stb
  PUBLIC
    dependencies/stb/stb_image.cpp
    dependencies/stb/stb_image_write.cpp
    dependencies/stb/stb_image.h
    dependencies/stb/stb_image_write.h)

target_include_directories(s_glad PUBLIC dependencies/stb)
target_link_libraries(${PROJECT_NAME} PUBLIC s_stb)

###################################################################################################################
#ImGuizmo
add_library(s_ImGuizmo)
target_sources(s_ImGuizmo
  PUBLIC
    dependencies/ImGuizmo/ImGuizmo.cpp
  PUBLIC
    dependencies/ImGuizmo/ImGuizmo.h
    )

  target_include_directories(s_ImGuizmo
    PUBLIC
      dependencies/imgui
      dependencies/ImGuizmo)

target_link_libraries(${PROJECT_NAME} PUBLIC s_ImGuizmo)

###################################################################################################################
#Assimp

#add_subdirectory(dependencies/assimp)

#set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
#set(ASSIMP_SHARED_LIBS OFF CACHE BOOL "" FORCE)
#set(ASSIMP_DOUBLE_PRECISION ON CACHE BOOL "" FORCE)

#target_link_libraries(${PROJECT_NAME} PUBLIC Assimp)

#******************************************************************************************************************
# INTERNAL LIBRARIES
#******************************************************************************************************************

add_subdirectory(src/cuda)
add_library(ui)
add_subdirectory(src/ui)
add_library(panels)
add_subdirectory(src/ui/panels)
add_library(io)
add_subdirectory(src/io)
#add_library(cuda_fluid)
add_library(main_src)
add_subdirectory(src)
add_library(graphics)
add_subdirectory(src/graphics)
add_library(scene)
add_subdirectory(src/graphics/scene)
add_library(OpenGL_graphics)
add_subdirectory(src/graphics/OpenGL)
add_library(geometry)
add_subdirectory(src/geometry)
add_library(math)
add_subdirectory(src/math)

include_directories(src)

target_link_libraries(${PROJECT_NAME} PUBLIC
  math
  ui
  panels
  io 
  geometry
  #cuda_fluid
  scene
  graphics
  OpenGL_graphics
  main_src)


#******************************************************************************************************************
# COPY ASSETS
#******************************************************************************************************************

add_custom_target(copy_assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets)

add_dependencies(${PROJECT_NAME} copy_assets)

add_custom_target(copy_resources
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/resources ${CMAKE_BINARY_DIR}/resources)

add_dependencies(${PROJECT_NAME} copy_resources)
