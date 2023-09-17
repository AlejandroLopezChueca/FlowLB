cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project("FlowLB" VERSION 1.0 LANGUAGES CUDA CXX C)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  add_compile_definitions(__OS_LINUX__)
elseif ((${CMAKE_SYSTEM_NAME} STREQUAL "Windows"))
  add_compile_definitions(__OS_WINDOWS__)
else()
  message("The Operating System is not found")
endif()

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

###################################################################
################################################

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

#******************************************************************************************************************
# INTERNAL LIBRARIES
#******************************************************************************************************************

add_library(ui)
add_subdirectory(src/ui)
add_library(io)
add_subdirectory(src/io)
#add_library(cuda_fluid)
add_subdirectory(src/cuda)
add_library(main_src)
add_subdirectory(src)
add_library(graphics)
add_subdirectory(src/graphics)
add_library(OpenGL_graphics)
add_subdirectory(src/graphics/OpenGL)
add_library(geometry)
add_subdirectory(src/geometry)

include_directories(src)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ui
  io 
  geometry
  #cuda_fluid 
  graphics
  OpenGL_graphics
  main_src)


#******************************************************************************************************************
# COPY ASSETS
#******************************************************************************************************************

add_custom_target(copy_assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets)

add_dependencies(${PROJECT_NAME} copy_assets)
