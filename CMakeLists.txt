cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project("FLowLB" VERSION 1.0 LANGUAGES C CXX CUDA)

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
#GLFW

#Not build the GLFW tests, examples and documentation
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(dependencies/glfw_3_3_8)
target_include_directories(${PROJECT_NAME} PUBLIC dependencies/glfw_3_3_8/include)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw ${GLFW_LIBRARIES})

###################################################################################################################
#GLAD
add_library(s_glad)
target_sources(s_glad
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/dependencies/glad/src/glad.c
  ${CMAKE_CURRENT_LIST_DIR}/dependencies/glad/include/glad/glad.h)

target_include_directories(s_glad PUBLIC dependencies/glad/include)
target_link_libraries(${PROJECT_NAME} PRIVATE s_glad)

###################################################################################################################
#Lean-VTK
add_subdirectory(dependencies/lean-vtk)
target_link_libraries(${PROJECT_NAME} PRIVATE LeanVTK)

###################################################################################################################
#CUDA libraries

find_package(CUDAToolkit)

# We need to explicitly state that we need all CUDA files 
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)

#******************************************************************************************************************
# INTERNAL LIBRARIES
#******************************************************************************************************************

add_library(io)
add_subdirectory(src/io)
add_library(cuda_fluid)
add_subdirectory(src/cuda)
add_library(main_src)
add_subdirectory(src)
add_library(graphics)
add_subdirectory(src/graphics)
add_library(OpenGL_graphics)
add_subdirectory(src/graphics/OpenGL)
#target_include_directories(${PROJECT_NAME} PUBLIC src src/io)
target_link_libraries(${PROJECT_NAME} PUBLIC 
  io 
  cuda_fluid 
  graphics
  OpenGL_graphics
  main_src)
#add_library(cuda_fluid STATIC
#  src/cuda/lbm_kernel.cu
#  src/cuda/lbm_kernel.cuh
#  )

#set_target_properties(cuda_fluid PROPERTIES CUDA_ARCHITECTURES native)
#target_link_libraries(${PROJECT_NAME} PRIVATE cuda_fluid)
