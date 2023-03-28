cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project("FLowLB" VERSION 1.0 LANGUAGES CXX CUDA)


###################################################################################################################
# MAIN
file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

#******************************************************************************************************************
# EXTERNAL LIBRARIES
#******************************************************************************************************************

###################################################################################################################
#GLFW

#add_subdirectory(libs/glfw_3_3_8)
#include_directories(libs/glfw_3_3_8/include)
#target_link_libraries(${PROJECT_NAME} PRIVATE glfw ${GLFW_LIBRARIES})



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


#add_library(cuda_fluid STATIC
#  src/cuda/lbm_kernel.cu
#  src/cuda/lbm_kernel.cuh
#  )

#set_target_properties(cuda_fluid PROPERTIES CUDA_ARCHITECTURES native)
#target_link_libraries(${PROJECT_NAME} PRIVATE cuda_fluid)

