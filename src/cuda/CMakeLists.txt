
target_sources(${PROJECT_NAME} #cuda_fluid
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/lbmKernel2D.cu
    ${CMAKE_CURRENT_LIST_DIR}/cudaInitData.cu
    ${CMAKE_CURRENT_LIST_DIR}/graphics.cu 
    ${CMAKE_CURRENT_LIST_DIR}/runCuda.cu
    ${CMAKE_CURRENT_LIST_DIR}/cudaUtils.cu
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/lbmKernel2D.cuh
    ${CMAKE_CURRENT_LIST_DIR}/cudaInitData.cuh
    ${CMAKE_CURRENT_LIST_DIR}/graphics.cuh 
    ${CMAKE_CURRENT_LIST_DIR}/runCuda.cuh
    ${CMAKE_CURRENT_LIST_DIR}/cellCalculations.cuh
    ${CMAKE_CURRENT_LIST_DIR}/cudaUtils.cuh)

  #include_directories(${CMAKE_SOURCE_DIR}/src)
target_include_directories( ${PROJECT_NAME} #cuda_fluid 
  PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/dependencies/glad/include
    ${CMAKE_SOURCE_DIR}/dependencies/glm
    ${CMAKE_SOURCE_DIR}/dependencies/entt/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/dependencies/fltk
    ${CMAKE_SOURCE_DIR}/src)

  #set_target_properties(cuda_fluid PROPERTIES CUDA_ARCHITECTURES native)

# We need to explicitly state that we need all CUDA files 
# library to be built with -dc as the member functions could be called by
# other libraries and executables
#set_target_properties(cuda_fluid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(cuda_fluid PRIVATE CUDA::cudart_static)
