target_sources(cuda_fluid
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/lbmKernel2D.cu
    ${CMAKE_CURRENT_LIST_DIR}/initData.cu
    ${CMAKE_CURRENT_LIST_DIR}/graphics.cu
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/lbmKernel2D.cuh
    ${CMAKE_CURRENT_LIST_DIR}/initData.cuh
    ${CMAKE_CURRENT_LIST_DIR}/graphics.cuh
    ${CMAKE_CURRENT_LIST_DIR}/cudaUtils.h)

target_include_directories(cuda_fluid PUBLIC ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_SOURCE_DIR}/src/io)

set_target_properties(cuda_fluid PROPERTIES CUDA_ARCHITECTURES native)

