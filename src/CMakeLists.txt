
target_sources(main_src
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/runCalculations.cpp
  ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/runCalculations.h
  ${CMAKE_CURRENT_LIST_DIR}/utils.h)

target_include_directories(main_src PUBLIC ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_SOURCE_DIR}/src/io
  ${CMAKE_SOURCE_DIR}/src/cuda
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

