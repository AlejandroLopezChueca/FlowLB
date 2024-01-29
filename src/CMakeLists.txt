
target_sources(main_src
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/initData.cpp
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/utils.h
  ${CMAKE_CURRENT_LIST_DIR}/initData.h)

target_include_directories(main_src 
  PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/dependencies/fltk
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

