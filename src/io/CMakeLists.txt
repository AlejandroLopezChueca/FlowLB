
target_sources(io
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/writer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvReader.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/writer.h
    ${CMAKE_CURRENT_LIST_DIR}/reader.h
    ${CMAKE_CURRENT_LIST_DIR}/csvReader.h)

target_include_directories(io
  PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
    #${CMAKE_SOURCE_DIR}/dependencies/lean-vtk/include
    ${CMAKE_SOURCE_DIR}/dependencies/fltk
    ${CMAKE_SOURCE_DIR}/dependencies/glm
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#target_link_libraries(io PRIVATE LeanVTK)


