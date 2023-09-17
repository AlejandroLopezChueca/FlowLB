target_sources(geometry
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/shapes.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/shapes.h
    ${CMAKE_CURRENT_LIST_DIR}/mesh.h
  )

target_include_directories(geometry 
  PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/dependencies/fltk)


