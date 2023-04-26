
target_sources(graphics
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/renderer2D.cpp
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/renderer2D.h)

  target_include_directories(graphics PUBLIC ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_SOURCE_DIR}/src/io)

