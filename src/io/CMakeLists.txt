
target_sources(io
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/VTUWriter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reader.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/VTUWriter.h
    ${CMAKE_CURRENT_LIST_DIR}/reader.h)

  target_include_directories(io PUBLIC ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_SOURCE_DIR}/src/io)


