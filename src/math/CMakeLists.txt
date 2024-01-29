target_sources(math
  PUBLIC  
    ${CMAKE_CURRENT_LIST_DIR}/math.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/math.h)

target_include_directories(math 
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/dependencies/glm
    ${CMAKE_SOURCE_DIR}/dependencies/fltk
    ${CMAKE_SOURCE_DIR}/src)
  #${CMAKE_SOURCE_DIR}/dependencies/glfw_3_3_8/include)
    #${CMAKE_SOURCE_DIR}/dependencies/msdf-atlas-gen/msdfgen
    #${CMAKE_SOURCE_DIR}/dependencies/msdf-atlas-gen/msdf-atlas-gen)
