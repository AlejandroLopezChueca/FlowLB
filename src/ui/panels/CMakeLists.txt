
#target_link_libraries(panels PRIVATE CUDA::cudart_static)

target_sources(panels
  PUBLIC  
    ${CMAKE_CURRENT_LIST_DIR}/addPanel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchyPanel.cpp
    #${CMAKE_CURRENT_LIST_DIR}/logPanel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/settingsPanel.cpp
    #${CMAKE_CURRENT_LIST_DIR}/contentBrowserPanel.cpp

  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/addPanel.h
    ${CMAKE_CURRENT_LIST_DIR}/hierarchyPanel.h
    #${CMAKE_CURRENT_LIST_DIR}/logPanel.h
    ${CMAKE_CURRENT_LIST_DIR}/settingsPanel.h
    #${CMAKE_CURRENT_LIST_DIR}/contentBrowserPanel.h
    )

target_include_directories(panels 
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/dependencies/fltk
    ${CMAKE_SOURCE_DIR}/dependencies/glm
    ${CMAKE_SOURCE_DIR}/dependencies/glad/include
    ${CMAKE_SOURCE_DIR}/dependencies/glfw_3_3_8/include
    ${CMAKE_SOURCE_DIR}/dependencies/entt/include
    ${CMAKE_SOURCE_DIR}/dependencies/imgui
    ${CMAKE_SOURCE_DIR}/dependencies/imgui/backends
    ${CMAKE_SOURCE_DIR}/dependencies/tinyfiledialogs
    ${CMAKE_SOURCE_DIR}/dependencies/msdf-atlas-gen/msdfgen
    ${CMAKE_SOURCE_DIR}/dependencies/msdf-atlas-gen/msdf-atlas-gen
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
