if(MNN_OPENGL)
  FILE(GLOB_RECURSE MNN_OpenGL_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp"
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_LIST_DIR}/makeshader.py"
    "${CMAKE_CURRENT_LIST_DIR}/glsl/"
    "${CMAKE_CURRENT_LIST_DIR}/AllShader.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp"
    COMMENT "OpenGL Code Generation"
  )
  add_custom_target (MNNOpenGLCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp")

  add_library(MNNOpenGL OBJECT ${MNN_OpenGL_SRC} "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp")
  list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNOpenGL>)
  list(APPEND MNN_TARGETS MNNOpenGL)
  SET(MNN_OBJECTS_TO_LINK "${MNN_OBJECTS_TO_LINK}" PARENT_SCOPE)
  SET(MNN_TARGETS "${MNN_TARGETS}" PARENT_SCOPE)
  target_include_directories(MNNOpenGL PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)

  IF(MNN_OPENGL_REGEN)
    add_dependencies(MNNOpenGL MNNOpenGLCodeGen)
  ENDIF()
endif()
