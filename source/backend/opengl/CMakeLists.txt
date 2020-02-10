FILE(GLOB MNN_OpenGL_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
option(MNN_OPENGL_REGEN "Regenerate OpenGL Shaders." OFF)
IF(MNN_SEP_BUILD)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    add_library(MNN_GL SHARED ${MNN_OpenGL_SRC} "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp")
    target_link_libraries(MNN_GL MNN GLESv3 EGL)
ELSE()
    add_library(MNN_GL OBJECT ${MNN_OpenGL_SRC} "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp")
ENDIF()
target_include_directories(MNN_GL PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)
IF(MNN_OPENGL_REGEN)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp"
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_LIST_DIR}/makeshader.py"
    "${CMAKE_CURRENT_LIST_DIR}/glsl/"
    "${CMAKE_SOURCE_DIR}/include/MNN/backend/opengl/AllShader.h"
    "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp"
    COMMENT "OpenGL Code Generation"
  )
  add_custom_target (MNN_GLCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp")
  add_dependencies(MNN_GL MNN_GLCodeGen)
ENDIF()
