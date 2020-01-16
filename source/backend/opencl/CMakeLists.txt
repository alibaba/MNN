if(MNN_OPENCL)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
      COMMAND ${PYTHON_EXECUTABLE}
      "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_codegen.py"
      "${CMAKE_CURRENT_LIST_DIR}/execution/cl/"
      "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
      COMMENT "OpenCL Code Generation"
  )
  add_custom_target (MNNOpenCLCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
  file(GLOB_RECURSE MNN_OpenCL_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cc ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
  add_library(MNNOpenCL OBJECT ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
  add_dependencies(MNNOpenCL MNNOpenCLCodeGen)
  target_include_directories(MNNOpenCL PRIVATE
      ${CMAKE_SOURCE_DIR}/include/
      ${CMAKE_SOURCE_DIR}/3rd_party/half
  )
  if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
      add_definitions(-DMNN_USE_OPENCL_WRAPPER)
  endif()
  list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNOpenCL>)
  list(APPEND MNN_TARGETS MNNOpenCL)
  SET(MNN_OBJECTS_TO_LINK "${MNN_OBJECTS_TO_LINK}" PARENT_SCOPE)
  SET(MNN_TARGETS "${MNN_TARGETS}" PARENT_SCOPE)
endif()
