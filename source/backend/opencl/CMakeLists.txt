add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_codegen.py"
    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/"
    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
    COMMENT "OpenCL Code Generation"
)
add_custom_target (MNN_CLCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
file(GLOB_RECURSE MNN_OpenCL_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cc ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
    add_definitions(-DMNN_USE_OPENCL_WRAPPER)
    add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=110)
else()
    find_package(OpenCL REQUIRED)
endif()
IF(MNN_SEP_BUILD)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    add_library(MNN_CL SHARED ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
    target_link_libraries(MNN_CL MNN ${OpenCL_LIBRARIES})
ELSE()
    add_library(MNN_CL OBJECT ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
    set(MNN_OCL_LIBS ${OpenCL_LIBRARIES} PARENT_SCOPE)
ENDIF()

add_dependencies(MNN_CL MNN_CLCodeGen)
target_include_directories(MNN_CL PRIVATE
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_SOURCE_DIR}/3rd_party/half
)
