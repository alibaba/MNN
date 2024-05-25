#add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
#    COMMAND ${PYTHON_EXECUTABLE}
#    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_codegen.py"
#    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/"
#    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
#    COMMENT "OpenCL Code Generation"
#)
#add_custom_target (MNN_CLCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
file(GLOB_RECURSE MNN_OpenCL_SRC ${CMAKE_CURRENT_LIST_DIR}/*)

option(MNN_OPENCL_SIZE_CUT "Disable MNN OpenCL Buffer Opt" OFF)
option(MNN_OPENCL_PROFILE "Enable MNN OpenCL Kernel Profile" OFF)

IF (MNN_OPENCL_PROFILE)
    add_definitions(-DENABLE_OPENCL_TIME_PROFILER)
ENDIF()

IF (MNN_OPENCL_SIZE_CUT)
    add_definitions(-DMNN_OPENCL_BUFFER_CLOSED)
ENDIF()

IF (MNN_LOW_MEMORY)
    add_definitions(-DMNN_LOW_MEMORY)
ENDIF()

if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
    add_definitions(-DMNN_USE_LIB_WRAPPER)
    add_definitions(-DMNN_OPENCL_SVM_ENABLE)
    add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows" OR ${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR MNN_SUPPORT_INTEL_SUBGROUP)
        add_definitions(-DMNN_SUPPORT_INTEL_SUBGROUP)
    endif()
    if (MNN_USE_SYSTEM_LIB)
        find_package(OpenCL REQUIRED)
    else()
        add_definitions(-DMNN_USE_LIB_WRAPPER)
        add_definitions(-DMNN_OPENCL_SVM_ENABLE)
        add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
    endif()
endif()
IF(MNN_SEP_BUILD)
    add_definitions(-DMNN_OPENCL_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    endif()
    add_library(MNN_CL SHARED ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
    target_link_libraries(MNN_CL MNN ${OpenCL_LIBRARIES})
ELSE()
    add_library(MNN_CL OBJECT ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
    set(MNN_OCL_LIBS ${OpenCL_LIBRARIES} PARENT_SCOPE)
ENDIF()

target_include_directories(MNN_CL PRIVATE
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_LIST_DIR}/schema/current
    ${CMAKE_SOURCE_DIR}/3rd_party/half
)
