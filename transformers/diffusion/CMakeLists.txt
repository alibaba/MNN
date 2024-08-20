cmake_minimum_required(VERSION 3.10)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../tools/cv/include)
add_executable(diffusion_demo ${SRCS})
IF(MNN_SEP_BUILD)
target_link_libraries(diffusion_demo MNN MNN_Express MNNOpenCV)
IF(MNN_OPENCL)
    target_link_libraries(diffusion_demo MNN_CL)
ENDIF()
ELSE()
target_link_libraries(diffusion_demo MNN)
ENDIF()
