file(GLOB BACKEND_SRCS ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp)
set(MNN_RKNN_SRCS ${BACKEND_SRCS})

set(_RKNN_API_INCLUDE "$ENV{RKNN_API_INCLUDE_DIR}")
if (DEFINED RKNN_API_INCLUDE_DIR AND NOT "${RKNN_API_INCLUDE_DIR}" STREQUAL "")
    set(_RKNN_API_INCLUDE "${RKNN_API_INCLUDE_DIR}")
endif()

if ("${_RKNN_API_INCLUDE}" STREQUAL "")
    message(FATAL_ERROR "MNN_RKNN=ON requires RKNN_API_INCLUDE_DIR (or env RKNN_API_INCLUDE_DIR) to point to the directory containing rknn_api.h")
endif()

add_library(MNN_RKNN OBJECT ${MNN_RKNN_SRCS})
target_include_directories(MNN_RKNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/backend/)
target_include_directories(MNN_RKNN PRIVATE ${_RKNN_API_INCLUDE})
