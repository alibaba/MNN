option(MNN_QNN_CONVERT_MODE "Enable the Convert mode of the QNN backend." OFF)

if (MNN_NPU_BACKENDS_SHARED AND
    CMAKE_SYSTEM_NAME MATCHES "^Android" AND
    NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    message(STATUS "QNN backend plugin is available only for Android arm64-v8a")
    return()
endif()

file(GLOB BACKEND_SRCS ${CMAKE_CURRENT_LIST_DIR}/backend/*)
file(GLOB EXECUTION_SRCS ${CMAKE_CURRENT_LIST_DIR}/execution/*)
if (NOT MNN_NPU_BACKENDS_SHARED)
    list(FILTER BACKEND_SRCS EXCLUDE REGEX "/QNNDspPerf\\.(cpp|hpp)$")
endif()
set(MNN_QNN_SRCS ${BACKEND_SRCS} ${EXECUTION_SRCS})

# QNNBackend.cpp 中使用了 flatbuffers/flexbuffers，需要 flatbuffers util.cpp 中提供的
# ClassicLocale::instance_ 静态成员定义。MSVC 链接时不会自动从其它 target 获取，因此
# 显式将 util.cpp 一并编译进 MNN_QNN OBJECT 库。
if (MSVC OR WIN32)
    list(APPEND MNN_QNN_SRCS ${CMAKE_SOURCE_DIR}/3rd_party/flatbuffers/src/util.cpp)
endif()

# Prefer CMake variable QNN_SDK_ROOT (passed via -DQNN_SDK_ROOT=...), fallback to environment
set(_QNN_ROOT "$ENV{QNN_SDK_ROOT}")
if (DEFINED QNN_SDK_ROOT AND NOT "${QNN_SDK_ROOT}" STREQUAL "")
    set(_QNN_ROOT "${QNN_SDK_ROOT}")
endif()
message(STATUS "QNN Root: ${_QNN_ROOT}")

if (MNN_QNN_CONVERT_MODE)
    file(GLOB CONVERTOR_SRCS ${CMAKE_CURRENT_LIST_DIR}/convertor/*.cpp)
    list(APPEND MNN_QNN_SRCS ${CONVERTOR_SRCS})
endif()

if (MNN_NPU_BACKENDS_SHARED)
    add_library(MNN_QNN SHARED
                ${MNN_QNN_SRCS}
                ${CMAKE_CURRENT_LIST_DIR}/QnnBackendPlugin.cpp)
    set_target_properties(MNN_QNN PROPERTIES
                          OUTPUT_NAME "MNN_Backend_QNN")
    target_link_libraries(MNN_QNN PRIVATE MNN ${CMAKE_DL_LIBS})
else()
    add_library(MNN_QNN OBJECT ${MNN_QNN_SRCS})
endif()

if (MNN_NPU_BACKENDS_SHARED)
    target_compile_definitions(MNN_QNN PRIVATE MNN_QNN_DSP_RUNTIME)
endif()

if (MNN_QNN_OFFLINE_CONTEXT)
    target_compile_definitions(MNN_QNN PRIVATE MNN_QNN_OFFLINE_CONTEXT)
endif()

if (MNN_ARM82)
    target_compile_options(MNN_QNN PRIVATE -DENABLE_ARMV82)
endif()

if (MNN_QNN_CONVERT_MODE)
    target_compile_options(MNN_QNN PRIVATE -DENABLE_QNN_CONVERT_MODE)
endif()

if (MNN_QNN_ONLINE_FINALIZE)
    target_compile_options(MNN_QNN PRIVATE -DENABLE_QNN_ONLINE_FINALIZE)
endif()

target_include_directories(MNN_QNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/backend/)
target_include_directories(MNN_QNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/)
target_include_directories(MNN_QNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/convertor/)
target_include_directories(MNN_QNN PRIVATE ${_QNN_ROOT}/include/QNN/)
# 在 MSVC / Windows 上 __fp16 不可用，需要使用 3rd_party/half 提供软件 fp16 转换。
target_include_directories(MNN_QNN PRIVATE ${CMAKE_SOURCE_DIR}/3rd_party/half)

set(MNN_NPU_PUB_HDRS
    ${MNN_NPU_PUB_HDRS}
    "${CMAKE_CURRENT_LIST_DIR}/include/MNNQnnBackend.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/MNNQnnPlugin.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/MNNQnnSession.hpp"
    PARENT_SCOPE)
