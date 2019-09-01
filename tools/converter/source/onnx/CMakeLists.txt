cmake_minimum_required(VERSION 3.0)

project(onnx)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

if (MSVC OR WIN32)
    set(Protobuf_SRC_ROOT_FOLDER $ENV{Protobuf_SRC_ROOT_FOLDER})
endif()
find_package(Protobuf REQUIRED)
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
    set(Protobuf_LIBRARY ${PROTOBUF_LIBRARY})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    onnx.proto
    onnx-operators.proto
    # onnx-ml.proto
    # onnx-operators-ml.proto
)

file(GLOB ONNX_SRC ${SRC_PATH}/onnx/*)

if (MSVC OR WIN32)
    add_library(onnx STATIC ${ONNX_SRC} ${ONNX_PROTO_SRCS})
    target_compile_options(onnx PRIVATE "/wd4267" "/wd4244" "/wd4305")
else()
    add_library(onnx SHARED ${ONNX_SRC} ${ONNX_PROTO_SRCS})
endif()

target_include_directories(onnx PRIVATE
    ${SRC_PATH}/IR
    ${SRC_PATH}/include
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(onnx ${Protobuf_LIBRARY})
