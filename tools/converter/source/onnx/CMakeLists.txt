cmake_minimum_required(VERSION 3.0)

project(onnx)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

find_package(Protobuf REQUIRED)
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

include_directories(${Protobuf_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    onnx.proto
    onnx-operators.proto
    # onnx-ml.proto
    # onnx-operators-ml.proto
)

include_directories(${SRC_PATH}/IR)
include_directories(${SRC_PATH}/include)

file(GLOB ONNX_SRC ${SRC_PATH}/onnx/*)
add_library(onnx SHARED ${ONNX_SRC} ${ONNX_PROTO_SRCS})
target_link_libraries(onnx ${Protobuf_LIBRARIES})