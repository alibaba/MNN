cmake_minimum_required(VERSION 2.8)

project(tensorflow)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

if (MSVC OR WIN32)
    set(Protobuf_SRC_ROOT_FOLDER $ENV{Protobuf_SRC_ROOT_FOLDER})
endif()
find_package(Protobuf REQUIRED)
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

include_directories(${Protobuf_INCLUDE_DIRS})

protobuf_generate_cpp(TENSORFLOW_PROTO_SRCS TENSORFLOW_PROTO_HDRS
    attr_value.proto
    function.proto
    graph.proto
    node_def.proto
    op_def.proto
    resource_handle.proto
    tensor.proto
    tensor_shape.proto
    types.proto
    versions.proto
)

file(GLOB TENSORFLOW_SRC ${SRC_PATH}/tensorflow/*)

if(MNN_BUILD_SHARED_LIBS)
    add_library(tensorflow SHARED ${TENSORFLOW_SRC} ${TENSORFLOW_PROTO_SRCS})
else()
    add_library(tensorflow STATIC ${TENSORFLOW_SRC} ${TENSORFLOW_PROTO_SRCS})
endif()

if (MSVC OR WIN32)
    target_compile_options(tensorflow PRIVATE "/wd4267" "/wd4244" "/wd4305")
endif()

target_include_directories(tensorflow PRIVATE 
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SRC_PATH}/IR
    ${SRC_PATH}/include
)

target_link_libraries(tensorflow ${Protobuf_LIBRARIES})
