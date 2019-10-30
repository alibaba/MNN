cmake_minimum_required(VERSION 2.8)

project(caffe)

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

protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS
    caffe.proto
)

file(GLOB CAFFE_SRC ${SRC_PATH}/caffe/*)
if(MNN_BUILD_SHARED_LIBS)
    add_library(caffe SHARED ${CAFFE_SRC} ${CAFFE_PROTO_SRCS})
else()
    add_library(caffe STATIC ${CAFFE_SRC} ${CAFFE_PROTO_SRCS})
endif()

if (MSVC OR WIN32)
    target_compile_options(caffe PRIVATE "/wd4267" "/wd4244" "/wd4305")
endif()

target_include_directories(caffe PRIVATE 
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SRC_PATH}/IR
    ${SRC_PATH}/include
)

target_link_libraries(caffe ${Protobuf_LIBRARIES})
