cmake_minimum_required(VERSION 3.0)

project(MNNConverter)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_VERSION "0.0.0.1")

configure_file(
    ./CONFIGURECONVERT.h.in
    ./CONFIGURECONVERT.h
)

# -----------debug or no-----------
set(CMAKE_BUILD_TYPE debug)
option(MNN_CONVERT_DEBUG "Enable MNN CONVERT DEBUG" ON)
if(MNN_CONVERT_DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
else()
    add_definitions(-DNDEBUG)
endif()

# -----------Enable tensorflow model optimizer or no-----------
option(TFMODEL_OPTIMIZE "Enable tensorflow model optimizer" OFF)
if(TFMODEL_OPTIMIZE)
    add_definitions(-DTFMODEL_OPTIMIZE)
endif()

# -----------find protobuf lib-----------
find_package(Protobuf REQUIRED)
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

# -----------set path-----------
set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(IR_PATH ${SRC_PATH}/IR)
set(COMMON_PATH ${SRC_PATH}/common)
set(CAFFE_PATH ${SRC_PATH}/caffe)
set(TENSORFLOW_PATH ${SRC_PATH}/tensorflow)
set(ONNX_PATH ${SRC_PATH}/onnx)
set(TFLITE_PATH ${SRC_PATH}/tflite)
set(MNN_PATH ${SRC_PATH}/MNN)
set(OPTIMIZER_PATH ${SRC_PATH}/optimizer)
set(INCLUDE_PATH ${SRC_PATH}/include)

# -----------set include path-----------
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${INCLUDE_PATH})
include_directories(${IR_PATH})
include_directories(${TFLITE_PATH}/schema)

add_subdirectory(${TENSORFLOW_PATH})
add_subdirectory(${ONNX_PATH})
add_subdirectory(${CAFFE_PATH})
add_subdirectory(${MNN_PATH})
add_subdirectory(${OPTIMIZER_PATH})
add_subdirectory(${TFLITE_PATH})

file(GLOB COMMON_SRC ${COMMON_PATH}/*)


add_library(COMMON_LIB STATIC ${COMMON_SRC})

add_executable(MNNDump2Json.out ${SRC_PATH}/MNNDump2Json.cpp)

include_directories(${SRC_PATH})
set(COMMAND_SRC
    ${SRC_PATH}/cli.cpp
    ${SRC_PATH}/config.cpp
)

add_executable(MNNConvert ${SRC_PATH}/MNNConverter.cpp ${COMMAND_SRC})
target_link_libraries(MNNConvert tensorflow caffe onnx mnn_bizcode tflite optimizer COMMON_LIB ${Protobuf_LIBRARIES})
