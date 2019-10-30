cmake_minimum_required(VERSION 2.8)

project(tflite)

set(CMAKE_CXX_STANDARD 11)


set(TFLITE_SRC_PATH ${SRC_PATH}/tflite)

include_directories(${SRC_PATH}/IR)
include_directories(${SRC_PATH}/include)
include_directories(${TFLITE_SRC_PATH}/schema)
include_directories(${TFLITE_SRC_PATH})

file(GLOB TFLITE_SRC ${TFLITE_SRC_PATH}/*)

if(MNN_BUILD_SHARED_LIBS)
    add_library(tflite SHARED ${TFLITE_SRC})
else()
    add_library(tflite STATIC ${TFLITE_SRC})
endif()

if (MSVC OR WIN32)
    target_compile_options(tflite PRIVATE "/wd4267" "/wd4244")
endif()
