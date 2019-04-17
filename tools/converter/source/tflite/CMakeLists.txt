cmake_minimum_required(VERSION 3.0)

project(tflite)

set(CMAKE_CXX_STANDARD 11)


set(TFLITE_SRC_PATH ${CMAKE_SOURCE_DIR}/source/tflite)

include_directories(${CMAKE_SOURCE_DIR}/source/IR)
include_directories(${CMAKE_SOURCE_DIR}/source/include)
include_directories(${TFLITE_SRC_PATH}/schema)
include_directories(${TFLITE_SRC_PATH})

file(GLOB TFLITE_SRC ${TFLITE_SRC_PATH}/*)

add_library(tflite SHARED ${TFLITE_SRC})