cmake_minimum_required(VERSION 3.0)

project(tflite)

set(CMAKE_CXX_STANDARD 11)


set(TFLITE_SRC_PATH ${SRC_PATH}/tflite)

include_directories(${SRC_PATH}/IR)
include_directories(${SRC_PATH}/include)
include_directories(${TFLITE_SRC_PATH}/schema)
include_directories(${TFLITE_SRC_PATH})

file(GLOB TFLITE_SRC ${TFLITE_SRC_PATH}/*)

add_library(tflite SHARED ${TFLITE_SRC})