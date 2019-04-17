cmake_minimum_required(VERSION 3.0)

project(MNN)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/source/IR)
include_directories(${CMAKE_SOURCE_DIR}/source/include)

file(GLOB MNN_SRC ${CMAKE_SOURCE_DIR}/source/MNN/*)

add_library(MNN SHARED ${MNN_SRC})
