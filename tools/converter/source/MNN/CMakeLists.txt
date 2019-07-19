cmake_minimum_required(VERSION 3.0)

project(mnn_bizcode)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/source/IR)
include_directories(${CMAKE_SOURCE_DIR}/source/include)

file(GLOB MNN_SRC ${CMAKE_SOURCE_DIR}/source/mnn/*)

add_library(mnn_bizcode SHARED ${MNN_SRC})
