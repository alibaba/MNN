cmake_minimum_required(VERSION 3.0)

project(optimizer)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/source/IR)
include_directories(${CMAKE_SOURCE_DIR}/source/include)

file(GLOB OPTIMIZER_SRC ${CMAKE_SOURCE_DIR}/source/optimizer/*)

add_library(optimizer SHARED ${OPTIMIZER_SRC})