cmake_minimum_required(VERSION 3.0)

project(optimizer)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

include_directories(${SRC_PATH}/IR)
include_directories(${SRC_PATH}/include)

file(GLOB OPTIMIZER_SRC ${SRC_PATH}/optimizer/*)

if (MSVC OR WIN32)
    add_library(optimizer STATIC ${OPTIMIZER_SRC})
    target_compile_options(optimizer PRIVATE "/wd4267")
else()
    add_library(optimizer SHARED ${OPTIMIZER_SRC})
endif()