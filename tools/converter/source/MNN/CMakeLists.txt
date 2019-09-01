cmake_minimum_required(VERSION 3.0)

project(mnn_bizcode)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

include_directories(${SRC_PATH}/IR)
include_directories(${SRC_PATH}/include)

file(GLOB MNN_SRC ${SRC_PATH}/MNN/*)

if (MSVC OR WIN32)
    add_library(mnn_bizcode STATIC ${MNN_SRC})
else()
    add_library(mnn_bizcode SHARED ${MNN_SRC})
endif()
