cmake_minimum_required(VERSION 2.6)

project(MNNEvaluation)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)
if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE -s")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_VERSION "0.0.0.1")

set(MNN_PATH ../../)

include_directories(${MNN_PATH}/3rd_party)
include_directories(${MNN_PATH}/3rd_party/imageHelper)
include_directories(${MNN_PATH}/source/core/)
include_directories(${MNN_PATH}/include)

add_executable(classficationTopkEval.out classficationTopkEval.cpp)
target_link_libraries(classficationTopkEval.out ${MNN_DEPEND})
