cmake_minimum_required(VERSION 2.6)

project(MNNTrain)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)
if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE -s")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_VERSION "0.0.0.1")

set(MNN_PATH ../../)

include_directories(../../3rd_party)
include_directories(${MNN_PATH}/tools)
include_directories(${MNN_PATH}/source/core/)
include_directories(include)
include_directories(${MNN_PATH}/express/include)
include_directories(source/transform)
include_directories(source/train)
file(GLOB BASIC_INCLUDE ${MNN_PATH}/include/*)
file(GLOB TRANSFORMER ./source/transform/*)
file(GLOB OPERATOR ./source/operator/*)

include_directories(source/operator)
add_library(transformer SHARED ${TRANSFORMER} ${BASIC_INCLUDE})
target_link_libraries(transformer MNN MNN_Express)

add_executable(transformer.out ./source/exec/transformer.cpp)
target_link_libraries(transformer.out transformer)

add_executable(train.out ./source/exec/train.cpp ${SCHEMA} ${BASIC_INCLUDE})
target_link_libraries(train.out ${MNN_DEPEND})

add_executable(rawDataTransform.out ./source/exec/rawDataTransform.cpp ${SCHEMA} ${BASIC_INCLUDE})

include_directories(../../3rd_party/imageHelper/)
add_executable(dataTransformer.out ./source/exec/dataTransformer.cpp ${SCHEMA} ${BASIC_INCLUDE})
target_link_libraries(dataTransformer.out MNN)
