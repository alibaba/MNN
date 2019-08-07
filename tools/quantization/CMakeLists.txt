cmake_minimum_required(VERSION 2.6)

project(MNNQuan)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)
if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE -s")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_VERSION "0.0.0.1")

set(MNN_PATH ../../)

include_directories(${MNN_PATH}/3rd_party/imageHelper)
include_directories(${MNN_PATH}/3rd_party)
include_directories(${MNN_PATH}/tools)
include_directories(${MNN_PATH}/source/core/)
include_directories(${MNN_PATH}/include)
file(GLOB BASIC_INCLUDE ${MNN_PATH}/include/*)
file(GLOB SCHEMA ${MNN_PATH}/tools/converter/source/IR/*.h ${MNN_PATH}/tools/converter/source/IR/flatbuffers/*.h )
file(GLOB QUANFILES TensorStatistic.hpp TensorStatistic.cpp quantizeWeight.hpp quantizeWeight.cpp calibration.hpp calibration.cpp Helper.hpp Helper.cpp)

add_executable(quantized.out ${QUANFILES} quantized.cpp)
target_link_libraries(quantized.out ${MNN_DEPEND})

add_executable(testQuanModel.out testQuanModel.cpp)
target_link_libraries(testQuanModel.out ${MNN_DEPEND})
