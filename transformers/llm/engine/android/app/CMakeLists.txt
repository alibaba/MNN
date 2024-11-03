cmake_minimum_required(VERSION 3.10)
project(mnn-llm)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/main/jni/include/)
find_library(log log)

# jni file
set(SRCS "")
list(APPEND SRCS "${CMAKE_CURRENT_LIST_DIR}/src/main/jni/llm_mnn_jni.cpp")
set(LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/src/main/jni/libs/arm64-v8a")

include_directories("${CMAKE_CURRENT_LIST_DIR}/../../include/")
include_directories("${CMAKE_CURRENT_LIST_DIR}/../../../../../include/")
include_directories("${CMAKE_CURRENT_LIST_DIR}/../../../../../3rd_party/")

add_library(chatbot SHARED ${SRCS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# compile dynamic so, support Linux/Mac
set_target_properties(chatbot PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
target_link_libraries(chatbot
                      ${LIB_PATH}/libllm.so
                      ${LIB_PATH}/libMNN.so
                      ${LIB_PATH}/libMNN_Express.so
                      ${LIB_PATH}/libMNN_CL.so
                      ${LIB_PATH}/libMNNOpenCV.so
                      log)
