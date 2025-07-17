cmake_minimum_required(VERSION 3.4.1)

project(taoavatar)

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/project/include)
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common)
set(CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
set(JNI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/jni)
set (THIRD_PARTY_INCLUDES_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp/third_party/)
set (A2BS_DROID_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/a2bs_droid)
set (TTS_DROID_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tts_droid)
set (LLM_DROID_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/llm_droid)
set (ASR_JNI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/asr_jni)
option(DEBUG_TTS_PIPE "tts use pipe" OFF)

include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CPP_DIR}/3rd_party
    ${CPP_DIR}/3rd_party/glm
#    ${CPP_DIR}/llm/include
    ${CPP_DIR}/tts/include
#    ${CPP_DIR}/tts_pipe/include
    ${CPP_DIR}/audio2bs/include
    ${CPP_DIR}
    ${JNI_DIR}
    ${PROJECT_INCLUDE_DIR}
    ${A2BS_DROID_DIR}
    ${COMMON_DIR}
    ${TTS_DROID_DIR})
#    ${ASR_JNI_DIR})


FILE(GLOB A2BS_DROID_SRC
        ${A2BS_DROID_DIR}/*.cpp)

FILE(GLOB CPP_SRCS
        ${CPP_DIR}/*.cpp
        ${COMMON_DIR}/*.cpp
        ${A2BS_DROID_SRC}
        ${TTS_DROID_DIR}/*.cpp
        ${LLM_DROID_DIR}/*.cpp)
#)

FILE(GLOB JNI_SRCS
        ${JNI_DIR}/*.cpp)

#add_subdirectory(${CPP_DIR}/audio2bs ${CMAKE_BINARY_DIR}/audio2bs)
#add_subdirectory(${CPP_DIR}/tts ${CMAKE_BINARY_DIR}/tts)
#add_subdirectory(${CPP_DIR}/llm ${CMAKE_BINARY_DIR}/llm)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall \
                     -DVK_USE_PLATFORM_ANDROID_KHR")

set(lib_path ${CMAKE_CURRENT_SOURCE_DIR}/libs)

add_library(MNN
        SHARED
        IMPORTED)
set(MNN_SOURCE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
message(STATUS "MNN_SOURCE_ROOT: ${MNN_SOURCE_ROOT}")
set_target_properties(MNN PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
        "${MNN_SOURCE_ROOT}/include;${MNN_SOURCE_ROOT}/tools/audio/include/;${MNN_SOURCE_ROOT}/transformers/llm/engine/include"
)
set_target_properties(MNN
        PROPERTIES IMPORTED_LOCATION
        ${lib_path}/MNN/lib/${ANDROID_ABI}/libMNN.so)

add_library(nnrruntime
        SHARED
        IMPORTED)

set_target_properties(nnrruntime
        PROPERTIES IMPORTED_LOCATION
        ${lib_path}/NNR/lib/${ANDROID_ABI}/libnnrruntime.so)
set_target_properties(nnrruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${lib_path}/NNR/include/)



list(APPEND MNN_LIBRARIES
        MNN
)

add_subdirectory(cpp/a2bs)
#add_subdirectory(cpp/a2bs_new)
if (DEBUG_TTS_PIPE)
    add_subdirectory(cpp/tts_pipe)
#    add_subdirectory(cpp/tts_pipe/third_party/espeak-ng)
else ()
    add_subdirectory(cpp/tts)
endif ()
#add_subdirectory(cpp/asr)


add_library(${CMAKE_PROJECT_NAME} SHARED ${CPP_SRCS} ${JNI_SRCS}
)


get_property(PROJECT_SOURCES TARGET ${CMAKE_PROJECT_NAME} PROPERTY SOURCES)

message(STATUS "All source files in project:")
foreach(src_file IN LISTS PROJECT_SOURCES)
    message(STATUS "${src_file}")
endforeach()


target_link_libraries(${CMAKE_PROJECT_NAME}
    log
    android
    MNN
    nnrruntime
    mnn_a2bs
    mnn_tts_SDK
    )
