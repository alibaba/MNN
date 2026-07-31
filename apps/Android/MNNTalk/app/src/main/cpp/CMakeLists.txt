cmake_minimum_required(VERSION 3.22.1)

project(mnnvoice)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MNN_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../")
set(MNN_INSTALL_ROOT "${MNN_SOURCE_ROOT}/project/android/build_64")
set(MNN_LIBRARY_PATH "${MNN_INSTALL_ROOT}/lib/libMNN.so")

if(NOT EXISTS "${MNN_LIBRARY_PATH}")
    message(FATAL_ERROR
            "Missing ${MNN_LIBRARY_PATH}. Run apps/Android/MNNTalk/build.sh first.")
endif()

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${MNN_LIBRARY_PATH}")

add_library(mnnvoice SHARED local_llm_jni.cpp)

target_include_directories(mnnvoice PRIVATE
        "${MNN_SOURCE_ROOT}/include"
        "${MNN_SOURCE_ROOT}/transformers/llm/engine/include"
)

target_link_options(mnnvoice PRIVATE "-Wl,-z,max-page-size=16384")

target_link_libraries(mnnvoice
        android
        log
        MNN
)
