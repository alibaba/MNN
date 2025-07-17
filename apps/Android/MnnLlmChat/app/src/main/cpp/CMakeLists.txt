# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("mnnllmapp")

add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        llm_mnn_jni.cpp
        diffusion_jni.cpp
        diffusion_session.cpp
        llm_session.cpp
        crash_util.cpp
)

#load local mnn,output dir will changed to .cxx
add_subdirectory(../../../../../../../ build-mnn)

#mnn
set (MNN_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/../../../../../../../")
#set (MNN_SOURCE_ROOT "/Users/songjinde/git/DAI/AliNNPrivate")

#set (MNN_ROOT "${CMAKE_SOURCE_DIR}/../../../../../../../")
message(INFO "MNN_SOURCE_ROOT: ${MNN_SOURCE_ROOT}")

include_directories("${MNN_SOURCE_ROOT}/include/")

#diffusion
include_directories("${MNN_SOURCE_ROOT}/transformers/diffusion/engine/include/")

#llm
include_directories("${MNN_SOURCE_ROOT}/transformers/llm/engine/include")
#include_directories("${CMAKE_SOURCE_DIR}/include")

#audio
include_directories("${MNN_SOURCE_ROOT}/tools/audio/include")
include_directories("${CMAKE_SOURCE_DIR}/third_party")


# Link your native library with the `.so` file

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
        log
        MNN
        )
