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

set(MNNCHAT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../")
set(GIT_COMMIT_ID_VALUE "unknown")
execute_process(
        COMMAND git -C "${MNNCHAT_REPO_ROOT}" rev-parse --short HEAD
        OUTPUT_VARIABLE GIT_COMMIT_ID_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (GIT_COMMIT_ID_OUTPUT)
    set(GIT_COMMIT_ID_VALUE "${GIT_COMMIT_ID_OUTPUT}")
endif ()

add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        llm_mnn_jni.cpp
        mnn_wrapper_jni.cpp
        llm_session.cpp
        crash_util.cpp
        processor.cpp
        # Video processing files (moved to video/ directory)
        video/video_decoder.cpp
        video/byte_buffer_decoder.cpp
        video/image_utils.cpp
        video/video_processor.cpp
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE GIT_COMMIT_ID="${GIT_COMMIT_ID_VALUE}")

# Add 16KB page size support (required for Android 15+ devices)
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")

#mnn
set (MNN_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/../../../../../../../")
#set (MNN_SOURCE_ROOT "/Users/songjinde/git/DAI/AliNNPrivate")
set (MNN_INSTALL_ROOT "${MNN_SOURCE_ROOT}/project/android/build_64")
set(LIB_PATH "${MNN_INSTALL_ROOT}/lib")
# set (MNN_INSTALL_ROOT "/Users/songjinde/MNN/project/android/build_64")

#set (MNN_ROOT "${CMAKE_SOURCE_DIR}/../../../../../../../")
message(INFO "MNN_SOURCE_ROOT: ${MNN_SOURCE_ROOT}")
message(INFO "MNN_INSTALL_ROOT: ${MNN_INSTALL_ROOT}")

include_directories("${MNN_SOURCE_ROOT}/include/")

#diffusion
include_directories("${MNN_SOURCE_ROOT}/transformers/diffusion/engine/include/")

#llm
include_directories("${MNN_SOURCE_ROOT}/transformers/llm/engine/include")
#include_directories("${CMAKE_SOURCE_DIR}/include")

#audio
include_directories("${MNN_SOURCE_ROOT}/tools/audio/include")
include_directories("${MNN_SOURCE_ROOT}/3rd_party/")
include_directories("${CMAKE_SOURCE_DIR}/third_party")
# Link the shared library

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN.so")

add_library(llm SHARED IMPORTED)
set_target_properties(llm PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libllm.so")

add_library(mnn_Express SHARED IMPORTED)
set_target_properties(mnn_Express PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_Express.so")


# Link your native library with the `.so` file

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
        log
        MNN
        llm
        mnn_Express
        # MediaCodec libraries for hardware video decoding
        mediandk

        )
