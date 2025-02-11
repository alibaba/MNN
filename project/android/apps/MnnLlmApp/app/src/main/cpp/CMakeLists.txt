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

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        llm_mnn_jni.cpp
        diffusion_session.cpp)
#mnn
set (MNN_ROOT "${CMAKE_SOURCE_DIR}/../../../../../../../../")
include_directories("${MNN_ROOT}/include/")

#diffusion
include_directories("${MNN_ROOT}/transformers/diffusion/engine/include/")

#llm
include_directories("${MNN_ROOT}/transformers/llm/engine/include")

# Add library paths
set(LIB_PATH "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

# Link the shared library
add_library(llm SHARED IMPORTED)
set_target_properties(llm PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libllm.so")

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN.so")

add_library(diffusion SHARED IMPORTED)
set_target_properties(diffusion PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libdiffusion.so")

add_library(mnn_cl SHARED IMPORTED)
set_target_properties(mnn_cl PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_CL.so")

add_library(mnn_audio SHARED IMPORTED)
set_target_properties(mnn_audio PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNNAudio.so")

add_library(mnn_cv SHARED IMPORTED)
set_target_properties(mnn_cv PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNNOpenCV.so")

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
        llm
        MNN
        mnn_Express
        mnn_cl
        mnn_audio
        mnn_cv
        diffusion
        )
