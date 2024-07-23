cmake_minimum_required(VERSION 3.0)
# Versioning stuff
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/MNN/MNNDefine.h" MNN_DEFINE)
string(REGEX MATCH "MNN_VERSION_MAJOR [0-9]+" MNN_VERSION_MAJOR_DEFINE ${MNN_DEFINE})
string(REGEX MATCH "[0-9]+" MNN_VERSION_MAJOR ${MNN_VERSION_MAJOR_DEFINE})
string(REGEX MATCH "MNN_VERSION_MINOR [0-9]+" MNN_VERSION_MINOR_DEFINE ${MNN_DEFINE})
string(REGEX MATCH "[0-9]+" MNN_VERSION_MINOR ${MNN_VERSION_MINOR_DEFINE})
string(REGEX MATCH "MNN_VERSION_PATCH [0-9]+" MNN_VERSION_PATCH_DEFINE ${MNN_DEFINE})
string(REGEX MATCH "[0-9]+" MNN_VERSION_PATCH ${MNN_VERSION_PATCH_DEFINE})
set(MNN_VERSION ${MNN_VERSION_MAJOR}.${MNN_VERSION_MINOR}.${MNN_VERSION_PATCH})

# Clear VERSION variables when no VERSION is given to project()
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
# MSVC runtime library flags are selected by an abstraction.
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()
project(MNN VERSION ${MNN_VERSION} LANGUAGES C CXX ASM)
# complier options
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_LIST_DIR}/cmake"
)
# build options
option(MNN_USE_SYSTEM_LIB "For opencl and vulkan, use system lib or use dlopen" OFF)
option(MNN_BUILD_HARD "Build -mfloat-abi=hard or not" OFF)
option(MNN_BUILD_SHARED_LIBS "MNN build shared or static lib" ON)
option(MNN_WIN_RUNTIME_MT "MNN use /MT on Windows dll" OFF)
option(MNN_FORBID_MULTI_THREAD "Disable Multi Thread" OFF)
option(MNN_OPENMP "Use OpenMP's thread pool implementation. Does not work on iOS or Mac OS" OFF)
option(MNN_USE_THREAD_POOL "Use MNN's own thread pool implementation" ON)
option(MNN_BUILD_TRAIN "Build MNN's training framework" OFF)
option(MNN_BUILD_DEMO "Build demo/exec or not" OFF)
option(MNN_BUILD_TOOLS "Build tools/cpp or not" ON)
option(MNN_BUILD_QUANTOOLS "Build Quantized Tools or not" OFF)
option(MNN_EVALUATION "Build Evaluation Tools or not" OFF)
option(MNN_BUILD_CONVERTER "Build Converter" OFF)
option(MNN_SUPPORT_DEPRECATED_OP "Enable MNN's tflite quantized op" ON)
option(MNN_DEBUG_MEMORY "MNN Debug Memory Access" OFF)
option(MNN_DEBUG_TENSOR_SIZE "Enable Tensor Size" OFF)
option(MNN_GPU_TRACE "Enable MNN Gpu Debug" OFF)
option(MNN_SUPPORT_RENDER "Enable MNN Render Ops" OFF)
option(MNN_SUPPORT_TRANSFORMER_FUSE "Enable MNN transformer Fuse Ops" OFF)
option(MNN_PORTABLE_BUILD "Link the static version of third party libraries where possible to improve the portability of built executables" OFF)
option(MNN_SEP_BUILD "Build MNN Backends and expression separately. Only works with MNN_BUILD_SHARED_LIBS=ON" ON)
option(NATIVE_LIBRARY_OUTPUT "Native Library Path" OFF)
option(NATIVE_INCLUDE_OUTPUT "Native Include Path" OFF)
option(MNN_AAPL_FMWK "Build MNN.framework instead of traditional .a/.dylib" OFF)
option(MNN_WITH_PLUGIN "Build with plugin op support." OFF)
option(MNN_BUILD_MINI "Build MNN-MINI that just supports fixed shape models." OFF)
option(MNN_USE_SSE "Use SSE optimization for x86 if possiable" ON)
option(MNN_BUILD_CODEGEN "Build with codegen" OFF)
option(MNN_ENABLE_COVERAGE "Build with coverage enable" OFF)
option(MNN_BUILD_PROTOBUFFER "Build with protobuffer in MNN" ON)
option(MNN_BUILD_OPENCV "Build OpenCV api in MNN." OFF)
option(MNN_BUILD_LLM "Build llm library based MNN." OFF)
option(MNN_BUILD_DIFFUSION "Build diffusion demo based MNN." OFF)
option(MNN_INTERNAL "Build with MNN internal features, such as model authentication, metrics logging" OFF)
option(MNN_JNI "Build MNN Jni for java to use" OFF)
option(MNN_SUPPORT_BF16 "Enable MNN's bf16 op" OFF)
option(MNN_LOW_MEMORY "Build MNN support low memory for weight quant model." OFF)

IF (OHOS)
  include($ENV{NODE_PATH}/@ali/tcpkg/tcpkg.cmake)
  export_headers(DIR ${CMAKE_SOURCE_DIR}/include/MNN)
  IF (MNN_BUILD_OPENCV)
    export_headers(DIR ${CMAKE_SOURCE_DIR}/tools/cv/include/cv)
  ENDIF()
ENDIF()

IF (NOT DEFINED MNN_USE_SPARSE_COMPUTE)
   set(MNN_USE_SPARSE_COMPUTE ON)
ENDIF()

IF(NOT MNN_BUILD_SHARED_LIBS AND MNN_SEP_BUILD)
  message(WARNING "Close MNN_SEP_BUILD for static library")
  SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
ENDIF()
IF(APPLE AND MNN_AAPL_FMWK AND MNN_SEP_BUILD)
  message(WARNING "MNN_SEP_BUILD AND MNN_AAPL_FMWK can't coexist. Turning off MNN_SEP_BUILD")
  SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
ENDIF()
IF(WIN32)
  IF(MNN_SEP_BUILD)
    message(WARNING "MNN_SEP_BUILD IS TROUBLESOME ON Windows. Forcing OFF...")
    SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
  ENDIF()
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)

  IF(MSVC)
    # generate optimized (release) exe and library with pdb debug file, https://stackoverflow.com/a/31264946
    SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")

    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4267 /wd4018 /wd4251 /wd4996 /wd4244 /wd4146 /wd4129 /wd4305 /wd4275 /wd4101")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4018 /wd4251 /wd4996 /wd4244 /wd4146 /wd4129 /wd4305 /wd4275 /wd4101")
  ENDIF()
ENDIF()

# for coverage test
IF( MNN_ENABLE_COVERAGE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
ENDIF()

if ((CMAKE_SYSTEM_NAME STREQUAL "Darwin") AND CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_OSX_ARCHITECTURES})
endif()

# do this before protobuf, make sure wincrt config of protobuf and MNN is same
if(MSVC)
    # same as protobuf, otherwise config is inconsistent
    if(CMAKE_VERSION VERSION_GREATER 3.15 OR CMAKE_VERSION VERSION_EQUAL 3.15)
      set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)
      if(NOT MNN_WIN_RUNTIME_MT)
        set(CMAKE_MSVC_RUNTIME_LIBRARY ${CMAKE_MSVC_RUNTIME_LIBRARY}DLL)
      endif()
    else()
      foreach(flag_var
          CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
          CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
          if (MNN_WIN_RUNTIME_MT)
              if(${flag_var} MATCHES "/MD")
                  string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
              endif()
          else ()
              if(${flag_var} MATCHES "/MT")
                  string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
              endif()
          endif ()
      endforeach()
    endif()
    set(protobuf_BUILD_SHARED_LIBS ${MNN_BUILD_SHARED_LIBS})
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/macros.cmake)
IF(MNN_BUILD_PROTOBUFFER)
IF(MNN_BUILD_CONVERTER)
  IF(MSVC)
    set(protobuf_BUILD_SHARED_LIBS ${MNN_BUILD_SHARED_LIBS})
    IF((NOT MNN_BUILD_SHARED_LIBS) AND (NOT MNN_WIN_RUNTIME_MT))
      message(FATAL_ERROR "When MNN_BUILD_CONVERTER=ON and MNN_BUILD_SHARED_LIBS=OFF, MNN_WIN_RUNTIME_MT must be ON. Because protobuf not support the config(static /MD)")
    ENDIF()
  ENDIF()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/3rd_party/protobuf/cmake)
ENDIF()
ENDIF()

# specify source file encoding explicitly, fix cross-platform garbled output issue
# we need do this after protobuf which set different execution-charset
IF(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} /source-charset:utf-8")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /source-charset:utf-8")
ENDIF()

IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT MNN_BUILD_SHARED_LIBS AND NOT (MSVC OR WIN32))
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
  IF(MNN_BUILD_CONVERTER)
    SET(MNN_PORTABLE_BUILD ON CACHE BOOL "<docstring>" FORCE)
  ENDIF()
ENDIF()

if(MNN_FORBID_MULTI_THREAD)
    add_definitions(-DMNN_FORBIT_MULTI_THREADS)
endif()
if(MNN_SUPPORT_DEPRECATED_OP)
    add_definitions(-DMNN_SUPPORT_DEPRECATED_OP)
endif()
if(MNN_SUPPORT_RENDER)
    add_definitions(-DMNN_SUPPORT_RENDER)
endif()
if(MNN_SUPPORT_TRANSFORMER_FUSE)
    add_definitions(-DMNN_SUPPORT_TRANSFORMER_FUSE)
endif()
# debug options
if(MNN_DEBUG_MEMORY)
    add_definitions(-DMNN_DEBUG_MEMORY)
endif()
if(MNN_DEBUG_TENSOR_SIZE)
    add_definitions(-DMNN_DEBUG_TENSOR_SIZE)
endif()
if(MNN_GPU_TRACE)
    add_definitions(-DMNN_GPU_FORCE_FINISH)
endif()

# backend options
option(MNN_METAL "Enable Metal" OFF)
option(MNN_OPENCL "Enable OpenCL" OFF)
option(MNN_OPENGL "Enable OpenGL" OFF)
option(MNN_VULKAN "Enable Vulkan" OFF)
option(MNN_ARM82 "Enable ARM82" OFF)
option(MNN_ONEDNN "Enable oneDNN" OFF)
option(MNN_AVX512 "Enable AVX512" OFF)
option(MNN_CUDA "Enable CUDA" OFF)
option(MNN_TENSORRT "Enable TensorRT" OFF)
option(MNN_COREML "Enable CoreML" OFF)
option(MNN_NNAPI "Enable NNAPI" OFF)

option(MNN_CUDA_PROFILE "Enable CUDA profile" OFF)

if (NOT MNN_CUDA OR NOT CMAKE_SYSTEM_NAME MATCHES "^Linux")
  set(MNN_CUDA_PROFILE OFF)
endif()

if (MNN_USE_THREAD_POOL)
    message(STATUS "Use Threadpool, forbid openmp")
    set(MNN_OPENMP OFF)
    add_definitions(-DMNN_USE_THREAD_POOL)
endif()

# When build Android based on arm32 by MTL, force turn off MNN_ARM82
if (CMAKE_SYSTEM_NAME MATCHES "^Android" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    message(STATUS "force turn off MNN_ARM82 when build for Android based on arm32 by MTL")
    SET(MNN_ARM82 OFF CACHE BOOL "Enable ARM82" FORCE)
endif()

# target options
option(MNN_BUILD_BENCHMARK "Build benchmark or not" OFF)
option(MNN_BUILD_TEST "Build tests or not" OFF)
option(MNN_BUILD_FOR_ANDROID_COMMAND "Build from command" OFF)
option(MNN_USE_LOGCAT "Use Logcat intead of print for info" ON)
set (MNN_HIDDEN FALSE)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
ELSE()
    set(MNN_HIDDEN TRUE)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "MNN BUILD INFO:")
message(STATUS "\tSystem: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tVersion: ${MNN_VERSION}")
message(STATUS "\tMetal: ${MNN_METAL}")
message(STATUS "\tOpenCL: ${MNN_OPENCL}")
message(STATUS "\tOpenGL: ${MNN_OPENGL}")
message(STATUS "\tVulkan: ${MNN_VULKAN}")
message(STATUS "\tARM82: ${MNN_ARM82}")
message(STATUS "\toneDNN: ${MNN_ONEDNN}")
message(STATUS "\tTensorRT: ${MNN_TENSORRT}")
message(STATUS "\tCoreML: ${MNN_COREML}")
message(STATUS "\tNNAPI: ${MNN_NNAPI}")
message(STATUS "\tCUDA: ${MNN_CUDA}")
message(STATUS "\tOpenMP: ${MNN_OPENMP}")
message(STATUS "\tBF16: ${MNN_SUPPORT_BF16}")
message(STATUS "\tThreadPool: ${MNN_USE_THREAD_POOL}")
message(STATUS "\tHidden: ${MNN_HIDDEN}")
message(STATUS "\tBuild Path: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "\tCUDA PROFILE: ${MNN_CUDA_PROFILE}")

if(CMAKE_SYSTEM_NAME MATCHES "^Android" OR CMAKE_SYSTEM_NAME MATCHES "^Linux")
    add_definitions(-fPIC)
endif()

# Raspberry Pi 32-bit fix
if(CMAKE_SYSTEM_NAME MATCHES "^Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
    add_definitions(-march=armv7-a -mfpu=neon-vfpv4)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    add_definitions(-DMNN_BUILD_FOR_ANDROID)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        add_definitions(-mfloat-abi=softfp -mfpu=neon)
    endif()
endif()
option(MNN_USE_CPP11 "Enable MNN use c++11" ON)
if (NOT MSVC)
    if(MNN_CUDA AND MNN_SUPPORT_TRANSFORMER_FUSE)
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    elseif(MNN_USE_CPP11)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
        add_definitions(-mfpu=neon)    #please define in project/cross-compile/arm.toolchain.cmake
    endif()
    if(MNN_BUILD_HARD)
        add_definitions(-mfloat-abi=hard)  #better define in project/cross-compile/arm.toolchain.cmake
    endif()
endif()

IF(MNN_DEBUG_MEMORY)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DMNN_DEBUG -DDEBUG)
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DEBUG")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEBUG")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    endif()
else()
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
        if(CMAKE_SYSTEM_NAME MATCHES "^Android")
            if(MNN_BUILD_FOR_ANDROID_COMMAND)
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE -s")
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
            endif()
        endif()
    endif()
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)
if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    IF(MNN_USE_LOGCAT)
        add_definitions(-DMNN_USE_LOGCAT)
    ENDIF()
    IF (NOT MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_LIBRARY_OUTPUT}/${ANDROID_ABI})
    ENDIF()
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "^Linux")
    if((CMAKE_SYSTEM_PROCESSOR MATCHES "^arm") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64"))
        set(aarch64_linux_include
            #/usr/include/c++/4.9
            #/usr/lib/gcc/x86_64-linux-gnu/4.9
            #/usr/lib/gcc/x86_64-linux-gnu/4.9/include
            #/usr/include/x86_64-linux-gnu/c++/4.9
        )
        include_directories(${aarch64_linux_include})
    endif()
endif()
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/
                    ${CMAKE_CURRENT_LIST_DIR}/source/
                    ${CMAKE_CURRENT_LIST_DIR}/express/
                    ${CMAKE_CURRENT_LIST_DIR}/tools/
                    ${CMAKE_CURRENT_LIST_DIR}/codegen/
                    ${CMAKE_CURRENT_LIST_DIR}/schema/current/
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/flatbuffers/include
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/half
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/imageHelper
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/OpenCLHeaders/
                  )


set(MNN_OBJECTS_TO_LINK "")
set(MNN_TARGETS "")

# Core
FILE(GLOB MNN_Core_SRC ${CMAKE_CURRENT_LIST_DIR}/source/core/*)
add_library(MNNCore OBJECT ${MNN_Core_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCore>)
list(APPEND MNN_TARGETS MNNCore)
if(MNN_BUILD_MINI)
    target_compile_options(MNNCore PRIVATE -DMNN_BUILD_MINI)
endif()

# CV
FILE(GLOB MNN_CV_SRC ${CMAKE_CURRENT_LIST_DIR}/source/cv/*)
add_library(MNNCV OBJECT ${MNN_CV_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCV>)
list(APPEND MNN_TARGETS MNNCV)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)|(i686)")
    if (APPLE)
        add_definitions(-fno-stack-check) # Workaround a Xcode 11.X bug
    endif()
endif()

# Math
FILE(GLOB MNN_Math_SRC ${CMAKE_CURRENT_LIST_DIR}/source/math/*)
add_library(MNNMath OBJECT ${MNN_Math_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMath>)
list(APPEND MNN_TARGETS MNNMath)

# Transform
FILE(GLOB_RECURSE MNN_Transform_SRC ${CMAKE_CURRENT_LIST_DIR}/source/shape/* ${CMAKE_CURRENT_LIST_DIR}/source/geometry/*)
add_library(MNNTransform OBJECT ${MNN_Transform_SRC})
IF (NOT MNN_BUILD_MINI)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNTransform>)
ENDIF()
list(APPEND MNN_TARGETS MNNTransform)

# Utils
FILE(GLOB MNN_Utils_SRC ${CMAKE_CURRENT_LIST_DIR}/source/utils/*)
add_library(MNNUtils OBJECT ${MNN_Utils_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNUtils>)
list(APPEND MNN_TARGETS MNNUtils)

include(${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/CMakeLists.txt)


SET(MNN_PUB_HDRS "")
SET(MNN_EXPR_PUB_HDRS "")
set(MNN_EXTRA_HEADERS "")

list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/MNNDefine.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/Interpreter.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/HalideRuntime.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/Tensor.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/ErrorCode.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/ImageProcess.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/Matrix.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/Rect.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/MNNForwardType.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/AutoTime.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/MNNSharedContext.h")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Expr.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/ExprCreator.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/MathOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/NeuralNetWorkOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Optimizer.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Executor.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Module.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/NeuralNetWorkOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/ExecutorScope.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Scope.hpp")

# Add Extra Header
IF(MNN_BUILD_OPENCV)
  file(GLOB MNN_CV_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/tools/cv/include/cv/*.hpp PARENT_SCOPE)
  file(GLOB MNN_CV_IMGHDRS ${CMAKE_CURRENT_SOURCE_DIR}/tools/cv/include/cv/imgproc/*.hpp PARENT_SCOPE)
  list(APPEND MNN_EXTRA_HEADERS ${MNN_CV_HDRS})
  list(APPEND MNN_EXTRA_HEADERS ${MNN_CV_IMGHDRS})
ENDIF()
IF(MNN_BUILD_LLM)
  file(GLOB MNN_LLM_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/transformers/llm/engine/include/llm/*)
  list(APPEND MNN_EXTRA_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/transformers/llm/engine/include/llm/llm.hpp)
ENDIF()


set(MNN_DEPS "")
set(MNN_EXTRA_DEPENDS "")

# Add Thread dependency
find_package(Threads)
list(APPEND MNN_EXTRA_DEPENDS ${CMAKE_THREAD_LIBS_INIT})

if (NOT APPLE)
  if(MNN_OPENMP)
      message(STATUS "[*] Checking OpenMP")
      find_package(OpenMP)
      # For CMake < 3.9, we need to make the target ourselves
      if(NOT TARGET OpenMP::OpenMP_CXX)
          add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
          set_property(TARGET OpenMP::OpenMP_CXX
              PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
          # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
          set_property(TARGET OpenMP::OpenMP_CXX
              PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
      endif()
      # TODO: Don't pollute global CFLAGS
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      if (MSVC)
          set(OpenMP_C_FLAGS "/openmp ${OpenMP_C_FLAGS}")
          set(OpenMP_CXX_FLAGS "/openmp ${OpenMP_CXX_FLAGS}")
      endif()
      list(APPEND MNN_EXTRA_DEPENDS OpenMP::OpenMP_CXX)
    endif()
endif()

if ((NOT MSVC) AND MNN_HIDDEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
    # Omit frame pointer may cause difficult debug
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
endif()
if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fp:fast")
endif()

# Metal
set(MNN_DEPS "")
set(MNN_EXTRA_DEPENDS "")
list(APPEND MNN_DEPS MNN)

# Plugin
if(MNN_WITH_PLUGIN)
    add_definitions(-DMNN_WITH_PLUGIN)
    include(${CMAKE_CURRENT_LIST_DIR}/source/plugin/CMakeLists.txt)
endif()

# Metal
if(MNN_METAL AND APPLE)
    target_compile_options(MNNCore PRIVATE -DMNN_METAL_ENABLED=1)
    include(${CMAKE_CURRENT_LIST_DIR}/source/backend/metal/CMakeLists.txt)
    list(APPEND MNN_TARGETS MNNMetal)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMetal>)
endif()

# CoreML
IF(MNN_COREML)
    add_definitions(-DMNN_COREML_ENABLED=1)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/coreml/)

    IF(MNN_SEP_BUILD)
      list(APPEND MNN_DEPS MNNCoreML)
      list(APPEND MNN_EXTRA_DEPENDS MNNCoreML)
    ELSE()
      list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCoreML>)
    ENDIF()

    find_library(COREML CoreML)
    find_library(FOUNDATION Foundation)
    find_library(METAL Metal)
    find_library(VIDEO CoreVideo)
    list(APPEND MNN_EXTRA_DEPENDS ${COREML})
    list(APPEND MNN_EXTRA_DEPENDS ${FOUNDATION})
    list(APPEND MNN_EXTRA_DEPENDS ${METAL})
    list(APPEND MNN_EXTRA_DEPENDS ${VIDEO})
ENDIF()

# NNAPI
IF(MNN_NNAPI)
    add_definitions(-DMNN_NNAPI_ENABLED=1)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/nnapi/)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_NNAPI>)
ENDIF()

# Vulkan
IF(MNN_VULKAN)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/vulkan/)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNN_Vulkan)
  ELSE()
    list(APPEND MNN_TARGETS MNN_Vulkan)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Vulkan>)
    list(APPEND MNN_EXTRA_DEPENDS ${MNN_VULKAN_LIBS})
  ENDIF()
ENDIF()

# oneDNN
IF(MNN_ONEDNN)
    target_compile_definitions(MNNCPU PRIVATE "-DMNN_USE_ONEDNN")
    add_dependencies(MNNCPU oneDNN)
    include(cmake/oneDNN.cmake)
    set(ONEDNN_DIR ${CMAKE_CURRENT_LIST_DIR}/3rd_party/oneDNN)
    add_library(ONEDNN_COMMON OBJECT IMPORTED)
    file(GLOB_RECURSE OBJECT_FILES ${ONEDNN_DIR}/src/common/CMakeFiles/dnnl_common.dir/*.o)
    set_property(TARGET ONEDNN_COMMON PROPERTY IMPORTED_OBJECTS ${OBJECT_FILES})
    add_library(ONEDNN_CPU OBJECT IMPORTED)
    file(GLOB_RECURSE OBJECT_FILES ${ONEDNN_DIR}/src/cpu/CMakeFiles/dnnl_cpu.dir/*.o)
    set_property(TARGET ONEDNN_CPU PROPERTY IMPORTED_OBJECTS ${OBJECT_FILES})
    add_library(ONEDNN_CPU_X64 OBJECT IMPORTED)
    file(GLOB_RECURSE OBJECT_FILES ${ONEDNN_DIR}/src/cpu/x64/CMakeFiles/dnnl_cpu_x64.dir/*.o)
    set_property(TARGET ONEDNN_CPU_X64 PROPERTY IMPORTED_OBJECTS ${OBJECT_FILES})
    include_directories(${ONEDNN_DIR}/include)
    list(APPEND MNN_TARGETS ${ONEDNN_COMMON})
    list(APPEND MNN_TARGETS ${ONEDNN_CPU})
    list(APPEND MNN_TARGETS ${ONEDNN_CPU_X64})
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:ONEDNN_COMMON>)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:ONEDNN_CPU>)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:ONEDNN_CPU_X64>)
ENDIF()

# OpenCL
IF(MNN_OPENCL)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/opencl/)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNN_CL)
  ELSE()
    add_definitions(-DMNN_OPENCL_ENABLED=1)
    list(APPEND MNN_TARGETS MNN_CL)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_CL>)
    list(APPEND MNN_EXTRA_DEPENDS ${MNN_OCL_LIBS})
  ENDIF()
ENDIF()

# OpenGL
IF(MNN_OPENGL)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/opengl/)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNN_GL)
  ELSE()
    list(APPEND MNN_TARGETS MNN_GL)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_GL>)
    list(APPEND MNN_EXTRA_DEPENDS GLESv3)
    list(APPEND MNN_EXTRA_DEPENDS EGL)
  ENDIF()
ENDIF()

# CUDA
IF(MNN_CUDA)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/cuda/)
  list(APPEND MNN_TARGETS MNN_CUDA)
  if (NOT MSVC)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_CUDA>)
  endif()
  list(APPEND MNN_EXTRA_DEPENDS ${MNN_CUDA_LIBS})
ENDIF()

# Express
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/express/)
IF(MNN_SEP_BUILD)
  list(APPEND MNN_DEPS MNN_Express)
ELSE()
   list(APPEND MNN_TARGETS MNN_Express)
   list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Express>)
ENDIF()

# Model Internal. Enable MNN internal features such as model authentication and metrics logging.
if (MNN_INTERNAL)
    target_compile_options(MNNCore PRIVATE -DMNN_INTERNAL_ENABLED)
    target_compile_options(MNN_Express PRIVATE -DMNN_INTERNAL_ENABLED)
    include(${CMAKE_CURRENT_LIST_DIR}/source/internal/logging/CMakeLists.txt)
    if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
        list(APPEND MNN_EXTRA_DEPENDS "-lcurl -lssl -lcrypto")
    endif()
endif()

# Train
IF(MNN_BUILD_TRAIN OR MNN_BUILD_QUANTOOLS)
  add_subdirectory(tools/train)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNNTrain)
    list(APPEND MNN_DEPS MNNTrainUtils)
  ELSE()
    list(APPEND MNN_TARGETS MNNTrain)
    list(APPEND MNN_TARGETS MNNTrainUtils)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNTrain>)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNTrainUtils>)
  ENDIF()
ENDIF()

#CodeGen
IF(MNN_BUILD_CODEGEN)
    add_definitions(-DMNN_BUILD_CODEGEN)
    include(${CMAKE_CURRENT_LIST_DIR}/codegen/CMakeLists.txt)
ENDIF()

# NPU
IF(MNN_NPU)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/)
    IF(MNN_SEP_BUILD)
        list(APPEND MNN_DEPS MNN_NPU)
    ELSE()
        list(APPEND MNN_TARGETS MNN_NPU)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_NPU>)
        list(APPEND MNN_EXTRA_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/3rdParty/${ANDROID_ABI}/libhiai.so)
        list(APPEND MNN_EXTRA_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/3rdParty/${ANDROID_ABI}/libhiai_ir_build.so)
        list(APPEND MNN_EXTRA_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/source/backend/hiai/3rdParty/${ANDROID_ABI}/libhiai_ir.so)
    ENDIF()
ENDIF()

# TensorRT
IF(MNN_TENSORRT)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/tensorrt/)
  list(APPEND MNN_TARGETS MNN_TRT)
  list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_TRT>)
  list(APPEND MNN_EXTRA_DEPENDS ${MNN_TRT_LIBS})
ENDIF()

IF(MNN_SEP_BUILD)
  add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS} ${MNN_EXTRA_HEADERS})
  target_link_libraries(MNN PUBLIC ${MNN_EXTRA_DEPENDS})
ELSE()
  IF(MNN_BUILD_SHARED_LIBS)
    add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS} ${MNN_EXTRA_HEADERS})
    if (WIN32)
      foreach(TARGET ${MNN_TARGETS})
        target_compile_definitions(${TARGET} PRIVATE "-DBUILDING_MNN_DLL")
        target_compile_definitions(${TARGET} INTERFACE "-DUSING_MNN_DLL")
      endforeach()
      target_compile_definitions(MNN PRIVATE "-DBUILDING_MNN_DLL")
      target_compile_definitions(MNN INTERFACE "-DUSING_MNN_DLL")
    endif()
  ELSE()
    add_library(MNN STATIC ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS} ${MNN_EXTRA_HEADERS})
  ENDIF()
  target_link_libraries(MNN PUBLIC ${MNN_EXTRA_DEPENDS})
ENDIF()
if (MSVC)
  target_link_options(MNN PRIVATE "/IGNORE:4049,4217")
  if (MNN_CUDA)
    if (MNN_BUILD_SHARED_LIBS)
      target_link_options(MNN PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:MNN_CUDA>")
    else()
      add_custom_command(
        TARGET MNN
        POST_BUILD
        COMMAND lib.exe ARGS /OUT:$<TARGET_FILE:MNN> $<TARGET_FILE:MNN> $<TARGET_FILE:MNN_CUDA>
      )
    endif()
  endif()
endif()
if (MNN_ONEDNN)
    add_dependencies(MNN ONEDNN_COMMON ONEDNN_CPU ONEDNN_CPU_X64)
endif()

if(APPLE)
    IF(MNN_AAPL_FMWK)
      set_target_properties(MNN PROPERTIES FRAMEWORK TRUE)
      set_target_properties(MNN PROPERTIES
          MACOSX_FRAMEWORK_IDENTIFIER com.alibaba.MNN
          MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${PACKAGE_VERSION}
          MACOSX_FRAMEWORK_BUNDLE_VERSION ${PACKAGE_VERSION}
          XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
      )
      set_target_properties(MNN PROPERTIES MACOSX_FRAMEWORK_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/project/ios/MNN/Info.plist)
    ENDIF()
    IF(MNN_METAL)
      find_library(FOUNDATION Foundation REQUIRED)
      target_link_libraries(MNN PUBLIC ${FOUNDATION})
      find_library(METAL Metal REQUIRED)
      target_link_libraries(MNN PUBLIC ${METAL})
      find_library(GRAPHIC CoreGraphics)
      target_link_libraries(MNN PUBLIC ${GRAPHIC})
    ENDIF()
endif()
add_dependencies(MNN MNNCore MNNCV MNNTransform MNNMath MNNCPU)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools/converter)
IF(WIN32 AND MNN_BUILD_CONVERTER AND MNN_BUILD_SHARED_LIBS)
# Because of dllimport/dllexport, we merge MNN and MNNConvertDeps together, which depend protobuf
  target_link_libraries(MNN PUBLIC ${Protobuf_LIBRARIES})
ENDIF()
# Merge MNN/MNNExpress/MNNOpenCV and other backends into one .lib/.dll on Windows
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools/cv)
IF(MNN_BUILD_OPENCV AND NOT MNN_SEP_BUILD)
  IF(MSVC)
    target_compile_definitions(MNNOpenCV PRIVATE "-DBUILDING_MNN_DLL" INTERFACE "-DUSING_MNN_DLL")
  ENDIF()
  target_sources(MNN PRIVATE $<TARGET_OBJECTS:MNNOpenCV>)
ENDIF()
IF(MNN_BUILD_LLM)
    # add_definitions(-DMNN_BUILD_LLM)
    include(${CMAKE_CURRENT_LIST_DIR}/transformers/llm/engine/CMakeLists.txt)
    IF(NOT MNN_SEP_BUILD)
      target_sources(MNN PRIVATE $<TARGET_OBJECTS:llm>)
    ENDIF()
ENDIF()

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
# Using -pthread, needed by thread-safe implemention of glibc, is better than only using -lpthread
# https://stackoverflow.com/questions/23250863/difference-between-pthread-and-lpthread-while-compiling
  target_link_libraries(MNN PUBLIC -pthread dl)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Android")
  target_link_libraries(MNN PUBLIC log m)
else()
endif()
if (NOT MNN_BUILD_SHARED_LIBS)
    if(APPLE)
        set(MNN_DEPS -Wl,-all_load ${MNN_DEPS} -Wl,-noall_load)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Static-link will not replace thread-related weak symbol in glibc with strong symbol
        # in pthread library, so we need use --whole-archive to pthread
        # https://stackoverflow.com/questions/35116327/when-g-static-link-pthread-cause-segmentation-fault-why
        if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
            set(MNN_DEPS -Wl,--whole-archive ${MNN_DEPS} -lpthread -Wl,--no-whole-archive)
        else()
            set(MNN_DEPS -Wl,--whole-archive ${MNN_DEPS} -Wl,--no-whole-archive)
        endif()
    endif()
endif()
list(APPEND MNN_TARGETS MNN)
list(REMOVE_ITEM MNN_TARGETS MNN)
IF(MNN_BUILD_DEMO)
include(${CMAKE_CURRENT_LIST_DIR}/demo/exec/CMakeLists.txt)
ENDIF()
IF(MNN_BUILD_DIFFUSION AND MNN_BUILD_OPENCV AND MNN_IMGCODECS)
include(${CMAKE_CURRENT_LIST_DIR}/transformers/diffusion/CMakeLists.txt)
ENDIF()
IF(MNN_BUILD_TOOLS)
include(${CMAKE_CURRENT_LIST_DIR}/tools/cpp/CMakeLists.txt)
ENDIF()
IF(MNN_BUILD_TEST)
include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
ENDIF()
IF(MNN_BUILD_BENCHMARK)
include(${CMAKE_CURRENT_LIST_DIR}/benchmark/CMakeLists.txt)
ENDIF()
IF(MNN_BUILD_QUANTOOLS)
include(${CMAKE_CURRENT_LIST_DIR}/tools/quantization/CMakeLists.txt)
ENDIF()
IF(MNN_EVALUATION)
include(${CMAKE_CURRENT_LIST_DIR}/tools/evaluation/CMakeLists.txt)
ENDIF()

# Install headers
IF(CMAKE_SYSTEM_NAME MATCHES "^Android" AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    IF(NOT NATIVE_INCLUDE_OUTPUT)
      set(NATIVE_INCLUDE_OUTPUT ".")
    ENDIF()
    set(MNN_INCLUDE_OUTPUT ${NATIVE_INCLUDE_OUTPUT}/MNN)
    add_custom_command(
      TARGET MNN
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      -E make_directory "${MNN_INCLUDE_OUTPUT}/"
    )
    add_custom_command(
      TARGET MNN
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      -E make_directory "${MNN_INCLUDE_OUTPUT}/expr/"
    )
    FOREACH(header ${MNN_PUB_HDRS})
      add_custom_command(
        TARGET MNN
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${header} "${MNN_INCLUDE_OUTPUT}/"
      )
    ENDFOREACH()
    FOREACH(header ${MNN_EXPR_PUB_HDRS})
      add_custom_command(
        TARGET MNN
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${header} "${MNN_INCLUDE_OUTPUT}/expr/"
      )
    ENDFOREACH()
ELSEIF(NOT APPLE)
  INSTALL(FILES ${MNN_PUB_HDRS} DESTINATION include/MNN/)
  INSTALL(FILES ${MNN_EXPR_PUB_HDRS} DESTINATION include/MNN/expr/)
  install(TARGETS MNN
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
  )
ELSE()
  install(TARGETS MNN
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      FRAMEWORK DESTINATION /Library/Frameworks/
  )
  IF(MNN_BUILD_OPENCV)
    if (NOT MNN_AAPL_FMWK)
        INSTALL(FILES ${MNN_CV_HDRS} DESTINATION include/MNN/cv)
        INSTALL(FILES ${MNN_CV_IMGHDRS} DESTINATION include/MNN/cv/imgproc)
    endif()
    FOREACH(HDR ${MNN_CV_HDRS})
      SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/cv/ )
    ENDFOREACH()
    FOREACH(HDR ${MNN_CV_IMGHDRS})
      SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/cv/imgproc )
    ENDFOREACH()
  ENDIF()
  IF(MNN_BUILD_LLM)
    if (NOT MNN_AAPL_FMWK)
        INSTALL(FILES ${MNN_LLM_HDRS} DESTINATION include/MNN/llm)
    endif()
    FOREACH(HDR ${MNN_LLM_HDRS})
      SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/llm )
    ENDFOREACH()
  ENDIF()

  if (NOT MNN_AAPL_FMWK)
      INSTALL(FILES ${MNN_PUB_HDRS} DESTINATION include/MNN/)
      INSTALL(FILES ${MNN_EXPR_PUB_HDRS} DESTINATION include/MNN/expr/)
  endif()
  FOREACH(HDR ${MNN_EXPR_PUB_HDRS})
    SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/expr/ )
  ENDFOREACH()
  FOREACH(HDR ${MNN_PUB_HDRS})
    SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/ )
  ENDFOREACH()
  IF(MNN_METAL)
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib PROPERTIES MACOSX_PACKAGE_LOCATION Resources/)
  ENDIF()
ENDIF()
if (MNN_JNI)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/jni/)
endif()
