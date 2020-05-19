cmake_minimum_required(VERSION 3.0)
# Versioning stuff
if(NOT DEFINED MNN_VERSION_MAJOR)
  set(MNN_VERSION_MAJOR 0)
endif()
if(NOT DEFINED MNN_VERSION_MINOR)
  set(MNN_VERSION_MINOR 2)
endif()
if(NOT DEFINED MNN_VERSION_PATCH)
  set(MNN_VERSION_PATCH 1)
endif()
if(NOT DEFINED MNN_VERSION_BUILD)
  set(MNN_VERSION_BUILD 5)
endif()
if(NOT DEFINED MNN_VERSION_SUFFIX)
  set(MNN_VERSION_SUFFIX git)
endif()
if (NOT PACKAGE_VERSION)
  set(PACKAGE_VERSION
    "${MNN_VERSION_MAJOR}.${MNN_VERSION_MINOR}.${MNN_VERSION_PATCH}.${MNN_VERSION_BUILD}${MNN_VERSION_SUFFIX}")
endif()
add_definitions("-DMNN_VERSION=\"${PACKAGE_VERSION}\"")
add_definitions("-DMNN_VERSION_MAJOR=${MNN_VERSION_MAJOR}")
add_definitions("-DMNN_VERSION_MINOR=${MNN_VERSION_MINOR}")
add_definitions("-DMNN_VERSION_PATCH=${MNN_VERSION_PATCH}")

# CMP0048 is related to letting CMake managing the package version for us

cmake_policy(SET CMP0048 NEW)
project(MNN VERSION ${MNN_VERSION_MAJOR}.${MNN_VERSION_MINOR}.${MNN_VERSION_PATCH}.${MNN_VERSION_BUILD} LANGUAGES C CXX ASM)
# complier options
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_LIST_DIR}/cmake"
)
add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/include/MNN/VCS.h"
    COMMAND ${CMAKE_COMMAND} "-DNAMES=MNN"
    "-DMNN_SOURCE_DIR=${CMAKE_CURRENT_LIST_DIR}"
    "-DHEADER_FILE=${CMAKE_CURRENT_LIST_DIR}/include/MNN/VCS.h"
    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/GenerateVersionFromVCS.cmake"
    COMMENT "Generating Version Control Info"
)
add_custom_target (GenVCSHDR DEPENDS "${CMAKE_CURRENT_LIST_DIR}/include/MNN/VCS.h")
# Required for OpenCL/OpenGL/Vulkan CodeGen
include(FindPythonInterp REQUIRED)
# build options
option(MNN_USE_SYSTEM_LIB "For opencl and vulkan, use system lib or use dlopen" OFF)
option(MNN_BUILD_HARD "Build -mfloat-abi=hard or not" OFF)
option(MNN_BUILD_SHARED_LIBS "MNN build shared or static lib" ON)
option(MNN_FORBID_MULTI_THREAD "Disable Multi Thread" OFF)
option(MNN_OPENMP "Use OpenMP's thread pool implementation. Does not work on iOS or Mac OS" OFF)
option(MNN_USE_THREAD_POOL "Use MNN's own thread pool implementation" ON)
option(MNN_BUILD_TRAIN "Build MNN's training framework" OFF)
option(MNN_BUILD_DEMO "Build demo/exec or not" OFF)
option(MNN_BUILD_TOOLS "Build tools/cpp or not" ON)
option(MNN_BUILD_QUANTOOLS "Build Quantized Tools or not" OFF)
option(MNN_EVALUATION "Build Evaluation Tools or not" OFF)
option(MNN_BUILD_CONVERTER "Build Converter" OFF)
option(MNN_SUPPORT_TFLITE_QUAN "Enable MNN's tflite quantized op" ON)
option(MNN_DEBUG_MEMORY "MNN Debug Memory Access" OFF)
option(MNN_DEBUG_TENSOR_SIZE "Enable Tensor Size" OFF)
option(MNN_GPU_TRACE "Enable MNN Gpu Debug" OFF)
option(MNN_PORTABLE_BUILD "Link the static version of third party libraries where possible to improve the portability of built executables" OFF)
option(MNN_SEP_BUILD "Build MNN Backends and expression seperately. Only works with MNN_BUILD_SHARED_LIBS=ON" ON)
option(NATIVE_LIBRARY_OUTPUT "Native Library Path" OFF)
option(NATIVE_INCLUDE_OUTPUT "Native Include Path" OFF)
option(MNN_AAPL_FMWK "Build MNN.framework instead of traditional .a/.dylib" OFF)
option(MNN_FMA_ENABLE "x86 routine use fma extension" OFF)
option(MNN_WITH_PLUGIN "Build with plugin op support." OFF)

IF(NOT MNN_BUILD_SHARED_LIBS)
  message(WARNING "Close MNN_SEP_BUILD for static library")
  SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
ENDIF()
IF(APPLE AND MNN_AAPL_FMWK AND MNN_SEP_BUILD)
  message(WARNING "MNN_SEP_BUILD AND MNN_AAPL_FMWK can't coexist. Turning off MNN_SEP_BUILD")
  SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
ENDIF()
IF(MSVC OR WIN32)
  IF(MNN_SEP_BUILD)
    message(WARNING "MNN_SEP_BUILD IS TROUBLESOME ON Windows. Forcing OFF...")
    SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
  ENDIF()
  SET(MNN_USE_SYSTEM_LIB ON CACHE BOOL "<docstring>" FORCE)

  # generate optimized (release) exe and library with pdb debug file, https://stackoverflow.com/a/31264946
  SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
  SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
  SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")

  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4267 /wd4018 /wd4251 /wd4996 /wd4244 /wd4146 /wd4129 /wd4305 /wd4275")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4018 /wd4251 /wd4996 /wd4244 /wd4146 /wd4129 /wd4305 /wd4275")
ENDIF()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/macros.cmake)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT MNN_BUILD_SHARED_LIBS AND NOT (MSVC OR WIN32))
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
  SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
  IF(MNN_BUILD_CONVERTER)
    SET(MNN_PORTABLE_BUILD ON CACHE BOOL "<docstring>" FORCE)
  ENDIF()
ENDIF()

if (MNN_USE_THREAD_POOL)
    set(MNN_OPENMP OFF)
    add_definitions(-DMNN_USE_THREAD_POOL)
endif()

if(MNN_FORBID_MULTI_THREAD)
    add_definitions(-DMNN_FORBIT_MULTI_THREADS)
endif()
if(MNN_SUPPORT_TFLITE_QUAN)
    add_definitions(-DMNN_SUPPORT_TFLITE_QUAN)
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

# codegen register ops
if (MNN_METAL)
    add_definitions(-DMNN_CODEGEN_REGISTER)
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
message(STATUS "\tMetal: ${MNN_METAL}")
message(STATUS "\tOpenCL: ${MNN_OPENCL}")
message(STATUS "\tOpenGL: ${MNN_OPENGL}")
message(STATUS "\tVulkan: ${MNN_VULKAN}")
message(STATUS "\tARM82: ${MNN_ARM82}")
message(STATUS "\tOpenMP: ${MNN_OPENMP}")
message(STATUS "\tHidden: ${MNN_HIDDEN}")
message(STATUS "\tBuild Path: ${CMAKE_CURRENT_BINARY_DIR}")

if(WIN32)
    if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
      message(FATAL_ERROR "MNN requires CMake 3.14+ to build on Windows!")
    endif()
    foreach(flag_var
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if (MNN_BUILD_SHARED_LIBS)
            if(${flag_var} MATCHES "/MT")
                string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
            endif()
        else ()
            if(${flag_var} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            endif()
        endif ()
    endforeach()
elseif(CMAKE_SYSTEM_NAME MATCHES "^Android" OR CMAKE_SYSTEM_NAME MATCHES "^Linux")
    add_definitions(-fPIC)
endif()
if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    add_definitions(-DMNN_BUILD_FOR_ANDROID)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        add_definitions(-mfloat-abi=softfp -mfpu=neon)
    endif()
endif()
option(MNN_USE_CPP11 "Enable MNN use c++11" ON)
if (NOT MSVC)
    if(MNN_USE_CPP11)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # This is to workaround libgcc.a
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
        add_definitions(-mfpu=neon)	#please define in project/cross-compile/arm.toolchain.cmake
    endif()
    if(MNN_BUILD_HARD)
        add_definitions(-mfloat-abi=hard)  #better define in project/cross-compile/arm.toolchain.cmake
    endif()
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

# CV
FILE(GLOB MNN_CV_SRC ${CMAKE_CURRENT_LIST_DIR}/source/cv/*)
add_library(MNNCV OBJECT ${MNN_CV_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCV>)
list(APPEND MNN_TARGETS MNNCV)

# Math
FILE(GLOB MNN_Math_SRC ${CMAKE_CURRENT_LIST_DIR}/source/math/*)
add_library(MNNMath OBJECT ${MNN_Math_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMath>)
list(APPEND MNN_TARGETS MNNMath)

# Shape
FILE(GLOB MNN_Shape_SRC ${CMAKE_CURRENT_LIST_DIR}/source/shape/*)
add_library(MNNShape OBJECT ${MNN_Shape_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNShape>)
list(APPEND MNN_TARGETS MNNShape)

# Compute
FILE(GLOB MNN_Compute_SRC ${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/compute/*)
add_library(MNNCompute OBJECT ${MNN_Compute_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCompute>)
list(APPEND MNN_TARGETS MNNCompute)

# CPU
FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/*)
add_library(MNNCPU OBJECT ${MNN_CPU_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCPU>)
list(APPEND MNN_TARGETS MNNCPU)

# X86_64 AVX/SSE
include(${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/x86_x64/CMakeLists.txt)

# AArch32/64 Assemblies
include(${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/arm/CMakeLists.txt)

IF(NOT DEFINED IOS_ARCH)
  set(IOS_ARCH "")
ENDIF()

SET(MNN_PUB_HDRS "")
SET(MNN_EXPR_PUB_HDRS "")
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
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Expr.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/ExprCreator.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/MathOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/NeuralNetWorkOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Optimizer.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/MNN/expr/Executor.hpp")

if (NOT APPLE)
  if(MNN_OPENMP)
      message(STATUS "[*] Checking OpenMP")
      find_package(OpenMP)
      # For CMake < 3.9, we need to make the target ourselves
      if(NOT TARGET OpenMP::OpenMP_CXX)
          find_package(Threads REQUIRED)
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
      if (WIN32)
          set(OpenMP_C_FLAGS "/openmp ${OpenMP_C_FLAGS}")
          set(OpenMP_CXX_FLAGS "/openmp ${OpenMP_CXX_FLAGS}")
      endif()
      FOREACH(TARGET ${MNN_TARGETS})
       target_link_libraries(${TARGET} OpenMP::OpenMP_CXX)
       IF(WIN32)
            target_compile_options(${TARGET} PUBLIC /openmp ${OpenMP_CXX_FLAGS} ${OpenMP_C_FLAGS})
       ELSE()
           target_compile_options(${TARGET} PUBLIC ${OpenMP_CXX_FLAGS} ${OpenMP_C_FLAGS})
       ENDIF()
     ENDFOREACH()
    endif()
endif()

set(CMAKE_CXX_FLAGS_ORIGIN ${CMAKE_CXX_FLAGS})
set(CMAKE_C_FLAGS_ORIGIN ${CMAKE_C_FLAGS})
if ((NOT (MSVC OR WIN32)) AND MNN_HIDDEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
    endif()
endif()

# Metal
include(${CMAKE_CURRENT_LIST_DIR}/source/backend/metal/CMakeLists.txt)
set(MNN_DEPS "")
set(MNN_EXTRA_DEPENDS "")
list(APPEND MNN_DEPS MNN)

# Plugin
if(MNN_WITH_PLUGIN)
    add_definitions(-DMNN_WITH_PLUGIN)
    include(${CMAKE_CURRENT_LIST_DIR}/source/plugin/CMakeLists.txt)
endif()

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

# OpenCL
IF(MNN_OPENCL)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/opencl/)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNN_CL)
  ELSE()
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

# ARM82 Assemblies
IF(MNN_ARM82)
  add_definitions(-DENABLE_ARMV82)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/arm82/)
  IF(MNN_SEP_BUILD)
    list(APPEND MNN_DEPS MNN_Arm82)
  ELSE()
    list(APPEND MNN_TARGETS MNN_Arm82)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Arm82>)
  ENDIF()
ENDIF()
# Express
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/express/)

IF(MNN_SEP_BUILD)
  add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
  target_link_libraries(MNN PUBLIC ${MNN_EXTRA_DEPENDS})
  list(APPEND MNN_DEPS MNN_Express)
ELSE()
  list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNExpress>)
  list(APPEND MNN_TARGETS MNNExpress)
  IF(MNN_BUILD_SHARED_LIBS)
    add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
    if (MSVC OR WIN32)
      foreach(TARGET ${MNN_TARGETS})
        target_compile_definitions(${TARGET} PRIVATE "-DBUILDING_MNN_DLL")
        target_compile_definitions(${TARGET} INTERFACE "-DUSING_MNN_DLL")
      endforeach()
      target_compile_definitions(MNN PRIVATE "-DBUILDING_MNN_DLL")
      target_compile_definitions(MNN INTERFACE "-DUSING_MNN_DLL")
    endif()
  ELSE()
    add_library(MNN STATIC ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
  ENDIF()
  target_link_libraries(MNN PUBLIC ${MNN_EXTRA_DEPENDS})
ENDIF()
if (MSVC OR WIN32)
  target_link_options(MNN PRIVATE "/IGNORE:4049,4217")
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
    ENDIF()
endif()
add_dependencies(MNN MNNCore MNNCV MNNShape MNNMath MNNCompute MNNCPU GenVCSHDR)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools/converter)

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
# Using -pthread, needed by thread-safe implemention of glibc, is better than only using -lpthread
# https://stackoverflow.com/questions/23250863/difference-between-pthread-and-lpthread-while-compiling
  target_link_libraries(MNN PUBLIC -pthread)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Android")
  target_link_libraries(MNN PUBLIC log android m)
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
  FOREACH(TARGET ${MNN_TARGETS})
    IF((NOT MSVC) AND (NOT WIN32))
    else()
      target_compile_definitions(${TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
  ENDFOREACH()
list(REMOVE_ITEM MNN_TARGETS MNN)
IF(MNN_BUILD_DEMO)
include(${CMAKE_CURRENT_LIST_DIR}/demo/exec/CMakeLists.txt)
ENDIF()
IF(MNN_BUILD_TOOLS)
include(${CMAKE_CURRENT_LIST_DIR}/tools/cpp/CMakeLists.txt)
ENDIF()
IF (MNN_BUILD_TRAIN)
  add_subdirectory(tools/train)
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
