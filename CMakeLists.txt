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
  set(MNN_VERSION_BUILD 7)
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
# CMP0079 is required for OpenMP

IF(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
ENDIF()
IF(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
ENDIF()
project(MNN VERSION ${MNN_VERSION_MAJOR}.${MNN_VERSION_MINOR}.${MNN_VERSION_PATCH}.${MNN_VERSION_BUILD} LANGUAGES C CXX ASM)
add_definitions(-fno-stack-check) # Workaround a Xcode 11.X bug
# complier options
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
option(MNN_BUILD_HARD "Build -mfloat-abi=hard or not" OFF)
option(MNN_BUILD_SHARED_LIBS "MNN build shared or static lib" ON)
option(MNN_FORBID_MULTI_THREAD "Disable Multi Thread" OFF)
option(MNN_OPENMP "Enable Multiple Thread Linux|Android" ON)
option(MNN_USE_THREAD_POOL "Use Multiple Thread by Self ThreadPool" ON)
option(MNN_BUILD_TRAIN "Build Train Tools" OFF)
option(MNN_BUILD_DEMO "Build demo/exec or not" OFF)
option(MNN_BUILD_QUANTOOLS "Build Quantized Tools or not" OFF)
option(MNN_EVALUATION "Build Evaluation Tools or not" OFF)
option(MNN_BUILD_CONVERTER "Build Converter" OFF)
option(MNN_SUPPORT_TFLITE_QUAN "Enable MNN's tflite quantized op" ON)
option(MNN_DEBUG_MEMORY "MNN Debug Memory Access" OFF)
option(MNN_DEBUG_TENSOR_SIZE "Enable Tensor Size" OFF)
option(MNN_GPU_TRACE "Enable MNN Gpu Debug" OFF)
option(MNN_PORTABLE_BUILD "Link the static version of third party libraries where possible to improve the portability of built executables" OFF)
option(MNN_SEP_BUILD "Build MNN Backends and expression seperately. Only works with MNN_BUILD_SHARED_LIBS=ON" ON)
option(MNN_AAPL_FMWK "Build MNN.framework instead of traditional .a/.dylib" OFF)
option(MNN_USE_SSE "Enable SSE Optimizations" ON)
option(MNN_USE_AVX "Enable AVX Optimizations" ON)
option(NATIVE_LIBRARY_OUTPUT "Native Library Path" OFF)
option(NATIVE_INCLUDE_OUTPUT "Native Include Path" OFF)


set(MNN_SCHEMA_SUFFIX "default" CACHE STRING "MNN Schema Source Path Suffix")
IF(APPLE AND MNN_AAPL_FMWK AND MNN_SEP_BUILD)
  message(WARNING "MNN_SEP_BUILD AND MNN_AAPL_FMWK can't coexist. Turning off MNN_SEP_BUILD")
  SET(MNN_SEP_BUILD OFF)
ENDIF()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/macros.cmake)

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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)")
  IF(MNN_USE_AVX)
    add_definitions(-DMNN_USE_AVX)
  ENDIF()
  IF(MNN_USE_SSE)
    add_definitions(-DMNN_USE_SSE)
  ENDIF()
ENDIF()

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
elseif(CMAKE_SYSTEM_NAME MATCHES "^Android" OR (UNIX AND NOT APPLE))
    add_definitions(-fPIC)
endif()
if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    add_definitions(-DMNN_BUILD_FOR_ANDROID)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
  if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    add_definitions(-mfloat-abi=softfp -mfpu=neon)
  else()
    IF(MNN_BUILD_HARD)
      add_definitions(-mfloat-abi=hard)
    ELSE()
      add_definitions(-mfloat-abi=softfp)
    ENDIF()
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
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        if(CMAKE_SYSTEM_NAME MATCHES "^Android")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE")
            if(MNN_BUILD_FOR_ANDROID_COMMAND)
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
            endif()
            IF(NOT MNN_BUILD_FOR_ANDROID_COMMAND)
              add_definitions(-DMNN_USE_LOGCAT)
            ENDIF(NOT MNN_BUILD_FOR_ANDROID_COMMAND)
            IF(NOT NATIVE_LIBRARY_OUTPUT)
              set(NATIVE_LIBRARY_OUTPUT ".")
            ENDIF()
            set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_LIBRARY_OUTPUT}/${ANDROID_ABI})
        endif()
    endif()
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

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


# Import FlatBuffers and use standard way to generate schemas
IF(CMAKE_CROSSCOMPILING)
  message(WARNING "Cross Compilation Detected. Third-Party tools like protobuf/flatbuffer are not built. You'll need to make sure they are available in your $PATH")
ELSE()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/3rd_party/flatbuffers ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build EXCLUDE_FROM_ALL)
ENDIF()
FILE(GLOB MNN_SCHEMA_SRC ${CMAKE_CURRENT_LIST_DIR}/schema/${MNN_SCHEMA_SUFFIX}/*.fbs)
SET(SCHEMA_TARGETS "")
FOREACH(SCHEMA_SRC ${MNN_SCHEMA_SRC})
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/schema/current/")
  get_filename_component(SCHEMA_NAME "${SCHEMA_SRC}" NAME_WE)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/schema/current/${SCHEMA_NAME}_generated.h"  COMMAND flatc -c -b --gen-object-api --reflect-names ${SCHEMA_SRC} COMMENT "Generating ${SCHEMA_NAME} Schema in ${CMAKE_CURRENT_LIST_DIR}/schema/${MNN_SCHEMA_SUFFIX}" WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/schema/current/ DEPENDS ${SCHEMA_SRC})
  ADD_CUSTOM_TARGET(MNN_SCHEMA_GEN_${SCHEMA_NAME} ALL DEPENDS "${CMAKE_CURRENT_LIST_DIR}/schema/current/${SCHEMA_NAME}_generated.h")
  IF(NOT CMAKE_CROSSCOMPILING)
    add_dependencies(MNN_SCHEMA_GEN_${SCHEMA_NAME} flatc)
  ENDIF()
  LIST(APPEND SCHEMA_TARGETS "${CMAKE_CURRENT_LIST_DIR}/schema/current/${SCHEMA_NAME}_generated.h")
ENDFOREACH()

# GenVCSHDR is not actually required. But this allows sub-targets using VCS.h without extra work in their CMake dependency declaration
add_custom_target(MNN_SCHEMA_GEN DEPENDS ${SCHEMA_TARGETS} GenVCSHDR)

set(MNN_OBJECTS_TO_LINK "")
set(MNN_TARGETS "")

# Core
FILE(GLOB MNN_Core_SRC ${CMAKE_CURRENT_LIST_DIR}/source/core/*.cpp ${CMAKE_CURRENT_LIST_DIR}/source/core/*.c)
add_library(MNNCore OBJECT ${MNN_Core_SRC} ${SCHEMA_TARGETS})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCore>)
list(APPEND MNN_TARGETS MNNCore)

# CV
FILE(GLOB MNN_CV_SRC ${CMAKE_CURRENT_LIST_DIR}/source/cv/*.cpp ${CMAKE_CURRENT_LIST_DIR}/source/cv/*.c)
add_library(MNNCV OBJECT ${MNN_CV_SRC} ${SCHEMA_TARGETS})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCV>)
list(APPEND MNN_TARGETS MNNCV)

# Math
FILE(GLOB MNN_Math_SRC ${CMAKE_CURRENT_LIST_DIR}/source/math/*.cpp ${CMAKE_CURRENT_LIST_DIR}/source/math/*.c)
add_library(MNNMath OBJECT ${MNN_Math_SRC} ${SCHEMA_TARGETS})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMath>)
list(APPEND MNN_TARGETS MNNMath)

# Shape
FILE(GLOB MNN_Shape_SRC ${CMAKE_CURRENT_LIST_DIR}/source/shape/*.cpp ${CMAKE_CURRENT_LIST_DIR}/source/shape/*.c)
add_library(MNNShape OBJECT ${MNN_Shape_SRC} ${SCHEMA_TARGETS})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNShape>)
list(APPEND MNN_TARGETS MNNShape)

# CPU
FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/*.cpp)
add_library(MNNCPU OBJECT ${MNN_CPU_SRC} ${SCHEMA_TARGETS})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCPU>)
list(APPEND MNN_TARGETS MNNCPU)

# Compute
FILE(GLOB MNN_Compute_SRC ${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/compute/*.cpp)
add_library(MNNCompute OBJECT ${MNN_Compute_SRC} ${SCHEMA_TARGETS})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCompute>)
list(APPEND MNN_TARGETS MNNCompute)

# Include sub components
## add_subdirectory() avoids recompilation if an option was toggled
## However due to variable scope issues, add the following two lines to the end of sub-scope CMakeLists
### SET(MNN_OBJECTS_TO_LINK "${MNN_OBJECTS_TO_LINK}" PARENT_SCOPE)
### SET(MNN_TARGETS "${MNN_TARGETS}" PARENT_SCOPE)

# X86_64 AVX/SSE
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/x86_x64/)

# AArch32/64 Assemblies
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/arm/)

# Metal
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/metal/)

# Vulkan
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/vulkan/)

# OpenCL
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/opencl/)

# OpenGL
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/opengl/)

# ARM82 Assemblies
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/backend/arm82/)

# Express
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/express/)

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

IF(MNN_BUILD_SHARED_LIBS)
  IF(MNN_SEP_BUILD)
  # TODO: Find better ways to do this
    IF(MNN_OPENCL)
      list(REMOVE_ITEM MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNOpenCL>)
      add_library(MNN_CL SHARED $<TARGET_OBJECTS:MNNOpenCL> ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp)
    ENDIF()
    IF(MNN_OPENGL)
      list(REMOVE_ITEM MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNOpenGL>)
      add_library(MNN_GL SHARED $<TARGET_OBJECTS:MNNOpenGL> ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp)
      target_link_libraries(MNN_GL PUBLIC GLESv3 EGL)
    ENDIF()
    IF(MNN_VULKAN)
      list(REMOVE_ITEM MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNVulkan>)
      add_library(MNN_Vulkan SHARED $<TARGET_OBJECTS:MNNVulkan> ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp)
    ENDIF()
    IF(MNN_ARM82)
      list(REMOVE_ITEM MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNARM82>)
      add_library(MNN_Arm82 SHARED $<TARGET_OBJECTS:MNNARM82> ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp)
    ENDIF()
    list(REMOVE_ITEM MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNExpress>)
    add_library(MNN_Express SHARED $<TARGET_OBJECTS:MNNExpress> ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp)
  ENDIF()
  add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
  if (WIN32)
    foreach(TARGET ${MNN_TARGETS})
      target_compile_definitions(${TARGET} PRIVATE "-DBUILDING_MNN_DLL")
      target_compile_definitions(${TARGET} INTERFACE "-DUSING_MNN_DLL")
    endforeach()
  endif()
ELSE()
  add_library(MNN STATIC ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS} ${MNN_METALLIB_PATH})
ENDIF()

set_target_properties(MNN PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include/")

if(APPLE)
    IF(MNN_AAPL_FMWK)
      SET_TARGET_PROPERTIES(MNN PROPERTIES FRAMEWORK TRUE)
      SET_TARGET_PROPERTIES(MNN PROPERTIES
          MACOSX_FRAMEWORK_IDENTIFIER com.alibaba.MNN
          MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${PACKAGE_VERSION}
          MACOSX_FRAMEWORK_BUNDLE_VERSION ${PACKAGE_VERSION}
          XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
      )
      SET_TARGET_PROPERTIES(MNN PROPERTIES MACOSX_FRAMEWORK_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/project/ios/MNN/Info.plist)
      IF(DEFINED MNN_METALLIB_PATH)
        message(STATUS "Metal Library Path:${MNN_METALLIB_PATH}")
        SET_TARGET_PROPERTIES(MNN PROPERTIES RESOURCE "${MNN_METALLIB_PATH}")
        SET_SOURCE_FILES_PROPERTIES("${MNN_METALLIB_PATH}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources/)
      ENDIF()
    ENDIF()
    find_library(FOUNDATION Foundation REQUIRED)
    target_link_libraries(MNN PUBLIC ${FOUNDATION})
    IF(MNN_METAL)
      find_library(METAL Metal REQUIRED)
      target_link_libraries(MNN PUBLIC ${METAL})
    ENDIF()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "^Android")
  target_link_libraries(MNN PUBLIC log android m)
elseif(UNIX AND NOT APPLE)
  target_link_libraries(MNN PUBLIC pthread)
else()
endif()
if(MNN_OPENGL)
  target_link_libraries(MNN PUBLIC GLESv3 EGL)
endif()
if (MSVC OR WIN32)
    target_link_options(MNN PRIVATE "/IGNORE:4049,4217")
    foreach(DEPEND ${MNN_DEPEND})
        if (${DEPEND} STREQUAL "MNN")
            continue()
        endif()
        target_link_options(MNN PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
        target_link_libraries(MNN PRIVATE ${DEPEND})
    endforeach()
endif()
set(MNN_DEPS "")
list(APPEND MNN_DEPS MNN)
IF(MNN_BUILD_SHARED_LIBS)
  IF(MNN_SEP_BUILD)
  # TODO: Find better ways to do this
    IF(MNN_OPENCL)
      target_link_libraries(MNN_CL PRIVATE MNN)
      list(APPEND MNN_DEPS MNN_CL)
    ENDIF()
    IF(MNN_OPENGL)
      target_link_libraries(MNN_GL PRIVATE MNN)
      list(APPEND MNN_DEPS MNN_GL)
    ENDIF()
    IF(MNN_VULKAN)
      target_link_libraries(MNN_Vulkan PRIVATE MNN)
      list(APPEND MNN_DEPS MNN_Vulkan)
    ENDIF()
    IF(MNN_ARM82)
        target_link_libraries(MNN_Arm82 PRIVATE MNN)
        list(APPEND MNN_DEPS MNN_Arm82)
    ENDIF()
      target_link_libraries(MNN_Express PRIVATE MNN)
      list(APPEND MNN_DEPS MNN_Express)
  ENDIF()
ENDIF()
if (NOT MNN_BUILD_SHARED_LIBS)
    if(APPLE)
        set(MNN_DEPEND -Wl,-all_load ${MNN_DEPEND} -Wl,-noall_load)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(MNN_DEPEND -Wl,--whole-archive ${MNN_DEPEND} -Wl,--no-whole-archive)
    endif()
endif()

# OpenCL Library
IF(MNN_OPENCL)
  IF(APPLE)
    find_library(OPENCL OpenCL REQUIRED)
    SET(MNN_OCL_LIBS ${OPENCL})
  ELSEIF(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Android")
    find_package(OpenCL REQUIRED)
    SET(MNN_OCL_LIBS ${OpenCL_LIBRARIES})
  ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Android")
      add_definitions(-DMNN_USE_OPENCL_WRAPPER)
      #add_definitions(-DENABLE_OPENCL_TURNING_PROFILER)
      #add_definitions(-DLOG_VERBOSE)
  ENDIF()
  IF(MNN_OCL_LIBS)
    IF(MNN_SEP_BUILD AND MNN_BUILD_SHARED_LIBS)
      target_link_libraries(MNN_CL PUBLIC ${MNN_OCL_LIBS})
    ELSE()
      target_link_libraries(MNN PUBLIC ${MNN_OCL_LIBS})
    ENDIF()
  ENDIF()
ENDIF()

# OpenMP
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
      FOREACH(TARGET ${MNN_DEPS})
       target_link_libraries(${TARGET} PUBLIC ${OpenMP_CXX_LIBRARIES})
       IF(WIN32)
            target_compile_options(${TARGET} PUBLIC /openmp ${OpenMP_CXX_FLAGS} ${OpenMP_C_FLAGS})
       ELSE()
           target_compile_options(${TARGET} PUBLIC ${OpenMP_CXX_FLAGS} ${OpenMP_C_FLAGS})
       ENDIF()
     ENDFOREACH()
    endif()
endif()
list(APPEND MNN_TARGETS MNN)
  FOREACH(TARGET ${MNN_TARGETS})
    add_dependencies(${TARGET} MNN_SCHEMA_GEN)
    IF(NOT MSVC)
      target_compile_options(${TARGET} PUBLIC -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti)
      target_compile_options(${TARGET} PRIVATE -fno-exceptions)
      if(MNN_HIDDEN)
          target_compile_options(${TARGET} PRIVATE -fvisibility-inlines-hidden -fvisibility=hidden)
      endif()
    else()
      add_compile_definitions("_CRT_SECURE_NO_WARNINGS")
      add_compile_options("/wd4267" "/wd4018" "/wd4251" "/wd4996" "/wd4244" "/wd4146" "/wd4129" "/wd4305")
    endif()
  ENDFOREACH()
list(REMOVE_ITEM MNN_TARGETS MNN)
include(${CMAKE_CURRENT_LIST_DIR}/demo/exec/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/tools/cpp/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/tools/train/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/benchmark/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/tools/quantization/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/tools/evaluation/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/tools/converter/CMakeLists.txt)

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
ENDIF()
