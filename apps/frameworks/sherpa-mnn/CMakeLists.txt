cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version. Used only for macOS")

set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

project(sherpa-mnn)

message(STATUS "MNN's dir: ${MNN_LIB_DIR}")
include_directories(${MNN_LIB_DIR}/include)
link_directories(${MNN_LIB_DIR}/lib)

# Remember to update
# ./CHANGELOG.md
# ./new-release.sh
set(SHERPA_MNN_VERSION "1.10.46")

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

option(SHERPA_MNN_ENABLE_PYTHON "Whether to build Python" OFF)
option(SHERPA_MNN_ENABLE_TESTS "Whether to build tests" OFF)
option(SHERPA_MNN_ENABLE_CHECK "Whether to build with assert" OFF)
option(BUILD_SHARED_LIBS "Whether to build shared libraries" OFF)
option(SHERPA_MNN_ENABLE_PORTAUDIO "Whether to build with portaudio" ON)
option(SHERPA_MNN_ENABLE_JNI "Whether to build JNI internface" OFF)
option(SHERPA_MNN_ENABLE_C_API "Whether to build C API" ON)
option(SHERPA_MNN_ENABLE_WEBSOCKET "Whether to build webscoket server/client" ON)
option(SHERPA_MNN_ENABLE_GPU "Enable ONNX Runtime GPU support" OFF)
option(SHERPA_MNN_ENABLE_DIRECTML "Enable ONNX Runtime DirectML support" OFF)
option(SHERPA_MNN_ENABLE_WASM "Whether to enable WASM" OFF)
option(SHERPA_MNN_ENABLE_WASM_SPEAKER_DIARIZATION "Whether to enable WASM for speaker diarization" OFF)
option(SHERPA_MNN_ENABLE_WASM_TTS "Whether to enable WASM for TTS" OFF)
option(SHERPA_MNN_ENABLE_WASM_ASR "Whether to enable WASM for ASR" OFF)
option(SHERPA_MNN_ENABLE_WASM_KWS "Whether to enable WASM for KWS" OFF)
option(SHERPA_MNN_ENABLE_WASM_VAD "Whether to enable WASM for VAD" OFF)
option(SHERPA_MNN_ENABLE_WASM_VAD_ASR "Whether to enable WASM for VAD+ASR" OFF)
option(SHERPA_MNN_ENABLE_WASM_NODEJS "Whether to enable WASM for NodeJS" OFF)
option(SHERPA_MNN_ENABLE_BINARY "Whether to build binaries" ON)
option(SHERPA_MNN_ENABLE_TTS "Whether to build TTS related code" ON)
option(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION "Whether to build speaker diarization related code" ON)
option(SHERPA_MNN_LINK_LIBSTDCPP_STATICALLY "True to link libstdc++ statically. Used only when BUILD_SHARED_LIBS is OFF on Linux" ON)
option(SHERPA_MNN_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE "True to use pre-installed onnxruntime if available" ON)
option(SHERPA_MNN_ENABLE_SANITIZER "Whether to enable ubsan and asan" OFF)
option(SHERPA_MNN_BUILD_C_API_EXAMPLES "Whether to enable C API examples" ON)
option(SHERPA_MNN_ENABLE_RKNN "Whether to build for RKNN NPU " OFF)

set(SHERPA_MNN_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION "1.11.0" CACHE STRING "Used only for Linux ARM64 GPU. If you use Jetson nano b01, then please set it to 1.11.0. If you use Jetson Orin NX, then set it to 1.16.0.If you use NVIDIA Jetson Orin Nano Engineering Reference Developer Kit
Super - Jetpack 6.2 [L4T 36.4.3], then set it to 1.18.1")


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT WIN32)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(BUILD_RPATH_USE_ORIGIN TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(NOT APPLE)
  set(SHERPA_MNN_RPATH_ORIGIN "$ORIGIN")
else()
  set(SHERPA_MNN_RPATH_ORIGIN "@loader_path")
endif()

if(NOT WIN32)
  set(CMAKE_INSTALL_RPATH ${SHERPA_MNN_RPATH_ORIGIN})
  set(CMAKE_BUILD_RPATH ${SHERPA_MNN_RPATH_ORIGIN})
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ANDROID_ABI AND NOT SHERPA_MNN_ENABLE_JNI AND NOT SHERPA_MNN_ENABLE_C_API)
  message(STATUS "Set SHERPA_MNN_ENABLE_JNI to ON for Android")
  set(SHERPA_MNN_ENABLE_JNI ON CACHE BOOL "" FORCE)
endif()

if(SHERPA_MNN_ENABLE_PYTHON AND NOT BUILD_SHARED_LIBS)
  message(STATUS "Set BUILD_SHARED_LIBS to ON since SHERPA_MNN_ENABLE_PYTHON is ON")
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

if(SHERPA_MNN_ENABLE_GPU)
  message(WARNING "\
Compiling for NVIDIA GPU is enabled. Please make sure cudatoolkit
is installed on your system. Otherwise, you will get errors at runtime.
Hint: You don't need sudo permission to install CUDA toolkit. Please refer to
  https://k2-fsa.github.io/k2/installation/cuda-cudnn.html
to install CUDA toolkit if you have not installed it.")
  if(NOT BUILD_SHARED_LIBS)
    message(STATUS "Set BUILD_SHARED_LIBS to ON since SHERPA_MNN_ENABLE_GPU is ON")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  endif()
endif()

if(SHERPA_MNN_ENABLE_DIRECTML)
  message(WARNING "\
Compiling with DirectML enabled. Please make sure Windows 10 SDK
is installed on your system. Otherwise, you will get errors at runtime.
Please refer to
  https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#requirements
to install Windows 10 SDK if you have not installed it.")
  if(NOT BUILD_SHARED_LIBS)
    message(STATUS "Set BUILD_SHARED_LIBS to ON since SHERPA_MNN_ENABLE_DIRECTML is ON")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  endif()
endif()

# see https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html
# https://stackoverflow.com/questions/14172856/compile-with-mt-instead-of-md-using-cmake
if(MSVC)
  add_compile_options(
      $<$<CONFIG:>:/MT> #---------|
      $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
      $<$<CONFIG:Release>:/MT> #--|
      $<$<CONFIG:RelWithDebInfo>:/MT>
      $<$<CONFIG:MinSizeRel>:/MT>
  )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL OHOS)
  set(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "-Wno-unused-command-line-argument ${CMAKE_C_FLAGS}")
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}")
message(STATUS "SHERPA_MNN_ENABLE_PYTHON ${SHERPA_MNN_ENABLE_PYTHON}")
message(STATUS "SHERPA_MNN_ENABLE_TESTS ${SHERPA_MNN_ENABLE_TESTS}")
message(STATUS "SHERPA_MNN_ENABLE_CHECK ${SHERPA_MNN_ENABLE_CHECK}")
message(STATUS "SHERPA_MNN_ENABLE_PORTAUDIO ${SHERPA_MNN_ENABLE_PORTAUDIO}")
message(STATUS "SHERPA_MNN_ENABLE_JNI ${SHERPA_MNN_ENABLE_JNI}")
message(STATUS "SHERPA_MNN_ENABLE_C_API ${SHERPA_MNN_ENABLE_C_API}")
message(STATUS "SHERPA_MNN_ENABLE_WEBSOCKET ${SHERPA_MNN_ENABLE_WEBSOCKET}")
message(STATUS "SHERPA_MNN_ENABLE_GPU ${SHERPA_MNN_ENABLE_GPU}")
message(STATUS "SHERPA_MNN_ENABLE_WASM ${SHERPA_MNN_ENABLE_WASM}")
message(STATUS "SHERPA_MNN_ENABLE_WASM_SPEAKER_DIARIZATION ${SHERPA_MNN_ENABLE_WASM_SPEAKER_DIARIZATION}")
message(STATUS "SHERPA_MNN_ENABLE_WASM_TTS ${SHERPA_MNN_ENABLE_WASM_TTS}")
message(STATUS "SHERPA_MNN_ENABLE_WASM_ASR ${SHERPA_MNN_ENABLE_WASM_ASR}")
message(STATUS "SHERPA_MNN_ENABLE_WASM_KWS ${SHERPA_MNN_ENABLE_WASM_KWS}")
message(STATUS "SHERPA_MNN_ENABLE_WASM_VAD ${SHERPA_MNN_ENABLE_WASM_VAD}")
message(STATUS "SHERPA_MNN_ENABLE_WASM_VAD_ASR ${SHERPA_MNN_ENABLE_WASM_VAD_ASR}")
message(STATUS "SHERPA_MNN_ENABLE_WASM_NODEJS ${SHERPA_MNN_ENABLE_WASM_NODEJS}")
message(STATUS "SHERPA_MNN_ENABLE_BINARY ${SHERPA_MNN_ENABLE_BINARY}")
message(STATUS "SHERPA_MNN_ENABLE_TTS ${SHERPA_MNN_ENABLE_TTS}")
message(STATUS "SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION ${SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION}")
message(STATUS "SHERPA_MNN_LINK_LIBSTDCPP_STATICALLY ${SHERPA_MNN_LINK_LIBSTDCPP_STATICALLY}")
message(STATUS "SHERPA_MNN_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE ${SHERPA_MNN_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE}")
message(STATUS "SHERPA_MNN_ENABLE_SANITIZER: ${SHERPA_MNN_ENABLE_SANITIZER}")
message(STATUS "SHERPA_MNN_BUILD_C_API_EXAMPLES: ${SHERPA_MNN_BUILD_C_API_EXAMPLES}")
message(STATUS "SHERPA_MNN_ENABLE_RKNN: ${SHERPA_MNN_ENABLE_RKNN}")

if(BUILD_SHARED_LIBS OR SHERPA_MNN_ENABLE_JNI)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_SHARED_LIBS AND NOT CMAKE_SYSTEM_NAME STREQUAL iOS AND CMAKE_BUILD_TYPE STREQUAL Release)
  # Don't use LTO for iOS since it causes the following error
  # error: unable to find any architecture information in the binary
  # at '/Users/fangjun/open-source/sherpa-onnx/build-ios/build/os64/sherpa-onnx.a':
  # Unknown header: 0xb17c0de
  # See also https://forums.developer.apple.com/forums/thread/714324

  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo)
  if(ipo)
    message(STATUS "IPO is enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(STATUS "IPO is not available")
  endif()
endif()

if(SHERPA_MNN_ENABLE_TTS)
  message(STATUS "TTS is enabled")
  add_definitions(-DSHERPA_MNN_ENABLE_TTS=1)
else()
  message(WARNING "TTS is disabled")
  add_definitions(-DSHERPA_MNN_ENABLE_TTS=0)
endif()

if(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
  message(STATUS "speaker diarization is enabled")
  add_definitions(-DSHERPA_MNN_ENABLE_SPEAKER_DIARIZATION=1)
else()
  message(WARNING "speaker diarization is disabled")
  add_definitions(-DSHERPA_MNN_ENABLE_SPEAKER_DIARIZATION=0)
endif()

if(SHERPA_MNN_ENABLE_DIRECTML)
  message(STATUS "DirectML is enabled")
  add_definitions(-DSHERPA_MNN_ENABLE_DIRECTML=1)
else()
  message(STATUS "DirectML is disabled")
  add_definitions(-DSHERPA_MNN_ENABLE_DIRECTML=0)
endif()

if(SHERPA_MNN_ENABLE_WASM_SPEAKER_DIARIZATION)
  if(NOT SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION to ON if you want to build WASM for speaker diarization")
  endif()

  if(NOT SHERPA_MNN_ENABLE_WASM)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_WASM to ON if you enable WASM for speaker diarization")
  endif()
endif()

if(SHERPA_MNN_ENABLE_WASM_TTS)
  if(NOT SHERPA_MNN_ENABLE_TTS)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_TTS to ON if you want to build WASM for TTS")
  endif()

  if(NOT SHERPA_MNN_ENABLE_WASM)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_WASM to ON if you enable WASM for TTS")
  endif()
endif()

if(SHERPA_MNN_ENABLE_WASM_ASR)
  if(NOT SHERPA_MNN_ENABLE_WASM)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_WASM to ON if you enable WASM for ASR")
  endif()
endif()

if(SHERPA_MNN_ENABLE_WASM_NODEJS)
  if(NOT SHERPA_MNN_ENABLE_WASM)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_WASM to ON if you enable WASM for NodeJS")
  endif()
  add_definitions(-DSHERPA_MNN_ENABLE_WASM_KWS=1)
endif()

if(SHERPA_MNN_ENABLE_WASM)
  add_definitions(-DSHERPA_MNN_ENABLE_WASM=1)
endif()

if(SHERPA_MNN_ENABLE_WASM_KWS)
  if(NOT SHERPA_MNN_ENABLE_WASM)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_WASM to ON if you enable WASM for KWS")
  endif()
  add_definitions(-DSHERPA_MNN_ENABLE_WASM_KWS=1)
endif()

if(SHERPA_MNN_ENABLE_WASM_VAD)
  if(NOT SHERPA_MNN_ENABLE_WASM)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_WASM to ON if you enable WASM for VAD")
  endif()
endif()

if(SHERPA_MNN_ENABLE_WASM_VAD_ASR)
  if(NOT SHERPA_MNN_ENABLE_WASM)
    message(FATAL_ERROR "Please set SHERPA_MNN_ENABLE_WASM to ON if you enable WASM for VAD+ASR")
  endif()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ version to be used.")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

include(CheckIncludeFileCXX)

if(SHERPA_MNN_ENABLE_RKNN)
  add_definitions(-DSHERPA_MNN_ENABLE_RKNN=1)
endif()

if(UNIX AND NOT APPLE AND NOT SHERPA_MNN_ENABLE_WASM AND NOT CMAKE_SYSTEM_NAME STREQUAL Android AND NOT CMAKE_SYSTEM_NAME STREQUAL OHOS)
  check_include_file_cxx(alsa/asoundlib.h SHERPA_MNN_HAS_ALSA)
  if(SHERPA_MNN_HAS_ALSA)
    message(STATUS "With Alsa")
    add_definitions(-DSHERPA_MNN_ENABLE_ALSA=1)
  else()
    message(WARNING "\
Could not find alsa/asoundlib.h !
We won't build sherpa-onnx-alsa
To fix that, please do:
  (1) sudo apt-get install alsa-utils libasound2-dev
  (2) rm -rf build
  (3) re-try
  ")
  endif()
endif()

check_include_file_cxx(cxxabi.h SHERPA_MNN_HAVE_CXXABI_H)
check_include_file_cxx(execinfo.h SHERPA_MNN_HAVE_EXECINFO_H)

if(WIN32)
  add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
endif()

if(WIN32 AND MSVC)
  # disable various warnings for MSVC
  # 4244: 'return': conversion from 'unsigned __int64' to 'int', possible loss of data
  # 4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
  # 4305: 'argument': truncation from 'double' to 'const float'
  # 4334: '<<': result of 32-bit shift implicitly converted to 64 bits
  # 4800: 'int': forcing value to bool 'true' or 'false'
  # 4996: 'fopen': This function or variable may be unsafe
  set(disabled_warnings
      /wd4244
      /wd4267
      /wd4305
      /wd4334
      /wd4800
      /wd4996
  )
  message(STATUS "Disabled warnings: ${disabled_warnings}")
  foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
  endforeach()

  add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(SHERPA_MNN_ENABLE_WASM)
  # Enable it for debugging in case there is something wrong.
  # string(APPEND CMAKE_CXX_FLAGS " -g4 -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=1 ")
endif()

if(NOT BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  if(SHERPA_MNN_LINK_LIBSTDCPP_STATICALLY)
    message(STATUS "Link libstdc++ statically")
    set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc ")
  else()
    message(STATUS "Link libstdc++ dynamically")
  endif()
endif()

include(kaldi-native-fbank)
include(kaldi-decoder)
include(simple-sentencepiece)
set(ONNXRUNTIME_DIR ${onnxruntime_SOURCE_DIR})
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

if(SHERPA_MNN_ENABLE_PORTAUDIO AND SHERPA_MNN_ENABLE_BINARY)
  # portaudio is used only in building demo binaries and the sherpa-onnx-core
  # library does not depend on it.
  include(portaudio)
endif()

if(SHERPA_MNN_ENABLE_PYTHON)
  include(pybind11)
endif()

if(SHERPA_MNN_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()

if(SHERPA_MNN_ENABLE_WEBSOCKET)
  include(websocketpp)
  include(asio)
endif()

if(SHERPA_MNN_ENABLE_TTS)
  include(espeak-ng-for-piper)
  set(ESPEAK_NG_DIR ${espeak_ng_SOURCE_DIR})
  message(STATUS "ESPEAK_NG_DIR: ${ESPEAK_NG_DIR}")
  include(piper-phonemize)
  include(cppjieba) # For Chinese TTS. It is a header-only C++ library
endif()

if(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
  include(hclust-cpp)
endif()

# if(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL Debug AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
if(SHERPA_MNN_ENABLE_SANITIZER)
  message(WARNING "enable ubsan and asan")
  set(CMAKE_REQUIRED_LIBRARIES -lubsan -lasan)
  include(CheckCCompilerFlag)

  set(flags -fsanitize=undefined )
  string(APPEND flags " -fno-sanitize-recover=undefined ")
  string(APPEND flags " -fsanitize=integer ")
  string(APPEND flags " -fsanitize=nullability ")
  string(APPEND flags " -fsanitize=implicit-conversion ")
  string(APPEND flags " -fsanitize=bounds ")
  string(APPEND flags " -fsanitize=address ")

  if(OFF)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flags} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flags} -Wall -Wextra")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flags}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flags}")
  endif()

  set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} ${flags}")

  add_compile_options(-fno-omit-frame-pointer)
endif()

add_subdirectory(sherpa-mnn)

if(SHERPA_MNN_ENABLE_C_API AND SHERPA_MNN_ENABLE_BINARY AND SHERPA_MNN_BUILD_C_API_EXAMPLES)
  set(SHERPA_MNN_PKG_WITH_CARGS "-lcargs")
  add_subdirectory(c-api-examples)
  add_subdirectory(cxx-api-examples)
endif()

if(SHERPA_MNN_ENABLE_WASM)
  add_subdirectory(wasm)
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

if(NOT BUILD_SHARED_LIBS)
  if(APPLE)
    set(SHERPA_MNN_PKG_CONFIG_EXTRA_LIBS "-lc++ -framework Foundation")
  endif()

  if(UNIX AND NOT APPLE)
    set(SHERPA_MNN_PKG_CONFIG_EXTRA_LIBS "-lstdc++ -lm -pthread -ldl")
  endif()
endif()

if(NOT BUILD_SHARED_LIBS)
# See https://people.freedesktop.org/~dbn/pkg-config-guide.html
  if(SHERPA_MNN_ENABLE_TTS)
    configure_file(cmake/sherpa-onnx-static.pc.in ${PROJECT_BINARY_DIR}/sherpa-onnx.pc @ONLY)
  else()
    configure_file(cmake/sherpa-onnx-static-no-tts.pc.in ${PROJECT_BINARY_DIR}/sherpa-onnx.pc @ONLY)
  endif()
else()
  configure_file(cmake/sherpa-onnx-shared.pc.in ${PROJECT_BINARY_DIR}/sherpa-onnx.pc @ONLY)
endif()

install(
  FILES
    ${PROJECT_BINARY_DIR}/sherpa-onnx.pc
  DESTINATION
    ./
)
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
