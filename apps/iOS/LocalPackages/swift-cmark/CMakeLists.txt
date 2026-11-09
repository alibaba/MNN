cmake_minimum_required(VERSION 3.12)

if(POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif()

if(POLICY CMP0092)
  cmake_policy(SET CMP0092 NEW)
endif()

project(cmark-gfm)
# NOTE: we cannot simply version the project due to the use of an invalid
# version (the infixed `.gfm.`).
set(PROJECT_VERSION 0.29.0.gfm.13)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED YES)
set(CMAKE_C_EXTENSIONS NO)

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

set(CMAKE_INCLUDE_CURRENT_DIR YES)

option(CMARK_FUZZ_QUADRATIC "Build quadratic fuzzing harness" OFF)
option(CMARK_LIB_FUZZER "Build libFuzzer fuzzing harness" OFF)
option(CMARK_THREADING "Add locks around static accesses" OFF)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "Do not build in-source.\nPlease remove CMakeCache.txt and the CMakeFiles/ directory.\nThen: mkdir build ; cd build ; cmake .. ; make")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
  "Choose the type of build, options are: Debug Profile Release Asan Ubsan." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

include(CheckFileOffsetBits)
include(CTest)
include(FindAsan)
if(CMARK_THREADING)
  set(THREADS_PREFER_PTHREAD_FLAG YES)
  include(FindThreads)
endif()
include(GNUInstallDirs)

if(NOT MSVC OR CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS ON)
  include(InstallRequiredSystemLibraries)
endif()

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/W4>)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/WX>)
  # FIXME(compnerd) why do we diverge from upstream?
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/wd5105>)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/wd4706>)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/wd4221>)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/wd4204>)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/wd4100>)
  add_compile_definitions($<$<COMPILE_LANGUAGE:C>:_CRT_SECURE_NO_WARNINGS>)
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES Clang)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wextra>)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wno-unused-parameter>)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-pedantic>)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Ubsan")
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-fsanitize=undefined>)
endif()

# Check integrity of node structure when compiled as debug
add_compile_definitions($<$<CONFIG:Debug>:CMARK_DEBUG_NODES>)
# FIXME(compnerd) why do we not use `!defined(NDEBUG)`?
add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
# Use CMake's generated headers instead of the Swift package prebuilt ones
add_compile_definitions(CMARK_USE_CMAKE_HEADERS)

add_compile_options($<$<AND:$<CONFIG:PROFILE>,$<COMPILE_LANGUAGE:C>>:-pg>)

if(CMARK_LIB_FUZZER)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-fsanitize-coverage=trace-pc-guard>)
endif()

if(CMARK_FUZZ_QUADRATIC)
  # FIXME(compnerd) why do we enable debug information?
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-g>)
  # FIXME(compnerd) why do we use fuzzer-no-link with a custom variable for the
  # runtime rather than `-fsanitize=fuzzer,address`?
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-fsanitize=fuzzer-no-link,address>)
  add_link_options($<$<COMPILE_LANGUAGE:C>:-fsanitize=address>)
endif()

check_file_offset_bits()

add_subdirectory(src)
add_subdirectory(extensions)
# TODO(compnerd) should this be enabled for MinGW, which sets CMAKE_SYSTEM_NAME
# to Windows, but defines `MINGW`.
if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  add_subdirectory(man)
endif()
if(BUILD_TESTING)
  add_subdirectory(test testdir)
  add_subdirectory(api_test)
endif()
if(CMARK_FUZZ_QUADRATIC)
  add_subdirectory(fuzz)
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(cmark-gfm-config.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/modules/cmark-gfm-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
install(FILES
  ${CMAKE_BINARY_DIR}/cmake/modules/cmark-gfm-config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)

