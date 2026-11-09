#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

cmake_minimum_required(VERSION 3.16)
project(SwiftCollections
  LANGUAGES C Swift)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)
set(CMAKE_Swift_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)

if(NOT SWIFT_SYSTEM_NAME)
  if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(SWIFT_SYSTEM_NAME macosx)
  else()
    set(SWIFT_SYSTEM_NAME "$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>")
  endif()
endif()

include(PlatformInfo)

set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)
set(CMAKE_Swift_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)

if(CMAKE_SYSTEM_NAME STREQUAL Windows OR CMAKE_SYSTEM_NAME STREQUAL Darwin)
  option(BUILD_SHARED_LIBS "Build shared libraries by default" YES)
endif()

option(COLLECTIONS_SINGLE_MODULE "Build as a single module" NO)
option(COLLECTIONS_FOUNDATION_TOOLCHAIN_MODULE "Build a module for Foundation in the toolchain" NO)

if(COLLECTIONS_FOUNDATION_TOOLCHAIN_MODULE)
  set(COLLECTIONS_SINGLE_MODULE YES)
endif()

include(CTest)
include(GNUInstallDirs)
include(SwiftSupport)

add_subdirectory(Sources)
# if(BUILD_TESTING)
#   add_subdirectory(Tests)
# endif()
add_subdirectory(cmake/modules)
