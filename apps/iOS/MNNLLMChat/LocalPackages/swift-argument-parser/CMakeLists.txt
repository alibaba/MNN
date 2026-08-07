cmake_minimum_required(VERSION 3.19)

project(swift-argument-parser
  LANGUAGES Swift)

option(BUILD_EXAMPLES "Build Example Programs" TRUE)
option(BUILD_SHARED_LIBS "Build shared libraries by default" YES)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

include(CTest)
include(SwiftSupport)

find_package(dispatch CONFIG)
find_package(Foundation CONFIG)
find_package(XCTest CONFIG)

add_subdirectory(Sources)
if(BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()
if(BUILD_TESTING)
  add_subdirectory(Tests)
endif()

add_subdirectory(cmake/modules)
