cmake_minimum_required(VERSION 3.18)
project(mnncli_unit_tests LANGUAGES CXX)

# Fetch Catch2
include(FetchContent)

# Try to find Catch2 installed on system first
find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found on system, fetching from GitHub...")
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Build test executable with only required source files
add_executable(model_name_utils_tests
    model_name_utils_case_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/model_name_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/model_sources.cpp
)

# Set C++ standard
set_property(TARGET model_name_utils_tests PROPERTY CXX_STANDARD 17)
set_property(TARGET model_name_utils_tests PROPERTY CXX_STANDARD_REQUIRED ON)

# Include directories - only the minimal needed
target_include_directories(model_name_utils_tests PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)

# Link Catch2
target_link_libraries(model_name_utils_tests PRIVATE Catch2::Catch2WithMain)

# Register tests with CTest
enable_testing()
include(CTest)
include(Catch)
catch_discover_tests(model_name_utils_tests)

# File Utils tests
add_executable(file_utils_tests
    file_utils_case_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/file_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/file_utils_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/cli_config_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/model_name_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/model_sources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/log_utils.cpp
)

set_property(TARGET file_utils_tests PROPERTY CXX_STANDARD 17)
set_property(TARGET file_utils_tests PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(file_utils_tests PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../../frameworks/3rd_party/include
)

# Link Catch2
target_link_libraries(file_utils_tests PRIVATE Catch2::Catch2WithMain)

catch_discover_tests(file_utils_tests)

# Local Model Utils tests
add_executable(local_model_utils_tests
    local_model_utils_case_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/local_model_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/cli_config_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/file_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/file_utils_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/model_name_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/model_sources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/log_utils.cpp
)

set_property(TARGET local_model_utils_tests PROPERTY CXX_STANDARD 17)
set_property(TARGET local_model_utils_tests PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(local_model_utils_tests PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../../frameworks/3rd_party/include
)

# Link Catch2
target_link_libraries(local_model_utils_tests PRIVATE Catch2::Catch2WithMain)

catch_discover_tests(local_model_utils_tests)

