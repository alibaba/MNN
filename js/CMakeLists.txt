cmake_minimum_required(VERSION 3.15)
project(mnn_node)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Node.js addon API
execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Remove quotes if present
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

message(STATUS "Node Addon API Dir: ${NODE_ADDON_API_DIR}")

# Source files
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
)

# LLM sources
set(LLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../transformers/llm/engine)
file(GLOB_RECURSE LLM_SRCS ${LLM_DIR}/src/*.cpp)
list(APPEND SRC_FILES ${LLM_SRCS})

# Create shared library
add_library(mnn_node SHARED ${SRC_FILES})

# Set library properties
set_target_properties(mnn_node PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    CXX_STANDARD 17
)

# Include directories
target_include_directories(mnn_node PRIVATE
    ${NODE_ADDON_API_DIR}
    ${CMAKE_JS_INC}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../express
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rd_party/flatbuffers/include
    ${LLM_DIR}/include
    ${LLM_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rd_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../tools
    ${CMAKE_CURRENT_SOURCE_DIR}/../source
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rd_party/half
)

add_definitions(-DLLM_USE_MINJA)

# Compiler flags
if(MSVC)
    target_compile_options(mnn_node PRIVATE
        /MT
        /EHsc
        /DNOMINMAX
        /DNAPI_CPP_EXCEPTIONS
    )
else()
    target_compile_options(mnn_node PRIVATE
        -fexceptions
        -fno-strict-aliasing
        -Wall
        -Wno-unused-parameter
        -DNAPI_CPP_EXCEPTIONS
    )
endif()

# Link MNN library
set(MNN_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build)

# Find MNN library
if(APPLE)
    set(MNN_LIB ${MNN_BUILD_DIR}/libMNN.dylib)
elseif(WIN32)
    set(MNN_LIB ${MNN_BUILD_DIR}/MNN.lib)
else()
    set(MNN_LIB ${MNN_BUILD_DIR}/libMNN.so)
endif()

target_link_libraries(mnn_node PRIVATE ${MNN_LIB})

# Platform-specific settings
if(APPLE)
    set_target_properties(mnn_node PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path/../../../build"
    )
elseif(UNIX)
    set_target_properties(mnn_node PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/../build"
    )
endif()

# Install target
install(TARGETS mnn_node DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
