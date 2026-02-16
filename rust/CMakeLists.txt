cmake_minimum_required(VERSION 3.15)
project(mnn_rust_c)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MNN root directory
set(MNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# C FFI wrapper static library
add_library(mnn_c STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mnn_c.cpp
)

target_include_directories(mnn_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${MNN_DIR}/include
    ${MNN_DIR}/transformers/llm/engine/include
    ${MNN_DIR}/transformers/llm/engine/src
    ${MNN_DIR}/express
    ${MNN_DIR}/3rd_party/flatbuffers/include
    ${MNN_DIR}/3rd_party/half
    ${MNN_DIR}/3rd_party
    ${MNN_DIR}/tools
    ${MNN_DIR}/source
)

target_compile_definitions(mnn_c PRIVATE LLM_USE_MINJA)

# Platform-specific settings
if(APPLE)
    target_compile_options(mnn_c PRIVATE -stdlib=libc++)
endif()

# LLM sources for static linking
file(GLOB_RECURSE LLM_SRCS ${MNN_DIR}/transformers/llm/engine/src/*.cpp)
target_sources(mnn_c PRIVATE ${LLM_SRCS})

# Link with MNN library
set(MNN_BUILD_DIR ${MNN_DIR}/build)

if(APPLE)
    set(MNN_LIB ${MNN_BUILD_DIR}/libMNN.dylib)
elseif(WIN32)
    set(MNN_LIB ${MNN_BUILD_DIR}/MNN.lib)
else()
    set(MNN_LIB ${MNN_BUILD_DIR}/libMNN.so)
endif()

target_link_libraries(mnn_c PUBLIC ${MNN_LIB})

# Install the static library
install(TARGETS mnn_c
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES csrc/mnn_c.h DESTINATION include)
