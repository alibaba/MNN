cmake_minimum_required(VERSION 3.10)

# SentencePiece integration
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER "internal" CACHE STRING "" FORCE)

# Force C++17 for sentencepiece compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prevent sentencepiece from installing itself
set(CMAKE_INSTALL_BINDIR bin_spm)
set(CMAKE_INSTALL_LIBDIR lib_spm)
set(CMAKE_INSTALL_INCLUDEDIR include_spm)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/tools/cv/include/)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

if (MNN_SEP_BUILD)
    if (MNN_BUILD_SHARED_LIBS)
        # compile dynamic so, support Linux/Mac
        add_library(diffusion SHARED ${SRCS})
        set_target_properties(diffusion PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        target_link_libraries(diffusion ${MNN_DEPS})
    else()
        add_library(diffusion STATIC ${SRCS})
    endif()
    list(APPEND MNN_DEPS diffusion)
else()
    add_library(diffusion OBJECT ${SRCS})
endif()

add_executable(diffusion_demo ${CMAKE_CURRENT_LIST_DIR}/diffusion_demo.cpp)
target_link_libraries(diffusion_demo  ${MNN_DEPS})

add_executable(diffusion_sd35_demo ${CMAKE_CURRENT_LIST_DIR}/diffusion_sd35_demo.cpp)
target_link_libraries(diffusion_sd35_demo  ${MNN_DEPS})