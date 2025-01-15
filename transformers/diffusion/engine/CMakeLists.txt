cmake_minimum_required(VERSION 3.10)

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