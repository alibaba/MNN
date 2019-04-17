cmake_minimum_required(VERSION 3.6)
if(POLICY CMP0065)
    cmake_policy(SET CMP0065 NEW)
endif()

set(MNN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
include_directories(${MNN_ROOT_DIR}/include)

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(MNN_VULKAN ON CACHE BOOL "" FORCE)
set(MNN_OPENCL ON CACHE BOOL "" FORCE)
set(MNN_OPENGL ON CACHE BOOL "" FORCE)
set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_FOR_ANDROID_COMMAND ON CACHE BOOL "" FORCE)
add_definitions(-DMNN_USE_LOGCAT)
add_subdirectory(${MNN_ROOT_DIR}/ ${CMAKE_CURRENT_SOURCE_DIR}/build)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -flax-vector-conversions")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions -flax-vector-conversions")
set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -Wl,--gc-sections")

file(GLOB_RECURSE CPP_SRCS src/main/jni/*.cpp )
add_library( mnncore SHARED ${CPP_SRCS} )

find_library( log-lib log )
find_library( jnigraphics-lib jnigraphics )

target_link_libraries( mnncore MNN MNN_CL MNN_Vulkan MNN_GL ${log-lib} ${jnigraphics-lib})


