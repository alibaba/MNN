cmake_minimum_required(VERSION 2.8)
project(MNN_Vulkan)
file(GLOB_RECURSE SRCS *.hpp *.cpp)
include_directories("../../../include/")
include_directories("include/")
include_directories("component/")
include_directories("runtime")
include_directories("execution")
include_directories("backend")

if(SYSTEM.Android AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_LIBRARY_OUTPUT}/${ANDROID_ABI})
endif()
add_library(
    MNN_Vulkan
    SHARED
    ${SRCS}
)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions")

target_include_directories(MNN_Vulkan PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(MNN_Vulkan
    m
    MNN
    )

if (SYSTEM.Android)
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
    target_link_libraries(MNN_Vulkan android log)
endif()

