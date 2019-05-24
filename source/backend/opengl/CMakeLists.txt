set(CMAKE_BUILD_TYPE release)

file(GLOB_RECURSE SRCS "*")
include_directories("../../../include/")
include_directories("./execution/")

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
option(GL_BUILD_FOR_ANDROID "Build OpenGL For Android" OFF)
add_definitions(-fvisibility=hidden)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
if(SYSTEM.Android AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_LIBRARY_OUTPUT}/${ANDROID_ABI})
endif()
add_library(
    MNN_GL
    SHARED
    ${SRCS}
)

target_include_directories(MNN_GL PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(MNN_GL
    GLESv3
    android
    EGL
    log
    m
    MNN
    )

