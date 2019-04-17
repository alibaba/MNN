cmake_minimum_required(VERSION 2.8)
project(MNN_Arm82)
file(GLOB SRCS "*.hpp" "*.cpp")
include_directories("../../include/")
include_directories("../../source/core")
include_directories("../../source/backend/cpu")
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
enable_language(ASM)
option(MNN_OPENMP "Enable Multiple Thread Linux|Android" ON)
if(SYSTEM.Android)
    if(MNN_OPENMP)
        FIND_PACKAGE(OpenMP REQUIRED)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
        file(GLOB SRCS_ASM "asm/arm32/*")
        add_definitions(-mfloat-abi=softfp -mfpu=neon)
        add_definitions(-march=armv8.2a+fp16)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
        add_definitions(-march=armv8.2a+fp16)
        file(GLOB SRCS_ASM "asm/arm64/*")
    endif()
    add_definitions(-DMNN_BUILD_FOR_ANDROID)
else()
    file(GLOB SRCS_ASM "asm/arm32/*" "asm/arm64/*")
endif()


if(SYSTEM.Android AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_LIBRARY_OUTPUT}/${ANDROID_ABI})
endif()
add_library(
    MNN_Arm82
    SHARED
    ${SRCS}
    ${SRCS_ASM}
)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
add_definitions(-fvisibility=hidden)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

target_link_libraries(MNN_Arm82
    MNN
    )

