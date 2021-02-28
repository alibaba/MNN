
file(GLOB MNN_ARM82_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

set(COMPILE_ARM64 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR IOS_ARCH STREQUAL "arm64")
    set(COMPILE_ARM64 ON)
endif()

file(GLOB MNN_ARM82_SRCS_ASM "${CMAKE_CURRENT_LIST_DIR}/asm/arm64/*")

add_library(
    MNN_Arm82
    OBJECT
    ${MNN_ARM82_SRCS}
    ${MNN_ARM82_SRCS_ASM}
    )

if(COMPILE_ARM64)
    target_compile_options(MNN_Arm82 PRIVATE -march=armv8.2-a+fp16)
endif()

target_include_directories(MNN_Arm82 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/asm/)
