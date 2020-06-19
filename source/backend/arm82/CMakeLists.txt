
file(GLOB MNN_ARM82_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

set(COMPILE_ARM64 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR IOS_ARCH STREQUAL "arm64")
    set(COMPILE_ARM64 ON)
endif()

if(COMPILE_ARM64)
    file(GLOB MNN_ARM82_SRCS_ASM "${CMAKE_CURRENT_LIST_DIR}/asm/arm64/*")
else()
    set(MNN_ARM82_SRCS_ASM "")
endif()

IF(MNN_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    endif()
    add_library(
        MNN_Arm82
        SHARED
        ${MNN_ARM82_SRCS}
        ${MNN_ARM82_SRCS_ASM}
        )
    target_link_libraries(MNN_Arm82 MNN)
ELSE()
    add_library(
        MNN_Arm82
        OBJECT
        ${MNN_ARM82_SRCS}
        ${MNN_ARM82_SRCS_ASM}
        )
ENDIF()

if(COMPILE_ARM64)
    target_compile_options(MNN_Arm82 PRIVATE -march=armv8.2-a+fp16)
endif()

target_include_directories(MNN_Arm82 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/asm/)
