
file(GLOB MNN_ARM82_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/compute/*.cpp")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?")
    file(GLOB MNN_ARM82_SRCS_ASM "${CMAKE_CURRENT_LIST_DIR}/asm/arm32/*")
    add_library(MNN_Arm82 OBJECT ${MNN_ARM82_SRCS} ${MNN_ARM82_SRCS_ASM})
    target_compile_options(MNN_Arm82 PRIVATE -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8 -mfloat-abi=softfp -DENABLE_ARMV82)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64")
    file(GLOB MNN_ARM82_SRCS_ASM "${CMAKE_CURRENT_LIST_DIR}/asm/arm64/*")
    if (MNN_LOW_MEMORY)
        file(GLOB MNN_ARM82_SRCS_ASM ${MNN_ARM82_SRCS_ASM} ${CMAKE_CURRENT_LIST_DIR}/asm/arm64/low_memory/*)
    endif()
    add_library(MNN_Arm82 OBJECT ${MNN_ARM82_SRCS} ${MNN_ARM82_SRCS_ASM})
    if (MNN_LOW_MEMORY)
        target_compile_options(MNN_Arm82 PRIVATE -DMNN_LOW_MEMORY)
    endif()
    target_compile_options(MNN_Arm82 PRIVATE -march=armv8.2-a+fp16 -DENABLE_ARMV82)
else()
# Building fat binary requires multiple separate builds and lipo-by-hand under CMake's design
endif()

target_include_directories(MNN_Arm82 PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(MNN_Arm82 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/compute/)
target_include_directories(MNN_Arm82 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/asm/)
