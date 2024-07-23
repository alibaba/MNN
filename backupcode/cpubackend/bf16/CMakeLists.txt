
file(GLOB MNN_BF16_SRCS "${CMAKE_CURRENT_LIST_DIR}/*")

file(GLOB MNN_BF16_SRCS_ASM "${CMAKE_CURRENT_LIST_DIR}/asm/*")

add_library(
    MNN_BF16
    OBJECT
    ${MNN_BF16_SRCS}
    )
target_compile_options(MNN_BF16 PRIVATE -DMNN_SUPPORT_BF16)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)|(i686)")
    if (MNN_USE_SSE)
        target_compile_options(MNN_BF16 PRIVATE -DMNN_USE_SSE)
        if (MNN_SSE_USE_FP16_INSTEAD)
            target_compile_options(MNN_BF16 PRIVATE -DMNN_SSE_USE_FP16_INSTEAD -mf16c)
        endif()
    endif()
endif()
