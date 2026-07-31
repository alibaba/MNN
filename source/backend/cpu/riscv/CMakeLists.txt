IF(NOT DEFINED ARCHS)
  set(ARCHS ${CMAKE_SYSTEM_PROCESSOR})
ENDIF()

FILE(GLOB MNN_RVV_SRC ${CMAKE_CURRENT_LIST_DIR}/rvv/*.cpp)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64" OR ARCHS STREQUAL "riscv64")
    message(STATUS "Enabling RVV Optimizations")
    set(MNN_RVV_MARCH "rv64gcv" CACHE STRING "RISC-V ISA string used for RVV optimized kernels")
    set(MNN_RVV_MCPU "" CACHE STRING "Optional RISC-V CPU name used to tune RVV optimized kernels")
    set(MNN_RVV_MTUNE "" CACHE STRING "Optional RISC-V CPU tune name used for RVV optimized kernels")
    option(MNN_RVV_FAST_MATH "Enable fast-math for RVV optimized kernels" OFF)
    option(MNN_RVV_SPACEMIT_IME2 "Enable SpacemiT K3 IME2 kernels" OFF)

    # Keep vendor-only ISA extensions out of the standard RVV objects, including
    # when an existing build cache still has the old combined ISA string.
    string(REPLACE "_xsmtvdotii" "" MNN_RVV_BASE_MARCH "${MNN_RVV_MARCH}")
    if (NOT MNN_RVV_BASE_MARCH STREQUAL MNN_RVV_MARCH)
        message(STATUS "Restricting xsmtvdotii to the SpacemiT IME2 target")
    endif()

    if (MNN_RVV_SPACEMIT_IME2)
        # The IME2 target supplies the mutually exclusive K3 registration
        # entry point. Standard RVV kernels stay in MNNRVV as fallbacks.
        list(REMOVE_ITEM MNN_RVV_SRC
            ${CMAKE_CURRENT_LIST_DIR}/rvv/MNNRvvFastPathRegistration.cpp
        )
    endif()

    add_library(MNNRVV OBJECT ${MNN_RVV_SRC})
    target_include_directories(MNNRVV PRIVATE ${CMAKE_CURRENT_LIST_DIR}/rvv/)
    target_compile_definitions(MNNCPU PRIVATE MNN_USE_RVV)
    target_compile_definitions(MNNRVV PRIVATE MNN_USE_RVV)
    target_compile_options(MNNRVV PRIVATE -march=${MNN_RVV_BASE_MARCH} -mabi=lp64d)
    if (MNN_RVV_MCPU)
        target_compile_options(MNNRVV PRIVATE -mcpu=${MNN_RVV_MCPU})
    endif()
    if (MNN_RVV_MTUNE)
        target_compile_options(MNNRVV PRIVATE -mtune=${MNN_RVV_MTUNE})
    endif()
    if (MNN_RVV_FAST_MATH)
        target_compile_options(MNNRVV PRIVATE -ffast-math)
    endif()
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNRVV>)
    list(APPEND MNN_TARGETS MNNRVV)

    if (MNN_RVV_SPACEMIT_IME2)
        set(MNN_SPACEMIT_IME2_RUNTIME_SRC
            ${CMAKE_CURRENT_LIST_DIR}/rvv/spacemit_ime2/MNNSpacemitIme2AttentionFunctions.cpp
            ${CMAKE_CURRENT_LIST_DIR}/rvv/spacemit_ime2/MNNSpacemitIme2FastPathRegistration.cpp
        )
        if (MNN_LOW_MEMORY)
            list(APPEND MNN_SPACEMIT_IME2_RUNTIME_SRC
                ${CMAKE_CURRENT_LIST_DIR}/rvv/spacemit_ime2/MNNSpacemitIme2ConvInt8Executor.cpp
            )
        endif()
        set(MNN_SPACEMIT_IME2_SRC
            ${CMAKE_CURRENT_LIST_DIR}/rvv/spacemit_ime2/MNNSpacemitIme2AttentionKernels.cpp
            ${CMAKE_CURRENT_LIST_DIR}/rvv/spacemit_ime2/MNNSpacemitIme2GemmI8I4Local.cpp
            ${CMAKE_CURRENT_LIST_DIR}/rvv/spacemit_ime2/MNNSpacemitIme2GemmInt8.cpp
        )
        set(MNN_SPACEMIT_IME2_MARCH "${MNN_RVV_BASE_MARCH}_xsmtvdotii")

        add_library(MNNSpacemitIme2Runtime OBJECT ${MNN_SPACEMIT_IME2_RUNTIME_SRC})
        target_include_directories(MNNSpacemitIme2Runtime PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/rvv/
            ${CMAKE_CURRENT_LIST_DIR}/rvv/spacemit_ime2/
        )
        target_compile_definitions(MNNSpacemitIme2Runtime PRIVATE MNN_USE_RVV MNN_USE_SPACEMIT_IME2)
        target_compile_options(MNNSpacemitIme2Runtime PRIVATE
            -march=${MNN_RVV_BASE_MARCH}
            -mabi=lp64d
        )
        if (MNN_RVV_MCPU)
            target_compile_options(MNNSpacemitIme2Runtime PRIVATE -mcpu=${MNN_RVV_MCPU})
        endif()
        if (MNN_RVV_MTUNE)
            target_compile_options(MNNSpacemitIme2Runtime PRIVATE -mtune=${MNN_RVV_MTUNE})
        endif()
        if (MNN_RVV_FAST_MATH)
            target_compile_options(MNNSpacemitIme2Runtime PRIVATE -ffast-math)
        endif()

        add_library(MNNSpacemitIme2 OBJECT ${MNN_SPACEMIT_IME2_SRC})
        target_include_directories(MNNSpacemitIme2 PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/rvv/
            ${CMAKE_CURRENT_LIST_DIR}/rvv/spacemit_ime2/
        )
        target_compile_definitions(MNNSpacemitIme2 PRIVATE MNN_USE_RVV MNN_USE_SPACEMIT_IME2)
        target_compile_options(MNNSpacemitIme2 PRIVATE
            -march=${MNN_SPACEMIT_IME2_MARCH}
            -mabi=lp64d
            -fno-stack-protector
        )
        if (MNN_RVV_MCPU)
            target_compile_options(MNNSpacemitIme2 PRIVATE -mcpu=${MNN_RVV_MCPU})
        endif()
        if (MNN_RVV_MTUNE)
            target_compile_options(MNNSpacemitIme2 PRIVATE -mtune=${MNN_RVV_MTUNE})
        endif()
        if (MNN_RVV_FAST_MATH)
            target_compile_options(MNNSpacemitIme2 PRIVATE -ffast-math)
        endif()
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNSpacemitIme2Runtime>)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNSpacemitIme2>)
        list(APPEND MNN_TARGETS MNNSpacemitIme2Runtime)
        list(APPEND MNN_TARGETS MNNSpacemitIme2)
    endif()
else()
    message(WARNING "RVV optimizations are only supported on riscv64 architecture")
endif()
