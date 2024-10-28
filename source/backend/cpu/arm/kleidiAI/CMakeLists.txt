#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

project(MNN_KleidiAI
    LANGUAGES C CXX ASM
)

set(KLEIDIAI_MIN_CLANG_VERSION 11)
set(KLEIDIAI_MIN_GNU_VERSION 11)

if(CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_C_COMPILER_VERSION VERSION_LESS ${KLEIDIAI_MIN_CLANG_VERSION})
    message(WARNING "KleidiAI: Using non-supported Clang version. Expected ${KLEIDIAI_MIN_CLANG_VERSION} or newer, received ${CMAKE_C_COMPILER_VERSION}.")
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS ${KLEIDIAI_MIN_GNU_VERSION})
    message(WARNING "KleidiAI: Using non-supported GCC version. Expected ${KLEIDIAI_MIN_GNU_VERSION} or newer, received ${CMAKE_C_COMPILER_VERSION}.")
endif()

list(APPEND MNN_KleidiAI_SOURCES ${CMAKE_CURRENT_LIST_DIR}/mnn_kleidiai.cpp)
list(APPEND MNN_KleidiAI_HEADERS ${CMAKE_CURRENT_LIST_DIR}/mnn_kleidiai.h)

add_library(
    MNN_KleidiAI
    SHARED
    ${MNN_KleidiAI_SOURCES} ${MNN_KleidiAI_HEADERS}
)

set(KLEIDIAI_SRC ${CMAKE_CURRENT_LIST_DIR})

include_directories(
    ${KLEIDIAI_SRC}/
    ${KLEIDIAI_SRC}/kai/
    ${KLEIDIAI_SRC}/kai/ukernels/
    ${KLEIDIAI_SRC}/kai/ukernels/matmul/
    ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/
    ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/)

set(KLEIDIAI_FILES_SCALAR
    ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
    ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c
)

set(KLEIDIAI_FILES_NEON_DOTPROD
    ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c
)

set(KLEIDIAI_FILES_NEON_I8MM
    ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c
)

# Selectively enable architecture features.
target_sources(MNN_KleidiAI PRIVATE ${KLEIDIAI_FILES_SCALAR})
if((CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64") AND NOT MSVC)
    target_sources(MNN_KleidiAI PRIVATE ${KLEIDIAI_FILES_NEON_DOTPROD})
    target_sources(MNN_KleidiAI PRIVATE ${KLEIDIAI_FILES_NEON_I8MM})

    set_source_files_properties(${KLEIDIAI_FILES_SCALAR}       PROPERTIES COMPILE_OPTIONS -march=armv8-a)
    set_source_files_properties(${KLEIDIAI_FILES_NEON_DOTPROD} PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+dotprod)
    set_source_files_properties(${KLEIDIAI_FILES_NEON_I8MM}    PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+i8mm)
endif()