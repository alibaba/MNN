IF(NOT DEFINED ARCHS)
  set(ARCHS ${CMAKE_SYSTEM_PROCESSOR})
ENDIF()
FILE(GLOB MNN_AArch32_SRC ${CMAKE_CURRENT_LIST_DIR}/arm32/*.[sS])
FILE(GLOB MNN_AArch64_SRC ${CMAKE_CURRENT_LIST_DIR}/arm64/*.[sS])

FILE(GLOB MNN_NEON_SRC ${CMAKE_CURRENT_LIST_DIR}/CommonOptFunctionNeon.cpp)
if (MNN_SUPPORT_BF16)
    FILE(GLOB MNN_NEON_SRC ${MNN_NEON_SRC} ${CMAKE_CURRENT_LIST_DIR}/CommonNeonBF16.cpp)
    FILE(GLOB MNN_AArch32_SRC ${MNN_AArch32_SRC} ${CMAKE_CURRENT_LIST_DIR}/arm32/bf16/*.[sS])
    FILE(GLOB MNN_AArch64_SRC ${MNN_AArch64_SRC} ${CMAKE_CURRENT_LIST_DIR}/arm64/bf16/*.[sS])
endif()

if (MNN_LOW_MEMORY)
    FILE(GLOB MNN_AArch64_SRC ${MNN_AArch64_SRC} ${CMAKE_CURRENT_LIST_DIR}/arm64/low_memory/*.[sS])
endif()

if (MNN_CPU_WEIGHT_DEQUANT_GEMM)
    FILE(GLOB MNN_AArch64_SRC ${MNN_AArch64_SRC} ${CMAKE_CURRENT_LIST_DIR}/arm64/normal_memory/*.[sS])
endif()

if (MNN_KLEIDIAI AND CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64" OR ARCHS STREQUAL "ARM64")
    # Disable the KleidiAI tests
    set(KLEIDIAI_BUILD_TESTS  OFF)
    # Fetch KleidiAI sources:
    include(FetchContent)
    set(KLEIDIAI_COMMIT_SHA "v1.9.0")
    set(KLEIDIAI_DOWNLOAD_URL "https://gitlab.arm.com/kleidi/kleidiai/-/archive/${KLEIDIAI_COMMIT_SHA}/kleidiai-${KLEIDIAI_COMMIT_SHA}.tar.gz")
    set(KLEIDIAI_ARCHIVE_MD5  "e4c9fcb5de397ba3532d593672d56e95")

    if (POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Declare(KleidiAI_Download
        URL ${KLEIDIAI_DOWNLOAD_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP NEW
        URL_HASH MD5=${KLEIDIAI_ARCHIVE_MD5})

    FetchContent_MakeAvailable(KleidiAI_Download)
    FetchContent_GetProperties(KleidiAI_Download
        SOURCE_DIR  KLEIDIAI_SRC
        POPULATED   KLEIDIAI_POPULATED)

    if (NOT KLEIDIAI_POPULATED)
        message(FATAL_ERROR "KleidiAI source downloaded failed.")
    endif()

    list(APPEND MNN_SOURCES_KLEIDIAI ${CMAKE_CURRENT_LIST_DIR}/mnn_kleidiai.cpp)
    list(APPEND MNN_SOURCES_KLEIDIAI ${CMAKE_CURRENT_LIST_DIR}/mnn_kleidiai_util.cpp)

    # KleidiAI
    include_directories(
        ${KLEIDIAI_SRC}/
        ${KLEIDIAI_SRC}/kai/
        ${KLEIDIAI_SRC}/kai/ukernels/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/
	${KLEIDIAI_SRC}/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/
	${KLEIDIAI_SRC}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/)

    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_asm.S)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_asm.S)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_asm.S)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_asm.S)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.c)

    set(KLEIDIAI_FILES_SME2
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.c
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.c
	${KLEIDIAI_SRC}/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.c
	${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.c
	${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.c
	${KLEIDIAI_SRC}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.c
	${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.c
	${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.c
    )

    set_source_files_properties(${MNN_SOURCES_KLEIDIAI} PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+i8mm+dotprod+sve+sve2+fp16")
    set_source_files_properties(${KLEIDIAI_FILES_SME2}  PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2")

endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?")
    message(STATUS "Enabling AArch32 Assemblies")
    add_library(MNNARM32 OBJECT ${MNN_AArch32_SRC} ${MNN_NEON_SRC})
    target_include_directories(MNNARM32 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNARM32>)
    list(APPEND MNN_TARGETS MNNARM32)
    add_definitions(-DMNN_USE_NEON)
    target_compile_options(MNNARM32 PRIVATE -D__arm__)
    if (MNN_SUPPORT_BF16)
        target_compile_options(MNNARM32 PRIVATE -DMNN_SUPPORT_BF16)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64" OR ARCHS STREQUAL "ARM64")
    message(STATUS "Enabling AArch64 Assemblies")
    if (MNN_KLEIDIAI)
       add_definitions(-DMNN_KLEIDIAI_ENABLED=1)
    endif()
    if (MNN_SME2)
        add_definitions(-DMNN_SME2)
        FILE(GLOB MNN_SME2_AArch64_SRC ${MNN_SME2_AArch64_SRC} ${CMAKE_CURRENT_LIST_DIR}/arm64/sme2_asm/*.[sS])
        #set_source_files_properties(${MNN_SME2_AArch64_SRC} PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.6-a+sve+sve2+sme+sme2+fp16")
        set_source_files_properties(${MNN_SME2_SRCS_ASM_FP16} PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+fp16")
    endif()
    add_library(MNNARM64 OBJECT ${MNN_AArch64_SRC} ${MNN_NEON_SRC} ${MNN_SOURCES_KLEIDIAI} ${KLEIDIAI_FILES_SME2} ${MNN_SME2_AArch64_SRC})
    target_include_directories(MNNARM64 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNARM64>)
    list(APPEND MNN_TARGETS MNNARM64)
    add_definitions(-DMNN_USE_NEON)
    target_compile_options(MNNARM64 PRIVATE -D__aarch64__)
    if (MNN_SUPPORT_BF16)
        target_compile_options(MNNARM64 PRIVATE -DMNN_SUPPORT_BF16)
    endif()

else()
# Building fat binary requires multiple separate builds and lipo-by-hand under CMake's design
endif()
