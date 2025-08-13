# ------------------------------------------------------------------------------
# Function: download_kleidiai_and_collect_sources
#
# Description:
#   Downloads the KleidiAI source code and collects relevant source files.
#   If the download fails, the function will not terminate the configuration
#   process but will return empty lists for the source files.
#
# Exposed Variables (via PARENT_SCOPE):
#   - MNN_SOURCES_KLEIDIAI : List of general KleidiAI source files.
#   - KLEIDIAI_FILES_SME2  : List of KleidiAI source files specific to SME2 architecture.
#
# Usage:
#   include(KleidiAI.cmake)
#   download_kleidiai_and_collect_sources()
#   Use MNN_SOURCES_KLEIDIAI and KLEIDIAI_FILES_SME2 in subsequent build steps.
# ------------------------------------------------------------------------------
function (download_kleidiai_and_collect_sources)
    set(MNN_SOURCES_KLEIDIAI "" PARENT_SCOPE)
    set(KLEIDIAI_FILES_SME2 "" PARENT_SCOPE)

    # Disable the KleidiAI tests
    set(KLEIDIAI_BUILD_TESTS OFF)

    set(KLEIDIAI_COMMIT_SHA "v1.9.0")
    set(KLEIDIAI_DOWNLOAD_URL "https://gitlab.arm.com/kleidi/kleidiai/-/archive/${KLEIDIAI_COMMIT_SHA}/kleidiai-${KLEIDIAI_COMMIT_SHA}.tar.gz")
    set(KLEIDIAI_ARCHIVE_MD5 "e4c9fcb5de397ba3532d593672d56e95")

    set(_kleidiai_src_dir "")
    if(DEFINED KLEIDIAI_SRC_DIR AND EXISTS "${KLEIDIAI_SRC_DIR}")
        set(_kleidiai_src_dir "${KLEIDIAI_SRC_DIR}")
    else()
        set(_deps_dir "${CMAKE_BINARY_DIR}/_deps")
        file(MAKE_DIRECTORY "${_deps_dir}")

        set(_tar_path "${_deps_dir}/kleidiai-${KLEIDIAI_COMMIT_SHA}.tar.gz")

        file(
            DOWNLOAD "${KLEIDIAI_DOWNLOAD_URL}" "${_tar_path}"
            STATUS _dl_status
            SHOW_PROGRESS
            TIMEOUT 180)
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            message(WARNING "KleidiAI download failed: ${_dl_status}. Building without KleidiAI. "
                            "If you have the source locally, set KLEIDIAI_SRC_DIR to skip downloading.")
            return()
        endif()

        file(MD5 "${_tar_path}" _actual_md5)
        if(NOT _actual_md5 STREQUAL "${KLEIDIAI_ARCHIVE_MD5}")
            message(WARNING "KleidiAI archive hash mismatch. Building without KleidiAI.")
            return()
        endif()

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${_tar_path}"
            WORKING_DIRECTORY "${_deps_dir}"
            RESULT_VARIABLE _untar_code)
        if(NOT _untar_code EQUAL 0)
            message(WARNING "KleidiAI extract failed (code=${_untar_code}). Building without KleidiAI.")
            return()
        endif()

        set(_kleidiai_src_dir "${_deps_dir}/kleidiai-${KLEIDIAI_COMMIT_SHA}")
        if(NOT EXISTS "${_kleidiai_src_dir}/kai")
            message(WARNING "KleidiAI source tree not found at expected location. Building without KleidiAI.")
            return()
        endif()

        set(KLEIDIAI_SRC_DIR
            "${_kleidiai_src_dir}"
            CACHE PATH "Path to KleidiAI source (downloaded or provided)" FORCE)
    endif()

    list(APPEND MNN_SOURCES_KLEIDIAI ${CMAKE_CURRENT_LIST_DIR}/mnn_kleidiai.cpp)
    list(APPEND MNN_SOURCES_KLEIDIAI ${CMAKE_CURRENT_LIST_DIR}/mnn_kleidiai_util.cpp)

    include_directories(
        ${KLEIDIAI_SRC_DIR}/
        ${KLEIDIAI_SRC_DIR}/kai/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/)

    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_asm.S)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_asm.S)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_asm.S)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_asm.S)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.c)

    set(KLEIDIAI_FILES_SME2
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.c
    )

    set_source_files_properties(
        ${MNN_SOURCES_KLEIDIAI}
        PROPERTIES COMPILE_OPTIONS
            "-fno-tree-vectorize;-march=armv8.2-a+i8mm+dotprod+sve+sve2+fp16")
    set_source_files_properties(
        ${KLEIDIAI_FILES_SME2}
        PROPERTIES COMPILE_OPTIONS
                   "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2")

    set(MNN_SOURCES_KLEIDIAI "${MNN_SOURCES_KLEIDIAI}" PARENT_SCOPE)
    set(KLEIDIAI_FILES_SME2 "${KLEIDIAI_FILES_SME2}" PARENT_SCOPE)

    # Define macro to indicate KleidiAI is enabled
    add_definitions(-DMNN_KLEIDIAI_ENABLED=1)
endfunction()
