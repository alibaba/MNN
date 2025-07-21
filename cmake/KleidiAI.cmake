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

    set(KLEIDIAI_COMMIT_SHA "v1.13.0")
    set(KLEIDIAI_DOWNLOAD_URL "https://gitlab.arm.com/kleidi/kleidiai/-/archive/${KLEIDIAI_COMMIT_SHA}/kleidiai-${KLEIDIAI_COMMIT_SHA}.tar.gz")
    set(KLEIDIAI_ARCHIVE_MD5 "7b73541c7ed442541b35e94725b2fd1f")

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

    file(GLOB kleidiai_pack_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c"
    )
    list(APPEND MNN_SOURCES_KLEIDIAI ${kleidiai_pack_sources})

    file(GLOB matmul_clamp_f32_qai8dxp_qsi4cxp_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/*dotprod.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/*i8mm.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/*dotprod_asm.S"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/*i8mm_asm.S"
    )
    list(APPEND MNN_SOURCES_KLEIDIAI ${matmul_clamp_f32_qai8dxp_qsi4cxp_sources})

    file(GLOB matmul_clamp_f16_qsi8d32p_qai4c32p_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/*dotprod.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/*i8mm.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/*dotprod_asm.S"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/*i8mm_asm.S"
    )
    list(APPEND MNN_SOURCES_KLEIDIAI ${matmul_clamp_f16_qsi8d32p_qai4c32p_sources})

    file(GLOB matmul_clamp_f32_qsi8d32p_qai4c32p_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/*dotprod.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/*i8mm.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/*dotprod_asm.S"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/*i8mm_asm.S"
    )
    list(APPEND MNN_SOURCES_KLEIDIAI ${matmul_clamp_f32_qsi8d32p_qai4c32p_sources})

    file(GLOB sme_pack_sources
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/*_sme.c
        ${KLEIDIAI_SRC_DIR}/kai/*_sme_asm.S
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/*_sme_asm.S
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon.c
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s1s0_f32_f32_f32_neon.c

    )
    list(APPEND KLEIDIAI_FILES_SME2 ${sme_pack_sources})

    file(GLOB matmul_clamp_f32_f32p_f32p_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/*mopa.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/*mopa_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${matmul_clamp_f32_f32p_f32p_sources})

    file(GLOB matmul_clamp_f32_f32_f32p_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/*sme2_mla.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/*sme2_mla_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${matmul_clamp_f32_f32_f32p_sources})

    file(GLOB matmul_clamp_f16_f16p_f16p_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/*mopa.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/*mopa_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${matmul_clamp_f16_f16p_f16p_sources})

    file(GLOB matmul_clamp_f16_f16_f16p_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/*sme2_dot.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/*sme2_dot_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${matmul_clamp_f16_f16_f16p_sources})

    file(GLOB matmul_clamp_f32_qai8dxp_qsi4cxp_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/*mopa.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/*mopa_asm.S"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/*sdot.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/*sdot_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${matmul_clamp_f32_qai8dxp_qsi4cxp_sources})

    file(GLOB imatmul_clamp_f32_f32p_f32p_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/*mopa.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/*mopa_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${imatmul_clamp_f32_f32p_f32p_sources})

    file(GLOB imatmul_clamp_f16_f16p_f16p_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/*mopa.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/*mopa_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${imatmul_clamp_f16_f16p_f16p_sources})

    file(GLOB matmul_clamp_f16_qsi8d32p_qai4c32p_sme2_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/*mopa.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/*mopa_asm.S"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/*dot.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/*dot_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${matmul_clamp_f16_qsi8d32p_qai4c32p_sme2_sources})

    file(GLOB matmul_clamp_f32_qsi8d32p_qai4c32p_sme2_sources
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/*mopa.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/*mopa_asm.S"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/*dot.c"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/*dot_asm.S"
    )
    list(APPEND KLEIDIAI_FILES_SME2 ${matmul_clamp_f32_qsi8d32p_qai4c32p_sme2_sources})

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
