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

# Capture the MNN cmake/ directory at include time. Inside a CMake function,
# CMAKE_CURRENT_LIST_DIR resolves to the *caller's* directory, so we must
# snapshot it here (file scope) to always get the MNN repo root regardless of
# whether MNN is the top-level project or an embedded dependency.
set(_MNN_KLEIDIAI_CMAKE_LIST_DIR "${CMAKE_CURRENT_LIST_DIR}")

function (download_kleidiai_and_collect_sources)
    set(MNN_SOURCES_KLEIDIAI "" PARENT_SCOPE)
    set(KLEIDIAI_FILES_SME2 "" PARENT_SCOPE)
    set(MNN_SOURCES_KLEIDIAI "")
    set(KLEIDIAI_FILES_NEON "")
    set(KLEIDIAI_FILES_SME2 "")

    # Disable the KleidiAI tests
    set(KLEIDIAI_BUILD_TESTS OFF)

    set(KLEIDIAI_VERSION "1.32.0")
    set(KLEIDIAI_DOWNLOAD_URL "https://github.com/ARM-software/kleidiai/releases/download/v${KLEIDIAI_VERSION}/kleidiai-v${KLEIDIAI_VERSION}-src.tar.gz")
    set(KLEIDIAI_ARCHIVE_SHA256 "a48d373e7c5d721a0aecbf5ddaf0bdf245bc8064e6b84086f13db0e102b553a6")

    # Anchor managed downloads to MNN's binary directory, not the parent
    # project's CMAKE_BINARY_DIR when MNN is embedded with add_subdirectory().
    set(_deps_dir "${CMAKE_CURRENT_BINARY_DIR}/_deps")
    set(_pinned_src_dir "${_deps_dir}/kleidiai-v${KLEIDIAI_VERSION}")
    set(_managed_source OFF)
    if(DEFINED KLEIDIAI_SRC_DIR)
        get_filename_component(_requested_src_dir "${KLEIDIAI_SRC_DIR}" ABSOLUTE)
        get_property(_src_cache_help CACHE KLEIDIAI_SRC_DIR PROPERTY HELPSTRING)
        get_property(_src_cache_type CACHE KLEIDIAI_SRC_DIR PROPERTY TYPE)
          # Only update dependencies owned by MNN; explicit overrides must satisfy
          # the same required official API contract as the pinned release.
        if(_src_cache_type STREQUAL "PATH"
           AND _src_cache_help STREQUAL "Path to KleidiAI source (downloaded or provided)"
              AND _requested_src_dir STREQUAL "${MNN_KLEIDIAI_DOWNLOADED_SRC_DIR}")
            set(_managed_source ON)
            if(NOT _requested_src_dir STREQUAL _pinned_src_dir)
                message(STATUS "KleidiAI: migrating managed dependency ${KLEIDIAI_SRC_DIR} to v${KLEIDIAI_VERSION}")
            endif()
        endif()
    endif()

    set(_kleidiai_src_dir "")
    if(DEFINED KLEIDIAI_SRC_DIR AND NOT "${KLEIDIAI_SRC_DIR}" STREQUAL "" AND NOT _managed_source)
        set(_kleidiai_src_dir "${_requested_src_dir}")
        if(NOT EXISTS "${_kleidiai_src_dir}/kai/kai_common.h")
            message(FATAL_ERROR "KLEIDIAI_SRC_DIR does not contain a KleidiAI source tree: ${_kleidiai_src_dir}")
        endif()
        message(STATUS "KleidiAI: using source override ${_kleidiai_src_dir}; required official APIs are checked below")
    else()
        file(MAKE_DIRECTORY "${_deps_dir}")
        set(_kleidiai_src_dir "${_pinned_src_dir}")
        if(NOT EXISTS "${_kleidiai_src_dir}/kai/kai_common.h")
            set(_tar_path "${_deps_dir}/kleidiai-v${KLEIDIAI_VERSION}-src.tar.gz")
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

            file(SHA256 "${_tar_path}" _actual_sha256)
            if(NOT _actual_sha256 STREQUAL "${KLEIDIAI_ARCHIVE_SHA256}")
                message(WARNING "KleidiAI archive SHA256 mismatch. Building without KleidiAI.")
                return()
            endif()

            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E tar xzf "${_tar_path}"
                WORKING_DIRECTORY "${_deps_dir}"
                RESULT_VARIABLE _untar_code)
            if(NOT _untar_code EQUAL 0)
                message(WARNING "KleidiAI extract failed (code=${_untar_code}). Building without KleidiAI.")
                return()
            endif()
        endif()

        if(NOT EXISTS "${_kleidiai_src_dir}/kai/kai_common.h")
            message(WARNING "KleidiAI source tree not found at expected location. Building without KleidiAI.")
            return()
        endif()

        set(KLEIDIAI_SRC_DIR
            "${_kleidiai_src_dir}"
            CACHE PATH "Path to KleidiAI source (downloaded or provided)" FORCE)
        set(MNN_KLEIDIAI_DOWNLOADED_SRC_DIR "${_kleidiai_src_dir}"
            CACHE INTERNAL "KleidiAI source directory managed by MNN" FORCE)
        message(STATUS "KleidiAI: using pinned release v${KLEIDIAI_VERSION}")
    endif()
    # Use the resolved path even when a caller supplies a normal (non-cache)
    # variable shadowing an old cache entry.
    set(KLEIDIAI_SRC_DIR "${_kleidiai_src_dir}")

    set(MNN_KLEIDIAI_DIR "${_MNN_KLEIDIAI_CMAKE_LIST_DIR}/../source/backend/cpu/kleidiai")
    if(NOT EXISTS "${MNN_KLEIDIAI_DIR}/KleidiAIConvInt8.cpp")
        # These sources live inside the MNN repo itself (committed alongside this
        # file), so a correctly-resolved path always finds them. Reaching here means
        # the MNN checkout is incomplete/corrupted, not an external dependency issue
        # -- fail loudly instead of silently disabling a feature the user enabled.
        message(FATAL_ERROR "KleidiAI MNN source files not found at ${MNN_KLEIDIAI_DIR}. "
                            "Your MNN checkout looks incomplete; update it to a version that "
                            "includes source/backend/cpu/kleidiai, or pass -DMNN_KLEIDIAI=OFF.")
    endif()
    set(MNN_KLEIDIAI_INTEGRATION_SOURCES
        ${MNN_KLEIDIAI_DIR}/KleidiAIConvolution.cpp
        ${MNN_KLEIDIAI_DIR}/KleidiAIConvolutionDepthwise.cpp
        ${MNN_KLEIDIAI_DIR}/KleidiAIConvInt8.cpp
        ${MNN_KLEIDIAI_DIR}/KleidiAIDenseConvolution.cpp)
    list(APPEND MNN_SOURCES_KLEIDIAI ${MNN_KLEIDIAI_INTEGRATION_SOURCES})

    include_directories(${MNN_KLEIDIAI_DIR})

    include_directories(
        ${KLEIDIAI_SRC_DIR}/
        ${KLEIDIAI_SRC_DIR}/kai/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f16p_qai4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/dwconv/pack/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/dwconv/dwconv_f32_f32_f32p/)

    # Explicit stems from the four MNN integration units and their headers.
    # Do not glob entire kernel families: newer releases contain unrelated
    # kernels that can crash AppleClang 16 and unnecessarily enlarge the build.
    set(_neon_stems
        kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon
        kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon
        kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon
        kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon
        kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod
        kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm)
    set(_sme2_stems
        kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme
        kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme
        kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme
        kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme
        kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme
        kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme
        kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme
        kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme
        kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa
        kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla
        kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa
        kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot
        kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa
        kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa
        kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa
        kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot
        kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme
        kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla)
    foreach(_kind neon sme2)
        if(_kind STREQUAL "neon")
            set(_source_list KLEIDIAI_FILES_NEON)
        else()
            set(_source_list KLEIDIAI_FILES_SME2)
        endif()
        foreach(_stem IN LISTS _${_kind}_stems)
            foreach(_extension .c .h)
                if(NOT EXISTS "${KLEIDIAI_SRC_DIR}/${_stem}${_extension}")
                    message(FATAL_ERROR "KleidiAI source tree is missing a required MNN dependency: ${_stem}${_extension}")
                endif()
            endforeach()
            list(APPEND ${_source_list} "${KLEIDIAI_SRC_DIR}/${_stem}.c")
            # Some official kernels are C-only; collect the exact assembler
            # companion for kernels implemented as C wrappers plus assembly.
            if(EXISTS "${KLEIDIAI_SRC_DIR}/${_stem}_asm.S")
                list(APPEND ${_source_list} "${KLEIDIAI_SRC_DIR}/${_stem}_asm.S")
            endif()
        endforeach()
    endforeach()

    # Official QSI4/QAI4 FP16 MOPA share FP32-converting and native FP16 LHS
    # packers. Do not add an official RHS intermediate packer: MNN constructs
    # the final RHS layout directly for quantization blocks that are multiples of 32.
    set(_f16_mopa_f32_pack "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon")
    set(_f16_mopa_f16_pack "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2")
    set(_f16_mopa_pack_sources "${_f16_mopa_f16_pack}.c" "${_f16_mopa_f16_pack}_asm.S")
    set(_f16_mopa_required_files
        ${_f16_mopa_pack_sources}
        "${_f16_mopa_f32_pack}.c" "${_f16_mopa_f32_pack}.h"
        "${KLEIDIAI_SRC_DIR}/kai/kai_common.h"
        "${KLEIDIAI_SRC_DIR}/kai/kai_common_sme_asm.S"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/kai_matmul.h"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/kai_matmul_types.h"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/kai_matmul_pack_lhs.h"
        "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/kai_matmul_pack_lhs_types.h")

    # The official QSI4 API wrapper depends on these implementation getters
    # and their compute assembler; they are not a source-version fallback.
    set(_qsi4_dir "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p")
    set(_qsi4_api "${_qsi4_dir}/kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s1s0sf16_f16p_4vsx16vs_sme2_mopa")
    set(_qsi4_impl "${_qsi4_dir}/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa")
    set(_qsi4_sme2_sources "${_qsi4_api}.c" "${_qsi4_impl}.c" "${_qsi4_impl}_asm.S")

    # Official QAI4 MOPA and DOT share one RHS layout.
    set(_qai4_dir "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f16p_qai4c32p")
    set(_qai4_api "${_qai4_dir}/kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa")
    set(_qai4_sme2_sources "${_qai4_api}.c" "${_qai4_api}_asm.S")
    set(_qai4_dot_name kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot)
    set(_qai4_dot "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/${_qai4_dot_name}")
    set(_qai4_dot_pack_name kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon)
    set(_qai4_dot_pack "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/${_qai4_dot_pack_name}")

    # These interfaces are mandatory, not optional features. Reject incomplete
    # overrides at configure time instead of silently compiling another path.
    set(_required_files ${_f16_mopa_required_files} ${_qsi4_sme2_sources}
        "${_qsi4_impl}.h" ${_qai4_sme2_sources}
        "${_qai4_dot}.c" "${_qai4_dot}_asm.S" "${_qai4_dot_pack}.c")
    foreach(_file IN LISTS _required_files)
        if(NOT EXISTS "${_file}")
            message(FATAL_ERROR "KleidiAI: required official kernel file missing: ${_file}. "
                "Use the pinned v${KLEIDIAI_VERSION} release or a source tree providing all required APIs.")
        endif()
    endforeach()
    foreach(_api "${_qsi4_api}" "${_qai4_api}" "${_qai4_dot}")
        get_filename_component(_api_name "${_api}" NAME)
        file(STRINGS "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/kai_matmul.h" _declaration
            REGEX "^[ \t]*struct kai_matmul_uker_api[ \t]+${_api_name}[ \t]*[(]void[)][ \t]*;")
        if(NOT _declaration)
            message(FATAL_ERROR "KleidiAI: required official API missing: ${_api_name}. Use v${KLEIDIAI_VERSION} or a compatible source tree.")
        endif()
    endforeach()
    foreach(_api_name kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2 ${_qai4_dot_pack_name})
        file(STRINGS "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/kai_matmul_pack_lhs.h" _declaration
            REGEX "^[ \t]*struct kai_matmul_pack_lhs_uker_api[ \t]+${_api_name}[ \t]*[(]void[)][ \t]*;")
        if(NOT _declaration)
            message(FATAL_ERROR "KleidiAI: required official pack API missing: ${_api_name}. Use v${KLEIDIAI_VERSION} or a compatible source tree.")
        endif()
    endforeach()
    list(APPEND KLEIDIAI_FILES_NEON "${_f16_mopa_f32_pack}.c" "${_qai4_dot_pack}.c")
    list(APPEND KLEIDIAI_FILES_SME2 "${KLEIDIAI_SRC_DIR}/kai/kai_common_sme_asm.S"
        ${_f16_mopa_pack_sources} ${_qsi4_sme2_sources} ${_qai4_sme2_sources}
        "${_qai4_dot}.c" "${_qai4_dot}_asm.S")
    message(STATUS "KleidiAI: official QSI4/QAI4 FP16 MOPA and shared-RHS DOT enabled; runtime gates select kernels")

    list(REMOVE_DUPLICATES KLEIDIAI_FILES_NEON)
    list(REMOVE_DUPLICATES KLEIDIAI_FILES_SME2)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_FILES_NEON})

    set_source_files_properties(
        ${MNN_KLEIDIAI_INTEGRATION_SOURCES}
        PROPERTIES COMPILE_OPTIONS
            "-fno-tree-vectorize;-march=armv8.2-a+fp16")
    set_source_files_properties(
        ${KLEIDIAI_FILES_NEON}
        PROPERTIES COMPILE_OPTIONS
            "-fno-tree-vectorize;-march=armv8.2-a+i8mm+dotprod+fp16")
    # Selected SME/SME2 assemblers encode SME instructions with .inst. Keep the
    # existing SVE2 compiler baseline (no global +sme/+sme2 requirement), but add
    # FP16 for the official QSI4/QAI4 C wrappers' architectural feature guards.
    set_source_files_properties(
        ${KLEIDIAI_FILES_SME2}
        PROPERTIES COMPILE_OPTIONS
                   "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2+fp16")

    set(MNN_SOURCES_KLEIDIAI "${MNN_SOURCES_KLEIDIAI}" PARENT_SCOPE)
    set(KLEIDIAI_FILES_SME2 "${KLEIDIAI_FILES_SME2}" PARENT_SCOPE)

    # Define macro to indicate KleidiAI is enabled (only on aarch64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        add_definitions(-DMNN_KLEIDIAI_ENABLED=1)
    endif()
endfunction()
