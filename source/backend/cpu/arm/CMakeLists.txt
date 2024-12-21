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

if (MNN_KLEIDIAI)
    add_definitions(-DMNN_KLEIDIAI_ENABLED=1)
    # Disable the KleidiAI tests
    set(KLEIDIAI_BUILD_TESTS  OFF)
    # Fetch KleidiAI sources:
    include(FetchContent)
    set(KLEIDIAI_COMMIT_SHA "v0.5.0")
    set(KLEIDIAI_DOWNLOAD_URL "${CMAKE_CURRENT_LIST_DIR}/kleidiai-v0.5.0.tar.gz")
    set(KLEIDIAI_ARCHIVE_MD5  "b75f2d3c1bd2a5457091cc3976e2fdde")

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
    list(APPEND MNN_HEADERS_KLEIDIAI ${CMAKE_CURRENT_LIST_DIR}/mnn_kleidiai.h)

    # KleidiAI
    include_directories(
        ${KLEIDIAI_SRC}/
        ${KLEIDIAI_SRC}/kai/
        ${KLEIDIAI_SRC}/kai/ukernels/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/
        ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/)

    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c)
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c)
    list(APPEND MNN_CDEF_PUBLIC MNN_USE_KLEIDIAI)

    set_source_files_properties(${MNN_SOURCES_KLEIDIAI} PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+i8mm+dotprod+sve+sve2+fp16)

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
    add_library(MNNARM64 OBJECT ${MNN_AArch64_SRC} ${MNN_NEON_SRC} ${MNN_SOURCES_KLEIDIAI})
    target_include_directories(MNNARM64 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)
    target_include_directories(MNNARM64 PRIVATE ${MNN_HEADERS_KLEIDIAI})
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
