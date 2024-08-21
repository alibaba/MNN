list(APPEND MNN_KleidiAI_SOURCES ${CMAKE_CURRENT_LIST_DIR}/MNNKleidiAI.cpp)
list(APPEND MNN_KleidiAI_HEADERS ${CMAKE_CURRENT_LIST_DIR}/MNNKleidiAI.h)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CMAKE_CURRENT_LIST_DIR}/kai/ukernels/
    ${CMAKE_CURRENT_LIST_DIR}/kai/ukernels/matmul/
    ${CMAKE_CURRENT_LIST_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/
    ${CMAKE_CURRENT_LIST_DIR}/kai/ukernels/matmul/pack/)

list(APPEND MNN_KleidiAI_SOURCES ${CMAKE_CURRENT_LIST_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c)
list(APPEND MNN_KleidiAI_SOURCES ${CMAKE_CURRENT_LIST_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0.c)
list(APPEND MNN_KleidiAI_SOURCES ${CMAKE_CURRENT_LIST_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c)
list(APPEND MNN_KleidiAI_SOURCES ${CMAKE_CURRENT_LIST_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c)


add_library(
    MNN_KleidiAI
    SHARED
    ${MNN_KleidiAI_SOURCES} ${MNN_KleidiAI_HEADERS}
)

# Enable ARMv8.6-A features
target_compile_definitions(MNN_KleidiAI PRIVATE
    __ARM_FEATURE_MATMUL_INT8
    __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    __ARM_FEATURE_BF16_SCALAR_ARITHMETIC
    __ARM_BF16_FORMAT_ALTERNATIVE
    __ARM_FEATURE_DOTPROD
)

target_compile_options(MNN_KleidiAI
    PRIVATE -march=armv8.6-a
)