//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural feature check

#include "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 1;  // multiple of vector length
static const size_t kai_nr = 4;  // multiple of vector length
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias_rhs = sizeof(float);

/**
 * Lut to be indexed by i4 resulting in its value in i8 (i.e. -2 = 1110 -> 1111 1110).
 **/

static const int8_t lut[64] = {0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0, 4,  0, 0,  0, 5, 0,
                               0,  0, 6, 0, 0,  0, 7, 0, 0,  0, -8, 0, 0,  0, -7, 0, 0,  0, -6, 0, 0, 0,
                               -5, 0, 0, 0, -4, 0, 0, 0, -3, 0, 0,  0, -2, 0, 0,  0, -1, 0, 0,  0};

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, 32);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa()) == 0);

    const size_t k_internal = kai_k_roundedup(k);

    return m_idx * (k_internal + kai_num_bytes_offset_lhs + kai_num_bytes_multiplier_lhs);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa()) == 0);

    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return n_idx * ((k_internal / 2) + kai_num_bytes_sum_rhs + kai_num_bytes_multiplier_rhs + kai_num_bytes_bias_rhs);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa()) == 0);
    KAI_ASSERT((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa()) == 0);

    return (n_idx * sizeof(float) + m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
    size_t m_in, size_t n_in, size_t k_in, const void* restrict lhs_packed, const void* restrict rhs_packed,
    float* restrict dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));
    KAI_ASSERT(dst_stride_row == n_in * sizeof(float));
    KAI_ASSERT(dst_stride_row == n_in * sizeof(float));
    KAI_ASSERT(n_in > 0);
    KAI_ASSERT(m_in > 0);

    // Constants
    uint64_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
    uint64_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
    uint64_t lhs_stride =
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(mr, k_in);
    uint64_t rhs_stride =
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(nr, k_in);
    uint64_t m_blk = (uint64_t)kai_k_roundedup(k_in) * mr;
    uint64_t n_blk = (uint64_t)kai_k_roundedup(k_in) * nr;
    uint64_t dst_inc = mr * n_in;
    float scalar_bounds[2] = {scalar_min, scalar_max};

    /* ---------------------------------------------------
                  Registers allocations
        x7:  u32 vector length (svls)
        x8:  RHS base address (rhs_base)
        x9:  Destination base address (dst_base)
        x10: LHS pointer (lhs_ptr)
        x11: RHS pointer (rhs_ptr)
        x12: Remaining M elements (m_rem)
        x13: Remaining N elements (n_rem)
        x14: k exit condition (k_cond)
             ZA tile index (l_idx)
        x15: LHS scaling factor pointer (lhs_sf_ptr)
        x16: ZA tile exit condition (l_cnd)
        x17: Destination pointer (dst_ptr)
        x19: Destination outer address (dst_o)
        x20: LHS base address (lhs_base)
    --------------------------------------------------- */
    __asm__ volatile(
        "   .inst 0xd503477f //smstart                       \n"
        "   mov   x19, %[dst]                                \n"
        "   mov   x20, %[lhs]                                \n"
        "   mov   x7, %[lut]                                 \n"
        "   .inst 0xe11f80e0 //ldr zt0, [x7]                 \n"
        "   cntw  x7                                         \n"
        "   ptrue p2.b                                       \n"
        "   ld1rw {z30.s}, p2/Z, [%[scalar_bounds]]          \n"
        "   ld1rw {z31.s}, p2/Z, [%[scalar_bounds], #4]      \n"

        // M loop head
        "   mov     x12, %[m]                                \n"
        "   .inst 0x25ac17e0 //whilelt p0.s, xzr, x12        \n"
        "1:                                                  \n"
        "   mov     x8, %[rhs]                               \n"
        "   mov     x9, x19                                  \n"
        "   mov     x13, %[n]                                \n"
        "   cmp     x7, x12                                  \n"
        "   csel    x16, x7, x12, lt                         \n"
        "   lsl     x16, x16, #2                             \n"

        // N loop head
        "   .inst 0x256d47f0 //whilelt pn8.h, xzr, x13, vlx2 \n"
        "2:                                                  \n"
        "   mov     x10, x20                                 \n"
        "   mov     x11, x8                                  \n"
        "   mov     x17, x9                                  \n"
        "   .inst 0x25ad67f1 //whilelt pn9.s, xzr, x13, vlx4 \n"

        // K loop
        "   .inst 0xc00800ff //zero    {za}                  \n"
        "   add     x14, x10, %[m_blk]                       \n"
        "3:                                                  \n"
        "   .inst 0xa540a144 //ld1w    { z4.s }, p0/z, [x10]         \n"
        "   .inst 0x042a502a //addvl   x10, x10, #1                  \n"
        "   .inst 0xa0402160 //ld1h    { z0.h-z1.h }, pn8/z, [x11]   \n"
        "   .inst 0x042b504b //addvl   x11, x11, #2                  \n"
        "   .inst 0xc08a4008 //luti4   { z8.b - z9.b }, zt0, z0[0]   \n"
        "   .inst 0xc08a402a //luti4   { z10.b - z11.b }, zt0, z1[0] \n"
        "   .inst 0xa0884880 //smopa   za0.s, p2/m, p2/m, z4.b, z8.b \n"
        "   .inst 0xa0894881 //smopa   za1.s, p2/m, p2/m, z4.b, z9.b \n"
        "   .inst 0xa08a4882 //smopa   za2.s, p2/m, p2/m, z4.b, z10.b\n"
        "   .inst 0xa08b4883 //smopa   za3.s, p2/m, p2/m, z4.b, z11.b\n"
        "   cmp     x10, x14                                 \n"
        "   b.lt    3b                                       \n"

        // RHS row sum, scale factor & bias
        "   .inst 0xa040c560 //ld1w    { z0.s-z3.s }, pn9/z, [x11]             \n"
        "   .inst 0xa041c564 //ld1w    { z4.s-z7.s }, pn9/z, [x11, #4, mul vl] \n"
        "   .inst 0xa042c568 //ld1w    { z8.s-z11.s }, pn9/z, [x11, #8, mul vl]\n"
        "   .inst 0x042b518b //addvl   x11, x11, #12               \n"
        "   .inst 0xc132e000 //scvtf   { z0.s-z3.s }, { z0.s-z3.s }\n"

        // Store loop
        "   mov     x14, #0                                  \n"
        "   addvl   x15, x10, #1          \n"
        "4:                                                  \n"
        // Load LHS Row-offset & SF
        "   ld1rw   {z16.s},  p2/z, [x10]  \n"
        "   ld1rw   {z17.s}, p2/z, [x15]   \n"
        "   add     x10, x10, #4           \n"
        "   add     x15, x15, #4           \n"
        "   scvtf   z16.s, p2/m, z16.s     \n"

        // offset x Row-sum
        "   fmul    z24.s, z16.s, z0.s     \n"
        "   fmul    z25.s, z16.s, z1.s     \n"
        "   fmul    z26.s, z16.s, z2.s     \n"
        "   fmul    z27.s, z16.s, z3.s     \n"

        // Scaling factors
        "   fmul    z20.s, z17.s, z4.s    \n"
        "   fmul    z21.s, z17.s, z5.s    \n"
        "   fmul    z22.s, z17.s, z6.s    \n"
        "   fmul    z23.s, z17.s, z7.s    \n"

        // Result = offset x Row-sum x SFs
        "   fmul    z24.s, z24.s, z20.s   \n"
        "   fmul    z25.s, z25.s, z21.s   \n"
        "   fmul    z26.s, z26.s, z22.s   \n"
        "   fmul    z27.s, z27.s, z23.s   \n"

        // Load inner accumulation & convert
        "   .inst 0xc006440c //mova    { z12.b-z15.b }, za0h.b[w14, 0:3]\n"
        "   .inst 0xc132e18c //scvtf   { z12.s-z15.s }, { z12.s-z15.s } \n"

        // Result += iacc x SF
        "   fmla    z24.s, p2/m, z20.s, z12.s \n"
        "   fmla    z25.s, p2/m, z21.s, z13.s \n"
        "   fmla    z26.s, p2/m, z22.s, z14.s \n"
        "   fmla    z27.s, p2/m, z23.s, z15.s \n"

        // Add the bias
        "   fadd    z24.s, p2/m, z24.s, z8.s  \n"
        "   fadd    z25.s, p2/m, z25.s, z9.s  \n"
        "   fadd    z26.s, p2/m, z26.s, z10.s \n"
        "   fadd    z27.s, p2/m, z27.s, z11.s \n"

        // CLAMP and store
        "   .inst 0xc1bfcbd8 //fclamp  { z24.s-z27.s }, z30.s, z31.s\n"
        "   .inst 0xa060c638 //st1w    { z24.s-z27.s }, pn9, [x17]  \n"

        "   add     x17, x17, %[n], lsl #2                   \n"
        "   add     x14, x14, #4                             \n"
        "   cmp     x14, x16                                 \n"
        "   b.lt    4b                                       \n"

        // N loop tail
        "   add   x8, x8, %[rhs_stride]                      \n"
        "   .inst 0x04295089 // ddvl x9, x9, #4              \n"
        "   sub   x13, x13, %[nr]                            \n"
        "   .inst 0x256d47f0 //whilelt pn8.h, xzr, x13, vlx2 \n"
        "   b.mi  2b                                         \n"

        // M loop tail
        "   add   x20, x20, %[lhs_stride]                    \n"
        "   add   x19, x19, %[dst_inc], lsl #2               \n"
        "   sub   x12, x12, %[mr]                            \n"
        "   whilelt p0.s, xzr, x12        \n"
        "   b.mi 1b                                          \n"

        "5:                                                  \n"
        "   .inst 0xd503467f //smstop                        \n"
        :
        : [m] "r"(m_in), [n] "r"(n_in), [k] "r"(k_in), [lhs_stride] "r"(lhs_stride), [rhs_stride] "r"(rhs_stride),
          [mr] "r"(mr), [nr] "r"(nr), [lut] "r"(lut), [m_blk] "r"(m_blk), [n_blk] "r"(n_blk), [lhs] "r"(lhs_packed),
          [rhs] "r"(rhs_packed), [dst_inc] "r"(dst_inc), [scalar_bounds] "r"(scalar_bounds), [dst] "r"(dst)
        : "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "p0", "p2", "p8",
          "p9", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z30", "z31",
#ifdef __ARM_STATE_ZA
          "za",
#endif
#ifdef __ARM_STATE_ZT0
          "zt0",
#endif
          "cc", "memory");
}

#endif  // Architectural feature check
