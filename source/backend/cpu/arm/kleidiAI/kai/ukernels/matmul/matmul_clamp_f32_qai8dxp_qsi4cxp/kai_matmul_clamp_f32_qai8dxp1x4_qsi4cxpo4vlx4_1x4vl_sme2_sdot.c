//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural feature check

#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot.h"

#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 1;
static const size_t kai_n_step = 1;
static const size_t kai_nr = 4;  // nr svl dependent
static const size_t kai_mr = 1;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

// Scaling factors
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
// q8_1 zero point
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
// Sum of quantized row for weights for faster zero point activations
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
// Bias
static const size_t kai_num_bytes_bias_rhs = sizeof(int32_t);

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, 32);
}

inline static size_t kai_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 32) == 0);

    return kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot() *
        (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 32) == 0);

    return kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot() *
        ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias_rhs);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(void) {
    return kai_m_step;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(void) {
    return kai_n_step * kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot();
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(void) {
    // For gemv mr must be 1 to consecutively read the data
    return kai_mr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + (m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(size_t m, size_t n) {
    return m * n * sizeof(float);
}

/**
 * Lut to be indexed by i4 resulting in its value in i8 (i.e. -2 = 1110 -> 1111 1110).
 **/
static const int8_t lut[64] = {0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0, 4,  0, 0,  0, 5, 0,
                               0,  0, 6, 0, 0,  0, 7, 0, 0,  0, -8, 0, 0,  0, -7, 0, 0,  0, -6, 0, 0, 0,
                               -5, 0, 0, 0, -4, 0, 0, 0, -3, 0, 0,  0, -2, 0, 0,  0, -1, 0, 0,  0};

/**
 *
 * Optimized for GEMV (matrix vector multiplication => m == 1).
 * Does a matmul for compatibility reasons, but should not be used that way.
 *
 **/
void kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed,
    float* dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    // Do function calls and calculations first to not overwrite registers we will use
    uint64_t k_internal = kai_k_roundedup(k);
    uint64_t A_vector_increment = kai_lhs_packed_stride(k);
    uint64_t W_row_stride = kai_rhs_packed_stride(k);
    uint64_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot();

    uint64_t W_row_bytes = nr * k_internal / 2;
    uint64_t W_row_bytes_net = nr * k_internal / 2;
    uint64_t A_matrix_end_ptr = ((uint64_t)lhs_packed) + (m * A_vector_increment);

    /*
     * x11: zero = 0 // MUST BE x8-x11
     * x15: n initialized as n
     * x19: nr initialized as nr
     * x20: lut_ptr initialized as lut
     * x21: A_vector_ptr initialized as lhs_packed
     * x22: n_idx
     * x23: k_idx
     * x24: W_k_block_ptr
     * x25: W_row_values_end_ptr
     * x26: W_row_ptr
     * x27: dst_ptr
     * x28: tmp_1
     */

    __asm__ volatile(

        // Setup
        " .inst 0xd503477f // smstart                                       \n"
        " mov     x11, #0                                                   \n"
        " mov     x15, %[n]                                                 \n"
        " mov     x19, %[nr]                                                \n"
        " mov     x21, %[lhs_packed]                                        \n"
        " mov     x20, %[lut]                                               \n"
        " .inst 0xe11f8280 // ldr     zt0, [x20]                            \n"
        " ptrue   p0.b                                                      \n"
        " .inst 0x25207810 // ptrue   pn8.b                                 \n"
        // predicate to load nr words for the W sums and scaling factors (should be exactly all true)
        " .inst 0x25b36571 // whilelt pn9.s, x11, x19, vlx4                 \n"
        " dup     z30.s, %w[scalar_min]                                     \n"
        " dup     z31.s, %w[scalar_max]                                     \n"

        // Activation matrix row loop
        "1:                                                                 \n"
        // Reset weight matrix ptr
        " mov     x26, %[rhs_packed]                                        \n"
        // Reset dst_ptr to dst of next GEMV result
        " mov     x27, %[dst_vector_ptr]                                    \n"
        // Reset n index
        " mov     x22, #0                                                   \n"
        // whilelt pn12.s, x22, %[n], vlx4
        " .inst 0x25af66d4 // whilelt pn12.s, x22, x15, vlx4                \n"

        // Weight matrix row loop (transposed so theoretical columns)
        "2:                                                                 \n"

        // Reset weights block ptr to start of row
        " mov     x24, x26                                                  \n"
        " add     x25, x26, %[W_row_bytes_net]                              \n"
        " .inst 0x25396712 // whilelt pn10.b, x24, x25, vlx4                \n"
        " addvl   x28, x24, #4                                              \n"
        " .inst 0x25396793 // whilelt pn11.b, x28, x25, vlx4                \n"
        " mov     x23, #0                                                   \n"
        " whilelt p1.b, x23, %[k_internal]                                  \n"
        // Zero for sdot accumulation in inner loop
        " .inst 0xc00800ff // zero    {za}                                  \n"

        // before k loop
        "3:                                                                 \n"

        // Load A
        " ld1rqb  { z0.b }, p1/z , [x21, x23]                               \n"

        // Load w
        " .inst 0xa0408b10 // ld1b    { z16.b - z19.b }, pn10/z, [x24]      \n"
        " .inst 0xa0418f14 // ld1b {z20.b-z23.b}, pn11/z, [x24,#0x4, mul vl]\n"

        // Weight i4 to i8 and sdot
        // k block + 0
        " .inst 0xc08a4218 // luti4   { z24.b, z25.b }, zt0, z16[0]         \n"
        " .inst 0xc08a423a // luti4   { z26.b, z27.b }, zt0, z17[0]         \n"
        " .inst 0xc150f320 // sdot za.s[w11,0, vgx4], {z24.b-z27.b}, z0.b[0]\n"
        // k block + 1
        " .inst 0xc08a4244 // luti4   { z4.b, z5.b }, zt0, z18[0]           \n"
        " .inst 0xc08a4266 // luti4   { z6.b, z7.b }, zt0, z19[0]           \n"
        " .inst 0xc150f4a0 // sdot za.s[w11,0, vgx4], {z4.b-z7.b}, z0.b[1]  \n"
        // k block + 2
        " .inst 0xc08a4288 // luti4   { z8.b, z9.b }, zt0, z20[0]           \n"
        " .inst 0xc08a42aa // luti4   { z10.b, z11.b }, zt0, z21[0]         \n"
        " .inst 0xc150f920 // sdot za.s[w11,0, vgx4], {z8.b-z11.b}, z0.b[2] \n"
        // k block + 3
        " .inst 0xc08a42cc // luti4   { z12.b, z13.b }, zt0, z22[0]         \n"
        " .inst 0xc08a42ee // luti4   { z14.b, z15.b }, zt0, z23[0]         \n"
        " .inst 0xc150fda0 // sdot za.s[w11,0, vgx4], {z12.b-z15.b}, z0.b[3]\n"

        // End K block loop
        " addvl   x24, x24, #8                                              \n"
        " .inst 0x25396712 // whilelt pn10.b, x24, x25, vlx4                \n"
        " addvl   x28, x24, #4                                              \n"
        " .inst 0x25396793 // whilelt pn11.b, x28, x25, vlx4                \n"
        " add     x23, x23, #16                                             \n"
        " whilelt p1.b, x23, %[k_internal]                                  \n"
        " b.first 3b                                                        \n"

        // Finish of accumulators with scaling factors and zero points

        // Load A zero point
        " add     x28, x21, %[k_internal]                                   \n"
        " ld1rw   { z2.s }, p0/z , [x28]                                    \n"
        // Load A scaling factor
        " ld1rw   { z3.s }, p0/z , [x28, #4]                                \n"
        // Load W sums
        " add     x28, x26, %[W_row_bytes]                                  \n"
        " .inst 0xa040c794 // ld1w    { z20.s - z23.s }, pn9/z, [x28]       \n"
        // Load W scaling factors
        " .inst 0xa041c798 // ld1w {z24.s-z27.s}, pn9/z, [x28, #0x4, mul vl]\n"
        // Load biases
        " .inst 0xa042c78c // ld1w {z12.s-z15.s}, pn9/z, [x28, #0x8, mul vl]\n"

        // Get accumulated value out of ZA
        " .inst 0xc0066c04 // mov     { z4.d - z7.d }, za.d[w11, 0, vgx4]   \n"

        // za contains a * w, which needs to be done + z * wsum -> smla
        // zero point * W row sum
        " mla     z4.s, p0/m, z20.s, z2.s                                   \n"
        " mla     z5.s, p0/m, z21.s, z2.s                                   \n"
        " mla     z6.s, p0/m, z22.s, z2.s                                   \n"
        " mla     z7.s, p0/m, z23.s, z2.s                                   \n"

        // Convert to float
        " .inst 0xc132e084 // scvtf   { z4.s - z7.s }, { z4.s - z7.s }      \n"

        // A scaling factor * W scaling factor
        " fmul    z24.s, z24.s, z3.s                                        \n"
        " fmul    z25.s, z25.s, z3.s                                        \n"
        " fmul    z26.s, z26.s, z3.s                                        \n"
        " fmul    z27.s, z27.s, z3.s                                        \n"

        // Bias + combined scaling factor * combined accumulator
        " fmla    z12.s, p0/m, z24.s, z4.s                                  \n"
        " fmla    z13.s, p0/m, z25.s, z5.s                                  \n"
        " fmla    z14.s, p0/m, z26.s, z6.s                                  \n"
        " fmla    z15.s, p0/m, z27.s, z7.s                                  \n"

        // Clamp
        " .inst 0xc1bfcbcc // fclamp  { z12.s - z15.s }, z30.s, z31.s       \n"

        // Store
        " .inst 0xa036d36c // st1w {z12.s-z15.s}, pn12, [x27, x22, lsl #2]  \n"

        // End W row loop
        " add     x26, x26, %[W_row_stride]                                 \n"
        // nr == svlb
        " addvl   x22, x22, #1                                              \n"
        // whilelt pn12.s, x22, %[n], vlx4
        " .inst 0x25af66d4 // whilelt pn12.s, x22, x15, vlx4                \n"
        " b.lt    2b                                                        \n"

        // End A row loop
        " add     %[dst_vector_ptr], %[dst_vector_ptr], %[dst_stride_row]   \n"
        " add     x21, x21, %[A_vector_increment]                           \n"
        " cmp     x21, %[A_matrix_end_ptr]                                  \n"
        " b.lt    1b                                                        \n"

        " .inst 0xd503467f // smstop                                        \n"

        : [dst_vector_ptr] "+r"(dst)
        : [lut] "r"(lut), [m] "r"(m), [n] "r"(n), [k] "r"(k), [lhs_packed] "r"(lhs_packed),
          [rhs_packed] "r"(rhs_packed), [dst_stride_row] "r"(dst_stride_row), [scalar_min] "r"(scalar_min),
          [scalar_max] "r"(scalar_max), [k_internal] "r"(k_internal), [A_vector_increment] "r"(A_vector_increment),
          [W_row_stride] "r"(W_row_stride), [nr] "r"(nr), [W_row_bytes] "r"(W_row_bytes),
          [W_row_bytes_net] "r"(W_row_bytes_net), [A_matrix_end_ptr] "r"(A_matrix_end_ptr)
        : "x11", "x15", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "p0", "p1", "p8", "p9",
          "p10", "p11", "p12", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13",
          "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28",
          "z29", "z30", "z31",
#ifdef __ARM_STATE_ZA
          "za",
#endif
#ifdef __ARM_STATE_ZT0
          "zt0",
#endif
          "memory", "cc");
}
#endif  // Architectural feature check
