//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;
static const size_t kai_kr = 16;
static const size_t kai_sr = 2;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_k_roundedup(size_t k) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for alignment
    size_t kr_sr_roundedup4 = kai_roundup(kai_kr * kai_sr, 4);
    return kai_roundup(k, kr_sr_roundedup4);
}

inline static size_t kai_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return kai_mr * (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return kai_nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed, float* restrict dst,
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t kai_k0 = kai_kr * kai_sr;

    const size_t num_rows = m;
    const size_t num_cols = n;

    const size_t lhs_packed_stride = kai_lhs_packed_stride(k);
    const size_t k_internal = kai_k_roundedup(k);

    const int8x16_t nibble_mask = vdupq_n_s8(0xF0);

    const uint8_t* lhs_ptr_start = lhs_packed;

    for (size_t row_idx = 0; row_idx < num_rows; row_idx += kai_mr) {
        const uint8_t* rhs_ptr = rhs_packed;
        for (size_t col_idx = 0; col_idx < num_cols; col_idx += kai_nr) {
            const uint8_t* lhs_ptr = lhs_ptr_start;

            // Main f32 accumulator
            int32x4_t iacc0011 = vdupq_n_s32(0);
            int32x4_t iacc2233 = vdupq_n_s32(0);

            for (size_t b = 0; b < k_internal; b += kai_k0) {
                // Set up RHS
                const int8x16_t rhs_raw_vec_0 = vld1q_s8((const int8_t*)(rhs_ptr + 0));
                const int8x16_t rhs_raw_vec_1 = vld1q_s8((const int8_t*)(rhs_ptr + 16));
                const int8x16_t rhs_raw_vec_2 = vld1q_s8((const int8_t*)(rhs_ptr + 32));
                const int8x16_t rhs_raw_vec_3 = vld1q_s8((const int8_t*)(rhs_ptr + 48));

                // Low nibble
                const int8x16_t rhs_vec_0_0 = vshlq_n_s8(rhs_raw_vec_0, 4);
                const int8x16_t rhs_vec_1_0 = vshlq_n_s8(rhs_raw_vec_1, 4);
                const int8x16_t rhs_vec_2_0 = vshlq_n_s8(rhs_raw_vec_2, 4);
                const int8x16_t rhs_vec_3_0 = vshlq_n_s8(rhs_raw_vec_3, 4);

                // High nibble
                const int8x16_t rhs_vec_0_1 = vandq_s8(rhs_raw_vec_0, nibble_mask);
                const int8x16_t rhs_vec_1_1 = vandq_s8(rhs_raw_vec_1, nibble_mask);
                const int8x16_t rhs_vec_2_1 = vandq_s8(rhs_raw_vec_2, nibble_mask);
                const int8x16_t rhs_vec_3_1 = vandq_s8(rhs_raw_vec_3, nibble_mask);

                const int8x16_t lhs_vec_0 = vld1q_s8((const int8_t*)(lhs_ptr + 0));
                const int8x16_t lhs_vec_1 = vld1q_s8((const int8_t*)(lhs_ptr + 16));

                lhs_ptr += 32;
                rhs_ptr += 64;

                int8x16_t t;

                t = vcombine_s8(vget_low_s8(lhs_vec_0), vget_low_s8(lhs_vec_0));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_0_0, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_1_0, t);
                t = vcombine_s8(vget_high_s8(lhs_vec_0), vget_high_s8(lhs_vec_0));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_2_0, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_3_0, t);
                t = vcombine_s8(vget_low_s8(lhs_vec_1), vget_low_s8(lhs_vec_1));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_0_1, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_1_1, t);
                t = vcombine_s8(vget_high_s8(lhs_vec_1), vget_high_s8(lhs_vec_1));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_2_1, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_3_1, t);
            }

            int32x4_t iacc = vpaddq_s32(iacc0011, iacc2233);

            // LHS offset
            const int32x4_t lhs_offset = vld1q_dup_s32((const int32_t*)lhs_ptr);
            lhs_ptr += sizeof(int32_t);

            // LHS scale
            const float32x4_t lhs_scale = vld1q_dup_f32((const float*)lhs_ptr);
            lhs_ptr += sizeof(float);

            // RHS sum values
            const int32x4_t sum_n_s32 = vld1q_s32((const int32_t*)(rhs_ptr));
            rhs_ptr += sizeof(int32x4_t);

            // RHS scale
            const float32x4_t rhs_scale = vld1q_f32((const float*)rhs_ptr);
            rhs_ptr += sizeof(float32x4_t);

            // Load the bias
            const float32x4_t bias0 = vld1q_f32((const float*)rhs_ptr);
            rhs_ptr += sizeof(float32x4_t);

            // Add the reduction sum
            iacc = vmlaq_s32(iacc, sum_n_s32, lhs_offset);

            float32x4_t main_acc = vmulq_f32(vcvtq_f32_s32(iacc), rhs_scale);

            main_acc = vmulq_f32(main_acc, lhs_scale);

            // Add the bias
            main_acc = vaddq_f32(main_acc, bias0);

            // clamp (min-max) operation
            const float32x4_t vmin_f32 = vdupq_n_f32(scalar_min);
            const float32x4_t vmax_f32 = vdupq_n_f32(scalar_max);

            main_acc = vmaxq_f32(main_acc, vmin_f32);
            main_acc = vminq_f32(main_acc, vmax_f32);

            if (col_idx + kai_nr <= n) {
                vst1q_f32((float*)((uint8_t*)dst + col_idx * sizeof(float) + row_idx * dst_stride_row), main_acc);
            } else {
                size_t leftover = n % kai_nr;
                *(float*)((uint8_t*)dst + (col_idx + 0) * sizeof(float) + row_idx * dst_stride_row) =
                    vgetq_lane_f32(main_acc, 0);
                if (leftover > 1) {
                    *(float*)((uint8_t*)dst + (col_idx + 1) * sizeof(float) + row_idx * dst_stride_row) =
                        vgetq_lane_f32(main_acc, 1);
                }
                if (leftover > 2) {
                    *(float*)((uint8_t*)dst + (col_idx + 2) * sizeof(float) + row_idx * dst_stride_row) =
                        vgetq_lane_f32(main_acc, 2);
                }
            }
        }
        lhs_ptr_start += lhs_packed_stride;
    }
}
#endif  // Architectural feature check
