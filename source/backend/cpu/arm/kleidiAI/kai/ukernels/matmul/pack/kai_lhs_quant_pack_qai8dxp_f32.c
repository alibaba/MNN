//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_lhs_quant_pack_qai8dxp_f32.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <float.h>
#include <math.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_per_multiplier = sizeof(float);
static const size_t kai_num_bytes_per_offset = sizeof(int32_t);

inline static size_t kai_k_roundedup(size_t k, size_t kr, size_t sr) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for memory alignment.
    size_t kr_sr_roundedup4 = kai_roundup(kr * sr, 4);
    return kai_roundup(k, kr_sr_roundedup4);
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t mr, size_t kr, size_t sr) {
    const size_t k_internal = kai_k_roundedup(k, kr, sr);

    KAI_ASSERT((k_internal % 2) == 0);

    return mr * (k_internal * sizeof(int8_t) + kai_num_bytes_per_multiplier + kai_num_bytes_per_offset);
}

size_t kai_get_m_step_lhs_quant_pack_qai8dxp_f32(size_t mr) {
    KAI_UNUSED(mr);
    return 1;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    // It always points to the beginning of the row
    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, sr);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(m, mr) / mr;

    return num_rows * kai_lhs_packed_stride(k, mr, kr, sr);
}

void kai_run_lhs_quant_pack_qai8dxp_f32(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* restrict lhs,
    size_t lhs_stride, void* restrict lhs_packed) {
    KAI_ASSERT((kr % sr) == 0);

    if (m == 0) {
        return;
    }

    const size_t num_rows = m;

    const float* src_ptr = lhs;

    const size_t dst_stride = kai_lhs_packed_stride(k, mr, kr, sr);
    const size_t k_internal = kai_k_roundedup(k, kr, sr);
    const int32_t k_block_len = (int32_t)(kr / sr);

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        float max0 = -FLT_MAX;
        float min0 = FLT_MAX;

        // Find min/max for each channel
        int32_t k_idx = 0;

#if defined(__aarch64__)
        float32x4_t vmax0 = vdupq_n_f32(-FLT_MAX);
        float32x4_t vmin0 = vdupq_n_f32(FLT_MAX);

        for (; k_idx <= ((int32_t)k - 8); k_idx += 8) {
            const float32x4_t src0_0 = vld1q_f32(src_ptr + 0 + (size_t)k_idx);
            const float32x4_t src0_1 = vld1q_f32(src_ptr + 4 + (size_t)k_idx);

            // Calculate the max
            vmax0 = vmaxq_f32(src0_0, vmax0);
            vmax0 = vmaxq_f32(vmax0, src0_1);

            // Calculate the min
            vmin0 = vminq_f32(src0_0, vmin0);
            vmin0 = vminq_f32(vmin0, src0_1);
        }
        // Get the max/min
        max0 = vmaxvq_f32(vmax0);
        min0 = vminvq_f32(vmin0);
#endif
        for (; k_idx < (int32_t)k; ++k_idx) {
            const float src0_0 = *(src_ptr + (size_t)k_idx);
            max0 = KAI_MAX(src0_0, max0);
            min0 = KAI_MIN(src0_0, min0);
        }

        // Maximum/minimum int8 values
        const float qmin = (float)INT8_MIN;
        const float qmax = (float)INT8_MAX;

        const float rmin0 = KAI_MIN(0.0F, min0);
        const float rmax0 = KAI_MAX(0.0F, max0);

        const float scale0 = rmin0 == rmax0 ? 1.F : (qmax - qmin) / (rmax0 - rmin0);

        // Reciprocal to quantize
        const float recip_scale0 = scale0 ? 1.0F / scale0 : 0.0F;

        const float descaled_min0 = rmin0 * scale0;
        const float descaled_max0 = rmax0 * scale0;

        const float zero_point_from_min_error0 = qmin + descaled_min0;
        const float zero_point_from_max_error0 = qmax + descaled_max0;

        float zero_point0 =
            zero_point_from_min_error0 + zero_point_from_max_error0 > 0 ? qmin - descaled_min0 : qmax - descaled_max0;

        zero_point0 = KAI_MAX(zero_point0, qmin);
        zero_point0 = KAI_MIN(zero_point0, qmax);

        // Round to nearest integer
        const int32_t nudged_zero_point0 = (int32_t)rintf(zero_point0);

        const size_t dst_x = ((row_idx + m_idx_start) % mr);

        uint8_t* dst_ptr = (uint8_t*)lhs_packed + dst_x * k_block_len * sizeof(int8_t);

        // Quantize the channels
        k_idx = 0;
        for (; k_idx < (int32_t)k_internal; k_idx += k_block_len) {
            for (size_t k_block_idx = 0; k_block_idx < (size_t)k_block_len; ++k_block_idx) {
                // Clamp at the last valid k-index
                const size_t k_idx_start = KAI_MIN((size_t)k_idx + k_block_idx, k - 1);

                const float src0_0 = *(src_ptr + k_idx_start);

                // Scale the values
                int32_t v0_s32 = (int32_t)(roundf(src0_0 * scale0));

                v0_s32 = v0_s32 + nudged_zero_point0;
                v0_s32 = KAI_MAX(v0_s32, INT8_MIN);
                v0_s32 = KAI_MIN(v0_s32, INT8_MAX);
                *((int8_t*)(dst_ptr)) = (int8_t)v0_s32;
                dst_ptr += sizeof(int8_t);
            }
            dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
        }

        dst_ptr = (uint8_t*)lhs_packed + mr * (k_internal * sizeof(int8_t));

        dst_ptr += dst_x * kai_num_bytes_per_offset;

        // LHS offset at the beginning of the row
        *((int32_t*)(dst_ptr)) = -nudged_zero_point0;

        // Assuming the same sizeof() for kai_num_bytes_per_offset and kai_num_bytes_per_multiplier
        KAI_ASSERT(kai_num_bytes_per_offset == kai_num_bytes_per_multiplier);

        dst_ptr += mr * kai_num_bytes_per_offset;

        // Store the scale quantization params
        *((float*)(dst_ptr)) = recip_scale0;

        src_ptr += (lhs_stride / sizeof(float));

        // Move to the next row if we have interleaved all Mr rows
        if ((((row_idx + 1) + m_idx_start) % mr) == 0) {
            lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
        }
    }
}
