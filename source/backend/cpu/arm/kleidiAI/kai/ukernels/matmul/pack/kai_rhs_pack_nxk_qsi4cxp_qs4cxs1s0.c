//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_k_roundedup(size_t k, size_t kr, size_t sr) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for alignment
    size_t kr_sr_roundedup4 = kai_roundup(kr * sr, 4);
    return kai_roundup(k, kr_sr_roundedup4);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t k_internal = kai_k_roundedup(k, kr, sr);

    KAI_ASSERT((k_internal % 2) == 0);

    return nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(k, nr, kr, sr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(size_t n, size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(k, nr, kr, sr);
}

void kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params* params) {
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->lhs_zero_point == 1);
    KAI_ASSERT(params->rhs_zero_point == 0 || params->rhs_zero_point == 8);

    const size_t rhs_zero_point = params->rhs_zero_point;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(k, nr, kr, sr);
    const size_t k_internal = kai_k_roundedup(k, kr, sr);
    const size_t dst_num_rows = kai_roundup(n, nr) / nr;
    const size_t dst_num_bytes_per_row = nr * (kai_k_roundedup(k, kr, sr) / 2);
    const size_t block_length_in_bytes = kr / sr;
    const size_t k_interleaved_v = 16U;
    const size_t rhs_stride = kai_roundup(k, 2) / 2;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
        uint8_t* dst_row = (uint8_t*)rhs_packed + dst_row_idx * rhs_packed_stride;

        int32_t* sums = (int32_t*)(dst_row + nr * (k_internal / 2));

        // Initialize to zero the RHS reduction sums
        memset(sums, 0, nr * sizeof(int32_t));

        for (size_t dst_byte_idx = 0; dst_byte_idx < dst_num_bytes_per_row; ++dst_byte_idx) {
            const size_t block_idx = dst_byte_idx / block_length_in_bytes;
            const size_t block_byte_idx = dst_byte_idx % block_length_in_bytes;
            const size_t super_block_idx = block_idx / nr;
            const size_t nr_idx = block_idx % nr;

            const size_t k_adjustment =
                ((block_byte_idx + super_block_idx * block_length_in_bytes) / k_interleaved_v) * k_interleaved_v;
            const size_t k0_idx = block_byte_idx + super_block_idx * block_length_in_bytes + k_adjustment;
            const size_t k1_idx = k0_idx + k_interleaved_v;
            const size_t n0_idx = dst_row_idx * nr + nr_idx;

            // Clamp the index to avoid out-of-bound reads
            const size_t n0_valid_idx = KAI_MIN(n0_idx, n - 1);

            const size_t src_addr_byte0 = (k0_idx / 2) + n0_valid_idx * rhs_stride;
            const size_t src_addr_byte1 = (k1_idx / 2) + n0_valid_idx * rhs_stride;

            if (params->rhs_zero_point == 8) {
                uint8_t byte0 = rhs_zero_point | rhs_zero_point << 4;
                uint8_t byte1 = rhs_zero_point | rhs_zero_point << 4;

                if (k0_idx < k) {
                    byte0 = rhs[src_addr_byte0];
                }

                if (k1_idx < k) {
                    byte1 = rhs[src_addr_byte1];
                }

                // The following operations where we extract the values from the bytes
                // can be also written in the following and less efficient manner:
                /*
                    uint8_t src_x0_lo = 0;
                    uint8_t src_x0_hi = 0;

                    if ((k0_idx % 2) == 0) {
                        src_x0_lo = (byte0 & 0x0F);
                    } else {
                        src_x0_lo = (byte0 >> 4);
                    }

                    if ((k1_idx % 2) == 0) {
                        src_x0_hi = (byte1 & 0x0F);
                    } else {
                        src_x0_hi = (byte1 >> 4);
                    }
                */
                const size_t shift_right_x0 = (k0_idx % 2) * 4;
                const size_t shift_right_x1 = (k1_idx % 2) * 4;

                const uint8_t src_x0_lo = (byte0 >> shift_right_x0) & 0x0F;
                const uint8_t src_x0_hi = (byte1 >> shift_right_x1) & 0x0F;

                sums[nr_idx] += (int32_t)src_x0_lo + (int32_t)src_x0_hi - 2 * (int32_t)rhs_zero_point;

                const uint8_t dst_qs0 = src_x0_lo | (src_x0_hi << 4);

                *dst_row = dst_qs0 ^ 0x88;
                dst_row += sizeof(uint8_t);
            } else {
                int8_t byte0 = 0;
                int8_t byte1 = 0;

                if (k0_idx < k) {
                    byte0 = rhs[src_addr_byte0];
                }

                if (k1_idx < k) {
                    byte1 = rhs[src_addr_byte1];
                }

                // The logic behind the following operations where we extract the
                // values from the bytes is same as unsigned

                const size_t shift_right_x0 = (k0_idx % 2) * 4;
                const size_t shift_right_x1 = (k1_idx % 2) * 4;

                int8_t src_x0_lo = (byte0 >> shift_right_x0) & 0x0F;
                int8_t src_x0_hi = (byte1 >> shift_right_x1) & 0x0F;

                const int8_t dst_qs0 = src_x0_lo | (src_x0_hi << 4);

                *(int8_t*)dst_row = dst_qs0;
                dst_row += sizeof(int8_t);

                src_x0_lo = kai_ext_sign_i8_i4(src_x0_lo);
                src_x0_hi = kai_ext_sign_i8_i4(src_x0_hi);
                sums[nr_idx] += (int32_t)src_x0_lo + (int32_t)src_x0_hi;
            }
        }

        // Adjust the reduction sums
        for (size_t i = 0; i < nr; ++i) {
            sums[i] = sums[i] * 16;
            dst_row += sizeof(int32_t);
        }

        // Adjust the scales
        for (size_t i = 0; i < nr; ++i) {
            // Clamp the row index to avoid out-of-bound reads
            const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);
            *((float*)(dst_row)) = scale[src_row_idx] * 0.0625F;
            dst_row += sizeof(float);
        }

        // Set the bias
        if (bias == NULL) {
            memset(dst_row, 0, nr * kai_num_bytes_bias);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);
                ((float*)dst_row)[i] = bias[src_row_idx];
            }
        }
    }
}
