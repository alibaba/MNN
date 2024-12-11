//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0.h"

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    size_t kai_k_multiple_of = 32;
    return kai_roundup(k, kai_k_multiple_of);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    const size_t k_internal = kai_k_roundedup(k);

    // multiple of 2 because 2 elements in a byte
    KAI_ASSERT((k_internal % 2) == 0);

    return nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(k, nr, kr, sr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(size_t n, size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(k, nr, kr, sr);
}

void kai_run_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0_params* params) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kr) == 0);
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->lhs_zero_point == 1);
    KAI_ASSERT(params->rhs_zero_point == 0 || params->rhs_zero_point == 8);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const int32_t rhs_zero_point = params->rhs_zero_point;
    const size_t rhs_stride = kai_roundup(k, 2) / 2;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(k, nr, kr, sr);
    const size_t dst_nr_block_size = nr * kr * sizeof(uint8_t) / 2;

    // Iterate over n src rows in blocks of nr rows
    for (size_t row_idx = 0; row_idx < n; row_idx += nr) {
        int8_t* const dst_row = (int8_t*)rhs_packed + ((row_idx / nr) * rhs_packed_stride);

        int32_t* const sums = (int32_t*)(dst_row + (nr * (k_internal / 2)));
        float32_t* const scaling_factors = (float32_t*)((uint8_t*)sums + (nr * kai_num_bytes_sum_rhs));
        // Update destination row pointer
        float* const biases = (float*)((uint8_t*)scaling_factors + (nr * kai_num_bytes_multiplier_rhs));

        // initialize sums to 0
        memset(sums, 0, nr * kai_num_bytes_sum_rhs);

        // Copy the scaling factors and bias
        size_t rows_left = n - row_idx;
        if (rows_left >= nr) {
            memcpy(scaling_factors, &scale[row_idx], nr * kai_num_bytes_multiplier_rhs);
            memcpy(biases, &bias[row_idx], nr * kai_num_bytes_bias);
        } else {
            // Fill remaining values
            memcpy(scaling_factors, &scale[row_idx], rows_left * kai_num_bytes_multiplier_rhs);
            memcpy(biases, &bias[row_idx], rows_left * kai_num_bytes_bias);
            // Set leftover to 0
            memset(&scaling_factors[rows_left], 0, (nr - rows_left) * kai_num_bytes_multiplier_rhs);
            memset(&biases[rows_left], 0, (nr - rows_left) * kai_num_bytes_bias);
        }

        // Iterate over rows in the nr row block
        for (size_t nr_block_idx = 0; nr_block_idx < nr; ++nr_block_idx) {
            const uint8_t* const src_row = rhs + ((row_idx + nr_block_idx) * rhs_stride);
            // Go to the first kr block for this row in the nr block
            int8_t* dst_kr_block = dst_row + (nr_block_idx * kr / 2);

            int32_t sum = 0;

            // Iterate over k src columns in blocks of kr columns
            for (size_t col_idx = 0; col_idx < k_internal; col_idx += kr) {
                // Iterate over columns in the kr block
                // Kr checked to be multiple of 2 (because 2 values per byte)
                for (size_t kr_block_idx = 0; kr_block_idx < kr; kr_block_idx += 2) {
                    // We pad dst with 0s if the rounded k or n values have been exceeded
                    if (row_idx + nr_block_idx >= n || col_idx + kr_block_idx >= k) {
                        dst_kr_block[kr_block_idx / 2] = 0;
                        continue;
                    }

                    // Load the 2 u4 values from source
                    const uint8_t dst_byte = src_row[(col_idx + kr_block_idx) / 2];

                    // extract i8 values from the 2 u4 values
                    const int32_t first_value = (dst_byte & 0xF) - rhs_zero_point;
                    const int32_t second_value = col_idx + kr_block_idx + 1 >= k ? 0 : (dst_byte >> 4) - rhs_zero_point;

                    // Add the i4 value to the row sum
                    sum += first_value + second_value;

                    // Truncate i8 to i4 and write to dst
                    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                    dst_kr_block[kr_block_idx / 2] = (second_value << 4) | (first_value & 0xF);
                    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                }

                // Go to the next kr block for this row in the nr rows
                dst_kr_block += dst_nr_block_size;
            }

            // save sum
            sums[nr_block_idx] = sum;
        }
    }
}
