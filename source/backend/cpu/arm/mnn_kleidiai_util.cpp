//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mnn_kleidiai_util.h"

using namespace MNN;

static const size_t kai_num_bytes_adder_rhs = 4; //sizeof(int32_t) or sizeof(float)
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_k_roundedup(size_t k, size_t kr, size_t sr) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for memory alignment.
    size_t kr_sr_roundedup4 = kai_roundup(kr * sr, 4);
    return kai_roundup(k, kr_sr_roundedup4);
}

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % bl) == 0);
    return k / bl;
}

void KleidiAIUtil::transferNCHWToNC4HW4(float* src, float* dst, size_t rowNum, size_t rowSize) {
    size_t blockNum = rowSize / 4;
    size_t blockSize = 4 * sizeof(float);

    for(size_t blockIndex = 0; blockIndex < blockNum; blockIndex++) {
        const float *rowSrc = src + blockIndex * 4;
        for(size_t rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            memcpy(dst, rowSrc, blockSize);
            dst += 4;
            rowSrc += rowSize;
        }
    }
}

void KleidiAIUtil::transferNCHWToNC4HW4(__fp16* src, __fp16* dst, size_t rowNum, size_t rowSize) {
    size_t blockNum = rowSize / 8;
    size_t blockSize = 8 * sizeof(__fp16);

    for(size_t blockIndex = 0; blockIndex < blockNum; blockIndex++) {
        const __fp16 *rowSrc = src + blockIndex * 8;
        for(size_t rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            memcpy(dst, rowSrc, blockSize);
            dst += 8;
            rowSrc += rowSize;
        }
    }
}

void KleidiAIUtil::transferNC4HW4ToNCHW(float* src, float* dst, size_t rowNum, size_t rowSize) {
    size_t blockNum = rowSize / 4;
    size_t blockSize = 4 * sizeof(float);

    for(size_t blockIndex = 0; blockIndex < blockNum; blockIndex++) {
        const float *rowSrc = src + blockIndex * 4 * rowNum;
        float *block_dst = dst + blockIndex * 4;
        for(size_t rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            memcpy(block_dst, rowSrc, blockSize);
            block_dst += rowSize;
            rowSrc += 4;
        }
    }
}

void KleidiAIUtil::transferNC4HW4ToNCHW(__fp16* src, __fp16* dst, size_t rowNum, size_t rowSize) {
    size_t blockNum = rowSize / 8;
    size_t blockSize = 8 * sizeof(__fp16);

    for(size_t blockIndex = 0; blockIndex < blockNum; blockIndex++) {
        const __fp16 *rowSrc = src + blockIndex * 8 * rowNum;
        __fp16 *block_dst = dst + blockIndex * 8;
        for(size_t rowIndex = 0; rowIndex < rowNum; rowIndex++) {
            memcpy(block_dst, rowSrc, blockSize);
            block_dst += rowSize;
            rowSrc += 8;
        }
    }
}

// Rhs pack functions for matmul_clamp_f32_qai8dxp_qsi4cxp.
void KleidiAIUtil::packQsi4cxps16s0Qs4cxs0s1(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct KleidiAIUtil::rhsPackParamCommon* paramsCommon) {
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);

    const struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params* params = (kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params *)paramsCommon;
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->rhs_zero_point == 8);
    KAI_ASSERT(params->lhs_zero_point == 1);

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
            const size_t shift_right_x0 = ((k0_idx + 1) % 2) * 4;
            const size_t shift_right_x1 = ((k1_idx + 1) % 2) * 4;

            const uint8_t src_x0_lo = (byte0 >> shift_right_x0) & 0x0F;
            const uint8_t src_x0_hi = (byte1 >> shift_right_x1) & 0x0F;

            sums[nr_idx] += (int32_t)src_x0_lo + (int32_t)src_x0_hi - 2 * (int32_t)rhs_zero_point;

            const uint8_t dst_qs0 = src_x0_lo | (src_x0_hi << 4);

            *dst_row = dst_qs0 ^ 0x88;
            dst_row += sizeof(uint8_t);
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
            memset(dst_row, 0, nr * sizeof(float));
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);
                ((float*)dst_row)[i] = bias[src_row_idx];
            }
        }
    }
}

void KleidiAIUtil::packQsi4cxps16s0Qs4cx(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct KleidiAIUtil::rhsPackParamCommon* paramsCommon) {
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);

    const struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params* params = (kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params *)paramsCommon;
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->rhs_zero_point == 8);
    KAI_ASSERT(params->lhs_zero_point == 1);

    const size_t rhs_zero_point = params->rhs_zero_point;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(k, nr, kr, sr);
    const size_t k_internal = kai_k_roundedup(k, kr, sr);
    const size_t dst_num_rows = kai_roundup(n, nr) / nr;
    const size_t dst_num_bytes_per_row = nr * (kai_k_roundedup(k, kr, sr) / 2);
    const size_t block_length_in_bytes = kr / sr;
    const size_t k_interleaved_v = 16U;
    const size_t rhs_stride = kai_roundup(k, 2);

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

            const size_t src_addr_byte0 = k0_idx + n0_valid_idx * rhs_stride;
            const size_t src_addr_byte1 = k1_idx + n0_valid_idx * rhs_stride;

            int8_t byte0 = 0;
            int8_t byte1 = 0;

            if (k0_idx < k) {
                byte0 = rhs[src_addr_byte0];
            }

            if (k1_idx < k) {
                byte1 = rhs[src_addr_byte1];
            }

            sums[nr_idx] += (int32_t)byte0 + (int32_t)byte1;

            const uint8_t dst_qs0 = (byte0 + rhs_zero_point) | ((byte1 + rhs_zero_point) << 4);

            *dst_row = dst_qs0 ^ 0x88;
            dst_row += sizeof(uint8_t);
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
            memset(dst_row, 0, nr * sizeof(float));
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);
                ((float*)dst_row)[i] = bias[src_row_idx];
            }
        }
    }
}

void KleidiAIUtil::packQsi4cxps1s0Qsu4cxs0s1(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct KleidiAIUtil::rhsPackParamCommon* paramsCommon) {
    const size_t k_internal = kai_k_roundedup(k, 16, 2);

    KAI_ASSERT((k_internal % kr) == 0);
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);

    const struct kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon_params* params = (kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon_params *)paramsCommon;
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->lhs_zero_point == 1);
    KAI_ASSERT(params->rhs_zero_point == 0 || params->rhs_zero_point == 8);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const int32_t rhs_zero_point = params->rhs_zero_point;
    const size_t rhs_stride = kai_roundup(k, 2) / 2;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(k, nr, kr, sr);
    const size_t dst_nr_block_size = nr * kr * sizeof(uint8_t) / 2;

    // Iterate over n src rows in blocks of nr rows
    for (size_t row_idx = 0; row_idx < n; row_idx += nr) {
        int8_t* const dst_row = (int8_t*)rhs_packed + ((row_idx / nr) * rhs_packed_stride);

        int32_t* const sums = (int32_t*)(dst_row + (nr * (k_internal / 2)));
        float* const scaling_factors = (float*)((uint8_t*)sums + (nr * kai_num_bytes_adder_rhs));
        // Update destination row pointer
        float* const biases = (float*)((uint8_t*)scaling_factors + (nr * kai_num_bytes_multiplier_rhs));

        // initialize sums to 0
        memset(sums, 0, nr * kai_num_bytes_adder_rhs);

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
                    const int32_t second_value = (dst_byte & 0xF) - rhs_zero_point;
                    const int32_t first_value = col_idx + kr_block_idx + 1 >= k ? 0 : (dst_byte >> 4) - rhs_zero_point;

                    // Add the i4 value to the row sum
                    sum += first_value + second_value;

                    // Truncate i8 to i4 and write to dst
                    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                    dst_kr_block[kr_block_idx / 2] = (second_value << 4) | (first_value & 0xF);
                }

                // Go to the next kr block for this row in the nr rows
                dst_kr_block += dst_nr_block_size;
            }

            // save sum
            sums[nr_block_idx] = sum;
        }
    }
}