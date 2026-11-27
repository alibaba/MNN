//
//  indexed_linear_buf.cl
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Weights retain their serialized expert-major row layout. Only the currently
// consumed tile is unpacked; alpha stores [offset, scale] in signed-Q convention.
inline float indexed_weight(__global const uchar* weights, __global const float2* alpha,
                            int expert, int output, int input, int unused_channels, int unused_outputs,
                            int unused_groups, int unused_group_size, int unused_weight_stride) {
    const int element = output * BANK_I + input;
    const size_t base = (size_t)expert * BANK_STRIDE;
#if WEIGHT_BITS == 4
    const int packed = weights[base + (element >> 1)];
    const float q = (float)(((element & 1) ? (packed & 15) : (packed >> 4)) - 8);
#elif WEIGHT_BITS == 8
    const float q = (float)((__global const char*)weights)[base + element];
#elif WEIGHT_BITS == 16
    const uint byte_offset = (uint)element * 2;
    // vload_half is available even when half arithmetic/storage is disabled.
    return vload_half(0, (__global const half*)(weights + base + byte_offset));
#else
    return ((__global const float*)(weights + base))[element];
#endif
#if WEIGHT_BITS == 4 || WEIGHT_BITS == 8
    const float2 parameters = alpha[((size_t)expert * BANK_O + output) * BANK_GROUPS + input / BANK_GROUP_SIZE];
    return mad(q, parameters.y, parameters.x);
#endif
}

#define BANK_RESULT_O (BANK_O / (FUSE_SILU_MUL ? 2 : 1))
inline FLOAT indexed_swiglu(float gate, float up) {
    const float g = (float)(FLOAT)gate;
    const FLOAT activated = (FLOAT)(g * native_recip(1.0f + native_exp(-g)));
    return (FLOAT)((float)activated * (float)(FLOAT)up);
}

__kernel void indexed_route_clear(__global int* counts, int experts) {
    const int expert = get_global_id(0);
    if (expert < experts) counts[expert] = 0;
}

__kernel void indexed_route_count(__global const int* ids, __global int* counts, int rows, int experts) {
    const int row = get_global_id(0);
    if (row < rows) atomic_inc(counts + clamp(ids[row], 0, experts - 1));
}

__kernel void indexed_route_prefix(__global int* counts, __global int* offsets,
                                   __global int2* tiles, int experts, int max_tiles) {
    int offset = 0;
    int tile_count = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const int count = counts[expert];
        offsets[expert] = offset;
        for (int row = 0; row < count; row += TILE_M) {
            tiles[tile_count++] = (int2)(expert, offset + row);
        }
        offset += count;
        counts[expert] = 0;
    }
    offsets[experts] = offset;
    for (int tile = tile_count; tile < max_tiles; ++tile) tiles[tile] = (int2)(-1, 0);
}

__kernel void indexed_route_scatter(__global const int* ids, __global int* counts,
                                    __global const int* offsets, __global int* sorted_rows,
                                    int rows, int experts) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    const int expert = clamp(ids[row], 0, experts - 1);
    const int position = atomic_inc(counts + expert);
    sorted_rows[offsets[expert] + position] = row;
}

__kernel void indexed_linear_gemv(__global const FLOAT* input, __global const int* ids,
                                  __global const uchar* weights, __global const float2* alpha,
                                  __global const float* bias, __global FLOAT* output,
                                  int unused_channels, int unused_outputs, int experts, int unused_topk, int unused_input_mode,
                                  int unused_groups, int unused_group_size, int unused_weight_stride) {
    const int lane = get_local_id(0);
    const int first_output = get_group_id(0) * 4;
    const int row = get_global_id(1);
    const int expert = ids[row];
    if (expert < 0 || expert >= experts) {
        if (lane == 0) {
            for (int j = 0; j < 4 && first_output + j < BANK_RESULT_O; ++j) output[row * BANK_RESULT_O + first_output + j] = (FLOAT)0;
        }
        return;
    }
    const int input_row = BANK_INPUT_MODE ? row : row / BANK_TOPK;
    float4 sum = (float4)(0);
#if FUSE_SILU_MUL
    float4 up_sum = (float4)(0);
#endif
    for (int k = lane; k < BANK_I; k += 64) {
#if BANK_XC4
        const float x = (float)input[((size_t)(k / 4) * BANK_TOKENS + input_row) * 4 + (k & 3)];
#else
        const float x = (float)input[(size_t)input_row * BANK_I + k];
#endif
        // Some Adreno compilers reject vector subscripting, even in a fixed
        // four-iteration loop. Keep each output channel an explicit component.
        float4 weight = (float4)(0);
#if FUSE_SILU_MUL
        float4 up_weight = (float4)(0);
#endif
        if (first_output + 0 < BANK_RESULT_O) {
            const int channel = (first_output + 0) * (FUSE_SILU_MUL ? 2 : 1);
            weight.s0 = indexed_weight(weights, alpha, expert, channel, k,
                BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
#if FUSE_SILU_MUL
            up_weight.s0 = indexed_weight(weights, alpha, expert, channel + 1, k,
                BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
#endif
        }
        if (first_output + 1 < BANK_RESULT_O) {
            const int channel = (first_output + 1) * (FUSE_SILU_MUL ? 2 : 1);
            weight.s1 = indexed_weight(weights, alpha, expert, channel, k,
                BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
#if FUSE_SILU_MUL
            up_weight.s1 = indexed_weight(weights, alpha, expert, channel + 1, k,
                BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
#endif
        }
        if (first_output + 2 < BANK_RESULT_O) {
            const int channel = (first_output + 2) * (FUSE_SILU_MUL ? 2 : 1);
            weight.s2 = indexed_weight(weights, alpha, expert, channel, k,
                BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
#if FUSE_SILU_MUL
            up_weight.s2 = indexed_weight(weights, alpha, expert, channel + 1, k,
                BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
#endif
        }
        if (first_output + 3 < BANK_RESULT_O) {
            const int channel = (first_output + 3) * (FUSE_SILU_MUL ? 2 : 1);
            weight.s3 = indexed_weight(weights, alpha, expert, channel, k,
                BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
#if FUSE_SILU_MUL
            up_weight.s3 = indexed_weight(weights, alpha, expert, channel + 1, k,
                BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
#endif
        }
        sum = mad((float4)(x), weight, sum);
#if FUSE_SILU_MUL
        up_sum = mad((float4)(x), up_weight, up_sum);
#endif
    }
    __local float4 partial[64];
    partial[lane] = sum;
#if FUSE_SILU_MUL
    __local float4 up_partial[64];
    up_partial[lane] = up_sum;
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
#if FUSE_SILU_MUL
            up_partial[lane] += up_partial[lane + stride];
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        const size_t out_base = (size_t)row * BANK_RESULT_O + first_output;
        const size_t bias_base = (size_t)expert * BANK_O + first_output * (FUSE_SILU_MUL ? 2 : 1);
        if (first_output + 0 < BANK_RESULT_O) {
#if FUSE_SILU_MUL
            output[out_base + 0] = indexed_swiglu(partial[0].s0 + bias[bias_base + 0],
                                                   up_partial[0].s0 + bias[bias_base + 1]);
#else
            output[out_base + 0] = (FLOAT)(partial[0].s0 + bias[bias_base + 0]);
#endif
        }
        if (first_output + 1 < BANK_RESULT_O) {
#if FUSE_SILU_MUL
            output[out_base + 1] = indexed_swiglu(partial[0].s1 + bias[bias_base + 2],
                                                   up_partial[0].s1 + bias[bias_base + 3]);
#else
            output[out_base + 1] = (FLOAT)(partial[0].s1 + bias[bias_base + 1]);
#endif
        }
        if (first_output + 2 < BANK_RESULT_O) {
#if FUSE_SILU_MUL
            output[out_base + 2] = indexed_swiglu(partial[0].s2 + bias[bias_base + 4],
                                                   up_partial[0].s2 + bias[bias_base + 5]);
#else
            output[out_base + 2] = (FLOAT)(partial[0].s2 + bias[bias_base + 2]);
#endif
        }
        if (first_output + 3 < BANK_RESULT_O) {
#if FUSE_SILU_MUL
            output[out_base + 3] = indexed_swiglu(partial[0].s3 + bias[bias_base + 6],
                                                   up_partial[0].s3 + bias[bias_base + 7]);
#else
            output[out_base + 3] = (FLOAT)(partial[0].s3 + bias[bias_base + 3]);
#endif
        }

    }
}

#ifdef INDEXED_VECTOR_GEMM
inline float4 indexed_weight4(__global const uchar* weights, __global const float2* alpha,
                              int expert, int output, int input) {
    const size_t base = (size_t)expert * BANK_STRIDE;
    const int element = output * BANK_I + input;
#if WEIGHT_BITS == 4
    const uchar2 packed = vload2(0, weights + base + element / 2);
    const float4 q = convert_float4((uchar4)(packed.x >> 4, packed.x & 15,
                                            packed.y >> 4, packed.y & 15)) - 8.0f;
#else
    const float4 q = convert_float4(vload4(0, (__global const char*)(weights + base) + element));
#endif
    const float2 sa = alpha[((size_t)expert * BANK_O + output) * BANK_GROUPS + input / BANK_GROUP_SIZE];
    return mad(q, (float4)sa.y, (float4)sa.x);
}

__kernel void indexed_linear_gemm_vec(__global const FLOAT* input, __global const int* ids,
                                      __global const uchar* weights, __global const float2* alpha,
                                      __global const float* bias, __global FLOAT* output,
                                      __global const int* offsets, __global const int* sorted_rows,
                                      __global const int2* tiles,
                                      int unused_channels, int unused_outputs, int unused_topk, int unused_input_mode,
                                      int unused_groups, int unused_group_size, int unused_weight_stride) {
    const int2 tile = tiles[get_group_id(1)];
    if (tile.x < 0) return;
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int local_index = ly * 16 + lx;
    const int first_output = get_group_id(0) * 64;
    const int expert = tile.x, end = offsets[expert + 1];
    __local float tile_input[32 * 32];
    __local float tile_weight[32 * 64];
    int row[4];
    float4 sum[4];
    for (int m = 0; m < 4; ++m) {
        const int pos = tile.y + ly + m * 8;
        row[m] = pos < end ? sorted_rows[pos] : -1;
        sum[m] = (float4)0;
    }
    for (int base_k = 0; base_k < BANK_I; base_k += 32) {
        for (int i = local_index; i < 32 * 32; i += 128) {
            const int pos = tile.y + i / 32;
            const int r = pos < end ? sorted_rows[pos] : -1;
            const int k = base_k + i % 32;
            const int input_row = BANK_INPUT_MODE ? r : r / BANK_TOPK;
            float value = 0.0f;
            if (r >= 0 && ids[r] == expert) {
#if BANK_XC4
                value = (float)input[((size_t)(k / 4) * BANK_TOKENS + input_row) * 4 + (k & 3)];
#else
                value = (float)input[(size_t)input_row * BANK_I + k];
#endif
            }
            tile_input[i] = value;
        }
        for (int i = local_index; i < 64 * 8; i += 128) {
            const int n = i / 8;
            const int k = (i % 8) * 4;
            float4 w = (float4)0;
            if (first_output + n < BANK_O) {
                w = indexed_weight4(weights, alpha, expert, first_output + n, base_k + k);
            }
            tile_weight[(k + 0) * 64 + n] = w.x;
            tile_weight[(k + 1) * 64 + n] = w.y;
            tile_weight[(k + 2) * 64 + n] = w.z;
            tile_weight[(k + 3) * 64 + n] = w.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float4 w = vload4(0, tile_weight + k * 64 + lx * 4);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const float x = tile_input[(ly + m * 8) * 32 + k];
                sum[m] = mad((float4)x, w, sum[m]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int m = 0; m < 4; ++m) {
        if (row[m] < 0) continue;
        const int n = first_output + lx * 4;
        if (n >= BANK_O) continue;
        const bool valid = ids[row[m]] == expert;
        const float4 value = sum[m] + vload4(0, bias + (size_t)expert * BANK_O + n);
#if FUSE_SILU_MUL
        output[(size_t)row[m] * BANK_RESULT_O + n / 2] =
            valid ? indexed_swiglu(value.x, value.y) : (FLOAT)0;
        output[(size_t)row[m] * BANK_RESULT_O + n / 2 + 1] =
            valid ? indexed_swiglu(value.z, value.w) : (FLOAT)0;
#else
        const size_t out = (size_t)row[m] * BANK_O + n;
        output[out + 0] = valid ? (FLOAT)value.x : (FLOAT)0;
        output[out + 1] = valid ? (FLOAT)value.y : (FLOAT)0;
        output[out + 2] = valid ? (FLOAT)value.z : (FLOAT)0;
        output[out + 3] = valid ? (FLOAT)value.w : (FLOAT)0;
#endif
    }
}
#endif

// Prefill uses transposed, padded [K][N + 1] local weights to avoid
// strided bank conflicts. Bank dimensions, quantization groups, input mode
// and top-k are compile constants; accumulation and local storage stay FP32.
#if TILE_M == 32
// M=32, N=64, K=16, workgroup=16x8, transpose=True
__kernel void indexed_linear_gemm(__global const FLOAT* input, __global const int* ids,
                                  __global const uchar* weights, __global const float2* alpha,
                                  __global const float* bias, __global FLOAT* output,
                                  __global const int* offsets, __global const int* sorted_rows,
                                  __global const int2* tiles,
                                  int unused_channels, int unused_outputs, int unused_topk, int unused_input_mode,
                                  int unused_groups, int unused_group_size, int unused_weight_stride) {
    const int2 tile = tiles[get_group_id(1)];
    if (tile.x < 0) return;
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int local_index = ly * 16 + lx;
    const int first_output = get_group_id(0) * 64;
    const int expert = tile.x, expert_end = offsets[expert + 1];
    __local float tile_input[512];
    __local float tile_weight[1040];
    const int pos0 = tile.y + ly + 0;
    const int row0 = pos0 < expert_end ? sorted_rows[pos0] : -1;
    const int valid0 = row0 >= 0 && ids[row0] == expert;
    float4 sum0 = (float4)(0);
    const int pos1 = tile.y + ly + 8;
    const int row1 = pos1 < expert_end ? sorted_rows[pos1] : -1;
    const int valid1 = row1 >= 0 && ids[row1] == expert;
    float4 sum1 = (float4)(0);
    const int pos2 = tile.y + ly + 16;
    const int row2 = pos2 < expert_end ? sorted_rows[pos2] : -1;
    const int valid2 = row2 >= 0 && ids[row2] == expert;
    float4 sum2 = (float4)(0);
    const int pos3 = tile.y + ly + 24;
    const int row3 = pos3 < expert_end ? sorted_rows[pos3] : -1;
    const int valid3 = row3 >= 0 && ids[row3] == expert;
    float4 sum3 = (float4)(0);
    for (int base_k = 0; base_k < BANK_I; base_k += 16) {
        for (int index = local_index; index < 512; index += 128) {
            const int token_position = tile.y + index / 16;
            const int token_row = token_position < expert_end ? sorted_rows[token_position] : -1;
            const int k = base_k + index % 16;
            const int input_row = BANK_INPUT_MODE ? token_row : token_row / BANK_TOPK;
            tile_input[index] = token_row >= 0 && ids[token_row] == expert && k < BANK_I ?
#if BANK_XC4
                (float)input[((size_t)(k / 4) * BANK_TOKENS + input_row) * 4 + (k & 3)] : 0.0f;
#else
                (float)input[(size_t)input_row * BANK_I + k] : 0.0f;
#endif
        }
        for (int index = local_index; index < 1024; index += 128) {
            const int n = first_output + index / 16;
            const int k = base_k + index % 16;
            tile_weight[(index % 16) * 65 + index / 16] = n < BANK_O && k < BANK_I ?
                indexed_weight(weights, alpha, expert, n, k,
                               BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) {
            const float x0 = tile_input[(ly + 0) * 16 + k];
            const float x1 = tile_input[(ly + 8) * 16 + k];
            const float x2 = tile_input[(ly + 16) * 16 + k];
            const float x3 = tile_input[(ly + 24) * 16 + k];
#if FUSE_SILU_MUL
            const float4 w0 = (float4)(tile_weight[k * 65 + 2 * lx], tile_weight[k * 65 + 2 * lx + 1],
                                      tile_weight[k * 65 + 2 * lx + 32],
                                      tile_weight[k * 65 + 2 * lx + 33]);
#else
            const float4 w0 = (float4)(tile_weight[k * 65 + (lx + 0)], tile_weight[k * 65 + (lx + 16)],
                                         tile_weight[k * 65 + (lx + 32)], tile_weight[k * 65 + (lx + 48)]);
#endif
            sum0 = mad((float4)(x0), w0, sum0);
            sum1 = mad((float4)(x1), w0, sum1);
            sum2 = mad((float4)(x2), w0, sum2);
            sum3 = mad((float4)(x3), w0, sum3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#if FUSE_SILU_MUL
    if (row0 >= 0) {
        if ((first_output + 2 * lx + 0) < BANK_O) {
            output[(size_t)row0 * BANK_RESULT_O + (first_output + 2 * lx + 0) / 2] = valid0 ?
                indexed_swiglu(sum0.s0 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0)],
                               sum0.s1 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0) + 1]) : (FLOAT)0;
        }
        if ((first_output + 2 * lx + 32) < BANK_O) {
            output[(size_t)row0 * BANK_RESULT_O + (first_output + 2 * lx + 32) / 2] = valid0 ?
                indexed_swiglu(sum0.s2 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 32)],
                               sum0.s3 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 32) + 1]) : (FLOAT)0;
        }
    }
    if (row1 >= 0) {
        if ((first_output + 2 * lx + 0) < BANK_O) {
            output[(size_t)row1 * BANK_RESULT_O + (first_output + 2 * lx + 0) / 2] = valid1 ?
                indexed_swiglu(sum1.s0 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0)],
                               sum1.s1 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0) + 1]) : (FLOAT)0;
        }
        if ((first_output + 2 * lx + 32) < BANK_O) {
            output[(size_t)row1 * BANK_RESULT_O + (first_output + 2 * lx + 32) / 2] = valid1 ?
                indexed_swiglu(sum1.s2 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 32)],
                               sum1.s3 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 32) + 1]) : (FLOAT)0;
        }
    }
    if (row2 >= 0) {
        if ((first_output + 2 * lx + 0) < BANK_O) {
            output[(size_t)row2 * BANK_RESULT_O + (first_output + 2 * lx + 0) / 2] = valid2 ?
                indexed_swiglu(sum2.s0 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0)],
                               sum2.s1 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0) + 1]) : (FLOAT)0;
        }
        if ((first_output + 2 * lx + 32) < BANK_O) {
            output[(size_t)row2 * BANK_RESULT_O + (first_output + 2 * lx + 32) / 2] = valid2 ?
                indexed_swiglu(sum2.s2 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 32)],
                               sum2.s3 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 32) + 1]) : (FLOAT)0;
        }
    }
    if (row3 >= 0) {
        if ((first_output + 2 * lx + 0) < BANK_O) {
            output[(size_t)row3 * BANK_RESULT_O + (first_output + 2 * lx + 0) / 2] = valid3 ?
                indexed_swiglu(sum3.s0 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0)],
                               sum3.s1 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0) + 1]) : (FLOAT)0;
        }
        if ((first_output + 2 * lx + 32) < BANK_O) {
            output[(size_t)row3 * BANK_RESULT_O + (first_output + 2 * lx + 32) / 2] = valid3 ?
                indexed_swiglu(sum3.s2 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 32)],
                               sum3.s3 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 32) + 1]) : (FLOAT)0;
        }
    }
#else
    if (row0 >= 0) {
        if ((first_output + lx + 0) < BANK_O) {
            output[(size_t)row0 * BANK_O + (first_output + lx + 0)] =
                valid0 ? (FLOAT)(sum0.s0 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 0)]) : (FLOAT)0;
        }
        if ((first_output + lx + 16) < BANK_O) {
            output[(size_t)row0 * BANK_O + (first_output + lx + 16)] =
                valid0 ? (FLOAT)(sum0.s1 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 16)]) : (FLOAT)0;
        }
        if ((first_output + lx + 32) < BANK_O) {
            output[(size_t)row0 * BANK_O + (first_output + lx + 32)] =
                valid0 ? (FLOAT)(sum0.s2 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 32)]) : (FLOAT)0;
        }
        if ((first_output + lx + 48) < BANK_O) {
            output[(size_t)row0 * BANK_O + (first_output + lx + 48)] =
                valid0 ? (FLOAT)(sum0.s3 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 48)]) : (FLOAT)0;
        }
    }
    if (row1 >= 0) {
        if ((first_output + lx + 0) < BANK_O) {
            output[(size_t)row1 * BANK_O + (first_output + lx + 0)] =
                valid1 ? (FLOAT)(sum1.s0 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 0)]) : (FLOAT)0;
        }
        if ((first_output + lx + 16) < BANK_O) {
            output[(size_t)row1 * BANK_O + (first_output + lx + 16)] =
                valid1 ? (FLOAT)(sum1.s1 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 16)]) : (FLOAT)0;
        }
        if ((first_output + lx + 32) < BANK_O) {
            output[(size_t)row1 * BANK_O + (first_output + lx + 32)] =
                valid1 ? (FLOAT)(sum1.s2 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 32)]) : (FLOAT)0;
        }
        if ((first_output + lx + 48) < BANK_O) {
            output[(size_t)row1 * BANK_O + (first_output + lx + 48)] =
                valid1 ? (FLOAT)(sum1.s3 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 48)]) : (FLOAT)0;
        }
    }
    if (row2 >= 0) {
        if ((first_output + lx + 0) < BANK_O) {
            output[(size_t)row2 * BANK_O + (first_output + lx + 0)] =
                valid2 ? (FLOAT)(sum2.s0 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 0)]) : (FLOAT)0;
        }
        if ((first_output + lx + 16) < BANK_O) {
            output[(size_t)row2 * BANK_O + (first_output + lx + 16)] =
                valid2 ? (FLOAT)(sum2.s1 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 16)]) : (FLOAT)0;
        }
        if ((first_output + lx + 32) < BANK_O) {
            output[(size_t)row2 * BANK_O + (first_output + lx + 32)] =
                valid2 ? (FLOAT)(sum2.s2 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 32)]) : (FLOAT)0;
        }
        if ((first_output + lx + 48) < BANK_O) {
            output[(size_t)row2 * BANK_O + (first_output + lx + 48)] =
                valid2 ? (FLOAT)(sum2.s3 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 48)]) : (FLOAT)0;
        }
    }
    if (row3 >= 0) {
        if ((first_output + lx + 0) < BANK_O) {
            output[(size_t)row3 * BANK_O + (first_output + lx + 0)] =
                valid3 ? (FLOAT)(sum3.s0 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 0)]) : (FLOAT)0;
        }
        if ((first_output + lx + 16) < BANK_O) {
            output[(size_t)row3 * BANK_O + (first_output + lx + 16)] =
                valid3 ? (FLOAT)(sum3.s1 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 16)]) : (FLOAT)0;
        }
        if ((first_output + lx + 32) < BANK_O) {
            output[(size_t)row3 * BANK_O + (first_output + lx + 32)] =
                valid3 ? (FLOAT)(sum3.s2 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 32)]) : (FLOAT)0;
        }
        if ((first_output + lx + 48) < BANK_O) {
            output[(size_t)row3 * BANK_O + (first_output + lx + 48)] =
                valid3 ? (FLOAT)(sum3.s3 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 48)]) : (FLOAT)0;
        }
    }
#endif
}
#else
// M=8, N=32, K=16, workgroup=8x8, transpose=True
__kernel void indexed_linear_gemm(__global const FLOAT* input, __global const int* ids,
                                  __global const uchar* weights, __global const float2* alpha,
                                  __global const float* bias, __global FLOAT* output,
                                  __global const int* offsets, __global const int* sorted_rows,
                                  __global const int2* tiles,
                                  int unused_channels, int unused_outputs, int unused_topk, int unused_input_mode,
                                  int unused_groups, int unused_group_size, int unused_weight_stride) {
    const int2 tile = tiles[get_group_id(1)];
    if (tile.x < 0) return;
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int local_index = ly * 8 + lx;
    const int first_output = get_group_id(0) * 32;
    const int expert = tile.x, expert_end = offsets[expert + 1];
    __local float tile_input[128];
    __local float tile_weight[528];
    const int pos0 = tile.y + ly + 0;
    const int row0 = pos0 < expert_end ? sorted_rows[pos0] : -1;
    const int valid0 = row0 >= 0 && ids[row0] == expert;
    float4 sum0 = (float4)(0);
    for (int base_k = 0; base_k < BANK_I; base_k += 16) {
        for (int index = local_index; index < 128; index += 64) {
            const int token_position = tile.y + index / 16;
            const int token_row = token_position < expert_end ? sorted_rows[token_position] : -1;
            const int k = base_k + index % 16;
            const int input_row = BANK_INPUT_MODE ? token_row : token_row / BANK_TOPK;
            tile_input[index] = token_row >= 0 && ids[token_row] == expert && k < BANK_I ?
#if BANK_XC4
                (float)input[((size_t)(k / 4) * BANK_TOKENS + input_row) * 4 + (k & 3)] : 0.0f;
#else
                (float)input[(size_t)input_row * BANK_I + k] : 0.0f;
#endif
        }
        for (int index = local_index; index < 512; index += 64) {
            const int n = first_output + index / 16;
            const int k = base_k + index % 16;
            tile_weight[(index % 16) * 33 + index / 16] = n < BANK_O && k < BANK_I ?
                indexed_weight(weights, alpha, expert, n, k,
                               BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) {
            const float x0 = tile_input[(ly + 0) * 16 + k];
#if FUSE_SILU_MUL
            const float4 w0 = (float4)(tile_weight[k * 33 + 2 * lx], tile_weight[k * 33 + 2 * lx + 1],
                                      tile_weight[k * 33 + 2 * lx + 16],
                                      tile_weight[k * 33 + 2 * lx + 17]);
#else
            const float4 w0 = (float4)(tile_weight[k * 33 + (lx + 0)], tile_weight[k * 33 + (lx + 8)],
                                         tile_weight[k * 33 + (lx + 16)], tile_weight[k * 33 + (lx + 24)]);
#endif
            sum0 = mad((float4)(x0), w0, sum0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#if FUSE_SILU_MUL
    if (row0 >= 0) {
        if ((first_output + 2 * lx + 0) < BANK_O) {
            output[(size_t)row0 * BANK_RESULT_O + (first_output + 2 * lx + 0) / 2] = valid0 ?
                indexed_swiglu(sum0.s0 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0)],
                               sum0.s1 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 0) + 1]) : (FLOAT)0;
        }
        if ((first_output + 2 * lx + 16) < BANK_O) {
            output[(size_t)row0 * BANK_RESULT_O + (first_output + 2 * lx + 16) / 2] = valid0 ?
                indexed_swiglu(sum0.s2 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 16)],
                               sum0.s3 + bias[(size_t)expert * BANK_O + (first_output + 2 * lx + 16) + 1]) : (FLOAT)0;
        }
    }
#else
    if (row0 >= 0) {
        if ((first_output + lx + 0) < BANK_O) {
            output[(size_t)row0 * BANK_O + (first_output + lx + 0)] =
                valid0 ? (FLOAT)(sum0.s0 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 0)]) : (FLOAT)0;
        }
        if ((first_output + lx + 8) < BANK_O) {
            output[(size_t)row0 * BANK_O + (first_output + lx + 8)] =
                valid0 ? (FLOAT)(sum0.s1 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 8)]) : (FLOAT)0;
        }
        if ((first_output + lx + 16) < BANK_O) {
            output[(size_t)row0 * BANK_O + (first_output + lx + 16)] =
                valid0 ? (FLOAT)(sum0.s2 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 16)]) : (FLOAT)0;
        }
        if ((first_output + lx + 24) < BANK_O) {
            output[(size_t)row0 * BANK_O + (first_output + lx + 24)] =
                valid0 ? (FLOAT)(sum0.s3 +
                    bias[(size_t)expert * BANK_O + (first_output + lx + 24)]) : (FLOAT)0;
        }
    }
#endif
}
#endif

__kernel void indexed_linear_gemv_rw(__global const FLOAT* input, __global const int* ids,
                                     __global const uchar* weights, __global const float2* alpha,
                                     __global const float* bias, __global const FLOAT* routeWeights,
                                     __global FLOAT* output, int experts, float routeEps, float routeScale) {
    const int lane = get_local_id(0);
    const int first_output = get_group_id(0) * 4;
    const int token = get_group_id(1);
    __local float4 partial[64];
    __local float accShare[4];
#if RW_RAW
    float rsum = routeEps;
    for (int kk = 0; kk < BANK_TOPK; ++kk) {
        const int e = ids[token * BANK_TOPK + kk];
        if (e >= 0 && e < experts) {
            rsum += (float)routeWeights[(size_t)token * experts + e];
        }
    }
#endif
    if (lane < 4) {
        accShare[lane] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < BANK_TOPK; ++k) {
        const int route = token * BANK_TOPK + k;
        const int expert = ids[route];
        const bool valid = expert >= 0 && expert < experts;
#if RW_RAW
        const float w = valid ? ((float)routeWeights[(size_t)token * experts + expert] / rsum * routeScale) : 0.0f;
#else
        const float w = valid ? (float)routeWeights[route] : 0.0f;
#endif
        float4 sum = (float4)(0);
        if (valid) {
            for (int k2 = lane; k2 < BANK_I; k2 += 64) {
                const float x = (float)input[(size_t)route * BANK_I + k2];
                float4 weight = (float4)(0);
                if (first_output + 0 < BANK_RESULT_O)
                    weight.s0 = indexed_weight(weights, alpha, expert, first_output + 0, k2,
                        BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
                if (first_output + 1 < BANK_RESULT_O)
                    weight.s1 = indexed_weight(weights, alpha, expert, first_output + 1, k2,
                        BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
                if (first_output + 2 < BANK_RESULT_O)
                    weight.s2 = indexed_weight(weights, alpha, expert, first_output + 2, k2,
                        BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
                if (first_output + 3 < BANK_RESULT_O)
                    weight.s3 = indexed_weight(weights, alpha, expert, first_output + 3, k2,
                        BANK_I, BANK_O, BANK_GROUPS, BANK_GROUP_SIZE, BANK_STRIDE);
                sum = mad((float4)(x), weight, sum);
            }
        }
        partial[lane] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = 32; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane == 0) {
            const size_t bias_base = (size_t)expert * BANK_O + first_output;
            if (valid) {
                if (first_output + 0 < BANK_RESULT_O) accShare[0] += w * (partial[0].s0 + bias[bias_base + 0]);
                if (first_output + 1 < BANK_RESULT_O) accShare[1] += w * (partial[0].s1 + bias[bias_base + 1]);
                if (first_output + 2 < BANK_RESULT_O) accShare[2] += w * (partial[0].s2 + bias[bias_base + 2]);
                if (first_output + 3 < BANK_RESULT_O) accShare[3] += w * (partial[0].s3 + bias[bias_base + 3]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        for (int j = 0; j < 4 && first_output + j < BANK_RESULT_O; ++j) {
            const int c = first_output + j;
            output[((size_t)(c / 4) * BANK_TOKENS + token) * 4 + (c & 3)] = (FLOAT)accShare[j];
        }
    }
}

__kernel void indexed_linear_rw_reduce(__global const FLOAT* tmp, __global const FLOAT* routeWeights,
                                       __global FLOAT* output, __global const int* ids, int experts,
                                       float routeEps, float routeScale) {
    const int c = get_global_id(0);
    const int token = get_global_id(1);
    if (c >= BANK_RESULT_O || token >= BANK_TOKENS) {
        return;
    }
#if RW_RAW
    float rsum = routeEps;
    for (int kk = 0; kk < BANK_TOPK; ++kk) {
        const int e = ids[token * BANK_TOPK + kk];
        if (e >= 0 && e < experts) {
            rsum += (float)routeWeights[(size_t)token * experts + e];
        }
    }
#endif
    float acc = 0.0f;
    for (int k = 0; k < BANK_TOPK; ++k) {
#if RW_RAW
        const int e = ids[token * BANK_TOPK + k];
        if (e < 0 || e >= experts) {
            continue;
        }
        const float w = (float)routeWeights[(size_t)token * experts + e] / rsum * routeScale;
        acc = mad(w, (float)tmp[(size_t)(token * BANK_TOPK + k) * BANK_RESULT_O + c], acc);
#else
        acc = mad((float)routeWeights[token * BANK_TOPK + k],
                  (float)tmp[(size_t)(token * BANK_TOPK + k) * BANK_RESULT_O + c], acc);
#endif
    }
    output[((size_t)(c / 4) * BANK_TOKENS + token) * 4 + (c & 3)] = (FLOAT)acc;
}
