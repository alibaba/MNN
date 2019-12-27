// TODO: use INIT_SCALAR_VALUE, OPERATOR, FINAL_OPERATOR_ON_CHANNEL macro abstract and simplify code
// TODO: support reduce dims include batch
// TODO: support keep_dim=False
// TODO: fix channel reduce result re-pack problem
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void reduce_sum_all(__read_only image2d_t input, __write_only image2d_t output, int width, int channel) {
    if (get_global_id(0) != 0) {
        return;
    }
    const int total_h = get_image_dim(input).y, channelDiv4 = channel / 4;
    FLOAT sum = 0;
    for (int c = 0; c < channelDiv4; ++c) {
        for (int i = 0; i < total_h; ++i) {
            for (int w = 0; w < width; ++w) {
                FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + w, i));
                sum = sum + in.x + in.y + in.z + in.w;
            }
        }
    }
    const int remain = channel % 4;
    if (remain != 0) {
        const int offset = channelDiv4 * width;
        for (int i = 0; i < total_h; ++i) {
            for (int w = 0; w < width; ++w) {
                FLOAT4 in = RI_F(input, SAMPLER, (int2)(offset + w, i));
                if (remain == 1) {
                    sum = sum + in.x;
                } else if (remain == 2) {
                    sum = sum + in.x + in.y;
                } else if (remain == 3) {
                    sum = sum + in.x + in.y + in.z;
                }
            }
        }
    }
    WI_F(output, (int2)(0, 0), (FLOAT4)sum);
}

__kernel void reduce_along_channel(__read_only image2d_t input, __write_only image2d_t output, int width, int meanAggregation) {
    const int channel_block_idx = get_global_id(0), total_h = get_image_dim(input).y;
    const int w_offset = channel_block_idx * width;
    FLOAT4 out = (FLOAT4){0, 0, 0, 0};
    for (int i = 0; i < total_h; ++i) {
        for (int j = 0; j < width; ++j) {
            FLOAT4 in = RI_F(input, SAMPLER, (int2)(w_offset + j, i));
            out = out + in;
        }
    }
    int2 pos = (int2)(0, channel_block_idx);
    if (get_image_dim(output).y == 1) {
        pos = (int2)(channel_block_idx, 0);
    }
    if (meanAggregation) {
        out = out / (total_h * width);
    }
    WI_F(output, pos, out);
}

__kernel void reduce_sum_use_local_along_channel(__read_only image2d_t input, __write_only image2d_t output, int width, int meanAggregation, int step, __local float* results, int local_size) {
    const int tile_index = get_global_id(0), channel_block_idx = get_global_id(1);
    const int h_start = tile_index * step, h_end = min((tile_index + 1) * step, get_image_dim(input).y);
    const int w_offset = channel_block_idx * width;
    FLOAT4 out = (FLOAT4){0, 0, 0, 0};
    for (int i = h_start; i < h_end; ++i) {
        for (int j = 0; j < width; ++j) {
            FLOAT4 in = RI_F(input, SAMPLER, (int2)(w_offset + j, i));
            out = out + in;
        }
    }
    results[tile_index * 4 + 0] = out.x;
    results[tile_index * 4 + 1] = out.y;
    results[tile_index * 4 + 2] = out.z;
    results[tile_index * 4 + 3] = out.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tile_index == 0) {
        out = 0;
        for (int i = 0; i < local_size; i += 4) {
            out.x = out.x + results[i];
            out.y = out.y + results[i + 1];
            out.z = out.z + results[i + 2];
            out.w = out.w + results[i + 3];
        }
        if (meanAggregation) {
            out = out / (get_image_dim(input).y * width);
        }
        int2 pos = (int2)(0, channel_block_idx);
        if (get_image_dim(output).y == 1) {
            pos = (int2)(channel_block_idx, 0);
        }
        WI_F(output, pos, out);
    }
}

__kernel void reduct_1d(GLOBAL_SIZE_3_DIMS
                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int groupWorkSize,
                        __private const int computeNum,
                        __private const int lastNum,
                        __private const int reductSize,
                        __private const int workNum,
                        __private const int groupNum,
                        __private const int channels
                        ) {
    const int w = get_local_id(0);
    const int h = get_local_id(1);
    const int bg= get_global_id(2);
    const int width = get_local_size(0);
    const int index = mad24(h, width, w);
    const int b = bg / groupNum;
    const int group_index  = mad24(b, -groupNum, bg);
    const int remain_channel = channels % 4;

    FLOAT4 in;
    FLOAT4 scale;
    int pos_x, pos_y;
// MAX
#if REDUCE_TYPE == 1
    FLOAT4 tempResult = (FLOAT4){-MAXFLOAT, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
    FLOAT4 out = (FLOAT4){-MAXFLOAT, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
// MIN
#elif REDUCE_TYPE == 2
    FLOAT4 tempResult = (FLOAT4){MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT};
    FLOAT4 out = (FLOAT4){MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT};
// PROD
#elif REDUCE_TYPE == 3
    FLOAT4 tempResult = (FLOAT4){1, 1, 1, 1};
    FLOAT4 out = (FLOAT4){1, 1, 1, 1};
#else
// MEAN or SUM
    FLOAT4 tempResult = (FLOAT4){0, 0, 0, 0};
    FLOAT4 out = (FLOAT4){0, 0, 0, 0};
#endif
    const bool greater_last = (lastNum > 0 && index >= lastNum);
    const int actual_computeNum = select(computeNum, computeNum - 1, greater_last);
    if (actual_computeNum == 0)
        return;
    const int base_offset = mul24(index, actual_computeNum);
    const int offset= select(base_offset, base_offset + lastNum, greater_last);
    scale = (FLOAT4)(1.f / reductSize);
#ifdef REDUCTION_C
    scale = (FLOAT4)(1.f / channels);
#endif
#pragma unroll
    for (int i = 0; i < actual_computeNum; ++i) {
        int element_idx = offset + i;
#pragma unroll
        for (int j = 0; j < reductSize; j++) {
#ifdef REDUCTION_H
            pos_x = mad24(group_index, workNum, element_idx);
            pos_y = mad24(b, reductSize, j);
            in = RI_F(input, SAMPLER, (int2)(pos_x, pos_y));
#endif
#ifdef REDUCTION_W
            pos_x = mad24(group_index, reductSize, j);
            pos_y = mad24(b, workNum, element_idx);
            in = RI_F(input, SAMPLER, (int2)(pos_x, pos_y));
#endif
#ifdef REDUCTION_C
            pos_x = mad24(j, workNum, element_idx);
            pos_y = mad24(b, groupNum, group_index);
            in = RI_F(input, SAMPLER, (int2)(pos_x, pos_y));
            if (remain_channel != 0 && j == (reductSize - 1)) {
                if (remain_channel == 1) {
#if REDUCE_TYPE == 1
                    in = (FLOAT4){in.x, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
#elif REDUCE_TYPE == 2
                    in = (FLOAT4){in.x, MAXFLOAT, MAXFLOAT, MAXFLOAT};
#elif REDUCE_TYPE == 3
                    in = (FLOAT4){in.x, 1, 1, 1};
#else
                    in = (FLOAT4){in.x, 0, 0, 0};
#endif
                } else if (remain_channel == 2) {
#if REDUCE_TYPE == 1
                    in = (FLOAT4){in.x, in.y, -MAXFLOAT, -MAXFLOAT};
#elif REDUCE_TYPE == 2
                    in = (FLOAT4){in.x, in.y, MAXFLOAT, MAXFLOAT};
#elif REDUCE_TYPE == 3
                    in = (FLOAT4){in.x, in.y, 1, 1};
#else
                    in = (FLOAT4){in.x, in.y, 0, 0};
#endif
                } else if (remain_channel == 3) {
#if REDUCE_TYPE == 1
                    in.w = -MAXFLOAT;
#elif REDUCE_TYPE == 2
                    in.w = MAXFLOAT;
#elif REDUCE_TYPE == 3
                    in.w = 1;
#else
                    in.w = 0;
#endif
                }
            }
#endif
#if REDUCE_TYPE == 1
            tempResult = fmax(tempResult, in);
#elif REDUCE_TYPE == 2
            tempResult = fmin(tempResult, in);
#elif REDUCE_TYPE == 3
            tempResult = tempResult * in;
#else
            tempResult = tempResult + in;
#endif
        }
        
#if REDUCE_TYPE == 0
        tempResult = tempResult * scale;
#endif
        out = tempResult;
#ifdef REDUCTION_H
        WI_F(output, (int2)(pos_x, b), out);
#endif
#ifdef REDUCTION_W
        WI_F(output, (int2)(group_index, pos_y), out);
#endif
#ifdef REDUCTION_C
#if REDUCE_TYPE == 1
        float tmp_value = fmax(out.x, out.y);
        tmp_value = fmax(out.z, tmp_value);
        out.x = fmax(out.w, tmp_value);
#elif REDUCE_TYPE == 2
        float tmp_value = fmin(out.x, out.y);
        tmp_value = fmin(out.z, tmp_value);
        out.x = fmin(out.w, tmp_value);
#elif REDUCE_TYPE == 3
        out.x = out.x * out.y * out.z * out.w;
#else
        out.x = out.x + out.y + out.z + out.w;
#endif
        out = (FLOAT4){out.x, 0, 0, 0};
        WI_F(output, (int2)(pos_x % workNum, pos_y), out);
#endif
    }
}

__kernel void reduct_2d(GLOBAL_SIZE_3_DIMS
                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __global FLOAT4 *groupBuffer,
                        __global FLOAT *leftBuffer,
                        __private const int groupWorkSize,
                        __private const int computeNum,
                        __private const int lastNum,
                        __private const int inputHeight,
                        __private const int inputWidth,
                        __private const int leftSize,
                        __private const int channels
                        ) {
    const int w = get_local_id(0);
    const int h = get_local_id(1);
    const int bl= get_global_id(2);
    const int width = get_local_size(0);
    const int index = mad24(h, width, w);
    const int b = bl / leftSize;
    const int left_index  = mad24(b, -leftSize, bl);
    const int remain_channel = channels % 4;

    FLOAT4 in;
    bool channel_flag;
    FLOAT4 scale;
// MAX
#if REDUCE_TYPE == 1
    FLOAT4 tempResult = (FLOAT4){-MAXFLOAT, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
    FLOAT4 allResult = (FLOAT4){-MAXFLOAT, 0, 0, 0};
// MIN
#elif REDUCE_TYPE == 2
    FLOAT4 tempResult = (FLOAT4){MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT};
    FLOAT4 allResult = (FLOAT4){MAXFLOAT, 0, 0, 0};
// PROD
#elif REDUCE_TYPE == 3
    FLOAT4 tempResult = (FLOAT4){1, 1, 1, 1};
    FLOAT4 allResult = (FLOAT4){1, 0, 0, 0};
#else
// MEAN or SUM
    FLOAT4 tempResult = (FLOAT4){0, 0, 0, 0};
    FLOAT4 allResult = (FLOAT4){0, 0, 0, 0};
#endif
    const bool greater_last = (lastNum > 0 && index >= lastNum);
    // After last index, each kernel only computes (computeNum - 1) elements.
    const int actual_computeNum = select(computeNum, computeNum - 1, greater_last);
    const int base_offset = mul24(index, actual_computeNum);
    const int offset= select(base_offset, base_offset + lastNum, greater_last);
#pragma unroll
    for (int i = 0; i < actual_computeNum; ++i) {
        int element_idx = offset + i;
#ifdef REDUCTION_HW
        int h_idx = element_idx / inputWidth;
        int w_idx = mad24(h_idx, -inputWidth, element_idx);
        int pos_x = mad24(left_index, inputWidth, w_idx);
        int pos_y = mad24(b, inputHeight, h_idx);
        in = RI_F(input, SAMPLER, (int2)(pos_x, pos_y));
#endif
#ifdef REDUCTION_HC
        int h_idx = element_idx / inputWidth;
        int w_idx = mad24(h_idx, -inputWidth, element_idx);
        int pos_x = mad24(w_idx, leftSize, left_index);
        int pos_y = mad24(b, inputHeight, h_idx);
        in = RI_F(input, SAMPLER, (int2)(pos_x, pos_y));
        channel_flag = (remain_channel != 0 && w_idx == (inputWidth - 1));
#endif
#ifdef REDUCTION_WC
        int c_idx = element_idx / inputWidth;
        int pos_x = element_idx;
        int pos_y = mad24(b, leftSize, left_index);
        in = RI_F(input, SAMPLER, (int2)(pos_x, pos_y));
        channel_flag = (remain_channel != 0 && c_idx == (inputHeight - 1));
#endif
#ifndef REDUCTION_HW
        if (channel_flag) {
            if (remain_channel == 1) {
#if REDUCE_TYPE == 1
                in = (FLOAT4){in.x, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
#elif REDUCE_TYPE == 2
                in = (FLOAT4){in.x, MAXFLOAT, MAXFLOAT, MAXFLOAT};
#elif REDUCE_TYPE == 3
                in = (FLOAT4){in.x, 1, 1, 1};
#else
                in = (FLOAT4){in.x, 0, 0, 0};
#endif
            } else if (remain_channel == 2) {
#if REDUCE_TYPE == 1
                in = (FLOAT4){in.x, in.y, -MAXFLOAT, -MAXFLOAT};
#elif REDUCE_TYPE == 2
                in = (FLOAT4){in.x, in.y, MAXFLOAT, MAXFLOAT};
#elif REDUCE_TYPE == 3
                in = (FLOAT4){in.x, in.y, 1, 1};
#else
                in = (FLOAT4){in.x, in.y, 0, 0};
#endif
            } else if (remain_channel == 3) {
#if REDUCE_TYPE == 1
                in.w = -MAXFLOAT;
#elif REDUCE_TYPE == 2
                in.w = MAXFLOAT;
#elif REDUCE_TYPE == 3
                in.w = 1;
#else
                in.w = 0;
#endif
            }
        }
#endif
#if REDUCE_TYPE == 1
        tempResult = fmax(tempResult, in);
#elif REDUCE_TYPE == 2
        tempResult = fmin(tempResult, in);
#elif REDUCE_TYPE == 3
        tempResult = tempResult * in;
#else
        tempResult = tempResult + in;
#endif
    }
#ifdef REDUCTION_HW
    scale = (FLOAT4)(1.f / (inputHeight * inputWidth));
#endif
#ifdef REDUCTION_HC
#if REDUCE_W == 1
    scale = (FLOAT4)(1.f / (inputHeight * channels * leftSize));
#else
    scale = (FLOAT4)(1.f / (inputHeight * channels));
#endif
#endif
#ifdef REDUCTION_WC
    scale = (FLOAT4)(1.f / (inputWidth * channels));
#endif

// MEAN
#if REDUCE_TYPE == 0
    tempResult = tempResult * scale;
#endif
    groupBuffer[index] = tempResult;

#ifdef NON_QUALCOMM_ADRENO
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    if (w == 0 && h == 0) {
#if REDUCE_TYPE == 1
        FLOAT4 out = (FLOAT4){-MAXFLOAT, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
#elif REDUCE_TYPE == 2
        FLOAT4 out = (FLOAT4){MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT};
#elif REDUCE_TYPE == 3
        FLOAT4 out = (FLOAT4){1, 1, 1, 1};
#else
        FLOAT4 out = (FLOAT4){0, 0, 0, 0};
#endif
#pragma unroll
        for (int i = 0; i < groupWorkSize; ++i) {
#if REDUCE_TYPE == 1
            out = fmax(out, groupBuffer[i]);
#elif REDUCE_TYPE == 2
            out = fmin(out, groupBuffer[i]);
#elif REDUCE_TYPE == 3
            out = out * groupBuffer[i];
#else
            out = out + groupBuffer[i];
#endif
        }
#ifdef REDUCTION_HW
        WI_F(output, (int2)(left_index, b), out);
#endif
#ifndef REDUCTION_HW
#if REDUCE_TYPE == 1
        float tmp_value = fmax(out.x, out.y);
        tmp_value = fmax(out.z, tmp_value);
        out.x = fmax(out.w, tmp_value);
#elif REDUCE_TYPE == 2
        float tmp_value = fmin(out.x, out.y);
        tmp_value = fmin(out.z, tmp_value);
        out.x = fmin(out.w, tmp_value);
#elif REDUCE_TYPE == 3
        out.x = out.x * out.y * out.z * out.w;
#else
        out.x = out.x + out.y + out.z + out.w;
#endif
        out = (FLOAT4){out.x, 0, 0, 0};
#endif
#ifdef REDUCTION_HC
#if REDUCE_W == 1
        leftBuffer[left_index] = out.x;
#ifdef NON_QUALCOMM_ADRENO
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
        if (left_index == 0) {
            for (int i = 0; i < leftSize; ++i) {
#if REDUCE_TYPE == 1
                allResult.x = fmax(allResult.x, leftBuffer[i]);
#elif REDUCE_TYPE == 2
                allResult.x = fmin(allResult.x, leftBuffer[i]);
#elif REDUCE_TYPE == 3
                allResult.x = allResult.x * leftBuffer[i];
#else
                allResult.x = allResult.x + leftBuffer[i];
#endif
            }
            WI_F(output, (int2)(0, 0), allResult);
        }
#else
        WI_F(output, (int2)(left_index, b), out);
#endif
#endif
#ifdef REDUCTION_WC
        WI_F(output, (int2)(0, b * leftSize + left_index), out);
#endif
    }
}
