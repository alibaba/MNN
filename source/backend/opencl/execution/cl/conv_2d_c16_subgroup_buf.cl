#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifdef MNN_SUPPORT_FP16
#define GROUP_READ(ptr, offset)            as_half(intel_sub_group_block_read_us((const __global ushort*)(ptr) + (offset)))
#define GROUP_READ2(ptr, offset)           as_half2(intel_sub_group_block_read_us2((const __global ushort*)(ptr) + (offset)))
#define GROUP_READ4(ptr, offset)           as_half4(intel_sub_group_block_read_us4((const __global ushort*)(ptr) + (offset)))
#define GROUP_READ8(ptr, offset)           as_half8(intel_sub_group_block_read_us8((const __global ushort*)(ptr) + (offset)))

#define GROUP_WRITE(ptr, offset, val)      intel_sub_group_block_write_us((const __global ushort*)(ptr) + (offset), as_ushort(val))
#define GROUP_WRITE2(ptr, offset, val)     intel_sub_group_block_write_us2((const __global ushort*)(ptr) + (offset), as_ushort2(val))
#define GROUP_WRITE4(ptr, offset, val)     intel_sub_group_block_write_us4((const __global ushort*)(ptr) + (offset), as_ushort4(val))
#define GROUP_WRITE8(ptr, offset, val)     intel_sub_group_block_write_us8((const __global ushort*)(ptr) + (offset), as_ushort8(val))

#define GROUP_SHUFFLE(data, id)            as_half(intel_sub_group_shuffle(as_ushort(data), id))
#define GROUP_SHUFFLE2(data, id)           as_half2(intel_sub_group_shuffle(as_ushort2(data), id))
#define GROUP_SHUFFLE4(data, id)           as_half4(intel_sub_group_shuffle(as_ushort4(data), id))
#define GROUP_SHUFFLE8(data, id)           as_half8(intel_sub_group_shuffle(as_ushort8(data), id))
#else
#define GROUP_READ(ptr, offset)            as_float(intel_sub_group_block_read((const __global uint*)(ptr) + (offset)))
#define GROUP_READ2(ptr, offset)           as_float2(intel_sub_group_block_read2((const __global uint*)(ptr) + (offset)))
#define GROUP_READ4(ptr, offset)           as_float4(intel_sub_group_block_read4((const __global uint*)(ptr) + (offset)))
#define GROUP_READ8(ptr, offset)           as_float8(intel_sub_group_block_read8((const __global uint*)(ptr) + (offset)))

#define GROUP_WRITE(ptr, offset, val)      intel_sub_group_block_write((const __global uint*)(ptr) + (offset), as_uint(val))
#define GROUP_WRITE2(ptr, offset, val)     intel_sub_group_block_write2((const __global uint*)(ptr) + (offset), as_uint2(val))
#define GROUP_WRITE4(ptr, offset, val)     intel_sub_group_block_write4((const __global uint*)(ptr) + (offset), as_uint4(val))
#define GROUP_WRITE8(ptr, offset, val)     intel_sub_group_block_write8((const __global uint*)(ptr) + (offset), as_uint8(val))

#define GROUP_SHUFFLE(data, id)            intel_sub_group_shuffle(data, id)
#define GROUP_SHUFFLE2(data, id)           intel_sub_group_shuffle(data, id)
#define GROUP_SHUFFLE4(data, id)           intel_sub_group_shuffle(data, id)
#define GROUP_SHUFFLE8(data, id)           intel_sub_group_shuffle(data, id)
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c16_c4_b2(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,    
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left, 
    __private const int input_pad_right,
    __private const int output_pad_left, 
    __private const int output_pad_right
) {
    const int sglid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 1;
    const int y = (xy / x_blocks);

    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    const int feature_block = (int)get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (input_pad_left + input_width + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (input_height);
    const uint input_b_pitch = input_fs_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch * output_width;
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch *  ((output_channel + 3) / 4);

    const uint output_offset = b * output_b_pitch +
                               (feature_block << 2) * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 16 * 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = feature_block * filter_os_pitch;

#if SLM_DIV_FACTOR == 1
    COMPUTE_FLOAT2 dst = (COMPUTE_FLOAT2)((GROUP_READ(biases, feature_block * 16)));
#else
    COMPUTE_FLOAT2 dst;

    if (feature_sub_block == 0) {
        dst = (COMPUTE_FLOAT2)((GROUP_READ(biases, feature_block * 16)));
    } else {
        dst = (COMPUTE_FLOAT2)0;
    }
#endif 

#if SLM_DIV_FACTOR > 1
    __local COMPUTE_FLOAT2 sum[WORK_GROUP_SIZE];
#endif


#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif 
            __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
            for (int kh = 0; kh < FILTER_HEIGHT; kh++) {
                if (input_y + kh * DILATION_HEIGHT < 0 || input_y + kh * DILATION_HEIGHT >= input_height)
                    continue;

                FLOAT line_cache[INPUT_LINE_SIZE];

                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        COMPUTE_FLOAT8 tmp = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                        line_cache[xb + 4] = tmp[4];
                        line_cache[xb + 5] = tmp[5];
                        line_cache[xb + 6] = tmp[6];
                        line_cache[xb + 7] = tmp[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        COMPUTE_FLOAT4 tmp = CONVERT_COMPUTE_FLOAT4(GROUP_READ4(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = GROUP_READ(input, input_offset +
                                                             icb * input_fs_pitch +
                                                             kh * DILATION_HEIGHT * input_y_pitch +
                                                             xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
                for (int kw = 0; kw < FILTER_WIDTH; kw++) {
                    FLOAT2 src;
                    __attribute__((opencl_unroll_hint(2)))
                    for (int i = 0; i < 2; i++) {
#if FILTER_WIDTH == 1 && DILATION_WIDTH == 1 && STRIDE_WIDTH == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_WIDTH + STRIDE_WIDTH * i];
#endif
                    }
                    COMPUTE_FLOAT8 weight0 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch));
                    COMPUTE_FLOAT8 weight1 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch +
                                                                    8 * filter_isv_pitch));
                    const COMPUTE_FLOAT2 src0  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 0));
                    const COMPUTE_FLOAT2 src1  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 1));
                    const COMPUTE_FLOAT2 src2  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 2));
                    const COMPUTE_FLOAT2 src3  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 3));
                    const COMPUTE_FLOAT2 src4  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 4));
                    const COMPUTE_FLOAT2 src5  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 5));
                    const COMPUTE_FLOAT2 src6  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 6));
                    const COMPUTE_FLOAT2 src7  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 7));
                    const COMPUTE_FLOAT2 src8  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 8));
                    const COMPUTE_FLOAT2 src9  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 9));
                    const COMPUTE_FLOAT2 src10 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 10));
                    const COMPUTE_FLOAT2 src11 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 11));
                    const COMPUTE_FLOAT2 src12 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 12));
                    const COMPUTE_FLOAT2 src13 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 13));
                    const COMPUTE_FLOAT2 src14 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 14));
                    const COMPUTE_FLOAT2 src15 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 15));

                    dst = mad(weight0.s0, src0,  dst);
                    dst = mad(weight0.s1, src1,  dst);
                    dst = mad(weight0.s2, src2,  dst);
                    dst = mad(weight0.s3, src3,  dst);
                    dst = mad(weight0.s4, src4,  dst);
                    dst = mad(weight0.s5, src5,  dst);
                    dst = mad(weight0.s6, src6,  dst);
                    dst = mad(weight0.s7, src7,  dst);
                    dst = mad(weight1.s0, src8,  dst);
                    dst = mad(weight1.s1, src9,  dst);
                    dst = mad(weight1.s2, src10, dst);
                    dst = mad(weight1.s3, src11, dst);
                    dst = mad(weight1.s4, src12, dst);
                    dst = mad(weight1.s5, src13, dst);
                    dst = mad(weight1.s6, src14, dst);
                    dst = mad(weight1.s7, src15, dst);
                }
            }
        }
        
#if SLM_DIV_FACTOR > 1
    sum[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        __attribute__((opencl_unroll_hint)) for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += sum[lid1 % feature_per_wg + i * feature_per_wg];
#endif

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif

    const uint lid_x = sglid % 4;
    const uint lid_y = sglid / 4;

    if ((feature_block+1)*16 >= output_channel) {
        for (int i = 0; i < 2 && (x + i) < output_width; i++) {
            if ((feature_block*16 + lid_y * 4 + lid_x < output_channel))
                output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
    else
    {
        for (int i = 0; i < 2 && (x + i) < output_width; i++) {
            output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
#if SLM_DIV_FACTOR > 1
    }
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c16_c4_b4(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,    
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left, 
    __private const int input_pad_right,
    __private const int output_pad_left, 
    __private const int output_pad_right
) {
    const int sglid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 2;
    const int y = (xy / x_blocks);

    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    const int feature_block = (int)get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (input_pad_left + input_width + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (input_height);
    const uint input_b_pitch = input_fs_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch * output_width;
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch *  ((output_channel + 3) / 4);

    const uint output_offset = b * output_b_pitch +
                               (feature_block << 2) * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 16 * 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = feature_block * filter_os_pitch;

#if SLM_DIV_FACTOR == 1
    COMPUTE_FLOAT4 dst = (COMPUTE_FLOAT4)((GROUP_READ(biases, feature_block * 16)));
#else
    COMPUTE_FLOAT4 dst;

    if (feature_sub_block == 0) {
        dst = (COMPUTE_FLOAT4)((GROUP_READ(biases, feature_block * 16)));
    } else {
        dst = (COMPUTE_FLOAT4)0;
    }
#endif 

#if SLM_DIV_FACTOR > 1
    __local COMPUTE_FLOAT4 sum[WORK_GROUP_SIZE];
#endif


#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif 
            __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
            for (int kh = 0; kh < FILTER_HEIGHT; kh++) {
                if (input_y + kh * DILATION_HEIGHT < 0 || input_y + kh * DILATION_HEIGHT >= input_height)
                    continue;

                FLOAT line_cache[INPUT_LINE_SIZE];

                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        COMPUTE_FLOAT8 tmp = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                        line_cache[xb + 4] = tmp[4];
                        line_cache[xb + 5] = tmp[5];
                        line_cache[xb + 6] = tmp[6];
                        line_cache[xb + 7] = tmp[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        COMPUTE_FLOAT4 tmp = CONVERT_COMPUTE_FLOAT4(GROUP_READ4(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = (COMPUTE_FLOAT)GROUP_READ(input, input_offset +
                                                             icb * input_fs_pitch +
                                                             kh * DILATION_HEIGHT * input_y_pitch +
                                                             xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
                for (int kw = 0; kw < FILTER_WIDTH; kw++) {
                    FLOAT4 src;
                    __attribute__((opencl_unroll_hint(4)))
                    for (int i = 0; i < 4; i++) {
#if FILTER_WIDTH == 1 && DILATION_WIDTH == 1 && STRIDE_WIDTH == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_WIDTH + STRIDE_WIDTH * i];
#endif
                    }
                    COMPUTE_FLOAT8 weight0 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch));
                    COMPUTE_FLOAT8 weight1 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch +
                                                                    8 * filter_isv_pitch));
                    const COMPUTE_FLOAT4 src0  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 0));
                    const COMPUTE_FLOAT4 src1  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 1));
                    const COMPUTE_FLOAT4 src2  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 2));
                    const COMPUTE_FLOAT4 src3  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 3));
                    const COMPUTE_FLOAT4 src4  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 4));
                    const COMPUTE_FLOAT4 src5  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 5));
                    const COMPUTE_FLOAT4 src6  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 6));
                    const COMPUTE_FLOAT4 src7  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 7));
                    const COMPUTE_FLOAT4 src8  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 8));
                    const COMPUTE_FLOAT4 src9  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 9));
                    const COMPUTE_FLOAT4 src10 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 10));
                    const COMPUTE_FLOAT4 src11 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 11));
                    const COMPUTE_FLOAT4 src12 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 12));
                    const COMPUTE_FLOAT4 src13 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 13));
                    const COMPUTE_FLOAT4 src14 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 14));
                    const COMPUTE_FLOAT4 src15 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 15));

                    dst = mad(weight0.s0, src0,  dst);
                    dst = mad(weight0.s1, src1,  dst);
                    dst = mad(weight0.s2, src2,  dst);
                    dst = mad(weight0.s3, src3,  dst);
                    dst = mad(weight0.s4, src4,  dst);
                    dst = mad(weight0.s5, src5,  dst);
                    dst = mad(weight0.s6, src6,  dst);
                    dst = mad(weight0.s7, src7,  dst);
                    dst = mad(weight1.s0, src8,  dst);
                    dst = mad(weight1.s1, src9,  dst);
                    dst = mad(weight1.s2, src10, dst);
                    dst = mad(weight1.s3, src11, dst);
                    dst = mad(weight1.s4, src12, dst);
                    dst = mad(weight1.s5, src13, dst);
                    dst = mad(weight1.s6, src14, dst);
                    dst = mad(weight1.s7, src15, dst);
                }
            }
        }
        
#if SLM_DIV_FACTOR > 1
    sum[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        __attribute__((opencl_unroll_hint)) for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += sum[lid1 % feature_per_wg + i * feature_per_wg];
#endif

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    const uint lid_x = sglid % 4;
    const uint lid_y = sglid / 4;

    if ((feature_block+1)*16 >= output_channel) {
        for (int i = 0; i < 4 && (x + i) < output_width; i++) {
            if ((feature_block*16 + lid_y * 4 + lid_x < output_channel))
                output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
    else
    {
        for (int i = 0; i < 4 && (x + i) < output_width; i++) {
            output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
#if SLM_DIV_FACTOR > 1
    }
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c16_c4_b8(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,    
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left, 
    __private const int input_pad_right,
    __private const int output_pad_left, 
    __private const int output_pad_right
) {
    const int sglid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 3;
    const int y = (xy / x_blocks);

    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    const int feature_block = (int)get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (input_pad_left + input_width + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (input_height);
    const uint input_b_pitch = input_fs_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch * output_width;
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch *  ((output_channel + 3) / 4);

    const uint output_offset = b * output_b_pitch +
                               (feature_block << 2) * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 16 * 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = feature_block * filter_os_pitch;

#if SLM_DIV_FACTOR == 1
    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(GROUP_READ(biases, feature_block * 16));
#else
    COMPUTE_FLOAT8 dst;

    if (feature_sub_block == 0) {
        dst = (COMPUTE_FLOAT8)(GROUP_READ(biases, feature_block * 16));
    } else {
        dst = (COMPUTE_FLOAT8)0;
    }
#endif 

#if SLM_DIV_FACTOR > 1
    __local COMPUTE_FLOAT8 sum[WORK_GROUP_SIZE];
#endif


#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif 
            __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
            for (int kh = 0; kh < FILTER_HEIGHT; kh++) {
                if (input_y + kh * DILATION_HEIGHT < 0 || input_y + kh * DILATION_HEIGHT >= input_height)
                    continue;

                FLOAT line_cache[INPUT_LINE_SIZE];

                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        COMPUTE_FLOAT8 tmp = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                        line_cache[xb + 4] = tmp[4];
                        line_cache[xb + 5] = tmp[5];
                        line_cache[xb + 6] = tmp[6];
                        line_cache[xb + 7] = tmp[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        COMPUTE_FLOAT4 tmp = CONVERT_COMPUTE_FLOAT4(GROUP_READ4(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = (COMPUTE_FLOAT)GROUP_READ(input, input_offset +
                                                             icb * input_fs_pitch +
                                                             kh * DILATION_HEIGHT * input_y_pitch +
                                                             xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
                for (int kw = 0; kw < FILTER_WIDTH; kw++) {
                    FLOAT8 src;
                    __attribute__((opencl_unroll_hint(8)))
                    for (int i = 0; i < 8; i++) {
#if FILTER_WIDTH == 1 && DILATION_WIDTH == 1 && STRIDE_WIDTH == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_WIDTH + STRIDE_WIDTH * i];
#endif
                    }
                    COMPUTE_FLOAT8 weight0 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch));
                    COMPUTE_FLOAT8 weight1 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch +
                                                                    8 * filter_isv_pitch));
                    const COMPUTE_FLOAT8 src0  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 0));
                    const COMPUTE_FLOAT8 src1  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 1));
                    const COMPUTE_FLOAT8 src2  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 2));
                    const COMPUTE_FLOAT8 src3  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 3));
                    const COMPUTE_FLOAT8 src4  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 4));
                    const COMPUTE_FLOAT8 src5  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 5));
                    const COMPUTE_FLOAT8 src6  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 6));
                    const COMPUTE_FLOAT8 src7  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 7));
                    const COMPUTE_FLOAT8 src8  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 8));
                    const COMPUTE_FLOAT8 src9  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 9));
                    const COMPUTE_FLOAT8 src10 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 10));
                    const COMPUTE_FLOAT8 src11 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 11));
                    const COMPUTE_FLOAT8 src12 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 12));
                    const COMPUTE_FLOAT8 src13 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 13));
                    const COMPUTE_FLOAT8 src14 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 14));
                    const COMPUTE_FLOAT8 src15 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 15));

                    dst = mad(weight0.s0, src0,  dst);
                    dst = mad(weight0.s1, src1,  dst);
                    dst = mad(weight0.s2, src2,  dst);
                    dst = mad(weight0.s3, src3,  dst);
                    dst = mad(weight0.s4, src4,  dst);
                    dst = mad(weight0.s5, src5,  dst);
                    dst = mad(weight0.s6, src6,  dst);
                    dst = mad(weight0.s7, src7,  dst);
                    dst = mad(weight1.s0, src8,  dst);
                    dst = mad(weight1.s1, src9,  dst);
                    dst = mad(weight1.s2, src10, dst);
                    dst = mad(weight1.s3, src11, dst);
                    dst = mad(weight1.s4, src12, dst);
                    dst = mad(weight1.s5, src13, dst);
                    dst = mad(weight1.s6, src14, dst);
                    dst = mad(weight1.s7, src15, dst);
                }
            }
        }
        
#if SLM_DIV_FACTOR > 1
    sum[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        __attribute__((opencl_unroll_hint)) for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += sum[lid1 % feature_per_wg + i * feature_per_wg];
#endif

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif

    const uint lid_x = sglid % 4;
    const uint lid_y = sglid / 4;

    if ((feature_block+1)*16 >= output_channel) {
        for (int i = 0; i < 8 && (x + i) < output_width; i++) {
            if ((feature_block*16 + lid_y * 4 + lid_x < output_channel))
                output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
    else
    {
        for (int i = 0; i < 8 && (x + i) < output_width; i++) {
            output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
#if SLM_DIV_FACTOR > 1
    }
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c16_c16_b2(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,    
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left, 
    __private const int input_pad_right,
    __private const int output_pad_left, 
    __private const int output_pad_right
) {
    const int sglid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 1;
    const int y = (xy / x_blocks);

    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    const int feature_block = (int)get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (input_pad_left + input_width + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (input_height);
    const uint input_b_pitch = input_fs_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch * (output_pad_left + output_width + output_pad_right);
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch *  ((output_channel + 15) / 16);

    const uint output_offset = b * output_b_pitch +
                               feature_block * output_fs_pitch +
                               y * output_y_pitch +
                               (x + output_pad_left) * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 16 * 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = feature_block * filter_os_pitch;

#if SLM_DIV_FACTOR == 1
    COMPUTE_FLOAT2 dst = (COMPUTE_FLOAT2)(GROUP_READ(biases, feature_block * 16));
#else
    COMPUTE_FLOAT2 dst;

    if (feature_sub_block == 0) {
        dst = (COMPUTE_FLOAT2)(GROUP_READ(biases, feature_block * 16));
    } else {
        dst = (COMPUTE_FLOAT2)0;
    }
#endif 

#if SLM_DIV_FACTOR > 1
    __local COMPUTE_FLOAT2 sum[WORK_GROUP_SIZE];
#endif


#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif 
            __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
            for (int kh = 0; kh < FILTER_HEIGHT; kh++) {
                if (input_y + kh * DILATION_HEIGHT < 0 || input_y + kh * DILATION_HEIGHT >= input_height)
                    continue;

                FLOAT line_cache[INPUT_LINE_SIZE];

                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        COMPUTE_FLOAT8 tmp = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                        line_cache[xb + 4] = tmp[4];
                        line_cache[xb + 5] = tmp[5];
                        line_cache[xb + 6] = tmp[6];
                        line_cache[xb + 7] = tmp[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        COMPUTE_FLOAT4 tmp = CONVERT_COMPUTE_FLOAT4(GROUP_READ4(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = (COMPUTE_FLOAT)GROUP_READ(input, input_offset +
                                                             icb * input_fs_pitch +
                                                             kh * DILATION_HEIGHT * input_y_pitch +
                                                             xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
                for (int kw = 0; kw < FILTER_WIDTH; kw++) {
                    FLOAT2 src;
                    __attribute__((opencl_unroll_hint(2)))
                    for (int i = 0; i < 2; i++) {
#if FILTER_WIDTH == 1 && DILATION_WIDTH == 1 && STRIDE_WIDTH == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_WIDTH + STRIDE_WIDTH * i];
#endif
                    }
                    COMPUTE_FLOAT8 weight0 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch));
                    COMPUTE_FLOAT8 weight1 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch +
                                                                    8 * filter_isv_pitch));
                    const COMPUTE_FLOAT2 src0  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 0));
                    const COMPUTE_FLOAT2 src1  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 1));
                    const COMPUTE_FLOAT2 src2  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 2));
                    const COMPUTE_FLOAT2 src3  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 3));
                    const COMPUTE_FLOAT2 src4  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 4));
                    const COMPUTE_FLOAT2 src5  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 5));
                    const COMPUTE_FLOAT2 src6  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 6));
                    const COMPUTE_FLOAT2 src7  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 7));
                    const COMPUTE_FLOAT2 src8  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 8));
                    const COMPUTE_FLOAT2 src9  = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 9));
                    const COMPUTE_FLOAT2 src10 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 10));
                    const COMPUTE_FLOAT2 src11 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 11));
                    const COMPUTE_FLOAT2 src12 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 12));
                    const COMPUTE_FLOAT2 src13 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 13));
                    const COMPUTE_FLOAT2 src14 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 14));
                    const COMPUTE_FLOAT2 src15 = CONVERT_COMPUTE_FLOAT2(GROUP_SHUFFLE2(src, 15));

                    dst = mad(weight0.s0, src0,  dst);
                    dst = mad(weight0.s1, src1,  dst);
                    dst = mad(weight0.s2, src2,  dst);
                    dst = mad(weight0.s3, src3,  dst);
                    dst = mad(weight0.s4, src4,  dst);
                    dst = mad(weight0.s5, src5,  dst);
                    dst = mad(weight0.s6, src6,  dst);
                    dst = mad(weight0.s7, src7,  dst);
                    dst = mad(weight1.s0, src8,  dst);
                    dst = mad(weight1.s1, src9,  dst);
                    dst = mad(weight1.s2, src10, dst);
                    dst = mad(weight1.s3, src11, dst);
                    dst = mad(weight1.s4, src12, dst);
                    dst = mad(weight1.s5, src13, dst);
                    dst = mad(weight1.s6, src14, dst);
                    dst = mad(weight1.s7, src15, dst);
                }
            }
        }

    if(x == 0){
        uint pad_offset = b * output_b_pitch + feature_block * output_fs_pitch + y * output_y_pitch;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
        pad_offset += (output_width + output_pad_left) * output_x_pitch;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
    }
        
#if SLM_DIV_FACTOR > 1
    sum[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        __attribute__((opencl_unroll_hint)) for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += sum[lid1 % feature_per_wg + i * feature_per_wg];
#endif

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif

    if ((feature_block+1)*16 >= output_channel) {
        for (int i = 0; i < 2; i++) {
            if ((feature_block*16 + sglid < output_channel) && (x + i) < output_width)
                output[output_offset + i * output_x_pitch + sglid] = (FLOAT)dst[i];
        }
    }
    else
    {
        if (x + 2 <= output_width || output_width % 2 == 0) {
            GROUP_WRITE2(output, output_offset, CONVERT_FLOAT2(dst));
        }else{
            for (int i = 0; i < output_width % 2; i++) {
                output[output_offset + i * output_x_pitch + sglid] = (FLOAT)dst[i];
            }
        }
    }
#if SLM_DIV_FACTOR > 1
    }
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c16_c16_b4(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,    
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left, 
    __private const int input_pad_right,
    __private const int output_pad_left, 
    __private const int output_pad_right
) {
    const int sglid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 2;
    const int y = (xy / x_blocks);

    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    const int feature_block = (int)get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (input_pad_left + input_width + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (input_height);
    const uint input_b_pitch = input_fs_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch * (output_pad_left + output_width + output_pad_right);
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch *  ((output_channel + 15) / 16);

    const uint output_offset = b * output_b_pitch +
                               feature_block * output_fs_pitch +
                               y * output_y_pitch +
                               (x + output_pad_left) * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 16 * 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = feature_block * filter_os_pitch;

#if SLM_DIV_FACTOR == 1
    COMPUTE_FLOAT4 dst = (COMPUTE_FLOAT4)(GROUP_READ(biases, feature_block * 16));
#else
    COMPUTE_FLOAT4 dst;

    if (feature_sub_block == 0) {
        dst = (COMPUTE_FLOAT4)(GROUP_READ(biases, feature_block * 16));
    } else {
        dst = (COMPUTE_FLOAT4)0;
    }
#endif 

#if SLM_DIV_FACTOR > 1
    __local COMPUTE_FLOAT4 sum[WORK_GROUP_SIZE];
#endif


#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif 
            __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
            for (int kh = 0; kh < FILTER_HEIGHT; kh++) {
                if (input_y + kh * DILATION_HEIGHT < 0 || input_y + kh * DILATION_HEIGHT >= input_height)
                    continue;

                FLOAT line_cache[INPUT_LINE_SIZE];

                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        COMPUTE_FLOAT8 tmp = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                        line_cache[xb + 4] = tmp[4];
                        line_cache[xb + 5] = tmp[5];
                        line_cache[xb + 6] = tmp[6];
                        line_cache[xb + 7] = tmp[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        COMPUTE_FLOAT4 tmp = CONVERT_COMPUTE_FLOAT4(GROUP_READ4(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = (COMPUTE_FLOAT)GROUP_READ(input, input_offset +
                                                             icb * input_fs_pitch +
                                                             kh * DILATION_HEIGHT * input_y_pitch +
                                                             xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
                for (int kw = 0; kw < FILTER_WIDTH; kw++) {
                    FLOAT4 src;
                    __attribute__((opencl_unroll_hint(4)))
                    for (int i = 0; i < 4; i++) {
#if FILTER_WIDTH == 1 && DILATION_WIDTH == 1 && STRIDE_WIDTH == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_WIDTH + STRIDE_WIDTH * i];
#endif
                    }
                    COMPUTE_FLOAT8 weight0 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch));
                    COMPUTE_FLOAT8 weight1 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch +
                                                                    8 * filter_isv_pitch));
                    const COMPUTE_FLOAT4 src0  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 0));
                    const COMPUTE_FLOAT4 src1  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 1));
                    const COMPUTE_FLOAT4 src2  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 2));
                    const COMPUTE_FLOAT4 src3  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 3));
                    const COMPUTE_FLOAT4 src4  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 4));
                    const COMPUTE_FLOAT4 src5  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 5));
                    const COMPUTE_FLOAT4 src6  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 6));
                    const COMPUTE_FLOAT4 src7  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 7));
                    const COMPUTE_FLOAT4 src8  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 8));
                    const COMPUTE_FLOAT4 src9  = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 9));
                    const COMPUTE_FLOAT4 src10 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 10));
                    const COMPUTE_FLOAT4 src11 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 11));
                    const COMPUTE_FLOAT4 src12 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 12));
                    const COMPUTE_FLOAT4 src13 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 13));
                    const COMPUTE_FLOAT4 src14 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 14));
                    const COMPUTE_FLOAT4 src15 = CONVERT_COMPUTE_FLOAT4(GROUP_SHUFFLE4(src, 15));

                    dst = mad(weight0.s0, src0,  dst);
                    dst = mad(weight0.s1, src1,  dst);
                    dst = mad(weight0.s2, src2,  dst);
                    dst = mad(weight0.s3, src3,  dst);
                    dst = mad(weight0.s4, src4,  dst);
                    dst = mad(weight0.s5, src5,  dst);
                    dst = mad(weight0.s6, src6,  dst);
                    dst = mad(weight0.s7, src7,  dst);
                    dst = mad(weight1.s0, src8,  dst);
                    dst = mad(weight1.s1, src9,  dst);
                    dst = mad(weight1.s2, src10, dst);
                    dst = mad(weight1.s3, src11, dst);
                    dst = mad(weight1.s4, src12, dst);
                    dst = mad(weight1.s5, src13, dst);
                    dst = mad(weight1.s6, src14, dst);
                    dst = mad(weight1.s7, src15, dst);
                }
            }
        }

    if(x == 0){
        uint pad_offset = b * output_b_pitch + feature_block * output_fs_pitch + y * output_y_pitch;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
        pad_offset += (output_width + output_pad_left) * output_x_pitch;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
    }

#if SLM_DIV_FACTOR > 1
    sum[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        __attribute__((opencl_unroll_hint)) for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += sum[lid1 % feature_per_wg + i * feature_per_wg];
#endif

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    if ((feature_block+1)*16 >= output_channel) {
        for (int i = 0; i < 4; i++) {
            if ((feature_block*16 + sglid < output_channel) && (x + i) < output_width)
                output[output_offset + i * output_x_pitch + sglid] = (FLOAT)dst[i];
        }
    }
    else
    {
        if (x + 4 <= output_width || output_width % 4 == 0) {
            GROUP_WRITE4(output, output_offset, CONVERT_FLOAT4(dst));
        }else{
            for (int i = 0; i < output_width % 4; i++) {
                output[output_offset + i * output_x_pitch + sglid] = (FLOAT)dst[i];
            }
        }
    }
#if SLM_DIV_FACTOR > 1
    }
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c16_c16_b8(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,    
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left, 
    __private const int input_pad_right,
    __private const int output_pad_left, 
    __private const int output_pad_right
) {
    const int sglid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 3;
    const int y = (xy / x_blocks);

    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    const int feature_block = (int)get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (input_pad_left + input_width + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (input_height);
    const uint input_b_pitch = input_fs_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch * (output_pad_left + output_width + output_pad_right);
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch *  ((output_channel + 15) / 16);

    const uint output_offset = b * output_b_pitch +
                               feature_block * output_fs_pitch +
                               y * output_y_pitch +
                               (x + output_pad_left) * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 16 * 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = feature_block * filter_os_pitch;

#if SLM_DIV_FACTOR == 1
    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(GROUP_READ(biases, feature_block * 16));
#else
    COMPUTE_FLOAT8 dst;

    if (feature_sub_block == 0) {
        dst = (COMPUTE_FLOAT8)(GROUP_READ(biases, feature_block * 16));
    } else {
        dst = (COMPUTE_FLOAT8)0;
    }
#endif 

#if SLM_DIV_FACTOR > 1
    __local COMPUTE_FLOAT8 sum[WORK_GROUP_SIZE];
#endif


#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif 
            __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
            for (int kh = 0; kh < FILTER_HEIGHT; kh++) {
                if (input_y + kh * DILATION_HEIGHT < 0 || input_y + kh * DILATION_HEIGHT >= input_height)
                    continue;

                FLOAT line_cache[INPUT_LINE_SIZE];

                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        COMPUTE_FLOAT8 tmp = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                        line_cache[xb + 4] = tmp[4];
                        line_cache[xb + 5] = tmp[5];
                        line_cache[xb + 6] = tmp[6];
                        line_cache[xb + 7] = tmp[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        COMPUTE_FLOAT4 tmp = CONVERT_COMPUTE_FLOAT4(GROUP_READ4(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch));
                    
                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = (COMPUTE_FLOAT)GROUP_READ(input, input_offset +
                                                             icb * input_fs_pitch +
                                                             kh * DILATION_HEIGHT * input_y_pitch +
                                                             xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
                for (int kw = 0; kw < FILTER_WIDTH; kw++) {
                    FLOAT8 src;
                    __attribute__((opencl_unroll_hint(8)))
                    for (int i = 0; i < 8; i++) {
#if FILTER_WIDTH == 1 && DILATION_WIDTH == 1 && STRIDE_WIDTH == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_WIDTH + STRIDE_WIDTH * i];
#endif
                    }
                    COMPUTE_FLOAT8 weight0 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch));
                    COMPUTE_FLOAT8 weight1 = CONVERT_COMPUTE_FLOAT8(GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch +
                                                                    8 * filter_isv_pitch));
                    const COMPUTE_FLOAT8 src0  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 0));
                    const COMPUTE_FLOAT8 src1  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 1));
                    const COMPUTE_FLOAT8 src2  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 2));
                    const COMPUTE_FLOAT8 src3  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 3));
                    const COMPUTE_FLOAT8 src4  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 4));
                    const COMPUTE_FLOAT8 src5  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 5));
                    const COMPUTE_FLOAT8 src6  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 6));
                    const COMPUTE_FLOAT8 src7  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 7));
                    const COMPUTE_FLOAT8 src8  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 8));
                    const COMPUTE_FLOAT8 src9  = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 9));
                    const COMPUTE_FLOAT8 src10 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 10));
                    const COMPUTE_FLOAT8 src11 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 11));
                    const COMPUTE_FLOAT8 src12 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 12));
                    const COMPUTE_FLOAT8 src13 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 13));
                    const COMPUTE_FLOAT8 src14 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 14));
                    const COMPUTE_FLOAT8 src15 = CONVERT_COMPUTE_FLOAT8(GROUP_SHUFFLE8(src, 15));

                    dst = mad(weight0.s0, src0,  dst);
                    dst = mad(weight0.s1, src1,  dst);
                    dst = mad(weight0.s2, src2,  dst);
                    dst = mad(weight0.s3, src3,  dst);
                    dst = mad(weight0.s4, src4,  dst);
                    dst = mad(weight0.s5, src5,  dst);
                    dst = mad(weight0.s6, src6,  dst);
                    dst = mad(weight0.s7, src7,  dst);
                    dst = mad(weight1.s0, src8,  dst);
                    dst = mad(weight1.s1, src9,  dst);
                    dst = mad(weight1.s2, src10, dst);
                    dst = mad(weight1.s3, src11, dst);
                    dst = mad(weight1.s4, src12, dst);
                    dst = mad(weight1.s5, src13, dst);
                    dst = mad(weight1.s6, src14, dst);
                    dst = mad(weight1.s7, src15, dst);
                }
            }
        }
        
        
    if(x == 0){
        uint pad_offset = b * output_b_pitch + feature_block * output_fs_pitch + y * output_y_pitch;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
        pad_offset += (output_width + output_pad_left) * output_x_pitch;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
    }
#if SLM_DIV_FACTOR > 1
    sum[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        __attribute__((opencl_unroll_hint)) for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += sum[lid1 % feature_per_wg + i * feature_per_wg];
#endif

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif



    if ((feature_block+1)*16 >= output_channel) {
        for (int i = 0; i < 8; i++) {
            if ((feature_block*16 + sglid < output_channel) && (x + i) < output_width)
                output[output_offset + i * output_x_pitch + sglid] = (FLOAT)dst[i];
        }
    }
    else
    {
        if (x + 8 <= output_width || output_width % 8 == 0) {
            GROUP_WRITE8(output, output_offset, CONVERT_FLOAT8(dst));
        }else{
            for (int i = 0; i < output_width % 8; i++) {
                output[output_offset + i * output_x_pitch + sglid] = (FLOAT)dst[i];
            }
        }
    }
#if SLM_DIV_FACTOR > 1
    }
#endif
}
