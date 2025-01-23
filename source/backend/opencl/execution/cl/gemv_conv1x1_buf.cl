#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define GLOBAL_SIZE_DIM_2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK_2(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

#define UCHAR4_TO_CHAR8(b, scale, offset) \
    wei.s0 = (COMPUTE_FLOAT)((b.s0 >> 4) - 8); \
    wei.s1 = (COMPUTE_FLOAT)((b.s0 & 15) - 8); \
    wei.s2 = (COMPUTE_FLOAT)((b.s1 >> 4) - 8); \
    wei.s3 = (COMPUTE_FLOAT)((b.s1 & 15) - 8); \
    wei.s4 = (COMPUTE_FLOAT)((b.s2 >> 4) - 8); \
    wei.s5 = (COMPUTE_FLOAT)((b.s2 & 15) - 8); \
    wei.s6 = (COMPUTE_FLOAT)((b.s3 >> 4) - 8); \
    wei.s7 = (COMPUTE_FLOAT)((b.s3 & 15) - 8); \
    wei = wei * scale + offset;


#if WGS >= 8
__kernel void gemv_conv_c8_int4_buf(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        __global const uchar *weight,
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); //oc/8
    const int oc8 = oc << 3;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop = max((srcChannel + 4 - 1) / 4 - 1, 0);
#else
    const int loop = (srcChannel + 4 - 1) / 4;
#endif
    __local COMPUTE_FLOAT8 sum[WGS];
    COMPUTE_FLOAT8 out0 = 0;
#ifndef USE_IMAGE
    const int weight_offset = oc * srcChannelC4 * 16;
#endif
    
    for(int j = lid; j < loop; j+=WGS){
        int k4 = j << 2;
#ifdef ASYMMETRIC
        COMPUTE_FLOAT8 scale, offset;
        {
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = scaleOffset.s02468ace;
            offset = scaleOffset.s13579bdf;
        }
#else
        COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
        COMPUTE_FLOAT8 offset = 0;
#endif
        COMPUTE_FLOAT8 wei;
        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4));
        #ifdef USE_IMAGE
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(j, oc)));
        #else
        uchar16 charWeightsInt40 = vload16(j, weight + weight_offset);
        #endif
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.scdef, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
        }
    }
#if INPUT_CHANNEL_LEAVES_NUM != 0
    {
        int k4 = loop << 2;
#ifdef ASYMMETRIC
        COMPUTE_FLOAT8 scale, offset;
        {
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = scaleOffset.s02468ace;
            offset = scaleOffset.s13579bdf;
        }
#else
        COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
        COMPUTE_FLOAT8 offset = 0;
#endif
        COMPUTE_FLOAT8 wei;
        #ifdef USE_IMAGE
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(j, oc)));
        #else
        uchar16 charWeightsInt40 = vload16(j, weight + weight_offset);
        #endif
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4], wei, out0);
        }
        #if INPUT_CHANNEL_LEAVES_NUM >= 2
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4 + 1], wei, out0);
        }
        #endif
        #if INPUT_CHANNEL_LEAVES_NUM >= 3
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4 + 2], wei, out0);
        }
        #endif
    }
#endif
    sum[lid] = out0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = WGS/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        out0 = sum[0] + CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc8));
    #ifdef RELU
        out0 = fmax(out0, (COMPUTE_FLOAT8)0);
    #endif

    #ifdef RELU6
        out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    #endif
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    }
}

__kernel void gemv_conv_c8_int8_buf(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        __global const char *weight,
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); //oc/8
    const int oc8 = oc << 3;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop = max((srcChannel + 2 - 1) / 2 - 1, 0);
#else
    const int loop = (srcChannel + 2 - 1) / 2;
#endif
    __local COMPUTE_FLOAT8 sum[WGS];
#ifndef USE_IMAGE
    const int weight_offset = oc * srcChannelC4 * 32;
#endif
    COMPUTE_FLOAT8 out0 = 0;
    for(int j = lid; j < loop; j+=WGS){
        int k2 = j << 1;
        COMPUTE_FLOAT16 scale, offset;
        {
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + (k2 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset.s02468ace, scaleOffset.s02468ace);
            offset = (COMPUTE_FLOAT16)(scaleOffset.s13579bdf, scaleOffset.s13579bdf);
            #else
            COMPUTE_FLOAT8 scaleOffset = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k2 / blockDim) * dstChannelC4 * 4)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset, scaleOffset);
            offset = 0;
            #endif
        }
        COMPUTE_FLOAT2 in = CONVERT_COMPUTE_FLOAT2(vload2(0, input + k2));
        #ifdef USE_IMAGE
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(j, oc)))) * scale + offset;
        #else
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(j, weight + weight_offset)) * scale + offset;
        #endif
        {
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei.s01234567, out0);
        }
        {
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei.s89abcdef, out0);
        }
    }
#if INPUT_CHANNEL_LEAVES_NUM != 0
    {
        int k2 = loop << 1;
        COMPUTE_FLOAT16 scale, offset;
        {
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + (k2 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset.s02468ace, scaleOffset.s02468ace);
            offset = (COMPUTE_FLOAT16)(scaleOffset.s13579bdf, scaleOffset.s13579bdf);
            #else
            COMPUTE_FLOAT8 scaleOffset = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k2 / blockDim) * dstChannelC4 * 4)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset, scaleOffset);
            offset = 0;
            #endif
        }
        #ifdef USE_IMAGE
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(j, oc)))) * scale + offset;
        #else
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(j, weight + weight_offset)) * scale + offset;
        #endif
        {
            out0 = mad((COMPUTE_FLOAT8)input[k2], wei.s01234567, out0);
        }
    }
#endif
    sum[lid] = out0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = WGS/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        out0 = sum[0] + CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc8));
    #ifdef RELU
        out0 = fmax(out0, (COMPUTE_FLOAT8)0);
    #endif

    #ifdef RELU6
        out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    #endif
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    }
}
#else
__kernel void gemv_conv_c8_int4_buf(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        __global const uchar *weight,
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int ic = get_global_id(0);
    const int oc = get_global_id(1); //oc/8
    
    UNIFORM_BOUNDRY_CHECK_2(ic, oc);
    const int oc8 = oc << 3;
    
    const int loop = (blockDim + 4 - 1) / 4;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif
    COMPUTE_FLOAT8 out0 = CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc8));
#ifndef USE_IMAGE
    const int weight_offset = oc * srcChannelC4 * 16;
#endif
    for (int i = 0; i < blockNum; i++){
#ifdef ASYMMETRIC
        COMPUTE_FLOAT8 scale, offset;
        {
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + i * dstChannelC4 * 8)) / coef);
            scale = scaleOffset.s02468ace;
            offset = scaleOffset.s13579bdf;
        }
#else
        COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + i * dstChannelC4 * 4)) / coef);
        COMPUTE_FLOAT8 offset = 0;
#endif
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            COMPUTE_FLOAT8 wei;
            COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + (k << 2)));
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, oc)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
            }
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
            }
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
            }
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.scdef, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
            }
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            int k4 = k << 2;
            COMPUTE_FLOAT8 wei;
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, oc)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4], wei, out0);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4 + 1], wei, out0);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4 + 2], wei, out0);
            }
            #endif
        }
        #endif
}
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif
    vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
}

__kernel void gemv_conv_c8_int8_buf(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        __global const char *weight,
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int ic = get_global_id(0);
    const int oc = get_global_id(1); //oc/8
    UNIFORM_BOUNDRY_CHECK_2(ic, oc);
    const int oc8 = oc << 3;
    const int loop = (blockDim + 2 - 1) / 2;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif
#ifndef USE_IMAGE
    const int weight_offset = oc * srcChannelC4 * 32;
#endif
    COMPUTE_FLOAT8 out0 = CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc8));
    for (int i = 0; i < blockNum; i++){
        COMPUTE_FLOAT16 scale, offset;
        {
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + i * dstChannelC4 * 8)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset.s02468ace, scaleOffset.s02468ace);
            offset = (COMPUTE_FLOAT16)(scaleOffset.s13579bdf, scaleOffset.s13579bdf);
            #else
            COMPUTE_FLOAT8 scaleOffset = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + i * dstChannelC4 * 4)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset, scaleOffset);
            offset = 0;
            #endif
        }
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            COMPUTE_FLOAT2 in = CONVERT_COMPUTE_FLOAT2(vload2(0, input + (k << 1)));
            #ifdef USE_IMAGE
            COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k, oc)))) * scale + offset;
            #else
            COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(k, weight + weight_offset)) * scale + offset;
            #endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei.s01234567, out0);
            }
            {
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei.s89abcdef, out0);
            }
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            #ifdef USE_IMAGE
            COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k, oc)))) * scale + offset;
            #else
            COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(k, weight + weight_offset)) * scale + offset;
            #endif
            {
                out0 = mad((COMPUTE_FLOAT8)input[k << 1], wei.s01234567, out0);
            }
        }
        #endif
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif
    vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
}
#endif
