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

#define UCHAR8_TO_CHAR16(b, scale, offset)\
    wei.s0 = (COMPUTE_FLOAT)((b.s0 >> 4) - 8); \
    wei.s1 = (COMPUTE_FLOAT)((b.s0 & 15) - 8); \
    wei.s2 = (COMPUTE_FLOAT)((b.s1 >> 4) - 8); \
    wei.s3 = (COMPUTE_FLOAT)((b.s1 & 15) - 8); \
    wei.s4 = (COMPUTE_FLOAT)((b.s2 >> 4) - 8); \
    wei.s5 = (COMPUTE_FLOAT)((b.s2 & 15) - 8); \
    wei.s6 = (COMPUTE_FLOAT)((b.s3 >> 4) - 8); \
    wei.s7 = (COMPUTE_FLOAT)((b.s3 & 15) - 8); \
    wei.s8 = (COMPUTE_FLOAT)((b.s4 >> 4) - 8); \
    wei.s9 = (COMPUTE_FLOAT)((b.s4 & 15) - 8); \
    wei.sa = (COMPUTE_FLOAT)((b.s5 >> 4) - 8);\
    wei.sb = (COMPUTE_FLOAT)((b.s5 & 15) - 8);\
    wei.sc = (COMPUTE_FLOAT)((b.s6 >> 4) - 8);\
    wei.sd = (COMPUTE_FLOAT)((b.s6 & 15) - 8);\
    wei.se = (COMPUTE_FLOAT)((b.s7 >> 4) - 8);\
    wei.sf = (COMPUTE_FLOAT)((b.s7 & 15) - 8);\
    wei = wei * scale + offset; 
   // wei.s11 = (COMPUTE_FLOAT)((b.s5 & 15) - 8); 
    
   //wei.s13 = (COMPUTE_FLOAT)((b.s6 & 15) - 8); \
   //wei.s14 = (COMPUTE_FLOAT)((b.s7 >> 4) - 8); \
   //wei.s15 = (COMPUTE_FLOAT)((b.s7 & 15) - 8); \
   

#define UCHAR4_TO_CHAR8_FIRST(b, scale, offset, wei_result) \
    wei_result.s0 = (COMPUTE_FLOAT)((b.s0 >> 4) - 8); \
    wei_result.s1 = (COMPUTE_FLOAT)((b.s0 & 15) - 8); \
    wei_result.s2 = (COMPUTE_FLOAT)((b.s1 >> 4) - 8); \
    wei_result.s3 = (COMPUTE_FLOAT)((b.s1 & 15) - 8); \
    wei_result.s4 = (COMPUTE_FLOAT)((b.s2 >> 4) - 8); \
    wei_result.s5 = (COMPUTE_FLOAT)((b.s2 & 15) - 8); \
    wei_result.s6 = (COMPUTE_FLOAT)((b.s3 >> 4) - 8); \
    wei_result.s7 = (COMPUTE_FLOAT)((b.s3 & 15) - 8); \
    wei_result = wei_result * scale + offset;

#define UCHAR4_TO_CHAR8_SECOND(b, scale, offset, wei_result) \
    wei_result.s0 = (COMPUTE_FLOAT)((b.s4 >> 4) - 8); \
    wei_result.s1 = (COMPUTE_FLOAT)((b.s4 & 15) - 8); \
    wei_result.s2 = (COMPUTE_FLOAT)((b.s5 >> 4) - 8); \
    wei_result.s3 = (COMPUTE_FLOAT)((b.s5 & 15) - 8); \
    wei_result.s4 = (COMPUTE_FLOAT)((b.s6 >> 4) - 8); \
    wei_result.s5 = (COMPUTE_FLOAT)((b.s6 & 15) - 8); \
    wei_result.s6 = (COMPUTE_FLOAT)((b.s7 >> 4) - 8); \
    wei_result.s7 = (COMPUTE_FLOAT)((b.s7 & 15) - 8); \
    wei_result = wei_result * scale + offset;


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
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(loop, oc)));
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
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
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
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(loop, oc)))) * scale + offset;
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
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
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
    #ifdef OUTPUT_CHANNEL_LEAVES
    vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
    if(oc8 + 4 < dstChannelC4 * 4)
        vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
    vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
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

    #ifdef OUTPUT_CHANNEL_LEAVES
    vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
    if(oc8 + 4 < dstChannelC4 * 4)
        vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
    vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
}
#endif


__kernel void gemv_conv_c8_int8_buf_sparse_simple(GLOBAL_SIZE_DIM_2
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
                        __private const float coef,
                        __private const float threshold) {
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
        COMPUTE_FLOAT8 scale, offset;
        {
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + (k2 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = (COMPUTE_FLOAT8)scaleOffset.s02468ace;
            offset = (COMPUTE_FLOAT8)scaleOffset.s13579bdf;
            #else
            COMPUTE_FLOAT8 scaleOffset = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k2 / blockDim) * dstChannelC4 * 4)) / coef);
            scale = (COMPUTE_FLOAT8)scaleOffset;
            offset = 0;
            #endif
        }
        COMPUTE_FLOAT2 in = CONVERT_COMPUTE_FLOAT2(vload2(0, input + k2));
        
        if (fabs(in.s0) >= threshold){
            COMPUTE_FLOAT8 wei = CONVERT_COMPUTE_FLOAT8(vload8(0, weight + weight_offset + k2 * 8)) * scale + offset;
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei.s01234567, out0);
        }   
        if (fabs(in.s1) >= threshold){
            COMPUTE_FLOAT8 wei = CONVERT_COMPUTE_FLOAT8(vload8(0, weight + weight_offset + (k2 + 1) * 8)) * scale + offset;
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei.s01234567, out0);
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
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(loop, oc)))) * scale + offset;
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
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
    }
}


__kernel void gemv_conv_c8_int8_buf_sparse(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
//#ifdef USE_IMAGE
//                        __read_only image2d_t weight,
//#else
                        __global const char *weight,
//#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef,
                        __private const float threshold) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); //oc/8
    const int oc8 = oc << 3;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop = max((srcChannel + 4 - 1) / 4 - 1, 0);
#else
    const int loop = (srcChannel + 4 - 1) / 4;
#endif
    __local COMPUTE_FLOAT8 sum[WGS];
//#ifndef USE_IMAGE
    const int weight_offset = oc * srcChannelC4 * 32;
//#endif
    COMPUTE_FLOAT8 out0 = 0;

    for(int j = lid; j < loop; j+=WGS){
        int k2 = j << 2;
        COMPUTE_FLOAT8 scale, offset;
        {
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + (k2 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = (COMPUTE_FLOAT8)scaleOffset.s02468ace;
            offset = (COMPUTE_FLOAT8)scaleOffset.s13579bdf;
            #else
            COMPUTE_FLOAT8 scaleOffset = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k2 / blockDim) * dstChannelC4 * 4)) / coef);
            scale = (COMPUTE_FLOAT8)scaleOffset;
            offset = 0;
            #endif
        }
        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k2));

        if (fabs(in.x) >= threshold){
            //#ifdef USE_IMAGE
            //COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 * 8, oc)))) * scale + offset;
            //#else
            COMPUTE_FLOAT8 wei = CONVERT_COMPUTE_FLOAT8(vload8(0, weight + weight_offset + k2 * 8)) * scale + offset;
            //#endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.x, wei.s01234567, out0);
            }
        }

        if (fabs(in.y) >= threshold){
            //#ifdef USE_IMAGE
            //COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)((k2 + 1) * 8, oc)))) * scale + offset;
            //#else
            COMPUTE_FLOAT8 wei = CONVERT_COMPUTE_FLOAT8(vload8(0, weight + weight_offset + (k2 + 1) * 8)) * scale + offset;
            //#endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.y, wei.s01234567, out0);
            }
        }

        if (fabs(in.z) >= threshold){
            //#ifdef USE_IMAGE
            //COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)((k2 + 2) * 8, oc)))) * scale + offset;
            //#else
            COMPUTE_FLOAT8 wei = CONVERT_COMPUTE_FLOAT8(vload8(0, weight + weight_offset + (k2 + 2) * 8)) * scale + offset;
            //#endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.z, wei.s01234567, out0);
            }
        }

        if (fabs(in.w) >= threshold){
            //#ifdef USE_IMAGE
            //COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT8(as_char16(read_imagei(weight, SAMPLER, (int2)((k2 + 3) * 8, oc)))) * scale + offset;
            //#else
            COMPUTE_FLOAT8 wei = CONVERT_COMPUTE_FLOAT8(vload8(0, weight + weight_offset + (k2 + 3) * 8)) * scale + offset;
            //#endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.w, wei.s01234567, out0);
            }
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
        //#ifdef USE_IMAGE
        //COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(loop, oc)))) * scale + offset;
        //#else
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(loop, weight + weight_offset)) * scale + offset;
        //#endif
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
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
    }
}

__kernel void gemv_conv_c8_int8_buf_sparse_group(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
//#ifdef USE_IMAGE
//                        __read_only image2d_t weight,
//#else
                        __global const char *weight,
//#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef,
                        __private const float threshold) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); //oc/8
    const int oc8 = oc << 3;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop = max((srcChannel + 2 - 1) / 2 - 1, 0);
#else
    const int loop = (srcChannel + 2 - 1) / 2;
#endif
    __local COMPUTE_FLOAT8 sum[WGS];
//#ifndef USE_IMAGE
    const int weight_offset = oc * srcChannelC4 * 32;
//#endif
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

        if (fabs(in.s0) + fabs(in.s1) >= 2 * threshold) {

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
        //#ifdef USE_IMAGE
        //COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(loop, oc)))) * scale + offset;
        //#else
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(loop, weight + weight_offset)) * scale + offset;
        //#endif
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
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
    }
}

__kernel void gemv_conv_c8_int4_buf_sparse(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
                        __global const uchar *weight,
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef,
                        __private const float threshold) {
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
    const int weight_offset = oc * srcChannelC4 * 16;
    
    for(int j = lid; j < loop / 2; j+= WGS){
        //int k4 = j << 2;
        int k8 = j << 3;
        //int k8 = j << 3;
        //COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4));
#ifdef ASYMMETRIC
        COMPUTE_FLOAT8 scale, offset;
        {
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + (k8 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = scaleOffset.s02468ace;
            offset = scaleOffset.s13579bdf;
        }
#else
        COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k8 / blockDim) * dstChannelC4 * 4)) / coef);
        COMPUTE_FLOAT8 offset = 0;
#endif
        
        COMPUTE_FLOAT8 wei;
        COMPUTE_FLOAT8 in = CONVERT_COMPUTE_FLOAT8(vload8(0, input + k8));
        uchar8 charWeightsInt;

        if (fabs(in.s0) >= threshold){
            uchar4 charWeightsInt = vload4(j * 8, weight + weight_offset);
            UCHAR4_TO_CHAR8(charWeightsInt, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
        }   
        if (fabs(in.s1) >= threshold){
            uchar4 charWeightsInt = vload4(j * 8 + 1, weight + weight_offset);
            UCHAR4_TO_CHAR8(charWeightsInt, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
        }
        if (fabs(in.s2) >= threshold){
            uchar4 charWeightsInt = vload4(j * 8 + 2, weight + weight_offset);
            UCHAR4_TO_CHAR8(charWeightsInt, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
        }
        if (fabs(in.s3) >= threshold){
            uchar4 charWeightsInt = vload4(j * 8 + 3, weight + weight_offset);
            UCHAR4_TO_CHAR8(charWeightsInt, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
        }
        if (fabs(in.s4) >= threshold){
            uchar4 charWeightsInt = vload4(j * 8 + 4, weight + weight_offset);
            UCHAR4_TO_CHAR8(charWeightsInt, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s4, wei, out0);
        }
        if (fabs(in.s5) >= threshold){
            uchar4 charWeightsInt = vload4(j * 8 + 5, weight + weight_offset);
            UCHAR4_TO_CHAR8(charWeightsInt, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s5, wei, out0);
        }
        if (fabs(in.s6) >= threshold){
            uchar4 charWeightsInt = vload4(j * 8 + 6, weight + weight_offset);
            UCHAR4_TO_CHAR8(charWeightsInt, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s6, wei, out0);
        }
        if (fabs(in.s7) >= threshold){
            uchar4 charWeightsInt = vload4(j * 8 + 7, weight + weight_offset);
            UCHAR4_TO_CHAR8(charWeightsInt, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s7, wei, out0);
        }
        //if (fabs(in.s0) >= threshold || fabs(in.s1) >= threshold){
        ////if (fabs(in.s0) + fabs(in.s1) >= 2 * threshold) {
        //    charWeightsInt = vload8(j * 2, weight + weight_offset);
        //    {
        //        UCHAR4_TO_CHAR8(charWeightsInt.s0123, scale, offset);
        //        out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
        //    }
        //    {
        //        UCHAR4_TO_CHAR8(charWeightsInt.s4567, scale, offset);
        //        out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
        //    }
        //    
        //}
        //if (fabs(in.s2) >= threshold || fabs(in.s3) >= threshold){
        ////if (fabs(in.s2) + fabs(in.s3) >= 2 * threshold) {
        //    charWeightsInt = vload8(j * 2 + 1, weight + weight_offset);
        //    {
        //        UCHAR4_TO_CHAR8(charWeightsInt.s0123, scale, offset);
        //        out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
        //    }
        //    {
        //        UCHAR4_TO_CHAR8(charWeightsInt.s4567, scale, offset);
        //        out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
        //    }
        //    
        //}
    
    
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
        uchar16 charWeightsInt40 = vload16(j, weight + weight_offset);
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
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
    }
}



__kernel void gemv_conv_c8_int4_buf_sparse_eager(GLOBAL_SIZE_DIM_2
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
                        __private const float coef,
                        __private const float threshold) {
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
        

        if (fabs(in.s0) >= threshold || fabs(in.s1) >= threshold || 
    fabs(in.s2) >= threshold || fabs(in.s3) >= threshold)
        {
            
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
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(loop, oc)));
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
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
        vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
    }
}

__kernel void gemv_conv_c8_int4_buf_sparse_for_raster(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
                        __global const uchar *weight,
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef,
                        __private const float threshold) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); //oc/16
    const int oc16 = oc << 4;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop = max((srcChannel + 4 - 1) / 4 - 1, 0);
#else
    const int loop = (srcChannel + 4 - 1) / 4;
#endif
    __local COMPUTE_FLOAT16 sum[WGS];
    const int weight_offset = oc * srcChannelC4 * 32;
    COMPUTE_FLOAT16 out0 = 0;
    //COMPUTE_FLOAT8 out0_1, out0_2;
    for(int j = lid; j < loop; j+=WGS){
        int k4 = j << 2;
        COMPUTE_FLOAT16 scale, offset;
        //COMPUTE_FLOAT8 scale1, scale2, offset1, offset2;
#ifdef ASYMMETRIC
        {
            COMPUTE_FLOAT16 scaleOffset1 = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc16 * 2 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
            COMPUTE_FLOAT16 scaleOffset2 = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc16 * 2 + 16 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset1.s02468ace, scaleOffset2.s02468ace);
            offset = (COMPUTE_FLOAT16)(scaleOffset1.s13579bdf, scaleOffset2.s13579bdf);
            //scale1 = scaleOffset1.s02468ace; 
            //scale2 = scaleOffset2.s02468ace; 
            //offset1 = scaleOffset1.s13579bdf;
            //offset2 = scaleOffset2.s13579bdf;
        }
#else
        scale = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc16 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
        offset = (COMPUTE_FLOAT16)0;
        //scale1 = scaleOffset.s01234567;
        //scale2 = scaleOffset.s89abcdef;
        //offset1 = (COMPUTE_FLOAT8)(0);
        //offset2 = (COMPUTE_FLOAT8)(0);
#endif
        
        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4));
        if (fabs(in.s0) >= threshold){
            uchar8 b = vload8(j * 4 + 0, weight + weight_offset);
            {
                COMPUTE_FLOAT16 wei;
                UCHAR8_TO_CHAR16(b, scale, offset);
                out0 = mad((COMPUTE_FLOAT16)in.s0, wei, out0);
                //UCHAR4_TO_CHAR8_FIRST(b, scale1, offset1, wei_local1);
                //out0_1 = mad((COMPUTE_FLOAT8)in.s0, wei_local1, out0_1);
            }
            
            //{
            //    COMPUTE_FLOAT8 wei_local2;
            //    UCHAR4_TO_CHAR8_SECOND(b, scale2, offset2, wei_local2);
            //    out0_2 = mad((COMPUTE_FLOAT8)in.s0, wei_local2, out0_2);
            //}
            
            //UCHAR8_TO_CHAR16(b, scale, offset);
            //UCHAR8_TO_CHAR16(b, scale, offset);
        }

        if (fabs(in.s1) >= threshold){
            uchar8 b = vload8(j * 4 + 1, weight + weight_offset);
            //{
            //    COMPUTE_FLOAT16 wei_local1;
            //    UCHAR4_TO_CHAR8_FIRST(b, scale1, offset1, wei_local1);
            //    out0_1 = mad((COMPUTE_FLOAT8)in.s1, wei_local1, out0_1);
            //}
            //
            //{
            //    COMPUTE_FLOAT8 wei_local2;
            //    UCHAR4_TO_CHAR8_SECOND(b, scale2, offset2, wei_local2);
            //    out0_2 = mad((COMPUTE_FLOAT8)in.s1, wei_local2, out0_2);
            //}
            {
                COMPUTE_FLOAT16 wei;
                UCHAR8_TO_CHAR16(b, scale, offset);
                out0 = mad((COMPUTE_FLOAT16)in.s1, wei, out0);
            }
        }

        if (fabs(in.s2) >= threshold){
            uchar8 b = vload8(j * 4 + 2, weight + weight_offset);
            //{
            //    COMPUTE_FLOAT8 wei_local1;
            //    UCHAR4_TO_CHAR8_FIRST(b, scale1, offset1, wei_local1);
            //    out0_1 = mad((COMPUTE_FLOAT8)in.s2, wei_local1, out0_1);
            //}
            //
            //{
            //    COMPUTE_FLOAT8 wei_local2;
            //    UCHAR4_TO_CHAR8_SECOND(b, scale2, offset2, wei_local2);
            //    out0_2 = mad((COMPUTE_FLOAT8)in.s2, wei_local2, out0_2);
            //}
            {
                COMPUTE_FLOAT16 wei;
                UCHAR8_TO_CHAR16(b, scale, offset);
                out0 = mad((COMPUTE_FLOAT16)in.s2, wei, out0);
            }
        }

        if (fabs(in.s3) >= threshold){
            uchar8 b = vload8(j * 4 + 3, weight + weight_offset);
            //{
            //    COMPUTE_FLOAT8 wei_local1;
            //    UCHAR4_TO_CHAR8_FIRST(b, scale1, offset1, wei_local1);
            //    out0_1 = mad((COMPUTE_FLOAT8)in.s3, wei_local1, out0_1);
            //}
            //
            //{
            //    COMPUTE_FLOAT8 wei_local2;
            //    UCHAR4_TO_CHAR8_SECOND(b, scale2, offset2, wei_local2);
            //    out0_2 = mad((COMPUTE_FLOAT8)in.s3, wei_local2, out0_2);
            //}
            {
                COMPUTE_FLOAT16 wei;
                UCHAR8_TO_CHAR16(b, scale, offset);
                out0 = mad((COMPUTE_FLOAT16)in.s3, wei, out0);
            }
        }

    }
#if INPUT_CHANNEL_LEAVES_NUM != 0
    {
        int k4 = loop << 2;
#ifdef ASYMMETRIC
        COMPUTE_FLOAT16 scale, offset;
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
        uchar16 charWeightsInt40 = vload16(j, weight + weight_offset);
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
    //out0 = (COMPUTE_FLOAT16)(out0_1.s0, out0_1.s1, out0_1.s2, out0_1.s3, out0_1.s4, out0_1.s5, out0_1.s6, out0_1.s7, out0_2.s0, out0_2.s1, out0_2.s2, out0_2.s3, out0_2.s4, out0_2.s5, out0_2.s6, out0_2.s7);
    //out0.s1 = out0_1.s1;
    //out0 = (COMPUTE_FLOAT16)(out0_1.s0, out0_1.s1, out0_1.s2, out0_1.s3, out0_1.s4, out0_1.s5, out0_1.s6, out0_1.s7, out0_2.s0, out0_2.s1, out0_2.s2, out0_2.s3, out0_2.s4, out0_2.s5, out0_2.s6, out0_2.s7);
    sum[lid] = out0;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = WGS/2; i > 0; i /= 2){
        if (lid < i){
            sum[lid] = sum[lid] + sum[lid + i];
        }
            
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(lid == 0){
        out0 = sum[0] + CONVERT_COMPUTE_FLOAT16(vload16(0, bias + oc16));
    #ifdef RELU
        out0 = fmax(out0, (COMPUTE_FLOAT8)0);
    #endif

    #ifdef RELU6
        out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    #endif
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
        vstore16(CONVERT_FLOAT16(out0), 0, output + oc16);
    #endif
    }
}

__kernel void gemv_conv_c8_int4_buf_sparse_for_raster_v2(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
                        __global const uchar *weight,
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef,
                        __private const float threshold) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); // oc / 8
    const int oc16 = oc << 4;

    const int loop = (srcChannel + 4 - 1) / 4;

    __local COMPUTE_FLOAT8 sum_low[WGS];
    __local COMPUTE_FLOAT8 sum_high[WGS];
    const int weight_offset = oc * srcChannelC4 * 32;

    COMPUTE_FLOAT8 out_low = 0;
    COMPUTE_FLOAT8 out_high = 0;

    for (int j = lid; j < loop; j += WGS){
        int k4 = j << 2;
        COMPUTE_FLOAT8 scale_low, scale_high, offset_low, offset_high;

        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4));

    #ifdef ASYMMETRIC
            {
                COMPUTE_FLOAT16 so1 = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc * 2 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
                COMPUTE_FLOAT16 so2 = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc * 2 + 16 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
                scale_low = so1.s02468ace;
                scale_high = so2.s02468ace;
                offset_low = so1.s13579bdf;
                offset_high = so2.s13579bdf;
            }
    #else
            {
                COMPUTE_FLOAT16 so = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc * 2 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
                scale_low = so.s01234567;
                scale_high = so.s89abcdef;
                offset_low = 0;
                offset_high = 0;
            }
    #endif
        if (fabs(in.s0) >= threshold){
            uchar8 b = vload8(j * 4 + 0, weight + weight_offset);
            {
                COMPUTE_FLOAT8 wei_low, wei_high;
                UCHAR4_TO_CHAR8_FIRST(b, scale_low, offset_low, wei_low);
                UCHAR4_TO_CHAR8_SECOND(b, scale_high, offset_high, wei_high);
                out_low = mad((COMPUTE_FLOAT8)in.s0, wei_low, out_low);
                out_high = mad((COMPUTE_FLOAT8)in.s0, wei_high, out_high);
            }
        }
        if (fabs(in.s1) >= threshold){
            uchar8 b = vload8(j * 4 + 1, weight + weight_offset);
            {
                COMPUTE_FLOAT8 wei_low, wei_high;
                UCHAR4_TO_CHAR8_FIRST(b, scale_low, offset_low, wei_low);
                UCHAR4_TO_CHAR8_SECOND(b, scale_high, offset_high, wei_high);
                out_low = mad((COMPUTE_FLOAT8)in.s1, wei_low, out_low);
                out_high = mad((COMPUTE_FLOAT8)in.s1, wei_high, out_high);
            }
        }
        if (fabs(in.s2) >= threshold){
            uchar8 b = vload8(j * 4 + 2, weight + weight_offset);
            {
                COMPUTE_FLOAT8 wei_low, wei_high;
                UCHAR4_TO_CHAR8_FIRST(b, scale_low, offset_low, wei_low);
                UCHAR4_TO_CHAR8_SECOND(b, scale_high, offset_high, wei_high);
                out_low = mad((COMPUTE_FLOAT8)in.s2, wei_low, out_low);
                out_high = mad((COMPUTE_FLOAT8)in.s2, wei_high, out_high);
            }
        }
        if (fabs(in.s3) >= threshold){
            uchar8 b = vload8(j * 4 + 3, weight + weight_offset);
            {
                COMPUTE_FLOAT8 wei_low, wei_high;
                UCHAR4_TO_CHAR8_FIRST(b, scale_low, offset_low, wei_low);
                UCHAR4_TO_CHAR8_SECOND(b, scale_high, offset_high, wei_high);
                out_low = mad((COMPUTE_FLOAT8)in.s3, wei_low, out_low);
                out_high = mad((COMPUTE_FLOAT8)in.s3, wei_high, out_high);
            }
        }
    }

    sum_low[lid] = out_low;
    sum_high[lid] = out_high;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int half_wgs = WGS / 2;
    
    if (lid < half_wgs) {
        // 前一半线程处理sum_low的reduction
        for(int i = half_wgs/2; i > 0; i /= 2){
            if (lid < i) {
                sum_low[lid] += sum_low[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    } else {
        // 后一半线程处理sum_high的reduction
        int local_lid = lid - half_wgs;
        for(int i = half_wgs/2; i > 0; i /= 2){
            if (local_lid < i) {
                sum_high[half_wgs + local_lid] += sum_high[half_wgs + local_lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    if(lid == 0){
        COMPUTE_FLOAT8 result_low = sum_low[0] + CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc16));
        COMPUTE_FLOAT8 result_high = sum_high[half_wgs] + CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc16 + 8));
        
    #ifdef RELU
        result_low = fmax(result_low, (COMPUTE_FLOAT8)0);
        result_high = fmax(result_high, (COMPUTE_FLOAT8)0);
    #endif
    #ifdef RELU6
        result_low = clamp(result_low, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
        result_high = clamp(result_high, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    #endif
        
        vstore8(CONVERT_FLOAT8(result_low), 0, output + oc16);
        vstore8(CONVERT_FLOAT8(result_high), 0, output + oc16 + 8);
    }
    

}

__kernel void gemv_conv_c8_int4_buf_sparse_for_raster_thread_split(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
                        __global const uchar *weight,
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef,
                        __private const float threshold) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1);
    const int oc16 = oc << 4;
    const int loop = (srcChannel + 4 - 1) / 4;
    
    __local COMPUTE_FLOAT8 sum_low[WGS/2];
    __local COMPUTE_FLOAT8 sum_high[WGS/2];
    const int weight_offset = oc * srcChannelC4 * 32;
    
    const int half_wgs = WGS / 2;
    const bool is_low_thread = (lid < half_wgs);
    const int local_lid = is_low_thread ? lid : (lid - half_wgs);
    
    COMPUTE_FLOAT8 out_result = 0;
    
    // 每个线程只处理自己负责的部分
    for(int j = local_lid; j < loop; j += half_wgs){
        int k4 = j << 2;
        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4));
        
        COMPUTE_FLOAT activity = fabs(in.s0) + fabs(in.s1) + fabs(in.s2) + fabs(in.s3);
        if (activity < threshold) continue;
        
        // 根据线程类型计算不同的scale和offset
        COMPUTE_FLOAT8 scale, offset;
        if (is_low_thread) {
            // 处理low部分 (前8个输出通道)
#ifdef ASYMMETRIC
            COMPUTE_FLOAT16 so1 = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc16 * 2 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = so1.s02468ace;
            offset = so1.s13579bdf;
#else
            COMPUTE_FLOAT16 s = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc16 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
            scale = s.s01234567;
            offset = (COMPUTE_FLOAT8)0;
#endif
        } else {
            // 处理high部分 (后8个输出通道)
#ifdef ASYMMETRIC
            COMPUTE_FLOAT16 so2 = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc16 * 2 + 16 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = so2.s02468ace;
            offset = so2.s13579bdf;
#else
            COMPUTE_FLOAT16 s = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc16 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
            scale = s.s89abcdef;
            offset = (COMPUTE_FLOAT8)0;
#endif
        }

        // 处理4个输入通道，每个线程只计算自己负责的权重部分
        COMPUTE_FLOAT inputs[4] = {in.s0, in.s1, in.s2, in.s3};
        
        for(int ch = 0; ch < 4; ch++) {
            if (fabs(inputs[ch]) >= threshold) {
                uchar8 w = vload8(j * 4 + ch, weight + weight_offset);
                COMPUTE_FLOAT8 wei;
                
                if (is_low_thread) {
                    // 只处理权重的前半部分
                    UCHAR4_TO_CHAR8_FIRST(w, scale, offset, wei);
                } else {
                    // 只处理权重的后半部分
                    UCHAR4_TO_CHAR8_SECOND(w, scale, offset, wei);
                }
                
                out_result = mad((COMPUTE_FLOAT8)inputs[ch], wei, out_result);
            }
        }
    }
    
    // 分别存储到不同的local memory区域
    if (is_low_thread) {
        sum_low[local_lid] = out_result;
    } else {
        sum_high[local_lid] = out_result;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 分别进行reduction，真正并行执行
    if (is_low_thread) {
        // low线程只处理sum_low
        for(int i = half_wgs/2; i > 0; i /= 2){
            if (local_lid < i) {
                sum_low[local_lid] += sum_low[local_lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    } else {
        // high线程只处理sum_high
        for(int i = half_wgs/2; i > 0; i /= 2){
            if (local_lid < i) {
                sum_high[local_lid] += sum_high[local_lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    // 只有线程0负责最终输出
    if(lid == 0){
        COMPUTE_FLOAT8 result_low = sum_low[0] + CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc16));
        COMPUTE_FLOAT8 result_high = sum_high[0] + CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc16 + 8));
        
    #ifdef RELU
        result_low = fmax(result_low, (COMPUTE_FLOAT8)0);
        result_high = fmax(result_high, (COMPUTE_FLOAT8)0);
    #endif
    #ifdef RELU6
        result_low = clamp(result_low, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
        result_high = clamp(result_high, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    #endif
        
        vstore8(CONVERT_FLOAT8(result_low), 0, output + oc16);
        vstore8(CONVERT_FLOAT8(result_high), 0, output + oc16 + 8);
    }
}

__kernel void gemv_conv_c8_int4_buf_sparse_for_raster_simple(GLOBAL_SIZE_DIM_2
                        __global const FLOAT* input,
                        __global const uchar *weight,
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef,
                        __private const float threshold) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); //oc/8
    const int oc8 = oc << 3;
    const int loop = (srcChannel + 4 - 1) / 4;
    
    __local COMPUTE_FLOAT8 sum[WGS];
    COMPUTE_FLOAT8 out0 = 0;
    const int weight_offset = oc * srcChannelC4 * 16;
    
    // 简化的主循环，专注于高效的内存访问
    for(int j = lid; j < loop; j += WGS){
        int k4 = j << 2;
        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4));
        
        // 快速活跃度检查
        if (fabs(in.s0) < threshold) 
            continue;
        
        // 计算scale和offset
        COMPUTE_FLOAT8 scale, offset;
#ifdef ASYMMETRIC
        {
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + (k4 / blockDim) * dstChannelC4 * 8)) / coef);
            scale = scaleOffset.s02468ace;
            offset = scaleOffset.s13579bdf;
        }
#else
        scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + (k4 / blockDim) * dstChannelC4 * 4)) / coef);
        offset = (COMPUTE_FLOAT8)0;
#endif

        uchar16 weights = vload16(j, weight + weight_offset);
        COMPUTE_FLOAT8 wei;

        if (fabs(in.s0) >= threshold) {
            UCHAR4_TO_CHAR8(weights.s0123, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
        }
        if (fabs(in.s1) >= threshold) {
            UCHAR4_TO_CHAR8(weights.s4567, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
        }
        if (fabs(in.s2) >= threshold) {
            UCHAR4_TO_CHAR8(weights.s89ab, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
        }
        if (fabs(in.s3) >= threshold) {
            UCHAR4_TO_CHAR8(weights.scdef, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
        }
    }
    
    // 标准reduction
    sum[lid] = out0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = WGS/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] += sum[lid + i];
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

__kernel void gemv_conv_c8_int4_buf_sparse_wo_local(GLOBAL_SIZE_DIM_2
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
                        __private const float coef,
                        __private const float threshold) {
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
            if (fabs(in.s0) < threshold) continue;
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
    #ifdef OUTPUT_CHANNEL_LEAVES
    vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + oc8);
    if(oc8 + 4 < dstChannelC4 * 4)
        vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + oc8 + 4);
    #else
    vstore8(CONVERT_FLOAT8(out0), 0, output + oc8);
    #endif
}