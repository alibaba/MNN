#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

#define UCHAR16_TO_2CHAR16(a, b, c) \
    a.s0 = (c.s0 >> 4) - 8; a.s1 = (c.s0 & 15) - 8; a.s2 = (c.s1 >> 4) - 8; a.s3 = (c.s1 & 15) - 8; a.s4 = (c.s2 >> 4) - 8; a.s5 = (c.s2 & 15) - 8; a.s6 = (c.s3 >> 4) - 8; a.s7 = (c.s3 & 15) - 8;         \
    a.s8 = (c.s4 >> 4) - 8; a.s9 = (c.s4 & 15) - 8; a.sa = (c.s5 >> 4) - 8; a.sb = (c.s5 & 15) - 8; a.sc = (c.s6 >> 4) - 8; a.sd = (c.s6 & 15) - 8; a.se = (c.s7 >> 4) - 8; a.sf = (c.s7 & 15) - 8;         \
    b.s0 = (c.s8 >> 4) - 8; b.s1 = (c.s8 & 15) - 8; b.s2 = (c.s9 >> 4) - 8; b.s3 = (c.s9 & 15) - 8; b.s4 = (c.sa >> 4) - 8; b.s5 = (c.sa & 15) - 8; b.s6 = (c.sb >> 4) - 8; b.s7 = (c.sb & 15) - 8;         \
    b.s8 = (c.sc >> 4) - 8; b.s9 = (c.sc & 15) - 8; b.sa = (c.sd >> 4) - 8; b.sb = (c.sd & 15) - 8; b.sc = (c.se >> 4) - 8; b.sd = (c.se & 15) - 8; b.se = (c.sf >> 4) - 8; b.sf = (c.sf & 15) - 8;

#define UCHAR8_TO_CHAR16(a, c) \
    a.s0 = (c.s0 >> 4) - 8; a.s1 = (c.s0 & 15) - 8; a.s2 = (c.s1 >> 4) - 8; a.s3 = (c.s1 & 15) - 8; a.s4 = (c.s2 >> 4) - 8; a.s5 = (c.s2 & 15) - 8; a.s6 = (c.s3 >> 4) - 8; a.s7 = (c.s3 & 15) - 8;         \
    a.s8 = (c.s4 >> 4) - 8; a.s9 = (c.s4 & 15) - 8; a.sa = (c.s5 >> 4) - 8; a.sb = (c.s5 & 15) - 8; a.sc = (c.s6 >> 4) - 8; a.sd = (c.s6 & 15) - 8; a.se = (c.s7 >> 4) - 8; a.sf = (c.s7 & 15) - 8;

#define DOT16X16(a, b, c) \
    c += dot(a.s0123, b.s0123); \
    c += dot(a.s4567, b.s4567); \
    c += dot(a.s89ab, b.s89ab); \
    c += dot(a.scdef, b.scdef);

#ifdef INPUT_CHANNEL_LEAVE
    #define PADZEROS(k, channel, data) {\
        COMPUTE_FLOAT* ptr = (COMPUTE_FLOAT*)&data; \
        int remain = k + 15 - channel; \
        for(int r = remain; r >= 0; r--){ \
            ptr[15 - r] = 0; \
        }  \
    }
#else
    #define PADZEROS(k, channel, data)
#endif

#if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
#define CHANNEL_PACK 32
#else
#define CHANNEL_PACK 16
#endif

#if (defined USE_LOW_BIT_WEIGHT_INT8)
#define WEIGHT_STRIDE 16
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
#define WEIGHT_STRIDE 8
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#ifdef USE_IMAGE
inline COMPUTE_FLOAT16 readWeight(__read_only image2d_t weight, int ix, int iy, COMPUTE_FLOAT scale, COMPUTE_FLOAT offset){
    return CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(ix, iy)))) * scale + offset;
}
#else

#if (defined USE_LOW_BIT_WEIGHT_INT8)
inline COMPUTE_FLOAT16 readWeight(__global const char *weight, int ix, int iy, COMPUTE_FLOAT scale, COMPUTE_FLOAT offset){
    return CONVERT_COMPUTE_FLOAT16(vload16(0, weight)) * scale + offset;
}
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
inline COMPUTE_FLOAT16 readWeight(__global const uchar *weight, int ix, int iy, COMPUTE_FLOAT scale, COMPUTE_FLOAT offset){
    uchar16 charWeightsInt40 = vload16(0, weight);
    uchar8 charWeightsInt4 = vload8(0, weight);
    char16 charWeights = 0;
    UCHAR8_TO_CHAR16(charWeights, charWeightsInt4);
    return CONVERT_COMPUTE_FLOAT16(charWeights) * scale + offset;
}
#endif
#endif


__kernel void gemv_conv_c4_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
#endif
#endif
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int bhw,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c/4
    const int y = get_global_id(1); //b h w

    UNIFORM_BOUNDRY_CHECK(x, y);

    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(x, bias));
    COMPUTE_FLOAT4 out0 = bias0;
    int idn = x << 2;
    int idm = y;
    
    int input_offset0 = idm * 4;

    int out_offset = (x * bhw + idm) * 4;
#ifndef USE_IMAGE
    int weight_offset = x * 4 * WEIGHT_STRIDE;
    int weight_oc_offset = dstChannelC4 * 4 * WEIGHT_STRIDE;
#endif

    const int loop = (blockDim + CHANNEL_PACK - 1) / CHANNEL_PACK;
#ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif
    
    for (int i = 0; i < blockNum; ++i){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(x, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; ++j) {
            int k = i * loop + j;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            int k32 = k << 5;
            COMPUTE_FLOAT16 weights00, weights01, weights10, weights11, weights20, weights21, weights30, weights31;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn, k)));
                uchar16 charWeightsInt41 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn + 1, k)));
                uchar16 charWeightsInt42 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn + 2, k)));
                uchar16 charWeightsInt43 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn + 3, k)));
                char16 charWeights0, charWeights1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                weights00 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights01 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
                weights10 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s2 + ScaleOffset.s3;
                weights11 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt42);
                weights20 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s4 + ScaleOffset.s5;
                weights21 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s4 + ScaleOffset.s5;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt43);
                weights30 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s6 + ScaleOffset.s7;
                weights31 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s6 + ScaleOffset.s7;
            }
            {
                COMPUTE_FLOAT16 in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + k32));
                COMPUTE_FLOAT16 in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + k32 + 16));
                DOT16X16(in0, weights00, out0.s0);DOT16X16(in1, weights01, out0.s0);
                DOT16X16(in0, weights10, out0.s1);DOT16X16(in1, weights11, out0.s1);
                DOT16X16(in0, weights20, out0.s2);DOT16X16(in1, weights21, out0.s2);
                DOT16X16(in0, weights30, out0.s3);DOT16X16(in1, weights31, out0.s3);
            }
            #else
            COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
            #ifdef USE_IMAGE
            weights0 = readWeight(weight, idn, k, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight, idn + 1, k, ScaleOffset.s2, ScaleOffset.s3);
            weights2 = readWeight(weight, idn + 2, k, ScaleOffset.s4, ScaleOffset.s5);
            weights3 = readWeight(weight, idn + 3, k, ScaleOffset.s6, ScaleOffset.s7);
            #else
            weights0 = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight + weight_offset + k * weight_oc_offset + WEIGHT_STRIDE, 0, 0, ScaleOffset.s2, ScaleOffset.s3);
            weights2 = readWeight(weight + weight_offset + k * weight_oc_offset + 2 * WEIGHT_STRIDE, 0, 0, ScaleOffset.s4, ScaleOffset.s5);
            weights3 = readWeight(weight + weight_offset + k * weight_oc_offset + 3 * WEIGHT_STRIDE, 0, 0, ScaleOffset.s6, ScaleOffset.s7);
            #endif
            {
                COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(k, input));
                DOT16X16(in, weights0, out0.s0);
                DOT16X16(in, weights1, out0.s1);
                DOT16X16(in, weights2, out0.s2);
                DOT16X16(in, weights3, out0.s3);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            int k8 = k << 3;
            COMPUTE_FLOAT16 weights00, weights01, weights10, weights11, weights20, weights21, weights30, weights31;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn, k)));
                uchar16 charWeightsInt41 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn + 1, k)));
                uchar16 charWeightsInt42 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn + 2, k)));
                uchar16 charWeightsInt43 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn + 3, k)));
                char16 charWeights0, charWeights1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                weights00 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights01 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
                weights10 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s2 + ScaleOffset.s3;
                weights11 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt42);
                weights20 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s4 + ScaleOffset.s5;
                weights21 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s4 + ScaleOffset.s5;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt43);
                weights30 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s6 + ScaleOffset.s7;
                weights31 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s6 + ScaleOffset.s7;
                
                PADZEROS(k, srcChannel, weights00);PADZEROS(k + 16, srcChannel, weights01);
                PADZEROS(k, srcChannel, weights10);PADZEROS(k + 16, srcChannel, weights11);
                PADZEROS(k, srcChannel, weights20);PADZEROS(k + 16, srcChannel, weights21);
                PADZEROS(k, srcChannel, weights30);PADZEROS(k + 16, srcChannel, weights31);
            }
            {
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k8 * 4));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + (k8 + 1) * 4) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + (k8 + 2) * 4) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + (k8 + 3) * 4) : (FLOAT4)0);
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + (k8 + 4) * 4) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + (k8 + 5) * 4) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + (k8 + 6) * 4) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + (k8 + 7) * 4) : (FLOAT4)0);
                DOT16X16(in0, weights00, out0.s0);DOT16X16(in1, weights01, out0.s0);
                DOT16X16(in0, weights10, out0.s1);DOT16X16(in1, weights11, out0.s1);
                DOT16X16(in0, weights20, out0.s2);DOT16X16(in1, weights21, out0.s2);
                DOT16X16(in0, weights30, out0.s3);DOT16X16(in1, weights31, out0.s3);
            }
            #else
            int k4 = k << 2;
            COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
            #ifdef USE_IMAGE
            weights0 = readWeight(weight, idn, k, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight, idn + 1, k, ScaleOffset.s2, ScaleOffset.s3);
            weights2 = readWeight(weight, idn + 2, k, ScaleOffset.s4, ScaleOffset.s5);
            weights3 = readWeight(weight, idn + 3, k, ScaleOffset.s6, ScaleOffset.s7);
            #else
            weights0 = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight + weight_offset + k * weight_oc_offset + WEIGHT_STRIDE, 0, 0, ScaleOffset.s2, ScaleOffset.s3);
            weights2 = readWeight(weight + weight_offset + k * weight_oc_offset + 2 * WEIGHT_STRIDE, 0, 0, ScaleOffset.s4, ScaleOffset.s5);
            weights3 = readWeight(weight + weight_offset + k * weight_oc_offset + 3 * WEIGHT_STRIDE, 0, 0, ScaleOffset.s6, ScaleOffset.s7);
            #endif
            PADZEROS(k, srcChannel, weights0);
            PADZEROS(k, srcChannel, weights1);
            PADZEROS(k, srcChannel, weights2);
            PADZEROS(k, srcChannel, weights3);
            {
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4 * 4));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + (k4 + 1) * 4) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + (k4 + 2) * 4) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + (k4 + 3) * 4) : (FLOAT4)0);
                DOT16X16(in, weights0, out0.s0);
                DOT16X16(in, weights1, out0.s1);
                DOT16X16(in, weights2, out0.s2);
                DOT16X16(in, weights3, out0.s3);
            }
            #endif
        }
        #endif
    }
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
}

__kernel void gemv_conv_c2_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
#endif
#endif
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int bhw,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c/2
    const int y = get_global_id(1); //b h w

    UNIFORM_BOUNDRY_CHECK(x, y);
    
    int idn = x << 1;
    int idm = y;
    COMPUTE_FLOAT2 bias0 = CONVERT_COMPUTE_FLOAT2(vload2(x, bias));
    COMPUTE_FLOAT2 out0 = bias0;
    int input_offset0 = idm * 4;
    int out_offset = ((x * 2) / 4 * bhw + idm) * 4 + ((x * 2) % 4);
#ifndef USE_IMAGE
    int weight_offset = x * 2 * WEIGHT_STRIDE;
    int weight_oc_offset = dstChannelC4 * 4 * WEIGHT_STRIDE;
#endif

    const int loop = (blockDim + CHANNEL_PACK - 1) / CHANNEL_PACK;
#ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif

    for (int i = 0; i < blockNum; ++i){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT4 ScaleOffset = CONVERT_COMPUTE_FLOAT4(vload4(x, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; ++j) {
            int k = i * loop + j;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            int k32 = k << 5;
            COMPUTE_FLOAT16 weights00, weights01, weights10, weights11;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn, k)));
                uchar16 charWeightsInt41 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn + 1, k)));
                char16 charWeights0, charWeights1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                weights00 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights01 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
                weights10 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s2 + ScaleOffset.s3;
                weights11 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
            }
            {
                COMPUTE_FLOAT16 in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + k32));
                COMPUTE_FLOAT16 in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + k32 + 16));
                DOT16X16(in0, weights00, out0.s0);DOT16X16(in1, weights01, out0.s0);
                DOT16X16(in0, weights10, out0.s1);DOT16X16(in1, weights11, out0.s1);
            }
            #else
            COMPUTE_FLOAT16 weights0, weights1;
            #ifdef USE_IMAGE
            weights0 = readWeight(weight, idn, k, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight, idn + 1, k, ScaleOffset.s2, ScaleOffset.s3);
            #else
            weights0 = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight + weight_offset + k * weight_oc_offset + WEIGHT_STRIDE, 0, 0, ScaleOffset.s2, ScaleOffset.s3);
            #endif
            {
                COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(k, input));
                DOT16X16(in, weights0, out0.s0);
                DOT16X16(in, weights1, out0.s1);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            int k8 = k << 3;
            COMPUTE_FLOAT16 weights00, weights01, weights10, weights11;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn, k)));
                uchar16 charWeightsInt41 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn + 1, k)));
                char16 charWeights0, charWeights1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                weights00 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights01 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
                weights10 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s2 + ScaleOffset.s3;
                weights11 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
                
                PADZEROS(k, srcChannel, weights00);PADZEROS(k + 16, srcChannel, weights01);
                PADZEROS(k, srcChannel, weights10);PADZEROS(k + 16, srcChannel, weights11);
            }
            {
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k8 * 4));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + (k8 + 1) * 4) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + (k8 + 2) * 4) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + (k8 + 3) * 4) : (FLOAT4)0);
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + (k8 + 4) * 4) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + (k8 + 5) * 4) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + (k8 + 6) * 4) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + (k8 + 7) * 4) : (FLOAT4)0);
                DOT16X16(in0, weights00, out0.s0);DOT16X16(in1, weights01, out0.s0);
                DOT16X16(in0, weights10, out0.s1);DOT16X16(in1, weights11, out0.s1);
            }
            #else
            int k4 = k << 2;
            COMPUTE_FLOAT16 weights0, weights1;
            #ifdef USE_IMAGE
            weights0 = readWeight(weight, idn, k, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight, idn + 1, k, ScaleOffset.s2, ScaleOffset.s3);
            #else
            weights0 = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight + weight_offset + k * weight_oc_offset + WEIGHT_STRIDE, 0, 0, ScaleOffset.s2, ScaleOffset.s3);
            #endif
            PADZEROS(k, srcChannel, weights0);
            PADZEROS(k, srcChannel, weights1);
            {
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4 * 4));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + (k4 + 1) * 4) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + (k4 + 2) * 4) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + (k4 + 3) * 4) : (FLOAT4)0);
                DOT16X16(in, weights0, out0.s0);
                DOT16X16(in, weights1, out0.s1);
            }
            #endif
        }
        #endif
    }
    
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif

    vstore2(CONVERT_FLOAT2(out0), 0, output+out_offset);
}

__kernel void gemv_conv_c1_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
#endif
#endif
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int bhw,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b h w

    UNIFORM_BOUNDRY_CHECK(x, y);
    int idn = x;
    int idm = y;

    COMPUTE_FLOAT bias0 = bias[x];
    COMPUTE_FLOAT out0 = bias0;
    
    int input_offset0 = idm * 4;
    
    int out_offset = ((x / 4) * bhw + idm) * 4 + (x % 4);
#ifndef USE_IMAGE
    int weight_offset = x * WEIGHT_STRIDE;
    int weight_oc_offset = dstChannelC4 * 4 * WEIGHT_STRIDE;
#endif

    const int loop = (blockDim + CHANNEL_PACK - 1) / CHANNEL_PACK;
#ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif
    
    for (int i = 0; i < blockNum; ++i){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT2 ScaleOffset = CONVERT_COMPUTE_FLOAT2(vload2(x, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; ++j) {
            int k = i * loop + j;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            int k32 = k << 5;
            COMPUTE_FLOAT16 weights00, weights01;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn, k)));
                char16 charWeights0, charWeights1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                weights00 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights01 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            {
                COMPUTE_FLOAT16 in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + k32));
                COMPUTE_FLOAT16 in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + k32 + 16));
                DOT16X16(in0, weights00, out0);DOT16X16(in1, weights01, out0);
            }
            #else
            COMPUTE_FLOAT16 weights;
            #ifdef USE_IMAGE
            weights = readWeight(weight, idn, k, ScaleOffset.s0, ScaleOffset.s1);
            #else
            weights = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            #endif
            {
                COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(k, input));
                DOT16X16(in, weights, out0);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            int k8 = k << 3;
            COMPUTE_FLOAT16 weights00, weights01;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(idn, k)));
                char16 charWeights0, charWeights1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                weights00 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights01 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
                
                PADZEROS(k, srcChannel, weights00);PADZEROS(k + 16, srcChannel, weights01);
            }
            {
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k8 * 4));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + (k8 + 1) * 4) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + (k8 + 2) * 4) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + (k8 + 3) * 4) : (FLOAT4)0);
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + (k8 + 4) * 4) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + (k8 + 5) * 4) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + (k8 + 6) * 4) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + (k8 + 7) * 4) : (FLOAT4)0);
                DOT16X16(in0, weights00, out0);DOT16X16(in1, weights01, out0);
            }
            #else
            int k4 = k << 2;
            COMPUTE_FLOAT16 weights;
            #ifdef USE_IMAGE
            weights = readWeight(weight, idn, k, ScaleOffset.s0, ScaleOffset.s1);
            #else
            weights = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            #endif
            PADZEROS(k, srcChannel, weights);
            {
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4 * 4));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + (k4 + 1) * 4) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + (k4 + 2) * 4) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + (k4 + 3) * 4) : (FLOAT4)0);
                DOT16X16(in, weights, out0);
            }
            #endif
        }
        #endif
    }
    
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
    output[out_offset] = out0;
}
