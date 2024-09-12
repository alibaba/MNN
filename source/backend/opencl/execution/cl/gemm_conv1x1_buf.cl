#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

#define GLOBAL_SIZE_DIM3 \
    __private int global_size_dim0, __private int global_size_dim1, __private int global_size_dim2,

#define UNIFORM_BOUNDRY_CHECK3(index0, index1, index2) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1 || index2 >= global_size_dim2) { \
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

__kernel void inverse_quant_weight(GLOBAL_SIZE_DIM2
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
    __global FLOAT* output,
    __private const int outputChannelAlign,
    __private const int outputChannel4Align,
    __private const int blockDim){
    const int x = get_global_id(0); //ic
    const int y = get_global_id(1); //oc

    UNIFORM_BOUNDRY_CHECK(x, y);
    #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
    
    const int ic = x << 5;
    const int oc = y << 2;
    const int output_offset = ic * outputChannelAlign + oc;

    int kindex = (ic / blockDim) * outputChannel4Align * 2;
    COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(0, dequantScaleOffset + kindex + oc * 2));
    COMPUTE_FLOAT16 weights00, weights01, weights10, weights11, weights20, weights21, weights30, weights31;
    {
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(oc, x)));
        uchar16 charWeightsInt41 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(oc + 1, x)));
        uchar16 charWeightsInt42 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(oc + 2, x)));
        uchar16 charWeightsInt43 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(oc + 3, x)));
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
    COMPUTE_FLOAT *weights00_ptr = (COMPUTE_FLOAT *)&weights00;
    COMPUTE_FLOAT *weights10_ptr = (COMPUTE_FLOAT *)&weights10;
    COMPUTE_FLOAT *weights20_ptr = (COMPUTE_FLOAT *)&weights20;
    COMPUTE_FLOAT *weights30_ptr = (COMPUTE_FLOAT *)&weights30;
    COMPUTE_FLOAT *weights01_ptr = (COMPUTE_FLOAT *)&weights01;
    COMPUTE_FLOAT *weights11_ptr = (COMPUTE_FLOAT *)&weights11;
    COMPUTE_FLOAT *weights21_ptr = (COMPUTE_FLOAT *)&weights21;
    COMPUTE_FLOAT *weights31_ptr = (COMPUTE_FLOAT *)&weights31;
    #pragma unroll
    for (int i = 0; i < 16; ++i){
        FLOAT4 out = CONVERT_FLOAT4((COMPUTE_FLOAT4)(weights00_ptr[i], weights10_ptr[i], weights20_ptr[i], weights30_ptr[i]));
        vstore4(out, 0, output+output_offset+i*outputChannelAlign);
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i){
        FLOAT4 out = CONVERT_FLOAT4((COMPUTE_FLOAT4)(weights01_ptr[i], weights11_ptr[i], weights21_ptr[i], weights31_ptr[i]));
        vstore4(out, 0, output+output_offset+(i + 16)*outputChannelAlign);
    }
    #else
    const int ic = x << 4;
    const int oc = y << 2;
#ifndef USE_IMAGE
    #if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = oc * 8;
    int weight_oc_offset = outputChannel4Align * 8;
    int weight_stride = 8;
    #else
    int weight_offset = oc * 16;
    int weight_oc_offset = outputChannel4Align * 16;
    int weight_stride = 16;
    #endif
#endif
    const int output_offset = ic * outputChannelAlign + oc;

    int kindex = (ic / blockDim) * outputChannel4Align * 2;
    COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(0, dequantScaleOffset + kindex + oc * 2));
    #ifdef USE_IMAGE
    COMPUTE_FLOAT16 weights0 = readWeight(weight, oc, x, ScaleOffset.s0, ScaleOffset.s1);
    COMPUTE_FLOAT16 weights1 = readWeight(weight, oc + 1, x, ScaleOffset.s2, ScaleOffset.s3);
    COMPUTE_FLOAT16 weights2 = readWeight(weight, oc + 2, x, ScaleOffset.s4, ScaleOffset.s5);
    COMPUTE_FLOAT16 weights3 = readWeight(weight, oc + 3, x, ScaleOffset.s6, ScaleOffset.s7);
    #else
    COMPUTE_FLOAT16 weights0 = readWeight(weight + weight_offset + x * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
    COMPUTE_FLOAT16 weights1 = readWeight(weight + weight_offset + x * weight_oc_offset + weight_stride, 0, 0, ScaleOffset.s2, ScaleOffset.s3);
    COMPUTE_FLOAT16 weights2 = readWeight(weight + weight_offset + x * weight_oc_offset + 2 * weight_stride, 0, 0, ScaleOffset.s4, ScaleOffset.s5);
    COMPUTE_FLOAT16 weights3 = readWeight(weight + weight_offset + x * weight_oc_offset + 3 * weight_stride, 0, 0, ScaleOffset.s6, ScaleOffset.s7);
    #endif
    COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT*)&weights0;
    COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT*)&weights1;
    COMPUTE_FLOAT *weights2_ptr = (COMPUTE_FLOAT*)&weights2;
    COMPUTE_FLOAT *weights3_ptr = (COMPUTE_FLOAT*)&weights3;
    #pragma unroll
    for (int i = 0; i < 16; ++i){
        FLOAT4 out = CONVERT_FLOAT4((COMPUTE_FLOAT4)(weights0_ptr[i], weights1_ptr[i], weights2_ptr[i], weights3_ptr[i]));
        vstore4(out, 0, output+output_offset+i*outputChannelAlign);
    }
    #endif
}

__kernel void reshape_nchw4_nhwc4(GLOBAL_SIZE_DIM2
__global const FLOAT* input,
__global FLOAT* output,
__private const int bhw,
__private const int channel,
__private const int channelAlign){
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //bhw

    UNIFORM_BOUNDRY_CHECK(x, y);
    
    const int x4 = x << 2;
    const int y4 = y << 2;
    const int input_offset = (x * bhw + y4) * 4;
    FLOAT4 in0 = vload4(0, input + input_offset);
    FLOAT4 in1 = (y4 + 1 < bhw) ? vload4(0, input + input_offset + 4) : (FLOAT4)0;
    FLOAT4 in2 = (y4 + 2 < bhw) ? vload4(0, input + input_offset + 8) : (FLOAT4)0;
    FLOAT4 in3 = (y4 + 3 < bhw) ? vload4(0, input + input_offset + 12) : (FLOAT4)0;
    
#ifdef INPUT_CHANNEL_LEAVE
    if(x4 + 3 >= channel){
        FLOAT *in0_ptr = (FLOAT*)&in0;
        FLOAT *in1_ptr = (FLOAT*)&in1;
        FLOAT *in2_ptr = (FLOAT*)&in2;
        FLOAT *in3_ptr = (FLOAT*)&in3;
        int remain = x4 + 3 - channel;
        for(int i = remain; i >= 0; i--){
            in0_ptr[3 - i] = 0;
            in1_ptr[3 - i] = 0;
            in2_ptr[3 - i] = 0;
            in3_ptr[3 - i] = 0;
        }
    }
#endif
    
#ifdef FORMAT_CNHW
    int idx = x / 4;
    int idy = x % 4;
    const int bhw4 = (bhw + 3) / 4 * 4;
    int output_offset = ((idx * bhw4 + y4) * 4 + idy) * 4; // [c/16 b 4 4]
    vstore4(in0, 0, output+output_offset);
    vstore4(in1, 0, output+output_offset+16);
    vstore4(in2, 0, output+output_offset+32);
    vstore4(in3, 0, output+output_offset+48);
#else
    FLOAT16 out = (FLOAT16)(in0.s0, in1.s0, in2.s0, in3.s0, in0.s1, in1.s1, in2.s1, in3.s1, in0.s2, in1.s2, in2.s2, in3.s2, in0.s3, in1.s3, in2.s3, in3.s3);
    const int output_offset = (y * channelAlign + x4) * 4;
    vstore16(out, 0, output+output_offset);
#endif
}

__kernel void reshape_nhwc4_nchw4(GLOBAL_SIZE_DIM2
__global const FLOAT* input,
__global FLOAT* output,
__private const int bhw,
__private const int channelAlign){
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //bhw

    UNIFORM_BOUNDRY_CHECK(x, y);
    
    const int x4 = x << 2;
    const int y4 = y << 2;
    const int output_offset = (x * bhw + y4) * 4;
    

    const int input_offset = (y * channelAlign + x4) * 4;
    FLOAT16 in = vload16(0, input + input_offset);
    
    FLOAT4 out0 = (FLOAT4)(in.s0, in.s4, in.s8, in.sc);
    FLOAT4 out1 = (FLOAT4)(in.s1, in.s5, in.s9, in.sd);
    FLOAT4 out2 = (FLOAT4)(in.s2, in.s6, in.sa, in.se);
    FLOAT4 out3 = (FLOAT4)(in.s3, in.s7, in.sb, in.sf);

    vstore4(out0, 0, output+output_offset);
    if(y4 + 1 >= bhw) return;
    vstore4(out1, 0, output+output_offset+4);
    if(y4 + 2 >= bhw) return;
    vstore4(out2, 0, output+output_offset+8);
    if(y4 + 3 >= bhw) return;
    vstore4(out3, 0, output+output_offset+12);
}


__kernel void gemm_b4_c4_buf(GLOBAL_SIZE_DIM2
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
                        __private const int bhw4,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b

    UNIFORM_BOUNDRY_CHECK(x, y);

    const int out_c_idx = x << 2;
    const int out_b_idx = y << 2;

    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(0, bias + out_c_idx));
    COMPUTE_FLOAT4 out = (COMPUTE_FLOAT4)bias0.s0;
    COMPUTE_FLOAT4 out1 = (COMPUTE_FLOAT4)bias0.s1, out2 = (COMPUTE_FLOAT4)bias0.s2, out3 = (COMPUTE_FLOAT4)bias0.s3;

#ifdef FORMAT_CNHW
    int input_offset = out_b_idx * 16;
#else
    int input_offset = out_b_idx * srcChannelAlign;
#endif
    int out_offset = out_b_idx * dstChannelAlign + out_c_idx * 4;
    
#ifndef USE_IMAGE
    int weight_offset = out_c_idx * WEIGHT_STRIDE;
    int weight_oc_offset = dstChannelAlign * WEIGHT_STRIDE;
#endif

    const int loop = (blockDim + CHANNEL_PACK - 1) / CHANNEL_PACK;
    
    for (int i = 0; i < blockNum; i++){
        int kindex = i * dstChannelAlign * 2;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(0, dequantScaleOffset + kindex + out_c_idx * 2));
        for (int j = 0; j < loop; j++) {
            int k = i * loop + j;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            COMPUTE_FLOAT16 weights00, weights01, weights10, weights11, weights20, weights21, weights30, weights31;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                uchar16 charWeightsInt41 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 1, k)));
                uchar16 charWeightsInt42 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 2, k)));
                uchar16 charWeightsInt43 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 3, k)));
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
            #ifdef FORMAT_CNHW
            int k2 = k << 1;
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16));
            DOT16X16(in, weights00, out.s0);
            DOT16X16(in, weights10, out1.s0);
            DOT16X16(in, weights20, out2.s0);
            DOT16X16(in, weights30, out3.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 16));
            DOT16X16(in, weights00, out.s1);
            DOT16X16(in, weights10, out1.s1);
            DOT16X16(in, weights20, out2.s1);
            DOT16X16(in, weights30, out3.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 32));
            DOT16X16(in, weights00, out.s2);
            DOT16X16(in, weights10, out1.s2);
            DOT16X16(in, weights20, out2.s2);
            DOT16X16(in, weights30, out3.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 48));
            DOT16X16(in, weights00, out.s3);
            DOT16X16(in, weights10, out1.s3);
            DOT16X16(in, weights20, out2.s3);
            DOT16X16(in, weights30, out3.s3);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16));
            DOT16X16(in, weights01, out.s0);
            DOT16X16(in, weights11, out1.s0);
            DOT16X16(in, weights21, out2.s0);
            DOT16X16(in, weights31, out3.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 16));
            DOT16X16(in, weights01, out.s1);
            DOT16X16(in, weights11, out1.s1);
            DOT16X16(in, weights21, out2.s1);
            DOT16X16(in, weights31, out3.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 32));
            DOT16X16(in, weights01, out.s2);
            DOT16X16(in, weights11, out1.s2);
            DOT16X16(in, weights21, out2.s2);
            DOT16X16(in, weights31, out3.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 48));
            DOT16X16(in, weights01, out.s3);
            DOT16X16(in, weights11, out1.s3);
            DOT16X16(in, weights21, out2.s3);
            DOT16X16(in, weights31, out3.s3);
            #else
            int k32 = k << 5;
            COMPUTE_FLOAT *weights00_ptr = (COMPUTE_FLOAT *)&weights00;
            COMPUTE_FLOAT *weights10_ptr = (COMPUTE_FLOAT *)&weights10;
            COMPUTE_FLOAT *weights20_ptr = (COMPUTE_FLOAT *)&weights20;
            COMPUTE_FLOAT *weights30_ptr = (COMPUTE_FLOAT *)&weights30;
            COMPUTE_FLOAT *weights01_ptr = (COMPUTE_FLOAT *)&weights01;
            COMPUTE_FLOAT *weights11_ptr = (COMPUTE_FLOAT *)&weights11;
            COMPUTE_FLOAT *weights21_ptr = (COMPUTE_FLOAT *)&weights21;
            COMPUTE_FLOAT *weights31_ptr = (COMPUTE_FLOAT *)&weights31;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights00_ptr[i], out);
                out1 = mad(in, weights10_ptr[i], out1);
                out2 = mad(in, weights20_ptr[i], out2);
                out3 = mad(in, weights30_ptr[i], out3);
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i + 16) * 4));
                out = mad(in, weights01_ptr[i], out);
                out1 = mad(in, weights11_ptr[i], out1);
                out2 = mad(in, weights21_ptr[i], out2);
                out3 = mad(in, weights31_ptr[i], out3);
            }
            #endif
            #else
            COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
            #ifdef USE_IMAGE
            weights0 = readWeight(weight, out_c_idx, k, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight, out_c_idx + 1, k, ScaleOffset.s2, ScaleOffset.s3);
            weights2 = readWeight(weight, out_c_idx + 2, k, ScaleOffset.s4, ScaleOffset.s5);
            weights3 = readWeight(weight, out_c_idx + 3, k, ScaleOffset.s6, ScaleOffset.s7);
            #else
            weights0 = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight + weight_offset + k * weight_oc_offset + WEIGHT_STRIDE, 0, 0, ScaleOffset.s2, ScaleOffset.s3);
            weights2 = readWeight(weight + weight_offset + k * weight_oc_offset + 2 * WEIGHT_STRIDE, 0, 0, ScaleOffset.s4, ScaleOffset.s5);
            weights3 = readWeight(weight + weight_offset + k * weight_oc_offset + 3 * WEIGHT_STRIDE, 0, 0, ScaleOffset.s6, ScaleOffset.s7);
            #endif
            #ifdef FORMAT_CNHW
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16));
            DOT16X16(in, weights0, out.s0);
            DOT16X16(in, weights1, out1.s0);
            DOT16X16(in, weights2, out2.s0);
            DOT16X16(in, weights3, out3.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 16));
            DOT16X16(in, weights0, out.s1);
            DOT16X16(in, weights1, out1.s1);
            DOT16X16(in, weights2, out2.s1);
            DOT16X16(in, weights3, out3.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 32));
            DOT16X16(in, weights0, out.s2);
            DOT16X16(in, weights1, out1.s2);
            DOT16X16(in, weights2, out2.s2);
            DOT16X16(in, weights3, out3.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 48));
            DOT16X16(in, weights0, out.s3);
            DOT16X16(in, weights1, out1.s3);
            DOT16X16(in, weights2, out2.s3);
            DOT16X16(in, weights3, out3.s3);
            #else
            int k16 = k << 4;
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            COMPUTE_FLOAT *weights2_ptr = (COMPUTE_FLOAT *)&weights2;
            COMPUTE_FLOAT *weights3_ptr = (COMPUTE_FLOAT *)&weights3;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights1_ptr[i], out1);
                out2 = mad(in, weights2_ptr[i], out2);
                out3 = mad(in, weights3_ptr[i], out3);
            }
            #endif
            #endif
        }
    }
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    vstore4(CONVERT_FLOAT4(out), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), 0, output+out_offset + 4);
    vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset + 8);
    vstore4(CONVERT_FLOAT4(out3), 0, output+out_offset + 12);
}

__kernel void gemm_b4_c2_buf(GLOBAL_SIZE_DIM2
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
                        __private const int bhw4,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b

    UNIFORM_BOUNDRY_CHECK(x, y);

    const int out_c_idx = x << 1;
    const int out_b_idx = y << 2;

    COMPUTE_FLOAT2 bias0 = CONVERT_COMPUTE_FLOAT2(vload2(0, bias + out_c_idx));
    COMPUTE_FLOAT4 out = (COMPUTE_FLOAT4)bias0.s0;
    COMPUTE_FLOAT4 out1 = (COMPUTE_FLOAT4)bias0.s1;
    
#ifdef FORMAT_CNHW
    int input_offset = out_b_idx * 16;
#else
    int input_offset = out_b_idx * srcChannelAlign;
#endif
    int out_offset = out_b_idx * dstChannelAlign + out_c_idx * 4;
    
#ifndef USE_IMAGE
    int weight_offset = out_c_idx * WEIGHT_STRIDE;
    int weight_oc_offset = dstChannelAlign * WEIGHT_STRIDE;
#endif

    const int loop = (blockDim + CHANNEL_PACK - 1) / CHANNEL_PACK;

    for (int i = 0; i < blockNum; i++){
        int kindex = i * dstChannelAlign * 2;
        COMPUTE_FLOAT4 ScaleOffset = CONVERT_COMPUTE_FLOAT4(vload4(0, dequantScaleOffset + kindex + out_c_idx * 2));
        for (int j = 0; j < loop; j++) {
            int k = i * loop + j;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            COMPUTE_FLOAT16 weights00, weights01, weights10, weights11;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                uchar16 charWeightsInt41 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 1, k)));
                char16 charWeights0, charWeights1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                weights00 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights01 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
                weights10 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s2 + ScaleOffset.s3;
                weights11 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
            }
            #ifdef FORMAT_CNHW
            int k2 = k << 1;
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16));
            DOT16X16(in, weights00, out.s0);
            DOT16X16(in, weights10, out1.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 16));
            DOT16X16(in, weights00, out.s1);
            DOT16X16(in, weights10, out1.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 32));
            DOT16X16(in, weights00, out.s2);
            DOT16X16(in, weights10, out1.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 48));
            DOT16X16(in, weights00, out.s3);
            DOT16X16(in, weights10, out1.s3);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16));
            DOT16X16(in, weights01, out.s0);
            DOT16X16(in, weights11, out1.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 16));
            DOT16X16(in, weights01, out.s1);
            DOT16X16(in, weights11, out1.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 32));
            DOT16X16(in, weights01, out.s2);
            DOT16X16(in, weights11, out1.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 48));
            DOT16X16(in, weights01, out.s3);
            DOT16X16(in, weights11, out1.s3);
            #else
            int k32 = k << 5;
            COMPUTE_FLOAT *weights00_ptr = (COMPUTE_FLOAT *)&weights00;
            COMPUTE_FLOAT *weights10_ptr = (COMPUTE_FLOAT *)&weights10;
            COMPUTE_FLOAT *weights01_ptr = (COMPUTE_FLOAT *)&weights01;
            COMPUTE_FLOAT *weights11_ptr = (COMPUTE_FLOAT *)&weights11;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights00_ptr[i], out);
                out1 = mad(in, weights10_ptr[i], out1);
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i + 16) * 4));
                out = mad(in, weights01_ptr[i], out);
                out1 = mad(in, weights11_ptr[i], out1);
            }
            #endif
            #else
            COMPUTE_FLOAT16 weights0, weights1;
            #ifdef USE_IMAGE
            weights0 = readWeight(weight, out_c_idx, k, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight, out_c_idx + 1, k, ScaleOffset.s2, ScaleOffset.s3);
            #else
            weights0 = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            weights1 = readWeight(weight + weight_offset + k * weight_oc_offset + WEIGHT_STRIDE, 0, 0, ScaleOffset.s2, ScaleOffset.s3);
            #endif
            #ifdef FORMAT_CNHW
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16));
            DOT16X16(in, weights0, out.s0);
            DOT16X16(in, weights1, out1.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 16));
            DOT16X16(in, weights0, out.s1);
            DOT16X16(in, weights1, out1.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 32));
            DOT16X16(in, weights0, out.s2);
            DOT16X16(in, weights1, out1.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 48));
            DOT16X16(in, weights0, out.s3);
            DOT16X16(in, weights1, out1.s3);
            #else
            int k16 = k << 4;
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights1_ptr[i], out1);
            }
            #endif
            #endif
        }
    }
    
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    vstore4(CONVERT_FLOAT4(out), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), 0, output+out_offset+4);
}

__kernel void gemm_b4_c1_buf(GLOBAL_SIZE_DIM2
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
                        __private const int bhw4,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b

    UNIFORM_BOUNDRY_CHECK(x, y);

    const int out_c_idx = x;
    const int out_b_idx = y << 2;

    COMPUTE_FLOAT bias0 = bias[out_c_idx];
    COMPUTE_FLOAT4 out = (COMPUTE_FLOAT4)bias0;
    
#ifdef FORMAT_CNHW
    int input_offset = out_b_idx * 16;
#else
    int input_offset = out_b_idx * srcChannelAlign;
#endif
    int out_offset = out_b_idx * dstChannelAlign + out_c_idx * 4;

#ifndef USE_IMAGE
    int weight_offset = out_c_idx * WEIGHT_STRIDE;
    int weight_oc_offset = dstChannelAlign * WEIGHT_STRIDE;
#endif

    const int loop = (blockDim + CHANNEL_PACK - 1) / CHANNEL_PACK;
    
    for (int i = 0; i < blockNum; i++){
        int kindex = i * dstChannelAlign * 2;
        COMPUTE_FLOAT2 ScaleOffset = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, dequantScaleOffset + kindex));
        for (int j = 0; j < loop; j++) {
            int k = i * loop + j;
            #if defined(USE_LOW_BIT_WEIGHT_INT4) && defined(USE_IMAGE)
            COMPUTE_FLOAT16 weights00, weights01, weights10, weights11;
            {
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                char16 charWeights0, charWeights1;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                weights00 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights01 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            #ifdef FORMAT_CNHW
            int k2 = k << 1;
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16));
            DOT16X16(in, weights00, out.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 16));
            DOT16X16(in, weights00, out.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 32));
            DOT16X16(in, weights00, out.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k2 * bhw4 * 16 + 48));
            DOT16X16(in, weights00, out.s3);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16));
            DOT16X16(in, weights01, out.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 16));
            DOT16X16(in, weights01, out.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 32));
            DOT16X16(in, weights01, out.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + (k2 + 1) * bhw4 * 16 + 48));
            DOT16X16(in, weights01, out.s3);
            #else
            int k32 = k << 5;
            COMPUTE_FLOAT *weights00_ptr = (COMPUTE_FLOAT *)&weights00;
            COMPUTE_FLOAT *weights01_ptr = (COMPUTE_FLOAT *)&weights01;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights00_ptr[i], out);
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i + 16) * 4));
                out = mad(in, weights01_ptr[i], out);
            }
            #endif
            #else
            COMPUTE_FLOAT16 weights;
            #ifdef USE_IMAGE
            weights = readWeight(weight, out_c_idx, k, ScaleOffset.s0, ScaleOffset.s1);
            #else
            weights = readWeight(weight + weight_offset + k * weight_oc_offset, 0, 0, ScaleOffset.s0, ScaleOffset.s1);
            #endif
            #ifdef FORMAT_CNHW
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16));
            DOT16X16(in, weights, out.s0);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 16));
            DOT16X16(in, weights, out.s1);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 32));
            DOT16X16(in, weights, out.s2);
            in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4 * 16 + 48));
            DOT16X16(in, weights, out.s3);
            #else
            int k16 = k << 4;
            COMPUTE_FLOAT *weights_ptr = (COMPUTE_FLOAT *)&weights;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights_ptr[i], out);
            }
            #endif
            #endif
        }
    }
    
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif
    vstore4(CONVERT_FLOAT4(out), 0, output+out_offset);
}
