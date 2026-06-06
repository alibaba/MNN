#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#ifdef USE_IMAGE1D_INPUT
#ifdef MNN_SUPPORT_FP16
#define RI_F(i, coord) read_imageh(i, coord)
#else
#define RI_F(i, coord) read_imagef(i, coord)
#endif
#endif
#define GLOBAL_SIZE_DIM_2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define GLOBAL_SIZE_DIM_3 \
    __private int global_size_dim0, __private int global_size_dim1, __private int global_size_dim2,

#define UNIFORM_BOUNDRY_CHECK_2(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

#define UCHAR4_TO_CHAR8(b, scale, offset) \
    wei.s0 = CONVERT_FLOAT((b.s0 >> 4)); \
    wei.s1 = CONVERT_FLOAT((b.s0 & 15)); \
    wei.s2 = CONVERT_FLOAT((b.s1 >> 4)); \
    wei.s3 = CONVERT_FLOAT((b.s1 & 15)); \
    wei.s4 = CONVERT_FLOAT((b.s2 >> 4)); \
    wei.s5 = CONVERT_FLOAT((b.s2 & 15)); \
    wei.s6 = CONVERT_FLOAT((b.s3 >> 4)); \
    wei.s7 = CONVERT_FLOAT((b.s3 & 15)); \
    wei = wei * scale + offset;

// 2bit unpack: 2 input bytes (b0,b1) → 8 weight floats. Caller must fold the
// origin offset (-2) into the bias term (offset = scaleOffset_asym - 2 * scale,
// or = -2 * scale in the non-asymmetric path).
// b0 holds OC[0..3] (bits 7:6,5:4,3:2,1:0), b1 holds OC[4..7].
#define UCHAR2_TO_CHAR8_W2(b0, b1, scale, offset) \
    wei.s0 = CONVERT_FLOAT(((int)((b0 >> 6) & 3))); \
    wei.s1 = CONVERT_FLOAT(((int)((b0 >> 4) & 3))); \
    wei.s2 = CONVERT_FLOAT(((int)((b0 >> 2) & 3))); \
    wei.s3 = CONVERT_FLOAT(((int)((b0 >> 0) & 3))); \
    wei.s4 = CONVERT_FLOAT(((int)((b1 >> 6) & 3))); \
    wei.s5 = CONVERT_FLOAT(((int)((b1 >> 4) & 3))); \
    wei.s6 = CONVERT_FLOAT(((int)((b1 >> 2) & 3))); \
    wei.s7 = CONVERT_FLOAT(((int)((b1 >> 0) & 3))); \
    wei = wei * scale + offset;

// 3bit unpack: 2 low bytes (b0,b1) + 1 high byte (h) → 8 weight floats. Caller
// must fold the origin offset (-4) into the bias the same way as w2.
// b0/b1 carry the same low-bit layout as w2; h carries the high bit (bit 7-(k*4+j)) for OC (k*4+j).
#define UCHAR3_TO_CHAR8_W3(b0, b1, h, scale, offset) \
    wei.s0 = CONVERT_FLOAT(((int)(((b0 >> 6) & 3) | (((h >> 7) & 1) << 2)))); \
    wei.s1 = CONVERT_FLOAT(((int)(((b0 >> 4) & 3) | (((h >> 6) & 1) << 2)))); \
    wei.s2 = CONVERT_FLOAT(((int)(((b0 >> 2) & 3) | (((h >> 5) & 1) << 2)))); \
    wei.s3 = CONVERT_FLOAT(((int)(((b0 >> 0) & 3) | (((h >> 4) & 1) << 2)))); \
    wei.s4 = CONVERT_FLOAT(((int)(((b1 >> 6) & 3) | (((h >> 3) & 1) << 2)))); \
    wei.s5 = CONVERT_FLOAT(((int)(((b1 >> 4) & 3) | (((h >> 2) & 1) << 2)))); \
    wei.s6 = CONVERT_FLOAT(((int)(((b1 >> 2) & 3) | (((h >> 1) & 1) << 2)))); \
    wei.s7 = CONVERT_FLOAT(((int)(((b1 >> 0) & 3) | (((h >> 0) & 1) << 2)))); \
    wei = wei * scale + offset;


#if WGS >= 8
__kernel void gemv_conv_c8_buf(GLOBAL_SIZE_DIM_3
#ifdef USE_IMAGE1D_INPUT
                        __read_only image1d_buffer_t input,
#else
                        __global const FLOAT* input,
#endif
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        #if QUANT_BIT == 8
                        __global const char *weight,
                        #else
                        __global const uchar *weight,
                        #endif
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1); //oc/8
    const int oc8 = oc << 3;
    
#if QUANT_BIT == 8
    #if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop = max((srcChannel + 2 - 1) / 2 - 1, 0);
    #else
    const int loop = (srcChannel + 2 - 1) / 2;
    #endif
    #ifndef USE_IMAGE
    const int weight_offset = oc * srcChannelC4 * 32;
    #endif
#else
    #if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop = max((srcChannel + 4 - 1) / 4 - 1, 0);
    #else
    const int loop = (srcChannel + 4 - 1) / 4;
    #endif
    #ifndef USE_IMAGE
        #if QUANT_BIT == 2
        const int weight_offset = oc * srcChannelC4 * 8;
        #elif QUANT_BIT == 3
        const int weight_offset = oc * srcChannelC4 * 12;
        #else
        const int weight_offset = oc * srcChannelC4 * 16;
        #endif
    #endif
#endif

    COMPUTE_FLOAT8 out0 = 0;
    int input_offset = 0, output_offset = oc8;
    __local COMPUTE_FLOAT8 sum0[WGS];
#ifdef COMPUTE_BATCH
    const int out_b_idx  = get_global_id(2) << 2; //b/4
    __local COMPUTE_FLOAT8 sum1[WGS];
    __local COMPUTE_FLOAT8 sum2[WGS];
    __local COMPUTE_FLOAT8 sum3[WGS];
    COMPUTE_FLOAT8 out1 = 0, out2 = 0, out3 = 0;
    input_offset = out_b_idx * srcChannelAlign;
    output_offset = oc8 + out_b_idx * dstChannelAlign;
#endif
    for(int j = lid; j < loop; j+=WGS){
        #if QUANT_BIT == 8
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
        COMPUTE_FLOAT2 in = CONVERT_COMPUTE_FLOAT2(vload2(0, input + input_offset + k2));
        #ifdef COMPUTE_BATCH
        COMPUTE_FLOAT2 in1 = CONVERT_COMPUTE_FLOAT2(vload2(0, input + input_offset + srcChannelAlign + k2));
        COMPUTE_FLOAT2 in2 = CONVERT_COMPUTE_FLOAT2(vload2(0, input + input_offset + srcChannelAlign * 2 + k2));
        COMPUTE_FLOAT2 in3 = CONVERT_COMPUTE_FLOAT2(vload2(0, input + input_offset + srcChannelAlign * 3 + k2));
        #endif
        #ifdef USE_IMAGE
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(j, oc)))) * scale + offset;
        #else
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(j, weight + weight_offset)) * scale + offset;
        #endif
        {
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei.s01234567, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s0, wei.s01234567, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s0, wei.s01234567, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s0, wei.s01234567, out3);
            #endif
        }
        {
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei.s89abcdef, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s1, wei.s89abcdef, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s1, wei.s89abcdef, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s1, wei.s89abcdef, out3);
            #endif
        }
        #else
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
        #if QUANT_BIT == 2
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-2) * scale;
        #elif QUANT_BIT == 3
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-4) * scale;
        #else
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-8) * scale;
        #endif
        #endif
        COMPUTE_FLOAT8 wei;
        #ifdef USE_IMAGE1D_INPUT
        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(RI_F(input, (k4 + input_offset) >> 2));
        #else
        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4 + input_offset));
        #endif
        #ifdef COMPUTE_BATCH
        #ifdef USE_IMAGE1D_INPUT
        COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(RI_F(input, (input_offset + srcChannelAlign + k4) >> 2));
        COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4(RI_F(input, (input_offset + srcChannelAlign * 2 + k4) >> 2));
        COMPUTE_FLOAT4 in3 = CONVERT_COMPUTE_FLOAT4(RI_F(input, (input_offset + srcChannelAlign * 3 + k4) >> 2));
        #else
        COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + srcChannelAlign + k4));
        COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + srcChannelAlign * 2 + k4));
        COMPUTE_FLOAT4 in3 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + srcChannelAlign * 3 + k4));
        #endif
        #endif
        #if QUANT_BIT == 2
        uchar8 charWeightsInt20 = vload8(j, weight + weight_offset);
        {
            UCHAR2_TO_CHAR8_W2(charWeightsInt20.s0, charWeightsInt20.s1, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s0, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s0, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s0, wei, out3);
            #endif
        }
        {
            UCHAR2_TO_CHAR8_W2(charWeightsInt20.s2, charWeightsInt20.s3, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s1, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s1, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s1, wei, out3);
            #endif
        }
        {
            UCHAR2_TO_CHAR8_W2(charWeightsInt20.s4, charWeightsInt20.s5, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s2, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s2, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s2, wei, out3);
            #endif
        }
        {
            UCHAR2_TO_CHAR8_W2(charWeightsInt20.s6, charWeightsInt20.s7, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s3, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s3, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s3, wei, out3);
            #endif
        }
        #elif QUANT_BIT == 3
        const int wo3_j = j * 12;
        uchar8 charWeightsInt30Lo = vload8(0, weight + weight_offset + wo3_j);
        uchar4 charWeightsInt30Hi = vload4(0, weight + weight_offset + wo3_j + 8);
        {
            UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s0, charWeightsInt30Lo.s1, charWeightsInt30Hi.s0, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s0, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s0, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s0, wei, out3);
            #endif
        }
        {
            UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s2, charWeightsInt30Lo.s3, charWeightsInt30Hi.s1, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s1, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s1, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s1, wei, out3);
            #endif
        }
        {
            UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s4, charWeightsInt30Lo.s5, charWeightsInt30Hi.s2, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s2, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s2, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s2, wei, out3);
            #endif
        }
        {
            UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s6, charWeightsInt30Lo.s7, charWeightsInt30Hi.s3, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s3, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s3, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s3, wei, out3);
            #endif
        }
        #else
        #ifdef USE_IMAGE
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(j, oc)));
        #else
        uchar16 charWeightsInt40 = vload16(j, weight + weight_offset);
        #endif
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s0, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s0, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s0, wei, out3);
            #endif
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s1, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s1, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s1, wei, out3);
            #endif
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s2, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s2, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s2, wei, out3);
            #endif
        }
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.scdef, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
            #ifdef COMPUTE_BATCH
            out1 = mad((COMPUTE_FLOAT8)in1.s3, wei, out1);
            out2 = mad((COMPUTE_FLOAT8)in2.s3, wei, out2);
            out3 = mad((COMPUTE_FLOAT8)in3.s3, wei, out3);
            #endif
        }
        #endif
        #endif
    }
#if INPUT_CHANNEL_LEAVES_NUM != 0
    {
        #if QUANT_BIT == 8
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
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(loop, weight + weight_offset)) * scale + offset;
        #endif
        {
            out0 = mad((COMPUTE_FLOAT8)input[k2], wei.s01234567, out0);
        }
        #else
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
        #if QUANT_BIT == 2
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-2) * scale;
        #elif QUANT_BIT == 3
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-4) * scale;
        #else
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-8) * scale;
        #endif
        #endif
        COMPUTE_FLOAT8 wei;
        #if QUANT_BIT == 2
        uchar8 charWeightsInt20 = vload8(loop, weight + weight_offset);
        {
            UCHAR2_TO_CHAR8_W2(charWeightsInt20.s0, charWeightsInt20.s1, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4], wei, out0);
        }
        #if INPUT_CHANNEL_LEAVES_NUM >= 2
        {
            UCHAR2_TO_CHAR8_W2(charWeightsInt20.s2, charWeightsInt20.s3, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4 + 1], wei, out0);
        }
        #endif
        #if INPUT_CHANNEL_LEAVES_NUM >= 3
        {
            UCHAR2_TO_CHAR8_W2(charWeightsInt20.s4, charWeightsInt20.s5, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4 + 2], wei, out0);
        }
        #endif
        #elif QUANT_BIT == 3
        const int wo3_loop = loop * 12;
        uchar8 charWeightsInt30Lo = vload8(0, weight + weight_offset + wo3_loop);
        uchar4 charWeightsInt30Hi = vload4(0, weight + weight_offset + wo3_loop + 8);
        {
            UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s0, charWeightsInt30Lo.s1, charWeightsInt30Hi.s0, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4], wei, out0);
        }
        #if INPUT_CHANNEL_LEAVES_NUM >= 2
        {
            UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s2, charWeightsInt30Lo.s3, charWeightsInt30Hi.s1, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4 + 1], wei, out0);
        }
        #endif
        #if INPUT_CHANNEL_LEAVES_NUM >= 3
        {
            UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s4, charWeightsInt30Lo.s5, charWeightsInt30Hi.s2, scale, offset);
            out0 = mad((COMPUTE_FLOAT8)input[k4 + 2], wei, out0);
        }
        #endif
        #else
        #ifdef USE_IMAGE
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(loop, oc)));
        #else
        uchar16 charWeightsInt40 = vload16(loop, weight + weight_offset);
        #endif
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
            #ifdef USE_IMAGE1D_INPUT
            COMPUTE_FLOAT4 _leaves_in = CONVERT_COMPUTE_FLOAT4(RI_F(input, k4 >> 2));
            out0 = mad((COMPUTE_FLOAT8)_leaves_in.s0, wei, out0);
            #else
            out0 = mad((COMPUTE_FLOAT8)input[k4], wei, out0);
            #endif
        }
        #if INPUT_CHANNEL_LEAVES_NUM >= 2
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
            #ifdef USE_IMAGE1D_INPUT
            out0 = mad((COMPUTE_FLOAT8)_leaves_in.s1, wei, out0);
            #else
            out0 = mad((COMPUTE_FLOAT8)input[k4 + 1], wei, out0);
            #endif
        }
        #endif
        #if INPUT_CHANNEL_LEAVES_NUM >= 3
        {
            UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
            #ifdef USE_IMAGE1D_INPUT
            out0 = mad((COMPUTE_FLOAT8)_leaves_in.s2, wei, out0);
            #else
            out0 = mad((COMPUTE_FLOAT8)input[k4 + 2], wei, out0);
            #endif
        }
        #endif
        #endif
        #endif
    }
#endif
    sum0[lid] = out0;
    #ifdef COMPUTE_BATCH
    sum1[lid] = out1; sum2[lid] = out2; sum3[lid] = out3;
    #endif
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = WGS/2; i > 0; i /= 2){
        if (lid < i){
            sum0[lid] = sum0[lid] + sum0[lid + i];
            #ifdef COMPUTE_BATCH
            sum1[lid] = sum1[lid] + sum1[lid + i];
            sum2[lid] = sum2[lid] + sum2[lid + i];
            sum3[lid] = sum3[lid] + sum3[lid + i];
            #endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        COMPUTE_FLOAT8 vBias = CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc8));
        out0 = sum0[0] + vBias;
    #ifdef RELU
        out0 = fmax(out0, (COMPUTE_FLOAT8)0);
    #endif

    #ifdef RELU6
        out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    #endif
    #ifdef OUTPUT_CHANNEL_LEAVES
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output + output_offset);
        if(oc8 + 4 < dstChannelC4 * 4)
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output + 4 + output_offset);
    #else
        vstore8(CONVERT_FLOAT8(out0), 0, output  + output_offset);
    #endif
    #ifdef COMPUTE_BATCH
        out1 = sum1[0] + vBias; out2 = sum2[0] + vBias; out3 = sum3[0] + vBias;
        #ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT8)0);out2 = fmax(out2, (COMPUTE_FLOAT8)0);out3 = fmax(out3, (COMPUTE_FLOAT8)0);
        #endif
        #ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);out2 = clamp(out2, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);out3 = clamp(out3, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
        #endif
        vstore8(CONVERT_FLOAT8(out1), 0, output + output_offset + dstChannelAlign);
        vstore8(CONVERT_FLOAT8(out2), 0, output + output_offset + dstChannelAlign + dstChannelAlign);
        vstore8(CONVERT_FLOAT8(out3), 0, output + output_offset + dstChannelAlign + dstChannelAlign + dstChannelAlign);
    #endif
    }
}
#else
__kernel void gemv_conv_c8_buf(GLOBAL_SIZE_DIM_3
#ifdef USE_IMAGE1D_INPUT
                        __read_only image1d_buffer_t input,
#else
                        __global const FLOAT* input,
#endif
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        #if QUANT_BIT == 8
                        __global const char *weight,
                        #else
                        __global const uchar *weight,
                        #endif
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
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

#if QUANT_BIT == 8
    const int loop = (blockDim + 2 - 1) / 2;
    #if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
    #else
    const int loop_end = loop;
    #endif
    #ifndef USE_IMAGE
    const int weight_offset = oc * srcChannelC4 * 32;
    #endif
#else
    const int loop = (blockDim + 4 - 1) / 4;
    #if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
    #else
    const int loop_end = loop;
    #endif
    #ifndef USE_IMAGE
        #if QUANT_BIT == 2
        const int weight_offset = oc * srcChannelC4 * 8;
        #elif QUANT_BIT == 3
        const int weight_offset = oc * srcChannelC4 * 12;
        #else
        const int weight_offset = oc * srcChannelC4 * 16;
        #endif
    #endif
#endif
    COMPUTE_FLOAT8 out0 = CONVERT_COMPUTE_FLOAT8(vload8(0, bias + oc8));
    for (int i = 0; i < blockNum; i++){
    #if QUANT_BIT == 8
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
    #else
        #ifdef ASYMMETRIC
        COMPUTE_FLOAT8 scale, offset;
        {
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + oc8 * 2 + i * dstChannelC4 * 8)) / coef);
            scale = scaleOffset.s02468ace;
            offset = scaleOffset.s13579bdf;
        }
        #else
        COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + oc8 + i * dstChannelC4 * 4)) / coef);
        #if QUANT_BIT == 2
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-2) * scale;
        #elif QUANT_BIT == 3
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-4) * scale;
        #else
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-8) * scale;
        #endif
        #endif
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            COMPUTE_FLOAT8 wei;
            #ifdef USE_IMAGE1D_INPUT
            COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(RI_F(input, (k << 2) >> 2));
            #else
            COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + (k << 2)));
            #endif
            #if QUANT_BIT == 2
            uchar8 charWeightsInt20 = vload8(k, weight + weight_offset);
            {
                UCHAR2_TO_CHAR8_W2(charWeightsInt20.s0, charWeightsInt20.s1, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
            }
            {
                UCHAR2_TO_CHAR8_W2(charWeightsInt20.s2, charWeightsInt20.s3, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
            }
            {
                UCHAR2_TO_CHAR8_W2(charWeightsInt20.s4, charWeightsInt20.s5, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
            }
            {
                UCHAR2_TO_CHAR8_W2(charWeightsInt20.s6, charWeightsInt20.s7, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
            }
            #elif QUANT_BIT == 3
            const int wo3_k = k * 12;
            uchar8 charWeightsInt30Lo = vload8(0, weight + weight_offset + wo3_k);
            uchar4 charWeightsInt30Hi = vload4(0, weight + weight_offset + wo3_k + 8);
            {
                UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s0, charWeightsInt30Lo.s1, charWeightsInt30Hi.s0, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
            }
            {
                UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s2, charWeightsInt30Lo.s3, charWeightsInt30Hi.s1, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
            }
            {
                UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s4, charWeightsInt30Lo.s5, charWeightsInt30Hi.s2, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
            }
            {
                UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s6, charWeightsInt30Lo.s7, charWeightsInt30Hi.s3, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
            }
            #else
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
            #endif
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            int k4 = k << 2;
            COMPUTE_FLOAT8 wei;
            #if QUANT_BIT == 2
            uchar8 charWeightsInt20 = vload8(k, weight + weight_offset);
            {
                UCHAR2_TO_CHAR8_W2(charWeightsInt20.s0, charWeightsInt20.s1, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4], wei, out0);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                UCHAR2_TO_CHAR8_W2(charWeightsInt20.s2, charWeightsInt20.s3, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4 + 1], wei, out0);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                UCHAR2_TO_CHAR8_W2(charWeightsInt20.s4, charWeightsInt20.s5, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4 + 2], wei, out0);
            }
            #endif
            #elif QUANT_BIT == 3
            const int wo3_k = k * 12;
            uchar8 charWeightsInt30Lo = vload8(0, weight + weight_offset + wo3_k);
            uchar4 charWeightsInt30Hi = vload4(0, weight + weight_offset + wo3_k + 8);
            {
                UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s0, charWeightsInt30Lo.s1, charWeightsInt30Hi.s0, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4], wei, out0);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s2, charWeightsInt30Lo.s3, charWeightsInt30Hi.s1, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4 + 1], wei, out0);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                UCHAR3_TO_CHAR8_W3(charWeightsInt30Lo.s4, charWeightsInt30Lo.s5, charWeightsInt30Hi.s2, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)input[k4 + 2], wei, out0);
            }
            #endif
            #else
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, oc)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s0123, scale, offset);
                #ifdef USE_IMAGE1D_INPUT
                COMPUTE_FLOAT4 _leaves_in2 = CONVERT_COMPUTE_FLOAT4(RI_F(input, k4 >> 2));
                out0 = mad((COMPUTE_FLOAT8)_leaves_in2.s0, wei, out0);
                #else
                out0 = mad((COMPUTE_FLOAT8)input[k4], wei, out0);
                #endif
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s4567, scale, offset);
                #ifdef USE_IMAGE1D_INPUT
                out0 = mad((COMPUTE_FLOAT8)_leaves_in2.s1, wei, out0);
                #else
                out0 = mad((COMPUTE_FLOAT8)input[k4 + 1], wei, out0);
                #endif
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                UCHAR4_TO_CHAR8(charWeightsInt40.s89ab, scale, offset);
                #ifdef USE_IMAGE1D_INPUT
                out0 = mad((COMPUTE_FLOAT8)_leaves_in2.s2, wei, out0);
                #else
                out0 = mad((COMPUTE_FLOAT8)input[k4 + 2], wei, out0);
                #endif
            }
            #endif
            #endif // QUANT_BIT branch in leaves (WGS<8)
        }
        #endif
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