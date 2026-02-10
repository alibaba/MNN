#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#ifdef INPUT_CHANNEL_LEAVE
    #define PADZEROSVEC(k, channel, data0, data1, data2, data3) \
        data0 = (k << 2) < channel ? data0 : 0; \
        data1 = (k << 2) + 1 < channel ? data1 : 0; \
        data2 = (k << 2) + 2 < channel ? data2 : 0; \
        data3 = (k << 2) + 3 < channel ? data3 : 0;
#else
    #define PADZEROSVEC(k, channel, data0, data1, data2, data3)
#endif

__kernel void gemm_conv(GLOBAL_SIZE_DIM2
                        __read_only image2d_t input,
#if QUANT_BIT == 8
                        __global const char *weight,
                        __global const float *dequantScaleOffset,
#else
                        __global const uchar *weight,
                        __global const float *dequantScaleOffset,
#endif
                        __read_only image2d_t bias,
                        __write_only image2d_t output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch
                        ,__private const int blockDim
                        ,__private const int srcChannel
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); //cout/4, b
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    FLOAT4 out = RI_F(bias, SAMPLER, (int2)(pos.x, 0));

#if QUANT_BIT == 8
    int weight_offset = pos.x * 16;
    int weight_oc_offset = dstChannelC4 * 16;
#else 
    int weight_offset = pos.x * 8;
    int weight_oc_offset = dstChannelC4 * 8;
#endif

    for (int k = 0; k < srcChannelC4; ++k) {
        int kindex = (k * 4) / blockDim * dstChannelC4 * 8;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(pos.x, dequantScaleOffset + kindex));
        COMPUTE_FLOAT16 scale = (COMPUTE_FLOAT16)(ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6,
        ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6,
        ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6,
        ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6);
        COMPUTE_FLOAT16 offset = (COMPUTE_FLOAT16)(ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7,
        ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7,
        ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7,
        ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7);
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(k, pos.y));
#if QUANT_BIT == 8
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * scale + offset;
#else
        uchar8 charWeightsInt4 = vload8(0, weight + weight_offset + k * weight_oc_offset);
        char16 charWeights = 0;
        charWeights.s0 = (charWeightsInt4.s0 >> 4) - 8;
        charWeights.s1 = (charWeightsInt4.s0 & 15) - 8;
        charWeights.s2 = (charWeightsInt4.s1 >> 4) - 8;
        charWeights.s3 = (charWeightsInt4.s1 & 15) - 8;
        charWeights.s4 = (charWeightsInt4.s2 >> 4) - 8;
        charWeights.s5 = (charWeightsInt4.s2 & 15) - 8;
        charWeights.s6 = (charWeightsInt4.s3 >> 4) - 8;
        charWeights.s7 = (charWeightsInt4.s3 & 15) - 8;
        charWeights.s8 = (charWeightsInt4.s4 >> 4) - 8;
        charWeights.s9 = (charWeightsInt4.s4 & 15) - 8;
        charWeights.sa = (charWeightsInt4.s5 >> 4) - 8;
        charWeights.sb = (charWeightsInt4.s5 & 15) - 8;
        charWeights.sc = (charWeightsInt4.s6 >> 4) - 8;
        charWeights.sd = (charWeightsInt4.s6 & 15) - 8;
        charWeights.se = (charWeightsInt4.s7 >> 4) - 8;
        charWeights.sf = (charWeightsInt4.s7 & 15) - 8;
        FLOAT16 weights = CONVERT_FLOAT16(charWeights) * scale + offset;
#endif
        PADZEROSVEC(k, srcChannel, weights.s0123, weights.s4567, weights.s89ab, weights.scdef);
        
        out = mad((FLOAT4)in.x, (FLOAT4)weights.s0123, out);
        out = mad((FLOAT4)in.y, (FLOAT4)weights.s4567, out);
        out = mad((FLOAT4)in.z, (FLOAT4)weights.s89ab, out);
        out = mad((FLOAT4)in.w, (FLOAT4)weights.scdef, out);
    }
    
#ifdef RELU
    out = fmax(out, (FLOAT4)0);
#endif

#ifdef RELU6
    out = clamp(out, (FLOAT4)0, (FLOAT4)6);
#endif

    WI_F(output, (int2)(pos.x, pos.y), out);
}

__kernel void gemm_conv_b2(GLOBAL_SIZE_DIM2
                        __read_only image2d_t input,
#if QUANT_BIT == 8
                        __global const char *weight,
                        __global const float *dequantScaleOffset,
#else
                        __global const uchar *weight,
                        __global const float *dequantScaleOffset,
#endif
                        __read_only image2d_t bias,
                        __write_only image2d_t output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch
                        ,__private const int blockDim
                        ,__private const int srcChannel
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); //cout/4, b
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    int pos_x = pos.x << 2;
    int pos_y = pos.y << 1;

    FLOAT4 bias0 = RI_F(bias, SAMPLER, (int2)(pos.x, 0));
    FLOAT4 out0 = bias0, out1 = bias0;
    
#if QUANT_BIT == 8
    int weight_offset = pos.x * 16;
    int weight_oc_offset = dstChannelC4 * 16;
#else
    int weight_offset = pos.x * 8;
    int weight_oc_offset = dstChannelC4 * 8;
#endif

    for (int k = 0; k < srcChannelC4; ++k) {
        int kindex = (k * 4) / blockDim * dstChannelC4 * 8;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(pos.x, dequantScaleOffset + kindex));
        COMPUTE_FLOAT16 scale = (COMPUTE_FLOAT16)(ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6,
        ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6,
        ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6,
        ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6);
        COMPUTE_FLOAT16 offset = (COMPUTE_FLOAT16)(ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7,
        ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7,
        ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7,
        ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7);
        FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(k, pos_y));
        FLOAT4 in1 = RI_F(input, SAMPLER, (int2)(k, pos_y + 1));
#if QUANT_BIT == 8
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * scale + offset;
#else
        uchar8 charWeightsInt4 = vload8(0, weight + weight_offset + k * weight_oc_offset);
        char16 charWeights = 0;
        charWeights.s0 = (charWeightsInt4.s0 >> 4) - 8;
        charWeights.s1 = (charWeightsInt4.s0 & 15) - 8;
        charWeights.s2 = (charWeightsInt4.s1 >> 4) - 8;
        charWeights.s3 = (charWeightsInt4.s1 & 15) - 8;
        charWeights.s4 = (charWeightsInt4.s2 >> 4) - 8;
        charWeights.s5 = (charWeightsInt4.s2 & 15) - 8;
        charWeights.s6 = (charWeightsInt4.s3 >> 4) - 8;
        charWeights.s7 = (charWeightsInt4.s3 & 15) - 8;
        charWeights.s8 = (charWeightsInt4.s4 >> 4) - 8;
        charWeights.s9 = (charWeightsInt4.s4 & 15) - 8;
        charWeights.sa = (charWeightsInt4.s5 >> 4) - 8;
        charWeights.sb = (charWeightsInt4.s5 & 15) - 8;
        charWeights.sc = (charWeightsInt4.s6 >> 4) - 8;
        charWeights.sd = (charWeightsInt4.s6 & 15) - 8;
        charWeights.se = (charWeightsInt4.s7 >> 4) - 8;
        charWeights.sf = (charWeightsInt4.s7 & 15) - 8;
        FLOAT16 weights = CONVERT_FLOAT16(charWeights) * scale + offset;
#endif
        PADZEROSVEC(k, srcChannel, weights.s0123, weights.s4567, weights.s89ab, weights.scdef);
        
        out0 = mad((FLOAT4)in0.x, (FLOAT4)weights.s0123, out0);
        out0 = mad((FLOAT4)in0.y, (FLOAT4)weights.s4567, out0);
        out0 = mad((FLOAT4)in0.z, (FLOAT4)weights.s89ab, out0);
        out0 = mad((FLOAT4)in0.w, (FLOAT4)weights.scdef, out0);
        
        out1 = mad((FLOAT4)in1.x, (FLOAT4)weights.s0123, out1);
        out1 = mad((FLOAT4)in1.y, (FLOAT4)weights.s4567, out1);
        out1 = mad((FLOAT4)in1.z, (FLOAT4)weights.s89ab, out1);
        out1 = mad((FLOAT4)in1.w, (FLOAT4)weights.scdef, out1);
    }
#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
#endif

    WI_F(output, (int2)(pos.x, pos_y), out0);
    if(pos_y + 1 < batch)
        WI_F(output, (int2)(pos.x, pos_y + 1), out1);
}
