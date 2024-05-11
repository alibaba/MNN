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

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gemm_conv_c4_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
#endif
                        __global const float *dequantScale,
                        __global const float *dequantOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch,
                        __private const int height,
                        __private const int width) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK2
    const int out_b_idx = (out_b_h_idx / height) << 1;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;

    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT sum = 0;
    COMPUTE_FLOAT4 out = 0;
#ifdef BACTH_BLOCK2
    COMPUTE_FLOAT sum1 = 0;
    COMPUTE_FLOAT4 out1 = 0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch = out_b_idx + 1 < batch;
#endif
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + out_c_idx) * height + out_h_idx) * width + out_w_idx) * 4;
#ifndef WIDTH_HEIGHT_1
    int wh = width * height * 4;
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 4 * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 4 * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif
    
    const COMPUTE_FLOAT4 Scale = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, dequantScale));
    const COMPUTE_FLOAT4 Offset = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, dequantOffset));
    
#ifdef INPUT_CHANNEL_LEAVE
    const int srcChannelC16 = (srcChannelC4 + 3) >> 2;
    for (int k = 0; k < srcChannelC16 - 1; ++k) {
#else
    for (int k = 0; k < srcChannelC4/4; ++k) {
#endif
#ifdef WIDTH_HEIGHT_1
        COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset));
#else
        int k4 = k << 2;
        COMPUTE_FLOAT16 in;
        in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
        in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
        in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));
#endif
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
#ifdef WIDTH_HEIGHT_1
            in1 = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset1));
#else
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
            in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
#endif
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16));
        COMPUTE_FLOAT16 weights2 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 32));
        COMPUTE_FLOAT16 weights3 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 48));
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt40 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        uchar16 charWeightsInt41 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
        COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
        {
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
            weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
        }
        
        {
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
            weights2 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            weights3 = CONVERT_COMPUTE_FLOAT16(charWeights1);
        }
#endif
        DOT16X16(in, weights0, out.s0);
        DOT16X16(in, weights1, out.s1);
        DOT16X16(in, weights2, out.s2);
        DOT16X16(in, weights3, out.s3);
#ifdef BACTH_BLOCK2
        DOT16X16(in1, weights0, out1.s0);
        DOT16X16(in1, weights1, out1.s1);
        DOT16X16(in1, weights2, out1.s2);
        DOT16X16(in1, weights3, out1.s3);
#endif
    }
#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC16 - 1;
        int k4 = k * 4;
        COMPUTE_FLOAT16 in;
        in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
        in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
        in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
        
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
            in1.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16));
        COMPUTE_FLOAT16 weights2 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 32));
        COMPUTE_FLOAT16 weights3 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 48));
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt40 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        uchar16 charWeightsInt41 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
        COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
        {
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
            weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
        }
        
        {
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
            weights2 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            weights3 = CONVERT_COMPUTE_FLOAT16(charWeights1);
        }
#endif
        DOT16X16(in, weights0, out.s0);
        DOT16X16(in, weights1, out.s1);
        DOT16X16(in, weights2, out.s2);
        DOT16X16(in, weights3, out.s3);
#ifdef BACTH_BLOCK2
        DOT16X16(in1, weights0, out1.s0);
        DOT16X16(in1, weights1, out1.s1);
        DOT16X16(in1, weights2, out1.s2);
        DOT16X16(in1, weights3, out1.s3);
#endif
    }
#endif
    
    out = bias0 + mad(out, Scale, sum * Offset);
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    vstore4(CONVERT_FLOAT4(out), 0, output+out_offset);
#ifdef BACTH_BLOCK2
    if(isValidBatch){
        out_offset += dstChannelC4 * height * width * 4;
        out1 = bias0 + mad(out1, Scale, sum1 * Offset);
#ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif
    
        vstore4(CONVERT_FLOAT4(out1), 0, output+out_offset);
    }
#endif
}

__kernel void gemm_conv_c2_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
#endif
                        __global const float *dequantScale,
                        __global const float *dequantOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch,
                        __private const int height,
                        __private const int width) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK2
    const int out_b_idx = (out_b_h_idx / height) << 1;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;

    COMPUTE_FLOAT2 bias0 = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, bias));
    COMPUTE_FLOAT sum = 0;
    COMPUTE_FLOAT2 out = 0;
#ifdef BACTH_BLOCK2
    COMPUTE_FLOAT sum1 = 0;
    COMPUTE_FLOAT2 out1 = 0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch = out_b_idx + 1 < batch;
#endif
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + (out_c_idx * 2) / 4) * height + out_h_idx) * width + out_w_idx) * 4 + ((out_c_idx * 2)%4);
#ifndef WIDTH_HEIGHT_1
    int wh = width * height * 4;
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 2 * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 2 * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif

    const COMPUTE_FLOAT2 Scale = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, dequantScale));
    const COMPUTE_FLOAT2 Offset = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, dequantOffset));

#ifdef INPUT_CHANNEL_LEAVE
    const int srcChannelC16 = (srcChannelC4 + 3) >> 2;
    for (int k = 0; k < srcChannelC16 - 1; ++k) {
#else
    for (int k = 0; k < srcChannelC4/4; ++k) {
#endif
#ifdef WIDTH_HEIGHT_1
        COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset));
#else
        COMPUTE_FLOAT16 in;
        int k4 = k << 2;
        in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
        in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
        in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));
#endif
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
#ifdef WIDTH_HEIGHT_1
            in1 = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset1));
#else
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
            in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
#endif
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16));
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt4 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        char16 charWeights0 = 0;
        char16 charWeights1 = 0;
        UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
        COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
        COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
#endif
        DOT16X16(in, weights0, out.s0);
        DOT16X16(in, weights1, out.s1);
#ifdef BACTH_BLOCK2
        DOT16X16(in1, weights0, out1.s0);
        DOT16X16(in1, weights1, out1.s1);
#endif
    }

#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC16 - 1;
        COMPUTE_FLOAT16 in = 0;
        int k4 = k * 4;
        in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
        in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
        in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
            in1.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16));
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt4 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        char16 charWeights0 = 0;
        char16 charWeights1 = 0;
        UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
        COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
        COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
#endif
        DOT16X16(in, weights0, out.s0);
        DOT16X16(in, weights1, out.s1);
#ifdef BACTH_BLOCK2
        DOT16X16(in1, weights0, out1.s0);
        DOT16X16(in1, weights1, out1.s1);
#endif
    }
#endif
    
    out = bias0 + mad(out, Scale, sum * Offset);
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif

    vstore2(CONVERT_FLOAT2(out), 0, output+out_offset);
#ifdef BACTH_BLOCK2
    if(isValidBatch){
        out_offset += dstChannelC4 * height * width * 4;
        out1 = bias0 + mad(out1, Scale, sum1 * Offset);
#ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
        
        vstore2(CONVERT_FLOAT2(out1), 0, output+out_offset);
    }
#endif
}

__kernel void gemm_conv_c1_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
#endif
                        __global const float *dequantScale,
                        __global const float *dequantOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch,
                        __private const int height,
                        __private const int width) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK2
    const int out_b_idx = (out_b_h_idx / height) << 1;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;

    COMPUTE_FLOAT bias0 = bias[out_c_idx];
    COMPUTE_FLOAT sum = 0;
    COMPUTE_FLOAT out = 0;
    
#ifdef BACTH_BLOCK2
    COMPUTE_FLOAT sum1 = 0;
    COMPUTE_FLOAT out1 = 0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch = out_b_idx + 1 < batch;
#endif
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + out_c_idx/4) * height + out_h_idx) * width + out_w_idx) * 4 + (out_c_idx%4);
#ifndef WIDTH_HEIGHT_1
    int wh = width * height * 4;
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif
    
    const COMPUTE_FLOAT Scale = dequantScale[out_c_idx];
    const COMPUTE_FLOAT Offset = dequantOffset[out_c_idx];
#ifdef INPUT_CHANNEL_LEAVE
    const int srcChannelC16 = (srcChannelC4 + 3) >> 2;
    for (int k = 0; k < srcChannelC16 - 1; ++k) {
#else
    for (int k = 0; k < srcChannelC4/4; ++k) {
#endif
#ifdef WIDTH_HEIGHT_1
        COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset));
#else
        COMPUTE_FLOAT16 in;
        int k4 = k << 2;
        in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
        in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
        in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));
#endif
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
#ifdef WIDTH_HEIGHT_1
            in1 = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset1));
#else
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
            in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
#endif
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        COMPUTE_FLOAT16 weights = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar8 charWeightsInt4 = vload8(0, weight + weight_offset + k * weight_oc_offset);
        char16 charWeights = 0;
        UCHAR8_TO_CHAR16(charWeights, charWeightsInt4);
        COMPUTE_FLOAT16 weights = CONVERT_COMPUTE_FLOAT16(charWeights);
#endif
        DOT16X16(in, weights, out);
#ifdef BACTH_BLOCK2
        DOT16X16(in1, weights, out1);
#endif
    }
#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC16 - 1;
        COMPUTE_FLOAT16 in = 0;
        int k4 = k * 4;
        in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
        in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
        in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
            in1.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        COMPUTE_FLOAT16 weights = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar8 charWeightsInt4 = vload8(0, weight + weight_offset + k * weight_oc_offset);
        char16 charWeights = 0;
        UCHAR8_TO_CHAR16(charWeights, charWeightsInt4);
        COMPUTE_FLOAT16 weights = CONVERT_COMPUTE_FLOAT16(charWeights);
#endif
        DOT16X16(in, weights, out);
#ifdef BACTH_BLOCK2
        DOT16X16(in1, weights, out1);
#endif
    }
#endif
    
    out = bias0 + mad(out, Scale, sum * Offset);
#ifdef RELU
    out = fmax(out, 0);
#endif

#ifdef RELU6
    out = clamp(out, 0, 6);
#endif
    output[out_offset] = out;
#ifdef BACTH_BLOCK2
    if(isValidBatch){
        out_offset += dstChannelC4 * height * width * 4;
        out1 = bias0 + mad(out1, Scale, sum1 * Offset);
#ifdef RELU
        out1 = fmax(out1, 0);
#endif

#ifdef RELU6
        out1 = clamp(out1, 0, 6);
#endif
            
        output[out_offset] = out1;
    }
#endif
}
    
__kernel void gemm_conv_c2_image(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
                        __read_only image2d_t weight,
                        __global const float *dequantScale,
                        __global const float *dequantOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch,
                        __private const int height,
                        __private const int width) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h
    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = (out_c_w_idx / width) << 1;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK2
    const int out_b_idx = (out_b_h_idx / height) << 1;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;
        
    COMPUTE_FLOAT2 bias0 = CONVERT_COMPUTE_FLOAT2(vload2(0, bias + out_c_idx));
    COMPUTE_FLOAT2 out = 0;
    COMPUTE_FLOAT sum = 0;
    
#ifdef BACTH_BLOCK2
    COMPUTE_FLOAT sum1 = 0;
    COMPUTE_FLOAT2 out1 = 0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch = out_b_idx + 1 < batch;
#endif

    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + out_c_idx/4) * height + out_h_idx) * width + out_w_idx) * 4 + (out_c_idx % 4);
#ifndef WIDTH_HEIGHT_1
    int wh = width * height * 4;
#endif

    const COMPUTE_FLOAT2 Scale = CONVERT_COMPUTE_FLOAT2(vload2(0, dequantScale + out_c_idx));
    const COMPUTE_FLOAT2 Offset = CONVERT_COMPUTE_FLOAT2(vload2(0, dequantOffset + out_c_idx));

#if (defined USE_LOW_BIT_WEIGHT_INT8)
#ifdef INPUT_CHANNEL_LEAVE
    const int srcChannelC16 = (srcChannelC4 + 3) >> 2;
    for (int k = 0; k < srcChannelC16-1; k++) {
#else
    for (int k = 0; k < srcChannelC4/4; k++) {
#endif
#ifdef WIDTH_HEIGHT_1
        COMPUTE_FLOAT16 in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 16));
#else
        COMPUTE_FLOAT16 in0 = 0;
        int k4 = k * 4;
        in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
        in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
        in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));

#endif
        sum += in0.s0 + in0.s1 + in0.s2 + in0.s3 + in0.s4 + in0.s5 + in0.s6 + in0.s7 + in0.s8 + in0.s9 + in0.sa + in0.sb + in0.sc + in0.sd + in0.se + in0.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
#ifdef WIDTH_HEIGHT_1
            in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 16));
#else
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
            in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
#endif
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif

        {
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagef(weight, SAMPLER, (int2)(out_c_idx, k))));
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagef(weight, SAMPLER, (int2)(out_c_idx + 1, k))));
            DOT16X16(in0, weights0, out.s0);
            DOT16X16(in0, weights1, out.s1);
#ifdef BACTH_BLOCK2
            DOT16X16(in1, weights0, out1.s0);
            DOT16X16(in1, weights1, out1.s1);
#endif
        }
    }
#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC16 - 1;
        COMPUTE_FLOAT16 in0 = 0;
        int k4 = k * 4;
        in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in0.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
        in0.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
        in0.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
        
        sum += in0.s0 + in0.s1 + in0.s2 + in0.s3 + in0.s4 + in0.s5 + in0.s6 + in0.s7 + in0.s8 + in0.s9 + in0.sa + in0.sb + in0.sc + in0.sd + in0.se + in0.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
            in1.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif

        {
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagef(weight, SAMPLER, (int2)(out_c_idx, k))));
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagef(weight, SAMPLER, (int2)(out_c_idx + 1, k))));
            DOT16X16(in0, weights0, out.s0);
            DOT16X16(in0, weights1, out.s1);
#ifdef BACTH_BLOCK2
            DOT16X16(in1, weights0, out1.s0);
            DOT16X16(in1, weights1, out1.s1);
#endif
        }
    }
#endif
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
#ifdef INPUT_CHANNEL_LEAVE
    const int srcChannelC32 = (srcChannelC4 + 7) >> 3;
    for (int k = 0; k < srcChannelC32-1; k++) {
#else
    for (int k = 0; k < srcChannelC4/8; k++) {
#endif
#ifdef WIDTH_HEIGHT_1
        COMPUTE_FLOAT16 in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 32));
        COMPUTE_FLOAT16 in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 32 + 16));
#else
        COMPUTE_FLOAT16 in0 = 0;
        COMPUTE_FLOAT16 in1 = 0;
        int k8 = k * 8;
        in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k8 * wh));
        in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 1) * wh));
        in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 2) * wh));
        in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 3) * wh));

        in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 4) * wh));
        in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 5) * wh));
        in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 6) * wh));
        in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 7) * wh));

#endif
        sum += in0.s0 + in0.s1 + in0.s2 + in0.s3 + in0.s4 + in0.s5 + in0.s6 + in0.s7 + in0.s8 + in0.s9 + in0.sa + in0.sb + in0.sc + in0.sd + in0.se + in0.sf;
        sum += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in2 = 0;
        COMPUTE_FLOAT16 in3 = 0;
        if(isValidBatch){
#ifdef WIDTH_HEIGHT_1
            in2 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 32));
            in3 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 32 + 16));
#else
            in2.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k8 * wh));
            in2.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 1) * wh));
            in2.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 2) * wh));
            in2.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 3) * wh));

            in3.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 4) * wh));
            in3.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 5) * wh));
            in3.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 6) * wh));
            in3.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 7) * wh));
#endif
            sum1 += in2.s0 + in2.s1 + in2.s2 + in2.s3 + in2.s4 + in2.s5 + in2.s6 + in2.s7 + in2.s8 + in2.s9 + in2.sa + in2.sb + in2.sc + in2.sd + in2.se + in2.sf;
            sum1 += in3.s0 + in3.s1 + in3.s2 + in3.s3 + in3.s4 + in3.s5 + in3.s6 + in3.s7 + in3.s8 + in3.s9 + in3.sa + in3.sb + in3.sc + in3.sd + in3.se + in3.sf;
        }
#endif

        {
            uchar16 charWeightsInt4 = as_uchar16(read_imagef(weight, SAMPLER, (int2)(out_c_idx, k)));
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
            DOT16X16(in0, weights0, out.s0);
            DOT16X16(in1, weights1, out.s0);
#ifdef BACTH_BLOCK2
            DOT16X16(in2, weights0, out1.s0);
            DOT16X16(in3, weights1, out1.s0);
#endif
        }
        
        {
            uchar16 charWeightsInt4 = as_uchar16(read_imagef(weight, SAMPLER, (int2)(out_c_idx + 1, k)));
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
            DOT16X16(in0, weights0, out.s1);
            DOT16X16(in1, weights1, out.s1);
#ifdef BACTH_BLOCK2
            DOT16X16(in2, weights0, out1.s1);
            DOT16X16(in3, weights1, out1.s1);
#endif
        }
    }
#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC32 - 1;
        COMPUTE_FLOAT16 in0 = 0;
        COMPUTE_FLOAT16 in1 = 0;
        int k8 = k * 8;
        in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k8 * wh));
        in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 1) * wh) : (FLOAT4)0);
        in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 2) * wh) : (FLOAT4)0);
        in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 3) * wh) : (FLOAT4)0);
        
        in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 4) * wh) : (FLOAT4)0);
        in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 5) * wh) : (FLOAT4)0);
        in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 6) * wh) : (FLOAT4)0);
        in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 7) * wh) : (FLOAT4)0);
        sum += in0.s0 + in0.s1 + in0.s2 + in0.s3 + in0.s4 + in0.s5 + in0.s6 + in0.s7 + in0.s8 + in0.s9 + in0.sa + in0.sb + in0.sc + in0.sd + in0.se + in0.sf;
        sum += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in2 = 0;
        COMPUTE_FLOAT16 in3 = 0;
        if(isValidBatch){
            in2.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k8 * wh));
            in2.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < vload4(0, input + input_offset1 + (k8 + 1) * wh) : (FLOAT4)0);
            in2.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < vload4(0, input + input_offset1 + (k8 + 2) * wh) : (FLOAT4)0);
            in2.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < vload4(0, input + input_offset1 + (k8 + 3) * wh) : (FLOAT4)0);
            
            in3.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 4) * wh) : (FLOAT4)0);
            in3.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 5) * wh) : (FLOAT4)0);
            in3.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 6) * wh) : (FLOAT4)0);
            in3.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 7) * wh) : (FLOAT4)0);
            sum1 += in2.s0 + in2.s1 + in2.s2 + in2.s3 + in2.s4 + in2.s5 + in2.s6 + in2.s7 + in2.s8 + in2.s9 + in2.sa + in2.sb + in2.sc + in2.sd + in2.se + in2.sf;
            sum1 += in3.s0 + in3.s1 + in3.s2 + in3.s3 + in3.s4 + in3.s5 + in3.s6 + in3.s7 + in3.s8 + in3.s9 + in3.sa + in3.sb + in3.sc + in3.sd + in3.se + in3.sf;
        }
#endif

        {
            uchar16 charWeightsInt4 = as_uchar16(read_imagef(weight, SAMPLER, (int2)(out_c_idx, k)));
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
            DOT16X16(in0, weights0, out.s0);
            DOT16X16(in1, weights1, out.s0);
#ifdef BACTH_BLOCK2
            DOT16X16(in2, weights0, out1.s0);
            DOT16X16(in3, weights1, out1.s0);
#endif
        }
        
        {
            uchar16 charWeightsInt4 = as_uchar16(read_imagef(weight, SAMPLER, (int2)(out_c_idx + 1, k)));
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
            DOT16X16(in0, weights0, out.s1);
            DOT16X16(in1, weights1, out.s1);
#ifdef BACTH_BLOCK2
            DOT16X16(in2, weights0, out1.s1);
            DOT16X16(in3, weights1, out1.s1);
#endif
        }
    }
#endif
#endif //USE_LOW_BIT_WEIGHT_INT4

    out = bias0 + mad(out, Scale, sum * Offset);
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT2)0);
#endif
#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
    vstore2(CONVERT_FLOAT2(out), 0, output + out_offset);
#ifdef BACTH_BLOCK2
    if(isValidBatch){
        out_offset += dstChannelC4 * height * width * 4;
        out1 = bias0 + mad(out1, Scale, sum1 * Offset);
#ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
            
        vstore2(CONVERT_FLOAT2(out1), 0, output+out_offset);
    }
#endif
}
__kernel void gemm_conv_c1_image(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
                        __read_only image2d_t weight,
                        __global const float *dequantScale,
                        __global const float *dequantOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch,
                        __private const int height,
                        __private const int width) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h
    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK2
    const int out_b_idx = (out_b_h_idx / height) << 1;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;
    
    COMPUTE_FLOAT bias0 = bias[out_c_idx];
    COMPUTE_FLOAT out = 0;
    COMPUTE_FLOAT sum = 0;
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + out_c_idx/4)* height + out_h_idx) * width + out_w_idx) * 4 + (out_c_idx%4);
#ifndef WIDTH_HEIGHT_1
    int wh = width * height * 4;
#endif
#ifdef BACTH_BLOCK2
    COMPUTE_FLOAT sum1 = 0;
    COMPUTE_FLOAT out1 = 0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch = out_b_idx + 1 < batch;
#endif
    const COMPUTE_FLOAT Scale = dequantScale[out_c_idx];
    const COMPUTE_FLOAT Offset = dequantOffset[out_c_idx];
    
#if (defined USE_LOW_BIT_WEIGHT_INT8)
#ifdef INPUT_CHANNEL_LEAVE
    const int srcChannelC16 = (srcChannelC4 + 3) >> 2;
    for (int k = 0; k < srcChannelC16-1; k++) {
#else
    for (int k = 0; k < srcChannelC4/4; k++) {
#endif
#ifdef WIDTH_HEIGHT_1
        COMPUTE_FLOAT16 in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 16));
#else
        COMPUTE_FLOAT16 in0 = 0;
        int k4 = k * 4;
        in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
        in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
        in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));

#endif
        sum += in0.s0 + in0.s1 + in0.s2 + in0.s3 + in0.s4 + in0.s5 + in0.s6 + in0.s7 + in0.s8 + in0.s9 + in0.sa + in0.sb + in0.sc + in0.sd + in0.se + in0.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
#ifdef WIDTH_HEIGHT_1
            in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 16));
#else
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
            in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
#endif
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif

        {
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagef(weight, SAMPLER, (int2)(out_c_idx, k))));
            DOT16X16(in0, weights0, out);
#ifdef BACTH_BLOCK2
            DOT16X16(in1, weights0, out1);
#endif
        }
    }
#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC16 - 1;
        COMPUTE_FLOAT16 in0 = 0;
        int k4 = k * 4;
        in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
        in0.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
        in0.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
        in0.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
        
        sum += in0.s0 + in0.s1 + in0.s2 + in0.s3 + in0.s4 + in0.s5 + in0.s6 + in0.s7 + in0.s8 + in0.s9 + in0.sa + in0.sb + in0.sc + in0.sd + in0.se + in0.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in1 = 0;
        if(isValidBatch){
            in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
            in1.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
            in1.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
            in1.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
            sum1 += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
        }
#endif

        {
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagef(weight, SAMPLER, (int2)(out_c_idx, k))));
            DOT16X16(in0, weights0, out);
#ifdef BACTH_BLOCK2
            DOT16X16(in1, weights0, out1);
#endif
        }
    }
#endif
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
#ifdef INPUT_CHANNEL_LEAVE
    const int srcChannelC32 = (srcChannelC4 + 7) >> 3;
    for (int k = 0; k < srcChannelC32-1; k++) {
#else
    for (int k = 0; k < srcChannelC4/8; k++) {
#endif
#ifdef WIDTH_HEIGHT_1
        COMPUTE_FLOAT16 in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 32));
        COMPUTE_FLOAT16 in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 32 + 16));
#else
        COMPUTE_FLOAT16 in0 = 0;
        COMPUTE_FLOAT16 in1 = 0;
        int k8 = k * 8;
        in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k8 * wh));
        in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 1) * wh));
        in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 2) * wh));
        in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 3) * wh));

        in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 4) * wh));
        in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 5) * wh));
        in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 6) * wh));
        in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 7) * wh));

#endif
        sum += in0.s0 + in0.s1 + in0.s2 + in0.s3 + in0.s4 + in0.s5 + in0.s6 + in0.s7 + in0.s8 + in0.s9 + in0.sa + in0.sb + in0.sc + in0.sd + in0.se + in0.sf;
        sum += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in2 = 0;
        COMPUTE_FLOAT16 in3 = 0;
        if(isValidBatch){
#ifdef WIDTH_HEIGHT_1
            in2 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 32));
            in3 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 32 + 16));
#else
            in2.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k8 * wh));
            in2.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 1) * wh));
            in2.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 2) * wh));
            in2.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 3) * wh));

            in3.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 4) * wh));
            in3.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 5) * wh));
            in3.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 6) * wh));
            in3.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 7) * wh));
#endif
            sum1 += in2.s0 + in2.s1 + in2.s2 + in2.s3 + in2.s4 + in2.s5 + in2.s6 + in2.s7 + in2.s8 + in2.s9 + in2.sa + in2.sb + in2.sc + in2.sd + in2.se + in2.sf;
            sum1 += in3.s0 + in3.s1 + in3.s2 + in3.s3 + in3.s4 + in3.s5 + in3.s6 + in3.s7 + in3.s8 + in3.s9 + in3.sa + in3.sb + in3.sc + in3.sd + in3.se + in3.sf;
        }
#endif
        {
            uchar16 charWeightsInt4 = as_uchar16(read_imagef(weight, SAMPLER, (int2)(out_c_idx, k)));
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
            DOT16X16(in0, weights0, out);
            DOT16X16(in1, weights1, out);
#ifdef BACTH_BLOCK2
            DOT16X16(in2, weights0, out1);
            DOT16X16(in3, weights1, out1);
#endif
        }
    }

#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC32 - 1;
        COMPUTE_FLOAT16 in0 = 0;
        COMPUTE_FLOAT16 in1 = 0;
        int k8 = k * 8;
        in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k8 * wh));
        in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 1) * wh) : (FLOAT4)0);
        in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 2) * wh) : (FLOAT4)0);
        in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 3) * wh) : (FLOAT4)0);
        
        in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 4) * wh) : (FLOAT4)0);
        in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 5) * wh) : (FLOAT4)0);
        in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 6) * wh) : (FLOAT4)0);
        in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 7) * wh) : (FLOAT4)0);
        sum += in0.s0 + in0.s1 + in0.s2 + in0.s3 + in0.s4 + in0.s5 + in0.s6 + in0.s7 + in0.s8 + in0.s9 + in0.sa + in0.sb + in0.sc + in0.sd + in0.se + in0.sf;
        sum += in1.s0 + in1.s1 + in1.s2 + in1.s3 + in1.s4 + in1.s5 + in1.s6 + in1.s7 + in1.s8 + in1.s9 + in1.sa + in1.sb + in1.sc + in1.sd + in1.se + in1.sf;
#ifdef BACTH_BLOCK2
        COMPUTE_FLOAT16 in2 = 0;
        COMPUTE_FLOAT16 in3 = 0;
        if(isValidBatch){
            in2.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k8 * wh));
            in2.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < vload4(0, input + input_offset1 + (k8 + 1) * wh) : (FLOAT4)0);
            in2.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < vload4(0, input + input_offset1 + (k8 + 2) * wh) : (FLOAT4)0);
            in2.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < vload4(0, input + input_offset1 + (k8 + 3) * wh) : (FLOAT4)0);
            
            in3.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 4) * wh) : (FLOAT4)0);
            in3.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 5) * wh) : (FLOAT4)0);
            in3.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 6) * wh) : (FLOAT4)0);
            in3.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 7) * wh) : (FLOAT4)0);
            sum1 += in2.s0 + in2.s1 + in2.s2 + in2.s3 + in2.s4 + in2.s5 + in2.s6 + in2.s7 + in2.s8 + in2.s9 + in2.sa + in2.sb + in2.sc + in2.sd + in2.se + in2.sf;
            sum1 += in3.s0 + in3.s1 + in3.s2 + in3.s3 + in3.s4 + in3.s5 + in3.s6 + in3.s7 + in3.s8 + in3.s9 + in3.sa + in3.sb + in3.sc + in3.sd + in3.se + in3.sf;
        }
#endif

        {
            uchar16 charWeightsInt4 = as_uchar16(read_imagef(weight, SAMPLER, (int2)(out_c_idx, k)));
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0);
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1);
            DOT16X16(in0, weights0, out);
            DOT16X16(in1, weights1, out);
#ifdef BACTH_BLOCK2
            DOT16X16(in2, weights0, out1);
            DOT16X16(in3, weights1, out1);
#endif
        }
    }
#endif
#endif //USE_LOW_BIT_WEIGHT_INT4

    out = bias0 + mad(out, Scale, sum * Offset);
#ifdef RELU
    out = fmax(out, 0);
#endif
#ifdef RELU6
    out = clamp(out, 0, 6);
#endif
    output[out_offset] = out;
#ifdef BACTH_BLOCK2
    if(isValidBatch){
        out_offset += dstChannelC4 * height * width * 4;
        out1 = bias0 + mad(out1, Scale, sum1 * Offset);
#ifdef RELU
        out1 = fmax(out1, 0);
#endif

#ifdef RELU6
        out1 = clamp(out1, 0, 6);
#endif
                
        output[out_offset] = out1;
    }
#endif
}
        
