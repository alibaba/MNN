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
            ptr[15 - remain] = 0; \
        }  \
    }
#else
    #define PADZEROS(k, channel, data)
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gemm_conv_c4_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
#endif
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int batch,
                        __private const int height,
                        __private const int width,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK4
    const int out_b_idx = (out_b_h_idx / height) << 2;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;

    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out = bias0;
#ifdef BACTH_BLOCK4
    COMPUTE_FLOAT4 out1 = bias0, out2 = bias0, out3 = bias0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset2 = (((out_b_idx + 2) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset3 = (((out_b_idx + 3) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch1 = out_b_idx + 1 < batch;
    bool isValidBatch2 = out_b_idx + 2 < batch;
    bool isValidBatch3 = out_b_idx + 3 < batch;
#endif
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + out_c_idx) * height + out_h_idx) * width + out_w_idx) * 4;
    int wh = width * height * 4;
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 4 * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 4 * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif

    const int loop = (blockDim + 15) / 16;
#ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif
    
    for (int i = 0; i < blockNum; ++i){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; ++j) {
            int k = i * loop + j;
            #ifndef WIDTH_HEIGHT_1
            int k4 = k << 2;
            #endif
            COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
            #if (defined USE_LOW_BIT_WEIGHT_INT8)
            weights0 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * ScaleOffset.s0 + ScaleOffset.s1;
            weights1 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16)) * ScaleOffset.s2 + ScaleOffset.s3;
            weights2 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 32)) * ScaleOffset.s4 + ScaleOffset.s5;
            weights3 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 48)) * ScaleOffset.s6 + ScaleOffset.s7;
            #elif (defined USE_LOW_BIT_WEIGHT_INT4)
            {
                uchar16 charWeightsInt40 = vload16(0, weight + weight_offset + k * weight_oc_offset);
                uchar16 charWeightsInt41 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
                {
                    char16 charWeights0 = 0;
                    char16 charWeights1 = 0;
                    UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                    weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                    weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
                    UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
                    weights2 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s4 + ScaleOffset.s5;
                    weights3 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s6 + ScaleOffset.s7;
                }
            }
            #endif
            {
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out.s0);
                DOT16X16(in, weights1, out.s1);
                DOT16X16(in, weights2, out.s2);
                DOT16X16(in, weights3, out.s3);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset1));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out1.s0);
                DOT16X16(in, weights1, out1.s1);
                DOT16X16(in, weights2, out1.s2);
                DOT16X16(in, weights3, out1.s3);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset2));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out2.s0);
                DOT16X16(in, weights1, out2.s1);
                DOT16X16(in, weights2, out2.s2);
                DOT16X16(in, weights3, out2.s3);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset3));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out3.s0);
                DOT16X16(in, weights1, out3.s1);
                DOT16X16(in, weights2, out3.s2);
                DOT16X16(in, weights3, out3.s3);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k4 = k << 2;
            COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
            #if (defined USE_LOW_BIT_WEIGHT_INT8)
            weights0 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * ScaleOffset.s0 + ScaleOffset.s1;
            weights1 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16)) * ScaleOffset.s2 + ScaleOffset.s3;
            weights2 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 32)) * ScaleOffset.s4 + ScaleOffset.s5;
            weights3 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 48)) * ScaleOffset.s6 + ScaleOffset.s7;
            #elif (defined USE_LOW_BIT_WEIGHT_INT4)
            {
                uchar16 charWeightsInt40 = vload16(0, weight + weight_offset + k * weight_oc_offset);
                uchar16 charWeightsInt41 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
                {
                    char16 charWeights0 = 0;
                    char16 charWeights1 = 0;
                    UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt40);
                    weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                    weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
                    UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt41);
                    weights2 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s4 + ScaleOffset.s5;
                    weights3 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s6 + ScaleOffset.s7;
                }
            }
            #endif
            PADZEROS(k, srcChannel, weights0);
            PADZEROS(k, srcChannel, weights1);
            PADZEROS(k, srcChannel, weights2);
            PADZEROS(k, srcChannel, weights3);
            {
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out.s0);
                DOT16X16(in, weights1, out.s1);
                DOT16X16(in, weights2, out.s2);
                DOT16X16(in, weights3, out.s3);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out1.s0);
                DOT16X16(in, weights1, out1.s1);
                DOT16X16(in, weights2, out1.s2);
                DOT16X16(in, weights3, out1.s3);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out2.s0);
                DOT16X16(in, weights1, out2.s1);
                DOT16X16(in, weights2, out2.s2);
                DOT16X16(in, weights3, out2.s3);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out3.s0);
                DOT16X16(in, weights1, out3.s1);
                DOT16X16(in, weights2, out3.s2);
                DOT16X16(in, weights3, out3.s3);
            }
            #endif
        }
        #endif
    }
    
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    vstore4(CONVERT_FLOAT4(out), 0, output+out_offset);
#ifdef BACTH_BLOCK4
    if(isValidBatch1){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif
    
        vstore4(CONVERT_FLOAT4(out1), 0, output+out_offset);
    }
    if(isValidBatch2){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out2 = fmax(out2, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
        out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif
    
        vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
    }
    if(isValidBatch3){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
        out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif
    
        vstore4(CONVERT_FLOAT4(out3), 0, output+out_offset);
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
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int batch,
                        __private const int height,
                        __private const int width,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK4
    const int out_b_idx = (out_b_h_idx / height) << 2;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;

    COMPUTE_FLOAT2 bias0 = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, bias));
    COMPUTE_FLOAT2 out = bias0;
#ifdef BACTH_BLOCK4
    COMPUTE_FLOAT2 out1 = bias0, out2 = bias0, out3 = bias0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset2 = (((out_b_idx + 2) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset3 = (((out_b_idx + 3) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch1 = out_b_idx + 1 < batch;
    bool isValidBatch2 = out_b_idx + 2 < batch;
    bool isValidBatch3 = out_b_idx + 3 < batch;
#endif
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + (out_c_idx * 2) / 4) * height + out_h_idx) * width + out_w_idx) * 4 + ((out_c_idx * 2)%4);
    int wh = width * height * 4;
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 2 * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 2 * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif

    const int loop = (blockDim + 15) / 16;
#ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif

    for (int i = 0; i < blockNum; ++i){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT4 ScaleOffset = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; ++j) {
            int k = i * loop + j;
            #ifndef WIDTH_HEIGHT_1
            int k4 = k << 2;
            #endif
            COMPUTE_FLOAT16 weights0, weights1;
            #if (defined USE_LOW_BIT_WEIGHT_INT8)
            weights0 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * ScaleOffset.s0 + ScaleOffset.s1;
            weights1 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16)) * ScaleOffset.s2 + ScaleOffset.s3;
            #elif (defined USE_LOW_BIT_WEIGHT_INT4)
            {
                uchar16 charWeightsInt4 = vload16(0, weight + weight_offset + k * weight_oc_offset);
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
            }
            #endif
            {
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out.s0);
                DOT16X16(in, weights1, out.s1);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset1));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out1.s0);
                DOT16X16(in, weights1, out1.s1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset2));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out2.s0);
                DOT16X16(in, weights1, out2.s1);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset3));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out3.s0);
                DOT16X16(in, weights1, out3.s1);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k4 = k << 2;
            COMPUTE_FLOAT16 weights0, weights1;
            #if (defined USE_LOW_BIT_WEIGHT_INT8)
            weights0 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * ScaleOffset.s0 + ScaleOffset.s1;
            weights1 = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16)) * ScaleOffset.s2 + ScaleOffset.s3;
            #elif (defined USE_LOW_BIT_WEIGHT_INT4)
            {
                uchar16 charWeightsInt4 = vload16(0, weight + weight_offset + k * weight_oc_offset);
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
            }
            #endif
            PADZEROS(k, srcChannel, weights0);
            PADZEROS(k, srcChannel, weights1);
            {
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out.s0);
                DOT16X16(in, weights1, out.s1);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out1.s0);
                DOT16X16(in, weights1, out1.s1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out2.s0);
                DOT16X16(in, weights1, out2.s1);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out3.s0);
                DOT16X16(in, weights1, out3.s1);
            }
            #endif
        }
        #endif
    }
    
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif

    vstore2(CONVERT_FLOAT2(out), 0, output+out_offset);
#ifdef BACTH_BLOCK4
    if(isValidBatch1){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
        
        vstore2(CONVERT_FLOAT2(out1), 0, output+out_offset);
    }
    if(isValidBatch2){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out2 = fmax(out2, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
        out2 = clamp(out2, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
        
        vstore2(CONVERT_FLOAT2(out2), 0, output+out_offset);
    }
    if(isValidBatch3){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out3 = fmax(out3, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
        out3 = clamp(out3, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
        
        vstore2(CONVERT_FLOAT2(out3), 0, output+out_offset);
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
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int batch,
                        __private const int height,
                        __private const int width,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK4
    const int out_b_idx = (out_b_h_idx / height) << 2;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;

    COMPUTE_FLOAT bias0 = bias[out_c_idx];
    COMPUTE_FLOAT out = bias0;
    
#ifdef BACTH_BLOCK4
    COMPUTE_FLOAT out1 = bias0, out2 = bias0, out3 = bias0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset2 = (((out_b_idx + 2) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset3 = (((out_b_idx + 3) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch1 = out_b_idx + 1 < batch;
    bool isValidBatch2 = out_b_idx + 2 < batch;
    bool isValidBatch3 = out_b_idx + 3 < batch;
#endif
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + out_c_idx/4) * height + out_h_idx) * width + out_w_idx) * 4 + (out_c_idx%4);
    int wh = width * height * 4;
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif

    const int loop = (blockDim + 15) / 16;
#ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif
    
    for (int i = 0; i < blockNum; ++i){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT2 ScaleOffset = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; ++j) {
            int k = i * loop + j;
            #ifndef WIDTH_HEIGHT_1
            int k4 = k << 2;
            #endif
            COMPUTE_FLOAT16 weights;
            #if (defined USE_LOW_BIT_WEIGHT_INT8)
            weights = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * ScaleOffset.s0 + ScaleOffset.s1;
            #elif (defined USE_LOW_BIT_WEIGHT_INT4)
            {
                uchar8 charWeightsInt4 = vload8(0, weight + weight_offset + k * weight_oc_offset);
                char16 charWeights = 0;
                UCHAR8_TO_CHAR16(charWeights, charWeightsInt4);
                weights = CONVERT_COMPUTE_FLOAT16(charWeights) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            #endif
            {
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights, out);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset1));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights, out1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset2));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights, out2);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(k, input + input_offset3));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights, out3);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k4 = k << 2;
            COMPUTE_FLOAT16 weights;
            #if (defined USE_LOW_BIT_WEIGHT_INT8)
            weights = CONVERT_COMPUTE_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * ScaleOffset.s0 + ScaleOffset.s1;
            #elif (defined USE_LOW_BIT_WEIGHT_INT4)
            {
                uchar8 charWeightsInt4 = vload8(0, weight + weight_offset + k * weight_oc_offset);
                char16 charWeights = 0;
                UCHAR8_TO_CHAR16(charWeights, charWeightsInt4);
                weights = CONVERT_COMPUTE_FLOAT16(charWeights) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            #endif
            PADZEROS(k, srcChannel, weights);
            {
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights, out);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights, out1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights, out2);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights, out3);
            }
            #endif
        }
        #endif
    }
    
#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT)0);
#endif

#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
    output[out_offset] = out;
#ifdef BACTH_BLOCK4
    if(isValidBatch1){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT)0);
#endif

#ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
            
        output[out_offset] = out1;
    }
    if(isValidBatch2){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out2 = fmax(out2, (COMPUTE_FLOAT)0);
#endif

#ifdef RELU6
        out2 = clamp(out2, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
            
        output[out_offset] = out2;
    }
    if(isValidBatch3){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out3 = fmax(out3, (COMPUTE_FLOAT)0);
#endif

#ifdef RELU6
        out3 = clamp(out3, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
            
        output[out_offset] = out3;
    }
#endif
}
__kernel void gemm_conv_c2_image(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
                        __read_only image2d_t weight,
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int batch,
                        __private const int height,
                        __private const int width,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h
    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = (out_c_w_idx / width) << 1;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK4
    const int out_b_idx = (out_b_h_idx / height) << 2;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;
        
    COMPUTE_FLOAT2 bias0 = CONVERT_COMPUTE_FLOAT2(vload2(0, bias + out_c_idx));
    COMPUTE_FLOAT2 out = bias0;
    
#ifdef BACTH_BLOCK4
    COMPUTE_FLOAT2 out1 = bias0, out2 = bias0, out3 = bias0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset2 = (((out_b_idx + 2) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset3 = (((out_b_idx + 3) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch1 = out_b_idx + 1 < batch;
    bool isValidBatch2 = out_b_idx + 2 < batch;
    bool isValidBatch3 = out_b_idx + 3 < batch;
#endif

    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + out_c_idx/4) * height + out_h_idx) * width + out_w_idx) * 4 + (out_c_idx % 4);
    int wh = width * height * 4;

#if (defined USE_LOW_BIT_WEIGHT_INT8)
    const int loop = (blockDim + 15) / 16;
    #ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
    #else
    const int loop_end = loop;
    #endif
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
    const int loop = (blockDim + 31) / 32;
    #ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
    #else
    const int loop_end = loop;
    #endif
#endif

    for (int i = 0; i < blockNum; ++i){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT4 ScaleOffset = CONVERT_COMPUTE_FLOAT4(vload4(0, dequantScaleOffset + out_c_idx * 2 + kindex));
        #if (defined USE_LOW_BIT_WEIGHT_INT8)
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            #ifndef WIDTH_HEIGHT_1
            int k4 = k << 2;
            #endif
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)))) * ScaleOffset.s0 + ScaleOffset.s1;
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 1, k)))) * ScaleOffset.s2 + ScaleOffset.s3;
            {
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 16));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out.s0);
                DOT16X16(in, weights1, out.s1);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 16));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out1.s0);
                DOT16X16(in, weights1, out1.s1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset2 + k * 16));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out2.s0);
                DOT16X16(in, weights1, out2.s1);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset3 + k * 16));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out3.s0);
                DOT16X16(in, weights1, out3.s1);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k4 = k << 2;
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)))) * ScaleOffset.s0 + ScaleOffset.s1;
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 1, k)))) * ScaleOffset.s2 + ScaleOffset.s3;
            PADZEROS(k, srcChannel, weights0);
            PADZEROS(k, srcChannel, weights1);
            {
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out.s0);
                DOT16X16(in, weights1, out.s1);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out1.s0);
                DOT16X16(in, weights1, out1.s1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out2.s0);
                DOT16X16(in, weights1, out2.s1);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out3.s0);
                DOT16X16(in, weights1, out3.s1);
            }
            #endif
        }
        #endif
        #elif (defined USE_LOW_BIT_WEIGHT_INT4)
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            #ifndef WIDTH_HEIGHT_1
            int k8 = k << 3;
            #endif
            COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
            {
                uchar16 charWeightsInt4 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            {
                uchar16 charWeightsInt4 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 1, k)));
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights2 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s2 + ScaleOffset.s3;
                weights3 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
            }
            {
                COMPUTE_FLOAT16 in0, in1;
                #ifdef WIDTH_HEIGHT_1
                in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 32));
                in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 32 + 16));
                #else
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 1) * wh));
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 2) * wh));
                in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 3) * wh));

                in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 4) * wh));
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 5) * wh));
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 6) * wh));
                in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 7) * wh));
                #endif
                
                DOT16X16(in0, weights0, out.s0);
                DOT16X16(in1, weights1, out.s0);
                DOT16X16(in0, weights2, out.s1);
                DOT16X16(in1, weights3, out.s1);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in0, in1;
                #ifdef WIDTH_HEIGHT_1
                in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 32));
                in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 32 + 16));
                #else
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 1) * wh));
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 2) * wh));
                in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 3) * wh));

                in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 4) * wh));
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 5) * wh));
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 6) * wh));
                in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 7) * wh));
                #endif
                DOT16X16(in0, weights0, out1.s0);
                DOT16X16(in1, weights1, out1.s0);
                DOT16X16(in0, weights2, out1.s1);
                DOT16X16(in1, weights3, out1.s1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in0, in1;
                #ifdef WIDTH_HEIGHT_1
                in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset2 + k * 32));
                in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset2 + k * 32 + 16));
                #else
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 1) * wh));
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 2) * wh));
                in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 3) * wh));

                in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 4) * wh));
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 5) * wh));
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 6) * wh));
                in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 7) * wh));
                #endif
                DOT16X16(in0, weights0, out2.s0);
                DOT16X16(in1, weights1, out2.s0);
                DOT16X16(in0, weights2, out2.s1);
                DOT16X16(in1, weights3, out2.s1);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in0, in1;
                #ifdef WIDTH_HEIGHT_1
                in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset3 + k * 32));
                in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset3 + k * 32 + 16));
                #else
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 1) * wh));
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 2) * wh));
                in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 3) * wh));

                in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 4) * wh));
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 5) * wh));
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 6) * wh));
                in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 7) * wh));
                #endif
                DOT16X16(in0, weights0, out3.s0);
                DOT16X16(in1, weights1, out3.s0);
                DOT16X16(in0, weights2, out3.s1);
                DOT16X16(in1, weights3, out3.s1);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k8 = k << 3;
            COMPUTE_FLOAT16 weights0, weights1, weights2, weights3;
            {
                uchar16 charWeightsInt4 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            {
                uchar16 charWeightsInt4 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 1, k)));
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights2 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s2 + ScaleOffset.s3;
                weights3 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s2 + ScaleOffset.s3;
            }
            PADZEROS(k, srcChannel, weights0);
            PADZEROS(k + 15, srcChannel, weights1);
            PADZEROS(k, srcChannel, weights2);
            PADZEROS(k + 15, srcChannel, weights3);
            {
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 1) * wh) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 2) * wh) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 3) * wh) : (FLOAT4)0);
                        
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 4) * wh) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 5) * wh) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 6) * wh) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 7) * wh) : (FLOAT4)0);
                
                DOT16X16(in0, weights0, out.s0);
                DOT16X16(in1, weights1, out.s0);
                DOT16X16(in0, weights2, out.s1);
                DOT16X16(in1, weights3, out.s1);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 1) * wh) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 2) * wh) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 3) * wh) : (FLOAT4)0);
                        
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 4) * wh) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 5) * wh) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 6) * wh) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 7) * wh) : (FLOAT4)0);
                
                DOT16X16(in0, weights0, out1.s0);
                DOT16X16(in1, weights1, out1.s0);
                DOT16X16(in0, weights2, out1.s1);
                DOT16X16(in1, weights3, out1.s1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 1) * wh) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 2) * wh) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 3) * wh) : (FLOAT4)0);
                        
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 4) * wh) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 5) * wh) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 6) * wh) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 7) * wh) : (FLOAT4)0);
                
                DOT16X16(in0, weights0, out2.s0);
                DOT16X16(in1, weights1, out2.s0);
                DOT16X16(in0, weights2, out2.s1);
                DOT16X16(in1, weights3, out2.s1);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 1) * wh) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 2) * wh) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 3) * wh) : (FLOAT4)0);
                        
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 4) * wh) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 5) * wh) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 6) * wh) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 7) * wh) : (FLOAT4)0);
                
                DOT16X16(in0, weights0, out3.s0);
                DOT16X16(in1, weights1, out3.s0);
                DOT16X16(in0, weights2, out3.s1);
                DOT16X16(in1, weights3, out3.s1);
            }
            #endif
        }
        #endif
    #endif //USE_LOW_BIT_WEIGHT_INT4
    }

#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT2)0);
#endif
#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
    vstore2(CONVERT_FLOAT2(out), 0, output + out_offset);
#ifdef BACTH_BLOCK4
    if(isValidBatch1){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
            
        vstore2(CONVERT_FLOAT2(out1), 0, output+out_offset);
    }
    if(isValidBatch2){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out2 = fmax(out2, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
        out2 = clamp(out2, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
            
        vstore2(CONVERT_FLOAT2(out2), 0, output+out_offset);
    }
    if(isValidBatch3){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out3 = fmax(out3, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
        out3 = clamp(out3, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif
            
        vstore2(CONVERT_FLOAT2(out3), 0, output+out_offset);
    }
#endif
}
__kernel void gemm_conv_c1_image(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
                        __read_only image2d_t weight,
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int srcChannel,
                        __private const int batch,
                        __private const int height,
                        __private const int width,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h
    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
#ifdef BACTH_BLOCK4
    const int out_b_idx = (out_b_h_idx / height) << 2;
#else
    const int out_b_idx = out_b_h_idx / height;
#endif
    const int out_h_idx = out_b_h_idx % height;
    
    COMPUTE_FLOAT bias0 = bias[out_c_idx];
    COMPUTE_FLOAT out = bias0;
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = (((out_b_idx * dstChannelC4 + out_c_idx/4)* height + out_h_idx) * width + out_w_idx) * 4 + (out_c_idx%4);
    int wh = width * height * 4;
#ifdef BACTH_BLOCK4
    COMPUTE_FLOAT out1 = bias0, out2 = bias0, out3 = bias0;
    int input_offset1 = (((out_b_idx + 1) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset2 = (((out_b_idx + 2) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int input_offset3 = (((out_b_idx + 3) * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    bool isValidBatch1 = out_b_idx + 1 < batch;
    bool isValidBatch2 = out_b_idx + 2 < batch;
    bool isValidBatch3 = out_b_idx + 3 < batch;
#endif

#if (defined USE_LOW_BIT_WEIGHT_INT8)
    const int loop = (blockDim + 15) / 16;
    #ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
    #else
    const int loop_end = loop;
    #endif
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
    const int loop = (blockDim + 31) / 32;
    #ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
    #else
    const int loop_end = loop;
    #endif
#endif
    
    for (int i = 0; i < blockNum; ++i){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT2 ScaleOffset = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, dequantScaleOffset + kindex));
        #if (defined USE_LOW_BIT_WEIGHT_INT8)
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            #ifndef WIDTH_HEIGHT_1
            int k4 = k << 2;
            #endif
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)))) * ScaleOffset.s0 + ScaleOffset.s1;
            {
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 16));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out);
            }
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 16));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset2 + k * 16));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out2);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                #ifdef WIDTH_HEIGHT_1
                in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset3 + k * 16));
                #else
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 1) * wh));
                in.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 2) * wh));
                in.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k4 + 3) * wh));
                #endif
                DOT16X16(in, weights0, out3);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k4 = k << 2;
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)))) * ScaleOffset.s0 + ScaleOffset.s1;
            {
               COMPUTE_FLOAT16 in;
               in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k4 * wh));
               in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0);
               in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0);
               in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0);
               DOT16X16(in, weights0, out);
            }
            PADZEROS(k, srcChannel, weights0);
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset2 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out2);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in;
                in.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k4 * wh));
                in.s4567 = CONVERT_COMPUTE_FLOAT4(k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 1) * wh) : (FLOAT4)0);
                in.s89ab = CONVERT_COMPUTE_FLOAT4(k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 2) * wh) : (FLOAT4)0);
                in.scdef = CONVERT_COMPUTE_FLOAT4(k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset3 + (k4 + 3) * wh) : (FLOAT4)0);
                DOT16X16(in, weights0, out3);
            }
            #endif
        }
        #endif
        #elif (defined USE_LOW_BIT_WEIGHT_INT4)
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            #ifndef WIDTH_HEIGHT_1
            int k8 = k << 3;
            #endif
            COMPUTE_FLOAT16 weights0, weights1;
            {
                uchar16 charWeightsInt4 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            {
                COMPUTE_FLOAT16 in0, in1;
                #ifdef WIDTH_HEIGHT_1
                in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 32));
                in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * 32 + 16));
                #else
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 1) * wh));
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 2) * wh));
                in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 3) * wh));

                in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 4) * wh));
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 5) * wh));
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 6) * wh));
                in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k8 + 7) * wh));
                #endif
                DOT16X16(in0, weights0, out);
                DOT16X16(in1, weights1, out);
            }
        
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in0, in1;
                #ifdef WIDTH_HEIGHT_1
                in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 32));
                in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset1 + k * 32 + 16));
                #else
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 1) * wh));
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 2) * wh));
                in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 3) * wh));

                in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 4) * wh));
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 5) * wh));
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 6) * wh));
                in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + (k8 + 7) * wh));
                #endif
                DOT16X16(in0, weights0, out1);
                DOT16X16(in1, weights1, out1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in0, in1;
                #ifdef WIDTH_HEIGHT_1
                in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset2 + k * 32));
                in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset2 + k * 32 + 16));
                #else
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 1) * wh));
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 2) * wh));
                in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 3) * wh));

                in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 4) * wh));
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 5) * wh));
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 6) * wh));
                in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + (k8 + 7) * wh));
                #endif
                DOT16X16(in0, weights0, out2);
                DOT16X16(in1, weights1, out2);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in0, in1;
                #ifdef WIDTH_HEIGHT_1
                in0 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset3 + k * 32));
                in1 = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset3 + k * 32 + 16));
                #else
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k8 + 1) * wh));
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k8 + 2) * wh));
                in0.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k8 + 3) * wh));

                in1.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k8 + 4) * wh));
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k8 + 5) * wh));
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k8 + 6) * wh));
                in1.scdef = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + (k8 + 7) * wh));
                #endif
                DOT16X16(in0, weights0, out3);
                DOT16X16(in1, weights1, out3);
            }
            #endif
        }
        #ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k8 = k << 3;
            COMPUTE_FLOAT16 weights0, weights1;
            {
                uchar16 charWeightsInt4 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            PADZEROS(k, srcChannel, weights0);
            PADZEROS(k + 15, srcChannel, weights1);
            {
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 1) * wh) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 2) * wh) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 3) * wh) : (FLOAT4)0);
                        
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 4) * wh) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 5) * wh) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 6) * wh) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset + (k8 + 7) * wh) : (FLOAT4)0);
                DOT16X16(in0, weights0, out);
                DOT16X16(in1, weights1, out);
            }
        
            #ifdef BACTH_BLOCK4
            if(isValidBatch1){
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset1 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 1) * wh) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 2) * wh) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 3) * wh) : (FLOAT4)0);
                        
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 4) * wh) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 5) * wh) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 6) * wh) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset1 + (k8 + 7) * wh) : (FLOAT4)0);
                DOT16X16(in0, weights0, out1);
                DOT16X16(in1, weights1, out1);
            }
            if(isValidBatch2){
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset2 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 1) * wh) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 2) * wh) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 3) * wh) : (FLOAT4)0);
                        
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 4) * wh) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 5) * wh) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 6) * wh) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset2 + (k8 + 7) * wh) : (FLOAT4)0);
                DOT16X16(in0, weights0, out2);
                DOT16X16(in1, weights1, out2);
            }
            if(isValidBatch3){
                COMPUTE_FLOAT16 in0, in1;
                in0.s0123 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset3 + k8 * wh));
                in0.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 1 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 1) * wh) : (FLOAT4)0);
                in0.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 2 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 2) * wh) : (FLOAT4)0);
                in0.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 3 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 3) * wh) : (FLOAT4)0);
                        
                in1.s0123 = CONVERT_COMPUTE_FLOAT4(k8 + 4 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 4) * wh) : (FLOAT4)0);
                in1.s4567 = CONVERT_COMPUTE_FLOAT4(k8 + 5 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 5) * wh) : (FLOAT4)0);
                in1.s89ab = CONVERT_COMPUTE_FLOAT4(k8 + 6 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 6) * wh) : (FLOAT4)0);
                in1.scdef = CONVERT_COMPUTE_FLOAT4(k8 + 7 < srcChannelC4 ? vload4(0, input + input_offset3 + (k8 + 7) * wh) : (FLOAT4)0);
                DOT16X16(in0, weights0, out3);
                DOT16X16(in1, weights1, out3);
            }
            #endif
        }
        #endif
    #endif //USE_LOW_BIT_WEIGHT_INT4
    }

#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT)0);
#endif
#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
    output[out_offset] = out;
#ifdef BACTH_BLOCK4
    if(isValidBatch1){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out1 = fmax(out1, (COMPUTE_FLOAT)0);
#endif

#ifdef RELU6
        out1 = clamp(out1, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
                
        output[out_offset] = out1;
    }
    if(isValidBatch2){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out2 = fmax(out2, (COMPUTE_FLOAT)0);
#endif

#ifdef RELU6
        out1 = clamp(out2, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
                
        output[out_offset] = out2;
    }
    if(isValidBatch3){
        out_offset += dstChannelC4 * height * width * 4;
#ifdef RELU
        out3 = fmax(out3, (COMPUTE_FLOAT)0);
#endif

#ifdef RELU6
        out3 = clamp(out3, (COMPUTE_FLOAT)0, (COMPUTE_FLOAT)6);
#endif
                
        output[out_offset] = out3;
    }
#endif
}
        
