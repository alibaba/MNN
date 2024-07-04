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

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void reshape_nchw4_nhwc4(GLOBAL_SIZE_DIM3
__global const FLOAT* input,
__global FLOAT* output,
__private const int width_height,
__private const int batch,
__private const int channel,
__private const int channelC4){
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b
    const int wh = get_global_id(2); // w*h

    UNIFORM_BOUNDRY_CHECK3(x, y, wh);
    
    const int x4 = x << 2;
    const int y4 = y << 2;
    const int channel4 = channelC4 * 4;
    const int stride = channel4 * width_height;
    const int input_offset = (y4 * channel4 + x4) * width_height + wh * 4;
    const int output_offset = ((y * width_height + wh) * channel4 + x4) * 4;
    FLOAT4 in0 = vload4(0, input + input_offset);
    FLOAT4 in1 = (y4 + 1 < batch) ? vload4(0, input + input_offset + stride) : (FLOAT4)0;
    FLOAT4 in2 = (y4 + 2 < batch) ? vload4(0, input + input_offset + 2 * stride) : (FLOAT4)0;
    FLOAT4 in3 = (y4 + 3 < batch) ? vload4(0, input + input_offset + 3 * stride) : (FLOAT4)0;
    
#ifdef INPUT_CHANNEL_LEAVE
    if(x4 + 3 >= channel){
        FLOAT *in0_ptr = (FLOAT*)&in0;
        FLOAT *in1_ptr = (FLOAT*)&in1;
        FLOAT *in2_ptr = (FLOAT*)&in2;
        FLOAT *in3_ptr = (FLOAT*)&in3;
        int remain = x4 + 3 - channel;
        for(int i = remain; i >= 0; i--){
            in0_ptr[3 - remain] = 0;
            in1_ptr[3 - remain] = 0;
            in2_ptr[3 - remain] = 0;
            in3_ptr[3 - remain] = 0;
        }
    }
#endif
    
    FLOAT16 out = (FLOAT16)(in0.s0, in1.s0, in2.s0, in3.s0, in0.s1, in1.s1, in2.s1, in3.s1, in0.s2, in1.s2, in2.s2, in3.s2, in0.s3, in1.s3, in2.s3, in3.s3);
    
    vstore16(out, 0, output+output_offset);
}

__kernel void reshape_nhwc4_nchw4(GLOBAL_SIZE_DIM3
__global const FLOAT* input,
__global FLOAT* output,
__private const int width_height,
__private const int batch,
__private const int channelC4){
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b
    const int wh  = get_global_id(2); //w*h

    UNIFORM_BOUNDRY_CHECK3(x, y, wh);
    
    const int x4 = x << 2;
    const int y4 = y << 2;
    const int channel4 = channelC4 * 4;
    const int stride = channel4 * width_height;
    const int input_offset = ((y * width_height + wh) * channel4 + x4) * 4;
    const int output_offset = (y4 * channel4 + x4) * width_height + wh * 4;
    FLOAT16 in = vload16(0, input + input_offset);
    
    FLOAT4 out0 = (FLOAT4)(in.s0, in.s4, in.s8, in.sc);
    FLOAT4 out1 = (FLOAT4)(in.s1, in.s5, in.s9, in.sd);
    FLOAT4 out2 = (FLOAT4)(in.s2, in.s6, in.sa, in.se);
    FLOAT4 out3 = (FLOAT4)(in.s3, in.s7, in.sb, in.sf);
    
    vstore4(out0, 0, output+output_offset);
    if(y4 + 1 >= batch) return;
    vstore4(out1, 0, output+output_offset+stride);
    if(y4 + 2 >= batch) return;
    vstore4(out2, 0, output+output_offset+2*stride);
    if(y4 + 3 >= batch) return;
    vstore4(out3, 0, output+output_offset+3*stride);
}


__kernel void gemm_b4_c4_buf(GLOBAL_SIZE_DIM2
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
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b

    UNIFORM_BOUNDRY_CHECK(x, y);

    const int out_c_idx = x;
    const int out_b_idx = y << 2;

    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out = (COMPUTE_FLOAT4)bias0.s0;
    COMPUTE_FLOAT4 out1 = (COMPUTE_FLOAT4)bias0.s1, out2 = (COMPUTE_FLOAT4)bias0.s2, out3 = (COMPUTE_FLOAT4)bias0.s3;
    
    int input_offset = out_b_idx * srcChannelC4 * 4;
    int out_offset = (out_b_idx * dstChannelC4 + out_c_idx * 4) * 4;

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
    const int remain = blockDim - loop_end*16;
#else
    const int loop_end = loop;
#endif
    
    for (int i = 0; i < blockNum; i++){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            int k16 = k << 4;
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
        }
#ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k16 = k << 4;
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
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            COMPUTE_FLOAT *weights2_ptr = (COMPUTE_FLOAT *)&weights2;
            COMPUTE_FLOAT *weights3_ptr = (COMPUTE_FLOAT *)&weights3;
            for (int i = 0; i < remain; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights1_ptr[i], out1);
                out2 = mad(in, weights2_ptr[i], out2);
                out3 = mad(in, weights3_ptr[i], out3);
            }
        }
#endif
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
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b

    UNIFORM_BOUNDRY_CHECK(x, y);

    const int out_c_idx = x;
    const int out_b_idx = y << 2;

    COMPUTE_FLOAT2 bias0 = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, bias));
    COMPUTE_FLOAT4 out = (COMPUTE_FLOAT4)bias0.s0;
    COMPUTE_FLOAT4 out1 = (COMPUTE_FLOAT4)bias0.s1;
    
    int input_offset = out_b_idx * srcChannelC4 * 4;
    int out_offset = (out_b_idx * dstChannelC4 + out_c_idx * 2) * 4;

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
    const int remain = blockDim - loop_end*16;
#else
    const int loop_end = loop;
#endif

    for (int i = 0; i < blockNum; i++){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT4 ScaleOffset = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            int k16 = k << 4;
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
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights1_ptr[i], out1);
            }
        }
#ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k16 = k << 4;
            
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
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            for (int i = 0; i < remain; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights1_ptr[i], out1);
            }
        }
#endif
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
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b

    UNIFORM_BOUNDRY_CHECK(x, y);

    const int out_c_idx = x;
    const int out_b_idx = y << 2;

    COMPUTE_FLOAT bias0 = bias[out_c_idx];
    COMPUTE_FLOAT4 out = (COMPUTE_FLOAT4)bias0;
    
    int input_offset = out_b_idx * srcChannelC4 * 4;
    int out_offset = (out_b_idx * dstChannelC4 + out_c_idx) * 4;

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
    const int remain = blockDim - loop_end*16;
#else
    const int loop_end = loop;
#endif
    
    for (int i = 0; i < blockNum; i++){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT2 ScaleOffset = CONVERT_COMPUTE_FLOAT2(vload2(out_c_idx, dequantScaleOffset + kindex));
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            int k16 = k << 4;
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
            COMPUTE_FLOAT *weights_ptr = (COMPUTE_FLOAT *)&weights;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights_ptr[i], out);
            }
        }
#ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k16 = k << 4;
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
            COMPUTE_FLOAT *weights_ptr = (COMPUTE_FLOAT *)&weights;
            for (int i = 0; i < remain; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights_ptr[i], out);
            }
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
}
__kernel void gemm_b4_c2_image(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
                        __read_only image2d_t weight,
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
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

    int input_offset = out_b_idx * srcChannelC4 * 4;
    int out_offset = (out_b_idx * dstChannelC4 + out_c_idx) * 4;
    
#if (defined USE_LOW_BIT_WEIGHT_INT8)
    const int loop = (blockDim + 15) / 16;
    #ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
    const int remain = blockDim - loop_end*16;
    #else
    const int loop_end = loop;
    #endif
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
    const int loop = (blockDim + 31) / 32;
    #ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
    const int remain = blockDim - loop_end*32;
    #else
    const int loop_end = loop;
    #endif
#endif
    
    for (int i = 0; i < blockNum; i++){
        int kindex = i * dstChannelC4 * 4 * 2;
        COMPUTE_FLOAT4 ScaleOffset = CONVERT_COMPUTE_FLOAT4(vload4(0, dequantScaleOffset + out_c_idx * 2 + kindex));
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            int k16 = k << 4;
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)))) * ScaleOffset.s0 + ScaleOffset.s1;
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 1, k)))) * ScaleOffset.s2 + ScaleOffset.s3;
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights1_ptr[i], out1);
            }
        }
#ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k16 = k << 4;
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)))) * ScaleOffset.s0 + ScaleOffset.s1;
            COMPUTE_FLOAT16 weights1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx + 1, k)))) * ScaleOffset.s2 + ScaleOffset.s3;
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            #pragma unroll
            for (int i = 0; i < remain; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights1_ptr[i], out1);
            }
        }
#endif
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            int k32 = k << 5;
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
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            COMPUTE_FLOAT *weights2_ptr = (COMPUTE_FLOAT *)&weights2;
            COMPUTE_FLOAT *weights3_ptr = (COMPUTE_FLOAT *)&weights3;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights2_ptr[i], out1);
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i + 16) * 4));
                out = mad(in, weights1_ptr[i], out);
                out1 = mad(in, weights3_ptr[i], out1);
            }
        }
#ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k32 = k << 5;
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
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            COMPUTE_FLOAT *weights2_ptr = (COMPUTE_FLOAT *)&weights2;
            COMPUTE_FLOAT *weights3_ptr = (COMPUTE_FLOAT *)&weights3;
            for (int i = 0; i < min(16, remain); ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
                out1 = mad(in, weights2_ptr[i], out1);
            }
            for (int i = 16; i < remain; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights1_ptr[i - 16], out);
                out1 = mad(in, weights3_ptr[i - 16], out1);
            }
        }
#endif
#endif //USE_LOW_BIT_WEIGHT_INT4
    }

#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif
    vstore4(CONVERT_FLOAT4(out), 0, output + out_offset);
    vstore4(CONVERT_FLOAT4(out1), 0, output + out_offset + 4);
}
__kernel void gemm_b4_c1_image(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
                        __read_only image2d_t weight,
                        __global const float *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int blockNum,
                        __private const int blockDim) {
    const int x = get_global_id(0); //c
    const int y  = get_global_id(1); //b
    UNIFORM_BOUNDRY_CHECK(x, y);

    const int out_c_idx = x;
    const int out_b_idx = y << 2;
    
    COMPUTE_FLOAT bias0 = bias[out_c_idx];
    COMPUTE_FLOAT4 out = (COMPUTE_FLOAT4)bias0;
    
    int input_offset = out_b_idx * srcChannelC4 * 4;
    int out_offset = (out_b_idx * dstChannelC4 + out_c_idx) * 4;
    
#if (defined USE_LOW_BIT_WEIGHT_INT8)
    const int loop = (blockDim + 15) / 16;
    #ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
    const int remain = blockDim - loop_end*16;
    #else
    const int loop_end = loop;
    #endif
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
    const int loop = (blockDim + 31) / 32;
    #ifdef INPUT_CHANNEL_LEAVE
    const int loop_end = max(loop - 1, 0);
    const int remain = blockDim - loop_end*32;
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
            int k16 = k << 4;
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)))) * ScaleOffset.s0 + ScaleOffset.s1;
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
            }
        }
#ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k16 = k << 4;
            COMPUTE_FLOAT16 weights0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)))) * ScaleOffset.s0 + ScaleOffset.s1;
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            #pragma unroll
            for (int i = 0; i < remain; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k16 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
            }
        }
#endif
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            int k32 = k << 5;
            COMPUTE_FLOAT16 weights0, weights1;
            {
                uchar16 charWeightsInt4 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i + 16) * 4));
                out = mad(in, weights1_ptr[i], out);
            }
        }
#ifdef INPUT_CHANNEL_LEAVE
        {
            int k = i * loop + loop_end;
            int k32 = k << 5;
            COMPUTE_FLOAT16 weights0, weights1;
            {
                uchar16 charWeightsInt4 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(out_c_idx, k)));
                char16 charWeights0 = 0;
                char16 charWeights1 = 0;
                UCHAR16_TO_2CHAR16(charWeights0, charWeights1, charWeightsInt4);
                weights0 = CONVERT_COMPUTE_FLOAT16(charWeights0) * ScaleOffset.s0 + ScaleOffset.s1;
                weights1 = CONVERT_COMPUTE_FLOAT16(charWeights1) * ScaleOffset.s0 + ScaleOffset.s1;
            }
            COMPUTE_FLOAT *weights0_ptr = (COMPUTE_FLOAT *)&weights0;
            COMPUTE_FLOAT *weights1_ptr = (COMPUTE_FLOAT *)&weights1;
            for (int i = 0; i < min(16, remain); ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights0_ptr[i], out);
            }
            for (int i = 16; i < remain; ++i){
                COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + (k32 + i) * 4));
                out = mad(in, weights1_ptr[i - 16], out);
            }
        }
#endif
#endif //USE_LOW_BIT_WEIGHT_INT4
    }

#ifdef RELU
    out = fmax(out, (COMPUTE_FLOAT4)0);
#endif
#ifdef RELU6
    out = clamp(out, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif
    vstore4(CONVERT_FLOAT4(out), 0, output+out_offset);
}
        
