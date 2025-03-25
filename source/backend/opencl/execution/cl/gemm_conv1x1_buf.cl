#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

#define UCHAR4_TO_CHAR8(a, c) \
    a.s0 = (c.s0 >> 4) - 8; a.s1 = (c.s0 & 15) - 8; a.s2 = (c.s1 >> 4) - 8; a.s3 = (c.s1 & 15) - 8; a.s4 = (c.s2 >> 4) - 8; a.s5 = (c.s2 & 15) - 8; a.s6 = (c.s3 >> 4) - 8; a.s7 = (c.s3 & 15) - 8;

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
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
    __global const FLOAT *dequantScaleOffset,
    __global FLOAT* output,
    __private const int inputChannel,
    __private const int inputChannel4Align,
    __private const int outputChannelAlign,
    __private const int outputChannel4Align,
    __private const int blockDim,
    __private const float coef){
    const int x = get_global_id(0); //ic
    const int y = get_global_id(1); //oc

    UNIFORM_BOUNDRY_CHECK(x, y);
    
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    const int ic = x << 2;
    const int oc = y << 3;
    const int output_offset = ic * outputChannelAlign + oc;

    #ifdef ASYMMETRIC
    COMPUTE_FLOAT8 scale, offset;
    {
        COMPUTE_FLOAT16 ScaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + ((ic / blockDim) * outputChannel4Align + oc) * 2)) / coef);
        scale = ScaleOffset.s02468ace;
        offset = ScaleOffset.s13579bdf;
    }
    #else
    COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (ic / blockDim) * outputChannel4Align + oc)) / coef);
    #endif
    COMPUTE_FLOAT8 weights0, weights1, weights2, weights3;
    {
        #ifdef USE_IMAGE
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(x, y)));
        #else
        uchar16 charWeightsInt40 = vload16(x, weight + y * inputChannel4Align * 4);
        #endif
        char8 charWeights0;
        #ifdef ASYMMETRIC
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s0123);
        weights0 = CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s4567);
        weights1 = ic + 1 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s89ab);
        weights2 = ic + 2 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.scdef);
        weights3 = ic + 3 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        #else
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s0123);
        weights0 = CONVERT_COMPUTE_FLOAT8(charWeights0) * scale;
        
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s4567);
        weights1 = ic + 1 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale;
        
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s89ab);
        weights2 = ic + 2 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale;
        
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.scdef);
        weights3 = ic + 3 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale;
        #endif
    }
    vstore8(CONVERT_FLOAT8(weights0), 0, output+output_offset);
    vstore8(CONVERT_FLOAT8(weights1), 0, output+output_offset+outputChannelAlign);
    vstore8(CONVERT_FLOAT8(weights2), 0, output+output_offset+2*outputChannelAlign);
    vstore8(CONVERT_FLOAT8(weights3), 0, output+output_offset+3*outputChannelAlign);
#else
    const int ic = x << 1;
    const int oc = y << 3;
    const int output_offset = ic * outputChannelAlign + oc;
    
    #ifdef ASYMMETRIC
    COMPUTE_FLOAT8 scale, offset;
    {
        COMPUTE_FLOAT16 ScaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + ((ic / blockDim) * outputChannel4Align + oc) * 2)) / coef);
        scale = ScaleOffset.s02468ace;
        offset = ScaleOffset.s13579bdf;
    }
    #else
    COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (ic / blockDim) * outputChannel4Align + oc)) / coef);
    #endif
    COMPUTE_FLOAT8 weights0, weights1;
    {
        #ifdef USE_IMAGE
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(x, y))));
        #else
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(x, weight + y * inputChannel4Align * 8));
        #endif
        #ifdef ASYMMETRIC
        weights0 = wei.s01234567 * scale + offset;
        weights1 = ic + 1 >= inputChannel ? 0 : wei.s89abcdef * scale + offset;
        #else
        weights0 = wei.s01234567 * scale;
        weights1 = ic + 1 >= inputChannel ? 0 : wei.s89abcdef * scale;
        #endif
    }
    vstore8(CONVERT_FLOAT8(weights0), 0, output+output_offset);
    vstore8(CONVERT_FLOAT8(weights1), 0, output+output_offset+outputChannelAlign);
    #endif
}

#define UCHAR4_TO_FLOAT8(b, scale, offset) \
    wei.s0 = (COMPUTE_FLOAT)((b.s0 >> 4) - 8); \
    wei.s1 = (COMPUTE_FLOAT)((b.s0 & 15) - 8); \
    wei.s2 = (COMPUTE_FLOAT)((b.s1 >> 4) - 8); \
    wei.s3 = (COMPUTE_FLOAT)((b.s1 & 15) - 8); \
    wei.s4 = (COMPUTE_FLOAT)((b.s2 >> 4) - 8); \
    wei.s5 = (COMPUTE_FLOAT)((b.s2 & 15) - 8); \
    wei.s6 = (COMPUTE_FLOAT)((b.s3 >> 4) - 8); \
    wei.s7 = (COMPUTE_FLOAT)((b.s3 & 15) - 8); \
    wei = wei * scale + offset;

__kernel void gemm_b4_c8_int4_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        __global const uchar *weight,
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int bhw,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int x = get_global_id(0); //b/4
    const int y  = get_global_id(1); //c/8

    UNIFORM_BOUNDRY_CHECK(x, y);
    
    const int out_b_idx = x << 2;
    const int out_c_idx = y << 1;

    COMPUTE_FLOAT8 out0 = CONVERT_COMPUTE_FLOAT8(vload8(0, bias + (out_c_idx << 2)));
    COMPUTE_FLOAT8 out1 = out0;
    COMPUTE_FLOAT8 out2 = out0;
    COMPUTE_FLOAT8 out3 = out0;
    
    const int bhw4 = bhw << 2;
    const int input_offset = out_b_idx * 4;
    int out_offset = out_c_idx * bhw4 + out_b_idx * 4;
#ifndef USE_IMAGE
    const int weight_offset = y * srcChannelAlign * 4;
#endif
    const int loop = (blockDim + 4 - 1) / 4;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif

#if INPUT_BATCH_LEAVES_NUM != 0
    if(out_b_idx + 3 >= bhw){
        for (int i = 0; i < blockNum; i++){
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT8 scale, offset;
            {
                COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + (out_c_idx << 3) + i * dstChannelAlign * 2)) / coef);
                scale = scaleOffset.s02468ace;
                offset = scaleOffset.s13579bdf;
            }
            #else
            COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
            COMPUTE_FLOAT8 offset = 0;
            #endif
            for (int j = 0; j < loop_end; j++) {
                int k = i * loop + j;
                COMPUTE_FLOAT8 wei;
                #ifdef USE_IMAGE
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
                #else
                uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
                #endif
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4));
                #if INPUT_BATCH_LEAVES_NUM >= 2
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4 + 4));
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4 + 8));
                #endif
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s0, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s0, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s0, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s1, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s1, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s1, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s2, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s2, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s2, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.scdef, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s3, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s3, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s3, wei, out2);
                    #endif
                }
            }
            #if INPUT_CHANNEL_LEAVES_NUM != 0
            {
                int k = i * loop + loop_end;
                COMPUTE_FLOAT8 wei;
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4));
                #if INPUT_BATCH_LEAVES_NUM >= 2
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4 + 4));
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4 + 8));
                #endif
                #ifdef USE_IMAGE
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
                #else
                uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
                #endif
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s0, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s0, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s0, wei, out2);
                    #endif
                }
                #if INPUT_CHANNEL_LEAVES_NUM >= 2
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s1, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s1, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s1, wei, out2);
                    #endif
                }
                #endif
                #if INPUT_CHANNEL_LEAVES_NUM >= 3
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s2, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s2, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s2, wei, out2);
                    #endif
                }
                #endif
            }
            #endif
        }
    } else {
#endif
    for (int i = 0; i < blockNum; i++){
        #ifdef ASYMMETRIC
        COMPUTE_FLOAT8 scale, offset;
        {
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + (out_c_idx << 3) + i * dstChannelAlign * 2)) / coef);
            scale = scaleOffset.s02468ace;
            offset = scaleOffset.s13579bdf;
        }
        #else
        COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
        COMPUTE_FLOAT8 offset = 0;
        #endif
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            COMPUTE_FLOAT8 wei;
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4));
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s4, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s8, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sc, wei, out3);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s5, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s9, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sd, wei, out3);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s6, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sa, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)in.se, wei, out3);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.scdef, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s3, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s7, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sb, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sf, wei, out3);
            }
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            COMPUTE_FLOAT8 wei;
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4));
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s4, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s8, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sc, wei, out3);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s5, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s9, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sd, wei, out3);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s6, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sa, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)in.se, wei, out3);
            }
            #endif
        }
        #endif
    }
#if INPUT_BATCH_LEAVES_NUM != 0
    }
#endif
    
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT8)0);
    out1 = fmax(out1, (COMPUTE_FLOAT8)0);
    out2 = fmax(out2, (COMPUTE_FLOAT8)0);
    out3 = fmax(out3, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out1 = clamp(out1, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out2 = clamp(out2, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out3 = clamp(out3, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif
    vstore4(CONVERT_FLOAT4(out0.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out0.s4567), 0, output+out_offset+bhw4);
    if(out_b_idx + 1 >= bhw) return;
    out_offset += 4;
    vstore4(CONVERT_FLOAT4(out1.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1.s4567), 0, output+out_offset+bhw4);
    if(out_b_idx + 2 >= bhw) return;
    out_offset += 4;
    vstore4(CONVERT_FLOAT4(out2.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2.s4567), 0, output+out_offset+bhw4);
    if(out_b_idx + 3 >= bhw) return;
    out_offset += 4;
    vstore4(CONVERT_FLOAT4(out3.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3.s4567), 0, output+out_offset+bhw4);
}


__kernel void gemm_b4_c8_int8_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        __global const char *weight,
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int bhw,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int x = get_global_id(0); //b/4
    const int y  = get_global_id(1); //c/8

    UNIFORM_BOUNDRY_CHECK(x, y);
    
    const int out_b_idx = x << 2;
    const int out_c_idx = y << 1;

    COMPUTE_FLOAT8 out0 = CONVERT_COMPUTE_FLOAT8(vload8(0, bias + (out_c_idx << 2)));
    COMPUTE_FLOAT8 out1 = out0;
    COMPUTE_FLOAT8 out2 = out0;
    COMPUTE_FLOAT8 out3 = out0;
    
    const int bhw4 = bhw << 2;
    const int input_offset = out_b_idx * 4;
    int out_offset = out_c_idx * bhw4 + out_b_idx * 4;
#ifndef USE_IMAGE
    const int weight_offset = y * srcChannelAlign * 8;
#endif
    const int loop = (blockDim + 4 - 1) / 4;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif

#if INPUT_BATCH_LEAVES_NUM != 0
    if(out_b_idx + 3 >= bhw){
        for (int i = 0; i < blockNum; i++){
            COMPUTE_FLOAT16 scale, offset;
            {
                #ifdef ASYMMETRIC
                COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + (out_c_idx << 3) + i * dstChannelAlign * 2)) / coef);
                scale = (COMPUTE_FLOAT16)(scaleOffset.s02468ace, scaleOffset.s02468ace);
                offset = (COMPUTE_FLOAT16)(scaleOffset.s13579bdf, scaleOffset.s13579bdf);
                #else
                scale.s01234567 = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
                scale.s89abcdef = scale.s01234567;
                offset = 0;
                #endif
            }
            for (int j = 0; j < loop_end; j++) {
                int k = i * loop + j;
                int k2 = k << 1;
                #ifdef USE_IMAGE
                COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2, y)))) * scale + offset;
                COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 + 1, y)))) * scale + offset;
                #else
                COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(vload16(k2, weight + weight_offset)) * scale + offset;
                COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(vload16(k2 + 1, weight + weight_offset)) * scale + offset;
                #endif
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4));
                #if INPUT_BATCH_LEAVES_NUM >= 2
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4 + 4));
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4 + 8));
                #endif
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s0, wei0.s01234567, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s0, wei0.s01234567, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s0, wei0.s01234567, out2);
                    #endif
                }
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s1, wei0.s89abcdef, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s1, wei0.s89abcdef, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s1, wei0.s89abcdef, out2);
                    #endif
                }
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s2, wei1.s01234567, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s2, wei1.s01234567, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s2, wei1.s01234567, out2);
                    #endif
                }
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s3, wei1.s89abcdef, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s3, wei1.s89abcdef, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s3, wei1.s89abcdef, out2);
                    #endif
                }
            }
            #if INPUT_CHANNEL_LEAVES_NUM != 0
            {
                int k = i * loop + loop_end;
                int k2 = k << 1;
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4));
                #if INPUT_BATCH_LEAVES_NUM >= 2
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4 + 4));
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4(vload4(0, input + input_offset + k * bhw4 + 8));
                #endif
                #ifdef USE_IMAGE
                COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2, y)))) * scale + offset;
                COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 + 1, y)))) * scale + offset;
                #else
                COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(vload16(k2, weight + weight_offset)) * scale + offset;
                COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(vload16(k2 + 1, weight + weight_offset)) * scale + offset;
                #endif
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s0, wei0.s01234567, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s0, wei0.s01234567, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s0, wei0.s01234567, out2);
                    #endif
                }
                #if INPUT_CHANNEL_LEAVES_NUM >= 2
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s1, wei0.s89abcdef, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s1, wei0.s89abcdef, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s1, wei0.s89abcdef, out2);
                    #endif
                }
                #endif
                #if INPUT_CHANNEL_LEAVES_NUM >= 3
                {
                    out0 = mad((COMPUTE_FLOAT8)in0.s2, wei1.s01234567, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s2, wei1.s01234567, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s2, wei1.s01234567, out2);
                    #endif
                }
                #endif
            }
            #endif
        }
    } else {
#endif
    for (int i = 0; i < blockNum; i++){
        COMPUTE_FLOAT16 scale, offset;
        {
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + (out_c_idx << 3) + i * dstChannelAlign * 2)) / coef);
            scale = (COMPUTE_FLOAT16)(scaleOffset.s02468ace, scaleOffset.s02468ace);
            offset = (COMPUTE_FLOAT16)(scaleOffset.s13579bdf, scaleOffset.s13579bdf);
            #else
            scale.s01234567 = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
            scale.s89abcdef = scale.s01234567;
            offset = 0;
            #endif
        }
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            int k2 = k << 1;
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4));
            #ifdef USE_IMAGE
            COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2, y)))) * scale + offset;
            COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 + 1, y)))) * scale + offset;
            #else
            COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(vload16(k2, weight + weight_offset)) * scale + offset;
            COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(vload16(k2 + 1, weight + weight_offset)) * scale + offset;
            #endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei0.s01234567, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s4, wei0.s01234567, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s8, wei0.s01234567, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sc, wei0.s01234567, out3);
            }
            {
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei0.s89abcdef, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s5, wei0.s89abcdef, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s9, wei0.s89abcdef, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sd, wei0.s89abcdef, out3);
            }
            {
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei1.s01234567, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s6, wei1.s01234567, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sa, wei1.s01234567, out2);
                out3 = mad((COMPUTE_FLOAT8)in.se, wei1.s01234567, out3);
            }
            {
                out0 = mad((COMPUTE_FLOAT8)in.s3, wei1.s89abcdef, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s7, wei1.s89abcdef, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sb, wei1.s89abcdef, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sf, wei1.s89abcdef, out3);
            }
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            int k2 = k << 1;
            COMPUTE_FLOAT16 in = CONVERT_COMPUTE_FLOAT16(vload16(0, input + input_offset + k * bhw4));
            #ifdef USE_IMAGE
            COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2, y)))) * scale + offset;
            COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(k2 + 1, y)))) * scale + offset;
            #else
            COMPUTE_FLOAT16 wei0 = CONVERT_COMPUTE_FLOAT16(vload16(k2, weight + weight_offset)) * scale + offset;
            COMPUTE_FLOAT16 wei1 = CONVERT_COMPUTE_FLOAT16(vload16(k2 + 1, weight + weight_offset)) * scale + offset;
            #endif
            {
                out0 = mad((COMPUTE_FLOAT8)in.s0, wei0.s01234567, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s4, wei0.s01234567, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s8, wei0.s01234567, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sc, wei0.s01234567, out3);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                out0 = mad((COMPUTE_FLOAT8)in.s1, wei0.s89abcdef, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s5, wei0.s89abcdef, out1);
                out2 = mad((COMPUTE_FLOAT8)in.s9, wei0.s89abcdef, out2);
                out3 = mad((COMPUTE_FLOAT8)in.sd, wei0.s89abcdef, out3);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                out0 = mad((COMPUTE_FLOAT8)in.s2, wei1.s01234567, out0);
                out1 = mad((COMPUTE_FLOAT8)in.s6, wei1.s01234567, out1);
                out2 = mad((COMPUTE_FLOAT8)in.sa, wei1.s01234567, out2);
                out3 = mad((COMPUTE_FLOAT8)in.se, wei1.s01234567, out3);
            }
            #endif
        }
        #endif
    }
#if INPUT_BATCH_LEAVES_NUM != 0
    }
#endif
    
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT8)0);
    out1 = fmax(out1, (COMPUTE_FLOAT8)0);
    out2 = fmax(out2, (COMPUTE_FLOAT8)0);
    out3 = fmax(out3, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out1 = clamp(out1, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out2 = clamp(out2, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out3 = clamp(out3, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif
    vstore4(CONVERT_FLOAT4(out0.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out0.s4567), 0, output+out_offset+bhw4);
    if(out_b_idx + 1 >= bhw) return;
    out_offset += 4;
    vstore4(CONVERT_FLOAT4(out1.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1.s4567), 0, output+out_offset+bhw4);
    if(out_b_idx + 2 >= bhw) return;
    out_offset += 4;
    vstore4(CONVERT_FLOAT4(out2.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2.s4567), 0, output+out_offset+bhw4);
    if(out_b_idx + 3 >= bhw) return;
    out_offset += 4;
    vstore4(CONVERT_FLOAT4(out3.s0123), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3.s4567), 0, output+out_offset+bhw4);
}
