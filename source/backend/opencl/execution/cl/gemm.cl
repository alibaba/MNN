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

__kernel void gemm(__read_only image2d_t uInput, __read_only image2d_t uKernel, __write_only image2d_t uOutput,
                   __private const int width, __private const int height, __private const int multiLength, __private const int alpha2) {
    
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    if (pos.x < width*height && pos.y < alpha2) {
        
        const int pos_x = pos.x % width;
        const int pos_y = pos.x / width;
        const int pos_z = pos.y;

        FLOAT4 o0 = (FLOAT4)(0);
        FLOAT4 o1 = (FLOAT4)(0);
        FLOAT4 o2 = (FLOAT4)(0);
        FLOAT4 o3 = (FLOAT4)(0);
        int kenerlY   = mad24(pos_z, height, pos_y);
        int srcY      = mad24(pos_z, width, pos_x);

        for (int k = 0; k < multiLength; ++k) {
            __private int index = mul24(k, 4);
            FLOAT4 k0 = RI_F(uKernel, SAMPLER, (int2)(index, kenerlY));
            FLOAT4 k1 = RI_F(uKernel, SAMPLER, (int2)(index+1, kenerlY));
            FLOAT4 k2 = RI_F(uKernel, SAMPLER, (int2)(index+2, kenerlY));
            FLOAT4 k3 = RI_F(uKernel, SAMPLER, (int2)(index+3, kenerlY));

            FLOAT4 s0 = RI_F(uInput, SAMPLER, (int2)(index, srcY));
            FLOAT4 s1 = RI_F(uInput, SAMPLER, (int2)(index+1, srcY));
            FLOAT4 s2 = RI_F(uInput, SAMPLER, (int2)(index+2, srcY));
            FLOAT4 s3 = RI_F(uInput, SAMPLER, (int2)(index+3, srcY));

            o0 = mad(s0.x, k0, o0);
            o0 = mad(s0.y, k1, o0);
            o0 = mad(s0.z, k2, o0);
            o0 = mad(s0.w, k3, o0);

            o1 = mad(s1.x, k0, o1);
            o1 = mad(s1.y, k1, o1);
            o1 = mad(s1.z, k2, o1);
            o1 = mad(s1.w, k3, o1);

            o2 = mad(s2.x, k0, o2);
            o2 = mad(s2.y, k1, o2);
            o2 = mad(s2.z, k2, o2);
            o2 = mad(s2.w, k3, o2);

            o3 = mad(s3.x, k0, o3);
            o3 = mad(s3.y, k1, o3);
            o3 = mad(s3.z, k2, o3);
            o3 = mad(s3.w, k3, o3);
        }

        __private int out_y_idx = mul24(pos_y, 4);
        WI_F(uOutput, (int2)(srcY, out_y_idx), o0);
        WI_F(uOutput, (int2)(srcY, out_y_idx + 1), o1);
        WI_F(uOutput, (int2)(srcY, out_y_idx + 2), o2);
        WI_F(uOutput, (int2)(srcY, out_y_idx + 3), o3);
    }
}

__kernel void gemmWinograd(__read_only image2d_t uInput, __read_only image2d_t uKernel, __write_only image2d_t uOutput,
                   __private const int unitWidth, __private const int unitHeight, __private const int dstChannelC4, __private const int multiLength, __private const int alpha2) {
    
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int unitWidth4 = (unitWidth + 3) / 4;
    if (pos.x < unitWidth4 * unitHeight && pos.y < alpha2 * dstChannelC4) {
        
        const int pos_x = pos.x % unitWidth4;
        const int pos_y = pos.x / unitWidth4;
        const int pos_z = pos.y % dstChannelC4;
        const int pos_w = pos.y / dstChannelC4;

        FLOAT4 o0 = (FLOAT4)(0);
        FLOAT4 o1 = (FLOAT4)(0);
        FLOAT4 o2 = (FLOAT4)(0);
        FLOAT4 o3 = (FLOAT4)(0);
        int srcY = mad24(pos_w, unitHeight, pos_y);
        int srcX = pos_x << 2;

        for (int k = 0; k < multiLength; ++k) {
            __private int index = mul24(k, 4);
            __private int x_offset = mul24(k, unitWidth);
            FLOAT4 k0 = RI_F(uKernel, SAMPLER, (int2)(index, pos.y));
            FLOAT4 k1 = RI_F(uKernel, SAMPLER, (int2)(index + 1, pos.y));
            FLOAT4 k2 = RI_F(uKernel, SAMPLER, (int2)(index + 2, pos.y));
            FLOAT4 k3 = RI_F(uKernel, SAMPLER, (int2)(index + 3, pos.y));

            FLOAT4 s0 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset, srcY));
            FLOAT4 s1 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 1, srcY));
            FLOAT4 s2 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 2, srcY));
            FLOAT4 s3 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 3, srcY));

            o0 = mad(s0.x, k0, o0);
            o0 = mad(s0.y, k1, o0);
            o0 = mad(s0.z, k2, o0);
            o0 = mad(s0.w, k3, o0);

            o1 = mad(s1.x, k0, o1);
            o1 = mad(s1.y, k1, o1);
            o1 = mad(s1.z, k2, o1);
            o1 = mad(s1.w, k3, o1);

            o2 = mad(s2.x, k0, o2);
            o2 = mad(s2.y, k1, o2);
            o2 = mad(s2.z, k2, o2);
            o2 = mad(s2.w, k3, o2);

            o3 = mad(s3.x, k0, o3);
            o3 = mad(s3.y, k1, o3);
            o3 = mad(s3.z, k2, o3);
            o3 = mad(s3.w, k3, o3);
        }

        __private int out_y_idx = mad24(pos_z, unitHeight, pos_y);
        __private int out_x_idx = mad24(pos_w, unitWidth, srcX);
        const int remain = unitWidth - srcX;
        if(remain >= 4){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
        }else if(remain == 3){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
        }else if(remain == 2){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
        }else if(remain == 1){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
        }
    }
}


__kernel void gemmWinogradW2(__read_only image2d_t uInput, __read_only image2d_t uKernel, __write_only image2d_t uOutput,
                   __private const int unitWidth, __private const int unitHeight, __private const int dstChannelC4, __private const int multiLength, __private const int alpha2) {
    
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int unitWidth8 = (unitWidth + 7) / 8;
    if (pos.x < unitWidth8 * unitHeight && pos.y < alpha2 * dstChannelC4) {
        
        const int pos_x = pos.x % unitWidth8;
        const int pos_y = pos.x / unitWidth8;
        const int pos_z = pos.y % dstChannelC4;
        const int pos_w = pos.y / dstChannelC4;

        FLOAT4 o0 = (FLOAT4)(0);
        FLOAT4 o1 = (FLOAT4)(0);
        FLOAT4 o2 = (FLOAT4)(0);
        FLOAT4 o3 = (FLOAT4)(0);
        FLOAT4 o4 = (FLOAT4)(0);
        FLOAT4 o5 = (FLOAT4)(0);
        FLOAT4 o6 = (FLOAT4)(0);
        FLOAT4 o7 = (FLOAT4)(0);
        int srcY = mad24(pos_w, unitHeight, pos_y);
        int srcX = pos_x << 3;

        for (int k = 0; k < multiLength; ++k) {
            __private int index = mul24(k, 4);
            __private int x_offset = mul24(k, unitWidth);
            FLOAT4 k0 = RI_F(uKernel, SAMPLER, (int2)(index, pos.y));
            FLOAT4 k1 = RI_F(uKernel, SAMPLER, (int2)(index + 1, pos.y));
            FLOAT4 k2 = RI_F(uKernel, SAMPLER, (int2)(index + 2, pos.y));
            FLOAT4 k3 = RI_F(uKernel, SAMPLER, (int2)(index + 3, pos.y));

            FLOAT4 s0 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset, srcY));
            FLOAT4 s1 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 1, srcY));
            FLOAT4 s2 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 2, srcY));
            FLOAT4 s3 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 3, srcY));
            FLOAT4 s4 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 4, srcY));
            FLOAT4 s5 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 5, srcY));
            FLOAT4 s6 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 6, srcY));
            FLOAT4 s7 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 7, srcY));

            o0 = mad(s0.x, k0, o0);
            o0 = mad(s0.y, k1, o0);
            o0 = mad(s0.z, k2, o0);
            o0 = mad(s0.w, k3, o0);

            o1 = mad(s1.x, k0, o1);
            o1 = mad(s1.y, k1, o1);
            o1 = mad(s1.z, k2, o1);
            o1 = mad(s1.w, k3, o1);

            o2 = mad(s2.x, k0, o2);
            o2 = mad(s2.y, k1, o2);
            o2 = mad(s2.z, k2, o2);
            o2 = mad(s2.w, k3, o2);

            o3 = mad(s3.x, k0, o3);
            o3 = mad(s3.y, k1, o3);
            o3 = mad(s3.z, k2, o3);
            o3 = mad(s3.w, k3, o3);
            
            o4 = mad(s4.x, k0, o4);
            o4 = mad(s4.y, k1, o4);
            o4 = mad(s4.z, k2, o4);
            o4 = mad(s4.w, k3, o4);

            o5 = mad(s5.x, k0, o5);
            o5 = mad(s5.y, k1, o5);
            o5 = mad(s5.z, k2, o5);
            o5 = mad(s5.w, k3, o5);

            o6 = mad(s6.x, k0, o6);
            o6 = mad(s6.y, k1, o6);
            o6 = mad(s6.z, k2, o6);
            o6 = mad(s6.w, k3, o6);

            o7 = mad(s7.x, k0, o7);
            o7 = mad(s7.y, k1, o7);
            o7 = mad(s7.z, k2, o7);
            o7 = mad(s7.w, k3, o7);
        }

        __private int out_y_idx = mad24(pos_z, unitHeight, pos_y);
        __private int out_x_idx = mad24(pos_w, unitWidth, srcX);
        const int remain = unitWidth - srcX;
        if(remain >= 8){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
            WI_F(uOutput, (int2)(out_x_idx + 4, out_y_idx), o4);
            WI_F(uOutput, (int2)(out_x_idx + 5, out_y_idx), o5);
            WI_F(uOutput, (int2)(out_x_idx + 6, out_y_idx), o6);
            WI_F(uOutput, (int2)(out_x_idx + 7, out_y_idx), o7);
        }else if(remain == 7){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
            WI_F(uOutput, (int2)(out_x_idx + 4, out_y_idx), o4);
            WI_F(uOutput, (int2)(out_x_idx + 5, out_y_idx), o5);
            WI_F(uOutput, (int2)(out_x_idx + 6, out_y_idx), o6);
        }else if(remain == 6){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
            WI_F(uOutput, (int2)(out_x_idx + 4, out_y_idx), o4);
            WI_F(uOutput, (int2)(out_x_idx + 5, out_y_idx), o5);
        }else if(remain == 5){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
            WI_F(uOutput, (int2)(out_x_idx + 4, out_y_idx), o4);
        }else if(remain == 4){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
        }else if(remain == 3){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
        }else if(remain == 2){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
        }else if(remain == 1){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
        }
    }
}

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
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
                        __global const float *dequantScaleOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
                        __global const float *dequantScaleOffset,
#else
                        __global const FLOAT *weight,
#endif
                        __read_only image2d_t bias,
                        __write_only image2d_t output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
                        ,__private const int blockDim
                        ,__private const int srcChannel
#endif
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); //cout/4, b
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    FLOAT4 out = RI_F(bias, SAMPLER, (int2)(pos.x, 0));

#if (defined USE_LOW_BIT_WEIGHT_INT8)
    int weight_offset = pos.x * 16;
    int weight_oc_offset = dstChannelC4 * 16;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = pos.x * 8;
    int weight_oc_offset = dstChannelC4 * 8;
#else
    int weight_offset = pos.x * 16;
    int weight_oc_offset = dstChannelC4 * 16;
#endif

    for (int k = 0; k < srcChannelC4; ++k) {
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
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
#endif
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(k, pos.y));
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * scale + offset;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
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
                
#else
        FLOAT16 weights = vload16(0, weight + weight_offset + k * weight_oc_offset);
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
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
                        __global const float *dequantScaleOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
                        __global const float *dequantScaleOffset,
#else
                        __global const FLOAT *weight,
#endif
                        __read_only image2d_t bias,
                        __write_only image2d_t output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
                        ,__private const int blockDim
                        ,__private const int srcChannel
#endif
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); //cout/4, b
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    int pos_x = pos.x << 2;
    int pos_y = pos.y << 1;

    FLOAT4 bias0 = RI_F(bias, SAMPLER, (int2)(pos.x, 0));
    FLOAT4 out0 = bias0, out1 = bias0;
    
#if (defined USE_LOW_BIT_WEIGHT_INT8)
    int weight_offset = pos.x * 16;
    int weight_oc_offset = dstChannelC4 * 16;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = pos.x * 8;
    int weight_oc_offset = dstChannelC4 * 8;
#else
    int weight_offset = pos.x * 16;
    int weight_oc_offset = dstChannelC4 * 16;
#endif

    for (int k = 0; k < srcChannelC4; ++k) {
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
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
#endif
        FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(k, pos_y));
        FLOAT4 in1 = RI_F(input, SAMPLER, (int2)(k, pos_y + 1));
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset)) * scale + offset;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
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
#else
        FLOAT16 weights = vload16(0, weight + weight_offset + k * weight_oc_offset);
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
