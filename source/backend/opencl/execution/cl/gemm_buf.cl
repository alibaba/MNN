#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

__kernel void gemm_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output,
                        __private const int width,//UP_DIV(wUnit*hUnit,4)
                        __private const int height,//dstChannelC4
                        __private const int srcChannelC4,
                        __private const int alpha2) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int pos_x = pos.x % width;
    const int pos_y = pos.x / width;
    const int pos_z = pos.y;

    FLOAT16 o = (FLOAT16)0;
    
    int kenerlY   = mad24(pos_z, height, pos_y);
    int srcY      = mad24(pos_z, width, pos_x);

    for (int k = 0; k < srcChannelC4; ++k) {
        __private int index = mul24(k, 4);
        
        //NHWC  [1, 1, alpha2*height, srcChannelC4*4] x 4
        //index:[0, 0, pos_z*width+pos_y,    index+0]
        //int inp1_offset = (((k * (alpha2*height) + kenerlY) * (srcChannelC4*4) + index)*4 + 0)*4;
        
        FLOAT16 k_v16 = vload16(kenerlY*(srcChannelC4) + k, input1);
        
        //NC4HW4 [alpha*alpha, srcChannelC4, width, 4] x 4
        //index: [pos_z,       k,            pos_x, 0]
        
        FLOAT16 s = vload16(((pos_z*srcChannelC4 + k) * width + pos_x), input0);

        o = mad((FLOAT16)((FLOAT4)s.s0, (FLOAT4)s.s4, (FLOAT4)s.s8, (FLOAT4)s.sc), (FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o);
        o = mad((FLOAT16)((FLOAT4)s.s1, (FLOAT4)s.s5, (FLOAT4)s.s9, (FLOAT4)s.sd), (FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o);
        o = mad((FLOAT16)((FLOAT4)s.s2, (FLOAT4)s.s6, (FLOAT4)s.sa, (FLOAT4)s.se), (FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o);
        o = mad((FLOAT16)((FLOAT4)s.s3, (FLOAT4)s.s7, (FLOAT4)s.sb, (FLOAT4)s.sf), (FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o);
    }

    __private int out_y_idx = mul24(pos_y, 4);
    //NC4HW4 [dstChannelC4, alpha2, 4, UP_DIV(wUnit*hUnit,4)] x 4

    //index: [pos_y,  pos_z,  0, pos_x]
    int out_offset = (((pos_y * alpha2 + pos_z) * 4 + 0) * width + pos_x) * 4;

    vstore4(o.s0123, 0, output+out_offset);
    vstore4(o.s4567, 0, output+out_offset+4*width);
    vstore4(o.s89ab, 0, output+out_offset+8*width);
    vstore4(o.scdef, 0, output+out_offset+12*width);
}



__kernel void gemm_buf2(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output,
                        __private const int width,//UP_DIV(wUnit*hUnit,8)
                        __private const int height,//dstChannelC4
                        __private const int srcChannelC4,
                        __private const int alpha2) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int width_block = (width+1) >> 1;
    const int pos_x = (pos.x % width_block) << 1;
    const int pos_y = pos.x / width_block;
    const int pos_z = pos.y;

    FLOAT16 o0 = (FLOAT16)0;
    FLOAT16 o1 = (FLOAT16)0;

    const int kenerlY   = mad24(pos_z, height, pos_y);
    const int kernel_base = mul24(kenerlY, srcChannelC4);
    const int inp_base = (pos_z*srcChannelC4 + 0) * width + pos_x;
    
    for (int k = 0; k < srcChannelC4; ++k) {
        //NHWC  [1, 1, alpha2*height, srcChannelC4*4] x 4
        //index:[0, 0, pos_z*width+pos_y,    index+0]
        //int inp1_offset = (((k * (alpha2*height) + kenerlY) * (srcChannelC4*4) + index)*4 + 0)*4;
        
        FLOAT16 k_v16 = vload16(kernel_base + k, input1);
        
        //NC4HW4 [alpha*alpha, srcChannelC4, width, 4] x 4
        //index: [pos_z,       k,            pos_x, 0]
        
        const int inp_offset = mad24(k, width, inp_base);
        FLOAT16 s = vload16(inp_offset, input0);

        o0 = mad((FLOAT16)((FLOAT4)s.s0, (FLOAT4)s.s4, (FLOAT4)s.s8, (FLOAT4)s.sc), (FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o0);
        o0 = mad((FLOAT16)((FLOAT4)s.s1, (FLOAT4)s.s5, (FLOAT4)s.s9, (FLOAT4)s.sd), (FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o0);
        o0 = mad((FLOAT16)((FLOAT4)s.s2, (FLOAT4)s.s6, (FLOAT4)s.sa, (FLOAT4)s.se), (FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o0);
        o0 = mad((FLOAT16)((FLOAT4)s.s3, (FLOAT4)s.s7, (FLOAT4)s.sb, (FLOAT4)s.sf), (FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o0);
        
        s = vload16(inp_offset + 1, input0);
        o1 = mad((FLOAT16)((FLOAT4)s.s0, (FLOAT4)s.s4, (FLOAT4)s.s8, (FLOAT4)s.sc), (FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o1);
        o1 = mad((FLOAT16)((FLOAT4)s.s1, (FLOAT4)s.s5, (FLOAT4)s.s9, (FLOAT4)s.sd), (FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o1);
        o1 = mad((FLOAT16)((FLOAT4)s.s2, (FLOAT4)s.s6, (FLOAT4)s.sa, (FLOAT4)s.se), (FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o1);
        o1 = mad((FLOAT16)((FLOAT4)s.s3, (FLOAT4)s.s7, (FLOAT4)s.sb, (FLOAT4)s.sf), (FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o1);
    }

    __private int out_y_idx = mul24(pos_y, 4);
    //NC4HW4 [dstChannelC4, alpha2, 4, UP_DIV(wUnit*hUnit,4)] x 4

    //index: [pos_y,  pos_z,  0, pos_x]
    int out_offset = (((pos_y * alpha2 + pos_z) * 4 + 0) * width + pos_x) * 4;

    vstore4(o0.s0123, 0, output+out_offset);
    vstore4(o0.s4567, 0, output+out_offset+4*width);
    vstore4(o0.s89ab, 0, output+out_offset+8*width);
    vstore4(o0.scdef, 0, output+out_offset+12*width);
    
    if(pos_x + 1 >= width) return;
    vstore4(o1.s0123, 1, output+out_offset);
    vstore4(o1.s4567, 1, output+out_offset+4*width);
    vstore4(o1.s89ab, 1, output+out_offset+8*width);
    vstore4(o1.scdef, 1, output+out_offset+12*width);
}


__kernel void gemm_conv_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
                        __global const FLOAT *dequantScale,
                        __global const FLOAT *dequantOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
                        __global const FLOAT *dequantScale,
                        __global const FLOAT *dequantOffset,
#else
                        __global const FLOAT *weight,
#endif
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); //cout/4, b
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    int pos_x = pos.x << 2;

    FLOAT4 bias0 = vload4(0, bias + pos_x);
    FLOAT sum = 0;
    FLOAT4 out = 0;
    
    int input_offset = pos.y * srcChannelC4 * 4;
    int out_offset = pos.y * dstChannelC4 * 4;
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = pos.x * 8;
    int weight_oc_offset = dstChannelC4 * 8;
#else
    int weight_offset = pos.x * 16;
    int weight_oc_offset = dstChannelC4 * 16;
#endif
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    const FLOAT4 Scale = vload4(pos.x, dequantScale);
    const FLOAT4 Offset = vload4(pos.x, dequantOffset);
#endif

    for (int k = 0; k < srcChannelC4; ++k) {
        FLOAT4 in = vload4(k, input + input_offset);
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        sum += in.x + in.y + in.z + in.w;
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
        FLOAT16 weights = CONVERT_FLOAT16(charWeights);
        sum += in.x + in.y + in.z + in.w;
                
#else
        FLOAT16 weights = vload16(0, weight + weight_offset + k * weight_oc_offset);
#endif
        
        out = mad((FLOAT4)in.x, (FLOAT4)weights.s0123, out);
        out = mad((FLOAT4)in.y, (FLOAT4)weights.s4567, out);
        out = mad((FLOAT4)in.z, (FLOAT4)weights.s89ab, out);
        out = mad((FLOAT4)in.w, (FLOAT4)weights.scdef, out);
    }
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    out = bias0 + mad(out, Scale, sum * Offset);
#endif
#ifdef RELU
    out = fmax(out, (FLOAT4)0);
#endif

#ifdef RELU6
    out = clamp(out, (FLOAT4)0, (FLOAT4)6);
#endif

    vstore4(out, pos.x, output+out_offset);
}

__kernel void gemm_conv_b2_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT *input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char *weight,
                        __global const FLOAT *dequantScale,
                        __global const FLOAT *dequantOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar *weight,
                        __global const FLOAT *dequantScale,
                        __global const FLOAT *dequantOffset,
#else
                        __global const FLOAT *weight,
#endif
                        __global const FLOAT *bias,
                        __global FLOAT *output,
                        __private const int dstChannelC4,
                        __private const int srcChannelC4,
                        __private const int batch) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); //cout/4, b
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    int pos_x = pos.x << 2;
    int pos_y = pos.y << 1;

    FLOAT4 bias0 = vload4(0, bias + pos_x);
    FLOAT sum0 = 0, sum1 = 0;
    FLOAT4 out0 = (FLOAT4)0, out1 = (FLOAT4)0;
    
    int input_offset = pos_y * srcChannelC4 * 4;
    int out_offset = pos_y * dstChannelC4 * 4;
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = pos.x * 8;
    int weight_oc_offset = dstChannelC4 * 8;
#else
    int weight_offset = pos.x * 16;
    int weight_oc_offset = dstChannelC4 * 16;
#endif
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    const FLOAT4 Scale = vload4(pos.x, dequantScale);
    const FLOAT4 Offset = vload4(pos.x, dequantOffset);
#endif

    for (int k = 0; k < srcChannelC4; ++k) {
        FLOAT4 in0 = vload4(k, input + input_offset);
        FLOAT4 in1 = vload4(k, input + input_offset + srcChannelC4 * 4);
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        sum0 += in0.x + in0.y + in0.z + in0.w;
        sum1 += in1.x + in1.y + in1.z + in1.w;
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
        FLOAT16 weights = CONVERT_FLOAT16(charWeights);
        sum0 += in0.x + in0.y + in0.z + in0.w;
        sum1 += in1.x + in1.y + in1.z + in1.w;
                
#else
        FLOAT16 weights = vload16(0, weight + weight_offset + k * weight_oc_offset);
#endif
        
        out0 = mad((FLOAT4)in0.x, (FLOAT4)weights.s0123, out0);
        out0 = mad((FLOAT4)in0.y, (FLOAT4)weights.s4567, out0);
        out0 = mad((FLOAT4)in0.z, (FLOAT4)weights.s89ab, out0);
        out0 = mad((FLOAT4)in0.w, (FLOAT4)weights.scdef, out0);
        
        out1 = mad((FLOAT4)in1.x, (FLOAT4)weights.s0123, out1);
        out1 = mad((FLOAT4)in1.y, (FLOAT4)weights.s4567, out1);
        out1 = mad((FLOAT4)in1.z, (FLOAT4)weights.s89ab, out1);
        out1 = mad((FLOAT4)in1.w, (FLOAT4)weights.scdef, out1);
    }
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    out0 = bias0 + mad(out0, Scale, sum0 * Offset);
    out1 = bias0 + mad(out1, Scale, sum1 * Offset);
#endif
#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
#endif

    vstore4(out0, pos.x, output+out_offset);
    if(pos_y + 1 < batch)
        vstore4(out1, pos.x, output+out_offset+dstChannelC4 * 4);
}
