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

__kernel void gemm_conv_c4_buf(GLOBAL_SIZE_DIM2
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
                        __private const int batch,
                        __private const int height,
                        __private const int width) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
    const int out_b_idx = out_b_h_idx / height;
    const int out_h_idx = out_b_h_idx % height;

    FLOAT4 bias0 = vload4(out_c_idx, bias);
    FLOAT sum = 0;
    FLOAT4 out = 0;
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = ((out_b_idx * dstChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int wh = width * height * 4;
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 4 * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 4 * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    const FLOAT4 Scale = vload4(out_c_idx, dequantScale);
    const FLOAT4 Offset = vload4(out_c_idx, dequantOffset);
#endif
#ifdef INPUT_CHANNEL_LEAVE
    for (int k = 0; k < srcChannelC4/4 - 1; ++k) {
#else
    for (int k = 0; k < srcChannelC4/4; ++k) {
#endif
#ifdef WIDTH_HEIGHT_1
        FLOAT16 in = vload16(k, input + input_offset);
#else
        int k4 = k << 2;
        FLOAT16 in;
        in.s0123 = vload4(0, input + input_offset + k4 * wh);
        in.s4567 = vload4(0, input + input_offset + (k4 + 1) * wh);
        in.s89ab = vload4(0, input + input_offset + (k4 + 2) * wh);
        in.scdef = vload4(0, input + input_offset + (k4 + 3) * wh);
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights0 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        FLOAT16 weights1 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16));
        FLOAT16 weights2 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 32));
        FLOAT16 weights3 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 48));
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt40 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        uchar16 charWeightsInt41 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
        FLOAT16 weights0, weights1, weights2, weights3;
        {
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            charWeights0.s0 = (charWeightsInt40.s0 >> 4) - 8;
            charWeights0.s1 = (charWeightsInt40.s0 & 15) - 8;
            charWeights0.s2 = (charWeightsInt40.s1 >> 4) - 8;
            charWeights0.s3 = (charWeightsInt40.s1 & 15) - 8;
            charWeights0.s4 = (charWeightsInt40.s2 >> 4) - 8;
            charWeights0.s5 = (charWeightsInt40.s2 & 15) - 8;
            charWeights0.s6 = (charWeightsInt40.s3 >> 4) - 8;
            charWeights0.s7 = (charWeightsInt40.s3 & 15) - 8;
            charWeights0.s8 = (charWeightsInt40.s4 >> 4) - 8;
            charWeights0.s9 = (charWeightsInt40.s4 & 15) - 8;
            charWeights0.sa = (charWeightsInt40.s5 >> 4) - 8;
            charWeights0.sb = (charWeightsInt40.s5 & 15) - 8;
            charWeights0.sc = (charWeightsInt40.s6 >> 4) - 8;
            charWeights0.sd = (charWeightsInt40.s6 & 15) - 8;
            charWeights0.se = (charWeightsInt40.s7 >> 4) - 8;
            charWeights0.sf = (charWeightsInt40.s7 & 15) - 8;
            charWeights1.s0 = (charWeightsInt40.s8 >> 4) - 8;
            charWeights1.s1 = (charWeightsInt40.s8 & 15) - 8;
            charWeights1.s2 = (charWeightsInt40.s9 >> 4) - 8;
            charWeights1.s3 = (charWeightsInt40.s9 & 15) - 8;
            charWeights1.s4 = (charWeightsInt40.sa >> 4) - 8;
            charWeights1.s5 = (charWeightsInt40.sa & 15) - 8;
            charWeights1.s6 = (charWeightsInt40.sb >> 4) - 8;
            charWeights1.s7 = (charWeightsInt40.sb & 15) - 8;
            charWeights1.s8 = (charWeightsInt40.sc >> 4) - 8;
            charWeights1.s9 = (charWeightsInt40.sc & 15) - 8;
            charWeights1.sa = (charWeightsInt40.sd >> 4) - 8;
            charWeights1.sb = (charWeightsInt40.sd & 15) - 8;
            charWeights1.sc = (charWeightsInt40.se >> 4) - 8;
            charWeights1.sd = (charWeightsInt40.se & 15) - 8;
            charWeights1.se = (charWeightsInt40.sf >> 4) - 8;
            charWeights1.sf = (charWeightsInt40.sf & 15) - 8;
            weights0 = CONVERT_FLOAT16(charWeights0);
            weights1 = CONVERT_FLOAT16(charWeights1);
        }
        
        {
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            charWeights0.s0 = (charWeightsInt41.s0 >> 4) - 8;
            charWeights0.s1 = (charWeightsInt41.s0 & 15) - 8;
            charWeights0.s2 = (charWeightsInt41.s1 >> 4) - 8;
            charWeights0.s3 = (charWeightsInt41.s1 & 15) - 8;
            charWeights0.s4 = (charWeightsInt41.s2 >> 4) - 8;
            charWeights0.s5 = (charWeightsInt41.s2 & 15) - 8;
            charWeights0.s6 = (charWeightsInt41.s3 >> 4) - 8;
            charWeights0.s7 = (charWeightsInt41.s3 & 15) - 8;
            charWeights0.s8 = (charWeightsInt41.s4 >> 4) - 8;
            charWeights0.s9 = (charWeightsInt41.s4 & 15) - 8;
            charWeights0.sa = (charWeightsInt41.s5 >> 4) - 8;
            charWeights0.sb = (charWeightsInt41.s5 & 15) - 8;
            charWeights0.sc = (charWeightsInt41.s6 >> 4) - 8;
            charWeights0.sd = (charWeightsInt41.s6 & 15) - 8;
            charWeights0.se = (charWeightsInt41.s7 >> 4) - 8;
            charWeights0.sf = (charWeightsInt41.s7 & 15) - 8;
            charWeights1.s0 = (charWeightsInt41.s8 >> 4) - 8;
            charWeights1.s1 = (charWeightsInt41.s8 & 15) - 8;
            charWeights1.s2 = (charWeightsInt41.s9 >> 4) - 8;
            charWeights1.s3 = (charWeightsInt41.s9 & 15) - 8;
            charWeights1.s4 = (charWeightsInt41.sa >> 4) - 8;
            charWeights1.s5 = (charWeightsInt41.sa & 15) - 8;
            charWeights1.s6 = (charWeightsInt41.sb >> 4) - 8;
            charWeights1.s7 = (charWeightsInt41.sb & 15) - 8;
            charWeights1.s8 = (charWeightsInt41.sc >> 4) - 8;
            charWeights1.s9 = (charWeightsInt41.sc & 15) - 8;
            charWeights1.sa = (charWeightsInt41.sd >> 4) - 8;
            charWeights1.sb = (charWeightsInt41.sd & 15) - 8;
            charWeights1.sc = (charWeightsInt41.se >> 4) - 8;
            charWeights1.sd = (charWeightsInt41.se & 15) - 8;
            charWeights1.se = (charWeightsInt41.sf >> 4) - 8;
            charWeights1.sf = (charWeightsInt41.sf & 15) - 8;
            weights2 = CONVERT_FLOAT16(charWeights0);
            weights3 = CONVERT_FLOAT16(charWeights1);
        }
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
                
#else
        FLOAT16 weights0 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        FLOAT16 weights1 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
        FLOAT16 weights2 = vload16(0, weight + weight_offset + k * weight_oc_offset + 32);
        FLOAT16 weights3 = vload16(0, weight + weight_offset + k * weight_oc_offset + 48);
#endif
        
        out.s0 += dot(in.s0123, weights0.s0123);
        out.s0 += dot(in.s4567, weights0.s4567);
        out.s0 += dot(in.s89ab, weights0.s89ab);
        out.s0 += dot(in.scdef, weights0.scdef);
        
        out.s1 += dot(in.s0123, weights1.s0123);
        out.s1 += dot(in.s4567, weights1.s4567);
        out.s1 += dot(in.s89ab, weights1.s89ab);
        out.s1 += dot(in.scdef, weights1.scdef);
        
        out.s2 += dot(in.s0123, weights2.s0123);
        out.s2 += dot(in.s4567, weights2.s4567);
        out.s2 += dot(in.s89ab, weights2.s89ab);
        out.s2 += dot(in.scdef, weights2.scdef);
        
        out.s3 += dot(in.s0123, weights3.s0123);
        out.s3 += dot(in.s4567, weights3.s4567);
        out.s3 += dot(in.s89ab, weights3.s89ab);
        out.s3 += dot(in.scdef, weights3.scdef);
    }
#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC4/4 - 1;
        int k4 = k * 4;
        FLOAT16 in;
        in.s0123 = vload4(0, input + input_offset + k4 * wh);
        in.s4567 = k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0;
        in.s89ab = k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0;
        in.scdef = k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0;
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights0 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        FLOAT16 weights1 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16));
        FLOAT16 weights2 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 32));
        FLOAT16 weights3 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 48));
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt40 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        uchar16 charWeightsInt41 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
        FLOAT16 weights0, weights1, weights2, weights3;
        {
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            charWeights0.s0 = (charWeightsInt40.s0 >> 4) - 8;
            charWeights0.s1 = (charWeightsInt40.s0 & 15) - 8;
            charWeights0.s2 = (charWeightsInt40.s1 >> 4) - 8;
            charWeights0.s3 = (charWeightsInt40.s1 & 15) - 8;
            charWeights0.s4 = (charWeightsInt40.s2 >> 4) - 8;
            charWeights0.s5 = (charWeightsInt40.s2 & 15) - 8;
            charWeights0.s6 = (charWeightsInt40.s3 >> 4) - 8;
            charWeights0.s7 = (charWeightsInt40.s3 & 15) - 8;
            charWeights0.s8 = (charWeightsInt40.s4 >> 4) - 8;
            charWeights0.s9 = (charWeightsInt40.s4 & 15) - 8;
            charWeights0.sa = (charWeightsInt40.s5 >> 4) - 8;
            charWeights0.sb = (charWeightsInt40.s5 & 15) - 8;
            charWeights0.sc = (charWeightsInt40.s6 >> 4) - 8;
            charWeights0.sd = (charWeightsInt40.s6 & 15) - 8;
            charWeights0.se = (charWeightsInt40.s7 >> 4) - 8;
            charWeights0.sf = (charWeightsInt40.s7 & 15) - 8;
            charWeights1.s0 = (charWeightsInt40.s8 >> 4) - 8;
            charWeights1.s1 = (charWeightsInt40.s8 & 15) - 8;
            charWeights1.s2 = (charWeightsInt40.s9 >> 4) - 8;
            charWeights1.s3 = (charWeightsInt40.s9 & 15) - 8;
            charWeights1.s4 = (charWeightsInt40.sa >> 4) - 8;
            charWeights1.s5 = (charWeightsInt40.sa & 15) - 8;
            charWeights1.s6 = (charWeightsInt40.sb >> 4) - 8;
            charWeights1.s7 = (charWeightsInt40.sb & 15) - 8;
            charWeights1.s8 = (charWeightsInt40.sc >> 4) - 8;
            charWeights1.s9 = (charWeightsInt40.sc & 15) - 8;
            charWeights1.sa = (charWeightsInt40.sd >> 4) - 8;
            charWeights1.sb = (charWeightsInt40.sd & 15) - 8;
            charWeights1.sc = (charWeightsInt40.se >> 4) - 8;
            charWeights1.sd = (charWeightsInt40.se & 15) - 8;
            charWeights1.se = (charWeightsInt40.sf >> 4) - 8;
            charWeights1.sf = (charWeightsInt40.sf & 15) - 8;
            weights0 = CONVERT_FLOAT16(charWeights0);
            weights1 = CONVERT_FLOAT16(charWeights1);
        }
        
        {
            char16 charWeights0 = 0;
            char16 charWeights1 = 0;
            charWeights0.s0 = (charWeightsInt41.s0 >> 4) - 8;
            charWeights0.s1 = (charWeightsInt41.s0 & 15) - 8;
            charWeights0.s2 = (charWeightsInt41.s1 >> 4) - 8;
            charWeights0.s3 = (charWeightsInt41.s1 & 15) - 8;
            charWeights0.s4 = (charWeightsInt41.s2 >> 4) - 8;
            charWeights0.s5 = (charWeightsInt41.s2 & 15) - 8;
            charWeights0.s6 = (charWeightsInt41.s3 >> 4) - 8;
            charWeights0.s7 = (charWeightsInt41.s3 & 15) - 8;
            charWeights0.s8 = (charWeightsInt41.s4 >> 4) - 8;
            charWeights0.s9 = (charWeightsInt41.s4 & 15) - 8;
            charWeights0.sa = (charWeightsInt41.s5 >> 4) - 8;
            charWeights0.sb = (charWeightsInt41.s5 & 15) - 8;
            charWeights0.sc = (charWeightsInt41.s6 >> 4) - 8;
            charWeights0.sd = (charWeightsInt41.s6 & 15) - 8;
            charWeights0.se = (charWeightsInt41.s7 >> 4) - 8;
            charWeights0.sf = (charWeightsInt41.s7 & 15) - 8;
            charWeights1.s0 = (charWeightsInt41.s8 >> 4) - 8;
            charWeights1.s1 = (charWeightsInt41.s8 & 15) - 8;
            charWeights1.s2 = (charWeightsInt41.s9 >> 4) - 8;
            charWeights1.s3 = (charWeightsInt41.s9 & 15) - 8;
            charWeights1.s4 = (charWeightsInt41.sa >> 4) - 8;
            charWeights1.s5 = (charWeightsInt41.sa & 15) - 8;
            charWeights1.s6 = (charWeightsInt41.sb >> 4) - 8;
            charWeights1.s7 = (charWeightsInt41.sb & 15) - 8;
            charWeights1.s8 = (charWeightsInt41.sc >> 4) - 8;
            charWeights1.s9 = (charWeightsInt41.sc & 15) - 8;
            charWeights1.sa = (charWeightsInt41.sd >> 4) - 8;
            charWeights1.sb = (charWeightsInt41.sd & 15) - 8;
            charWeights1.sc = (charWeightsInt41.se >> 4) - 8;
            charWeights1.sd = (charWeightsInt41.se & 15) - 8;
            charWeights1.se = (charWeightsInt41.sf >> 4) - 8;
            charWeights1.sf = (charWeightsInt41.sf & 15) - 8;
            weights2 = CONVERT_FLOAT16(charWeights0);
            weights3 = CONVERT_FLOAT16(charWeights1);
        }
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
                
#else
        FLOAT16 weights0 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        FLOAT16 weights1 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
        FLOAT16 weights2 = vload16(0, weight + weight_offset + k * weight_oc_offset + 32);
        FLOAT16 weights3 = vload16(0, weight + weight_offset + k * weight_oc_offset + 48);
#endif
        
        out.s0 += dot(in.s0123, weights0.s0123);
        out.s0 += dot(in.s4567, weights0.s4567);
        out.s0 += dot(in.s89ab, weights0.s89ab);
        out.s0 += dot(in.scdef, weights0.scdef);
        
        out.s1 += dot(in.s0123, weights1.s0123);
        out.s1 += dot(in.s4567, weights1.s4567);
        out.s1 += dot(in.s89ab, weights1.s89ab);
        out.s1 += dot(in.scdef, weights1.scdef);
        
        out.s2 += dot(in.s0123, weights2.s0123);
        out.s2 += dot(in.s4567, weights2.s4567);
        out.s2 += dot(in.s89ab, weights2.s89ab);
        out.s2 += dot(in.scdef, weights2.scdef);
        
        out.s3 += dot(in.s0123, weights3.s0123);
        out.s3 += dot(in.s4567, weights3.s4567);
        out.s3 += dot(in.s89ab, weights3.s89ab);
        out.s3 += dot(in.scdef, weights3.scdef);
    }
#endif
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    out = bias0 + mad(out, Scale, sum * Offset);
#endif
#ifdef RELU
    out = fmax(out, (FLOAT4)0);
#endif

#ifdef RELU6
    out = clamp(out, (FLOAT4)0, (FLOAT4)6);
#endif

    vstore4(out, out_c_idx, output+out_offset);
}

__kernel void gemm_conv_c2_buf(GLOBAL_SIZE_DIM2
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
                        __private const int batch,
                        __private const int height,
                        __private const int width) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
    const int out_b_idx = out_b_h_idx / height;
    const int out_h_idx = out_b_h_idx % height;

    FLOAT2 bias0 = vload2(out_c_idx, bias);
    FLOAT sum = 0;
    FLOAT2 out = 0;
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = ((out_b_idx * dstChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int wh = width * height * 4;
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 2 * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 2 * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    const FLOAT2 Scale = vload2(out_c_idx, dequantScale);
    const FLOAT2 Offset = vload2(out_c_idx, dequantOffset);
#endif

#ifdef INPUT_CHANNEL_LEAVE
    for (int k = 0; k < srcChannelC4/4 - 1; ++k) {
#else
    for (int k = 0; k < srcChannelC4/4; ++k) {
#endif
#ifdef WIDTH_HEIGHT_1
        FLOAT16 in = vload16(k, input + input_offset);
#else
        FLOAT16 in;
        int k4 = k << 2;
        in.s0123 = vload4(0, input + input_offset + k4 * wh);
        in.s4567 = vload4(0, input + input_offset + (k4 + 1) * wh);
        in.s89ab = vload4(0, input + input_offset + (k4 + 2) * wh);
        in.scdef = vload4(0, input + input_offset + (k4 + 3) * wh);
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights0 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        FLOAT16 weights1 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16));
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt4 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        char16 charWeights0 = 0;
        char16 charWeights1 = 0;
        charWeights0.s0 = (charWeightsInt4.s0 >> 4) - 8;
        charWeights0.s1 = (charWeightsInt4.s0 & 15) - 8;
        charWeights0.s2 = (charWeightsInt4.s1 >> 4) - 8;
        charWeights0.s3 = (charWeightsInt4.s1 & 15) - 8;
        charWeights0.s4 = (charWeightsInt4.s2 >> 4) - 8;
        charWeights0.s5 = (charWeightsInt4.s2 & 15) - 8;
        charWeights0.s6 = (charWeightsInt4.s3 >> 4) - 8;
        charWeights0.s7 = (charWeightsInt4.s3 & 15) - 8;
        charWeights0.s8 = (charWeightsInt4.s4 >> 4) - 8;
        charWeights0.s9 = (charWeightsInt4.s4 & 15) - 8;
        charWeights0.sa = (charWeightsInt4.s5 >> 4) - 8;
        charWeights0.sb = (charWeightsInt4.s5 & 15) - 8;
        charWeights0.sc = (charWeightsInt4.s6 >> 4) - 8;
        charWeights0.sd = (charWeightsInt4.s6 & 15) - 8;
        charWeights0.se = (charWeightsInt4.s7 >> 4) - 8;
        charWeights0.sf = (charWeightsInt4.s7 & 15) - 8;
        
        charWeights1.s0 = (charWeightsInt4.s8 >> 4) - 8;
        charWeights1.s1 = (charWeightsInt4.s8 & 15) - 8;
        charWeights1.s2 = (charWeightsInt4.s9 >> 4) - 8;
        charWeights1.s3 = (charWeightsInt4.s9 & 15) - 8;
        charWeights1.s4 = (charWeightsInt4.sa >> 4) - 8;
        charWeights1.s5 = (charWeightsInt4.sa & 15) - 8;
        charWeights1.s6 = (charWeightsInt4.sb >> 4) - 8;
        charWeights1.s7 = (charWeightsInt4.sb & 15) - 8;
        charWeights1.s8 = (charWeightsInt4.sc >> 4) - 8;
        charWeights1.s9 = (charWeightsInt4.sc & 15) - 8;
        charWeights1.sa = (charWeightsInt4.sd >> 4) - 8;
        charWeights1.sb = (charWeightsInt4.sd & 15) - 8;
        charWeights1.sc = (charWeightsInt4.se >> 4) - 8;
        charWeights1.sd = (charWeightsInt4.se & 15) - 8;
        charWeights1.se = (charWeightsInt4.sf >> 4) - 8;
        charWeights1.sf = (charWeightsInt4.sf & 15) - 8;
        FLOAT16 weights0 = CONVERT_FLOAT16(charWeights0);
        FLOAT16 weights1 = CONVERT_FLOAT16(charWeights1);
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
                
#else
        FLOAT16 weights0 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        FLOAT16 weights1 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
#endif
        out.s0 += dot(in.s0123, weights0.s0123);
        out.s0 += dot(in.s4567, weights0.s4567);
        out.s0 += dot(in.s89ab, weights0.s89ab);
        out.s0 += dot(in.scdef, weights0.scdef);
        out.s1 += dot(in.s0123, weights1.s0123);
        out.s1 += dot(in.s4567, weights1.s4567);
        out.s1 += dot(in.s89ab, weights1.s89ab);
        out.s1 += dot(in.scdef, weights1.scdef);
    }

#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC4/4 - 1;
        FLOAT16 in = 0;
        int k4 = k * 4;
        in.s0123 = vload4(0, input + input_offset + k4 * wh);
        in.s4567 = k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0;
        in.s89ab = k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0;
        in.scdef = k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0;
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights0 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        FLOAT16 weights1 = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset + 16));
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt4 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        char16 charWeights0 = 0;
        char16 charWeights1 = 0;
        charWeights0.s0 = (charWeightsInt4.s0 >> 4) - 8;
        charWeights0.s1 = (charWeightsInt4.s0 & 15) - 8;
        charWeights0.s2 = (charWeightsInt4.s1 >> 4) - 8;
        charWeights0.s3 = (charWeightsInt4.s1 & 15) - 8;
        charWeights0.s4 = (charWeightsInt4.s2 >> 4) - 8;
        charWeights0.s5 = (charWeightsInt4.s2 & 15) - 8;
        charWeights0.s6 = (charWeightsInt4.s3 >> 4) - 8;
        charWeights0.s7 = (charWeightsInt4.s3 & 15) - 8;
        charWeights0.s8 = (charWeightsInt4.s4 >> 4) - 8;
        charWeights0.s9 = (charWeightsInt4.s4 & 15) - 8;
        charWeights0.sa = (charWeightsInt4.s5 >> 4) - 8;
        charWeights0.sb = (charWeightsInt4.s5 & 15) - 8;
        charWeights0.sc = (charWeightsInt4.s6 >> 4) - 8;
        charWeights0.sd = (charWeightsInt4.s6 & 15) - 8;
        charWeights0.se = (charWeightsInt4.s7 >> 4) - 8;
        charWeights0.sf = (charWeightsInt4.s7 & 15) - 8;
        
        charWeights1.s0 = (charWeightsInt4.s8 >> 4) - 8;
        charWeights1.s1 = (charWeightsInt4.s8 & 15) - 8;
        charWeights1.s2 = (charWeightsInt4.s9 >> 4) - 8;
        charWeights1.s3 = (charWeightsInt4.s9 & 15) - 8;
        charWeights1.s4 = (charWeightsInt4.sa >> 4) - 8;
        charWeights1.s5 = (charWeightsInt4.sa & 15) - 8;
        charWeights1.s6 = (charWeightsInt4.sb >> 4) - 8;
        charWeights1.s7 = (charWeightsInt4.sb & 15) - 8;
        charWeights1.s8 = (charWeightsInt4.sc >> 4) - 8;
        charWeights1.s9 = (charWeightsInt4.sc & 15) - 8;
        charWeights1.sa = (charWeightsInt4.sd >> 4) - 8;
        charWeights1.sb = (charWeightsInt4.sd & 15) - 8;
        charWeights1.sc = (charWeightsInt4.se >> 4) - 8;
        charWeights1.sd = (charWeightsInt4.se & 15) - 8;
        charWeights1.se = (charWeightsInt4.sf >> 4) - 8;
        charWeights1.sf = (charWeightsInt4.sf & 15) - 8;
        FLOAT16 weights0 = CONVERT_FLOAT16(charWeights0);
        FLOAT16 weights1 = CONVERT_FLOAT16(charWeights1);
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
                
#else
        FLOAT16 weights0 = vload16(0, weight + weight_offset + k * weight_oc_offset);
        FLOAT16 weights1 = vload16(0, weight + weight_offset + k * weight_oc_offset + 16);
#endif
        out.s0 += dot(in.s0123, weights0.s0123);
        out.s0 += dot(in.s4567, weights0.s4567);
        out.s0 += dot(in.s89ab, weights0.s89ab);
        out.s0 += dot(in.scdef, weights0.scdef);
        out.s1 += dot(in.s0123, weights1.s0123);
        out.s1 += dot(in.s4567, weights1.s4567);
        out.s1 += dot(in.s89ab, weights1.s89ab);
        out.s1 += dot(in.scdef, weights1.scdef);
    }
#endif
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    out = bias0 + mad(out, Scale, sum * Offset);
#endif
#ifdef RELU
    out = fmax(out, (FLOAT2)0);
#endif

#ifdef RELU6
    out = clamp(out, (FLOAT2)0, (FLOAT2)6);
#endif

    vstore2(out, out_c_idx, output+out_offset);
}

__kernel void gemm_conv_c1_buf(GLOBAL_SIZE_DIM2
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
                        __private const int batch,
                        __private const int height,
                        __private const int width) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    UNIFORM_BOUNDRY_CHECK(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / width;
    const int out_w_idx = out_c_w_idx % width;
    const int out_b_idx = out_b_h_idx / height;
    const int out_h_idx = out_b_h_idx % height;

    FLOAT bias0 = bias[out_c_idx];
    FLOAT sum = 0;
    FLOAT out = 0;
    
    int input_offset = ((out_b_idx * srcChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int out_offset = ((out_b_idx * dstChannelC4 * height + out_h_idx) * width + out_w_idx) * 4;
    int wh = width * height * 4;
#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_offset = out_c_idx * 8;
    int weight_oc_offset = dstChannelC4 * 32;
#else
    int weight_offset = out_c_idx * 16;
    int weight_oc_offset = dstChannelC4 * 64;
#endif
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    const FLOAT Scale = dequantScale[out_c_idx];
    const FLOAT Offset = dequantOffset[out_c_idx];
#endif
#ifdef INPUT_CHANNEL_LEAVE
    for (int k = 0; k < srcChannelC4/4 - 1; ++k) {
#else
    for (int k = 0; k < srcChannelC4/4; ++k) {
#endif
#ifdef WIDTH_HEIGHT_1
        FLOAT16 in = vload16(k, input + input_offset);
#else
        FLOAT16 in;
        int k4 = k << 2;
        in.s0123 = vload4(0, input + input_offset + k4 * wh);
        in.s4567 = vload4(0, input + input_offset + (k4 + 1) * wh);
        in.s89ab = vload4(0, input + input_offset + (k4 + 2) * wh);
        in.scdef = vload4(0, input + input_offset + (k4 + 3) * wh);
#endif
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
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
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#else
        FLOAT16 weights = vload16(0, weight + weight_offset + k * weight_oc_offset);
#endif
        out += dot(in.s0123, weights.s0123);
        out += dot(in.s4567, weights.s4567);
        out += dot(in.s89ab, weights.s89ab);
        out += dot(in.scdef, weights.scdef);
    }
#ifdef INPUT_CHANNEL_LEAVE
    {
        int k = srcChannelC4/4 - 1;
        FLOAT16 in = 0;
        int k4 = k * 4;
        in.s0123 = vload4(0, input + input_offset + k4 * wh);
        in.s4567 = k4 + 1 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 1) * wh) : (FLOAT4)0;
        in.s89ab = k4 + 2 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 2) * wh) : (FLOAT4)0;
        in.scdef = k4 + 3 < srcChannelC4 ? vload4(0, input + input_offset + (k4 + 3) * wh) : (FLOAT4)0;
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, weight + weight_offset + k * weight_oc_offset));
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
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
        sum += in.s0 + in.s1 + in.s2 + in.s3 + in.s4 + in.s5 + in.s6 + in.s7 + in.s8 + in.s9 + in.sa + in.sb + in.sc + in.sd + in.se + in.sf;
#else
        FLOAT16 weights = vload16(0, weight + weight_offset + k * weight_oc_offset);
#endif
        out += dot(in.s0123, weights.s0123);
        out += dot(in.s4567, weights.s4567);
        out += dot(in.s89ab, weights.s89ab);
        out += dot(in.scdef, weights.scdef);
    }
#endif
    
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
    out = bias0 + mad(out, Scale, sum * Offset);
#endif
#ifdef RELU
    out = fmax(out, )0);
#endif

#ifdef RELU6
    out = clamp(out, 0, 6);
#endif

    //vstore4(out, pos.x, output+out_offset);
    output[out_offset + out_c_idx] = out;
}
