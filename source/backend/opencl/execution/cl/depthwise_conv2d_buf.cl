#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define READ_INPUT_IMAGE(i, base)                                                                         \
    int inOffset##i = inWidthOffset##i + base;                                                           \
    inOffset##i =                                                                                   \
        select(inCurIdx + inOffset##i, -1, (inOffset##i < 0 || inOffset##i >= inputShape.y)); \
    in_c_block_idx = inOffset##i / inputShape.y; \
    in_w_idx = inOffset##i % inputShape.y; \
    inpOffset = (((in_b_idx*channelBlocks + in_c_block_idx)*inputShape.x + in_h_idx)* inputShape.y + in_w_idx)*4; \
    inValue##i = ((inOffset##i)==-1 || inHeightIdx == -1) ? (FLOAT4)0 : vload4(0, input+inpOffset);


#define CALCULATE_OUTPUT(i)                  \
    outValue##i = mad(inValue##i.x, weights0, outValue##i); \
    outValue##i = mad(inValue##i.y, weights1, outValue##i); \
    outValue##i = mad(inValue##i.z, weights2, outValue##i); \
    outValue##i = mad(inValue##i.w, weights3, outValue##i);

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }


__kernel
void depthwise_conv2d_c4h1w4(GLOBAL_SIZE_2_DIMS __global const FLOAT *input,
                                  __global const FLOAT *filter,
                                  __global const FLOAT *bias,
                                  __global FLOAT *output,
                                  __private const int2 in_hw,
                                  __private const int channel,
                                  __private const int2 out_hw,
                                  __private const int2 filter_hw,
                                  __private const int2 pad_hw,
                                  __private const int2 dilate_hw,
                                  __private const int2 stride_hw,
                                  __private const int out_w_blocks,
                                  __private const int c_blocks) {

    const int out_c_w_idx = get_global_id(0);// oc/4 * ow/4
    const int out_b_h_idx  = get_global_id(1);// b * h
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int b_idx = out_b_h_idx / out_hw.x;
    const int out_h_idx = out_b_h_idx % out_hw.x;

    FLOAT4 outValue0 = vload4(c_idx, bias);
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;

    const int out_w4_idx = out_w_idx << 2;
    const int in_w_start_0 = out_w4_idx * stride_hw.y - pad_hw.y;
    const int in_w_start_1 = in_w_start_0 + stride_hw.y;
    const int in_w_start_2 = in_w_start_1 + stride_hw.y;
    const int in_w_start_3 = in_w_start_2 + stride_hw.y;

    const int in_h_start = out_h_idx * stride_hw.x - pad_hw.x;
    
    for (int kh = 0; kh < filter_hw.x; kh++) {
        const int in_h_cur = in_h_start + kh * dilate_hw.x;
        if(in_h_cur < 0 || in_h_cur >= in_hw.x) continue;
        
        int inp_offset = (((b_idx*c_blocks + c_idx)*in_hw.x + in_h_cur)* in_hw.y + in_w_start_0)*4;
        for (int kw = 0; kw < filter_hw.y; kw++) {
            const int filter_idx = mad24(kh, filter_hw.y, kw);
            const int kw_dilate = kw * dilate_hw.y;
            FLOAT4 inValue0 = (in_w_start_0+kw_dilate < 0 || in_w_start_0+kw_dilate >= in_hw.y) ? (FLOAT4)0 : vload4(kw_dilate+0, input+inp_offset);
            FLOAT4 inValue1 = (in_w_start_1+kw_dilate < 0 || in_w_start_1+kw_dilate >= in_hw.y) ? (FLOAT4)0 : vload4(kw_dilate+1*stride_hw.y, input+inp_offset);
            FLOAT4 inValue2 = (in_w_start_2+kw_dilate < 0 || in_w_start_2+kw_dilate >= in_hw.y) ? (FLOAT4)0 : vload4(kw_dilate+2*stride_hw.y, input+inp_offset);
            FLOAT4 inValue3 = (in_w_start_3+kw_dilate < 0 || in_w_start_3+kw_dilate >= in_hw.y) ? (FLOAT4)0 : vload4(kw_dilate+3*stride_hw.y, input+inp_offset);

            //NC4HW4 [1, filterShape.x*filterShape.y, 1, channelBlocks] x oc4
            //index: [0, filterIdx,                   0, inChannelBlockIdx]
            FLOAT4 weights = vload4(0, filter+(filter_idx*c_blocks+c_idx)*4);

            outValue0 = mad(inValue0, weights, outValue0);
            outValue1 = mad(inValue1, weights, outValue1);
            outValue2 = mad(inValue2, weights, outValue2);
            outValue3 = mad(inValue3, weights, outValue3);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((b_idx*c_blocks + c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w4_idx)*4;

    const int remain     = out_hw.y - out_w4_idx;
    if (remain >= 4) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
        vstore4(outValue2, 2, output+out_offset);
        vstore4(outValue3, 3, output+out_offset);
    } else if (remain == 3) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
        vstore4(outValue2, 2, output+out_offset);
    } else if (remain == 2) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(outValue0, 0, output+out_offset);
    }
    
}

__kernel
void depthwise_conv2d_c4h1w2(GLOBAL_SIZE_2_DIMS __global const FLOAT *input,
                                  __global const FLOAT *filter,
                                  __global const FLOAT *bias,
                                  __global FLOAT *output,
                                  __private const int2 in_hw,
                                  __private const int channel,
                                  __private const int2 out_hw,
                                  __private const int2 filter_hw,
                                  __private const int2 pad_hw,
                                  __private const int2 dilate_hw,
                                  __private const int2 stride_hw,
                                  __private const int out_w_blocks,
                                  __private const int c_blocks) {

    const int out_c_w_idx = get_global_id(0);// oc/4 * ow/4
    const int out_b_h_idx  = get_global_id(1);// b * h
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int b_idx = out_b_h_idx / out_hw.x;
    const int out_h_idx = out_b_h_idx % out_hw.x;

    FLOAT4 outValue0 = vload4(c_idx, bias);
    FLOAT4 outValue1 = outValue0;

    const int out_w2_idx = out_w_idx << 1;
    const int in_w_start_0 = out_w2_idx * stride_hw.y - pad_hw.y;
    const int in_w_start_1 = in_w_start_0 + stride_hw.y;
    
    const int in_h_start = out_h_idx * stride_hw.x - pad_hw.x;
    
    for (int kh = 0; kh < filter_hw.x; kh++) {
        const int in_h_cur = in_h_start + kh * dilate_hw.x;
        if(in_h_cur < 0 || in_h_cur >= in_hw.x) continue;
        
        int inp_offset = (((b_idx*c_blocks + c_idx)*in_hw.x + in_h_cur)* in_hw.y + in_w_start_0)*4;
        for (int kw = 0; kw < filter_hw.y; kw++) {
            const int filter_idx = mad24(kh, filter_hw.y, kw);
            const int kw_dilate = kw * dilate_hw.y;
            FLOAT4 inValue0 = (in_w_start_0+kw_dilate < 0 || in_w_start_0+kw_dilate >= in_hw.y) ? (FLOAT4)0 : vload4(kw_dilate+0, input+inp_offset);
            FLOAT4 inValue1 = (in_w_start_1+kw_dilate < 0 || in_w_start_1+kw_dilate >= in_hw.y) ? (FLOAT4)0 : vload4(kw_dilate+1*stride_hw.y, input+inp_offset);

            //NC4HW4 [1, filterShape.x*filterShape.y, 1, channelBlocks] x oc4
            //index: [0, filterIdx,                   0, inChannelBlockIdx]
            FLOAT4 weights = vload4(0, filter+(filter_idx*c_blocks+c_idx)*4);

            outValue0 = mad(inValue0, weights, outValue0);
            outValue1 = mad(inValue1, weights, outValue1);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((b_idx*c_blocks + c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w2_idx)*4;

    const int remain     = out_hw.y - out_w2_idx;
    if (remain >= 2) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(outValue0, 0, output+out_offset);
    }
    
}

__kernel
void depthwise_conv2d_c4h1w1(GLOBAL_SIZE_2_DIMS __global const FLOAT *input,
                                  __global const FLOAT *filter,
                                  __global const FLOAT *bias,
                                  __global FLOAT *output,
                                  __private const int2 in_hw,
                                  __private const int channel,
                                  __private const int2 out_hw,
                                  __private const int2 filter_hw,
                                  __private const int2 pad_hw,
                                  __private const int2 dilate_hw,
                                  __private const int2 stride_hw,
                                  __private const int out_w_blocks,
                                  __private const int c_blocks) {

    const int out_c_w_idx = get_global_id(0);// oc/4 * ow/4
    const int out_b_h_idx  = get_global_id(1);// b * h
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int b_idx = out_b_h_idx / out_hw.x;
    const int out_h_idx = out_b_h_idx % out_hw.x;

    FLOAT4 outValue0 = vload4(c_idx, bias);
    FLOAT4 outValue1 = outValue0;

    const int in_w_start_0 = out_w_idx * stride_hw.y - pad_hw.y;
    const int in_h_start = out_h_idx * stride_hw.x - pad_hw.x;
    
    for (int kh = 0; kh < filter_hw.x; kh++) {
        const int in_h_cur = in_h_start + kh * dilate_hw.x;
        if(in_h_cur < 0 || in_h_cur >= in_hw.x) continue;
        
        int inp_offset = (((b_idx*c_blocks + c_idx)*in_hw.x + in_h_cur)* in_hw.y + in_w_start_0)*4;
        for (int kw = 0; kw < filter_hw.y; kw++) {
            const int filter_idx = mad24(kh, filter_hw.y, kw);
            const int kw_dilate = kw * dilate_hw.y;
            FLOAT4 inValue0 = (in_w_start_0+kw_dilate < 0 || in_w_start_0+kw_dilate >= in_hw.y) ? (FLOAT4)0 : vload4(kw_dilate+0, input+inp_offset);

            //NC4HW4 [1, filterShape.x*filterShape.y, 1, channelBlocks] x oc4
            //index: [0, filterIdx,                   0, inChannelBlockIdx]
            FLOAT4 weights = vload4(0, filter+(filter_idx*c_blocks+c_idx)*4);

            outValue0 = mad(inValue0, weights, outValue0);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((b_idx*c_blocks + c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;

    vstore4(outValue0, 0, output+out_offset);
}

__kernel
void depthwise_conv2d_s1_c8h1w4(GLOBAL_SIZE_2_DIMS __global const FLOAT *input,
                                  __global const FLOAT *filter,
                                  __global const FLOAT *bias,
                                  __global FLOAT *output,
                                  __private const int2 in_hw,
                                  __private const int channel,
                                  __private const int2 out_hw,
                                  __private const int2 filter_hw,
                                  __private const int2 pad_hw,
                                  __private const int2 dilate_hw,
                                  __private const int2 stride_hw,
                                  __private const int out_w_blocks,
                                  __private const int c_blocks) {

    const int out_c_w_idx = get_global_id(0);// oc/4 * ow/4
    const int out_b_h_idx  = get_global_id(1);// b * h
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int c_idx = (out_c_w_idx / out_w_blocks) << 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int b_idx = out_b_h_idx / out_hw.x;
    const int out_h_idx = out_b_h_idx % out_hw.x;

    FLOAT4 outValue0 = vload4(c_idx+0, bias);
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;
    FLOAT4 outValue4 = vload4(c_idx+1, bias);
    FLOAT4 outValue5 = outValue4;
    FLOAT4 outValue6 = outValue4;
    FLOAT4 outValue7 = outValue4;

    const int out_w4_idx = out_w_idx << 2;
    const int in_w_start_0 = out_w4_idx - pad_hw.y;
    const int in_w_start_1 = in_w_start_0 + 1;
    const int in_w_start_2 = in_w_start_0 + 2;
    const int in_w_start_3 = in_w_start_0 + 3;

    const int in_h_start = out_h_idx - pad_hw.x;
    
    for (int kh = 0; kh < filter_hw.x; kh++) {
        const int in_h_cur = in_h_start + kh;
        if(in_h_cur < 0 || in_h_cur >= in_hw.x) continue;
        
        int inp_offset_c0 = (((b_idx*c_blocks + c_idx+0)*in_hw.x + in_h_cur)* in_hw.y + in_w_start_0)*4;
        int inp_offset_c1 = (((b_idx*c_blocks + c_idx+1)*in_hw.x + in_h_cur)* in_hw.y + in_w_start_0)*4;
        for (int kw = 0; kw < filter_hw.y; kw++) {
            const int filter_idx = mad24(kh, filter_hw.y, kw);
            FLOAT4 inValue0 = (in_w_start_0+kw < 0 || in_w_start_0+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+0, input+inp_offset_c0);
            FLOAT4 inValue1 = (in_w_start_1+kw < 0 || in_w_start_1+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+1, input+inp_offset_c0);
            FLOAT4 inValue2 = (in_w_start_2+kw < 0 || in_w_start_2+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+2, input+inp_offset_c0);
            FLOAT4 inValue3 = (in_w_start_3+kw < 0 || in_w_start_3+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+3, input+inp_offset_c0);

            FLOAT4 inValue4 = (in_w_start_0+kw < 0 || in_w_start_0+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+0, input+inp_offset_c1);
            FLOAT4 inValue5 = (in_w_start_1+kw < 0 || in_w_start_1+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+1, input+inp_offset_c1);
            FLOAT4 inValue6 = (in_w_start_2+kw < 0 || in_w_start_2+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+2, input+inp_offset_c1);
            FLOAT4 inValue7 = (in_w_start_3+kw < 0 || in_w_start_3+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+3, input+inp_offset_c1);
            
            //NC4HW4 [1, filterShape.x*filterShape.y, 1, channelBlocks] x oc4
            //index: [0, filterIdx,                   0, inChannelBlockIdx]
            FLOAT4 weights_0 = vload4(0, filter+(filter_idx*c_blocks+c_idx+0)*4);
            FLOAT4 weights_1 = vload4(0, filter+(filter_idx*c_blocks+c_idx+1)*4);

            outValue0 = mad(inValue0, weights_0, outValue0);
            outValue1 = mad(inValue1, weights_0, outValue1);
            outValue2 = mad(inValue2, weights_0, outValue2);
            outValue3 = mad(inValue3, weights_0, outValue3);
            
            outValue4 = mad(inValue4, weights_1, outValue4);
            outValue5 = mad(inValue5, weights_1, outValue5);
            outValue6 = mad(inValue6, weights_1, outValue6);
            outValue7 = mad(inValue7, weights_1, outValue7);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
    
    outValue4 = fmax(outValue4, (FLOAT4)0);
    outValue5 = fmax(outValue5, (FLOAT4)0);
    outValue6 = fmax(outValue6, (FLOAT4)0);
    outValue7 = fmax(outValue7, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
    
    outValue4 = clamp(outValue4, (FLOAT4)0, (FLOAT4)6);
    outValue5 = clamp(outValue5, (FLOAT4)0, (FLOAT4)6);
    outValue6 = clamp(outValue6, (FLOAT4)0, (FLOAT4)6);
    outValue7 = clamp(outValue7, (FLOAT4)0, (FLOAT4)6);
#endif

    int out_offset = (((b_idx*c_blocks + c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w4_idx)*4;

    const int remain     = out_hw.y - out_w4_idx;
    if (remain >= 4) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
        vstore4(outValue2, 2, output+out_offset);
        vstore4(outValue3, 3, output+out_offset);
    } else if (remain == 3) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
        vstore4(outValue2, 2, output+out_offset);
    } else if (remain == 2) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(outValue0, 0, output+out_offset);
    }
    
    if(c_idx + 1 >= c_blocks) return;
    
    out_offset += out_hw.x * out_hw.y * 4;
    if (remain >= 4) {
        vstore4(outValue4, 0, output+out_offset);
        vstore4(outValue5, 1, output+out_offset);
        vstore4(outValue6, 2, output+out_offset);
        vstore4(outValue7, 3, output+out_offset);
    } else if (remain == 3) {
        vstore4(outValue4, 0, output+out_offset);
        vstore4(outValue5, 1, output+out_offset);
        vstore4(outValue6, 2, output+out_offset);
    } else if (remain == 2) {
        vstore4(outValue4, 0, output+out_offset);
        vstore4(outValue5, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(outValue4, 0, output+out_offset);
    }
}


__kernel
void depthwise_conv2d_s1_c8h1w2(GLOBAL_SIZE_2_DIMS __global const FLOAT *input,
                                  __global const FLOAT *filter,
                                  __global const FLOAT *bias,
                                  __global FLOAT *output,
                                  __private const int2 in_hw,
                                  __private const int channel,
                                  __private const int2 out_hw,
                                  __private const int2 filter_hw,
                                  __private const int2 pad_hw,
                                  __private const int2 dilate_hw,
                                  __private const int2 stride_hw,
                                  __private const int out_w_blocks,
                                  __private const int c_blocks) {

    const int out_c_w_idx = get_global_id(0);// oc/4 * ow/4
    const int out_b_h_idx  = get_global_id(1);// b * h
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int c_idx = (out_c_w_idx / out_w_blocks) << 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int b_idx = out_b_h_idx / out_hw.x;
    const int out_h_idx = out_b_h_idx % out_hw.x;

    FLOAT4 outValue0 = vload4(c_idx+0, bias);
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue4 = vload4(c_idx+1, bias);
    FLOAT4 outValue5 = outValue4;

    const int out_w2_idx = out_w_idx << 1;
    const int in_w_start_0 = out_w2_idx - pad_hw.y;
    const int in_w_start_1 = in_w_start_0 + 1;

    const int in_h_start = out_h_idx - pad_hw.x;
    
    for (int kh = 0; kh < filter_hw.x; kh++) {
        const int in_h_cur = in_h_start + kh;
        if(in_h_cur < 0 || in_h_cur >= in_hw.x) continue;
        
        int inp_offset_c0 = (((b_idx*c_blocks + c_idx+0)*in_hw.x + in_h_cur)* in_hw.y + in_w_start_0)*4;
        int inp_offset_c1 = (((b_idx*c_blocks + c_idx+1)*in_hw.x + in_h_cur)* in_hw.y + in_w_start_0)*4;
        for (int kw = 0; kw < filter_hw.y; kw++) {
            const int filter_idx = mad24(kh, filter_hw.y, kw);
            FLOAT4 inValue0 = (in_w_start_0+kw < 0 || in_w_start_0+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+0, input+inp_offset_c0);
            FLOAT4 inValue1 = (in_w_start_1+kw < 0 || in_w_start_1+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+1, input+inp_offset_c0);

            FLOAT4 inValue4 = (in_w_start_0+kw < 0 || in_w_start_0+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+0, input+inp_offset_c1);
            FLOAT4 inValue5 = (in_w_start_1+kw < 0 || in_w_start_1+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+1, input+inp_offset_c1);

            //NC4HW4 [1, filterShape.x*filterShape.y, 1, channelBlocks] x oc4
            //index: [0, filterIdx,                   0, inChannelBlockIdx]
            FLOAT4 weights_0 = vload4(0, filter+(filter_idx*c_blocks+c_idx+0)*4);
            FLOAT4 weights_1 = vload4(0, filter+(filter_idx*c_blocks+c_idx+1)*4);

            outValue0 = mad(inValue0, weights_0, outValue0);
            outValue1 = mad(inValue1, weights_0, outValue1);
            
            outValue4 = mad(inValue4, weights_1, outValue4);
            outValue5 = mad(inValue5, weights_1, outValue5);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    
    outValue4 = fmax(outValue4, (FLOAT4)0);
    outValue5 = fmax(outValue5, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    
    outValue4 = clamp(outValue4, (FLOAT4)0, (FLOAT4)6);
    outValue5 = clamp(outValue5, (FLOAT4)0, (FLOAT4)6);
#endif

    int out_offset = (((b_idx*c_blocks + c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w2_idx)*4;

    const int remain     = out_hw.y - out_w2_idx;
    if (remain >= 2) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(outValue0, 0, output+out_offset);
    }
    
    if(c_idx + 1 >= c_blocks) return;
    
    out_offset += out_hw.x * out_hw.y * 4;
    if (remain >= 2) {
        vstore4(outValue4, 0, output+out_offset);
        vstore4(outValue5, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(outValue4, 0, output+out_offset);
    }
}

__kernel
void depthwise_conv2d_s1_c4h1w4(GLOBAL_SIZE_2_DIMS __global const FLOAT *input,
                                  __global const FLOAT *filter,
                                  __global const FLOAT *bias,
                                  __global FLOAT *output,
                                  __private const int2 in_hw,
                                  __private const int channel,
                                  __private const int2 out_hw,
                                  __private const int2 filter_hw,
                                  __private const int2 pad_hw,
                                  __private const int2 dilate_hw,
                                  __private const int2 stride_hw,
                                  __private const int out_w_blocks,
                                  __private const int c_blocks) {

    const int out_c_w_idx = get_global_id(0);// oc/4 * ow/4
    const int out_b_h_idx  = get_global_id(1);// b * h
    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int b_idx = out_b_h_idx / out_hw.x;
    const int out_h_idx = out_b_h_idx % out_hw.x;

    FLOAT4 outValue0 = vload4(c_idx, bias);
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;

    const int out_w4_idx = out_w_idx << 2;
    const int in_w_start_0 = out_w4_idx - pad_hw.y;
    const int in_w_start_1 = in_w_start_0 + 1;
    const int in_w_start_2 = in_w_start_0 + 2;
    const int in_w_start_3 = in_w_start_0 + 3;

    const int in_h_start = out_h_idx - pad_hw.x;
    
    FLOAT4 inValue0, inValue1, inValue2, inValue3;
    for (int kh = 0; kh < filter_hw.x; kh++) {
        const int in_h_cur = in_h_start + kh;
        if(in_h_cur < 0 || in_h_cur >= in_hw.x) continue;
        
        int inp_offset = (((b_idx*c_blocks + c_idx)*in_hw.x + in_h_cur)* in_hw.y + in_w_start_0)*4;
        for (int kw = 0; kw < filter_hw.y; kw++) {
            const int filter_idx = mad24(kh, filter_hw.y, kw);
            inValue0 = (in_w_start_0+kw < 0 || in_w_start_0+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+0, input+inp_offset);
            inValue1 = (in_w_start_1+kw < 0 || in_w_start_1+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+1, input+inp_offset);
            inValue2 = (in_w_start_2+kw < 0 || in_w_start_2+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+2, input+inp_offset);
            inValue3 = (in_w_start_3+kw < 0 || in_w_start_3+kw >= in_hw.y) ? (FLOAT4)0 : vload4(kw+3, input+inp_offset);

            //NC4HW4 [1, filterShape.x*filterShape.y, 1, channelBlocks] x oc4
            //index: [0, filterIdx,                   0, inChannelBlockIdx]
            FLOAT4 weights = vload4(0, filter+(filter_idx*c_blocks+c_idx)*4);

            outValue0 = mad(inValue0, weights, outValue0);
            outValue1 = mad(inValue1, weights, outValue1);
            outValue2 = mad(inValue2, weights, outValue2);
            outValue3 = mad(inValue3, weights, outValue3);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((b_idx*c_blocks + c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w4_idx)*4;

    const int remain     = out_hw.y - out_w4_idx;
    if (remain >= 4) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
        vstore4(outValue2, 2, output+out_offset);
        vstore4(outValue3, 3, output+out_offset);
    } else if (remain == 3) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
        vstore4(outValue2, 2, output+out_offset);
    } else if (remain == 2) {
        vstore4(outValue0, 0, output+out_offset);
        vstore4(outValue1, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(outValue0, 0, output+out_offset);
    }
}
