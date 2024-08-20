#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#define MOD_NUM 15
#ifdef INPUT_CHANNEL_LEAVE
    #define PADZEROSVEC(k, channel, data0, data1, data2, data3) \
        data0 = (k << 2) < channel ? data0 : 0; \
        data1 = (k << 2) + 1 < channel ? data1 : 0; \
        data2 = (k << 2) + 2 < channel ? data2 : 0; \
        data3 = (k << 2) + 3 < channel ? data3 : 0;
#else
    #define PADZEROSVEC(k, channel, data0, data1, data2, data3)
#endif

__kernel
void conv_2d_int_c4h1w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *weight,
#else
                      __global const uchar *weight,
#endif
                      __global const float *dequantScaleOffset,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_hw.y;
    const int out_w_idx = out_c_w_idx % out_hw.y;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    COMPUTE_FLOAT4 out0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);

    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        int kindex = (in_c_idx * 4) / blockDim * out_c_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale = (COMPUTE_FLOAT4)(ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6);
        COMPUTE_FLOAT4 offset = (COMPUTE_FLOAT4)(ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7);
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + kw_start) * 4;
        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            for(int ix = in_w_idx_start; ix < in_w_idx_end; ix += dilate_hw.y) {
                int inp_offset = (((out_b_idx * in_c_blocks + in_c_idx) * in_hw.x + iy) * in_hw.y + ix) * 4;
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset));
                
                const int filter_w_inc = (ix-in_w_idx_start)/dilate_hw.y;

#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(filter_w_inc, weight+weight_offset);
                char4 charWeight1 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset*3);
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale + offset;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale + offset;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale + offset;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale + offset;
#else
                uchar2 charWeightInt40 = vload2(filter_w_inc, weight+weight_offset/2);
                uchar2 charWeightInt41 = vload2(filter_w_inc, weight+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(filter_w_inc, weight+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(filter_w_inc, weight+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale + offset;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale + offset;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale + offset;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale + offset;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);
                
                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);

            }
            weight_offset += 4*filter_hw.y;
        }
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
 
}

__kernel
void conv_2d_int_c4h1w2(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *weight,
#else
                      __global const uchar *weight,
#endif
                      __global const float *dequantScaleOffset,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,//generate width's num
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 1;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out0 = bias0;
    COMPUTE_FLOAT4 out1 = bias0;
    
    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        int kindex = (in_c_idx * 4) / blockDim * out_c_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale = (COMPUTE_FLOAT4)(ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6);
        COMPUTE_FLOAT4 offset = (COMPUTE_FLOAT4)(ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7);
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            const int inp_offset_base = (((out_b_idx * in_c_blocks + in_c_idx) * in_hw.x + iy) * in_hw.y + 0) * 4;

            for(int fw = 0; fw < filter_hw.y; fw++) {
                const int in_w0_idx = fw * dilate_hw.y + in_w0_idx_base;
                const int in_w1_idx = fw * dilate_hw.y + in_w1_idx_base;

                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4((in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4((in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w1_idx, input+inp_offset_base));
                
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, weight+weight_offset);
                char4 charWeight1 = vload4(0, weight+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(0, weight+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(0, weight+weight_offset+weight_oc_offset*3);
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale + offset;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale + offset;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale + offset;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale + offset;
#else
                uchar2 charWeightInt40 = vload2(0, weight+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, weight+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(0, weight+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, weight+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale + offset;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale + offset;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale + offset;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale + offset;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);
                
                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                weight_offset += 4;
            }
        }
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    if(out_w_idx + 1 >= out_hw.y) return;
    vstore4(CONVERT_FLOAT4(out1), 1, output+out_offset);
#else
    vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
#endif
}

__kernel
void conv_2d_int_c4h1w4(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *weight,
#else
                      __global const uchar *weight,
#endif
                      __global const float *dequantScaleOffset,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 2;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;

    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out0 = bias0;
    COMPUTE_FLOAT4 out1 = bias0;
    COMPUTE_FLOAT4 out2 = bias0;
    COMPUTE_FLOAT4 out3 = bias0;

    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;
    const int in_w2_idx_base = in_w1_idx_base + stride_hw.y;
    const int in_w3_idx_base = in_w2_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        int kindex = (in_c_idx * 4) / blockDim * out_c_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale = (COMPUTE_FLOAT4)(ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6);
        COMPUTE_FLOAT4 offset = (COMPUTE_FLOAT4)(ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7);
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            const int inp_offset_base = (((out_b_idx * in_c_blocks + in_c_idx) * in_hw.x + iy) * in_hw.y + 0) * 4;

            for(int fw = 0; fw < filter_hw.y; fw++) {
                const int in_w0_idx = fw * dilate_hw.y + in_w0_idx_base;
                const int in_w1_idx = fw * dilate_hw.y + in_w1_idx_base;
                const int in_w2_idx = fw * dilate_hw.y + in_w2_idx_base;
                const int in_w3_idx = fw * dilate_hw.y + in_w3_idx_base;

                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4((in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4((in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w1_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4((in_w2_idx < 0 || in_w2_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w2_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = CONVERT_COMPUTE_FLOAT4((in_w3_idx < 0 || in_w3_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w3_idx, input+inp_offset_base));

#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, weight+weight_offset);
                char4 charWeight1 = vload4(0, weight+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(0, weight+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(0, weight+weight_offset+weight_oc_offset*3);
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale + offset;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale + offset;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale + offset;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale + offset;
#else
                uchar2 charWeightInt40 = vload2(0, weight+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, weight+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(0, weight+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, weight+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale + offset;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale + offset;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale + offset;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale + offset;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);
                
                weight_offset += 4;
            }
        }
    }
    
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.y - out_w_idx;

    if (remain >= 4) {
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
    }else if(remain == 3){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2, output+out_offset);
    }else if(remain == 2){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#else
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
#endif
}

__kernel
void conv_2d_int_c4h4w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *weight,
#else
                      __global const uchar *weight,
#endif
                      __global const float *dequantScaleOffset,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 2;
    
    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out0 = bias0;
    COMPUTE_FLOAT4 out1 = bias0;
    COMPUTE_FLOAT4 out2 = bias0;
    COMPUTE_FLOAT4 out3 = bias0;

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);

    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int in_h1_idx_base = in_h0_idx_base + stride_hw.x;
    const int in_h2_idx_base = in_h1_idx_base + stride_hw.x;
    const int in_h3_idx_base = in_h2_idx_base + stride_hw.x;
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    const int in_hw_size = in_hw.x * in_hw.y;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        int kindex = (in_c_idx * 4) / blockDim * out_c_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale = (COMPUTE_FLOAT4)(ScaleOffset.s0, ScaleOffset.s2, ScaleOffset.s4, ScaleOffset.s6);
        COMPUTE_FLOAT4 offset = (COMPUTE_FLOAT4)(ScaleOffset.s1, ScaleOffset.s3, ScaleOffset.s5, ScaleOffset.s7);
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx * in_c_blocks + in_c_idx) * in_hw.x * in_hw.y * 4;

        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;
            const int in_h1_idx = (iy * dilate_hw.x + in_h1_idx_base) * in_hw.y;
            const int in_h2_idx = (iy * dilate_hw.x + in_h2_idx_base) * in_hw.y;
            const int in_h3_idx = (iy * dilate_hw.x + in_h3_idx_base) * in_hw.y;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4((in_h0_idx < 0 || in_h0_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h0_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4((in_h1_idx < 0 || in_h1_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h1_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4((in_h2_idx < 0 || in_h2_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h2_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = CONVERT_COMPUTE_FLOAT4((in_h3_idx < 0 || in_h3_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h3_idx + fw, input+inp_offset_base));

#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, weight+weight_offset);
                char4 charWeight1 = vload4(0, weight+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(0, weight+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(0, weight+weight_offset+weight_oc_offset*3);
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale + offset;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale + offset;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale + offset;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale + offset;
#else
                uchar2 charWeightInt40 = vload2(0, weight+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, weight+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(0, weight+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, weight+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale + offset;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale + offset;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale + offset;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale + offset;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);
                
                weight_offset += 4;
            }
        }
    }
    
#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
#endif
}

__kernel
void conv_2d_int_c8h4w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *weight,
#else
                      __global const uchar *weight,
#endif
                      __global const float *dequantScaleOffset,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = (out_c_w_idx / out_w_blocks) << 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 2;
    
    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out0 = bias0;
    COMPUTE_FLOAT4 out1 = bias0;
    COMPUTE_FLOAT4 out2 = bias0;
    COMPUTE_FLOAT4 out3 = bias0;
    COMPUTE_FLOAT4 bias1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx + 1, bias));
    COMPUTE_FLOAT4 out4 = bias1;
    COMPUTE_FLOAT4 out5 = bias1;
    COMPUTE_FLOAT4 out6 = bias1;
    COMPUTE_FLOAT4 out7 = bias1;

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);

    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int in_h1_idx_base = in_h0_idx_base + stride_hw.x;
    const int in_h2_idx_base = in_h1_idx_base + stride_hw.x;
    const int in_h3_idx_base = in_h2_idx_base + stride_hw.x;
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    const int in_hw_size = in_hw.x * in_hw.y;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        int kindex = (in_c_idx * 4) / blockDim * out_c_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset0 = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT8 ScaleOffset1 = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx + 1, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale0 = (COMPUTE_FLOAT4)(ScaleOffset0.s0, ScaleOffset0.s2, ScaleOffset0.s4, ScaleOffset0.s6);
        COMPUTE_FLOAT4 offset0 = (COMPUTE_FLOAT4)(ScaleOffset0.s1, ScaleOffset0.s3, ScaleOffset0.s5, ScaleOffset0.s7);
        COMPUTE_FLOAT4 scale1 = (COMPUTE_FLOAT4)(ScaleOffset1.s0, ScaleOffset1.s2, ScaleOffset1.s4, ScaleOffset1.s6);
        COMPUTE_FLOAT4 offset1 = (COMPUTE_FLOAT4)(ScaleOffset1.s1, ScaleOffset1.s3, ScaleOffset1.s5, ScaleOffset1.s7);
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx * in_c_blocks + in_c_idx) * in_hw.x * in_hw.y * 4;

        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;
            const int in_h1_idx = (iy * dilate_hw.x + in_h1_idx_base) * in_hw.y;
            const int in_h2_idx = (iy * dilate_hw.x + in_h2_idx_base) * in_hw.y;
            const int in_h3_idx = (iy * dilate_hw.x + in_h3_idx_base) * in_hw.y;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4((in_h0_idx < 0 || in_h0_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h0_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4((in_h1_idx < 0 || in_h1_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h1_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4((in_h2_idx < 0 || in_h2_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h2_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = CONVERT_COMPUTE_FLOAT4((in_h3_idx < 0 || in_h3_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h3_idx + fw, input+inp_offset_base));

#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, weight+weight_offset);
                char4 charWeight1 = vload4(0, weight+weight_offset+weight_ic_offset);
                char4 charWeight2 = vload4(0, weight+weight_offset+weight_ic_offset*2);
                char4 charWeight3 = vload4(0, weight+weight_offset+weight_ic_offset*3);
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale0 + offset0;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale0 + offset0;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale0 + offset0;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale0 + offset0;
#else
                uchar2 charWeightInt40 = vload2(0, weight+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, weight+weight_offset/2+weight_ic_offset/2);
                uchar2 charWeightInt42 = vload2(0, weight+weight_offset/2+weight_ic_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, weight+weight_offset/2+weight_ic_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM)- 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale0 + offset0;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale0 + offset0;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale0 + offset0;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale0 + offset0;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);

#if (defined USE_LOW_BIT_WEIGHT_INT8)
                charWeight0 = vload4(0, weight+weight_offset+weight_oc_offset);
                charWeight1 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset);
                charWeight2 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2);
                charWeight3 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3);
                weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale1 + offset1;
                weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale1 + offset1;
                weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale1 + offset1;
                weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale1 + offset1;
#else
                charWeightInt40 = vload2(0, weight+weight_offset/2+weight_oc_offset/2);
                charWeightInt41 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset/2);
                charWeightInt42 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset*2/2);
                charWeightInt43 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset*3/2);
                charWeight0 = (char4)(0, 0, 0, 0);
                charWeight1 = (char4)(0, 0, 0, 0);
                charWeight2 = (char4)(0, 0, 0, 0);
                charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM)- 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM)- 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale1 + offset1;
                weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale1 + offset1;
                weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale1 + offset1;
                weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale1 + offset1;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);

                out4 = mad(in0.x, weight0, out4);
                out4 = mad(in0.y, weight1, out4);
                out4 = mad(in0.z, weight2, out4);
                out4 = mad(in0.w, weight3, out4);
                
                out5 = mad(in1.x, weight0, out5);
                out5 = mad(in1.y, weight1, out5);
                out5 = mad(in1.z, weight2, out5);
                out5 = mad(in1.w, weight3, out5);
                
                out6 = mad(in2.x, weight0, out6);
                out6 = mad(in2.y, weight1, out6);
                out6 = mad(in2.z, weight2, out6);
                out6 = mad(in2.w, weight3, out6);
                
                out7 = mad(in3.x, weight0, out7);
                out7 = mad(in3.y, weight1, out7);
                out7 = mad(in3.z, weight2, out7);
                out7 = mad(in3.w, weight3, out7);
                
                weight_offset += 4;
            }
        }
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
    out4 = fmax(out4, (COMPUTE_FLOAT4)0);
    out5 = fmax(out5, (COMPUTE_FLOAT4)0);
    out6 = fmax(out6, (COMPUTE_FLOAT4)0);
    out7 = fmax(out7, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out4 = clamp(out4, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out5 = clamp(out5, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out6 = clamp(out6, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out7 = clamp(out7, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#ifdef CHANNEL_LEAVE
    if(out_c_idx + 1 >= out_c_blocks){
        return;
    }
#endif
    out_offset = (((out_b_idx*out_c_blocks + out_c_idx + 1)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    if(remain >= 4){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out6), 2 * out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out7), 3 * out_hw.y, output+out_offset);
    }else if(remain == 3){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
        vstore4(CONVERT_FLOAT4(out6), 2 * out_hw.y, output+out_offset);
    }else if(remain == 2){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out2), 2 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), 3 * out_hw.y, output+out_offset);
#ifdef CHANNEL_LEAVE
    if(out_c_idx + 1 >= out_c_blocks){
        return;
    }
#endif
    out_offset = (((out_b_idx*out_c_blocks + out_c_idx + 1)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out5), out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out6), 2 * out_hw.y, output+out_offset);
    vstore4(CONVERT_FLOAT4(out7), 3 * out_hw.y, output+out_offset);
#endif
}

__kernel
void conv_2d_int_c8h2w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *weight,
#else
                      __global const uchar *weight,
#endif
                      __global const float *dequantScaleOffset,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = (out_c_w_idx / out_w_blocks) << 1;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h_blocks;//equal to in_b_idx
    const int out_h_idx = (out_b_h_idx % out_h_blocks) << 1;

    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out0 = bias0;
    COMPUTE_FLOAT4 out1 = bias0;
    COMPUTE_FLOAT4 bias1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx + 1, bias));
    COMPUTE_FLOAT4 out2 = bias1;
    COMPUTE_FLOAT4 out3 = bias1;

    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);

    const int in_h0_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    const int in_h1_idx_base = in_h0_idx_base + stride_hw.x;
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    const int in_hw_size = in_hw.x * in_hw.y;
    // weight: [ic/4, oc, 4], loop: ic/4
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        int kindex = (in_c_idx * 4) / blockDim * out_c_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset0 = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT8 ScaleOffset1 = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx + 1, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale0 = (COMPUTE_FLOAT4)(ScaleOffset0.s0, ScaleOffset0.s2, ScaleOffset0.s4, ScaleOffset0.s6);
        COMPUTE_FLOAT4 offset0 = (COMPUTE_FLOAT4)(ScaleOffset0.s1, ScaleOffset0.s3, ScaleOffset0.s5, ScaleOffset0.s7);
        COMPUTE_FLOAT4 scale1 = (COMPUTE_FLOAT4)(ScaleOffset1.s0, ScaleOffset1.s2, ScaleOffset1.s4, ScaleOffset1.s6);
        COMPUTE_FLOAT4 offset1 = (COMPUTE_FLOAT4)(ScaleOffset1.s1, ScaleOffset1.s3, ScaleOffset1.s5, ScaleOffset1.s7);
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        const int inp_offset_base = (out_b_idx * in_c_blocks + in_c_idx) * in_hw.x * in_hw.y * 4;

        for(int iy = 0; iy < filter_hw.x; iy++) {
            int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + iy)*filter_hw.y + kw_start) * 4;
            const int in_h0_idx = (iy * dilate_hw.x + in_h0_idx_base) * in_hw.y;
            const int in_h1_idx = (iy * dilate_hw.x + in_h1_idx_base) * in_hw.y;

            for(int fw = in_w_idx_start; fw < in_w_idx_end; fw += dilate_hw.y) {
                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4((in_h0_idx < 0 || in_h0_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h0_idx + fw, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4((in_h1_idx < 0 || in_h1_idx >= in_hw_size) ? (FLOAT4)0 : vload4(in_h1_idx + fw, input+inp_offset_base));
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, weight+weight_offset);
                char4 charWeight1 = vload4(0, weight+weight_offset+weight_ic_offset);
                char4 charWeight2 = vload4(0, weight+weight_offset+weight_ic_offset*2);
                char4 charWeight3 = vload4(0, weight+weight_offset+weight_ic_offset*3);
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale0 + offset0;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale0 + offset0;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale0 + offset0;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale0 + offset0;
#else
                uchar2 charWeightInt40 = vload2(0, weight+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, weight+weight_offset/2+weight_ic_offset/2);
                uchar2 charWeightInt42 = vload2(0, weight+weight_offset/2+weight_ic_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, weight+weight_offset/2+weight_ic_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale0 + offset0;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale0 + offset0;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale0 + offset0;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale0 + offset0;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);
                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                charWeight0 = vload4(0, weight+weight_offset+weight_oc_offset);
                charWeight1 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset);
                charWeight2 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2);
                charWeight3 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3);
                weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale1 + offset1;
                weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale1 + offset1;
                weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale1 + offset1;
                weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale1 + offset1;
#else
                charWeightInt40 = vload2(0, weight+weight_offset/2+weight_oc_offset/2);
                charWeightInt41 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset/2);
                charWeightInt42 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset*2/2);
                charWeightInt43 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset*3/2);
                charWeight0 = (char4)(0, 0, 0, 0);
                charWeight1 = (char4)(0, 0, 0, 0);
                charWeight2 = (char4)(0, 0, 0, 0);
                charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0& MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1& MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0& MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1& MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0& MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1& MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0& MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1& MOD_NUM) - 8;
                weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale1 + offset1;
                weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale1 + offset1;
                weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale1 + offset1;
                weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale1 + offset1;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);
                out2 = mad(in0.x, weight0, out2);
                out2 = mad(in0.y, weight1, out2);
                out2 = mad(in0.z, weight2, out2);
                out2 = mad(in0.w, weight3, out2);
                
                out3 = mad(in1.x, weight0, out3);
                out3 = mad(in1.y, weight1, out3);
                out3 = mad(in1.z, weight2, out3);
                out3 = mad(in1.w, weight3, out3);
                
                weight_offset += 4;
            }
        }
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.x - out_h_idx;
    if(remain >= 2){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#ifdef CHANNEL_LEAVE
    if(out_c_idx + 1 >= out_c_blocks){
        return;
    }
#endif
    out_offset = (((out_b_idx*out_c_blocks + out_c_idx + 1)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    if(remain >= 2){
        vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out3), out_hw.y, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
    }
#else
    vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out1), out_hw.y, output+out_offset);
#ifdef CHANNEL_LEAVE
    if(out_c_idx + 1 >= out_c_blocks){
        return;
    }
#endif
    out_offset = (((out_b_idx*out_c_blocks + out_c_idx + 1)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(CONVERT_FLOAT4(out2), 0, output+out_offset);
    vstore4(CONVERT_FLOAT4(out3), out_hw.y, output+out_offset);
#endif
}

__kernel
void conv_2d_int_c8h1w4(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *weight,
#else
                      __global const uchar *weight,
#endif
                      __global const float *dequantScaleOffset,
                      __global const FLOAT *bias,
                      __global FLOAT *output,
                      __private const int2 in_hw,
                      __private const int inChannel,
                      __private const int in_c_blocks,
                      __private const int2 out_hw,
                      __private const int2 filter_hw,
                      __private const int2 stride_hw,
                      __private const int2 pad_hw,
                      __private const int2 dilate_hw,
                      __private const int out_w_blocks,
                      __private const int out_c_blocks,
                      __private const int out_h_blocks,
                      __private const int blockDim) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = (out_c_w_idx / out_w_blocks) << 1;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 2;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    COMPUTE_FLOAT4 bias0 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out0 = bias0;
    COMPUTE_FLOAT4 out1 = bias0;
    COMPUTE_FLOAT4 out2 = bias0;
    COMPUTE_FLOAT4 out3 = bias0;
    COMPUTE_FLOAT4 bias1 = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx + 1, bias));
    COMPUTE_FLOAT4 out4 = bias1;
    COMPUTE_FLOAT4 out5 = bias1;
    COMPUTE_FLOAT4 out6 = bias1;
    COMPUTE_FLOAT4 out7 = bias1;

    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;
    const int in_w2_idx_base = in_w1_idx_base + stride_hw.y;
    const int in_w3_idx_base = in_w2_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    for(ushort in_c_idx = 0; in_c_idx < in_c_blocks; in_c_idx++) {
        int kindex = (in_c_idx * 4) / blockDim * out_c_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset0 = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT8 ScaleOffset1 = CONVERT_COMPUTE_FLOAT8(vload8(out_c_idx + 1, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale0 = (COMPUTE_FLOAT4)(ScaleOffset0.s0, ScaleOffset0.s2, ScaleOffset0.s4, ScaleOffset0.s6);
        COMPUTE_FLOAT4 offset0 = (COMPUTE_FLOAT4)(ScaleOffset0.s1, ScaleOffset0.s3, ScaleOffset0.s5, ScaleOffset0.s7);
        COMPUTE_FLOAT4 scale1 = (COMPUTE_FLOAT4)(ScaleOffset1.s0, ScaleOffset1.s2, ScaleOffset1.s4, ScaleOffset1.s6);
        COMPUTE_FLOAT4 offset1 = (COMPUTE_FLOAT4)(ScaleOffset1.s1, ScaleOffset1.s3, ScaleOffset1.s5, ScaleOffset1.s7);
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            const int inp_offset_base = (((out_b_idx * in_c_blocks + in_c_idx) * in_hw.x + iy) * in_hw.y + 0) * 4;

            for(int fw = 0; fw < filter_hw.y; fw++) {
                const int in_w0_idx = fw * dilate_hw.y + in_w0_idx_base;
                const int in_w1_idx = fw * dilate_hw.y + in_w1_idx_base;
                const int in_w2_idx = fw * dilate_hw.y + in_w2_idx_base;
                const int in_w3_idx = fw * dilate_hw.y + in_w3_idx_base;

                COMPUTE_FLOAT4 in0 = CONVERT_COMPUTE_FLOAT4((in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w0_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in1 = CONVERT_COMPUTE_FLOAT4((in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w1_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in2 = CONVERT_COMPUTE_FLOAT4((in_w2_idx < 0 || in_w2_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w2_idx, input+inp_offset_base));
                COMPUTE_FLOAT4 in3 = CONVERT_COMPUTE_FLOAT4((in_w3_idx < 0 || in_w3_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w3_idx, input+inp_offset_base));

#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, weight+weight_offset);
                char4 charWeight1 = vload4(0, weight+weight_offset+weight_ic_offset);
                char4 charWeight2 = vload4(0, weight+weight_offset+weight_ic_offset*2);
                char4 charWeight3 = vload4(0, weight+weight_offset+weight_ic_offset*3);
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale0 + offset0;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale0 + offset0;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale0 + offset0;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale0 + offset0;
#else
                uchar2 charWeightInt40 = vload2(0, weight+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, weight+weight_offset/2+weight_ic_offset/2);
                uchar2 charWeightInt42 = vload2(0, weight+weight_offset/2+weight_ic_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, weight+weight_offset/2+weight_ic_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                COMPUTE_FLOAT4 weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale0 + offset0;
                COMPUTE_FLOAT4 weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale0 + offset0;
                COMPUTE_FLOAT4 weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale0 + offset0;
                COMPUTE_FLOAT4 weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale0 + offset0;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);

                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                out1 = mad(in1.x, weight0, out1);
                out1 = mad(in1.y, weight1, out1);
                out1 = mad(in1.z, weight2, out1);
                out1 = mad(in1.w, weight3, out1);
                
                out2 = mad(in2.x, weight0, out2);
                out2 = mad(in2.y, weight1, out2);
                out2 = mad(in2.z, weight2, out2);
                out2 = mad(in2.w, weight3, out2);
                
                out3 = mad(in3.x, weight0, out3);
                out3 = mad(in3.y, weight1, out3);
                out3 = mad(in3.z, weight2, out3);
                out3 = mad(in3.w, weight3, out3);
                
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                charWeight0 = vload4(0, weight+weight_offset+weight_oc_offset);
                charWeight1 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset);
                charWeight2 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*2);
                charWeight3 = vload4(0, weight+weight_offset+weight_oc_offset+weight_ic_offset*3);
                weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale1 + offset1;
                weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale1 + offset1;
                weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale1 + offset1;
                weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale1 + offset1;
#else
                charWeightInt40 = vload2(0, weight+weight_offset/2+weight_oc_offset/2);
                charWeightInt41 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset/2);
                charWeightInt42 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset*2/2);
                charWeightInt43 = vload2(0, weight+weight_offset/2+weight_oc_offset/2+weight_ic_offset*3/2);
                charWeight0 = (char4)(0, 0, 0, 0);
                charWeight1 = (char4)(0, 0, 0, 0);
                charWeight2 = (char4)(0, 0, 0, 0);
                charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM)- 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                weight0 = CONVERT_COMPUTE_FLOAT4(charWeight0) * scale1 + offset1;
                weight1 = CONVERT_COMPUTE_FLOAT4(charWeight1) * scale1 + offset1;
                weight2 = CONVERT_COMPUTE_FLOAT4(charWeight2) * scale1 + offset1;
                weight3 = CONVERT_COMPUTE_FLOAT4(charWeight3) * scale1 + offset1;
#endif
                PADZEROSVEC(in_c_idx, inChannel, weight0, weight1, weight2, weight3);
                
                out4 = mad(in0.x, weight0, out4);
                out4 = mad(in0.y, weight1, out4);
                out4 = mad(in0.z, weight2, out4);
                out4 = mad(in0.w, weight3, out4);
                
                out5 = mad(in1.x, weight0, out5);
                out5 = mad(in1.y, weight1, out5);
                out5 = mad(in1.z, weight2, out5);
                out5 = mad(in1.w, weight3, out5);
                
                out6 = mad(in2.x, weight0, out6);
                out6 = mad(in2.y, weight1, out6);
                out6 = mad(in2.z, weight2, out6);
                out6 = mad(in2.w, weight3, out6);
                
                out7 = mad(in3.x, weight0, out7);
                out7 = mad(in3.y, weight1, out7);
                out7 = mad(in3.z, weight2, out7);
                out7 = mad(in3.w, weight3, out7);
                
                weight_offset += 4;
            }
        }
    }

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT4)0);
    out1 = fmax(out1, (COMPUTE_FLOAT4)0);
    out2 = fmax(out2, (COMPUTE_FLOAT4)0);
    out3 = fmax(out3, (COMPUTE_FLOAT4)0);
    out4 = fmax(out4, (COMPUTE_FLOAT4)0);
    out5 = fmax(out5, (COMPUTE_FLOAT4)0);
    out6 = fmax(out6, (COMPUTE_FLOAT4)0);
    out7 = fmax(out7, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out1 = clamp(out1, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out2 = clamp(out2, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out3 = clamp(out3, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out4 = clamp(out4, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out5 = clamp(out5, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out6 = clamp(out6, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
    out7 = clamp(out7, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
#ifdef BLOCK_LEAVE
    const int remain = out_hw.y - out_w_idx;
    if(remain >= 4){
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
    }else if(remain == 3){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2), 2, output+out_offset);
    }else if(remain == 2){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0, out1)), 0, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out0), 0, output+out_offset);
    }
#ifdef CHANNEL_LEAVE
    if(out_c_idx + 1 >= out_c_blocks)return;
#endif
    out_offset = (((out_b_idx*out_c_blocks + out_c_idx + 1)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    if(remain >= 4){
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4, out5, out6, out7)), 0, output+out_offset);
    }else if(remain == 3){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4, out5)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out6), 2, output+out_offset);
    }else if(remain == 2){
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4, out5)), 0, output+out_offset);
    }else if(remain == 1){
        vstore4(CONVERT_FLOAT4(out4), 0, output+out_offset);
    }
#else
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0, out1, out2, out3)), 0, output+out_offset);
#ifdef CHANNEL_LEAVE
    if(out_c_idx + 1 >= out_c_blocks)return;
#endif
    out_offset = (((out_b_idx*out_c_blocks + out_c_idx + 1)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4, out5, out6, out7)), 0, output+out_offset);
#endif
}
