#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define READ_INPUT_IMAGE(i, base)                                                                         \
    int in_width_value##i = in_width##i + base;                                                           \
    in_width_value##i =                                                                                   \
        select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_shape.y)); \
    in_w_idx = in_width_value##i % input_shape.y; \
    inp_offset = (((in_b_idx*in_channel_block_length + in_channel_block_idx)*input_shape.x + in_h_idx)* input_shape.y + in_w_idx)*4; \
    in##i = (in_width_value##i)==-1 ? (FLOAT4)0 : vload4(0, input+inp_offset);

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

__kernel
void conv_2d_c4h1w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
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
                      __private const int out_c_blocks) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_hw.y;
    const int out_w_idx = out_c_w_idx % out_hw.y;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    FLOAT4 out0 = vload4(out_c_idx, bias);
    
    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);

    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < (ushort)IN_C_BLOCK; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + kw_start) * 4;
        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            for(int ix = in_w_idx_start; ix < in_w_idx_end; ix += dilate_hw.y) {
                int inp_offset = (((out_b_idx * in_c_blocks + in_c_idx) * in_hw.x + iy) * in_hw.y + ix) * 4;
                FLOAT4 in0 = vload4(0, input+inp_offset);
                
                const int filter_w_inc = (ix-in_w_idx_start)/dilate_hw.y;
                FLOAT4 weight0 = vload4(filter_w_inc, weight+weight_offset);
                FLOAT4 weight1 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset);
                FLOAT4 weight2 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset*2);
                FLOAT4 weight3 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset*3);
                
                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
            }
            weight_offset += 4*filter_hw.y;
        }
    }
#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(out0, 0, output+out_offset);
 
}


__kernel
void conv_2d_c8h1w1(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
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
                      __private const int out_c_blocks) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = (out_c_w_idx / out_hw.y) << 1;
    const int out_w_idx = out_c_w_idx % out_hw.y;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    FLOAT4 out0 = vload4(out_c_idx, bias);
    FLOAT4 out1 = vload4(out_c_idx+1, bias);
    
    const int in_w_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kw_start = select(0, (-in_w_idx_base + dilate_hw.y - 1) / dilate_hw.y, in_w_idx_base < 0);
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);

    const int in_w_idx_start = mad24(kw_start, dilate_hw.y, in_w_idx_base);
    const int in_w_idx_end = min(mad24(filter_hw.y, dilate_hw.y, in_w_idx_base), in_hw.y);
    
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = filter_hw.x * filter_hw.y * 4;
    const int weight_ic_offset = out_c_blocks * weight_oc_offset;
    for(ushort in_c_idx = 0; in_c_idx < (ushort)IN_C_BLOCK; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + kw_start) * 4;
        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            for(int ix = in_w_idx_start; ix < in_w_idx_end; ix += dilate_hw.y) {
                int inp_offset = (((out_b_idx * in_c_blocks + in_c_idx) * in_hw.x + iy) * in_hw.y + ix) * 4;
                FLOAT4 in0 = vload4(0, input+inp_offset);
                
                const int filter_w_inc = (ix-in_w_idx_start)/dilate_hw.y;
                FLOAT4 weight0 = vload4(filter_w_inc, weight+weight_offset);
                FLOAT4 weight1 = vload4(filter_w_inc, weight+weight_offset+weight_ic_offset);
                FLOAT4 weight2 = vload4(filter_w_inc, weight+weight_offset+weight_ic_offset*2);
                FLOAT4 weight3 = vload4(filter_w_inc, weight+weight_offset+weight_ic_offset*3);
                
                out0 = mad(in0.x, weight0, out0);
                out0 = mad(in0.y, weight1, out0);
                out0 = mad(in0.z, weight2, out0);
                out0 = mad(in0.w, weight3, out0);
                
                weight0 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset);
                weight1 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset+weight_ic_offset);
                weight2 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset+weight_ic_offset*2);
                weight3 = vload4(filter_w_inc, weight+weight_offset+weight_oc_offset+weight_ic_offset*3);
                
                out1 = mad(in0.x, weight0, out1);
                out1 = mad(in0.y, weight1, out1);
                out1 = mad(in0.z, weight2, out1);
                out1 = mad(in0.w, weight3, out1);
            }
            weight_offset += 4*filter_hw.y;
        }
    }
#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(out0, 0, output+out_offset);
    if(out_c_idx+1 >= out_c_blocks) return;
    vstore4(out1, 0, output+out_offset+out_hw.x*out_hw.y*4);

}

__kernel
void conv_2d_c4h1w2(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
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
                      __private const int out_c_blocks) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 1;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    FLOAT4 out0 = vload4(out_c_idx, bias);
    FLOAT4 out1 = out0;
    
    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < (ushort)IN_C_BLOCK; in_c_idx++) {
        //weights  NC4HW4  [1,  4*icC4,  ocC4*kh*kw,  1] xic4
        //index:   [0, 4*in_c_idx, out_c_idx*kh*kw + kh_start*kw + kw_start, 0]
        int weight_offset = ((((4*in_c_idx+0)* out_c_blocks + out_c_idx) *filter_hw.x + kh_start)*filter_hw.y + 0) * 4;

        for(int iy = in_h_idx_start; iy < in_h_idx_end; iy += dilate_hw.x) {
            const int inp_offset_base = (((out_b_idx * in_c_blocks + in_c_idx) * in_hw.x + iy) * in_hw.y + 0) * 4;

            for(int fw = 0; fw < filter_hw.y; fw++) {
                const int in_w0_idx = fw * dilate_hw.y + in_w0_idx_base;
                const int in_w1_idx = fw * dilate_hw.y + in_w1_idx_base;

                FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w0_idx, input+inp_offset_base);
                FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w1_idx, input+inp_offset_base);
                
                FLOAT4 weight0 = vload4(0, weight+weight_offset);
                FLOAT4 weight1 = vload4(0, weight+weight_offset+weight_oc_offset);
                FLOAT4 weight2 = vload4(0, weight+weight_offset+weight_oc_offset*2);
                FLOAT4 weight3 = vload4(0, weight+weight_offset+weight_oc_offset*3);
                
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
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(out0, 0, output+out_offset);
    if(out_w_idx + 1 >= out_hw.y) return;
    vstore4(out1, 1, output+out_offset);
}

__kernel
void conv_2d_c4h1w4(GLOBAL_SIZE_2_DIMS
                      __global const FLOAT *input,
                      __global const FLOAT *weight,
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
                      __private const int out_c_blocks) {
    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = (out_c_w_idx % out_w_blocks) << 2;
    const int out_b_idx = out_b_h_idx / out_hw.x;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_hw.x;
    
    FLOAT4 out0 = vload4(out_c_idx, bias);
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int in_w0_idx_base = mad24(out_w_idx, stride_hw.y, -pad_hw.y);
    const int in_w1_idx_base = in_w0_idx_base + stride_hw.y;
    const int in_w2_idx_base = in_w1_idx_base + stride_hw.y;
    const int in_w3_idx_base = in_w2_idx_base + stride_hw.y;

    const int in_h_idx_base = mad24(out_h_idx, stride_hw.x, -pad_hw.x);
    
    const int kh_start = select(0, (-in_h_idx_base + dilate_hw.x - 1) / dilate_hw.x, in_h_idx_base < 0);
    const int in_h_idx_start = mad24(kh_start, dilate_hw.x, in_h_idx_base);
    const int in_h_idx_end = min(mad24(filter_hw.x, dilate_hw.x, in_h_idx_base), in_hw.x);
    
    const int weight_oc_offset = out_c_blocks * filter_hw.x * filter_hw.y * 4;
    for(ushort in_c_idx = 0; in_c_idx < (ushort)IN_C_BLOCK; in_c_idx++) {
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

                FLOAT4 in0 = (in_w0_idx < 0 || in_w0_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w0_idx, input+inp_offset_base);
                FLOAT4 in1 = (in_w1_idx < 0 || in_w1_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w1_idx, input+inp_offset_base);
                FLOAT4 in2 = (in_w2_idx < 0 || in_w2_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w2_idx, input+inp_offset_base);
                FLOAT4 in3 = (in_w3_idx < 0 || in_w3_idx >= in_hw.y) ? (FLOAT4)0 : vload4(in_w3_idx, input+inp_offset_base);

                FLOAT4 weight0 = vload4(0, weight+weight_offset);
                FLOAT4 weight1 = vload4(0, weight+weight_offset+weight_oc_offset);
                FLOAT4 weight2 = vload4(0, weight+weight_offset+weight_oc_offset*2);
                FLOAT4 weight3 = vload4(0, weight+weight_offset+weight_oc_offset*3);
                
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
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_blocks + out_c_idx)*out_hw.x + out_h_idx)*out_hw.y + out_w_idx)*4;
    vstore4(out0, 0, output+out_offset);
    if(out_w_idx + 1 >= out_hw.y) return;
    vstore4(out1, 1, output+out_offset);
    if(out_w_idx + 2 >= out_hw.y) return;
    vstore4(out2, 2, output+out_offset);
    if(out_w_idx + 3 >= out_hw.y) return;
    vstore4(out3, 3, output+out_offset);
}

__kernel
void conv_2d_1x1_c4h1w4(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_c_block) {

    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx
    
    const int out_w4_idx = mul24(out_w_idx, 4);
    FLOAT4 out0 = vload4(out_c_idx, (__global FLOAT *)bias_ptr);
    
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int intput_width_idx0 = out_w4_idx;
    
    int offset = mul24(out_c_idx, in_c_block) << 2;
    int inp_offset =
    (((out_b_idx*in_c_block)*out_h + out_h_idx)* out_w + intput_width_idx0) << 2;
    
    const inp_add = out_h*out_w*4;
    for (ushort in_channel_block_idx = 0; in_channel_block_idx < (ushort)IN_C_BLOCK; ++in_channel_block_idx) {

        FLOAT4 in0 = vload4(0, input+inp_offset);
        FLOAT4 in1 = vload4(1, input+inp_offset);;
        FLOAT4 in2 = vload4(2, input+inp_offset);;
        FLOAT4 in3 = vload4(3, input+inp_offset);;

        FLOAT4 weights0 = vload4(offset, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights1 = vload4(offset + 1, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights2 = vload4(offset + 2, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights3 = vload4(offset + 3, (__global FLOAT *)kernel_ptr);
        
        out0.x += dot(weights0, in0);
        out0.y += dot(weights1, in0);
        out0.z += dot(weights2, in0);
        out0.w += dot(weights3, in0);

        out1.x += dot(weights0, in1);
        out1.y += dot(weights1, in1);
        out1.z += dot(weights2, in1);
        out1.w += dot(weights3, in1);

        out2.x += dot(weights0, in2);
        out2.y += dot(weights1, in2);
        out2.z += dot(weights2, in2);
        out2.w += dot(weights3, in2);

        out3.x += dot(weights0, in3);
        out3.y += dot(weights1, in3);
        out3.z += dot(weights2, in3);
        out3.w += dot(weights3, in3);
        
        offset += 4;
        inp_offset += inp_add;
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_block + out_c_idx)*out_h + out_h_idx)* out_w + out_w4_idx)*4;

    const int remain = out_w - out_w4_idx;

    if (remain >= 4) {
        vstore4(out0, 0, output+out_offset);
        vstore4(out1, 1, output+out_offset);
        vstore4(out2, 2, output+out_offset);
        vstore4(out3, 3, output+out_offset);
    } else if (remain == 3) {
        vstore4(out0, 0, output+out_offset);
        vstore4(out1, 1, output+out_offset);
        vstore4(out2, 2, output+out_offset);
    } else if (remain == 2) {
        vstore4(out0, 0, output+out_offset);
        vstore4(out1, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(out0, 0, output+out_offset);
    }

}



__kernel
void conv_2d_1x1_c8h1w4(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_c_block) {

    const int out_c_w_idx = get_global_id(0); //c/8 w/4
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx
    
    const int out_w4_idx = mul24(out_w_idx, 4);
    FLOAT4 out0 = vload4(out_c_idx<<1, (__global FLOAT *)bias_ptr);
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;
    
    FLOAT4 out4 = vload4((out_c_idx<<1)+1, (__global FLOAT *)bias_ptr);
    FLOAT4 out5 = out4;
    FLOAT4 out6 = out4;
    FLOAT4 out7 = out4;

    const int intput_width_idx0 = out_w4_idx;
    
    for (int in_channel_block_idx = 0; in_channel_block_idx < IN_C_BLOCK; ++in_channel_block_idx) {
        int input_width_base  = mul24(in_channel_block_idx, out_w);

        int offset = mad24(out_c_idx, in_c_block, in_channel_block_idx)*8;
        const int inp_offset =
        (((out_b_idx*in_c_block + in_channel_block_idx)*out_h + out_h_idx)* out_w + intput_width_idx0)*4;
        
        FLOAT4 in0 = vload4(0, input+inp_offset);
        FLOAT4 in1 = vload4(1, input+inp_offset);;
        FLOAT4 in2 = vload4(2, input+inp_offset);;
        FLOAT4 in3 = vload4(3, input+inp_offset);;

        FLOAT4 weights0 = vload4(offset, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights1 = vload4(offset + 1, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights2 = vload4(offset + 2, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights3 = vload4(offset + 3, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights4 = vload4(offset + 4, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights5 = vload4(offset + 5, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights6 = vload4(offset + 6, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights7 = vload4(offset + 7, (__global FLOAT *)kernel_ptr);
        
        out0.x += dot(weights0, in0);
        out0.y += dot(weights1, in0);
        out0.z += dot(weights2, in0);
        out0.w += dot(weights3, in0);

        out1.x += dot(weights0, in1);
        out1.y += dot(weights1, in1);
        out1.z += dot(weights2, in1);
        out1.w += dot(weights3, in1);

        out2.x += dot(weights0, in2);
        out2.y += dot(weights1, in2);
        out2.z += dot(weights2, in2);
        out2.w += dot(weights3, in2);

        out3.x += dot(weights0, in3);
        out3.y += dot(weights1, in3);
        out3.z += dot(weights2, in3);
        out3.w += dot(weights3, in3);
        
        out4.x += dot(weights4, in0);
        out4.y += dot(weights5, in0);
        out4.z += dot(weights6, in0);
        out4.w += dot(weights7, in0);

        out5.x += dot(weights4, in1);
        out5.y += dot(weights5, in1);
        out5.z += dot(weights6, in1);
        out5.w += dot(weights7, in1);

        out6.x += dot(weights4, in2);
        out6.y += dot(weights5, in2);
        out6.z += dot(weights6, in2);
        out6.w += dot(weights7, in2);

        out7.x += dot(weights4, in3);
        out7.y += dot(weights5, in3);
        out7.z += dot(weights6, in3);
        out7.w += dot(weights7, in3);

    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
    
    out4 = fmax(out4, (FLOAT4)0);
    out5 = fmax(out5, (FLOAT4)0);
    out6 = fmax(out6, (FLOAT4)0);
    out7 = fmax(out7, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
    
    out4 = clamp(out4, (FLOAT4)0, (FLOAT4)6);
    out5 = clamp(out5, (FLOAT4)0, (FLOAT4)6);
    out6 = clamp(out6, (FLOAT4)0, (FLOAT4)6);
    out7 = clamp(out7, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_block + out_c_idx*2)*out_h + out_h_idx)* out_w + out_w4_idx)*4;

    const int remain = out_w - out_w4_idx;

    __global FLOAT* _tempoutput = output + out_offset;
    __global FLOAT* _tempoutput1 = _tempoutput + 4*out_h*out_w;

    if (remain >= 4) {
        vstore4(out0, 0, _tempoutput);
        vstore4(out1, 1, _tempoutput);
        vstore4(out2, 2, _tempoutput);
        vstore4(out3, 3, _tempoutput);
    } else if (remain == 3) {
        vstore4(out0, 0, _tempoutput);
        vstore4(out1, 1, _tempoutput);
        vstore4(out2, 2, _tempoutput);
    } else if (remain == 2) {
        vstore4(out0, 0, _tempoutput);
        vstore4(out1, 1, _tempoutput);
    } else if (remain == 1) {
        vstore4(out0, 0, _tempoutput);
    }
    if(out_c_idx*2+1 >= out_c_block) {
        return;
    }
    if (remain >= 4) {
        vstore4(out4, 0, _tempoutput1);
        vstore4(out5, 1, _tempoutput1);
        vstore4(out6, 2, _tempoutput1);
        vstore4(out7, 3, _tempoutput1);
    } else if (remain == 3) {
        vstore4(out4, 0, _tempoutput1);
        vstore4(out5, 1, _tempoutput1);
        vstore4(out6, 2, _tempoutput1);
    } else if (remain == 2) {
        vstore4(out4, 0, _tempoutput1);
        vstore4(out5, 1, _tempoutput1);
    } else if (remain == 1) {
        vstore4(out4, 0, _tempoutput1);
    }
}


__kernel
void conv_2d_1x1_c8h1w2(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_c_block) {

    const int out_c_w_idx = get_global_id(0); //c/8 w/4
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx
    
    const int out_w2_idx = mul24(out_w_idx, 2);
    FLOAT4 out0 = vload4(out_c_idx<<1, (__global FLOAT *)bias_ptr);
    FLOAT4 out1 = out0;
    
    FLOAT4 out4 = vload4((out_c_idx<<1)+1, (__global FLOAT *)bias_ptr);
    FLOAT4 out5 = out4;

    const int intput_width_idx0 = out_w2_idx;
    
    for (int in_channel_block_idx = 0; in_channel_block_idx < IN_C_BLOCK; ++in_channel_block_idx) {
        int input_width_base  = mul24(in_channel_block_idx, out_w);

        int offset = mad24(out_c_idx, in_c_block, in_channel_block_idx)*8;
        const int inp_offset =
        (((out_b_idx*in_c_block + in_channel_block_idx)*out_h + out_h_idx)* out_w + intput_width_idx0)*4;
        
        FLOAT4 in0 = vload4(0, input+inp_offset);
        FLOAT4 in1 = vload4(1, input+inp_offset);;

        FLOAT4 weights0 = vload4(offset, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights1 = vload4(offset + 1, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights2 = vload4(offset + 2, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights3 = vload4(offset + 3, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights4 = vload4(offset + 4, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights5 = vload4(offset + 5, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights6 = vload4(offset + 6, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights7 = vload4(offset + 7, (__global FLOAT *)kernel_ptr);
        
        out0.x += dot(weights0, in0);
        out0.y += dot(weights1, in0);
        out0.z += dot(weights2, in0);
        out0.w += dot(weights3, in0);

        out1.x += dot(weights0, in1);
        out1.y += dot(weights1, in1);
        out1.z += dot(weights2, in1);
        out1.w += dot(weights3, in1);
        
        out4.x += dot(weights4, in0);
        out4.y += dot(weights5, in0);
        out4.z += dot(weights6, in0);
        out4.w += dot(weights7, in0);

        out5.x += dot(weights4, in1);
        out5.y += dot(weights5, in1);
        out5.z += dot(weights6, in1);
        out5.w += dot(weights7, in1);
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);

    out4 = fmax(out4, (FLOAT4)0);
    out5 = fmax(out5, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);

    out4 = clamp(out4, (FLOAT4)0, (FLOAT4)6);
    out5 = clamp(out5, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_block + out_c_idx*2)*out_h + out_h_idx)* out_w + out_w2_idx)*4;

    const int remain = out_w - out_w2_idx;

    __global FLOAT* _tempoutput = output + out_offset;
    __global FLOAT* _tempoutput1 = _tempoutput + 4*out_h*out_w;

    if (remain >= 2) {
        vstore4(out0, 0, _tempoutput);
        vstore4(out1, 1, _tempoutput);
    } else if (remain == 1) {
        vstore4(out0, 0, _tempoutput);
    }
    if(out_c_idx*2+1 >= out_c_block) {
        return;
    }
    if (remain >= 2) {
        vstore4(out4, 0, _tempoutput1);
        vstore4(out5, 1, _tempoutput1);
    } else if (remain == 1) {
        vstore4(out4, 0, _tempoutput1);
    }
}

__kernel
void conv_2d_1x1_c4h1w1(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_c_block) {

    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w;
    const int out_w_idx = out_c_w_idx % out_w;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx
    
    FLOAT4 out0 = vload4(out_c_idx, (__global FLOAT *)bias_ptr);
    const int intput_width_idx0 = out_w_idx;
    
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {
        int input_width_base  = mul24(in_channel_block_idx, out_w);

        int offset = mad24(out_c_idx, in_c_block, in_channel_block_idx)*4;
        const int inp_offset =
        (((out_b_idx*in_c_block + in_channel_block_idx)*out_h + out_h_idx)* out_w + intput_width_idx0)*4;
        
        FLOAT4 in0 = vload4(0, input+inp_offset);

        FLOAT4 weights0 = vload4(offset, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights1 = vload4(offset + 1, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights2 = vload4(offset + 2, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights3 = vload4(offset + 3, (__global FLOAT *)kernel_ptr);
        
        out0.x += dot(weights0, in0);
        out0.y += dot(weights1, in0);
        out0.z += dot(weights2, in0);
        out0.w += dot(weights3, in0);
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_block + out_c_idx)*out_h + out_h_idx)* out_w + out_w_idx)*4;

    vstore4(out0, 0, output+out_offset);
}


__kernel
void conv_2d_1x1_c4h1w2(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks,
                          __global const FLOAT *input,
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          __global FLOAT *output,
                          __private const int in_c_block,
                          __private const int out_h,
                          __private const int out_w,
                          __private const int out_c_block) {

    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;
    const int out_b_idx = out_b_h_idx / out_h;//equal to in_b_idx
    const int out_h_idx = out_b_h_idx % out_h;//equal to in_h_idx
    
    const int out_w2_idx = mul24(out_w_idx, 2);

    FLOAT4 out0 = vload4(out_c_idx, (__global FLOAT *)bias_ptr);
    FLOAT4 out1 = out0;

    const int intput_width_idx0 = out_w2_idx;
    
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {
        int input_width_base  = mul24(in_channel_block_idx, out_w);

        int offset = mad24(out_c_idx, in_c_block, in_channel_block_idx)*4;
        const int inp_offset =
        (((out_b_idx*in_c_block + in_channel_block_idx)*out_h + out_h_idx)* out_w + intput_width_idx0)*4;
        
        FLOAT4 in0 = vload4(0, input+inp_offset);
        FLOAT4 in1 = vload4(1, input+inp_offset);;

        FLOAT4 weights0 = vload4(offset, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights1 = vload4(offset + 1, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights2 = vload4(offset + 2, (__global FLOAT *)kernel_ptr);
        FLOAT4 weights3 = vload4(offset + 3, (__global FLOAT *)kernel_ptr);
        
        out0.x += dot(weights0, in0);
        out0.y += dot(weights1, in0);
        out0.z += dot(weights2, in0);
        out0.w += dot(weights3, in0);

        out1.x += dot(weights0, in1);
        out1.y += dot(weights1, in1);
        out1.z += dot(weights2, in1);
        out1.w += dot(weights3, in1);
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_offset = (((out_b_idx*out_c_block + out_c_idx)*out_h + out_h_idx)* out_w + out_w2_idx)*4;

    const int remain = out_w - out_w2_idx;

    if (remain >= 2) {
        vstore4(out0, 0, output+out_offset);
        vstore4(out1, 1, output+out_offset);
    } else if (remain == 1) {
        vstore4(out0, 0, output+out_offset);
    }

}
