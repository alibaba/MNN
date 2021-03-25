#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void pooling(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                      __private const int2 input_shape,
                      __private const int2 output_shape,
                      __private const int2 pad_shape,
                      __private const int2 stride_shape,
                      __private const int2 kernel_shape,
                      __global FLOAT *output,
                      __private const int channel_block) {
                          
    const int ow_idx   = get_global_id(0);
    const int b_oh_idx = get_global_id(1);
    const int c_idx    = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(ow_idx, b_oh_idx, c_idx);
    
    const int b_idx = b_oh_idx / output_shape.x;
    const int oh_idx = b_oh_idx % output_shape.x;
    const int iw_start = mad24(ow_idx, stride_shape.y, -pad_shape.y);
    const int ih_start = mad24(oh_idx, stride_shape.x, -pad_shape.x);
    
    #ifdef POOL_AVG
    FLOAT4 result = (FLOAT4)(0);
    const int inp_offset = (((b_idx*channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start)*4;
    int total_count = 0;
    for(int kh=0; kh<kernel_shape.x; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }
        for(int kw=0; kw<kernel_shape.y; kw++) {
            int iw_cur = iw_start + kw;
            if(iw_cur < 0 || iw_cur >= input_shape.y) {
                continue;
            }
            FLOAT4 inp_data = vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4);
            result += inp_data;
            total_count++;
        }
    }
    result = result / (FLOAT4)(1.0*total_count);
    #else
    FLOAT4 result = (FLOAT4)(-FLT_MAX);
    const int inp_offset = (((b_idx*channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start)*4;
    for(int kh=0; kh<kernel_shape.x; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }
        for(int kw=0; kw<kernel_shape.y; kw++) {
            int iw_cur = iw_start + kw;
            if(iw_cur < 0 || iw_cur >= input_shape.y) {
                continue;
            }
            FLOAT4 inp_data = vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4);
            result = fmax(result, inp_data);
        }
    }
    #endif
    
    const int out_offset = (((b_idx*channel_block + c_idx)*output_shape.x + oh_idx)* output_shape.y + ow_idx)*4;
    vstore4(result, 0, output+out_offset);
}
