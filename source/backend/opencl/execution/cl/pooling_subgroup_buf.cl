#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void pooling_c4_c4(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                      __private const int2 input_shape,
                      __private const int2 output_shape,
                      __private const int2 pad_shape,
                      __global FLOAT *output,
                      __private const int channel,
                      __private const int in_channel_block,
                      __private const int out_channel_block,
                      __private const int input_pad_left, 
                      __private const int input_pad_right, 
                      __private const int output_pad_left,
                      __private const int output_pad_right) {
                          
    const int ow_idx   = get_global_id(0);
    const int b_oh_idx = get_global_id(1);
    const int c_idx    = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(ow_idx, b_oh_idx, c_idx);
    
    const int b_idx = b_oh_idx / output_shape.x;
    const int oh_idx = b_oh_idx % output_shape.x;
    const int iw_start = mad24(ow_idx, STRIDE_X, -pad_shape.y);
    const int ih_start = mad24(oh_idx, STRIDE_Y, -pad_shape.x);
    
    #ifdef POOL_AVG
    FLOAT4 result = (FLOAT4)(0);
    const int inp_offset = (((b_idx*in_channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start+input_pad_left)*4;
    int total_count = 0;
    for(int kh=0; kh<KERNEL_Y; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }
        for(int kw=0; kw<KERNEL_X; kw++) {
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
    const int inp_offset = (((b_idx*in_channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start+input_pad_left)*4;
    for(int kh=0; kh<KERNEL_Y; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }
        for(int kw=0; kw<KERNEL_X; kw++) {
            int iw_cur = iw_start + kw;
            if(iw_cur < 0 || iw_cur >= input_shape.y) {
                continue;
            }
            FLOAT4 inp_data = vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4);
            result = fmax(result, inp_data);
        }
    }
    #endif
    
    const int out_offset = (((b_idx*in_channel_block + c_idx)*output_shape.x + oh_idx)* output_shape.y + ow_idx + output_pad_left)*4;
    vstore4(result, 0, output+out_offset);
}

__kernel void pooling_c4_c16(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                      __private const int2 input_shape,
                      __private const int2 output_shape,
                      __private const int2 pad_shape,
                      __global FLOAT *output,
                      __private const int channel,
                      __private const int in_channel_block,
                      __private const int out_channel_block,
                      __private const int input_pad_left, 
                      __private const int input_pad_right, 
                      __private const int output_pad_left,
                      __private const int output_pad_right) {
                          
    const int ow_idx   = get_global_id(0);
    const int b_oh_idx = get_global_id(1);
    const int c_idx    = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(ow_idx, b_oh_idx, c_idx);
    
    const int b_idx = b_oh_idx / output_shape.x;
    const int oh_idx = b_oh_idx % output_shape.x;
    const int iw_start = mad24(ow_idx, STRIDE_X, -pad_shape.y);
    const int ih_start = mad24(oh_idx, STRIDE_Y, -pad_shape.x);
    const int dst_width = output_shape.y + output_pad_left + output_pad_right;
    
    #ifdef POOL_AVG
    FLOAT4 result = (FLOAT4)(0);
    const int inp_offset = (((b_idx*in_channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start+input_pad_left)*4;
    int total_count = 0;
    for(int kh=0; kh<KERNEL_Y; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }
        for(int kw=0; kw<KERNEL_X; kw++) {
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
    const int inp_offset = (((b_idx*in_channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start+input_pad_left)*4;
    for(int kh=0; kh<KERNEL_Y; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }
        for(int kw=0; kw<KERNEL_X; kw++) {
            int iw_cur = iw_start + kw;
            if(iw_cur < 0 || iw_cur >= input_shape.y) {
                continue;
            }
            FLOAT4 inp_data = vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4);
            result = fmax(result, inp_data);
        }
    }
    #endif

    const int c_left = (c_idx % 4) * 4;
    const int out_offset = (((b_idx*out_channel_block + c_idx/4)*output_shape.x + oh_idx)* dst_width + ow_idx + output_pad_left)*16 + c_left;
    vstore4(result, 0, output+out_offset);
    if(ow_idx == 0){
        int pad_offset = (((b_idx*out_channel_block + c_idx/4)*output_shape.x + oh_idx)* dst_width + 0)*16 + c_left;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (output_shape.x + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pooling_c16_c16(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                      __private const int2 input_shape,
                      __private const int2 output_shape,
                      __private const int2 pad_shape,
                      __global FLOAT *output,
                      __private const int channel,
                      __private const int in_channel_block,
                      __private const int out_channel_block,
                      __private const int input_pad_left, 
                      __private const int input_pad_right, 
                      __private const int output_pad_left,
                      __private const int output_pad_right) {
                          
    const int ow_idx   = get_global_id(1) << 3;
    const int b_oh_idx = get_global_id(2);
    const int c_idx    = get_group_id(0);
    const int sglid = get_sub_group_local_id();
    
    const int b_idx = b_oh_idx / output_shape.x;
    const int oh_idx = b_oh_idx % output_shape.x;
    const int iw_start = mad24(ow_idx, STRIDE_X, -pad_shape.y);
    const int ih_start = mad24(oh_idx, STRIDE_Y, -pad_shape.x);
    const int src_width = input_shape.y + input_pad_left + input_pad_right;
    const int dst_width = output_shape.y + output_pad_left + output_pad_right;

#ifdef POOL_AVG
    FLOAT8 result = (FLOAT8)(0);
    FLOAT8 w_start = (FLOAT8)(iw_start, iw_start + STRIDE_X, iw_start + STRIDE_X * 2, iw_start + STRIDE_X * 3, iw_start + STRIDE_X * 4, iw_start + STRIDE_X * 5, iw_start + STRIDE_X * 6, iw_start + STRIDE_X * 7);
    w_start = fmax(w_start, (FLOAT8)0);
    FLOAT8 w_end = fmin(w_start + KERNEL_X, (FLOAT8)input_shape.y);
    float h_start = fmax((float)ih_start, 0);
    float h_end = fmin(h_start + KERNEL_Y, (float)input_shape.x);
    FLOAT8 total_count = (w_end - w_start) * (FLOAT8)(h_end - h_start);
#else
    FLOAT8 result = (FLOAT8)(-FLT_MAX);
#endif
    const int inp_offset = mul24(mad24(mad24(mad24(b_idx,in_channel_block,c_idx),input_shape.x,ih_start),src_width,iw_start+input_pad_left),16);
    for(int kh=0; kh<KERNEL_Y; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }

        FLOAT line_cache[INPUT_LINE_SIZE];
        for (int i = 0; i < INPUT_LINE_SIZE; i++) {
            if ((iw_start + i) >= 0 && (iw_start + i) < input_shape.y){
#ifdef MNN_SUPPORT_FP16
                line_cache[i] = as_half(intel_sub_group_block_read_us((__global ushort*)(input + inp_offset + mul24(mad24(kh,src_width,i),16))));
#else
                line_cache[i] = as_float(intel_sub_group_block_read((__global uint*)(input + inp_offset + mul24(mad24(kh,src_width,i),16))));
#endif
                //line_cache[i] = input[inp_offset + (kh*src_width + i)*16];
            } else{
#ifdef POOL_AVG
                line_cache[i] = 0;
#else
                line_cache[i] = (FLOAT)(-FLT_MAX);
#endif
            }
        }


        for(int kw=0; kw<KERNEL_X; kw++) {
            FLOAT8 src;
            __attribute__((opencl_unroll_hint(8)))
            for (int i = 0; i < 8; i++) {
                src[i] = line_cache[kw + STRIDE_X*i];
//                if ((iw_start + kw + STRIDE_X*i) >= 0 && (iw_start + kw + STRIDE_X*i) < input_shape_y){
//#ifdef MNN_SUPPORT_FP16
//                    src[i] = as_half(intel_sub_group_block_read_us((__global ushort*)(input + inp_offset + (kh*src_width + kw + STRIDE_X*i)*16)));
//#else
//                    src[i] = as_float(intel_sub_group_block_read((__global uint*)(input + inp_offset + (kh*src_width + kw + STRIDE_X*i)*16)));
//#endif
//                }else{
//#ifdef POOL_AVG
//                    src[i] = 0;
//#else
//                    src[i] = (FLOAT)(-FLT_MAX);
//#endif
//                }
            }
#ifdef POOL_AVG
            result += src;
#else
            result = fmax(result, src);
#endif
        }
    }
#ifdef POOL_AVG
    result = result / total_count;
#endif


    if(ow_idx == 0){
        int pad_offset = (((b_idx*out_channel_block + c_idx)*output_shape.x + oh_idx)* dst_width + 0)*16 + sglid;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset+i*16] = 0;
        }
        pad_offset += (output_shape.y + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset+i*16] = 0;
        }
    }
    
    const int out_offset = (((b_idx*out_channel_block + c_idx)*output_shape.x + oh_idx)* dst_width + ow_idx + output_pad_left)*16;
#if OUTPUT_LEFTOVERS
    if ((c_idx+1)*16 >= channel) {
        for (int i = 0; i < 8; i++) {
            if ((c_idx*16 + sglid < channel) && (ow_idx + i) < output_shape.y)
                output[out_offset + i * 16 + sglid] = result[i];
        }
    }
    else
#endif  
    {
        if (ow_idx + 8 <= output_shape.y) {
#ifdef MNN_SUPPORT_FP16
            intel_sub_group_block_write_us8((__global ushort*)(output + out_offset), as_ushort8(result));
#else
            intel_sub_group_block_write8((__global uint*)(output + out_offset), as_uint8(result));
#endif
        }else{
            for (int i = 0; i < output_shape.y % 8; i++) {
                output[out_offset + i * 16 + sglid] = result[i];
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pooling_c16_c4(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                      __private const int2 input_shape,
                      __private const int2 output_shape,
                      __private const int2 pad_shape,
                      __global FLOAT *output,
                      __private const int channel,
                      __private const int in_channel_block,
                      __private const int out_channel_block,
                      __private const int input_pad_left, 
                      __private const int input_pad_right, 
                      __private const int output_pad_left,
                      __private const int output_pad_right) {
                          
    const int ow_idx   = get_global_id(1) << 3;
    const int b_oh_idx = get_global_id(2);
    const int c_idx    = get_group_id(0);
    const int sglid = get_sub_group_local_id();
    
    const int b_idx = b_oh_idx / output_shape.x;
    const int oh_idx = b_oh_idx % output_shape.x;
    const int iw_start = mad24(ow_idx, STRIDE_X, -pad_shape.y);
    const int ih_start = mad24(oh_idx, STRIDE_Y, -pad_shape.x);
    const int src_width = input_shape.y + input_pad_left + input_pad_right;

#ifdef POOL_AVG
    FLOAT8 result = (FLOAT8)(0);
    FLOAT8 w_start = (FLOAT8)(iw_start, iw_start + STRIDE_X, iw_start + STRIDE_X * 2, iw_start + STRIDE_X * 3, iw_start + STRIDE_X * 4, iw_start + STRIDE_X * 5, iw_start + STRIDE_X * 6, iw_start + STRIDE_X * 7);
    w_start = fmax(w_start, (FLOAT8)0);
    FLOAT8 w_end = fmin(w_start + KERNEL_X, (FLOAT8)input_shape.y);
    float h_start = fmax((float)ih_start, 0);
    float h_end = fmin(h_start + KERNEL_Y, (float)input_shape.x);
    FLOAT8 total_count = (w_end - w_start) * (FLOAT8)(h_end - h_start);
#else
    FLOAT8 result = (FLOAT8)(-FLT_MAX);
#endif
    const int inp_offset = mul24(mad24(mad24(mad24(b_idx,in_channel_block,c_idx),input_shape.x,ih_start),src_width,iw_start+input_pad_left),16);
    for(int kh=0; kh<KERNEL_Y; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }

        FLOAT line_cache[INPUT_LINE_SIZE];
        for (int i = 0; i < INPUT_LINE_SIZE; i++) {
            if ((iw_start + i) >= 0 && (iw_start + i) < input_shape.y){
#ifdef MNN_SUPPORT_FP16
                line_cache[i] = as_half(intel_sub_group_block_read_us((__global ushort*)(input + inp_offset + mul24(mad24(kh,src_width,i),16))));
#else
                line_cache[i] = as_float(intel_sub_group_block_read((__global uint*)(input + inp_offset + mul24(mad24(kh,src_width,i),16))));
#endif
                //line_cache[i] = input[inp_offset + (kh*src_width + i)*16];
            } else{
#ifdef POOL_AVG
                line_cache[i] = 0;
#else
                line_cache[i] = (FLOAT)(-FLT_MAX);
#endif
            }
        }


        for(int kw=0; kw<KERNEL_X; kw++) {
            FLOAT8 src;
            __attribute__((opencl_unroll_hint(8)))
            for (int i = 0; i < 8; i++) {
                src[i] = line_cache[kw + STRIDE_X*i];
//                if ((iw_start + kw + STRIDE_X*i) >= 0 && (iw_start + kw + STRIDE_X*i) < input_shape_y){
//#ifdef MNN_SUPPORT_FP16
//                    src[i] = as_half(intel_sub_group_block_read_us((__global ushort*)(input + inp_offset + (kh*src_width + kw + STRIDE_X*i)*16)));
//#else
//                    src[i] = as_float(intel_sub_group_block_read((__global uint*)(input + inp_offset + (kh*src_width + kw + STRIDE_X*i)*16)));
//#endif
//                }else{
//#ifdef POOL_AVG
//                    src[i] = 0;
//#else
//                    src[i] = (FLOAT)(-FLT_MAX);
//#endif
//                }
            }
#ifdef POOL_AVG
            result += src;
#else
            result = fmax(result, src);
#endif
        }
    }
#ifdef POOL_AVG
    result = result / total_count;
#endif


    const uint lid_x = sglid % 4;
    const uint lid_y = sglid / 4;
    
    const int out_offset = (((b_idx*out_channel_block + c_idx * 4)*output_shape.x + oh_idx)* output_shape.y + ow_idx + output_pad_left)*4;
    const int width_height = output_shape.y * output_shape.x * 4;
#if OUTPUT_LEFTOVERS
    if ((c_idx+1)*16 >= channel) {
        for (int i = 0; i < 8; i++) {
            if ((c_idx*16 + lid_y * 4 + lid_x < channel) && (ow_idx + i) < output_shape.y)
                output[out_offset + lid_y * width_height + i * 4 + lid_x] = result[i];
        }
    }
    else
#endif  
    {
        for (int i = 0; i < 8 && (ow_idx + i) < output_shape.y; i++) {
            output[out_offset + lid_y * width_height + i * 4 + lid_x] = result[i];
        }
    }
}