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
                      __global FLOAT *rediceOutput,
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
    COMPUTE_FLOAT4 result = (COMPUTE_FLOAT4)(0);
    const int inp_offset = (((b_idx*in_channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start+input_pad_left)*4;
#ifdef COUNT_INCLUDE_PADDING
    int total_count = (min(ih_start + KERNEL_Y, input_shape.x + pad_shape.x) - ih_start) * (min(iw_start + KERNEL_X, input_shape.y + pad_shape.y) - iw_start);
#else
    int total_count = 0;
#endif
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
            COMPUTE_FLOAT4 inp_data = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4));
            result += inp_data;
#ifndef COUNT_INCLUDE_PADDING
            total_count++;
#endif
        }
    }
    result = result / (COMPUTE_FLOAT4)(1.0*total_count);
    #else
    COMPUTE_FLOAT4 result = (COMPUTE_FLOAT4)(-FLT_MAX);
    #if RETURN_REDICE
    int4 redice = (int4)0;
    #endif
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
            COMPUTE_FLOAT4 inp_data = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4));
            #if RETURN_REDICE
            redice = inp_data > result ? (int4)((ih_start + kh) * input_shape.y + iw_start + kw) : redice;
            #endif
            result = fmax(result, inp_data);
        }
    }
    #endif
    
    const int out_offset = (((b_idx*in_channel_block + c_idx)*output_shape.x + oh_idx)* output_shape.y + ow_idx + output_pad_left)*4;
    vstore4(CONVERT_FLOAT4(result), 0, output+out_offset);
    #if RETURN_REDICE
    vstore4(CONVERT_FLOAT4(redice),  0, rediceOutput+(((b_idx*in_channel_block + c_idx)*output_shape.x + oh_idx)* output_shape.y + ow_idx)*4);
    #endif
}

__kernel void pooling_c4_c16(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                      __private const int2 input_shape,
                      __private const int2 output_shape,
                      __private const int2 pad_shape,
                      __global FLOAT *output,
                      __global FLOAT *rediceOutput,
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
    COMPUTE_FLOAT4 result = (COMPUTE_FLOAT4)(0);
    const int inp_offset = (((b_idx*in_channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start+input_pad_left)*4;
 #ifdef COUNT_INCLUDE_PADDING
    int total_count = (min(ih_start + KERNEL_Y, input_shape.x + pad_shape.x) - ih_start) * (min(iw_start + KERNEL_X, input_shape.y + pad_shape.y) - iw_start);
#else
    int total_count = 0;
#endif
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
            COMPUTE_FLOAT4 inp_data = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4));
            result += inp_data;
#ifndef COUNT_INCLUDE_PADDING
            total_count++;
#endif
        }
    }
    result = result / (COMPUTE_FLOAT4)(1.0*total_count);
    #else
    COMPUTE_FLOAT4 result = (COMPUTE_FLOAT4)(-FLT_MAX);
    #if RETURN_REDICE
    int4 redice = (int4)0;
    #endif
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
            COMPUTE_FLOAT4 inp_data = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4));
            #if RETURN_REDICE
            redice = inp_data > result ? (int4)((ih_start + kh) * input_shape.y + iw_start + kw) : redice;
            #endif
            result = fmax(result, inp_data);
        }
    }
    #endif

    const int c_left = (c_idx % 4) * 4;
    const int out_offset = (((b_idx*out_channel_block + c_idx/4)*output_shape.x + oh_idx)* dst_width + ow_idx + output_pad_left)*16 + c_left;
    vstore4(CONVERT_FLOAT4(result), 0, output+out_offset);
    #if RETURN_REDICE
    vstore4(CONVERT_FLOAT4(redice),  0, rediceOutput+(((b_idx*out_channel_block + c_idx)*output_shape.x + oh_idx)* output_shape.y + ow_idx)*4);
    #endif
    if(ow_idx == 0){
        int pad_offset = (((b_idx*out_channel_block + c_idx/4)*output_shape.x + oh_idx)* dst_width + 0)*16 + c_left;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (output_shape.y + output_pad_left) * 16;
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
                      __global FLOAT *rediceOutput,
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
    COMPUTE_FLOAT8 result = (COMPUTE_FLOAT8)(0);
    COMPUTE_FLOAT8 w_start = (COMPUTE_FLOAT8)(iw_start, iw_start + STRIDE_X, iw_start + STRIDE_X * 2, iw_start + STRIDE_X * 3, iw_start + STRIDE_X * 4, iw_start + STRIDE_X * 5, iw_start + STRIDE_X * 6, iw_start + STRIDE_X * 7);
#ifdef COUNT_INCLUDE_PADDING
    COMPUTE_FLOAT8 w_size      = fmin(w_start + KERNEL_X, input_shape.y + pad_shape.y) - w_start;
    COMPUTE_FLOAT8 total_count = (COMPUTE_FLOAT8)(min(ih_start + KERNEL_Y, input_shape.x + pad_shape.x) - ih_start) * w_size;
#else
    w_start = fmax(w_start, (COMPUTE_FLOAT8)0);
    COMPUTE_FLOAT8 w_end = fmin(w_start + KERNEL_X, (COMPUTE_FLOAT8)input_shape.y);
    float h_start = fmax((float)ih_start, 0);
    float h_end = fmin(h_start + KERNEL_Y, (float)input_shape.x);
    COMPUTE_FLOAT8 total_count = (w_end - w_start) * (COMPUTE_FLOAT8)(h_end - h_start);
#endif
#else
    COMPUTE_FLOAT8 result = (COMPUTE_FLOAT8)(-FLT_MAX);
#if RETURN_REDICE
    int8 redice = (int8)0;
#endif
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
            } else{
#ifdef POOL_AVG
                line_cache[i] = 0;
#else
                line_cache[i] = (COMPUTE_FLOAT)(-FLT_MAX);
#endif
            }
        }


        for(int kw=0; kw<KERNEL_X; kw++) {
            COMPUTE_FLOAT8 src;
            __attribute__((opencl_unroll_hint(8)))
            for (int i = 0; i < 8; i++) {
                src[i] = line_cache[kw + STRIDE_X*i];
            }
#ifdef POOL_AVG
            result += src;
#else
#if RETURN_REDICE
            redice = src > result ? (int8)((ih_start + kh) * input_shape.y + iw_start + kw) : redice;
#endif
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
            intel_sub_group_block_write_us8((__global ushort*)(output + out_offset), as_ushort8(CONVERT_FLOAT8(result)));
#else
            intel_sub_group_block_write8((__global uint*)(output + out_offset), as_uint8(CONVERT_FLOAT8(result)));
#endif
        }else{
            for (int i = 0; i < output_shape.y % 8; i++) {
                output[out_offset + i * 16 + sglid] = result[i];
            }
        }
    }
#ifdef RETURN_REDICE
    const uint lid_x = sglid % 4;
    const uint lid_y = sglid / 4;
    
    const int width_height = output_shape.y * output_shape.x * 4;
    const int redice_offset = (((b_idx*out_channel_block + c_idx * 4)*output_shape.x + oh_idx)* output_shape.y + ow_idx)*4;
#if OUTPUT_LEFTOVERS
    if ((c_idx+1)*16 >= channel) {
        for (int i = 0; i < 8; i++) {
            if ((c_idx*16 + lid_y * 4 + lid_x < channel) && (ow_idx + i) < output_shape.y)
                rediceOutput[redice_offset + lid_y * width_height + i * 4 + lid_x] = redice[i];
        }
    }
    else
#endif
    {
        for (int i = 0; i < 8 && (ow_idx + i) < output_shape.y; i++) {
            rediceOutput[redice_offset + lid_y * width_height + i * 4 + lid_x] = redice[i];
        }
    }
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pooling_c16_c4(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                      __private const int2 input_shape,
                      __private const int2 output_shape,
                      __private const int2 pad_shape,
                      __global FLOAT *output,
                      __global FLOAT *rediceOutput,
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
    COMPUTE_FLOAT8 result = (COMPUTE_FLOAT8)(0);
    COMPUTE_FLOAT8 w_start = (COMPUTE_FLOAT8)(iw_start, iw_start + STRIDE_X, iw_start + STRIDE_X * 2, iw_start + STRIDE_X * 3, iw_start + STRIDE_X * 4, iw_start + STRIDE_X * 5, iw_start + STRIDE_X * 6, iw_start + STRIDE_X * 7);
#ifdef COUNT_INCLUDE_PADDING
    COMPUTE_FLOAT8 w_size      = fmin(w_start + KERNEL_X, input_shape.y + pad_shape.y) - w_start;
    COMPUTE_FLOAT8 total_count = (COMPUTE_FLOAT8)(min(ih_start + KERNEL_Y, input_shape.x + pad_shape.x) - ih_start) * w_size;
#else
    w_start = fmax(w_start, (COMPUTE_FLOAT8)0);
    COMPUTE_FLOAT8 w_end = fmin(w_start + KERNEL_X, (COMPUTE_FLOAT8)input_shape.y);
    float h_start = fmax((float)ih_start, 0);
    float h_end = fmin(h_start + KERNEL_Y, (float)input_shape.x);
    COMPUTE_FLOAT8 total_count = (w_end - w_start) * (COMPUTE_FLOAT8)(h_end - h_start);
#endif
#else
    COMPUTE_FLOAT8 result = (COMPUTE_FLOAT8)(-FLT_MAX);
#if RETURN_REDICE
    int8 redice = (int8)0;
#endif
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
            } else{
#ifdef POOL_AVG
                line_cache[i] = 0;
#else
                line_cache[i] = (FLOAT)(-FLT_MAX);
#endif
            }
        }


        for(int kw=0; kw<KERNEL_X; kw++) {
            COMPUTE_FLOAT8 src;
            __attribute__((opencl_unroll_hint(8)))
            for (int i = 0; i < 8; i++) {
                src[i] = line_cache[kw + STRIDE_X*i];
            }
#ifdef POOL_AVG
            result += src;
#else
#if RETURN_REDICE
            redice = src > result ? (int8)((ih_start + kh) * input_shape.y + iw_start + kw) : redice;
#endif
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
#if RETURN_REDICE
    const int redice_offset = (((b_idx*out_channel_block + c_idx * 4)*output_shape.x + oh_idx)* output_shape.y + ow_idx)*4;
#endif
#if OUTPUT_LEFTOVERS
    if ((c_idx+1)*16 >= channel) {
        for (int i = 0; i < 8; i++) {
            if ((c_idx*16 + lid_y * 4 + lid_x < channel) && (ow_idx + i) < output_shape.y)
                output[out_offset + lid_y * width_height + i * 4 + lid_x] = result[i];
#if RETURN_REDICE
                rediceOutput[redice_offset + lid_y * width_height + i * 4 + lid_x] = redice[i];
#endif
        }
    }
    else
#endif  
    {
        for (int i = 0; i < 8 && (ow_idx + i) < output_shape.y; i++) {
            output[out_offset + lid_y * width_height + i * 4 + lid_x] = result[i];
#if RETURN_REDICE
            rediceOutput[redice_offset + lid_y * width_height + i * 4 + lid_x] = redice[i];
#endif
        }
    }
}
