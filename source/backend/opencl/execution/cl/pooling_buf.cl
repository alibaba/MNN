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
                      __global FLOAT *rediceOutput,
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
    COMPUTE_FLOAT4 result = (COMPUTE_FLOAT4)(0);
    const int inp_offset = (((b_idx*channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start)*4;
    #ifdef COUNT_INCLUDE_PADDING
    int total_count = (min(ih_start + kernel_shape.x, input_shape.x + pad_shape.x) - ih_start) * (min(iw_start + kernel_shape.y, input_shape.y + pad_shape.y) - iw_start);
    #else
    int total_count = 0;
    #endif
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
            COMPUTE_FLOAT4 inp_data = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4));
            #if RETURN_REDICE
            redice = inp_data > result ? (int4)((ih_start + kh) * input_shape.y + iw_start + kw) : redice;
            #endif
            result = fmax(result, inp_data);
        }
    }
    #endif
    
    const int out_offset = (((b_idx*channel_block + c_idx)*output_shape.x + oh_idx)* output_shape.y + ow_idx)*4;
    vstore4(CONVERT_FLOAT4(result), 0, output+out_offset);
    #if RETURN_REDICE
    vstore4(CONVERT_FLOAT4(redice),  0, rediceOutput+out_offset);
    #endif
}

#ifdef LOCAL_SIZE
__kernel void global_pooling_buf(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                                __private const int2 input_shape,
                                __private const int2 output_shape,
                                __private const int2 pad_shape,
                                __private const int2 stride_shape,
                                __private const int2 kernel_shape,
                                __global FLOAT *output,
                                __global FLOAT *rediceOutput,
                                __private const int channel_block) {
    const int local_id                = get_local_id(0);
    const int output_channel_idx      = get_global_id(1);
    const int output_batch_idx        = get_global_id(2);

#ifdef POOL_AVG
    COMPUTE_FLOAT4 output_result = 0;
#else
    COMPUTE_FLOAT4 output_result = (COMPUTE_FLOAT4)(-FLT_MAX);
#if RETURN_REDICE
    int4 redice = (int4)0;
    int4 local rediceId[LOCAL_SIZE];
#endif
#endif

    COMPUTE_FLOAT4 local sum[LOCAL_SIZE];
    const int inp_offset = ((output_batch_idx*channel_block+output_channel_idx)*input_shape.x)*input_shape.y*4;
    const int size = input_shape.x * input_shape.y;
    for(int i = local_id; i < size; i+=LOCAL_SIZE){
        int w = i % input_shape.y;;
        int h = i / input_shape.y;
        COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input+inp_offset+(h*input_shape.y+w)*4));
#ifdef POOL_AVG
        output_result += in;
#else
        output_result = fmax(output_result, in);
#if RETURN_REDICE
        redice = in > output_result ? (int4)(i) : redice;
#endif
#endif
    }
    
    sum[local_id] = output_result;
#if RETURN_REDICE
    rediceId[local_id] = redice;
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
        if (local_id < i)
#ifdef POOL_AVG
            sum[local_id] = sum[local_id] + sum[local_id + i];
#else
        {
            sum[local_id] = fmax(sum[local_id], sum[local_id + i]);
#if RETURN_REDICE
            rediceId[local_id] = sum[local_id] > sum[local_id + i] ? rediceId[local_id] : rediceId[local_id + i];
#endif
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    output_result = sum[0];
#ifdef POOL_AVG
    output_result /= (input_shape.x * input_shape.y);
#endif

    const int out_offset = (output_batch_idx*channel_block + output_channel_idx)*4;
    vstore4(CONVERT_FLOAT4(output_result), 0, output+out_offset);
#if RETURN_REDICE
    redice = rediceId[0];
    vstore4(CONVERT_FLOAT4(redice),  0, rediceOutput+out_offset);
#endif
}
#endif
