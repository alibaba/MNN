#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void pooling(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                      __private const int2 input_shape, __private const int output_height, __private const int2 pad_shape,
                      __private const int2 stride_shape,
                      __private const int2 kernel_shape,
                      __write_only image2d_t output,
                      __write_only image2d_t rediceOutput) {
    const int output_channel_idx      = get_global_id(0);
    const int output_width_idx        = get_global_id(1);
    const int output_batch_height_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_idx, output_width_idx, output_batch_height_idx);
    const int output_width = global_size_dim1;

    const int output_batch_idx    = output_batch_height_idx / output_height;
    const int output_height_idx   = output_batch_height_idx - mul24(output_batch_idx, output_height);
    const int input_start         = mul24(output_batch_idx, input_shape.x);
    const int input_height_start  = mad24(output_height_idx, stride_shape.x, -pad_shape.x);
    const int input_width_start   = mad24(output_width_idx, stride_shape.y, -pad_shape.y);
    const int input_channel_start = mul24(output_channel_idx, input_shape.y);

#ifdef POOL_AVG
    FLOAT4 output_result = 0;
    for (int height = 0; height < kernel_shape.x; height++) {
        int input_height_idx = input_height_start + height;
        input_height_idx =
            select(input_start + input_height_idx, -1, (input_height_idx < 0 || input_height_idx >= input_shape.x));
        for (int width = 0; width < kernel_shape.y; width++) {
            int input_width_idx = input_width_start + width;
            input_width_idx =
                select(input_channel_start + input_width_idx, -1, (input_width_idx < 0 || input_width_idx >= input_shape.y));

            FLOAT4 input_data = RI_F(input, SAMPLER, (int2)(input_width_idx, input_height_idx));
            output_result         = output_result + input_data;
        }
    }

    const int kernel_height_start = max(0, input_height_start);
    const int kernel_width_start  = max(0, input_width_start);
    const int kernel_height_end   = min(input_height_start + kernel_shape.x, input_shape.x);
    const int kernel_width_end    = min(input_width_start + kernel_shape.y, input_shape.y);
    #ifdef COUNT_INCLUDE_PADDING
    const int block_size = (min(input_height_start + kernel_shape.x, input_shape.x + pad_shape.x) - input_height_start) * (min(input_width_start + kernel_shape.y, input_shape.y + pad_shape.y) - input_width_start);
    #else
    const int block_size = mul24((kernel_height_end - kernel_height_start), (kernel_width_end - kernel_width_start));
    #endif
    const FLOAT block_float_req = (FLOAT)1.0f / (FLOAT)block_size;
    output_result = output_result * block_float_req;
#else
    FLOAT4 output_result = (FLOAT4)(-FLT_MAX);
    #if RETURN_REDICE
    int4 redice = (int4)0;
    #endif
    for (int height = 0; height < kernel_shape.x; height++) {
        int input_height_idx = input_height_start + height;
        input_height_idx =
            select(input_start + input_height_idx, -1, (input_height_idx < 0 || input_height_idx >= input_shape.x));
        if (input_height_idx != -1) {
            for (int width = 0; width < kernel_shape.y; width++) {
                int input_width_idx = input_width_start + width;
                input_width_idx     = select(input_channel_start + input_width_idx, -1,
                                         (input_width_idx < 0 || input_width_idx >= input_shape.y));

                if (input_width_idx != -1) {
                    FLOAT4 input_data = RI_F(input, SAMPLER, (int2)(input_width_idx, input_height_idx));
                    #if RETURN_REDICE
                    redice = input_data > output_result ? (int4)((input_height_start + height) * input_shape.y + input_width_start + width) : redice;
                    #endif
                    output_result         = fmax(output_result, input_data);
                }
            }
        }
    }
#endif

    const int output_channel_width_idx = mad24(output_channel_idx, output_width, output_width_idx);
    WI_F(output, (int2)(output_channel_width_idx, output_batch_height_idx), output_result);
    #if RETURN_REDICE
    WI_F(rediceOutput, (int2)(output_channel_width_idx, output_batch_height_idx), CONVERT_FLOAT4(redice));
    #endif
}

#ifdef LOCAL_SIZE
__kernel void global_pooling(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                            __private const int2 input_shape, __private const int output_height, __private const int2 pad_shape,
                            __private const int2 stride_shape,
                            __private const int2 kernel_shape,
                            __write_only image2d_t output,
                            __write_only image2d_t rediceOutput) {
    const int local_id                = get_local_id(0);
    const int output_channel_idx      = get_global_id(1);
    const int output_batch_idx        = get_global_id(2);

#ifdef POOL_AVG
    FLOAT4 output_result = 0;
#else
    FLOAT4 output_result = (FLOAT4)(-FLT_MAX);
#if RETURN_REDICE
    int4 redice = (int4)0;
    int4 local rediceId[LOCAL_SIZE];
#endif
#endif

    FLOAT4 local sum[LOCAL_SIZE];
    int wc = output_channel_idx * input_shape.y;
    int bh = output_batch_idx * input_shape.x;
    for(int i = local_id; i < input_shape.x * input_shape.y; i+=LOCAL_SIZE){
        int w = i % input_shape.y;;
        int h = i / input_shape.y;
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(wc+w, bh+h));
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

    WI_F(output, (int2)(output_channel_idx, output_batch_idx), output_result);
    #if RETURN_REDICE
    redice = rediceId[0];
    WI_F(rediceOutput, (int2)(output_channel_idx, output_batch_idx), CONVERT_FLOAT4(redice));
    #endif
}
#endif
