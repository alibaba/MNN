#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifdef MNN_SUPPORT_FP16
#define GROUP_READ(ptr, offset)            as_half(intel_sub_group_block_read_us((const __global ushort*)(ptr) + (offset)))
#define GROUP_READ2(ptr, offset)           as_half2(intel_sub_group_block_read_us2((const __global ushort*)(ptr) + (offset)))
#define GROUP_READ4(ptr, offset)           as_half4(intel_sub_group_block_read_us4((const __global ushort*)(ptr) + (offset)))
#define GROUP_READ8(ptr, offset)           as_half8(intel_sub_group_block_read_us8((const __global ushort*)(ptr) + (offset)))

#define GROUP_WRITE(ptr, offset, val)      intel_sub_group_block_write_us((const __global ushort*)(ptr) + (offset), as_ushort(val))
#define GROUP_WRITE2(ptr, offset, val)     intel_sub_group_block_write_us2((const __global ushort*)(ptr) + (offset), as_ushort2(val))
#define GROUP_WRITE4(ptr, offset, val)     intel_sub_group_block_write_us4((const __global ushort*)(ptr) + (offset), as_ushort4(val))
#define GROUP_WRITE8(ptr, offset, val)     intel_sub_group_block_write_us8((const __global ushort*)(ptr) + (offset), as_ushort8(val))

#define GROUP_SHUFFLE(data, id)            as_half(intel_sub_group_shuffle(as_ushort(data), id))
#define GROUP_SHUFFLE2(data, id)           as_half2(intel_sub_group_shuffle(as_ushort2(data), id))
#define GROUP_SHUFFLE4(data, id)           as_half4(intel_sub_group_shuffle(as_ushort4(data), id))
#define GROUP_SHUFFLE8(data, id)           as_half8(intel_sub_group_shuffle(as_ushort8(data), id))
#else
#define GROUP_READ(ptr, offset)            as_float(intel_sub_group_block_read((const __global uint*)(ptr) + (offset)))
#define GROUP_READ2(ptr, offset)           as_float2(intel_sub_group_block_read2((const __global uint*)(ptr) + (offset)))
#define GROUP_READ4(ptr, offset)           as_float4(intel_sub_group_block_read4((const __global uint*)(ptr) + (offset)))
#define GROUP_READ8(ptr, offset)           as_float8(intel_sub_group_block_read8((const __global uint*)(ptr) + (offset)))

#define GROUP_WRITE(ptr, offset, val)      intel_sub_group_block_write((const __global uint*)(ptr) + (offset), as_uint(val))
#define GROUP_WRITE2(ptr, offset, val)     intel_sub_group_block_write2((const __global uint*)(ptr) + (offset), as_uint2(val))
#define GROUP_WRITE4(ptr, offset, val)     intel_sub_group_block_write4((const __global uint*)(ptr) + (offset), as_uint4(val))
#define GROUP_WRITE8(ptr, offset, val)     intel_sub_group_block_write8((const __global uint*)(ptr) + (offset), as_uint8(val))

#define GROUP_SHUFFLE(data, id)            intel_sub_group_shuffle(data, id)
#define GROUP_SHUFFLE2(data, id)           intel_sub_group_shuffle(data, id)
#define GROUP_SHUFFLE4(data, id)           intel_sub_group_shuffle(data, id)
#define GROUP_SHUFFLE8(data, id)           intel_sub_group_shuffle(data, id)
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c1_c4_b2(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left,
    __private const int input_pad_right,
    __private const int output_pad_left,
    __private const int output_pad_right
)
{
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 1;
    const int y = (xy / x_blocks);

    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;


    const uint input_x_pitch = 1;
    const uint input_y_pitch = input_x_pitch * input_width;
    const uint input_f_pitch = input_y_pitch * input_height;
    const uint input_b_pitch = input_f_pitch * INPUT_CHANNEL;

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              input_x * input_x_pitch;


    const uint output_pack = (output_channel + 3) / 4;
    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch * output_width;
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch * output_pack;
    
    
    const uint output_offset = b * output_b_pitch +
                               f_block * 4 * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 256;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = f_block * filter_os_pitch;

    uint bias_offset = f_block * 16;
    COMPUTE_FLOAT2 dst = (COMPUTE_FLOAT2)(GROUP_READ(biases, bias_offset));
    
    FLOAT line_cache[INPUT_CHANNEL * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < INPUT_CHANNEL; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (int i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const int in_elem = i * 16 + lid;
            const int xb = in_elem % INPUT_LINE_SIZE;
            const int yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < input_height &&
                input_x + xb >= 0 && input_x + xb < input_width)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = 0;
        }
    }

    __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
    for (int kh = 0; kh < FILTER_HEIGHT; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
        for (int kw = 0; kw < FILTER_WIDTH; kw++)
        {
            uint offset = filter_offset + kh * filter_y_pitch + kw * filter_x_pitch;
    
            COMPUTE_FLOAT wei[INPUT_CHANNEL];
            __attribute__((opencl_unroll_hint(INPUT_CHANNEL)))
            for (int ic = 0; ic < INPUT_CHANNEL; ic++)
                wei[ic] = GROUP_READ(weights, offset + ic * filter_isv_pitch);
    
            __attribute__((opencl_unroll_hint(2)))
            for (int i = 0; i < 2; i++)
            {
                const uint buf_offset = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) / 16;
                const uint buf_group  = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) % 16;
    
                for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                    COMPUTE_FLOAT src = GROUP_SHUFFLE(line_cache[ic * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                    dst[i] = mad(wei[ic], src, dst[i]);
                }
            }
        }
    }

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif

    const uint lid_x = lid % 4;
    const uint lid_y = lid / 4;

    if ((f_block+1)*16 >= output_channel) {
        for (int i = 0; i < 2 && (x + i) < output_width; i++) {
            if ((f_block*16 + lid_y * 4 < output_pack * 4))
                output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
    else
    {
        for (int i = 0; i < 2 && (x + i) < output_width; i++) {
            output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c1_c4_b4(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left,
    __private const int input_pad_right,
    __private const int output_pad_left,
    __private const int output_pad_right
)
{
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 2;
    const int y = (xy / x_blocks);

    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;


    const uint input_x_pitch = 1;
    const uint input_y_pitch = input_x_pitch * input_width;
    const uint input_f_pitch = input_y_pitch * input_height;
    const uint input_b_pitch = input_f_pitch * INPUT_CHANNEL;

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              input_x * input_x_pitch;


    const uint output_pack = (output_channel + 3) / 4;
    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch * output_width;
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch * output_pack;
    
    
    const uint output_offset = b * output_b_pitch +
                               f_block * 4 * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 256;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = f_block * filter_os_pitch;

    uint bias_offset = f_block * 16;
    COMPUTE_FLOAT4 dst = (COMPUTE_FLOAT4)(GROUP_READ(biases, bias_offset));
    
    FLOAT line_cache[INPUT_CHANNEL * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < INPUT_CHANNEL; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (int i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const int in_elem = i * 16 + lid;
            const int xb = in_elem % INPUT_LINE_SIZE;
            const int yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < input_height &&
                input_x + xb >= 0 && input_x + xb < input_width)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = 0;
        }
    }

    __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
    for (int kh = 0; kh < FILTER_HEIGHT; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
        for (int kw = 0; kw < FILTER_WIDTH; kw++)
        {
            uint offset = filter_offset + kh * filter_y_pitch + kw * filter_x_pitch;
    
            COMPUTE_FLOAT wei[INPUT_CHANNEL];
            __attribute__((opencl_unroll_hint(INPUT_CHANNEL)))
            for (int ic = 0; ic < INPUT_CHANNEL; ic++)
                wei[ic] = GROUP_READ(weights, offset + ic * filter_isv_pitch);
    
            __attribute__((opencl_unroll_hint(4)))
            for (int i = 0; i < 4; i++)
            {
                const uint buf_offset = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) / 16;
                const uint buf_group  = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) % 16;
    
                for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                    COMPUTE_FLOAT src = GROUP_SHUFFLE(line_cache[ic * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                    dst[i] = mad(wei[ic], src, dst[i]);
                }
            }
        }
    }

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    const uint lid_x = lid % 4;
    const uint lid_y = lid / 4;

    if ((f_block+1)*16 >= output_channel) {
        for (int i = 0; i < 4 && (x + i) < output_width; i++) {
            if ((f_block*16 + lid_y * 4 < output_pack * 4))
                output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
    else
    {
        for (int i = 0; i < 4 && (x + i) < output_width; i++) {
            output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c1_c4_b8(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left,
    __private const int input_pad_right,
    __private const int output_pad_left,
    __private const int output_pad_right
)
{
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 3;
    const int y = (xy / x_blocks);

    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;


    const uint input_x_pitch = 1;
    const uint input_y_pitch = input_x_pitch * input_width;
    const uint input_f_pitch = input_y_pitch * input_height;
    const uint input_b_pitch = input_f_pitch * INPUT_CHANNEL;

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              input_x * input_x_pitch;


    const uint output_pack = (output_channel + 3) / 4;
    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch * output_width;
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch * output_pack;
    
    
    const uint output_offset = b * output_b_pitch +
                               f_block * 4 * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 256;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = f_block * filter_os_pitch;

    uint bias_offset = f_block * 16;
    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(GROUP_READ(biases, bias_offset));
    
    FLOAT line_cache[INPUT_CHANNEL * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < INPUT_CHANNEL; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (int i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const int in_elem = i * 16 + lid;
            const int xb = in_elem % INPUT_LINE_SIZE;
            const int yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < input_height &&
                input_x + xb >= 0 && input_x + xb < input_width)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = 0;
        }
    }

    __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
    for (int kh = 0; kh < FILTER_HEIGHT; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
        for (int kw = 0; kw < FILTER_WIDTH; kw++)
        {
            uint offset = filter_offset + kh * filter_y_pitch + kw * filter_x_pitch;
    
            COMPUTE_FLOAT wei[INPUT_CHANNEL];
            __attribute__((opencl_unroll_hint(INPUT_CHANNEL)))
            for (int ic = 0; ic < INPUT_CHANNEL; ic++)
                wei[ic] = GROUP_READ(weights, offset + ic * filter_isv_pitch);
    
            __attribute__((opencl_unroll_hint(8)))
            for (int i = 0; i < 8; i++)
            {
                const uint buf_offset = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) / 16;
                const uint buf_group  = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) % 16;
    
                for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                    COMPUTE_FLOAT src = GROUP_SHUFFLE(line_cache[ic * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                    dst[i] = mad(wei[ic], src, dst[i]);
                }
            }
        }
    }

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif

    const uint lid_x = lid % 4;
    const uint lid_y = lid / 4;

    if ((f_block+1)*16 >= output_channel) {
        for (int i = 0; i < 8 && (x + i) < output_width; i++) {
            if ((f_block*16 + lid_y * 4 < output_pack * 4))
                output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
    else
    {
        for (int i = 0; i < 8 && (x + i) < output_width; i++) {
            output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = (FLOAT)dst[i];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c1_c16_b2(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left,
    __private const int input_pad_right,
    __private const int output_pad_left,
    __private const int output_pad_right
)
{
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 1;
    const int y = (xy / x_blocks);

    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;


    const uint input_x_pitch = 1;
    const uint input_y_pitch = input_x_pitch * input_width;
    const uint input_f_pitch = input_y_pitch * input_height;
    const uint input_b_pitch = input_f_pitch * INPUT_CHANNEL;

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              input_x * input_x_pitch;


    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch * (output_pad_left + output_width + output_pad_right);
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch * ((output_channel + 15) / 16);
    
    
    const uint output_offset = b * output_b_pitch +
                               f_block * output_fs_pitch +
                               y * output_y_pitch +
                               (x + output_pad_left) * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 256;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = f_block * filter_os_pitch;

    uint bias_offset = f_block * 16;
    COMPUTE_FLOAT2 dst = (COMPUTE_FLOAT2)(GROUP_READ(biases, bias_offset));
    
    FLOAT line_cache[INPUT_CHANNEL * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < INPUT_CHANNEL; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (int i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const int in_elem = i * 16 + lid;
            const int xb = in_elem % INPUT_LINE_SIZE;
            const int yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < input_height &&
                input_x + xb >= 0 && input_x + xb < input_width)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = 0;
        }
    }

    __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
    for (int kh = 0; kh < FILTER_HEIGHT; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
        for (int kw = 0; kw < FILTER_WIDTH; kw++)
        {
            uint offset = filter_offset + kh * filter_y_pitch + kw * filter_x_pitch;
    
            COMPUTE_FLOAT wei[INPUT_CHANNEL];
            __attribute__((opencl_unroll_hint(INPUT_CHANNEL)))
            for (int ic = 0; ic < INPUT_CHANNEL; ic++)
                wei[ic] = GROUP_READ(weights, offset + ic * filter_isv_pitch);
    
            __attribute__((opencl_unroll_hint(2)))
            for (int i = 0; i < 2; i++)
            {
                const uint buf_offset = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) / 16;
                const uint buf_group  = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) % 16;
    
                for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                    COMPUTE_FLOAT src = GROUP_SHUFFLE(line_cache[ic * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                    dst[i] = mad(wei[ic], src, dst[i]);
                }
            }
        }
    }

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT2)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT2)0, (COMPUTE_FLOAT2)6);
#endif

    if(x == 0){
        uint pad_offset = b * output_b_pitch + f_block * output_fs_pitch + y * output_y_pitch;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * output_x_pitch + lid] = 0;
        }
        pad_offset += (output_width + output_pad_left) * output_x_pitch;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * output_x_pitch + lid] = 0;
        }
    }


    if ((f_block+1)*16 >= output_channel) {
        for (int i = 0; i < 2; i++) {
            if ((f_block*16 + lid < output_channel) && (x + i) < output_width)
                output[output_offset + i * output_x_pitch + lid] = (FLOAT)dst[i];
        }
    }
    else
    {
        if (x + 2 <= output_width || output_width % 2 == 0) {
            GROUP_WRITE2(output, output_offset, CONVERT_FLOAT2(dst));
        }else{
            for (int i = 0; i < output_width % 2; i++) {
                output[output_offset + i * output_x_pitch + lid] = (FLOAT)dst[i];
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c1_c16_b4(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left,
    __private const int input_pad_right,
    __private const int output_pad_left,
    __private const int output_pad_right
)
{
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 2;
    const int y = (xy / x_blocks);

    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;


    const uint input_x_pitch = 1;
    const uint input_y_pitch = input_x_pitch * input_width;
    const uint input_f_pitch = input_y_pitch * input_height;
    const uint input_b_pitch = input_f_pitch * INPUT_CHANNEL;

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              input_x * input_x_pitch;


    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch * (output_pad_left + output_width + output_pad_right);
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch * ((output_channel + 15) / 16);
    
    
    const uint output_offset = b * output_b_pitch +
                               f_block * output_fs_pitch +
                               y * output_y_pitch +
                               (x + output_pad_left) * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 256;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = f_block * filter_os_pitch;

    uint bias_offset = f_block * 16;
    COMPUTE_FLOAT4 dst = (COMPUTE_FLOAT4)(GROUP_READ(biases, bias_offset));
    
    FLOAT line_cache[INPUT_CHANNEL * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < INPUT_CHANNEL; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (int i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const int in_elem = i * 16 + lid;
            const int xb = in_elem % INPUT_LINE_SIZE;
            const int yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < input_height &&
                input_x + xb >= 0 && input_x + xb < input_width)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = 0;
        }
    }

    __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
    for (int kh = 0; kh < FILTER_HEIGHT; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
        for (int kw = 0; kw < FILTER_WIDTH; kw++)
        {
            uint offset = filter_offset + kh * filter_y_pitch + kw * filter_x_pitch;
    
            COMPUTE_FLOAT wei[INPUT_CHANNEL];
            __attribute__((opencl_unroll_hint(INPUT_CHANNEL)))
            for (int ic = 0; ic < INPUT_CHANNEL; ic++)
                wei[ic] = GROUP_READ(weights, offset + ic * filter_isv_pitch);
    
            __attribute__((opencl_unroll_hint(4)))
            for (int i = 0; i < 4; i++)
            {
                const uint buf_offset = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) / 16;
                const uint buf_group  = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) % 16;
    
                for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                    COMPUTE_FLOAT src = GROUP_SHUFFLE(line_cache[ic * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                    dst[i] = mad(wei[ic], src, dst[i]);
                }
            }
        }
    }

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT4)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT4)0, (COMPUTE_FLOAT4)6);
#endif

    if(x == 0){
        uint pad_offset = b * output_b_pitch + f_block * output_fs_pitch + y * output_y_pitch;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * output_x_pitch + lid] = 0;
        }
        pad_offset += (output_width + output_pad_left) * output_x_pitch;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * output_x_pitch + lid] = 0;
        }
    }


    if ((f_block+1)*16 >= output_channel) {
        for (int i = 0; i < 4; i++) {
            if ((f_block*16 + lid < output_channel) && (x + i) < output_width)
                output[output_offset + i * output_x_pitch + lid] = (FLOAT)dst[i];
        }
    }
    else
    {
        if (x + 4 <= output_width || output_width % 4 == 0) {
            GROUP_WRITE4(output, output_offset, CONVERT_FLOAT4(dst));
        }else{
            for (int i = 0; i < output_width % 4; i++) {
                output[output_offset + i * output_x_pitch + lid] = (FLOAT)dst[i];
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_2d_buf_subgroup_c1_c16_b8(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
    __private const int pad_width,
    __private const int pad_height,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int output_channel,
    __private const int x_blocks,
    __private const int input_pad_left,
    __private const int input_pad_right,
    __private const int output_pad_left,
    __private const int output_pad_right
)
{
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) << 3;
    const int y = (xy / x_blocks);

    const int input_x = x * STRIDE_WIDTH - pad_width;
    const int input_y = y * STRIDE_HEIGHT - pad_height;


    const uint input_x_pitch = 1;
    const uint input_y_pitch = input_x_pitch * input_width;
    const uint input_f_pitch = input_y_pitch * input_height;
    const uint input_b_pitch = input_f_pitch * INPUT_CHANNEL;

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              input_x * input_x_pitch;


    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch * (output_pad_left + output_width + output_pad_right);
    const uint output_fs_pitch = output_y_pitch * output_height;
    const uint output_b_pitch = output_fs_pitch * ((output_channel + 15) / 16);
    
    
    const uint output_offset = b * output_b_pitch +
                               f_block * output_fs_pitch +
                               y * output_y_pitch +
                               (x + output_pad_left) * output_x_pitch;

    const uint filter_isv_pitch = 16;
    const uint filter_x_pitch = 256;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + 15) / 16);

    const uint filter_offset = f_block * filter_os_pitch;

    uint bias_offset = f_block * 16;
    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(GROUP_READ(biases, bias_offset));
    
    FLOAT line_cache[INPUT_CHANNEL * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < INPUT_CHANNEL; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (int i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const int in_elem = i * 16 + lid;
            const int xb = in_elem % INPUT_LINE_SIZE;
            const int yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < input_height &&
                input_x + xb >= 0 && input_x + xb < input_width)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = 0;
        }
    }

    __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
    for (int kh = 0; kh < FILTER_HEIGHT; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
        for (int kw = 0; kw < FILTER_WIDTH; kw++)
        {
            uint offset = filter_offset + kh * filter_y_pitch + kw * filter_x_pitch;
    
            COMPUTE_FLOAT wei[INPUT_CHANNEL];
            __attribute__((opencl_unroll_hint(INPUT_CHANNEL)))
            for (int ic = 0; ic < INPUT_CHANNEL; ic++)
                wei[ic] = GROUP_READ(weights, offset + ic * filter_isv_pitch);
    
            __attribute__((opencl_unroll_hint(8)))
            for (int i = 0; i < 8; i++)
            {
                const uint buf_offset = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) / 16;
                const uint buf_group  = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh*DILATION_HEIGHT) * INPUT_LINE_SIZE) % 16;
    
                for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                    COMPUTE_FLOAT src = GROUP_SHUFFLE(line_cache[ic * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                    dst[i] = mad(wei[ic], src, dst[i]);
                }
            }
        }
    }

#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif

    if(x == 0){
        uint pad_offset = b * output_b_pitch + f_block * output_fs_pitch + y * output_y_pitch;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * output_x_pitch + lid] = 0;
        }
        pad_offset += (output_width + output_pad_left) * output_x_pitch;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * output_x_pitch + lid] = 0;
        }
    }


    if ((f_block+1)*16 >= output_channel) {
        for (int i = 0; i < 8; i++) {
            if ((f_block*16 + lid < output_channel) && (x + i) < output_width)
                output[output_offset + i * output_x_pitch + lid] = (FLOAT)dst[i];
        }
    }
    else
    {
        if (x + 8 <= output_width || output_width % 8 == 0) {
            GROUP_WRITE8(output, output_offset, CONVERT_FLOAT8(dst));
        }else{
            for (int i = 0; i < output_width % 8; i++) {
                output[output_offset + i * output_x_pitch + lid] = (FLOAT)dst[i];
            }
        }
    }
}