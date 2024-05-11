#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void depthwise_conv_2d_buf_c16_c16(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
   __private const int inputHeight,
   __private const int inputWidth,
   __private const int Channel,
   __private const int input_pad_left,
   __private const int input_pad_right,
   __private const int outputHeight,
   __private const int outputWidth,
   __private const int output_pad_left,
   __private const int output_pad_right,
   __private const int pad_w,
   __private const int pad_h
) {
    const int x_blocks = (outputWidth + 7) / 8;
    const int sglid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) * 8;
    const int y = (xy / x_blocks);

    const int c = get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_w;
    const int input_y = y * STRIDE_HEIGHT - pad_h;
    const int channel_pack = ((Channel + 15) / 16);

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (inputWidth + input_pad_left + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (inputHeight);
    const uint input_b_pitch = input_fs_pitch * channel_pack;

    const uint input_offset = b * input_b_pitch +
                              c * input_fs_pitch + 
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch *  (outputWidth + output_pad_left + output_pad_right);
    const uint output_fs_pitch = output_y_pitch * outputHeight;
    const uint output_b_pitch = output_fs_pitch * channel_pack;

    const uint output_offset = b * output_b_pitch +
                               c * output_fs_pitch +
                               y * output_y_pitch +
                               (x + output_pad_left) * output_x_pitch;

    const uint filter_x_pitch = 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;

    const uint filter_offset = c * filter_is_pitch;

#ifdef MNN_SUPPORT_FP16
    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(as_half(intel_sub_group_block_read_us((__global ushort*)(biases + c * 16))));

    for(int i = 0; i < FILTER_HEIGHT; ++i){
        if ((input_y + i * DILATION_HEIGHT) < 0 || (input_y + i * DILATION_HEIGHT) >= inputHeight)
            continue;
        for(int j = 0; j < FILTER_WIDTH; ++j){
            COMPUTE_FLOAT wei = as_half(intel_sub_group_block_read_us((__global ushort*)(weights + filter_offset + i * filter_y_pitch + j * filter_x_pitch)));
            for(int k = 0; k < 8; ++k){
                COMPUTE_FLOAT src = as_half(intel_sub_group_block_read_us((__global ushort*)(input + input_offset + i * DILATION_HEIGHT * input_y_pitch + (j * DILATION_WIDTH + k * STRIDE_WIDTH) * input_x_pitch)));
                dst[k] = mad(src, wei, dst[k]);
            }
        }
    }
    
#else
    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(as_float(intel_sub_group_block_read((__global uint*)(biases + c * 16))));

    for(int i = 0; i < FILTER_HEIGHT; ++i){
        if ((input_y + i * DILATION_HEIGHT) < 0 || (input_y + i * DILATION_HEIGHT) >= inputHeight)
            continue;
        for(int j = 0; j < FILTER_WIDTH; ++j){
            COMPUTE_FLOAT wei = as_float(intel_sub_group_block_read((__global ushort*)(weights + filter_offset + i * filter_y_pitch + j * filter_x_pitch)));
            for(int k = 0; k < 8; ++k){
                COMPUTE_FLOAT src = as_float(intel_sub_group_block_read((__global ushort*)(input + input_offset + i * DILATION_HEIGHT * input_y_pitch + (j * DILATION_WIDTH + k * STRIDE_WIDTH) * input_x_pitch)));
                dst[k] = mad(src, wei, dst[k]);
            }
        }
    }
#endif


#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif
    
    for (int i = 0; i < 8 && (x + i) < outputWidth; i++) {
#ifdef MNN_SUPPORT_FP16
        intel_sub_group_block_write_us((__global ushort*)(output + output_offset + i * output_x_pitch), as_ushort((FLOAT)dst[i]));
#else
        intel_sub_group_block_write((__global uint*)(output + output_offset + i * output_x_pitch), as_uint((FLOAT)dst[i]));
#endif
    }
    if(x == 0){
        uint pad_offset = b * output_b_pitch + c * output_fs_pitch + y * output_y_pitch;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
        pad_offset += (outputWidth + output_pad_left) * output_x_pitch;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void depthwise_conv_2d_buf_c16_c4(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
   __private const int inputHeight,
   __private const int inputWidth,
   __private const int Channel,
   __private const int input_pad_left,
   __private const int input_pad_right,
   __private const int outputHeight,
   __private const int outputWidth,
   __private const int output_pad_left,
   __private const int output_pad_right,
   __private const int pad_w,
   __private const int pad_h
) {
    const int x_blocks = (outputWidth + 7) / 8;
    const int sglid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) * 8;
    const int y = (xy / x_blocks);

    const int c = get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_w;
    const int input_y = y * STRIDE_HEIGHT - pad_h;
    const int channel_pack = ((Channel + 15) / 16);

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (inputWidth + input_pad_left + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (inputHeight);
    const uint input_b_pitch = input_fs_pitch * channel_pack;

    const uint input_offset = b * input_b_pitch +
                              c * input_fs_pitch + 
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch * outputWidth;
    const uint output_fs_pitch = output_y_pitch * outputHeight;
    const uint output_b_pitch = output_fs_pitch * ((Channel + 3) / 4);

    const uint output_offset = b * output_b_pitch +
                               (c << 2) * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_x_pitch = 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;

    const uint filter_offset = c * filter_is_pitch;

#ifdef MNN_SUPPORT_FP16
    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(as_half(intel_sub_group_block_read_us((__global ushort*)(biases + c * 16))));

    for(int i = 0; i < FILTER_HEIGHT; ++i){
        if ((input_y + i * DILATION_HEIGHT) < 0 || (input_y + i * DILATION_HEIGHT) >= inputHeight)
            continue;
        for(int j = 0; j < FILTER_WIDTH; ++j){
            COMPUTE_FLOAT wei = as_half(intel_sub_group_block_read_us((__global ushort*)(weights + filter_offset + i * filter_y_pitch + j * filter_x_pitch)));
            for(int k = 0; k < 8; ++k){
                COMPUTE_FLOAT src = as_half(intel_sub_group_block_read_us((__global ushort*)(input + input_offset + i * DILATION_HEIGHT * input_y_pitch + (j * DILATION_WIDTH + k * STRIDE_WIDTH) * input_x_pitch)));
                dst[k] = mad(src, wei, dst[k]);
            }
        }
    }
    
#else
    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(as_float(intel_sub_group_block_read((__global uint*)(biases + c * 16))));

    for(int i = 0; i < FILTER_HEIGHT; ++i){
        if ((input_y + i * DILATION_HEIGHT) < 0 || (input_y + i * DILATION_HEIGHT) >= inputHeight)
            continue;
        for(int j = 0; j < FILTER_WIDTH; ++j){
            COMPUTE_FLOAT wei = as_float(intel_sub_group_block_read((__global ushort*)(weights + filter_offset + i * filter_y_pitch + j * filter_x_pitch)));
            for(int k = 0; k < 8; ++k){
                COMPUTE_FLOAT src = as_float(intel_sub_group_block_read((__global ushort*)(input + input_offset + i * DILATION_HEIGHT * input_y_pitch + (j * DILATION_WIDTH + k * STRIDE_WIDTH) * input_x_pitch)));
                dst[k] = mad(src, wei, dst[k]);
            }
        }
    }
#endif


#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif

    const uint lid_x = sglid % 4;
    const uint lid_y = sglid / 4;
    for (int i = 0; i < 8 && (x + i) < outputWidth; i++) {
        output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = dst[i];
    }
}