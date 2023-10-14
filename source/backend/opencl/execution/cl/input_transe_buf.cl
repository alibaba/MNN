#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_transe_c4_c1(
    int global_size_dim0,
    int global_size_dim1,
    int global_size_dim2,
    __global FLOAT* input,
    __global FLOAT* output,
    __private const int input_width,
    __private const int input_height,
    __private const int input_channel,
    __private const int channel_blocks,
    __private const int input_pad_left,
    __private const int input_pad_right)
{
    int x = get_global_id(0);
    int w = x % input_width;
    int h = x / input_width;
    int c = get_global_id(1);
    int b = get_global_id(2);
    int cout = c << 2;
    if(x >= global_size_dim0 || c >= global_size_dim1 || b >= global_size_dim2)
        return;

    // Input offset calculations:
    const uint input_x_pitch = 4;
    const uint input_y_pitch = input_x_pitch * input_width;
    const uint input_f_pitch = input_y_pitch * input_height;
    const uint input_b_pitch = input_f_pitch * channel_blocks;

    const uint input_offset = b * input_b_pitch +
                              c * input_f_pitch +
                              h * input_y_pitch +
                              w * input_x_pitch;

    // Output offset calculations:
    const uint output_x_pitch = 1;
    const uint output_y_pitch = output_x_pitch * input_width;
    const uint output_f_pitch = output_y_pitch * input_height;
    const uint output_b_pitch = output_f_pitch * input_channel;

    const uint output_offset = b * output_b_pitch +
                               cout * output_f_pitch + 
                               h * output_y_pitch +
                               w * output_x_pitch;
    
    FLOAT4 value = vload4(0, input + input_offset);
    for(int i = 0; i < 4 && cout + i < input_channel; ++i){
        output[output_offset + i * output_f_pitch] = value[i];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void conv_transe_c4_c16(
    int global_size_dim0,
    int global_size_dim1,
    int global_size_dim2,
    __global FLOAT* input,
    __global FLOAT* output,
    int input_width,
    int input_height,
    int input_channel,
    int channel_blocks,
    int input_pad_left,
    int input_pad_right)
{
    int x = get_global_id(0);
    int w = x % input_width;
    int h = x / input_width;
    int c = get_global_id(1);
    int b = get_global_id(2);
    int cout = c >> 2;
    if(x >= global_size_dim0 || c >= global_size_dim1 || b >= global_size_dim2)
        return;
    
    // Input offset calculations:
    const uint input_x_pitch = 4;
    const uint input_y_pitch = input_x_pitch * input_width;
    const uint input_f_pitch = input_y_pitch * input_height;
    const uint input_b_pitch = input_f_pitch * channel_blocks;
    
    const uint input_offset = b * input_b_pitch +
                              c * input_f_pitch +
                              h * input_y_pitch +
                              w * input_x_pitch;
    
    // Output offset calculations:
    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch * (input_pad_left + input_width + input_pad_right);
    const uint output_f_pitch = output_y_pitch * input_height;
    const uint output_b_pitch = output_f_pitch * ((input_channel + 15) / 16);
    
    const uint output_offset = b * output_b_pitch +
                               cout * output_f_pitch + 
                               h * output_y_pitch +
                               (w + input_pad_left) * output_x_pitch + (c % 4) * 4;
    
    FLOAT4 value = vload4(0, input + input_offset);
    vstore4(value, 0, output + output_offset);
    if(w == 0){
        uint pad_offset =  b * output_b_pitch + cout * output_f_pitch + h * output_y_pitch + (c % 4) * 4;
        for(int i = 0; i < input_pad_left; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * output_x_pitch);
        }
        pad_offset += (input_pad_left + input_width) * output_x_pitch;
        for(int i = 0; i < input_pad_right; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * output_x_pitch);
        }
    }
}