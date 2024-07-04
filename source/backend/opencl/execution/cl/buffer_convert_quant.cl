#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#ifdef USE_LOW_BIT_WEIGHT_INT8
// convert kernel : from int8 buffer(oihw) to int8 image(oc/4 h w , ic oc4)
__kernel void conv2d_filter_buffer_to_nc4hw4_buffer_int8(GLOBAL_SIZE_2_DIMS
                                            __global const char *input_ptr,
                                            __private const int output_channel,
                                            __private const int2 kernel_shape,
                                            __private const int ic_h_w_size,
                                            __private const int height_width_size,
                                            __global char *output) {
    int image_width_idx  = get_global_id(0); // ic
    int image_height_idx = get_global_id(1); // oc/4 h w
 
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);
 
    const int input_channel_4_idx  = image_width_idx;
    const int output_channel_4_idx = (image_height_idx / height_width_size) * 4;
    const int height_width_idx     = image_height_idx % height_width_size;
    const int buffer_height_idx    = height_width_idx / kernel_shape.y;
    const int buffer_width_idx     = height_width_idx % kernel_shape.y;
 
    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_shape.y + buffer_width_idx;
 
    char4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = (char)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (char)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (char)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.w = (char)(*(input_ptr + offset));
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = (char)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (char)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (char)(*(input_ptr + offset));
 
        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = (char)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (char)(*(input_ptr + offset));
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = (char)(*(input_ptr + offset));
        }
    }
    const int out_offset = (image_width_idx*height_width_size*((output_channel+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}
#endif

#ifdef USE_LOW_BIT_WEIGHT_INT4
// convert kernel : from int8 buffer(oihw) to int4 image(oc/4 h w , ic oc4)
__kernel void conv2d_filter_buffer_to_nc4hw4_buffer_int4(GLOBAL_SIZE_2_DIMS
                                            __global const char *input_ptr,
                                            __private const int output_channel,
                                            __private const int2 kernel_shape,
                                            __private const int ic_h_w_size,
                                            __private const int height_width_size,
                                            __global uchar *output) {
    int image_width_idx  = get_global_id(0); // ic
    int image_height_idx = get_global_id(1); // oc/4 h w
 
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);
 
    const int input_channel_4_idx  = image_width_idx;
    const int output_channel_4_idx = (image_height_idx / height_width_size) * 4;
    const int height_width_idx     = image_height_idx % height_width_size;
    const int buffer_height_idx    = height_width_idx / kernel_shape.y;
    const int buffer_width_idx     = height_width_idx % kernel_shape.y;
 
    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_shape.y + buffer_width_idx;
 
    char4 output_values_int8 = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values_int8.x = (char)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values_int8.y = (char)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values_int8.z = (char)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values_int8.w = (char)(*(input_ptr + offset));
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values_int8.x = (char)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values_int8.y = (char)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values_int8.z = (char)(*(input_ptr + offset));
 
        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values_int8.x = (char)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values_int8.y = (char)(*(input_ptr + offset));
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values_int8.x = (char)(*(input_ptr + offset));
        }
    }
 
    uchar2 output_values_int4 = (uchar2)(0, 0);
    output_values_int4.s0 = (output_values_int8.x + 8) * 16 + (output_values_int8.y + 8);
    output_values_int4.s1 = (output_values_int8.z + 8) * 16 + (output_values_int8.w + 8);
 
    const int out_offset = (image_width_idx*height_width_size*((output_channel+3)/4)+image_height_idx)*2;
    vstore2(output_values_int4, 0, output+out_offset);
}
#endif

#define CHAR16_TO_UCHAR8(a, b) \
    a = (uchar8)(((b.s0 + 8) << 4) + b.s1 + 8, ((b.s2 + 8) << 4) + b.s3 + 8, ((b.s4 + 8) << 4) + b.s5 + 8, ((b.s6 + 8) << 4) + b.s7 + 8, ((b.s8 + 8) << 4) + b.s9 + 8, ((b.sa + 8) << 4) + b.sb + 8, ((b.sc + 8) << 4) + b.sd + 8, ((b.se + 8) << 4) + b.sf + 8);

#define CHAR32_TO_UCHAR16(a, b, c) \
    a = (uchar16)(((b.s0 + 8) << 4) + b.s1 + 8, ((b.s2 + 8) << 4) + b.s3 + 8, ((b.s4 + 8) << 4) + b.s5 + 8, ((b.s6 + 8) << 4) + b.s7 + 8, ((b.s8 + 8) << 4) + b.s9 + 8, ((b.sa + 8) << 4) + b.sb + 8, ((b.sc + 8) << 4) + b.sd + 8, ((b.se + 8) << 4) + b.sf + 8,  \
                  ((c.s0 + 8) << 4) + c.s1 + 8, ((c.s2 + 8) << 4) + c.s3 + 8, ((c.s4 + 8) << 4) + c.s5 + 8, ((c.s6 + 8) << 4) + c.s7 + 8, ((c.s8 + 8) << 4) + c.s9 + 8, ((c.sa + 8) << 4) + c.sb + 8, ((c.sc + 8) << 4) + c.sd + 8, ((c.se + 8) << 4) + c.sf + 8);
__kernel void conv2d_1x1_weight_quant_buffer(GLOBAL_SIZE_2_DIMS
                                            __global const char *input_ptr,
#ifdef USE_LOW_BIT_WEIGHT_INT4
                                            __global uchar *output_ptr,
#else
                                            __global char *output_ptr,
#endif
                                            __private const int input_channel,
                                            __private const int output_channel) {
    int x  = get_global_id(0); // ic / 16
    int y = get_global_id(1); // oc
 
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int xin = x << 4;
    const int outputChannelC4 = (output_channel + 3) >> 2;
    const int inputOffset = y * input_channel + xin;
    char16 weight = 0;
#ifdef INPUT_CHANNEL_LEAVE
    if(xin + 15 >= input_channel){
        char *weight_ptr = (char*)&weight;
        for(int i = 0, j = 0; xin + i < input_channel && j < 16; ++i, ++j){
            weight_ptr[j] = input_ptr[inputOffset + i];
        }
    }else {
        weight = vload16(0, input_ptr + inputOffset);
    }
#else
    weight = vload16(0, input_ptr + inputOffset);
#endif
    
#ifdef USE_LOW_BIT_WEIGHT_INT4
    const int outputOffset = ((x * outputChannelC4 * 4 * 8 + y * 8));
    uchar8 outWeight;
    CHAR16_TO_UCHAR8(outWeight, weight);
    vstore8(outWeight, 0, output_ptr + outputOffset);
#else
    const int outputOffset = (x * outputChannelC4 * 4 + y) << 4;
    vstore16(weight, 0, output_ptr + outputOffset);
#endif
}

__kernel void conv2d_1x1_weight_quant_image(GLOBAL_SIZE_2_DIMS
                                            __global const char *input_ptr,
                                            __write_only image2d_t output,
                                            __private const int input_channel,
                                            __private const int output_channel) {
    
#ifdef USE_LOW_BIT_WEIGHT_INT4
    int x  = get_global_id(0); // ic / 32
    int y = get_global_id(1); // oc
 
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int outputChannelC4 = (output_channel + 3) >> 2;
    const int xin = x << 5;
    const int inputOffset = y * input_channel + xin;
    char16 weight00 = 0, weight01 = 0;
#ifdef INPUT_CHANNEL_LEAVE
    if(xin + 31 >= input_channel){
        char *weight00_ptr = (char*)&weight00;
        char *weight01_ptr = (char*)&weight01;
        int i = 0;
        for(int j = 0; xin + i < input_channel && j < 16; ++i, ++j){
            weight00_ptr[j] = input_ptr[inputOffset + i];
        }
        for(int j = 0; xin + i < input_channel && j < 16; ++i, ++j){
            weight01_ptr[j] = input_ptr[inputOffset + i];
        }
    }else {
        weight00 = vload16(0, input_ptr + inputOffset);
        weight01 = vload16(0, input_ptr + inputOffset + 16);
    }
#else
    weight00 = vload16(0, input_ptr + inputOffset);
    weight01 = vload16(0, input_ptr + inputOffset + 16);
#endif
    
    uchar16 outWeight;
    CHAR32_TO_UCHAR16(outWeight, weight00, weight01);
    write_imagei(output, (int2)(y, x), as_int4(outWeight));
#else
    int x  = get_global_id(0); // ic / 16
    int y = get_global_id(1); // oc
 
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int xin = x << 4;
    const int inputOffset = y * input_channel + xin;
    const int outputChannelC4 = (output_channel + 3) >> 2;
    char16 weight = 0;
#ifdef INPUT_CHANNEL_LEAVE
    if(xin + 15 >= input_channel){
        char *weight_ptr = (char*)&weight;
        for(int i = 0, j = 0; xin + i < input_channel && j < 16; ++i, ++j){
            weight_ptr[j] = input_ptr[inputOffset + i];
        }
    }else {
        weight = vload16(0, input_ptr + inputOffset);
    }
#else
    weight = vload16(0, input_ptr + inputOffset);
#endif
    
    write_imagei(output, (int2)(y, x), as_int4(weight));
#endif
}
