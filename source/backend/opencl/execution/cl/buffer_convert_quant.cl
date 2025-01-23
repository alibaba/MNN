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
                                            __global const uchar *input_ptr,
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

    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size + buffer_height_idx * kernel_shape.y + buffer_width_idx;
    int index0 = buffer_offset, index1 = buffer_offset + ic_h_w_size, index2 = buffer_offset + 2 * ic_h_w_size, index3 = buffer_offset + 3 * ic_h_w_size;

    uchar2 output_values_int4 = (uchar2)(0, 0);
    uchar s0 = input_ptr[index0/2];
    uchar s1 = output_channel_4_idx + 1 >= output_channel ? 0 : input_ptr[index1/2];
    uchar s2 = output_channel_4_idx + 1 >= output_channel ? 0 : input_ptr[index2/2];
    uchar s3 = output_channel_4_idx + 1 >= output_channel ? 0 : input_ptr[index3/2];
    output_values_int4.x = ((index0 % 2) == 0 ? (s0 & 0xf0) : (s0 << 4)) | ((index1 % 2) == 0 ? (s1 >> 4) : (s1 & 0x0f));
    output_values_int4.y = ((index2 % 2) == 0 ? (s2 & 0xf0) : (s2 << 4)) | ((index3 % 2) == 0 ? (s3 >> 4) : (s3 & 0x0f));

    const int out_offset = (image_width_idx*height_width_size*((output_channel+3)/4)+image_height_idx)*2;
    vstore2(output_values_int4, 0, output+out_offset);
}
#endif

__kernel void conv2d_1x1_weight_quant_image(GLOBAL_SIZE_2_DIMS
#ifdef USE_LOW_BIT_WEIGHT_INT4
                                            __global const uchar *input_ptr,
#else
                                            __global const char *input_ptr,
#endif
                                            __write_only image2d_t output,
                                            __private const int input_channel,
                                            __private const int output_channel) {

    int x  = get_global_id(0); // ic / 4
    int y = get_global_id(1); // oc / 8

    DEAL_NON_UNIFORM_DIM2(x, y);
    const int xin = x << 2;
    const int yin = y << 3;
#ifdef USE_LOW_BIT_WEIGHT_INT4
    uchar16 out = 0;
    uchar *out_ptr = (uchar*)&out;
    for(int i = 0; i < 4; ++i){
        int index0 = yin * input_channel + xin + i;
        int index1 = (yin + 1) * input_channel + xin + i;
        int index2 = (yin + 2) * input_channel + xin + i;
        int index3 = (yin + 3) * input_channel + xin + i;
        int index4 = (yin + 4) * input_channel + xin + i;
        int index5 = (yin + 5) * input_channel + xin + i;
        int index6 = (yin + 6) * input_channel + xin + i;
        int index7 = (yin + 7) * input_channel + xin + i;
        uchar s0 = input_ptr[index0/2];
        uchar s1 = input_ptr[index1/2];
        uchar s2 = input_ptr[index2/2];
        uchar s3 = input_ptr[index3/2];
        uchar s4 = input_ptr[index4/2];
        uchar s5 = input_ptr[index5/2];
        uchar s6 = input_ptr[index6/2];
        uchar s7 = input_ptr[index7/2];
        out_ptr[i * 4] = ((index0 % 2) == 0 ? (s0 & 0xf0) : (s0 << 4)) | ((index1 % 2) == 0 ? (s1 >> 4) : (s1 & 0x0f));
        out_ptr[i * 4 + 1] = ((index2 % 2) == 0 ? (s2 & 0xf0) : (s2 << 4)) | ((index3 % 2) == 0 ? (s3 >> 4) : (s3 & 0x0f));
        out_ptr[i * 4 + 2] = ((index4 % 2) == 0 ? (s4 & 0xf0) : (s4 << 4)) | ((index5 % 2) == 0 ? (s5 >> 4) : (s5 & 0x0f));
        out_ptr[i * 4 + 3] = ((index6 % 2) == 0 ? (s6 & 0xf0) : (s6 << 4)) | ((index7 % 2) == 0 ? (s7 >> 4) : (s7 & 0x0f));
    }
    write_imagei(output, (int2)(x, y), as_int4(out));
#else
    const int inputOffset = yin * input_channel + xin;
    char4 s0 = vload4(0, input_ptr + inputOffset);
    char4 s1 = vload4(0, input_ptr + inputOffset + input_channel);
    char4 s2 = vload4(0, input_ptr + inputOffset + input_channel * 2);
    char4 s3 = vload4(0, input_ptr + inputOffset + input_channel * 3);
    char4 s4 = vload4(0, input_ptr + inputOffset + input_channel * 4);
    char4 s5 = vload4(0, input_ptr + inputOffset + input_channel * 5);
    char4 s6 = vload4(0, input_ptr + inputOffset + input_channel * 6);
    char4 s7 = vload4(0, input_ptr + inputOffset + input_channel * 7);
    char16 out0 = (char16)(s0.s0, s1.s0, s2.s0, s3.s0, s4.s0, s5.s0, s6.s0, s7.s0, s0.s1, s1.s1, s2.s1, s3.s1, s4.s1, s5.s1, s6.s1, s7.s1);
    char16 out1 = (char16)(s0.s2, s1.s2, s2.s2, s3.s2, s4.s2, s5.s2, s6.s2, s7.s2, s0.s3, s1.s3, s2.s3, s3.s3, s4.s3, s5.s3, s6.s3, s7.s3);
    write_imagei(output, (int2)(x * 2, y), as_int4(out0));
    write_imagei(output, (int2)(x * 2 + 1, y), as_int4(out1));
#endif
}

__kernel void conv2d_1x1_weight_quant_buffer(GLOBAL_SIZE_2_DIMS
#ifdef USE_LOW_BIT_WEIGHT_INT4
                                            __global const uchar *input_ptr,
#else
                                            __global const char *input_ptr,
#endif
                                            __global char *output_ptr,
                                            __private const int input_channel,
                                            __private const int output_channel) {
    int x  = get_global_id(0); // ic / 4
    int y = get_global_id(1); // oc / 8

    DEAL_NON_UNIFORM_DIM2(x, y);
    const int xin = x << 2;
    const int yin = y << 3;
    const int outputChannelC8 = (output_channel + 7) >> 3;
    const int inputChannelC4 = (input_channel + 3) >> 2;
#ifdef USE_LOW_BIT_WEIGHT_INT4
    uchar16 out = 0;
    uchar *out_ptr = (uchar*)&out;
    for(int i = 0; i < 4; ++i){
        int index0 = yin * input_channel + xin + i;
        int index1 = (yin + 1) * input_channel + xin + i;
        int index2 = (yin + 2) * input_channel + xin + i;
        int index3 = (yin + 3) * input_channel + xin + i;
        int index4 = (yin + 4) * input_channel + xin + i;
        int index5 = (yin + 5) * input_channel + xin + i;
        int index6 = (yin + 6) * input_channel + xin + i;
        int index7 = (yin + 7) * input_channel + xin + i;
        uchar s0 = input_ptr[index0/2];
        uchar s1 = input_ptr[index1/2];
        uchar s2 = input_ptr[index2/2];
        uchar s3 = input_ptr[index3/2];
        uchar s4 = input_ptr[index4/2];
        uchar s5 = input_ptr[index5/2];
        uchar s6 = input_ptr[index6/2];
        uchar s7 = input_ptr[index7/2];
        out_ptr[i * 4] = ((index0 % 2) == 0 ? (s0 & 0xf0) : (s0 << 4)) | ((index1 % 2) == 0 ? (s1 >> 4) : (s1 & 0x0f));
        out_ptr[i * 4 + 1] = ((index2 % 2) == 0 ? (s2 & 0xf0) : (s2 << 4)) | ((index3 % 2) == 0 ? (s3 >> 4) : (s3 & 0x0f));
        out_ptr[i * 4 + 2] = ((index4 % 2) == 0 ? (s4 & 0xf0) : (s4 << 4)) | ((index5 % 2) == 0 ? (s5 >> 4) : (s5 & 0x0f));
        out_ptr[i * 4 + 3] = ((index6 % 2) == 0 ? (s6 & 0xf0) : (s6 << 4)) | ((index7 % 2) == 0 ? (s7 >> 4) : (s7 & 0x0f));
    }
    const int outputOffset = (y * inputChannelC4 + x) * 16;
    vstore16(as_char16(out),0,output_ptr+outputOffset);
#else
    const int inputOffset = yin * input_channel + xin;
    char4 s0 = vload4(0, input_ptr + inputOffset);
    char4 s1 = vload4(0, input_ptr + inputOffset + input_channel);
    char4 s2 = vload4(0, input_ptr + inputOffset + input_channel * 2);
    char4 s3 = vload4(0, input_ptr + inputOffset + input_channel * 3);
    char4 s4 = vload4(0, input_ptr + inputOffset + input_channel * 4);
    char4 s5 = vload4(0, input_ptr + inputOffset + input_channel * 5);
    char4 s6 = vload4(0, input_ptr + inputOffset + input_channel * 6);
    char4 s7 = vload4(0, input_ptr + inputOffset + input_channel * 7);
    char16 out0 = (char16)(s0.s0, s1.s0, s2.s0, s3.s0, s4.s0, s5.s0, s6.s0, s7.s0, s0.s1, s1.s1, s2.s1, s3.s1, s4.s1, s5.s1, s6.s1, s7.s1);
    char16 out1 = (char16)(s0.s2, s1.s2, s2.s2, s3.s2, s4.s2, s5.s2, s6.s2, s7.s2, s0.s3, s1.s3, s2.s3, s3.s3, s4.s3, s5.s3, s6.s3, s7.s3);
    const int outputOffset = (y * inputChannelC4 + x) * 8 * 4;
    vstore16(out0, 0, output_ptr + outputOffset);
    vstore16(out1, 0, output_ptr + outputOffset + 16);
#endif
}

__kernel void conv2d_1x1_ic_oc_weight_quant_buffer(GLOBAL_SIZE_2_DIMS
#ifdef USE_LOW_BIT_WEIGHT_INT4
                                            __global const uchar *input_ptr,
                                            __global uchar *output_ptr, //(Ci/packCin， Co/packCout, packCin， packCout)
#else
                                            __global const char *input_ptr,
                                            __global char *output_ptr, //(Ci/packCin， Co/packCout, packCin， packCout)
#endif
                                            __private const int input_channel,
                                            __private const int output_channel,
                                            __private const int icPack,
                                            __private const int ocPack) {
    int x  = get_global_id(0); // ic / icPack
    int y = get_global_id(1); // oc / ocPack

    DEAL_NON_UNIFORM_DIM2(x, y);
    const int xin = x * icPack;
    const int yin = y * ocPack;
    const int inputChannelC4 = (input_channel + icPack - 1) / icPack;
    const int outputChannelC4 = (output_channel + ocPack - 1) / ocPack;
#ifdef USE_LOW_BIT_WEIGHT_INT4
    const int inputOffset = (yin * input_channel + xin) / 2;
    const int outputOffset = ((x * outputChannelC4 + y) * icPack * ocPack) / 2;
    for(int i = 0; i < icPack; ++i){
        for(int j = 0; j < ocPack / 2; ++j){
            int index0 = (yin + j * 2) * input_channel + xin + i;
            int index1 = (yin + j * 2 + 1) * input_channel + xin + i;
            uchar s0 = input_ptr[index0/2];
            uchar s1 = input_ptr[index1/2];
            s0 = (index0 % 2) == 0 ? (s0 & 0xf0) : ((s0 & 0x0f) << 4);
            s1 = (index1 % 2) == 0 ? (s1 >> 4) : (s1 & 0x0f);
            output_ptr[outputOffset + i * (ocPack / 2) + j] = s0 | s1;
        }
    }
#else
    const int inputOffset = yin * input_channel + xin;
    const int outputOffset = (x * outputChannelC4 + y) * icPack * ocPack;
    for(int i = 0; i < icPack; ++i){
        for(int j = 0; j < ocPack; ++j){
            output_ptr[outputOffset + i * ocPack + j] = input_ptr[inputOffset + j * input_channel + i];
        }
    }
#endif
}
