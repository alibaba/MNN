#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

// convert data from buffer(nhwc) to buffer(nc16hw16) float input
__kernel void nhwc_buffer_to_nc16hw16_buffer(GLOBAL_SIZE_2_DIMS
                                   __global const INPUT_TYPE *input_ptr,
                                   __private const int height,
                                   __private const int width, __private const int channels,
                                   __global OUTPUT_TYPE *output,
                                   __private const int input_pad_left, __private const int input_pad_right,
                                   __private const int output_pad_left, __private const int output_pad_right) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_16_idx = (image_width_idx / width) << 4;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_16_idx;

    const int remain_channel                = min(channels - channel_16_idx, 16);
    INPUT_TYPE16 values                          = 0;
    INPUT_TYPE* values_ptr = (INPUT_TYPE*)(&values);

    __global const INPUT_TYPE *input_current_ptr = input_ptr + buffer_offset;

    for(int i = 0; i < remain_channel; ++i){
        values_ptr[i] = *(input_current_ptr + i);
    }
    const int out_offset = (((batch_idx * ((channels+15)/16) + channel_16_idx/16) * height + height_idx) * (output_pad_left + width + output_pad_right) + width_idx + output_pad_left)*16;
    vstore16(CONVERT_OUTPUT16(values), 0, output+out_offset);
    if(width_idx == 0){
        int pad_offset = (((batch_idx * ((channels+15)/16) + channel_16_idx/16) * height + height_idx) * (output_pad_left + width + output_pad_right))*16;
        for(int i = 0; i < output_pad_left; ++i){
            vstore16((OUTPUT_TYPE16)0, 0, output+pad_offset+i*16);
        }
        pad_offset += (output_pad_right + width) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore16((OUTPUT_TYPE16)0, 0, output+pad_offset+i*16);
        }
    }
}

// convert data from buffer(nchw) to buffer(nc16hw16)
__kernel void nchw_buffer_to_nc16hw16_buffer(GLOBAL_SIZE_2_DIMS
                                   __global const INPUT_TYPE *input_ptr,
                                   __private const int height, __private const int width, __private const int channels,
                                   __global OUTPUT_TYPE *output,
                                   __private const int input_pad_left, __private const int input_pad_right,
                                   __private const int output_pad_left, __private const int output_pad_right) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int src_width = width + input_pad_left + input_pad_right;
    const int dst_width = width + output_pad_left + output_pad_right;
    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_16_idx = image_width_idx / width << 4;
    const int buffer_offset = ((batch_idx * channels + channel_16_idx) * height + height_idx) * src_width + width_idx + input_pad_left;

    const int remain_channel    = min(channels - channel_16_idx, 16);
    const int height_width_size = height * width;
    INPUT_TYPE16 output_values  = 0;
    INPUT_TYPE *output_values_ptr = (INPUT_TYPE*)(&output_values);
    for(int i = 0; i < remain_channel; ++i){
        output_values_ptr[i] = *(input_ptr + buffer_offset + height_width_size * i);
    }

    if(width_idx == 0){
        int pad_offset = (((batch_idx * ((channels+15)/16) + channel_16_idx/16) * height + height_idx) * dst_width + 0)*16;
        for(int i = 0; i < output_pad_left; ++i){
            vstore16((OUTPUT_TYPE16)0, 0, output+pad_offset + 16 * i);
        }
        pad_offset += 16 * (width + output_pad_left);
        for(int i = 0; i < output_pad_right; ++i){
            vstore16((OUTPUT_TYPE16)0, 0, output+pad_offset + 16 * i);
        }
    }
    const int out_offset = (((batch_idx * ((channels+15)/16) + channel_16_idx/16) * height + height_idx) * dst_width + width_idx + output_pad_left)*16;
    vstore16(CONVERT_OUTPUT16(output_values), 0, output+out_offset);
}

// convert data from image(b h, ic/16 w ic16) to buffer(nhwc)
__kernel void nc16hw16_buffer_to_nhwc_buffer(GLOBAL_SIZE_2_DIMS
                                    __global OUTPUT_TYPE *output,
                                    __private const int height, __private const int width,
                                    __private const int channels,
                                    __global INPUT_TYPE *input_ptr,
                                   __private const int input_pad_left, __private const int input_pad_right,
                                   __private const int output_pad_left, __private const int output_pad_right) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_16_idx = (image_width_idx / width) << 4;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_16_idx;

    const int in_offset = (((batch_idx * ((channels+15)/16) + channel_16_idx/16) * height + height_idx) * (input_pad_left + width + input_pad_right) + width_idx + input_pad_left)*16;

    INPUT_TYPE16 values        = vload16(0, input_ptr+in_offset);
    INPUT_TYPE* values_ptr = (INPUT_TYPE*)(&values);
    const int remain_channel = min(channels - channel_16_idx, 16);
    for(int i = 0; i < remain_channel; ++i){
        output[buffer_offset + i] = (OUTPUT_TYPE)values_ptr[i];
    }
}

// convert data from buffer(nc16hw16) to buffer(nchw)
__kernel void nc16hw16_buffer_to_nchw_buffer(GLOBAL_SIZE_2_DIMS
                                    __global OUTPUT_TYPE *output,
                                    __private const int height, __private const int width,
                                    __private const int channels,
                                    __global INPUT_TYPE *input_ptr,
                                   __private const int input_pad_left, __private const int input_pad_right,
                                   __private const int output_pad_left, __private const int output_pad_right) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);
    
    const int src_width = width + input_pad_left + input_pad_right;
    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;
    const int width_idx  = image_width_idx % width;
    int channel_16_idx    = (image_width_idx / width) << 4;
    int buffer_offset    = ((batch_idx * channels + channel_16_idx) * height + height_idx) * width + width_idx;
    
    const int in_offset = (((batch_idx * ((channels+15)/16) + channel_16_idx/16) * height + height_idx) * src_width + width_idx + input_pad_left)*16;
    INPUT_TYPE16 values    = vload16(0, input_ptr+in_offset);
    INPUT_TYPE *values_ptr = (INPUT_TYPE*)(&values);

    const int height_width_size = height * width;

    const int remain_channel = min(channels - channel_16_idx, 16);
    for(int i = 0; i < remain_channel; ++i){
        output[buffer_offset + i * height_width_size] = (OUTPUT_TYPE)values_ptr[i];
    }
}

__kernel void nc4hw4_buffer_to_nc16hw16_buffer(GLOBAL_SIZE_2_DIMS
                                    __global const INPUT_TYPE *input_ptr,
                                    __private const int2 output_shape,
                                    __private const int2 src_stride,
                                    __private const int2 dst_stride,
                                    __global OUTPUT_TYPE *output,
                                    __private const int input_pad_left,
                                    __private const int input_pad_right,
                                    __private const int output_pad_left,
                                    __private const int output_pad_right,
                                    __private const int channelc4
) {

    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / output_shape.x;
    const int height_idx        = image_height_idx % output_shape.x;
    const int width_idx         = image_width_idx % output_shape.y;
    const int channel_block_idx = image_width_idx / output_shape.y;
    const int in_channel_block_idx = channel_block_idx << 2;
    const int dst_width = output_pad_left + output_shape.y + output_pad_right;
    int2 src_bc_offset = src_stride * (int2)(batch_idx, in_channel_block_idx);
    int2 dst_bc_offset = dst_stride * (int2)(batch_idx, channel_block_idx);
    int src_buffer_offset =
        (((src_bc_offset.x + src_bc_offset.y) * output_shape.x + height_idx) * output_shape.y + width_idx) * 4;
    int dst_buffer_offset =
        (((dst_bc_offset.x + dst_bc_offset.y) * output_shape.x + height_idx) * dst_width + width_idx + output_pad_left) * 16;
    int width_height_size4 = output_shape.x * output_shape.y * 4;
    INPUT_TYPE4 values0 = vload4(0, input_ptr + src_buffer_offset);
    INPUT_TYPE4 values1 = in_channel_block_idx + 1 >= src_bc_offset.x ? (INPUT_TYPE4)0 : vload4(0, input_ptr + src_buffer_offset + width_height_size4);
    INPUT_TYPE4 values2 = in_channel_block_idx + 2 >= src_bc_offset.x ? (INPUT_TYPE4)0 : vload4(0, input_ptr + src_buffer_offset + width_height_size4 * 2);
    INPUT_TYPE4 values3 = in_channel_block_idx + 3 >= src_bc_offset.x ? (INPUT_TYPE4)0 : vload4(0, input_ptr + src_buffer_offset + width_height_size4 * 3);
    
    vstore16(CONVERT_OUTPUT16((INPUT_TYPE16)(values0.s0123, values1.s0123, values2.s0123, values3.s0123)), 0, output+dst_buffer_offset);
    if(width_idx == 0){
        int pad_offset = (((dst_bc_offset.x + dst_bc_offset.y) * output_shape.x + height_idx) * dst_width) * 16;
        for(int i = 0; i < output_pad_left; ++i){
            vstore16((OUTPUT_TYPE16)0, 0, output+pad_offset + 16 * i);
        }
        pad_offset += 16 * (output_shape.y + output_pad_left);
        for(int i = 0; i < output_pad_right; ++i){
            vstore16((OUTPUT_TYPE16)0, 0, output+pad_offset + 16 * i);
        }
    }
}

__kernel void nc16hw16_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                    __global const INPUT_TYPE *input_ptr,
                                    __private const int2 output_shape,
                                    __private const int2 src_stride,
                                    __private const int2 dst_stride,
                                    __global OUTPUT_TYPE *output,
                                    __private const int input_pad_left,__private const int input_pad_right,
                                    __private const int output_pad_left,__private const int output_pad_right,
                                    __private const int channelc4
) {

    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / output_shape.x;
    const int height_idx        = image_height_idx % output_shape.x;
    const int width_idx         = image_width_idx % output_shape.y;
    const int channel_block_idx = image_width_idx / output_shape.y;
    const int out_channel_block_idx = channel_block_idx << 2;
    int2 src_bc_offset = src_stride * (int2)(batch_idx, channel_block_idx);
    int2 dst_bc_offset = dst_stride * (int2)(batch_idx, out_channel_block_idx);
    int width_height_size4 = output_shape.x * output_shape.y * 4;
    int src_buffer_offset =
        (((src_bc_offset.x + src_bc_offset.y) * output_shape.x + height_idx) * (input_pad_left + output_shape.y + input_pad_right) + width_idx + input_pad_left) * 16;
    int dst_buffer_offset =
        (((dst_bc_offset.x + dst_bc_offset.y) * output_shape.x + height_idx) * output_shape.y + width_idx) * 4;
    INPUT_TYPE16 values = vload16(0, input_ptr + src_buffer_offset);
    
    vstore4(CONVERT_OUTPUT4(values.s0123), 0, output+dst_buffer_offset);
    if(out_channel_block_idx + 1 >= channelc4) return;
    vstore4(CONVERT_OUTPUT4(values.s4567), 0, output+dst_buffer_offset + width_height_size4);
    if(out_channel_block_idx + 2 >= channelc4) return;
    vstore4(CONVERT_OUTPUT4(values.s89ab), 0, output+dst_buffer_offset + 2 * width_height_size4);
    if(out_channel_block_idx + 3 >= channelc4) return;
    vstore4(CONVERT_OUTPUT4(values.scdef), 0, output+dst_buffer_offset + 3 * width_height_size4);
}
