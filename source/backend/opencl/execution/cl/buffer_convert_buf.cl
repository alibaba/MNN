#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }
// convert data from buffer(nhwc) to buffer(nc4hw4)
__kernel void nhwc_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                   __global const INPUT_TYPE *input_ptr,
                                   __private const int height,
                                   __private const int width, __private const int channels,
                                   __global OUTPUT_TYPE *output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_4_idx;

    const int remain_channel                = channels - channel_4_idx;
    float4 values                           = convert_float4(vload4(0, input_ptr + buffer_offset));

    if (remain_channel == 3) {
        values.w = 0;
    } else if (remain_channel == 2) {
        values.z = 0;
        values.w = 0;
    } else if (remain_channel == 1) {
        values.y = 0;
        values.z = 0;
        values.w = 0;
    }
    const int out_offset = (((batch_idx * ((channels+3)/4) + channel_4_idx/4) * height + height_idx) * width + width_idx)*4;
    vstore4(CONVERT_OUTPUT4(values), 0, output+out_offset);
}

// convert data from buffer(nchw) to buffer(nc4hw4)
__kernel void nchw_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                   __global const INPUT_TYPE *input_ptr,
                                   __private const int height, __private const int width, __private const int channels,
                                   __global OUTPUT_TYPE *output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = image_width_idx / width << 2;
    const int buffer_offset = ((batch_idx * channels + channel_4_idx) * height + height_idx) * width + width_idx;

    const int remain_channel    = channels - channel_4_idx;
    const int height_width_size = height * width;
    float4 output_values    = 0;

    if (remain_channel >= 4) {
        int offset      = buffer_offset;
        output_values.x = (float)*(input_ptr + offset);
        offset += height_width_size;
        output_values.y = (float)*(input_ptr + offset);
        offset += height_width_size;
        output_values.z = (float)*(input_ptr + offset);
        offset += height_width_size;
        output_values.w = (float)*(input_ptr + offset);
    } else if (remain_channel == 3) {
        int offset      = buffer_offset;
        output_values.x = (float)*(input_ptr + offset);
        offset += height_width_size;
        output_values.y = (float)*(input_ptr + offset);
        offset += height_width_size;
        output_values.z = (float)*(input_ptr + offset);
    } else if (remain_channel == 2) {
        int offset      = buffer_offset;
        output_values.x = (float)*(input_ptr + offset);
        offset += height_width_size;
        output_values.y = (float)*(input_ptr + offset);
    } else if (remain_channel == 1) {
        int offset      = buffer_offset;
        output_values.x = (float)*(input_ptr + offset);
    }

    const int out_offset = (((batch_idx * ((channels+3)/4) + channel_4_idx/4) * height + height_idx) * width + width_idx)*4;
    vstore4(CONVERT_OUTPUT4(output_values), 0, output+out_offset);
}


__kernel void nchw_buffer_to_nchw_buffer(GLOBAL_SIZE_2_DIMS
                                   __global INPUT_TYPE *input_ptr,
                                   __private const int height, __private const int width, __private const int channels,
                                   __private const int input_pad_left, __private const int input_pad_right,
                                   __private const int output_pad_left, __private const int output_pad_right,
                                   __global OUTPUT_TYPE *output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int src_width = width + input_pad_left + input_pad_right;
    const int dst_width = width + output_pad_left + output_pad_right;
    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_idx = image_width_idx / width;
    const int in_offset = ((batch_idx * channels + channel_idx) * height + height_idx) * src_width + width_idx + input_pad_left;
    const int out_offset = ((batch_idx * channels + channel_idx) * height + height_idx) * dst_width + width_idx + output_pad_left;

    output[out_offset] = (OUTPUT_TYPE)input_ptr[in_offset];
}

// convert data from image(b h, ic/4 w ic4) to buffer(nhwc)
__kernel void nc4hw4_buffer_to_nhwc_buffer(GLOBAL_SIZE_2_DIMS
                                    __global OUTPUT_TYPE *output,
                                    __private const int height, __private const int width,
                                    __private const int channels,
                                    __global INPUT_TYPE *input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_4_idx;

    const int in_offset = (((batch_idx * ((channels+3)/4) + channel_4_idx/4) * height + height_idx) * width + width_idx)*4;
    
    float4 values        = convert_float4(vload4(0, input_ptr+in_offset));
    const int remain_channel = channels - channel_4_idx;
    if (remain_channel >= 4) {
        vstore4(CONVERT_OUTPUT4(values), 0, output + buffer_offset);
    } else if (remain_channel == 3) {
        int offset     = buffer_offset;
        output[offset] = (OUTPUT_TYPE)values.x;
        offset++;
        output[offset] = (OUTPUT_TYPE)values.y;
        offset++;
        output[offset] = (OUTPUT_TYPE)values.z;
    } else if (remain_channel == 2) {
        int offset     = buffer_offset;
        output[offset] = (OUTPUT_TYPE)values.x;
        offset++;
        output[offset] = (OUTPUT_TYPE)values.y;
    } else if (remain_channel == 1) {
        int offset     = buffer_offset;
        output[offset] = (OUTPUT_TYPE)values.x;
    }
}

// convert data from buffer(nc4hw4) to buffer(nchw)
__kernel void nc4hw4_buffer_to_nchw_buffer(GLOBAL_SIZE_2_DIMS
                                    __global OUTPUT_TYPE *output,
                                    __private const int height, __private const int width,
                                    __private const int channels,
                                    __global INPUT_TYPE *input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;
    const int width_idx  = image_width_idx % width;
    int channel_4_idx    = (image_width_idx / width) * 4;
    int buffer_offset    = ((batch_idx * channels + channel_4_idx) * height + height_idx) * width + width_idx;
    
    const int in_offset = (((batch_idx * ((channels+3)/4) + channel_4_idx/4) * height + height_idx) * width + width_idx)*4;
    float4 values    = convert_float4(vload4(0, input_ptr+in_offset));

    const int height_width_size = height * width;

    const int remain_channel = channels - channel_4_idx;

    if (remain_channel >= 4) {
        int offset     = buffer_offset;
        output[offset] = (OUTPUT_TYPE)values.x;
        offset += height_width_size;
        output[offset] = (OUTPUT_TYPE)values.y;
        offset += height_width_size;
        output[offset] = (OUTPUT_TYPE)values.z;
        offset += height_width_size;
        output[offset] = (OUTPUT_TYPE)values.w;
    } else if (remain_channel == 3) {
        int offset     = buffer_offset;
        output[offset] = (OUTPUT_TYPE)values.x;
        offset += height_width_size;
        output[offset] = (OUTPUT_TYPE)values.y;
        offset += height_width_size;
        output[offset] = (OUTPUT_TYPE)values.z;
    } else if (remain_channel == 2) {
        int offset     = buffer_offset;
        output[offset] = (OUTPUT_TYPE)values.x;
        offset += height_width_size;
        output[offset] = (OUTPUT_TYPE)values.y;
    } else if (remain_channel == 1) {
        int offset     = buffer_offset;
        output[offset] = (OUTPUT_TYPE)values.x;
    }
}

__kernel void nc4hw4_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                    __global const INPUT_TYPE *input_ptr,
                                    __private const int2 output_shape,
                                    __private const int2 src_stride,
                                    __private const int2 dst_stride,
                                    __global OUTPUT_TYPE *output
) {

    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / output_shape.x;
    const int height_idx        = image_height_idx % output_shape.x;
    const int width_idx         = image_width_idx % output_shape.y;
    const int channel_block_idx = image_width_idx / output_shape.y;
    int2 src_bc_offset = src_stride * (int2)(batch_idx, channel_block_idx);
    int2 dst_bc_offset = dst_stride * (int2)(batch_idx, channel_block_idx);
    int src_buffer_offset =
        (((src_bc_offset.x + src_bc_offset.y) * output_shape.x + height_idx) * output_shape.y + width_idx) * 4;
    int dst_buffer_offset =
        (((dst_bc_offset.x + dst_bc_offset.y) * output_shape.x + height_idx) * output_shape.y + width_idx) * 4;
    
    vstore4(CONVERT_OUTPUT4(vload4(0, input_ptr + src_buffer_offset)), 0, output+dst_buffer_offset);
}

// convert kernel : from buffer(oihw) to image(oc/4 h w , ic oc4)
__kernel void conv2d_filter_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                            __global const FLOAT *input_ptr,
                                            __private const int output_channel,
                                            __private const int2 kernel_shape,
                                            __private const int ic_h_w_size,
                                            __private const int height_width_size,
                                            __global FLOAT *output) {
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

    FLOAT4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.w = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));

        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
        }
    }
    const int out_offset = (image_width_idx*height_width_size*((output_channel+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}

// convert kernel : from buffer(oihw) to image(oc/4 h w , ic oc4)
__kernel void conv2d_filter_buffer_to_nc4hw4_buffer_floatin(GLOBAL_SIZE_2_DIMS
                                            __global const float *input_ptr,
                                            __private const int output_channel,
                                            __private const int2 kernel_shape,
                                            __private const int ic_h_w_size,
                                            __private const int height_width_size,
                                            __global FLOAT *output) {
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

    FLOAT4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.w = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));

        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
        }
    }
    const int out_offset = (image_width_idx*height_width_size*((output_channel+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}


// convert kernel from buffer(mihw) to image(ic/4, ic4 h w m)
// but now dw only support m == 1
__kernel void dw_filter_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                        __global const FLOAT *input_ptr,
                                        __private const int4 kernel_shape,//[1, Cout, fh, fw]
                                        __private const int height_width_size,
                                        __global FLOAT *output) {
    const int image_width_idx  = get_global_id(0);//fh*fw
    const int image_height_idx = get_global_id(1);//UP_DIV(Cout, 4)

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    FLOAT4 output_values = 0;
    if (kernel_shape.x == 1) {
        const int input_channel_4_idx = image_height_idx * 4;
        const int buffer_height_idx   = image_width_idx / kernel_shape.w;
        const int buffer_width_idx    = image_width_idx % kernel_shape.w;

        const int buffer_offset =
            mad24(mad24(input_channel_4_idx, kernel_shape.z, buffer_height_idx), kernel_shape.w, buffer_width_idx);

        //input [1, Cout,                fh,                 fw]
        //index:[0, input_channel_4_idx, buffer_height_idx,  buffer_width_idx]
        const int remain_channel = kernel_shape.y - input_channel_4_idx;
        if (input_channel_4_idx < kernel_shape.y) {
            if (remain_channel >= 4) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.w = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 3) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));

            } else if (remain_channel == 2) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 1) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
            }
        }
    }

    //output NC4HW4 [1, fw*fh,            1, Cout/4]x oc4
    //index:        [0, image_width_idx,  0, image_height_idx]
    const int out_offset = (image_width_idx*((kernel_shape.y+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}

__kernel void dw_filter_buffer_to_nc4hw4_buffer_floatin(GLOBAL_SIZE_2_DIMS
                                        __global const float *input_ptr,
                                        __private const int4 kernel_shape,//[1, Cout, fh, fw]
                                        __private const int height_width_size,
                                        __global FLOAT *output) {
    const int image_width_idx  = get_global_id(0);//fh*fw
    const int image_height_idx = get_global_id(1);//UP_DIV(Cout, 4)

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    FLOAT4 output_values = 0;
    if (kernel_shape.x == 1) {
        const int input_channel_4_idx = image_height_idx * 4;
        const int buffer_height_idx   = image_width_idx / kernel_shape.w;
        const int buffer_width_idx    = image_width_idx % kernel_shape.w;

        const int buffer_offset =
            mad24(mad24(input_channel_4_idx, kernel_shape.z, buffer_height_idx), kernel_shape.w, buffer_width_idx);

        //input [1, Cout,                fh,                 fw]
        //index:[0, input_channel_4_idx, buffer_height_idx,  buffer_width_idx]
        const int remain_channel = kernel_shape.y - input_channel_4_idx;
        if (input_channel_4_idx < kernel_shape.y) {
            if (remain_channel >= 4) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.w = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 3) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));

            } else if (remain_channel == 2) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 1) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
            }
        }
    }

    //output NC4HW4 [1, fw*fh,            1, Cout/4]x oc4
    //index:        [0, image_width_idx,  0, image_height_idx]
    const int out_offset = (image_width_idx*((kernel_shape.y+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}
